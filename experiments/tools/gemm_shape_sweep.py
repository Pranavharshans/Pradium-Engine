#!/usr/bin/env python3
"""Sweep EXL3 GEMM kernel shapes for the M=1 decode shapes.

For a representative EXL3 linear of each decode shape, calls the raw
``ext.exl3_gemm`` entry with every ``force_shape_idx`` / ``force_num_sms``
combination and times each inside one captured CUDA graph. The point is to test
whether the default dispatch (autotuned cooperative kernels) leaves a
significantly faster kernel on the table for batch-1 decode.

Reports ms and effective GB/s per (shape, kernel config); GPU-side only.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch


def find_exl3_linears(root, max_depth: int = 5):
    found: dict[tuple[int, int], object] = {}

    def rec(obj, depth, seen):
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor) or isinstance(obj, (str, bytes, int, float)):
            return
        if getattr(obj, "quant_type", None) == "exl3" and hasattr(obj, "in_features"):
            key = (int(obj.in_features), int(obj.out_features))
            found.setdefault(key, obj)
            return
        data = getattr(obj, "__dict__", None)
        if not isinstance(data, dict):
            return
        for value in data.values():
            if isinstance(value, (list, tuple)):
                for item in value:
                    rec(item, depth + 1, seen)
            else:
                rec(value, depth + 1, seen)

    rec(root, 0, set())
    return found


def weight_bytes(inner) -> int:
    total = 0
    for name in ("trellis", "suh", "svh", "mul1_tensor", "mcg_tensor"):
        t = getattr(inner, name, None)
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


def time_graph(fn, iters: int, capture_n: int = 16, warmup: int = 5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(capture_n):
            fn()
    torch.cuda.synchronize()
    replays = max(1, iters // capture_n)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (replays * capture_n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--iters", type=int, default=128)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import exllamav3
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.ext import exllamav3_ext as ext

    cfg = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(cfg)
    model = Model.from_config(cfg)
    model.load(progressbar=False, device="cuda:0")
    torch.cuda.synchronize()

    n_shapes = int(ext.exl3_gemm_num_kernel_shapes())
    linears = find_exl3_linears(model)
    print(f"module={exllamav3.__file__}  kernel shapes={n_shapes}")

    results = []
    for (in_features, out_features), liner in sorted(linears.items()):
        inner = liner.inner
        trellis = inner.trellis
        suh = inner.suh
        svh = inner.svh
        mul1 = bool(getattr(inner, "mul1", inner.mul1_tensor is not None))
        mcg = bool(getattr(inner, "mcg", inner.mcg_tensor is not None))
        if trellis is None:
            continue
        wb = weight_bytes(inner)
        x = torch.randn(1, in_features, dtype=torch.half, device="cuda:0")
        y = torch.empty(1, out_features, dtype=torch.half, device="cuda:0")
        xh = torch.empty_like(x)

        row = {
            "in_features": in_features,
            "out_features": out_features,
            "weight_mb": wb / 2**20,
            "trials": [],
        }
        print(f"\n== {out_features} x {in_features}  ({wb / 2**20:.2f} MB)  mul1={mul1} mcg={mcg}")
        for shape_idx in range(1, n_shapes + 1):
            for num_sms in (0,):
                def call(shape_idx=shape_idx, num_sms=num_sms):
                    ext.exl3_gemm(x, trellis, y, suh, xh, svh, shape_idx, mcg, mul1, num_sms)

                try:
                    ms = time_graph(call, args.iters)
                except Exception as exc:  # noqa: BLE001
                    row["trials"].append({"shape_idx": shape_idx, "num_sms": num_sms,
                                          "error": str(exc)[:120]})
                    continue
                entry = {
                    "shape_idx": shape_idx,
                    "num_sms": num_sms,
                    "ms": ms,
                    "gb_s": wb / (ms / 1000.0) / 1e9 if ms else None,
                }
                row["trials"].append(entry)
                print(f"   shape={shape_idx} sms={num_sms}: {ms * 1000:8.2f} us  {entry['gb_s']:7.1f} GB/s")
        ok = [t for t in row["trials"] if "ms" in t]
        if ok:
            best = min(ok, key=lambda t: t["ms"])
            row["best"] = best
            print(f"   BEST shape={best['shape_idx']} {best['ms'] * 1000:.2f} us {best['gb_s']:.1f} GB/s")
        # default dispatch for comparison (force = -1)
        def default_call():
            ext.exl3_gemm(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0)

        ms = time_graph(default_call, args.iters)
        row["default_ms"] = ms
        row["default_gb_s"] = wb / (ms / 1000.0) / 1e9 if ms else None
        print(f"   DEFAULT (-1): {ms * 1000:.2f} us {row['default_gb_s']:.1f} GB/s")
        results.append(row)
        del x, y, xh

    payload = {
        "exllamav3_file": exllamav3.__file__,
        "torch": torch.__version__,
        "kernel_shapes": n_shapes,
        "results": results,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
