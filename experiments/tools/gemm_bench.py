#!/usr/bin/env python3
"""Per-shape EXL3 decode-GEMM microbenchmark (M=1).

Loads the frozen model, groups the EXL3 linear modules by (in_features,
out_features) and measures one representative per shape with:

  eager  : N back-to-back ``bc.run_alloc(x)`` calls, CUDA-event timed (includes
           host launch overhead)
  graph  : the same calls inside one captured CUDA graph replayed N/16 times
           (pure kernel time; isolates launch overhead)

Effective bandwidth is computed from the measured weight bytes of the module
(trellis + suh + svh + mul1 + mcg tensors), the bytes a decode step actually
reads.

Reports JSON so the same shapes can be compared across revisions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def find_exl3_linears(root, max_depth: int = 5):
    """Bounded reflection walk; returns EXL3 linear wrappers, one per shape."""
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
    for name in ("trellis", "suh", "svh", "mul1_tensor", "mcg_tensor", "bias"):
        t = getattr(inner, name, None)
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


def bench_eager(liner, x, out_features, iters, warmup):
    inner = liner.inner
    for _ in range(warmup):
        liner.inner.bc.run_alloc(x, out_features, False)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        liner.inner.bc.run_alloc(x, out_features, False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_graph(liner, x, out_features, iters, warmup, capture_n=16):
    """Capture capture_n calls in one graph; time replay over iters/capture_n."""
    try:
        for _ in range(warmup):
            liner.inner.bc.run_alloc(x, out_features, False)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # Warm the private pool so capture has stable allocations
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                liner.inner.bc.run_alloc(x, out_features, False)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(capture_n):
                liner.inner.bc.run_alloc(x, out_features, False)
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
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-shapes", type=int, default=0, help="0 = all")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import exllamav3
    from exllamav3 import Config, Model, Tokenizer

    cfg = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(cfg)
    model = Model.from_config(cfg)
    model.load(progressbar=False, device="cuda:0")
    torch.cuda.synchronize()

    linears = find_exl3_linears(model)
    shapes = sorted(linears.items(), key=lambda kv: -weight_bytes(kv[1].inner))
    if args.max_shapes:
        shapes = shapes[: args.max_shapes]

    results = []
    print(f"module={exllamav3.__file__}")
    print(f"{'shape (out x in)':>20} {'MB':>8} {'eager ms':>9} {'GB/s':>7} {'graph ms':>9} {'GB/s':>7}")
    for (in_features, out_features), liner in shapes:
        x = torch.randn(1, in_features, dtype=torch.half, device="cuda:0")
        wb = weight_bytes(liner.inner)
        eager_ms = bench_eager(liner, x, out_features, args.iters, args.warmup)
        graph_ms = bench_graph(liner, x, out_features, args.iters, args.warmup)
        row = {
            "in_features": in_features,
            "out_features": out_features,
            "weight_mb": wb / 2**20,
            "eager_ms": eager_ms,
            "eager_gb_s": wb / (eager_ms / 1000.0) / 1e9 if eager_ms else None,
        }
        if isinstance(graph_ms, float):
            row["graph_ms"] = graph_ms
            row["graph_gb_s"] = wb / (graph_ms / 1000.0) / 1e9 if graph_ms else None
        else:
            row["graph_error"] = graph_ms["error"]
        results.append(row)
        print(
            f"{out_features:>9} x {in_features:<8} {row['weight_mb']:>8.2f} {eager_ms:>9.3f} "
            f"{row['eager_gb_s']:>7.1f} "
            + (
                f"{row.get('graph_ms', float('nan')):>9.3f} {row.get('graph_gb_s', 0):>7.1f}"
                if "graph_ms" in row
                else f"{'capture failed':>9}"
            )
        )
        del x

    payload = {
        "label": args.json_out or "",
        "exllamav3_file": exllamav3.__file__,
        "torch": torch.__version__,
        "iters": args.iters,
        "env": {k: v for k, v in __import__("os").environ.items() if k.startswith("EXL3_")},
        "shapes": results,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
