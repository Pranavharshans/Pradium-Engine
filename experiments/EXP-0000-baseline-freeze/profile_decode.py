#!/usr/bin/env python3
"""EXP-0000: decode-step decomposition profiler for ExLlamaV3 / Pradium.

Drives ``exllamav3``'s ``Generator`` exactly like
``benchmarks/adapters/exllamav3.py`` does for an official run (ArgmaxSampler,
EOS-masked fixed-length generation, same Generator/Cache construction) and
reports, for one fixed ``(prompt_tokens, max_new_tokens)`` request:

  decode_tok_s / ITL   : same definitions as benchmarks/core/timing.py
  kernel_ms_per_step   : summed CUDA kernel time inside a profiled window
  gap_ms_per_step      : profiled window span - busy time (host/sync not hidden)
  launches_per_step    : CUDA kernel launches per decode step
  h2d/d2h per step     : device copies (readbacks) per decode step
  top kernels          : by total CUDA time in the window

This script must run in an environment where ``exllamav3`` resolves to the
revision under test. It does not import anything from the Pradium patch series,
so it runs against both the upstream wheel and a patched build.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    rank = (q / 100.0) * (len(data) - 1)
    low = int(rank)
    high = min(low + 1, len(data) - 1)
    frac = rank - low
    return data[low] * (1.0 - frac) + data[high] * frac


def _stats(values_ms: list[float]) -> dict:
    if not values_ms:
        return {}
    return {
        "n": len(values_ms),
        "mean_ms": statistics.fmean(values_ms),
        "median_ms": statistics.median(values_ms),
        "p95_ms": _percentile(values_ms, 95),
        "p99_ms": _percentile(values_ms, 99),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
        "stdev_ms": statistics.stdev(values_ms) if len(values_ms) > 1 else 0.0,
    }


def _interval_union_us(intervals: list[tuple[float, float]]) -> float:
    """Total covered time of [start, end) intervals, in microseconds."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def _nvidia_smi_sample() -> dict:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,power.draw,clocks.sm,temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "util_pct": float(parts[0]),
            "power_w": float(parts[1]),
            "sm_clock_mhz": float(parts[2]),
            "temp_c": float(parts[3]),
            "mem_used_mb": float(parts[4]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _git_provenance() -> dict:
    """Revision of the checkout this script was launched from."""
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[2]
    out: dict = {"repo": str(repo)}
    try:
        out["revision"] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        out["branch"] = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        out["dirty"] = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout.strip()
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def _eos_ids(cfg) -> list[int]:
    raw = getattr(cfg, "eos_token_id_list", None) or getattr(cfg, "eos_token_id", None)
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return []


def _profiler_breakdown(profiler, torch, top_n: int) -> dict:
    """Extract kernel/copy totals and the device timeline coverage."""
    ka = profiler.key_averages()
    events: list[tuple[float, float]] = []
    try:
        for ev in profiler.events():
            if "CUDA" not in str(getattr(ev, "device_type", "")):
                continue
            tr = getattr(ev, "time_range", None)
            start = getattr(tr, "start", None)
            end = getattr(tr, "end", None)
            if start is not None and end is not None and end > start:
                events.append((float(start), float(end)))
    except Exception:  # noqa: BLE001
        events = []

    kernel_us = 0.0
    h2d = 0
    d2h = 0
    top: list[dict] = []
    for e in ka:
        key = str(e.key)
        count = int(getattr(e, "count", 0))
        self_us = float(
            getattr(e, "self_device_time_total", None)
            or getattr(e, "self_cuda_time_total", 0.0)
            or 0.0
        )
        if key.startswith("Memcpy HtoD"):
            h2d += count
            continue
        if key.startswith("Memcpy DtoH"):
            d2h += count
            continue
        if key.startswith("Memcpy") or key.startswith("Memset"):
            continue
        kernel_us += self_us
        top.append({"key": key, "count": count, "self_device_us": self_us})
    top.sort(key=lambda r: -r["self_device_us"])
    return {
        "kernel_us": kernel_us,
        "busy_us": _interval_union_us(events),
        "span_us": (max(e[1] for e in events) - min(e[0] for e in events)) if events else 0.0,
        "h2d": h2d,
        "d2h": d2h,
        "top": top[:top_n],
    }


def run(args) -> dict:
    import torch
    import exllamav3
    from exllamav3 import ArgmaxSampler, Cache, Config, Generator, Job, Model, Tokenizer

    cfg = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(cfg)
    model = Model.from_config(cfg)
    cache = Cache(
        model,
        max_num_tokens=int(args.max_num_tokens),
        max_batch_size=int(args.max_batch_size),
        max_history=0,
    )
    model.load(progressbar=False, device="cuda:0")
    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        max_batch_size=int(args.max_batch_size),
        max_chunk_size=int(args.max_chunk_size),
        cpu_cache_size=0,
    )
    torch.cuda.synchronize()

    prompt = [1000 + (i * 7919) % 30000 for i in range(args.prompt_tokens)]

    def one_request() -> dict:
        ids = torch.tensor([prompt], dtype=torch.long)
        job = Job(
            input_ids=ids,
            max_new_tokens=int(args.max_new_tokens),
            min_new_tokens=int(args.max_new_tokens),
            stop_conditions=_eos_ids(cfg),
            identifier="profile",
            sampler=ArgmaxSampler(),
        )
        torch.cuda.synchronize()
        submit_ns = time.perf_counter_ns()
        generator.enqueue(job)

        token_ts_ns: list[int] = []
        step_wall_ns: list[int] = []
        profiled_wall_ns: list[int] = []
        profiler = None
        profiled_steps = 0
        profiled_done = False
        profiled: dict = {}
        steps = 0
        eos_reason = None
        safety = args.prompt_tokens // args.max_chunk_size + args.max_new_tokens + 64
        done = False

        while steps < safety and not done:
            if (
                profiler is None
                and not profiled_done
                and len(token_ts_ns) >= args.profile_after_tokens
            ):
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                )
                profiler.start()

            t0 = time.perf_counter_ns()
            results = generator.iterate()
            t1 = time.perf_counter_ns()
            step_wall_ns.append(t1 - t0)
            if profiler is not None:
                profiled_steps += 1
                profiled_wall_ns.append(t1 - t0)

            for res in results:
                new_ids = res.get("token_ids")
                if new_ids is not None:
                    for _ in range(int(new_ids.numel())):
                        token_ts_ns.append(t1)
                if res.get("eos"):
                    done = True
                    eos_reason = res.get("eos_reason")

            if profiler is not None and (profiled_steps >= args.profile_steps or done):
                profiler.stop()
                profiled = _profiler_breakdown(profiler, torch, args.top_kernels)
                profiler = None
                profiled_done = True
            steps += 1

        torch.cuda.synchronize()
        return {
            "submit_ns": submit_ns,
            "token_ts_ns": token_ts_ns,
            "step_wall_ns": step_wall_ns,
            "steps": steps,
            "eos_reason": eos_reason,
            "profiled_steps": profiled_steps,
            "profiled_wall_ns": profiled_wall_ns,
            "profiled": profiled,
        }

    summary: dict = {
        "label": args.label,
        "repo": _git_provenance(),
        "torch": torch.__version__,
        "exllamav3_file": exllamav3.__file__,
        "exllamav3_version": getattr(exllamav3, "__version__", None),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "config": {
            "model_dir": args.model_dir,
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_num_tokens": args.max_num_tokens,
            "max_batch_size": args.max_batch_size,
            "max_chunk_size": args.max_chunk_size,
            "sampler": "ArgmaxSampler",
            "eos_masked": True,
        },
        "env": {
            k: os.environ.get(k)
            for k in sorted(os.environ)
            if k.startswith("EXL3_") or k.startswith("EXLLAMA_")
        },
        "requests": [],
    }

    for i in range(args.warmup_requests + 1):
        r = one_request()
        is_measured = i >= args.warmup_requests
        entry: dict = {
            "role": "measured" if is_measured else "warmup",
            "steps": r["steps"],
            "tokens": len(r["token_ts_ns"]),
            "eos_reason": r["eos_reason"],
        }
        ts = r["token_ts_ns"]
        if ts:
            entry["ttft_ms"] = (ts[0] - r["submit_ns"]) / 1e6
        if len(ts) >= 2:
            itl_ms = [(ts[k] - ts[k - 1]) / 1e6 for k in range(1, len(ts))]
            drop = min(args.drop_first_itl, max(len(itl_ms) - 1, 0))
            entry.update(
                {
                    "decode_ms": (ts[-1] - ts[0]) / 1e6,
                    "decode_tok_s": (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9),
                    "itl_ms": _stats(itl_ms[drop:]),
                    "itl_dropped_first": drop,
                }
            )
        if is_measured:
            entry["step_wall_ms"] = _stats([v / 1e6 for v in r["step_wall_ns"]])
            prof = r["profiled"] or {}
            n = r["profiled_steps"] or 0
            entry["profiled"] = {
                "steps": n,
                "wall_ms_per_step": (
                    statistics.fmean(r["profiled_wall_ns"]) / 1e6
                    if r["profiled_wall_ns"]
                    else None
                ),
                "kernel_ms_per_step": (prof.get("kernel_us", 0.0) / 1000.0 / n) if n else None,
                "busy_ms_per_step": (prof.get("busy_us", 0.0) / 1000.0 / n) if n else None,
                "span_ms_per_step": (prof.get("span_us", 0.0) / 1000.0 / n) if n else None,
                "h2d_per_step": (prof.get("h2d", 0) / n) if n else None,
                "d2h_per_step": (prof.get("d2h", 0) / n) if n else None,
                "top_kernels": prof.get("top", []),
            }
        summary["requests"].append(entry)

    summary["telemetry_after"] = _nvidia_smi_sample()
    summary["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-num-tokens", type=int, default=32768)
    ap.add_argument("--max-batch-size", type=int, default=16)
    ap.add_argument("--max-chunk-size", type=int, default=2048)
    ap.add_argument("--warmup-requests", type=int, default=1)
    ap.add_argument("--profile-after-tokens", type=int, default=8)
    ap.add_argument("--profile-steps", type=int, default=24)
    ap.add_argument("--drop-first-itl", type=int, default=2)
    ap.add_argument("--top-kernels", type=int, default=25)
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--print-json", action="store_true", help="print the full JSON instead of a compact summary")
    args = ap.parse_args()

    summary = run(args)
    text = json.dumps(summary, indent=2, default=str)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(text + "\n")
    if args.print_json:
        print(text)
        return 0

    measured = [r for r in summary["requests"] if r["role"] == "measured"]
    print(f"== {summary['label']} == {summary['exllamav3_file']}")
    print(
        f"config: prompt={summary['config']['prompt_tokens']} "
        f"new={summary['config']['max_new_tokens']} "
        f"cache={summary['config']['max_num_tokens']} env={summary['env']}"
    )
    for r in measured:
        print(
            f"tokens={r['tokens']} eos={r['eos_reason']} "
            f"ttft_ms={r.get('ttft_ms', float('nan')):.2f} "
            f"decode_tok_s={r.get('decode_tok_s', float('nan')):.2f}"
        )
        itl = r.get("itl_ms", {})
        print(
            f"  itl ms: med={itl.get('median_ms', float('nan')):.3f} "
            f"mean={itl.get('mean_ms', float('nan')):.3f} "
            f"p99={itl.get('p99_ms', float('nan')):.3f} max={itl.get('max_ms', float('nan')):.3f}"
        )
        prof = r.get("profiled", {})
        print(
            f"  profiled steps={prof.get('steps')} "
            f"wall={prof.get('wall_ms_per_step')} ms/step "
            f"kernel={_fmt(prof.get('kernel_ms_per_step'))} ms/step "
            f"busy={_fmt(prof.get('busy_ms_per_step'))} ms/step "
            f"span={_fmt(prof.get('span_ms_per_step'))} ms/step "
            f"h2d={prof.get('h2d_per_step')} d2h={prof.get('d2h_per_step')} per step"
        )
        for k in prof.get("top_kernels", [])[:12]:
            per_step_us = k["self_device_us"] / max(prof.get("steps") or 1, 1)
            print(f"    {per_step_us / 1000.0:7.3f} ms/step x{k['count']:>5}  {k['key'][:96]}")
    return 0


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    sys.exit(main())
