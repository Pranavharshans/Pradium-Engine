#!/usr/bin/env python3
"""Attention fixed-cost curve: split/combine kernel time vs live context.

Loads the model once and, for each prompt length, drives one decode request
exactly like profile_decode.py while profiling a window of decode steps, then
reports the per-step GPU time of the paged-attention kernels (split, combine,
rope, kv update) and of the whole step.

The point: separate the part of decode attention that scales with live context
from the part that does not (the fixed cost the direct single-pass regime
targets in EXP-0002).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

from importlib import import_module

KERNELS = ("_paged_attn_decode_split_kernel",
           "_paged_attn_decode_combine_kernel",
           "_paged_attn_decode_kernel",
           "rope_kernel",
           "_paged_kv_update_kernel")


def run_window(generator, prompt, max_new_tokens, profile_after, profile_steps):
    from exllamav3 import ArgmaxSampler, Job

    cfg = generator.model.config
    ids = torch.tensor([prompt], dtype=torch.long)
    job = Job(
        input_ids=ids,
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=int(max_new_tokens),
        stop_conditions=[],
        identifier="attncurve",
        sampler=ArgmaxSampler(),
    )
    generator.enqueue(job)
    emitted = 0
    steps = 0
    safety = max_new_tokens + 32
    profiler = None
    profiled_steps = 0
    window = None
    done = False
    while steps < safety and not done:
        if profiler is None and window is None and emitted >= profile_after:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            )
            profiler.start()
        results = generator.iterate()
        if profiler is not None:
            profiled_steps += 1
        for res in results:
            nid = res.get("token_ids")
            if nid is not None:
                emitted += int(nid.numel())
            if res.get("eos"):
                done = True
        steps += 1
        if profiler is not None and (profiled_steps >= profile_steps or done):
            profiler.stop()
            window = profiler.key_averages()
            profiler = None
    out = {"steps": profiled_steps}
    if window is not None:
        for e in window:
            key = str(e.key)
            for wanted in KERNELS:
                if wanted in key and "seq" not in key:
                    us = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
                    out.setdefault(wanted, 0.0)
                    out[wanted] += us / 1000.0
        total_us = 0.0
        for e in window:
            key = str(e.key)
            if key.startswith("Memcpy") or key.startswith("Memset"):
                continue
            if " " in key and "(" in key or True:
                pass
            us = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
            total_us += us
        out["kernel_total_ms"] = total_us / 1000.0
    return out, profiled_steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lengths", default="16,64,128,256,512,1024,2048,4096")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--profile-after", type=int, default=6)
    ap.add_argument("--profile-steps", type=int, default=12)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import exllamav3
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer

    cfg = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(cfg)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=32768, max_batch_size=16, max_history=0)
    model.load(progressbar=False, device="cuda:0")
    generator = Generator(
        model=model, cache=cache, tokenizer=tokenizer,
        max_batch_size=16, max_chunk_size=2048, cpu_cache_size=0,
    )
    torch.cuda.synchronize()

    results = {"exllamav3_file": exllamav3.__file__, "lengths": {}}
    print(f"module={exllamav3.__file__}")
    print(f"{'prompt':>7} {'split ms/step':>14} {'combine':>9} {'rope':>7} {'kvupd':>7} {'kernel total':>13}")
    for length in [int(x) for x in args.lengths.split(",")]:
        prompt = [1000 + (i * 7919) % 30000 for i in range(length)]
        out, steps = run_window(
            generator, prompt, args.max_new_tokens, args.profile_after, args.profile_steps
        )
        steps = max(steps, 1)
        row = {
            "steps": steps,
            "split_ms_per_step": out.get("_paged_attn_decode_split_kernel", 0.0) / steps,
            "combine_ms_per_step": out.get("_paged_attn_decode_combine_kernel", 0.0) / steps,
            "rope_ms_per_step": out.get("rope_kernel", 0.0) / steps,
            "kv_update_ms_per_step": out.get("_paged_kv_update_kernel", 0.0) / steps,
            "kernel_total_ms_per_step": out.get("kernel_total_ms", 0.0) / steps,
        }
        results["lengths"][str(length)] = row
        print(
            f"{length:>7} {row['split_ms_per_step']:>14.3f} {row['combine_ms_per_step']:>9.3f} "
            f"{row['rope_ms_per_step']:>7.3f} {row['kv_update_ms_per_step']:>7.3f} "
            f"{row['kernel_total_ms_per_step']:>13.3f}"
        )
        sys.stdout.flush()

    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
