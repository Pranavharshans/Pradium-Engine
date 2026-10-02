#!/usr/bin/env python3
"""Peak achievable GPU memory bandwidth on this VM (calibration for the
GEMM efficiency question). Torch copy (read+write) and reduction (read-only)
over 256 MB tensors, deep async loops, CUDA-event timed."""

from __future__ import annotations

import json
import sys

import torch


def timed(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main() -> int:
    n = 64 * 1024 * 1024  # 64M floats = 256 MB
    x = torch.randn(n, dtype=torch.float32, device="cuda:0")
    y = torch.empty_like(x)
    out = torch.empty((), dtype=torch.float32, device="cuda:0")

    ms = timed(lambda: y.copy_(x))
    copy_gb_s = (2 * n * 4) / (ms / 1000.0) / 1e9

    ms_sum = timed(lambda: torch.sum(x, out=out))
    sum_gb_s = (n * 4) / (ms_sum / 1000.0) / 1e9

    # 16-bit read (closer to the GEMM's weight stream)
    xh = torch.randn(n, dtype=torch.half, device="cuda:0")
    outh = torch.empty((), dtype=torch.half, device="cuda:0")
    ms_h = timed(lambda: torch.sum(xh, out=outh))
    sum_h_gb_s = (n * 2) / (ms_h / 1000.0) / 1e9

    result = {
        "device": torch.cuda.get_device_name(0),
        "copy_ms": round(ms, 3),
        "copy_read_write_gb_s": round(copy_gb_s, 1),
        "sum_fp32_ms": round(ms_sum, 3),
        "sum_fp32_read_gb_s": round(sum_gb_s, 1),
        "sum_fp16_ms": round(ms_h, 3),
        "sum_fp16_read_gb_s": round(sum_h_gb_s, 1),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
