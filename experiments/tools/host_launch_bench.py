#!/usr/bin/env python3
"""Host launch-cost probe: how much CPU time each per-step operation costs.

Times (async, no sync inside the loop) the individual host-side operations a
decode step performs ~127 times: an ext.rms_norm call, a torch elementwise add,
a torch.empty_like allocation, and the full Module.forward wrapper for a norm.
Reports microseconds per call so the elementwise launch budget can be compared
against a C++ fusion alternative.
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import torch


def time_calls(fn, iters: int) -> dict:
    for _ in range(200):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / iters / 1000.0)
    torch.cuda.synchronize()
    return {
        "us_per_call_median": statistics.median(samples),
        "us_per_call_min": min(samples),
    }


def main() -> int:
    import exllamav3
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3 import Config, Model, Tokenizer

    model_dir = sys.argv[1]
    cfg = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(cfg)
    model = Model.from_config(cfg)
    model.load(progressbar=False, device="cuda:0")
    torch.cuda.synchronize()

    rows = 1
    dim = 2048
    x = torch.randn(rows, dim, dtype=torch.half, device="cuda:0")
    w = torch.randn(dim, dtype=torch.half, device="cuda:0")
    y = torch.empty_like(x)
    z = torch.empty_like(x)

    out = {}

    out["ext.rms_norm (direct)"] = time_calls(
        lambda: ext.rms_norm(x, w, y, 1e-5, 0.0, 1.0, False, False), 2000
    )
    out["ext.rms_norm_res_in (direct)"] = time_calls(
        lambda: ext.rms_norm_res_in(x, w, y, z, 1e-5, 0.0, 1.0), 2000
    )
    out["torch add_ in place"] = time_calls(lambda: z.add_(y), 2000)
    out["torch empty_like"] = time_calls(lambda: torch.empty_like(x), 2000)
    out["torch view -1,dim"] = time_calls(lambda: x.view(-1, dim), 2000)
    out["ext.rms_norm with view+alloc"] = time_calls(
        lambda: ext.rms_norm(
            x.view(-1, dim), w, torch.empty_like(x).view(-1, dim), 1e-5, 0.0, 1.0, False, False
        ),
        2000,
    )

    # Module.forward wrapper for the model's actual attn_norm module
    norm_module = None
    for module in model.modules:
        for name, attr in vars(module).items():
            if type(attr).__name__ == "RMSNorm":
                norm_module = attr
                break
        if norm_module is not None:
            break
    if norm_module is not None:
        xx = x.view(1, 1, dim)
        params = {}
        out["RMSNorm.forward (module)"] = time_calls(
            lambda: norm_module.forward(xx, params, out_dtype=torch.half), 2000
        )
        out["RMSNorm.forward res_in (module)"] = time_calls(
            lambda: norm_module.forward(xx, params, out_dtype=torch.half, residual_in=z.view(1, 1, dim)),
            2000,
        )

    print(json.dumps(
        {
            "exllamav3_file": exllamav3.__file__,
            "torch": torch.__version__,
            "calls": out,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
