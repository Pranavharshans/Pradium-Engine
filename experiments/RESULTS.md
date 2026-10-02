# Pradium engine optimization — measured results

This file is filled from committed evidence only. Every number below is tied to
a session directory, a code revision and a command in the experiment folders.
Nothing here is projected; see `LEDGER.md` for the per-experiment verdicts.

## 1. Frozen baseline (official ExLlamaV3 v1.5.3, RTX 3060, MiniCPM5-2B EXL3)

Committed campaign: `benchmarks/campaigns/rtx3060-minicpm5-exl3-b0`.
Re-verified in EXP-0000 on the same VM (medians of 10): SS 112.42, MM 102.02,
LL 75.28 tok/s vs frozen 109.80 / 101.31 / 77.19 (+2.4 / +0.7 / -2.5%).

| | SS | SM | SL | MS | MM | ML | LS | LM | LL |
|---|---|---|---|---|---|---|---|---|---|
| frozen decode tok/s | 109.80 | 85.58 | 96.45 | 109.36 | 101.31 | 87.95 | 99.81 | 89.29 | 77.19 |

## 2. Where a decoded token's time goes (EXP-0000, measured)

Per-step budget at SS (128/64), 24-step window, torch/CUPTI:

| component | ms/step | share |
|---|---|---|
| EXL3 layer GEMMs (294 linears, 4 bpw) | 4.92 | 68% |
| attention (split + combine + rope + kv) | 1.38 | 19% |
| LM head (6 bpw, 191 MB) | 0.61 | 8% |
| norms + residual add + activation | 0.27 | 4% |
| sampler | 0.01 | 0% |
| host gap (213 launches + readback + sync) | 0.6-1.3 | - |

Weights read per token: 991 MB (layers) + 191 MB (head) = **1.19 GB**. Measured
peak read bandwidth on this VM: **~330 GB/s** (`tools/bw_probe.py`), so the
pure weight-read floor is **3.6 ms/token** — i.e. a 2.00x decode step
(4.45 ms) would have to fit attention, activations, sampling, host work and
sync into 0.85 ms. That is not reachable; measured best-case composition is
~5.5-6 ms (1.5-1.6x).

## 3. Experiment results

(filled in as rounds complete)

| experiment | status | incremental | cumulative | evidence |
|---|---|---|---|---|
| EXP-0001 patch 0001 (replay bindings) | pending | | | |
| EXP-0002 patch 0002 (direct attention) | pending | | | |
| EXP-0003 patch 0003 (static buffers) | pending | | | |

## 4. Closed hypotheses (measured, negative)

| hypothesis | result |
|---|---|
| int8-activation GEMV default is slower on Ampere; disabling it helps | **wrong**: disabling is 35-70% slower per shape (6144x2048: 160 vs 249 GB/s) |
| a better EXL3 kernel shape exists for the M=1 layer shapes | **no**: forced-shape sweep (int8 on and off) finds nothing faster than the default dispatch |
| the layer GEMM kernel is at its bandwidth ceiling | **no**: 199-250 GB/s vs 321 GB/s the head kernel reaches, but no available knob or alternate kernel captures it |
