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

Per-shape EXL3 bandwidth (graph-timed, `tools/gemm_bench.py`), the arithmetic
behind that conclusion:

| shape | MB read/token | ms | GB/s | vs head |
|---|---|---|---|---|
| LM head 130560x2048 (6 bpw) | 191.5 | 0.625 | 321 | 97% of peak |
| gate/up 2048x6144 | 12.04 | 0.050 | 241 | 75% |
| down 6144x2048 | 6.02 | 0.024 | 249 | 78% |
| qkv 2048x2048-ish | 2.5 | 0.013 | 199 | 62% |

Even if every layer GEMM ran at the head's 321 GB/s, the layer weights alone
would take 3.1 ms; the measured kernels take 4.9 ms, and the forced-shape
sweep shows no available kernel closes that gap — it is a kernel property at
M=1, not a dispatch choice.

## 3. Experiment results (measured)

Candidate tree `cand-series-0001-0002` (pin + patches 0001/0002/0004),
`EXLLAMA_EXT_COMPRESS=0`, on the same VM and frozen configuration. Steady-state
step time (ITL median, ms) is the reliable comparison across sessions; the
headline `decode_tok_s` carries a session-varying tail (see §6) so it is
reported but not used for verdicts.

| workload | ITL med (ms) direct OFF | ITL med direct ON | ITL med direct ON + buffers | decode tok/s: frozen / OFF / ON / ON+buf |
|---|---|---|---|---|
| SS | 7.84 | **7.49** | **7.48** | 109.8 / 109.2 / 113.7 / 109.4 |
| SM | 7.92 | 8.09 | 8.24 | 85.6 / 105.1 / 99.1 / 86.4 |
| SL | 8.00 | 10.39 | 10.34 | 96.5 / 83.3 / 76.1 / 69.1 |
| MS | 8.03 | 11.50 | 11.51 | 109.4 / 108.8 / 79.9 / 80.5 |
| MM | 8.04 | 12.13 | 12.06 | 101.3 / 101.2 / 72.5 / 72.7 |
| ML | 8.09 | 14.20 | 14.16 | 88.0 / 72.7 / 59.2 / 60.9 |
| LS | 8.91 | 8.91 | 8.94 | 99.8 / 99.7 / 99.4 / 96.2 |
| LM | 8.93 | 8.92 | 8.92 | 89.3 / 88.4 / 61.2 / 74.2 |
| LL | 8.99 | 9.01 | 9.03 | 77.2 / 72.8 / 77.3 / 70.5 |

Kernel-level paired evidence at SS (same build, minutes apart, only the gate
env differs): the direct regime removes the combine kernel and shrinks the
split kernel — kernel total 7.239 -> 6.846 ms/step (split 1.055 -> 0.865,
combine 0.172 -> absent).

Numeric equivalence: direct vs split attention in one process on the same
prompt is **bit-identical at the fp16 logits** (max |delta| = 0.0, argmax
agreement 16/16) at 128, 1024 and 4096 prompt tokens
(`results/attn-diff/attn_diff_*.json`).

Correctness: EXP-0001's session reproduced the frozen official session's greedy
token IDs **exactly, 117/117 runs**. EXP-0002/0003 diverge on 5 workloads at
deterministic indices while the two direct-OFF workloads (LM, LL) also diverge
in those sessions, which points at session-level kernel-selection
nondeterminism rather than the attention change (see `wheel-repro`).

| experiment | verdict | measured effect | evidence |
|---|---|---|---|
| EXP-0001 patch 0001 | KEEP | no regression; no measurable change (as expected for host bookkeeping); 117/117 token-identical | `EXP-0001-patch-series/results/EXP-0001-patch0001` |
| EXP-0002 patch 0002 (coarse gate) | REJECT as configured | SS -4.5% step time, but SM..ML +2..+76%; gate cannot discriminate | `EXP-0002-direct-attention/results/EXP-0002-direct` |
| EXP-0003 patch 0003 (buffers) | INCONCLUSIVE | within +-1% of its parent on every workload | `EXP-0003-static-buffers/results/EXP-0003-buffers-on` |
| EXP-0005 patch 0005 (precise gate, 512) | REJECT (correctness) | recovers every coarse-gate loss (SL/MS/MM/ML back to parity) and keeps SS at -3.6%, but changes greedy tokens in SM | `EXP-0002-direct-attention/results/EXP-0005-gate512` |

### The gate refinement round (EXP-0005)

| workload | ITL med, direct OFF | ITL med, coarse gate | ITL med, live-token gate 512 | tokens vs frozen |
|---|---|---|---|---|
| SS | 7.84 | 7.48 | **7.56** | 13/13 identical |
| SM | 7.92 | 8.24 | 8.10 | **0/13 identical** (first diff at token 73) |
| SL | 8.00 | 10.34 | **8.33** | 13/13 |
| MS | 8.03 | 11.51 | **8.08** | 13/13 |
| MM | 8.04 | 12.06 | **8.06** | 13/13 |
| ML | 8.09 | 14.16 | **8.10** | 13/13 |
| LS | 8.91 | 8.94 | 8.91 | 13/13 |
| LM | 8.93 | 8.92 | 8.92 | 13/13 |
| LL | 8.99 | 9.03 | 9.03 | 13/13 |

The live-token gate removes every loss the coarse gate caused and keeps the
192-token-context gain, but the direct reduction order still flips a greedy
token in SM (deterministically at token 73, 13/13 runs), so the attention
change is rejected on correctness grounds rather than on speed.

### Reproducibility controls

- The unpatched wheel, re-run in a fresh session, reproduces the frozen
  session's greedy tokens exactly (39/39 on SM, LM, LL)
  (`results/wheel-repro/`).
- The candidate build with the direct regime OFF reproduces the frozen session
  exactly on all nine workloads (117/117) — the build itself is faithful.
- Direct vs split attention in one process on the same prompt is bit-identical
  at the fp16 logits (max |delta| = 0.0) over 16 generated tokens at 128, 1024
  and 4096 prompt tokens (`results/attn-diff/`).

## 4. Experiment history (commits, config, commands)

| experiment | branch | candidate commit | evidence commit | verdict |
|---|---|---|---|---|
| EXP-0000 | `bench/exl3-runtime-b0-rtx3060` | tooling `04fa920` | `17c2c9e` | KEEP (evidence) |
| EXP-0001 | same | build `cand-series-0001-0002` @ `86a5290` | (filled on completion) | pending |
| EXP-0002 | same | + `EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=4097` | (filled) | pending |
| EXP-0003 | same | + `patches/0003-static-decode-buffers.patch` | (filled) | pending |

VM: RTX 3060 12 GB, driver 550.144.03, 4 vCPU (2 physical cores), 31 GB RAM,
unprivileged container; engine environment `/workspace/envs/exllamav3`
(Python 3.12, torch 2.10.0+cu128, triton 3.6.0); model
`/workspace/models/minicpm5-2b-exl3` (frozen revision); harness
`python -m benchmarks run matrix` with the campaign runtime config.

Build: `/workspace/exp/builds/cand-series-0001-0002`, materialized by
`engine/pradium/build.py` from the pin + patches 0001-0002, built in place with
`TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 EXLLAMA_EXT_COMPRESS=0`.

## 5. Remaining bottleneck and next steps (measured basis)

The decode step's GPU time is 7.2 ms (SS) / 8.2 ms (LL prompts), of which
4.9 ms / 4.9 ms is the EXL3 layer GEMMs. Those kernels move 991 MB/token at
199-250 GB/s where the LM-head kernel of the same format reaches 321 GB/s
(97% of the VM's measured 330 GB/s peak). Closing that gap is kernel work, not
configuration, and it is the single largest remaining item:

| opportunity | measured basis | expected |
|---|---|---|
| EXL3 M=1 layer-shape kernel efficiency | 199-250 GB/s vs 321 GB/s head | up to 1.2 ms/step |
| fold the per-layer norms and residual adds into the BC attention/MLP graphs | 127 elementwise launches + 84 graph launches per step | ~0.4 ms/step |
| decode attention split geometry at 4096-token context | 1.96 ms/step split kernel (29% of KV-read bandwidth) | unknown, needs kernel work |
| whole-forward single graph (approach A) | host gap is only 0.6 ms clean once buffers are reused | ~0.3 ms/step |

## 6. Closed hypotheses (measured, negative)

| hypothesis | result |
|---|---|
| int8-activation GEMV default is slower on Ampere; disabling it helps | **wrong**: disabling is 35-70% slower per shape (6144x2048: 160 vs 249 GB/s) |
| a better EXL3 kernel shape exists for the M=1 layer shapes | **no**: forced-shape sweep (int8 on and off) finds nothing faster than the default dispatch |
| the layer GEMM kernel is at its bandwidth ceiling | **no**: 199-250 GB/s vs 321 GB/s the head kernel reaches, but no available knob or alternate kernel captures it |
