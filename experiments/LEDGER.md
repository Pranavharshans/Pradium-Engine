# Pradium engine experiment ledger

Goal: **>= 2x sustained single-request (batch 1) decode throughput** on each of
the nine frozen core workloads vs the frozen official ExLlamaV3 baseline, with no
quality loss, measured on the authorized RTX 3060 VM.

- Accepted engine branch: `bench/exl3-runtime-b0-rtx3060`
- Frozen official baseline results:
  `benchmarks/campaigns/rtx3060-minicpm5-exl3-b0` (ExLlamaV3 v1.5.3 wheel,
  upstream commit `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`)
- Frozen baseline decode medians (tok/s): SS 109.80, SM 85.58, SL 96.45,
  MS 109.36, MM 101.31, ML 87.95, LS 99.81, LM 89.29, LL 77.19
- Benchmark scope: **only** the 3x3 core matrix (SS..LL). Other suites are out
  of scope for this program.

Status vocabulary: PLANNED / RUNNING / KEEP / REJECT / INCONCLUSIVE.

## Experiments

| ID | Hypothesis (one line) | Status | Parent commit | Candidate commit | Evidence commit | Incremental vs parent | Cumulative vs frozen baseline |
|---|---|---|---|---|---|---|---|
| EXP-0000 | Freeze/verify the baseline and decompose a decode step into GPU kernel, host gap and launch/readback counts | KEEP (evidence) | `eba4849` | tooling `04fa920` | `17c2c9e` | n/a | n/a |
| EXP-0001 | Patch 0001 (cache graph-replay argument bindings) removes host bookkeeping per replay | RUNNING | `86a5290` | build `cand-series-0001-0002` (direct attn OFF) | — | — | — |
| EXP-0002 | Patch 0002 (direct single-request attention, regime 2) removes split/combine fixed cost for contexts <= 4096 tokens | RUNNING | EXP-0001 | same build, `EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=4097` | — | — | — |

## EXP-0000 results (measured)

Baseline verification (harness matrix suite, SS/MM/LL, 3 warmups + 10 measured,
medians): SS 112.42, MM 102.02, LL 75.28 tok/s vs frozen 109.80 / 101.31 /
77.19 (+2.4% / +0.7% / -2.5%). Baseline reproduced; the frozen baseline stays
the comparison anchor.

Decode-step decomposition (torch/CUPTI, 24-step window, nine short/medium/long
configurations). Per-step budget at SS (128/64), profiled window wall
8.63 ms/step (harness median 8.90-9.11 ms):

| component | ms/step | launches/step | note |
|---|---|---|---|
| `exl3_mgemm<4,...,512>` | 2.254 | 42 | MLP gate+up |
| `exl3_gemv_int8_sq<4,...>` | 1.632 | 84 | 2 per layer |
| `_paged_attn_decode_split_kernel` | 1.056 | 42 | 25 µs/layer at 192-token context |
| `exl3_mgemm<4,...,256>` | 1.037 | 42 | MLP down |
| `exl3_gemm<6,...>` (LM head) | 0.610 | 1 | 6-bit; ~91% of BW |
| attention combine + rope + kv_update | 0.327 | 126 | |
| rms_norm (2 variants) + residual add | 0.266 | 127 | |
| act + sampler + misc | 0.056 | 43 | |
| **GPU kernel total** | **7.238** | | |
| host gap (wall - busy) | 1.30 | 129 kernel + 84 graph launches, 5 memcpy, 1 sync | |

At 4096-token prompts the attention split kernel grows to 1.96 ms/step and the
kernel total to 8.22 ms/step; everything else is unchanged. The largest
non-weight, non-attention cost is the 200+ host launches per step.

## Ceiling analysis (why 2x is bounded)

Weights read once per token (294 EXL3 linears + 6-bit LM head):
42 layers × 47.17M weights at 4.0127 bpw = 993.6 MB, head 267.4M at 6.0079 bpw
= 200.8 MB, total **1.194 GB/token**. At the RTX 3060's 360 GB/s that is
**3.32 ms/token of pure weight reads**; the measured GEMM+head time (4.49 ms)
is ~74% of bandwidth. A 2x decode step (4.45 ms for SS) therefore leaves only
~1.1 ms for attention, KV, activations, sampling, host dispatch and sync —
below any practical bound (attention alone costs 0.5 ms of KV traffic at 4096
tokens, and the harness's own adapter/scheduler path adds ~0.5-1.0 ms/token).

Realistic program target: ~1.4-1.7x, pursued in this order:

| Approach | Measured basis | Status |
|---|---|---|
| D (attention fixed cost) | 1.06-1.96 ms/step split kernel; 25-47 µs/layer for microseconds of work | RUNNING (EXP-0002) |
| G (host stalls/launches) | 129 kernel + 84 graph launches/step; 1.30 ms/step gap; tails (ITL p99 14-70 ms) amplify host jitter | PLANNED (EXP-0003) |
| C (EXL3 kernel efficiency) | GEMMs at ~74% of 360 GB/s | PLANNED (EXP-0004) |
| A/B/E/F | no measured basis yet | PLANNED / low priority |

## Notes

- The VM is an unprivileged container: no kernel profilers; profiling uses
  torch/CUPTI plus harness telemetry.
- Harness medians are stable to ~±3%; individual runs have heavy tails
  (an SS run can drop from 114 to 70 tok/s), so decisions use medians of 10
  with the candidate and baseline measured on the same VM session.
