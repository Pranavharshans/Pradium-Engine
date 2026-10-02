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
| EXP-0000 | Freeze/verify the official baseline and decompose a decode step into GPU-busy vs host-gap time on the VM | RUNNING | `eba4849` | (tooling only) | — | — | — |

## Planned candidates (order decided by EXP-0000 evidence)

| Approach | Idea | Status |
|---|---|---|
| A | Whole-forward CUDA graph: capture the entire batch-1 decode step (incl. sampling) as one graph replayed with static buffers; removes per-op host dispatch and per-token sync from the critical path | PLANNED |
| B | Persistent execution / GPU-side scheduling (MPK-inspired) | PLANNED |
| C | EXL3 matrix-kernel improvements (packed-weight load coalescing, fragment decode, prefill/GEMM regressions) | PLANNED |
| D | Adaptive decode attention (tiling/split policy, direct-output path; validate existing patch 0002) | PLANNED |
| E | Cross-op fusion beyond upstream fused blocks | PLANNED |
| F | LM-head / sampling integration (avoid logits materialization where semantics allow) | PLANNED |
| G | Periodic stall removal (readbacks, syncs, allocations, token delivery) | PLANNED |
| H | Existing patch series validation (0001 graph replay bindings, 0002 direct attention) | PLANNED |

## Notes

- Physical ceiling context (to be confirmed by EXP-0000 measurements): the
  Frozen checkpoint is 1.74 GB of weights read once per decode token. At the
  RTX 3060's ~360 GB/s memory bandwidth the pure weight-read floor is ~4.8 ms
  per token (~207 tok/s), i.e. ~1.9x the SS baseline before any attention,
  LM-head or host overhead. The remaining headroom is therefore bounded; the
  loop aims for the best measured result and reports the true ceiling honestly.
- The VM is an unprivileged container: no kernel profilers; use userspace
  profiling (torch/CUPTI) and the harness's telemetry.
