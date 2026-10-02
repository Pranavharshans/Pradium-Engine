# EXP-0000 — Baseline freeze and decode-step decomposition

Status: RUNNING (evidence collection; no engine change)

## Purpose

1. Verify that the frozen official ExLlamaV3 baseline is reproducible on the VM
   with the campaign's exact configuration.
2. Decompose one batch-1 decode step into GPU-busy time, host/launch/sync gap,
   launch counts and readbacks, so the first optimization targets measured
   evidence instead of assumption.

This experiment changes no engine code. All files here are tooling and evidence.

## Frozen baseline (never replaced)

Source of truth:
`benchmarks/campaigns/rtx3060-minicpm5-exl3-b0/` (committed) — ExLlamaV3 v1.5.3
upstream wheel, commit `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`, model
`ewin-reg/MiniCPM5-2B-EXL3-Quantized` @ `e36fbb568f466739d7c244a5100d01acf93d3b7f`,
KV cache fp16, `max_num_tokens=32768`, `max_batch_size=16`,
`max_chunk_size=2048`, `cpu_cache_size=0`, ArgmaxSampler, 3 warmups +
10 measured runs per workload, median as headline statistic.

Frozen decode medians (tok/s), core matrix:

| SS | SM | SL | MS | MM | ML | LS | LM | LL |
|---|---|---|---|---|---|---|---|---|
| 109.80 | 85.58 | 96.45 | 109.36 | 101.31 | 87.95 | 99.81 | 89.29 | 77.19 |

(Full table with TTFT/prefill/TPOT/ITL/VRAM/power:
`benchmarks/campaigns/rtx3060-minicpm5-exl3-b0/exllamav3/derived-tables.md`.)

## Hypotheses to test against measurement

- H1 (host-bound): the campaign's telemetry shows GPU utilisation of only
  65–92% and batch-1 decode dominated by host dispatch on a 2-core CPU; a
  decode step should show a large non-overlapped gap (wall − kernel busy).
- H2 (bandwidth floor): the checkpoint is 1.740 GB of weights read once per
  token; at the 3060's ~360 GB/s the pure weight-read floor is ~4.8 ms/token,
  so kernel time is expected to be in the 5–7 ms range and any claim of a 2x
  must be checked against this floor.
- H3 (readbacks): each decode step must sync to obtain the sampled token; per
  step there should be a small number of D2H copies, and they are part of the
  serialized critical path.

## Method

- `profile_decode.py` drives `Generator` exactly like the official adapter
  (same construction, ArgmaxSampler, EOS-masked fixed-length generation) and
  measures per-step wall time (identical ITL definition to
  `benchmarks/core/timing.py`), CUDA kernel busy time (torch/CUPTI profiler),
  launch counts and copies.
- `verify_baseline.sh` re-runs the harness's own `matrix` suite for
  SS/MM/LL with the campaign's frozen runtime config into a fresh results dir.

## Acceptance for this experiment

- Harness re-run of SS/MM/LL lands within ±10% of the frozen decode medians;
  otherwise the environment changed and must be investigated before any engine
  experiment is judged.
- Profiling produces a per-step budget that sums to ~100% of the measured ITL
  (kernel + gap + copies), with top kernels attributed.

## Commands

```sh
# on the VM
/workspace/exp/pradium/experiments/EXP-0000-baseline-freeze/setup_vm_checkout.sh bench/exl3-runtime-b0-rtx3060
/workspace/exp/pradium/experiments/EXP-0000-baseline-freeze/verify_baseline.sh SS,MM,LL
/workspace/exp/pradium/experiments/EXP-0000-baseline-freeze/run_profile.sh exllamav3 baseline-SS --prompt-tokens 128 --max-new-tokens 64
/workspace/exp/pradium/experiments/EXP-0000-baseline-freeze/run_profile.sh exllamav3 baseline-LL --prompt-tokens 4096 --max-new-tokens 1024 --profile-steps 16
```

## Evidence

(Filled in as results arrive; see `results/`.)
