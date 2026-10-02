# EXP-0001 / EXP-0002 — Pradium patch series (0001, 0002)

Status: RUNNING

## Builds

One materialized tree serves both experiments:
`/workspace/exp/builds/cand-series-0001-0002`
(upstream pin `d3739fd` + `patches/0001-cache-graph-replay-bindings.patch` +
`patches/0002-direct-single-request-attention.patch`, built in place with
`TORCH_CUDA_ARCH_LIST=8.6`).

Attribution method: patch 0002 is inert unless
`EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS` makes `direct_attention_eligible()` true,
so the same binary measures both experiments:

- EXP-0001 (patch 0001 only): run with the env var unset/0.
- EXP-0002 (patch 0002 on top): run with the env var set.

## EXP-0001 — cache graph replay argument bindings

Hypothesis: replacing the per-replay O(n·m) site scan in `Graph::launch` with a
cached binding vector (recomputed only when the parameter layout changes)
removes host bookkeeping in the graph replay path. Affected bottleneck: host
gap inside a decode step (~1.0–1.3 ms measured in EXP-0000). Risk: binding
layout mismatch (guarded by the ADDRESS/UB-sanitized native test in
`engine/pradium/tests`). Acceptance: no correctness divergence, decode
medians not below the frozen baseline, and a measurable gap reduction in the
step profile.

## EXP-0002 — direct single-request attention (regime 2)

Hypothesis: for short-context batch-1 decode, the split/combine flash-decoding
path is dominated by fixed cost (EXP-0000: 1.06 ms split + 0.17 ms combine per
step at 192-token context, ~25 µs per layer for work that is microseconds of
real compute). A single-pass kernel writing the final output directly (no
combine launch, no partial buffers) should cut most of that.

Known defect carried from the patch series: the shipped gate compares
`block_table.shape[1] * PAGE_SIZE + q_len` (the **padded** generator table
width, minimum 16 pages = 4096 tokens) against the env limit, so the README's
example value (513) can never select regime 2. With the padded width, one
limit covers every workload whose table width is 16 pages
(SS, SM, SL, MS, MM, ML = prompt+output ≤ 4096 tokens) and excludes
LS/LM/LL (width 32). A precise live-length gate is possible without a GPU
sync because the generator's `cache_seqlens` staging buffer is pinned host
memory; that refinement is a Python-only follow-up if the coarse gate shows a
win.

Acceptance: greedy token IDs identical to the accepted parent on the core
matrix, no core workload regressed, decode medians improved on the workloads
where the gate applies.

## Predeclared decision criteria

- Correctness: greedy token IDs identical to the frozen official session for
  every matched (profile, run) pair, or a recorded, explained divergence with
  evidence that it is float-reduction noise (for EXP-0002: the direct-vs-split
  differential on logits, `tools/attn_path_compare.py`).
- Improvement counts only if the geomean of the nine per-workload decode
  medians improves by more than the observed baseline reproducibility band
  (frozen vs re-verified baseline: -2.5%..+2.4%, geomean +0.2%).
- Any workload whose median drops below its parent's median by more than 5%
  is a regression that rejects the experiment unless the drop is explained by
  a measurement artefact reproduced under the same conditions.
- Medians come from the frozen protocol (3 warmups + 10 measured runs per
  workload); no run is dropped.

## Commands

```sh
# build (VM)
experiments/EXP-0001-patch-series/build_candidate.sh /workspace/exp/builds/cand-series-0001-0002

# EXP-0001 measurement (patch 0001 only)
experiments/tools/run_matrix.sh exllamav3 <results>/EXP-0001 SS,SM,SL,MS,MM,ML,LS,LM,LL /workspace/exp/builds/cand-series-0001-0002

# EXP-0002 measurement (patch 0002 enabled)
EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=4097 experiments/tools/run_matrix.sh exllamav3 <results>/EXP-0002 SS,SM,SL,MS,MM,ML,LS,LM,LL /workspace/exp/builds/cand-series-0001-0002
```
