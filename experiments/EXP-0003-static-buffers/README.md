# EXP-0003 — static decode buffers (persistent per-module outputs)

Status: PLANNED (implementation ready; measurement after EXP-0001/0002)

## Hypothesis

A decoded token allocates, in every one of the 42 layers, four small
activations — the attention input-norm output, the attention block output, the
MLP input-norm output and the MLP output (≈168 allocations/token) — plus churn
of the addresses that the CUDA-graph replay argument comparison has to update
every step. Reusing one persistent buffer per (module, key) removes the
allocations and makes the replay pointers static.

Mechanism: host-side only. No kernel changes, no numerical change (same
kernels, same values, same order). Affected bottleneck: the host gap inside a
decode step (EXP-0000: ~1.3 ms/step under CUPTI, ~0.6 ms clean, 213 launches
+ ~168 allocations per step).

## Implementation

`patches/0003-static-decode-buffers.patch` (Python only — no extension
rebuild):

- `exllamav3/util/decode_buffers.py`: `decode_out(owner, key, shape, dtype,
  device)` returns a persistent buffer, reallocating only if the shape/dtype/
  device changes; `reuse(out, shape, dtype)` validates a caller-supplied
  buffer.
- `RMSNorm.forward(..., out=None)`, `MLP.forward(..., out=None)`,
  `BC_Attention.step(..., out=None)` / `Attention.forward(..., out=None)`:
  write into `out` when it matches the shape and dtype they would allocate.
- `TransformerBlock.forward`: passes persistent buffers on the single-token
  decode path only (bsz == 1 and q_len == 1); prefill and batched shapes keep
  allocating. `EXL3_DECODE_BUFFER_REUSE=0` disables the whole path for A/B
  attribution.

Safety argument: each buffer is written by one producer and read by the next
operation on the same stream before the producer runs again; the four keys
never alias each other, and none of them aliases the residual stream.

## Measurement plan

With the patch applied to the candidate tree (Python-only):

```sh
EXL3_DECODE_BUFFER_REUSE=1 experiments/tools/run_candidate.sh ... EXP-0003-<rev>buffs <tree>
EXL3_DECODE_BUFFER_REUSE=0 experiments/tools/run_candidate.sh ... EXP-0003-<rev>nobuff <tree>
```

Acceptance: token IDs identical to the accepted parent (numerically identical
by construction), decode medians improved beyond noise, no workload regressed.
