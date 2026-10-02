# EXP-0002 — direct single-request attention (regime 2)

Status: RUNNING (measurement queued behind the candidate build)

## Hypothesis

For short-context batch-1 decode the flash-decoding split/combine schedule is
dominated by its fixed cost, not by KV work: EXP-0000 measured 1.06 ms/step at
192-token context and 1.16 ms/step at 1280 tokens (56-block grid per layer,
`programs = 2`, `splits_cap = 28`, only 1-3 live splits). A single-pass kernel
that writes the final output directly (no combine launch, one CTA per program)
should remove most of that fixed cost.

## Gate

`EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS` is compared against
`block_table.shape[1] * PAGE_SIZE + q_len`, and `block_table.shape[1]` is the
generator's padded staging width (multiples of 16 pages = 4096 tokens). With
4097 the direct regime applies to every workload whose padded width is 16
pages — SS, SM, SL, MS, MM, ML (prompt+output <= 4096) — and never to
LS/LM/LL (width 32). The gate is stable for the whole request because
`max_seq_len` (prompt + max_new_tokens) is fixed when the job is enqueued.

Numerics: the split geometry is forced to `num_splits = 1` and
`split_len = ceil(bound/block_n)*block_n` covers the full bound
(`split_config` clamps `num_splits` to `splits_cap`), so all keys are still
covered; only the reduction order changes (one pass instead of
partials + combine). The expected difference is float-reduction noise, which
`tools/attn_path_compare.py` measures per prompt length (logit deltas and
argmax agreement) in the same process.

## Evidence

`results/EXP-0002-direct/` (matrix + profiles + correctness) and
`results/attn-diff/` (numeric differential).
