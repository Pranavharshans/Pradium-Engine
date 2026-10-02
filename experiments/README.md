# Pradium engine optimization experiments

Measurement-driven optimization loop for the Pradium engine. One accepted
engine branch, one branch per experiment, evidence committed per experiment.

## Layout

```
experiments/
  LEDGER.md                 # central ledger: every experiment, status, commits, results
  EXP-0000-baseline-freeze/ # frozen baseline + profiling tooling (this experiment)
  EXP-00NN-<slug>/          # per-experiment evidence: plan, raw results, summaries, decision
```

## Branches

- Accepted engine branch: `bench/exl3-runtime-b0-rtx3060`. Experiments integrate
  here only after KEEP.
- One branch per experiment: `experiment/EXP-00NN-<slug>`, cut from the accepted
  branch.

## VM workflow (GitHub is the transport)

1. Author changes locally in this repository, commit, push the experiment branch.
2. On the benchmark VM: `/workspace/exp/pradium` (isolated checkout, see
   `EXP-0000-baseline-freeze/setup_vm_checkout.sh`) fetches and checks out the
   exact candidate commit.
3. Run correctness + the nine core workloads on the VM. Save raw records,
   summaries and the decision under `experiments/EXP-00NN-<slug>/` in the VM
   checkout.
4. Copy the evidence files back to the local checkout (rsync over SSH), commit
   locally, push. Record the commit IDs in `LEDGER.md`.

The VM has read access to the repository but no push credentials; evidence is
pushed from the local checkout after being copied back, so local and remote
history stay synchronized with the exact VM-run revision recorded in the
evidence files.

## Ground rules

- Only the frozen 3x3 core matrix (SS..LL) is used for performance decisions.
- Every result is associated with the exact executed code revision.
- Rejected experiments keep their branch, commits and evidence; nothing is
  deleted or force-pushed away.
- No secrets, model weights or unrelated files are committed.
- The official ExLlamaV3 baseline is frozen at
  `benchmarks/campaigns/rtx3060-minicpm5-exl3-b0` and is never replaced or
  re-measured into its own session directories.
