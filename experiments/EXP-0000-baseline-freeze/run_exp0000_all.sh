#!/bin/bash
# EXP-0000 full evidence run: frozen-baseline verification (harness) and
# decode-step profiles for the nine core-matrix input/output configurations.
# Runs on the VM; writes everything under the experiment's results directory.
set -uo pipefail

REPO=${REPO:-/workspace/exp/pradium}
E=$REPO/experiments/EXP-0000-baseline-freeze
OUT=${OUT:-$E/results}
LOG=$OUT/run_all.log
mkdir -p "$OUT"

{
  echo "EXP-0000 evidence run"; date -u
  echo; echo "=== [1/2] baseline verification (harness matrix suite, profiles SS,MM,LL) ==="
  "$E/verify_baseline.sh" SS,MM,LL
  echo; echo "=== [2/2] decode-step profiles, nine core configurations ==="
  while read -r name pin pout; do
    [ -z "$name" ] && continue
    echo; echo "--- profile $name (${pin}/${pout}) ---"
    "$E/run_profile.sh" exllamav3 "baseline-$name" \
      --prompt-tokens "$pin" --max-new-tokens "$pout" \
      --profile-steps 24 --profile-after-tokens 8
  done <<'CFG'
SS 128 64
SM 128 256
SL 128 1024
MS 1024 64
MM 1024 256
ML 1024 1024
LS 4096 64
LM 4096 256
LL 4096 1024
CFG
  echo; echo "=== done ==="; date -u
} > "$LOG" 2>&1

echo "EXP0000_RUN_DONE"
