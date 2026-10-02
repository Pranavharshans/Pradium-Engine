#!/bin/bash
# Instrument-cost diagnostic: harness measurement (telemetry + adapter, the
# frozen configuration) vs a direct engine-driven measurement (no telemetry,
# no adapter, no profiler) on the same workloads, same session, same VM.
#
# The difference isolates the benchmark harness's own per-token cost from the
# engine's decode time. It does not change the frozen baseline numbers; it
# explains them.
set -uo pipefail

REPO=${REPO:-/workspace/exp/pradium}
OUT=${OUT:-/workspace/exp/diag}
PROFILES=${1:-SS,LL}

mkdir -p "$OUT"
EV=$REPO/experiments/EXP-0000-baseline-freeze

echo "=== harness ($PROFILES, telemetry on) ==="
"$REPO/experiments/tools/run_matrix.sh" exllamav3 "$OUT/telemetry_on" "$PROFILES"

for cfg in "SS 128 64" "LL 4096 1024"; do
  set -- $cfg
  echo "=== direct $1 (prompt=$2 new=$3, no telemetry/adapter/profiler) ==="
  python "$EV/profile_decode.py" \
    --model-dir /workspace/models/minicpm5-2b-exl3 \
    --label "clean-$1" --prompt-tokens "$2" --max-new-tokens "$3" \
    --profile-steps 0 --json-out "$OUT/clean_$1.json"
done
echo "DIAG_DONE"
