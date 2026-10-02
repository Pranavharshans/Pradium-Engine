#!/bin/bash
# One candidate measurement round on the VM: core-matrix harness run, decode
# profiles for the nine configurations, and greedy-token correctness against
# the frozen official matrix session.
#
#   usage: run_candidate.sh <exp-dir> <tag> <tree|-> [env] [profiles]
#
# The freeze baseline session defaults to the campaign's official matrix
# session; override with FROZEN_SESSION.
set -uo pipefail

EXP=${1:?experiment dir, e.g. EXP-0001-patch-series}
TAG=${2:?tag}
TREE=${3:?-}
ENV_NAME=${4:-exllamav3}
PROFILES=${5:-SS,SM,SL,MS,MM,ML,LS,LM,LL}

REPO=${REPO:-/workspace/exp/pradium}
OUT=${OUT:-$REPO/experiments/$EXP/results/$TAG}
FROZEN_SESSION=${FROZEN_SESSION:-/workspace/pradium/benchmarks/campaigns/rtx3060-minicpm5-exl3-b0/sessions/20261002-090349_exllamav3_official_537aee}
PYPREFIX=""
[ "$TREE" != "-" ] && PYPREFIX="$TREE"

mkdir -p "$OUT"
LOG=$OUT/round.log

{
  echo "=== candidate round: $TAG ($EXP) ==="; date -u
  echo "TREE=$TREE ENV=$ENV_NAME PROFILES=$PROFILES"
  if [ "$TREE" != "-" ] && [ -f "$TREE/PRADIUM_BUILD.json" ]; then
    echo "--- PRADIUM_BUILD.json ---"
    cat "$TREE/PRADIUM_BUILD.json"
  fi

  echo; echo "--- matrix (core workloads: $PROFILES) ---"
  PY_PREFIX="$PYPREFIX" "$REPO/experiments/tools/run_matrix.sh" "$ENV_NAME" "$OUT/session" "$PROFILES" "$PYPREFIX"

  SESSION=$(ls -td "$OUT/session"/*/ 2>/dev/null | head -1)
  echo "SESSION=$SESSION"

  echo; echo "--- decode profiles ---"
  if [ "${SKIP_PROFILES:-0}" = "1" ]; then
    echo "SKIP_PROFILES=1: matrix-only round (decision metric); profiles run separately"
  fi
  while read -r name pin pout; do
    [ "${SKIP_PROFILES:-0}" = "1" ] && break
    [ -z "$name" ] && continue
    echo; echo "--- profile $name (${pin}/${pout}) ---"
    EXTRA_PYTHONPATH="$PYPREFIX" OUT="$OUT" \
      "$REPO/experiments/EXP-0000-baseline-freeze/run_profile.sh" "$ENV_NAME" "$TAG-$name" \
      --prompt-tokens "$pin" --max-new-tokens "$pout" --profile-steps 24 --profile-after-tokens 8
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

  echo; echo "--- correctness: greedy token IDs vs frozen official matrix session ---"
  if [ -d "$FROZEN_SESSION" ] && [ -n "$SESSION" ]; then
    python "$REPO/experiments/tools/compare_tokens.py" "$FROZEN_SESSION" "$SESSION" \
      --json-out "$OUT/correctness.json"
    echo "CORRECTNESS_EXIT=$?"
  else
    echo "FROZEN_SESSION or session dir missing; correctness NOT RUN"
  fi

  echo; echo "--- decode summary ---"
  python "$REPO/experiments/EXP-0000-baseline-freeze/summarize_profiles.py" "$OUT/profile_$TAG-*.json" \
    | tee "$OUT/summary_table.md"
  echo; echo "=== round done ==="; date -u
} > "$LOG" 2>&1

echo "ROUND_DONE $TAG"
