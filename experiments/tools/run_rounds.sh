#!/bin/bash
# Runs one full measurement round for the patch-series candidate, or a subset.
#
#   usage: run_rounds.sh <round> [tree]
#     round = exp0001 | exp0002 | exp0003-on | exp0003-off | attn-diff | all
#
# Every round writes into experiments/<EXP dir>/results/<tag>/ and appends to
# the round log; the tree revision, PRADIUM_BUILD.json and EXL3_* env vars are
# captured in that log.
set -uo pipefail

ROUND=${1:?round name}
TREE=${2:-/workspace/exp/builds/cand-series-0001-0002}
REPO=${REPO:-/workspace/exp/pradium}
BASE=$REPO/experiments
PROFILES=${PROFILES:-SS,SM,SL,MS,MM,ML,LS,LM,LL}

run_round() {
  local expdir=$1 tag=$2 treearg=$3
  echo "### round $tag ($expdir) tree=$treearg"
  "$BASE/tools/run_candidate.sh" "$expdir" "$tag" "$treearg"
}

case "$ROUND" in
  exp0001)
    # patch series built; direct attention OFF -> isolates patch 0001.
    # Patch 0001 only removes host bookkeeping inside an existing graph-replay path,
    # so the round uses SS/MM/LL (documented) plus the SS step profile; the full
    # nine-workload scope is used for EXP-0002 and EXP-0003.
    PROFILES=SS,MM,LL EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=0 \
      run_round EXP-0001-patch-series EXP-0001-patch0001 "$TREE"
    # one decode-step profile at SS: build-equivalence check against the wheel
    OUT=$BASE/EXP-0001-patch-series/results/EXP-0001-patch0001 \
    EXTRA_PYTHONPATH="$TREE" EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=0 \
      "$BASE/EXP-0000-baseline-freeze/run_profile.sh" exllamav3 exp0001-SS \
      --prompt-tokens 128 --max-new-tokens 64 --profile-steps 24
    ;;
  exp0002)
    # direct attention enabled for contexts <= 4096 tokens
    EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=4097 \
      run_round EXP-0002-direct-attention EXP-0002-direct "$TREE"
    OUT=$BASE/EXP-0002-direct-attention/results/EXP-0002-direct \
    EXTRA_PYTHONPATH="$TREE" EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=4097 \
      "$BASE/EXP-0000-baseline-freeze/run_profile.sh" exllamav3 exp0002-SS \
      --prompt-tokens 128 --max-new-tokens 64 --profile-steps 24
    ;;
  exp0003-on)
    EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=4097 EXL3_DECODE_BUFFER_REUSE=1 \
      run_round EXP-0003-static-buffers EXP-0003-buffers-on "$TREE"
    ;;
  exp0003-off)
    EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=4097 EXL3_DECODE_BUFFER_REUSE=0 \
      run_round EXP-0003-static-buffers EXP-0003-buffers-off "$TREE"
    ;;
  attn-diff)
    OUT=$BASE/EXP-0002-direct-attention/results/attn-diff
    mkdir -p "$OUT"
    for cfg in "128 16" "1024 16" "4096 16"; do
      set -- $cfg
      echo "### attn differential prompt=$1 new=$2"
      EXTRA_PYTHONPATH="$TREE" python "$BASE/tools/attn_path_compare.py" \
        --model-dir /workspace/models/minicpm5-2b-exl3 \
        --prompt-tokens "$1" --max-new-tokens "$2" \
        --json-out "$OUT/attn_diff_$1.json" 2>&1 | tail -25
    done
    ;;
  all)
    bash "$0" exp0001 "$TREE"
    bash "$0" exp0002 "$TREE"
    bash "$0" attn-diff "$TREE"
    ;;
  *)
    echo "unknown round: $ROUND" >&2
    exit 2
    ;;
esac
echo "ROUNDS_DONE $ROUND"
