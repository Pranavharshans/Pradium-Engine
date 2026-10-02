#!/bin/bash
# Unattended measurement campaign for the Pradium patch-series candidate.
#
#   1. EXP-0001  candidate tree, direct attention OFF   (isolates patch 0001)
#   2. EXP-0002  direct attention ON (gate 4097)        (patch 0002 on top)
#   3. direct-vs-split numeric differential (3 prompt lengths)
#   4. apply patch 0003 to the tree (Python only, no rebuild) and run it with
#      EXL3_DECODE_BUFFER_REUSE=1 (patch 0003 on top of EXP-0002). EXP-0002 is
#      its parent, so the incremental effect is the same-build difference.
#   5. print the cross-round summary
#
# Every round writes into experiments/<EXP>/results/<tag>/ (committed evidence).
set -uo pipefail

TREE=${TREE:-/workspace/exp/builds/cand-series-0001-0002}
R=${REPO:-/workspace/exp/pradium}
LOG=${LOG:-/workspace/exp/full_campaign.log}
TOOLS=$R/experiments/tools

{
  echo "=== full campaign start ==="; date -u
  echo "tree: $TREE"
  cat "$TREE/PRADIUM_BUILD.json" 2>/dev/null
  echo

  bash "$TOOLS/run_rounds.sh" exp0001 "$TREE"
  bash "$TOOLS/run_rounds.sh" exp0002 "$TREE"
  bash "$TOOLS/run_rounds.sh" attn-diff "$TREE"

  echo; echo "=== applying patch 0003 to the candidate tree (Python only) ==="
  cd "$TREE" || exit 1
  GIT_CEILING_DIRECTORIES="$(dirname "$TREE")" \
    git apply "$R/engine/pradium/patches/0003-static-decode-buffers.patch" \
    && echo "patch 0003 applied" || echo "patch 0003 FAILED to apply"

  bash "$TOOLS/run_rounds.sh" exp0003-on "$TREE"

  echo; echo "=== cross-round summary ==="
  python "$TOOLS/summarize_rounds.py" \
    "exp0001=$R/experiments/EXP-0001-patch-series/results/EXP-0001-patch0001/session" \
    "exp0002=$R/experiments/EXP-0002-direct-attention/results/EXP-0002-direct/session" \
    "exp0003-on=$R/experiments/EXP-0003-static-buffers/results/EXP-0003-buffers-on/session"
  echo "=== full campaign done ==="; date -u
} > "$LOG" 2>&1

echo "CAMPAIGN_DONE"
