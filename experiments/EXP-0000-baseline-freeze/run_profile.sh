#!/bin/bash
# Run the decode-step decomposition profiler on the VM against whichever
# exllamav3 revision the named environment resolves to.
#
#   usage: run_profile.sh <env-name> <tag> [profile_decode.py args...]
set -uo pipefail

ENV_NAME=${1:?env name}
TAG=${2:?tag}
shift 2

REPO=${REPO:-/workspace/exp/pradium}
OUT=${OUT:-$REPO/experiments/EXP-0000-baseline-freeze/results}
MODEL_DIR=${MODEL_DIR:-/workspace/models/minicpm5-2b-exl3}

source "/workspace/envs/${ENV_NAME}/bin/activate"
export PYTHONPATH="${EXTRA_PYTHONPATH:+${EXTRA_PYTHONPATH}:}${REPO}"
export HF_HOME=/workspace/.hf_home
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"
cd "$REPO"

PATHS="$OUT/profile_${TAG}.json"
echo "COMMAND: profile_decode.py --label ${TAG} --json-out ${PATHS} $*"
python experiments/EXP-0000-baseline-freeze/profile_decode.py \
  --model-dir "$MODEL_DIR" \
  --label "$TAG" \
  --json-out "$PATHS" \
  "$@"
echo "PROFILE_EXIT=$?"
