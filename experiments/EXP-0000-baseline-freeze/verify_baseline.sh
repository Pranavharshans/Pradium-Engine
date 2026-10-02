#!/bin/bash
# Re-run the frozen official campaign configuration on selected core-matrix
# profiles to confirm the baseline is reproducible on the VM. Writes a fresh
# session under the experiment's results directory; the frozen campaign
# directory is never written to.
#
#   usage: verify_baseline.sh [profiles] e.g. verify_baseline.sh SS,MM,LL
set -uo pipefail

PROFILES=${1:-SS,MM,LL}

REPO=${REPO:-/workspace/exp/pradium}
OUT=${OUT:-$REPO/experiments/EXP-0000-baseline-freeze/results/baseline-verify}
ENV_NAME=${ENV_NAME:-exllamav3}
MODEL_DIR=${MODEL_DIR:-/workspace/models/minicpm5-2b-exl3}

source "/workspace/envs/${ENV_NAME}/bin/activate"
export PYTHONPATH="$REPO"
export HF_HOME=/workspace/.hf_home
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REV=e36fbb568f466739d7c244a5100d01acf93d3b7f
TOK="hf:ewin-reg/MiniCPM5-2B-EXL3-Quantized@${REV}"
RC="{\"model_dir\":\"${MODEL_DIR}\",\"max_num_tokens\":32768,\"max_batch_size\":16,\"max_chunk_size\":2048,\"model\":\"ewin-reg/MiniCPM5-2B-EXL3-Quantized\",\"model_revision\":\"${REV}\"}"

mkdir -p "$OUT"
cd "$REPO"

echo "COMMAND: python -m benchmarks run matrix --profiles ${PROFILES} (env ${ENV_NAME})"
python -m benchmarks run matrix \
  --runtime exllamav3 \
  --tokenizer "$TOK" \
  --model ewin-reg/MiniCPM5-2B-EXL3-Quantized \
  --model-revision "$REV" \
  --quantization EXL3 \
  --bpw 4.0 \
  --profiles "$PROFILES" \
  --results-dir "$OUT" \
  --runtime-config "$RC"
echo "SUITE_EXIT=$?"
