#!/bin/bash
# Run the frozen-configuration core-matrix harness against an engine revision.
#
#   usage: run_matrix.sh <env-name> <results-dir> <profiles> [pythonpath-prefix]
#
# pythonpath-prefix, when given, is prepended to PYTHONPATH so that a candidate
# exllamav3 tree shadows the environment's installed build.
set -uo pipefail

ENV_NAME=${1:?env name}
RESULTS_DIR=${2:?results dir}
PROFILES=${3:?profiles, e.g. SS,SM,SL,MS,MM,ML,LS,LM,LL or a subset}
PY_PREFIX=${4:-}

REPO=${REPO:-/workspace/exp/pradium}
MODEL_DIR=${MODEL_DIR:-/workspace/models/minicpm5-2b-exl3}

source "/workspace/envs/${ENV_NAME}/bin/activate"
export PYTHONPATH="${PY_PREFIX:+${PY_PREFIX}:}${REPO}"
export HF_HOME=/workspace/.hf_home
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REV=e36fbb568f466739d7c244a5100d01acf93d3b7f
TOK="hf:ewin-reg/MiniCPM5-2B-EXL3-Quantized@${REV}"
RC="{\"model_dir\":\"${MODEL_DIR}\",\"max_num_tokens\":32768,\"max_batch_size\":16,\"max_chunk_size\":2048,\"model\":\"ewin-reg/MiniCPM5-2B-EXL3-Quantized\",\"model_revision\":\"${REV}\"}"

mkdir -p "$RESULTS_DIR"
cd "$REPO"

echo "COMMAND: run_matrix env=${ENV_NAME} pythonpath_prefix=${PY_PREFIX:-<none>} profiles=${PROFILES}"
env | grep -E '^(EXL3_|EXLLAMA_)' | sort | sed 's/^/ENV: /' || true
python -m benchmarks run matrix \
  --runtime exllamav3 \
  --tokenizer "$TOK" \
  --model ewin-reg/MiniCPM5-2B-EXL3-Quantized \
  --model-revision "$REV" \
  --quantization EXL3 \
  --bpw 4.0 \
  --profiles "$PROFILES" \
  --results-dir "$RESULTS_DIR" \
  --runtime-config "$RC"
echo "SUITE_EXIT=$?"
