#!/bin/bash
# Wheel-based A/B measurement of patch 0003 (Python-only) via the overlay:
# same process content, toggle EXL3_DECODE_BUFFER_REUSE 0 vs 1, identical
# frozen configuration. Supplementary to the tree-based EXP-0003 round; run
# under whatever load the VM has and label the evidence accordingly.
set -uo pipefail

R=${REPO:-/workspace/exp/pradium}
OUT=$R/experiments/EXP-0003-static-buffers/results
OVERLAY=${OVERLAY:-/workspace/exp/overlay-wheel-0003}
PROFILES=${PROFILES:-SS,MM,LL}

source /workspace/envs/exllamav3/bin/activate
export PYTHONPATH="$OVERLAY:$R"
export HF_HOME=/workspace/.hf_home
export TORCH_CUDA_ARCH_LIST=8.6
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REV=e36fbb568f466739d7c244a5100d01acf93d3b7f
RC="{\"model_dir\":\"/workspace/models/minicpm5-2b-exl3\",\"max_num_tokens\":32768,\"max_batch_size\":16,\"max_chunk_size\":2048,\"model\":\"ewin-reg/MiniCPM5-2B-EXL3-Quantized\",\"model_revision\":\"${REV}\"}"
cd "$R"

python -c "import exllamav3, sys; print('using', exllamav3.__file__)"

for mode in 0 1; do
  echo "=== EXL3_DECODE_BUFFER_REUSE=$mode ==="
  EXL3_DECODE_BUFFER_REUSE=$mode python -m benchmarks run matrix \
    --runtime exllamav3 \
    --tokenizer "hf:ewin-reg/MiniCPM5-2B-EXL3-Quantized@${REV}" \
    --model ewin-reg/MiniCPM5-2B-EXL3-Quantized \
    --model-revision "$REV" \
    --quantization EXL3 \
    --bpw 4.0 \
    --profiles "$PROFILES" \
    --results-dir "$OUT/wheel-buffers-$mode" \
    --runtime-config "$RC"
  echo "EXIT=$?"
done
echo "WHEEL_AB_DONE"
