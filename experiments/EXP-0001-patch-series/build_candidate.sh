#!/bin/bash
# Materialize the Pradium patch series onto the pinned upstream revision and
# build the CUDA extension in place (VM only). The tree is then usable by
# prepending it to PYTHONPATH; no environment is modified.
#
#   usage: build_candidate.sh <tree-dir>
#
# Rebuilds are incremental: only changed sources recompile.
set -euo pipefail

DEST=${1:?output tree}
REPO=${REPO:-/workspace/exp/pradium}
ENV_BIN=${ENV_BIN:-/workspace/envs/exllamav3}

mkdir -p "$(dirname "$DEST")"
cd "$REPO"

if [ ! -f "$DEST/PRADIUM_BUILD.json" ]; then
  python engine/pradium/build.py --output "$DEST"
fi

source "$ENV_BIN/bin/activate"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-3}"

cd "$DEST"
python setup.py build_ext --inplace
ls -la exllamav3_ext*.so
python - <<'EOF'
import importlib.machinery, importlib.util, json, os, sys
root = os.getcwd()
sys.path.insert(0, root)
spec = importlib.util.find_spec("exllamav3_ext")
print("ext:", spec.origin if spec else None)
EOF
echo "BUILD_DONE"
