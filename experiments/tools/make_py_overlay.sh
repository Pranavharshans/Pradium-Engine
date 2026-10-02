#!/bin/bash
# Build a Python-only overlay of the installed exllamav3 package with the
# Pradium Python patches applied, so Python-level experiments can be measured
# against the official wheel without rebuilding the CUDA extension.
#
#   usage: make_py_overlay.sh <overlay-dir> <patch> [patch...]
#
# The overlay directory is prepended to PYTHONPATH; `exllamav3` resolves to the
# overlay while `exllamav3_ext` still resolves to the wheel's compiled module
# (ext.py falls back to the installed extension when it is importable).
set -euo pipefail

DEST=${1:?overlay dir}
shift
PATCHES=("$@")
ENV_BIN=${ENV_BIN:-/workspace/envs/exllamav3}
SP=$("$ENV_BIN/bin/python" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")

rm -rf "$DEST"
mkdir -p "$DEST"
cp -a "$SP/exllamav3" "$DEST/"
cd "$DEST"
for patch in "${PATCHES[@]}"; do
  echo "applying $patch"
  GIT_CEILING_DIRECTORIES="$(dirname "$DEST")" git apply "$(realpath "$patch")"
done
"$ENV_BIN/bin/python" -c "
import importlib.machinery, importlib.util, os, sys
sys.path.insert(0, os.getcwd())
spec = importlib.util.find_spec('exllamav3_ext')
print('ext:', spec.origin if spec else None)
"
echo "overlay ready: $DEST"
