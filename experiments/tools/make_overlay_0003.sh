#!/bin/bash
# Build a Python-only overlay of the installed (wheel) exllamav3 package with
# patch 0003 applied, for a wheel-based measurement of the Python-only change.
#
#   usage: make_overlay_0003.sh <overlay-dir> [patch-0003]
#
# patch 0003 was generated against the pin + patches 0001/0002, so its
# `bc_attn.py` hunk carries context that only exists after patch 0002. The hunk
# applied here is the same two-line change (an `out` parameter for the step
# output and the reuse check), applied directly to the wheel's file.
set -euo pipefail

DEST=${1:?overlay dir}
PATCH=${2:-/workspace/exp/pradium/engine/pradium/patches/0003-static-decode-buffers.patch}
ENV_BIN=${ENV_BIN:-/workspace/envs/exllamav3}
SP=$("$ENV_BIN/bin/python" -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")

DEST=$(mkdir -p "$DEST" && cd "$DEST" && pwd)
rm -rf "$DEST/exllamav3"
cp -a "$SP/exllamav3" "$DEST/"

cd "$DEST"
GIT_CEILING_DIRECTORIES="$DEST" git apply -p1 \
  --include='exllamav3/util/decode_buffers.py' \
  --include='exllamav3/modules/rmsnorm.py' \
  --include='exllamav3/modules/mlp.py' \
  --include='exllamav3/modules/attn.py' \
  --include='exllamav3/modules/transformer.py' \
  "$PATCH"

"$ENV_BIN/bin/python" - "$DEST" <<'PY'
import pathlib, sys

root = pathlib.Path(sys.argv[1])
path = root / "exllamav3/modules/attention_fn/bc_attn.py"
text = path.read_text()

old_sig = """        causal: bool = True,
        host_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:"""
new_sig = """        causal: bool = True,
        host_seqlens: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:"""
assert text.count(old_sig) == 1, "step() signature anchor not found"
text = text.replace(old_sig, new_sig)

old_alloc = """        y = torch.empty((bsz, q_len, self.hidden_size), dtype = self.o_dtype, device = x.device)"""
new_alloc = """        if not _db.reuse(out, (bsz, q_len, self.hidden_size), self.o_dtype):
            out = None
        y = out if out is not None else torch.empty((bsz, q_len, self.hidden_size), dtype = self.o_dtype, device = x.device)"""
assert text.count(old_alloc) == 1, "step() allocation anchor not found"
text = text.replace(old_alloc, new_alloc)

old_imp = """from ...util.tensor import g_tensor_cache"""
new_imp = """from ...util import decode_buffers as _db
from ...util.tensor import g_tensor_cache"""
assert text.count(old_imp) == 1, "bc_attn import anchor not found"
text = text.replace(old_imp, new_imp)

path.write_text(text)
print("bc_attn.py patched for the wheel base")
PY

"$ENV_BIN/bin/python" -c "
import importlib.util, os, sys
sys.path.insert(0, os.environ.get('OVERLAY') or '$DEST')
import importlib.machinery
spec = importlib.util.find_spec('exllamav3_ext')
print('ext:', spec.origin if spec else None)
"
echo "overlay ready: $DEST"
