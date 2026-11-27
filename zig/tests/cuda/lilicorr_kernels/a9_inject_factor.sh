#!/bin/bash
# A9 fail-before injector — FACTOR control: the cross-term W-slab fold.
# Re-introduces the slab mis-index class in the scratch copy of the kernel header:
# the cross term w_cross points at the anchor-term slab (w_row + hd) instead of its own
# (w_row + 2*hd). The device factor output then mis-folds every row and the FACTOR
# control's host mirror grades RED. Never edits the repo tree: the caller runs this
# against a COPY (see a9_run.sh).
# Usage: bash a9_inject_factor.sh <scratch-dir>
set -eu
D="${1:?usage: bash a9_inject_factor.sh <scratch-dir>}"
F="$D/qwen35_draft_lilicorr.cuh"
grep -q "A9-INJECTOR-FACTOR" "$F" && { echo "already injected"; exit 1; }
grep -q "const float\* w_cross = w_row + 2 \* hd;" "$F" || { echo "cure line not found in $F — refusing"; exit 1; }
sed -i 's|const float\* w_cross = w_row + 2 \* hd;|const float* w_cross = w_row + hd;  // A9-INJECTOR-FACTOR: slab fold (defenseless mis-index)|' "$F"
grep -q "A9-INJECTOR-FACTOR" "$F" || { echo "sed did not apply"; exit 1; }
echo "A9-INJECTOR-FACTOR: applied (cross term folded onto the w2 slab)"
