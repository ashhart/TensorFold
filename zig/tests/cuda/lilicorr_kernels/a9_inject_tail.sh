#!/bin/bash
# A9 fail-before injector — TAIL control: the abandoned tail-warp load duty.

# Re-introduces the recorded A1 fuzz defect class (tensorflow/fuzz 11/25: "exiting them

# left whole Ain rows uninitialized") WITHOUT its host-side barrier violation: inactive

# tail warps (t >= hd when hd % 8 != 0) skip their shared-row load-work participation but

# stay for every __syncthreads. With hd=6, rows=9, the Ain columns of rows 6,7 at i0=0

# are filled ONLY by threads 192..255 (the tail warps), so those shared rows read zero,

# the factor outputs for exactly those rows fall off the closed form, and the TAIL

# control grades RED on the r in {6,7} band (its red-band-fails>0 line). Never edits the

# repo tree: the caller runs this against a COPY (see a9_run.sh). Usage: bash a9_inject_tail.sh <scratch-dir>
set -eu
D="${1:?usage: bash a9_inject_tail.sh <scratch-dir>}"
F="$D/qwen35_draft_lilicorr.cuh"
grep -q "A9-INJECTOR-TAIL" "$F" && { echo "already injected"; exit 1; }
grep -q "Ain\[r\]\[kk\] = (i0 + kk < hd) ? A_IN\[(size_t)r \* hd + i0 + kk\] : 0.f;" "$F" || { echo "cure line not found in $F — refusing"; exit 1; }
sed -i 's|Ain\[r\]\[kk\] = (i0 + kk < hd) ? A_IN\[(size_t)r \* hd + i0 + kk\] : 0.f;|Ain[r][kk] = (!active) ? 0.f : ((i0 + kk < hd) ? A_IN[(size_t)r * hd + i0 + kk] : 0.f);  // A9-INJECTOR-TAIL|' "$F"
grep -q "A9-INJECTOR-TAIL" "$F" || { echo "sed did not apply"; exit 1; }
echo "A9-INJECTOR-TAIL: applied (tail warps abandon the load duty)"
