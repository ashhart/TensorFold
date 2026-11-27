#!/bin/bash
# A9 fail-before injector — CHAIN control: the S4-era l2norm tail width.

# Re-introduces the recorded S4 normalization defect verbatim (the kernel's own comment:

# "the S4-era kernel used `stride` here and over-ran the tail's hd elements into whatever

# sat behind it — the tail's inv carried their stale ssq — a real normalization defect"):

# the anchor-tail block normalizes tail_width=2048 floats instead of the tail's 1024 in

# the flat chain face, so the anchor row carries the stale-ssq class and every start

# score is wrong. The buffer carries two guard rows, so the over-run stays in-bounds and

# deterministic — a wrong-answer RED, never a crash (the pre-fix face is graded on

# values, not on allocation accidents). Never edits the repo tree: the caller runs this against a COPY (see a9_run.sh).

# Usage: bash a9_inject_chain.sh <scratch-dir>
set -eu
D="${1:?usage: bash a9_inject_chain.sh <scratch-dir>}"
F="$D/qwen35_draft_lilicorr.cuh"
grep -q "A9-INJECTOR-CHAIN" "$F" && { echo "already injected"; exit 1; }
grep -q "const int width = edge_row ? stride / 2 : tail_width;" "$F" || { echo "cure line not found in $F — refusing"; exit 1; }
sed -i 's|const int width = edge_row ? stride / 2 : tail_width;|const int width = edge_row ? stride / 2 : stride;  // A9-INJECTOR-CHAIN|' "$F"
grep -q "A9-INJECTOR-CHAIN" "$F" || { echo "sed did not apply"; exit 1; }
echo "A9-INJECTOR-CHAIN: applied (S4-era tail width)"
