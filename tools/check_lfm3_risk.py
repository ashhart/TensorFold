#!/usr/bin/env python3
"""Check the compiled Metal predicate source against the original bit predicates."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'zig/kernels/metal/lfm3/attention.metal').read_text()
start = source.index('struct l3_risk_factor')
end = source.index('inline void l3_x_merge', start)
with tempfile.TemporaryDirectory(prefix='lfm3-risk-proof-') as folder:
    work = Path(folder)
    (work / 'predicate.inc').write_text(source[start:end])
    (work / 'proof.cpp').write_text((root / 'tools/check_lfm3_risk.cpp').read_text())
    subprocess.run(['clang++', '-O2', '-std=c++17', str(work / 'proof.cpp'), '-o', str(work / 'proof')], check=True)
    subprocess.run([str(work / 'proof')], check=True)
