"""Compare every tensor of matching dump files (reference vs vLLM): relative error overall, first 16 rows, row 0.

    python tools/dsv41_dump_diff.py out/refdump out/vllmdump [name-glob]
"""

import sys
from pathlib import Path

import torch

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
pattern = sys.argv[3] if len(sys.argv) > 3 else "*.pt"
for pa in sorted(a_dir.glob(pattern)):
    pb = b_dir / pa.name
    if not pb.exists():
        continue
    a, b = torch.load(pa), torch.load(pb)
    for key in a:
        if key not in b:
            continue
        x, y = a[key], b[key]
        if x.shape[0] != y.shape[0]:
            print(f"{pa.stem}.{key}: rows {tuple(x.shape)} vs {tuple(y.shape)}")
            continue
        x, y = x.float().reshape(x.shape[0], -1), y.float().reshape(y.shape[0], -1)
        if x.shape != y.shape:
            print(f"{pa.stem}.{key}: shape {tuple(x.shape)} vs {tuple(y.shape)}")
            continue
        if key == "hashes":
            print(f"{pa.stem}.{key}: equal {bool((x == y).all())}, rows equal {int((x == y).all(1).sum())}/{x.shape[0]}")
            continue
        rel = (x - y).norm(dim=-1) / y.norm(dim=-1).clamp(min=1e-9)
        print(f"{pa.stem}.{key:9s} rel mean {rel.mean():.4f} first16 {rel[:16].mean():.4f} row0 {rel[0]:.4f}  "
              f"|x| {x.norm(dim=-1).mean():.2f} |y| {y.norm(dim=-1).mean():.2f}")
