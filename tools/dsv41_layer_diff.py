"""Layer-by-layer difference between two stream dumps (reference vs vLLM), per layer: relative error and cosine.

    python tools/dsv41_layer_diff.py out/refdump out/vllmdump
"""

import sys
from pathlib import Path

import torch

a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
for pa in sorted(a_dir.glob("layer*.pt")):
    pb = b_dir / pa.name
    if not pb.exists():
        continue
    a, b = torch.load(pa), torch.load(pb)
    row = [pa.stem]
    for key in ("stream", "ffn_out", "pre"):
        x, y = a[key].float().reshape(a[key].shape[0], -1), b[key].float().reshape(b[key].shape[0], -1)
        rel = (x - y).norm(dim=-1) / y.norm(dim=-1).clamp(min=1e-9)
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        q = rel.shape[0]
        row.append(f"{key}: rel mean {rel.mean():.4f} [first16 {rel[:16].mean():.4f} last64 {rel[-64:].mean():.4f}] "
                   f"cos min {cos.min():.4f}")
    print("  ".join(row))
