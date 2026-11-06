"""Compare tf-minimax-cuda's dumped rows and logits with minimax_m3_ref.py's, layer by layer."""

import argparse
import os

USAGE = """
The CUDA family's dumps (tf-minimax-cuda: r{i}.f32 or r{i}.c{k}.f32, logits.bf16) against the reference's (r{i}.npy as
bf16 bits, logits.npy): per layer the largest and mean absolute difference, the share of rows within bf16 noise, the
worst row's relative L2; for the logits the greedy ids and the top-5 of the last row.

  python tools/zig/minimax_m3_compare.py REF_DIR CUDA_DIR [--rows N]
"""

import numpy as np


def bf16_bits_to_f32(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("cuda")
    ap.add_argument("--rows", type=int, default=0, help="compare only the first N rows")
    a = ap.parse_args()
    i = 0
    worst = 0.0
    def cuda_rows(i):  # r{i}.c{k}.f32 chunk by chunk (tf-minimax-cuda), or one r{i}.f32
        one = os.path.join(a.cuda, f"r{i}.f32")
        if os.path.exists(one):
            return np.fromfile(one, dtype=np.float32)
        parts, k = [], 0
        while os.path.exists(os.path.join(a.cuda, f"r{i}.c{k}.f32")):
            parts.append(np.fromfile(os.path.join(a.cuda, f"r{i}.c{k}.f32"), dtype=np.float32))
            k += 1
        return np.concatenate(parts) if parts else None
    while os.path.exists(os.path.join(a.ref, f"r{i}.npy")) and cuda_rows(i) is not None:
        ref = bf16_bits_to_f32(np.load(os.path.join(a.ref, f"r{i}.npy")).view(np.uint16))
        got = cuda_rows(i).reshape(ref.shape)
        if a.rows:
            ref, got = ref[: a.rows], got[: a.rows]
        diff = np.abs(ref - got)
        rel = np.linalg.norm(ref - got, axis=-1) / np.maximum(np.linalg.norm(ref, axis=-1), 1e-30)
        exact = float((diff == 0).all(axis=-1).mean())
        print(f"layer {i:2d}: max |d| {diff.max():.4g} mean {diff.mean():.3g}  rows exact {exact:.3f}  worst row rel L2 {rel.max():.4g} (row {int(rel.argmax())}) "
              f"ref |x| max {np.abs(ref).max():.3g}")
        worst = max(worst, float(rel.max()))
        i += 1
    rl, cl = os.path.join(a.ref, "logits.npy"), os.path.join(a.cuda, "logits.bf16")
    if os.path.exists(rl) and os.path.exists(cl):
        ref = bf16_bits_to_f32(np.load(rl).view(np.uint16))[-1]
        got = bf16_bits_to_f32(np.fromfile(cl, dtype=np.uint16))
        print(f"logits: max |d| {np.abs(ref - got).max():.4g}; greedy ref {int(ref.argmax())} cuda {int(got.argmax())}")
        print(" ref top5", np.argsort(-ref)[:5].tolist(), " cuda top5", np.argsort(-got)[:5].tolist())
    print(f"{i} layers compared, worst row rel L2 {worst:.4g}")


if __name__ == "__main__":
    main()
