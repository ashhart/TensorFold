"""The served CUDA engine's teacher-forced logits against the release's: KL, top-1 agreement and every miss."""

import argparse
import json
import os

USAGE = """
The release's logits come from minimax_m3_ref.py --logits (logits.npy, bf16 bits [rows][vocab]); the served engine's
from tf-minimax-cuda's every-row mode on the same ids (logits.r{rank}.bf16 and .json, each rank's vocabulary share).
Row r is the distribution of token r + 1 given tokens 0..r. Per row: KL(release || served) in float64 over the whole
vocabulary, both greedy picks and, where they differ, each side's top contenders with logprobs and the margins.

  python tools/zig/minimax_m3_kl.py REF_DIR ENGINE_DIR --ids ids.json [--tokenizer HF_DIR] [--out report.json]
"""

import numpy as np


def bf16_bits_to_f32(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def engine_logits(folder: str) -> np.ndarray:
    """Every rank's share joined along the vocabulary, as bf16 bits [rows][vocab]."""
    parts = []
    for name in sorted(os.listdir(folder)):
        if name.startswith("logits.r") and name.endswith(".json"):
            meta = json.load(open(os.path.join(folder, name)))
            raw = np.fromfile(os.path.join(folder, name[:-5] + ".bf16"), dtype=np.uint16)
            parts.append((meta["vocab_lo"], raw.reshape(meta["rows"], meta["vocab_hi"] - meta["vocab_lo"])))
    parts.sort(key=lambda p: p[0])
    at = 0
    for lo, p in parts:
        if lo != at:
            raise SystemExit(f"the ranks' vocabulary shares leave a gap at {at}")
        at += p.shape[1]
    return np.concatenate([p for _, p in parts], axis=1)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir")
    ap.add_argument("engine_dir")
    ap.add_argument("--ids", required=True)
    ap.add_argument("--tokenizer", default="", help="a folder with tokenizer.json, to print the contenders as text")
    ap.add_argument("--top", type=int, default=5, help="contenders listed for a miss")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--sparse-from", type=int, default=2048, help="rows from here also reported alone (the indexer selects)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ids = json.load(open(a.ids))
    ids = ids["ids"] if isinstance(ids, dict) else ids
    ref = np.load(os.path.join(a.ref_dir, "logits.npy"), mmap_mode="r")
    eng = engine_logits(a.engine_dir)
    if ref.shape != eng.shape or ref.shape[0] != len(ids):
        raise SystemExit(f"shapes differ: release {ref.shape}, served {eng.shape}, {len(ids)} ids")
    tok = None
    if a.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
    show = (lambda t: tok.decode([int(t)])) if tok else (lambda t: str(int(t)))
    rows = ref.shape[0]
    kl = np.zeros(rows)
    agree = np.zeros(rows, dtype=bool)
    misses = []
    for lo in range(0, rows, a.chunk):
        hi = min(rows, lo + a.chunk)
        lp = log_softmax(bf16_bits_to_f32(np.asarray(ref[lo:hi])))
        lq = log_softmax(bf16_bits_to_f32(eng[lo:hi]))
        kl[lo:hi] = (np.exp(lp) * (lp - lq)).sum(-1)
        for j in range(hi - lo):
            r = lo + j
            rt = np.argsort(-lp[j])[: a.top]
            qt = np.argsort(-lq[j])[: a.top]
            agree[r] = rt[0] == qt[0]
            if agree[r]:
                continue
            misses.append({
                "row": r,
                "next_token": show(ids[r + 1]) if r + 1 < len(ids) else None,
                "release": [[show(t), round(float(lp[j][t]), 4)] for t in rt],
                "served": [[show(t), round(float(lq[j][t]), 4)] for t in qt],
                "release_margin": round(float(lp[j][rt[0]] - lp[j][rt[1]]), 4),
                "served_margin": round(float(lq[j][qt[0]] - lq[j][qt[1]]), 4),
                "kl": round(float(kl[r]), 5),
            })
    rep = {
        "rows": rows,
        "kl_mean": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_max": float(kl.max()),
        "top1_agreement": float(agree.mean()),
        "sparse_rows": int(max(0, rows - a.sparse_from)),
        "sparse_kl_mean": float(kl[a.sparse_from:].mean()) if rows > a.sparse_from else None,
        "sparse_top1_agreement": float(agree[a.sparse_from:].mean()) if rows > a.sparse_from else None,
        "kl_rows": [round(float(x), 6) for x in kl],
        "misses": misses,
    }
    print(f"{rows} rows: KL(release || served) mean {rep['kl_mean']:.5f}, median {rep['kl_median']:.5f}, "
          f"p99 {rep['kl_p99']:.4f}, max {rep['kl_max']:.4f}; top-1 agreement {100 * rep['top1_agreement']:.2f}% "
          f"({len(misses)} misses)")
    if rep["sparse_rows"]:
        print(f"rows from {a.sparse_from} ({rep['sparse_rows']}, sparse selection): KL mean {rep['sparse_kl_mean']:.5f}, "
              f"top-1 {100 * rep['sparse_top1_agreement']:.2f}%")
    for m in misses:
        print(f"row {m['row']}: release {m['release'][:3]} (margin {m['release_margin']}), "
              f"served {m['served'][:3]} (margin {m['served_margin']}), KL {m['kl']}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
