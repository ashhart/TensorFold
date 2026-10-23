"""Teacher-forced comparison of two sets of logits (reference against engine): top-1, log-prob gap, KL, margins."""
# A near tie: a top-1 disagreement whose margin is at most 0.25 on both sides; KL is over the full vocabulary.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def logsoftmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ref", type=Path)
    p.add_argument("test", type=Path)
    p.add_argument("--only")
    p.add_argument("--json", type=Path, help="write the totals here")
    args = p.parse_args()
    only = set(args.only.split(",")) if args.only else None
    tot = {"n": 0, "top1": 0, "tie": 0, "dlp": [], "kl": []}
    print(f"{'prompt':16s} {'pos':>4s} {'top1':>7s} {'dlp mean':>9s} {'dlp p95':>8s} {'dlp max':>8s} {'KL mean':>9s} {'KL max':>8s}")
    for f in sorted(args.ref.glob("*.json")):
        meta = json.loads(f.read_text())
        if not isinstance(meta, dict) or "name" not in meta:
            continue
        name = meta["name"]
        if "prompt_ids" not in meta or (only and name not in only) or not (args.test / f"{name}.logits.npy").exists():
            continue
        a = np.load(args.ref / f"{name}.logits.npy")
        b = np.load(args.test / f"{name}.logits.npy")
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        gen = np.array(meta["generated_ids"][:n])
        la, lb = logsoftmax(a), logsoftmax(b)
        ta, tb = a.argmax(-1), b.argmax(-1)
        same = ta == tb
        dlp = np.abs(la[np.arange(n), gen] - lb[np.arange(n), gen])
        kl = (np.exp(la) * (la - lb)).sum(-1)
        ties = 0
        for i in np.flatnonzero(~same):
            sa, sb = np.sort(a[i])[-2:], np.sort(b[i])[-2:]
            if sa[1] - sa[0] <= 0.25 and sb[1] - sb[0] <= 0.25:
                ties += 1
        print(f"{name:16s} {n:4d} {same.sum():3d}/{n:<3d} {dlp.mean():9.4f} {np.percentile(dlp, 95):8.4f} {dlp.max():8.4f} {kl.mean():9.5f} {kl.max():8.4f}  misses {int((~same).sum())} (near ties {ties})")
        tot["n"] += n
        tot["top1"] += int(same.sum())
        tot["tie"] += ties
        tot["dlp"].append(dlp)
        tot["kl"].append(kl)
    if not tot["n"]:
        raise SystemExit("nothing compared")
    dlp, kl = np.concatenate(tot["dlp"]), np.concatenate(tot["kl"])
    res = {"positions": tot["n"], "top1": tot["top1"] / tot["n"], "misses": tot["n"] - tot["top1"], "near_ties": tot["tie"], "dlp_mean": float(dlp.mean()), "dlp_p95": float(np.percentile(dlp, 95)), "dlp_max": float(dlp.max()), "kl_mean": float(kl.mean()), "kl_max": float(kl.max())}
    print("TOTAL " + json.dumps(res))
    if args.json:
        args.json.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
