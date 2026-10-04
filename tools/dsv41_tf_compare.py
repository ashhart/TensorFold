"""Compare teacher-forced runs of tools/dsv41_serial_run.py --tf-compare (one per KV format).

    python tools/dsv41_tf_compare.py REF.pt RUN.pt [RUN.pt ...]

Per document and per length: top-1 agreement of each run with REF, mean NLL of both, mean |dNLL| of the actual
next token, and whether the actual token stays the top-1 (accuracy) in each.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import torch


def knobs(run: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(run["env"].items()) if v and k != "TF_DSV41_KV_FP8")


def main() -> None:
    ref_path, *runs = sys.argv[1:]
    ref = torch.load(ref_path)
    for path in runs:
        run = torch.load(path)
        print(f"\n{run['kv']} {knobs(run)} vs {ref['kv']} {knobs(ref)}")
        print(f"{'doc':<14}{'agree%':>8}{'NLL ref':>9}{'NLL run':>9}{'dNLL':>9}{'|dNLL|':>8}{'acc ref':>8}{'acc run':>8}")
        by_len = defaultdict(list)
        for name, a in ref["docs"].items():
            b = run["docs"].get(name)
            if b is None:
                continue
            agree = (a["top_id"][:, 0] == b["top_id"][:, 0]).float()
            d = b["nll"] - a["nll"]
            acc_a = (a["top_id"][:, 0] == a["tgt"]).float()
            acc_b = (b["top_id"][:, 0] == b["tgt"]).float()
            row = (agree, a["nll"], b["nll"], d, acc_a, acc_b)
            by_len[name.split("/")[0]].append(row)
            by_len["all"].append(row)
            print(f"{name:<14}{100 * agree.mean():8.2f}{a['nll'].mean():9.4f}{b['nll'].mean():9.4f}"
                  f"{d.mean():+9.4f}{d.abs().mean():8.4f}{100 * acc_a.mean():8.2f}{100 * acc_b.mean():8.2f}")
        for L, rows in by_len.items():
            agree, na, nb, d, aa, ab = (torch.cat(x) for x in zip(*rows))
            # a paired standard error of dNLL over positions (positions are correlated: a rough floor)
            se = d.std() / d.numel() ** 0.5
            print(f"{'= ' + L:<14}{100 * agree.mean():8.2f}{na.mean():9.4f}{nb.mean():9.4f}{d.mean():+9.4f}"
                  f"{d.abs().mean():8.4f}{100 * aa.mean():8.2f}{100 * ab.mean():8.2f}   (n={d.numel()}, "
                  f"dNLL se {se:.4f})")


if __name__ == "__main__":
    main()
