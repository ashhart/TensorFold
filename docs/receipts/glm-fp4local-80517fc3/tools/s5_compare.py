"""S5's comparison (<redacted-plan>): each engine run of s5_prompt_lp.py against the bf16 reference (s5_export.py's
bf16-ref-b2.npz), position by position over B-2 (the engine's row t of sequence s is the reference's position t-1 of s:
both score token t from tokens 0..t-1). numpy only.
  - top-1: the engine's best token equals bf16's (exact).
  - NLL: the mean of -logprob of the given tokens, the engine's against bf16's (exact; bf16's equals F3's nll_ref).
  - KL(bf16 || engine) over bf16's top 20 only, an approximation: sum p_bf16 (lp_bf16 - lp_engine) over bf16's 20, the
    engine's logprob of a token outside its own 20 taken as its 20th (so that term is at most what it would be); the
    mass bf16's 20 cover is printed beside it.

    python s5_compare.py REF_NPZ OUT LABEL=ENGINE_JSONL...
"""

import json
import sys

import numpy as np


def load(path):
    given, top, counts = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            counts.append((r["seq"], r["n"] - 1))
            given.extend(r["given"])
            top.extend(r["top"])
    g = np.asarray(given, dtype=np.float64)
    t = np.asarray(top, dtype=np.float64)
    return counts, g[:, 0], g[:, 1].astype(np.int64), t[:, :, 0].astype(np.int64), t[:, :, 1]


def compare(ref, path):
    counts, lp, rank, ids, tlp = load(path)
    seq = ref["seq"]
    want = [(int(s), int((seq == s).sum())) for s in np.unique(seq)]
    assert counts == want, "the engine's rows are not B-2's positions, sequence for sequence"
    r_lp, r_ids, r_tlp, nxt = ref["next_lp"].astype(np.float64), ref["top_ids"], ref["top_lp"].astype(np.float64), ref["next"]
    eq = r_ids[:, :, None] == ids[:, None, :]
    found = eq.any(-1)
    e_for_r = np.where(found, (eq * tlp[:, None, :]).sum(-1), np.where(r_ids == nxt[:, None], lp[:, None], tlp[:, -1:]))
    p = np.exp(r_tlp)
    kl = (p * (r_tlp - e_for_r)).sum(-1)
    d = lp - r_lp
    # both sides' logits are bf16, so best tokens tie; "_tied": the engine's first is one of bf16's tied best
    mine = r_ids == ids[:, :1]
    tied = np.where(mine.any(-1), (mine * r_tlp).sum(-1), -np.inf) == r_tlp[:, 0]
    return {"positions": int(lp.size), "top1_equal_bf16": float((ids[:, 0] == r_ids[:, 0]).mean()),
            "top1_equal_bf16_tied": float(tied.mean()),
            "top1_is_given": float((ids[:, 0] == nxt).mean()), "top1_is_given_tied": float((lp == tlp[:, 0]).mean()),
            "bf16_top1_is_given": float((r_ids[:, 0] == nxt).mean()),
            "bf16_top1_is_given_tied": float((r_lp == r_tlp[:, 0]).mean()),
            "nll": float(-lp.mean()), "nll_bf16": float(-r_lp.mean()), "nll_rise": float(-lp.mean() + r_lp.mean()),
            "nll_rise_rel": float((-lp.mean() + r_lp.mean()) / -r_lp.mean()),
            "given_lp_abs_diff": {"mean": float(np.abs(d).mean()), "p99": float(np.quantile(np.abs(d), 0.99)),
                                  "max": float(np.abs(d).max())},
            "kl_top20": {"mean": float(kl.mean()), "p99": float(np.quantile(kl, 0.99))},
            "bf16_top20_mass": float(p.sum(-1).mean()), "bf16_top20_found_in_engine_top20": float(found.mean())}


def main(ref_path, out, runs):
    ref = np.load(ref_path)
    res = {}
    for arg in runs:
        label, path = arg.split("=", 1)
        res[label] = compare(ref, path)
        print(label, json.dumps(res[label]), flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
