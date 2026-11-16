"""The FP8 cache's teacher-forced comparison: two s5_prompt_lp.py runs of one binary, the reference with the default
BF16 latent cache and the candidate with --kv-cache fp8 on the same checkpoint (or, for scale, another checkpoint with
the BF16 cache), position by position over mkcorpus.py's sequences
(row t of a sequence scores its token t from tokens 0..t-1). numpy only. Over all positions, each domain, and each
position band (where the dense limit and the long tail sit):
  - top-1: the candidate's best token equals the reference's; "_tied": it is one of the reference's tied best.
  - NLL: the mean of -logprob of the given tokens, each side, and the rise.
  - KL(ref || cand) over the reference's top 20 only, as s5_compare.py takes it (a token outside the candidate's 20
    gets the candidate's 20th logprob, so that term is at most what it would be); the mass the 20 cover beside it.
  - the given token sure on one side (p > 0.99) and below even odds on the other, each way round; NLL a sequence.
  - every position whose best token differs, with both sides' margins: the reference's best minus its logprob of the
    candidate's best (the reference's 20th when it is outside), and the candidate's the other way round.
Two runs that are the same bytes give zero differences: run it on the reference against itself as the control.

    python3 kv8_compare.py PLAN_JSON REF_JSONL CAND_JSONL OUT_JSON OUT_FLIPS_JSONL
"""

import json
import sys

import numpy as np

BANDS = [(1, 2048), (2048, 8192), (8192, 32768), (32768, 65536)]


def load(path):
    given, top, rows = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            given.extend(r["given"])
            top.extend(r["top"])
            rows.extend((r["seq"], t) for t in range(1, r["n"]))
    g = np.asarray(given, dtype=np.float64)
    t = np.asarray(top, dtype=np.float64)
    return np.asarray(rows, dtype=np.int64), g[:, 0], t[:, :, 0].astype(np.int64), t[:, :, 1]


def other(ids, tlp, want):
    """Each row's logprob of token want[row] from its own top 20, or its 20th when it is not there."""
    eq = ids == want[:, None]
    return np.where(eq.any(-1), (eq * tlp).sum(-1), tlp[:, -1])


def stats(m, lp, ids, tlp, r_lp, r_ids, r_tlp):
    eq = r_ids[:, :, None] == ids[:, None, :]
    found = eq.any(-1)
    c_for_r = np.where(found, (eq * tlp[:, None, :]).sum(-1), tlp[:, -1:])
    p = np.exp(r_tlp)
    kl = (p * (r_tlp - c_for_r)).sum(-1)
    d = np.abs(lp - r_lp)
    tied = other(r_ids, r_tlp, ids[:, 0]) == r_tlp[:, 0]
    return {"positions": int(m.sum()), "top1_equal": float((ids[:, 0] == r_ids[:, 0])[m].mean()),
            "top1_equal_tied": float(tied[m].mean()), "nll_ref": float(-r_lp[m].mean()), "nll_cand": float(-lp[m].mean()),
            "nll_rise": float(r_lp[m].mean() - lp[m].mean()),
            "nll_rise_rel": float((r_lp[m].mean() - lp[m].mean()) / -r_lp[m].mean()),
            "given_lp_abs_diff": {"mean": float(d[m].mean()), "p99": float(np.quantile(d[m], 0.99)), "max": float(d[m].max())},
            "kl_top20": {"mean": float(kl[m].mean()), "p99": float(np.quantile(kl[m], 0.99)), "max": float(kl[m].max())},
            "ref_top20_mass": float(p.sum(-1)[m].mean()), "ref_top20_found_in_cand_top20": float(found[m].mean())}


def main(plan_path, ref_path, cand_path, out, flips_out):
    plan = {s["seq"]: s for s in json.load(open(plan_path))["sequences"]}
    rows, lp, ids, tlp = load(cand_path)
    r_rows, r_lp, r_ids, r_tlp = load(ref_path)
    assert np.array_equal(rows, r_rows), "the runs' rows are not the same positions"
    domain = np.array([plan[s]["domain"] for s in rows[:, 0]])
    pos = rows[:, 1]
    args = (lp, ids, tlp, r_lp, r_ids, r_tlp)
    res = {"all": stats(np.ones(len(pos), bool), *args)}
    for dname in ("prose", "code"):
        res[dname] = stats(domain == dname, *args)
    for lo, hi in BANDS:
        m = (pos >= lo) & (pos < hi)
        if m.any():
            res[f"pos_{lo}_{hi}"] = stats(m, *args)
    flip = np.nonzero(ids[:, 0] != r_ids[:, 0])[0]
    r_margin = r_tlp[:, 0] - other(r_ids, r_tlp, ids[:, 0])
    c_margin = tlp[:, 0] - other(ids, tlp, r_ids[:, 0])
    res["flips"] = {"count": int(flip.size), "ref_margin_max": float(r_margin[flip].max()) if flip.size else 0.0,
                    "ref_margin_p50": float(np.median(r_margin[flip])) if flip.size else 0.0,
                    "cand_margin_max": float(c_margin[flip].max()) if flip.size else 0.0,
                    "ref_margin_le": {str(x): int((r_margin[flip] <= x).sum()) for x in (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)}}
    # both ways: the given token near certain on one side (p > 0.99) and below even odds on the other
    conf = {}
    for name, m in [("all", np.ones(len(pos), bool)), ("prose", domain == "prose"), ("code", domain == "code")] + \
                   [(f"pos_{lo}_{hi}", (pos >= lo) & (pos < hi)) for lo, hi in BANDS]:
        conf[name] = {"ref_sure": int((m & (r_lp > np.log(0.99))).sum()),
                      "ref_sure_cand_below_half": int((m & (r_lp > np.log(0.99)) & (lp < np.log(0.5))).sum()),
                      "cand_sure_ref_below_half": int((m & (lp > np.log(0.99)) & (r_lp < np.log(0.5))).sum())}
    res["sure_tokens"] = conf
    res["sequences"] = [{"seq": int(s), "domain": plan[int(s)]["domain"], "n": plan[int(s)]["n"],
                         "nll_ref": float(-r_lp[rows[:, 0] == s].mean()), "nll_cand": float(-lp[rows[:, 0] == s].mean())}
                        for s in np.unique(rows[:, 0])]
    res["identical_rows"] = bool(np.array_equal(lp, r_lp) and np.array_equal(ids, r_ids) and np.array_equal(tlp, r_tlp))
    with open(flips_out, "w") as f:
        for i in flip:
            f.write(json.dumps({"seq": int(rows[i, 0]), "pos": int(pos[i]), "domain": str(domain[i]),
                                "ref_top1": [int(r_ids[i, 0]), float(r_tlp[i, 0])], "cand_top1": [int(ids[i, 0]), float(tlp[i, 0])],
                                "ref_margin": float(r_margin[i]), "cand_margin": float(c_margin[i])}) + "\n")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("all", "flips", "identical_rows")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
