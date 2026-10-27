#!/usr/bin/env python3
"""Compare the independent reference golden (tools/exl3_ref_golden.py) with our engine's outputs.

  exl3_ref_compare.py golden.json --gen NAME=gen.log ... --tf NAME=tf.log ...

``gen.log``: ``tf-cuda-test exl3-forward <pack> gen 48 <n> <ids>`` (line ``GEN ids=...``).
``tf.log``: ``tf-cuda-test exl3-forward <pack> tf 48 <ids|default> hc`` (the per-position table: our device
top-1 and logit, the f64 oracle's top-1 and logit, our top-5 ids). Positions past the golden prompt are matched
against the golden greedy steps when the tf ids are the prompt followed by the golden continuation.
"""

from __future__ import annotations

import argparse
import json
import re


def parse_gen(path):
    for line in open(path):
        if line.startswith("GEN ids="):
            return [int(x) for x in line.split("=", 1)[1].strip().split(",") if x]
    raise ValueError(f"{path}: no GEN ids line")


ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+) \(\s*([-\d.]+)\)\s+(\d+) \(\s*([-\d.]+)\)\s+(\S+)\s+([-\d.]+)\s+([\d,]+)\s*$")


def parse_tf(path):
    rows = []
    for line in open(path):
        m = ROW.match(line)
        if m:
            p, inp, g, gl, o, ol, match, gap, five = m.groups()
            rows.append({"pos": int(p), "input": int(inp), "gpu": int(g), "gpu_logit": float(gl), "oracle": int(o),
                         "oracle_logit": float(ol), "match": match, "oracle_gap_of_gpu_pick": float(gap),
                         "gpu_top5": [int(x) for x in five.split(",")]})
    return rows


def ref_positions(p):
    """Reference rows by absolute position along prompt + greedy continuation: (input id, top5 ids, top5 logits)."""
    ids = p["ids"] + p["greedy_ids"]
    out = {}
    for r in p["teacher_forced"]:
        out[r["pos"]] = (ids[r["pos"]], r["top5_ids"], r["top5_logits"])
    n = len(p["ids"])
    for k, r in enumerate(p["greedy_steps"][1:], start=1):
        out[n - 1 + k] = (ids[n - 1 + k], r["top5_ids"], r["top5_logits"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--gen", action="append", default=[])
    ap.add_argument("--tf", action="append", default=[])
    ap.add_argument("--json", help="write the comparison here")
    a = ap.parse_args()
    g = json.load(open(a.golden))
    by = {p["name"]: p for p in g["prompts"]}
    report = {"gen": {}, "tf": {}}
    for spec in a.gen:
        name, path = spec.split("=", 1)
        p, ours = by[name], parse_gen(path)
        ref = p["greedy_ids"]
        n = min(len(ref), len(ours))
        first = next((i for i in range(n) if ref[i] != ours[i]), None)
        agree = sum(ref[i] == ours[i] for i in range(n))
        rec = {"ref": ref, "ours": ours, "agree_prefix": first if first is not None else n, "same_positions": agree,
               "n": n, "first_divergence": first}
        if first is not None:
            st = p["greedy_steps"][first]
            rec["ref_top5_at_divergence"] = list(zip(st["top5_ids"], st["top5_logits"]))
            ours_tok = ours[first]
            rec["ref_logit_of_our_pick"] = (st["top5_logits"][st["top5_ids"].index(ours_tok)]
                                            if ours_tok in st["top5_ids"] else None)
            rec["ref_margin_top1_minus_our_pick"] = (None if rec["ref_logit_of_our_pick"] is None
                                                     else st["top5_logits"][0] - rec["ref_logit_of_our_pick"])
        report["gen"][name] = rec
        print(f"GEN {name!r}: {agree}/{n} positions equal, identical prefix {rec['agree_prefix']}"
              + ("" if first is None else f"; first divergence at step {first}: ref {ref[first]} ours {ours[first]}, "
                 f"ref top-5 {rec['ref_top5_at_divergence']}, ref margin over our pick "
                 f"{rec['ref_margin_top1_minus_our_pick']}"))
    for spec in a.tf:
        name, path = spec.split("=", 1)
        p, rows = by[name], parse_tf(path)
        refpos = ref_positions(p)
        out = []
        print(f"TF {name!r} ({len(rows)} positions)")
        print("  pos input  ref_top1(logit)  gpu_top1(logit)  f64_top1(logit)  ref=gpu ref=f64 gpu=f64  "
              "|ref-gpu| |ref-f64| top5(ref)==top5(gpu)")
        for r in rows:
            if r["pos"] not in refpos:
                continue
            inp, ti, tv = refpos[r["pos"]]
            if inp != r["input"]:
                raise ValueError(f"pos {r['pos']}: input {r['input']} vs golden {inp}")
            d_g = abs(tv[0] - r["gpu_logit"]) if ti[0] == r["gpu"] else None
            d_o = abs(tv[0] - r["oracle_logit"]) if ti[0] == r["oracle"] else None
            row = {**r, "ref_top1": ti[0], "ref_logit": tv[0], "ref_top5": ti, "ref_top5_logits": tv,
                   "ref_eq_gpu": ti[0] == r["gpu"], "ref_eq_oracle": ti[0] == r["oracle"],
                   "abs_ref_gpu": d_g, "abs_ref_oracle": d_o, "top5_set_eq": set(ti) == set(r["gpu_top5"])}
            out.append(row)
            f = lambda x: "  -  " if x is None else f"{x:.3f}"  # noqa: E731
            print(f"  {r['pos']:>3} {r['input']:>6}  {ti[0]:>6}({tv[0]:7.3f})  {r['gpu']:>6}({r['gpu_logit']:7.3f})"
                  f"  {r['oracle']:>6}({r['oracle_logit']:7.3f})  {'yes' if row['ref_eq_gpu'] else 'NO ':>7}"
                  f" {'yes' if row['ref_eq_oracle'] else 'NO ':>7} {'yes' if r['gpu'] == r['oracle'] else 'NO ':>7}"
                  f"  {f(d_g):>8} {f(d_o):>8}  {row['top5_set_eq']}")
        n = len(out)
        dg = [x["abs_ref_gpu"] for x in out if x["abs_ref_gpu"] is not None]
        do = [x["abs_ref_oracle"] for x in out if x["abs_ref_oracle"] is not None]
        summ = {"positions": n, "ref_eq_gpu": sum(x["ref_eq_gpu"] for x in out),
                "ref_eq_oracle": sum(x["ref_eq_oracle"] for x in out),
                "gpu_eq_oracle": sum(x["gpu"] == x["oracle"] for x in out),
                "top5_set_eq": sum(x["top5_set_eq"] for x in out),
                "max_abs_ref_gpu_top1_logit": max(dg) if dg else None,
                "mean_abs_ref_gpu_top1_logit": sum(dg) / len(dg) if dg else None,
                "max_abs_ref_oracle_top1_logit": max(do) if do else None,
                "mean_abs_ref_oracle_top1_logit": sum(do) / len(do) if do else None}
        print("  " + json.dumps(summ))
        report["tf"][name + "@" + path] = {"summary": summ, "rows": out}
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
