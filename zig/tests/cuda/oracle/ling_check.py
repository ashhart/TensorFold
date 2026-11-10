#!/usr/bin/env python3
"""Score ling-tp --ids runs against ling_ref_stream.py's exllamav3 answer key, on the CPU."""

# python3 ling_check.py REF_DIR --run a=DIR [--run b=DIR ...] [--json out.json] [--misses]

# DIR/<text>.bin from `ling-tp PACK OUT --ids <text>.ids` (TFLTP001): row r is the forward after ids[:T//2 + r].

# Per text and run: KL(ref || ours), top-1 agreement, ratio to exllamav3's floor, true-token NLL, worst rows' margins.

# Floors: floor_dec is the T-17 prefix run against the full run on shared decode rows; floor_max the larger prefix mean.

# --misses lists each row whose top-1 differs: both picks, both margins, the true token and the prefix run's pick.

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read_run(path, V):
    b = Path(path).read_bytes()
    assert b[:8] == b"TFLTP001", path
    world, vocab, rows = np.frombuffer(b, np.uint32, 3, 8)
    assert vocab == V, (vocab, V)
    toks = np.frombuffer(b, np.uint32, rows, 20)
    lg = np.frombuffer(b, np.uint16, rows * vocab, 20 + 4 * rows).astype(np.uint32) << 16
    return int(world), toks, torch.from_numpy(lg.view(np.float32).reshape(rows, vocab).copy())


def kl_rows(p_logits, q_logits):
    lp = torch.log_softmax(p_logits.float(), -1)
    lq = torch.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def nll_rows(logits, targets):
    return -torch.log_softmax(logits.float(), -1).gather(1, targets[:, None]).squeeze(1)


def margin(row):
    t = torch.topk(row.float(), 2)
    return int(t.indices[0]), int(t.indices[1]), float(t.values[0] - t.values[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--run", action="append", required=True, help="label=DIR")
    ap.add_argument("--json")
    ap.add_argument("--worst", type=int, default=5)
    ap.add_argument("--misses", action="store_true")
    a = ap.parse_args()
    ref = Path(a.ref)
    meta = json.loads((ref / "meta.json").read_text())
    V = meta["vocab"]
    runs = dict(r.split("=", 1) for r in a.run)
    labels = list(runs)
    report = {}
    for name in (ref / "names.txt").read_text().split():
        ids = torch.from_numpy(np.fromfile(ref / f"{name}.ids", dtype=np.int32).astype(np.int64))
        T = ids.numel()
        P = T // 2
        full = torch.from_numpy(np.fromfile(ref / f"{name}.ref_logits.f32", dtype=np.float32).reshape(T, V))
        rl = full[P - 1:T - 1]                     # the rows our runs hold: positions P-1 .. T-2
        tgt = ids[P:T]
        r = {"tokens": T, "prompt": P, "decode_rows": T - P}
        # exllamav3's own floor
        k2 = T - 17
        pre = torch.from_numpy(np.fromfile(ref / f"{name}@{k2}.ref_logits.f32", dtype=np.float32).reshape(k2, V))
        fkl = kl_rows(full[P - 1:k2], pre[P - 1:k2])
        r["floor_dec"] = float(fkl.mean())
        r["floor_dec_top1"] = float((full[P - 1:k2].argmax(-1) == pre[P - 1:k2].argmax(-1)).float().mean())
        r["floor_dec_kl_max"] = float(fkl.max())
        r["floor_max"] = max(v["kl"] for v in meta["texts"][name]["noise"].values())
        r["floor_runs"] = meta["texts"][name]["noise"]
        nref = nll_rows(rl, tgt)
        r["nll_ref"] = float(nref.mean())
        r["nll_ref_full_text"] = meta["texts"][name]["nll"]
        got = {}
        for lab in labels:
            f = Path(runs[lab]) / f"{name}.bin"
            if not f.exists():
                continue
            world, toks, lg = read_run(f, V)
            assert lg.shape[0] == T - P, (lab, name, lg.shape, T - P)
            got[lab] = lg
            kl = kl_rows(rl, lg)
            n = nll_rows(lg, tgt)
            e = {"world": world, "kl": float(kl.mean()), "kl_median": float(kl.median()), "kl_max": float(kl.max()),
                 "top1": float((rl.argmax(-1) == lg.argmax(-1)).float().mean()),
                 "strict_misses": int(sum(1 for i in range(len(kl)) if int(lg[i].argmax()) != int(rl[i].argmax())
                                          and float(rl[i].max() - rl[i][int(lg[i].argmax())]) > 0)),
                 "kl_over_floor_dec": float(kl.mean()) / r["floor_dec"] if r["floor_dec"] > 0 else None,
                 "kl_over_floor_max": float(kl.mean()) / r["floor_max"] if r["floor_max"] > 0 else None,
                 "nll": float(n.mean()), "nll_delta_pct": 100 * (float(n.mean()) - r["nll_ref"]) / r["nll_ref"]}
            worst = []
            for i in torch.topk(kl, min(a.worst, len(kl))).indices.tolist():
                pos = P - 1 + i
                ra, rb, rm = margin(rl[i])
                oa, ob, om = margin(lg[i])
                worst.append({"pos": pos, "kl": round(float(kl[i]), 4), "ref_top2": [ra, rb], "ref_margin": round(rm, 4),
                              "ours_top2": [oa, ob], "ours_margin": round(om, 4), "true": int(tgt[i]),
                              "nll_ref": round(float(nref[i]), 3), "nll_ours": round(float(n[i]), 3),
                              "floor_kl_here": round(float(fkl[i]), 4) if pos < k2 else None,
                              "floor_kl_rank": (int((fkl > fkl[i]).sum()) + 1) if pos < k2 else None})
            e["worst"] = worst
            misses = []
            for i in torch.nonzero(rl.argmax(-1) != lg.argmax(-1)).flatten().tolist():
                pos, ra, oa = P - 1 + i, int(rl[i].argmax()), int(lg[i].argmax())
                misses.append({"pos": pos, "ref_pick": ra, "ours_pick": oa, "true": int(tgt[i]),
                               "ref_margin": round(float(rl[i][ra] - rl[i][oa]), 4),
                               "ours_margin": round(float(lg[i][oa] - lg[i][ra]), 4),
                               "floor_pick": int(pre[pos].argmax()) if pos < k2 else None})
            e["misses"] = misses
            if a.misses:
                print(f"[misses] {name} {lab}: {len(misses)} of {T - P} rows (pos ref_pick ours_pick true ref_margin "
                      "ours_margin floor_pick)", flush=True)
                for m in misses:
                    print(f"  {m['pos']} {m['ref_pick']} {m['ours_pick']} {m['true']} {m['ref_margin']} {m['ours_margin']} "
                          f"{m['floor_pick']}", flush=True)
            r[lab] = e
        if len(got) >= 2:
            x, y = labels[0], labels[1]
            if x in got and y in got:
                kl = kl_rows(got[x], got[y])
                r[f"{y}_vs_{x}"] = {"kl": float(kl.mean()), "kl_max": float(kl.max()),
                                    "top1": float((got[x].argmax(-1) == got[y].argmax(-1)).float().mean()),
                                    "byte_equal_rows": int((got[x] == got[y]).all(-1).sum())}
            for z in labels[2:]:
                if z in got and labels[1] in got:
                    r[f"{z}_vs_{labels[1]}"] = {"byte_equal_rows": int((got[z] == got[labels[1]]).all(-1).sum()),
                                                "rows": T - P}
        report[name] = r
        line = f"[check] {name}: T {T}, floor dec {r['floor_dec']:.5f} max {r['floor_max']:.5f}, nll ref {r['nll_ref']:.4f}"
        for lab in labels:
            if lab in r:
                e = r[lab]
                line += (f" | {lab}: KL {e['kl']:.5f} ({e['kl_over_floor_dec']:.2f}x dec, {e['kl_over_floor_max']:.2f}x max)"
                         f" top1 {e['top1']:.4f} nll {e['nll']:.4f} ({e['nll_delta_pct']:+.2f}%)")
        for k, v in r.items():
            if "_vs_" in k:
                line += f" | {k}: {json.dumps(v)}"
        print(line, flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
