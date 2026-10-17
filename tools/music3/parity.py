"""MiniMax Music 3 on CUDA vs a ComfyUI reference run, stage by stage: the MLX port's parity.json table.

    python tools/music3/parity.py REF MINE [--report parity.json] [--ar-cfg 1.5]

REF is a ComfyUI reference dump (see docs/recipes/minimax-music3-cuda.md); MINE is what
`music3-generate --parity-{ar,dit,probe,dav} REF --out MINE` wrote. numpy only (no GPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def snr_db(ref, test) -> float:
    ref, test = np.asarray(ref, np.float64), np.asarray(test, np.float64)
    return float(10 * np.log10((ref ** 2).sum() / max(((ref - test) ** 2).sum(), 1e-30)))


def rows_cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)


def summary(values) -> dict:
    v = np.asarray(values, np.float64)
    return {"mean": round(float(v.mean()), 6), "min": round(float(v.min()), 6), "p01": round(float(np.percentile(v, 1)), 6)}


def check_ar(ref: Path, mine: Path, report: dict, ar_cfg: float) -> None:
    report["prompt_ids_equal"] = json.loads((mine / "ids.json").read_text())["ids"] == json.loads((ref / "ids.json").read_text())["ids"]
    hidden = np.load(ref / "hidden.npy")
    states = np.load(mine / "states.npy")
    n = min(len(states), len(hidden))
    layer = np.stack([rows_cos(states[:n, i * 4096:(i + 1) * 4096], hidden[:n, i * 4096:(i + 1) * 4096]) for i in range(8)])
    report["frame_states"] = {"frames": [int(len(states)), int(len(hidden))], "cos": summary(rows_cos(states[:n], hidden[:n])),
                              "cos_by_part": [round(float(x), 6) for x in layer.mean(axis=1)]}
    codes = np.load(ref / "codes.npy")
    ref_c0 = np.load(ref / "c0_logits.npy").astype(np.float32)
    my_c0 = np.load(mine / "c0_logits.npy")
    m = min(len(ref_c0), len(my_c0))
    g_ref = ref_c0[:m, 1] + (ref_c0[:m, 0] - ref_c0[:m, 1]) * ar_cfg
    g_my = my_c0[:m, 1] + (my_c0[:m, 0] - my_c0[:m, 1]) * ar_cfg
    top50 = np.mean([len(set(np.argsort(-a)[:50]) & set(np.argsort(-b)[:50])) / 50 for a, b in zip(my_c0[:m, 0], ref_c0[:m, 0])])
    forced = codes[:m, 0] + 1
    report["c0_logits"] = {"frames": int(m), "cos_cond": summary(rows_cos(my_c0[:m, 0], ref_c0[:m, 0])),
                           "cos_uncond": summary(rows_cos(my_c0[:m, 1], ref_c0[:m, 1])),
                           "greedy_agree_cond": round(float((my_c0[:m, 0].argmax(-1) == ref_c0[:m, 0].argmax(-1)).mean()), 4),
                           "greedy_agree_guided": round(float((g_my.argmax(-1) == g_ref.argmax(-1)).mean()), 4),
                           "top50_overlap": round(float(top50), 4),
                           "ref_drawn_code_rank_mine_median": int(np.median([(row > row[f]).sum() for row, f in zip(g_my, forced)]))}
    ref_d = np.load(ref / "depth_logits.npy").astype(np.float32)
    my_d = np.load(mine / "depth_logits.npy")
    m = min(len(ref_d), len(my_d))
    report["depth_logits"] = {
        "cos_cond_by_codebook": [round(float(rows_cos(my_d[:m, i, 0], ref_d[:m, i, 0]).mean()), 6) for i in range(7)],
        "greedy_agree_by_codebook": [round(float((my_d[:m, i, 0].argmax(-1) == ref_d[:m, i, 0].argmax(-1)).mean()), 4) for i in range(7)]}


def check_probe(ref: Path, mine: Path, report: dict) -> None:
    outs = np.load(ref / "probe_dit.npz")
    res = {}
    for sigma in (1.0, 0.5, 0.1):
        c, u = np.load(mine / f"probe_cond_{sigma:.1f}.npy"), np.load(mine / f"probe_uncond_{sigma:.1f}.npy")
        rc, ru = outs[f"cond_{sigma}"], outs[f"uncond_{sigma}"]
        res[str(sigma)] = {"cos_cond": round(cos(c, rc), 6), "cos_uncond": round(cos(u, ru), 6),
                           "rel_err_cond": round(float(np.linalg.norm(c - rc) / np.linalg.norm(rc)), 5),
                           "cos_guided_1.7": round(cos(u + (c - u) * 1.7, ru + (rc - ru) * 1.7), 6)}
    report["dit_probe"] = res


def check_dit(ref: Path, mine: Path, report: dict) -> None:
    r, m = np.load(ref / "latent.npy"), np.load(mine / "latent.npy")
    report["dit_final_latent"] = {"cos": round(cos(m, r), 6), "snr_db": round(snr_db(r, m), 2)}


def check_dav(ref: Path, mine: Path, report: dict) -> None:
    if (mine / "probe_dav.npy").exists():
        r, m = np.load(ref / "probe_dav.npy"), np.load(mine / "probe_dav.npy")
        n = min(r.shape[-1], m.shape[-1])
        report["dav_probe"] = {"cos": round(cos(m[..., :n], r[..., :n]), 7), "snr_db": round(snr_db(r[..., :n], m[..., :n]), 2)}
    if (mine / "audio.npy").exists():
        r, m = np.load(ref / "audio.npy"), np.load(mine / "audio.npy")
        n = min(m.shape[1], r.shape[1])
        report["dav_waveform"] = {"samples": [int(m.shape[1]), int(r.shape[1])], "cos": round(cos(m[:, :n], r[:, :n]), 7),
                                  "snr_db": round(snr_db(r[:, :n], m[:, :n]), 2)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("ref", type=Path)
    p.add_argument("mine", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--ar-cfg", type=float, default=1.5)
    p.add_argument("--label", default="")
    a = p.parse_args()
    report: dict = {"ref": str(a.ref), "mine": str(a.mine), "label": a.label}
    for name, fn in (("states.npy", lambda: check_ar(a.ref, a.mine, report, a.ar_cfg)),
                     ("probe_cond_1.0.npy", lambda: check_probe(a.ref, a.mine, report)),
                     ("latent.npy", lambda: check_dit(a.ref, a.mine, report)),
                     ("audio.npy", lambda: check_dav(a.ref, a.mine, report))):
        if (a.mine / name).exists() or name == "audio.npy":
            fn()
    text = json.dumps(report, indent=1)
    print(text)
    if a.report:
        a.report.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
