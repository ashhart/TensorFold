"""The stage table over parity.sh's output: python summarize_parity.py REF_DIR OUT_DIR [NAME] (writes summary.json)."""
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

FORCED = re.compile(r"FORCED text (\d+)/(\d+) codes (\d+)/(\d+) frames (\d+)/(\d+)")


def cos(a, b):
    """Cosine in float64."""
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def pct(a, b):
    """a/b with its percentage."""
    return f"{a}/{b} ({100 * a / max(1, b):.1f}%)"


def summarize(r, t, name):
    """Per-stage minima and medians over every utterance with a parity.json."""
    keys = ("mel_cos", "mel_max", "enc", "proj", "ids", "pre_med", "pre_min", "lin", "snr")
    rows = {k: [] for k in keys}
    forced, depth, dmed, dmin = [0] * 6, [0, 0], [], []
    us = sorted(p.name for p in t.iterdir() if (p / "parity.json").exists())
    for u in us:
        p = json.load(open(t / u / "parity.json"))
        rows["mel_cos"].append(p["mel"]["cos"])
        rows["mel_max"].append(p["mel"]["max_abs"])
        rows["enc"].append(p["encoder"]["cos"])
        rows["proj"].append(p["projector"]["cos"])
        rows["ids"].append(p["prompt_ids_equal"])
        ph = np.load(r / u / "prefill_hidden.npy")
        tph = np.fromfile(t / u / "prefill_hidden.f32", dtype=np.float32).reshape(-1, ph.shape[1])[: len(ph)]
        rcos = [cos(ph[i], tph[i]) for i in range(len(ph))]
        rows["pre_med"].append(statistics.median(rcos))
        rows["pre_min"].append(min(rcos))
        if "codec_lin" in p:
            rows["lin"].append(p["codec_lin"]["cos"])
            rows["snr"].append(p["codec_wave"]["snr_db"])
        m = FORCED.search(open(t / u / "forced.log").read())
        if m:
            forced = [x + int(m.group(i + 1)) for i, x in enumerate(forced)]
        dp = t / u / "depth_parity.json"
        if dp.exists() and dp.stat().st_size:
            d = json.load(open(dp))
            depth = [depth[0] + d["argmax_agree"], depth[1] + d["scored"]]
            dmed.append(d["logit_cos_median"])
            dmin.append(d["logit_cos_min"])
    med = statistics.median
    return {
        "name": name, "utterances": len(us),
        "log-mel cos (min), max abs (max)": [round(min(rows["mel_cos"]), 6), max(rows["mel_max"])],
        "speech encoder cos (min)": round(min(rows["enc"]), 6),
        "projector cos (min)": round(min(rows["proj"]), 6),
        "prompt ids identical": f'{sum(rows["ids"])}/{len(us)}',
        "prefill row cos (median of medians, min of medians, min row)":
            [round(med(rows["pre_med"]), 6), round(min(rows["pre_med"]), 6), round(min(rows["pre_min"]), 6)],
        "teacher-forced text top-1": pct(forced[0], forced[1]),
        "teacher-forced codes": pct(forced[2], forced[3]),
        "depth on the reference's states: greedy agreement": pct(depth[0], depth[1]),
        "depth logits cos (median, min)": [round(med(dmed), 6), round(min(dmin), 6)] if dmed else None,
        "codec lin cos (min)": round(min(rows["lin"]), 6) if rows["lin"] else None,
        "codec SNR dB (median, min)": [round(med(rows["snr"]), 2), round(min(rows["snr"]), 2)] if rows["snr"] else None,
    }


if __name__ == "__main__":
    R, T = Path(sys.argv[1]), Path(sys.argv[2])
    out = summarize(R, T, sys.argv[3] if len(sys.argv) > 3 else T.name)
    print(json.dumps(out, indent=1))
    json.dump(out, open(T / "summary.json", "w"), indent=1)
