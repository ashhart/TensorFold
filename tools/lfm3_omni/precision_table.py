"""Every stage against bf16 and fp32 references: precision_table.py REF_BF16 REF_F32 TF_BF16 TF_F32 [OUT_JSON]."""
import json
import statistics
import sys
from pathlib import Path

import numpy as np

import summarize_parity as sp


def snr(ref, x):
    """Waveform SNR in dB over the common length."""
    k = min(len(ref), len(x))
    a, b = ref[:k].astype(np.float64), x[:k].astype(np.float64)
    return float(10 * np.log10(np.sum(a * a) / (np.sum((a - b) ** 2) + 1e-30)))


def rows_cos(a, b):
    """Row cosines of two [rows, d] arrays over the common rows."""
    k = min(len(a), len(b))
    return [sp.cos(a[i], b[i]) for i in range(k)]


def references(rb, rf, us):
    """The bf16 reference against the fp32 one, stage by stage."""
    enc, proj, pre, mel = [], [], [], []
    for u in us:
        mel.append(bool(np.array_equal(np.load(rb / u / "mel.npy"), np.load(rf / u / "mel.npy"))))
        enc.append(sp.cos(np.load(rb / u / "enc.npy"), np.load(rf / u / "enc.npy")))
        proj.append(sp.cos(np.load(rb / u / "audio_emb.npy"), np.load(rf / u / "audio_emb.npy")))
        pre.append(statistics.median(rows_cos(np.load(rb / u / "prefill_hidden.npy"),
                                              np.load(rf / u / "prefill_hidden.npy"))))
    ids = sum(np.array_equal(np.load(rb / u / "input_ids.npy"), np.load(rf / u / "input_ids.npy")) for u in us)
    med = [round(statistics.median(pre), 6), round(min(pre), 6)]
    return {"log-mel identical": f"{sum(mel)}/{len(us)}", "speech encoder cos (min)": round(min(enc), 6),
            "projector cos (min)": round(min(proj), 6), "prompt ids identical": f"{ids}/{len(us)}",
            "prefill row cos (median of medians, min of medians)": med}


def forced(d, us, name):
    """Teacher-forced text and code agreement summed over NAME (reference_forced.py's scores)."""
    t, c = [0, 0], [0, 0]
    for u in us:
        r = json.load(open(d / u / name))
        t = [t[0] + r["text"][0], t[1] + r["text"][1]]
        c = [c[0] + r["codes"][0], c[1] + r["codes"][1]]
    return {"teacher-forced text top-1": sp.pct(*t), "teacher-forced codes": sp.pct(*c)}


def depth(d, us, name):
    """Depth decoder agreement and logits cosine summed over NAME (reference_depth.py's scores)."""
    a, n, med, lo = 0, 0, [], []
    for u in us:
        r = json.load(open(d / u / name))
        a, n = a + r["argmax_agree"], n + r["scored"]
        med.append(r["logit_cos_median"])
        lo.append(r["logit_cos_min"])
    return {"depth on the reference's states: greedy agreement": sp.pct(a, n),
            "depth logits cos (median, min)": [round(statistics.median(med), 6), round(min(lo), 6)]}


def steps(ref, other, us, name):
    """Backbone step rows along the reference's reply: OTHER/<u>/NAME (.npy, or the engine's raw fp32) against it."""
    med = []
    for u in us:
        a = np.load(ref / u / "step_hidden.npy")
        f = other / u / name
        b = np.load(f) if name.endswith(".npy") else np.fromfile(f, dtype=np.float32).reshape(-1, a.shape[1])
        med.append(statistics.median(rows_cos(a, b)))
    return {"step row cos, teacher-forced (median of medians, min of medians)":
            [round(statistics.median(med), 6), round(min(med), 6)]}


def codec(rb, tb, us):
    """The bf16 reference's codes through three codecs: the engine's, the reference's as stored and cast to fp32."""
    out = {k: {"snr": [], "lin": []} for k in ("engine vs bf16", "engine vs fp32", "bf16 vs fp32")}
    for u in us:
        if not (rb / u / "reply_f32.npy").exists():
            continue
        w = {"bf16": np.load(rb / u / "reply.npy"), "fp32": np.load(rb / u / "reply_f32.npy"),
             "engine": np.fromfile(tb / u / "reply.f32", dtype=np.float32)}
        l = {"bf16": np.load(rb / u / "codec_lin.npy"), "fp32": np.load(rb / u / "codec_lin_f32.npy")}
        l["engine"] = np.fromfile(tb / u / "codec_lin.f32", dtype=np.float32).reshape(-1, l["fp32"].shape[1])
        for k, v in out.items():
            x, ref = k.split(" vs ")
            v["snr"].append(snr(w[ref], w[x]))
            v["lin"].append(sp.cos(l[ref][: len(l[x])], l[x][: len(l[ref])]))
    return {k: {"codec SNR dB (median, min)": [round(statistics.median(v["snr"]), 2), round(min(v["snr"]), 2)],
                "codec lin cos (min)": round(min(v["lin"]), 6)} for k, v in out.items()}


def main():
    """Prints and writes the three comparisons: engine vs bf16, engine vs fp32, bf16 vs fp32."""
    rb, rf, tb, tf = (Path(x) for x in sys.argv[1:5])
    us = sorted(p.name for p in tb.iterdir() if (p / "parity.json").exists())
    a, b = sp.summarize(rb, tb, "engine vs bf16"), sp.summarize(rf, tf, "engine vs fp32")
    c = {"name": "bf16 vs fp32", "utterances": len(us)} | references(rb, rf, us) | forced(rf, us, "forced_bf16.json")
    c |= depth(rf, us, "depth_noise.json") | steps(rf, rf, us, "forced_bf16_hidden.npy")
    if (tb / us[0] / "forced" / "step_hidden.f32").exists():
        a |= steps(rb, tb, us, "forced/step_hidden.f32")
        b |= steps(rf, tf, us, "forced/step_hidden.f32")
    cd = codec(rb, tb, us)
    for x in (a, b):
        x.pop("codec lin cos (min)", None)
        x.pop("codec SNR dB (median, min)", None)
    a |= cd["engine vs bf16"]
    b |= cd["engine vs fp32"]
    c |= cd["bf16 vs fp32"]
    out = {"engine vs bf16": a, "engine vs fp32": b, "bf16 vs fp32": c}
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 5:
        json.dump(out, open(sys.argv[5], "w"), indent=1)


if __name__ == "__main__":
    main()
