"""Stage parity of lfm3-omni-generate's dumps against reference_dump.py's: python compare_stages.py REF_DIR TF_DIR."""
import json
import sys
from pathlib import Path

import numpy as np

MIXED_START = 125015


def stat(a, b):
    """Cosine, relative L2 and largest absolute difference in float64."""
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
    return {"cos": round(cos, 6), "rel_l2": round(rel, 5), "max_abs": round(float(np.abs(a - b).max()), 5)}


def first_split(a, b):
    """The length of the common prefix of two sequences."""
    m = min(len(a), len(b))
    return next((i for i in range(m) if np.any(a[i] != b[i])), m)


def compare(r, t):
    """Every stage the two dumps share."""
    def raw(name, dtype=np.float32):
        p = t / name
        return np.fromfile(p, dtype=dtype) if p.exists() else None

    out = {}
    mel = np.load(r / "mel.npy")
    tm = raw("mel.f32").reshape(128, -1)
    out["mel"] = stat(mel, tm[:, : mel.shape[1]]) | {"frames": [mel.shape[1], tm.shape[1]]}
    enc = np.load(r / "enc.npy")
    te = raw("enc.f32").reshape(-1, enc.shape[1])
    out["encoder"] = stat(enc, te) | {"tokens": [enc.shape[0], te.shape[0]]}
    emb = np.load(r / "audio_emb.npy")
    out["projector"] = stat(emb, raw("audio_emb.f32").reshape(emb.shape))
    ids, ti = np.load(r / "input_ids.npy"), raw("input_ids.i32", np.int32)
    out["prompt_ids_equal"] = bool(len(ti) == len(ids) + 1 and (ti[: len(ids)] == ids).all() and ti[-1] == MIXED_START)
    ph = np.load(r / "prefill_hidden.npy")
    tph = raw("prefill_hidden.f32").reshape(-1, ph.shape[1])
    n = ph.shape[0]
    out["prefill_hidden"] = stat(ph, tph[:n]) | {"last_row": stat(ph[-1], tph[n - 1])}
    out["prefill_hidden"]["min_row_cos"] = min(stat(ph[i], tph[i])["cos"] for i in range(n))
    dl = np.load(r / "depth_logits0.npy")
    tdl = raw("depth_logits0.f32").reshape(dl.shape)
    out["depth_logits_frame0"] = stat(dl, tdl) | {"argmax_equal": int((dl.argmax(-1) == tdl.argmax(-1)).sum())}
    seq = np.load(r / "gen_seq.npy")
    ts = np.concatenate([[MIXED_START], raw("gen_seq.i32", np.int32)])
    div = first_split(seq, ts)
    out["free_run_slots"] = {"ref": int(len(seq)), "tf": int(len(ts)), "identical_prefix": int(div)}
    codes, tc = np.load(r / "gen_codes.npy").T, raw("gen_codes.i32", np.int32).reshape(-1, 12)
    same = int(first_split(codes, tc))
    out["free_run_frames"] = {"ref": int(len(codes)), "tf": int(len(tc)), "identical_prefix": same}
    sh, tsh = np.load(r / "step_hidden.npy"), raw("step_hidden.f32").reshape(-1, ph.shape[1])
    k = min(len(sh), len(tsh), div)
    min_cos = round(min(stat(sh[i], tsh[i])["cos"] for i in range(k)), 6) if k else None
    out["step_hidden_before_divergence"] = {"steps": int(k), "min_cos": min_cos, "first": stat(sh[0], tsh[0])}
    cl = raw("codec_lin.f32")
    if cl is not None and (r / "codec_lin.npy").exists():
        rl = np.load(r / "codec_lin.npy")
        cl = cl.reshape(-1, rl.shape[1])
        k = min(len(rl), len(cl))
        out["codec_lin"] = stat(rl[:k], cl[:k]) | {"rows": [len(rl), len(cl)]}
        rw, tw = np.load(r / "reply.npy"), raw("reply.f32")
        k = min(len(rw), len(tw))
        snr = 10 * np.log10(np.sum(rw[:k] ** 2) / (np.sum((rw[:k] - tw[:k]) ** 2) + 1e-30))
        out["codec_wave"] = {"samples": [len(rw), len(tw)], "snr_db": round(float(snr), 2)} | stat(rw[:k], tw[:k])
    return out


if __name__ == "__main__":
    R, T = Path(sys.argv[1]), Path(sys.argv[2])
    res = compare(R, T)
    print(json.dumps(res, indent=1))
    json.dump(res, open(T / "parity.json", "w"), indent=1)
