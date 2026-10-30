"""The depth decoder on the reference reply's states: reference_depth.py MODEL DUMP prep|score [DEV] [DTYPE]."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import refcommon as rc

AUDIO_OUT = 125032


def inputs(d):
    """The state before each reference frame and its codes, as lfm3-omni-generate --depth-check reads them."""
    seq, sh, codes = np.load(d / "gen_seq.npy"), np.load(d / "step_hidden.npy"), np.load(d / "gen_codes.npy").T
    slots = [j for j in range(1, len(seq)) if seq[j] == AUDIO_OUT][: len(codes)]
    return np.stack([sh[j - 1] for j in slots]).astype(np.float32), codes[: len(slots)].astype(np.int32)


def prep(model_dir, d, dev, dtype):
    """Writes depth_h.f32 and depth_codes.i32, and the reference's three passes of logits per frame."""
    h_all, c_all = inputs(d)
    h_all.tofile(d / "depth_h.f32")
    c_all.tofile(d / "depth_codes.i32")
    model, _ = rc.load(model_dir, dev, dtype)
    dd, out = model.model.depth_decoder, []
    with torch.no_grad():
        for h, c in zip(h_all, c_all):
            ht = torch.from_numpy(h)[None].to(dev, dtype)
            ct = torch.zeros((1, 12), dtype=torch.long, device=dev)
            masked = torch.ones((1, 12), dtype=torch.bool, device=dev)
            rows = []
            for p in range(3):
                rows.append(dd(ht, ct, masked)[0].float().cpu().numpy())
                for q in range(4 * p, 4 * p + 4):
                    ct[0, q] = int(c[q])
                    masked[0, q] = False
            out.append(np.stack(rows))
    np.save(d / "ref_depth.npy", np.stack(out))


def score(d):
    """Greedy agreement and logits cosine of depth_check.f32 against the reference's passes."""
    ref = np.load(d / "ref_depth.npy")
    tf = np.fromfile(d / "depth_check.f32", dtype=np.float32).reshape(ref.shape)
    agree = total = 0
    cos = []
    for p in range(3):
        for q in range(4 * p, 4 * p + 4):
            a, b = ref[:, p, q], tf[:, p, q]
            agree += int((a.argmax(-1) == b.argmax(-1)).sum())
            total += len(a)
            for x, y in zip(a.astype(np.float64), b.astype(np.float64)):
                cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))))
    r = {"frames": int(ref.shape[0]), "argmax_agree": agree, "scored": total, "rate": round(agree / total, 4),
         "logit_cos_min": round(min(cos), 6), "logit_cos_median": round(float(np.median(cos)), 6)}
    print(json.dumps(r))
    json.dump(r, open(d / "depth_parity.json", "w"))


if __name__ == "__main__":
    D = Path(sys.argv[2])
    if sys.argv[3] == "prep":
        dev = sys.argv[4] if len(sys.argv) > 4 else "cuda"
        prep(sys.argv[1], D, dev, rc.dtype_of(sys.argv[5] if len(sys.argv) > 5 else "bf16"))
    else:
        score(D)
