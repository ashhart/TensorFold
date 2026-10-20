"""S5's bf16 reference (<redacted-plan>), with no engine: B-2's tokens (ids.pt) through BF16's
lm_head on the BF16 model's final hidden (eval_glm.py ref's hidden.pt: after the hyper-connection head and the norm),
with eval_glm.py compare's arithmetic (the bf16 linear's output to float32, log_softmax a 512-position chunk). For
every predicted position (each sequence's 0..T-2): the next token's logprob, the logsumexp, the top 20 ids and
logprobs. CHECK (a score's summary.json of the same set) must give the same mean NLL as its nll_ref.

    python s5_bf16_ref.py IDS HIDDEN SNAPSHOT CHECK OUT
"""
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open

CHUNK = 512  # eval_glm.py's
TOP = 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(ids_path, hidden_path, snapshot, check_path, out):
    ids = torch.load(ids_path)["input_ids"]
    seqs = [s.long().reshape(-1) for s in (ids if isinstance(ids, list) else ids.unbind(0))]
    h = hashlib.sha256()
    for s in seqs:
        h.update(s.to(torch.int32).numpy().tobytes())
    finals = torch.load(hidden_path)
    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(snapshot, index["lm_head.weight"]), "pt") as f:
        w = f.get_tensor("lm_head.weight")
    dev = torch.device("cuda")
    head = w.to(dev)
    cols = {k: [] for k in ("seq", "pos", "next", "next_lp", "lse", "top_ids", "top_lp")}
    with torch.no_grad():
        for s, (t, a) in enumerate(zip(seqs, finals)):
            assert a.shape[0] == len(t), (s, a.shape, len(t))
            for lo in range(0, len(t) - 1, CHUNK):
                hi = min(lo + CHUNK, len(t) - 1)
                logits = torch.nn.functional.linear(a[lo:hi].to(dev), head).float()
                lse = torch.logsumexp(logits, -1)
                lp = torch.log_softmax(logits, -1)  # eval_glm.py compare's form: a position's value is its nll_ref
                nxt = t[lo + 1:hi + 1].to(dev)
                top = lp.topk(TOP, -1)
                cols["seq"].append(torch.full((hi - lo,), s, dtype=torch.int32))
                cols["pos"].append(torch.arange(lo, hi, dtype=torch.int32))
                cols["next"].append(nxt.to(torch.int32).cpu())
                cols["next_lp"].append(lp.gather(1, nxt[:, None])[:, 0].cpu())
                cols["lse"].append(lse.cpu())
                cols["top_ids"].append(top.indices.to(torch.int32).cpu())
                cols["top_lp"].append(top.values.cpu())
    res = {k: torch.cat(v) for k, v in cols.items()}
    check = json.load(open(check_path))
    nll = float(-res["next_lp"].double().mean())
    meta = {"ids": os.path.basename(ids_path), "ids_file_sha256": sha256_file(ids_path), "ids_sha256": h.hexdigest()[:16],
            "hidden_sha256": sha256_file(hidden_path), "hidden_dtype": str(finals[0].dtype),
            "lm_head": {"file": index["lm_head.weight"], "shape": list(w.shape), "dtype": str(w.dtype),
                        "sha256": hashlib.sha256(w.view(torch.uint16).numpy().tobytes()).hexdigest()},
            "snapshot": os.path.basename(snapshot.rstrip("/")), "sequences": len(seqs),
            "positions": int(res["next_lp"].numel()), "top": TOP, "chunk": CHUNK, "nll_mean": nll,
            "top1_is_next": float((res["top_ids"][:, 0] == res["next"]).float().mean()),
            "check": {"file": check_path, "ids_sha256": check.get("ids_sha256"), "nll_ref": check.get("nll_ref"),
                      "positions": check.get("positions")}}
    meta["check"]["equal"] = (check.get("ids_sha256") == meta["ids_sha256"] and check.get("positions") == meta["positions"]
                              and abs(check.get("nll_ref", 0.0) - nll) < 1e-5)  # float32 against float64 sums
    torch.save({"meta": meta, **res}, out)
    json.dump(meta, open(out + ".json", "w"), indent=1)
    print(json.dumps(meta))
    return 0 if meta["check"]["equal"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
