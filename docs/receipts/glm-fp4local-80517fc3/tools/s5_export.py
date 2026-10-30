"""S5's bf16 reference for hosts without torch: bf16-ref-b2.pt's columns into an .npz (numpy) and B-2's sequences into
ids.json (lists of ints, for the engine's /v1/completions prompts). Runs where s5_bf16_ref.py ran (the host and
Python environment with torch).

    python s5_export.py IDS REF OUTDIR
"""
import json
import os
import sys

import numpy as np
import torch


def main(ids_path, ref_path, outdir):
    ids = torch.load(ids_path)["input_ids"]
    seqs = [s.long().reshape(-1).tolist() for s in (ids if isinstance(ids, list) else ids.unbind(0))]
    with open(os.path.join(outdir, "ids.json"), "w") as f:
        json.dump(seqs, f)
    ref = torch.load(ref_path)
    cols = {k: ref[k].numpy() for k in ("seq", "pos", "next", "next_lp", "lse", "top_ids", "top_lp")}
    np.savez_compressed(os.path.join(outdir, "bf16-ref-b2.npz"), **cols)
    print(json.dumps({"sequences": len(seqs), "tokens": sum(map(len, seqs)), "positions": int(cols["pos"].size),
                      "meta": ref["meta"]["check"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
