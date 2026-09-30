"""Where does row 1 of a 2-row static call first differ from two one-row calls? (both ranks; rank 0 prints)"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda.comm import NCCL
from tensorfold.families.deepseek_v41 import engram as E
from tensorfold.families.deepseek_v41.cuda import weights as W
from tensorfold.families.deepseek_v41.cuda.serial import MAX_ROWS, Comm, SerialEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("engram", type=Path)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29561)
    ap.add_argument("--cases", type=Path, default=Path("notes/dsv41/vllm_accept.json"))
    args = ap.parse_args()
    torch.cuda.set_device(0)
    nccl = NCCL(args.rank, 2, args.master, args.port)
    w = W.load(args.model, rank=args.rank, log=lambda *a, **k: None, draft=False)
    eng = SerialEngine(w, Comm(nccl), str(args.engram), str(args.model / "tokenizer.json"), cap=1024)
    c = eng.c
    prompt = next(x for x in json.loads(args.cases.read_text())["results"] if x["name"] == "reasoning")["ids"]
    ap2 = int(__import__("os").environ.get("ROWS", "2"))
    toks = prompt[-ap2:]
    base = prompt[:-ap2]

    def rows_for(ids_all, p0, R):
        start = max(0, p0 - 3)
        h = E.hashes(np.array(ids_all[start:p0 + R]), eng.tmap, eng.layout, c.engram_pad_token_id)[p0 - start:]
        got = eng.tables.raw([(ell, h[:, ell, :]) for ell in range(2)])
        wv = torch.from_numpy(np.stack([g[0] for g in got])).cuda().view(2, R, -1, 256)
        sc = torch.from_numpy(np.stack([g[1] for g in got])).cuda().view(2, R, wv.shape[2], -1)
        return E.dequant(wv, sc)

    with torch.no_grad():
        eng.reset()
        for i in range(0, len(base), MAX_ROWS):
            eng.forward(base[i:i + MAX_ROWS])
        saved = eng._save_caches()
        p0 = len(base)
        ids_all = base + toks
        ones = []
        for j in range(len(toks)):
            eng.debug = []
            eng.core(torch.tensor(toks[j:j + 1], device="cuda"), torch.tensor([p0 + j], device="cuda"),
                     rows_for(ids_all, p0 + j, 1), static=True)
            ones.append(eng.debug)
        one = ones[-1]
        eng._restore_caches(saved)
        eng.debug = []
        eng.core(torch.tensor(toks, device="cuda"), torch.arange(p0, p0 + len(toks), device="cuda"),
                 rows_for(ids_all, p0, len(toks)), static=True)
        two = eng.debug
        eng.debug = None
        # graphs vs eager from the same state
        eng._restore_caches(saved)
        e2 = eng.core(torch.tensor(toks, device="cuda"), torch.arange(p0, p0 + len(toks), device="cuda"),
                      rows_for(ids_all, p0, len(toks)), static=True).clone()
        eng.capture(1)
        eng.capture(len(toks))
        eng._restore_caches(saved)
        del eng.state.ids[p0:]
        eng.step_rows(toks)
        g2 = eng.graphs[len(toks)]["logits"].clone()
        eng._restore_caches(saved)
        del eng.state.ids[p0:]
        g1 = []
        for t in toks:
            eng.step_rows([t])
            g1.append(eng.graphs[1]["logits"][0].clone())
        graph_vs = {"graphR vs eagerR": float((g2 - e2).abs().max()),
                    "graph1 vs eagerR per row": [float((g1[j] - e2[j]).abs().max()) for j in range(len(toks))],
                    "graph1 vs graphR per row": [float((g1[j] - g2[j]).abs().max()) for j in range(len(toks))]}
    if args.rank == 0:
        for L, (a, b) in enumerate(zip(one, two)):
            last = len(toks) - 1
            diffs = {k: float((a[k][0].float() - b[k][last].float()).abs().max())
                     for k in ("attn_in", "X")}
            fa = a["f"][:, 0] if a["f"].dim() == 3 else a["f"][0]
            fb = b["f"][:, last] if b["f"].dim() == 3 else b["f"][last]
            diffs["f"] = float((fa.float() - fb.float()).abs().max())
            if any(v > 0 for v in diffs.values()):
                print(f"first difference at layer {L}: {diffs}", flush=True)
                break
        else:
            print("row 1 identical through every layer", flush=True)
        print(graph_vs, flush=True)
    nccl.barrier()


if __name__ == "__main__":
    main()
