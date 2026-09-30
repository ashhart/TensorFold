"""Is an R-row verify step bit-identical to R one-row steps? Greedy-decode a prompt one row at a time recording
logits, then from the same states replay windows of R tokens through the R-row graph and compare bits.

Run on both ranks like tools/dsv41_serial_run.py (rank 0 prints).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tensorfold.cuda.comm import NCCL
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
    ap.add_argument("--case", default="reasoning")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--windows", default="0,5,11,23")
    args = ap.parse_args()
    torch.cuda.set_device(0)
    nccl = NCCL(args.rank, 2, args.master, args.port)
    w = W.load(args.model, rank=args.rank, log=lambda *a, **k: None, draft=False)
    eng = SerialEngine(w, Comm(nccl), str(args.engram), str(args.model / "tokenizer.json"), cap=1024)
    R = args.rows
    starts = [int(s) for s in args.windows.split(",")]
    prompt = next(c for c in json.loads(args.cases.read_text())["results"] if c["name"] == args.case)["ids"]
    with torch.no_grad():
        eng.capture(1)
        eng.capture(R)

        def prefill():
            eng.reset()
            logits = None
            for i in range(0, len(prompt), MAX_ROWS):
                logits = eng.forward(prompt[i:i + MAX_ROWS])
            return int(logits[-1].argmax())

        tok = prefill()
        toks, logits = [], []
        for _ in range(max(starts) + R):
            toks.append(tok)
            tok = eng.step(tok)
            logits.append(eng.graphs[1]["logits"][0].clone())
        results = []
        for s in starts:
            prefill()
            for t in toks[:s]:
                eng.step(t)
            eng.step_rows(toks[s:s + R])
            got = eng.graphs[R]["logits"].clone()
            same = [bool(torch.equal(got[j], logits[s + j])) for j in range(R)]
            diff = max(float((got[j] - logits[s + j]).abs().max()) for j in range(R))
            results.append((s, same, diff))
    if args.rank == 0:
        for s, same, diff in results:
            print(f"window at {s}: rows identical {same}, max |dlogit| {diff:.3g}", flush=True)
    nccl.barrier()


if __name__ == "__main__":
    main()
