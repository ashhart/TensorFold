"""Teacher-forced DSpark draft accuracy: prefill a known text, draft at many anchors, compare with the next tokens.

Run on both ranks like tools/dsv41_serial_run.py; variants via DSPARK_VARIANT (comma list, see cuda/dspark.py).
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
    ap.add_argument("--text", type=Path, default=Path("notes/dsv41/golden.json"))
    ap.add_argument("--prompt", type=int, default=6)
    ap.add_argument("--anchors", type=int, default=60)
    ap.add_argument("--dspark", type=int, default=3)
    args = ap.parse_args()
    torch.cuda.set_device(0)
    nccl = NCCL(args.rank, 2, args.master, args.port)
    w = W.load(args.model, rank=args.rank, log=lambda *a, **k: None)
    eng = SerialEngine(w, Comm(nccl), str(args.engram), str(args.model / "tokenizer.json"), cap=1024)
    eng.enable_dspark(args.dspark)
    ids = json.loads(args.text.read_text())["goldens"][args.prompt]["ids"]
    N = args.dspark
    with torch.no_grad():
        eng.reset()
        tgt = torch.cat([eng.forward(ids[i:i + MAX_ROWS]).argmax(-1) for i in range(0, len(ids), MAX_ROWS)]).tolist()
        saved = [t.clone() for t in eng.drafter.swa]
        hits = torch.zeros(N)
        prefix_ok = torch.zeros(N)
        vs_target = 0
        anchors = list(range(len(ids) - N - args.anchors, len(ids) - N))
        for P in anchors:
            for dst, src in zip(eng.drafter.swa, saved):
                dst.copy_(src)
            d = eng.drafter.draft(torch.tensor([ids[P]], device="cuda"), torch.tensor([P], device="cuda")).tolist()
            want = ids[P + 1:P + 1 + N]
            vs_target += d[0] == tgt[P]
            ok = True
            for j in range(N):
                hits[j] += d[j] == want[j]
                ok = ok and d[j] == want[j]
                prefix_ok[j] += ok
    if args.rank == 0:
        n = len(anchors)
        print(f"draft accuracy by position {[f'{100 * h / n:.0f}%' for h in hits.tolist()]}; "
              f"accepted prefix mean {prefix_ok.sum().item() / n:.2f} of {N}; first draft = target argmax "
              f"{100 * vs_target / n:.0f}%; target argmax = text {100 * sum(tgt[P] == ids[P + 1] for P in anchors) / n:.0f}%",
              flush=True)
    nccl.barrier()


if __name__ == "__main__":
    main()
