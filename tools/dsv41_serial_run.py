"""Run the serial two-rank DeepSeek-V4.1 engine on one golden prompt: prefill parity vs the reference, then decode speed.

    rank 1: python tools/dsv41_serial_run.py MODEL ENGRAM --rank 1 --master 10.42.0.1
    rank 0: python tools/dsv41_serial_run.py MODEL ENGRAM --rank 0 --master 10.42.0.1 --golden g.json --ref ref-6.pt
"""

from __future__ import annotations

import argparse
import json
import time
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
    ap.add_argument("--golden", type=Path, default=Path("notes/dsv41/golden.json"))
    ap.add_argument("--prompt", type=int, default=6)
    ap.add_argument("--ref", type=Path, help="reference .pt of the same prompt (rank 0 compares)")
    ap.add_argument("--decode", type=int, default=32)
    args = ap.parse_args()

    torch.cuda.set_device(0)
    nccl = NCCL(args.rank, 2, args.master, args.port)
    nccl.barrier()
    t0 = time.time()
    w = W.load(args.model, rank=args.rank, log=lambda *a, **k: None)
    torch.cuda.synchronize()
    print(f"[rank {args.rank}] weights loaded in {time.time() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    eng = SerialEngine(w, Comm(nccl), str(args.engram), str(args.model / "tokenizer.json"), cap=1024)
    nccl.barrier()

    ids = json.loads(args.golden.read_text())["goldens"][args.prompt]["ids"]
    with torch.no_grad():
        eng.reset()
        t1 = time.time()
        logits = torch.cat([eng.forward(ids[i:i + MAX_ROWS]) for i in range(0, len(ids), MAX_ROWS)])
        torch.cuda.synchronize()
        t2 = time.time()
    if args.rank == 0:
        print(f"prefill {len(ids)} tokens in {t2 - t1:.2f} s ({len(ids) / (t2 - t1):.0f} tok/s)", flush=True)
        if args.ref:
            ref = torch.load(args.ref)["logits"].float()
            ours = logits.float().cpu()
            agree = (ours.argmax(-1) == ref.argmax(-1)).float().mean().item()
            lp_o, lp_r = torch.log_softmax(ours, -1), torch.log_softmax(ref, -1)
            tgt = torch.tensor(ids[1:])
            d = (lp_o[:-1].gather(1, tgt[:, None]) - lp_r[:-1].gather(1, tgt[:, None])).abs()
            print(f"vs reference: top-1 agree {100 * agree:.1f}%, |dlogprob| mean {d.mean():.4f} max {d.max():.3f}, "
                  f"NLL ours {-lp_o[:-1].gather(1, tgt[:, None]).mean():.3f} ref "
                  f"{-lp_r[:-1].gather(1, tgt[:, None]).mean():.3f}", flush=True)
    res = eng.generate(ids, args.decode)
    if args.rank == 0:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        print(f"decode {len(res['tokens'])} tokens: {res['decode_tps']:.2f} tok/s "
              f"(prefill {res['prefill_tps']:.0f} tok/s)", flush=True)
        print("text:", repr(tok.decode(res["tokens"])), flush=True)
    nccl.barrier()


if __name__ == "__main__":
    main()
