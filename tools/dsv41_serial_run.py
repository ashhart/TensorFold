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
    ap.add_argument("--profile", action="store_true", help="profile 4 decode steps after the run (rank 0 prints)")
    ap.add_argument("--profile-rows", type=int, default=1, help="rows a profiled step takes (a verify window)")
    ap.add_argument("--profile-prefill", type=int, default=0, help="profile one prompt chunk of N rows and exit")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--graph", action="store_true", help="capture the one-row decode step as a CUDA graph")
    ap.add_argument("--dspark", type=int, default=0, help="draft N tokens a round with the checkpoint's DSpark blocks")
    ap.add_argument("--cap", type=int, default=1024, help="context capacity (cache rows)")
    ap.add_argument("--long-golden", type=Path, help="tools/dsv41_golden_long.py output: prefill parity per prefix")
    ap.add_argument("--save", type=Path, help="rank 0 writes each case's generated tokens here (JSON)")
    ap.add_argument("--temperature", type=float, default=0.0, help="position-keyed sampling (0: greedy)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fixed-k", action="store_true", help="verify every draft each round (no adaptive policy)")
    ap.add_argument("--cases", type=Path, help="tools/dsv41_vllm_accept.py results: replay every case's prompt ids")
    ap.add_argument("--stack-after", type=int, default=0, help="dump every thread's stack after N seconds")
    args = ap.parse_args()
    if args.stack_after:
        import faulthandler
        import sys

        faulthandler.dump_traceback_later(args.stack_after, repeat=True, file=sys.stderr)

    torch.cuda.set_device(0)
    nccl = NCCL(args.rank, 2, args.master, args.port)
    nccl.barrier()
    t0 = time.time()
    w = W.load(args.model, rank=args.rank, log=lambda *a, **k: None, draft=args.dspark > 0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"[rank {args.rank}] weights loaded in {time.time() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    eng = SerialEngine(w, Comm(nccl), str(args.engram), str(args.model / "tokenizer.json"), cap=args.cap)
    nccl.barrier()
    if args.dspark:
        eng.enable_dspark(args.dspark)
    if args.graph or args.dspark:
        t0 = time.time()
        with torch.no_grad():
            eng.capture(1)
            if args.dspark:
                for rows in range(2, args.dspark + 2):
                    eng.capture(rows)
                eng.drafter.capture()
                eng.adaptive = not args.fixed_k
        print(f"[rank {args.rank}] decode graphs captured in {time.time() - t0:.1f} s", flush=True)

    from tensorfold.engine.exact_sampling import Sampling

    if args.profile_prefill:
        from torch.profiler import ProfilerActivity, profile

        doc = json.loads(Path("notes/dsv41/golden_long2.json").read_text())["goldens"][-1]["ids"]
        n = args.profile_prefill
        with torch.no_grad():
            eng.reset()
            eng.prefill(doc[:n], n)                               # warm: kernels, Engram pages
            torch.cuda.synchronize()
            t = time.perf_counter()
            eng.prefill(doc[n:5 * n], n)                          # four chunks, rows read ahead
            torch.cuda.synchronize()
            steady = time.perf_counter() - t
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                t = time.perf_counter()
                eng.prefill(doc[5 * n:6 * n], n)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t
        if args.rank == 0:
            ev = prof.key_averages()
            gpu = sum(e.self_device_time_total for e in ev) / 1e3
            print(f"prefill {4 * n} tokens in chunks of {n}: {4 * n / steady:.0f} tok/s; profiled chunk wall "
                  f"{wall * 1e3:.0f} ms, GPU busy {gpu:.0f} ms", flush=True)
            print(ev.table(sort_by="self_device_time_total", row_limit=25, max_name_column_width=60), flush=True)
        nccl.barrier()
        return

    sampling = Sampling(seed=args.seed, temperature=args.temperature, top_k=0, top_p=0.95) \
        if args.temperature > 0 else None
    if args.long_golden:
        for g in json.loads(args.long_golden.read_text())["goldens"]:
            ids = g["ids"]
            nll, best = [], []                          # per position, reduced on the GPU (full logits are GBs)
            nxt = torch.tensor(ids[1:] + [0], device="cuda")
            with torch.no_grad():
                eng.reset()
                t1 = time.time()
                for i in range(0, len(ids), MAX_ROWS):
                    lp = torch.log_softmax(eng.forward(ids[i:i + MAX_ROWS]).float(), -1)
                    nll.append(-lp.gather(1, nxt[i:i + lp.shape[0], None]).squeeze(1).cpu())
                    best.append(lp.argmax(-1).cpu())
                torch.cuda.synchronize()
                t2 = time.time()
            if args.rank == 0:
                ours = torch.cat(nll)[:-1]
                theirs = torch.tensor([-a for a in g["prompt_actual"][1:]])
                top = torch.tensor(g["prompt_top1"][1:])
                agree = (torch.cat(best)[:-1] == top).float()
                q = len(ids) // 4
                print(f"{len(ids)} tokens: prefill {len(ids) / (t2 - t1):.0f} tok/s, top-1 agree "
                      f"{100 * agree.mean():.1f}% (last quarter {100 * agree[-q:].mean():.1f}%), NLL ours "
                      f"{ours.mean():.3f} vLLM {theirs.mean():.3f} (last quarter {ours[-q:].mean():.3f} vs "
                      f"{theirs[-q:].mean():.3f})", flush=True)
        nccl.barrier()
        return
    ids = json.loads(args.golden.read_text())["goldens"][args.prompt]["ids"]
    with torch.no_grad():
        if args.no_parity:
            ids_parity = []
        else:
            ids_parity = ids
        eng.reset()
        t1 = time.time()
        logits = torch.cat([eng.forward(ids_parity[i:i + MAX_ROWS]) for i in range(0, len(ids_parity), MAX_ROWS)]) \
            if ids_parity else None
        torch.cuda.synchronize()
        t2 = time.time()
    if args.rank == 0 and ids_parity:
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
    if args.cases:
        cases = json.loads(args.cases.read_text())["results"]
        saved_tokens = {}
        for case in cases:
            r = eng.generate(case["ids"], args.decode, sampling=sampling)
            saved_tokens[case["name"]] = r["tokens"]
            if args.rank == 0:
                same = case.get("out_ids") is not None and r["tokens"][:len(case["out_ids"])] == case["out_ids"][:len(
                    r["tokens"])]
                extra = (f", {r['accepted_per_round']:.2f} accepted a round (vLLM {case['accepted_per_round']:.2f}), "
                         f"draft {r['draft_ms']:.1f} ms + verify {r['verify_ms']:.1f} ms a round, k used {r['k_histogram']}"
                         if "rounds" in r else "")
                print(f"{case['name']}: {len(r['tokens'])} tokens {r['decode_tps']:.1f} tok/s{extra}; "
                      f"same tokens as vLLM: {same}", flush=True)
        if args.save and args.rank == 0:
            args.save.write_text(json.dumps(saved_tokens))
        nccl.barrier()
        return
    res = eng.generate(ids, args.decode)
    if args.rank == 0:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        print(f"decode {len(res['tokens'])} tokens: {res['decode_tps']:.2f} tok/s "
              f"(prefill {res['prefill_tps']:.0f} tok/s)", flush=True)
        if "rounds" in res:
            print(f"dspark: {res['rounds']} rounds, {res['accepted_per_round']:.2f} drafts accepted a round, "
                  f"{res['tokens_per_round']:.2f} tokens a round", flush=True)
        print("text:", repr(tok.decode(res["tokens"])), flush=True)
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t = time.perf_counter()
            R = args.profile_rows
            for k in range(4):
                eng.forward(res["tokens"][k * R:(k + 1) * R])
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t) / 4
        if args.rank == 0:
            events = prof.key_averages()
            gpu = sum(e.self_device_time_total for e in events) / 4 / 1e3
            launches = sum(e.count for e in events if e.self_device_time_total > 0) / 4
            print(f"profile: wall {wall * 1e3:.1f} ms/step, GPU busy {gpu:.1f} ms/step, ~{launches:.0f} kernels/step",
                  flush=True)
            print(events.table(sort_by="self_device_time_total", row_limit=45, max_name_column_width=60), flush=True)
    nccl.barrier()


if __name__ == "__main__":
    main()
