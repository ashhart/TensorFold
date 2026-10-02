"""Run the serial two-rank DeepSeek-V4.1 engine on one golden prompt: prefill parity vs the reference, then decode speed.

    rank 1: python tools/dsv41_serial_run.py MODEL ENGRAM --rank 1 --master 10.42.0.1
    rank 0: python tools/dsv41_serial_run.py MODEL ENGRAM --rank 0 --master 10.42.0.1 --golden g.json --ref ref-6.pt
"""

from __future__ import annotations

import argparse
import json
import os
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
    ap.add_argument("--sublayer-bench", action="store_true",
                    help="after the cases: graph time of attention-only / MoE-only / HC-only passes over layers 4-39 (1 row)")
    ap.add_argument("--profile-step", type=int, default=0,
                    help="after the cases: profile this many 1-row decode graph replays (kernels inside the graphs)")
    ap.add_argument("--profile-rows", type=int, default=1, help="rows a profiled step takes (a verify window)")
    ap.add_argument("--profile-prefill", type=int, default=0, help="profile one prompt chunk of N rows and exit")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--slots", type=int, default=1, help="stream slots (concurrent decoding)")
    ap.add_argument("--step-test", type=int, default=0, help="N: decode after a prompt: greedy vs drafted rounds")
    ap.add_argument("--step-len", type=int, default=36000, help="--step-test prompt tokens (the document repeated)")
    ap.add_argument("--decoder-test", type=int, default=0, help="1: MultiDecoder (rank 1 following) vs greedy; 2: +pool")
    ap.add_argument("--multi-test", type=int, default=0, help="N steps: two streams batched vs alone (needs --slots 2)")
    ap.add_argument("--prefill-bench", default="", help="comma lengths: whole-prompt prefill time of each (fresh request)")
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
    if os.environ.get("TF_COMM", "nccl") == "rdma":                # small all-gathers over our RoCE transport
        from tensorfold.cuda.rdma import RdmaComm

        nccl = RdmaComm(nccl, args.rank)
        if args.rank == 0:
            print(f"[rank 0] all-gathers up to {nccl.rdma.slot_bytes >> 10} KiB over RoCE ({nccl.rdma.device})", flush=True)
    t0 = time.time()
    w = W.load(args.model, rank=args.rank, log=lambda *a, **k: None, draft=args.dspark > 0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"[rank {args.rank}] weights loaded in {time.time() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    eng = SerialEngine(w, Comm(nccl), str(args.engram), str(args.model / "tokenizer.json"), cap=args.cap,
                       slots=args.slots)
    nccl.barrier()
    if args.dspark:
        eng.enable_dspark(args.dspark)
    if args.graph or args.dspark:
        t0 = time.time()
        with torch.no_grad():
            eng.capture(1)
            if args.dspark:
                top = args.dspark + 1
                if os.environ.get("TF_COPY_DRAFTS", "1") != "0":         # copy drafts verify longer windows
                    top = max(top, int(os.environ.get("TF_COPY_MAX") or 15) + 1)
                for rows in range(2, top + 1):
                    eng.capture(rows)
                eng.drafter.capture()
                eng.adaptive = not args.fixed_k
        print(f"[rank {args.rank}] decode graphs captured in {time.time() - t0:.1f} s", flush=True)
    if hasattr(nccl, "settle"):
        nccl.settle()

    from tensorfold.engine.exact_sampling import Sampling

    if args.decoder_test:
        from tensorfold.cuda.kv_pool import PrefixPool
        from tensorfold.cuda.streams import Stream
        from tensorfold.families.deepseek_v41.cuda.multi import MultiDecoder

        base = json.loads(Path("notes/dsv41/golden_long2.json").read_text())["goldens"][-1]["ids"]
        doc = (base * (1 + args.step_len // len(base)))[:args.step_len]

        def share(values):
            n = torch.tensor([len(values) if args.rank == 0 else 0], dtype=torch.int64, device="cuda")
            got = torch.empty((2,), dtype=torch.int64, device="cuda")
            nccl.all_gather(n, got)
            count = int(got[0])
            if count == 0:
                return []
            mine = (torch.tensor(values, dtype=torch.int64, device="cuda") if args.rank == 0
                    else torch.zeros((count,), dtype=torch.int64, device="cuda"))
            out = torch.empty((2 * count,), dtype=torch.int64, device="cuda")
            nccl.all_gather(mine, out)
            return out[:count].tolist()

        def gather(values):
            mine = torch.tensor(values, dtype=torch.int64, device="cuda")
            out = torch.empty((2 * len(values),), dtype=torch.int64, device="cuda")
            nccl.all_gather(mine, out)
            return [out[:len(values)].tolist(), out[len(values):].tolist()]

        with torch.no_grad():
            for rows in range(2, 17):
                if rows not in eng.graphs:
                    eng.capture(rows)
            eng.select_slot(args.slots - 1)                       # the reference: whole prefill, greedy steps
            eng.reset()
            nxt = int(eng.prefill(doc)[-1].argmax())
            ref = [nxt]
            for _ in range(23):
                nxt = eng.step(nxt)
                ref.append(nxt)
            if args.decoder_test == 2:
                eng.pool = PrefixPool(4 << 30)
            dec = MultiDecoder(eng, share, rank=args.rank, drafts=3)
            dec.calibrate(gather)
            if args.rank == 0:
                s = Stream(list(doc), 24, None, draft=True, stop_eos=False)
                dec.admit(s)
                while not s.done:
                    dec.round()
                dec.finish([s])
                share([])
                print(f"decoder-test: equal {s.out == ref}\n  ref {ref[:16]}\n  dec {s.out[:16]}", flush=True)
            else:
                dec.follow()
        nccl.barrier()
        return
    if args.step_test:
        base = json.loads(Path("notes/dsv41/golden_long2.json").read_text())["goldens"][-1]["ids"]
        doc = (base * (1 + args.step_len // len(base)))[:args.step_len]
        res = {}
        with torch.no_grad():
            for mode in ("whole", "steps", "steps_drafted"):
                slot = args.slots - 1
                eng.select_slot(slot)
                eng.reset()
                if mode.startswith("steps"):
                    for p0 in range(0, len(doc), 16384):
                        logits = eng.prefill(doc[p0:p0 + 16384])
                else:
                    logits = eng.prefill(doc)
                nxt = int(logits[-1].argmax())
                toks = [nxt]
                if mode in ("whole", "steps"):
                    for _ in range(23):
                        nxt = eng.step(nxt)
                        toks.append(nxt)
                else:                                   # MultiDecoder's round: drafts, step_multi, accept, roll back
                    log = []
                    while len(toks) < 24:
                        ids = eng.views[slot].ids
                        p0 = len(ids)
                        drafts = eng.drafter.propose(nxt, p0)[:3]
                        _, target = eng.step_multi([(slot, t) for t in [nxt, *drafts]])
                        m = 0
                        while m < len(drafts) and drafts[m] == target[m]:
                            m += 1
                        del eng.views[slot].ids[p0 + 1 + m:]
                        log.append((drafts, target, m))
                        toks += drafts[:m] + [target[m]]
                        nxt = target[m]
                    if args.rank == 0:
                        print("rounds:", log[:4], flush=True)
                res[mode] = toks[:24]
        if args.rank == 0:
            print(f"step-test: steps equal {res['whole'] == res['steps']}, steps_drafted equal "
                  f"{res['whole'] == res['steps_drafted']}\n  whole {res['whole'][:16]}\n  steps {res['steps'][:16]}\n"
                  f"  s+d   {res['steps_drafted'][:16]}", flush=True)
        nccl.barrier()
        return
    if args.multi_test:
        doc = json.loads(Path("notes/dsv41/golden_long2.json").read_text())["goldens"][-1]["ids"]
        prompts = [doc[:900], doc[5000:6300]]
        N = args.multi_test
        with torch.no_grad():
            for rows in range(2, 5):
                if rows not in eng.graphs:
                    eng.capture(rows)
            alone = []
            for slot, p in enumerate(prompts):                      # each stream decoded by itself
                eng.select_slot(slot)
                eng.reset()
                nxt = int(eng.prefill(p)[-1].argmax())
                toks = [nxt]
                for _ in range(N - 1):
                    nxt = eng.step(nxt)
                    toks.append(nxt)
                alone.append(toks)
            pend = []
            for slot, p in enumerate(prompts):                      # both again, then batched rows
                eng.select_slot(slot)
                eng.reset()
                pend.append(int(eng.prefill(p)[-1].argmax()))
            both = [[pend[0]], [pend[1]]]
            for _ in range(N - 1):
                _, nxt = eng.step_multi([(0, pend[0]), (1, pend[1])])
                pend = nxt
                both[0].append(nxt[0])
                both[1].append(nxt[1])
            # two rows of stream 0 (its pending token and the alone run's next one: a verify window) beside stream 1
            eng.select_slot(0)
            eng.reset()
            a0 = int(eng.prefill(prompts[0])[-1].argmax())
            eng.select_slot(1)
            eng.reset()
            b0 = int(eng.prefill(prompts[1])[-1].argmax())
            _, win = eng.step_multi([(0, a0), (0, alone[0][1]), (1, b0)])
        if args.rank == 0:
            print(f"multi-test: stream 0 batched == alone {both[0] == alone[0]}, stream 1 {both[1] == alone[1]}; "
                  f"window rows {win[:2]} vs alone {alone[0][1:3]}, stream 1 row {win[2]} vs {alone[1][1]}", flush=True)
            print("  alone0", alone[0][:12], "\n  both0 ", both[0][:12], flush=True)
        nccl.barrier()
        return
    if args.prefill_bench:
        doc = json.loads(Path("notes/dsv41/golden_long2.json").read_text())["goldens"][-1]["ids"]
        with torch.no_grad():
            eng.reset()
            eng.prefill(doc[:2048], 2048)                         # warm kernels
            torch.cuda.synchronize()
            for n in (int(v) for v in args.prefill_bench.split(",")):
                best = 0.0
                for _ in range(2):
                    eng.reset()
                    nccl.barrier()
                    t = time.perf_counter()
                    eng.prefill(doc[:n], 2048)
                    torch.cuda.synchronize()
                    best = max(best, n / (time.perf_counter() - t))
                if args.rank == 0:
                    print(f"whole prompt {n} tokens: {best:.0f} tok/s", flush=True)
        nccl.barrier()
        return
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
                copies = (f", copy rounds {r['copy_rounds']} ({r['copy_accepted']} accepted)" if "copy_rounds" in r else "")
                extra = (f"{copies}, {r['accepted_per_round']:.2f} accepted a round (vLLM {case['accepted_per_round']:.2f}), "
                         f"draft {r['draft_ms']:.1f} ms + verify {r['verify_ms']:.1f} ms a round, k used {r['k_histogram']}"
                         if "rounds" in r else "")
                print(f"{case['name']}: {len(r['tokens'])} tokens {r['decode_tps']:.1f} tok/s{extra}; "
                      f"same tokens as vLLM: {same}", flush=True)
        if os.environ.get("TF_ROUND_PROF") and eng.graphs:     # replays at the live context vs capture-time costs
            for R in sorted(eng.graphs):
                g = eng.graphs[R]
                nccl.barrier()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(5):
                    eng._replay_free(g)
                torch.cuda.synchronize()
                if args.rank == 0:
                    print(f"live-context replay {R} rows: {(time.perf_counter() - t) / 5 * 1e3:.1f} ms "
                          f"(context {len(eng.state.ids)})", flush=True)
        if args.sublayer_bench and eng.graphs:                # where a layer's time goes, piece by piece
            c = eng.c
            g1 = eng.graphs[1]
            eng._sid = g1["sid"]
            pos = g1["pos"]
            Ls = eng.w.layers[4:40]
            x = (torch.randn((1, c.hidden_size), device="cuda") * 0.05).to(torch.bfloat16)
            X = (torch.randn((1, c.hc_mult, c.hidden_size), device="cuda") * 0.05).to(torch.bfloat16)
            pre0 = torch.full((1, c.hc_mult), 0.25, device="cuda")
            saved = eng._save_rows(eng.slot, torch.arange(0, 4), torch.arange(0, 4))
            pieces = {
                "attention": lambda: [eng.attention(L, x, pos, True) for L in Ls],
                "moe": lambda: [eng.moe(L, x, 1) for L in Ls],
                "hc pre": lambda: [eng.hc(L.hc_attn, X, pre0) for L in Ls],
            }
            for name, fn in pieces.items():
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    fn(); fn()
                torch.cuda.current_stream().wait_stream(side)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn()
                for _ in range(3):
                    gr.replay()
                nccl.barrier()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(20):
                    gr.replay()
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) / 20 * 1e3
                if args.rank == 0:
                    print(f"sublayer-bench: {name}: {ms / len(Ls) * 1e3:.1f} us a layer ({ms:.2f} ms over {len(Ls)})",
                          flush=True)
                if os.environ.get("TF_SUBPROF") and name == "attention":
                    from torch.profiler import ProfilerActivity, profile

                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(5):
                            gr.replay()
                        torch.cuda.synchronize()
                    if args.rank == 0:
                        for e in sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:30]:
                            if e.self_device_time_total > 0:
                                print(f"    {e.self_device_time_total / 5 / len(Ls):7.2f} us/layer {e.count / 5 / len(Ls):5.2f}x "
                                      f"{e.key[:90]}", flush=True)
            eng._restore_rows(saved)
        if args.profile_step and eng.graphs:                  # kernels of the 1-row decode graphs, back to back
            from torch.profiler import ProfilerActivity, profile

            g = eng.graphs[1]
            for _ in range(3):
                eng._replay_free(g)
            nccl.barrier()
            torch.cuda.synchronize()
            n = args.profile_step
            if "one" in g:                                    # the whole step as one graph vs three launches
                for label, fn in (("three launches", lambda: (eng._replay_free(g))),
                                  ("one launch", lambda: eng._replay_free(g))) * 2:
                    nccl.barrier()
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    for _ in range(n):
                        fn()
                    torch.cuda.synchronize()
                    if args.rank == 0:
                        print(f"full-graph test: {label}: {(time.perf_counter() - t) / n * 1e3:.2f} ms a step", flush=True)
            if os.environ.get("TF_NSYS"):                     # an Nsight Systems capture range instead
                torch.cuda.profiler.start()
                for _ in range(n):
                    eng._replay_free(g)
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                t = time.perf_counter()
                for _ in range(n):
                    eng._replay_free(g)
                torch.cuda.synchronize()
                wall = (time.perf_counter() - t) / n * 1e3
            if args.rank == 0:
                ev = prof.key_averages()
                busy = sum(e.self_device_time_total for e in ev) / n / 1e3
                print(f"profile-step: {wall:.2f} ms a step, kernels {busy:.2f} ms (side streams overlap: may exceed)",
                      flush=True)
                for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:40]:
                    if e.self_device_time_total > 0:
                        print(f"  {e.self_device_time_total / n / 1e3:7.3f} ms  {e.count / n:6.1f}x  {e.key[:110]}",
                              flush=True)
        if args.rank == 0 and getattr(eng, "_round_costs", None):
            print("graph round costs (ms, k = 0..n, draft included):", [round(c, 1) for c in eng._round_costs],
                  flush=True)
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
