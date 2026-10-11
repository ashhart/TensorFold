#!/usr/bin/env python3
"""Decode reference: TP=2 prompt and greedy decoding with state dumps, logits and block digests, AOT set, rope."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(t) -> str:
    import torch

    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def rank_main(rank: int, a) -> None:
    sys.path.insert(0, str(HERE))
    import shutil

    import torch

    dev_index = rank if a.rank < 0 else 0     # two hosts: each rank on its own GPU 0
    torch.cuda.set_device(dev_index)
    from dsv41_triton_capture import IndexRows, pack_aot, specialization
    from tensorfold.cuda.comm import open_comm
    from tensorfold.families.deepseek_v41.cuda import rope as RO
    from tensorfold.families.deepseek_v41.cuda import weights as W
    from tensorfold.families.deepseek_v41.cuda.config import Config
    from tensorfold.families.deepseek_v41.cuda.csa2.backend import Triton
    from tensorfold.families.deepseek_v41.cuda.engram_host import NgramHasher, token_map
    from tensorfold.families.deepseek_v41.cuda.forward import Forward, Seg, greedy

    t0 = time.time()
    out = Path(a.out)
    dev = torch.device(f"cuda:{dev_index}")
    kvmode = a.kv
    comm = open_comm(rank, 2, a.master, a.port)
    pack = Path(a.pack)
    cfg = Config.from_file(pack / "config.json")
    tmap, _ = token_map(pack / "tokenizer.json")
    src = W.Shards(pack)
    loader = W.Loader(cfg, src, rank, 2, dev)
    vh = cfg.vocab_size // 2
    embed = loader._dev(src.rows("embed.weight", rank * vh, (rank + 1) * vh).to(torch.bfloat16))
    head, norm = loader.x3("head", "col"), loader.native("norm.weight")
    lo, hi = (int(v) for v in a.layers.split("-"))
    layers = []
    for L in range(lo, hi + 1):
        layers.append(loader.layer(L))
        if rank == 0:
            print(f"loaded layer {L} at {time.time() - t0:.0f} s", flush=True)
    tree = W.RankW(rank, 2, rank * vh, embed, layers, norm, head, None, {})
    rows = {L: IndexRows(cfg.engram_head_dim) for L in cfg.engram_layer_ids}
    if a.engram_dir:               # the packed shards (engram_host.ShardRows), as served; --engram-make writes them
        from tensorfold.families.deepseek_v41.cuda.engram_host import open_shards

        if a.engram_make:
            sys.path.insert(0, str(Path(__file__).parent / "dsv41_perf"))
            import engram_shards

            ps = [torch.randint(3, cfg.vocab_size, (n,), generator=torch.Generator().manual_seed(a.seed)).tolist()
                  for n in (a.prompts or [a.prompt])]
            engram_shards.make(a.engram_dir, NgramHasher(cfg, tmap), ps, rank, 2)
        rows = open_shards(a.engram_dir, rank, 2, cfg.engram_layer_ids, dim=cfg.engram_head_dim)
    kv = None
    if kvmode == "contig":
        fw = Forward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=comm, device=dev,
                     limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap),
                     slots=len(a.prompts) if a.batched else 1, contiguous=True, pchunk=2048)
    else:                          # the slot in a (split) KV pool, bound as serving binds it
        import dsv41_kv_ref as KR
        from tensorfold.families.deepseek_v41.cuda.slots import SlotsForward

        fw = SlotsForward(cfg, tree, csa2=Triton(128, cfg.num_attention_heads // 2, dev), comm=comm, device=dev,
                          limit=a.limit, chunk=128, engram_rows=rows, hasher=NgramHasher(cfg, tmap), slots=1,
                          contiguous=False, pchunk=2048)
        kv = KR.KV(fw, pack, kvmode, rank, 2, a.limit, dev, scramble=not a.no_scramble)
    load_s = time.time() - t0
    rec = None
    both = kvmode != "contig" and a.rank < 0     # pooled KV: rank 1's Triton variants too (pack2 specializes on the rank)
    if rank == 0 or a.rank >= 0 or both:   # the AOT set (and rope tables) on each host of a two-host run, the
        import triton_aot_manifest as aot   # prompt's prefill kernels included (the Zig prefill's too)

        rec = aot.Recorder().install()
    dirs = [Path(a.out) / f"p{p}" for p in a.prompts] if a.prompts else [Path(a.out)]
    if a.batched:
        batched(rank, a, fw, cfg, greedy, Seg, digest, dirs, lo, hi, t0, load_s)
    for i, (prompt, out) in enumerate(zip([] if a.batched else (a.prompts or [a.prompt]), dirs)):
        if i > 0:
            if kv is None:
                fw.slots[0] = fw.new_slot(contiguous=True)
            else:
                kv.fresh(fw)
        one(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, i == len(dirs) - 1, kv)
    # --profile "1 4 8": per row count, 3 warm + 16 timed windows and kernel time by name into DIR/prof-rank<r>.json
    if a.profile:
        prof = {"rows": {}, "kernels": {}}
        tok = 11
        for n in [int(x) for x in a.profile.split()]:
            ts = []
            for i in range(19):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                fw.run([Seg(0, fw.slot.pos, tuple([tok] * n))])
                torch.cuda.synchronize()
                if i >= 3:
                    ts.append((time.perf_counter() - t1) * 1e3)
                fw.keep(0, n - 1)
            prof["rows"][str(n)] = {"ms_mean": sum(ts) / len(ts), "ms_min": min(ts)}
            from torch.profiler import ProfilerActivity, profile as tprofile

            with tprofile(activities=[ProfilerActivity.CUDA]) as pr:
                for _ in range(8):
                    fw.run([Seg(0, fw.slot.pos, tuple([tok] * n))])
                    fw.keep(0, n - 1)
                torch.cuda.synchronize()
            ks = {}
            for ev in pr.key_averages():
                t = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
                if t:
                    ks[ev.key] = {"calls": ev.count / 8, "ms": t / 1e3 / 8}
            prof["kernels"][str(n)] = ks
            if rank == 0:
                print(f"profile: {n}-row windows {prof['rows'][str(n)]['ms_mean']:.3f} ms mean, "
                      f"kernels {sum(v['ms'] for v in ks.values()):.3f} ms a window", flush=True)
        (Path(a.out) / f"prof-rank{rank}.json").write_text(json.dumps(prof, indent=1) + "\n")
    # --profile-prefill N: the gate's prompt shape from a fresh slot, warm / timed / profiled, into prof-rank<r>.json
    if a.profile_prefill:
        if kv is not None:
            raise SystemExit("--profile-prefill: the contiguous slot only (no --pool / --split / --park)")
        from torch.profiler import ProfilerActivity, profile as tprofile

        g = torch.Generator().manual_seed(a.seed)
        ids = torch.randint(3, cfg.vocab_size, (a.profile_prefill,), generator=g).tolist()

        def prefill_once():
            fw.slots[0] = fw.new_slot(contiguous=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            served_prompt(fw, Seg, ids, a.prompt_tail, a.prefill)
            torch.cuda.synchronize()
            return (time.perf_counter() - t1) * 1e3

        prefill_once()
        wall = prefill_once()
        with tprofile(activities=[ProfilerActivity.CUDA]) as pr:
            prefill_once()
        ks = {}
        for ev in pr.key_averages():
            t = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
            if t:
                ks[ev.key] = {"calls": ev.count, "ms": t / 1e3}
        f = Path(a.out) / f"prof-rank{rank}.json"
        doc = json.loads(f.read_text()) if f.exists() else {}
        doc["prefill"] = {"rows": len(ids), "segments": -(-len(ids) // fw.pchunk), "wall_ms": wall,
                          "gpu_ms": sum(v["ms"] for v in ks.values()), "prompt_tail": a.prompt_tail,
                          "prefill": a.prefill, "kernels": ks}
        f.write_text(json.dumps(doc, indent=1) + "\n")
        if rank == 0:
            print(f"profile: prefill of {len(ids)} ({a.prefill}, tail {a.prompt_tail}) {wall:.1f} ms wall, "
                  f"kernels {doc['prefill']['gpu_ms']:.1f} ms", flush=True)
    # --sampled: the keyed sampling reference on this load (DIR/samp/trace.jsonl), before the AOT dump
    if a.sampled:
        if kv is not None:
            raise SystemExit("--sampled: the contiguous slot only (no --pool / --split / --park)")
        import dsv41_samp_ref

        dsv41_samp_ref.run_cases(fw, cfg, rank, Path(a.out) / "samp", dsv41_samp_ref.PROMPTS, a.steps, a.seed)
    # --grammar: the structured-output reference on this load (grammar masks over serial decoding, DIR/grammar/)
    if a.grammar:
        if kv is not None:
            raise SystemExit("--grammar: the contiguous slot only (no --pool / --split / --park)")
        sys.path.insert(0, str(Path(__file__).resolve().parent / "dsv41_grammar"))
        import grammar_ref

        grammar_ref.run_cases(fw, cfg, rank, Path(a.out) / "grammar", Path(a.pack), 48, a.seed)
    # --aot-prompts: more prompt lengths, each from a fresh slot, prefilled and decoded a step for their Triton variants
    if a.aot_taps:      # as a drafting server prefills: DSpark's taps kept (the prefill `_site`'s TAP_ON variants)
        fw.keep_taps = True
    for n in a.aot_prompts:
        if kv is None:
            fw.slots[0] = fw.new_slot(contiguous=True)
        else:
            kv.fresh(fw)
        g = torch.Generator().manual_seed(a.seed + n)
        ids = torch.randint(3, cfg.vocab_size, (n,), generator=g).tolist()
        tok = int(greedy(fw, fw.prompt(ids)[-1:])[0])
        fw.run([Seg(0, fw.slot.pos, (tok,))])
        fw.keep(0, 0)
        if a.prefill == "replay":       # the replay's passes at this length too (encoder segments, the decoder's tail)
            fw.slots[0] = fw.new_slot(contiguous=True)
            tok = int(greedy(fw, replay_prompt(fw, Seg, ids))[0])
            fw.run([Seg(0, fw.slot.pos, (tok,))])
            fw.keep(0, 0)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"aot prompt of {n}: done", flush=True)
    run_s = time.time() - t0 - load_s
    top = Path(a.out)
    cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    if both and rank == 1:          # rank 1's set beside rank 0's, merged by rank 0 after the barrier
        r1 = top / "aot-rank1"
        r1.mkdir(parents=True, exist_ok=True)
        rec.dump(r1 / "launches.json")
        aot.build(cache, r1 / "launches.json", r1 / "manifest.json", None)
        pack_aot(r1 / "manifest.json", cache, r1 / "aot", specialization())
    if both:
        comm.barrier()
    if rank == 0 or a.rank >= 0:   # the AOT set and rope tables on each host of a two-host run
        tabs = RO.tables(cfg, a.limit + 2048)
        (top / "rope.bin").write_bytes(b"DSV41RP1" + struct.pack("<II", *tabs[0].cs.shape)
                                       + tabs[0].cs.contiguous().numpy().tobytes()
                                       + tabs[1].cs.contiguous().numpy().tobytes())
        rec.dump(top / "launches.json")
        aot.build(cache, top / "launches.json", top / "manifest.json", None)
        pack_aot(top / "manifest.json", cache, top / "aot", specialization())
        if both:
            print(f"rank 0: rank 1's AOT set merged: {merge_aot(top / 'aot', top / 'aot-rank1' / 'aot')}", flush=True)
        for d in dirs:
            if d != top:
                shutil.copy(top / "rope.bin", d / "rope.bin")
                shutil.copytree(top / "aot", d / "aot", dirs_exist_ok=True)
        print(f"rank 0: {len(dirs)} prompt(s), load {load_s:.0f} s, run {run_s:.0f} s", flush=True)
    comm.barrier()


def merge_aot(dst: Path, src: Path) -> dict:
    """The kernels of AOT set ``src`` that ``dst`` lacks (by hash), with their cubins (dsv41_aot_merge.py's rule)."""

    import shutil

    a = json.loads((dst / "aot.json").read_text())
    b = json.loads((src / "aot.json").read_text())
    have = {k["hash"] for k in a["kernels"]}
    added = 0
    for k in b["kernels"]:
        if k["hash"] in have:
            continue
        shutil.copyfile(src / "cubins" / f"{k['hash']}.cubin", dst / "cubins" / f"{k['hash']}.cubin")
        a["kernels"].append(k)
        have.add(k["hash"])
        added += 1
    (dst / "aot.json").write_text(json.dumps(a, indent=1) + "\n")
    return {"added": added, "kernels": len(a["kernels"])}


def decode(fw, greedy, Seg, digest, tok, steps, kv, trace) -> tuple[list, list, dict]:
    """``steps`` greedy windows of one row from ``tok``: (tokens, logits digests, the first window's block digests)."""

    tokens, digests, layers_d = [tok], [], {}
    for step in range(steps):
        fw.trace = {} if step == 0 and trace else None
        if kv is not None:
            kv.grow(fw.slot.pos + 1)
        lg = fw.run([Seg(0, fw.slot.pos, (tok,))])
        if step == 0 and trace:
            layers_d = {str(L): digest(t) for L, t in sorted(fw.trace.items())}
        digests.append(digest(lg))
        tok = int(greedy(fw, lg)[0])
        tokens.append(tok)
        fw.keep(0, 0)
    fw.trace = None
    return tokens, digests, layers_d


def served_prompt(fw, Seg, ids, tail: str = "prefill", prefill: str = "full"):
    """The prompt's choice logits [1, V / W] in ``tail``'s shape: prefill, verify (last token a decode row), replay."""

    if prefill == "replay":
        if tail != "verify":
            raise ValueError("--prefill replay: the served engine's shape only (--prompt-tail verify)")
        return replay_prompt(fw, Seg, ids)
    if tail == "prefill":
        return fw.prompt(ids)[-1:]
    if len(ids) > 1:
        fw.prompt(ids[:-1])
    lg = fw.run([Seg(0, fw.slot.pos, (ids[-1],))])
    fw.keep(0, 0)
    return lg[-1:]


def replay_prompt(fw, Seg, ids):
    """``TF_DSV41_PREFILL=replay``: bounded-replay encode over [0, n - 1), finish over the tail, then the verify row."""

    from tensorfold.families.deepseek_v41.cuda import replay as RP
    from tensorfold.families.deepseek_v41.cuda.protocol import REPLAY, Piece
    from tensorfold.families.deepseek_v41.cuda.slots import prefill_runs

    if not isinstance(getattr(fw, "stash", None), dict):    # a plain Forward: SlotsForward's stash dict
        fw.stash = {}
    fw.stash.pop(0, None)
    n, pos = len(ids), fw.slot.pos
    if n > 1:
        RP.encode(fw, prefill_runs([Piece(0, pos, tuple(ids[:-1]))], fw.pchunk))
        RP.finish(fw, 0, pos + n - 1, tuple(ids[max(0, n - REPLAY):n - 1]))
    lg = fw.run([Seg(0, fw.slot.pos, (ids[-1],))])
    fw.keep(0, 0)
    return lg[-1:]


def prompt_ids(a, cfg, prompt: int) -> list[int]:
    import torch

    g = torch.Generator().manual_seed(a.seed)
    return torch.randint(3, cfg.vocab_size, (prompt,), generator=g).tolist()


def batched(rank, a, fw, cfg, greedy, Seg, digest, dirs, lo, hi, t0, load_s) -> None:
    """``--batched``: every prompt admitted in one round's encoder run, then each slot's final, then each as ``one``."""

    from tensorfold.families.deepseek_v41.cuda import replay as RP
    from tensorfold.families.deepseek_v41.cuda.protocol import REPLAY, Piece
    from tensorfold.families.deepseek_v41.cuda.slots import prefill_runs

    if a.prefill != "replay" or a.prompt_tail != "verify":
        raise ValueError("--batched: the served engine's shape in replay mode only (--prefill replay --prompt-tail verify)")
    idss = [prompt_ids(a, cfg, p) for p in a.prompts]
    if not isinstance(getattr(fw, "stash", None), dict):
        fw.stash = {}
    runs = prefill_runs([Piece(j, 0, tuple(ids[:-1])) for j, ids in enumerate(idss)], fw.pchunk)
    if rank == 0:
        print(f"batched: {len(idss)} prompts, {len(runs)} encoder run(s) of {[len(r) for r in runs]} segments", flush=True)
    RP.encode(fw, runs)
    for j, ids in enumerate(idss):
        n = len(ids)
        RP.finish(fw, j, n - 1, tuple(ids[max(0, n - REPLAY):n - 1]))
    for j, (prompt, out) in enumerate(zip(a.prompts, dirs)):
        fw.slots[0], fw.slots[j] = fw.slots[j], fw.slots[0]     # slot j's state where `one` reads it
        one(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, j == len(dirs) - 1, prefilled=True)
        fw.slots[0], fw.slots[j] = fw.slots[j], fw.slots[0]


def one(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, last, kv=None,
        prefilled: bool = False) -> None:
    """One prompt from the slot as it is: state dump, greedy steps and digests (``prefilled``: tail only)."""

    import torch

    if kv is not None:
        one_kv(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, last, kv)
        return
    ids = prompt_ids(a, cfg, prompt)
    t1 = time.time()
    if prefilled:
        logits = fw.run([Seg(0, fw.slot.pos, (ids[-1],))])
        fw.keep(0, 0)
        logits = logits[-1:]
    else:
        logits = served_prompt(fw, Seg, ids, a.prompt_tail, a.prefill)
    tok = int(greedy(fw, logits[-1:])[0])
    torch.cuda.synchronize()
    if rank == 0:
        print(f"prompt of {prompt} at {time.time() - t0:.0f} s, first token {tok} (tail {a.prompt_tail}, prefill {a.prefill})", flush=True)
    # the slot after the prompt, by our forward's role names (block.zig: s.L<i>.swa / comp .v .s, .ik, .carry)
    sd = out / f"rank{rank}" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    roles, s0 = {}, fw.slot

    def put(role, t):
        f = sd / f"{role}.bin"
        f.write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        roles[role] = f.name

    for L in range(lo, hi + 1):
        v, sc = s0.swa[L].kernel()
        put(f"s.L{L}.swa.v", v)
        put(f"s.L{L}.swa.s", sc)
        if L in s0.comp:
            v, sc = s0.comp[L].kernel()
            put(f"s.L{L}.comp.v", v)
            put(f"s.L{L}.comp.s", sc)
            put(f"s.L{L}.ik", s0.ikeys[L].kernel())
        if L in s0.carry:
            put(f"s.L{L}.carry", s0.carry[L])
    (sd / "state.json").write_text(json.dumps({"pos": s0.pos, "tail": list(s0.tail), "roles": roles}, indent=1) + "\n")
    tokens, digests, layers_d = [tok], [], {}
    for step in range(a.steps):
        fw.trace = {} if step == 0 else None
        lg = fw.run([Seg(0, fw.slot.pos, (tok,))])
        if step == 0:
            layers_d = {str(L): digest(t) for L, t in sorted(fw.trace.items())}
        digests.append(digest(lg))
        tok = int(greedy(fw, lg)[0])
        tokens.append(tok)
        fw.keep(0, 0)
    fw.trace = None
    # windows of 2.. rows after every recorded step: their Triton variants join the AOT set (nothing recorded)
    if last:
        for n in a.extra_rows:
            fw.run([Seg(0, fw.slot.pos, tuple([tok] * n))])
            fw.keep(0, n - 1)
    torch.cuda.synchronize()
    mine = {"rank": rank, "load_s": round(load_s, 1), "run_s": round(time.time() - t1, 1), "tokens": tokens,
            "logits_sha256": digests, "layers_first_window": layers_d, "prompt_ids": ids, "prompt_tail": a.prompt_tail,
            "prefill": a.prefill}
    (out / f"rank{rank}.json").write_text(json.dumps(mine) + "\n")


def one_kv(rank, a, fw, cfg, greedy, Seg, digest, prompt, out, lo, hi, t0, load_s, last, kv) -> None:
    """``one`` on a pool slot: pages mapped before each run, state by the pool's roles; ``--park``: session trip."""

    import torch

    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(3, cfg.vocab_size, (prompt,), generator=g).tolist()
    t1 = time.time()
    kv.grow(len(ids))
    logits = fw.prompt(ids)
    tok = int(greedy(fw, logits[-1:])[0])
    torch.cuda.synchronize()
    if rank == 0:
        print(f"prompt of {prompt} at {time.time() - t0:.0f} s, first token {tok} ({kv.mode} pool)", flush=True)
    sd = out / f"rank{rank}" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    roles, shapes, s0 = {}, {}, fw.slot

    def put(role, t):
        f = sd / f"{role}.bin"
        f.write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        roles[role] = f.name
        shapes[role] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape)}

    for L in range(lo, hi + 1):
        v, sc = s0.swa[L].kernel()
        put(f"s.L{L}.swa.v", v)
        put(f"s.L{L}.swa.s", sc)
        if L in s0.carry:
            put(f"s.L{L}.carry", s0.carry[L])
    kvinfo = kv.dump(fw, put, sd)
    (sd / "state.json").write_text(json.dumps({"pos": s0.pos, "tail": list(s0.tail), "roles": roles,
                                               "shapes": shapes, "kv": kvinfo}, indent=1) + "\n")
    park = None
    if a.park:
        import dsv41_kv_ref as KR

        park = KR.Park(kv, out / "park")
        park.save_park(fw, ids)
        if rank == 0:
            print(f"parked {park.info.get('file_bytes', 0) / 2**20:.1f} MiB in {park.info.get('park_s')} s", flush=True)
    tokens, digests, layers_d = decode(fw, greedy, Seg, digest, tok, a.steps, kv, True)
    mine = {"rank": rank, "load_s": round(load_s, 1), "tokens": tokens, "logits_sha256": digests,
            "layers_first_window": layers_d, "prompt_ids": ids, "kv": kv.mode}
    if park is not None:
        park.restore(fw, ids + [tok])
        t2, d2, _ = decode(fw, greedy, Seg, digest, tok, a.steps, kv, False)
        park.info.update(tokens=t2, logits_sha256=d2, equal=bool(t2 == tokens and d2 == digests))
        if rank == 0:
            print(f"restored in {park.info.get('restore_s')} s: {a.steps} steps "
                  f"{'== the uninterrupted run' if park.info['equal'] else 'DIFFER from the uninterrupted run'}",
                  flush=True)
        park.close()
        mine["park"] = park.info
    if last:
        tok = mine["park"]["tokens"][-1] if park is not None else tokens[-1]
        for n in a.extra_rows:
            kv.grow(fw.slot.pos + n)
            fw.run([Seg(0, fw.slot.pos, tuple([tok] * n))])
            fw.keep(0, n - 1)
    torch.cuda.synchronize()
    if kv.kvx is not None:
        mine["exchange"] = kv.kvx.describe()
    mine["run_s"] = round(time.time() - t1, 1)
    (out / f"rank{rank}.json").write_text(json.dumps(mine) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-24")
    ap.add_argument("--prompt", type=int, default=1536)
    ap.add_argument("--engram-dir", default="", help="Engram rows from these packed shards (the Zig ranks' TF_DSV41_ENGRAM_DIR), "
                    "not the index function")
    ap.add_argument("--engram-make", action="store_true", help="write --engram-dir first: sparse shards, records under the prompts' rows")
    ap.add_argument("--profile", default="", help="row counts (\"1 4 8\"): time windows of each after the run")
    ap.add_argument("--profile-prefill", type=int, default=0,
                    help="after the run: an N-id prompt (the gate's ids at N = --prompt) prefilled from a fresh slot, timed "
                         "and under torch.profiler, into DIR/prof-rank<r>.json's \"prefill\" (gap.py prefill)")
    ap.add_argument("--aot-prompts", type=lambda v: [int(x) for x in v.split(",") if x], default=[],
                    help="more prompt lengths after the run, for their Triton variants only (e.g. 1,9,17,45,64,65,173,300,1025,2049,2600)")
    ap.add_argument("--aot-taps", action="store_true",
                    help="--aot-prompts with DSpark's taps kept (Forward taps=True), for a drafting server's prefill variants")
    ap.add_argument("--prompts", type=lambda v: [int(x) for x in v.split(",") if x], default=[],
                    help="several prompts in one load, each from a fresh slot into DIR/p<N> (with --rank: this host's rank)")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--batched", action="store_true",
                    help="--prompts in one batched round (the served batcher: one encoder run over every prompt's piece, "
                         "slots.prefill_runs), each from its own slot; needs --prefill replay --prompt-tail verify")
    ap.add_argument("--prompt-tail", choices=("prefill", "verify"), default="prefill",
                    help="verify: the prompt as the served engine runs it ([0, n - 1) prefilled, the last token a decode row; "
                         "the Zig side's TF_DSV41_PROMPT_TAIL default); prefill: Forward.prompt (contiguous slot only)")
    ap.add_argument("--prefill", choices=("full", "replay"), default="full",
                    help="replay: the served TF_DSV41_PREFILL=replay (CED bounded replay; needs --prompt-tail verify); the Zig "
                         "prefill gate's ranks then need TF_DSV41_PREFILL=replay")
    ap.add_argument("--grammar", action="store_true",
                    help="also the structured-output gate's replies (dsv41_grammar/grammar_ref.py) into DIR/grammar/trace.jsonl")
    ap.add_argument("--sampled", action="store_true",
                    help="also the keyed sampling gate's replies (dsv41_samp_ref.run_cases) into DIR/samp/trace.jsonl")
    ap.add_argument("--limit", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=4101)
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--rank", type=int, default=-1, help="two hosts: this host's rank (one process); -1: both ranks here")
    ap.add_argument("--extra-rows", type=lambda v: [int(x) for x in v.split(",") if x], default=[2, 3, 4, 5, 6, 16],
                    help="windows of these rows after the steps (their Triton variants in the AOT set)")
    ap.add_argument("--master", default="127.0.0.1", help="rank 0's address (two hosts: the head's fabric address)")
    ap.add_argument("--pool", action="store_true", help="KV pool: the slot in a replicated KV pool (paged stores)")
    ap.add_argument("--split", action="store_true", help="KV pool: TF_DSV41_KV_SPLIT=1, the pool split over the ranks")
    ap.add_argument("--compact", default=None, help="KV pool, --split: TF_DSV41_KV_SPLIT_COMPACT (0 / 1 / force)")
    ap.add_argument("--park", action="store_true", help="KV pool: save / park / restore after the prompt (implies --pool)")
    ap.add_argument("--no-scramble", action="store_true", help="KV pool: no page holder (the slot's pages 0, 1, 2 ...)")
    a = ap.parse_args()
    a.kv = "split" if a.split else ("pool" if a.pool or a.park else "contig")
    if a.prompt_tail == "verify" and a.kv != "contig":
        ap.error("--prompt-tail verify: the contiguous slot only (one_kv prefills with Forward.prompt)")
    if a.prefill == "replay" and a.prompt_tail != "verify":
        ap.error("--prefill replay: the served engine's shape only (--prompt-tail verify)")
    if a.compact is not None:
        if not a.split:
            ap.error("--compact needs --split")
        os.environ["TF_DSV41_KV_SPLIT_COMPACT"] = a.compact
    if a.split:
        os.environ["TF_DSV41_KV_SPLIT"] = "1"           # both ranks (spawned with this environment)
    elif a.kv == "pool":
        os.environ.pop("TF_DSV41_KV_SPLIT", None)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.rank >= 0:                         # two hosts: this host's rank alone (the other host runs the other)
        rank_main(a.rank, a)
        for prompt in a.prompts or [a.prompt]:      # --prompts: each prompt's DIR/p<N> as a one-prompt run's DIR
            d = Path(a.out) / f"p{prompt}" if a.prompts else Path(a.out)
            mine = json.loads((d / f"rank{a.rank}.json").read_text())
            ranks = [None, None]
            ranks[a.rank] = {k: mine[k] for k in ("load_s", "run_s", "logits_sha256", "layers_first_window")}
            meta = {"layers": a.layers, "prompt": prompt, "steps": a.steps, "seed": a.seed, "limit": a.limit,
                    "tokens": mine["tokens"], "prompt_ids": mine["prompt_ids"], "host_rank": a.rank,
                    "prompt_tail": mine.get("prompt_tail", "prefill"), "prefill": mine.get("prefill", "full"),
                    "env": {k: v for k, v in os.environ.items() if k.startswith("TF_DSV41")}, "ranks": ranks}
            if a.kv != "contig":
                meta["kv"] = a.kv
                if "park" in mine:
                    ranks[a.rank]["park"] = mine["park"]
            (d / "ref.json").write_text(json.dumps(meta, indent=1) + "\n")
            print(json.dumps({"rank": a.rank, "prompt": prompt, "tokens": len(mine["tokens"]), "first": mine["tokens"][:8]}), flush=True)
        return 0
    import torch.multiprocessing as mp

    mp.spawn(rank_main, args=(a,), nprocs=2, join=True)
    st = 0
    for prompt in a.prompts or [a.prompt]:
        d = Path(a.out) / f"p{prompt}" if a.prompts else Path(a.out)
        r0 = json.loads((d / "rank0.json").read_text())
        r1 = json.loads((d / "rank1.json").read_text())
        if r0["tokens"] != r1["tokens"]:
            print(f"prompt {prompt}: the ranks' greedy tokens differ", flush=True)
            st = 1
            continue
        meta = {"layers": a.layers, "prompt": prompt, "steps": a.steps, "seed": a.seed, "limit": a.limit,
                "tokens": r0["tokens"], "prompt_ids": r0["prompt_ids"], "prompt_tail": r0.get("prompt_tail", "prefill"),
                "prefill": r0.get("prefill", "full"),
                "env": {k: v for k, v in os.environ.items() if k.startswith("TF_DSV41")},
                "ranks": [{k: r[k] for k in ("load_s", "run_s", "logits_sha256", "layers_first_window")} for r in (r0, r1)]}
        if a.kv != "contig":            # pooled KV: the pool mode, the exchange's counts, the park round trip a rank
            meta["kv"] = a.kv
            for m, r in zip(meta["ranks"], (r0, r1)):
                for k in ("park", "exchange"):
                    if k in r:
                        m[k] = r[k]
            if "park" in r0:
                eq = bool(r0["park"].get("equal") and r1.get("park", {}).get("equal"))
                meta["park_equal"] = eq
                if not eq:
                    print(f"prompt {prompt}: the restored session's steps differ from the uninterrupted run", flush=True)
                    st = 1
        (d / "ref.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps({"prompt": prompt, "tokens": len(meta["tokens"]), "first": meta["tokens"][:8],
                          **({"park_equal": meta["park_equal"]} if "park_equal" in meta else {})}), flush=True)
    return st


if __name__ == "__main__":
    raise SystemExit(main())
