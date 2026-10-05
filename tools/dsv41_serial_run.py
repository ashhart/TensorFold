"""Run the serial two-rank DeepSeek-V4.1 engine on one golden prompt: prefill parity vs the reference, then decode speed.

    rank 1: python tools/dsv41_serial_run.py MODEL ENGRAM --rank 1 --master 10.42.0.1
    rank 0: python tools/dsv41_serial_run.py MODEL ENGRAM --rank 0 --master 10.42.0.1 --golden g.json --ref ref-6.pt

Several checks in one process, the weights loaded once (tools/dsv41_suite.py: tests, groups, tiers):

    ... --suite quick --group q          # one group's tests; tools/dsv41_suite2.sh runs every group of a tier

A test in a suite parses the argument list a fresh run of it gets and starts from the request state a fresh engine
has (``fresh_state``): its ``[result]`` line (status and a fingerprint of its tokens / diffs / hashes) is the one the
fresh run prints. Every mode prints a ``[result]`` line; ``--suite`` adds a ``[suite]`` summary on both ranks.

Memory (tools/dsv41_memguard.py): a run refuses to start when MemAvailable is under TF_MEM_MIN_START_GIB and caps
torch's allocator at MemAvailable less TF_MEM_RESERVE_GIB + TF_MEM_SLACK_GIB (TF_MEM_CAP_GIB=0: no cap).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # the suite and memory-guard helpers beside this file

import dsv41_memguard as MG
import dsv41_suite as SU

# the engine's modules, imported by main (the parser and the suite plumbing import without CUDA)
torch = NCCL = W = Comm = SerialEngine = None
KV_MODE, MAX_ROWS = None, 2048


def _engine_modules() -> None:
    global torch, NCCL, W, Comm, SerialEngine, KV_MODE, MAX_ROWS
    import torch as _torch

    from tensorfold.cuda.comm import NCCL as _NCCL
    from tensorfold.families.deepseek_v41.cuda import weights as _W
    from tensorfold.families.deepseek_v41.cuda.serial import KV_MODE as _KV
    from tensorfold.families.deepseek_v41.cuda.serial import MAX_ROWS as _MR
    from tensorfold.families.deepseek_v41.cuda.serial import Comm as _Comm
    from tensorfold.families.deepseek_v41.cuda.serial import SerialEngine as _SE

    torch, NCCL, W, Comm, SerialEngine, KV_MODE, MAX_ROWS = _torch, _NCCL, _W, _Comm, _SE, _KV, _MR
    if os.environ.get("TF_STUDY_TIE") == "0":       # (fp4 study) the prompt top-k without the lower-index tie keys
        from tensorfold.families.deepseek_v41.cuda import kernels as _K

        _K.TIE_KEYS = False


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--views-test", type=int, default=0, help="N: after an N-token prefill, the kept state's bytes "
                    "(kept_views) hashed on both ranks and compared")
    ap.add_argument("--resume-test", default="", help="N,Q: a kept N-token state resumed (COPY to another extent, "
                    "its bytes through a kept_views round trip, TAKEOVER) and Q more prompt tokens vs a fresh prefill: "
                    "16 greedy tokens each (needs --slots 2)")
    ap.add_argument("--tf-compare", default="", help="comma lengths: teacher-forced scoring of the last --tf-score "
                    "positions of four documents of each length (this repo's markdown and code, the golden); rank 0 saves per-position "
                    "NLL / top-k to --out (tools/dsv41_tf_compare.py compares two runs)")
    ap.add_argument("--tf-score", type=int, default=None, help="--tf-compare: positions scored a document (default: "
                    "one prompt chunk, serial.MAX_ROWS)")
    ap.add_argument("--tf-rows", type=int, default=0, help="--tf-compare: score them in calls of this many rows (<= 32: "
                    "the decode / verify attention path) instead of one prompt chunk")
    ap.add_argument("--needle", default="", help="comma lengths: a magic number at --needle-depths of repo code filler, "
                    "asked for after it (chat template, greedy)")
    ap.add_argument("--needle-trials", type=int, default=1, help="--needle: trials a depth (different numbers)")
    ap.add_argument("--needle-depths", default="0.1,0.5,0.9", help="--needle: where the number sits (fractions)")
    ap.add_argument("--out", type=Path, help="--tf-compare / --needle: rank 0 writes results here")
    ap.add_argument("--decode-bench", default="", help="comma lengths: one prefill (timed), its state copied into "
                    "every slot, then greedy decode steps over all --slots streams timed (graphs)")
    ap.add_argument("--chunk-test", default="", help="N,k[,k..]: a prompt's rows as one chunk vs split at k: the "
                    "first sublayer where a row's values depend on its chunk (and, with --graph, a <=32-row tail)")
    # several tests, one weight load (tools/dsv41_suite.py)
    ap.add_argument("--suite", default="", help="quick | full | long | test names (comma separated): run them in this "
                    "process; the construction (--cap/--slots/--graph/--dspark) comes from the group")
    ap.add_argument("--group", default="", help="--suite: the group to run when the tests span several (q, fp8, long)")
    ap.add_argument("--suite-out", type=Path, help="--suite: rank 0 writes the results here (JSON)")
    ap.add_argument("--suite-baseline", type=Path, help="--suite: an earlier --suite-out to judge tf-compare and "
                    "decode-bench against")
    # loading and memory
    ap.add_argument("--draft-weights", choices=("auto", "on", "off"), default="auto",
                    help="load the DSpark blocks: auto = with --dspark, or when TF_DSV41_PREPARED holds a current folder "
                    "built with them (make prepare's: the 24 s read instead of the 82 s build)")
    ap.add_argument("--mem-cap-gib", default=None, help="torch allocator cap: auto | 0 (none) | GiB (TF_MEM_CAP_GIB)")
    ap.add_argument("--mem-reserve-gib", type=float, default=None, help="GiB kept available (TF_MEM_RESERVE_GIB, 3)")
    ap.add_argument("--mem-slack-gib", type=float, default=None,
                    help="GiB for allocations outside torch's allocator (TF_MEM_SLACK_GIB, 3)")
    ap.add_argument("--mem-min-start-gib", type=float, default=None,
                    help="refuse to start below this MemAvailable (TF_MEM_MIN_START_GIB, 100; 0: never)")
    ap.add_argument("--run-tag", default="", help="names this run's processes for tools/dsv41_memwatch.sh (run2.sh)")
    return ap


# flags that never change what a test computes (where it runs, what it is called, where results go)
NEUTRAL = {"rank", "master", "port", "out", "save", "stack_after", "suite", "group", "suite_out", "suite_baseline",
           "draft_weights", "mem_cap_gib", "mem_reserve_gib", "mem_slack_gib", "mem_min_start_gib", "run_tag",
           "model", "engram"}


def test_args(ap: argparse.ArgumentParser, args: argparse.Namespace, test: SU.Test) -> argparse.Namespace:
    """The namespace a fresh run of ``test`` parses (same model, ranks and output flags as this run)."""

    base = [str(args.model), str(args.engram), "--rank", str(args.rank), "--master", args.master, "--port",
            str(args.port)]
    return ap.parse_args(base + test.argv())


def match_test(ap: argparse.ArgumentParser, args: argparse.Namespace, env: dict) -> str | None:
    """The suite test this single run is (same flags but the neutral ones, same mode environment), or None."""

    mine = {k: v for k, v in vars(args).items() if k not in NEUTRAL}
    for t in SU.TESTS:
        theirs = {k: v for k, v in vars(test_args(ap, args, t)).items() if k not in NEUTRAL}
        if theirs == mine and SU.check_env(t.group, env) is None and \
                all(env.get(k, "") == dict(t.env).get(k, "") for k in SU.MODE_ENV):
            if t.group != "fp8" and env.get("TF_DSV41_KV", "") not in ("", "fp4"):
                continue
            return t.name
    return None


# -- loading -------------------------------------------------------------------------------------------------------
def want_draft(args) -> bool:
    """Whether to load the DSpark blocks (see --draft-weights)."""

    if args.draft_weights != "auto":
        return args.draft_weights == "on"
    if args.dspark:
        return True
    from tensorfold.families.deepseek_v41.cuda import fastboot

    root = fastboot.prepared_root()
    if root is None:
        return False
    for draft in (False, True):                  # a current folder of this exact tree first, then make prepare's
        key = fastboot.weights_key(args.model, args.rank, W.WORLD, draft=draft)
        if fastboot.valid(fastboot.folder(root, key), key)[0]:
            return draft
    return False


def make_engine(args, nccl, w):
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
    return eng


def fresh_state(eng) -> None:
    """The request state a newly built engine has, for the next test of a suite: every per-token cache, ring and
    staging ring zeroed in place (captured graphs hold their addresses), each slot back on its construction extent
    with no tokens, no kept states (pool, bank), the drafter's rings clear, slot 0 current. The decode graphs stay."""

    def zero(t) -> None:
        if hasattr(t, "_bytes"):                     # K.QRows: fp8 / fp4 planes
            for b in t._bytes():
                b.zero_()
        elif torch.is_tensor(t):
            t.zero_()

    big = eng.big
    for t in [*big.swa, *big.raw.values(), *big.comp.values(), *big.ik.values(), *eng.stage_swa,
              *eng.stage_raw.values()]:
        zero(t)
    if eng.drafter is not None:
        for t in [*eng.drafter.swa_big, *eng.drafter.stage]:
            zero(t)
    if hasattr(eng, "bank"):
        del eng.bank
    eng.pool = None
    eng.debug = None
    eng._round_costs = None
    eng.topk.clear()
    eng.candidates = None
    S = eng.slots
    eng.ring_from, eng.prefilled, eng.deep_from = [0] * S, [0] * S, [0] * S
    fixed = eng.pool_tokens >= S * eng.span                # as reset() places them on construction
    for i in range(S):
        eng.bind(i, i * eng.span if fixed else 0, eng.span)
    eng.select_slot(0)
    torch.cuda.synchronize()


def golden_ids() -> list[int]:
    return json.loads(Path("notes/dsv41/golden_long2.json").read_text())["goldens"][-1]["ids"]


# -- modes (each returns a suite Result; rank 0 prints its lines as before) -------------------------------------------
def mode_chunk_test(eng, nccl, args, env) -> SU.Result:
    N, *splits = [int(v) for v in args.chunk_test.split(",")]
    base = golden_ids()
    doc = (base * (1 + N // len(base)))[:N]

    def run(pieces):
        eng.select_slot(0)
        eng.reset()
        eng.debug = []
        logits = None
        for a, b in pieces:
            logits = eng.forward(doc[a:b], last_only=True)
        dbg, eng.debug = eng.debug, None
        per = len(dbg) // len(pieces)              # sublayer records a forward
        return [dbg[i * per:(i + 1) * per] for i in range(len(pieces))], logits.float().clone()

    def report(name, recs_a, recs_b, rows_a, rows_b):
        first = None
        worst = 0.0
        for ra, rb in zip(recs_a, recs_b):
            for key in ra:
                if key == "layer" or not torch.is_tensor(ra[key]):
                    continue
                d = (ra[key][rows_a].float() - rb[key][rows_b].float()).abs().max().item()
                worst = max(worst, d)
                if d > 0 and first is None:
                    first = (ra["layer"], key, d)
        if args.rank == 0:
            print(f"chunk-test {name}: first difference {first}, max {worst:.3g}", flush=True)
        return first, worst

    def pre(cuts):                                     # through prefill (its chunk plan): last-row logits
        eng.select_slot(0)
        eng.reset()
        logits = None
        for a, b in zip([0, *cuts], [*cuts, N]):
            logits = eng.prefill(doc[a:b])
        return logits.float().clone()

    if env.get("TF_TAIL_TEST"):                       # N: bounded-tail prefill vs every layer everywhere
        from tensorfold.families.deepseek_v41.cuda import serial as SER

        def full_run(bounded):
            SER.BOUNDED_TAIL = bounded
            eng.select_slot(0)
            eng.reset()
            logits = eng.prefill(doc[:N], final=N).float().clone()
            c = eng.c
            caches = [eng.state.comp[s_][:N // c.layer_ratios[s_]] for s_ in c.kv_source_layer_ids]
            caches = [t.q.view(torch.uint8).clone() if hasattr(t, "q") else t.clone() for t in caches]
            caches += [eng.state.ik[s_][:N // c.layer_ratios[s_]] for s_ in c.kv_source_layer_ids]
            caches = [t.q.view(torch.uint8).clone() if hasattr(t, "q") else t.clone() for t in caches]
            ring = torch.arange(N - 200, N, device=eng.dev) % 256
            rings = [t[ring].clone() for t in eng._rings()]
            toks, nxt = [], int(logits[-1].argmax())
            for _ in range(32):
                toks.append(nxt)
                nxt = eng.step(nxt)
            return logits, caches, rings, toks, list(eng.deep_from)

        with torch.no_grad():
            a = full_run(False)
            b = full_run(True)
            SER.BOUNDED_TAIL = True
        same_c = all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
        same_r = all(torch.equal(x, y) for x, y in zip(a[2], b[2]))
        dl = (a[0] - b[0]).abs().max().item()
        if args.rank == 0:
            names = [f"comp{s_}" for s_ in eng.c.kv_source_layer_ids] + [f"ik{s_}" for s_ in eng.c.kv_source_layer_ids]
            for nm, x, y in zip(names, a[1], b[1]):
                if not torch.equal(x, y):
                    bad = (x != y).reshape(x.shape[0], -1).any(1).nonzero().flatten()
                    print(f"tail-test {nm}: {len(bad)} rows differ, first {bad[:5].tolist()} last {bad[-3:].tolist()}"
                          f" of {x.shape[0]}", flush=True)
            print(f"tail-test N={N} tail_min={eng.tail_min}: logits max|diff| {dl:.3g},"
                  f" caches equal {same_c}, last-200 ring rows equal {same_r}, 32 decode tokens equal "
                  f"{a[3] == b[3]}, deep_from {b[4][0]}", flush=True)
        ok = dl == 0 and same_c and same_r and a[3] == b[3]
        return SU.Result("tail-test", "PASS" if ok else "FAIL",
                         {"dl": dl, "caches": same_c, "rings": same_r, "toks": [a[3], b[3]]},
                         f"N={N}: logits max|diff| {dl:.3g}, caches equal {same_c}, decode equal {a[3] == b[3]}")
    if env.get("TF_CHUNK_PREFILL"):                   # N,cut[,cut..]: one prefill vs calls ending at each cut
        diffs = {}
        with torch.no_grad():
            ref = pre([])
            for k in splits:
                got = pre([k])
                diffs[k] = (ref - got).abs().max().item()
                if args.rank == 0:
                    print(f"chunk-test prefill split {k} (tail {N - k}): logits max|diff| "
                          f"{diffs[k]:.3g}", flush=True)
        ok = all(d == 0 for d in diffs.values())
        # the whole prefill's last-row logits themselves (bytes and argmax): a suite run must reproduce them exactly
        sha = hashlib.sha256(ref.cpu().numpy().tobytes()).hexdigest()[:16]
        return SU.Result("chunk-prefill", "PASS" if ok else "FAIL",
                         {"N": N, "diffs": diffs, "logits_sha": sha, "argmax": int(ref[-1].argmax())},
                         f"N={N}: last-row logits max|diff| " + ", ".join(f"split {k} {d:.3g}" for k, d in diffs.items()))
    facts = []
    with torch.no_grad():
        whole, lw = run([(0, N)])
        for k in splits:
            parts, lp = run([(0, k), (k, N)])
            # rows [0, k): computed in a chunk of N rows vs one of k rows
            r1 = report(f"rows 0..{k} in a {N}-row vs {k}-row chunk", whole[0], parts[0], slice(0, k), slice(0, k))
            # rows [k, N): same chunk start? no: positions k.. in the big chunk vs a chunk starting at k
            r2 = report(f"rows {k}..{N} in a {N}-row chunk vs a chunk at {k}", whole[0], parts[1], slice(k, N),
                        slice(0, N - k))
            dl = (lw - lp).abs().max().item()
            if args.rank == 0:
                print(f"chunk-test split {k}: last-row logits max|diff| {dl:.3g}, "
                      f"argmax {int(lw.argmax())} vs {int(lp.argmax())}", flush=True)
            facts.append([k, r1, r2, dl])
    return SU.Result("chunk-test", "INFO", facts, f"N={N} sublayer diffs at splits {splits}")


def mode_views_test(eng, nccl, args, env) -> SU.Result:
    """TP=2: a kept state's bytes are the same on both ranks."""

    base = golden_ids()
    doc = (base * (1 + args.views_test // len(base)))[:args.views_test]
    with torch.no_grad():
        eng.select_slot(0)
        eng.reset()
        eng.prefill(doc)
        eng.make_bank(1)
        eng.save_window(0, 0)
        torch.cuda.synchronize()
        views = eng.kept_views(eng.extents[0][0], len(doc), 0)
        mine = [int.from_bytes(hashlib.sha256(v.contiguous().cpu().numpy().tobytes()).digest()[:7], "little")
                for _, v in views]
        got = torch.empty((2 * len(mine),), dtype=torch.int64, device="cuda")
        nccl.all_gather(torch.tensor(mine, dtype=torch.int64, device="cuda"), got)
        both = got.view(2, -1).tolist()
    same = [n for (n, _), a, b in zip(views, *both) if a == b]
    differ = [n for (n, _), a, b in zip(views, *both) if a != b]
    if args.rank == 0:
        print(f"views-test N={len(doc)}: {len(same)}/{len(views)} views equal across ranks "
              f"({sum(v.numel() for _, v in views) / 2 ** 20:.1f} MiB); names {[n for n, _ in views][:6]}..., "
              f"differ {differ}", flush=True)
    return SU.Result("views-test", "PASS" if not differ else "FAIL",
                     {"names": [n for n, _ in views], "hashes": both},
                     f"N={len(doc)}: {len(same)}/{len(views)} views equal across ranks" +
                     (f", differ {differ}" if differ else ""))


def mode_quality(eng, nccl, args, env) -> SU.Result:
    """Quality across KV formats: documents both ranks build (--tf-compare, --needle)."""

    import hashlib
    import random

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.model / "tokenizer.json"))

    def corpus(pattern):                                # this repo's own text: not in the training data
        files = sorted(p for p in Path(".").glob(pattern) if not {"ref", "out", ".venv", ".git", ".claude"}
                       & set(p.parts))
        text = "".join(f"# {p}\n{p.read_text(errors='ignore')}\n" for p in files)
        return tok.encode(text, add_special_tokens=False).ids

    def from_rank0(values):                             # rank 0's list on both ranks (the trees may differ)
        n = torch.tensor([len(values) if args.rank == 0 else 0], dtype=torch.int64, device="cuda")
        got = torch.empty((2,), dtype=torch.int64, device="cuda")
        nccl.all_gather(n, got)
        mine = (torch.tensor(values, dtype=torch.int64, device="cuda") if args.rank == 0
                else torch.zeros((int(got[0]),), dtype=torch.int64, device="cuda"))
        both = torch.empty((2 * int(got[0]),), dtype=torch.int64, device="cuda")
        nccl.all_gather(mine, both)
        return both[:int(got[0])].tolist()

    code = from_rank0(corpus("src/**/*.py") if args.rank == 0 else [])
    prose = from_rank0(corpus("**/*.md") if args.rank == 0 else [])
    golden = golden_ids()

    def doc_hash(ids):
        return int.from_bytes(hashlib.sha256(json.dumps(ids).encode()).digest()[:7], "little")

    def same_on_both(ids):                              # the ranks must feed identical tokens
        h = doc_hash(ids)
        got = torch.empty((2,), dtype=torch.int64, device="cuda")
        nccl.all_gather(torch.tensor([h], dtype=torch.int64, device="cuda"), got)
        if int(got[0]) != int(got[1]):
            raise RuntimeError("the ranks built different documents")

    out = {"kv": KV_MODE, "env": {k: v for k, v in env.items() if k.startswith("TF_DSV41")}, "docs": {}}
    facts, metrics, notes = {"tf": {}, "needle": []}, {"docs": {}}, []
    S = args.tf_score if args.tf_score is not None else MAX_ROWS
    if args.rank == 0:
        print(f"[quality] repo code {len(code)} tokens, markdown {len(prose)}, golden {len(golden)}", flush=True)
    with torch.no_grad():
        for L in [int(v) for v in args.tf_compare.split(",") if v]:
            docs = {}
            if L <= len(golden):
                docs["golden"] = golden[:L]
            docs["md"] = (prose + golden)[:L]
            for j in range(3 if L > len(golden) else 2):
                a = j * len(code) // 3
                docs[f"code{j}"] = (code[a:] + code)[:L]
            for name, doc in docs.items():
                assert len(doc) == L, (name, len(doc))
                same_on_both(doc)
                eng.select_slot(0)
                eng.reset()
                t = time.perf_counter()
                eng.prefill(doc[:L - S])
                if args.tf_rows:
                    lg = torch.cat([eng.forward(doc[a:min(a + args.tf_rows, L)]).float().clone()
                                    for a in range(L - S, L, args.tf_rows)])
                else:
                    lg = eng.forward(doc[L - S:L])
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                tgt = torch.tensor(doc[L - S + 1:L], device=lg.device)
                parts = []
                for i in range(0, S - 1, 256):              # (a whole [S, vocab] fp32 softmax is GBs)
                    lp = torch.log_softmax(lg[i:min(i + 256, S - 1)].float(), -1)
                    top_lp, top_id = lp.topk(8, -1)
                    parts.append((-lp.gather(1, tgt[i:i + lp.shape[0], None]).squeeze(1), top_id, top_lp,
                                  -(lp.exp() * lp).sum(-1)))
                    del lp
                nll, top_id, top_lp, ent = (torch.cat(x) for x in zip(*parts))
                rec = {"nll": nll.cpu(), "top_id": top_id.int().cpu(), "top_lp": top_lp.cpu(), "ent": ent.cpu(),
                       "tgt": tgt.int().cpu()}
                out["docs"][f"{L}/{name}"] = rec
                acc = (rec["top_id"][:, 0] == rec["tgt"]).float().mean()
                if args.rank == 0:
                    print(f"[tf] {L}/{name}: NLL {nll.mean():.4f}, top-1 = actual {100 * acc:.1f}%, "
                          f"entropy {ent.mean():.3f} ({dt:.0f} s)", flush=True)
                m = {"nll": float(nll.mean()), "top1": float(100 * acc), "ent": float(ent.mean()),
                     "sha": f"{doc_hash(doc):014x}"}
                facts["tf"][f"{L}/{name}"] = [m["nll"], m["top1"], m["ent"], rec["top_id"][:, 0].tolist()]
                metrics["docs"][f"{L}/{name}"] = m
                del lg
        needles = [int(v) for v in args.needle.split(",") if v]
        if needles:
            from tensorfold.cuda.chat_template import ChatTemplate

            tmpl = ChatTemplate(args.model)
            mark = "⁣NEEDLEDOC⁣"
            pre, post = tmpl.render([{"role": "user", "content": mark}], tools=None,
                                    enable_thinking=False).split(mark)
            pre_ids = tok.encode(pre, add_special_tokens=False).ids
            post_ids = tok.encode(post, add_special_tokens=False).ids
            names = ["Hawthorn", "Bluefin", "Larkspur", "Quillon", "Marigold", "Tesseract", "Obsidian", "Juniper"]
            out["needle"] = []
            for L in needles:
                for depth in [float(v) for v in args.needle_depths.split(",")]:
                    for trial in range(args.needle_trials):
                        rng = random.Random(f"{L}/{depth}/{trial}")
                        num, name = rng.randint(100000, 999999), rng.choice(names)
                        nd = tok.encode(f"\nThe special magic number for project {name} is {num}. Remember "
                                        f"it.\n", add_special_tokens=False).ids
                        q = tok.encode(f"\n\nWhat is the special magic number for project {name}? Reply with "
                                       f"the number only.", add_special_tokens=False).ids
                        room = L - len(pre_ids) - len(post_ids) - len(nd) - len(q)
                        a = int(depth * room)
                        off = (trial + 1) * len(code) // 7
                        fill = (code[off:] + code[:off])[:room]
                        ids = pre_ids + fill[:a] + nd + fill[a:] + q + post_ids
                        same_on_both(ids)
                        eng.select_slot(0)
                        eng.reset()
                        t = time.perf_counter()
                        nxt = int(eng.prefill(ids, final=len(ids))[-1].argmax())
                        torch.cuda.synchronize()
                        dt = time.perf_counter() - t
                        toks = [nxt]
                        for _ in range(23):
                            nxt = eng.step(nxt)
                            toks.append(nxt)
                        reply = tok.decode(toks)
                        ok = str(num) in reply
                        out["needle"].append({"len": len(ids), "depth": depth, "trial": trial, "num": num,
                                              "reply": reply, "pass": ok, "prefill_s": dt})
                        facts["needle"].append([len(ids), depth, trial, toks, ok])
                        if args.rank == 0:
                            print(f"[needle] {len(ids)} tokens depth {depth:.0%} trial {trial}: "
                                  f"{'PASS' if ok else 'FAIL'} ({num}) reply {reply!r} (prefill {dt:.0f} s, "
                                  f"{len(ids) / dt:.0f} tok/s)", flush=True)
    if args.rank == 0 and args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, args.out)
    status = "INFO"
    bad_nll = [k for k, m in metrics["docs"].items() if not (m["nll"] == m["nll"] and m["nll"] < 50)]
    if bad_nll:
        status = "FAIL"
        notes.append(f"NLL not finite / absurd: {bad_nll}")
    if facts["needle"]:
        found = sum(r[-1] for r in facts["needle"])
        notes.append(f"needle {found}/{len(facts['needle'])}")
        if found < len(facts["needle"]):
            status = "FAIL"
        elif status == "INFO" and not facts["tf"]:
            status = "PASS"
    if facts["tf"]:
        notes.append("NLL " + ", ".join(f"{k} {m['nll']:.4f}" for k, m in metrics["docs"].items()))
    name = "needle" if args.needle and not args.tf_compare else "tf-compare" if args.tf_compare and not args.needle \
        else "quality"
    return SU.Result(name, status, facts, "; ".join(notes), metrics)


def mode_decode_bench(eng, nccl, args, env) -> SU.Result:
    """Decode speed at a context length, 1..slots streams."""

    base = golden_ids()
    S = args.slots
    facts, metrics = {}, {"ms": {}, "prefill_tps": {}}
    with torch.no_grad():
        for rows in range(1, S + 1):
            if rows not in eng.graphs:
                eng.capture(rows)
        eng.make_bank(1)
        for L in [int(v) for v in args.decode_bench.split(",")]:
            doc = (base * (1 + L // len(base)))[:L]
            eng.select_slot(0)
            eng.reset()
            nccl.barrier()
            torch.cuda.synchronize()
            t = time.perf_counter()
            nxt = int(eng.prefill(doc, final=L)[-1].argmax())
            torch.cuda.synchronize()
            pf = time.perf_counter() - t
            facts[L] = nxt
            metrics["prefill_tps"][str(L)] = L / pf
            if args.rank == 0:
                print(f"decode-bench L={L}: after prefill torch peak {torch.cuda.max_memory_allocated() / 2**30:.1f} "
                      f"GiB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)
            eng.save_window(0, 0)
            vs, vd, b0 = eng.ring_from[0], max(eng.ring_from[0], eng.deep_from[0]), eng.extents[0][0]
            for slot in range(1, S):
                x0, size = eng.extents[slot]
                assert x0 != b0, "fixed extents needed (pool too small for the slots)"
                eng.copy_rows(b0, x0, L)
                eng.bind(slot, x0, size, ids=doc)
                eng.select_slot(slot)
                eng.load_window(0, slot)
                eng.ring_from[slot], eng.deep_from[slot] = vs, vd
            for n_streams in sorted({1, S}):
                pend = [nxt] * n_streams
                lens = [len(eng.views[s_].ids) for s_ in range(n_streams)]
                steps, warm = 64, 4
                for i in range(warm + steps):
                    if i == warm:
                        nccl.barrier()
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                    if n_streams == 1:
                        eng.select_slot(0)
                        pend = [eng.step(pend[0])]
                    else:
                        _, pend = eng.step_multi([(s_, pend[s_]) for s_ in range(n_streams)])
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                if os.environ.get("TF_DECODE_PROF") and n_streams == 1:   # the step's kernels at this context
                    from torch.profiler import ProfilerActivity, profile

                    eng.select_slot(0)
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        for _ in range(8):
                            pend = [eng.step(pend[0])]
                        torch.cuda.synchronize()
                    if args.rank == 0:
                        ev = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
                        print(f"  [step1] kernels {sum(e.self_device_time_total for e in ev) / 8e3:.3f} ms", flush=True)
                        for e in ev[:70]:
                            print(f"  {e.self_device_time_total / 8 / 1e3:7.3f} ms  {e.count / 8:6.1f}x  "
                                  f"{e.key[:100]}", flush=True)
                for s_ in range(n_streams):                 # back to the prompt for the next measurement
                    del eng.views[s_].ids[lens[s_]:]
                if os.environ.get("TF_DECODE_ROWS") and n_streams == 1:   # (fp4 study) verify rows, drafter
                    from torch.profiler import ProfilerActivity, profile

                    eng.select_slot(0)
                    n0 = len(eng.views[0].ids)
                    for R in [int(v) for v in os.environ["TF_DECODE_ROWS"].split(",")]:
                        if R not in eng.graphs:
                            eng.capture(R)
                        toks = list(base[1000:1000 + R])

                        def vstep():
                            eng.step_rows(toks)
                            del eng.views[0].ids[n0:]
                        for _ in range(4):
                            vstep()
                        nccl.barrier()
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        for _ in range(32):
                            vstep()
                        torch.cuda.synchronize()
                        vms = (time.perf_counter() - t) / 32 * 1e3
                        if args.rank == 0:
                            print(f"decode-bench L={L}: verify R={R}: {vms:.2f} ms a step", flush=True)
                        if os.environ.get("TF_DECODE_PROF"):
                            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                                for _ in range(8):
                                    vstep()
                                torch.cuda.synchronize()
                            if args.rank == 0:
                                ev = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
                                print(f"  [R={R}] kernels {sum(e.self_device_time_total for e in ev) / 8e3:.3f} ms",
                                      flush=True)
                                for e in ev[:70]:
                                    print(f"  {e.self_device_time_total / 8 / 1e3:7.3f} ms  {e.count / 8:6.1f}x  "
                                          f"{e.key[:100]}", flush=True)
                    dsp = eng.drafter
                    if dsp is not None and dsp.graph is not None:
                        ids_ = eng.views[0].ids
                        dsp.g_anchor.fill_(ids_[-1])
                        dsp.g_P.fill_(len(ids_))
                        dsp.g_base.fill_(0)
                        for _ in range(3):
                            dsp.graph.replay()
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        for _ in range(32):
                            dsp.graph.replay()
                        torch.cuda.synchronize()
                        dms = (time.perf_counter() - t) / 32 * 1e3
                        if os.environ.get("TF_DECODE_PROF"):
                            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                                for _ in range(8):
                                    dsp.graph.replay()
                                torch.cuda.synchronize()
                        if args.rank == 0:
                            print(f"decode-bench L={L}: drafter graph {dms:.2f} ms", flush=True)
                            if os.environ.get("TF_DECODE_PROF"):
                                ev = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)
                                for e in ev[:40]:
                                    print(f"  {e.self_device_time_total / 8 / 1e3:7.3f} ms  {e.count / 8:6.1f}x  "
                                          f"{e.key[:100]}", flush=True)
                        eng._round_costs = None
                        rc = eng.round_costs()
                        if args.rank == 0:
                            print(f"decode-bench L={L}: round costs k=0..n {[round(c, 2) for c in rc]}", flush=True)
                metrics["ms"][f"{L}/{n_streams}"] = dt / steps * 1e3
                if args.rank == 0:
                    print(f"decode-bench L={L}: prefill {pf:.1f} s ({L / pf:.0f} tok/s); {n_streams} stream(s): "
                          f"{steps / dt:.2f} steps/s, {n_streams * steps / dt:.1f} tok/s total, "
                          f"{dt / steps * 1e3:.2f} ms a step; torch peak {torch.cuda.max_memory_allocated() / 2**30:.1f}"
                          f" GiB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)
    return SU.Result("decode-bench", "INFO", {"first": facts},
                     ", ".join(f"L={k.split('/')[0]} x{k.split('/')[1]} {ms:.2f} ms a step"
                               for k, ms in metrics["ms"].items()), metrics)


def mode_resume_test(eng, nccl, args, env) -> SU.Result:
    """Kept COPY / disk bytes / TAKEOVER resume == fresh."""

    N, Q = [int(v) for v in args.resume_test.split(",")]
    base = golden_ids()
    doc = (base * (1 + (N + Q) // len(base)))[:N + Q]

    def decode(nxt):
        toks = [nxt]
        for _ in range(15):
            nxt = eng.step(nxt)
            toks.append(nxt)
        return toks

    res = {}
    with torch.no_grad():
        eng.make_bank(1)
        eng.select_slot(1)
        eng.reset()
        res["fresh"] = decode(int(eng.prefill(doc)[-1].argmax()))
        eng.select_slot(0)                              # the kept state: slot 0's extent and rings
        eng.reset()
        eng.prefill(doc[:N])
        eng.save_window(0, 0)
        vs = eng.ring_from[0]
        vd = max(vs, eng.deep_from[0])
        b0 = eng.extents[0][0]

        def resume(slot):
            x0, size = eng.extents[slot]
            if x0 != b0:
                eng.copy_rows(b0, x0, N)
            eng.bind(slot, x0, size, ids=doc[:N])
            eng.select_slot(slot)
            eng.load_window(0, slot)
            eng.ring_from[slot] = vs
            eng.deep_from[slot] = vd
            return decode(int(eng.prefill(doc[N:])[-1].argmax()))

        res["copy"] = resume(1)
        views = eng.kept_views(b0, N, 0)                # the NVMe tier's bytes: out, scrambled, back
        saved = [v.cpu().clone() for _, v in views]
        for _, v in views:
            v.random_(0, 256)
        for (_, v), b in zip(views, saved):
            v.copy_(b)
        res["disk"] = resume(1)
        res["takeover"] = resume(0)
    eq = {k: v == res["fresh"] for k, v in res.items() if k != "fresh"}
    if args.rank == 0:
        print(f"resume-test N={N} Q={Q}: " + ", ".join(f"{k} == fresh {v}" for k, v in eq.items())
              + f"\n  fresh {res['fresh']}\n  copy  {res['copy']}", flush=True)
    return SU.Result("resume-test", "PASS" if all(eq.values()) else "FAIL", res,
                     f"N={N} Q={Q}: " + ", ".join(f"{k} == fresh {v}" for k, v in eq.items()))


def mode_step_test(eng, nccl, args, env) -> SU.Result:
    base = golden_ids()
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
    ok = res["whole"] == res["steps"] == res["steps_drafted"]
    return SU.Result("step-test", "PASS" if ok else "FAIL", res,
                     f"{args.step_len} tokens: steps equal {res['whole'] == res['steps']}, steps_drafted equal "
                     f"{res['whole'] == res['steps_drafted']}")


def mode_multi_test(eng, nccl, args, env) -> SU.Result:
    doc = golden_ids()
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
    win = list(win)
    ok = both[0] == alone[0] and both[1] == alone[1] and win[:2] == alone[0][1:3] and win[2] == alone[1][1]
    return SU.Result("multi-test", "PASS" if ok else "FAIL", {"alone": alone, "both": both, "win": win},
                     f"{N} steps: stream 0 batched == alone {both[0] == alone[0]}, stream 1 {both[1] == alone[1]}, "
                     f"window rows {win[:2] == alone[0][1:3] and win[2] == alone[1][1]}")


def mode_prefill_bench(eng, nccl, args, env) -> SU.Result:
    doc = golden_ids()
    top = max(int(v) for v in args.prefill_bench.split(","))
    doc = (doc * (1 + top // len(doc)))[:top]                 # (lengths past the document: it repeated)
    metrics = {"tps": {}}
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
                eng.prefill(doc[:n], 2048, final=n)
                torch.cuda.synchronize()
                best = max(best, n / (time.perf_counter() - t))
            metrics["tps"][str(n)] = best
            if args.rank == 0:
                print(f"whole prompt {n} tokens: {best:.0f} tok/s", flush=True)
    return SU.Result("prefill-bench", "INFO", {"lengths": sorted(int(k) for k in metrics["tps"])},
                     ", ".join(f"{k} tokens {v:.0f} tok/s" for k, v in metrics["tps"].items()), metrics)


MODES = {"chunk_test": mode_chunk_test, "views_test": mode_views_test, "quality": mode_quality,
         "decode_bench": mode_decode_bench, "resume_test": mode_resume_test, "step_test": mode_step_test,
         "multi_test": mode_multi_test, "prefill_bench": mode_prefill_bench}


def single_mode(args) -> str | None:
    """The mode this run's flags select (main's order), or None (the other paths)."""

    if args.decoder_test:
        return None
    for attr, mode in (("chunk_test", "chunk_test"), ("views_test", "views_test"), ("tf_compare", "quality"),
                       ("needle", "quality"), ("decode_bench", "decode_bench"), ("resume_test", "resume_test"),
                       ("step_test", "step_test"), ("multi_test", "multi_test"), ("prefill_bench", "prefill_bench")):
        if getattr(args, attr):
            return mode
    return None


# -- the suite -----------------------------------------------------------------------------------------------------
def run_suite(ap, args, nccl, eng, tests: list[SU.Test]) -> bool:
    """Each test of this group on the engine, from a fresh request state; [result] lines and a [suite] summary on both
    ranks (rank 1's go to its log). Stops at the first error: the ranks may be out of step after one."""

    seed = torch.initial_seed()
    results: list[SU.Result] = []
    for t in tests:
        targs = test_args(ap, args, t)
        fresh_state(eng)
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        if args.rank == 0:
            print(f"[suite] {t.name}: {t.what}", flush=True)
        t0 = time.time()
        try:
            r = MODES[t.mode](eng, nccl, targs, SU.mode_env(t, dict(os.environ)))
            r.name = t.name
        except Exception as exc:                    # noqa: BLE001 - reported, then the suite stops
            oom = isinstance(exc, torch.OutOfMemoryError)
            r = SU.Result(t.name, "ERROR", None, f"{'OOM under the allocator cap: ' if oom else ''}"
                          f"{type(exc).__name__}: {str(exc)[:300]}")
            r.seconds = time.time() - t0
            results.append(r)
            print(f"[rank {args.rank}] " + SU.line(r), flush=True)
            import traceback

            traceback.print_exc()
            break
        r.seconds = time.time() - t0
        results.append(r)
        nccl.barrier()
        print(SU.line(r) if args.rank == 0 else f"[rank 1] {SU.line(r)}", flush=True)
    if args.suite_baseline:
        SU.compare(results, json.loads(args.suite_baseline.read_text()))
        if args.rank == 0:
            for r in results:
                print("[baseline] " + SU.line(r), flush=True)
    text, ok = SU.summary(results, f"suite {args.suite} group {args.group}")
    ran = {r.name for r in results}
    skipped = [t.name for t in tests if t.name not in ran]
    if skipped:
        text += f"; not run: {', '.join(skipped)}"
        ok = False
    print(text if args.rank == 0 else f"[rank 1] {text}", flush=True)
    if args.rank == 0 and args.suite_out:
        args.suite_out.parent.mkdir(parents=True, exist_ok=True)
        args.suite_out.write_text(json.dumps(SU.to_json(results, suite=args.suite, group=args.group, kv=KV_MODE,
                                                        env={k: v for k, v in os.environ.items()
                                                             if k.startswith("TF_")}), indent=1, default=str))
    return ok


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    suite_tests: list[SU.Test] = []
    if args.suite:                                   # the group's construction replaces the command line's
        suite_tests = SU.resolve(args.suite)
        groups = SU.groups_of(suite_tests)
        if not args.group:
            if len(groups) > 1:
                raise SystemExit(f"--suite {args.suite} spans groups {groups}: pass --group (one process and one "
                                 "weight load a group) or use tools/dsv41_suite2.sh")
            args.group = groups[0]
        if args.group not in groups:
            raise SystemExit(f"--group {args.group}: --suite {args.suite} has groups {groups}")
        why = SU.check_env(args.group, os.environ)
        if why:
            raise SystemExit(why)
        suite_tests = [t for t in suite_tests if t.group == args.group]
        g = SU.GROUPS[args.group]
        args.cap, args.slots, args.graph, args.dspark, args.fixed_k = g.cap, g.slots, g.graph, g.dspark, False
    if args.stack_after:
        import faulthandler

        faulthandler.dump_traceback_later(args.stack_after, repeat=True, file=sys.stderr)

    _engine_modules()
    if args.group and args.group != "fp8" and KV_MODE != "fp4":
        raise SystemExit(f"--group {args.group} runs on fp4 caches (TF_DSV41_KV={KV_MODE} in the environment)")
    torch.cuda.set_device(0)
    try:
        MG.guard(args.rank, cap=args.mem_cap_gib, reserve=args.mem_reserve_gib, slack=args.mem_slack_gib,
                 min_start=args.mem_min_start_gib)
    except MemoryError as exc:
        raise SystemExit(f"[rank {args.rank}] memory guard: {exc}") from None
    nccl = NCCL(args.rank, 2, args.master, args.port)
    nccl.barrier()
    if os.environ.get("TF_COMM", "nccl") == "rdma":                # small all-gathers over our RoCE transport
        from tensorfold.cuda.rdma import RdmaComm

        nccl = RdmaComm(nccl, args.rank)
        if args.rank == 0:
            print(f"[rank 0] all-gathers up to {nccl.rdma.slot_bytes >> 10} KiB over RoCE ({nccl.rdma.device})", flush=True)
    t0 = time.time()
    draft = want_draft(args)
    w = W.load(args.model, rank=args.rank, log=lambda *a, **k: None, draft=draft)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"[rank {args.rank}] weights loaded in {time.time() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB{' (with the DSpark blocks)' if draft else ''}", flush=True)
    eng = make_engine(args, nccl, w)
    if hasattr(nccl, "settle"):
        nccl.settle()

    if suite_tests:
        ok = run_suite(ap, args, nccl, eng, suite_tests)
        nccl.barrier()
        sys.exit(0 if ok else 1)
    mode = single_mode(args)
    if mode is not None:
        r = MODES[mode](eng, nccl, args, dict(os.environ))
        r.name = match_test(ap, args, dict(os.environ)) or r.name
        line = SU.line(r)
        print(line if args.rank == 0 else f"[rank 1] {line}", flush=True)
        nccl.barrier()
        return

    from tensorfold.engine.exact_sampling import Sampling

    if args.decoder_test:
        from tensorfold.cuda.kv_pool import PrefixPool
        from tensorfold.cuda.streams import Stream
        from tensorfold.families.deepseek_v41.cuda.multi import MultiDecoder

        base = golden_ids()
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
    if args.profile_prefill:
        from torch.profiler import ProfilerActivity, profile

        doc = golden_ids()
        n = args.profile_prefill
        at = int(os.environ.get("TF_PROF_AT") or 0)              # (fp4 study) the profiled chunk's start position
        doc = (doc * (1 + (at + 6 * n) // len(doc)))[:at + 6 * n]
        with torch.no_grad():
            eng.reset()
            if at:
                eng.prefill(doc[:at], n)
                doc = doc[at:]
            eng.prefill(doc[:n], n)                               # warm: kernels, Engram pages
            torch.cuda.synchronize()
            t = time.perf_counter()
            eng.prefill(doc[n:5 * n], n)                          # four chunks, rows read ahead
            torch.cuda.synchronize()
            steady = time.perf_counter() - t
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                t = time.perf_counter()
                eng.prefill(doc[5 * n:6 * n], n)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t
        if args.rank == 0:
            ev = prof.key_averages()
            gpu = sum(e.self_device_time_total for e in ev) / 1e3
            print(f"prefill {4 * n} tokens in chunks of {n}: {4 * n / steady:.0f} tok/s; profiled chunk wall "
                  f"{wall * 1e3:.0f} ms, GPU busy {gpu:.0f} ms", flush=True)
            print(ev.table(sort_by="self_device_time_total", row_limit=45, max_name_column_width=90), flush=True)
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
