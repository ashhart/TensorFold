"""GLM-5.3-Flash's CUDA engine behind ``tensorfold.cuda.server``: two ranks over NCCL, one per machine.

Rank 0 serves HTTP and sends each request's header and prompt to rank 1 over the engine's all-gather; both ranks
then run the same prefill and drafted decode. Both sample every row with the same keyed rule from the same
gathered candidates, so they agree without a broadcast, and the reply is byte-identical to serial decoding on
the same two ranks.

Prefix reuse: the committed state after the last request's prompt and after its reply are kept (``decode.
Snapshot``), and a prompt that extends either resumes from it. Rows never depend on their chunk-mates, so a
resumed prompt ends in the state a fresh prefill gives. Both ranks keep the same snapshots; rank 0 names the one
it resumes from in the header.

A request's draft policy is a spec (the engine's default, or the request's through ``app.GlmApp``):

    0             serial: one token a round
    N             N MTP drafts a round (the checkpoint's MTP head)
    a[:LOW:HIGH]  1 to 3 MTP drafts from the running acceptance (the default, a:0.6:0.85)
    cN:P          up to N MTP drafts while the product of the drafts' own probabilities stays at or above P
    f...          the same with DFlash2 drafts (fN, fcN:P, fa:...), when both ranks loaded the draft model
    auto          the default. Greedy requests: each round drafts with the MTP head (c3:0.35) or DFlash2
                  (fc5:0.3), whichever has committed more tokens per millisecond in this request
                  (``decode.DrafterChoice``: 2 rounds of each first, then a 3% margin to switch and one round of
                  the other every 8). Sampled requests: MTP drafts, 1 to 3 from the running acceptance
                  (a:0.6:0.85), where DFlash2's sampled chains measured slower. MTP only without the draft model.
    auto:E:EVERY:MARGIN
                  the same choice with E rounds of each first, a probe every EVERY rounds and a MARGIN to switch,
                  for sampled requests too (MTP a:0.6:0.85 against DFlash2 there)
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_POLICY = "auto"
GRAPH_ROWS = (1, 2, 3, 4, 5, 6)       # verify windows captured as CUDA graphs
MAX_ROWS = 8                          # the widest verify window (a pending token and up to 7 drafts)
DENSE_CAPACITY = 2560                 # cache slots while DSA attention stays dense (contexts up to 2,051 tokens)


def encode_policy(spec: str) -> list[int]:
    """A policy spec as 4 ints: kind (0 serial, 1 fixed, 2 running acceptance, 3 confidence; plus 10 for DFlash2
    drafts), most drafts, two parameters in millionths."""

    spec = str(spec).strip()
    bad = ValueError(f"draft policy {spec!r}: expected auto[:E:EVERY:MARGIN], 0, N, a[:LOW:HIGH], cN:P, or one of "
                     f"these after f (N from 1 to {MAX_ROWS - 1})")
    try:
        if spec == "auto" or spec.startswith("auto:"):
            parts = spec.split(":")
            if len(parts) not in (1, 4):
                raise bad
            explore, every, margin = (int(parts[1]), int(parts[2]), float(parts[3])) if len(parts) == 4 else (2, 8, 0.03)
            if explore < 1 or every < 0 or not 0 <= margin < 1:
                raise bad
            return [4 if len(parts) == 1 else 5, explore, every, int(round(margin * 1e6))]
        if spec.startswith("f"):
            code = encode_policy(spec[1:])
            return [code[0] + 10] + code[1:] if code[0] else code
        if spec.startswith("a"):
            parts = spec.split(":")
            if parts[0] != "a" or len(parts) not in (1, 3):
                raise bad
            low, high = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.8, 0.9)
            return [2, 3, int(round(low * 1e6)), int(round(high * 1e6))]
        if spec.startswith("c"):
            most_text, conf = spec[1:].split(":")
            most = int(most_text)
            if not 0 < most < MAX_ROWS:
                raise bad
            return [3, most, int(round(float(conf) * 1e6)), 0]
        most = int(spec)
    except ValueError:
        raise bad from None
    if not 0 <= most < MAX_ROWS:
        raise bad
    return [1 if most > 0 else 0, most, 0, 0]


def decode_policy(code: list[int]):
    """The ``decode.DepthPolicy`` for a code, None for serial decoding, or ("auto", explore, every, margin,
    choose for sampled requests too)."""

    from .decode import DepthPolicy

    kind, most, a, b = code
    if kind in (4, 5):
        return ("auto", most, a, b / 1e6, kind == 5)
    kind %= 10
    if kind == 2:
        return DepthPolicy(min(most, MAX_ROWS - 1), low=a / 1e6, high=b / 1e6)
    if kind == 3:
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, confidence=a / 1e6)
    return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True) if kind == 1 else None


def _f64_ints(x: float) -> list[int]:
    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _ints_f64(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


class GlmEngine:
    """GLM-5.3-Flash on two ranks (this one ``rank``): weights, MTP and DFlash2 drafting, per-request policies."""

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, policy: str = DEFAULT_POLICY,
                 drafter: Path | None = None, context: int = 0, serial_only: bool = False, comm=None) -> None:
        """``comm``: a communicator with ``all_gather`` and ``barrier`` instead of NCCL between two machines (tests)."""

        import torch

        from .comm import NCCL
        from .decode import Engine
        from .weights import Config, load

        encode_policy(policy)                           # a bad default fails here, not in the first request
        torch.cuda.set_device(0)
        self.torch = torch
        self.rank = rank
        self.policy = "0" if serial_only else policy
        self.serial_only = serial_only
        cfg = Config.read(model_dir)
        long_context = context > cfg.dense_limit
        capacity = max(DENSE_CAPACITY, context + MAX_ROWS) if long_context else DENSE_CAPACITY
        self.limit = capacity - MAX_ROWS if long_context else cfg.dense_limit
        self.comm = comm if comm is not None else NCCL(rank, 2, master, port)
        self.comm.barrier()
        # both ranks must run the same calls: refuse to start when they were given different settings
        mine = [int(drafter is not None), capacity, int(long_context), int(serial_only)]
        both = self._gather_ints(mine)
        if both[0] != both[1]:
            raise RuntimeError("the two ranks were started with different settings (draft model, context, drafts): "
                               f"rank 0 {both[0]}, rank 1 {both[1]}; pull the draft model on both machines (or pass "
                               "--drafter none to both) and give both the same flags")
        w = load(model_dir, rank=rank)
        w.comm = self.comm
        self.comm.barrier()
        self.w = w
        self.drafter = None
        if drafter is not None:
            from .dflash2 import Drafter

            self.drafter = Drafter(drafter, w, capacity=capacity)
        self.e = Engine(w, capacity=capacity, max_rows=MAX_ROWS, prefill_rows=64, graphs=True, graph_rows=GRAPH_ROWS,
                        long_context=long_context, taps=self.drafter.tap_layers if self.drafter is not None else ())
        if self.drafter is not None:
            self.drafter.capture()
        self.costs = self._calibrate()
        if rank == 0:
            c = self.costs
            print(f"[tensorfold] drafter timings (ms, fastest of 7): {c['timed']}", flush=True)
            print("[tensorfold] drafter costs (ms): verify " + " ".join(f"{v:.1f}" for v in c["verify"]) +
                  f"; MTP draft {c['mtp']:.2f} (+{c['mtp_step']:.2f} a chained draft, +{c['mtp_row']:.2f} a row); "
                  f"DFlash2 block {c['block']:.2f} (+{c['taps_row']:.3f} a tap row)", flush=True)
        self.eos = tuple(w.cfg.eos)
        self.request = threading.local()    # the calling request's policy and stop-at-EOS (``app.GlmApp``)
        self.cache: list = []               # decode.Snapshot entries, each a prefix of the next

    def _calibrate(self) -> dict:
        """Milliseconds for ``decode.DrafterChoice``, timed on random tokens as rounds use them and made the same on
        both ranks (the slower rank's time of each): a verify window of 1 to MAX_ROWS rows; an MTP draft (absorbing
        one row, sampling its draft), each further chained draft and each further absorbed row; a DFlash2 block with
        its host chain and each tap row DFlash2 takes. The machine has slow moments of a few seconds (page
        migration, most of all right after loading), so every piece is timed in turns over several passes and
        keeps its fastest run, and the windows of 2 rows and more follow a line through their times fitted with
        the median of pairwise slopes."""

        import statistics

        import numpy as np

        from .decode import draft, prefill

        torch = self.torch
        e, st = self.e, self.e.st
        rng = np.random.default_rng(0)
        vocab = self.w.cfg.vocab

        def tokens(n: int) -> list[int]:
            return [int(t) for t in rng.integers(0, vocab, n)]

        prefill(e, tokens(64), None, mtp=True, drafter=self.drafter)
        hidden = e.main_hidden(slice(0, MAX_ROWS)).clone()
        one, six = tokens(1), tokens(6)
        start = st.mtp_len

        def rewind() -> None:
            st.set_mtp_len(start)
            st.mtp_drafted = 0

        pieces: dict[str, tuple] = {f"v{r}": (lambda w=tokens(r): e.forward(w), None) for r in range(1, MAX_ROWS + 1)}
        pieces["m1"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 1, None), rewind)
        pieces["m3"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 3, None), rewind)
        pieces["m6"] = (lambda: draft(e, hidden[:6], six, st.pos + 1, 1, None), rewind)
        if self.drafter is not None:
            d = self.drafter
            taps = e.tap_rows(8).clone()
            ctx = d.context_end

            def back() -> None:
                if d.context_end != ctx:
                    d.pos_dev.sub_(d.context_end - ctx)
                    d.context_end = ctx

            pieces["block"] = (lambda: d.propose(one[0], 5, None, 0.0), None)
            pieces["taps8"] = (lambda: d.add_taps(taps), back)
        best = {name: float("inf") for name in pieces}
        for turn in range(9):
            for name, (fn, prep) in pieces.items():
                if prep is not None:
                    prep()
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                if turn >= 2:
                    best[name] = min(best[name], (time.perf_counter() - t) * 1e3)
            rewind()
            if self.drafter is not None:
                back()
        names = list(best)
        mine = torch.tensor([best[n] for n in names], dtype=torch.float32, device="cuda")
        got = torch.empty((2 * mine.numel(),), dtype=torch.float32, device="cuda")
        self.comm.all_gather(mine, got)
        both = dict(zip(names, got.view(2, -1).max(dim=0).values.tolist()))
        e.reset()
        if self.drafter is not None:
            self.drafter.reset()
        rows = list(range(2, MAX_ROWS + 1))
        ys = [both[f"v{r}"] for r in rows]
        slope = statistics.median((ys[j] - ys[i]) / (rows[j] - rows[i]) for i in range(len(rows))
                                  for j in range(i + 1, len(rows)))
        base = statistics.median(y - slope * r for r, y in zip(rows, ys))
        verify = [both["v1"]] + [base + slope * r for r in rows]
        mtp = both["m1"]
        return {"verify": verify, "mtp": mtp, "mtp_step": max((both["m3"] - mtp) / 2, 0.0),
                "mtp_row": max((both["m6"] - mtp) / 5, 0.0), "block": both.get("block", 0.0),
                "taps_row": max(both.get("taps8", 0.0) / 8, 0.0), "timed": {k: round(v, 2) for k, v in both.items()}}

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int32, device="cuda")
        got = torch.empty((2 * len(values),), dtype=torch.int32, device="cuda")
        self.comm.all_gather(mine, got)
        return [got[:len(values)].tolist(), got[len(values):].tolist()]

    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank (a length, then the values, through the all-gather)."""

        torch = self.torch
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32, device="cuda")
        got = torch.empty((2,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(n, got)
        count = int(got[0].item())
        buf = (torch.tensor(values, dtype=torch.int32, device="cuda") if self.rank == 0
               else torch.zeros((count,), dtype=torch.int32, device="cuda"))
        allv = torch.empty((2 * count,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]

    def _drafters(self, code: list[int]) -> tuple[bool, bool, bool]:
        """(auto, MTP drafts, DFlash2 drafts) for a policy code."""

        auto = code[0] in (4, 5)
        dflash = (auto or code[0] // 10 == 1) and self.drafter is not None
        return auto, auto or not dflash, dflash

    def _resume(self, prompt: list[int], code: list[int]):
        """The longest snapshot of a strict prefix of ``prompt`` whose draft caches fit the request's drafters."""

        _, mtp, dflash = self._drafters(code)
        best = None
        for snap in self.cache:
            fits = (not dflash or snap.drafter_end == len(snap.ids)) and (not mtp or snap.mtp_len >= 0)
            if fits and len(snap.ids) < len(prompt) and prompt[:len(snap.ids)] == snap.ids and (
                    best is None or len(snap.ids) > len(best.ids)):
                best = snap
        return best

    def _remember(self, snap) -> None:
        self.cache = [c for c in self.cache if len(c.ids) < len(snap.ids) and snap.ids[:len(c.ids)] == c.ids]
        self.cache = self.cache[-1:] + [snap]

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool, on_tokens: Callable[[list[int]], Any],
             code: list[int], hit, draft: bool) -> dict[str, Any]:
        from .decode import (DepthPolicy, DrafterChoice, auto_decode, dflash_decode, mtp_decode, prefill,
                             serial_decode, take_snapshot)

        auto, use_mtp, use_dflash = self._drafters(code)
        drafter = self.drafter if use_dflash else None
        t0 = time.perf_counter()
        # a request writes the attention caches from its resume point on: every longer snapshot is overwritten
        cut = len(hit.ids) if hit is not None else 0
        self.cache = [c for c in self.cache if len(c.ids) <= cut]
        first = prefill(self.e, prompt, sampling, mtp=use_mtp, drafter=drafter, resume=hit)
        prefill_s = time.perf_counter() - t0
        if draft:
            self._remember(take_snapshot(self.e, prompt, self.e.last_hidden if use_mtp else None, mtp=use_mtp,
                                         drafter=drafter))
        stats: dict[str, Any] = {"prefill_s": prefill_s, "cached": cut}
        on_tokens([first])
        if max_tokens <= 1 or (stop_eos and first in self.eos):
            return stats
        policy = decode_policy(code)
        if policy is None:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens)
        elif auto:
            greedy = sampling is None or sampling.temperature <= 0
            m_policy = DepthPolicy(3, fixed=True, confidence=0.35) if greedy else DepthPolicy(3, low=0.6, high=0.85)
            _, explore, every, margin, sampled_too = policy
            choice = None
            if drafter is not None and (greedy or sampled_too):
                choice = DrafterChoice(self.costs, first="f" if greedy else "m", explore=explore, every=every,
                                       margin=margin)
            res = auto_decode(self.e, drafter, first, max_tokens, sampling, choice=choice, m_policy=m_policy,
                              f_policy=DepthPolicy(5, fixed=True, confidence=0.3), stop_eos=stop_eos,
                              on_tokens=on_tokens)
        elif use_dflash:
            res = dflash_decode(self.e, self.drafter, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                                on_tokens=on_tokens)
        else:
            res = mtp_decode(self.e, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                             on_tokens=on_tokens)
        if draft and policy is not None and res.keeps:
            committed = list(prompt) + res.tokens[:self.e.st.pos - len(prompt)]
            if auto:
                pending = res.pending
            else:
                pending = None if use_dflash else self.e.main_hidden(slice(0, res.keeps[-1]))
            self._remember(take_snapshot(self.e, committed, pending, mtp=use_mtp, drafter=drafter))
        elif policy is None:
            self.cache = [c for c in self.cache if len(c.ids) <= len(prompt)]   # the reply's rows are not kept
        stats.update(decode_s=res.seconds, rounds=res.rounds,
                     tokens_per_round=round((len(res.tokens) - 1) / max(res.rounds, 1), 3),
                     sha256=hashlib.sha256(json.dumps(res.tokens).encode()).hexdigest()[:16])
        if res.arms:
            stats.update(drafters=res.arms, keeps=res.keeps)
        if res.stages:
            stats["stages_ms"] = {k: round(v * 1e3, 1) for k, v in res.stages.items()}
        return stats

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True) -> dict[str, Any]:
        """Rank 0: one request, mirrored by rank 1 (``follow``). ``draft=False``: serial decoding and a fresh
        prefill, the reference drafted replies must equal."""

        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        if not draft or self.serial_only:
            spec = "0"
        else:
            spec = getattr(self.request, "policy", None) or self.policy
        code = encode_policy(spec)
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        hit = self._resume(list(prompt), code) if draft else None
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        header = [max_tokens, int(stop_eos), int(draft), len(hit.ids) if hit is not None else 0,
                  seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
                  *_f64_ints(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
                  *_f64_ints(sampling.top_p if sampling else 1.0)] + code
        self._share(header)
        self._share(list(prompt))
        stats = self._run(list(prompt), max_tokens, sampling, stop_eos, on_tokens, code, hit, draft)
        stats.update(policy=spec, drafts=draft)
        return stats

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling

        while True:
            max_tokens, stop_eos, draft, cached, s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, *code = \
                self._share(None)
            prompt = self._share(None)
            temperature = _ints_f64(t_lo, t_hi)
            seed = (s_top << 62) | (s_hi << 31) | s_lo
            sampling = Sampling(seed, temperature, top_k, _ints_f64(p_lo, p_hi)) if temperature > 0 else None
            hit = None
            if cached:
                hit = next((c for c in self.cache if len(c.ids) == cached and prompt[:cached] == c.ids), None)
                if hit is None:
                    raise RuntimeError(f"rank 1 has no snapshot of the {cached} tokens rank 0 resumes from")
            self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None, code, hit, bool(draft))
