"""GLM-5.3-Flash on two NCCL ranks; both sample by one keyed rule from the same gathered candidates, so no broadcast."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_POLICY = "auto"
DFLASH_POLICY = "fc5:0.3"             # DFlash2 drafts every round: up to 5 while their probability product holds 0.3
EXL3_AUTO = DFLASH_POLICY             # what auto runs on an EXL3 checkpoint with the draft model
GRAPH_ROWS = (1, 2, 3, 4, 5, 6)       # verify windows captured as CUDA graphs
MAX_ROWS = 8                          # the widest verify window (a pending token and up to 7 drafts)
DENSE_CAPACITY = 2560                 # cache slots while DSA attention stays dense (contexts up to 2,051 tokens)
# TF_GLM_DRAFT_RING=0: DFlash2's context in a flat buffer of the whole window, not a 2,176-row ring (the same drafts)
DRAFT_RING = os.environ.get("TF_GLM_DRAFT_RING", "1").strip() != "0"


def encode_policy(spec: str) -> list[int]:
    """Encode policy kind, maximum drafts, and two parameters in millionths as four integers, with 10 added to kind for DFlash2."""

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
    """Decode a serial, MTP, or automatic policy, retaining exploration, sampling, and margin settings."""

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


MTP_DEFAULT = "1"                     # TF_GLM_MTP when unset: the MTP head stays beside DFlash2 (auto, 0: left out)


def mtp_head(drafter: bool, serial_only: bool, layers: int, value: str | None = None) -> bool:
    """TF_GLM_MTP: load the MTP head? auto: not beside DFlash2 or with --no-drafts; 1: whenever it exists; 0: never."""

    value = os.environ.get("TF_GLM_MTP", "") if value is None else value
    value = value.strip().lower() or MTP_DEFAULT
    if value not in ("0", "1", "auto"):
        raise ValueError(f"TF_GLM_MTP: 0, 1 or auto, not {value!r}")
    if value == "auto":
        return bool(layers) and not drafter and not serial_only
    return value == "1" and bool(layers)


def without_mtp(transform, layers: int):
    """A startup weight transform that leaves out the MTP layer's tensors (``layers.<num_hidden_layers>.``)."""

    prefix = f"model.language_model.layers.{layers}."
    return lambda name, info: (0, 0) if name.startswith(prefix) else transform(name, info)


class GlmEngine:
    """GLM-5.3-Flash on two ranks (this one ``rank``): weights, MTP and DFlash2 drafting, per-request policies."""

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, policy: str = DEFAULT_POLICY,
                 drafter: Path | None = None, context: int | None = None, context_explicit: bool | None = None, serial_only: bool = False, comm=None,
                 prefill_rows: int | None = None) -> None:
        """``comm``: a communicator with ``all_gather`` and ``barrier`` instead of NCCL between two machines (tests)."""

        import torch

        from tensorfold.cuda.comm import NCCL
        from .decode import Engine
        from .weights import Config, load
        from .split import rule
        from tensorfold.cuda.capacity import admit
        from tensorfold.cuda.geometry import (PREFILL_ROWS, dflash2_geometry, dflash2_weights, mla_geometry,
                                              split_weights)

        encode_policy(policy)                           # a bad default fails here, not in the first request
        torch.cuda.set_device(0)
        self.torch = torch
        self.rank = rank
        self.policy = "0" if serial_only else policy
        self.serial_only = serial_only
        self.comm = comm if comm is not None else NCCL(rank, 2, master, port)
        self.comm.barrier()
        cfg = Config.read(model_dir)
        # Without --context the window stays dense, attending every key without indexer work.
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        from . import LATENT

        # TF_GLM_MTP off: the MTP layer's tensors, caches and buffers are neither loaded nor estimated
        self.mtp_on = mtp_head(drafter is not None, serial_only, cfg.mtp_layers)
        weights_estimate = split_weights(rule)
        if not self.mtp_on:
            weights_estimate = without_mtp(weights_estimate, cfg.layers)
        self.capacity_plan = admit(model_dir, context if explicit else cfg.dense_limit, explicit, torch,
                                   lambda text: mla_geometry(text, 2, MAX_ROWS, minimum_slots=DENSE_CAPACITY,
                                                             latent=LATENT, mtp=self.mtp_on),
                                   weights_estimate, rank=rank, world=2, gather=self._gather_ints,
                                   draft_dir=drafter, draft_weights=lambda d: dflash2_weights(d, 2),
                                   draft_geometry=lambda text: dflash2_geometry(text, 2, MAX_ROWS, ring=DRAFT_RING))
        self.limit = self.capacity_plan["context_window"]
        capacity = self.capacity_plan["cache_slots"]
        long_context = self.limit > cfg.dense_limit
        # both ranks must run the same calls: refuse to start when they were given different settings
        prefill_rows = PREFILL_ROWS if prefill_rows is None else int(prefill_rows)
        mine = [int(drafter is not None), capacity, int(long_context), int(serial_only), int(LATENT),
                prefill_rows, int(self.mtp_on), int(DRAFT_RING)]
        # other conversations' kept prompts get what the window leaves, at most TF_GLM_CACHE_GIB, the same on both ranks
        plan = self.capacity_plan
        wanted = int(float(os.environ.get("TF_GLM_CACHE_GIB", "3")) * 2 ** 30)
        spare = max(0, min(wanted, plan["budget_bytes"] - plan["total_bytes_estimate"]))
        both = self._gather_ints(mine + [spare >> 20])
        if both[0][:-1] != both[1][:-1]:
            raise RuntimeError("the two ranks were started with different settings (draft model, context, drafts, "
                               "TF_GLM_LATENT, TF_GLM_MTP, TF_GLM_DRAFT_RING): "
                               f"rank 0 {both[0][:-1]}, rank 1 {both[1][:-1]}; pull the draft model on both machines "
                               "(or pass --drafter none to both) and give both the same flags")
        self.cache_bytes = min(both[0][-1], both[1][-1]) << 20
        plan["kept_bytes"] = self.cache_bytes
        for key in ("serving_peak_bytes_estimate", "total_bytes_estimate"):
            plan[key] = plan[key] + self.cache_bytes
        if rank == 0 and self.cache_bytes < wanted:
            print(f"[tensorfold] other conversations' prompts are kept in {self.cache_bytes / 2 ** 30:.1f} GiB, what "
                  f"the {self.limit}-token window leaves (TF_GLM_CACHE_GIB asks {wanted / 2 ** 30:.1f})", flush=True)
        if not self.mtp_on and drafter is None and not serial_only:
            raise ValueError(("TF_GLM_MTP=0 leaves" if cfg.mtp_layers else "this checkpoint has") + " no MTP head and "
                             "no DFlash2 draft model was given, so every round would decode one token: pull the draft "
                             "model on both machines (--drafter), or pass --no-drafts to both for the serial reference")
        w = load(model_dir, rank=rank, mtp=self.mtp_on)
        w.comm = self.comm
        self.comm.ready("loading")                   # a peer stuck loading is named, not waited on in NCCL
        self.comm.barrier()
        self.w = w
        if rank == 0 and cfg.mtp_layers and not self.mtp_on:
            print("[tensorfold] the checkpoint's MTP head is not loaded (TF_GLM_MTP=" +
                  (os.environ.get("TF_GLM_MTP", "").strip() or MTP_DEFAULT) + "): " +
                  ("DFlash2 drafts every request" if drafter is not None else "--no-drafts"), flush=True)
        self.drafter = None
        if drafter is not None:
            from .dflash2 import Drafter

            self.drafter = Drafter(drafter, w, capacity=capacity, ring=DRAFT_RING)
        self.e = Engine(w, capacity=capacity, max_rows=MAX_ROWS, prefill_rows=prefill_rows, graphs=True, graph_rows=GRAPH_ROWS,
                        long_context=long_context, taps=self.drafter.tap_layers if self.drafter is not None else ())
        if self.drafter is not None:
            self.drafter.capture()
        self.costs = self._calibrate()
        if rank == 0:
            c = self.costs
            print(f"[tensorfold] drafter timings (ms, fastest of 7): {c['timed']}", flush=True)
            mtp = (f"; MTP draft {c['mtp']:.2f} (+{c['mtp_step']:.2f} a chained draft, +{c['mtp_row']:.2f} a row)"
                   if w.mtp is not None else "")
            print("[tensorfold] drafter costs (ms): verify " + " ".join(f"{v:.1f}" for v in c["verify"]) + mtp +
                  f"; DFlash2 block {c['block']:.2f} (+{c['taps_row']:.3f} a tap row)", flush=True)
        self.eos = tuple(w.cfg.eos)
        self.model_dir = Path(model_dir)
        self.request = threading.local()    # the calling request's policy and stop-at-EOS (``app.GlmApp``)
        # kept conversations (decode.Snapshot, least recently used first) and the live caches' ids; states and saved rows stay within cache_bytes
        self.cache: list = []
        self.live: list[int] = []
        self.cache_entries = int(os.environ.get("TF_GLM_CACHE_ENTRIES", "8"))

    def _calibrate(self) -> dict:
        """Per-piece ms for ``drafter_choice.DrafterChoice``: fastest of interleaved passes, equal on both ranks."""

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
        hidden = e.pbuf.fnormed[:MAX_ROWS].clone()          # rows for timing the draft steps
        one, six = tokens(1), tokens(6)
        start = st.mtp_len

        def rewind() -> None:
            st.set_mtp_len(start)
            st.mtp_drafted = 0

        pieces: dict[str, tuple] = {f"v{r}": (lambda w=tokens(r): e.forward(w), None) for r in range(1, MAX_ROWS + 1)}
        if self.w.mtp is not None:
            pieces["m1"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 1, None), rewind)
            pieces["m3"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 3, None), rewind)
            pieces["m6"] = (lambda: draft(e, hidden[:6], six, st.pos + 1, 1, None), rewind)
        if self.drafter is not None:
            d = self.drafter
            taps = e.tap_rows(8, e.pbuf).clone()
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
        mtp = both.get("m1", 0.0)
        return {"verify": verify, "mtp": mtp, "mtp_step": max((both.get("m3", 0.0) - mtp) / 2, 0.0),
                "mtp_row": max((both.get("m6", 0.0) - mtp) / 5, 0.0), "block": both.get("block", 0.0),
                "taps_row": max(both.get("taps8", 0.0) / 8, 0.0), "timed": {k: round(v, 2) for k, v in both.items()}}

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int32, device="cuda")
        got = torch.empty((2 * len(values),), dtype=torch.int32, device="cuda")
        self.comm.all_gather(mine, got)
        return [got[:len(values)].tolist(), got[len(values):].tolist()]

    # the idle doorbell: rank 1 waits for each request on the rendezvous store (no store: no doorbell), not in NCCL
    def _store(self):
        return getattr(self.comm, "store", None)

    def _ring(self) -> None:
        store = self._store()
        if store is not None:
            self._bell = getattr(self, "_bell", 0) + 1
            store.set(f"tf_glm_request_{self._bell}", b"1")

    def _await_bell(self) -> None:
        store = self._store()
        if store is None:
            return
        from datetime import timedelta

        key = f"tf_glm_request_{getattr(self, '_bell', 0) + 1}"
        while True:
            try:
                store.wait([key], timedelta(hours=1))
                break
            except Exception as e:            # an idle hour: wait again (a lost rank 0 is a connection error instead)
                if "timeout" not in str(e).lower():
                    raise
        store.delete_key(key)
        self._bell = getattr(self, "_bell", 0) + 1

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

    def _effective(self, code: list[int]) -> list[int]:
        """Resolve auto and MTP policies to the available heads, using EXL3_AUTO for EXL3 with DFlash2 and DFlash2 when MTP is absent."""

        if code[0] == 4 and self.drafter is not None and self.w.cfg.quant == "exl3":
            return encode_policy(EXL3_AUTO)
        if self.w.mtp is None and code[0] in (1, 2, 3, 4, 5):
            return encode_policy(DFLASH_POLICY) if code[0] in (4, 5) else [code[0] + 10] + code[1:]
        return code

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

    def _drop(self, snap) -> None:
        """Forget a kept snapshot and free its saved rows now, even while a caller still holds the object."""
        snap.rows, snap.nbytes, snap.drafter_rows = None, 0, None
        self.cache.remove(snap)

    def _remember(self, snap) -> None:
        for c in [c for c in self.cache if c.ids == snap.ids and c is not snap]:
            self._drop(c)
        self.cache[:] = [c for c in self.cache if c is not snap] + [snap]   # a resumed prompt kept again moves last
        dropped = False
        while len(self.cache) > 1 and (len(self.cache) > self.cache_entries or self._held_bytes() > self.cache_bytes):
            self._drop(self.cache[0])
            dropped = True
        if dropped:
            import torch

            torch.cuda.empty_cache()

    def _take_over(self, keep: list[int]) -> None:
        """Save the rows of every kept snapshot the next prefill overwrites, dropping the oldest entries past the memory budget; both ranks decide alike."""
        from .decode import row_bytes, save_rows

        live = self.live
        dropped = False

        def resumes(c) -> bool:
            return len(c.ids) <= len(keep) and keep[:len(c.ids)] == c.ids

        for snap in list(self.cache):
            n = len(snap.ids)
            if snap not in self.cache or snap.rows is not None or resumes(snap):
                continue
            if live[:n] != snap.ids:                  # its rows are already gone: nothing to resume from
                self._drop(snap)
                continue
            need = row_bytes(self.e, snap)
            while self._held_bytes() + need > self.cache_bytes:
                old = next((c for c in self.cache if c is not snap and not resumes(c)), None)
                if old is None:
                    break
                self._drop(old)
                dropped = True
            if self._held_bytes() + need > self.cache_bytes:
                self._drop(snap)
                dropped = True
                continue
            save_rows(self.e, snap)
        if dropped:
            import torch

            torch.cuda.empty_cache()             # give the freed rows back rather than keep them in torch's pool

    def _held_bytes(self) -> int:
        from .decode import snapshot_bytes

        return sum(snapshot_bytes(c) for c in self.cache)

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool, on_tokens: Callable[[list[int]], Any],
             code: list[int], hit, draft: bool, constraint=None) -> dict[str, Any]:
        self.e.constraint, self.e.window = constraint, None       # both ranks walk and mask the same rows
        try:
            return self._run_once(prompt, max_tokens, sampling, stop_eos, on_tokens, code, hit, draft)
        finally:
            self.e.constraint = self.e.window = None

    def _run_once(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool,
                  on_tokens: Callable[[list[int]], Any], code: list[int], hit, draft: bool) -> dict[str, Any]:
        from .decode import DepthPolicy, dflash_decode, mtp_decode, prefill, serial_decode
        from .drafter_choice import DrafterChoice, auto_decode

        auto, use_mtp, use_dflash = self._drafters(code)
        drafter = self.drafter if use_dflash else None
        t0 = time.perf_counter()
        # a request writes the caches from its resume point: other conversations' rows are saved first, a saved resume point's restored
        from .decode import load_rows

        cut = len(hit.ids) if hit is not None else 0
        self._take_over(list(hit.ids) if hit is not None else [])
        if hit is not None and hit.rows is not None:
            load_rows(self.e, hit)
            hit.rows, hit.nbytes = None, 0            # live again
        self.live = list(prompt)
        first = prefill(self.e, prompt, sampling, mtp=use_mtp, drafter=drafter, resume=hit,
                        keep_at=max(1, len(prompt) - 1) if draft else None, keep=self._remember)
        prefill_s = time.perf_counter() - t0
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
        # the caches now hold prompt and reply; only prompts are snapshotted, since a later prompt prefills the reply again
        self.live = list(prompt) + res.tokens[:self.e.st.pos - len(prompt)]
        stats.update(decode_s=res.seconds, rounds=res.rounds, min_rows=1 + min(res.depths, default=0),
                     tokens_per_round=round((len(res.tokens) - 1) / max(res.rounds, 1), 3),
                     sha256=hashlib.sha256(json.dumps(res.tokens).encode()).hexdigest()[:16])
        if res.arms:
            stats.update(drafters=res.arms, keeps=res.keeps)
        if res.stages:
            stats["stages_ms"] = {k: round(v * 1e3, 1) for k, v in res.stages.items()}
        return stats

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True,
                 constraint=None) -> dict[str, Any]:
        """Mirror one rank-0 request on rank 1; draft=False uses serial decoding and fresh prefill as the reference drafted replies must equal."""

        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        if not draft or self.serial_only:
            spec = "0"
        else:
            spec = getattr(self.request, "policy", None) or self.policy
        code = self._effective(encode_policy(spec))
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        hit = self._resume(list(prompt), code) if draft else None
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        header = [max_tokens, int(stop_eos), int(draft), len(hit.ids) if hit is not None else 0,
                  seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
                  *_f64_ints(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
                  *_f64_ints(sampling.top_p if sampling else 1.0), *_f64_ints(sampling.min_p if sampling else 0.0),
                  int(constraint is not None)] + code
        from tensorfold.engine.grammar import pack

        self._ring()                                   # wakes rank 1, which idles on the store, not in the all-gather
        self._share(header)
        self._share(list(prompt))
        if constraint is not None:                     # the request's grammar: rank 1 compiles the same
            self._share(pack(constraint))
        stats = self._run(list(prompt), max_tokens, sampling, stop_eos, on_tokens, code, hit, draft, constraint)
        stats.update(policy=spec, drafts=draft)
        return stats

    def score_labels(self, prompt_ids: list[int], label_ids: list[int]) -> tuple[list[float], float]:
        """Both ranks prefill the prompt and return its label logits plus the full-vocabulary logsumexp."""

        prompt = [int(token) for token in prompt_ids]
        labels = [int(token) for token in label_ids]
        if not prompt:
            raise ValueError("empty prompt")
        if not labels:
            raise ValueError("empty labels")
        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        self._ring()                    # rank 1 waits on the store before this header, as a chat request does
        self._share([0, len(labels)])   # max_tokens on a chat header is at least 1, so 0 is a score
        self._share(prompt)
        self._share(labels)
        return self._score_local(prompt, labels)

    def _score_local(self, prompt: list[int], labels: list[int]) -> tuple[list[float], float]:
        from tensorfold.families.glm5_next.cuda.score import prompt_logits
        from tensorfold.server.decisions import reduce_vocab_shards

        # The score prefill writes attention rows from position 0. Save or drop every snapshot those rows
        # still belong to, then stop naming them: the next chat must not resume the decision as that conversation.
        self._take_over([])
        self.live = []
        if self.drafter is not None:
            self.drafter.reset()
        local = prompt_logits(self.e, prompt)
        rows = [local] if self.comm is None else self._gather_floats(local)
        if self.rank != 0:
            return [], 0.0
        return reduce_vocab_shards(rows, labels, len(rows[0]))

    def _gather_floats(self, values: list[float]) -> list[list[float]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.float32, device="cuda")
        got = torch.empty((2 * len(values),), dtype=torch.float32, device="cuda")
        self.comm.all_gather(mine, got)
        width = len(values)
        return [[float(item) for item in got[:width].tolist()], [float(item) for item in got[width:].tolist()]]

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling

        while True:
            self._await_bell()
            header = self._share(None)
            if len(header) == 2 and header[0] == 0:     # a decision: both ranks prefill, neither samples
                prompt = self._share(None)
                labels = self._share(None)
                self._score_local(prompt, labels)
                continue
            (max_tokens, stop_eos, draft, cached, s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, m_lo, m_hi, shaped,
             *code) = header
            prompt = self._share(None)
            packed = self._share(None) if shaped else []
            constraint = None
            if packed:                                  # compiled here as on rank 0
                from tensorfold.engine import grammar

                constraint = grammar.compiler(self, self.model_dir, self.eos).follow(packed)
            temperature = _ints_f64(t_lo, t_hi)
            seed = (s_top << 62) | (s_hi << 31) | s_lo
            sampling = (Sampling(seed, temperature, top_k, _ints_f64(p_lo, p_hi), _ints_f64(m_lo, m_hi))
                        if temperature > 0 else None)
            hit = None
            if cached:
                hit = next((c for c in self.cache if len(c.ids) == cached and prompt[:cached] == c.ids), None)
                if hit is None:
                    raise RuntimeError(f"rank 1 has no snapshot of the {cached} tokens rank 0 resumes from")
            self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None, code, hit, bool(draft),
                      constraint)
