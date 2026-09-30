"""Full GLM-5.3's fused decoding (M3-M4): eager prompt chunks, then MTP-drafted rounds replayed as CUDA graphs.

A round: the MTP layer takes the rows the last round accepted (their target hiddens with the tokens that followed) -
which rewrites its cache at those positions with target hiddens - and its last row drafts the first token; k - 1 more
single-row MTP steps chain the rest; the target verifies [token, drafts] in one window. Draft tokens never leave the
GPU; the host reads the target's picks once a round. Every emitted token is the target's own, and a verify row has a
serial step's bits (fused.py), so drafted replies equal serial ones.

MTP positions follow vLLM's DeepSeek MTP: the row at position i reads (target hidden i, token i + 1) and predicts token
i + 2; the embedding at position 0 is zeroed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import torch

from . import fused

PROMPT_ROWS = fused.PROMPT_ROWS
PROFILE_FLAG = os.environ.get("TF_GLM53_PROFILE_FLAG", "/tf/PROFILE")      # touch it: the next PROFILE_ROUNDS traced
PROFILE_ROUNDS = 8


class RoundProfiler:
    """torch.profiler over a few decode rounds when PROFILE_FLAG exists: per-kernel GPU time, round wall time."""

    def __init__(self, rank: int) -> None:
        self.rank, self.prof, self.left, self.wall = rank, None, 0, []

    def begin(self) -> None:
        if self.prof is None and os.path.exists(PROFILE_FLAG):
            from torch.profiler import ProfilerActivity, profile

            torch.cuda.synchronize()
            self.prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])
            self.prof.__enter__()
            self.left, self.wall = PROFILE_ROUNDS, []
        if self.prof is not None:
            self.t0 = time.perf_counter()

    def end(self) -> None:
        if self.prof is None:
            return
        torch.cuda.synchronize()
        self.wall.append(time.perf_counter() - self.t0)
        self.left -= 1
        if self.left:
            return
        self.prof.__exit__(None, None, None)
        rows = {}
        for e in self.prof.events():
            if e.device_type.name == "CUDA":
                n = e.name
                t, c = rows.get(n, (0.0, 0))
                rows[n] = (t + e.device_time, c + 1)
        busy = sum(t for t, _ in rows.values()) / 1e3 / len(self.wall)
        wall = 1e3 * sum(self.wall) / len(self.wall)
        os.makedirs(os.path.dirname(PROFILE_FLAG) + "/prof", exist_ok=True)
        with open(os.path.dirname(PROFILE_FLAG) + f"/prof/rank{self.rank}.txt", "w") as f:
            f.write(f"rounds {len(self.wall)}  wall/round {wall:.2f} ms  gpu kernel time/round {busy:.2f} ms  "
                    f"(gaps/overlap {wall - busy:+.2f} ms)\n")
            for n, (t, c) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
                f.write(f"{t / 1e3 / len(self.wall):9.3f} ms/round {c // len(self.wall):6d}x  {n[:150]}\n")
        try:
            self.prof.export_chrome_trace(os.path.dirname(PROFILE_FLAG) + f"/prof/rank{self.rank}.json")
        except Exception:                        # noqa: BLE001  the table is what matters
            pass
        self.prof = None
        try:
            os.remove(PROFILE_FLAG)
        except OSError:
            pass


def _pct(v: list[float]) -> dict:
    if not v:
        return {}
    v = sorted(v)
    at = lambda q: round(v[min(len(v) - 1, int(q * len(v)))], 1)  # noqa: E731
    return {"p10": at(0.1), "p50": at(0.5), "p90": at(0.9), "max": round(v[-1], 1), "mean": round(sum(v) / len(v), 1)}


class GraphSet:
    """CUDA graphs by key, captured on first use (after an eager run that is the step's real result)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.pool = torch.cuda.graph_pool_handle() if enabled else None
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.capture_s = 0.0

    def run(self, key: tuple, fn: Callable[[], None]) -> None:
        g = self.graphs.get(key)
        if g is not None:
            g.replay()
            return
        fn()
        if not self.enabled:
            return
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            fn()
        torch.cuda.synchronize()
        self.graphs[key] = g
        self.capture_s += time.perf_counter() - t0


class Runner:
    def __init__(self, w: fused.Weights, capacity: int, k: int, graphs: bool = True) -> None:
        self.w, self.k = w, k
        self.capacity = capacity
        c = w.cfg
        self.topk = c.index_topk
        cols = fused.bucket(capacity, self.topk) or 0
        self.st = fused.State(w, capacity)
        self.drafter = None                              # DFlash2 (engine attaches; set_drafter)
        self.vrows = max(k + 1, int(os.environ.get("TF_GLM53_VERIFY_ROWS", "8")) if w.tap_slot else 1)
        self.vb = fused.Buffers(w, self.vrows, max(cols, 1))                    # verify windows
        self.mb = (fused.Buffers(w, max(k + 1, self.vrows), max(cols, 1))          # MTP windows (+ a backlog of
                   if k else None)                                                # rows DFlash2 rounds kept)
        self.pb = fused.Buffers(w, PROMPT_ROWS, max(capacity, 1))                  # prompt chunks (exact key range)
        self.overlap = (fused.PROMPT_OVERLAP and w.world > 1 and hasattr(w.comm, "all_reduce")
                        and fused.PREFILL_REDUCE == "ring")
        if self.overlap:                                 # the second micro-batch of a prompt chunk
            self.pb1 = fused.Buffers(w, PROMPT_ROWS - PROMPT_ROWS // 2, max(capacity, 1))
            self.pos1 = torch.zeros((1,), dtype=torch.int32, device=w.device)
            self.comm_stream = torch.cuda.Stream()
        self.carry = torch.zeros((c.hidden_size,), dtype=torch.bfloat16, device=w.device)
        self.G = GraphSet(graphs)
        self.profiler = RoundProfiler(w.rank)
        self.set_mode(fused.MTP_MODE)

    # ------------------------------------------------------------------------------------------------ prompt ---
    def _T(self, end: int) -> int | None:
        """Keys the indexer scores for a window ending at position end - 1 (graphs: a bucket; eager: exact)."""
        return fused.bucket(end, self.topk)

    @torch.no_grad()
    def prewarm(self) -> None:
        """Capture every decode graph up front (each index-key bucket up to the capacity, windows of 1..k+1 rows, the
        default MTP inputs), so no request waits for a capture. Caches get scratch values; every request starts at 0."""
        t0 = time.perf_counter()
        buckets = [None]
        t = 2 * self.topk
        while self.topk < self.capacity and t <= max(fused.bucket(self.capacity, self.topk) or 0, 2 * self.topk):
            buckets.append(t)
            t *= 2
        n = min(self.capacity - 1, PROMPT_ROWS + self.topk + 301)   # a full chunk, a remainder, rows past topk
        self.prefill([1000 + (i * 7919) % 150000 for i in range(n)])  # compiles the prompt path's kernels
        cn, full = self.chain_normed, self.draft_full
        self.st.pos.fill_(0)
        for T in buckets:
            for R in range(1, max(self.k + 1, self.vrows) + 1):
                for pick in ("argmax", "full"):
                    self._verify(R, 0, T, pick)
            if self.k:
                for m in range(1, (max(self.k + 1, self.vrows) if self.w.tap_slot else self.k + 1) + 1):
                    self._mtp(m, 1, 0, T)
                for j in range(2, self.k + 1):
                    self._mtp(1, j, 0, T)
        torch.cuda.synchronize()
        print(f"[tensorfold] rank {self.w.rank}: {len(self.G.graphs)} decode graphs captured in "
              f"{time.perf_counter() - t0:.1f}s (key buckets {buckets})", flush=True)

    def set_mode(self, mode: str) -> None:
        if mode not in fused.MTP_MODES:
            raise ValueError(f"MTP mode {mode!r}: one of {fused.MTP_MODES}")
        self.mode = mode
        self.dflash = mode == "dflash"
        self.auto = mode == "auto"
        if self.dflash or self.auto:
            if self.drafter is None:
                raise ValueError("no DFlash2 drafter loaded (TF_GLM53_DFLASH)")
            mode = fused.MTP_MODE
        base, _, head = mode.partition(":")
        self.hid_normed, self.chain_normed = (part == "normed" for part in base.split("/"))
        self.draft_full = head == "full"

    @torch.no_grad()
    def prefill(self, prompt: list[int]) -> torch.Tensor:
        flag = PROFILE_FLAG + "_PREFILL"
        if not os.path.exists(flag):
            return self._prefill(prompt)
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            out = self._prefill(prompt)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        rows = {}
        for e in prof.events():
            if e.device_type.name == "CUDA":
                t, c = rows.get(e.name, (0.0, 0))
                rows[e.name] = (t + e.device_time, c + 1)
        busy = sum(t for t, _ in rows.values()) / 1e6
        d = os.path.dirname(PROFILE_FLAG) + "/prof"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/prefill_rank{self.w.rank}.txt", "w") as f:
            f.write(f"prompt {len(prompt)} tokens  wall {wall:.2f} s  ({len(prompt) / wall:.0f} tok/s)  gpu kernels "
                    f"{busy:.2f} s\n")
            for n, (t, c) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:40]:
                f.write(f"{t / 1e6:8.3f} s {c:7d}x  {n[:140]}\n")
        try:
            os.remove(flag)
        except OSError:
            pass
        return out

    def _prefill(self, prompt: list[int]) -> torch.Tensor:
        """Every prompt row through the target (and the MTP layer); returns the last row's logits [1, vocab]."""
        w, st, b = self.w, self.st, self.pb
        L0 = len(prompt)
        toks = torch.tensor(prompt, dtype=torch.long, device=w.device)
        logits = None
        if self.drafter is not None:
            self.drafter.reset()
        n = -(-L0 // PROMPT_ROWS)                        # equal chunks: a short remainder would read every
        step = -(-L0 // n)                               # weight again for a few rows (4102 = 2 x 2051, not 4096 + 6)
        for a in range(0, L0, step):
            e = min(a + step, L0)
            R = e - a
            T = e if e > self.topk else None
            b.ids[:R].copy_(toks[a:e])
            st.pos.fill_(a)
            if self.overlap and R >= 2 * fused.MAX_ROWS + 2:
                self.pos1.fill_(a + R // 2)
                fused.compute_prompt(w, st, b, self.pb1, R, T, self.pos1, self.comm_stream,
                                     logits="last" if e == L0 else "none")
            else:
                fused.compute(w, st, b, R, T, logits="last" if e == L0 else "none")
            if self.drafter is not None:                 # the drafter's context: this chunk's committed taps
                self.drafter.add_taps(b.taps[:R])
            if e == L0:
                logits = b.logits[:1].clone()
            if self.k:
                self._mtp_prompt(b, toks, a, e, T)
        return logits

    def _mtp_prompt(self, b: fused.Buffers, toks: torch.Tensor, a: int, e: int, T: int | None) -> None:
        """MTP rows for positions a - 1 .. e - 2: (hidden q, token q + 1); row e - 1 waits for token e (carry)."""
        w, st = self.w, self.st
        R = e - a
        h = torch.empty((R, w.cfg.hidden_size), dtype=torch.bfloat16, device=w.device)
        fused.target_hidden_for_mtp(w, b, slice(0, R), h, self.hid_normed)
        first = a == 0
        rows = R - 1 if first else R
        if rows > 0:
            if first:
                b.hin[:rows].copy_(h[:rows])
            else:
                b.hin[0].copy_(self.carry)
                b.hin[1:rows].copy_(h[:rows - 1])
            b.ids[:rows].copy_(toks[a + 1:e] if first else toks[a:e])
            st.mpos.fill_(0 if first else a - 1)
            fused.mtp_compute(w, st, b, rows, T, logits="none", zero_first=first,
                              chain_normed=self.chain_normed)
        self.carry.copy_(h[R - 1])

    # ------------------------------------------------------------------------------------------------ decode ---
    def _mtp(self, m: int, j: int, P0: int, T: int | None) -> None:
        """MTP window of m rows at P0 ..; its last row's draft goes to verify slot j and feeds the next MTP step."""
        w, st, mb, vb = self.w, self.st, self.mb, self.vb
        cn, full = self.chain_normed, self.draft_full
        st.mpos.fill_(P0)

        def fn():
            fused.mtp_compute(w, st, mb, m, T, logits="last", chain_normed=cn, draft_full=full)
            vb.ids[j:j + 1].copy_(mb.argmax[:1])
            mb.ids[:1].copy_(mb.argmax[:1])
            mb.hin[:1].copy_(mb.hidden[m - 1:m])
        self.G.run(("mtp", m, j, T, cn, full), fn)

    def _verify(self, R: int, P: int, T: int | None, pick: str) -> None:
        w, st, vb = self.w, self.st, self.vb
        st.pos.fill_(P)
        self.G.run(("tgt", R, T, pick), lambda: fused.compute(w, st, vb, R, T, logits="all", pick=pick))

    def _dflash_cfg(self) -> tuple[int, float]:
        cfg = {"depth": int(os.environ.get("TF_GLM53_DFLASH_DEPTH", "7")),
               "confidence": float(os.environ.get("TF_GLM53_DFLASH_CONFIDENCE", "0.4"))}
        try:
            import json as _json

            cfg.update(_json.loads(open(os.path.dirname(PROFILE_FLAG) + "/DFLASH_CFG").read()))
        except (OSError, ValueError):
            pass
        return min(self.drafter.block - 1, self.vrows - 1, int(cfg["depth"])), float(cfg["confidence"])

    def _generate_auto(self, out, tok, P, max_tokens, sample, stop, on_tokens, prefill_s, sampling, k):
        """Each round MTP ("m") or DFlash2 ("f"), whichever has been emitting more tokens a second (EMAs of rank 0's
        round times, shared so every rank takes the same arm; a probe of the other every PROBE rounds). Both stay current: DFlash2 absorbs every round's kept taps; the MTP layer carries a
        backlog of kept rows (target hidden, next token) that its next merged call writes - flushed when it outgrows the
        window. The target picks every token, so the reply equals the serial one whichever arm drafts."""
        w, vb, mb, dr = self.w, self.vb, self.mb, self.drafter
        depth, conf = self._dflash_cfg()
        probe = int(os.environ.get("TF_GLM53_AUTO_PROBE", "16"))
        ema = {"m": None, "f": None}
        since = {"m": 0, "f": 0}
        arms = []
        m = 1                                            # backlog rows in mb.hin/ids: positions P - m .. P - 1
        rounds = drafted = accepted = 0
        done = len(out) >= max_tokens or stop(tok)
        round_ms: list[float] = []
        t1 = time.perf_counter()
        while not done:
            tr = time.perf_counter()
            rounds += 1
            if ema["m"] is None or ema["f"] is None:
                arm = "m" if ema["m"] is None else "f"
            else:
                arm = "m" if ema["m"] >= ema["f"] else "f"
                other = "f" if arm == "m" else "m"
                if since[other] >= probe:
                    arm = other
            since[arm] = 0
            since["f" if arm == "m" else "m"] += 1
            arms.append(arm)
            if arm == "m":
                room = max(0, min(k, max_tokens - len(out) - 1, self.capacity - P - 1))
                vb.ids[:1].fill_(tok)
                if room:
                    Tm = self._T(P + room)
                    self._mtp(m, 1, P - m, Tm)
                    for j in range(2, room + 1):
                        self._mtp(1, j, P + j - 2, Tm)
                    m = 0                                # the merged call wrote the backlog
                R = room + 1
            else:
                room = max(0, min(depth, max_tokens - len(out) - 1, self.capacity - P - 1))
                drafts = dr.propose(tok, room, sampling, conf) if room else []
                R = 1 + len(drafts)
                vb.ids[:R].copy_(torch.tensor([tok] + drafts, dtype=torch.long), non_blocking=True)
            self._verify(R, P, self._T(P + R), "argmax" if sample is None else "full")
            if sample is None:
                both = torch.cat([vb.argmax[:R], vb.ids[1:R]]).tolist()
                picks, drafts = both[:R], both[R:]
            else:
                drafts = vb.ids[1:R].tolist()
                picks = []
                for i in range(R):
                    picks.append(sample(vb.logits[i:i + 1], P + i + 1))
                    if i >= len(drafts) or drafts[i] != picks[-1]:
                        break
            n = 0
            while n < len(drafts) and n < len(picks) - 1 and drafts[n] == picks[n]:
                n += 1
            emit = picks[:n + 1]
            drafted += len(drafts)
            accepted += n
            dr.add_taps(vb.taps[:n + 1])                 # DFlash2 context: every round's kept rows
            if m + n + 1 > mb.rows:                      # MTP backlog would outgrow its window: write it now
                if m:
                    self.st.mpos.fill_(P - m)
                    fused.mtp_compute(w, self.st, mb, m, self._T(P), logits="none", chain_normed=self.chain_normed)
                m = 0
            fused.target_hidden_for_mtp(w, vb, slice(0, n + 1), mb.hin[m:m + n + 1], self.hid_normed)
            mb.ids[m:m + n + 1].copy_(torch.tensor(emit, dtype=torch.long), non_blocking=True)
            m += n + 1
            for e_tok in emit:
                out.append(e_tok)
                on_tokens([e_tok])
                if len(out) >= max_tokens or stop(e_tok):
                    done = True
                    break
            P, tok = P + n + 1, emit[-1]
            dt = time.perf_counter() - tr
            round_ms.append(1e3 * dt)
            if w.world > 1:                              # rank 0's round time on every rank: the arm choice must
                mine = torch.tensor([dt], dtype=torch.float32, device=w.device)       # be identical on all ranks
                allt = torch.empty((w.world,), dtype=torch.float32, device=w.device)
                w.comm.all_gather(mine, allt)
                dt = float(allt[0].item())
            rate = (n + 1) / dt
            ema[arm] = rate if ema[arm] is None else 0.8 * ema[arm] + 0.2 * rate
        torch.cuda.synchronize()
        dec = time.perf_counter() - t1
        arm_s = "".join(arms)
        return {"prefill_s": prefill_s, "decode_s": dec, "tokens": len(out), "rounds": rounds,
                "tokens_per_round": round((len(out) - 1) / max(rounds, 1), 3),
                "accept": round(accepted / drafted, 3) if drafted else None,
                "tok_s": round((len(out) - 1) / dec, 2) if dec > 0 and len(out) > 1 else 0.0,
                "round_ms": _pct(round_ms), "mtp_mode": "auto", "arms": {"m": arm_s.count("m"), "f": arm_s.count("f")},
                "graphs": len(self.G.graphs), "capture_s": round(self.G.capture_s, 1), "out": out}

    def _generate_dflash(self, prompt, out, tok, P, max_tokens, sample, stop, on_tokens, prefill_s, sampling):
        """DFlash2 rounds: the drafter proposes up to its block - 1 tokens after the pending one, the target verifies
        [pending, drafts] in one window (graph), the kept rows' taps extend the drafter's context."""
        vb, dr = self.vb, self.drafter
        cfg = {"depth": int(os.environ.get("TF_GLM53_DFLASH_DEPTH", "7")),
               "confidence": float(os.environ.get("TF_GLM53_DFLASH_CONFIDENCE", "0.4"))}
        try:                                             # runtime override (the same file on every node)
            import json as _json

            cfg.update(_json.loads(open(os.path.dirname(PROFILE_FLAG) + "/DFLASH_CFG").read()))
        except (OSError, ValueError):
            pass
        depth_max = min(dr.block - 1, self.vrows - 1, int(cfg["depth"]))
        conf = float(cfg["confidence"])
        t_draft = t_verify = t_taps = 0.0
        rounds = drafted = accepted = 0
        done = len(out) >= max_tokens or stop(tok)
        round_ms: list[float] = []
        t1 = time.perf_counter()
        while not done:
            tr = time.perf_counter()
            rounds += 1
            room = max(0, min(depth_max, max_tokens - len(out) - 1, self.capacity - P - 1))
            drafts = dr.propose(tok, room, sampling, conf) if room else []
            ta = time.perf_counter()
            R = 1 + len(drafts)
            vb.ids[:R].copy_(torch.tensor([tok] + drafts, dtype=torch.long), non_blocking=True)
            self._verify(R, P, self._T(P + R), "argmax" if sample is None else "full")
            if sample is None:
                picks = vb.argmax[:R].tolist()
                tb = time.perf_counter()
            else:
                picks = []
                for i in range(R):
                    picks.append(sample(vb.logits[i:i + 1], P + i + 1))
                    if i >= len(drafts) or drafts[i] != picks[-1]:
                        break
            n = 0
            while n < len(drafts) and n < len(picks) - 1 and drafts[n] == picks[n]:
                n += 1
            emit = picks[:n + 1]
            drafted += len(drafts)
            accepted += n
            if sample is not None:
                tb = time.perf_counter()
            dr.add_taps(vb.taps[:n + 1])
            torch.cuda.synchronize()
            tc = time.perf_counter()
            t_draft += ta - tr
            t_verify += tb - ta
            t_taps += tc - tb
            for e_tok in emit:
                out.append(e_tok)
                on_tokens([e_tok])
                if len(out) >= max_tokens or stop(e_tok):
                    done = True
                    break
            P, tok = P + n + 1, emit[-1]
            round_ms.append(1e3 * (time.perf_counter() - tr))
        torch.cuda.synchronize()
        dec = time.perf_counter() - t1
        return {"prefill_s": prefill_s, "decode_s": dec, "tokens": len(out), "rounds": rounds,
                "tokens_per_round": round((len(out) - 1) / max(rounds, 1), 3),
                "accept": round(accepted / drafted, 3) if drafted else None,
                "tok_s": round((len(out) - 1) / dec, 2) if dec > 0 and len(out) > 1 else 0.0,
                "round_ms": _pct(round_ms), "mtp_mode": "dflash", "depth": depth_max, "confidence": conf,
                "ms_per_round": {k: round(1e3 * v / max(rounds, 1), 2) for k, v in
                                 (("draft", t_draft), ("verify", t_verify), ("taps", t_taps))},
                "graphs": len(self.G.graphs), "capture_s": round(self.G.capture_s, 1), "out": out}

    @torch.no_grad()
    def generate(self, prompt: list[int], max_tokens: int, sample: Callable, stop: Callable[[int], bool],
                 on_tokens: Callable[[list[int]], Any], k: int, mode: str | None = None,
                 sampling=None) -> dict[str, Any]:
        """sample(logits_row [1, V], position) -> token (None: greedy on the device's argmax)."""
        w, vb, mb = self.w, self.vb, self.mb
        k = min(k, self.k)
        self.set_mode(mode or fused.MTP_MODE)
        t0 = time.perf_counter()
        L0 = len(prompt)
        lg = self.prefill(prompt)
        tok = sample(lg, L0) if sample else int(torch.argmax(lg[0]).item())
        prefill_s = time.perf_counter() - t0
        out = [tok]
        on_tokens([tok])
        P = L0                                   # tok sits at P, not yet in the caches
        m = 1                                    # MTP rows pending: (carry = hidden P - 1, tok) at P - 1
        if k:
            mb.hin[:1].copy_(self.carry)
            mb.ids[:1].fill_(tok)
        if self.dflash:
            return self._generate_dflash(prompt, out, tok, P, max_tokens, sample, stop, on_tokens, prefill_s,
                                         sampling)
        if self.auto and k:
            return self._generate_auto(out, tok, P, max_tokens, sample, stop, on_tokens, prefill_s, sampling, k)
        rounds = drafted = accepted = 0
        done = len(out) >= max_tokens or stop(tok)
        t1 = time.perf_counter()
        t_sample = t_stream = 0.0
        round_ms: list[float] = []
        while not done:
            rounds += 1
            self.profiler.begin()
            tr = time.perf_counter()
            room = max(0, min(k, max_tokens - len(out) - 1, self.capacity - P - 1))
            vb.ids[:1].fill_(tok)
            if room:
                Tm = self._T(P + room)
                self._mtp(m, 1, P - m, Tm)
                for j in range(2, room + 1):
                    self._mtp(1, j, P + j - 2, Tm)
            R = room + 1
            self._verify(R, P, self._T(P + R), "argmax" if sample is None else "full")
            if sample is None:
                both = torch.cat([vb.argmax[:R], vb.ids[1:R]]).tolist()
                picks, drafts = both[:R], both[R:]
            else:
                drafts = vb.ids[1:R].tolist()
                picks = []
                for i in range(R):
                    ts = time.perf_counter()
                    picks.append(sample(vb.logits[i:i + 1], P + i + 1))
                    t_sample += time.perf_counter() - ts
                    if i >= len(drafts) or drafts[i] != picks[-1]:
                        break
            n = 0
            while n < len(drafts) and n < len(picks) - 1 and drafts[n] == picks[n]:
                n += 1
            emit = picks[:n + 1]
            drafted += len(drafts)
            accepted += n
            for e_tok in emit:
                out.append(e_tok)
                ts = time.perf_counter()
                on_tokens([e_tok])
                t_stream += time.perf_counter() - ts
                if len(out) >= max_tokens or stop(e_tok):
                    done = True
                    break
            if k:                                # next round's MTP rows: (target hidden P + i, token P + i + 1)
                m = n + 1
                fused.target_hidden_for_mtp(w, vb, slice(0, m), mb.hin[:m], self.hid_normed)
                mb.ids[:m].copy_(torch.tensor(emit, dtype=torch.long), non_blocking=True)
            P, tok = P + n + 1, emit[-1]
            self.profiler.end()
            round_ms.append(1e3 * (time.perf_counter() - tr))
        torch.cuda.synchronize()
        dec = time.perf_counter() - t1
        return {"prefill_s": prefill_s, "decode_s": dec, "tokens": len(out), "rounds": rounds,
                "tokens_per_round": round((len(out) - 1) / max(rounds, 1), 3),
                "accept": round(accepted / drafted, 3) if drafted else None,
                "tok_s": round((len(out) - 1) / dec, 2) if dec > 0 and len(out) > 1 else 0.0,
                "round_ms": _pct(round_ms), "sample_ms_per_round": round(1e3 * t_sample / max(rounds, 1), 2),
                "stream_ms_per_round": round(1e3 * t_stream / max(rounds, 1), 2),
                "mtp_mode": self.mode, "graphs": len(self.G.graphs), "capture_s": round(self.G.capture_s, 1), "out": out}
