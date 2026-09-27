"""GLM-5.3-Flash as the lane engine serves it: the backbone (``model.py``) plus the checkpoint's MTP head.

``GLMFlash`` gives the lane engine's family rounds (``engine.lane_family.FamilyRounds``) what they drive, the way
``qwen4_exp.runtime`` does for Flash Next:

- ``exact_width``: the widest window (up to 16 rows) whose every row gets a one-row forward's bits, checked on the
  real weights at load (``check_windows``), and ``window_costs``, each exact width's forward time; drafting is
  off when no window past one row is exact;
- ``keep_rows``: roll every cache back to a prefix of a verified window (KDA states replayed from the window's
  entry state with the same kernel, attention caches trimmed);
- the MTP head: ``speculate`` absorbs a verify round's rows (the backbone's raw hidden and the token sampled
  after each) and draws each row's first draft before anything is read; ``settle`` keeps the kept rows and chains
  up to ``drafts`` drafts from the last one, sampled with the target's keyed sampler at their positions.

Drafts change speed only: every emitted token is the target's own sample.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.glm5_next.model import DECODE_ROWS, GLM5, MLACache


class MTPCache(MLACache):
    """The MTP head's attention cache; ``drafted``: how many of its last entries are chained drafts (trimmed before
    the next kept rows are absorbed)."""

    drafted = 0


class GLMFlash:
    """GLM-5.3-Flash with the backbone and head apart, the row-exact decode path, and MTP drafting."""

    lane_family = True
    # the decode path's widest window (``model.DECODE_ROWS``: wider calls take the prefill path)
    fused_rows = DECODE_ROWS
    # tokens drawn by gpu_sampling (the exact keyed rule on the GPU): the host sampler's top-k over 154,880 logits
    # would run twice a drafted round
    gpu_sampling = True
    # ``hidden`` takes an unread GPU token: one-token rounds (the serial reference, a checkpoint without the head)
    # run one step ahead
    gpu_tokens = True
    speculate_early = os.environ.get("TF_GLM_SPEC_EARLY", "1") == "1"

    def __init__(self, model: GLM5, head: Any | None = None, *, drafts: int = 1, check: bool = True) -> None:
        self.model = model
        self.args = model.args
        self.layer_count = len(model.layers)
        self.mtp = None
        self.drafts = int(drafts)
        self.check_report: dict[int, bool] = {}
        self.exact_width, self.window_costs = self.check_windows() if check else (1, {})
        self.multi_row_exact = self.exact_width >= 2
        if check and not self.multi_row_exact:
            print(f"[glm5] a multi-row forward does not reproduce serial steps on this MLX/GPU "
                  f"({self.check_report}): no drafts", flush=True)
        self._raw: mx.array | None = None
        self._spec: tuple[mx.array, int] | None = None
        self.mtp_step_ms = 0.0
        if self.multi_row_exact and head is not None and self.drafts > 0:
            self.mtp = head
            self.mtp_step_ms = self._time_mtp_step()

    # -- the engine's model interface ---------------------------------------------------
    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        caches = self.model.make_cache()
        if self.mtp is not None:
            caches.append(MTPCache())                          # last: the model's layers never reach it
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """A stored or copied cache without an MTP entry gets an empty one (drafts then see less context)."""

        if self.mtp is not None and len(cache) == self.layer_count:
            cache.append(MTPCache())
        return cache

    def hidden(self, inputs: Any, cache: list[Any]) -> mx.array:
        """Hidden states [1, R, D] of R consecutive tokens (a [1, R] array, a GPU token not read yet, or a list),
        advancing the backbone's caches. Up to ``fused_rows`` rows take the row-exact decode path; a prompt
        chunk takes the batched prefill path."""

        tokens = inputs if isinstance(inputs, mx.array) else mx.array(np.asarray(inputs, dtype=np.int64))
        out = self.model.hidden(tokens, cache[: self.layer_count])
        self._raw = self.model.last_raw
        return out

    def head(self, hidden: mx.array) -> mx.array:
        return self.model.head(hidden)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        self.model.keep_rows(cache, rows, keep)

    # -- drafting ---------------------------------------------------------------------
    @property
    def last_streams(self) -> mx.array:
        """The last hidden() call's raw hidden states (streams collapsed, before the final norm), [R, D]."""

        return self._raw

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any]) -> None:
        """The MTP cache takes the last hidden() call's first len(next_tokens) positions (prompt rows)."""

        tokens = next_tokens if isinstance(next_tokens, mx.array) else mx.array(np.asarray(next_tokens).reshape(-1))
        tokens = tokens.reshape(-1).astype(mx.uint32)
        self._absorb(self._raw[: int(tokens.shape[0])], tokens, cache[-1])

    # TF_GLM_MTP_NORMED=1: the head reads the backbone's final-normed hidden row instead of the streams' mean before
    # the norm (the CUDA engine found the head agrees more often that way; oMLX feeds the mean). Drafts only: every
    # emitted token is still the target's sample, so either setting is exact.
    mtp_normed = os.environ.get("TF_GLM_MTP_NORMED", "0") == "1"

    def _absorb(self, raw: mx.array, tokens: mx.array, mtp_cache: MTPCache) -> mx.array:
        """Rows (raw hidden [n, D], the tokens that follow them [n]) into the head; its output rows [n, D]."""

        if mtp_cache.drafted:
            mtp_cache.trim(mtp_cache.drafted)
            mtp_cache.drafted = 0
        if self.mtp_normed:
            raw = mx.fast.rms_norm(raw, self.model.norm, self.args.rms_norm_eps)
        return self.mtp(self.model, raw, tokens, mtp_cache, int(tokens.shape[0]) <= self.fused_rows)

    def _draft_draw(self, out: mx.array, sampling: Any, positions: Any) -> mx.array:
        """Drafts (uint32 [n], lazy) from the head's output rows [n, D]: the target's keyed rule at ``positions``."""

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        return gpu_sample(self.mtp.logits(self.model, out), sampling, positions)

    def speculate(self, cache: list[Any], tokens: mx.array, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False) -> mx.array:
        """Before a verify round's tokens are read: the MTP head absorbs rows ``start`` .. of the last hidden()
        call (their raw hidden; ``tokens`` [n], the tokens that follow them, still on the GPU) and draws each row's
        first draft, for positions ``position`` + 2 + i (``position``: row ``start``'s). ``settle`` then keeps the
        kept rows' part. Rows go through the head exactly as the kept ones alone would (the decode path gives a
        row the same bits at any row count up to ``exact_width``). Returns the drafts [n] (lazy)."""

        mtp_cache = cache[-1]
        tokens = tokens.reshape(-1).astype(mx.uint32)
        rows = int(tokens.shape[0])
        total = int(self._raw.shape[0])
        start = start + total if start < 0 else start
        out = self._absorb(self._raw[start:start + rows], tokens, mtp_cache)
        self._spec = (out, rows)
        if last_only:                      # the last row's draft only (every row still enters the head's cache)
            return self._draft_draw(out[-1:], sampling, [position + 1 + rows])
        return self._draft_draw(out, sampling, [position + 2 + r for r in range(rows)])

    def settle(self, cache: list[Any], keep: int, first: int, position: int, sampling: Any, count: int) -> Any:
        """After ``speculate``: forget the head's entries of the rows past ``keep``, then the drafts for positions
        ``position``, ``position`` + 1, ...: the kept row's first draft ``first`` and ``count`` - 1 chained ones,
        each drawn on the GPU and fed to the next step unread (a lazy array the next round's inputs take)."""

        mtp_cache = cache[-1]
        out, rows = self._spec
        self._spec = None
        if rows > keep:
            mtp_cache.trim(rows - keep)
        if count <= 0:
            return []
        first = int(first.item()) if isinstance(first, mx.array) else int(first)
        if count == 1:
            return [first]
        last = out[keep - 1:keep]
        chain = [mx.array([first], dtype=mx.uint32)]
        for j in range(1, count):
            last = self.mtp(self.model, last, chain[-1], mtp_cache, True)
            mtp_cache.drafted += 1
            chain.append(self._draft_draw(last, sampling, [position + j]))
        drafts = mx.concatenate(chain)
        mx.async_eval(drafts)
        return drafts

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""

        if self._spec is not None:
            cache[-1].trim(self._spec[1])
            self._spec = None

    def draft(self, cache: list[Any], streams: mx.array, tokens: list[int], position: int, sampling: Any,
              count: int | None = None) -> list[int]:
        """Absorb positions whose raw hidden states are ``streams`` [n, D] and whose next tokens are ``tokens``,
        then chain ``count`` (default ``drafts``) drafts for positions ``position``, ``position`` + 1, ... (read
        back as ints: tests and tools; the engine takes ``speculate`` and ``settle``)."""

        mtp_cache = cache[-1]
        out = self._absorb(streams, mx.array([int(t) for t in tokens], dtype=mx.uint32), mtp_cache)[-1:]
        drafts: list[int] = []
        count = self.drafts if count is None else int(count)
        for j in range(count):
            d = int(self._draft_draw(out, sampling, [position + j]).item())
            drafts.append(d)
            if j + 1 < count:
                out = self.mtp(self.model, out, mx.array([d], dtype=mx.uint32), mtp_cache, True)
                mtp_cache.drafted += 1
        return drafts

    def _time_mtp_step(self) -> float:
        """One chained draft step as ``settle`` takes it (the head's layer, the vocabulary head, a draw read back),
        ms, fastest of 6: the engine's depth rule adds it per draft before measured rounds replace the estimate."""

        import time

        vocab = int(self.args.vocab_size)
        cache = MTPCache()
        raw = mx.zeros((1, int(self.args.hidden_size)), dtype=mx.bfloat16)
        out = self.mtp(self.model, raw, mx.array([3001 % vocab], dtype=mx.uint32), cache, True)
        mx.eval(out)
        best = float("inf")
        for i in range(6):
            started = time.perf_counter()
            out = self.mtp(self.model, out, mx.array([(3002 + i) % vocab], dtype=mx.uint32), cache, True)
            self._draft_draw(out, None, [100 + i]).item()
            best = min(best, (time.perf_counter() - started) * 1e3)
        return round(best, 3)

    # -- load-time check ------------------------------------------------------------------
    def check_windows(self, widest: int | None = None) -> tuple[int, dict[int, float]]:
        """The widest window (up to ``fused_rows``) whose every narrower window gives each row a one-row forward's
        logits bit for bit, from a 48-token prompt; and every exact width's forward time in ms (fastest of 3).
        ``check_report``: each width tried and whether it matched."""

        import time

        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        widest = int(widest or self.fused_rows)
        vocab = int(self.args.vocab_size)
        prompt = mx.array([[((37 * i + 11) % 50_000 + 1000) % vocab for i in range(48)]], dtype=mx.uint32)
        window = [(3001 + 17 * r) % vocab for r in range(widest)]
        base = self.model.make_cache()
        mx.eval(self.model.hidden(prompt, base))
        one = copy(base)
        serial = []
        for token in window:
            logits = self.model.head(self.model.hidden(mx.array([[token]], dtype=mx.uint32), one))
            mx.eval(logits)
            serial.append(logits[0, -1])
        exact = 1
        self.check_report = {}
        for width in range(2, widest + 1):
            logits = self.model.head(self.model.hidden(mx.array([window[:width]], dtype=mx.uint32), copy(base)))
            mx.eval(logits)
            same = all(bool(mx.array_equal(logits[0, i], serial[i]).item()) for i in range(width))
            self.check_report[width] = same
            if not same:
                break
            exact = width
        costs: dict[int, float] = {}
        for width in range(1, exact + 1):
            best = float("inf")
            for _ in range(3):
                cache = copy(base)
                started = time.perf_counter()
                mx.eval(self.model.head(self.model.hidden(mx.array([window[:width]], dtype=mx.uint32), cache)))
                best = min(best, (time.perf_counter() - started) * 1e3)
            costs[width] = round(best, 3)
        return exact, costs

    def rows_match_serial(self, widths: tuple[int, ...] = (2, 3, 4, 8)) -> bool:
        """Drafted rounds are exact only if a forward of k rows gives each row a one-row forward's bits (the
        load-time check, over the listed widths)."""

        exact, _ = self.check_windows(max(widths))
        return all(w <= exact for w in widths)


def load(model_dir: Path, *, drafts: int | None = None, check: bool = True) -> tuple[GLMFlash, Any]:
    """``drafts`` (default TF_GLM_MTP, else 3; 0: none): the most drafts a round from the MTP head. The engine
    picks up to this many a round from the stream's recent acceptance at each depth and the measured window
    costs, so a deeper chain is tried only where it pays."""

    import os

    from tensorfold.families.glm5_next import model as glm
    from tensorfold.families.glm5_next import mtp as mtp_module

    model, tokenizer = glm.load(Path(model_dir))
    drafts = int(os.environ.get("TF_GLM_MTP", "3")) if drafts is None else int(drafts)
    head = mtp_module.load(model) if drafts > 0 and mtp_module.has_mtp(model_dir) else None
    model.weights = None                                                 # the checkpoint's shard index is done
    runtime = GLMFlash(model, head, drafts=drafts, check=check)
    print(f"[glm5] exact window {runtime.exact_width} rows, forward ms by width {runtime.window_costs}, "
          f"MTP step {runtime.mtp_step_ms} ms, drafts up to {runtime.drafts if runtime.mtp else 0}", flush=True)
    return runtime, tokenizer
