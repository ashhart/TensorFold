"""Flash Next as the lane engine serves it: the fused decode plus the checkpoint's MTP head.

``FlashNext`` wraps the model (``model.py``) with what the lane engine's family rounds
(``engine.lane_family``) drive:

- ``exact_width``: the widest window (up to 16 rows) whose every row gets a one-row
  forward's bits, checked at load on this MLX and GPU, and ``window_costs``, each
  exact width's forward time; drafted rows are verified exactly;
- ``keep_rows``: roll every cache back to a prefix of a verified window;
- the MTP head: its cache absorbs each kept position (main-model streams and the
  sampled next token), then it chains ``drafts`` drafts from the last one. Drafts
  are sampled with the target's keyed sampler at their positions, so a draft is
  the target's own sample whenever the two distributions agree there.

Drafts change speed only: every emitted token is the target's sample.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.qwen4_exp.model import AttentionCache


class MTPCache(AttentionCache):
    """The MTP head's attention cache; ``drafted``: how many of its last entries are chained drafts (trimmed before
    the next kept rows are absorbed)."""

    drafted = 0


class FlashNext:
    """Flash Next with the backbone and head apart, the fused decode, and MTP drafting."""

    fused_rows = 16
    lane_family = True
    # tokens drawn by gpu_sampling (the exact keyed rule in one kernel): the host sampler's argpartition of
    # 248k logits and readback cost ~2 ms a draw on the M3 Ultra, twice a drafted round (2026-09-25)
    gpu_sampling = True

    def __init__(self, model: Any, head: Any | None = None, *, drafts: int = 1) -> None:
        self.model = model
        self.args = model.args
        self.fused = model.__dict__.get("fused")
        self.layer_count = len(model.layers)
        self.mtp = None
        self.drafts = int(drafts)
        self.exact_width, self.window_costs = self.check_windows() if self.fused is not None else (1, {})
        self.multi_row_exact = self.exact_width >= 2
        if self.fused is not None and not self.multi_row_exact:
            print("[flash-next] a multi-row forward does not reproduce serial steps on this MLX/GPU: no drafts",
                  flush=True)
        if self.multi_row_exact and head is not None and self.drafts > 0:
            self.mtp = head
            # the head's decoder layer and mixer through the same fused kernels as the model's layers
            from types import SimpleNamespace

            from tensorfold.families.qwen4_exp.decode import FusedDecode

            shell = SimpleNamespace(args=self._head_config(), layers=head.layers,
                                    model=SimpleNamespace(hyper_connection_mixer=head.hyper_connection_mixer,
                                                          embed_tokens=model.model.embed_tokens))
            self.mtp_fused = FusedDecode(shell)
            self._mtp_scales = [1.0 + head.pre_fc_norm_embedding.weight.astype(mx.float32),
                                1.0 + head.pre_fc_norm_hidden.weight.astype(mx.float32)]
            mx.eval(*self._mtp_scales)
            import os

            # the head scores a fixed list of 79,591 ids (TF_FLASH_DRAFT_VOCAB=0: the whole vocabulary), and a
            # round's chained drafts stay on the GPU until the next round reads them (TF_FLASH_QUEUED=0: each read)
            self._draft_ids = self._draft_head = None
            if os.environ.get("TF_FLASH_DRAFT_VOCAB", "1") != "0":
                from tensorfold.families.qwen4_exp.draft_head import cut_head, draft_ids

                ids = draft_ids()
                self._draft_ids = mx.array(ids)
                self._draft_head = cut_head(model.lm_head, ids)
            self.queued_chains = os.environ.get("TF_FLASH_QUEUED", "1") != "0"
            self.mtp_step_ms = self._time_mtp_step()

    queued_chains = False

    mtp_step_ms = 0.0

    def _time_mtp_step(self) -> float:
        """One chained draft step as ``settle`` takes it (the head's layer, the vocabulary head, a draw read back),
        ms, fastest of 6: the engine's depth rule adds it per draft before measured rounds replace the estimate."""

        import time

        cache = MTPCache()
        mixed, out = self._mtp_step([3001], self.fused.last_streams[-1:], cache)
        mx.eval(mixed, out)
        best = float("inf")
        for i in range(6):
            started = time.perf_counter()
            mixed, out = self._mtp_step([3002 + i], out, cache)
            self._draft_draw(mixed, None, [100 + i]).item()
            best = min(best, (time.perf_counter() - started) * 1e3)
        return round(best, 3)

    # -- the serial engine's model interface ----------------------------------------
    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        caches = self.model.make_cache()
        if self.mtp is not None:
            caches.append(MTPCache())                         # last: the model's layers never reach it
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """A stored or copied cache without an MTP entry gets an empty one (drafts then see less context)."""

        if self.mtp is not None and len(cache) == self.layer_count:
            cache.append(MTPCache())
        return cache

    def hidden(self, inputs: Any, cache: list[Any]) -> mx.array:
        """Mixed hidden states [1, R, D]: up to ``fused_rows`` rows through the fused decode kernels, a longer
        prompt chunk through MLX's forward, whose bits depend on the chunk; the engine's aligned prefill gives a
        resumed prompt the chunks of the same prompt fed fresh."""

        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        out = self.model.hidden(tokens, cache[: self.layer_count])
        fused = self.fused is not None and tokens.shape[0] == 1 and tokens.shape[1] <= self.fused_rows
        self._streams = self.fused.last_streams if fused else self.model.__dict__["last_streams"]
        return out

    def head(self, hidden: mx.array) -> mx.array:
        from tensorfold.families.qwen4_exp.decode import project

        return project(hidden, self.model.lm_head)

    def __call__(self, inputs: Any, cache: list[Any]) -> mx.array:
        return self.head(self.hidden(inputs, cache))

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        self.fused.keep_rows(cache, rows, keep)

    # -- drafting -------------------------------------------------------------------
    @property
    def last_streams(self) -> mx.array:
        """The last hidden() call's residual streams before the final mixer, [L, S*D]."""

        return self._streams

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any]) -> None:
        """The MTP cache takes the last hidden() call's first len(next_tokens) positions (prompt rows)."""

        tokens = [int(t) for t in np.asarray(next_tokens).reshape(-1)]
        self._absorb(self._streams[:len(tokens)], tokens, cache[-1])

    def _head_config(self) -> Any:
        from dataclasses import replace

        return replace(self.args, num_hidden_layers=1, layer_types=["sparse_attention"], ple_layer_ids=[])

    def _mtp_step(self, tokens: Any, streams: mx.array, mtp_cache: MTPCache) -> tuple[mx.array, mx.array]:
        """The MTP head on rows (next tokens, residual streams [n, S*D]): (mixed [1, n, D], its streams [n, S*D]).
        Prompt-long inputs take the reference modules; decode rows the fused kernels."""

        head = self.mtp
        rows, wide = streams.shape
        dims = wide // head.streams
        if rows <= self.fused_rows:
            # the prologue in a few kernels (was ~20 MLX ops a draft step): embedding rows, the two (1 + w) norms,
            # the two projections through ``project``
            from tensorfold.kernels.qwen.flash_next.v1 import kernels as K
            from tensorfold.families.qwen4_exp.decode import project

            eps = self.mtp_fused.eps
            emb = K.embed_rows(tokens, self.model.model.embed_tokens)                      # [n, D]
            e = project(K.rms_norm_rows(emb, self._mtp_scales[0], eps), head.fc_embedding)
            normed = K.rms_norm_rows(streams, self._mtp_scales[1], eps).reshape(rows * head.streams, dims)
            hs = project(normed, head.fc_hidden) if rows * head.streams <= 32 else head.fc_hidden(normed)
            x = (e[:, None, :] + hs.reshape(rows, head.streams, dims)).reshape(rows, wide)
            mixed = self.mtp_fused.run(x, None, [mtp_cache])
            return mixed, self.mtp_fused.last_streams
        ids = tokens.astype(mx.int32) if isinstance(tokens, mx.array) else mx.array(tokens, dtype=mx.int32)
        emb = self.model.model.embed_tokens(ids)                                            # [n, D]
        e = head.fc_embedding(head.pre_fc_norm_embedding(emb))
        hs = head.fc_hidden(head.pre_fc_norm_hidden(streams).reshape(rows, head.streams, dims))
        x = (e[:, None, :] + hs).reshape(rows, wide)
        x = head.layers[0](x[None], None, mtp_cache)
        return head.hyper_connection_mixer(x), x[0]

    def _draft_draw(self, mixed: mx.array, sampling: Any, positions: Any) -> mx.array:
        """Drafts (uint32 [n], lazy) from the head's mixed hidden states [1, n, D]: the target's keyed rule over the
        listed ids when the head is cut, else over the whole vocabulary."""

        from tensorfold.families.qwen4_exp.decode import project

        x = mixed.reshape(-1, mixed.shape[-1])
        if self._draft_head is not None:
            from tensorfold.families.qwen4_exp.draft_head import sample as draft_sample

            return draft_sample(project(x, self._draft_head), self._draft_ids, sampling, positions)
        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        return gpu_sample(self.head(x[None]).reshape(x.shape[0], -1), sampling, positions)

    _draft_head: Any = None
    _draft_ids: Any = None

    def _absorb(self, streams: mx.array, tokens: list[int], mtp_cache: MTPCache) -> tuple[mx.array, mx.array]:
        if mtp_cache.drafted:
            mtp_cache.trim(mtp_cache.drafted, self.args.indexer_compress_ratio)
            mtp_cache.drafted = 0
        mixed, out = self._mtp_step(tokens, streams, mtp_cache)
        return mixed[:, -1:], out[-1:]

    def draft(self, cache: list[Any], streams: mx.array, tokens: list[int], position: int, sampling: Any,
              count: int | None = None) -> list[int]:
        """Absorb positions whose residual streams are ``streams`` [n, S*D] and whose next tokens are ``tokens``,
        then chain ``count`` (default ``drafts``) drafts for positions ``position``, ``position`` + 1, ..."""

        mtp_cache = cache[-1]
        mixed, out = self._absorb(streams, [int(t) for t in tokens], mtp_cache)
        drafts: list[int] = []
        count = self.drafts if count is None else int(count)
        for j in range(count):
            d = int(self._draft_draw(mixed, sampling, [position + j]).item())
            drafts.append(d)
            if j + 1 < count:
                mixed, out = self._mtp_step([d], out, mtp_cache)
                mtp_cache.drafted += 1
        return drafts

    def speculate(self, cache: list[Any], tokens: mx.array, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False) -> mx.array:
        """Before a verify round's tokens are read: the MTP head absorbs rows ``start`` .. of the last hidden()
        call (their streams; ``tokens`` [n], the tokens that follow them, still on the GPU) and draws each row's
        first draft, for positions ``position`` + 2 + i (``position``: row ``start``'s). One read then returns
        both; ``settle`` keeps the kept rows' part. Rows go through the MTP head exactly as ``draft`` would take
        the kept ones (the fused kernels give a row the same bits at any row count), so the drafts and the MTP
        cache are the same, a host round trip earlier. Returns the drafts [n] (lazy)."""

        mtp_cache = cache[-1]
        if mtp_cache.drafted:
            mtp_cache.trim(mtp_cache.drafted, self.args.indexer_compress_ratio)
            mtp_cache.drafted = 0
        tokens = tokens.reshape(-1)
        rows = int(tokens.shape[0])
        total = int(self._streams.shape[0])
        start = start + total if start < 0 else start
        mixed, out = self._mtp_step(tokens, self._streams[start:start + rows], mtp_cache)
        self._spec = (out, rows)
        if last_only:                      # the last row's draft only (every row still enters the head's cache)
            return self._draft_draw(mixed[:, -1:], sampling, [position + 1 + rows])
        return self._draft_draw(mixed, sampling, [position + 2 + r for r in range(rows)])

    def settle(self, cache: list[Any], keep: int, first: int, position: int, sampling: Any, count: int) -> list[int]:
        """After ``speculate``: forget the MTP entries of the rows past ``keep``, then the drafts for positions
        ``position``, ``position`` + 1, ...: the kept row's first draft ``first`` and ``count`` - 1 chained ones."""

        mtp_cache = cache[-1]
        out, rows = self._spec
        self._spec = None
        if rows > keep:
            mtp_cache.trim(rows - keep, self.args.indexer_compress_ratio)
        if count <= 0:
            return []
        streams = out[keep - 1:keep]
        if not self.queued_chains:
            drafts = [int(first)]
            for j in range(1, count):
                mixed, streams = self._mtp_step([drafts[-1]], streams, mtp_cache)
                mtp_cache.drafted += 1
                drafts.append(int(self._draft_draw(mixed, sampling, [position + j]).item()))
            return drafts
        if count == 1:
            return [int(first)]
        # each chained draft is drawn on the GPU and fed to the next step as an array: no read until the next
        # round's inputs are built
        chain = [mx.array([int(first)], dtype=mx.uint32)]
        for j in range(1, count):
            mixed, streams = self._mtp_step(chain[-1], streams, mtp_cache)
            mtp_cache.drafted += 1
            chain.append(self._draft_draw(mixed, sampling, [position + j]))
        drafts = mx.concatenate(chain)
        mx.async_eval(drafts)
        return drafts

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""

        if getattr(self, "_spec", None) is not None:
            cache[-1].trim(self._spec[1], self.args.indexer_compress_ratio)
            self._spec = None

    # -- load-time check ------------------------------------------------------------------
    def check_windows(self, widest: int | None = None) -> tuple[int, dict[int, float]]:
        """The widest window (up to ``fused_rows``) whose every narrower window gives each row a one-row forward's
        logits bit for bit, from a 48-token prompt; and every exact width's forward time in ms (fastest of 3)."""

        import time

        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        widest = int(widest or self.fused_rows)
        prompt = np.array([[(37 * i + 11) % 50_000 + 1000 for i in range(48)]], dtype=np.int64)
        window = [3001 + 17 * r for r in range(widest)]
        base = self.model.make_cache()
        mx.eval(self.model.hidden(prompt, base))
        one = copy(base)
        serial = []
        for token in window:
            logits = self.head(self.model.hidden(np.array([[token]], dtype=np.int64), one))
            mx.eval(logits)
            serial.append(logits[0, -1])
        exact = 1
        for width in range(2, widest + 1):
            logits = self.head(self.model.hidden(np.array([window[:width]], dtype=np.int64), copy(base)))
            mx.eval(logits)
            if not all(bool(mx.array_equal(logits[0, i], serial[i]).item()) for i in range(width)):
                break
            exact = width
        costs: dict[int, float] = {}
        for width in range(1, exact + 1):
            best = float("inf")
            for _ in range(3):
                cache = copy(base)
                started = time.perf_counter()
                mx.eval(self.head(self.model.hidden(np.array([window[:width]], dtype=np.int64), cache)))
                best = min(best, (time.perf_counter() - started) * 1e3)
            costs[width] = round(best, 3)
        return exact, costs


def load(model_dir: Path, *, drafts: int | None = None) -> tuple[FlashNext, Any]:
    """``drafts`` (default TF_FLASH_MTP, else 3; 0: none): the most drafts a round from the MTP head. The engine
    picks up to this many a round from the stream's recent acceptance: a chained draft step (~1.3 ms) and its
    verify row (~5 ms) pay only at high acceptance. On an M3 Ultra (2026-09-25) a 23k-token agent prompt with a
    512-token thinking budget, then a 2,550-token tool call, ran at 91.1 tok/s with a cap of 3 against 86.8 at 1."""

    import os

    from tensorfold.families.qwen4_exp import model as q4
    from tensorfold.families.qwen4_exp import mtp as mtp_module

    model, tokenizer = q4.load(Path(model_dir))
    drafts = int(os.environ.get("TF_FLASH_MTP", "3")) if drafts is None else int(drafts)
    head = mtp_module.load(Path(model_dir), model.args) if drafts > 0 and model.__dict__.get("fused") else None
    return FlashNext(model, head, drafts=drafts), tokenizer
