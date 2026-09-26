"""Flash Next as the serial engine serves it: the fused decode plus the checkpoint's MTP head.

``FlashNext`` wraps the model (``model.py``) with what ``family_engine.SerialEngine``
drafts with:

- ``multi_row_exact``: a forward of 2-4 consecutive rows gives each row a
  one-row forward's bits (checked at load on this MLX and GPU), so drafted rows
  are verified exactly;
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
        self.multi_row_exact = self.fused is not None and self.rows_match_serial()
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
        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        out = self.model.hidden(tokens, cache[: self.layer_count])
        fused = self.fused is not None and tokens.shape[1] <= self.fused_rows
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
            d = _sample(self.model.head(mixed), position + j, sampling)
            drafts.append(d)
            if j + 1 < count:
                mixed, out = self._mtp_step([d], out, mtp_cache)
                mtp_cache.drafted += 1
        return drafts

    def speculate(self, cache: list[Any], tokens: mx.array, position: int, sampling: Any) -> mx.array:
        """Before a verify round's tokens are read: the MTP head absorbs every row of the last hidden() call (its
        streams; ``tokens`` [R], the round's sampled tokens still on the GPU, as their next tokens) and draws each
        row's first draft, for positions ``position`` + 2 + r (``position``: the first row's). One read then
        returns both; ``settle`` keeps the kept rows' part. Rows go through the MTP head exactly as ``draft``
        would take the kept ones (the fused kernels give a row the same bits at any row count), so the drafts
        and the MTP cache are the same, a host round trip earlier. Returns the drafts [R] (lazy)."""

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        mtp_cache = cache[-1]
        if mtp_cache.drafted:
            mtp_cache.trim(mtp_cache.drafted, self.args.indexer_compress_ratio)
            mtp_cache.drafted = 0
        rows = int(self._streams.shape[0])
        mixed, out = self._mtp_step(tokens, self._streams, mtp_cache)
        self._spec = (out, rows)
        logits = self.head(mixed)
        return gpu_sample(logits.reshape(logits.shape[1:]), sampling, [position + 2 + r for r in range(rows)])

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
        drafts = [int(first)]
        streams = out[keep - 1:keep]
        for j in range(1, count):
            mixed, streams = self._mtp_step([drafts[-1]], streams, mtp_cache)
            mtp_cache.drafted += 1
            drafts.append(_sample(self.model.head(mixed), position + j, sampling))
        return drafts

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""

        if getattr(self, "_spec", None) is not None:
            cache[-1].trim(self._spec[1], self.args.indexer_compress_ratio)
            self._spec = None

    # -- load-time check ------------------------------------------------------------------
    def rows_match_serial(self) -> bool:
        """Drafted rounds are exact only if a forward of 2-4 rows gives each row a one-row forward's bits."""

        from tensorfold.engine.lane_engine import LaneEngine

        prompt = np.array([[(37 * i + 11) % 50_000 + 1000 for i in range(48)]], dtype=np.int64)
        base = self.model.make_cache()
        mx.eval(self.model.hidden(prompt, base))
        for width in (2, 3, 4):
            one, many = LaneEngine.copy_single_cache(base), LaneEngine.copy_single_cache(base)
            rows = [np.array([[3001 + 17 * r]], dtype=np.int64) for r in range(width)]
            serial = mx.concatenate([self.head(self.model.hidden(t, one)) for t in rows], axis=1)
            window = self.head(self.model.hidden(np.concatenate(rows, axis=1), many))
            if not bool(mx.array_equal(serial, window).item()):
                return False
        return True


def _sample(logits: mx.array, position: int, sampling: Any) -> int:
    if sampling is None:
        return int(mx.argmax(logits.reshape(-1)).item())
    from tensorfold.engine.gpu_sampling import sample as gpu_sample

    return int(gpu_sample(logits.reshape(1, -1), sampling, [position]).item())


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
