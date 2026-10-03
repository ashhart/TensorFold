"""Flash Next's MTP drafts for one stream: the head's step, its draws, and the chain a round queues after its read."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.qwen4_exp.model import _write_back
from tensorfold.families.qwen4_exp.mtp_cache import MTPCache


class MTPDrafts:
    """The MTP head's one-stream drafting, mixed into FlashNext (which holds the head, its fused decode and caches)."""

    def _head_config(self) -> Any:
        from dataclasses import replace

        return replace(self.args, num_hidden_layers=1, layer_types=["sparse_attention"], ple_layer_ids=[])

    def _mtp_step(self, tokens: Any, streams: mx.array, mtp_cache: MTPCache,
                  last_only: bool = False) -> tuple[mx.array, mx.array]:
        """MTP on next tokens and residual streams: reference modules for prompts, fused kernels for decode."""

        head = self.mtp
        rows, wide = streams.shape
        dims = wide // head.streams
        if rows <= self.fused_rows:
            # Fuse embedding rows and centred norms, then run both projections through ``project``.
            from tensorfold.kernels.qwen.flash_next.v1 import attention, base, embed, experts, gdn, hc
            from tensorfold.families.qwen4_exp.decode import project

            eps = self.mtp_fused.eps
            emb = embed.embed_rows(tokens, self.model.model.embed_tokens)                      # [n, D]
            e = project(embed.rms_norm_rows(emb, self._mtp_scales[0], eps), head.fc_embedding)
            normed = embed.rms_norm_rows(streams, self._mtp_scales[1], eps).reshape(rows * head.streams, dims)
            hs = project(normed, head.fc_hidden)
            x = (e[:, None, :] + hs.reshape(rows, head.streams, dims)).reshape(rows, wide)
            mixed = self.mtp_fused.run(x, None, [mtp_cache])
            return mixed, self.mtp_fused.last_streams
        ids = tokens.astype(mx.int32) if isinstance(tokens, mx.array) else mx.array(tokens, dtype=mx.int32)
        emb = self.model.model.embed_tokens(ids)                                            # [n, D]
        e = head.fc_embedding(head.pre_fc_norm_embedding(emb))
        hs = head.fc_hidden(head.pre_fc_norm_hidden(streams).reshape(rows, head.streams, dims))
        x = (e[:, None, :] + hs).reshape(rows, wide)
        layer = head.layers[0]
        if not last_only:
            x = layer(x[None], None, mtp_cache)
            return head.hyper_connection_mixer(x), x[0]
        h = last_row_layer(layer, x[None], mtp_cache)
        return head.hyper_connection_mixer(h), h[0]

    def _draft_draw(self, mixed: mx.array, sampling: Any, positions: Any) -> mx.array:
        """Draw lazy uint32 drafts [n] with the target's keyed rule over the cut head's ids or the whole vocabulary."""

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
        mixed, out = self._mtp_step(tokens, streams, mtp_cache, last_only=True)
        return mixed[:, -1:], out[-1:]

    def draft(self, cache: list[Any], streams: mx.array, tokens: list[int], position: int, sampling: Any,
              count: int | None = None) -> list[int]:
        """Absorb the given residual streams and next tokens, then chain ``count`` drafts starting at ``position``."""

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
        """Absorb rows and draw lazy first drafts at position + 2 + i before the read; settle keeps the kept prefix."""

        mtp_cache = cache[-1]
        self._prepared.pop(id(mtp_cache), None)
        if mtp_cache.drafted:
            mtp_cache.trim(mtp_cache.drafted, self.args.indexer_compress_ratio)
            mtp_cache.drafted = 0
        tokens = tokens.reshape(-1)
        rows = int(tokens.shape[0])
        total = int(self._streams.shape[0])
        start = start + total if start < 0 else start
        mixed, out = self._mtp_step(tokens, self._streams[start:start + rows], mtp_cache)
        self._specs[id(mtp_cache)] = (out, rows)
        if last_only:                      # the last row's draft only (every row still enters the head's cache)
            return self._draft_draw(mixed[:, -1:], sampling, [position + 1 + rows])
        return self._draft_draw(mixed, sampling, [position + 2 + r for r in range(rows)])

    def prepare_settle(self, cache: list[Any], firsts: mx.array, position: int, sampling: Any) -> None:
        """Before the read, build (not queue) the first chained step for keeping every row and all but one."""

        mtp_cache = cache[-1]
        spec = self._specs.get(id(mtp_cache))
        if spec is None or not self.queued_chains:
            return
        out, rows = spec
        if rows < 3:                            # one draft: the next is one too in ~80% of rounds (no chain to build)
            return
        before, built = dict(vars(mtp_cache)), {}
        fused = self.mtp_fused
        every, fused.eval_every = fused.eval_every, 0      # built, not queued: settle's async_eval queues the one used
        try:
            for keep in (rows, rows - 1):
                if keep < 1:
                    continue
                try:
                    if keep < rows:
                        mtp_cache.trim(rows - keep, self.args.indexer_compress_ratio)
                    mtp_cache.chaining = True    # as settle's steps: rows beside the buffers, nothing written in place
                    head = firsts[keep - 1:keep].astype(mx.uint32)
                    mixed, streams = self._mtp_step(head, out[keep - 1:keep], mtp_cache)
                    draw = self._draft_draw(mixed, sampling, [position + keep + 2])
                    mtp_cache.chaining = False
                    built[keep] = (position + keep + 1, streams, draw, dict(vars(mtp_cache)))
                finally:                        # the head's cache as speculate left it
                    mtp_cache.__dict__.clear()
                    mtp_cache.__dict__.update(before)
        finally:
            fused.eval_every = every
        self._prepared[id(mtp_cache)] = built

    def settle(self, cache: list[Any], keep: int, first: int, position: int, sampling: Any, count: int) -> list[int]:
        """Trim speculative MTP entries past ``keep``; return ``first``, then chained drafts from ``position``."""

        mtp_cache = cache[-1]
        out, rows = self._specs.pop(id(mtp_cache))
        built = self._prepared.pop(id(mtp_cache), {}).get(keep)
        if built is not None and (built[0] != position or count < 2 or not self.queued_chains):
            built = None
        if rows > keep and built is None:
            mtp_cache.trim(rows - keep, self.args.indexer_compress_ratio)
        if count <= 0:
            return []
        streams = out[keep - 1:keep]
        if not self.queued_chains:
            drafts = [int(first.item() if isinstance(first, mx.array) else first)]
            for j in range(1, count):
                mixed, streams = self._mtp_step([drafts[-1]], streams, mtp_cache)
                mtp_cache.drafted += 1
                drafts.append(int(self._draft_draw(mixed, sampling, [position + j]).item()))
            return drafts
        # Keep ``first`` and chained draws on the GPU until the next round builds its inputs.
        head = (first.reshape(1).astype(mx.uint32) if isinstance(first, mx.array)
                else mx.array([int(first)], dtype=mx.uint32))
        if count == 1:
            return head if isinstance(first, mx.array) else [int(first)]
        chain = [head]
        mtp_cache.chaining = True              # the steps' rows go beside the buffers the last step still reads
        try:
            for j in range(1, count):
                if j == 1 and built is not None:   # built before the read: its head cache state, then queue it
                    _, streams, draw, after = built
                    mtp_cache.__dict__.clear()
                    mtp_cache.__dict__.update(after)
                    mtp_cache.chaining = True
                    chain.append(draw)
                else:
                    mixed, streams = self._mtp_step(chain[-1], streams, mtp_cache)
                    chain.append(self._draft_draw(mixed, sampling, [position + j]))
                mtp_cache.drafted += 1
                mx.async_eval(chain[-1])       # the GPU starts each step while the host builds the next
        finally:
            mtp_cache.chaining = False
        drafts = mx.concatenate(chain)
        mx.async_eval(drafts)
        return drafts

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""

        self._prepared.pop(id(cache[-1]), None)
        spec = self._specs.pop(id(cache[-1]), None)
        if spec is not None:
            cache[-1].trim(spec[1], self.args.indexer_compress_ratio)

    # Shared rounds preserve each stream's serial bits and obey per-stream and total row limits.


def last_row_layer(layer: Any, x: mx.array, cache: Any) -> mx.array:
    """``layer`` on rows ``x`` [1, R, W]: every row enters its attention cache, only the last row is carried on."""

    mixed, inject = layer.attn_hyper_connection(x)
    h = _write_back(x[:, -1:], layer.self_attn(mixed, cache)[:, -1:], inject[:, -1:])
    mixed, inject = layer.mlp_hyper_connection(h)
    return _write_back(h, layer.mlp(mixed), inject)
