"""Flash Next fused decoding and MTP drafting, with every emitted token verified against the target's sample."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.qwen4_exp.model import AttentionCache, _write_back, select_by_kernels


class MTPCache(AttentionCache):
    """Track MTP attention entries and chained drafts, trimming drafts before absorbing kept rows."""

    drafted = 0


class FlashNext:
    """Flash Next with the backbone and head apart, the fused decode, and MTP drafting."""

    fused_rows = 16
    lane_family = True
    # Draw with gpu_sampling's keyed rule on the GPU.
    gpu_sampling = True

    def __init__(self, model: Any, head: Any | None = None, *, drafts: int = 1) -> None:
        self.model = model
        self.args = model.args
        self.fused = model.__dict__.get("fused")
        self.layer_count = len(model.layers)
        self.mtp = None
        self.drafts = int(drafts)
        self._specs: dict[int, tuple[mx.array, int]] = {}        # head cache id -> (streams out, rows) of speculate
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
            select_by_kernels(head.layers)
            self._mtp_scales = [1.0 + head.pre_fc_norm_embedding.weight.astype(mx.float32),
                                1.0 + head.pre_fc_norm_hidden.weight.astype(mx.float32)]
            mx.eval(*self._mtp_scales)
            import os

            # TF_FLASH_DRAFT_VOCAB=0 scores the full vocabulary; TF_FLASH_QUEUED=0 reads each chained draft immediately.
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
        """Estimate one chained MTP step in milliseconds for the depth rule until measured rounds replace the estimate."""

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

    def release_rounds(self) -> None:
        """Drop the last forward's rollback and draft rows when no stream is live; the next forward rebuilds them."""

        for fused in (self.fused, getattr(self, "mtp_fused", None)):
            if fused is not None:
                fused.row_states.clear()
                fused._last_heads.clear()
                fused.last_streams = None
        self._streams = None
        self._specs.clear()
        self.model.__dict__.pop("last_streams", None)

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """A stored or copied cache without an MTP entry gets an empty one (drafts then see less context)."""

        if self.mtp is not None and len(cache) == self.layer_count:
            cache.append(MTPCache())
        return cache

    @property
    def prefill_key(self) -> str:
        """How prompt chunks are prefilled (path, matmul route, GPU), for snapshot keys: each rounds differently."""

        from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

        fast = self.fused is not None and prefill_mm.fast_prefill()
        key = "flash-prefill=" + ("fast;" + prefill_mm.prefill_identity() if fast else "mlx")
        resolved = self.__dict__.get("_resolved_prefill_identity")
        if resolved is not None and key != resolved:
            raise RuntimeError("Flash Next's prefill arithmetic changed after the snapshot key was fixed: reload")
        return key

    def prefetch_prompt(self, tokens: Any, begin: int, end: int) -> None:
        """Start reading the host n-gram rows prompt chunk [begin, end) will look up (tables on the host only)."""

        for layer in self.model.layers:
            emb = layer.ple.ple_embedding if "ple" in layer else None
            read_ahead = getattr(getattr(emb, "host", None), "read_ahead", None)
            if read_ahead is None:
                continue
            before = [emb.eos] * emb.context + [int(t) for t in tokens[max(0, begin - emb.context):begin]]
            history = np.array([before[len(before) - emb.context:]], dtype=np.int64)
            read_ahead(emb.ids(history, np.array([[int(t) for t in tokens[begin:end]]], dtype=np.int64)))

    def tighten_prefill(self) -> bool:
        """Queue a prompt chunk one layer at a time (about half its working memory, the same bits); False if it does."""

        if self.model.__dict__.get("prefill_queue") == 1:
            return False
        self.model.__dict__["prefill_queue"] = 1
        return True

    def resolve_prefill_identity(self) -> None:
        """Fix the actual matmul route before snapshot keying; its required self-check belongs to startup."""

        from tensorfold.kernels.qwen.flash_next.v1 import prefill_mm

        if self.fused is not None and prefill_mm.fast_prefill():
            prefill_mm.tiles()
        self._resolved_prefill_identity = self.prefill_key

    def hidden(self, inputs: Any, cache: list[Any]) -> mx.array:
        """Mixed hidden states [1, R, D]: the fused kernels up to ``fused_rows`` rows, else a prompt chunk's path."""

        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        if tokens.shape[1] > self.fused_rows and "_resolved_prefill_identity" in self.__dict__:
            self.prefill_key  # refuse a changed prefill mode before reading or updating a keyed cache
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

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """The MTP cache takes rows ``start`` .. of the last forward, one a next token (prompt rows)."""

        tokens = [int(t) for t in np.asarray(next_tokens).reshape(-1)]
        self._absorb(self._streams[start:start + len(tokens)], tokens, cache[-1])

    def _head_config(self) -> Any:
        from dataclasses import replace

        return replace(self.args, num_hidden_layers=1, layer_types=["sparse_attention"], ple_layer_ids=[])

    def _mtp_step(self, tokens: Any, streams: mx.array, mtp_cache: MTPCache,
                  last_only: bool = False) -> tuple[mx.array, mx.array]:
        """Run MTP on next tokens and residual streams, using reference modules for prompts and fused kernels for decode."""

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
        """Absorb rows and draw lazy first drafts at position + 2 + i before readback; ``settle`` keeps the accepted prefix."""

        mtp_cache = cache[-1]
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

    def settle(self, cache: list[Any], keep: int, first: int, position: int, sampling: Any, count: int) -> list[int]:
        """Trim speculative MTP entries past ``keep``, then return ``first`` followed by chained drafts from ``position``."""

        mtp_cache = cache[-1]
        out, rows = self._specs.pop(id(mtp_cache))
        if rows > keep:
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
        for j in range(1, count):
            mixed, streams = self._mtp_step(chain[-1], streams, mtp_cache)
            mtp_cache.drafted += 1
            chain.append(self._draft_draw(mixed, sampling, [position + j]))
        drafts = mx.concatenate(chain)
        mx.async_eval(drafts)
        return drafts

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""

        spec = self._specs.pop(id(cache[-1]), None)
        if spec is not None:
            cache[-1].trim(spec[1], self.args.indexer_compress_ratio)

    # Shared rounds preserve each stream's serial bits and obey per-stream and total row limits.
    max_streams = 32
    batch_rows = 64
    rows_per_call = 128

    def hidden_rows(self, windows: list[Any], caches: list[list[Any]]) -> mx.array:
        """Return stream-ordered mixed states [1, N, D], advancing each cache independently and hashing token ids on the host."""

        lazy = [w for w in windows if isinstance(w, mx.array)]
        if lazy:
            mx.eval(*lazy)          # Read every stream's window in one GPU round trip.
        host = [[int(t) for t in (w.reshape(-1).tolist() if isinstance(w, mx.array) else w)] for w in windows]
        return self.hidden_multi(host, caches)

    def keep_rows_streams(self, caches: list[list[Any]], lengths: Any, keeps: Any) -> None:
        """Keep each stream's requested prefix after ``hidden_rows``; ``settle`` manages the head cache."""

        for cache, length, keep in zip(caches, lengths, keeps):
            if int(keep) < int(length):
                self.fused.keep_rows(cache, int(length), int(keep))

    def hidden_multi(self, inputs: list[Any], caches: list[list[Any]]) -> mx.array:
        """Return mixed states [1, N, D] in stream order from host token windows and their separate caches."""

        from tensorfold.kernels.qwen.flash_next.v1 import embed

        tokens = [np.asarray(t, dtype=np.int64).reshape(1, -1) for t in inputs]
        rows = [int(t.shape[1]) for t in tokens]
        if len(rows) == 1:
            return self.hidden(tokens[0], caches[0])
        if len(rows) > 64 or sum(rows) > self.rows_per_call or self.fused is None:
            raise ValueError(f"hidden_multi: at most 64 streams and {self.rows_per_call} rows in all")
        h = embed.embed_rows(np.concatenate(tokens, axis=1).reshape(-1), self.model.model.embed_tokens,
                             tile=self.args.hc_count)
        out = self.fused.run_multi(h, tokens, [c[: self.layer_count] for c in caches], rows)
        self._streams = self.fused.last_streams
        return out

    def _mtp_step_multi(self, tokens: Any, streams: mx.array, mtp_caches: list[MTPCache], rows: list[int]
                        ) -> tuple[mx.array, mx.array]:
        """``_mtp_step`` over several streams' rows (each stream's rows through its own head cache)."""

        if len(rows) == 1:
            return self._mtp_step(tokens, streams, mtp_caches[0])
        from tensorfold.kernels.qwen.flash_next.v1 import attention, base, embed, experts, gdn, hc
        from tensorfold.families.qwen4_exp.decode import project

        head = self.mtp
        total, wide = streams.shape
        dims = wide // head.streams
        eps = self.mtp_fused.eps
        emb = embed.embed_rows(tokens, self.model.model.embed_tokens)
        e = project(embed.rms_norm_rows(emb, self._mtp_scales[0], eps), head.fc_embedding)
        normed = embed.rms_norm_rows(streams, self._mtp_scales[1], eps).reshape(total * head.streams, dims)
        hs = project(normed, head.fc_hidden)
        x = (e[:, None, :] + hs.reshape(total, head.streams, dims)).reshape(total, wide)
        mixed = self.mtp_fused.run_multi(x, None, [[m] for m in mtp_caches], rows)
        return mixed, self.mtp_fused.last_streams

    def draft_streams(self, caches: list[list[Any]], follows: list[list[int]], rows: list[list[int]],
                      positions: list[int], samplings: list[Any], depths: list[int]) -> list[Any]:
        """Draft each stream to its requested depth after absorbing its kept rows and following tokens from ``hidden_rows``."""

        mtp = [c[-1] for c in caches]
        for m in mtp:
            if m.drafted:
                m.trim(m.drafted, self.args.indexer_compress_ratio)
                m.drafted = 0
        index = mx.array([int(r) for kept in rows for r in kept], dtype=mx.int32)
        tokens = mx.array([int(t) for f in follows for t in f], dtype=mx.uint32)
        mixed, out = self._mtp_step_multi(tokens, mx.take(self._streams, index, axis=0), mtp, [len(k) for k in rows])
        lasts = [sum(len(k) for k in rows[:i + 1]) - 1 for i in range(len(rows))]
        chains: list[list[mx.array]] = [[] for _ in rows]
        live = [i for i, d in enumerate(depths) if d > 0]
        for i in live:
            chains[i].append(self._draft_draw(mixed[:, lasts[i]:lasts[i] + 1], samplings[i], [positions[i]]))
        streams = {i: out[lasts[i]:lasts[i] + 1] for i in live}
        step = 1
        while True:
            live = [i for i in live if depths[i] > step]
            if not live:
                break
            mixed, grown = self._mtp_step_multi(mx.concatenate([chains[i][-1] for i in live]),
                                                mx.concatenate([streams[i] for i in live]),
                                                [mtp[i] for i in live], [1] * len(live))
            for j, i in enumerate(live):
                mtp[i].drafted += 1
                chains[i].append(self._draft_draw(mixed[:, j:j + 1], samplings[i], [positions[i] + step]))
                streams[i] = grown[j:j + 1]
            step += 1
        drafts = [mx.concatenate(chain) if chain else [] for chain in chains]
        mx.async_eval(*[d for d in drafts if isinstance(d, mx.array)])
        return drafts

    # -- load-time check ------------------------------------------------------------------
    def check_windows(self, widest: int | None = None) -> tuple[int, dict[int, float]]:
        """Check the widest window whose prefixes match serial logits bit for bit, and return each exact width's timing."""

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
        if exact >= 2:
            self.exact_width = exact                         # hidden_multi's per-stream limit, for the check
            self.streams_exact = self._check_streams(base, window)
            if not self.streams_exact:
                self.max_streams = 1
                print("[flash-next] a forward over several streams' rows does not reproduce each stream's own call "
                      "here: one stream a call", flush=True)
        return exact, costs

    streams_exact = False

    def _check_streams(self, base: list[Any], window: list[int]) -> bool:
        """Check multi-stream results against separate calls, including another step after retaining partial windows."""

        from tensorfold.engine.lane_engine import LaneEngine

        copy = LaneEngine.copy_single_cache
        lengths, keeps = [3, 1, 4], [2, 1, 1]

        def streams() -> list[list[Any]]:
            out = []
            for b in range(len(lengths)):
                cache = copy(base)
                for extra in range(b):                      # different lengths: b more tokens each
                    mx.eval(self.model.hidden(np.array([[5001 + 13 * b + extra]], dtype=np.int64), cache))
                out.append(cache)
            return out

        inputs = [[window[(3 * b + r) % len(window)] for r in range(n)] for b, n in enumerate(lengths)]
        follow = [[window[(5 * b + 7) % len(window)]] for b in range(len(lengths))]
        alone, multi = streams(), streams()
        for step in range(2):
            wins = inputs if step == 0 else follow
            single = []
            for b, cache in enumerate(alone):
                logits = self.head(self.model.hidden(np.array([wins[b]], dtype=np.int64), cache))
                mx.eval(logits)
                single.append(logits[0])
                if step == 0 and keeps[b] < len(wins[b]):
                    self.fused.keep_rows(cache, len(wins[b]), keeps[b])
            together = self.head(self.hidden_multi(wins, multi))[0]
            mx.eval(together)
            at = 0
            for b, win in enumerate(wins):
                if not bool(mx.array_equal(together[at:at + len(win)], single[b]).item()):
                    return False
                at += len(win)
            if step == 0:
                self.keep_rows_streams(multi, [len(w) for w in wins], keeps)
        return True


def last_row_layer(layer: Any, x: mx.array, cache: Any) -> mx.array:
    """``layer`` on rows ``x`` [1, R, W]: every row enters its attention cache, only the last row is carried on."""

    mixed, inject = layer.attn_hyper_connection(x)
    h = _write_back(x[:, -1:], layer.self_attn(mixed, cache)[:, -1:], inject[:, -1:])
    mixed, inject = layer.mlp_hyper_connection(h)
    return _write_back(h, layer.mlp(mixed), inject)


def load(model_dir: Path, *, drafts: int | None = None, ple_on_ssd: bool = False,
         ssd_experts: float | None = None) -> tuple[FlashNext, Any]:
    """Load with an MTP draft cap from ``drafts`` or TF_FLASH_MTP, defaulting to 3; zero disables drafts."""

    import os

    from tensorfold.families.qwen4_exp import MODELS, decode
    from tensorfold.families.qwen4_exp import model as q4
    from tensorfold.families.qwen4_exp import mtp as mtp_module

    model, tokenizer = q4.load(Path(model_dir), ple_on_ssd=ple_on_ssd, ssd_experts=ssd_experts)
    drafts = int(os.environ.get("TF_FLASH_MTP", "3")) if drafts is None else int(drafts)
    head = mtp_module.load(Path(model_dir), model.args) if drafts > 0 and model.__dict__.get("fused") else None
    missed = decode.unreadable(model, head) if decode.DENSE == "lane" else {}   # simd_qmm checks a shape on first use
    if missed:
        kinds = ", ".join(f"{n} {kind}" for kind, n in sorted(missed.items()))
        raise SystemExit(f"[tensorfold] Qwen3.8 Flash Next: the lane matmul does not read this checkpoint's {kinds} "
                         f"linears. Use {MODELS[0]}")
    runtime = FlashNext(model, head, drafts=drafts)
    q4.prefetch_ngrams(model)
    return runtime, tokenizer
