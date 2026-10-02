"""Flash Next fused decoding and MTP drafting, with every emitted token verified against the target's sample."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from tensorfold.families.qwen4_exp.model import select_by_kernels
from tensorfold.families.qwen4_exp.mtp_cache import MTPCache
from tensorfold.families.qwen4_exp.mtp_chain import MTPDrafts, last_row_layer  # noqa: F401 (tests import it here)


class FlashNext(MTPDrafts):
    """Flash Next with the backbone and head apart, the fused decode, and MTP drafting."""

    fused_rows = 16
    lane_family = True
    # Draw with gpu_sampling's keyed rule on the GPU.
    gpu_sampling = True
    # the engine fills a prompt a chunk a forward: a pass holds every chunk's layer temporaries (+44-60 GiB served)
    prompt_pass = False

    def __init__(self, model: Any, head: Any | None = None, *, drafts: int = 1) -> None:
        self.model = model
        self.args = model.args
        self.fused = model.__dict__.get("fused")
        self.layer_count = len(model.layers)
        self.mtp = None
        self.drafts = int(drafts)
        self._specs: dict[int, tuple[mx.array, int]] = {}        # head cache id -> (streams out, rows) of speculate
        self._prepared: dict[int, dict[int, tuple]] = {}         # head cache id -> keep -> its built first step
        self.exact_width, self.window_costs = self.check_windows() if self.fused is not None else (1, {})
        if self.fused is not None:
            self._warm_sparse()
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

    def _warm_sparse(self) -> None:
        """The sparse attention kernels' decode variants built at load, not inside the first long request."""

        from tensorfold.kernels.qwen.flash_next.v1 import attention

        entry = next((e for e in self.fused.layers if "attn" in e), None)
        if entry is None:
            return
        c, a = self.args, entry["attn"][-1]
        width = ((2 * c.num_attention_heads + 2 * c.num_key_value_heads) * c.head_dim
                 + (c.indexer_n_heads + 1) * c.indexer_head_dim)      # [q|gate] pairs, k, v, indexer q, raw key
        attention.warm_decode(heads=c.num_attention_heads, kv_heads=c.num_key_value_heads, dims=c.head_dim,
                              index_heads=c.indexer_n_heads, index_dims=c.indexer_head_dim, top=a.indexer.top_blocks,
                              scale=a.scale, width=width, norm=entry["attn"][4], eps=self.fused.eps,
                              rotary_dim=c.rotary_dim, base=c.rope_theta)

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
        self.__dict__.get("_prepared", {}).clear()
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

        if isinstance(inputs, mx.array) and self.fused is not None and inputs.size <= self.fused_rows:
            window = inputs.reshape(1, -1)
            tables = self.fused.ple_tables
            if tables is not None and tables.host is not None:
                mx.async_eval(window)      # host tables: the n-gram layer reads the ids, which get their own buffer
            out = self.fused(window, cache[: self.layer_count])       # the n-gram ids are hashed on the GPU
            self._streams = self.fused.last_streams
            return out
        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        if tokens.shape[1] > self.fused_rows and "_resolved_prefill_identity" in self.__dict__:
            self.prefill_key  # refuse a changed prefill mode before reading or updating a keyed cache
        out = self.model.hidden(tokens, cache[: self.layer_count])
        fused = self.fused is not None and tokens.shape[0] == 1 and tokens.shape[1] <= self.fused_rows
        self._streams = self.fused.last_streams if fused else self.model.__dict__["last_streams"]
        return out

    def hidden_pass(self, inputs: Any, cache: list[Any], sizes: Any) -> mx.array:
        """Consecutive prompt chunks in one forward (``sizes`` rows each), every chunk with its own forward's bits."""

        tokens = np.asarray(inputs, dtype=np.int64)
        if tokens.ndim == 1:
            tokens = tokens[None]
        if min(int(n) for n in sizes) <= self.fused_rows:     # such a chunk alone takes the fused decode kernels
            raise ValueError(f"hidden_pass: every chunk needs over {self.fused_rows} rows, got {tuple(sizes)}")
        if "_resolved_prefill_identity" in self.__dict__:
            self.prefill_key  # refuse a changed prefill mode before reading or updating a keyed cache
        out = self.model.hidden_pass(tokens, cache[: self.layer_count], sizes)
        self._streams = self.model.__dict__["last_streams"]
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
        from tensorfold.kernels.qwen.flash_next.v1 import rows

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
        # the allocator prices rounds at the per-row kernels' costs: at the tiles' cheaper 8+ rows, 2 streams lost 4%
        before, rows.hc_tiles_on = rows.hc_tiles_on, False
        try:
            for width in range(1, exact + 1):
                best = float("inf")
                for _ in range(3):
                    cache = copy(base)
                    started = time.perf_counter()
                    mx.eval(self.head(self.model.hidden(np.array([window[:width]], dtype=np.int64), cache)))
                    best = min(best, (time.perf_counter() - started) * 1e3)
                costs[width] = round(best, 3)
        finally:
            rows.hc_tiles_on = before
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
