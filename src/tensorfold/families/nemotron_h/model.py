"""Expose Nemotron-H backbone/head separation, cache rollback and MTP drafts to the lane engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Use real text for load-time checks so tokens exercise the experts reached during decoding.
_CHECK_TEXT = ("def merge(intervals):\n    \"\"\"Merge overlapping intervals and return them sorted.\"\"\"\n"
               "    intervals = sorted(intervals)\n    out = [intervals[0]]\n    for start, end in intervals[1:]:\n"
               "        if start <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], end)\n        else:\n"
               "            out.append([start, end])\n    return out\n\nThe river ran high that spring, and the ferry "
               "stopped for the first time anyone could remember.")


class NemotronH:
    """Separate backbone and head, using fused kernels for short windows and mlx_lm for longer inputs with the same cache layout."""

    fused_rows = 16
    lane_family = True
    gpu_sampling = True
    # a shared forward's rows and streams (``hidden_rows``): the lane matmul (M5) and ``rows.qmv`` keep a row's bits
    batch_rows = 128
    max_streams = 64
    # Draft from kept rows after reading a round, overlapping GPU work with host preparation of the next round.
    speculate_early = False
    # the head's acceptance at depth 1, 2, ... given the ones before it, until a stream has its own
    draft_prior = (0.8, 0.72, 0.68, 0.62, 0.58, 0.55, 0.5, 0.5)

    def __init__(self, model: Any, *, fused: bool = True, mtp_path: Path | None = None, drafts: int = 4,
                 tokenizer: Any = None) -> None:
        self.model = model
        self.args = model.args
        self.fused = None
        self._last_hidden: Any = None
        self._spec: tuple[Any, int] | None = None
        if fused:
            from tensorfold.kernels.nemotron.lightning.v1 import rows
            from tensorfold.kernels.nemotron.lightning.v1.kernels import FusedDecode, tensor_units

            # Use row-exact expert and projection kernels so row counts cannot change a row's bits.
            self.fused = FusedDecode(model)
            if tensor_units():
                self._install_lane_matmul()
            else:
                print(f"[nemotron] row-exact kernels: {rows.install(self)}", flush=True)
        # the decode step takes its token as a GPU array: one-token rounds run one step ahead
        self.gpu_tokens = self.fused is not None
        # Keep the widest serial-exact window and each exact width's forward time.
        self.exact_width, self.window_costs = (1, {})
        if self.fused is not None:
            self.exact_width, self.window_costs = self.check_windows(tokenizer)
            if self.exact_width < 2:
                print("[nemotron] no verify window reproduces one-token steps with these kernels here: one token a "
                      "round", flush=True)
        self.multi_row_exact = self.exact_width >= 2
        # a shared forward's time past the widest exact window (the engine sizes rounds on it)
        self.shared_costs = (self.time_shared_rows(tokenizer)
                             if self.fused is not None and self.multi_row_exact and self.batch_rows > self.exact_width
                             else {})
        # The MTP head drafts the token after next.
        self.mtp = None
        self.drafts = max(0, int(drafts))
        self.mtp_step_ms = 0.0
        self._draft_ids: Any = None
        self._draft_head: Any = None
        if self.multi_row_exact and self.drafts and mtp_path is not None and Path(mtp_path).is_file():
            from tensorfold.families.nemotron_h import mtp as nemotron_mtp

            self.mtp = nemotron_mtp.load(Path(mtp_path), model.args)
            self._draft_ids, self._draft_head = self._load_draft_head()
            # Drafts need no row-exactness, so head projections use MLX kernels.
            _plain_matmuls(self.mtp)
            if self._draft_head is not None:
                _plain_matmuls(self._draft_head)
            self.mtp_step_ms = self._time_mtp_step()
        if self.multi_row_exact:
            timing = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.window_costs.items()))
            shared = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.shared_costs.items()))
            print(f"[nemotron] windows of up to {self.exact_width} rows reproduce one-token steps here (ms by rows "
                  f"{timing}; shared forwards of up to {self.batch_rows} rows {shared or 'none'}); MTP head "
                  f"{'off' if self.mtp is None else f'on, a step {self.mtp_step_ms:.2f} ms'}", flush=True)

    def _install_lane_matmul(self) -> None:
        """Route dense 4-bit projections and the head through row-exact lane_qmm on tensor-unit GPUs, regrouping weights in place."""

        import mlx.nn as nn

        from tensorfold.kernels.nemotron.lightning.v1.kernels import tensor_units
        from tensorfold.kernels.qwen.dense.v1 import lane_qmm

        if not tensor_units():
            return
        holder = nn.Module()
        holder.model = self.model
        holder.stacked = [stacked for stacked, _ in self.fused.qkv.values()]    # q/k/v, outside the model tree
        lane_qmm.install(holder, rows=lane_qmm.MAX_ROWS, tile=True, wide=True)
        lane_qmm.warm(holder, rows=(1,))
        self.lane_matmul = True
        self.fused.lane_xs = True           # the norm kernels hand the projections their input sums

    lane_matmul = False

    # -- load-time checks ------------------------------------------------------------------------------------------
    def _check_tokens(self, tokenizer: Any, count: int) -> list[int]:
        ids: list[int] = []
        if tokenizer is not None:
            try:
                ids = [int(t) for t in tokenizer.encode(_CHECK_TEXT)]
            except Exception:  # noqa: BLE001 - fall back to fixed ids
                ids = []
        if len(ids) < count:     # fixed ids, inside the vocabulary (MLX's gather reads past the table unchecked)
            ids = [((37 * i + 11) % 50_000 + 1000) % int(self.args.vocab_size) for i in range(count)]
        return ids[:count]

    def release_rounds(self) -> None:
        """Drop the last call's per-row states and hidden rows (no stream keeps rows of it; the next call sets them)."""

        if self.fused is not None:
            self.fused.row_states = {}
        self._last_hidden = None

    def check_windows(self, tokenizer: Any = None, *, widest: int | None = None) -> tuple[int, dict[int, float]]:
        """Return the widest window whose narrower windows all match serial logits bit for bit, with forward times for each exact width."""

        import time

        import mlx.core as mx

        from tensorfold.engine.lane_engine import LaneEngine
        from tensorfold.engine.family_common import cache_arrays

        copy = LaneEngine.copy_single_cache
        widest = int(widest or self.fused_rows)
        ids = self._check_tokens(tokenizer, 48 + widest)
        prompt, window = ids[:48], ids[48:48 + widest]
        base = self.model.make_cache()                    # the model's own caches (no MTP entry)
        mx.eval(self.hidden(mx.array([prompt], dtype=mx.uint32), base), *cache_arrays(base))
        one = copy(base)
        serial = []
        for token in window:
            logits = self.head(self.hidden(mx.array([[token]], dtype=mx.uint32), one))
            mx.eval(logits)
            serial.append(logits[0, -1])
        exact = 1
        for width in range(2, widest + 1):
            logits = self.head(self.hidden(mx.array([window[:width]], dtype=mx.uint32), copy(base)))
            mx.eval(logits)
            if not all(bool(mx.array_equal(logits[0, i], serial[i]).item()) for i in range(width)):
                break
            exact = width
        costs: dict[int, float] = {}
        for width in range(1, exact + 1):
            best = float("inf")
            for _ in range(3):
                cache = copy(base)
                mx.eval(*cache_arrays(cache))
                started = time.perf_counter()
                mx.eval(self.head(self.hidden(mx.array([window[:width]], dtype=mx.uint32), cache)))
                best = min(best, (time.perf_counter() - started) * 1e3)
            costs[width] = round(best, 3)
        return exact, costs

    def time_shared_rows(self, tokenizer: Any = None, totals: tuple[int, ...] = (17, 32, 48, 64, 96, 128)
                         ) -> dict[int, float]:
        """Shared forwards' ms (best of 3) by total rows (2-row streams), until one differs from the solo rounds."""

        import time

        import mlx.core as mx

        from tensorfold.engine.lane_engine import LaneEngine
        from tensorfold.engine.family_common import cache_arrays

        copy = LaneEngine.copy_single_cache
        ids = self._check_tokens(tokenizer, 50)
        base = self.model.make_cache()
        mx.eval(self.hidden(mx.array([ids[:48]], dtype=mx.uint32), base), *cache_arrays(base))
        alone = [self.head(self.hidden(mx.array([ids[48:48 + n]], dtype=mx.uint32), copy(base))) for n in (1, 2)]
        costs: dict[int, float] = {}
        for total in (t for t in totals if t <= self.batch_rows):
            windows = [ids[48:50]] * (total // 2) + [ids[48:49]] * (total % 2)
            best = float("inf")
            for _ in range(3):
                caches = [copy(base) for _ in windows]
                mx.eval(*[a for c in caches for a in cache_arrays(c)])
                started = time.perf_counter()
                mx.eval(logits := self.head(self.hidden_rows(windows, caches)))
                best = min(best, (time.perf_counter() - started) * 1e3)
            if not bool(mx.array_equal(logits, mx.concatenate([alone[len(w) - 1] for w in windows], axis=1)).item()):
                self.batch_rows = max([self.exact_width, *costs])      # a wider round changes a stream's bits here
                break
            costs[total] = round(best, 3)
        return costs

    def _load_draft_head(self) -> tuple[Any, Any]:
        """Load vocabulary-head rows for draft_ids.txt; excluding tokens restricts drafts but cannot change verified output."""

        import mlx.core as mx
        import mlx.nn as nn

        path = Path(__file__).with_name("draft_ids.txt")
        if not path.is_file():
            return None, None
        ids = [int(t) for t in path.read_text().split()]
        full = self.model.lm_head
        weight = full["weight"]
        if getattr(full, "_lane_tiled", False):
            # the lane kernel regrouped the head's weight in place: its rows are not token rows any more
            from tensorfold.kernels.qwen.dense.v1 import lane_qmm

            weight = lane_qmm.untile_weight(weight, full._lane_nt, full.group_size, bits=full.bits)
        rows = mx.array(ids, dtype=mx.int32)
        head = nn.QuantizedLinear(int(weight.shape[1]) * 32 // full.bits, len(ids), bias=False,
                                  group_size=full.group_size, bits=full.bits)
        head.weight, head.scales, head.biases = weight[rows], full.scales[rows], full.biases[rows]
        mx.eval(head.weight, head.scales, head.biases)
        return mx.array(ids, dtype=mx.uint32), head

    def _draft_logits(self, state: Any) -> Any:
        return (self._draft_head if self._draft_head is not None else self.model.lm_head)(state)

    def _time_mtp_step(self) -> float:
        """Return the shortest measured duration of a chained draft step in milliseconds."""

        import time

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        cache = self.mtp.make_cache()
        hidden = mx.zeros((1, 1, int(self.args.hidden_size)), dtype=mx.bfloat16)
        token = mx.array([1000], dtype=mx.uint32)
        best = float("inf")
        for _ in range(6):
            started = time.perf_counter()
            out = self._head_step(hidden, self.model.backbone.embeddings(token.reshape(1, 1)), cache, 1)
            mx.eval(gpu_sample(self._draft_logits(out).reshape(1, -1), None, [0], ids=self._draft_ids))
            best = min(best, (time.perf_counter() - started) * 1e3)
        return round(best, 3)

    # -- the model the lane engine drives ---------------------------------------------------------------------------
    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        from mlx_lm.models.cache import KVCache

        from tensorfold.engine.alternating_kv import AlternatingKVCache

        from mlx_lm.models.cache import ArraysCache

        from tensorfold.families.nemotron_h.state_cache import RowStateCache

        # Alternate attention buffers while prior steps read them; retain Mamba rows for shared-forward commits.
        caches = [AlternatingKVCache() if type(c) is KVCache else RowStateCache(2) if type(c) is ArraysCache else c
                  for c in self.model.make_cache()]
        if self.mtp is not None:
            caches.append(self.mtp.make_cache())     # last: the model's layers never reach it
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """``cache`` (a stored or copied prefix, possibly with mlx_lm's ``KVCache``) with this model's classes."""

        from mlx_lm.models.cache import ArraysCache, KVCache

        from tensorfold.engine.alternating_kv import AlternatingKVCache
        from tensorfold.families.nemotron_h.state_cache import RowStateCache

        for i, item in enumerate(cache):
            if type(item) is KVCache:
                adopted = AlternatingKVCache()
                adopted.keys, adopted.values, adopted.offset = item.keys, item.values, item.offset
                cache[i] = adopted
            elif type(item) is ArraysCache:
                states = RowStateCache(len(item.cache))
                states.cache = list(item.cache)
                cache[i] = states
        if self.mtp is not None and len(cache) == self._layer_caches():
            cache.append(self.mtp.make_cache())      # a prefix stored without the head: drafts see less context
        return cache

    def _layer_caches(self) -> int:
        return sum(1 for layer in self.model.layers if layer.block_type in "M*")

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        """After a forward on ``rows`` rows, keep the first ``keep`` in the model's caches (not the MTP's)."""

        self.fused.keep_rows(cache, rows, self._kept(keep))

    def hidden(self, inputs: Any, cache: list[Any] | None = None, parents: Any = None) -> Any:
        self._chain_only(parents)
        if self.fused is not None and cache is not None and inputs.shape[-1] <= self.fused_rows:
            out = self.fused(inputs, cache)
        else:
            out = self.model.backbone(inputs, cache=cache)
        self._last_hidden = out
        return out

    @staticmethod
    def _chain_only(parents: Any) -> None:
        """Nemotron's head drafts chains: a window whose rows are not a chain is a caller's error."""

        if parents is not None and list(parents) != list(range(-1, len(parents) - 1)):
            raise NotImplementedError("Nemotron verifies draft chains, not trees")

    @staticmethod
    def _kept(keep: Any) -> int:
        """A kept-row count from a count or a path from the root (a chain's path is a prefix)."""

        if isinstance(keep, int):
            return keep
        path = [int(r) for r in keep]
        if path != list(range(len(path))):
            raise NotImplementedError("Nemotron keeps a prefix of a window's rows")
        return len(path)

    def hidden_rows(self, windows: list[Any], caches: list[list[Any]], parents: list[Any] | None = None) -> Any:
        """Return grouped stream rows [1, N, D], advancing only each stream's own cache and preserving standalone bits."""

        import mlx.core as mx

        for rows in parents or ():
            self._chain_only(rows)

        parts = [w.reshape(-1).astype(mx.uint32) if isinstance(w, mx.array) else mx.array([int(t) for t in w],
                                                                                        dtype=mx.uint32)
                 for w in windows]
        lengths = tuple(int(p.shape[0]) for p in parts)
        if sum(lengths) > self.batch_rows:
            raise ValueError(f"hidden_rows: at most {self.batch_rows} rows a call, got {sum(lengths)}")
        out = self.fused.run_streams(mx.concatenate(parts) if len(parts) > 1 else parts[0], lengths, caches)
        self._last_hidden = out
        return out

    def keep_rows_streams(self, caches: list[list[Any]], lengths: tuple[int, ...], keeps: tuple[Any, ...]) -> None:
        """Keep each stream's window prefix after hidden_rows, leaving MTP caches to their own commit path."""

        self.fused.keep_rows_streams(caches, lengths, tuple(self._kept(k) for k in keeps))

    def head(self, hidden: Any) -> Any:
        return self.model.lm_head(hidden)

    def __call__(self, inputs: Any, cache: list[Any] | None = None) -> Any:
        return self.head(self.hidden(inputs, cache))

    # -- the MTP head ---------------------------------------------------------------------------------------------
    @staticmethod
    def _trim_chained(mcache: Any) -> None:
        if getattr(mcache, "drafted", 0):
            mcache.trim(mcache.drafted)
            mcache.drafted = 0

    def _head_step(self, hidden: Any, embeddings: Any, mcache: Any, tail: int | None) -> Any:
        """Cache all hidden/embedding rows, then run the last tail rows through the MTP MoE block, or all rows if tail is None."""

        return self.mtp(hidden, embeddings, mcache, tail=tail)

    def _head_rows(self, hidden: Any, embeddings: Any, mcaches: list[Any], lengths: tuple[int, ...],
                   last_only: bool) -> Any:
        """Cache each stream's rows and return [1, S or N, D] after running its last row or all rows through the MTP MoE block."""

        import mlx.core as mx

        from tensorfold.kernels.nemotron.lightning.v1 import rows
        from tensorfold.kernels.nemotron.lightning.v1.kernels import add_norm, add_norm_moe, route, router_logits

        fused = self.fused
        first, second = self.mtp.layers
        eps = fused.eps_value
        emb2, hid2 = embeddings.reshape(embeddings.shape[1:]), hidden.reshape(hidden.shape[1:])
        x = first.eh_proj(mx.concatenate([mx.fast.rms_norm(emb2, first.enorm.weight, eps),
                                          mx.fast.rms_norm(hid2, first.hnorm.weight, eps)], axis=-1))
        attended = fused._attention_streams(first.mixer, mx.fast.rms_norm(x, first.norm.weight, eps), mcaches,
                                            lengths)
        if last_only and len(lengths) and sum(lengths) != len(lengths):
            last = mx.array([sum(lengths[:i + 1]) - 1 for i in range(len(lengths))], dtype=mx.int32)
            x, attended = x[last], attended[last]
        h, normed = add_norm(x, attended, second.norm.weight, fused.eps)
        mixer = second.mixer
        bias = self.__dict__.get("_mtp_gate_bias")
        if bias is None:
            bias = mixer.gate.e_score_correction_bias.astype(mx.float32)
            mx.eval(bias)
            self._mtp_gate_bias = bias
        experts, weights = route(router_logits(normed, mixer.gate.weight), bias, fused.top_k, fused.scaling)
        routed = rows.experts(mixer.switch_mlp, normed, experts)
        _, out = add_norm_moe(h, routed, weights, mixer.shared_experts(normed), second.final_layernorm.weight,
                              fused.eps)
        return out[None]

    def draft_streams(self, caches: list[list[Any]], follows: list[list[int]], rows: list[list[int]],
                      positions: list[int], samplings: list[Any], depths: list[int]) -> list[Any]:
        """Absorb each stream's consecutive kept rows with their following tokens, then draft chains together as lazy uint32 arrays, or [] at depth zero."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample_rows

        mcaches = [c[-1] for c in caches]
        for mcache in mcaches:
            self._trim_chained(mcache)
        lengths = tuple(len(f) for f in follows)
        starts = [int(r[0]) for r in rows]
        for r, st in zip(rows, starts):
            if list(r) != list(range(st, st + len(r))):
                raise NotImplementedError("Nemotron's head reads consecutive rows")
        hidden = mx.concatenate([self._last_hidden[:, st:st + n] for st, n in zip(starts, lengths)], axis=1)
        tokens = mx.array([int(t) for f in follows for t in f], dtype=mx.uint32)
        state = self._head_rows(hidden, self.model.backbone.embeddings(tokens.reshape(1, -1)), mcaches, lengths,
                                last_only=True)                                   # [1, S, D]
        # a depth's drafts for every stream still drafting in one sampler call; its rows follow ``streams``
        streams = [i for i, depth in enumerate(depths) if depth > 0]
        levels: list[tuple[list[int], Any]] = []
        if streams:
            keep = mx.array(streams, dtype=mx.int32)
            state = state[:, keep] if len(streams) < len(follows) else state
            level = sample_rows(self._draft_logits(state)[0], [samplings[i] for i in streams],
                                [positions[i] for i in streams], ids=self._draft_ids)
            levels.append((streams, level))
        for j in range(1, max(depths, default=0)):
            prior, level = levels[-1]
            active = [k for k, i in enumerate(prior) if depths[i] > j]
            if not active:
                break
            pick = mx.array(active, dtype=mx.int32)
            streams = [prior[k] for k in active]
            state = self._head_rows(state[:, pick], self.model.backbone.embeddings(level[pick].reshape(1, -1)),
                                    [mcaches[i] for i in streams], tuple(1 for _ in streams), last_only=False)
            for i in streams:
                mcaches[i].drafted += 1
            level = sample_rows(self._draft_logits(state)[0], [samplings[i] for i in streams],
                                [positions[i] + j for i in streams], ids=self._draft_ids)
            levels.append((streams, level))
        found: dict[int, list[Any]] = {}
        for streams, level in levels:
            for k, i in enumerate(streams):
                found.setdefault(i, []).append(level[k:k + 1])
        result = [mx.concatenate(found[i]) if i in found else [] for i in range(len(follows))]
        mx.async_eval(*[r for r in result if isinstance(r, mx.array)])
        return result

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """Extend the MTP cache with prompt rows through attention only, without drafting from them."""

        mcache = cache[-1]
        self._trim_chained(mcache)
        embeddings = self.model.backbone.embeddings(next_tokens.reshape(1, -1))
        self._head_step(hidden, embeddings, mcache, 0)

    def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False, rows: list[int] | None = None) -> Any:
        """Absorb rows from start with their following tokens and lazily draft at position + 2 + i, optionally drafting only from the final row."""

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        if rows is not None:
            start = int(rows[0])
            if list(rows) != list(range(start, start + len(rows))):
                raise NotImplementedError("Nemotron's head reads consecutive rows")
        mcache = cache[-1]
        self._trim_chained(mcache)
        tokens = tokens.reshape(-1)
        count = int(tokens.shape[0])
        rows = int(self._last_hidden.shape[1])
        start = start + rows if start < 0 else start
        hidden = self._last_hidden[:, start:start + count]
        out = self._head_step(hidden, self.model.backbone.embeddings(tokens.reshape(1, -1)), mcache,
                              1 if last_only else None)
        self._spec = (out, count, last_only)
        logits = self._draft_logits(out)
        drafted = [position + 1 + count] if last_only else [position + 2 + i for i in range(count)]
        return gpu_sample(logits.reshape(logits.shape[1:]), sampling, drafted, ids=self._draft_ids)

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        """Keep the first keep MTP rows and return first plus chained drafts at successive positions as a lazy uint32 array."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        mcache = cache[-1]
        out, rows, last_only = self._spec
        self._spec = None
        if rows > keep:
            mcache.trim(rows - keep)
        if count <= 0:
            return []
        drafts = [first.reshape(1).astype(mx.uint32) if isinstance(first, mx.array)
                  else mx.array([int(first)], dtype=mx.uint32)]
        state = out[:, -1:] if last_only else out[:, keep - 1:keep]
        for j in range(1, count):
            state = self._head_step(state, self.model.backbone.embeddings(drafts[-1].reshape(1, 1)), mcache, 1)
            mcache.drafted += 1
            logits = self._draft_logits(state)
            drafts.append(gpu_sample(logits.reshape(1, -1), sampling, [position + j], ids=self._draft_ids)
                          .astype(mx.uint32))
        result = mx.concatenate(drafts) if len(drafts) > 1 else drafts[0]
        mx.async_eval(result)
        return result

    def unspeculate(self, cache: list[Any]) -> None:
        """Undo ``speculate`` entirely (the round's rows are absorbed another way)."""

        if self._spec is not None:
            cache[-1].trim(self._spec[1])
            self._spec = None


def _plain_matmuls(module: Any) -> None:
    """Route module and its quantized linears through MLX quantized matmul regardless of installed lane routing."""

    import mlx.nn as nn

    items = [("", module), *module.named_modules()] if hasattr(module, "named_modules") else [("", module)]
    for _, item in items:
        if type(item) is nn.QuantizedLinear:
            item.__class__ = _MLXQuantizedLinear


def _mlx_quantized_linear_class() -> Any:
    import mlx.core as mx
    import mlx.nn as nn

    class MLXQuantizedLinear(nn.QuantizedLinear):
        """``nn.QuantizedLinear`` through ``mx.quantized_matmul`` itself (MLX's own call)."""

        def __call__(self, x: Any) -> Any:
            y = mx.quantized_matmul(x, self["weight"], scales=self["scales"], biases=self.get("biases"),
                                    transpose=True, group_size=self.group_size, bits=self.bits,
                                    mode=getattr(self, "mode", "affine"))
            return y + self["bias"] if "bias" in self else y

    return MLXQuantizedLinear


_MLXQuantizedLinear = _mlx_quantized_linear_class()

MTP_FILE = "mtp-4bit.safetensors"


def find_mtp_head(model_dir: Path, choice: str = "") -> Path | None:
    """Resolve an explicit or environment-selected MTP head, then a sibling checkpoint or the default directory; "0" disables it."""

    import os

    from tensorfold.families.nemotron_h.mtp import DEFAULT_DIR

    choice = choice or os.environ.get("TF_NEMOTRON_MTP", "")
    if choice == "0" or os.environ.get("TF_MTP_ROUNDS", "1") == "0":
        return None
    if choice:
        return Path(choice).expanduser()
    for candidate in (Path(model_dir) / MTP_FILE, DEFAULT_DIR / MTP_FILE):
        if candidate.is_file():
            return candidate
    return None


def load(model_dir: Path, *, mtp_head: str = "", mtp_drafts: int | None = None) -> tuple[Any, Any]:
    """Load the model with an available MTP head unless mtp_drafts is zero, verifying every draft within the configured depth limit."""

    from mlx_lm import load as mlx_load

    mtp_path = None if mtp_drafts == 0 else find_mtp_head(Path(model_dir), mtp_head)
    loaded = mlx_load(str(model_dir))
    drafts = 4 if mtp_drafts is None else int(mtp_drafts)
    return NemotronH(loaded[0], mtp_path=mtp_path, drafts=drafts, tokenizer=loaded[1]), loaded[1]
