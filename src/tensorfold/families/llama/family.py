"""Llama rows keep the stock arithmetic per query and never mix streams' attention prefixes."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache

from .config import _shape, normalize


class _NativeQuantizedLinear(nn.QuantizedLinear):
    """Keep stock prefill arithmetic even when another model installs global quantized routing."""

    def __call__(self, x: Any) -> Any:
        y = mx.quantized_matmul(x, self["weight"], scales=self["scales"], biases=self.get("biases"),
                                transpose=True, group_size=self.group_size, bits=self.bits, mode=self.mode)
        return y + self["bias"] if "bias" in self else y


class LlamaFamily:
    lane_family = True
    exact_width = 1
    batch_rows = 16
    max_streams = 16

    def __init__(self, model: Any, *, widest: int = 16) -> None:
        if not 1 <= widest <= self.batch_rows:
            raise ValueError("Llama width must be in 1..16")
        config = normalize(vars(model.args))
        self.inner = model
        self.core = model.model
        self.lm_head = model.lm_head
        if (not isinstance(self.lm_head, nn.Linear) or isinstance(self.lm_head, nn.QuantizedLinear)
                or not isinstance(self.core.embed_tokens, nn.Embedding)
                or isinstance(self.core.embed_tokens, nn.QuantizedEmbedding)):
            raise ValueError("Llama requires unquantized bf16 head and embedding")  # noqa: TRY004
        if len(self.core.layers) != config["num_hidden_layers"]:
            raise ValueError("Llama: unsupported layer shape")
        for name, module in model.named_modules():
            packed = isinstance(module, nn.QuantizedLinear)
            bias_required = (".self_attn." in name and config.get("attention_bias", False)
                             or ".mlp." in name and config.get("mlp_bias", False))
            if ((packed or isinstance(module, nn.Linear)) and bias_required and "bias" not in module
                    or "bias" in module and not isinstance(module["bias"], mx.array)):
                raise ValueError(f"Llama: invalid model bias shape: {name}")
            if packed:
                if (module.bits, module.group_size, module.mode) != (8, 64, "affine"):
                    raise ValueError("Llama reads affine 8-bit/group-64 projections only")
                if any(not isinstance(module.get(key), mx.array) for key in ("weight", "scales", "biases")):
                    raise ValueError(f"Llama: incomplete packed weight shape: {name}")
                from tensorfold.kernels.qwen.dense.v1.affine_rows import shape

                if shape(module.weight, module.scales, module.biases, 64, 8) != tuple(_shape(name, config)):
                    raise ValueError(f"Llama: unsupported packed weight shape: {name}")
            elif isinstance(module, (nn.Linear, nn.Embedding, nn.RMSNorm)) and not isinstance(module.get("weight"), mx.array):
                raise ValueError(f"Llama: missing weight shape: {name}")
            for key, value in module.items():
                if not isinstance(value, mx.array):
                    continue
                dtype = mx.uint32 if packed and key == "weight" else mx.bfloat16
                if value.dtype != dtype:
                    raise ValueError(f"Llama requires {'packed U32' if dtype == mx.uint32 else 'bf16'} weights: {name}.{key}")
                expected = _shape(name, config)
                if packed and key in ("weight", "scales", "biases"):
                    expected = [expected[0], expected[1] // (4 if key == "weight" else 64)]
                elif key == "bias" and len(expected) == 2:
                    expected = [expected[0]]
                elif key != "weight":
                    raise ValueError(f"Llama: unsupported parameter shape: {name}.{key}")
                if list(value.shape) != expected:
                    raise ValueError(f"Llama: unsupported parameter shape: {name}.{key}")
        self.backend = None
        self.qmm_paths = ()
        quantized = [m for _, m in self.core.named_modules() if isinstance(m, nn.QuantizedLinear)]
        if quantized:
            for module in quantized:
                module.__class__ = _NativeQuantizedLinear
            from tensorfold.kernels.qwen.dense.v1.row_matmul import simd_qmm_backend

            self.backend = simd_qmm_backend()
            self.backend.prepare([(m.weight, m.scales, m.biases, m.group_size, m.bits) for m in quantized])
            from tensorfold.kernels.qwen.dense.v1 import affine_rows, simd_qmm_bits

            # Preparation can change process-wide fallback. Capture direct primitives once, before qualification.
            paths = {(int(m.weight.shape[0]), int(m.weight.shape[1]) * 4, m.bits, m.group_size):
                     simd_qmm_bits.fits(m.weight, m.scales, m.biases, m.group_size, m.bits) for m in quantized}
            simd, affine = simd_qmm_bits.qmm, affine_rows.qmm
            self.qmm_paths = tuple((*shape, "simd_qmm_bits" if fast else "affine_rows")
                                   for shape, fast in sorted(paths.items()))

            def frozen_qmm(x: Any, w: Any, s: Any, b: Any, gs: int, bits: int) -> Any:
                if paths[(int(w.shape[0]), int(w.shape[1]) * 4, bits, gs)]:
                    return simd(x, w, s, b, bits, gs)
                return affine(x, w, s, b, gs, bits)

            self.backend.qmm = frozen_qmm
        self._last: dict[int, tuple[int, int]] = {}
        self.exact_width, self.window_costs = self.check_windows(widest)
        self.batch_rows = self.exact_width
        self.streams_exact = self.check_streams()

    def make_cache(self) -> list[Any]:
        return self.inner.make_cache()

    @staticmethod
    def _each(module: Any, x: Any) -> Any:
        return mx.concatenate([module(x[:, i:i + 1]) for i in range(int(x.shape[1]))], axis=1)

    def project(self, module: Any, x: Any) -> Any:
        """Affine offsets and model bias stay separate; dense calls retain one-row MLX dispatch."""
        if isinstance(module, nn.QuantizedLinear):
            y = self.backend(x, module.weight, module.scales, module.biases, module.group_size, module.bits)
            return y + module.bias if "bias" in module else y
        return self._each(module, x)

    def hidden(self, inputs: Any, cache: list[Any], parents: Sequence[int] | None = None) -> Any:
        if inputs.ndim != 2 or inputs.shape[0] != 1:
            raise ValueError("Llama: token rows must have shape [1, rows]")
        return self.hidden_rows([inputs.reshape(-1)], [cache], None if parents is None else [parents])

    def hidden_rows(self, windows: Sequence[Any], caches: Sequence[list[Any]],
                    parents: Sequence[Sequence[int]] | None = None) -> Any:
        lengths = [int(w.size) if isinstance(w, mx.array) else len(w) for w in windows]
        if (not lengths or len(lengths) != len(caches) or len(caches) > self.max_streams
                or len({id(c) for cache in caches for c in cache}) != sum(map(len, caches))):
            raise ValueError("Llama requires independent stream caches")
        if (sum(lengths) > self.batch_rows or any(n < 1 or n > self.exact_width for n in lengths)
                or len(lengths) > 1 and not getattr(self, "streams_exact", True)):
            raise ValueError("Llama: unsupported window width")
        if parents is not None and (len(parents) != len(lengths) or any(
                list(p) != [-1, *range(n - 1)] for p, n in zip(parents, lengths))):
            raise ValueError("Llama supports chain parents only")
        arrays = [w.reshape(1, -1) if isinstance(w, mx.array) else mx.array([w]) for w in windows]
        for array, cache in zip(arrays, caches):
            self._validate(array, cache)
        for cache, n in zip(caches, lengths):
            self._last[id(cache)] = (int(cache[0].offset), n)
        x = self.core.embed_tokens(mx.concatenate(arrays, axis=1))
        rows = sum(lengths)
        for depth, layer in enumerate(self.core.layers):
            attn = layer.self_attn
            normed = self._each(layer.input_layernorm, x)
            q, k, v = [self.project(m, normed) for m in (attn.q_proj, attn.k_proj, attn.v_proj)]
            q = q.reshape(1, rows, attn.n_heads, -1).transpose(0, 2, 1, 3)
            k = k.reshape(1, rows, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            v = v.reshape(1, rows, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            outputs, at = [], 0
            for cache, length in zip(caches, lengths):
                state = cache[depth]
                for i in range(at, at + length):
                    query = attn.rope(q[:, :, i:i + 1], offset=state.offset)
                    key = attn.rope(k[:, :, i:i + 1], offset=state.offset)
                    keys, values = state.update_and_fetch(key, v[:, :, i:i + 1])
                    out = scaled_dot_product_attention(query, keys, values, cache=state, scale=attn.scale, mask=None)
                    outputs.append(out.transpose(0, 2, 1, 3).reshape(1, 1, -1))
                at += length
            x = x + self.project(attn.o_proj, mx.concatenate(outputs, axis=1))
            normed = self._each(layer.post_attention_layernorm, x)
            gate, up = [self.project(m, normed) for m in (layer.mlp.gate_proj, layer.mlp.up_proj)]
            x = x + self.project(layer.mlp.down_proj, swiglu(gate, up))
        return self._each(self.core.norm, x)

    def head(self, hidden: Any) -> Any:
        return self._each(self.lm_head, hidden)

    def _validate(self, inputs: Any, cache: list[Any]) -> None:
        if (inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.size < 1
                or not mx.issubdtype(inputs.dtype, mx.integer)
                or not bool(mx.all((inputs >= 0) & (inputs < self.inner.args.vocab_size)).item())):
            raise ValueError("Llama: invalid token IDs")
        if (len(cache) != len(self.core.layers) or any(type(c) is not KVCache for c in cache)
                or len({int(c.offset) for c in cache}) != 1):
            raise ValueError("Llama: incompatible cache")
        for layer, state in zip(self.core.layers, cache):
            keys, values = state.keys, state.values
            if (keys is None) != (values is None) or (keys is None and state.offset != 0) or state.offset < 0:
                raise ValueError("Llama: incomplete cache buffers")
            if keys is not None:
                attn = layer.self_attn
                if (keys.shape != values.shape or keys.ndim != 4 or keys.shape[:2] != (1, attn.n_kv_heads)
                        or keys.shape[-1] != attn.head_dim or keys.shape[2] < state.offset
                        or keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16):
                    raise ValueError("Llama: invalid cache geometry or dtype")
        limit = self.inner.args.max_position_embeddings
        if limit is not None and cache[0].offset + inputs.size > limit:
            raise ValueError("Llama: context limit exceeded")

    def prefill(self, inputs: Any, cache: list[Any]) -> Any:
        self._validate(inputs, cache)
        self._last.pop(id(cache), None)
        return self.core(inputs, cache=cache)

    def _kept(self, cache: list[Any], rows: int, keep: Any) -> int:
        record = self._last.get(id(cache))
        if record is None:
            raise ValueError("Llama: missing rollback cache")
        start, width = record
        if isinstance(keep, int) and not 0 <= keep <= rows:
            raise ValueError("Llama keeps a chain prefix only")
        path = list(range(keep)) if isinstance(keep, int) else list(keep)
        if rows != width or path != list(range(len(path))) or not 0 <= len(path) <= rows:
            raise ValueError("Llama keeps a chain prefix only")
        if any(int(c.offset) != start + rows for c in cache):
            raise ValueError("Llama: stale rollback cache")
        return len(path)

    def keep_rows(self, cache: list[Any], rows: int, keep: Any) -> None:
        count = self._kept(cache, rows, keep)
        for state in cache:
            state.trim(rows - count)
        self._last.pop(id(cache))

    def keep_rows_streams(self, caches: Sequence[list[Any]], lengths: Sequence[int], keeps: Sequence[Any]) -> None:
        if not len(caches) == len(lengths) == len(keeps):
            raise ValueError("Llama: mismatched stream keeps")
        if len({id(c) for c in caches}) != len(caches):
            raise ValueError("Llama requires independent stream keeps")
        for cache, rows, keep in zip(caches, lengths, keeps):
            self._kept(cache, rows, keep)
        for cache, rows, keep in zip(caches, lengths, keeps):
            self.keep_rows(cache, rows, keep)

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        self._validate(mx.array([[0]]), cache)
        return cache

    def release_rounds(self) -> None:
        self._last.clear()

    @staticmethod
    def _same_cache(cache: list[Any], reference: list[Any]) -> bool:
        return all(a.offset == b.offset and all(bool(mx.array_equal(x[:, :, :a.offset], y[:, :, :b.offset]).item())
                   for x, y in ((a.keys, b.keys), (a.values, b.values))) for a, b in zip(cache, reference))

    def check_windows(self, widest: int) -> tuple[int, dict[int, float]]:
        """Check every supported width's hidden and head bits, and time evaluated forwards."""
        from tensorfold.engine.lane_engine import LaneEngine

        vocab = int(self.inner.args.vocab_size)
        base = self.make_cache()
        mx.eval(self.prefill(mx.array([[t % vocab for t in range(32)]]), base))
        tokens = [t % vocab for t in range(41, 41 + widest)]
        serial = LaneEngine.copy_single_cache(base)
        parts, prefixes = [], []
        for token in tokens:
            parts.append(self.hidden(mx.array([[token]]), serial))
            prefixes.append(LaneEngine.copy_single_cache(serial))
        ref = mx.concatenate(parts, axis=1)
        ref_logits = self.head(ref)
        mx.eval(ref, ref_logits)
        costs, exact = {}, 1
        self.exact_width = widest
        for width in range(1, widest + 1):
            best = float("inf")
            for rep in range(2):
                cache = LaneEngine.copy_single_cache(base)
                began = time.perf_counter()
                hidden = self.hidden(mx.array([tokens[:width]]), cache)
                logits = self.head(hidden)
                mx.eval(hidden, logits)
                best = min(best, (time.perf_counter() - began) * 1000)
                if rep == 0 and not (bool(mx.array_equal(hidden, ref[:, :width]).item())
                                     and bool(mx.array_equal(logits, ref_logits[:, :width]).item())
                                     and self._same_cache(cache, prefixes[width - 1])):
                    self.release_rounds()
                    if width == 1:
                        raise ValueError("Llama: width-one probe is not exact")
                    return exact, costs
                if rep == 0:
                    keep = width // 2
                    self.keep_rows(cache, width, keep)
                    continuation = self.hidden(mx.array([[tokens[keep]]]), cache)
                    if not (bool(mx.array_equal(continuation, ref[:, keep:keep + 1]).item())
                            and bool(mx.array_equal(self.head(continuation), ref_logits[:, keep:keep + 1]).item())
                            and self._same_cache(cache, prefixes[keep])):
                        self.release_rounds()
                        raise ValueError("Llama: rollback continuation is not exact")
            exact = width
            costs[width] = best
        self.release_rounds()
        return exact, costs

    def check_streams(self) -> bool:
        """Qualify uneven streams at different offsets and measure shared forwards, not solo proxies."""
        from tensorfold.engine.lane_engine import LaneEngine

        vocab = int(self.inner.args.vocab_size)
        bases = [self.make_cache() for _ in range(3)]
        for length, cache in zip((11, 23, 37), bases):
            mx.eval(self.prefill(mx.array([[t % vocab for t in range(length)]]), cache))
        costs = {}
        for total in range(2, self.batch_rows + 1):
            lengths = [1, 1] if total == 2 else [1, total // 2, (total - 1) // 2]
            lengths = [n for n in lengths if n > 0]
            windows = [[(41 + i * 7 + t) % vocab for t in range(n)] for i, n in enumerate(lengths)]
            references = [LaneEngine.copy_single_cache(base) for base in bases[:len(lengths)]]
            refs = [self.hidden(mx.array([window]), cache) for window, cache in zip(windows, references)]
            ref = mx.concatenate(refs, axis=1)
            ref_logits = self.head(ref)
            mx.eval(ref, ref_logits)
            cache = [LaneEngine.copy_single_cache(base) for base in bases[:len(lengths)]]
            began = time.perf_counter()
            hidden = self.hidden_rows(windows, cache)
            logits = self.head(hidden)
            mx.eval(hidden, logits)
            costs[total] = (time.perf_counter() - began) * 1000
            if not (bool(mx.array_equal(hidden, ref).item()) and bool(mx.array_equal(logits, ref_logits).item())
                    and all(self._same_cache(c, r) for c, r in zip(cache, references))):
                self.release_rounds()
                self.shared_costs = {}
                return False
        self.release_rounds()
        self.shared_costs = costs
        return bool(costs)
