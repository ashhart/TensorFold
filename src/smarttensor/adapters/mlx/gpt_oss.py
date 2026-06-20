"""GPT-OSS streaming runner."""

from __future__ import annotations

from .core import *

class GptOssStreamingForwardRunner(StreamingChatRunner):
    """Generate with GPT-OSS through SmartTensor selective-expert streaming.

    GPT-OSS layers are ~875MB of which ~846MB is the 32-expert bank; only the
    top-4 experts fire per token. Each layer streams its ~28MB base
    (attention, norms, router) and then loads only the selected expert rows.
    """

    EXPERT_MARKER = ".mlp.experts."
    # Cold expert must beat the weakest resident by this many selection
    # counts before a hot-table membership rebuild is considered.
    REFRESH_HYSTERESIS = 4

    def __init__(
        self,
        model_dir: str | Path,
        *,
        evaluate: bool = True,
        retain_layers: set[int] | None = None,
        resident_budget_bytes: int | None = None,
        backend: str = "native",
        clear_on_evict: bool = True,
        pin_policy: str = "all",
        warm_embeddings: bool = False,
        expert_hot_set: int = 0,
        trace: bool = False,
        sliding_cache: str = "rotating",
        pack_dir: str | Path | None = None,
        native_layers: int = 0,
        weight_page_budget_bytes: int | None = None,
        weight_page_policy: str = "auto",
        weight_page_rows: int = 1,
        decode_scheduler: str = "auto",
        expert_compute_mode: str = "table",
        expert_prefetch: str = "off",
        expert_prefetch_cap: int = 32,
    ) -> None:
        if pin_policy not in {"all", "phase"}:
            raise ValueError("pin_policy must be 'all' or 'phase'")
        if warm_embeddings and pin_policy != "phase":
            raise ValueError("warm_embeddings only applies to pin_policy='phase'")
        if expert_hot_set < 0:
            raise ValueError("expert_hot_set must be non-negative")
        if sliding_cache not in {"rotating", "kv", "temporal"}:
            raise ValueError("sliding_cache must be 'rotating', 'kv', or 'temporal'")
        if native_layers < 0:
            raise ValueError("native_layers must be non-negative")
        if weight_page_rows < 1:
            raise ValueError("weight_page_rows must be positive")
        if weight_page_rows != 1 and weight_page_budget_bytes is None:
            raise ValueError("weight_page_rows requires weight_page_budget_bytes")
        if decode_scheduler not in {"auto", "serial", "async-lookahead"}:
            raise ValueError("decode_scheduler must be 'auto', 'serial', or 'async-lookahead'")
        if expert_compute_mode not in {"table", "direct_qmm"}:
            raise ValueError("GPT-OSS expert_compute_mode must be 'table' or 'direct_qmm'")
        if expert_compute_mode == "direct_qmm" and expert_hot_set:
            raise ValueError("GPT-OSS direct_qmm currently bypasses hot expert tables")
        if expert_prefetch not in {"off", "previous"}:
            raise ValueError("GPT-OSS expert_prefetch must be 'off' or 'previous'")
        if expert_prefetch != "off" and weight_page_budget_bytes is None:
            raise ValueError("GPT-OSS expert_prefetch requires --weight-page-budget")
        if expert_prefetch_cap < 1:
            raise ValueError("expert_prefetch_cap must be positive")
        # Lean mode (trace=False, the default for serve and benchmarks) skips
        # per-layer/per-token event construction in the hot loop; trace mode
        # restores the full event stream and collects --trace-pass buckets.
        self._trace = trace
        self._pass_trace = PassTraceCollector() if trace else None
        self.session = MlxModelSession(
            model_dir,
            evaluate=evaluate,
            retain_layers=retain_layers,
            resident_budget_bytes=None,
            backend=backend,
            clear_on_evict=clear_on_evict,
            pin_policy=pin_policy,
            warm_embeddings=warm_embeddings,
            trace=trace,
        )
        self.pin_policy = pin_policy
        self.warm_embeddings = warm_embeddings
        self.warm_output = pin_policy == "phase"
        self._embed_prefix = "model.embed_tokens"
        self._base_loaded: set[int] = set()
        self.sliding_cache = sliding_cache
        self.pack_dir = pack_dir
        self.resident_budget_bytes = resident_budget_bytes
        if pack_dir is not None:
            self.session.loader.attach_pack_dir(pack_dir)
        weight_page_policy = resolve_weight_page_policy(
            "gpt_oss",
            weight_page_policy,
            has_weight_page_budget=weight_page_budget_bytes is not None,
        )
        self.weight_page_budget_bytes = weight_page_budget_bytes
        self.weight_page_policy = weight_page_policy
        self.weight_page_rows = weight_page_rows
        self.expert_compute_mode = expert_compute_mode
        self.expert_prefetch = expert_prefetch
        self.expert_prefetch_cap = expert_prefetch_cap
        if weight_page_budget_bytes is not None:
            self.session.loader.attach_weight_page_cache(
                weight_page_budget_bytes,
                eviction_policy=weight_page_policy,
                rows_per_page=weight_page_rows,
            )
        self.decode_scheduler = decode_scheduler
        # Native-resident layers: their full expert bank + base is loaded once
        # into the model shell and the MLX-native fused layer forward runs
        # (native SwitchGLU does top-k internally), skipping the manual
        # router-sync / numpy expert-select / remap / per-layer eval that make
        # the streamed path ~3.2x slower than native on resident weights.
        # Cold layers keep the manual selective streaming path.
        self.native_layers: set[int] = set(range(native_layers))
        self._native_loaded: set[int] = set()
        self._native_resident_bytes = 0
        # Overlapped (async_eval-pipelined) greedy decode is only enabled when
        # the full forward is resident/native and lazy end-to-end. Streaming
        # layers still perform host routing and evictions, so they remain on
        # the serial scheduler until in-flight weight fences exist.
        self._supports_overlap = True
        if self.session.config.get("model_type") != "gpt_oss":
            raise ValueError("GptOssStreamingForwardRunner currently supports model_type=gpt_oss only")
        self.top_k = int(self.session.config.get("num_experts_per_tok", 4))
        # Hot expert residency: keep up to N most-recently-first-seen expert
        # rows per layer resident (lazy-grow: rows enter the set when a miss
        # loads them anyway, so warming costs no extra IO). Measured working
        # set: top-8 of 32 experts serve 81% of selections, top-12 serve 92%.
        self.expert_hot_set = expert_hot_set
        self._hot_tables: dict[int, dict[str, Any]] = {}
        self._hot_counts: dict[int, dict[int, int]] = {}
        # Layers whose expert modules are bound to a temporary (over-cap) pass
        # table rather than the resident hot table; only these need the
        # placeholder clear after the layer forward.
        self._bound_temporary_layers: set[int] = set()
        # Stable slots: the arrays dict currently bound to each layer's expert
        # modules, so all-hit passes skip rebinding identical tables.
        self._bound_tables: dict[int, Any] = {}
        self._expert_names_cache: dict[int, tuple[str, ...]] = {}
        self._expert_cache_bytes = 0
        self._hot_stats = {
            "hit_passes": 0,
            "miss_passes": 0,
            "overflow_passes": 0,
            "refreshes": 0,
            "served_rows": 0,
            "loaded_rows": 0,
            "temporary_rows": 0,
        }
        self._expert_history: dict[int, list[int]] = {}
        self._pending_expert_prefetch: dict[str, Any] | None = None
        self._pending_expert_prefetch_bytes = 0
        self._prefetch_stats = ExpertPrefetchStats()
        self.loader_executor = ThreadPoolExecutor(max_workers=1)
        first_layer = self.session.loader.manifest.layers[min(self.session.loader.manifest.layers)]
        self._expert_row_bytes = sum(
            self.session.loader.manifest.tensors[name].nbytes
            for name in first_layer.tensor_names
            if self.EXPERT_MARKER in name
        ) // 32
        if retain_layers is not None:
            self.base_retain_layers = set(retain_layers)
        elif resident_budget_bytes is not None:
            self.base_retain_layers = select_qwen_base_layers_for_budget(
                self.session.loader.manifest,
                resident_budget_bytes,
                top_k=self.top_k,
                expert_marker=self.EXPERT_MARKER,
            )
        else:
            self.base_retain_layers = set()

        from mlx_lm.utils import load_tokenizer

        self.tokenizer = load_tokenizer(self.session.model_dir)

    def _resident_sidecar_bytes(self) -> int:
        return (
            self._expert_cache_bytes
            + self.session.loader.weight_page_resident_bytes
            + self._pending_expert_prefetch_bytes
        )

    def _async_lookahead_support(self) -> tuple[bool, str]:
        total_layers = len(self.session.model.model.layers)
        if self.pin_policy != "all":
            return False, "requires pin-policy all so embeddings/output stay resident"
        if len(self.native_layers) != total_layers:
            return False, "requires every layer to be native-resident; streamed layers can evict in-flight weights"
        if set(self.native_layers) != set(range(total_layers)):
            return False, "native-resident layers must cover the full contiguous model"
        if self.expert_hot_set:
            return False, "hot expert tables are a streamed-layer optimization, not overlap-safe yet"
        if self.weight_page_budget_bytes is not None:
            return False, "weight page cache can evict streamed weights; async fences are not implemented yet"
        if self.resident_budget_bytes is not None:
            overlap_margin = 256 * 1024 * 1024
            estimated_peak = self.session.resident_bytes + overlap_margin
            if estimated_peak > self.resident_budget_bytes:
                return (
                    False,
                    "resident budget leaves no async-lookahead headroom "
                    f"({estimated_peak} > {self.resident_budget_bytes})",
                )
        return True, "full native-resident GPT-OSS forward is overlap-safe"

    def _make_cache(self) -> list[Any]:
        cache = super()._make_cache()
        if self.sliding_cache == "rotating":
            return cache

        from mlx_lm.models.cache import KVCache

        inner = self.session.model.model
        for index, layer_type in enumerate(inner.layer_types):
            if layer_type != "full_attention":
                if self.sliding_cache == "kv":
                    cache[index] = KVCache()
                else:
                    cache[index] = TemporalSlidingKVCache(inner.window_size)
        return cache

    def _embed_module(self) -> Any:
        return self.session.model.model.embed_tokens

    def next_token(self, prompt: str, *, max_layers: int | None = None) -> MlxForwardResult:
        import mlx.core as mx

        started = time.perf_counter()
        events: list[dict[str, Any]] = []

        pin_event = self._pin_for_run()
        if self._trace:
            events.append({"kind": "load", **compact_event(pin_event)})

        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            raise ValueError("prompt produced no tokens")

        inner = self.session.model.model
        total_layers = len(inner.layers)
        layer_limit = total_layers if max_layers is None else min(max_layers, total_layers)
        cache = [None] * total_layers
        x = self._stream_forward_tokens(
            [tokens],
            cache=cache,
            events=events,
            max_layers=layer_limit,
            pass_kind="prefill",
        )

        next_token: int | None = None
        next_text: str | None = None
        if layer_limit == total_layers:
            logits = self._logits_from_hidden(x, events)
            token_array = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token_array)
            next_token = int(token_array.item())
            next_text = self.tokenizer.decode([next_token])

        return MlxForwardResult(
            prompt=prompt,
            prompt_tokens=len(tokens),
            completed_layers=layer_limit,
            total_layers=total_layers,
            seconds=time.perf_counter() - started,
            next_token=next_token,
            next_text=next_text,
            resident_peak_bytes=self.session.peak_resident_bytes,
            events=tuple(events),
        )

    def close(self) -> None:
        self._drain_gptoss_expert_prefetch()
        self.loader_executor.shutdown(wait=True)
        self._hot_tables.clear()
        self._hot_counts.clear()
        self._bound_temporary_layers.clear()
        self._bound_tables.clear()
        self._expert_cache_bytes = 0
        self.session.close()

    def _reset_stream_state(self) -> None:
        self._drain_gptoss_expert_prefetch()
        self._expert_history.clear()
        self._prefetch_stats = ExpertPrefetchStats()

    def _drain_gptoss_expert_prefetch(self) -> None:
        pending = getattr(self, "_pending_expert_prefetch", None)
        self._pending_expert_prefetch = None
        self._pending_expert_prefetch_bytes = 0
        if pending is not None:
            pending["future"].result()
            self._set_external_resident_bytes()

    def _gptoss_slice_nbytes(self, layer_index: int, expert_ids: list[int]) -> int:
        if not expert_ids:
            return 0
        total = 0
        for name in self._expert_slice_names(layer_index):
            record = self.session.loader.manifest.tensors[name]
            total += record.nbytes * len(expert_ids) // record.shape[0]
        return total

    def _gptoss_slice_page_miss_nbytes(
        self,
        layer_index: int,
        expert_ids: list[int],
        *,
        cap_to_headroom: bool = True,
    ) -> int:
        return self.session.loader.estimate_first_dim_slice_page_miss_bytes(
            self._expert_slice_names(layer_index),
            expert_ids,
            cap_to_headroom=cap_to_headroom,
        )

    def _gptoss_budget_allows(
        self,
        *,
        additional_sidecar_bytes: int = 0,
        temporary_bytes: int = 0,
    ) -> bool:
        budget = getattr(self, "resident_budget_bytes", None)
        if budget is None:
            return True
        return self.session.resident_bytes + additional_sidecar_bytes + temporary_bytes <= budget

    def _maybe_submit_gptoss_expert_prefetch(self, layer_index: int) -> None:
        if self.expert_prefetch != "previous":
            return
        if self._pending_expert_prefetch is not None:
            return
        history = self._expert_history.get(layer_index)
        if not history:
            self._prefetch_stats.skipped_no_history += 1
            return
        if len(history) > self.expert_prefetch_cap:
            self._prefetch_stats.skipped_over_cap += 1
            return

        predicted = list(history)
        page_growth_bytes = self._gptoss_slice_page_miss_nbytes(layer_index, predicted)
        predicted_table_bytes = self._gptoss_slice_nbytes(layer_index, predicted)
        if not self._gptoss_budget_allows(
            additional_sidecar_bytes=page_growth_bytes,
            temporary_bytes=predicted_table_bytes,
        ):
            self._prefetch_stats.skipped_over_budget += 1
            return

        future = self.loader_executor.submit(
            self.session.loader.warm_first_dim_slices,
            self._expert_slice_names(layer_index),
            predicted,
        )
        self._pending_expert_prefetch_bytes = page_growth_bytes
        self._set_external_resident_bytes()
        self._pending_expert_prefetch = {
            "layer": layer_index,
            "experts": predicted,
            "future": future,
        }

    def _consume_gptoss_expert_prefetch(
        self,
        layer_index: int,
        selected_experts: list[int],
        *,
        pass_kind: str,
        token_step: int | None,
        events: list[dict[str, Any]],
    ) -> None:
        pending = self._pending_expert_prefetch
        self._pending_expert_prefetch = None
        self._pending_expert_prefetch_bytes = 0
        if pending is None:
            return

        join_started = time.perf_counter()
        batch = pending["future"].result()
        join_wait = time.perf_counter() - join_started
        self._set_external_resident_bytes()
        if pending["layer"] != layer_index:
            self._prefetch_stats.skipped_no_history += 1
            return

        predicted = pending["experts"]
        _ordering, hits, missing = expert_merge_plan(predicted, selected_experts)
        wasted = sorted(set(predicted) - set(selected_experts))
        stats = self._prefetch_stats
        stats.attempted_layers += 1
        stats.predicted_rows += len(predicted)
        stats.true_rows += len(selected_experts)
        stats.hit_rows += len(hits)
        stats.missing_rows += len(missing)
        stats.wasted_rows += len(wasted)
        stats.prefetched_bytes += batch.nbytes
        row_bytes = batch.nbytes // len(predicted) if predicted else 0
        stats.wasted_bytes += row_bytes * len(wasted)
        stats.prefetch_load_seconds += batch.seconds
        stats.join_wait_seconds += join_wait
        if not missing:
            stats.full_hits += 1
        if self._trace:
            events.append(
                {
                    "kind": "load",
                    "action": "prefetch-warm-expert-pages",
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    "predicted_experts": predicted,
                    "hit_count": len(hits),
                    "missing_count": len(missing),
                    "join_wait_seconds": join_wait,
                    "prefetch_seconds": batch.seconds,
                    "resident_bytes": self.session.resident_bytes,
                    "nbytes_loaded": batch.nbytes,
                }
            )

    def _stream_forward_tokens(
        self,
        token_rows: list[list[int]],
        *,
        cache: list[Any],
        events: list[dict[str, Any]],
        pass_kind: str,
        max_layers: int | None = None,
        token_step: int | None = None,
        manage_embedding: bool = True,
        on_layer: Any | None = None,
        on_layer_stage: Any | None = None,
        input_array: Any | None = None,
    ) -> Any:
        import mlx.core as mx
        from mlx_lm.models.gpt_oss import create_attention_mask

        trace = self._trace
        pass_trace = self._pass_trace
        pass_started = time.perf_counter()
        inner = self.session.model.model
        if self.pin_policy == "phase" and manage_embedding:
            load_embedding, local_token_rows = self._load_embedding_slices(token_rows)
            if trace:
                events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_embedding)})
            inputs = mx.array(local_token_rows)
            embed_started = time.perf_counter()
            try:
                x = inner.embed_tokens(inputs)
                if pass_trace is not None:
                    pass_trace.add("embedding_eval", time.perf_counter() - embed_started)
                if trace:
                    events.append(
                        {
                            "kind": "compute",
                            "action": "embedding",
                            "pass": pass_kind,
                            "token_step": token_step,
                            "seconds": time.perf_counter() - embed_started,
                            "hidden_shape": list(x.shape),
                        }
                    )
            finally:
                self._clear_embedding_slices()
                if trace:
                    evict_embedding = MlxStreamEvent(
                        action="evict-embedding-slices",
                        seconds=0.0,
                        resident_bytes=self.session.resident_bytes,
                        requested=load_embedding.requested,
                        evicted=load_embedding.loaded,
                    )
                    self.session.events.append(evict_embedding)
                    events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_embedding)})
        else:
            # input_array (a lazy [batch,1] token array) lets the overlapped
            # decoder feed the previous step's still-unmaterialized token
            # straight into this forward, so the GPU pipelines instead of
            # waiting on a Python .item() between tokens.
            inputs = input_array if input_array is not None else mx.array(token_rows)
            embed_started = time.perf_counter()
            x = inner.embed_tokens(inputs)
            if pass_trace is not None:
                pass_trace.add("embedding_eval", time.perf_counter() - embed_started)
            if trace:
                events.append(
                    {
                        "kind": "compute",
                        "action": "embedding",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "seconds": time.perf_counter() - embed_started,
                        "hidden_shape": list(x.shape),
                    }
                )

        full_mask = create_attention_mask(x, cache[inner.ga_idx])
        swa_mask = create_attention_mask(x, cache[inner.swa_idx], window_size=inner.window_size)

        total_layers = len(inner.layers)
        layer_limit = total_layers if max_layers is None else min(max_layers, total_layers)

        for layer_index in range(layer_limit):
            layer = inner.layers[layer_index]
            layer_cache = cache[layer_index]
            layer_type = inner.layer_types[layer_index]
            mask = full_mask if layer_type == "full_attention" else swa_mask
            self._maybe_submit_gptoss_expert_prefetch(layer_index)

            # Native-resident layer: full bank + base loaded once into the
            # shell; run MLX's fused layer forward, no manual MoE / clear / evict.
            if layer_index in self.native_layers:
                if layer_index not in self._native_loaded:
                    native_started = time.perf_counter()
                    self._load_native_layer(layer_index)
                    if pass_trace is not None:
                        pass_trace.add("native_layer_load", time.perf_counter() - native_started)
                compute_started = time.perf_counter()
                x = layer(x, mask, layer_cache)
                if pass_trace is not None:
                    pass_trace.add("native_layer_compute", time.perf_counter() - compute_started)
                if trace:
                    events.append({"kind": "compute", "action": "native-layer", "pass": pass_kind,
                                   "token_step": token_step, "layer": layer_index,
                                   "seconds": time.perf_counter() - compute_started, "hidden_shape": list(x.shape)})
                if on_layer is not None:
                    mx.eval(x)
                    on_layer(layer_index, x, cache[layer_index])
                continue

            if layer_index in self._base_loaded:
                pass  # retained base resident since its first pass: skip bookkeeping
            else:
                load_event = self._load_layer_base(layer_index)
                if pass_trace is not None:
                    pass_trace.add("layer_base_load", load_event.seconds)
                if trace:
                    events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_event)})
                if layer_index in self.base_retain_layers:
                    self._base_loaded.add(layer_index)

            compute_started = time.perf_counter()
            try:
                x = self._layer_forward_selective_experts(
                    layer_index,
                    layer,
                    x,
                    mask=mask,
                    cache=layer_cache,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                    on_stage=on_layer_stage,
                )
            finally:
                if self.expert_compute_mode != "direct_qmm":
                    clear_started = time.perf_counter()
                    self._clear_selected_experts(layer_index, layer)
                    if pass_trace is not None:
                        pass_trace.add("clear_experts", time.perf_counter() - clear_started)
            if trace:
                events.append(
                    {
                        "kind": "compute",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "seconds": time.perf_counter() - compute_started,
                        "hidden_shape": list(x.shape),
                    }
                )
            if on_layer is not None:
                mx.eval(x)
                on_layer(layer_index, x, cache[layer_index])

            if layer_index not in self.base_retain_layers:
                evict_started = time.perf_counter()
                evict_event = self.session.evict_layer(layer_index)
                if pass_trace is not None:
                    pass_trace.add("evict", time.perf_counter() - evict_started)
                if trace:
                    events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_event)})

        if pass_trace is not None:
            pass_trace.add_pass(time.perf_counter() - pass_started)
        return x

    def _load_layer_base(self, layer_index: int) -> MlxStreamEvent:
        layer = self.session.loader.manifest.layers[layer_index]
        names = tuple(name for name in layer.tensor_names if self.EXPERT_MARKER not in name)
        return self.session._load_into_model(names, action="load-layer-base", layer=layer_index)

    def _load_native_layer(self, layer_index: int) -> None:
        """Load a layer's full bank + base resident into the shell, once.

        After this the layer's modules hold native (32-expert) weights, so the
        fused ``layer(x, mask, cache)`` forward runs without any manual expert
        selection. Resident for the runner's lifetime (not evicted/cleared).
        """

        layer = self.session.loader.manifest.layers[layer_index]
        names = tuple(layer.tensor_names)
        self.session._load_into_model(names, action="load-native-layer", layer=layer_index)
        self._native_loaded.add(layer_index)
        self._native_resident_bytes += sum(
            self.session.loader.manifest.tensors[name].nbytes for name in names
        )
        self._set_external_resident_bytes()

    def _layer_forward_selective_experts(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
        on_stage: Any | None = None,
    ) -> Any:
        residual = x
        hidden = layer.input_layernorm(x)
        if on_stage is not None:
            on_stage(layer_index, "attention_norm", hidden)
        if isinstance(cache, TemporalSlidingKVCache) and 1 < hidden.shape[1] <= 16:
            hidden = self._temporal_sliding_attention(layer.self_attn, hidden, cache)
        else:
            hidden = layer.self_attn(hidden, mask, cache)
        if on_stage is not None:
            on_stage(layer_index, "attention", hidden)
        x = residual + hidden
        if on_stage is not None:
            on_stage(layer_index, "attention_residual", x)

        residual = x
        hidden = layer.post_attention_layernorm(x)
        if on_stage is not None:
            on_stage(layer_index, "moe_norm", hidden)
        moe_out = self._moe_forward_selective_experts(
            layer_index,
            layer.mlp,
            hidden,
            events=events,
            pass_kind=pass_kind,
            token_step=token_step,
        )
        if on_stage is not None:
            on_stage(layer_index, "moe", moe_out)
        return residual + moe_out

    def _temporal_sliding_attention(self, attention: Any, x: Any, cache: TemporalSlidingKVCache) -> Any:
        import mlx.core as mx
        from mlx_lm.models.gpt_oss import scaled_dot_product_attention

        batch, length, _ = x.shape
        head_dim = attention.head_dim
        query_heads = attention.num_attention_heads
        kv_heads = attention.num_key_value_heads
        previous_offset = cache.offset

        q = attention.q_proj(x).reshape(batch, length, query_heads, head_dim).swapaxes(1, 2)
        k = attention.k_proj(x).reshape(batch, length, kv_heads, head_dim).swapaxes(1, 2)
        v = attention.v_proj(x).reshape(batch, length, kv_heads, head_dim).swapaxes(1, 2)

        q = attention.rope(q, offset=previous_offset)
        k = attention.rope(k, offset=previous_offset)
        keys, values = cache.update_and_fetch(k, v)

        outputs = []
        for position in range(length):
            absolute = previous_offset + position
            start_absolute = max(cache.start_position, absolute - cache.max_size + 1)
            end_absolute = absolute + 1
            start = start_absolute - cache.start_position
            end = end_absolute - cache.start_position
            outputs.append(
                scaled_dot_product_attention(
                    q[:, :, position : position + 1, :],
                    keys[:, :, start:end, :],
                    values[:, :, start:end, :],
                    cache,
                    attention.sm_scale,
                    mask=None,
                    sinks=attention.sinks,
                )
            )

        attended = mx.concatenate(outputs, axis=2)
        return attention.o_proj(attended.swapaxes(1, 2).reshape(batch, length, -1))

    def _moe_forward_selective_experts(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        *,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx
        from mlx_lm.models.gpt_oss import mlx_topk

        trace = self._trace
        pass_trace = self._pass_trace

        gates = mlp.router(x)
        values, expert_indices = mlx_topk(gates, k=mlp.num_experts_per_tok, axis=-1)
        expert_weights = mx.softmax(values, axis=-1, precise=True)

        sync_started = time.perf_counter() if pass_trace is not None else 0.0
        # One host round-trip per layer: np.asarray materializes the indices
        # (the load plan is host-driven), and ids/counts/remap all reuse it.
        np_indices = np.asarray(expert_indices)
        unique_ids, unique_counts = np.unique(np_indices, return_counts=True)
        selected_experts = [int(expert) for expert in unique_ids]
        expert_counts = {
            int(expert): int(count) for expert, count in zip(unique_ids, unique_counts)
        }
        if pass_trace is not None:
            pass_trace.add("router_indices_sync", time.perf_counter() - sync_started)
        self._consume_gptoss_expert_prefetch(
            layer_index,
            selected_experts,
            pass_kind=pass_kind,
            token_step=token_step,
            events=events,
        )
        self._expert_history[layer_index] = selected_experts

        if self.expert_compute_mode == "direct_qmm":
            return self._moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                np_indices,
                selected_experts,
                expert_weights,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        if self.expert_hot_set > 0:
            table_order, load_event = self._serve_experts_from_hot_set(
                layer_index, mlp, selected_experts, expert_counts
            )
        else:
            load_event = self._load_selected_experts(layer_index, mlp, selected_experts)
            table_order = selected_experts
        if trace and load_event is not None:
            events.append(
                {
                    "kind": "load",
                    "action": load_event.action,
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    **compact_event(load_event),
                }
            )

        remap_started = time.perf_counter() if pass_trace is not None else 0.0
        local_indices = remap_expert_indices(np_indices, table_order)
        if pass_trace is not None:
            pass_trace.add("remap", time.perf_counter() - remap_started)
        expert_y = mlp.experts(x, local_indices)
        expert_y = expert_y * mx.expand_dims(expert_weights, axis=-1)
        return expert_y.sum(axis=-2)

    def _moe_forward_direct_qmm(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        expert_weights: Any,
        *,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx

        trace = self._trace
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        batch = self.session.loader.load_first_dim_slices(
            names,
            selected_experts,
            evaluate=self.session.evaluate,
        )
        self._set_external_resident_bytes(batch.nbytes)
        if self._pass_trace is not None:
            self._pass_trace.add("expert_load", time.perf_counter() - started)
        try:
            local_indices = remap_expert_indices(expert_indices, selected_experts)
            x_expanded = mx.expand_dims(x, (-2, -3))
            up = self._gptoss_projection_qmm(
                mlp,
                layer_index,
                "up_proj",
                x_expanded,
                local_indices,
                batch.arrays,
            )
            gate = self._gptoss_projection_qmm(
                mlp,
                layer_index,
                "gate_proj",
                x_expanded,
                local_indices,
                batch.arrays,
            )
            expert_y = self._gptoss_projection_qmm(
                mlp,
                layer_index,
                "down_proj",
                mlp.experts.activation(up, gate),
                local_indices,
                batch.arrays,
            ).squeeze(-2)
            if trace:
                events.append(
                    {
                        "kind": "load",
                        "action": "load-selected-experts-direct-qmm",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        "seconds": time.perf_counter() - started,
                        "resident_bytes": self.session.resident_bytes,
                        "nbytes_loaded": batch.nbytes,
                    }
                )
            expert_y = expert_y * mx.expand_dims(expert_weights, axis=-1)
            return expert_y.sum(axis=-2)
        finally:
            self._set_external_resident_bytes()

    def _gptoss_projection_qmm(
        self,
        mlp: Any,
        layer_index: int,
        projection: str,
        inputs: Any,
        local_indices: Any,
        arrays: dict[str, Any],
    ) -> Any:
        import mlx.core as mx

        prefix = f"model.layers.{layer_index}.mlp.experts.{projection}"
        module = getattr(mlp.experts, projection)
        projected = mx.gather_qmm(
            inputs,
            arrays[f"{prefix}.weight"],
            arrays[f"{prefix}.scales"],
            arrays.get(f"{prefix}.biases"),
            rhs_indices=local_indices,
            transpose=True,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
            sorted_indices=False,
        )
        bias = arrays.get(f"{prefix}.bias")
        if bias is not None:
            projected = projected + mx.expand_dims(bias[local_indices], -2)
        return projected

    def _expert_slice_names(self, layer_index: int) -> tuple[str, ...]:
        names = self._expert_names_cache.get(layer_index)
        if names is None:
            prefix = f"model.layers.{layer_index}.mlp.experts."
            names = tuple(
                sorted(
                    name
                    for name in self.session.loader.manifest.layers[layer_index].tensor_names
                    if name.startswith(prefix)
                )
            )
            self._expert_names_cache[layer_index] = names
        return names

    def _load_selected_experts(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
    ) -> MlxStreamEvent | None:
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        batch = self.session.loader.load_first_dim_slices(
            names,
            selected_experts,
            evaluate=self.session.evaluate,
        )
        for name in names:
            parts = name.split(".")
            module = getattr(mlp.experts, parts[-2])
            setattr(module, parts[-1], batch.arrays[name])
        self._set_external_resident_bytes(batch.nbytes)
        if self._pass_trace is not None:
            self._pass_trace.add("expert_load", time.perf_counter() - started)
        if not self._trace:
            return None
        event = MlxStreamEvent(
            action="load-selected-experts",
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names,
            nbytes_loaded=batch.nbytes,
        )
        self.session.events.append(event)
        return event

    def _serve_experts_from_hot_set(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
        selected_counts: dict[int, int],
    ) -> tuple[list[int], MlxStreamEvent | None]:
        import mlx.core as mx

        pass_trace = self._pass_trace
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        hot = self._hot_tables.setdefault(layer_index, {"order": [], "arrays": {}})
        counts = self._hot_counts.setdefault(layer_index, {})
        for expert, count in selected_counts.items():
            counts[expert] = counts.get(expert, 0) + count

        hot_ids = set(hot["order"])
        missing = [expert for expert in selected_experts if expert not in hot_ids]

        nbytes_loaded = 0
        timed_sub_seconds = 0.0
        if missing:
            self._hot_stats["miss_passes"] += 1
            self._hot_stats["loaded_rows"] += len(missing)
            load_started = time.perf_counter()
            batch = self.session.loader.load_first_dim_slices(
                names, missing, evaluate=self.session.evaluate
            )
            if pass_trace is not None:
                load_seconds = time.perf_counter() - load_started
                timed_sub_seconds += load_seconds
                pass_trace.add("expert_load", load_seconds)
            assembly_started = time.perf_counter()
            nbytes_loaded = batch.nbytes
            if hot["order"]:
                arrays = {
                    name: mx.concatenate([hot["arrays"][name], batch.arrays[name]], axis=0)
                    for name in names
                }
            else:
                arrays = dict(batch.arrays)
            order = hot["order"] + missing
            if len(order) <= self.expert_hot_set:
                # Lazy-grow: adopt the merged table as the new resident set.
                mx.eval(list(arrays.values()))
                hot["order"], hot["arrays"] = order, arrays
                self._expert_cache_bytes = self._expert_row_bytes * sum(
                    len(table["order"]) for table in self._hot_tables.values()
                )
                temporary_bytes = nbytes_loaded
                table_order, table_arrays = order, arrays
            else:
                # The layer's resident set is full. Serve hot hits from the
                # cached table, merge in the just-loaded misses, but keep only
                # the active rows for this pass instead of duplicating the
                # whole hot table into a temporary over-cap table.
                self._hot_stats["overflow_passes"] += 1
                self._hot_stats["temporary_rows"] += len(selected_experts)
                table_order, table_arrays = self._gather_expert_table(
                    names, hot, batch.arrays, missing, selected_experts
                )
                refresh_order = (
                    choose_frequency_hot_order(
                        hot["order"],
                        missing,
                        counts,
                        self.expert_hot_set,
                    )
                    if should_refresh_hot_set(
                        hot["order"], missing, counts, self.REFRESH_HYSTERESIS
                    )
                    else hot["order"]
                )
                if set(refresh_order) != set(hot["order"]):
                    # Membership changed: rebuild the resident table. Pure
                    # reorderings are skipped — frequency counts permute the
                    # ranking almost every pass, and a rebuild copies 16 rows
                    # x 12 tensors for zero served-row difference (measured
                    # 328 rebuilds in 65 passes before this check).
                    hot["order"], hot["arrays"] = self._gather_expert_table(
                        names, hot, batch.arrays, missing, refresh_order
                    )
                    self._expert_cache_bytes = self._expert_row_bytes * sum(
                        len(table["order"]) for table in self._hot_tables.values()
                    )
                    self._hot_stats["refreshes"] += 1
                temporary_bytes = nbytes_loaded + self._expert_row_bytes * len(table_order)
            if pass_trace is not None:
                assembly_seconds = time.perf_counter() - assembly_started
                timed_sub_seconds += assembly_seconds
                pass_trace.add("expert_assembly", assembly_seconds)
        else:
            self._hot_stats["hit_passes"] += 1
            table_order, table_arrays = hot["order"], hot["arrays"]
            temporary_bytes = 0
        self._hot_stats["served_rows"] += len(selected_experts) - len(missing)

        if self._bound_tables.get(layer_index) is not table_arrays:
            for name in names:
                parts = name.split(".")
                setattr(getattr(mlp.experts, parts[-2]), parts[-1], table_arrays[name])
            self._bound_tables[layer_index] = table_arrays
        if table_arrays is hot["arrays"]:
            self._bound_temporary_layers.discard(layer_index)
        else:
            self._bound_temporary_layers.add(layer_index)
        self._set_external_resident_bytes(temporary_bytes)

        if pass_trace is not None:
            pass_trace.add(
                "expert_serve_python",
                time.perf_counter() - started - timed_sub_seconds,
            )
        if not self._trace:
            return table_order, None
        event = MlxStreamEvent(
            action="hot-set-experts" if not missing else "hot-set-experts-miss",
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names if missing else (),
            nbytes_loaded=nbytes_loaded,
        )
        self.session.events.append(event)
        return table_order, event

    def _gather_expert_table(
        self,
        names: tuple[str, ...],
        hot: dict[str, Any],
        missing_arrays: dict[str, Any],
        missing: list[int],
        members: list[int],
    ) -> tuple[list[int], dict[str, Any]]:
        """Assemble a table containing ``members`` as hot-rows-first.

        Membership is what matters — remap adapts to any row order — so the
        table is built with two gathers (one from the resident hot table, one
        from the just-loaded miss batch) instead of per-row slice+concat.
        Returns the actual row order used and the materialized arrays.
        """

        import mlx.core as mx

        hot_position = {expert: index for index, expert in enumerate(hot["order"])}
        missing_position = {expert: index for index, expert in enumerate(missing)}
        hot_members = [expert for expert in members if expert in hot_position]
        cold_members = [expert for expert in members if expert not in hot_position]
        hot_idx = (
            mx.array([hot_position[expert] for expert in hot_members])
            if hot_members
            else None
        )
        cold_idx = (
            mx.array([missing_position[expert] for expert in cold_members])
            if cold_members
            else None
        )
        order = hot_members + cold_members
        arrays: dict[str, Any] = {}
        for name in names:
            parts = []
            if hot_idx is not None:
                parts.append(mx.take(hot["arrays"][name], hot_idx, axis=0))
            if cold_idx is not None:
                parts.append(mx.take(missing_arrays[name], cold_idx, axis=0))
            arrays[name] = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        mx.eval(list(arrays.values()))
        return order, arrays

    def _expert_telemetry(self) -> dict[str, Any] | None:
        weight_pages = self.session.loader.weight_page_summary()
        native_resident = {
            "resident_layers": len(self._native_loaded),
            "requested_layers": len(self.native_layers),
            "resident_bytes": self._native_resident_bytes,
        }
        prefetch_stats = (
            self._prefetch_stats.to_dict()
            if self.expert_prefetch != "off"
            else None
        )
        if self.expert_hot_set <= 0:
            if (
                weight_pages is None
                and not self.native_layers
                and self.expert_prefetch == "off"
            ):
                return None
            return {
                "mode": "resident-sidecars",
                "native_layers": native_resident,
                "weight_page_budget_bytes": self.weight_page_budget_bytes,
                "weight_pages": weight_pages,
                "weight_page_policy": self.weight_page_policy,
                "weight_page_rows": self.weight_page_rows,
                "expert_prefetch_mode": self.expert_prefetch,
                "expert_prefetch": prefetch_stats,
            }
        stats = dict(self._hot_stats)
        total_rows = stats["served_rows"] + stats["loaded_rows"]
        return {
            "mode": "expert-hot-set",
            "rows_cap_per_layer": self.expert_hot_set,
            "resident_rows": sum(len(table["order"]) for table in self._hot_tables.values()),
            "resident_expert_bytes": self._expert_cache_bytes,
            "native_layers": native_resident,
            "weight_pages": weight_pages,
            "weight_page_budget_bytes": self.weight_page_budget_bytes,
            "weight_page_policy": self.weight_page_policy,
            "weight_page_rows": self.weight_page_rows,
            "expert_prefetch_mode": self.expert_prefetch,
            "expert_prefetch": prefetch_stats,
            "row_hit_rate": stats["served_rows"] / total_rows if total_rows else 0.0,
            **stats,
        }

    def _clear_selected_experts(self, layer_index: int, layer: Any) -> None:
        import mlx.core as mx

        if self.expert_hot_set > 0 and layer_index not in self._bound_temporary_layers:
            # The bound table IS the resident hot table: rebinding zero
            # placeholders here and the real table next pass was pure churn
            # (24 layers x 12 tensors per pass) and released no memory.
            return
        self._bound_temporary_layers.discard(layer_index)
        self._bound_tables.pop(layer_index, None)
        manifest = self.session.loader.manifest
        for name in self._expert_slice_names(layer_index):
            record = manifest.tensors[name]
            parts = name.split(".")
            module = getattr(layer.mlp.experts, parts[-2])
            empty_shape = (0,) * len(record.shape)
            setattr(module, parts[-1], mx.zeros(empty_shape, dtype=mlx_dtype(record.dtype)))
        self._set_external_resident_bytes()

    def _logits_from_hidden(self, hidden: Any, events: list[dict[str, Any]]) -> Any:
        import mlx.core as mx

        if self.pin_policy == "phase" and not self.warm_output:
            before_output_load = getattr(self, "_before_phase_output_load", None)
            if before_output_load is not None:
                before_output_load(events)
            load_output = self.session.load_output()
            if self._trace:
                events.append({"kind": "load", **compact_event(load_output)})

        logits_started = time.perf_counter()
        inner = self.session.model.model
        hidden = inner.norm(hidden)
        logits = self.session.model.lm_head(hidden)
        if self._pass_trace is not None:
            self._pass_trace.add("logits_eval", time.perf_counter() - logits_started)
        if self._trace:
            events.append(
                {
                    "kind": "compute",
                    "action": "logits",
                    "seconds": time.perf_counter() - logits_started,
                    "logits_shape": list(logits.shape),
                }
            )
        if self.pin_policy == "phase" and not self.warm_output:
            evict_output = self.session.evict_output()
            if self._trace:
                events.append({"kind": "evict", **compact_event(evict_output)})
        return logits
