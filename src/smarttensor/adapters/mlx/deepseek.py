"""DeepSeek V3/R1 streaming runner."""

from __future__ import annotations

from .core import *

class DeepSeekV3StreamingForwardRunner(StreamingChatRunner):
    """Generate with DeepSeek V3/R1 through SmartTensor selective MoE streaming."""

    EXPERT_MARKER = ".mlp.switch_mlp."

    def __init__(
        self,
        model_dir: str | Path,
        *,
        evaluate: bool = True,
        retain_layers: set[int] | None = None,
        resident_budget_bytes: int | None = None,
        backend: str = "native",
        clear_on_evict: bool = False,
        pin_policy: str = "all",
        warm_embeddings: bool = False,
        weight_page_budget_bytes: int | None = None,
        weight_page_policy: str = "auto",
        weight_page_rows: int = 1,
        expert_prefetch: str = "off",
        expert_prefetch_cap: int = 32,
        expert_compute_mode: str = "table",
        expert_slot_capacity: int | None = None,
        trace: bool = False,
    ) -> None:
        if pin_policy not in {"all", "phase"}:
            raise ValueError("pin_policy must be 'all' or 'phase'")
        if warm_embeddings and pin_policy != "phase":
            raise ValueError("warm_embeddings only applies to pin_policy='phase'")
        if expert_prefetch not in {"off", "previous", "previous_table"}:
            raise ValueError("expert_prefetch must be 'off', 'previous', or 'previous_table'")
        if expert_compute_mode not in {
            "table",
            "direct_qmm",
            "per_expert",
            "table_overlap_shared",
            "split_overlap_down",
            "split_overlap_shared_down",
            "slot_arena_direct_qmm",
        }:
            raise ValueError(
                "expert_compute_mode must be 'table', 'direct_qmm', 'per_expert', "
                "'table_overlap_shared', 'split_overlap_down', or "
                "'split_overlap_shared_down', or 'slot_arena_direct_qmm'"
            )
        if expert_prefetch_cap < 1:
            raise ValueError("expert_prefetch_cap must be positive")
        if expert_slot_capacity is not None and expert_slot_capacity < 1:
            raise ValueError("expert_slot_capacity must be positive")
        if weight_page_rows < 1:
            raise ValueError("weight_page_rows must be positive")
        if weight_page_rows != 1 and weight_page_budget_bytes is None:
            raise ValueError("weight_page_rows requires weight_page_budget_bytes")
        if expert_prefetch != "off" and weight_page_budget_bytes is None:
            raise ValueError("DeepSeek expert prefetch requires --weight-page-budget")
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
        self._trace = trace
        self._pass_trace = None
        self.pin_policy = pin_policy
        self.warm_embeddings = warm_embeddings
        # DeepSeek's output head is large enough that holding it resident
        # throughout the streamed layer pass can exceed a 2 GiB budget before
        # the first routed expert table is even loaded. Load it only for the
        # logits projection in low-memory phase mode.
        self.warm_output = False
        self._embed_prefix = "model.embed_tokens"
        self._expert_cache_bytes = 0
        weight_page_policy = resolve_weight_page_policy(
            "deepseek_v3",
            weight_page_policy,
            has_weight_page_budget=weight_page_budget_bytes is not None,
        )
        self.resident_budget_bytes = resident_budget_bytes
        self.weight_page_budget_bytes = weight_page_budget_bytes
        self.weight_page_policy = weight_page_policy
        self.weight_page_rows = weight_page_rows
        self.expert_prefetch = expert_prefetch
        self.expert_prefetch_cap = expert_prefetch_cap
        self.expert_compute_mode = expert_compute_mode
        self.expert_slot_capacity = expert_slot_capacity
        self._expert_history: dict[int, list[int]] = {}
        self._expert_slot_arenas: dict[int, DeepSeekExpertSlotArena] = {}
        self._expert_slot_arena_bytes = 0
        self._expert_slot_clock = 0
        self._expert_slot_stats = DeepSeekExpertSlotArenaStats()
        self._pending_expert_prefetch: dict[str, Any] | None = None
        self._pending_expert_prefetch_bytes = 0
        self._pending_layer_base_prefetch: dict[str, Any] | None = None
        self._pending_layer_base_prefetch_bytes = 0
        self._prefetch_stats = ExpertPrefetchStats()
        self.loader_executor = ThreadPoolExecutor(max_workers=1)
        if weight_page_budget_bytes is not None:
            self.session.loader.attach_weight_page_cache(
                weight_page_budget_bytes,
                eviction_policy=weight_page_policy,
                rows_per_page=weight_page_rows,
            )
        if self.session.config.get("model_type") != "deepseek_v3":
            raise ValueError(
                "DeepSeekV3StreamingForwardRunner currently supports model_type=deepseek_v3 only"
            )

        top_k = int(self.session.config.get("num_experts_per_tok", 1))
        if self.expert_slot_capacity is None:
            self.expert_slot_capacity = top_k
        if retain_layers is not None:
            self.base_retain_layers = set(retain_layers)
        elif resident_budget_bytes is not None:
            base_budget_bytes = reserve_weight_page_budget_for_base_retention(
                resident_budget_bytes,
                weight_page_budget_bytes,
            )
            self.base_retain_layers = select_qwen_base_layers_for_budget(
                self.session.loader.manifest,
                base_budget_bytes,
                top_k=top_k,
                expert_marker=self.EXPERT_MARKER,
            )
        else:
            self.base_retain_layers = set()

        from mlx_lm.utils import load_tokenizer

        self.tokenizer = load_tokenizer(self.session.model_dir)

    def close(self) -> None:
        error: BaseException | None = None
        try:
            self._drain_deepseek_expert_prefetch()
            self._drain_deepseek_layer_base_prefetch()
        except BaseException as exc:
            error = exc
        try:
            self.loader_executor.shutdown(wait=True)
        finally:
            self.session.close()
        if error is not None:
            raise error

    def _reset_stream_state(self) -> None:
        try:
            self._drain_deepseek_expert_prefetch()
            self._drain_deepseek_layer_base_prefetch()
        finally:
            self._expert_cache_bytes = 0
            self._expert_history.clear()
            self._clear_deepseek_slot_arenas()
            self._prefetch_stats = ExpertPrefetchStats()
            self._expert_slot_stats = DeepSeekExpertSlotArenaStats()
            self._set_deepseek_external_resident_bytes()

    def _drain_deepseek_expert_prefetch(self) -> None:
        pending = self._pending_expert_prefetch
        self._pending_expert_prefetch = None
        self._pending_expert_prefetch_bytes = 0
        if pending is not None:
            pending["future"].result()

    def _drain_deepseek_layer_base_prefetch(self) -> None:
        pending = getattr(self, "_pending_layer_base_prefetch", None)
        self._pending_layer_base_prefetch = None
        self._pending_layer_base_prefetch_bytes = 0
        if pending is not None:
            pending["future"].result()

    def _resident_sidecar_bytes(self) -> int:
        return (
            self._expert_cache_bytes
            + int(getattr(self, "_expert_slot_arena_bytes", 0) or 0)
            + self._pending_expert_prefetch_bytes
            + int(getattr(self, "_pending_layer_base_prefetch_bytes", 0) or 0)
            + self.session.loader.weight_page_resident_bytes
        )

    def _set_deepseek_external_resident_bytes(self, temporary_bytes: int = 0) -> None:
        total = self._resident_sidecar_bytes() + temporary_bytes
        if total:
            self.session.set_external_resident_bytes(total)
        else:
            self.session.clear_external_resident_bytes()

    def _deepseek_budget_base_bytes(self) -> int:
        current = int(getattr(self.session, "resident_bytes", 0) or 0)
        external = int(getattr(self.session, "external_resident_bytes", 0) or 0)
        return max(current - external, 0)

    def _deepseek_budget_allows(
        self,
        *,
        additional_sidecar_bytes: int = 0,
        temporary_bytes: int = 0,
    ) -> bool:
        budget = getattr(self, "resident_budget_bytes", None)
        if budget is None:
            return True
        proposed_external = (
            self._resident_sidecar_bytes()
            + additional_sidecar_bytes
            + temporary_bytes
        )
        return self._deepseek_budget_base_bytes() + proposed_external <= budget

    def _deepseek_total_resident_allows(self, additional_bytes: int = 0) -> bool:
        budget = getattr(self, "resident_budget_bytes", None)
        if budget is None:
            return True
        return self.session.resident_bytes + additional_bytes <= budget

    def _require_deepseek_budget(
        self,
        *,
        action: str,
        additional_sidecar_bytes: int = 0,
        temporary_bytes: int = 0,
    ) -> None:
        if self._deepseek_budget_allows(
            additional_sidecar_bytes=additional_sidecar_bytes,
            temporary_bytes=temporary_bytes,
        ):
            return
        proposed_external = (
            self._resident_sidecar_bytes()
            + additional_sidecar_bytes
            + temporary_bytes
        )
        proposed_peak = self._deepseek_budget_base_bytes() + proposed_external
        budget = getattr(self, "resident_budget_bytes", None)
        raise MemoryError(
            f"{action} would exceed resident budget "
            f"({proposed_peak} > {budget})"
        )

    def _embed_module(self) -> Any:
        return self.session.model.model.embed_tokens

    def _stream_forward_tokens(
        self,
        token_rows: list[list[int]],
        *,
        cache: list[Any],
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None = None,
        manage_embedding: bool = True,
    ) -> Any:
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        trace = self._trace
        core = self.session.model.model
        if self.pin_policy == "phase" and manage_embedding:
            load_embedding, local_token_rows = self._load_embedding_slices(token_rows)
            if trace and load_embedding is not None:
                events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_embedding)})
            inputs = mx.array(local_token_rows)
            try:
                x = core.embed_tokens(inputs)
                mx.eval(x)
            finally:
                self._clear_embedding_slices()
                if trace and load_embedding is not None:
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
            inputs = mx.array(token_rows)
            x = core.embed_tokens(inputs)
            mx.eval(x)

        mask = create_attention_mask(x, cache[0], return_array=True)

        for layer_index, (layer, layer_cache) in enumerate(zip(core.layers, cache)):
            load_event = self._load_deepseek_layer_base(layer_index)
            if trace:
                events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_event)})
            self._maybe_submit_deepseek_expert_prefetch(
                layer_index,
                layer,
                events=events if trace else None,
                pass_kind=pass_kind,
                token_step=token_step,
            )

            compute_started = time.perf_counter() if trace else 0.0
            try:
                x = self._deepseek_layer_forward_selective_experts(
                    layer_index,
                    layer,
                    x,
                    mask=mask,
                    cache=layer_cache,
                    events=events if trace else None,
                    pass_kind=pass_kind,
                    token_step=token_step,
                )
                self._maybe_submit_deepseek_layer_base_prefetch(
                    layer_index + 1,
                    len(core.layers),
                )
                mx.eval(x)
            finally:
                if (
                    hasattr(layer.mlp, "switch_mlp")
                    and self._deepseek_compute_binds_switch_mlp()
                ):
                    self._clear_deepseek_selected_experts(layer)
            if trace:
                events.append(
                    {
                        "kind": "compute",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "seconds": time.perf_counter() - compute_started,
                        "hidden_shape": list(x.shape),
                        "is_moe": hasattr(layer.mlp, "switch_mlp"),
                    }
                )

            if layer_index in self.base_retain_layers:
                evict_event = MlxStreamEvent(
                    action="retain-layer-base",
                    layer=layer_index,
                    seconds=0.0,
                    resident_bytes=self.session.resident_bytes,
                    requested=self.session.loader.manifest.layers[layer_index].tensor_names,
                )
                if trace:
                    self.session.events.append(evict_event)
            else:
                self._clear_deepseek_mla_projection_tables(layer)
                evict_event = self.session.evict_layer(layer_index)
            if trace:
                events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_event)})

        return x

    def _load_deepseek_layer_base(self, layer_index: int) -> MlxStreamEvent:
        pending = self._pending_layer_base_prefetch
        self._pending_layer_base_prefetch = None
        if pending is not None and pending["layer"] == layer_index:
            started = time.perf_counter()
            names = pending["names"]
            batch: MlxTensorBatch | None = None
            try:
                batch = pending["future"].result()
                missing = tuple(name for name in names if name not in self.session.resident)
                skipped = tuple(name for name in names if name in self.session.resident)
                if missing:
                    self.session._apply_tensor_batch(missing, batch)
                resident_bytes = self.session.resident_bytes
                self.session.peak_resident_bytes = max(
                    self.session.peak_resident_bytes,
                    resident_bytes,
                )
                event = MlxStreamEvent(
                    action="load-layer-base-prefetched",
                    layer=layer_index,
                    seconds=time.perf_counter() - started,
                    resident_bytes=resident_bytes,
                    requested=names,
                    loaded=missing,
                    skipped=skipped,
                    nbytes_loaded=batch.nbytes if missing else 0,
                )
                if self._trace:
                    self.session.events.append(event)
                return event
            finally:
                self._pending_layer_base_prefetch_bytes = 0
                self._set_deepseek_external_resident_bytes()
        if pending is not None:
            try:
                pending["future"].result()
            finally:
                self._pending_layer_base_prefetch_bytes = 0
                self._set_deepseek_external_resident_bytes()

        names = self._deepseek_layer_base_names(layer_index)
        return self.session._load_into_model(names, action="load-layer-base", layer=layer_index)

    def _deepseek_layer_base_names(self, layer_index: int) -> tuple[str, ...]:
        layer = self.session.loader.manifest.layers[layer_index]
        return tuple(name for name in layer.tensor_names if self.EXPERT_MARKER not in name)

    def _maybe_submit_deepseek_layer_base_prefetch(
        self,
        layer_index: int,
        total_layers: int,
    ) -> None:
        if self._pending_layer_base_prefetch is not None:
            return
        if layer_index >= total_layers:
            return
        if layer_index in self.base_retain_layers:
            return
        if self.expert_prefetch != "off":
            return
        if self.session.clear_on_evict:
            return
        names = tuple(
            name
            for name in self._deepseek_layer_base_names(layer_index)
            if name not in self.session.resident
        )
        if not names:
            return
        nbytes = sum(self.session.loader.manifest.tensors[name].nbytes for name in names)
        if not self._deepseek_budget_allows(additional_sidecar_bytes=nbytes):
            return
        future = self.loader_executor.submit(
            self.session.loader.load_tensors,
            names,
            evaluate=self.session.evaluate,
        )
        self._pending_layer_base_prefetch_bytes = nbytes
        self._set_deepseek_external_resident_bytes()
        self._pending_layer_base_prefetch = {
            "layer": layer_index,
            "names": names,
            "future": future,
        }

    def _deepseek_layer_forward_selective_experts(
        self,
        layer_index: int,
        layer: Any,
        x: Any,
        *,
        mask: Any,
        cache: Any,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        residual = layer.self_attn(layer.input_layernorm(x), mask, cache)
        hidden = x + residual
        mlp_input = layer.post_attention_layernorm(hidden)
        if not hasattr(layer.mlp, "switch_mlp"):
            return hidden + layer.mlp(mlp_input)
        return hidden + self._deepseek_moe_forward_selective_experts(
            layer_index,
            layer.mlp,
            mlp_input,
            events=events,
            pass_kind=pass_kind,
            token_step=token_step,
        )

    def _before_phase_output_load(self, events: list[dict[str, Any]]) -> None:
        if not getattr(self, "_expert_slot_arenas", None):
            return
        output_bytes = role_budget_bytes(self.session.loader.manifest, "output")
        # Loading lm_head also materializes logits/norm work. The raw output
        # tensor size alone underestimates the phase peak on tight tiers.
        output_required_bytes = output_bytes + 1536 * 1024 * 1024
        if self._deepseek_total_resident_allows(output_required_bytes):
            return
        before_count = len(self._expert_slot_arenas)
        before_bytes = self._expert_slot_arena_bytes
        self._evict_deepseek_slot_arenas_until(additional_total_bytes=output_required_bytes)
        if not self._deepseek_total_resident_allows(output_required_bytes):
            self._clear_deepseek_slot_arenas()
            self._set_deepseek_external_resident_bytes()
        if self._trace:
            events.append(
                {
                    "kind": "evict",
                    "action": "evict-slot-arenas-for-output",
                    "evicted_count": before_count - len(self._expert_slot_arenas),
                    "evicted_bytes": before_bytes - self._expert_slot_arena_bytes,
                    "resident_bytes": self.session.resident_bytes,
                }
            )

    def _deepseek_moe_forward_selective_experts(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx

        expert_indices, scores = mlp.gate(x)
        mx.eval(expert_indices)
        selected_experts = selected_expert_ids(expert_indices)
        if (
            self.expert_compute_mode == "slot_arena_direct_qmm"
            and self.expert_prefetch == "off"
        ):
            self._expert_history[layer_index] = selected_experts
            return self._deepseek_moe_forward_slot_arena_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        if (
            self.expert_compute_mode == "direct_qmm"
            and self.expert_prefetch == "off"
        ):
            self._expert_history[layer_index] = selected_experts
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        if (
            self.expert_compute_mode in {"split_overlap_down", "split_overlap_shared_down"}
            and self.expert_prefetch == "off"
        ):
            self._expert_history[layer_index] = selected_experts
            return self._deepseek_moe_forward_split_overlap_down(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
                overlap_shared=self.expert_compute_mode == "split_overlap_shared_down",
            )
        if (
            self.expert_compute_mode == "table_overlap_shared"
            and self.expert_prefetch == "off"
            and getattr(mlp, "shared_experts", None) is not None
        ):
            self._expert_history[layer_index] = selected_experts
            return self._deepseek_moe_forward_table_overlap_shared(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        if (
            self.expert_compute_mode == "per_expert"
            and self.expert_prefetch == "off"
            and self._can_stream_deepseek_experts_per_expert(expert_indices)
        ):
            topk_experts = [int(value) for value in np.asarray(expert_indices).reshape(-1)]
            self._expert_history[layer_index] = selected_experts
            return self._deepseek_moe_forward_per_expert(
                layer_index,
                mlp,
                x,
                topk_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        table_order = self._consume_deepseek_expert_prefetch(
            layer_index,
            mlp,
            selected_experts,
            events=events,
            pass_kind=pass_kind,
            token_step=token_step,
        )
        self._expert_history[layer_index] = selected_experts

        if table_order is None:
            load_event = self._load_deepseek_selected_experts(layer_index, mlp, selected_experts)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "load-selected-experts",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        **compact_event(load_event),
                    }
                )
            table_order = selected_experts

        local_indices = remap_expert_indices(expert_indices, table_order)
        expert_y = mlp.switch_mlp(x, local_indices)
        expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
        shared_experts = getattr(mlp, "shared_experts", None)
        if shared_experts is not None:
            expert_y = expert_y + shared_experts(x)
        return expert_y

    def _deepseek_moe_forward_table_overlap_shared(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx

        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        use_weight_page_cache = True
        if getattr(self, "resident_budget_bytes", None) is not None:
            estimated_table_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
            estimated_page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=True,
            )
            estimated_transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=False,
            )
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=estimated_table_bytes + estimated_page_growth_bytes,
                temporary_bytes=estimated_transient_page_bytes,
            ):
                if estimated_transient_page_bytes and self._deepseek_budget_allows(
                    additional_sidecar_bytes=estimated_table_bytes,
                ):
                    use_weight_page_cache = False
                    estimated_transient_page_bytes = 0
                else:
                    self._require_deepseek_budget(
                        action="load-selected-experts-overlap-shared",
                        additional_sidecar_bytes=estimated_table_bytes,
                        temporary_bytes=estimated_transient_page_bytes,
                    )
            self._set_deepseek_external_resident_bytes(
                estimated_table_bytes + estimated_transient_page_bytes
            )

        future = self.loader_executor.submit(
            self._load_deepseek_slice_batch,
            layer_index,
            selected_experts,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=self.session.evaluate,
        )
        shared_y = mlp.shared_experts(x)
        mx.async_eval(shared_y)
        try:
            batch = future.result()
            batch_transient_page_bytes = int(
                getattr(batch, "transient_page_bytes", 0) or 0
            )
            if batch_transient_page_bytes:
                self._set_deepseek_external_resident_bytes(
                    batch.nbytes + batch_transient_page_bytes
                )
            self._assign_deepseek_expert_tables(mlp, layer_index, batch.arrays)
            self._set_deepseek_external_resident_bytes(batch.nbytes)
            local_indices = remap_expert_indices(expert_indices, selected_experts)
            expert_y = mlp.switch_mlp(x, local_indices)
            expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
            event = MlxStreamEvent(
                action=(
                    "load-selected-experts-overlap-shared"
                    if use_weight_page_cache
                    else "load-selected-experts-overlap-shared-direct-budget"
                ),
                layer=layer_index,
                seconds=time.perf_counter() - started,
                resident_bytes=self.session.resident_bytes,
                requested=names,
                loaded=names,
                nbytes_loaded=batch.nbytes,
                transient_page_bytes=batch_transient_page_bytes,
            )
            if self._trace:
                self.session.events.append(event)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": event.action,
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        **compact_event(event),
                    }
                )
            return expert_y + shared_y
        finally:
            self._set_deepseek_external_resident_bytes()

    def _deepseek_moe_forward_direct_qmm(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
        overlap_shared: bool = False,
    ) -> Any:
        import mlx.core as mx

        started = time.perf_counter()
        use_weight_page_cache = True
        estimated_table_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
        estimated_transient_page_bytes = 0
        if getattr(self, "resident_budget_bytes", None) is not None:
            estimated_page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=True,
            )
            estimated_transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=False,
            )
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=estimated_table_bytes + estimated_page_growth_bytes,
                temporary_bytes=estimated_transient_page_bytes,
            ):
                if estimated_transient_page_bytes and self._deepseek_budget_allows(
                    additional_sidecar_bytes=estimated_table_bytes,
                    temporary_bytes=0,
                ):
                    use_weight_page_cache = False
                    estimated_transient_page_bytes = 0
                else:
                    self._require_deepseek_budget(
                        action="load-selected-experts-direct-qmm",
                        additional_sidecar_bytes=estimated_table_bytes,
                        temporary_bytes=estimated_transient_page_bytes,
                    )

        batch = self._load_deepseek_slice_batch(
            layer_index,
            selected_experts,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=False,
        )
        batch_transient_page_bytes = int(
            getattr(batch, "transient_page_bytes", 0) or 0
        )
        self._set_deepseek_external_resident_bytes(
            batch.nbytes + batch_transient_page_bytes
        )
        try:
            local_indices = remap_expert_indices(expert_indices, selected_experts)
            x_expanded = mx.expand_dims(x, (-2, -3))
            up = self._deepseek_projection_qmm(
                mlp,
                layer_index,
                "up_proj",
                x_expanded,
                local_indices,
                batch.arrays,
            )
            gate = self._deepseek_projection_qmm(
                mlp,
                layer_index,
                "gate_proj",
                x_expanded,
                local_indices,
                batch.arrays,
            )
            activated = mlp.switch_mlp.activation(up, gate)
            expert_y = self._deepseek_projection_qmm(
                mlp,
                layer_index,
                "down_proj",
                activated,
                local_indices,
                batch.arrays,
            )
            expert_y = expert_y.squeeze(-2)
            expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
            shared_experts = getattr(mlp, "shared_experts", None)
            if shared_experts is not None:
                expert_y = expert_y + shared_experts(x)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": (
                            "load-selected-experts-direct-qmm"
                            if use_weight_page_cache
                            else "load-selected-experts-direct-qmm-direct-budget"
                        ),
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        "seconds": time.perf_counter() - started,
                        "resident_bytes": self.session.resident_bytes,
                        "nbytes_loaded": batch.nbytes,
                        "transient_page_bytes": batch_transient_page_bytes,
                    }
                )
            return expert_y
        finally:
            self._set_deepseek_external_resident_bytes()

    def _deepseek_moe_forward_slot_arena_direct_qmm(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        started = time.perf_counter()
        self._expert_slot_stats.attempts += 1
        self._expert_slot_stats.selected_rows += len(selected_experts)
        arena_result = self._ensure_deepseek_slot_arena(
            layer_index,
            selected_experts,
        )
        if arena_result is None:
            self._expert_slot_stats.fallback_direct += 1
            direct_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=direct_bytes
            ) or not self._deepseek_total_resident_allows(direct_bytes):
                self._evict_deepseek_slot_arenas_until(
                    additional_sidecar_bytes=direct_bytes,
                    additional_total_bytes=direct_bytes,
                )
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=direct_bytes
            ) or not self._deepseek_total_resident_allows(direct_bytes):
                self._clear_deepseek_slot_arenas()
                self._set_deepseek_external_resident_bytes()
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "slot-arena-fallback-direct-qmm",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                    }
                )
            return self._deepseek_moe_forward_direct_qmm(
                layer_index,
                mlp,
                x,
                expert_indices,
                selected_experts,
                scores,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )

        arena, arena_info = arena_result
        local_indices = remap_expert_indices_to_slots(expert_indices, arena.expert_to_slot)
        expert_y = self._deepseek_routed_qmm_from_arrays(
            mlp,
            layer_index,
            x,
            local_indices,
            scores,
            arena.arrays,
        )
        if events is not None:
            events.append(
                {
                    "kind": "load",
                    "action": arena_info["action"],
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "expert_count": len(selected_experts),
                    "experts": selected_experts,
                    "seconds": time.perf_counter() - started,
                    "resident_bytes": self.session.resident_bytes,
                    **arena_info,
                }
            )
        return expert_y

    def _deepseek_routed_qmm_from_arrays(
        self,
        mlp: Any,
        layer_index: int,
        x: Any,
        local_indices: Any,
        scores: Any,
        arrays: dict[str, Any],
    ) -> Any:
        import mlx.core as mx

        x_expanded = mx.expand_dims(x, (-2, -3))
        up = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "up_proj",
            x_expanded,
            local_indices,
            arrays,
        )
        gate = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "gate_proj",
            x_expanded,
            local_indices,
            arrays,
        )
        activated = mlp.switch_mlp.activation(up, gate)
        expert_y = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "down_proj",
            activated,
            local_indices,
            arrays,
        )
        expert_y = expert_y.squeeze(-2)
        expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
        shared_experts = getattr(mlp, "shared_experts", None)
        if shared_experts is not None:
            expert_y = expert_y + shared_experts(x)
        return expert_y

    def _deepseek_moe_forward_split_overlap_down(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        expert_indices: Any,
        selected_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
        overlap_shared: bool = False,
    ) -> Any:
        import mlx.core as mx

        started = time.perf_counter()
        use_weight_page_cache = True
        estimated_table_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
        estimated_page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            selected_experts,
            cap_to_headroom=True,
        )
        estimated_transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            selected_experts,
            cap_to_headroom=False,
        )
        if getattr(self, "resident_budget_bytes", None) is not None:
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=estimated_table_bytes + estimated_page_growth_bytes,
                temporary_bytes=estimated_transient_page_bytes,
            ):
                if estimated_transient_page_bytes and self._deepseek_budget_allows(
                    additional_sidecar_bytes=estimated_table_bytes,
                ):
                    use_weight_page_cache = False
                    estimated_transient_page_bytes = 0
                else:
                    self._require_deepseek_budget(
                        action="load-selected-experts-split-overlap-down",
                        additional_sidecar_bytes=estimated_table_bytes,
                        temporary_bytes=estimated_transient_page_bytes,
                    )
        gate_up_names = self._expert_projection_slice_names(layer_index, ("gate_proj", "up_proj"))
        down_names = self._expert_projection_slice_names(layer_index, ("down_proj",))
        gate_up_bytes = self._deepseek_named_slice_nbytes(gate_up_names, selected_experts)
        down_bytes = self._deepseek_named_slice_nbytes(down_names, selected_experts)
        gate_up_page_transient = (
            self.session.loader.estimate_first_dim_slice_page_miss_bytes(
                gate_up_names,
                selected_experts,
                cap_to_headroom=False,
            )
            if use_weight_page_cache
            else 0
        )
        self._set_deepseek_external_resident_bytes(
            gate_up_bytes + gate_up_page_transient
        )
        shared_experts = getattr(mlp, "shared_experts", None)
        shared_y = None
        gate_up_future = None
        if overlap_shared and shared_experts is not None:
            gate_up_future = self.loader_executor.submit(
                self.session.loader.load_first_dim_slices,
                gate_up_names,
                selected_experts,
                evaluate=False,
                use_weight_page_cache=use_weight_page_cache,
            )
            shared_y = shared_experts(x)
            mx.async_eval(shared_y)
            gate_up = gate_up_future.result()
        else:
            gate_up = self.session.loader.load_first_dim_slices(
                gate_up_names,
                selected_experts,
                evaluate=False,
                use_weight_page_cache=use_weight_page_cache,
            )
        local_indices = remap_expert_indices(expert_indices, selected_experts)
        x_expanded = mx.expand_dims(x, (-2, -3))
        up = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "up_proj",
            x_expanded,
            local_indices,
            gate_up.arrays,
        )
        gate = self._deepseek_projection_qmm(
            mlp,
            layer_index,
            "gate_proj",
            x_expanded,
            local_indices,
            gate_up.arrays,
        )
        activated = mlp.switch_mlp.activation(up, gate)
        mx.async_eval(activated)
        self._set_deepseek_external_resident_bytes(
            gate_up.nbytes + down_bytes + int(gate_up.transient_page_bytes or 0)
        )
        future = self.loader_executor.submit(
            self.session.loader.load_first_dim_slices,
            down_names,
            selected_experts,
            evaluate=False,
            use_weight_page_cache=use_weight_page_cache,
        )
        try:
            down = future.result()
            down_transient = int(getattr(down, "transient_page_bytes", 0) or 0)
            self._set_deepseek_external_resident_bytes(
                gate_up.nbytes + down.nbytes + down_transient
            )
            expert_y = self._deepseek_projection_qmm(
                mlp,
                layer_index,
                "down_proj",
                activated,
                local_indices,
                down.arrays,
            )
            expert_y = expert_y.squeeze(-2)
            expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
            if shared_y is not None:
                expert_y = expert_y + shared_y
            elif shared_experts is not None:
                expert_y = expert_y + shared_experts(x)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": (
                            "load-selected-experts-split-overlap-shared-down"
                            if overlap_shared and shared_experts is not None
                            else "load-selected-experts-split-overlap-down"
                        ),
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(selected_experts),
                        "experts": selected_experts,
                        "seconds": time.perf_counter() - started,
                        "resident_bytes": self.session.resident_bytes,
                        "nbytes_loaded": gate_up.nbytes + down.nbytes,
                        "transient_page_bytes": int(gate_up.transient_page_bytes or 0) + down_transient,
                    }
                )
            return expert_y
        finally:
            self._set_deepseek_external_resident_bytes()

    def _can_stream_deepseek_experts_per_expert(self, expert_indices: Any) -> bool:
        shape = tuple(int(dim) for dim in getattr(expert_indices, "shape", ()))
        return len(shape) == 3 and shape[0] == 1 and shape[1] == 1

    def _deepseek_moe_forward_per_expert(
        self,
        layer_index: int,
        mlp: Any,
        x: Any,
        topk_experts: list[int],
        scores: Any,
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> Any:
        import mlx.core as mx

        started = time.perf_counter()
        x_expanded = mx.expand_dims(x, (-2, -3))
        local_index = mx.array([[[0]]], dtype=mx.int32)
        outputs: list[Any] = []
        nbytes_loaded = 0
        transient_page_bytes = 0

        try:
            for expert_id in topk_experts:
                row_bytes = self._deepseek_slice_nbytes(layer_index, [expert_id])
                page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
                    layer_index,
                    [expert_id],
                    cap_to_headroom=True,
                )
                page_transient_bytes = self._deepseek_slice_page_miss_nbytes(
                    layer_index,
                    [expert_id],
                    cap_to_headroom=False,
                )
                use_weight_page_cache = True
                if not self._deepseek_budget_allows(
                    additional_sidecar_bytes=row_bytes + page_growth_bytes,
                    temporary_bytes=page_transient_bytes,
                ):
                    if page_transient_bytes and self._deepseek_budget_allows(
                        additional_sidecar_bytes=row_bytes,
                    ):
                        use_weight_page_cache = False
                        page_transient_bytes = 0
                    else:
                        self._require_deepseek_budget(
                            action="stream-selected-expert",
                            additional_sidecar_bytes=row_bytes,
                            temporary_bytes=page_transient_bytes,
                        )

                self._set_deepseek_external_resident_bytes(
                    row_bytes + page_transient_bytes
                )
                batch = self._load_deepseek_slice_batch(
                    layer_index,
                    [expert_id],
                    use_weight_page_cache=use_weight_page_cache,
                )
                batch_transient_page_bytes = int(
                    getattr(batch, "transient_page_bytes", 0) or 0
                )
                nbytes_loaded += batch.nbytes
                transient_page_bytes += batch_transient_page_bytes
                self._set_deepseek_external_resident_bytes(
                    batch.nbytes + batch_transient_page_bytes
                )
                out = self._deepseek_single_expert_glu(
                    mlp,
                    layer_index,
                    expert_id,
                    x_expanded,
                    local_index,
                    batch.arrays,
                )
                mx.eval(out)
                outputs.append(out)
                self._set_deepseek_external_resident_bytes()

            expert_y = mx.concatenate(outputs, axis=-2).squeeze(-3)
            expert_y = (expert_y * scores[..., None]).sum(axis=-2).astype(expert_y.dtype)
            shared_experts = getattr(mlp, "shared_experts", None)
            if shared_experts is not None:
                expert_y = expert_y + shared_experts(x)
            return expert_y
        finally:
            self._set_deepseek_external_resident_bytes()
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "stream-selected-experts-per-expert",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "expert_count": len(topk_experts),
                        "experts": topk_experts,
                        "seconds": time.perf_counter() - started,
                        "resident_bytes": self.session.resident_bytes,
                        "nbytes_loaded": nbytes_loaded,
                        "transient_page_bytes": transient_page_bytes,
                    }
                )

    def _deepseek_single_expert_glu(
        self,
        mlp: Any,
        layer_index: int,
        expert_id: int,
        x_expanded: Any,
        local_index: Any,
        arrays: dict[str, Any],
    ) -> Any:
        import mlx.core as mx

        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"

        def project(name: str, inputs: Any) -> Any:
            module = getattr(mlp.switch_mlp, name)
            return mx.gather_qmm(
                inputs,
                arrays[f"{prefix}.{name}.weight"],
                arrays[f"{prefix}.{name}.scales"],
                arrays[f"{prefix}.{name}.biases"],
                rhs_indices=local_index,
                transpose=True,
                group_size=module.group_size,
                bits=module.bits,
                mode=module.mode,
                sorted_indices=False,
            )

        up = project("up_proj", x_expanded)
        gate = project("gate_proj", x_expanded)
        return project("down_proj", mlp.switch_mlp.activation(up, gate))

    def _ensure_deepseek_slot_arena(
        self,
        layer_index: int,
        selected_experts: list[int],
    ) -> tuple[DeepSeekExpertSlotArena, dict[str, Any]] | None:
        capacity = int(self.expert_slot_capacity or len(selected_experts))
        if not selected_experts or len(selected_experts) > capacity:
            return None
        if self.expert_prefetch != "off":
            return None
        if getattr(self, "resident_budget_bytes", None) is None:
            return None
        # A retained slot arena adds persistent sidecar memory on top of the
        # already-tight DeepSeek phase path. Below ~8 GiB the direct lazy-QMM
        # path has a lower measured peak, so keep the hard low-memory tier
        # honest and use the arena only when there is real headroom.
        if self.resident_budget_bytes < 8 * 1024 * 1024 * 1024:
            return None

        arena = self._expert_slot_arenas.get(layer_index)
        if arena is not None and arena.capacity != capacity:
            self._evict_deepseek_slot_arena(layer_index)
            arena = None

        if arena is None:
            return self._create_deepseek_slot_arena(layer_index, selected_experts, capacity)

        self._expert_slot_clock += 1
        arena.last_used = self._expert_slot_clock
        hits = [expert for expert in selected_experts if expert in arena.expert_to_slot]
        missing = [expert for expert in selected_experts if expert not in arena.expert_to_slot]
        self._expert_slot_stats.hit_rows += len(hits)
        self._expert_slot_stats.missing_rows += len(missing)
        if not missing:
            self._expert_slot_stats.arena_hits += 1
            return arena, {
                "action": "slot-arena-hit",
                "hit_count": len(hits),
                "missing_count": 0,
                "nbytes_loaded": 0,
                "arena_resident_bytes": self._expert_slot_arena_bytes,
                "arena_count": len(self._expert_slot_arenas),
            }
        self._expert_slot_stats.arena_misses += 1
        updated = self._update_deepseek_slot_arena(layer_index, arena, selected_experts, missing)
        if updated is None:
            return None
        return updated

    def _create_deepseek_slot_arena(
        self,
        layer_index: int,
        selected_experts: list[int],
        capacity: int,
    ) -> tuple[DeepSeekExpertSlotArena, dict[str, Any]] | None:
        import mlx.core as mx

        names = self._expert_slice_names(layer_index)
        arena_bytes = self._deepseek_slot_arena_nbytes(layer_index, capacity)
        load_bytes = self._deepseek_named_slice_nbytes(names, selected_experts)
        page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            selected_experts,
            cap_to_headroom=True,
        )
        transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            selected_experts,
            cap_to_headroom=False,
        )
        use_weight_page_cache = True
        temporary_bytes = transient_page_bytes + (load_bytes if capacity > len(selected_experts) else 0)
        total_required_bytes = arena_bytes + temporary_bytes + 512 * 1024 * 1024
        if not self._deepseek_budget_allows(
            additional_sidecar_bytes=arena_bytes + page_growth_bytes,
            temporary_bytes=temporary_bytes,
        ) or not self._deepseek_total_resident_allows(total_required_bytes):
            self._evict_deepseek_slot_arenas_until(
                additional_sidecar_bytes=arena_bytes + page_growth_bytes,
                temporary_bytes=temporary_bytes,
                additional_total_bytes=total_required_bytes,
                protected_layer=layer_index,
                current_layer=layer_index,
            )
        if not self._deepseek_budget_allows(
            additional_sidecar_bytes=arena_bytes + page_growth_bytes,
            temporary_bytes=temporary_bytes,
        ) or not self._deepseek_total_resident_allows(total_required_bytes):
            direct_total_required = (
                arena_bytes
                + (load_bytes if capacity > len(selected_experts) else 0)
                + 512 * 1024 * 1024
            )
            if transient_page_bytes and self._deepseek_budget_allows(
                additional_sidecar_bytes=arena_bytes,
                temporary_bytes=(load_bytes if capacity > len(selected_experts) else 0),
            ) and self._deepseek_total_resident_allows(direct_total_required):
                use_weight_page_cache = False
                transient_page_bytes = 0
                temporary_bytes = load_bytes if capacity > len(selected_experts) else 0
            else:
                return None

        started = time.perf_counter()
        batch = self._load_deepseek_slice_batch(
            layer_index,
            selected_experts,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=False,
        )
        load_seconds = time.perf_counter() - started
        batch_transient_page_bytes = int(getattr(batch, "transient_page_bytes", 0) or 0)
        arrays = batch.arrays
        update_seconds = 0.0
        if capacity > len(selected_experts):
            update_started = time.perf_counter()
            padded: dict[str, Any] = {}
            self._set_deepseek_external_resident_bytes(
                arena_bytes + batch.nbytes + batch_transient_page_bytes
            )
            for name, array in batch.arrays.items():
                slot_array = mx.zeros((capacity, *array.shape[1:]), dtype=array.dtype)
                for row_index in range(len(selected_experts)):
                    slot_array = mx.slice_update(
                        slot_array,
                        array[row_index : row_index + 1],
                        start_indices=mx.array(row_index),
                        axes=(0,),
                    )
                padded[name] = slot_array
            mx.eval(list(padded.values()))
            update_seconds = time.perf_counter() - update_started
            arrays = padded
        else:
            self._set_deepseek_external_resident_bytes(arena_bytes + batch_transient_page_bytes)

        slot_to_expert: list[int | None] = [None] * capacity
        expert_to_slot: dict[int, int] = {}
        for slot, expert in enumerate(selected_experts):
            slot_to_expert[slot] = expert
            expert_to_slot[expert] = slot
        self._expert_slot_clock += 1
        arena = DeepSeekExpertSlotArena(
            layer_index=layer_index,
            capacity=capacity,
            arrays=arrays,
            slot_to_expert=slot_to_expert,
            expert_to_slot=expert_to_slot,
            nbytes=arena_bytes,
            last_used=self._expert_slot_clock,
        )
        self._expert_slot_arenas[layer_index] = arena
        self._expert_slot_arena_bytes += arena_bytes
        self._expert_slot_stats.arena_creates += 1
        self._expert_slot_stats.arena_misses += 1
        self._expert_slot_stats.missing_rows += len(selected_experts)
        self._expert_slot_stats.loaded_bytes += batch.nbytes
        self._expert_slot_stats.load_seconds += load_seconds
        self._expert_slot_stats.update_seconds += update_seconds
        self._set_deepseek_external_resident_bytes()
        return arena, {
            "action": (
                "slot-arena-create"
                if use_weight_page_cache
                else "slot-arena-create-direct-budget"
            ),
            "hit_count": 0,
            "missing_count": len(selected_experts),
            "nbytes_loaded": batch.nbytes,
            "transient_page_bytes": batch_transient_page_bytes,
            "load_seconds": load_seconds,
            "update_seconds": update_seconds,
            "arena_resident_bytes": self._expert_slot_arena_bytes,
            "arena_count": len(self._expert_slot_arenas),
        }

    def _update_deepseek_slot_arena(
        self,
        layer_index: int,
        arena: DeepSeekExpertSlotArena,
        selected_experts: list[int],
        missing: list[int],
    ) -> tuple[DeepSeekExpertSlotArena, dict[str, Any]] | None:
        import mlx.core as mx

        selected_set = set(selected_experts)
        free_slots = [
            slot for slot, expert in enumerate(arena.slot_to_expert) if expert is None
        ]
        evictable_slots = [
            slot
            for slot, expert in enumerate(arena.slot_to_expert)
            if expert is not None and expert not in selected_set
        ]
        target_slots = (free_slots + evictable_slots)[: len(missing)]
        if len(target_slots) < len(missing):
            return None
        slot_assignments = list(zip(sorted(target_slots), missing, strict=True))
        ordered_slots = [slot for slot, _expert in slot_assignments]
        ordered_missing = [expert for _slot, expert in slot_assignments]
        update_runs: list[tuple[int, int, int]] = []
        if ordered_slots:
            run_start_row = 0
            run_start_slot = ordered_slots[0]
            previous_slot = run_start_slot
            for row_index, slot in enumerate(ordered_slots[1:], start=1):
                if slot != previous_slot + 1:
                    update_runs.append(
                        (run_start_slot, run_start_row, row_index - run_start_row)
                    )
                    run_start_row = row_index
                    run_start_slot = slot
                previous_slot = slot
            update_runs.append(
                (run_start_slot, run_start_row, len(ordered_slots) - run_start_row)
            )

        names = self._expert_slice_names(layer_index)
        load_bytes = self._deepseek_named_slice_nbytes(names, missing)
        page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            missing,
            cap_to_headroom=True,
        )
        transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
            layer_index,
            missing,
            cap_to_headroom=False,
        )
        use_weight_page_cache = True
        temporary_bytes = arena.nbytes + load_bytes + transient_page_bytes
        total_required_bytes = temporary_bytes + 512 * 1024 * 1024
        if not self._deepseek_budget_allows(
            additional_sidecar_bytes=page_growth_bytes,
            temporary_bytes=temporary_bytes,
        ) or not self._deepseek_total_resident_allows(total_required_bytes):
            self._evict_deepseek_slot_arenas_until(
                additional_sidecar_bytes=page_growth_bytes,
                temporary_bytes=temporary_bytes,
                additional_total_bytes=total_required_bytes,
                protected_layer=layer_index,
                current_layer=layer_index,
            )
        if not self._deepseek_budget_allows(
            additional_sidecar_bytes=page_growth_bytes,
            temporary_bytes=temporary_bytes,
        ) or not self._deepseek_total_resident_allows(total_required_bytes):
            if transient_page_bytes and self._deepseek_budget_allows(
                additional_sidecar_bytes=0,
                temporary_bytes=arena.nbytes + load_bytes,
            ) and self._deepseek_total_resident_allows(
                arena.nbytes + load_bytes + 512 * 1024 * 1024
            ):
                use_weight_page_cache = False
                transient_page_bytes = 0
                temporary_bytes = arena.nbytes + load_bytes
            else:
                return None

        load_started = time.perf_counter()
        batch = self._load_deepseek_slice_batch(
            layer_index,
            ordered_missing,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=False,
        )
        load_seconds = time.perf_counter() - load_started
        batch_transient_page_bytes = int(getattr(batch, "transient_page_bytes", 0) or 0)
        update_started = time.perf_counter()
        self._set_deepseek_external_resident_bytes(
            arena.nbytes + batch.nbytes + batch_transient_page_bytes
        )
        updated_arrays: dict[str, Any] = {}
        for name, array in arena.arrays.items():
            updated = array
            loaded = batch.arrays[name]
            for slot, row_index, count in update_runs:
                updated = mx.slice_update(
                    updated,
                    loaded[row_index : row_index + count],
                    start_indices=mx.array(slot),
                    axes=(0,),
                )
            updated_arrays[name] = updated
        mx.eval(list(updated_arrays.values()))
        update_seconds = time.perf_counter() - update_started

        for slot, expert in slot_assignments:
            old_expert = arena.slot_to_expert[slot]
            if old_expert is not None:
                arena.expert_to_slot.pop(old_expert, None)
            arena.slot_to_expert[slot] = expert
            arena.expert_to_slot[expert] = slot
        arena.arrays = updated_arrays
        self._expert_slot_clock += 1
        arena.last_used = self._expert_slot_clock
        self._expert_slot_stats.arena_updates += 1
        self._expert_slot_stats.loaded_bytes += batch.nbytes
        self._expert_slot_stats.load_seconds += load_seconds
        self._expert_slot_stats.update_seconds += update_seconds
        self._expert_slot_stats.slot_update_ops += len(update_runs) * len(updated_arrays)
        self._set_deepseek_external_resident_bytes()
        return arena, {
            "action": (
                "slot-arena-update"
                if use_weight_page_cache
                else "slot-arena-update-direct-budget"
            ),
            "hit_count": len(selected_experts) - len(missing),
            "missing_count": len(missing),
            "nbytes_loaded": batch.nbytes,
            "transient_page_bytes": batch_transient_page_bytes,
            "load_seconds": load_seconds,
            "update_seconds": update_seconds,
            "slot_update_ops": len(update_runs) * len(updated_arrays),
            "arena_resident_bytes": self._expert_slot_arena_bytes,
            "arena_count": len(self._expert_slot_arenas),
        }

    def _deepseek_slot_arena_nbytes(self, layer_index: int, capacity: int) -> int:
        if capacity <= 0:
            return 0
        total = 0
        for name in self._expert_slice_names(layer_index):
            record = self.session.loader.manifest.tensors[name]
            total += record.nbytes * capacity // record.shape[0]
        return total

    def _evict_deepseek_slot_arena(self, layer_index: int) -> int:
        arena = self._expert_slot_arenas.pop(layer_index, None)
        if arena is None:
            return 0
        self._expert_slot_arena_bytes -= arena.nbytes
        self._expert_slot_stats.evictions += 1
        self._set_deepseek_external_resident_bytes()
        return arena.nbytes

    def _evict_deepseek_slot_arenas_until(
        self,
        *,
        additional_sidecar_bytes: int = 0,
        temporary_bytes: int = 0,
        additional_total_bytes: int = 0,
        protected_layer: int | None = None,
        current_layer: int | None = None,
    ) -> None:
        while (
            self._expert_slot_arenas
            and (
                not self._deepseek_budget_allows(
                    additional_sidecar_bytes=additional_sidecar_bytes,
                    temporary_bytes=temporary_bytes,
                )
                or not self._deepseek_total_resident_allows(
                    additional_total_bytes
                )
            )
        ):
            candidates = [
                (arena.last_used, layer)
                for layer, arena in self._expert_slot_arenas.items()
                if layer != protected_layer
            ]
            if current_layer is not None:
                past_candidates = [
                    (last_used, layer)
                    for last_used, layer in candidates
                    if layer < current_layer
                ]
                if past_candidates:
                    candidates = past_candidates
                else:
                    return
            if not candidates:
                return
            _, layer_to_evict = min(candidates)
            self._evict_deepseek_slot_arena(layer_to_evict)

    def _clear_deepseek_slot_arenas(self) -> None:
        arenas = getattr(self, "_expert_slot_arenas", None)
        if not arenas:
            self._expert_slot_arena_bytes = 0
            return
        arenas.clear()
        self._expert_slot_arena_bytes = 0

    def _deepseek_projection_qmm(
        self,
        mlp: Any,
        layer_index: int,
        projection: str,
        inputs: Any,
        local_indices: Any,
        arrays: dict[str, Any],
    ) -> Any:
        import mlx.core as mx

        prefix = f"model.layers.{layer_index}.mlp.switch_mlp.{projection}"
        module = getattr(mlp.switch_mlp, projection)
        return mx.gather_qmm(
            inputs,
            arrays[f"{prefix}.weight"],
            arrays[f"{prefix}.scales"],
            arrays[f"{prefix}.biases"],
            rhs_indices=local_indices,
            transpose=True,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
            sorted_indices=False,
        )

    def _expert_slice_names(self, layer_index: int) -> tuple[str, ...]:
        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"
        return tuple(
            f"{prefix}.{projection}.{field}"
            for projection in ("gate_proj", "up_proj", "down_proj")
            for field in ("weight", "scales", "biases")
        )

    def _expert_projection_slice_names(
        self,
        layer_index: int,
        projections: tuple[str, ...],
    ) -> tuple[str, ...]:
        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"
        return tuple(
            f"{prefix}.{projection}.{field}"
            for projection in projections
            for field in ("weight", "scales", "biases")
        )

    def _deepseek_named_slice_nbytes(
        self,
        names: tuple[str, ...],
        expert_ids: list[int],
    ) -> int:
        if not expert_ids:
            return 0
        total = 0
        for name in names:
            record = self.session.loader.manifest.tensors[name]
            total += record.nbytes * len(expert_ids) // record.shape[0]
        return total

    def _load_deepseek_selected_experts(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
    ) -> MlxStreamEvent:
        started = time.perf_counter()
        names = self._expert_slice_names(layer_index)
        use_weight_page_cache = True
        if getattr(self, "resident_budget_bytes", None) is not None:
            estimated_table_bytes = self._deepseek_slice_nbytes(layer_index, selected_experts)
            estimated_page_growth_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=True,
            )
            estimated_transient_page_bytes = self._deepseek_slice_page_miss_nbytes(
                layer_index,
                selected_experts,
                cap_to_headroom=False,
            )
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=estimated_table_bytes + estimated_page_growth_bytes,
                temporary_bytes=estimated_transient_page_bytes,
            ):
                if estimated_transient_page_bytes and self._deepseek_budget_allows(
                    additional_sidecar_bytes=estimated_table_bytes,
                    temporary_bytes=0,
                ):
                    use_weight_page_cache = False
                else:
                    self._require_deepseek_budget(
                        action="load-selected-experts",
                        additional_sidecar_bytes=estimated_table_bytes,
                        temporary_bytes=estimated_transient_page_bytes,
                    )
        batch = self._load_deepseek_slice_batch(
            layer_index,
            selected_experts,
            use_weight_page_cache=use_weight_page_cache,
            evaluate=False,
        )
        batch_transient_page_bytes = int(
            getattr(batch, "transient_page_bytes", 0) or 0
        )
        if batch_transient_page_bytes:
            self._set_deepseek_external_resident_bytes(
                batch.nbytes + batch_transient_page_bytes
            )
        self._assign_deepseek_expert_tables(mlp, layer_index, batch.arrays)
        self._set_deepseek_external_resident_bytes(batch.nbytes)
        event = MlxStreamEvent(
            action=(
                "load-selected-experts"
                if use_weight_page_cache
                else "load-selected-experts-direct-budget"
            ),
            layer=layer_index,
            seconds=time.perf_counter() - started,
            resident_bytes=self.session.resident_bytes,
            requested=names,
            loaded=names,
            nbytes_loaded=batch.nbytes,
            transient_page_bytes=batch_transient_page_bytes,
        )
        if self._trace:
            self.session.events.append(event)
        return event

    def _load_deepseek_slice_batch(
        self,
        layer_index: int,
        expert_ids: list[int],
        *,
        use_weight_page_cache: bool = True,
        evaluate: bool | None = None,
    ) -> MlxTensorBatch:
        if evaluate is None:
            evaluate = self.session.evaluate
        return self.session.loader.load_first_dim_slices(
            self._expert_slice_names(layer_index),
            expert_ids,
            evaluate=evaluate,
            use_weight_page_cache=use_weight_page_cache,
        )

    def _deepseek_slice_nbytes(self, layer_index: int, expert_ids: list[int]) -> int:
        if not expert_ids:
            return 0
        total = 0
        for name in self._expert_slice_names(layer_index):
            record = self.session.loader.manifest.tensors[name]
            total += record.nbytes * len(expert_ids) // record.shape[0]
        return total

    def _deepseek_slice_page_miss_nbytes(
        self,
        layer_index: int,
        expert_ids: list[int],
        *,
        cap_to_headroom: bool = False,
    ) -> int:
        return self.session.loader.estimate_first_dim_slice_page_miss_bytes(
            self._expert_slice_names(layer_index),
            expert_ids,
            cap_to_headroom=cap_to_headroom,
        )

    def _assign_deepseek_expert_tables(
        self,
        mlp: Any,
        layer_index: int,
        arrays: dict[str, Any],
    ) -> None:
        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(mlp.switch_mlp, projection)
            module.weight = arrays[f"{prefix}.{projection}.weight"]
            module.scales = arrays[f"{prefix}.{projection}.scales"]
            module.biases = arrays[f"{prefix}.{projection}.biases"]

    def _maybe_submit_deepseek_expert_prefetch(
        self,
        layer_index: int,
        layer: Any,
        *,
        events: list[dict[str, Any]] | None = None,
        pass_kind: str | None = None,
        token_step: int | None = None,
    ) -> None:
        if self.expert_prefetch not in {"previous", "previous_table"}:
            return
        if not hasattr(layer.mlp, "switch_mlp"):
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
        mode = self.expert_prefetch
        pending_page_bytes = self._deepseek_slice_page_miss_nbytes(layer_index, predicted)
        if mode == "previous_table":
            predicted_table_bytes = self._deepseek_slice_nbytes(layer_index, predicted)
            predicted_pending_bytes = predicted_table_bytes + pending_page_bytes
            if not self._deepseek_budget_allows(
                additional_sidecar_bytes=predicted_pending_bytes,
            ):
                if self._deepseek_budget_allows(additional_sidecar_bytes=pending_page_bytes):
                    mode = "previous"
                    self._prefetch_stats.table_prefetch_downgrades += 1
                    if events is not None:
                        events.append(
                            {
                                "kind": "load",
                                "action": "prefetch-table-downgraded-budget",
                                "pass": pass_kind,
                                "token_step": token_step,
                                "layer": layer_index,
                                "predicted_experts": predicted,
                                "predicted_table_bytes": predicted_table_bytes,
                                "pending_page_bytes": pending_page_bytes,
                                "resident_budget_bytes": self.resident_budget_bytes,
                            }
                        )
                else:
                    self._prefetch_stats.skipped_over_budget += 1
                    if events is not None:
                        events.append(
                            {
                                "kind": "load",
                                "action": "prefetch-skipped-over-budget",
                                "pass": pass_kind,
                                "token_step": token_step,
                                "layer": layer_index,
                                "predicted_experts": predicted,
                                "predicted_table_bytes": predicted_table_bytes,
                                "pending_page_bytes": pending_page_bytes,
                                "resident_budget_bytes": self.resident_budget_bytes,
                            }
                        )
                    return
        elif not self._deepseek_budget_allows(additional_sidecar_bytes=pending_page_bytes):
            self._prefetch_stats.skipped_over_budget += 1
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "prefetch-skipped-over-budget",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "predicted_experts": predicted,
                        "pending_page_bytes": pending_page_bytes,
                        "resident_budget_bytes": self.resident_budget_bytes,
                    }
                )
            return

        if mode == "previous_table":
            future = self.loader_executor.submit(
                self._load_deepseek_slice_batch,
                layer_index,
                predicted,
            )
            self._pending_expert_prefetch_bytes = (
                self._deepseek_slice_nbytes(layer_index, predicted) + pending_page_bytes
            )
        else:
            future = self.loader_executor.submit(
                self.session.loader.warm_first_dim_slices,
                self._expert_slice_names(layer_index),
                predicted,
            )
            self._pending_expert_prefetch_bytes = pending_page_bytes
        self._set_deepseek_external_resident_bytes()
        self._pending_expert_prefetch = {
            "layer": layer_index,
            "experts": predicted,
            "future": future,
            "mode": mode,
        }

    def _consume_deepseek_expert_prefetch(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
        *,
        events: list[dict[str, Any]] | None,
        pass_kind: str,
        token_step: int | None,
    ) -> list[int] | None:
        pending = self._pending_expert_prefetch
        self._pending_expert_prefetch = None
        self._pending_expert_prefetch_bytes = 0
        if pending is None:
            return None

        join_started = time.perf_counter()
        batch = pending["future"].result()
        join_wait = time.perf_counter() - join_started
        if pending["layer"] != layer_index:
            self._prefetch_stats.skipped_no_history += 1
            return None

        predicted = pending["experts"]
        ordering, hits, missing = expert_merge_plan(predicted, selected_experts)
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

        fallback_seconds = 0.0
        fallback_bytes = 0
        fallback_transient_page_bytes = 0
        assemble_seconds = 0.0
        assemble_temporary_bytes = 0
        if pending.get("mode") == "previous_table":
            arrays = dict(batch.arrays)
            batch_transient_page_bytes = int(
                getattr(batch, "transient_page_bytes", 0) or 0
            )
            if not missing and not self._deepseek_budget_allows(
                additional_sidecar_bytes=batch.nbytes,
            ):
                stats.skipped_over_budget += 1
                if events is not None:
                    events.append(
                        {
                            "kind": "load",
                            "action": "prefetch-selected-expert-table-over-budget",
                            "pass": pass_kind,
                            "token_step": token_step,
                            "layer": layer_index,
                            "predicted_experts": predicted,
                            "experts": selected_experts,
                            "estimated_table_bytes": batch.nbytes,
                        }
                    )
                self._set_deepseek_external_resident_bytes()
                return None
            if missing:
                if getattr(self, "resident_budget_bytes", None) is not None:
                    estimated_fallback_bytes = self._deepseek_slice_nbytes(layer_index, missing)
                    estimated_fallback_transient = self._deepseek_slice_page_miss_nbytes(
                        layer_index,
                        missing,
                    )
                    estimated_assembly_peak = (
                        batch_transient_page_bytes
                        + estimated_fallback_transient
                        + 2 * (batch.nbytes + estimated_fallback_bytes)
                    )
                    if not self._deepseek_budget_allows(
                        temporary_bytes=estimated_assembly_peak,
                    ):
                        stats.skipped_over_budget += 1
                        if events is not None:
                            events.append(
                                {
                                    "kind": "load",
                                    "action": "prefetch-selected-expert-table-over-budget",
                                    "pass": pass_kind,
                                    "token_step": token_step,
                                    "layer": layer_index,
                                    "predicted_experts": predicted,
                                    "experts": selected_experts,
                                    "estimated_assembly_peak": estimated_assembly_peak,
                                }
                            )
                        self._set_deepseek_external_resident_bytes()
                        return None

                fallback_started = time.perf_counter()
                missing_batch = self._load_deepseek_slice_batch(layer_index, missing)
                fallback_seconds = time.perf_counter() - fallback_started
                fallback_bytes = missing_batch.nbytes
                fallback_transient_page_bytes = int(
                    getattr(missing_batch, "transient_page_bytes", 0) or 0
                )
                stats.fallback_bytes += fallback_bytes
                stats.fallback_load_seconds += fallback_seconds

                import mlx.core as mx

                assemble_started = time.perf_counter()
                assemble_temporary_bytes = (
                    batch_transient_page_bytes
                    + fallback_transient_page_bytes
                    + 2 * (batch.nbytes + missing_batch.nbytes)
                )
                stats.max_assemble_temporary_bytes = max(
                    stats.max_assemble_temporary_bytes,
                    assemble_temporary_bytes,
                )
                self._set_deepseek_external_resident_bytes(assemble_temporary_bytes)
                arrays = {
                    name: mx.concatenate([arrays[name], missing_batch.arrays[name]], axis=0)
                    for name in arrays
                }
                mx.eval(list(arrays.values()))
                assemble_seconds = time.perf_counter() - assemble_started
                stats.assemble_seconds += assemble_seconds

            self._assign_deepseek_expert_tables(mlp, layer_index, arrays)
            self._set_deepseek_external_resident_bytes(batch.nbytes + fallback_bytes)

            event = MlxStreamEvent(
                action="prefetch-selected-expert-table",
                layer=layer_index,
                seconds=join_wait + fallback_seconds + assemble_seconds,
                resident_bytes=self.session.resident_bytes,
                requested=self._expert_slice_names(layer_index),
                loaded=self._expert_slice_names(layer_index),
                nbytes_loaded=batch.nbytes + fallback_bytes,
                transient_page_bytes=batch_transient_page_bytes + fallback_transient_page_bytes,
            )
            if self._trace:
                self.session.events.append(event)
            if events is not None:
                events.append(
                    {
                        "kind": "load",
                        "action": "prefetch-selected-expert-table",
                        "pass": pass_kind,
                        "token_step": token_step,
                        "layer": layer_index,
                        "predicted_experts": predicted,
                        "experts": selected_experts,
                        "table_order": ordering,
                        "hit_count": len(hits),
                        "missing_count": len(missing),
                        "wasted_count": len(wasted),
                        "join_wait_seconds": join_wait,
                        "fallback_seconds": fallback_seconds,
                        "assemble_seconds": assemble_seconds,
                        "assemble_temporary_bytes": assemble_temporary_bytes,
                        "fallback_transient_page_bytes": fallback_transient_page_bytes,
                        "weight_page_cache_bytes": batch.weight_page_cache_bytes,
                        **compact_event(event),
                    }
                )
            return ordering

        if events is not None:
            events.append(
                {
                    "kind": "load",
                    "action": "warm-selected-expert-pages",
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "predicted_experts": predicted,
                    "experts": selected_experts,
                    "hit_count": len(hits),
                    "missing_count": len(missing),
                    "wasted_count": len(wasted),
                    "join_wait_seconds": join_wait,
                    "nbytes_loaded": batch.nbytes,
                    "weight_page_cache_bytes": batch.weight_page_cache_bytes,
                }
            )
        return None

    def _clear_deepseek_selected_experts(self, layer: Any) -> None:
        import mlx.core as mx

        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp.switch_mlp, projection)
            module.weight = mx.zeros((0, 0, 0), dtype=mx.uint32)
            module.scales = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
            module.biases = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
        self._set_deepseek_external_resident_bytes()

    def _deepseek_compute_binds_switch_mlp(self) -> bool:
        if self.expert_prefetch != "off":
            return True
        return self.expert_compute_mode in {"table", "table_overlap_shared"}

    def _expert_telemetry(self) -> dict[str, Any] | None:
        weight_pages = self.session.loader.weight_page_summary()
        slot_arena_active = self.expert_compute_mode == "slot_arena_direct_qmm"
        if weight_pages is None and self.expert_prefetch == "off" and not slot_arena_active:
            return None
        return {
            "mode": "deepseek-weight-pages",
            "weight_page_budget_bytes": self.weight_page_budget_bytes,
            "weight_page_policy": self.weight_page_policy,
            "weight_page_rows": self.weight_page_rows,
            "weight_pages": weight_pages,
            "expert_prefetch_mode": self.expert_prefetch,
            "expert_prefetch": (
                self._prefetch_stats.to_dict()
                if self.expert_prefetch != "off"
                else None
            ),
            "expert_slot_arena": (
                self._expert_slot_stats.to_dict(
                    resident_bytes=self._expert_slot_arena_bytes,
                    arena_count=len(self._expert_slot_arenas),
                    capacity=int(self.expert_slot_capacity or 0),
                )
                if slot_arena_active
                else None
            ),
        }

    def _clear_deepseek_mla_projection_tables(self, layer: Any) -> None:
        import mlx.core as mx

        for module in (layer.self_attn.embed_q, layer.self_attn.unembed_out):
            module.weight = mx.zeros((0, 0, 0), dtype=mx.uint32)
            if hasattr(module, "scales"):
                module.scales = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
            if hasattr(module, "biases"):
                module.biases = mx.zeros((0, 0, 0), dtype=mx.bfloat16)

    def _deepseek_missing_role_bytes(self, role: str) -> int:
        return sum(
            record.nbytes
            for record in self.session.loader.manifest.tensors.values()
            if record.role == role and record.name not in self.session.resident
        )

    def _clear_weight_pages_for_deepseek_output_budget(
        self,
        events: list[dict[str, Any]] | None,
    ) -> None:
        budget = getattr(self, "resident_budget_bytes", None)
        page_bytes = self.session.loader.weight_page_resident_bytes
        if budget is None or not page_bytes:
            return
        output_bytes = self._deepseek_missing_role_bytes("output")
        if self.session.resident_bytes + output_bytes <= budget:
            return

        evicted_bytes = self.session.loader.clear_weight_page_cache()
        self._set_deepseek_external_resident_bytes()
        if events is not None:
            events.append(
                {
                    "kind": "evict",
                    "action": "clear-weight-pages-for-output-budget",
                    "seconds": 0.0,
                    "resident_bytes": self.session.resident_bytes,
                    "nbytes_evicted": evicted_bytes,
                    "output_bytes": output_bytes,
                    "budget_bytes": budget,
                }
            )

    def _logits_from_hidden(self, hidden: Any, events: list[dict[str, Any]]) -> Any:
        import mlx.core as mx

        trace = self._trace
        if self.pin_policy == "phase" and not self.warm_output:
            self._before_phase_output_load(events if trace else [])
            self._clear_weight_pages_for_deepseek_output_budget(events if trace else None)
            load_output = self.session.load_output()
            if trace:
                events.append({"kind": "load", **compact_event(load_output)})

        logits_started = time.perf_counter() if trace else 0.0
        hidden = self.session.model.model.norm(hidden)
        logits = self.session.model.lm_head(hidden)
        mx.eval(logits)
        if trace:
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
            if trace:
                events.append({"kind": "evict", **compact_event(evict_output)})
        return logits
