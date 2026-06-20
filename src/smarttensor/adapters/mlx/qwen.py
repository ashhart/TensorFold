"""Qwen MoE streaming runner."""

from __future__ import annotations

from .core import *

class Qwen35MoeStreamingForwardRunner(StreamingChatRunner):
    """Generate with Qwen3.5-MoE through SmartTensor layer streaming."""

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
        expert_prefetch: str = "off",
        expert_prefetch_cap: int = 32,
        expert_reuse: bool = False,
        expert_reuse_cap: int = 16,
    ) -> None:
        if pin_policy not in {"all", "phase"}:
            raise ValueError("pin_policy must be 'all' or 'phase'")
        if warm_embeddings and pin_policy != "phase":
            raise ValueError("warm_embeddings only applies to pin_policy='phase'")
        if expert_prefetch not in {"off", "previous"}:
            raise ValueError("expert_prefetch must be 'off' or 'previous'")
        if expert_prefetch_cap < 1:
            raise ValueError("expert_prefetch_cap must be positive")
        if expert_reuse_cap < 1:
            raise ValueError("expert_reuse_cap must be positive")
        self.session = MlxModelSession(
            model_dir,
            evaluate=evaluate,
            retain_layers=retain_layers,
            resident_budget_bytes=None,
            backend=backend,
            clear_on_evict=clear_on_evict,
            pin_policy=pin_policy,
            warm_embeddings=warm_embeddings,
        )
        self.pin_policy = pin_policy
        self.warm_embeddings = warm_embeddings
        self.warm_output = pin_policy == "phase"
        self.expert_prefetch = expert_prefetch
        self.expert_prefetch_cap = expert_prefetch_cap
        self.expert_reuse = expert_reuse
        self.expert_reuse_cap = expert_reuse_cap
        self._expert_history: dict[int, list[int]] = {}
        self._pending_expert_prefetch: dict[str, Any] | None = None
        self._prefetch_stats = ExpertPrefetchStats()
        self._expert_table_cache: dict[int, dict[str, Any]] = {}
        self._expert_cache_bytes = 0
        self.loader_executor = ThreadPoolExecutor(max_workers=1)
        self._embed_prefix = "language_model.model.embed_tokens"
        if self.session.config.get("model_type") != "qwen3_5_moe":
            raise ValueError("Qwen35MoeStreamingForwardRunner currently supports model_type=qwen3_5_moe only")
        if retain_layers is not None:
            self.base_retain_layers = set(retain_layers)
        elif resident_budget_bytes is not None:
            self.base_retain_layers = select_qwen_base_layers_for_budget(
                self.session.loader.manifest,
                resident_budget_bytes,
                top_k=self.session.model.language_model.args.num_experts_per_tok,
            )
        else:
            self.base_retain_layers = set()

        from mlx_lm.utils import load_tokenizer

        self.tokenizer = load_tokenizer(self.session.model_dir)

    def close(self) -> None:
        self.loader_executor.shutdown(wait=True)
        self._expert_table_cache.clear()
        self._expert_cache_bytes = 0
        self.session.close()

    def _reset_stream_state(self) -> None:
        self._expert_history.clear()
        self._pending_expert_prefetch = None
        self._prefetch_stats = ExpertPrefetchStats()
        self._expert_table_cache.clear()
        self._expert_cache_bytes = 0

    def _expert_telemetry(self) -> dict[str, Any] | None:
        if self.expert_prefetch != "off" or self.expert_reuse:
            return self._prefetch_stats.to_dict()
        return None

    def _embed_module(self) -> Any:
        return self.session.model.language_model.model.embed_tokens

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
        from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

        core = self.session.model.language_model.model
        if self.pin_policy == "phase" and manage_embedding:
            load_embedding, local_token_rows = self._load_embedding_slices(token_rows)
            events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_embedding)})
            inputs = mx.array(local_token_rows)
            try:
                x = core.embed_tokens(inputs)
                mx.eval(x)
            finally:
                self._clear_embedding_slices()
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

        fa_mask = create_attention_mask(x, cache[core.fa_idx])
        ssm_mask = create_ssm_mask(x, cache[core.ssm_idx])

        for layer_index, (layer, layer_cache) in enumerate(zip(core.layers, cache)):
            self._maybe_submit_expert_prefetch(layer_index)
            load_event = self._load_qwen_layer_base(layer_index)
            events.append({"kind": "load", "pass": pass_kind, "token_step": token_step, **compact_event(load_event)})

            compute_started = time.perf_counter()
            mask = ssm_mask if layer.is_linear else fa_mask
            try:
                x = self._qwen_layer_forward_selective_experts(
                    layer_index,
                    layer,
                    x,
                    mask=mask,
                    cache=layer_cache,
                    events=events,
                    pass_kind=pass_kind,
                    token_step=token_step,
                )
            finally:
                self._clear_qwen_selected_experts(layer)
            events.append(
                {
                    "kind": "compute",
                    "pass": pass_kind,
                    "token_step": token_step,
                    "layer": layer_index,
                    "seconds": time.perf_counter() - compute_started,
                    "hidden_shape": list(x.shape),
                    "is_linear": bool(layer.is_linear),
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
                self.session.events.append(evict_event)
            else:
                evict_event = self.session.evict_layer(layer_index)
            events.append({"kind": "evict", "pass": pass_kind, "token_step": token_step, **compact_event(evict_event)})

        return x

    def _load_qwen_layer_base(self, layer_index: int) -> MlxStreamEvent:
        layer = self.session.loader.manifest.layers[layer_index]
        names = tuple(name for name in layer.tensor_names if ".mlp.switch_mlp." not in name)
        return self.session._load_into_model(names, action="load-layer-base", layer=layer_index)

    def _qwen_layer_forward_selective_experts(
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
    ) -> Any:
        if layer.is_linear:
            residual = layer.linear_attn(layer.input_layernorm(x), mask, cache)
        else:
            residual = layer.self_attn(layer.input_layernorm(x), mask, cache)
        hidden = x + residual
        moe_input = layer.post_attention_layernorm(hidden)
        return hidden + self._qwen_moe_forward_selective_experts(
            layer_index,
            layer.mlp,
            moe_input,
            events=events,
            pass_kind=pass_kind,
            token_step=token_step,
        )

    def _qwen_moe_forward_selective_experts(
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

        gates = mlp.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        top_k = mlp.top_k
        expert_indices = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
        scores = mx.take_along_axis(gates, expert_indices, axis=-1)
        if mlp.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        mx.eval(expert_indices)
        selected_experts = selected_expert_ids(expert_indices)
        self._expert_history[layer_index] = selected_experts

        pending = self._pending_expert_prefetch
        self._pending_expert_prefetch = None
        use_prefetch = pending is not None and pending["layer"] == layer_index
        if pending is not None and not use_prefetch:
            pending["future"].result()

        expert_load = None
        reuse_plan = None
        if self.expert_reuse:
            reuse_plan = self._plan_expert_reuse(layer_index, selected_experts)
            if reuse_plan["load_ids"]:
                expert_load = self.loader_executor.submit(
                    self._load_expert_slice_batch,
                    layer_index,
                    reuse_plan["load_ids"],
                )
        elif not use_prefetch:
            expert_load = self.loader_executor.submit(
                self._load_qwen_selected_experts,
                layer_index,
                mlp,
                selected_experts,
            )

        shared_y = mlp.shared_expert(x)
        shared_y = mx.sigmoid(mlp.shared_expert_gate(x)) * shared_y
        mx.async_eval(shared_y)

        if reuse_plan is not None:
            table_order = self._consume_expert_reuse(
                reuse_plan,
                expert_load,
                mlp,
                layer_index,
                selected_experts,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        elif use_prefetch:
            table_order = self._consume_expert_prefetch(
                pending,
                mlp,
                selected_experts,
                events=events,
                pass_kind=pass_kind,
                token_step=token_step,
            )
        else:
            load_event = expert_load.result()
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
        expert_y = (expert_y * scores[..., None]).sum(axis=-2)
        return expert_y + shared_y

    def _plan_expert_reuse(self, layer_index: int, selected_experts: list[int]) -> dict[str, Any]:
        cached = self._expert_table_cache.get(layer_index)
        if cached is None:
            return {"mode": "rebuild", "load_ids": list(selected_experts), "base": None}

        ordering, hits, missing = expert_merge_plan(cached["order"], selected_experts)
        if not missing:
            return {"mode": "hit", "load_ids": [], "base": cached, "order": list(cached["order"]), "hits": hits}
        if len(ordering) > self.expert_reuse_cap:
            return {
                "mode": "rebuild",
                "load_ids": list(selected_experts),
                "base": None,
                "evicted_rows": len(cached["order"]),
            }
        return {"mode": "extend", "load_ids": missing, "base": cached, "order": ordering, "hits": hits}

    def _consume_expert_reuse(
        self,
        plan: dict[str, Any],
        expert_load: Any,
        mlp: Any,
        layer_index: int,
        selected_experts: list[int],
        *,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> list[int]:
        import mlx.core as mx

        stats = self._prefetch_stats
        stats.true_rows += len(selected_experts)
        mode = plan["mode"]
        join_wait = 0.0
        assemble_seconds = 0.0
        loaded_bytes = 0

        if mode == "hit":
            stats.attempted_layers += 1
            stats.hit_rows += len(plan["hits"])
            stats.full_hits += 1
            arrays = plan["base"]["arrays"]
            order = plan["order"]
            nbytes = plan["base"]["nbytes"]
        elif mode == "extend":
            stats.attempted_layers += 1
            stats.hit_rows += len(plan["hits"])
            stats.missing_rows += len(plan["load_ids"])
            join_started = time.perf_counter()
            batch = expert_load.result()
            join_wait = time.perf_counter() - join_started
            loaded_bytes = batch.nbytes
            assemble_started = time.perf_counter()
            base_arrays = plan["base"]["arrays"]
            arrays = {
                name: mx.concatenate([base_arrays[name], batch.arrays[name]], axis=0)
                for name in base_arrays
            }
            mx.eval(list(arrays.values()))
            assemble_seconds = time.perf_counter() - assemble_started
            order = plan["order"]
            nbytes = plan["base"]["nbytes"] + batch.nbytes
        else:
            stats.skipped_no_history += 1
            stats.missing_rows += len(plan["load_ids"])
            stats.wasted_rows += plan.get("evicted_rows", 0)
            join_started = time.perf_counter()
            batch = expert_load.result()
            join_wait = time.perf_counter() - join_started
            loaded_bytes = batch.nbytes
            arrays = batch.arrays
            order = list(plan["load_ids"])
            nbytes = batch.nbytes

        stats.fallback_bytes += loaded_bytes
        stats.fallback_load_seconds += join_wait
        stats.assemble_seconds += assemble_seconds

        previous = self._expert_table_cache.pop(layer_index, None)
        if previous is not None:
            self._expert_cache_bytes -= previous["nbytes"]
        # Oversized tables (prompt-pass unions) are used but never cached: they
        # would pin several GB across layers and will not recur next token.
        if len(order) <= self.expert_reuse_cap:
            self._expert_table_cache[layer_index] = {"order": order, "arrays": arrays, "nbytes": nbytes}
            self._expert_cache_bytes += nbytes

        self._assign_expert_tables(mlp, layer_index, arrays)
        self.session.set_external_resident_bytes(self._expert_cache_bytes)

        event = MlxStreamEvent(
            action=f"reuse-experts-{mode}",
            layer=layer_index,
            seconds=join_wait + assemble_seconds,
            resident_bytes=self.session.resident_bytes,
            requested=self._expert_slice_names(layer_index),
            loaded=self._expert_slice_names(layer_index) if loaded_bytes else (),
            nbytes_loaded=loaded_bytes,
        )
        self.session.events.append(event)
        events.append(
            {
                "kind": "load",
                "action": f"reuse-experts-{mode}",
                "pass": pass_kind,
                "token_step": token_step,
                "layer": layer_index,
                "expert_count": len(selected_experts),
                "table_rows": len(order),
                "loaded_rows": len(plan["load_ids"]),
                "cache_bytes": self._expert_cache_bytes,
                **compact_event(event),
            }
        )
        return order

    def _maybe_submit_expert_prefetch(self, layer_index: int) -> None:
        if self.expert_reuse:
            return
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
        future = self.loader_executor.submit(
            self._load_expert_slice_batch, layer_index, predicted
        )
        self._pending_expert_prefetch = {
            "layer": layer_index,
            "experts": predicted,
            "future": future,
        }

    def _consume_expert_prefetch(
        self,
        pending: dict[str, Any],
        mlp: Any,
        selected_experts: list[int],
        *,
        events: list[dict[str, Any]],
        pass_kind: str,
        token_step: int | None,
    ) -> list[int]:
        import mlx.core as mx

        layer_index = pending["layer"]
        predicted = pending["experts"]

        join_started = time.perf_counter()
        batch = pending["future"].result()
        join_wait = time.perf_counter() - join_started

        ordering, hits, missing = expert_merge_plan(predicted, selected_experts)
        stats = self._prefetch_stats
        stats.attempted_layers += 1
        stats.predicted_rows += len(predicted)
        stats.true_rows += len(selected_experts)
        stats.hit_rows += len(hits)
        stats.missing_rows += len(missing)
        wasted = len(predicted) - len(hits)
        stats.wasted_rows += wasted
        stats.prefetched_bytes += batch.nbytes
        row_bytes = batch.nbytes // len(predicted) if predicted else 0
        stats.wasted_bytes += row_bytes * wasted
        stats.prefetch_load_seconds += batch.seconds
        stats.join_wait_seconds += join_wait
        if not missing:
            stats.full_hits += 1

        arrays = dict(batch.arrays)
        fallback_bytes = 0
        fallback_seconds = 0.0
        if missing:
            fallback_started = time.perf_counter()
            missing_batch = self._load_expert_slice_batch(layer_index, missing)
            fallback_seconds = time.perf_counter() - fallback_started
            fallback_bytes = missing_batch.nbytes
            stats.fallback_bytes += fallback_bytes
            stats.fallback_load_seconds += fallback_seconds

            assemble_started = time.perf_counter()
            arrays = {
                name: mx.concatenate([arrays[name], missing_batch.arrays[name]], axis=0)
                for name in arrays
            }
            mx.eval(list(arrays.values()))
            stats.assemble_seconds += time.perf_counter() - assemble_started

        self._assign_expert_tables(mlp, layer_index, arrays)
        self.session.set_external_resident_bytes(batch.nbytes + fallback_bytes)

        event = MlxStreamEvent(
            action="prefetch-selected-experts",
            layer=layer_index,
            seconds=batch.seconds + fallback_seconds,
            resident_bytes=self.session.resident_bytes,
            requested=self._expert_slice_names(layer_index),
            loaded=self._expert_slice_names(layer_index),
            nbytes_loaded=batch.nbytes + fallback_bytes,
        )
        self.session.events.append(event)
        events.append(
            {
                "kind": "load",
                "action": "prefetch-selected-experts",
                "pass": pass_kind,
                "token_step": token_step,
                "layer": layer_index,
                "expert_count": len(selected_experts),
                "experts": selected_experts,
                "predicted_experts": predicted,
                "hit_count": len(hits),
                "missing_count": len(missing),
                "join_wait_seconds": join_wait,
                "fallback_seconds": fallback_seconds,
                **compact_event(event),
            }
        )
        return ordering

    def _expert_slice_names(self, layer_index: int) -> tuple[str, ...]:
        prefix = f"language_model.model.layers.{layer_index}.mlp.switch_mlp"
        return tuple(
            f"{prefix}.{projection}.{field}"
            for projection in ("gate_proj", "up_proj", "down_proj")
            for field in ("weight", "scales", "biases")
        )

    def _load_expert_slice_batch(
        self,
        layer_index: int,
        expert_ids: list[int],
    ) -> MlxTensorBatch:
        return self.session.loader.load_first_dim_slices(
            self._expert_slice_names(layer_index),
            expert_ids,
            evaluate=self.session.evaluate,
        )

    def _assign_expert_tables(self, mlp: Any, layer_index: int, arrays: dict[str, Any]) -> None:
        prefix = f"language_model.model.layers.{layer_index}.mlp.switch_mlp"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(mlp.switch_mlp, projection)
            module.weight = arrays[f"{prefix}.{projection}.weight"]
            module.scales = arrays[f"{prefix}.{projection}.scales"]
            module.biases = arrays[f"{prefix}.{projection}.biases"]

    def _load_qwen_selected_experts(
        self,
        layer_index: int,
        mlp: Any,
        selected_experts: list[int],
    ) -> MlxStreamEvent:
        started = time.perf_counter()
        batch = self._load_expert_slice_batch(layer_index, selected_experts)
        self._assign_expert_tables(mlp, layer_index, batch.arrays)
        self.session.set_external_resident_bytes(batch.nbytes)
        names = self._expert_slice_names(layer_index)
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

    def _clear_qwen_selected_experts(self, layer: Any) -> None:
        import mlx.core as mx

        for projection in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp.switch_mlp, projection)
            module.weight = mx.zeros((0, 0, 0), dtype=mx.uint32)
            module.scales = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
            module.biases = mx.zeros((0, 0, 0), dtype=mx.bfloat16)
        if self._expert_cache_bytes:
            self.session.set_external_resident_bytes(self._expert_cache_bytes)
        else:
            self.session.clear_external_resident_bytes()

    def _logits_from_hidden(self, hidden: Any, events: list[dict[str, Any]]) -> Any:
        import mlx.core as mx

        language_model = self.session.model.language_model
        core = language_model.model
        tied_embeddings = language_model.args.tie_word_embeddings

        if self.pin_policy == "phase" and not self.warm_output:
            load_event = self.session.load_embedding() if tied_embeddings else self.session.load_output()
            events.append({"kind": "load", **compact_event(load_event)})

        hidden = core.norm(hidden)
        if tied_embeddings:
            logits = core.embed_tokens.as_linear(hidden)
        else:
            logits = language_model.lm_head(hidden)
        mx.eval(logits)

        if self.pin_policy == "phase" and not self.warm_output:
            evict_event = self.session.evict_embedding() if tied_embeddings else self.session.evict_output()
            events.append({"kind": "evict", **compact_event(evict_event)})
        return logits
