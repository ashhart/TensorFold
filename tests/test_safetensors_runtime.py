from __future__ import annotations

import inspect
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

from smarttensor.adapters.mlx import (
    AdaptiveDraftGate,
    CacheTransaction,
    DeepSeekV3StreamingForwardRunner,
    ExactnessModeSettings,
    ExternalProcessDrafter,
    GptOssStreamingForwardRunner,
    PromptLookupTreeDrafter,
    ExpertPrefetchStats,
    MlxSelectiveLoader,
    MlxStreamingSession,
    PromptLookupDrafter,
    StreamingChatRunner,
    TemporalSlidingKVCache,
    choose_frequency_hot_order,
    should_refresh_hot_set,
    count_accepted_drafts,
    expert_merge_plan,
    prompt_lookup_draft,
    remap_expert_indices,
    qwen_layer_base_bytes,
    qwen_selected_expert_bytes,
    reserve_weight_page_budget_for_base_retention,
    resolve_weight_page_policy,
    select_exactness_mode,
    select_retained_layers_for_budget,
    selected_expert_counts,
    selected_expert_ids,
)
from smarttensor.manifest import (
    LayerRecord,
    SmartTensorManifest,
    TensorRecord,
    build_layers,
    infer_layer_index,
)
from smarttensor.planner import build_streaming_plan, parse_bytes
from smarttensor.runtime import SmartTensorRuntime
from smarttensor.safetensors import SafeTensorFile
from smarttensor.server import (
    longest_reusable_prefix,
    parse_harmony_output,
    pass_economics,
    streaming_visible_text,
    strip_trailing_stops,
)


class SmartTensorRuntimeTests(unittest.TestCase):
    def test_safetensors_metadata_is_parsed_without_loading_tensor_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_toy_safetensors(Path(directory) / "toy.safetensors")

            with SafeTensorFile(path) as safe_file:
                self.assertEqual(safe_file.user_metadata["name"], "toy")
                self.assertEqual(
                    safe_file.tensor_names(),
                    [
                        "lm_head.weight",
                        "model.embed_tokens.weight",
                        "model.layers.0.mlp.up_proj.weight",
                        "model.layers.0.self_attn.q_proj.weight",
                    ],
                )
                tensor = safe_file.tensors["model.layers.0.self_attn.q_proj.weight"]
                self.assertEqual(tensor.dtype, "F32")
                self.assertEqual(tensor.shape, (2, 2))
                self.assertEqual(tensor.nbytes, 16)

    def test_manifest_groups_common_transformer_layer_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_toy_safetensors(Path(directory) / "toy.safetensors")

            manifest = SmartTensorManifest.from_safetensors([path])

            self.assertEqual(manifest.total_bytes, 64)
            self.assertEqual(manifest.bytes_by_dtype(), {"F32": 64})
            self.assertEqual(list(manifest.layers), [0])
            self.assertEqual(manifest.layers[0].nbytes, 32)
            self.assertIn("model.embed_tokens.weight", manifest.unlayered_tensors())

    def test_runtime_reads_tensor_bytes_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_toy_safetensors(Path(directory) / "toy.safetensors")

            with SmartTensorRuntime([path]) as runtime:
                data = runtime.read_tensor_bytes("model.layers.0.mlp.up_proj.weight", limit=4)

            self.assertEqual(data, bytes([32, 33, 34, 35]))

    def test_runtime_can_keep_tensors_under_a_residency_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_toy_safetensors(Path(directory) / "toy.safetensors")

            with SmartTensorRuntime([path], cache_budget_bytes=16) as runtime:
                first = runtime.get_tensor(
                    "model.layers.0.self_attn.q_proj.weight", keep_resident=True
                )
                self.assertEqual(runtime.cache.names(), ["model.layers.0.self_attn.q_proj.weight"])
                second = runtime.get_tensor("model.layers.0.mlp.up_proj.weight", keep_resident=True)
                self.assertEqual(runtime.cache.names(), ["model.layers.0.mlp.up_proj.weight"])
                self.assertEqual(second.view[:2].tobytes(), bytes([32, 33]))
                self.assertEqual(first.name, "model.layers.0.self_attn.q_proj.weight")

    def test_streaming_plan_estimates_memory_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_toy_safetensors(Path(directory) / "toy.safetensors")
            manifest = SmartTensorManifest.from_safetensors([path])

            plan = build_streaming_plan(
                manifest,
                budget_bytes=parse_bytes("128B"),
                prefetch_window=0,
            )

            self.assertTrue(plan.fits_budget)
            self.assertEqual(plan.pinned_bytes, 32)
            self.assertEqual(plan.largest_layer_bytes, 32)
            self.assertEqual(plan.peak_estimated_bytes, 64)
            self.assertEqual(plan.steps[0].action, "pin")

    def test_mlx_adapter_loads_selected_layer_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_toy_safetensors(Path(directory) / "toy.safetensors")
            loader = MlxSelectiveLoader.from_model_dir(path.parent)

            try:
                batch = loader.load_layer(0)
            finally:
                loader.close()

            self.assertEqual(batch.tensor_count, 2)
            self.assertEqual(batch.nbytes, 32)
            self.assertEqual(
                sorted(batch.arrays),
                [
                    "model.layers.0.mlp.up_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                ],
            )

    def test_mlx_streaming_session_dedupes_pinned_layer_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_toy_safetensors(Path(directory) / "toy.safetensors")
            session = MlxStreamingSession.from_model_dir(path.parent)
            try:
                pin_event = session.pin()
                load_event = session.load_layer(0)
                evict_event = session.evict_layer(0)

                self.assertEqual(pin_event.loaded, ("lm_head.weight", "model.embed_tokens.weight"))
                self.assertEqual(load_event.loaded, (
                    "model.layers.0.mlp.up_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                ))
                self.assertEqual(evict_event.evicted, (
                    "model.layers.0.mlp.up_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                ))
                self.assertEqual(sorted(session.resident), ["lm_head.weight", "model.embed_tokens.weight"])
            finally:
                session.close()

    def test_retained_layer_budget_selects_prefix_that_fits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_toy_safetensors(Path(directory) / "toy.safetensors")
            manifest = SmartTensorManifest.from_safetensors([path])

            retained = select_retained_layers_for_budget(manifest, resident_budget_bytes=64)

            self.assertEqual(retained, {0})

    def test_phase_budget_keeps_one_gpt_oss_sized_layer_under_2gb(self) -> None:
        manifest = sized_manifest(
            pin_small_bytes=144_000,
            embedding_bytes=615_329_280,
            output_bytes=615_329_280,
            layer_bytes=(874_902_592, 874_902_592, 874_902_592),
        )

        phase_retained = select_retained_layers_for_budget(
            manifest,
            resident_budget_bytes=2_000_000_000,
            pin_policy="phase",
        )
        all_retained = select_retained_layers_for_budget(
            manifest,
            resident_budget_bytes=2_000_000_000,
            pin_policy="all",
        )
        warm_embedding_retained = select_retained_layers_for_budget(
            manifest,
            resident_budget_bytes=2_000_000_000,
            pin_policy="phase",
            warm_embeddings=True,
        )

        self.assertEqual(phase_retained, {0})
        self.assertEqual(all_retained, set())
        self.assertEqual(warm_embedding_retained, set())

    def test_manifest_does_not_group_vision_blocks_as_text_layers(self) -> None:
        self.assertIsNone(infer_layer_index("vision_tower.blocks.0.attn.qkv.weight"))
        self.assertEqual(infer_layer_index("language_model.model.layers.0.mlp.gate.weight"), 0)

    def test_deepseek_switch_mlp_rows_are_planned_as_routed_experts(self) -> None:
        def record(name: str, nbytes: int, shape: tuple[int, ...]) -> TensorRecord:
            return TensorRecord(
                name=name,
                file="deepseek.safetensors",
                dtype="U32",
                shape=shape,
                data_offsets=(0, nbytes),
                absolute_offsets=(0, nbytes),
                nbytes=nbytes,
                layer=infer_layer_index(name),
            )

        tensors = {
            "model.layers.3.self_attn.kv_b_proj.weight": record(
                "model.layers.3.self_attn.kv_b_proj.weight",
                1024,
                (256, 1),
            ),
            "model.layers.3.mlp.shared_experts.gate_proj.weight": record(
                "model.layers.3.mlp.shared_experts.gate_proj.weight",
                2048,
                (256, 2),
            ),
            "model.layers.3.mlp.switch_mlp.gate_proj.weight": record(
                "model.layers.3.mlp.switch_mlp.gate_proj.weight",
                2560,
                (256, 10),
            ),
            "model.layers.3.mlp.switch_mlp.up_proj.weight": record(
                "model.layers.3.mlp.switch_mlp.up_proj.weight",
                5120,
                (256, 20),
            ),
            "model.layers.3.mlp.switch_mlp.down_proj.weight": record(
                "model.layers.3.mlp.switch_mlp.down_proj.weight",
                7680,
                (256, 30),
            ),
        }
        manifest = SmartTensorManifest(
            format="safetensors",
            files=("deepseek.safetensors",),
            tensors=tensors,
            layers=build_layers(tensors),
        )

        self.assertEqual(
            qwen_layer_base_bytes(
                manifest,
                3,
                expert_marker=".mlp.switch_mlp.",
            ),
            3072,
        )
        self.assertEqual(
            qwen_selected_expert_bytes(
                manifest,
                top_k=8,
                expert_marker=".mlp.switch_mlp.",
            ),
            480,
        )

    def test_deepseek_runner_exposes_weight_page_cache_knobs(self) -> None:
        signature = inspect.signature(DeepSeekV3StreamingForwardRunner)

        self.assertIn("weight_page_budget_bytes", signature.parameters)
        self.assertIn("weight_page_policy", signature.parameters)
        self.assertIn("weight_page_rows", signature.parameters)
        self.assertIn("expert_prefetch", signature.parameters)
        self.assertIn("expert_prefetch_cap", signature.parameters)
        self.assertIn("trace", signature.parameters)
        self.assertIs(signature.parameters["clear_on_evict"].default, False)
        self.assertIs(signature.parameters["trace"].default, False)

    def test_weight_page_policy_auto_prefers_frequency_for_deepseek_budget(self) -> None:
        self.assertEqual(
            resolve_weight_page_policy(
                "deepseek_v3",
                "auto",
                has_weight_page_budget=True,
            ),
            "frequency",
        )
        self.assertEqual(
            resolve_weight_page_policy(
                "gpt_oss",
                "auto",
                has_weight_page_budget=True,
            ),
            "lru",
        )
        self.assertEqual(
            resolve_weight_page_policy(
                "deepseek_v3",
                "two_queue",
                has_weight_page_budget=True,
            ),
            "two_queue",
        )

    def test_weight_page_budget_is_reserved_before_base_retention(self) -> None:
        self.assertEqual(
            reserve_weight_page_budget_for_base_retention(4 * 1024, 2 * 1024),
            2 * 1024,
        )
        self.assertEqual(
            reserve_weight_page_budget_for_base_retention(4 * 1024, None),
            4 * 1024,
        )
        self.assertEqual(
            reserve_weight_page_budget_for_base_retention(2 * 1024, 4 * 1024),
            0,
        )

    def test_weight_page_rows_requires_positive_budgeted_cache(self) -> None:
        gpt_signature = inspect.signature(GptOssStreamingForwardRunner)
        deepseek_signature = inspect.signature(DeepSeekV3StreamingForwardRunner)

        self.assertIn("weight_page_rows", gpt_signature.parameters)
        self.assertIn("weight_page_rows", deepseek_signature.parameters)

        for runner_class in (GptOssStreamingForwardRunner, DeepSeekV3StreamingForwardRunner):
            with self.subTest(runner=runner_class.__name__, rows=0):
                with self.assertRaisesRegex(ValueError, "weight_page_rows must be positive"):
                    runner = object.__new__(runner_class)
                    runner_class.__init__(
                        runner,
                        Path("/nonexistent"),
                        weight_page_rows=0,
                    )
            with self.subTest(runner=runner_class.__name__, no_budget=True):
                with self.assertRaisesRegex(ValueError, "weight_page_rows requires"):
                    runner = object.__new__(runner_class)
                    runner_class.__init__(
                        runner,
                        Path("/nonexistent"),
                        weight_page_rows=2,
                    )


class ExpertPrefetchPlanningTests(unittest.TestCase):
    def test_decode_scheduler_auto_falls_back_when_overlap_is_unsafe(self) -> None:
        class FakeRunner:
            decode_scheduler = "auto"

            def _async_lookahead_support(self) -> tuple[bool, str]:
                return False, "streamed layers can evict in-flight weights"

        active, telemetry = StreamingChatRunner._select_decode_scheduler(
            FakeRunner(),
            prompt_count=1,
            temperature=0.0,
            max_tokens=8,
        )

        self.assertFalse(active)
        self.assertEqual(telemetry["active"], "serial")
        self.assertIn("evict", telemetry["reason"])

    def test_decode_scheduler_explicit_async_rejects_unsafe_forward(self) -> None:
        class FakeRunner:
            decode_scheduler = "async-lookahead"

            def _async_lookahead_support(self) -> tuple[bool, str]:
                return False, "streamed layers can evict in-flight weights"

        with self.assertRaises(ValueError):
            StreamingChatRunner._select_decode_scheduler(
                FakeRunner(),
                prompt_count=1,
                temperature=0.0,
                max_tokens=8,
            )

    def test_decode_scheduler_blocks_unmarked_callbacks(self) -> None:
        class FakeRunner:
            decode_scheduler = "auto"

            def _async_lookahead_support(self) -> tuple[bool, str]:
                return True, "safe"

        def on_step(step_tokens: list[int], finished: list[bool]) -> None:
            pass

        active, telemetry = StreamingChatRunner._select_decode_scheduler(
            FakeRunner(),
            prompt_count=1,
            temperature=0.0,
            max_tokens=8,
            on_step=on_step,
        )

        self.assertFalse(active)
        self.assertEqual(telemetry["reason_code"], "on_step")

    def test_decode_scheduler_allows_marked_callbacks(self) -> None:
        class FakeRunner:
            decode_scheduler = "auto"

            def _async_lookahead_support(self) -> tuple[bool, str]:
                return True, "safe"

        def on_step(step_tokens: list[int], finished: list[bool]) -> None:
            pass

        on_step.smarttensor_async_safe = True

        active, telemetry = StreamingChatRunner._select_decode_scheduler(
            FakeRunner(),
            prompt_count=1,
            temperature=0.0,
            max_tokens=8,
            on_step=on_step,
        )

        self.assertTrue(active)
        self.assertEqual(telemetry["active"], "async-lookahead")

    def test_gpt_oss_native_resident_bytes_are_not_sidecar_bytes(self) -> None:
        class FakeLoader:
            weight_page_resident_bytes = 17

        class FakeSession:
            def __init__(self) -> None:
                self.loader = FakeLoader()
                self.external_resident_bytes = 0

            def set_external_resident_bytes(self, nbytes: int) -> None:
                self.external_resident_bytes = nbytes

            def clear_external_resident_bytes(self) -> None:
                self.external_resident_bytes = 0

        runner = object.__new__(GptOssStreamingForwardRunner)
        runner.session = FakeSession()
        runner._expert_cache_bytes = 11
        runner._native_resident_bytes = 1_000
        runner._pending_expert_prefetch_bytes = 13

        self.assertEqual(runner._resident_sidecar_bytes(), 41)

        GptOssStreamingForwardRunner._set_external_resident_bytes(runner)

        self.assertEqual(runner.session.external_resident_bytes, 41)

    def test_expert_merge_plan_orders_predicted_rows_before_missing_rows(self) -> None:
        ordering, hits, missing = expert_merge_plan([3, 7, 9], [1, 7, 9])

        self.assertEqual(ordering, [3, 7, 9, 1])
        self.assertEqual(hits, [7, 9])
        self.assertEqual(missing, [1])

    def test_expert_merge_plan_full_hit_keeps_predicted_table(self) -> None:
        ordering, hits, missing = expert_merge_plan([2, 5, 8], [2, 8])

        self.assertEqual(ordering, [2, 5, 8])
        self.assertEqual(hits, [2, 8])
        self.assertEqual(missing, [])

    def test_remap_expert_indices_handles_unused_table_rows(self) -> None:
        indices = np.array([[[7, 1]]])

        local = remap_expert_indices(indices, [3, 7, 9, 1])

        np.testing.assert_array_equal(np.array(local), [[[1, 3]]])

    def test_selected_expert_ids_unions_batch_rows(self) -> None:
        indices = np.array([[[3, 1]], [[1, 5]]])

        self.assertEqual(selected_expert_ids(indices), [1, 3, 5])

    def test_selected_expert_counts_preserves_multiplicity(self) -> None:
        indices = np.array([[[3, 1]], [[1, 5]], [[3, 1]]])

        self.assertEqual(selected_expert_counts(indices), {1: 3, 3: 2, 5: 1})

    def test_frequency_hot_order_keeps_resident_rows_on_ties(self) -> None:
        order = choose_frequency_hot_order(
            [2, 5],
            [8],
            {2: 3, 5: 1, 8: 1},
            cap=2,
        )

        self.assertEqual(order, [2, 5])

    def test_refresh_gate_requires_count_margin(self) -> None:
        current = [1, 2, 3]
        counts = {1: 10, 2: 8, 3: 5, 9: 6}
        self.assertFalse(should_refresh_hot_set(current, [9], counts, hysteresis=4))
        counts[9] = 9
        self.assertTrue(should_refresh_hot_set(current, [9], counts, hysteresis=4))

    def test_refresh_gate_edge_cases(self) -> None:
        self.assertTrue(should_refresh_hot_set([], [4], {}, hysteresis=4))
        self.assertFalse(should_refresh_hot_set([1], [], {1: 1}, hysteresis=4))
        # Unknown experts count as zero.
        self.assertFalse(should_refresh_hot_set([1], [2], {1: 0}, hysteresis=1))
        self.assertTrue(should_refresh_hot_set([1], [2], {1: 0, 2: 1}, hysteresis=1))

    def test_frequency_hot_order_replaces_cold_resident_rows(self) -> None:
        order = choose_frequency_hot_order(
            [2, 5],
            [8],
            {2: 3, 5: 1, 8: 4},
            cap=2,
        )

        self.assertEqual(order, [8, 2])

    def test_prefetch_stats_reports_coverage_and_waste(self) -> None:
        stats = ExpertPrefetchStats(
            attempted_layers=4,
            full_hits=1,
            predicted_rows=32,
            true_rows=32,
            hit_rows=14,
            wasted_rows=18,
            missing_rows=18,
            prefetch_load_seconds=0.4,
            join_wait_seconds=0.1,
        )

        data = stats.to_dict()

        self.assertAlmostEqual(data["coverage"], 14 / 32)
        self.assertAlmostEqual(data["waste_rate"], 18 / 32)
        self.assertAlmostEqual(data["full_hit_rate"], 0.25)
        self.assertAlmostEqual(data["hidden_load_seconds"], 0.3)

    def test_deepseek_reset_drains_pending_prefetch_before_forgetting_it(self) -> None:
        class FakeFuture:
            def __init__(self) -> None:
                self.result_calls = 0

            def result(self) -> None:
                self.result_calls += 1

        class FakeLoader:
            weight_page_resident_bytes = 17

        class FakeSession:
            def __init__(self) -> None:
                self.loader = FakeLoader()
                self.external_resident_bytes = 0

            def set_external_resident_bytes(self, nbytes: int) -> None:
                self.external_resident_bytes = nbytes

            def clear_external_resident_bytes(self) -> None:
                self.external_resident_bytes = 0

        future = FakeFuture()
        runner = object.__new__(DeepSeekV3StreamingForwardRunner)
        runner.session = FakeSession()
        runner._expert_cache_bytes = 123
        runner._expert_history = {7: [1, 3, 5]}
        runner._pending_expert_prefetch = {"future": future}
        runner._pending_expert_prefetch_bytes = 80
        runner._prefetch_stats = ExpertPrefetchStats(attempted_layers=1)

        DeepSeekV3StreamingForwardRunner._reset_stream_state(runner)

        self.assertEqual(future.result_calls, 1)
        self.assertIsNone(runner._pending_expert_prefetch)
        self.assertEqual(runner._pending_expert_prefetch_bytes, 0)
        self.assertEqual(runner._expert_cache_bytes, 0)
        self.assertEqual(runner._expert_history, {})
        self.assertEqual(runner._prefetch_stats, ExpertPrefetchStats())
        self.assertEqual(runner.session.external_resident_bytes, 17)

    def test_deepseek_reset_settles_state_when_pending_prefetch_raises(self) -> None:
        class FakeFuture:
            def __init__(self) -> None:
                self.result_calls = 0

            def result(self) -> None:
                self.result_calls += 1
                raise RuntimeError("prefetch failed")

        class FakeLoader:
            weight_page_resident_bytes = 17

        class FakeSession:
            def __init__(self) -> None:
                self.loader = FakeLoader()
                self.external_resident_bytes = 0

            def set_external_resident_bytes(self, nbytes: int) -> None:
                self.external_resident_bytes = nbytes

            def clear_external_resident_bytes(self) -> None:
                self.external_resident_bytes = 0

        future = FakeFuture()
        runner = object.__new__(DeepSeekV3StreamingForwardRunner)
        runner.session = FakeSession()
        runner._expert_cache_bytes = 123
        runner._expert_history = {7: [1, 3, 5]}
        runner._pending_expert_prefetch = {"future": future}
        runner._pending_expert_prefetch_bytes = 80
        runner._prefetch_stats = ExpertPrefetchStats(attempted_layers=1)

        with self.assertRaisesRegex(RuntimeError, "prefetch failed"):
            DeepSeekV3StreamingForwardRunner._reset_stream_state(runner)

        self.assertEqual(future.result_calls, 1)
        self.assertIsNone(runner._pending_expert_prefetch)
        self.assertEqual(runner._pending_expert_prefetch_bytes, 0)
        self.assertEqual(runner._expert_cache_bytes, 0)
        self.assertEqual(runner._expert_history, {})
        self.assertEqual(runner._prefetch_stats, ExpertPrefetchStats())
        self.assertEqual(runner.session.external_resident_bytes, 17)

    def test_deepseek_previous_table_prefetch_full_hit_assigns_predicted_table(self) -> None:
        class FakeFuture:
            def __init__(self, batch: object) -> None:
                self.batch = batch
                self.result_calls = 0

            def result(self) -> object:
                self.result_calls += 1
                return self.batch

        class FakeLoader:
            weight_page_resident_bytes = 32

        class FakeSession:
            def __init__(self) -> None:
                self.loader = FakeLoader()
                self.events: list[object] = []
                self.resident_bytes = 0
                self.external_resident_bytes = 0

            def set_external_resident_bytes(self, nbytes: int) -> None:
                self.external_resident_bytes = nbytes

            def clear_external_resident_bytes(self) -> None:
                self.external_resident_bytes = 0

        class Projection:
            weight: object | None = None
            scales: object | None = None
            biases: object | None = None

        layer_index = 3
        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"
        arrays = {
            f"{prefix}.{projection}.{field}": f"{projection}-{field}"
            for projection in ("gate_proj", "up_proj", "down_proj")
            for field in ("weight", "scales", "biases")
        }
        batch = type(
            "Batch",
            (),
            {
                "nbytes": 80,
                "seconds": 0.02,
                "arrays": arrays,
                "weight_page_cache_bytes": 32,
            },
        )()
        future = FakeFuture(batch)
        runner = object.__new__(DeepSeekV3StreamingForwardRunner)
        runner.session = FakeSession()
        runner._expert_cache_bytes = 0
        runner._pending_expert_prefetch_bytes = 80
        runner._pending_expert_prefetch = {
            "layer": layer_index,
            "experts": [2, 5],
            "future": future,
            "mode": "previous_table",
        }
        runner._prefetch_stats = ExpertPrefetchStats()
        switch_mlp = type(
            "SwitchMlp",
            (),
            {
                "gate_proj": Projection(),
                "up_proj": Projection(),
                "down_proj": Projection(),
            },
        )()
        mlp = type("Mlp", (), {"switch_mlp": switch_mlp})()
        events: list[dict[str, object]] = []

        order = DeepSeekV3StreamingForwardRunner._consume_deepseek_expert_prefetch(
            runner,
            layer_index,
            mlp,
            [5],
            events=events,
            pass_kind="decode",
            token_step=4,
        )

        self.assertEqual(order, [2, 5])
        self.assertIsNone(runner._pending_expert_prefetch)
        self.assertEqual(runner._pending_expert_prefetch_bytes, 0)
        self.assertEqual(future.result_calls, 1)
        self.assertEqual(switch_mlp.gate_proj.weight, "gate_proj-weight")
        self.assertEqual(switch_mlp.up_proj.scales, "up_proj-scales")
        self.assertEqual(switch_mlp.down_proj.biases, "down_proj-biases")
        self.assertEqual(runner.session.external_resident_bytes, 112)
        self.assertEqual(runner._prefetch_stats.full_hits, 1)
        self.assertEqual(runner._prefetch_stats.hit_rows, 1)
        self.assertEqual(runner._prefetch_stats.wasted_rows, 1)
        self.assertEqual(events[0]["action"], "prefetch-selected-expert-table")

    def test_deepseek_previous_table_prefetch_downgrades_when_table_budget_is_tight(self) -> None:
        class FakeFuture:
            def result(self) -> None:
                return None

        class FakeExecutor:
            def __init__(self) -> None:
                self.calls: list[tuple[object, tuple[object, ...]]] = []

            def submit(self, func: object, *args: object) -> FakeFuture:
                self.calls.append((func, args))
                return FakeFuture()

        class FakeLoader:
            weight_page_resident_bytes = 10

            def warm_first_dim_slices(self, *args: object) -> None:
                return None

        class FakeSession:
            def __init__(self) -> None:
                self.loader = FakeLoader()
                self.resident_bytes = 0
                self.external_resident_bytes = 0

            def set_external_resident_bytes(self, nbytes: int) -> None:
                self.external_resident_bytes = nbytes

            def clear_external_resident_bytes(self) -> None:
                self.external_resident_bytes = 0

        runner = object.__new__(DeepSeekV3StreamingForwardRunner)
        runner.session = FakeSession()
        runner.resident_budget_bytes = 40
        runner.expert_prefetch = "previous_table"
        runner.expert_prefetch_cap = 8
        runner._expert_history = {3: [2, 5]}
        runner._pending_expert_prefetch = None
        runner._pending_expert_prefetch_bytes = 0
        runner._expert_cache_bytes = 0
        runner._prefetch_stats = ExpertPrefetchStats()
        runner.loader_executor = FakeExecutor()
        runner._deepseek_slice_page_miss_nbytes = lambda layer, experts, **kwargs: 20
        runner._deepseek_slice_nbytes = lambda layer, experts: 80
        switch_mlp = object()
        layer = type("Layer", (), {"mlp": type("Mlp", (), {"switch_mlp": switch_mlp})()})()
        events: list[dict[str, object]] = []

        DeepSeekV3StreamingForwardRunner._maybe_submit_deepseek_expert_prefetch(
            runner,
            3,
            layer,
            events=events,
            pass_kind="decode",
            token_step=9,
        )

        self.assertIsNotNone(runner._pending_expert_prefetch)
        assert runner._pending_expert_prefetch is not None
        self.assertEqual(runner._pending_expert_prefetch["mode"], "previous")
        self.assertEqual(runner._pending_expert_prefetch_bytes, 20)
        self.assertEqual(runner.session.external_resident_bytes, 30)
        self.assertEqual(runner._prefetch_stats.table_prefetch_downgrades, 1)
        self.assertEqual(runner._prefetch_stats.skipped_over_budget, 0)
        self.assertEqual(len(runner.loader_executor.calls), 1)
        self.assertEqual(runner.loader_executor.calls[0][0].__name__, "warm_first_dim_slices")
        self.assertEqual(events[0]["action"], "prefetch-table-downgraded-budget")
        self.assertEqual(events[0]["pass"], "decode")
        self.assertEqual(events[0]["token_step"], 9)
        self.assertEqual(events[0]["predicted_table_bytes"], 80)
        self.assertEqual(events[0]["pending_page_bytes"], 20)

    def test_deepseek_prefetch_skips_when_warm_pages_exceed_budget(self) -> None:
        class FakeExecutor:
            def __init__(self) -> None:
                self.calls: list[tuple[object, tuple[object, ...]]] = []

            def submit(self, func: object, *args: object) -> object:
                self.calls.append((func, args))
                return object()

        class FakeLoader:
            weight_page_resident_bytes = 10

            def warm_first_dim_slices(self, *args: object) -> None:
                return None

        class FakeSession:
            def __init__(self) -> None:
                self.loader = FakeLoader()
                self.resident_bytes = 0
                self.external_resident_bytes = 0

        runner = object.__new__(DeepSeekV3StreamingForwardRunner)
        runner.session = FakeSession()
        runner.resident_budget_bytes = 20
        runner.expert_prefetch = "previous"
        runner.expert_prefetch_cap = 8
        runner._expert_history = {3: [2, 5]}
        runner._pending_expert_prefetch = None
        runner._pending_expert_prefetch_bytes = 0
        runner._expert_cache_bytes = 0
        runner._prefetch_stats = ExpertPrefetchStats()
        runner.loader_executor = FakeExecutor()
        runner._deepseek_slice_page_miss_nbytes = lambda layer, experts: 20
        layer = type("Layer", (), {"mlp": type("Mlp", (), {"switch_mlp": object()})()})()
        events: list[dict[str, object]] = []

        DeepSeekV3StreamingForwardRunner._maybe_submit_deepseek_expert_prefetch(
            runner,
            3,
            layer,
            events=events,
            pass_kind="decode",
            token_step=9,
        )

        self.assertIsNone(runner._pending_expert_prefetch)
        self.assertEqual(runner._pending_expert_prefetch_bytes, 0)
        self.assertEqual(runner._prefetch_stats.skipped_over_budget, 1)
        self.assertEqual(runner.loader_executor.calls, [])
        self.assertEqual(events[0]["action"], "prefetch-skipped-over-budget")
        self.assertEqual(events[0]["pass"], "decode")
        self.assertEqual(events[0]["token_step"], 9)
        self.assertEqual(events[0]["pending_page_bytes"], 20)

    def test_deepseek_previous_table_prefetch_miss_accounts_assemble_peak(self) -> None:
        import mlx.core as mx

        class FakeFuture:
            def __init__(self, batch: object) -> None:
                self.batch = batch
                self.result_calls = 0

            def result(self) -> object:
                self.result_calls += 1
                return self.batch

        class FakeLoader:
            weight_page_resident_bytes = 32

            def __init__(self, missing_batch: object) -> None:
                self.missing_batch = missing_batch
                self.requests: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

            def load_first_dim_slices(
                self,
                names: tuple[str, ...],
                indices: list[int],
                *,
                evaluate: bool = False,
                use_weight_page_cache: bool = True,
            ) -> object:
                self.requests.append((tuple(names), tuple(indices)))
                return self.missing_batch

        class FakeSession:
            def __init__(self, loader: FakeLoader) -> None:
                self.loader = loader
                self.events: list[object] = []
                self.resident_bytes = 0
                self.external_resident_bytes = 0
                self.peak_external_resident_bytes = 0
                self.evaluate = False

            def set_external_resident_bytes(self, nbytes: int) -> None:
                self.external_resident_bytes = nbytes
                self.peak_external_resident_bytes = max(
                    self.peak_external_resident_bytes,
                    nbytes,
                )

            def clear_external_resident_bytes(self) -> None:
                self.external_resident_bytes = 0

        class Projection:
            weight: object | None = None
            scales: object | None = None
            biases: object | None = None

        layer_index = 3
        prefix = f"model.layers.{layer_index}.mlp.switch_mlp"

        def batch(
            nbytes: int,
            rows: int,
            value: int,
            *,
            transient_page_bytes: int = 0,
        ) -> object:
            arrays = {
                f"{prefix}.{projection}.{field}": mx.array(
                    np.full((rows, 1), value, dtype=np.float32)
                )
                for projection in ("gate_proj", "up_proj", "down_proj")
                for field in ("weight", "scales", "biases")
            }
            return type(
                "Batch",
                (),
                {
                    "nbytes": nbytes,
                    "seconds": 0.02,
                    "arrays": arrays,
                    "weight_page_cache_bytes": 32,
                    "transient_page_bytes": transient_page_bytes,
                },
            )()

        predicted_batch = batch(80, 2, 1, transient_page_bytes=30)
        missing_batch = batch(40, 1, 2, transient_page_bytes=10)
        future = FakeFuture(predicted_batch)
        loader = FakeLoader(missing_batch)
        runner = object.__new__(DeepSeekV3StreamingForwardRunner)
        runner.session = FakeSession(loader)
        runner._expert_cache_bytes = 0
        runner._pending_expert_prefetch_bytes = 120
        runner._pending_expert_prefetch = {
            "layer": layer_index,
            "experts": [2, 5],
            "future": future,
            "mode": "previous_table",
        }
        runner._prefetch_stats = ExpertPrefetchStats()
        switch_mlp = type(
            "SwitchMlp",
            (),
            {
                "gate_proj": Projection(),
                "up_proj": Projection(),
                "down_proj": Projection(),
            },
        )()
        mlp = type("Mlp", (), {"switch_mlp": switch_mlp})()
        events: list[dict[str, object]] = []

        order = DeepSeekV3StreamingForwardRunner._consume_deepseek_expert_prefetch(
            runner,
            layer_index,
            mlp,
            [5, 7],
            events=events,
            pass_kind="decode",
            token_step=4,
        )

        self.assertEqual(order, [2, 5, 7])
        self.assertEqual(loader.requests[0][1], (7,))
        self.assertEqual(future.result_calls, 1)
        self.assertEqual(runner._pending_expert_prefetch_bytes, 0)
        self.assertEqual(runner._prefetch_stats.hit_rows, 1)
        self.assertEqual(runner._prefetch_stats.missing_rows, 1)
        self.assertEqual(runner._prefetch_stats.fallback_bytes, 40)
        self.assertEqual(runner._prefetch_stats.max_assemble_temporary_bytes, 280)
        self.assertEqual(events[0]["assemble_temporary_bytes"], 280)
        self.assertEqual(events[0]["fallback_transient_page_bytes"], 10)
        self.assertEqual(events[0]["transient_page_bytes"], 40)
        self.assertEqual(runner.session.external_resident_bytes, 152)
        self.assertEqual(runner.session.peak_external_resident_bytes, 312)

    def test_deepseek_previous_table_prefetch_miss_falls_back_when_assembly_exceeds_budget(self) -> None:
        class FakeFuture:
            def __init__(self, batch: object) -> None:
                self.batch = batch
                self.result_calls = 0

            def result(self) -> object:
                self.result_calls += 1
                return self.batch

        class FakeLoader:
            weight_page_resident_bytes = 32

            def __init__(self) -> None:
                self.requests: list[tuple[tuple[str, ...], tuple[int, ...]]] = []

            def load_first_dim_slices(
                self,
                names: tuple[str, ...],
                indices: list[int],
                *,
                evaluate: bool = False,
            ) -> object:
                self.requests.append((tuple(names), tuple(indices)))
                raise AssertionError("partial over-budget path should not load fallback rows")

        class FakeSession:
            def __init__(self, loader: FakeLoader) -> None:
                self.loader = loader
                self.events: list[object] = []
                self.resident_bytes = 0
                self.external_resident_bytes = 0
                self.evaluate = False

            def set_external_resident_bytes(self, nbytes: int) -> None:
                self.external_resident_bytes = nbytes

            def clear_external_resident_bytes(self) -> None:
                self.external_resident_bytes = 0

        layer_index = 3
        batch = type(
            "Batch",
            (),
            {
                "nbytes": 80,
                "seconds": 0.02,
                "arrays": {},
                "weight_page_cache_bytes": 32,
                "transient_page_bytes": 30,
            },
        )()
        future = FakeFuture(batch)
        loader = FakeLoader()
        runner = object.__new__(DeepSeekV3StreamingForwardRunner)
        runner.session = FakeSession(loader)
        runner.resident_budget_bytes = 200
        runner._expert_cache_bytes = 0
        runner._pending_expert_prefetch_bytes = 120
        runner._pending_expert_prefetch = {
            "layer": layer_index,
            "experts": [2, 5],
            "future": future,
            "mode": "previous_table",
        }
        runner._prefetch_stats = ExpertPrefetchStats()
        runner._deepseek_slice_nbytes = lambda layer, experts: 40
        runner._deepseek_slice_page_miss_nbytes = lambda layer, experts: 10
        events: list[dict[str, object]] = []

        order = DeepSeekV3StreamingForwardRunner._consume_deepseek_expert_prefetch(
            runner,
            layer_index,
            object(),
            [5, 7],
            events=events,
            pass_kind="decode",
            token_step=4,
        )

        self.assertIsNone(order)
        self.assertEqual(future.result_calls, 1)
        self.assertEqual(loader.requests, [])
        self.assertEqual(runner._prefetch_stats.skipped_over_budget, 1)
        self.assertEqual(events[0]["action"], "prefetch-selected-expert-table-over-budget")
        self.assertEqual(events[0]["estimated_assembly_peak"], 280)
        self.assertEqual(runner.session.external_resident_bytes, 32)

    def test_prompt_lookup_draft_continues_repeated_ngrams(self) -> None:
        # ... 7 8 9 10 [5 6] ... [5 6] -> propose what followed last time: 7 8 9 10
        context = [1, 2, 3, 5, 6, 7, 8, 9, 10, 4, 5, 6]

        self.assertEqual(prompt_lookup_draft(context, 4), [7, 8, 9, 10])
        self.assertEqual(prompt_lookup_draft(context, 2), [7, 8])

    def test_prompt_lookup_draft_returns_empty_without_repeats(self) -> None:
        self.assertEqual(prompt_lookup_draft([1, 2, 3, 4, 5, 6], 4), [])
        self.assertEqual(prompt_lookup_draft([], 4), [])
        self.assertEqual(prompt_lookup_draft([1, 2], 0), [])

    def test_prompt_lookup_draft_prefers_longer_ngrams(self) -> None:
        # [2 3] occurs early followed by 9; [1 2 3] occurs later followed by 4.
        context = [2, 3, 9, 1, 2, 3, 4, 8, 1, 2, 3]

        self.assertEqual(prompt_lookup_draft(context, 1), [4])

    def test_count_accepted_drafts_stops_at_first_mismatch(self) -> None:
        self.assertEqual(count_accepted_drafts([5, 6, 7], [5, 6, 7]), 3)
        self.assertEqual(count_accepted_drafts([5, 6, 7], [5, 9, 7]), 1)
        self.assertEqual(count_accepted_drafts([5, 6, 7], [1, 6, 7]), 0)
        self.assertEqual(count_accepted_drafts([], []), 0)

    def test_count_accepted_drafts_rejects_near_tie_agreements(self) -> None:
        drafts = [5, 6, 7]
        greedy = [5, 6, 7]

        confident = count_accepted_drafts(drafts, greedy, gaps=[4.0, 3.0, 2.0], margin=0.5)
        near_tie = count_accepted_drafts(drafts, greedy, gaps=[4.0, 0.1, 2.0], margin=0.5)
        no_margin = count_accepted_drafts(drafts, greedy, gaps=[4.0, 0.1, 2.0], margin=0.0)

        self.assertEqual(confident, 3)
        self.assertEqual(near_tie, 1)
        self.assertEqual(no_margin, 3)

    def test_draft_gate_single_accepts_never_trigger(self) -> None:
        gate = AdaptiveDraftGate()
        for _ in range(20):
            self.assertTrue(gate.allow_draft())
            gate.observe(1)
        self.assertEqual(gate.level, 0)
        self.assertEqual(gate.triggers, 0)
        self.assertEqual(gate.gated_rounds, 0)

    def test_draft_gate_double_zero_triggers_mild_cooldown(self) -> None:
        gate = AdaptiveDraftGate()
        self.assertTrue(gate.allow_draft())
        gate.observe(0)
        self.assertTrue(gate.allow_draft())
        gate.observe(0)
        for _ in range(AdaptiveDraftGate.MILD_COOLDOWN):
            self.assertFalse(gate.allow_draft())
        self.assertTrue(gate.allow_draft())
        self.assertEqual(gate.level, 1)
        self.assertEqual(gate.triggers, 1)
        self.assertEqual(gate.gated_rounds, AdaptiveDraftGate.MILD_COOLDOWN)

    def test_draft_gate_escalates_to_hard_then_disabled(self) -> None:
        gate = AdaptiveDraftGate()

        def fail_through_streak() -> None:
            # Drain any active cooldown, then drive zero-accept rounds until
            # the gate triggers its next rung.
            triggers = gate.triggers
            while gate.triggers == triggers and not gate.disabled:
                if gate.allow_draft():
                    gate.observe(0)

        fail_through_streak()
        self.assertEqual(gate.level, 1)
        self.assertEqual(gate.cooldown_remaining, AdaptiveDraftGate.MILD_COOLDOWN)
        fail_through_streak()
        self.assertEqual(gate.level, 2)
        self.assertEqual(gate.cooldown_remaining, AdaptiveDraftGate.HARD_COOLDOWN)
        fail_through_streak()
        self.assertTrue(gate.disabled)
        for _ in range(100):
            self.assertFalse(gate.allow_draft())

    def test_draft_gate_acceptance_resets_streak_and_de_escalates(self) -> None:
        gate = AdaptiveDraftGate()
        gate.observe(0)
        gate.observe(3)
        gate.observe(0)
        self.assertEqual(gate.level, 0)
        self.assertEqual(gate.triggers, 0)

        gate.observe(0)
        self.assertEqual(gate.level, 1)
        while not gate.allow_draft():
            pass
        gate.observe(5)
        self.assertEqual(gate.level, 0)

    def test_draft_gate_telemetry_counts_zero_accept_rounds(self) -> None:
        gate = AdaptiveDraftGate()
        gate.observe(0)
        gate.observe(2)
        gate.observe(0)
        gate.observe(0)
        telemetry = gate.telemetry()
        self.assertEqual(telemetry["zero_accept_rounds"], 3)
        self.assertEqual(telemetry["gate_triggers"], 1)
        self.assertFalse(telemetry["gate_disabled"])

    def test_draft_gate_telemetry_counts_disabled_and_regression_avoided(self) -> None:
        gate = AdaptiveDraftGate()
        # Drive the gate to the disabled rung: 2+2+2 consecutive zero-accept
        # rounds trip mild, hard, then disabled.
        for _ in range(6):
            gate.observe(0)
        self.assertTrue(gate.disabled)
        # Once disabled, every allow_draft is a gated round and a disabled
        # round; gated rounds are the estimated-regression-avoided count.
        for _ in range(5):
            self.assertFalse(gate.allow_draft())
        telemetry = gate.telemetry()
        self.assertEqual(telemetry["disabled_rounds"], 5)
        self.assertGreaterEqual(telemetry["gated_rounds"], 5)
        self.assertEqual(
            telemetry["estimated_regression_avoided_passes"], telemetry["gated_rounds"]
        )

    def test_draft_gate_cooldown_rounds_are_gated_but_not_disabled(self) -> None:
        gate = AdaptiveDraftGate()
        gate.observe(0)
        gate.observe(0)  # trips mild cooldown
        self.assertFalse(gate.disabled)
        for _ in range(AdaptiveDraftGate.MILD_COOLDOWN):
            self.assertFalse(gate.allow_draft())
        telemetry = gate.telemetry()
        self.assertEqual(telemetry["gated_rounds"], AdaptiveDraftGate.MILD_COOLDOWN)
        self.assertEqual(telemetry["disabled_rounds"], 0)


class ExactnessModeSelectionTests(unittest.TestCase):
    def test_target_verified_is_default_and_honors_requested_cache(self) -> None:
        for model_type in ("gpt_oss", "qwen3_5_moe", "deepseek_v3", None):
            settings = select_exactness_mode(model_type, sliding_cache="kv")
            self.assertIsInstance(settings, ExactnessModeSettings)
            self.assertEqual(settings.mode, "target-verified")
            self.assertEqual(settings.sliding_cache, "kv")
            self.assertTrue(settings.speculation)

    def test_exact_strict_gpt_oss_forces_temporal_and_keeps_speculation(self) -> None:
        for requested in ("rotating", "kv", "temporal"):
            settings = select_exactness_mode(
                "gpt_oss", "exact-strict", sliding_cache=requested
            )
            self.assertEqual(settings.mode, "exact-strict")
            self.assertEqual(settings.sliding_cache, "temporal")
            self.assertTrue(settings.speculation)

    def test_exact_strict_gpt_oss_reason_notes_override_when_not_temporal(self) -> None:
        overridden = select_exactness_mode(
            "gpt_oss", "exact-strict", sliding_cache="rotating"
        )
        self.assertIn("overrode", overridden.reason)
        already = select_exactness_mode(
            "gpt_oss", "exact-strict", sliding_cache="temporal"
        )
        self.assertNotIn("overrode", already.reason)

    def test_exact_strict_qwen_disables_speculation(self) -> None:
        settings = select_exactness_mode("qwen3_5_moe", "exact-strict")
        self.assertEqual(settings.mode, "exact-strict")
        self.assertFalse(settings.speculation)

    def test_exact_strict_unknown_model_disables_speculation(self) -> None:
        settings = select_exactness_mode("some_future_moe", "exact-strict")
        self.assertFalse(settings.speculation)

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_exactness_mode("gpt_oss", "bitwise-exact")

    def test_settings_to_dict_round_trips_fields(self) -> None:
        payload = select_exactness_mode("gpt_oss", "exact-strict").to_dict()
        self.assertEqual(payload["mode"], "exact-strict")
        self.assertEqual(payload["sliding_cache"], "temporal")
        self.assertTrue(payload["speculation"])
        self.assertIn("reason", payload)


class TemporalSlidingKVCacheSeamTests(unittest.TestCase):
    """meta_state round-trip and trim() invariant on CPU (no model load)."""

    @staticmethod
    def _kv(offset: int, length: int):
        import mlx.core as mx

        base = mx.arange(offset, offset + length).reshape(1, 1, length, 1)
        base = mx.broadcast_to(base.astype(mx.float32), (1, 2, length, 4))
        return base, base

    @staticmethod
    def _invariant(cache: TemporalSlidingKVCache) -> bool:
        if cache.keys is None:
            return cache.offset == cache.start_position
        return cache.keys.shape[2] == cache.offset - cache.start_position

    def test_trim_slices_arrays_and_keeps_invariant(self) -> None:
        cache = TemporalSlidingKVCache(8)
        cache.update_and_fetch(*self._kv(0, 5))
        self.assertTrue(self._invariant(cache))
        trimmed = cache.trim(2)
        self.assertEqual(trimmed, 2)
        self.assertEqual(cache.offset, 3)
        self.assertEqual(cache.keys.shape[2], 3)
        self.assertTrue(self._invariant(cache))

    def test_trim_then_refeed_restores_correct_temporal_positions(self) -> None:
        cache = TemporalSlidingKVCache(8)
        cache.update_and_fetch(*self._kv(0, 5))
        cache.trim(2)
        keys, _ = cache.update_and_fetch(*self._kv(3, 2))
        positions = [int(x) for x in keys[0, 0, :, 0].tolist()]
        self.assertEqual(positions, [0, 1, 2, 3, 4])
        self.assertTrue(self._invariant(cache))

    def test_trim_more_than_content_empties_cache(self) -> None:
        cache = TemporalSlidingKVCache(8)
        cache.update_and_fetch(*self._kv(0, 3))
        trimmed = cache.trim(100)
        self.assertEqual(trimmed, 3)
        self.assertIsNone(cache.keys)
        self.assertEqual(cache.offset, 0)
        self.assertTrue(self._invariant(cache))

    def test_trim_zero_is_noop(self) -> None:
        cache = TemporalSlidingKVCache(8)
        cache.update_and_fetch(*self._kv(0, 4))
        self.assertEqual(cache.trim(0), 0)
        self.assertEqual(cache.offset, 4)
        self.assertEqual(cache.keys.shape[2], 4)

    def test_meta_state_round_trip_on_saturated_window(self) -> None:
        cache = TemporalSlidingKVCache(8)
        for i in range(12):
            cache.update_and_fetch(*self._kv(i, 1))
        self.assertEqual(cache.start_position, 4)
        state, meta = cache.state, cache.meta_state
        # Mirror the server restore order: state setter, then meta_state setter.
        restored = TemporalSlidingKVCache(0)
        restored.state = state
        restored.meta_state = meta
        self.assertEqual(restored.max_size, 8)
        self.assertEqual(restored.start_position, 4)
        self.assertEqual(restored.offset, 12)
        self.assertTrue(self._invariant(restored))
        # Restored cache continues exactly where it left off.
        keys, _ = restored.update_and_fetch(*self._kv(12, 1))
        positions = [int(x) for x in keys[0, 0, :, 0].tolist()]
        self.assertEqual(positions, [5, 6, 7, 8, 9, 10, 11, 12])

    def test_is_trimmable_true(self) -> None:
        self.assertTrue(TemporalSlidingKVCache(8).is_trimmable())

    def test_cache_transaction_rollback_restores_pre_state_exactly(self) -> None:
        class FakeCacheItem:
            def __init__(self) -> None:
                self.cache = [1, 2, 3]
                self.offset = 7

        item = FakeCacheItem()
        transaction = CacheTransaction([item])
        item.cache.append(4)
        item.offset = 99

        transaction.rollback()

        self.assertEqual(item.cache, [1, 2, 3])
        self.assertEqual(item.offset, 7)
        with self.assertRaises(RuntimeError):
            transaction.rollback()

    def test_cache_transaction_commit_keeps_mutations(self) -> None:
        class FakeCacheItem:
            def __init__(self) -> None:
                self.offset = 1

        item = FakeCacheItem()
        transaction = CacheTransaction([item])
        item.offset = 5

        transaction.commit()

        self.assertEqual(item.offset, 5)
        with self.assertRaises(RuntimeError):
            transaction.commit()

    def test_lz_drafter_prefers_longest_suffix_match(self) -> None:
        # Suffix [1, 2, 3] occurs twice: the later occurrence also matches the
        # preceding token (7), so its continuation (5, ...) must win over the
        # early occurrence's continuation (4, ...).
        context = [1, 2, 3, 4, 9, 9, 7, 1, 2, 3, 5, 8, 7, 1, 2, 3]
        drafter = PromptLookupDrafter()

        self.assertEqual(drafter.propose(context, 4), [5, 8, 7, 1])

    def test_lz_drafter_returns_empty_without_repeats(self) -> None:
        drafter = PromptLookupDrafter()

        self.assertEqual(drafter.propose([1, 2, 3, 4, 5, 6], 4), [])
        self.assertEqual(drafter.propose([], 4), [])

    def test_lz_drafter_shrinks_after_repeated_rejection(self) -> None:
        context = [1, 2, 3, 4, 9, 9, 7, 1, 2, 3, 5, 8, 7, 1, 2, 3]
        drafter = PromptLookupDrafter()
        drafter.observe_result(8, 0)
        drafter.observe_result(8, 0)

        shrunk = drafter.propose(context, 8)
        drafter.observe_result(len(shrunk), len(shrunk))
        drafter.observe_result(8, 8)
        recovered = drafter.propose(context, 8)

        self.assertEqual(len(shrunk), 2)
        self.assertGreater(len(recovered), len(shrunk))

    def test_lz_drafter_reports_match_telemetry(self) -> None:
        context = [1, 2, 3, 4, 9, 9, 7, 1, 2, 3, 5, 8, 7, 1, 2, 3]
        drafter = PromptLookupDrafter()
        drafter.propose(context, 4)
        drafter.propose([5, 6, 7, 8, 9, 10], 4)

        telemetry = drafter.telemetry()

        self.assertEqual(telemetry["matches_attempted"], 2)
        self.assertEqual(telemetry["matches_found"], 1)
        self.assertEqual(telemetry["average_match_length"], 4.0)

    def test_tree_drafter_returns_unique_full_length_branches(self) -> None:
        context = [
            7,
            1,
            2,
            3,
            4,
            8,
            1,
            2,
            3,
            5,
            9,
            1,
            2,
            3,
        ]
        drafter = PromptLookupTreeDrafter(ngram=3, max_candidates=8)

        branches = drafter.propose_branches(context, max_draft=2, max_branches=4)

        self.assertEqual(branches, [[5, 9], [4, 8]])

    def test_tree_drafter_respects_branch_cap(self) -> None:
        context = [1, 2, 9, 1, 2, 8, 1, 2, 7, 1, 2]
        drafter = PromptLookupTreeDrafter(ngram=2, max_candidates=8)

        branches = drafter.propose_branches(context, max_draft=1, max_branches=2)

        self.assertEqual(branches, [[7], [8]])

    def test_tree_drafter_finds_shorter_ngram_fallbacks(self) -> None:
        # The 3-gram suffix (6, 7, 8) never repeats, but the 2-gram (7, 8)
        # does — the fallback only works if shorter n-grams are indexed.
        context = [1, 7, 8, 5, 5, 2, 3, 6, 7, 8]
        drafter = PromptLookupTreeDrafter()

        branches = drafter.propose_branches(context, 1, 4)

        self.assertIn([5], branches)

    def test_tree_drafter_returns_multiple_distinct_branches(self) -> None:
        # (1, 2) repeats with two different continuations; both must surface.
        context = [1, 2, 9, 9, 3, 1, 2, 4, 4, 7, 1, 2]
        drafter = PromptLookupTreeDrafter()

        branches = drafter.propose_branches(context, 2, 8)

        self.assertGreaterEqual(len(branches), 2)
        self.assertIn([4, 4], branches)
        self.assertIn([9, 9], branches)

    def test_tree_drafter_keeps_partial_continuations(self) -> None:
        # The earlier (1, 2) match's continuation runs into the end of the
        # corpus, so it is shorter than max_draft. The old full-length-only
        # filter dropped it entirely, starving the verifier; it must surface.
        context = [1, 2, 5, 1, 2]
        drafter = PromptLookupTreeDrafter(ngram=2)

        branches = drafter.propose_branches(context, max_draft=4, max_branches=4)

        self.assertIn([5, 1, 2], branches)
        self.assertTrue(any(len(branch) < 4 for branch in branches))

    def test_tree_drafter_session_seed_mines_prior_output(self) -> None:
        # The current request's context alone has no repeat of (1, 2, 3), but a
        # prior request's output did — seeding the session index must make that
        # historical continuation a candidate branch.
        drafter = PromptLookupTreeDrafter(ngram=3)
        self.assertEqual(drafter.propose_branches([1, 2, 3], 2, 4), [])

        drafter.seed_session([9, 1, 2, 3, 8, 8, 0])
        branches = drafter.propose_branches([1, 2, 3], 2, 4)

        self.assertIn([8, 8], branches)

    def test_drafter_session_seed_persists_across_reset(self) -> None:
        # reset() runs at the top of every generate; the session prefix must
        # survive it so the index spans the whole serve session.
        drafter = PromptLookupTreeDrafter(ngram=3)
        drafter.seed_session([9, 1, 2, 3, 8, 8, 0])
        drafter.reset()

        branches = drafter.propose_branches([1, 2, 3], 2, 4)
        self.assertIn([8, 8], branches)

        drafter.reset_session()
        self.assertEqual(drafter.propose_branches([1, 2, 3], 2, 4), [])

    def test_drafter_session_seed_does_not_change_plain_propose(self) -> None:
        # propose() must also see the session corpus (single-branch path).
        drafter = PromptLookupDrafter(ngram=3)
        drafter.seed_session([5, 1, 2, 3, 6, 7, 8])

        self.assertEqual(drafter.propose([1, 2, 3], 3), [6, 7, 8])

    def test_external_process_drafter_protocol(self) -> None:
        helper = """
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    kind = request.get("type")
    if kind == "propose":
        context = request.get("context") or []
        max_draft = int(request.get("max_draft") or 0)
        response = {"tokens": list(reversed(context[-max_draft:]))}
    elif kind == "telemetry":
        response = {"telemetry": {"helper": "fake"}}
    else:
        response = {"ok": True}
    print(json.dumps(response), flush=True)
    if kind == "close":
        break
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "fake_drafter.py"
            script.write_text(helper, encoding="utf-8")
            drafter = ExternalProcessDrafter([sys.executable, str(script)])
            try:
                drafter.reset()
                drafter.seed_session([1, 2, 3])
                self.assertEqual(drafter.propose([10, 20, 30], 2), [30, 20])
                drafter.observe_result(2, 1)

                telemetry = drafter.telemetry()
                self.assertEqual(telemetry["requests"], 1)
                self.assertEqual(telemetry["tokens_proposed"], 2)
                self.assertEqual(telemetry["helper"], "fake")
            finally:
                drafter.close()

    def test_pass_economics_is_none_without_speculation(self) -> None:
        self.assertIsNone(pass_economics(None))
        self.assertIsNone(pass_economics({}))

    def test_pass_economics_surfaces_tree_block(self) -> None:
        block = pass_economics(
            {
                "drafter": "prompt-lookup-tree",
                "accepted_tokens_per_pass": 1.8,
                "acceptance_rate": 0.4,
                "streamed_passes": 40,
                "branch_rounds": 12,
                "branches_verified": 96,
                "average_branches": 8.0,
                "zero_accept_rounds": 3,
                "gate_disabled": False,
            }
        )

        self.assertEqual(block["drafter"], "prompt-lookup-tree")
        self.assertEqual(block["accepted_tokens_per_pass"], 1.8)
        self.assertEqual(block["branches_verified"], 96)
        self.assertEqual(block["zero_accept_rounds"], 3)

    def test_pass_economics_tolerates_single_branch_shape(self) -> None:
        # generate_speculative emits SpeculativeStats (no branch keys).
        block = pass_economics(
            {
                "drafter": "prompt-lookup",
                "accepted_tokens_per_pass": 2.1,
                "acceptance_rate": 0.5,
                "streamed_passes": 20,
            }
        )

        self.assertEqual(block["average_branches"], 0.0)
        self.assertEqual(block["branch_rounds"], 0)
        self.assertEqual(block["zero_accept_rounds"], 0)

    def test_longest_reusable_prefix_requires_strict_prefix(self) -> None:
        self.assertEqual(longest_reusable_prefix([1, 2, 3], [1, 2, 3, 4, 5]), 3)
        self.assertEqual(longest_reusable_prefix([], [1, 2]), 0)
        self.assertEqual(longest_reusable_prefix([1, 2, 3], [1, 2, 3]), 0)
        self.assertEqual(longest_reusable_prefix([1, 9], [1, 2, 3]), 0)
        self.assertEqual(longest_reusable_prefix([1, 2, 3, 4], [1, 2]), 0)

    def test_strip_trailing_stops_removes_only_trailing_stop_tokens(self) -> None:
        self.assertEqual(strip_trailing_stops([5, 6, 7, 0, 0], {0}), [5, 6, 7])
        self.assertEqual(strip_trailing_stops([0, 5, 0, 6], {0}), [0, 5, 0, 6])
        self.assertEqual(strip_trailing_stops([], {0}), [])

    def test_parse_harmony_output_extracts_final_and_reasoning(self) -> None:
        text = (
            "<|channel|>analysis<|message|>thinking about it<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello there, friend.<|return|>"
        )

        content, reasoning = parse_harmony_output(text)

        self.assertEqual(content, "Hello there, friend.")
        self.assertEqual(reasoning, "thinking about it")

    def test_parse_harmony_output_passes_plain_text_through(self) -> None:
        self.assertEqual(parse_harmony_output("just a reply"), ("just a reply", None))

    def test_parse_harmony_output_handles_truncated_analysis(self) -> None:
        content, reasoning = parse_harmony_output(
            "<|channel|>analysis<|message|>still thinking"
        )

        self.assertEqual(content, "")
        self.assertEqual(reasoning, "still thinking")

    def test_streaming_visible_text_suppresses_analysis(self) -> None:
        self.assertEqual(streaming_visible_text("plain"), "plain")
        self.assertEqual(
            streaming_visible_text("<|channel|>analysis<|message|>thinking"), ""
        )
        self.assertEqual(
            streaming_visible_text(
                "<|channel|>analysis<|message|>x<|end|><|start|>assistant"
                "<|channel|>final<|message|>Hel"
            ),
            "Hel",
        )

    def test_first_dim_slices_preserve_requested_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_toy_safetensors(Path(directory) / "toy.safetensors")
            loader = MlxSelectiveLoader.from_model_dir(path.parent)
            name = "model.layers.0.self_attn.q_proj.weight"

            try:
                batch = loader.load_first_dim_slices((name,), [1, 0])
            finally:
                loader.close()

            full = np.frombuffer(bytes(range(16, 32)), dtype="<f4").reshape(2, 2)
            np.testing.assert_array_equal(np.array(batch.arrays[name]), full[[1, 0]])

    def test_first_dim_slices_preserve_duplicate_requested_rows(self) -> None:
        import mlx.core as mx

        def raw_bytes(array: object, dtype: str) -> bytes:
            if dtype == "BF16":
                return np.array(array.view(mx.uint16)).tobytes()
            return np.array(array).tobytes()

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            manifest = SmartTensorManifest.from_safetensors([src])
            names = tuple(
                sorted(
                    name
                    for name in manifest.tensors
                    if ".mlp.experts." in name
                )
            )
            loader = MlxSelectiveLoader(manifest)
            paged_loader = MlxSelectiveLoader(manifest)
            paged_loader.attach_weight_page_cache(
                2048,
                eviction_policy="frequency",
                rows_per_page=2,
            )

            try:
                direct = loader.load_first_dim_slices(names, [2, 2, 1])
                paged = paged_loader.load_first_dim_slices(names, [2, 2, 1])
            finally:
                loader.close()
                paged_loader.close()

            expected_weight = np.arange(5 * 4 * 3, dtype="<u4").reshape(5, 4, 3)[[2, 2, 1]]
            expected_scales = (np.arange(5 * 6, dtype="<u2") + 7).reshape(5, 6)[[2, 2, 1]]
            expected_by_name = {
                "model.layers.0.mlp.experts.gate_proj.weight": expected_weight.tobytes(),
                "model.layers.0.mlp.experts.gate_proj.scales": expected_scales.tobytes(),
            }

            for name in names:
                dtype = manifest.tensors[name].dtype
                self.assertEqual(raw_bytes(direct.arrays[name], dtype), expected_by_name[name])
                self.assertEqual(raw_bytes(paged.arrays[name], dtype), expected_by_name[name])
            self.assertEqual(direct.transient_page_bytes, 0)
            self.assertEqual(paged.transient_page_bytes, 240)

    def test_weight_page_cache_reuses_first_dim_rows(self) -> None:
        import mlx.core as mx

        def raw_bytes(array: object, dtype: str) -> bytes:
            if dtype == "BF16":
                return np.array(array.view(mx.uint16)).tobytes()
            return np.array(array).tobytes()

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            manifest = SmartTensorManifest.from_safetensors([src])
            names = tuple(
                sorted(
                    name
                    for name in manifest.tensors
                    if ".mlp.experts." in name
                )
            )
            baseline = MlxSelectiveLoader(manifest)
            loader = MlxSelectiveLoader(manifest)
            loader.attach_weight_page_cache(1024, eviction_policy="frequency")

            try:
                expected = baseline.load_first_dim_slices(names, [2])
                first = loader.load_first_dim_slices(names, [2])
                second = loader.load_first_dim_slices(names, [2])
                summary = loader.weight_page_summary()
            finally:
                baseline.close()
                loader.close()

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(summary["misses"], len(names))
            self.assertEqual(summary["hits"], len(names))
            self.assertEqual(summary["eviction_policy"], "frequency")
            self.assertEqual(summary["bytes_read"], expected.nbytes)
            self.assertEqual(summary["resident_count"], len(names))
            self.assertEqual(first.weight_page_cache_bytes, expected.nbytes)
            self.assertEqual(second.weight_page_cache_bytes, expected.nbytes)
            self.assertEqual(first.transient_page_bytes, expected.nbytes)
            self.assertEqual(second.transient_page_bytes, 0)
            for name in names:
                dtype = manifest.tensors[name].dtype
                self.assertEqual(
                    raw_bytes(first.arrays[name], dtype),
                    raw_bytes(expected.arrays[name], dtype),
                    msg=name,
                )
                self.assertEqual(
                    raw_bytes(second.arrays[name], dtype),
                    raw_bytes(expected.arrays[name], dtype),
                    msg=name,
                )

    def test_first_dim_slice_load_can_bypass_weight_page_cache(self) -> None:
        import mlx.core as mx

        def raw_bytes(array: object, dtype: str) -> bytes:
            if dtype == "BF16":
                return np.array(array.view(mx.uint16)).tobytes()
            return np.array(array).tobytes()

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            manifest = SmartTensorManifest.from_safetensors([src])
            names = tuple(
                sorted(
                    name
                    for name in manifest.tensors
                    if ".mlp.experts." in name
                )
            )
            baseline = MlxSelectiveLoader(manifest)
            loader = MlxSelectiveLoader(manifest)
            loader.attach_weight_page_cache(1024, eviction_policy="frequency")

            try:
                expected = baseline.load_first_dim_slices(names, [2])
                loaded = loader.load_first_dim_slices(
                    names,
                    [2],
                    use_weight_page_cache=False,
                )
                summary = loader.weight_page_summary()
            finally:
                baseline.close()
                loader.close()

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(summary["hits"], 0)
            self.assertEqual(summary["misses"], 0)
            self.assertEqual(summary["resident_bytes"], 0)
            self.assertEqual(loaded.weight_page_cache_bytes, 0)
            self.assertEqual(loaded.transient_page_bytes, 0)
            for name in names:
                dtype = manifest.tensors[name].dtype
                self.assertEqual(
                    raw_bytes(loaded.arrays[name], dtype),
                    raw_bytes(expected.arrays[name], dtype),
                    msg=name,
                )

    def test_weight_page_warm_first_dim_rows_without_assembling_tables(self) -> None:
        import mlx.core as mx

        def raw_bytes(array: object, dtype: str) -> bytes:
            if dtype == "BF16":
                return np.array(array.view(mx.uint16)).tobytes()
            return np.array(array).tobytes()

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            manifest = SmartTensorManifest.from_safetensors([src])
            names = tuple(
                sorted(
                    name
                    for name in manifest.tensors
                    if ".mlp.experts." in name
                )
            )
            baseline = MlxSelectiveLoader(manifest)
            loader = MlxSelectiveLoader(manifest)
            loader.attach_weight_page_cache(1024, eviction_policy="two_queue")

            try:
                expected = baseline.load_first_dim_slices(names, [1])
                warm = loader.warm_first_dim_slices(names, [1])
                loaded = loader.load_first_dim_slices(names, [1])
                summary = loader.weight_page_summary()
            finally:
                baseline.close()
                loader.close()

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(warm.arrays, {})
            self.assertEqual(warm.nbytes, expected.nbytes)
            self.assertEqual(warm.weight_page_cache_bytes, expected.nbytes)
            self.assertEqual(loaded.weight_page_cache_bytes, expected.nbytes)
            self.assertEqual(warm.transient_page_bytes, expected.nbytes)
            self.assertEqual(loaded.transient_page_bytes, 0)
            self.assertEqual(summary["misses"], len(names))
            self.assertEqual(summary["hits"], len(names))
            self.assertEqual(summary["eviction_policy"], "two_queue")
            self.assertEqual(summary["bytes_read"], expected.nbytes)
            for name in names:
                dtype = manifest.tensors[name].dtype
                self.assertEqual(
                    raw_bytes(loaded.arrays[name], dtype),
                    raw_bytes(expected.arrays[name], dtype),
                    msg=name,
                )

    def test_weight_page_miss_estimator_can_report_uncapped_transient_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            manifest = SmartTensorManifest.from_safetensors([src])
            names = tuple(
                sorted(
                    name
                    for name in manifest.tensors
                    if ".mlp.experts." in name
                )
            )
            first_row_bytes = sum(
                manifest.tensors[name].nbytes // manifest.tensors[name].shape[0]
                for name in names
            )
            loader = MlxSelectiveLoader(manifest)
            loader.attach_weight_page_cache(first_row_bytes, eviction_policy="lru")

            try:
                loader.load_first_dim_slices(names, [0])
                capped = loader.estimate_first_dim_slice_page_miss_bytes(names, [1])
                uncapped = loader.estimate_first_dim_slice_page_miss_bytes(
                    names,
                    [1],
                    cap_to_headroom=False,
                )
            finally:
                loader.close()

            self.assertEqual(capped, 0)
            self.assertEqual(uncapped, first_row_bytes)

    def test_weight_page_multi_row_pages_preserve_requested_order(self) -> None:
        import mlx.core as mx

        def raw_bytes(array: object, dtype: str) -> bytes:
            if dtype == "BF16":
                return np.array(array.view(mx.uint16)).tobytes()
            return np.array(array).tobytes()

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            manifest = SmartTensorManifest.from_safetensors([src])
            names = tuple(
                sorted(
                    name
                    for name in manifest.tensors
                    if ".mlp.experts." in name
                )
            )
            baseline = MlxSelectiveLoader(manifest)
            loader = MlxSelectiveLoader(manifest)
            loader.attach_weight_page_cache(
                2048,
                eviction_policy="frequency",
                rows_per_page=2,
            )

            try:
                estimated_page_bytes = loader.estimate_first_dim_slice_page_miss_bytes(
                    names,
                    [3, 1, 2],
                )
                expected = baseline.load_first_dim_slices(names, [3, 1, 2])
                loaded = loader.load_first_dim_slices(names, [3, 1, 2])
                summary = loader.weight_page_summary()
            finally:
                baseline.close()
                loader.close()

            self.assertIsNotNone(summary)
            assert summary is not None
            expected_page_bytes = sum(
                manifest.tensors[name].nbytes * 4 // manifest.tensors[name].shape[0]
                for name in names
            )
            self.assertEqual(summary["rows_per_page"], 2)
            self.assertEqual(summary["resident_count"], len(names) * 2)
            self.assertEqual(summary["misses"], len(names) * 2)
            self.assertEqual(summary["bytes_read"], expected_page_bytes)
            self.assertEqual(estimated_page_bytes, expected_page_bytes)
            self.assertEqual(loaded.nbytes, expected.nbytes)
            self.assertEqual(loaded.weight_page_cache_bytes, expected_page_bytes)
            self.assertEqual(loaded.transient_page_bytes, expected_page_bytes)
            for name in names:
                dtype = manifest.tensors[name].dtype
                self.assertEqual(
                    raw_bytes(loaded.arrays[name], dtype),
                    raw_bytes(expected.arrays[name], dtype),
                    msg=name,
                )

    def test_weight_page_empty_first_dim_request_does_not_touch_cache(self) -> None:
        import mlx.core as mx

        def raw_bytes(array: object, dtype: str) -> bytes:
            if dtype == "BF16":
                return np.array(array.view(mx.uint16)).tobytes()
            return np.array(array).tobytes()

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            manifest = SmartTensorManifest.from_safetensors([src])
            names = tuple(
                sorted(
                    name
                    for name in manifest.tensors
                    if ".mlp.experts." in name
                )
            )
            baseline = MlxSelectiveLoader(manifest)
            loader = MlxSelectiveLoader(manifest)
            loader.attach_weight_page_cache(
                2048,
                eviction_policy="frequency",
                rows_per_page=2,
            )

            try:
                expected = baseline.load_first_dim_slices(names, [])
                loaded = loader.load_first_dim_slices(names, [])
                summary = loader.weight_page_summary()
            finally:
                baseline.close()
                loader.close()

            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(loaded.nbytes, 0)
            self.assertEqual(loaded.weight_page_cache_bytes, 0)
            self.assertEqual(loaded.transient_page_bytes, 0)
            self.assertEqual(summary["resident_count"], 0)
            self.assertEqual(summary["misses"], 0)
            self.assertEqual(summary["bytes_read"], 0)
            for name in names:
                dtype = manifest.tensors[name].dtype
                self.assertEqual(loaded.arrays[name].shape[0], 0)
                self.assertEqual(
                    raw_bytes(loaded.arrays[name], dtype),
                    raw_bytes(expected.arrays[name], dtype),
                    msg=name,
                )


class ExpertPackStoreTests(unittest.TestCase):
    """Byte-equivalence + layout/telemetry checks for the contiguous pack store.

    All CPU/numpy; no MLX, no GPU. The pack must reproduce the exact bytes the
    safetensors first-dim gather would, with experts laid out contiguously so a
    selected id (or contiguous run) extracts as a single basic slice.
    """

    def _gather_fix_module(self):
        from smarttensor.adapters.mlx import (
            contiguous_runs,
            gather_first_dim_rows,
        )

        return contiguous_runs, gather_first_dim_rows

    def test_pack_round_trips_expert_bytes_identically(self) -> None:
        from smarttensor.packstore import ExpertPackReader, ExpertPackWriter

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            with SafeTensorFile(src) as safe_file:
                names = ExpertPackWriter.expert_tensor_names(safe_file)
                self.assertEqual(
                    names,
                    [
                        "model.layers.0.mlp.experts.gate_proj.scales",
                        "model.layers.0.mlp.experts.gate_proj.weight",
                    ],
                )
                pack_path = Path(directory) / "experts.pack"
                ExpertPackWriter.write(safe_file, pack_path, names)

                # Ground truth, computed directly from the safetensors path.
                truth: dict[str, np.ndarray] = {}
                for name in names:
                    meta = safe_file.tensors[name]
                    with safe_file.tensor(name) as slice_obj:
                        full = np.frombuffer(
                            slice_obj.copy(), dtype=_np_dtype(meta.dtype)
                        ).reshape(meta.shape)
                    truth[name] = full

                with ExpertPackReader(pack_path) as reader:
                    for ids in ([0], [2, 4], [1, 2, 3]):
                        got = reader.load_expert_union(names, ids, layer=0)
                        for name in names:
                            expected = truth[name][sorted(set(ids))]
                            self.assertEqual(
                                got[name].shape,
                                expected.shape,
                                msg=f"{name} ids={ids}",
                            )
                            self.assertEqual(
                                got[name].tobytes(),
                                expected.tobytes(),
                                msg=f"{name} ids={ids}",
                            )

    def test_pack_records_are_16kb_page_aligned(self) -> None:
        from smarttensor.packstore import PAGE_SIZE, ExpertPackReader, ExpertPackWriter

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            with SafeTensorFile(src) as safe_file:
                names = ExpertPackWriter.expert_tensor_names(safe_file)
                pack_path = Path(directory) / "experts.pack"
                records = ExpertPackWriter.write(safe_file, pack_path, names)
                for record in records.values():
                    self.assertEqual(record.data_offset % PAGE_SIZE, 0)
                    self.assertEqual(record.expert_stride % PAGE_SIZE, 0)
                    self.assertGreaterEqual(record.expert_stride, record.expert_nbytes)
                with ExpertPackReader(pack_path) as reader:
                    for name, record in records.items():
                        for expert_id in range(record.expert_count):
                            self.assertEqual(
                                reader.records[name].expert_offset(expert_id)
                                % PAGE_SIZE,
                                0,
                            )

    def test_contiguous_run_resolves_to_a_single_basic_slice(self) -> None:
        from smarttensor.packstore import ExpertPackReader, ExpertPackWriter

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            with SafeTensorFile(src) as safe_file:
                names = ExpertPackWriter.expert_tensor_names(safe_file)
                pack_path = Path(directory) / "experts.pack"
                ExpertPackWriter.write(safe_file, pack_path, names)
                with ExpertPackReader(pack_path) as reader:
                    # Adjacent ids -> one contiguous range per tensor.
                    reader.load_expert_union(names, [1, 2, 3], layer=0)
                    call = reader.telemetry.ranges_by_call[-1]
                    self.assertEqual(call["ranges"], len(names))
                    # Scattered ids -> a range per island per tensor.
                    reader.load_expert_union(names, [0, 2, 4], layer=0)
                    call = reader.telemetry.ranges_by_call[-1]
                    self.assertEqual(call["ranges"], 3 * len(names))

    def test_pack_telemetry_tracks_waste_and_useful_bytes(self) -> None:
        from smarttensor.packstore import ExpertPackReader, ExpertPackWriter

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            with SafeTensorFile(src) as safe_file:
                names = ExpertPackWriter.expert_tensor_names(safe_file)
                pack_path = Path(directory) / "experts.pack"
                ExpertPackWriter.write(safe_file, pack_path, names)
                with ExpertPackReader(pack_path) as reader:
                    reader.load_expert_union(names, [1, 2], layer=0)
                    summary = reader.telemetry.to_dict()
                    self.assertEqual(summary["union_calls"], 1)
                    self.assertGreater(summary["read_bytes"], 0)
                    self.assertGreater(summary["useful_bytes"], 0)
                    # 16KB padding means read >= useful, so waste in [0, 1).
                    self.assertGreaterEqual(summary["read_bytes"], summary["useful_bytes"])
                    self.assertGreaterEqual(summary["waste_ratio"], 0.0)
                    self.assertLess(summary["waste_ratio"], 1.0)

    def test_gather_first_dim_rows_is_bitwise_identical_to_fancy_index(self) -> None:
        _, gather_first_dim_rows = self._gather_fix_module()
        arr = np.arange(8 * 3 * 2, dtype="<u4").reshape(8, 3, 2)
        for ids in ([0], [3], [4, 5, 6, 7], [1, 4, 6], [0, 7]):
            got = gather_first_dim_rows(arr, tuple(ids))
            expected = arr[list(ids)]
            self.assertEqual(got.shape, expected.shape, msg=str(ids))
            self.assertEqual(got.tobytes(), expected.tobytes(), msg=str(ids))

    def test_gather_first_dim_rows_falls_back_for_non_ascending_ids(self) -> None:
        _, gather_first_dim_rows = self._gather_fix_module()
        arr = np.arange(5 * 2, dtype="<u4").reshape(5, 2)
        # Descending / repeated ids must preserve the requested order exactly,
        # matching np_array[list(indices)] (the fancy-index fallback).
        for ids in ([3, 1], [2, 2], [4, 0, 4]):
            got = gather_first_dim_rows(arr, tuple(ids))
            expected = arr[list(ids)]
            self.assertEqual(got.tobytes(), expected.tobytes(), msg=str(ids))

    def test_contiguous_runs_merges_adjacent_ascending_ids(self) -> None:
        contiguous_runs, _ = self._gather_fix_module()
        self.assertEqual(contiguous_runs((4,)), [(4, 5)])
        self.assertEqual(contiguous_runs((1, 2, 3)), [(1, 4)])
        self.assertEqual(contiguous_runs((0, 2, 4)), [(0, 1), (2, 3), (4, 5)])
        self.assertEqual(contiguous_runs((3, 4, 7, 8)), [(3, 5), (7, 9)])

    def test_loader_first_dim_slices_match_pack_union(self) -> None:
        """The runtime loader and the pack reader return identical bytes."""

        from smarttensor.packstore import ExpertPackReader, ExpertPackWriter

        with tempfile.TemporaryDirectory() as directory:
            src = write_expert_safetensors(Path(directory) / "src.safetensors")
            with SafeTensorFile(src) as safe_file:
                names = ExpertPackWriter.expert_tensor_names(safe_file)
                pack_path = Path(directory) / "experts.pack"
                ExpertPackWriter.write(safe_file, pack_path, names)
                truth = {}
                for name in names:
                    meta = safe_file.tensors[name]
                    with safe_file.tensor(name) as slice_obj:
                        truth[name] = np.frombuffer(
                            slice_obj.copy(), dtype=_np_dtype(meta.dtype)
                        ).reshape(meta.shape)
                with ExpertPackReader(pack_path) as reader:
                    ids = [1, 2, 4]
                    union = reader.load_expert_union(names, ids, layer=0)
                    for name in names:
                        expected = truth[name][sorted(ids)]
                        self.assertEqual(union[name].tobytes(), expected.tobytes())


def _np_dtype(dtype: str) -> np.dtype:
    return {
        "U32": np.dtype("<u4"),
        "BF16": np.dtype("<u2"),
        "F32": np.dtype("<f4"),
    }[dtype]


def write_expert_safetensors(path: Path) -> Path:
    """A synthetic shard with two leading-expert-axis MoE tensors (E=5)."""

    experts = 5
    weight = np.arange(experts * 4 * 3, dtype="<u4").reshape(experts, 4, 3)
    scales = (np.arange(experts * 6, dtype="<u2") + 7).reshape(experts, 6)
    tensors = {
        "model.layers.0.mlp.experts.gate_proj.weight": (
            "U32",
            list(weight.shape),
            weight.tobytes(),
        ),
        "model.layers.0.mlp.experts.gate_proj.scales": (
            "BF16",
            list(scales.shape),
            scales.tobytes(),
        ),
        # A non-expert tensor that must be ignored by the writer's selector.
        "model.layers.0.self_attn.q_proj.weight": (
            "F32",
            [2, 2],
            np.arange(4, dtype="<f4").tobytes(),
        ),
    }

    offset = 0
    payload = bytearray()
    header: dict[str, object] = {"__metadata__": {"name": "expert-synth"}}
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.extend(data)

    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    return path


def write_toy_safetensors(path: Path) -> Path:
    tensors = {
        "model.embed_tokens.weight": ("F32", [2, 2], bytes(range(0, 16))),
        "model.layers.0.self_attn.q_proj.weight": ("F32", [2, 2], bytes(range(16, 32))),
        "model.layers.0.mlp.up_proj.weight": ("F32", [2, 2], bytes(range(32, 48))),
        "lm_head.weight": ("F32", [2, 2], bytes(range(48, 64))),
    }

    offset = 0
    payload = bytearray()
    header = {"__metadata__": {"name": "toy"}}
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.extend(data)

    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
    return path


def sized_manifest(
    *,
    pin_small_bytes: int,
    embedding_bytes: int,
    output_bytes: int,
    layer_bytes: tuple[int, ...],
) -> SmartTensorManifest:
    tensors: dict[str, TensorRecord] = {}

    def add_tensor(
        name: str,
        nbytes: int,
        *,
        layer: int | None = None,
        role: str | None = None,
        residency_hint: str = "stream",
    ) -> None:
        tensors[name] = TensorRecord(
            name=name,
            file="synthetic.safetensors",
            dtype="U8",
            shape=(nbytes,),
            data_offsets=(0, nbytes),
            absolute_offsets=(0, nbytes),
            nbytes=nbytes,
            layer=layer,
            role=role,
            residency_hint=residency_hint,
        )

    add_tensor("model.norm.weight", pin_small_bytes, role="norm", residency_hint="pin-small")
    add_tensor("model.embed_tokens.weight", embedding_bytes, role="embedding", residency_hint="pin")
    add_tensor("lm_head.weight", output_bytes, role="output", residency_hint="pin")

    layers: dict[int, LayerRecord] = {}
    for index, nbytes in enumerate(layer_bytes):
        name = f"model.layers.{index}.mlp.weight"
        add_tensor(name, nbytes, layer=index, role="mlp")
        layers[index] = LayerRecord(
            index=index,
            tensor_names=(name,),
            nbytes=nbytes,
            dtypes=("U8",),
        )

    return SmartTensorManifest(
        format="synthetic",
        files=("synthetic.safetensors",),
        tensors=tensors,
        layers=layers,
    )


if __name__ == "__main__":
    unittest.main()
