from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import nemotron_qmm_table_probe as probe


class NemotronQmmTableProbeTests(unittest.TestCase):
    def test_build_cases_estimates_table_bytes_for_two_quantized_projections(self) -> None:
        cases = probe.build_cases(
            table_rows=(22, 88),
            selected_rows=22,
            input_dims=1024,
            hidden_dims=2688,
            group_size=64,
            bits=4,
        )

        self.assertEqual([case.table_rows for case in cases], [22, 88])
        self.assertEqual(cases[0].selected_rows, 22)
        self.assertLess(cases[0].table_bytes, cases[1].table_bytes)
        self.assertEqual(cases[1].table_bytes, cases[0].table_bytes * 4)

    def test_summarize_results_reports_slowest_relative_to_compact(self) -> None:
        result = probe.summarize_results(
            [
                {"table_rows": 22, "median_ms": 1.0, "table_bytes": 100},
                {"table_rows": 88, "median_ms": 1.7, "table_bytes": 400},
            ]
        )

        self.assertEqual(result["baseline_table_rows"], 22)
        self.assertEqual(result["slowest_table_rows"], 88)
        self.assertAlmostEqual(result["slowest_vs_baseline"], 1.7)

    def test_summarize_component_results_reports_slowest_component(self) -> None:
        result = probe.summarize_component_results(
            [
                {"component": "routed_qmm", "median_ms": 1.2},
                {"component": "shared_expert_mlp", "median_ms": 2.4},
            ]
        )

        self.assertEqual(result["slowest_component"], "shared_expert_mlp")
        self.assertAlmostEqual(result["slowest_median_ms"], 2.4)
        self.assertIn("routed_qmm", result["results"])

    def test_component_probe_requires_single_table_row_value(self) -> None:
        with self.assertRaises(ValueError):
            probe.main(["--component-probe", "--table-rows", "22,88"])

    def test_parse_side_table_dtype_rejects_unknown_dtype(self) -> None:
        with self.assertRaises(Exception):
            probe._parse_side_table_dtype("int8")

    def test_summarize_first_use_results_reports_first_vs_repeat(self) -> None:
        result = probe.summarize_first_use_results(
            [
                {"first_ms": 10.0, "repeat_ms": 2.0},
                {"first_ms": 6.0, "repeat_ms": 3.0},
            ]
        )

        self.assertEqual(result["block_count"], 2)
        self.assertAlmostEqual(result["first_median_ms"], 8.0)
        self.assertAlmostEqual(result["repeat_median_ms"], 2.5)
        self.assertAlmostEqual(result["first_vs_repeat"], 3.2)

    def test_nemotron_switch_tensor_names_match_manifest_naming(self) -> None:
        self.assertEqual(
            probe.nemotron_switch_tensor_names(19),
            (
                "backbone.layers.19.mixer.switch_mlp.fc1.biases",
                "backbone.layers.19.mixer.switch_mlp.fc1.scales",
                "backbone.layers.19.mixer.switch_mlp.fc1.weight",
                "backbone.layers.19.mixer.switch_mlp.fc2.biases",
                "backbone.layers.19.mixer.switch_mlp.fc2.scales",
                "backbone.layers.19.mixer.switch_mlp.fc2.weight",
            ),
        )

    def test_local_indices_for_table_preserve_selected_order(self) -> None:
        self.assertEqual(
            probe.local_indices_for_table(
                selected_experts=[30, 10, 40],
                table_order=[10, 20, 30, 40],
            ),
            [2, 0, 3],
        )

    def test_local_indices_for_table_raise_on_missing_expert(self) -> None:
        with self.assertRaises(ValueError):
            probe.local_indices_for_table(
                selected_experts=[30, 99],
                table_order=[10, 20, 30, 40],
            )

    def test_parse_byte_budget_accepts_decimal_and_binary_units(self) -> None:
        self.assertEqual(probe.parse_byte_budget("300MB"), 300_000_000)
        self.assertEqual(probe.parse_byte_budget("2MiB"), 2 * 1024 * 1024)

    def test_enforce_real_weight_probe_byte_ceiling_rejects_large_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "real-weight table bytes"):
            probe.enforce_real_weight_probe_byte_ceiling(
                table_bytes=301,
                max_table_bytes=300,
            )

    def test_real_weight_probe_dry_run_reports_without_loading_weights(self) -> None:
        result = probe.real_weight_probe_dry_run(
            model_dir="/models/nemotron",
            layer_index=19,
            table_experts=[1, 2, 3],
            selected_experts=[1, 3],
            table_bytes=9_000,
        )

        self.assertEqual(result["mode"], "real-weight-probe-dry-run")
        self.assertEqual(result["table_rows"], 3)
        self.assertEqual(result["selected_rows"], 2)
        self.assertEqual(result["table_bytes"], 9_000)


if __name__ == "__main__":
    unittest.main()
