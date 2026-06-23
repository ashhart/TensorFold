"""Tests for smarttensor.nemotron_layout.plan_blocks.

Ground truth is mlx_lm's NemotronHModel: each test asserts that plan_blocks
agrees with a freshly-instantiated real model's backbone.fa_idx/ssm_idx,
len(make_cache()), and per-layer cache presence. The real model wins; if
plan_blocks disagrees, plan_blocks is the bug.
"""
from __future__ import annotations

import unittest

from smarttensor.nemotron_layout import plan_blocks


def _real_model(pattern: list[str]):
    """Instantiate the authoritative mlx_lm NemotronH model for a pattern."""
    from mlx_lm.models.nemotron_h import Model, ModelArgs
    from tests.fixtures.tiny_nemotron import tiny_config

    cfg = dict(tiny_config())
    cfg["layers_block_type"] = list(pattern)
    cfg["num_hidden_layers"] = len(pattern)
    return Model(ModelArgs.from_dict(cfg))


class PlanBlocksTests(unittest.TestCase):
    def _assert_agrees_with_real_model(self, pattern: list[str]) -> None:
        plan = plan_blocks(pattern)
        model = _real_model(pattern)
        backbone = model.backbone

        # fa_idx / ssm_idx must match the real model's scan exactly.
        self.assertEqual(plan["fa_idx"], backbone.fa_idx, f"fa_idx mismatch for {pattern}")
        self.assertEqual(plan["ssm_idx"], backbone.ssm_idx, f"ssm_idx mismatch for {pattern}")

        # n_cache must equal the real model's sparse cache length.
        caches = model.make_cache()
        self.assertEqual(plan["n_cache"], len(caches), f"n_cache mismatch for {pattern}")

        # cache_index is None exactly for the E/- (no-cache) layers, and the
        # non-None indices advance contiguously over M/* layers in order.
        expected_present = [b.block_type in ("M", "*") for b in backbone.layers]
        got_present = [b["cache_index"] is not None for b in plan["layers"]]
        self.assertEqual(got_present, expected_present, f"cache presence mismatch for {pattern}")

        counter = 0
        for layer in plan["layers"]:
            if layer["cache_index"] is None:
                continue
            self.assertEqual(layer["cache_index"], counter)
            counter += 1
        self.assertEqual(counter, len(caches))

    def test_fixture_pattern_matches_real_model(self) -> None:
        # tiny_config()'s pattern: mamba-first -> exercises the fa_idx branch.
        pattern = ["mamba", "moe", "attention", "mamba", "moe", "mlp"]
        self._assert_agrees_with_real_model(pattern)

        plan = plan_blocks(pattern)
        # Block-type chars derived from the word forms.
        self.assertEqual(
            [b["block_type"] for b in plan["layers"]],
            ["M", "E", "*", "M", "E", "-"],
        )
        # cache_index advances only on M/*, None for E/-.
        self.assertEqual(
            [b["cache_index"] for b in plan["layers"]],
            [0, None, 1, 2, None, None],
        )
        # mask: '*' uses attention, everything else uses ssm.
        self.assertEqual(
            [b["mask"] for b in plan["layers"]],
            ["ssm", "ssm", "attn", "ssm", "ssm", "ssm"],
        )
        # Authoritative values from the real model (the plan file's
        # ssm_idx==1 assertion is WRONG; first char 'M' breaks the ssm loop -> 0).
        self.assertEqual(plan["fa_idx"], 1)
        self.assertEqual(plan["ssm_idx"], 0)
        self.assertEqual(plan["n_cache"], 3)

    def test_attention_first_pattern_matches_real_model(self) -> None:
        # attention-first -> exercises the ssm_idx branch (counts '*' before 'M').
        pattern = ["attention", "mamba", "moe"]
        self._assert_agrees_with_real_model(pattern)

        plan = plan_blocks(pattern)
        self.assertEqual(
            [b["block_type"] for b in plan["layers"]],
            ["*", "M", "E"],
        )
        self.assertEqual(
            [b["cache_index"] for b in plan["layers"]],
            [0, 1, None],
        )
        self.assertEqual(
            [b["mask"] for b in plan["layers"]],
            ["attn", "ssm", "ssm"],
        )
        self.assertEqual(plan["fa_idx"], 0)
        self.assertEqual(plan["ssm_idx"], 1)
        self.assertEqual(plan["n_cache"], 2)

    def test_accepts_char_forms(self) -> None:
        # plan_blocks must accept char forms directly, matching word forms.
        word = plan_blocks(["mamba", "moe", "attention", "mlp"])
        char = plan_blocks(["M", "E", "*", "-"])
        self.assertEqual(word, char)


class PartitionTests(unittest.TestCase):
    def test_experts_split_from_base(self) -> None:
        from smarttensor.nemotron_layout import partition_layer_tensors
        names = [
            "backbone.layers.1.mixer.gate.weight",
            "backbone.layers.1.mixer.switch_mlp.fc1.weight",
            "backbone.layers.1.mixer.switch_mlp.fc2.weight",
            "backbone.layers.1.mixer.shared_experts.up_proj.weight",
            "backbone.layers.1.mixer.fc1_latent_proj.weight",
            "backbone.layers.1.norm.weight",
        ]
        base, experts = partition_layer_tensors(names)
        self.assertIn("backbone.layers.1.mixer.gate.weight", base)
        self.assertIn("backbone.layers.1.mixer.shared_experts.up_proj.weight", base)
        self.assertIn("backbone.layers.1.mixer.fc1_latent_proj.weight", base)
        self.assertEqual(sorted(experts), [
            "backbone.layers.1.mixer.switch_mlp.fc1.weight",
            "backbone.layers.1.mixer.switch_mlp.fc2.weight",
        ])

    def test_quantized_expert_siblings_land_in_experts(self) -> None:
        # The on-disk Ultra is 4-bit: switch_mlp tensors carry .scales/.biases
        # siblings. They contain '.mixer.switch_mlp.' so they must partition as
        # routed experts, not base (else the pager leaves quant params resident).
        from smarttensor.nemotron_layout import partition_layer_tensors
        names = [
            "backbone.layers.1.mixer.gate.weight",
            "backbone.layers.1.mixer.switch_mlp.fc1.weight",
            "backbone.layers.1.mixer.switch_mlp.fc1.scales",
            "backbone.layers.1.mixer.switch_mlp.fc1.biases",
            "backbone.layers.1.norm.weight",
        ]
        base, experts = partition_layer_tensors(names)
        self.assertIn("backbone.layers.1.mixer.switch_mlp.fc1.scales", experts)
        self.assertIn("backbone.layers.1.mixer.switch_mlp.fc1.biases", experts)
        # And base stays free of any switch_mlp tensor.
        self.assertNotIn("backbone.layers.1.mixer.switch_mlp.fc1.scales", base)
        self.assertNotIn("backbone.layers.1.mixer.switch_mlp.fc1.biases", base)

    def test_order_preserved(self) -> None:
        # Both partitions must preserve the input order (pager relies on it).
        from smarttensor.nemotron_layout import partition_layer_tensors
        names = [
            "a.norm.weight",
            "a.mixer.switch_mlp.fc1.weight",
            "a.mixer.gate.weight",
            "a.mixer.switch_mlp.fc2.weight",
            "a.mixer.shared_experts.up_proj.weight",
        ]
        base, experts = partition_layer_tensors(names)
        self.assertEqual(base, [
            "a.norm.weight",
            "a.mixer.gate.weight",
            "a.mixer.shared_experts.up_proj.weight",
        ])
        self.assertEqual(experts, [
            "a.mixer.switch_mlp.fc1.weight",
            "a.mixer.switch_mlp.fc2.weight",
        ])


if __name__ == "__main__":
    unittest.main()
