"""Exactness gate for the fused Mamba2 gated-RMSNorm Metal kernel.

The fused kernel must reproduce mlx_lm's ``MambaRMSNormGated.__call__`` (swiglu
gate -> per-group RMS over ``group_size`` -> per-channel weight) to within the
input dtype's ULP, AND a float64 oracle must show the single fused dispatch is
NOT LESS accurate than the stock multi-op sequence. Covers several shapes, all
three input dtypes, a couple of n_groups splits, and the gate=None fallback.
"""
from __future__ import annotations

import unittest

import numpy as np


def _stock_norm(hidden_size: int, eps: float, group_size: int):
    from mlx_lm.models.nemotron_h import MambaRMSNormGated

    return MambaRMSNormGated(hidden_size, eps=eps, group_size=group_size)


def _oracle_f64(x_np: np.ndarray, gate_np, weight_np: np.ndarray, n_groups: int, eps: float):
    x = x_np.astype(np.float64)
    if gate_np is not None:
        g = gate_np.astype(np.float64)
        x = (g / (1.0 + np.exp(-g))) * x
    shape = x.shape
    grouped = x.reshape(*shape[:-1], n_groups, shape[-1] // n_groups)
    ms = np.mean(grouped * grouped, axis=-1, keepdims=True)
    grouped = grouped / np.sqrt(ms + eps)
    normed = grouped.reshape(shape)
    return weight_np.astype(np.float64) * normed


class FusedMambaRMSNormGatedTests(unittest.TestCase):
    def _eps(self) -> float:
        return 1e-5

    def test_matches_stock_across_shapes_dtypes_groups(self) -> None:
        import mlx.core as mx

        from smarttensor.ssm_glue import fused_mamba_rmsnorm_gated

        eps = self._eps()
        rng = np.random.default_rng(0)
        # (batch, seq, hidden), n_groups  -- hidden divisible by n_groups.
        cases = [
            ((1, 1, 64), 1),
            ((1, 1, 64), 8),
            ((1, 3, 128), 8),
            ((2, 4, 256), 4),
            ((1, 1, 96), 3),
        ]
        # Per-dtype mantissa bits -> a ULP-relative max-abs-diff tolerance scaled
        # by the output magnitude. fp32: 23 bits, fp16: 10, bf16: 7. We allow a
        # few ULPs of slack since the fused kernel and the stock op-sequence use
        # different (both valid) fp32 reduction orders for the per-group RMS.
        dtypes = [
            (mx.float32, 8, 23),
            (mx.float16, 8, 10),
            (mx.bfloat16, 8, 7),
        ]
        for shape, n_groups in cases:
            hidden = shape[-1]
            group_size = hidden // n_groups
            x_np = rng.standard_normal(shape).astype(np.float32)
            gate_np = rng.standard_normal(shape).astype(np.float32)
            weight_np = rng.standard_normal((hidden,)).astype(np.float32) * 0.5 + 1.0
            oracle = _oracle_f64(x_np, gate_np, weight_np, n_groups, eps)
            scale = float(np.max(np.abs(oracle)))
            for mxdt, ulps, mant_bits in dtypes:
                with self.subTest(shape=shape, n_groups=n_groups, dtype=str(mxdt)):
                    x = mx.array(x_np).astype(mxdt)
                    gate = mx.array(gate_np).astype(mxdt)
                    weight = mx.array(weight_np).astype(mxdt)

                    norm = _stock_norm(hidden, eps, group_size)
                    norm.weight = weight
                    stock = norm(x, gate)
                    fused = fused_mamba_rmsnorm_gated(
                        x, gate, weight, n_groups=n_groups, eps=eps
                    )
                    mx.eval(stock, fused)

                    self.assertEqual(tuple(fused.shape), tuple(stock.shape))
                    self.assertEqual(fused.dtype, stock.dtype)

                    ulp = scale * (2.0 ** -mant_bits)
                    tol = ulps * ulp
                    stock_f = np.asarray(stock.astype(mx.float32))
                    fused_f = np.asarray(fused.astype(mx.float32))
                    max_abs = float(np.max(np.abs(fused_f - stock_f)))
                    self.assertLessEqual(
                        max_abs,
                        tol,
                        f"fused vs stock max-abs-diff {max_abs:.3e} > {ulps} ULP "
                        f"({tol:.3e})",
                    )

                    # float64 oracle: the fused single dispatch must be within a
                    # few ULPs of ground truth AND not materially worse than the
                    # stock op-sequence. The kernel and stock use different (both
                    # valid) fp32 reduction orders, so per-run fused-vs-stock can
                    # vary by <1 ULP; a broken kernel is tens of ULPs off (caught).
                    stock_err = float(np.max(np.abs(stock_f.astype(np.float64) - oracle)))
                    fused_err = float(np.max(np.abs(fused_f.astype(np.float64) - oracle)))
                    gate_thresh = max(stock_err, 2.0 * ulp) * 1.5
                    self.assertLessEqual(
                        fused_err,
                        gate_thresh,
                        f"fused err {fused_err:.3e} ({fused_err/ulp:.2f} ULP) "
                        f"exceeds gate {gate_thresh:.3e} (stock {stock_err:.3e})",
                    )

    def test_gate_none_fallback_matches_stock(self) -> None:
        import mlx.core as mx

        from smarttensor.ssm_glue import fused_mamba_rmsnorm_gated

        eps = self._eps()
        rng = np.random.default_rng(1)
        shape = (1, 2, 128)
        n_groups = 8
        hidden = shape[-1]
        group_size = hidden // n_groups
        x_np = rng.standard_normal(shape).astype(np.float32)
        weight_np = rng.standard_normal((hidden,)).astype(np.float32) * 0.5 + 1.0

        for mxdt in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=str(mxdt)):
                x = mx.array(x_np).astype(mxdt)
                weight = mx.array(weight_np).astype(mxdt)
                norm = _stock_norm(hidden, eps, group_size)
                norm.weight = weight
                stock = norm(x, None)
                fused = fused_mamba_rmsnorm_gated(
                    x, None, weight, n_groups=n_groups, eps=eps
                )
                mx.eval(stock, fused)
                stock_f = np.asarray(stock.astype(mx.float32))
                fused_f = np.asarray(fused.astype(mx.float32))
                self.assertTrue(
                    np.array_equal(fused_f, stock_f),
                    "gate=None fallback must be bit-identical to stock",
                )


class FusedRMSNormGateRunnerInstallTests(unittest.TestCase):
    """The runner flag swaps the fused norm into every Mamba2 layer and the
    resulting logits stay within fp32 ULP noise of the stock (flag-OFF) path."""

    def test_runner_flag_installs_and_logits_match_within_ulp(self) -> None:
        import tempfile

        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
        from tests.fixtures.tiny_nemotron import build_tiny_nemotron

        prompt = [1, 5, 9, 3]
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            stock_runner = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", fuse_ssm_norm_gate=False
            )
            self.addCleanup(stock_runner.close)
            stock = stock_runner.forward_logits(prompt)
            mx.eval(stock)

            fused_runner = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", fuse_ssm_norm_gate=True
            )
            self.addCleanup(fused_runner.close)
            self.assertTrue(fused_runner.fuse_ssm_norm_gate)
            fused = fused_runner.forward_logits(prompt)
            mx.eval(fused)

            self.assertEqual(tuple(fused.shape), tuple(stock.shape))
            self.assertEqual(fused.dtype, stock.dtype)
            stock_f = np.asarray(stock.astype(mx.float32))
            fused_f = np.asarray(fused.astype(mx.float32))
            max_abs = float(np.max(np.abs(fused_f - stock_f)))
            scale = float(np.max(np.abs(stock_f)))
            tol = 16.0 * scale * (2.0 ** -23)
            self.assertLessEqual(
                max_abs,
                tol,
                f"fused-norm runner logits diverged {max_abs:.3e} > {tol:.3e}",
            )

    def test_runner_flag_default_off(self) -> None:
        import tempfile

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
        from tests.fixtures.tiny_nemotron import build_tiny_nemotron

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            runner = NemotronHStreamingForwardRunner(str(path), pin_policy="all")
            self.addCleanup(runner.close)
            self.assertFalse(runner.fuse_ssm_norm_gate)


class FusedRMSNormGateDeferredInstallTests(unittest.TestCase):
    """The fused norm must actually install in the DEFERRED decode path (the fast
    ~16 tok/s path we measure), not just the synced forward_logits path. A broken
    install would leave the kernel inert -> fused==unfused, so each test asserts
    ``_fused_ssm_norm_installed`` flipped True (catches the wiring gap directly),
    and the deferred logits stay within fp32 ULP of the flag-OFF deferred path."""

    def _cover(self):
        from tests.test_nemotron_runner_fixed_hotset import MOE_LAYERS

        return {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}

    def test_forward_logits_deferred_installs_fused_within_ulp(self) -> None:
        import tempfile

        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
        from tests.fixtures.tiny_nemotron import build_tiny_nemotron
        from tests.test_nemotron_runner_fixed_hotset import PROMPT

        cover = self._cover()
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            off = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True,
                fixed_hotset_experts=4, fuse_ssm_norm_gate=False,
            )
            self.addCleanup(off.close)
            off.build_fixed_hotset(PROMPT, override=cover)
            off_logits = off.forward_logits_deferred(PROMPT)
            mx.eval(off_logits)
            self.assertFalse(off._fused_ssm_norm_installed)

            on = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True,
                fixed_hotset_experts=4, fuse_ssm_norm_gate=True,
            )
            self.addCleanup(on.close)
            on.build_fixed_hotset(PROMPT, override=cover)
            on_logits = on.forward_logits_deferred(PROMPT)
            mx.eval(on_logits)
            self.assertTrue(
                on._fused_ssm_norm_installed,
                "fused norm did not install in forward_logits_deferred",
            )

            self.assertEqual(tuple(on_logits.shape), tuple(off_logits.shape))
            off_f = np.asarray(off_logits.astype(mx.float32))
            on_f = np.asarray(on_logits.astype(mx.float32))
            max_abs = float(np.max(np.abs(on_f - off_f)))
            scale = float(np.max(np.abs(off_f)))
            tol = 16.0 * scale * (2.0 ** -23)
            self.assertLessEqual(
                max_abs, tol,
                f"deferred fused-norm logits diverged {max_abs:.3e} > {tol:.3e}",
            )

    def test_generate_greedy_deferred_installs_fused(self) -> None:
        import tempfile

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
        from tests.fixtures.tiny_nemotron import build_tiny_nemotron
        from tests.test_nemotron_runner_fixed_hotset import PROMPT

        cover = self._cover()
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            on = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True,
                fixed_hotset_experts=4, fuse_ssm_norm_gate=True,
            )
            self.addCleanup(on.close)
            on.build_fixed_hotset(PROMPT, override=cover)
            out = on.generate_greedy_deferred(PROMPT, 2)
            self.assertTrue(
                on._fused_ssm_norm_installed,
                "fused norm did not install in generate_greedy_deferred",
            )
            self.assertEqual(len(out["tokens"]), 2)


if __name__ == "__main__":
    unittest.main()
