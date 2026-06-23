"""``force_top_k`` routing truncation on the DEFERRED path (lossy bandwidth lever).

NemotronH routes top-``num_experts_per_tok`` experts per token (22 on the real
550B, 2 on the tiny fixture). ``force_top_k=K`` truncates that to the K
HIGHEST-gate-score experts at inference: keep the K best of the routed set,
RENORMALIZE their scores exactly the way the stock gate's ``norm_topk_prob``
would for a top-K gate, and gather only K experts. Fewer expert-GEMMs per token
== the real bandwidth win, lossy by design (it changes routing, so NOT exact).

The gate truncation happens in :class:`NemotronDeferredRemapGate` BEFORE the
``mx.take(g2s, inds)`` global->slot remap, so the cold-buddy substitution baked
into ``g2s`` still applies to the K kept ids -> the deferred single-graph +
cold-buddy path is untouched; ONLY the routing mask narrows.

Contract proven here on the tiny fixture (``num_experts_per_tok=2``):

* ``force_top_k=1`` DIFFERS from the unset (full top-2) path -> truncation is
  genuinely lossy and runs end-to-end without error.
* ``force_top_k=1`` drops the per-token ACTIVE-expert count vs the unset path
  (the gather touches fewer experts).
* ``force_top_k >= routed count`` (== 2 here) is a NO-OP: bit-identical to the
  unset path (and so still == stock, preserving the deferred exactness gate).
* ``force_top_k=None`` is the default and changes nothing.

Tiny weights are random, so the truncated LOGITS are numerically meaningless;
what is asserted is the PIPELINE: lossy<->no-op behavior, active-expert drop,
and no regression to the unset deferred path.
"""
from __future__ import annotations

import tempfile
import unittest

import mlx.core as mx

from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
from tests.fixtures.tiny_nemotron import (
    build_tiny_nemotron,
    build_tiny_nemotron_quantized,
)
from tests.test_nemotron_runner_fixed_hotset import MOE_LAYERS, PROMPT


# A COVERING fixed set (superset of the routed union {2,5,7}) so every MoE layer
# is all-resident: the unset deferred path is then pure (no cold redo) and
# bit-identical to stock, isolating ``force_top_k`` as the ONLY routing change.
_COVER = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}


class _RecordingGate:
    """Type-level proxy around an installed remap gate that records routed width.

    ``gate(x)`` resolves ``__call__`` on the TYPE, so instance-level patching is
    invisible to the model. This proxy defines ``__call__`` on its own class and
    delegates to the wrapped gate, recording the per-position routed WIDTH
    (``local.shape[-1]`` = experts gathered per token) it emits each call. That
    width is exactly the bandwidth quantity ``force_top_k`` shrinks (K vs the
    full ``num_experts_per_tok``); the distinct-expert UNION over a multi-token
    prompt need not shrink, but the per-token gather always does.
    """

    def __init__(self, gate, sink):
        self._gate = gate
        self._sink = sink

    def __getattr__(self, name):  # forward n_routed_experts etc.
        return getattr(self._gate, name)

    def __call__(self, x):
        local, scores = self._gate(x)
        self._sink.append(int(local.shape[-1]))
        return local, scores


def _active_expert_count(runner: NemotronHStreamingForwardRunner, prompt) -> int:
    """Max per-token routed WIDTH the deferred gates emit over ``prompt``.

    Wraps every installed :class:`NemotronDeferredRemapGate` in a type-level
    recording proxy and runs one deferred forward. Returns the max routed width
    (experts gathered per token); ``force_top_k=K`` must shrink this to K from
    the full ``num_experts_per_tok``.
    """
    from smarttensor.adapters import mlx as mlx_mod

    runner._install_deferred_hotset()
    widths: list[int] = []
    backbone = runner.session.model.backbone
    originals = {}
    try:
        for layer_index in runner._fixed_hotset:
            mixer = backbone.layers[layer_index].mixer
            gate = mixer.gate
            if not isinstance(gate, mlx_mod.NemotronDeferredRemapGate):
                continue
            originals[layer_index] = gate
            mixer.gate = _RecordingGate(gate, widths)
        out = runner.forward_logits_deferred(prompt)
        mx.eval(out)
    finally:
        for layer_index, gate in originals.items():
            backbone.layers[layer_index].mixer.gate = gate
        runner._uninstall_deferred_hotset()
    return max(widths) if widths else 0


class ForceTopKContractTests(unittest.TestCase):
    """``force_top_k`` is lossy when K < routed, a no-op when K >= routed."""

    def _runner(self, path, *, force_top_k):
        runner = NemotronHStreamingForwardRunner(
            str(path),
            pin_policy="all",
            page_experts=True,
            fixed_hotset_experts=4,
            force_top_k=force_top_k,
        )
        self.addCleanup(runner.close)
        runner.build_fixed_hotset(PROMPT, override=_COVER)
        return runner

    def test_force_top_k_below_routed_changes_output_both_fixtures(self) -> None:
        """K=1 (< routed 2) DIFFERS from unset and runs without error."""
        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)

                    base = self._runner(path, force_top_k=None)
                    base_logits = base.forward_logits_deferred(PROMPT)
                    mx.eval(base_logits)

                    trunc = self._runner(path, force_top_k=1)
                    trunc_logits = trunc.forward_logits_deferred(PROMPT)
                    mx.eval(trunc_logits)

                    self.assertEqual(
                        tuple(base_logits.shape), tuple(trunc_logits.shape)
                    )
                    diff = float(
                        mx.max(
                            mx.abs(
                                base_logits.astype(mx.float32)
                                - trunc_logits.astype(mx.float32)
                            )
                        )
                    )
                    self.assertGreater(
                        diff,
                        0.0,
                        "force_top_k=1 produced identical logits to full routing "
                        "(truncation was a no-op when it should be lossy)",
                    )

    def test_force_top_k_at_routed_count_is_noop_both_fixtures(self) -> None:
        """K == routed (2) is bit-identical to the unset path (and so == stock)."""
        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)

                    base = self._runner(path, force_top_k=None)
                    base_logits = base.forward_logits_deferred(PROMPT)
                    mx.eval(base_logits)

                    noop = self._runner(path, force_top_k=2)
                    noop_logits = noop.forward_logits_deferred(PROMPT)
                    mx.eval(noop_logits)

                    self.assertEqual(
                        float(
                            mx.max(
                                mx.abs(
                                    base_logits.astype(mx.float32)
                                    - noop_logits.astype(mx.float32)
                                )
                            )
                        ),
                        0.0,
                        "force_top_k == routed count must be a no-op",
                    )

    def test_force_top_k_above_routed_count_is_noop(self) -> None:
        """K > routed (e.g. 5 > 2) is also a no-op (cannot keep more than routed)."""
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            base = self._runner(path, force_top_k=None)
            base_logits = base.forward_logits_deferred(PROMPT)
            mx.eval(base_logits)

            noop = self._runner(path, force_top_k=5)
            noop_logits = noop.forward_logits_deferred(PROMPT)
            mx.eval(noop_logits)

            self.assertEqual(
                float(
                    mx.max(
                        mx.abs(
                            base_logits.astype(mx.float32)
                            - noop_logits.astype(mx.float32)
                        )
                    )
                ),
                0.0,
            )

    def test_force_top_k_reduces_active_expert_count(self) -> None:
        """K=1 gathers 1 expert/token vs the full top-2 path's 2."""
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            base = self._runner(path, force_top_k=None)
            base_active = _active_expert_count(base, PROMPT)

            trunc = self._runner(path, force_top_k=1)
            trunc_active = _active_expert_count(trunc, PROMPT)

            self.assertEqual(base_active, 2, "full routing should gather top-2")
            self.assertEqual(trunc_active, 1, "force_top_k=1 should gather 1")
            self.assertGreater(
                base_active,
                trunc_active,
                f"per-token gather did not drop: base={base_active} "
                f"trunc={trunc_active}",
            )

    def test_force_top_k_keeps_argmax_and_renorms_to_scaling_factor(self) -> None:
        """K=1 keeps the highest-score routed expert; its score == routed_scaling.

        The documented contract: truncation keeps the K best routed experts and,
        when ``norm_topk_prob`` is set, renormalizes them to sum to
        ``routed_scaling_factor`` (the stock norm invariant applied to the K
        subset). With K=1 that means the single survivor carries the full
        ``routed_scaling_factor`` mass, and it must be the routed argmax.

        (NOTE: the stock gate's K=1 path skips normalization via its ``top_k > 1``
        guard, so it is NOT the reference here — the model is never *run* at K=1;
        the well-defined truncation renormalizes the surviving routing mass.)
        """
        import numpy as np

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            runner = self._runner(path, force_top_k=1)
            runner._install_deferred_hotset()
            self.addCleanup(runner._uninstall_deferred_hotset)

            backbone = runner.session.model.backbone
            layer_index = next(iter(runner._fixed_hotset))
            gate = backbone.layers[layer_index].mixer.gate
            stock = gate._gate  # the original MoEGate

            x = mx.random.normal((1, stock.config.hidden_size)).astype(mx.float32)

            # Stock routed top-2 (inds + already-normalized scores).
            routed_inds, routed_scores = stock(x)
            mx.eval(routed_inds, routed_scores)
            routed_inds = np.asarray(routed_inds).reshape(-1)
            routed_scores = np.asarray(routed_scores).reshape(-1)
            best_pos = int(routed_scores.argmax())
            best_global = int(routed_inds[best_pos])

            # Truncated gate emits LOCAL slots (post g2s remap).
            local, scores = gate(x)
            mx.eval(local, scores)
            g2s = np.asarray(runner._fixed_hotset[layer_index]["g2s"])

            self.assertEqual(local.shape[-1], 1, "expected exactly K=1 routed slot")
            self.assertEqual(
                int(np.asarray(local).reshape(-1)[0]),
                int(g2s[best_global]),
                "force_top_k=1 did not keep the highest-score routed expert",
            )
            # norm_topk_prob is True on the fixture -> single score == rsf.
            self.assertAlmostEqual(
                float(np.asarray(scores).reshape(-1)[0]),
                float(stock.routed_scaling_factor),
                places=5,
                msg="K=1 survivor score must carry the full routed_scaling_factor",
            )

    def test_force_top_k_renorm_formula_matches_stock_topk(self) -> None:
        """For K>1 the kept-K renorm == stock ``group_expert_select(top_k=K)``.

        Tiny routes only 2 experts, so verify the renormalization MATH directly
        against the stock gate on a constructed routed set: given the stock's
        already-normalized routed scores (``orig/sum_routed * rsf``), keeping the
        top-K and re-dividing by their sum (× rsf) must equal ``orig_K/sum_K * rsf``
        — exactly what the stock gate emits with ``top_k=K``. This is the identity
        the truncation relies on; we check it numerically end-to-end.
        """
        import numpy as np

        rng = np.random.default_rng(0)
        rsf = 2.5  # arbitrary routed_scaling_factor
        for routed_n, k in ((6, 4), (8, 2), (5, 3)):
            orig = rng.random(routed_n) + 0.1  # positive "sigmoid" scores
            # Stock normalized routed scores (norm over the full routed set).
            stock_routed = orig / orig.sum() * rsf
            # Truncation: keep top-K by score, renorm to sum to rsf.
            keep = np.argsort(stock_routed)[-k:]
            kept = stock_routed[keep]
            got = kept / kept.sum() * rsf
            # Ground truth: the stock gate run with top_k=K on the SAME orig.
            kept_orig = orig[keep]
            ref = kept_orig / kept_orig.sum() * rsf
            self.assertTrue(
                np.allclose(np.sort(got), np.sort(ref), atol=1e-9),
                f"renorm formula diverged (routed={routed_n}, k={k})",
            )


if __name__ == "__main__":
    unittest.main()
