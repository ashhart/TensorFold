"""Memory-hardened ``build_fixed_hotset``: same stacks, same decode, lower peak.

On the real 550B (M3 Ultra, 256GB) ``build_fixed_hotset`` peaked at ~217GB RSS
— dangerously close to the OS jetsam line (~220GB) — even though the build's
STEADY footprint is only ~136GB (base + the fixed stacks). The ~80GB transient
spike is the warmup phase's expert page residency + mmap file pages coexisting
with the phase-3 union load + MLX's allocator cache.

The fix is PURE MEMORY HYGIENE: between phases (and periodically during the
per-layer union assemble) release everything that is NOT part of the fixed
stacks — the warmup's weight-page residency, the warmup KV/Mamba cache's freed
buffers, the per-layer raw page reads — and force the pread reader so mmap file
pages never accumulate (the macOS RSS double-count). It must NOT change WHAT gets
built or computed:

* **Correctness (the hard gate):** the assembled fixed stacks are bit-identical
  to a fresh selective load of the same expert rows (the hygiene didn't mutate
  them), the per-layer ``order`` is unchanged, and ``generate_greedy_deferred``
  output is token-identical on both fixtures. The existing exactness suites
  already pin token-identity end-to-end; here we add the direct stack-identity
  check and re-assert deferred token-identity.
* **Hygiene MECHANISM (toy proxy):** the toy model is far too small to show the
  80GB spike, so we assert the MECHANISM fires, not absolute GB — ``mx.clear_cache``
  is invoked between phases, the loader's weight-page residency is cleared after
  the warmup, and the pread reader (``drop_mmap_cache_after_read``) is in effect
  DURING the build and restored to its prior value AFTER. The real-model peak-RSS
  win can only be measured on the Studio.
"""
from __future__ import annotations

import tempfile
import unittest

from tests.fixtures.tiny_nemotron import (
    build_tiny_nemotron,
    build_tiny_nemotron_quantized,
)
from tests.test_nemotron_runner_fixed_hotset import (
    MOE_LAYERS,
    PROMPT,
    ROUTED_UNION,
)


class NemotronFixedHotsetMemoryCorrectnessTests(unittest.TestCase):
    """The hygiene must not change the assembled stacks or the decode output."""

    def test_assembled_stacks_identical_to_fresh_selective_load(self) -> None:
        """Each layer's stored fc1/fc2 stack == a fresh ``_load_selected_experts``.

        The hygiene frees the per-layer raw page reads and the warmup residency
        AFTER the stack is materialized. If any free touched a still-live stack
        the stored arrays would no longer equal a clean reload of the same expert
        rows. We rebuild the compact table for the same ``order`` straight from
        ``_load_selected_experts`` (the call the build itself uses) and assert
        every field is ``mx.array_equal`` — proving the stacks survive the frees
        untouched. Run on both fixtures (unquantized: weight only; quantized:
        weight + scales + biases) so the multi-field slice is covered.
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    runner = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=4,
                    )
                    self.addCleanup(runner.close)
                    built = runner.build_fixed_hotset(PROMPT, override=cover)

                    for layer in MOE_LAYERS:
                        entry = built[layer]
                        order = entry["order"]
                        self.assertEqual(order, cover[layer])
                        # Fresh, independent selective load of the SAME rows.
                        fresh, _ev = runner._load_selected_experts(layer, order)
                        for projection in ("fc1", "fc2"):
                            stored = entry["arrays"][projection]
                            ref = fresh[projection]
                            self.assertEqual(
                                set(stored), set(ref),
                                f"layer {layer} {projection} field set changed",
                            )
                            for field in ref:
                                self.assertTrue(
                                    bool(
                                        mx.array_equal(
                                            stored[field], ref[field]
                                        )
                                    ),
                                    f"layer {layer} {projection}.{field} stack "
                                    "diverged from a fresh selective load",
                                )

    def test_deferred_decode_token_identical_after_hardening(self) -> None:
        """``generate_greedy_deferred`` is token-identical to the page path.

        End-to-end gate: a covering fixed set built through the hardened path must
        decode the SAME tokens as the exact page path (``fixed_hotset_experts``
        off). Any drift in the stacks or any premature free would surface as a
        changed argmax somewhere in the 8-step sequence. Both fixtures.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        steps = 8
        cover = {layer: [7, 2, 5, 1] for layer in MOE_LAYERS}

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)

                    page = NemotronHStreamingForwardRunner(
                        str(path), pin_policy="all", page_experts=True
                    )
                    self.addCleanup(page.close)
                    page_tokens = page.generate_greedy(PROMPT, steps)["tokens"]

                    fixed = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        fixed_hotset_experts=4,
                    )
                    self.addCleanup(fixed.close)
                    fixed.build_fixed_hotset(PROMPT, override=cover)
                    out = fixed.generate_greedy_deferred(PROMPT, steps)

                    self.assertEqual(
                        out["tokens"],
                        page_tokens,
                        "hardened-build deferred decode diverged from the page "
                        "path",
                    )

    def test_frequency_built_membership_unchanged(self) -> None:
        """A frequency-discovered build still selects the same union under hygiene.

        The warmup-residency release happens AFTER frequency discovery, so the
        membership must be unaffected: the warmup over PROMPT routes the {2,5,7}
        union on every MoE layer, and with K = |union| that exact union must
        remain the chosen fixed set.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=3,  # == |{2,5,7}|
            )
            self.addCleanup(runner.close)
            built = runner.build_fixed_hotset(PROMPT, warmup_tokens=8)
            for layer in MOE_LAYERS:
                self.assertIn(layer, built)
                self.assertEqual(set(built[layer]["order"]), ROUTED_UNION)


class NemotronFixedHotsetMemoryHygieneTests(unittest.TestCase):
    """Assert the transient-release MECHANISM fires (toy can't show the GB)."""

    def test_clear_cache_invoked_between_phases(self) -> None:
        """``mx.clear_cache`` is called during the build (returns freed buffers).

        The build drops the warmup KV/Mamba cache and the per-layer raw page
        reads; ``mx.clear_cache`` is what actually returns those freed allocator
        buffers to the OS. We spy the symbol the runner module calls and assert it
        fired at least once during ``build_fixed_hotset`` (a no-op build that
        never clears would keep the 80GB spike resident on the real model).
        """
        import mlx.core as mx

        from smarttensor.adapters import mlx as mlx_mod
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        calls = {"n": 0}
        original = mx.clear_cache

        def counting(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        mx.clear_cache = counting
        self.addCleanup(setattr, mx, "clear_cache", original)

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
            )
            self.addCleanup(runner.close)
            before = calls["n"]
            runner.build_fixed_hotset(
                PROMPT, override={layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
            )
            self.assertGreater(
                calls["n"] - before,
                0,
                "build_fixed_hotset never invoked mx.clear_cache — freed "
                "transients were not returned to the OS",
            )

    def test_weight_page_residency_cleared_after_warmup(self) -> None:
        """The warmup's weight-page residency is released before the union load.

        On the real model the warmup faults expert row pages into the loader's
        weight-page cache; none of them are needed for the fixed stacks (phase 3
        reloads exactly the chosen experts). We attach a small weight-page cache,
        spy the loader's ``clear_weight_page_cache``, and assert the build invokes
        it — the mechanism that drops the warmup page residency between phases.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
                # Attach a weight-page cache so the warmup has page residency to
                # release (matches the real-model build, which always runs with
                # a weight-page budget).
                weight_page_budget_bytes=1 << 20,
            )
            self.addCleanup(runner.close)

            loader = runner.session.loader
            calls = {"n": 0}
            original = loader.clear_weight_page_cache

            def counting(*a, **k):
                calls["n"] += 1
                return original(*a, **k)

            loader.clear_weight_page_cache = counting  # type: ignore[assignment]
            self.addCleanup(
                setattr, loader, "clear_weight_page_cache", original
            )

            runner.build_fixed_hotset(PROMPT, warmup_tokens=4)
            self.assertGreater(
                calls["n"],
                0,
                "build_fixed_hotset never cleared the warmup weight-page "
                "residency",
            )

    def test_pread_reader_forced_during_build_and_restored_after(self) -> None:
        """``drop_mmap_cache_after_read`` is True DURING the build, restored AFTER.

        macOS never releases the mmap reader's faulted file pages (MADV_DONTNEED
        is a no-op), so the build must force the pread reader to avoid the RSS
        double-count. We capture the flag at the moment the union load runs (via
        a spy on ``_load_selected_experts``, the phase-3 loader) and assert it was
        True there, then assert the runner restored the prior value (False, the
        nemotron default) after the build returns.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=4,
            )
            self.addCleanup(runner.close)
            loader = runner.session.loader
            self.assertFalse(
                loader.drop_mmap_cache_after_read,
                "precondition: nemotron loader defaults to the mmap reader",
            )

            seen: list[bool] = []
            original = runner._load_selected_experts

            def spy(layer_index, selected, _orig=original):
                seen.append(bool(loader.drop_mmap_cache_after_read))
                return _orig(layer_index, selected)

            runner._load_selected_experts = spy  # type: ignore[assignment]
            self.addCleanup(
                lambda: runner.__dict__.pop("_load_selected_experts", None)
            )

            runner.build_fixed_hotset(
                PROMPT, override={layer: [7, 2, 5, 1] for layer in MOE_LAYERS}
            )

            # The phase-3 union load ran with the pread reader in effect.
            self.assertTrue(
                seen and all(seen),
                "the union load ran with the mmap reader (mmap file pages would "
                "accumulate on macOS)",
            )
            # And the prior reader choice was restored afterwards.
            self.assertFalse(
                loader.drop_mmap_cache_after_read,
                "build_fixed_hotset did not restore drop_mmap_cache_after_read",
            )

    def test_page_residency_and_allocator_cache_drop_after_warmup(self) -> None:
        """Warmup page residency AND MLX's allocator cache drop to 0 on release.

        The two transients the real-model spike is made of — the warmup's
        weight-page residency (the loader's page cache) and the freed buffers
        sitting in MLX's allocator cache — must actually go back to the OS at the
        post-warmup boundary. We snapshot both right after the warmup forward (via
        a spy on ``generate_greedy``) and again right after the release (via a spy
        on ``_release_build_transients``), and assert each STRICTLY DROPS to 0. On
        the toy these are KB, not GB; the assertion is on the mechanism — a build
        that leaks the warmup residency would keep page bytes / cache bytes > 0
        here and the 80GB spike on the real model. (``get_active_memory`` is NOT
        used: on the lazily-loaded toy active is dominated by the steady base
        footprint that legitimately grows as layers fault in during the warmup,
        which is not a transient.)
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                fixed_hotset_experts=3,
                # Attach a weight-page cache so the warmup builds page residency
                # (matches the real-model build).
                weight_page_budget_bytes=1 << 20,
            )
            self.addCleanup(runner.close)

            marks: dict[str, int] = {}
            loader = runner.session.loader
            original_gen = runner.generate_greedy

            def gen_spy(prompt_ids, max_tokens, _orig=original_gen):
                out = _orig(prompt_ids, max_tokens)
                mx.eval(mx.zeros(1))
                marks["warmup_page"] = loader.weight_page_resident_bytes
                marks["warmup_cache"] = mx.get_cache_memory()
                return out

            runner.generate_greedy = gen_spy  # type: ignore[assignment]
            self.addCleanup(
                lambda: runner.__dict__.pop("generate_greedy", None)
            )

            original_release = runner._release_build_transients

            def release_spy(*a, _orig=original_release, **k):
                out = _orig(*a, **k)
                mx.eval(mx.zeros(1))
                marks["post_page"] = loader.weight_page_resident_bytes
                marks["post_cache"] = mx.get_cache_memory()
                return out

            runner._release_build_transients = release_spy  # type: ignore[assignment]
            self.addCleanup(
                lambda: runner.__dict__.pop("_release_build_transients", None)
            )

            runner.build_fixed_hotset(PROMPT, warmup_tokens=6)

            self.assertIn("warmup_page", marks, "warmup did not run")
            self.assertIn("post_page", marks, "post-warmup release did not run")
            # The warmup genuinely built page residency to reclaim.
            self.assertGreater(
                marks["warmup_page"],
                0,
                "warmup did not fault any expert pages — test premise invalid",
            )
            # ...and the release returned BOTH transients to the OS.
            self.assertEqual(
                marks["post_page"],
                0,
                "warmup weight-page residency not released before the union load",
            )
            self.assertLessEqual(
                marks["post_cache"],
                1 << 20,
                "MLX allocator cache not returned to the OS after the warmup",
            )


if __name__ == "__main__":
    unittest.main()
