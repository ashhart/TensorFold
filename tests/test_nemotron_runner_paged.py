"""Selective-expert (paged) NemotronH MoE forward: token-exact vs stock."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.fixtures.tiny_nemotron import (
    build_tiny_nemotron,
    build_tiny_nemotron_quantized,
)


def _stock_logits(path, ids):
    import mlx.core as mx

    from smarttensor.adapters.mlx import build_mlx_model_shell
    from smarttensor.manifest import SmartTensorManifest

    cfg = json.loads((Path(path) / "config.json").read_text())
    man = SmartTensorManifest.from_safetensors(
        [str(Path(path) / "model.safetensors")]
    )
    m = build_mlx_model_shell(cfg, man)
    w = m.sanitize(mx.load(str(Path(path) / "model.safetensors")))
    m.load_weights(list(w.items()))
    out = m(mx.array([ids]), cache=m.make_cache())
    mx.eval(out)
    return out


def _measure_weight_page_footprint(path, prompt_ids):
    """Derive the toy's per-MoE-layer union footprint + largest page from disk.

    Runs a generous-budget (whole-model) multi-token decode so every routed
    row page is admitted and stays resident (no eviction), then reads the live
    ``weight_page_summary()`` pages to learn, per MoE layer, the total bytes of
    its routed-expert union and the single largest page. The tight-budget test
    derives its budget from these MEASURED sizes rather than a magic constant.

    Returns ``(per_layer_max_bytes, all_layers_bytes, max_page_bytes)``.
    """
    import re
    from collections import defaultdict

    from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
    from smarttensor.manifest import SmartTensorManifest

    man = SmartTensorManifest.from_safetensors(
        [str(Path(path) / "model.safetensors")]
    )
    runner = NemotronHStreamingForwardRunner(
        str(path),
        pin_policy="all",
        page_experts=True,
        # Footprint derivation measures the per-token page re-request pattern of
        # the page-rebuild path; the persistent-table path reuses rows across
        # steps and would not re-touch every page each token. Pin the old path
        # so the measured per-layer union sizes match what the budget tests
        # (which also pin the old path) actually exercise.
        persist_expert_tables=False,
        weight_page_budget_bytes=man.total_bytes,
    )
    try:
        runner.generate_greedy(list(prompt_ids), max_tokens=8)
        summary = runner.session.loader.weight_page_summary()
        # Generous budget => nothing evicted, so every touched page is resident
        # and these sizes are the true per-layer union footprints.
        assert summary["evictions"] == 0, "measurement pass evicted unexpectedly"
        layer_bytes: dict[int, int] = defaultdict(int)
        max_page = 0
        for page in summary["pages"]:
            match = re.search(r"layers\.(\d+)\.", page["tensor_name"])
            assert match is not None, page["tensor_name"]
            layer_bytes[int(match.group(1))] += page["nbytes"]
            max_page = max(max_page, page["nbytes"])
        assert layer_bytes, "no resident expert pages measured"
        return max(layer_bytes.values()), sum(layer_bytes.values()), max_page
    finally:
        runner.close()


def _assert_non_contiguous_routing(test, runner) -> None:
    """Every MoE layer's routed union is a genuine non-identity remap."""
    test.assertTrue(runner._expert_history, "no MoE layers routed")
    for layer_index, union in runner._expert_history.items():
        union_set = set(union)
        test.assertNotEqual(
            union_set,
            {0, 1},
            f"layer {layer_index} routing is degenerate {{0,1}} (identity remap)",
        )
        test.assertTrue(
            any(e >= 2 for e in union_set),
            f"layer {layer_index} union {sorted(union_set)} has no expert id >= 2",
        )


class NemotronPagedForwardTests(unittest.TestCase):
    def test_paged_matches_stock_quantized(self) -> None:
        """4-bit STACKED switch_mlp: selective paged forward is token-exact.

        Mirrors the real on-disk Ultra format (stacked, 4-bit affine). The
        selective loader must slice weight + scales + biases of the routed rows
        only, and the paged forward must be bit-identical to the stock quantized
        model.
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        ids = [5, 9, 1, 17, 3, 8]
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            expected = _stock_logits(path, ids)

            runner = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True
            )
            self.addCleanup(runner.close)
            got = runner.forward_logits(ids)
            mx.eval(got)

            self.assertEqual(tuple(got.shape), tuple(expected.shape))
            self.assertEqual(
                float(
                    mx.max(mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)))
                ),
                0.0,
            )

            # Selectivity: the loader must slice ONLY the routed experts, never
            # the full table of 8. Each MoE layer routes to the non-contiguous
            # union {2,5,7} (3 experts) for the fixed token sequence.
            _assert_non_contiguous_routing(self, runner)
            for layer_index, union in runner._expert_history.items():
                self.assertLess(
                    len(union),
                    8,
                    f"layer {layer_index} loaded all experts (not selective)",
                )

    def test_paged_matches_stock(self) -> None:
        import mlx.core as mx

        from smarttensor.adapters.mlx import (
            NemotronHStreamingForwardRunner,
            build_mlx_model_shell,
        )
        from smarttensor.manifest import SmartTensorManifest

        ids = [5, 9, 1, 17, 3, 8]
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            cfg = json.loads((Path(path) / "config.json").read_text())
            man = SmartTensorManifest.from_safetensors(
                [str(Path(path) / "model.safetensors")]
            )
            m = build_mlx_model_shell(cfg, man)
            w = m.sanitize(mx.load(str(Path(path) / "model.safetensors")))
            m.load_weights(list(w.items()))
            expected = m(mx.array([ids]), cache=m.make_cache())
            mx.eval(expected)

            runner = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True
            )
            self.addCleanup(runner.close)
            got = runner.forward_logits(ids)
            mx.eval(got)

            self.assertEqual(tuple(got.shape), tuple(expected.shape))
            self.assertEqual(
                float(
                    mx.max(mx.abs(got.astype(mx.float32) - expected.astype(mx.float32)))
                ),
                0.0,
            )

            # The toy fixture forces NON-CONTIGUOUS routing so global->local is
            # a genuine remap (not the identity {0,1}). Prove the remap was
            # non-trivial AND that the loader sliced ONLY the routed rows of the
            # stacked switch_mlp (never the full table of 8).
            _assert_non_contiguous_routing(self, runner)
            for layer_index, union in runner._expert_history.items():
                self.assertLess(
                    len(union),
                    8,
                    f"layer {layer_index} loaded all experts (not selective)",
                )

    def test_paged_cache_on_matches_cache_off(self) -> None:
        """Read-side weight-page cache is exact: cache-ON == cache-OFF == stock.

        The resident row-page cache (``PagedWeightCache``) only changes whether a
        sliced expert row comes from RAM (hit) or disk (miss); it never changes
        WHICH bytes are read or how the compact table is assembled. So turning it
        on with a budget large enough to hold the toy's whole expert set (no
        eviction) must be bit-identical to the cache-off path AND to stock
        mlx_lm. Run for BOTH the unquantized and the 4-bit quantized toy.
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
        from smarttensor.manifest import SmartTensorManifest

        ids = [5, 9, 1, 17, 3, 8]
        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_logits(path, ids)

                    # Budget = whole toy model on disk, so the routed-expert
                    # row pages always fit and NOTHING is ever evicted. (Total
                    # model size is a comfortable over-estimate of just the
                    # stacked expert tables, which is exactly what we want.)
                    man = SmartTensorManifest.from_safetensors(
                        [str(Path(path) / "model.safetensors")]
                    )
                    total_bytes = man.total_bytes

                    runner_off = NemotronHStreamingForwardRunner(
                        str(path), pin_policy="all", page_experts=True
                    )
                    self.addCleanup(runner_off.close)
                    self.assertIsNone(
                        runner_off.session.loader._weight_pages,
                        "cache-off path attached a weight-page cache",
                    )
                    cache_off = runner_off.forward_logits(ids)
                    mx.eval(cache_off)

                    runner_on = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        weight_page_budget_bytes=total_bytes,
                    )
                    self.addCleanup(runner_on.close)
                    self.assertIsNotNone(
                        runner_on.session.loader._weight_pages,
                        "cache-on path did not attach a weight-page cache",
                    )
                    cache_on = runner_on.forward_logits(ids)
                    mx.eval(cache_on)

                    self.assertEqual(
                        tuple(cache_on.shape), tuple(cache_off.shape)
                    )
                    self.assertEqual(
                        float(
                            mx.max(
                                mx.abs(
                                    cache_on.astype(mx.float32)
                                    - cache_off.astype(mx.float32)
                                )
                            )
                        ),
                        0.0,
                        "cache-on diverged from cache-off",
                    )
                    self.assertEqual(
                        float(
                            mx.max(
                                mx.abs(
                                    cache_on.astype(mx.float32)
                                    - stock.astype(mx.float32)
                                )
                            )
                        ),
                        0.0,
                        "cache-on diverged from stock mlx_lm",
                    )

    def test_tiny_budget_evicts_but_stays_exact(self) -> None:
        """Read-side page cache stays TOKEN-EXACT even when eviction re-reads.

        The cache is a pure read-side memo: whether a routed expert row is served
        from RAM (hit) or freshly re-read from disk after eviction (miss), the
        bytes are identical, so the logits cannot move. This proves the hard case
        the large-budget tests skip: a budget tight enough that previously-evicted
        pages must be re-loaded mid-run, and the forward is STILL bit-identical to
        stock mlx_lm.

        Budget derivation (NOT a magic constant): a generous-budget measurement
        pass reports, per MoE layer, its routed-expert union footprint and the
        single largest page. The tight budget is ``one_layer_union + max_page`` —
        comfortably ABOVE any single page (so every page is admissible) yet BELOW
        the footprint of BOTH MoE layers (so loading the second layer must evict
        the first). The toy has 2 MoE layers each routing to {2,5,7}, so this sits
        at ~1.1x one layer / ~0.56x both layers.
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        prompt = [5, 9, 1, 17, 3, 8]
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)

            per_layer, all_layers, max_page = _measure_weight_page_footprint(
                path, prompt
            )
            # Tight: above any single page (admissible) yet below both layers
            # (loading layer 2's experts must evict layer 1's). Derived from the
            # measured fixture sizes, never hardcoded.
            budget = per_layer + max_page
            self.assertGreater(
                budget, max_page, "budget below a single page is inadmissible"
            )
            self.assertLess(
                budget,
                all_layers,
                "budget large enough to hold both layers would never evict",
            )

            # --- Multi-token decode under the tight budget: decode steps
            # re-request the same (layer, expert) rows; the tight budget evicts
            # them between layers, so they must be RE-READ from disk each step.
            # This test exercises the read-side weight-page cache's
            # evict-then-reread semantics, which only fire when rows are
            # re-requested every token — the per-token page-rebuild path. The
            # persistent-table path (default) reuses rows across steps and is
            # covered for eviction exactness by its own tiny-cap test. Pin the
            # old path here.
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                persist_expert_tables=False,
                weight_page_budget_bytes=budget,
            )
            self.addCleanup(runner.close)
            result = runner.generate_greedy(prompt, max_tokens=8)
            page_summary = result["summary"]["weight_page_summary"]

            # Eviction genuinely fired.
            self.assertGreater(
                page_summary["evictions"],
                0,
                "tight budget did not evict (footprint derivation is wrong)",
            )
            # Every page is reloaded on a miss (insertions == misses), and there
            # are MORE misses than evictions, so at least one evicted page was
            # re-read after eviction (the memo's re-read path, not just admission).
            self.assertEqual(
                page_summary["insertions"],
                page_summary["misses"],
                "every miss should re-load and re-admit its page",
            )
            self.assertGreater(
                page_summary["misses"],
                page_summary["evictions"],
                "no evicted page was ever re-read (evict-then-reread not exercised)",
            )

            # --- Exactness under the SAME tight budget. A fresh runner: warm
            # once, then a REPEAT identical forward must re-read the pages that
            # were resident-then-evicted between the two passes (misses rise),
            # and the logits are still bit-identical to stock.
            expected = _stock_logits(path, prompt)
            runner2 = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                persist_expert_tables=False,
                weight_page_budget_bytes=budget,
            )
            self.addCleanup(runner2.close)

            warm = runner2.forward_logits(prompt)
            mx.eval(warm)
            before = dict(runner2.session.loader.weight_page_summary())
            got = runner2.forward_logits(prompt)
            mx.eval(got)
            after = dict(runner2.session.loader.weight_page_summary())

            # The repeat forward re-touched the same rows; because the tight
            # budget evicted them between passes they were re-read from disk.
            self.assertGreater(
                after["evictions"], 0, "single forward never evicted under tight budget"
            )
            self.assertGreater(
                after["misses"],
                before["misses"],
                "repeat forward did not re-read any evicted page",
            )

            # The whole point: re-reading an evicted page yields identical bytes,
            # so the logits are EXACTLY stock (0.0, not approximately).
            self.assertEqual(tuple(got.shape), tuple(expected.shape))
            self.assertEqual(
                float(
                    mx.max(
                        mx.abs(
                            got.astype(mx.float32) - expected.astype(mx.float32)
                        )
                    )
                ),
                0.0,
                "logits diverged from stock under eviction pressure",
            )

    def test_budget_below_single_page_raises(self) -> None:
        """A budget below one expert row-page hits the cache's admission guard.

        ``PagedWeightCache.get_or_load`` refuses to admit a page larger than the
        whole budget (it can never fit), raising ``ValueError`` from the
        single-page-over-budget guard in ``weight_pager.py`` (the
        ``nbytes > self.budget_bytes`` check). With the budget set one byte below
        the largest measured expert page, the first selective expert load trips
        that guard.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        prompt = [5, 9, 1, 17, 3, 8]
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            _per_layer, _all_layers, max_page = _measure_weight_page_footprint(
                path, prompt
            )

            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                weight_page_budget_bytes=max_page - 1,
            )
            self.addCleanup(runner.close)
            # The cache's own guard (a plain ValueError) — verified against the
            # real class in weight_pager.py, not assumed.
            with self.assertRaises(ValueError) as ctx:
                runner.forward_logits(prompt)
            self.assertIn("larger than cache budget", str(ctx.exception))

    def test_paged_emits_load_selected_experts_event(self) -> None:
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        ids = [5, 9, 1, 17, 3, 8]
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            # Pin the per-token page-rebuild path: this test asserts the
            # ``load-selected-experts`` event contract of that path specifically.
            # The persistent-table path (default) emits ``assemble-experts-*``
            # events instead and is covered by NemotronPersistentExpertTableTests.
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                persist_expert_tables=False,
            )
            self.addCleanup(runner.close)
            events: list[dict] = []
            runner._stream_forward_tokens(
                [ids], cache=runner.session.model.make_cache(), events=events,
                pass_kind="prefill",
            )
            loads = [
                e
                for e in events
                if e.get("action") == "load-selected-experts"
            ]
            # two MoE layers (block_type 'E') in the tiny pattern
            self.assertEqual(len(loads), 2)
            for ev in loads:
                self.assertEqual(ev["kind"], "load")
                self.assertGreaterEqual(ev["expert_count"], 1)
                self.assertLessEqual(ev["expert_count"], 8)

    def test_emits_load_telemetry(self) -> None:
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)
            # Old page-rebuild path: this test asserts the per-token
            # ``load-selected-experts`` load event + summary keys of that path.
            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                persist_expert_tables=False,
            )
            self.addCleanup(runner.close)
            events: list[dict] = []
            cache = runner.session.model.make_cache()
            runner._stream_forward_tokens(
                [[5, 9, 1]], cache=cache, events=events, pass_kind="prefill"
            )
            loads = [e for e in events if e["kind"] == "load"]
            self.assertTrue(
                any(e.get("action") == "load-selected-experts" for e in loads),
                "no selective-expert load event",
            )
            self.assertTrue(
                all("layer" in e for e in loads),
                "load event missing 'layer'",
            )
            # resident-path loads (non-MoE layers) now also emit events:
            self.assertTrue(
                any(
                    e.get("action") in ("load-layer-full", "load-layer-base")
                    for e in loads
                ),
                "no base/full load event",
            )
            summary = runner.run_summary(events, tokens=3, elapsed_s=1.5)
            for k in (
                "loaded_bytes",
                "load_seconds",
                "load_events",
                "resident_bytes",
                "peak_rss_bytes",
                "tokens",
                "tok_per_s",
            ):
                self.assertIn(k, summary, f"summary missing {k}")
            self.assertEqual(summary["tok_per_s"], 2.0)

    def test_run_summary_reports_weight_page_stats(self) -> None:
        """run_summary surfaces the resident expert cache's live hit/miss signal.

        The Studio measurement reads the cache signal from ``run_summary``, so the
        two getters on the loader (``weight_page_resident_bytes`` /
        ``weight_page_summary``) must be additively exposed. The decisive proof is
        that the cache actually HITS: a single forward is all first-touch misses,
        so we run a multi-token greedy decode and require ``hits > 0`` (repeat
        touches across steps) AND ``misses > 0`` (first touches). The toy's gate
        biases the routed union of every MoE layer to a stable subset of {2,5,7},
        so the same (layer, expert) row pages recur step-to-step and HIT.

        Mirrors the real model with the 4-bit quantized fixture. The additive keys
        must also degrade cleanly to ``None`` / ``0`` with no budget (cache off).
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
        from smarttensor.manifest import SmartTensorManifest

        original_keys = (
            "loaded_bytes",
            "load_seconds",
            "load_events",
            "resident_bytes",
            "peak_rss_bytes",
            "tokens",
            "tok_per_s",
        )
        prompt_ids = [5, 9, 1, 17, 3, 8]
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)

            # Budget = whole toy model on disk: a comfortable over-estimate of
            # just the stacked expert tables, so every routed row page fits and
            # NOTHING is ever evicted (repeat touches are guaranteed hits).
            man = SmartTensorManifest.from_safetensors(
                [str(Path(path) / "model.safetensors")]
            )
            total_bytes = man.total_bytes

            # --- Cache ON: multi-token decode so the same expert pages recur.
            # This asserts the weight-page cache HITS on repeat row touches
            # across decode steps, which only happens when each step re-requests
            # the rows from the cache — the per-token page-rebuild path. The
            # persistent-table path serves repeats from its own assembled table
            # (the point of this change) and would not re-touch the cache, so
            # pin the old path to keep measuring the cache's hit/miss signal.
            runner_on = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                persist_expert_tables=False,
                weight_page_budget_bytes=total_bytes,
            )
            self.addCleanup(runner_on.close)
            result = runner_on.generate_greedy(prompt_ids, max_tokens=8)
            summary = result["summary"]

            # Original keys all still present (purely additive change).
            for k in original_keys:
                self.assertIn(k, summary, f"summary dropped original key {k}")

            # Additive keys present with the right shapes.
            self.assertIn("weight_page_resident_bytes", summary)
            self.assertIn("weight_page_summary", summary)
            self.assertIsInstance(summary["weight_page_resident_bytes"], int)
            self.assertGreater(
                summary["weight_page_resident_bytes"],
                0,
                "experts loaded but resident_bytes is 0",
            )
            page_summary = summary["weight_page_summary"]
            self.assertIsInstance(page_summary, dict)
            for k in ("hits", "misses", "hit_rate"):
                self.assertIn(k, page_summary, f"weight_page_summary missing {k}")

            # The cache is genuinely LIVE: first touches miss, repeat touches hit.
            self.assertGreater(
                page_summary["misses"], 0, "no first-touch misses recorded"
            )
            self.assertGreater(
                page_summary["hits"],
                0,
                "cache never hit across decode steps (inert cache)",
            )
            requests = page_summary["hits"] + page_summary["misses"]
            self.assertAlmostEqual(
                page_summary["hit_rate"],
                page_summary["hits"] / requests,
                msg="hit_rate inconsistent with hits/misses",
            )

            # --- Cache OFF: additive keys degrade to None / 0.
            runner_off = NemotronHStreamingForwardRunner(
                str(path), pin_policy="all", page_experts=True
            )
            self.addCleanup(runner_off.close)
            off = runner_off.generate_greedy(prompt_ids, max_tokens=4)
            off_summary = off["summary"]
            for k in original_keys:
                self.assertIn(k, off_summary, f"cache-off dropped original key {k}")
            self.assertIsNone(
                off_summary["weight_page_summary"],
                "cache-off weight_page_summary must be None",
            )
            self.assertEqual(
                off_summary["weight_page_resident_bytes"],
                0,
                "cache-off weight_page_resident_bytes must be 0",
            )


class NemotronPagedDropMmapCacheTests(unittest.TestCase):
    """The weight-page-cache read path must honor ``drop_mmap_cache_after_read``.

    On macOS ``madvise(MADV_DONTNEED)`` is a NO-OP, so the prior fix (commit
    dd85ac7: call ``drop_tensor_cache`` on the cache branch when the flag is set)
    did NOT actually release the mmap'd shard pages faulted in by cache-MISS
    expert reads — file-backed RSS still accumulated alongside the owned MLX
    copy (~2x working set) until jetsam killed the process. The real fix reads
    the selected expert rows via ``os.pread`` into a TRANSIENT buffer (freed
    right after the ``mx.array`` copy) so NO persistent mmap pages accumulate at
    all. These tests pin that fix: when the flag is set the cache path reads
    through ``pread_range`` (not the mmap), it no longer needs the now-redundant
    ``drop_tensor_cache`` call on that path, it stays bit-identical to stock, and
    the pread substitution stays gated on the opt-in flag (default mmap reader).
    """

    PROMPT = [5, 9, 1, 17, 3, 8]

    def _spy_drop_tensor_cache(self):
        """Counter for switch_mlp ``drop_tensor_cache`` calls, delegating to real.

        The resident/base-layer loads (``load_tensors``) honor the drop flag on
        their OWN non-cache branch, so a blanket counter would be > 0 even when
        the expert weight-page-cache branch never drops. We count ONLY the
        experts' stacked ``switch_mlp`` tensors. With the pread fix this stays 0
        on the cache path even when the flag is set: pread leaves no mmap pages,
        so the drop is unnecessary and is not called.
        """
        from smarttensor.safetensors import SafeTensorFile

        original = SafeTensorFile.drop_tensor_cache
        counter = {"calls": 0}

        def counting_drop(self, name, *, _orig=original, _counter=counter):
            if "switch_mlp" in name:
                _counter["calls"] += 1
            return _orig(self, name)

        SafeTensorFile.drop_tensor_cache = counting_drop
        self.addCleanup(setattr, SafeTensorFile, "drop_tensor_cache", original)
        return counter

    def _spy_pread_range(self):
        """Counter for ``SafeTensorFile.pread_range`` calls, delegating to real.

        The pread expert reader is the ONLY caller of ``pread_range`` on this
        path, so a > 0 count proves the routed-expert rows were read via pread
        (no mmap residency) rather than the mmap ``tensor()`` slice. With the
        flag set this fires > 0; with the flag off it stays 0 (mmap reader).
        """
        from smarttensor.safetensors import SafeTensorFile

        original = SafeTensorFile.pread_range
        counter = {"calls": 0}

        def counting_pread(self, start, end, *, _orig=original, _counter=counter):
            _counter["calls"] += 1
            return _orig(self, start, end)

        SafeTensorFile.pread_range = counting_pread
        self.addCleanup(setattr, SafeTensorFile, "pread_range", original)
        return counter

    def _run_cache_forward(self, *, drop_flag):
        """Run one cache-ON forward with the flag set; return logits + counts.

        Budget = whole toy model on disk, so every routed row page is admitted
        and the cache is genuinely attached (cache-ON path), exactly like the
        existing cache-exactness test. Returns
        ``(logits, drop_calls, pread_calls, stock)``.
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
        from smarttensor.manifest import SmartTensorManifest

        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron_quantized(d)
            stock = _stock_logits(path, self.PROMPT)

            man = SmartTensorManifest.from_safetensors(
                [str(Path(path) / "model.safetensors")]
            )
            drop_counter = self._spy_drop_tensor_cache()
            pread_counter = self._spy_pread_range()

            runner = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                weight_page_budget_bytes=man.total_bytes,
            )
            self.addCleanup(runner.close)
            # The weight-page cache is genuinely attached on this path.
            self.assertIsNotNone(
                runner.session.loader._weight_pages,
                "cache-on path did not attach a weight-page cache",
            )
            runner.session.loader.drop_mmap_cache_after_read = drop_flag

            got = runner.forward_logits(self.PROMPT)
            mx.eval(got)
            return got, drop_counter["calls"], pread_counter["calls"], stock

    def test_cache_path_reads_via_pread_when_flag_set(self) -> None:
        """Flag set: cache-ON expert reads go through ``pread_range`` (no mmap).

        The pread reader is the only ``pread_range`` caller on this path, so a
        > 0 count proves the routed-expert rows were read transiently (no mmap
        residency accumulates). Because pread leaves no mmap pages, the now
        redundant ``drop_tensor_cache`` is NOT called on this path — the spy for
        switch_mlp drops stays at zero.
        """
        _got, drop_calls, pread_calls, _stock = self._run_cache_forward(
            drop_flag=True
        )
        self.assertGreater(
            pread_calls,
            0,
            "weight-page-cache read path did not use pread when the flag is set "
            "(mmap residency would still accumulate -> macOS jetsam)",
        )
        self.assertEqual(
            drop_calls,
            0,
            "pread path should not call drop_tensor_cache (it is redundant): "
            "pread leaves no mmap pages to drop",
        )

    def test_cache_path_with_pread_is_token_exact(self) -> None:
        """Critical safety gate: reading via pread does NOT corrupt the result.

        ``load_native_mlx_array_first_dim_indices_pread`` is byte-identical to
        the mmap reader (np.frombuffer of the SAME bytes -> mx.array, same
        ``.view(mx.bfloat16)``). The SAME cache-ON + flag-set forward must stay
        bit-identical to stock mlx_lm (0.0, not approximately).
        """
        import mlx.core as mx

        got, _drop_calls, pread_calls, stock = self._run_cache_forward(
            drop_flag=True
        )
        # Sanity: the pread path actually fired on this exact forward.
        self.assertGreater(
            pread_calls, 0, "pread did not fire on the exact-gate forward"
        )
        self.assertEqual(tuple(got.shape), tuple(stock.shape))
        self.assertEqual(
            float(
                mx.max(
                    mx.abs(got.astype(mx.float32) - stock.astype(mx.float32))
                )
            ),
            0.0,
            "pread expert reader corrupted the result (logits moved off stock)",
        )

    def test_cache_path_uses_mmap_reader_when_flag_off(self) -> None:
        """Flag OFF (default): the cache path must NOT use pread.

        Models that do not opt in keep the prior behavior — the mmap reader, so
        ``pread_range`` is never called and no switch_mlp pages are dropped.
        """
        _got, drop_calls, pread_calls, _stock = self._run_cache_forward(
            drop_flag=False
        )
        self.assertEqual(
            pread_calls,
            0,
            "cache path used pread even though the flag is off",
        )
        self.assertEqual(
            drop_calls,
            0,
            "cache path dropped switch_mlp mmap pages even though the flag is off",
        )


class FakeGate:
    """Minimal stock-gate stub: returns fixed global ids + pass-through scores."""

    def __init__(self, indices, scores) -> None:
        import mlx.core as mx

        self._indices = mx.array(indices)
        self._scores = mx.array(scores)

    def __call__(self, x):
        return self._indices, self._scores


class NemotronHotsetGateAdapterTests(unittest.TestCase):
    def test_remaps_non_identity_hotset(self) -> None:
        import numpy as np
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHotsetGateAdapter

        # Hotset order [2, 5, 7] maps global->slot as 2->0, 5->1, 7->2.
        # Routed global ids [5, 2, 7] must therefore remap to local [1, 0, 2]
        # (a genuine non-identity permutation), with scores untouched.
        scores = [[0.6, 0.3, 0.1]]
        gate = FakeGate([[5, 2, 7]], scores)
        adapter = NemotronHotsetGateAdapter(gate, hotset=[2, 5, 7], strict=True)

        local, out_scores = adapter([[0.0]])
        self.assertEqual(np.asarray(local).reshape(-1).tolist(), [1, 0, 2])
        # scores must pass through unchanged (bit-identical)
        self.assertEqual(
            float(mx.max(mx.abs(out_scores - mx.array(scores)))), 0.0
        )

    def test_strict_miss_raises(self) -> None:
        from smarttensor.adapters.mlx import (
            NemotronHotsetGateAdapter,
            NemotronHotsetMiss,
        )

        # routed id 9 is outside the hotset -> strict miss
        gate = FakeGate([[2, 9]], [[0.5, 0.5]])
        adapter = NemotronHotsetGateAdapter(gate, hotset=[2, 5, 7], strict=True)
        with self.assertRaises(NemotronHotsetMiss) as ctx:
            adapter([[0.0]])
        self.assertEqual(ctx.exception.missing_experts, (9,))


def _spy_load_selected_experts(testcase, runner):
    """Count ``_load_selected_experts`` calls on ``runner`` (delegates to real).

    Returns a ``{"calls": int}`` dict that increments once per real selective
    load (the per-token compact-table rebuild on the OLD path; the missing-only
    append on the NEW persistent-table path). The counter distinguishes the two
    regimes: the old path calls it on EVERY MoE layer EVERY token, the new path
    calls it only when routing leaves the resident membership.
    """
    from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

    original = NemotronHStreamingForwardRunner._load_selected_experts
    counter = {"calls": 0}

    def counting(self, *args, _orig=original, _counter=counter, **kwargs):
        _counter["calls"] += 1
        return _orig(self, *args, **kwargs)

    NemotronHStreamingForwardRunner._load_selected_experts = counting
    testcase.addCleanup(
        setattr,
        NemotronHStreamingForwardRunner,
        "_load_selected_experts",
        original,
    )
    return counter


class NemotronPersistentExpertTableTests(unittest.TestCase):
    """Persistent per-layer assembled expert table: exact AND not-rebuilt-every-token.

    The persistent table holds the SAME expert weights as the old per-token
    page-rebuild path — just assembled once and reused across decode steps,
    appended-to only when routing leaves the resident membership, and LRU-evicted
    under a per-layer cap. The forward math is unchanged, so the gate is hard
    token-exactness: bit-identical to (a) stock mlx_lm and (b) the old
    page-rebuild path, across reuse + extension + eviction, on BOTH fixtures.
    """

    # Single-token prompt: prefill assembles a NARROW membership ({2,7}) per MoE
    # layer; a later decode step routes to expert 5 (membership extension), and
    # all remaining steps are subsets of {2,5,7} (pure reuse). With a cap of 2,
    # admitting expert 5 over-fills the layer and forces an LRU eviction. So one
    # multi-token decode exercises reuse, extension AND eviction in one shot.
    PROMPT = [5]
    STEPS = 12

    def _stock_greedy(self, path, prompt, n):
        import mlx.core as mx

        from smarttensor.adapters.mlx import build_mlx_model_shell
        from smarttensor.manifest import SmartTensorManifest

        cfg = json.loads((Path(path) / "config.json").read_text())
        man = SmartTensorManifest.from_safetensors(
            [str(Path(path) / "model.safetensors")]
        )
        m = build_mlx_model_shell(cfg, man)
        w = m.sanitize(mx.load(str(Path(path) / "model.safetensors")))
        m.load_weights(list(w.items()))
        cache = m.make_cache()
        cur = mx.array([prompt])
        out: list[int] = []
        for _ in range(n):
            lg = m(cur, cache=cache)
            mx.eval(lg)
            nxt = int(mx.argmax(lg[:, -1, :], axis=-1)[0])
            out.append(nxt)
            cur = mx.array([[nxt]])
        return out

    def test_persistent_table_decode_exact_reuse_extend_evict(self) -> None:
        """generate_greedy: persistent-table tokens == stock == old page-rebuild path.

        Run on BOTH fixtures with a tiny per-layer cap (2) so the single decode
        run forces table reuse, membership extension AND LRU eviction — and the
        emitted token list is still bit-for-bit identical to stock mlx_lm AND to
        the old per-token page-rebuild path. Decode-exactness (identical token
        list across all steps) is the gate: any drift in the assembled weights,
        the membership remap, or the eviction would change a downstream argmax.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = self._stock_greedy(path, self.PROMPT, self.STEPS)

                    # Old path: per-token compact-table rebuild (reference).
                    old = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        persist_expert_tables=False,
                    )
                    self.addCleanup(old.close)
                    old_tokens = old.generate_greedy(self.PROMPT, self.STEPS)[
                        "tokens"
                    ]

                    # New path: persistent per-layer assembled table, tiny cap.
                    new = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        persist_expert_tables=True,
                        max_resident_experts_per_layer=2,
                    )
                    self.addCleanup(new.close)
                    new_tokens = new.generate_greedy(self.PROMPT, self.STEPS)[
                        "tokens"
                    ]

                    self.assertEqual(
                        new_tokens,
                        stock,
                        "persistent-table decode diverged from stock mlx_lm",
                    )
                    self.assertEqual(
                        new_tokens,
                        old_tokens,
                        "persistent-table decode diverged from old page-rebuild path",
                    )

                    # Eviction genuinely fired: with cap 2 and a union of 3
                    # experts ({2,5,7}) per MoE layer, the table must have evicted.
                    stats = new._assembled_stats
                    self.assertGreater(
                        stats["evictions"],
                        0,
                        "cap=2 over a {2,5,7} union never evicted (test is not "
                        "exercising the eviction path)",
                    )
                    # Extension genuinely fired: prefill membership was narrower
                    # than the eventual union, so at least one decode-step append
                    # happened beyond the initial assembly.
                    self.assertGreater(
                        stats["append_passes"],
                        0,
                        "membership never extended (no append observed)",
                    )

    def test_persistent_table_forward_logits_exact(self) -> None:
        """forward_logits: persistent-table prefill is bit-identical to stock + old path.

        A single prefill cannot reuse across tokens, but it must still produce
        EXACTLY the stock logits with the persistent table active (the table is
        assembled once during prefill). Checked on both fixtures, 0.0 vs stock
        and vs the old page-rebuild path.
        """
        import mlx.core as mx

        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        ids = [5, 9, 1, 17, 3, 8]
        for builder in (build_tiny_nemotron, build_tiny_nemotron_quantized):
            with self.subTest(fixture=builder.__name__):
                with tempfile.TemporaryDirectory() as d:
                    path = builder(d)
                    stock = _stock_logits(path, ids)

                    old = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        persist_expert_tables=False,
                    )
                    self.addCleanup(old.close)
                    old_logits = old.forward_logits(ids)
                    mx.eval(old_logits)

                    new = NemotronHStreamingForwardRunner(
                        str(path),
                        pin_policy="all",
                        page_experts=True,
                        persist_expert_tables=True,
                    )
                    self.addCleanup(new.close)
                    new_logits = new.forward_logits(ids)
                    mx.eval(new_logits)

                    self.assertEqual(
                        tuple(new_logits.shape), tuple(stock.shape)
                    )
                    self.assertEqual(
                        float(
                            mx.max(
                                mx.abs(
                                    new_logits.astype(mx.float32)
                                    - stock.astype(mx.float32)
                                )
                            )
                        ),
                        0.0,
                        "persistent-table prefill diverged from stock",
                    )
                    self.assertEqual(
                        float(
                            mx.max(
                                mx.abs(
                                    new_logits.astype(mx.float32)
                                    - old_logits.astype(mx.float32)
                                )
                            )
                        ),
                        0.0,
                        "persistent-table prefill diverged from old page path",
                    )

    def test_persistent_table_does_not_rebuild_every_token(self) -> None:
        """No-rebuild proof: stable-membership decode loads << old per-token path.

        The toy's gate biases every MoE layer's routed union to a stable subset
        of {2,5,7}. With a generous cap the persistent table assembles that union
        once (during prefill + the first couple of decode steps) and then SERVES
        every later step with no load. Counting real ``_load_selected_experts``
        calls:

        * OLD path loads on EVERY MoE layer EVERY pass: 2 layers x (1 prefill +
          STEPS-1 decode) calls — it grows linearly with the token count.
        * NEW path loads only while membership is still being discovered, then
          stops. So new-path loads are FAR fewer than old-path loads, and the
          per-token decode load rate collapses toward zero after warmup.
        """
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        prompt = [5, 9, 1, 17, 3, 8]
        steps = 10
        with tempfile.TemporaryDirectory() as d:
            path = build_tiny_nemotron(d)

            # --- OLD path: rebuilds the compact table every MoE layer every pass.
            old_counter = _spy_load_selected_experts(self, None)
            old = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                persist_expert_tables=False,
            )
            self.addCleanup(old.close)
            old.generate_greedy(prompt, steps)
            old_calls = old_counter["calls"]

            # --- NEW path: persistent table, generous cap (no eviction churn).
            new_counter = _spy_load_selected_experts(self, None)
            new = NemotronHStreamingForwardRunner(
                str(path),
                pin_policy="all",
                page_experts=True,
                persist_expert_tables=True,
                max_resident_experts_per_layer=8,
            )
            self.addCleanup(new.close)
            new.generate_greedy(prompt, steps)
            new_calls = new_counter["calls"]

            # The old path loads on every MoE layer every pass (2 layers x #passes).
            passes = 1 + (steps - 1)  # prefill + decode steps
            self.assertEqual(
                old_calls,
                2 * passes,
                "old path is expected to rebuild every MoE layer every pass",
            )
            # The new path loads only during membership discovery, then stops:
            # one assembly per layer's narrow union and at most a couple of
            # appends — strictly and dramatically fewer than the old path.
            self.assertLess(
                new_calls,
                old_calls,
                "persistent table did not reduce loads vs per-token rebuild",
            )
            # Concretely: at most one load per MoE layer per DISTINCT expert in
            # its union (3) — never proportional to the token count.
            self.assertLessEqual(
                new_calls,
                2 * 3,
                f"persistent table loaded too often ({new_calls}); membership "
                "is stable so loads should not scale with tokens",
            )
            # And the table really was reused (most passes served with no load).
            new_stats = new._assembled_stats
            self.assertGreater(
                new_stats["hit_passes"],
                new_stats["append_passes"],
                "persistent table was not predominantly served from residency",
            )


if __name__ == "__main__":
    unittest.main()
