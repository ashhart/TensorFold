# Nemotron-3-Ultra-550B — Ranked Next Steps

Engineering plan for the `NemotronHStreamingForwardRunner` stack on the Studio
(M3 Ultra, 256 GB, ~819 GB/s). Companion to
`docs/nemotron_ultra_benchmarks.md` — read the honesty rules there first.

**Where we are:** ~2.0 tok/s bit-exact (fresh), ~16.5–16.7 tok/s near-exact
(fresh, coherent). Decode is **launch-bound** (~54 ms fixed per-forward kernel
overhead + ~9–15 ms/token marginal), not bandwidth-bound once the working set is
resident. The fast number already exists; the open frontier is (a) turning
speculation into a *real* number, (b) holding speed as context grows, and (c)
shrinking the resident/peak envelope.

Each item below states: **expected win**, **risk**, **validation command**,
**stop condition**. "Validation command" points at the real exactness/eval gate
that must pass; the headline speed number itself is a Studio measurement (the
toy fixtures gate *correctness*, the Studio gates *speed*).

---

## #1 — Real-drafter speculation on HELD-OUT prompts  ← top priority

**Why first:** the entire 22.76 / 41 / 56.9 tok/s "speculation" history is an
**oracle artefact** (the draft tape was the generation's own output ⇒ 100%
accept by construction). Real-drafter speculation is currently **UNMEASURED**.
The verify machinery (`generate_greedy_speculative_deferred`, batched
block-verify, full-accept-skip, exact rollback over the Mamba SSM state) is
already proven token-exact in tests — the **only** missing piece is a drafter
that proposes tokens it has not already seen, scored on prompts the harness has
not generated.

- **Expected win:** honest. Speculation pays as a *multiplier on top of
  residency*, and only when per-token cost is dominated by fixed per-forward
  overhead / expert load (our launch-bound regime fits). Realistic acceptance
  for a small/prompt-lookup drafter on held-out text is modest (block accept
  length ~1.3–2.x), so expect a **real** multiplier well below the oracle's —
  plausibly **1.2–1.8× over L3** if a drafter clears net-positive, possibly
  net-zero/negative if acceptance is low. The point is to **get the true number**,
  not to beat the fake one.
- **Risk:** (a) acceptance too low ⇒ the commit re-forward overhead makes
  speculation a net loss (documented in `generate_greedy_speculative`'s own
  performance-honesty note); (b) the route-union over a draft block widens the
  cold-expert set, raising redos on the bit-exact path (mitigated on the
  near-exact substitution path, which never redoes); (c) a trained drafter needs
  a head/checkpoint we do not yet have for this model.
- **Validation command:**
  `python -m pytest tests/test_nemotron_runner_speculative_deferred.py tests/test_nemotron_runner_speculative.py -q`
  must stay green (verify is exact: `test_exact_quantized`,
  `test_partial_accept_still_rolls_back_exact`, `test_full_accept_skip_is_exact_and_fewer_forwards`).
  Then a Studio run: drafter trained/seeded **independently of the eval prompts**,
  decode a held-out prompt set, record accept-length mean and tok/s vs L3.
- **Stop condition:** stop and report the number whichever comes first —
  (i) measured net tok/s on held-out prompts lands within a stable band over ≥3
  runs, or (ii) acceptance-length mean ≤ ~1.1 (drafter not earning its keep) ⇒
  declare speculation net-neutral/negative at this tier and move on. **Do not**
  keep tuning to chase the oracle figure; it is unreachable by definition.

---

## #2 — Long-context KV / SSM state cost

**Why:** all current numbers are short-window decode. NemotronH is hybrid
Mamba-2 / attention (`max_position_embeddings` 262144 in the export; paper
claims 1M — **discrepancy to flag**). At length, the attention KV cache and the
non-trimmable Mamba SSM state change the per-token cost and the rollback cost in
speculation (the commit re-forward is the only exact way to rewind SSM state).
We have not measured how tok/s degrades with context.

- **Expected win:** not a speedup — a **characterization + a hardening**. Win is
  (a) a known tok/s-vs-context curve, (b) bounded peak RSS at long context (KV +
  SSM growth), (c) confirming speculation rollback stays exact and affordable as
  SSM state grows.
- **Risk:** KV/SSM growth pushes peak RSS toward the 256 GB ceiling at long
  context (the fixed hot-set already holds ~110 GB); rollback re-forward cost
  grows with committed-block SSM state and could erode speculation's win.
- **Validation command:**
  `python -m pytest tests/test_nemotron_runner_deferred.py tests/test_nemotron_runner_speculative_deferred.py -q`
  (exactness holds across the deferred + verify paths regardless of context),
  then a Studio sweep of decode at increasing prompt lengths recording tok/s and
  peak RSS at each.
- **Stop condition:** stop when the tok/s-vs-context curve is recorded out to a
  context that either (i) hits a peak-RSS guard (< 256 GB) or (ii) shows tok/s
  has stabilized/plateaued. Report the curve; do not optimize further until a
  long-context workload is actually required.

---

## #3 — Mixed-precision cold tail (drop fewer bytes, not fewer experts)

**Why:** the resident envelope is dominated by the fixed hot-set (~110 GB at
K=200) plus base (~25 GB). The cold tail (experts outside the hot-set) is what
forces either a redo (bit-exact) or a buddy substitution (near-exact drift).
Re-quantizing the **cold tail** to a lower precision shrinks the streamed/resident
bytes for exactly the experts we currently substitute, per the findings-verdict
lever ("flip the streamed wedge toward resident").

- **Expected win:** the near-fit lever. Lower-precision cold experts mean either
  a **larger resident hot-set in the same budget** (fewer cold-routed tokens ⇒
  less drift, closer to bit-exact) or **cheaper cold reads** (faster L0/L1).
  This is the documented "strong fit" for near-fit models: drop the cold wedge
  precision to keep peak under 256 GB with the working set RAM-resident.
- **Risk:** lower-precision cold experts add quality drift *on top of* buddy
  substitution — the two error sources compound; must be measured, not assumed
  ≤0.1%. Also adds a quantization/build step and a mixed-precision gather path
  (the stacked `switch_mlp` slicer currently assumes a single bit-width).
- **Validation command:** new toy fixture in the spirit of
  `tests/fixtures/tiny_nemotron.py::build_tiny_nemotron_quantized` with a
  mixed-precision tail; gate that all-resident tokens stay **bit-exact**
  (max-abs-diff `0.0`) and cold-tail tokens stay **coherent** with measured
  drift. Existing
  `python -m pytest tests/test_nemotron_runner_fixed_hotset.py tests/test_nemotron_runner_cold_substitution.py -q`
  must stay green for the unchanged-precision path.
- **Stop condition:** stop if combined drift (cold-tail precision + substitution)
  exceeds the user-accepted ~0.1% PPL band on held-out prompts, or if the
  mixed-precision gather erases the resident-budget saving via added launch
  overhead. Otherwise stop once peak RSS and drift are both recorded for one tail
  precision.

---

## #4 — Packed expert layout (cut launch overhead / read amplification)

**Why:** decode is launch-bound (~54 ms fixed/forward). On-disk experts are
stored stacked but are sliced per routed-expert row; the fixed hot-set already
pre-stacks the top-K into a single `switch_mlp` for a native on-GPU gather. A
**packed contiguous layout** of the hot-set (and of the per-load cold slice)
reduces gather/slice setup and read amplification on the page path.

- **Expected win:** marginal-but-real reduction in per-forward overhead and in
  cold-slice read cost; helps the L0/L1 bit-exact rates most (where reads
  dominate) and trims the launch tax on every path. Single-digit-percent class,
  not a step change.
- **Risk:** layout work touches the loader's slicing (`_expert_slice_names`,
  `load_first_dim_slices`) and the build of the stacked hot-set; easy to break
  the row-order invariant that makes the global→slot remap exact. Low ceiling —
  do not over-invest.
- **Validation command:**
  `python -m pytest tests/test_nemotron_layout.py tests/test_nemotron_runner_fixed_hotset.py tests/test_nemotron_runner_paged.py -q`
  (layout/slice invariants + exact forward), then a Studio A/B of L1 tok/s and
  bytes/token before vs after packing.
- **Stop condition:** stop if the layout change does not move L1 tok/s or
  bytes/token by a measurable margin over ≥3 runs, or if it threatens the
  row-order exactness invariant. This is a low-rank item; bail fast.

---

## #5 — Build-peak hardening (transient RSS during hot-set build)

**Why:** `build_fixed_hotset` reads each MoE layer's experts and the gate row
(for the buddy cosine) under `drop_mmap_cache_after_read=True`, then stacks the
top-K. The transient peak during this one-time build can spike RSS even though
steady-state residency is bounded (~110 GB hot-set + ~25 GB base). A
build-time OOM on the 256 GB box would block the whole fast path.

- **Expected win:** robustness, not speed. Bounded **build-time** peak RSS so the
  fast path is reachable with margin under 256 GB; protects #3's larger hot-sets.
- **Risk:** chunking the build adds complexity to the buddy-map computation
  (`_compute_buddy_map` needs each layer's gate rows) and could slow the one-time
  build; must not change the resulting stacked arrays (exactness).
- **Validation command:**
  `python -m pytest tests/test_nemotron_runner_fixed_hotset_memory.py tests/test_nemotron_runner_fixed_hotset.py -q`
  (the memory-bounded build test + exact-build test), then a Studio build of the
  real model recording **transient peak RSS during build** vs steady-state.
- **Stop condition:** stop once measured build-time peak RSS is comfortably under
  256 GB (target a clear margin, e.g. < ~200 GB) for the intended K, with the
  built stacks bit-identical to the un-chunked build. No further work if build
  peak already sits under budget.

---

## #6 — Route telemetry (instrument before optimizing further)

**Why:** every item above needs route facts we currently sample only loosely:
true cold-routed-token fraction on held-out prompts (drives #3's drift budget and
the hot-set K), accept-length distribution (drives #1's go/no-go), and per-layer
hit/miss occupancy (drives #4/#5). `run_summary` already exposes
`weight_page_summary()`, `hotset_native_hits` / `hotset_cold_fallbacks`, and
`_deferred_stats` (`cold_redos`, `cold_substitutions`, `deferred_tokens`) — this
item is about **using and extending** that signal, not inventing it.

- **Expected win:** no direct speedup — it **de-risks #1–#5** and prevents
  chasing the wrong lever. Turns "we think the working set is 82 GB / drift is
  0.1%" into measured per-prompt distributions.
- **Risk:** telemetry that itself adds per-token Python/eval overhead would
  poison the launch-bound decode loop. Must stay on the additive, zero-eval path
  (`run_summary` reads `kind=='load'` totals and existing counters; the deferred
  hot path deliberately skips per-layer event appends).
- **Validation command:**
  `python -m pytest tests/test_nemotron_runner_deferred_hotpath.py tests/test_nemotron_runner_deferred.py -q`
  (confirms the hot decode path stays zero-byte / zero-eval and output
  byte-identical with telemetry on), plus an assertion that added counters do
  not change the generated token sequence.
- **Stop condition:** stop once the per-prompt cold-routed fraction,
  accept-length distribution, and hot-set occupancy are recorded for the
  held-out eval set and feed #1/#3. Do not add telemetry that moves any reported
  speed metric or the decode token sequence.

---

## Cross-cutting guardrails (apply to every item)

- **Never present a same-prompt re-decode or oracle-drafter number as real.** L2
  (~16.3) is leaky; L4 (22.76/41/56.9) is invalid. The only real headline
  numbers are L1 (~2.0 bit-exact) and L3 (~16.5–16.7 near-exact), both fresh.
- **Bit-exact features gate at `0.0` max-abs-diff on the toy
  (`tests/fixtures/tiny_nemotron.py`).** Near-exact features gate on
  **coherence + scoped drift**, not `0.0`.
- **Speed is a Studio measurement; correctness is a toy-fixture measurement.** A
  validation command passing proves correctness; it does not by itself produce a
  headline tok/s — that comes from a fresh Studio run with the exactness mode
  stated.
- **Interpreter:** `python`. Real model on the
  Studio (323 GB, 109/109 shards), dispatched via the serve path
  (`model_type=nemotron_h`, `--page-experts`).
- **Config discrepancies to flag in any writeup:** export is **affine-4bit**
  (`group_size 32, bits 4`) though the paper says NVFP4 pretraining;
  `max_position_embeddings` is **262144** in the export though the paper claims
  1M; **MTP is not in the checkpoint** (config lists `num_nextn_predict_layers 1`
  + `mtp_layers_block_type`, but the 1791 exported tensors carry no mtp/nextn —
  `mlx_lm` strips it). Do not plan an MTP head against weights that are not
  present.
