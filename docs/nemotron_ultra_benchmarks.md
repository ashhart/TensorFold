# Nemotron-3-Ultra-550B — Measured Benchmarks

Honest record of every measured decode run of the
`Nemotron-3-Ultra-550B-A55B-4bit` streaming runner
(`src/smarttensor/adapters/mlx.py::NemotronHStreamingForwardRunner`).

**Read the rules below before quoting any number.** Several figures in our
history are leaky (same-prompt re-decode) or invalid (oracle drafter). They are
recorded here *because* they are leaky — to keep the lie out of the headline,
not to bury it. Every leaky/invalid row is labelled inline.

---

## 0. Exactness vocabulary (what each "mode" means)

| Mode | Definition | How verified |
|---|---|---|
| **BIT-EXACT** | Argmax-identical to stock `mlx_lm` at every position; tensors max-abs-diff `0.0` on the toy. | `tests/test_nemotron_runner_*.py` against `tests/fixtures/tiny_nemotron.py` (`build_tiny_nemotron` / `build_tiny_nemotron_quantized`). Production: token sequence equals plain greedy. |
| **NEAR-EXACT** | Output drifts on tokens that route to a cold (non-resident) expert; all-resident tokens stay bit-exact. Drift ~0.1% PPL-class; output must remain **coherent** on unseen prompts. | Coherence + scoped-drift A/B, *not* `0.0`. Cold-routed tokens substitute a buddy (see §4). |
| **LEAKY** | A real, exact mechanism measured under a setup that inflates the number — chiefly **same-prompt re-decode** (the tokens were already produced once, so the run is not fresh work). | Flagged inline; never a headline. |
| **INVALID (oracle)** | Speed produced by an **oracle drafter**: the draft tape *is* the generation's own greedy output, so acceptance is 100% by construction. Measures nothing. | Flagged inline; **must never** be cited as a real number. |

**Discipline:** a speed number is only "real" if the work was *fresh* (the tokens
were not already known to the harness) and the exactness mode is stated. A
bit-exact feature must show `0.0` max-abs-diff on the toy fixture; a near-exact
feature must show coherent output under distribution shift, not `0.0`.

---

## 1. The honest speed ladder (headline table)

Studio class: M3 Ultra, 256 GB unified, ~819 GB/s. Model on disk: 323 GB total
(~298 GB experts; ~12 MB/expert at 4-bit, latent-dim). 55B active params ⇒
~27.5 GB active 4-bit weight read/token; base (embeddings + lm_head + norm +
per-layer gate/latent/shared/mamba/attn) ~25 GB.

| # | Config | Exactness | Fresh? | tok/s | Status |
|---|---|---|---|---|---|
| L0 | Cold baseline, selective page load, no cache | BIT-EXACT | yes | **~0.30** | REAL (floor) |
| L1 | Stage-2 adaptive weight-page cache | BIT-EXACT | yes | **~2.0** | **REAL** — fresh-generation bit-exact number |
| L2 | Deferred fixed-hot-set, no substitution | BIT-EXACT | **NO — same-prompt re-decode** | ~16.3 | **LEAKY** — do not headline |
| L3 | Deferred fixed-hot-set **+ cold→buddy substitution** | NEAR-EXACT | **yes (unseen/shifted prompts)** | **~16.5–16.7** | **REAL** — the genuine fast number |
| L4a | Speculation, block-verify, **oracle drafter** | (verify is exact) | **NO — oracle tape = own output** | 22.76 | **INVALID (oracle)** |
| L4b | Speculation, block-verify, **oracle drafter** | (verify is exact) | **NO — oracle tape = own output** | 41 | **INVALID (oracle)** |
| L4c | Speculation, block-verify, **oracle drafter** | (verify is exact) | **NO — oracle tape = own output** | 56.9 | **INVALID (oracle)** |

**The two numbers you are allowed to quote as real:**
- **~2.0 tok/s bit-exact** (L1) on fresh generation.
- **~16.5–16.7 tok/s near-exact** (L3) on fresh, unseen prompts, output coherent.

**Real-drafter speculation is UNMEASURED.** L4a–c are an oracle artefact (see §5).
A real speculation number requires a real drafter on held-out prompts and does
not yet exist.

---

## 2. L0 — Cold baseline (BIT-EXACT, REAL)

- **Config:** `NemotronHStreamingForwardRunner`, selective per-(layer,expert)
  page load from mmap'd shards (`pread`, `drop_mmap_cache_after_read=True`), no
  resident expert cache, no fixed hot-set, no deferred path.
- **Exactness:** BIT-EXACT (selective loader slices the stacked quantized
  `switch_mlp.{fc1,fc2}.{weight,scales,biases}` by routed expert rows; math
  unchanged vs stock).
- **Prompt / tokens:** short prompt, 16-token decode.
- **Warm/cold:** cold (no cache warmed).
- **Resident budget:** none beyond base (~25 GB) + the per-token routed expert
  slice.
- **tok/s:** **~0.30**.
- **Why this slow (measured, from Stage-2 plan):** 91% of wall time is SSD
  streaming — load 83 s of 91.8 s for the 16-token decode. **~17 GB streamed per
  token** at ~3.3 GB/s read tier, with a **reload multiplier ~3.06×** (each
  routed expert read ~3× then discarded across the window). This is the
  un-cached, redundant-read floor.

---

## 3. L1 — Stage-2 adaptive weight-page cache (BIT-EXACT, REAL)

- **Config:** L0 plus the resident weight-page cache
  (`PagedWeightCache` via `attach_weight_page_cache`), LRU policy
  (`resolve_weight_page_policy('nemotron_h','auto')` → `lru`), target cache
  ~150 GiB so the ~82 GB working set (top-22/512 × ~48 MoE layers, ~131 unique
  experts/layer over the window) never evicts a hot row.
- **Exactness:** BIT-EXACT. The cache returns the identical bytes; output token
  sequence equals plain greedy.
- **Warm/cold:** fresh generation. First-token fill still pays cold reads;
  steady decode hits RAM.
- **Resident budget:** base ~25 GB + ≤150 GB cache ⇒ ~175–185 GB peak,
  well under the 256 GB box.
- **tok/s:** **~2.0** on fresh generation.
- **Mechanism win:** collapses the 3.06× reload multiplier toward ~1× on the
  82 GB working set. This is un-redundant streaming, not magic — it cannot make a
  323 GB model *fast* (the bytes still move), it just stops reading each expert
  three times. Headline-eligible: fresh + bit-exact.
- **Telemetry to trust:** `loader.weight_page_summary()`
  (hits/misses/hit_rate/evictions) and `loader.weight_page_resident_bytes`.
  Do **not** read `run_summary.loaded_bytes` for the hit signal — it is computed
  from the full selected set per load and does **not** drop on a cache hit
  (`load_first_dim_slices` computes `nbytes` from the full set regardless).

---

## 4. L2 / L3 — Deferred fixed-hot-set (and buddy substitution)

Both build, **once**, a per-MoE-layer stacked `switch_mlp` of the top-K
warmup-frequency experts (`build_fixed_hotset`, default `K=200`,
~110 GB resident) and run the model's own native on-GPU gather. The deferred
path (`_stream_forward_tokens_deferred`) removes every per-layer eval: the
global→slot remap is done on-GPU (`mx.take` of a prebuilt `g2s` lookup),
residency is checked on-GPU (a lazy OR of per-layer cold flags), and the whole
forward composes into ONE graph evaluated once at `lm_head` (~1–2 evals/token).

### L2 — deferred fixed-hot-set, NO substitution (BIT-EXACT, **LEAKY**)

- **Config:** fixed hot-set + deferred path, `cold_substitution=False`.
- **Exactness:** BIT-EXACT. A routed expert outside the resident set maps to the
  **sentinel**, trips the on-GPU cold flag, and forces a full-token rollback +
  exact synced redo. So the output is byte-identical — but every cold-routed
  token costs a redo.
- **Fresh?** **NO.** The ~16.3 tok/s figure was measured on a **same-prompt
  re-decode**: the tokens were already produced, so the residency/route pattern
  was effectively pre-conditioned and cold redos were suppressed. On a *fresh*
  prompt the pure-exact deferred path goes cold often and the redos dominate, so
  this rate does **not** hold for new work.
- **Status:** **LEAKY.** Records the exact-path ceiling under re-decode; not a
  headline. The honest exact number for fresh work is L1 (~2.0).

### L3 — deferred fixed-hot-set + cold→buddy substitution (NEAR-EXACT, REAL)

- **Config:** fixed hot-set + deferred path, `cold_substitution=True`.
- **Mechanism:** at build, `_compute_buddy_map` maps each non-resident expert to
  its nearest **resident buddy slot** — the resident expert whose `gate.weight`
  row is most cosine-similar. `g2s[cold] = buddy_slot`, so every global id maps
  to a *valid resident slot*, the `local == sentinel` signal never fires, and the
  deferred path **never redoes** → fast on **any** generation.
- **Exactness:** **NEAR-EXACT.** The routed cold expert's gate *score* is
  unchanged; only *which* expert computes that slot changes (cold → buddy). So
  the output drifts ~0.1% (PPL-class) on tokens that route to a cold expert.
  All-resident tokens are still **bit-exact** (substitution is a no-op for
  resident ids). This is *not* a `0.0`-diff feature and must not be claimed as
  one.
- **Fresh?** **YES** — measured on fresh, unseen / distribution-shifted prompts
  (see §6).
- **tok/s:** **~16.5–16.7** on fresh generation, output verified coherent.
- **Status:** **REAL — the genuine fast number.** This is the headline fast
  figure: fresh work, coherent output, ~0.1% scoped drift.
- **Telemetry:** `_deferred_stats["cold_substitutions"]` > 0 confirms
  substitution fired; `cold_redos` should be ~0 on the substitution path;
  `hotset_native_hits` vs `hotset_cold_fallbacks` in `run_summary`.

---

## 5. L4 — Speculation (INVALID: oracle drafter)

- **Config:** `generate_greedy_speculative_deferred` (block-verify framework,
  batched verify of a whole block in one `_stream_forward_tokens` pass,
  full-accept-skip), on top of the deferred fixed-hot-set.
- **The verify logic is genuinely token-exact.** A block is verified in one
  batched forward, the longest matching greedy prefix is accepted, a bonus token
  is emitted, and the cache is rolled back + re-forwarded over exactly the
  committed tokens (the only provably-exact way to rewind the non-trimmable
  Mamba SSM state). Returned tokens are bit-for-bit plain greedy. **That is not
  the problem.**
- **The problem — the DRAFTER:** the draft tape used in these runs **was the
  generation's own greedy output**. Feeding the answer back as the proposal makes
  acceptance **100% by construction** — every block is fully accepted, the
  commit re-forward is skipped, and the measured rate is an artefact of perfect
  foreknowledge. **This is leakage.**
- **Numbers (ALL INVALID, recorded only to label them):** 22.76, 41, 56.9 tok/s.
- **Status:** **INVALID (oracle).** None of these is a real speed. Do **not**
  present "41–57 tok/s" (or any of them) as a Nemotron result, in any summary,
  ever.
- **What a real number needs:** a *real* drafter (small model / trained head /
  prompt-lookup) proposing tokens it has **not** already seen, on **held-out**
  prompts, with acceptance measured. **Real-drafter speculation is currently
  UNMEASURED.** See `docs/nemotron_ultra_next_steps.md` #1.

---

## 6. Quality under distribution shift (supports L3)

The near-exact substitution path (L3) was checked for coherence when the resident
hot-set was warmed on one prompt and generation ran on a *different*, unseen task
— i.e. the routed experts genuinely fall outside the warmed set and the buddy
substitution actually fires:

- **Warm-up prompt:** "capital of France" (correctly answers Paris-class).
- **Held-out generations (different distribution), all ~17 tok/s, coherent:**
  - asked for **factorial code** → produced correct factorial code;
  - asked about **colors** → coherent color answer;
  - asked about the **sky** → "sky-blue"-class correct answer.

Conclusion: the ~0.1% drift from cold→buddy substitution stays **coherent under
distribution shift**, supporting L3 as a real near-exact fast number — *not* a
same-prompt artefact. This is qualitative coherence evidence, not a `0.0`-diff
exactness claim.

---

## 7. Memory / hardware envelope (context for all rows)

- **Disk:** 323 GB total; ~298 GB experts; ~12 MB/expert (4-bit fc1+fc2+scales+
  biases, latent dim).
- **Active read/token:** ~27.5 GB (4-bit, 55B active params).
- **Base resident:** ~25 GB (embeddings + lm_head + norm + per-layer
  gate/latent/shared/mamba/attn).
- **Fixed hot-set K=200:** ~110 GB resident.
- **Bandwidth floor:** M3 Ultra ~819 GB/s ⇒ ~34 ms/token if perfectly streamed
  from RAM.
- **Profiled decode forward:** ~54 ms FIXED per-forward overhead (kernel
  launches) + ~9–15 ms/token marginal ⇒ **launch-bound**, not (once cached)
  bandwidth-bound. This is why the wins above come from cutting evals/kernel
  launches (deferred path) and from stopping redundant reads (page cache), and
  why the next levers target launch overhead and amortization rather than raw
  bandwidth.
- **RSS:** `run_summary.peak_rss_bytes` is `ru_maxrss`, **bytes on this Darwin
  host**. Stage-2 cache rows target peak RSS < 190 GB (< 256 GB hard).

---

## 8. Reproduction notes

- Toy exactness gate: `tests/fixtures/tiny_nemotron.py`
  (`build_tiny_nemotron`, `build_tiny_nemotron_quantized`) +
  `tests/test_nemotron_runner_*.py`. Bit-exact features must show max-abs-diff
  `0.0` here.
- Real model lives on the Studio (323 GB, 109/109 shards verified). Production
  runs are dispatched via the serve path (`model_type=nemotron_h`,
  `--page-experts`).
- When recording a NEW run, fill: config, exactness mode, prompt, tokens,
  warm/cold, resident budget, tok/s, peak RSS — and **state whether the work was
  fresh.** If it was a same-prompt re-decode or used an oracle drafter, label it
  LEAKY / INVALID in the same row. No exceptions.
