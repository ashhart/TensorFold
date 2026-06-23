# Reproducing the ~16.5–17.6 tok/s Nemotron-3-Ultra-550B decode

The fast near-exact decode path for `NemotronHStreamingForwardRunner` (this directory's
`mlx.py`). Single-stream (batch-1), memory-bounded, on a 256 GB M3 Ultra.

> **Two configs, not quality-equivalent — read this first.**
> - **Override path (~17.6 tok/s, robust today):** seeds the hot-set with experts
>   `range(K)` per MoE layer. Skips the cold warmup → no SIGKILL. But **every** token
>   routes to a cold expert → buddy-substituted → *maximum* drift. This is the path
>   validated end-to-end in the 2026-06-22 session.
> - **Frequency path (~16.5 tok/s, documented):** a warmup pass discovers the top-K
>   most-routed experts → most tokens route to *resident* experts (bit-exact), drift
>   only on the cold tail. **Its one-time build can SIGKILL** on a cold page cache
>   (a full forward of a 323 GB model on a 256 GB box exhausts RAM). Mitigations below.
>
> Both yield the same *speed*; they differ in *quality* (how many tokens are substituted).

---

## 1. Prerequisites

| | |
|---|---|
| Runtime | **MLX 0.31.2**, **mlx-lm 0.31.3**, transformers 5.12.1, numpy. No MLX/mlx-lm patches. |
| Interpreter (Studio) | `~/st-venv/bin/python` |
| Model | `/path/to/models/Nemotron-3-Ultra-550B-A55B-4bit` (323 GB, 109 shards, 4-bit affine, MTP-stripped by `mlx_lm` `sanitize()`) |
| Hardware | M3 Ultra, 256 GB unified, ~819 GB/s |
| Repo | run from `~/SmartTensor` with `PYTHONPATH=src`; the runner is `src/smarttensor/adapters/mlx.py` |

```bash
# sanity: the import graph + tokenizer deps resolve
cd ~/SmartTensor
PYTHONPATH=src ~/st-venv/bin/python -c "
import mlx.core, mlx_lm, transformers
from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
print('ok', transformers.__version__)"
```

---

## 2. Quick start — override path (robust, ~17.6 tok/s)

The session-validated A/B harness; read the `OFF#1`/`OFF#2 decode_tok/s` lines (the
`ON` lines exercise the dead fused-norm kernel — ignore them):

```bash
cd ~/SmartTensor
PYTHONPATH=src ~/st-venv/bin/python benchmarks/nemotron_fused_norm_ab.py \
  --arbitrary-hotset --k 96 --budget-gib 110 --decode 64
# -> [ab] OFF#1 ... decode_tok/s=17.6 ...     (build ~13–20s, peak ~67 GB, free stays > 50 GB)
```

Minimal standalone equivalent (no harness):

```python
from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner
from mlx_lm.utils import load_tokenizer

MODEL = "/path/to/models/Nemotron-3-Ultra-550B-A55B-4bit"
K = 96

r = NemotronHStreamingForwardRunner(
    MODEL, pin_policy="all", page_experts=True,
    weight_page_budget_bytes=110 * 1024**3,
    fixed_hotset_experts=K, cold_substitution=True,
)
ids = list(load_tokenizer(MODEL).encode("Explain how a transformer predicts the next token."))
moe = list(r._moe_layer_indices())                       # the 48 MoE layer indices
r.build_fixed_hotset(ids, override={int(L): list(range(K)) for L in moe})  # full override => NO warmup => NO SIGKILL
out = r.generate_greedy_deferred(ids, 64)
print("decode_tok_s", round(out["decode_tok_s"], 2),
      "cold_subs", out["cold_substitutions"], "cold_redos", out["cold_redos"])
r.close()
```

---

## 3. Frequency-optimized path (~16.5 tok/s) + SIGKILL mitigation

Same config, but `K=200`, a larger budget, and a **real warmup** to discover the
frequency-optimal hot-set:

```python
r = NemotronHStreamingForwardRunner(
    MODEL, pin_policy="all", page_experts=True,
    weight_page_budget_bytes=150 * 1024**3,
    fixed_hotset_experts=200, cold_substitution=True,
)
ids = list(load_tokenizer(MODEL).encode(WARMUP_PROMPT))
r.build_fixed_hotset(ids, warmup_tokens=32)              # phase-1 cold warmup => SIGKILL RISK (see below)
out = r.generate_greedy_deferred(ids, 64)
```

**The trap:** `build_fixed_hotset(warmup_tokens=...)` runs a cold `generate_greedy`
over the *whole* model to record routing frequencies. A full forward of a model
**larger than RAM** faults every shard into the page cache → system `free` → 0 →
**memory-pressure SIGKILL** (no traceback, no Jetsam report; observed at both K=200
and K=96 on a cold cache this session). The kill is driven by *system free*, not
process RSS — so watch both.

**Mitigations (none re-validated this session — prior runs built it when the box
state allowed):**
- **Pre-warm the page cache** before the build (one bounded sequential read of the
  shards) so the warmup forward hits warm pages instead of faulting cold.
- **Shrink `warmup_tokens`** (e.g. 4–8) — fewer cold forwards, less page-cache churn
  (note: even one forward touches all shards, so this only reduces, not eliminates).
- **Build when the model is already RAM-resident** from a prior run.
- If none work, fall back to the **override path** (§2) and accept the higher
  substitution rate.

**Watchdog (always run the build under this):**
```bash
PID=<python pid>
LIMIT_KB=$((215*1024*1024))            # 215 GiB process-RSS cap
while kill -0 "$PID" 2>/dev/null; do
  RSS=$(ps -o rss= -p "$PID" | tr -d ' ')                                  # KB
  FREE=$(vm_stat | awk '/Pages free/{gsub(/\./,"",$3); printf "%.0f", $3*16384/1073741824}')
  echo "rss=$((RSS/1048576))GB free=${FREE}GB"
  if [ "${FREE:-99}" -lt 3 ]; then echo "FREE->0, kill imminent"; fi        # the real signal
  [ "$RSS" -gt "$LIMIT_KB" ] && { echo "RSS cap, killing"; kill -9 "$PID"; break; }
  sleep 5
done
```
On the **shared Studio**, also honor the coordination protocol: check the process
table, append a CLAIMING line to `~/SmartTensor/findings.md`, run **one** model
process, append RELEASING.

---

## 4. Config breakdown (ctor args, `mlx.py:4649`)

| Arg | Value | Behavioral impact |
|---|---|---|
| `pin_policy` | `"all"` | Base (non-expert) weights — embeddings, lm_head, norms, per-layer gate/latent/shared/mamba/attn — pinned resident (~25 GB). |
| `page_experts` | `True` | Activates the weight-page expert cache (the streaming/residency machinery). Required for the hot-set path. |
| `weight_page_budget_bytes` | `150*1024**3` (freq) / `110*1024**3` (override) | Resident expert-cache budget. K=200 hot-set ≈ 110 GB resident; budget must exceed it. |
| `weight_page_rows` | `1` (default) | Page granularity. **Keep 1** — top-22-of-512 routing is scattered, so rows>1 over-reads non-routed neighbors (measured 3× slower). |
| `fixed_hotset_experts` | `200` (freq) / `96` (override) | **K** — resident experts per MoE layer. Built once by `build_fixed_hotset`. |
| `cold_substitution` | `True` | The near-exact unlock: cold experts → resident buddy slot (no redo). `False` = byte-exact but cold tokens force a full-token rollback+redo (slow on fresh prompts). |
| `max_resident_experts_per_layer` | `160` (default) | Upper bound on resident experts/layer. |
| `persist_expert_tables` | `False` (default) | Leave OFF — the naive persistent table regressed (MLX immutable-array concatenate churn). |
| `cold_tier_bits` | `None` (default) | The mixed-precision cold tier (a *separate* resident-fit lever); unused here. |
| `fuse_ssm_norm_gate` | `False` (default) | The fused gated-RMSNorm Metal kernel — measured **0.99× (flat)**, leave OFF. |

Build call (`mlx.py:5489`): `build_fixed_hotset(warmup_prompt_ids, *, warmup_tokens=32, override=None, cold_tier_override=None)`. A **full** `override` (all MoE layers) sets `needs_discovery=False` → the warmup is skipped entirely.

Decode call (`mlx.py:6619`): `generate_greedy_deferred(prompt_ids, max_tokens) -> dict`.

---

## 5. Telemetry & interpretation (the `out` dict)

| Field | Meaning | What to expect |
|---|---|---|
| `decode_tok_s` | **decode-only** rate = `max_tokens / decode_s` (excludes prefill) | the headline; ~16.5 (freq) / ~17.6 (override) |
| `prefill_s` / `decode_s` | wall time of prefill vs the decode loop | decode_s dominates at decode≥64 |
| `deferred_tokens` | tokens that took the fast 1-eval deferred path | should ≈ `max_tokens` |
| `cold_redos` | tokens that hit a non-resident expert and fell back to the exact synced redo | **≈ 0** with `cold_substitution=True` (the unlock); >0 = slow exact path firing |
| `cold_substitutions` | total cold→buddy remaps that fired across the run | **override:** ≈ 22 × 48 × tokens (every token, all-cold). **freq:** low/zero if routing stays resident. This is the **drift knob** — higher = more substitution = more drift. |
| `summary.weight_page_summary.hit_rate` | expert-cache hit rate | high once the hot-set covers routing |

Per-token cost model: the whole 108-layer forward composes into **one MLX graph**;
each token does a **single `mx.eval(next_token_argmax, cold_flag)`** — that's the only
GPU sync. (`cold_redos>0` breaks this for that token.)

---

## 6. Known limits

**Drift (cold→buddy substitution).** *Not* measured via logits/PPL on Nemotron. What
exists: the `cold_substitutions` rate (telemetry) and **qualitative coherence** on a
few held-out, distribution-shifted prompts (factorial code, colors, sky — all coherent
at ~17 tok/s; `docs/nemotron_ultra_benchmarks.md` §6). The **"~0.1%" is a borrowed
PPL-class estimate** (cache-conditional-experts literature, arXiv 2412.00099 on
DeepSeek-MoE), not a Nemotron number. **What changes:** only *which* resident expert's
weights compute a cold-routed slot (nearest gate-row-cosine buddy). **What doesn't:**
the routing **scores** (the gate still emits the cold expert's real top-k weight) and
**all-resident tokens stay bit-exact**. Drift scales with **cold-routing density** —
domain/distribution dependent; the override path is the 100%-substitution worst case.
A hard number needs teacher-forced `KL(stock ‖ substituted)` over a held-out set,
bucketed by cold-routing rate — **not yet run**.

**Speed ceiling.** ~16.5–17.6 tok/s near-exact is the **batch-1 wall**, measured from
every angle: glue-fusion flat (0.99×), `MLX_MAX_OPS_PER_BUFFER` flat → **GPU-bound**
(~58% of peak bandwidth = small batch-1 kernels not saturating), and the only
ceiling-raiser (a Metal megakernel) measured **1.06–1.35× slower** (atomic
cross-threadgroup tax > launch saving; `benchmarks/megakernel_perf_poc.py`).
30–40 tok/s is foreclosed on this model + MLX + hardware at batch-1.

**Bit-exact alternative.** Drop `cold_substitution` (and skip the hot-set) → the plain
expert-cache path is **token-identical to stock** at **~2.0 tok/s** (4.4 warm-ideal).

**Exactness gates (CI, tiny model).** `tests/test_nemotron_runner_*.py` +
`tests/fixtures/tiny_nemotron.py` — bit-exact features must show max-abs-diff `0.0`
on the toy; the deferred path is proven `== stock == synced-fixed` there.
