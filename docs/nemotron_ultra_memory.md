# Nemotron-3-Ultra-550B — TensorFold Memory Model (Phase 2)

Model: `Nemotron-3-Ultra-550B-A55B-4bit` (`model_type=nemotron_h`,
`NemotronHForCausalLM`). Hybrid Mamba2 / attention backbone with a LatentMoE in
the MoE layers. This document is the byte budget and resident-floor model for
running it under TensorFold's streaming/paging stack.

All arithmetic below is shown explicitly. Numbers are derived from the exported
checkpoint config and cross-checked against the `mlx_lm` reference
implementation (`mlx_lm/models/nemotron_h.py`, `cache.py`, `ssm.py`) that our
runner (`NemotronHStreamingForwardRunner` in `src/smarttensor/adapters/mlx.py`)
mirrors. Where a number cannot be pinned exactly on this machine (the 323 GB
checkpoint lives on the Studio, not this 24 GB Mac mini), it is banded and
flagged, not invented.

Units: **GB/MB = 10^9 / 10^6** (decimal, matches disk reporting);
**GiB/MiB/KiB = 2^30 / 2^20 / 2^10** (binary, matches RSS / allocator). Mixed
deliberately — disk vendors and `df` use decimal, the allocator uses binary.

---

## 1. Config facts (the inputs)

| Field | Value | Notes |
|---|---|---|
| `num_hidden_layers` | 108 | |
| `layers_block_type` | list of `mamba`/`moe`/`attention` | ~48 `moe`, mostly `mamba`, a few `attention` |
| `hidden_size` | 8192 | |
| `num_attention_heads` | 64 | |
| `num_key_value_heads` | 2 | GQA — KV cache is tiny per token |
| `head_dim` | 128 | |
| `n_routed_experts` | 512 | per MoE layer |
| `num_experts_per_tok` | 22 | top-k |
| `n_shared_experts` | 1 | always-on, runs on hidden dim |
| `moe_latent_size` | 2048 | LatentMoE: experts operate in latent dim |
| `moe_intermediate_size` | 5120 | expert MLP inner dim |
| `moe_shared_expert_intermediate_size` | 10240 | shared expert inner dim |
| `mamba_num_heads` | 256 | |
| `mamba_head_dim` | 64 | → mamba `intermediate_size` = 256×64 = **16384** |
| `ssm_state_size` | 128 | |
| `conv_kernel` | 4 | conv state depth = `conv_kernel-1` = 3 |
| `n_groups` | 8 | |
| `max_position_embeddings` | 262144 | **paper claims 1M — flag the discrepancy** |
| `quantization` | `{group_size:32, bits:4, mode:affine}` | **paper says NVFP4 pretrain; export is affine-4bit — flag** |
| MTP / nextn | config lists 1 nextn layer | **export has ZERO mtp/nextn tensors** (`sanitize` strips `mtp.`) → **0 bytes** |

Derived constants used throughout:

- 4-bit weight cost `W = 0.5 byte/param` (pre-overhead; scales+biases add on top).
- Cache/activation dtype = **bf16 → 2 bytes** (mlx_lm runs the KV and Mamba
  state unquantized by default; `KVCache.update_and_fetch` allocates at
  `keys.dtype`).
- Mamba `conv_dim = intermediate_size + 2·n_groups·ssm_state_size`
  = 16384 + 2·8·128 = **18432**.

### Layer-type split (banded)

The exact `attention` count is in the config's `layers_block_type` list, which
is on the Studio, not here. 48 layers are `moe`. The remainder is mamba-heavy
with a small number of attention layers (the Nemotron-H hybrid pattern). The
band that matters for KV growth:

| Assumed `attention` layers | `mamba` layers | `moe` layers |
|---|---|---|
| 4 | 56 | 48 |
| **6 (representative)** | **54** | **48** |
| 8 | 52 | 48 |

The representative split (6 attn / 54 mamba / 48 moe) is used for the tables
below. KV-cache numbers scale linearly with the attention count — recompute when
the real count is confirmed.

---

## 2. The two state regimes (the core of the hybrid)

This is the one thing a memory model for this architecture must get right: the
hybrid has **two completely different state regimes**, and they must be budgeted
**separately**. Conflating them is the classic misread.

### 2a. Attention KV cache — GROWS with context

Only `attention` layers carry a KV cache (`make_cache()` appends a `KVCache()`
for `*` layers, nothing for `moe`/`mamba`). Each attention layer stores keys and
values shaped `(B, num_kv_heads, seq_len, head_dim)` at bf16.

Per attention layer, per token:

```
k + v = 2 × num_kv_heads(2) × head_dim(128) × 2 bytes = 1024 bytes/token/layer
```

GQA with only **2 KV heads** is what makes this cheap. Across 6 attention
layers:

```
KV/token = 1024 × 6 = 6144 bytes/token = 6.0 KiB/token
```

| Context (tokens) | KV cache total |
|---|---|
| 1,024 | 6.0 MiB |
| 8,192 | 48.0 MiB |
| 32,768 | 192.0 MiB |
| 131,072 | 768.0 MiB |
| 262,144 (max pos) | **1.50 GiB** |

(`KVCache` over-allocates in steps of 256 tokens, so true resident rounds up to
the next 256-token block per layer — negligible at these scales.)

### 2b. Mamba / SSM state — FIXED per layer, context-independent

Mamba layers carry an `ArraysCache(size=2)`: slot 0 = conv state, slot 1 = SSM
recurrent state. **Neither depends on `seq_len`.** This is the whole point of
the SSM — constant state regardless of how long the context is.

Per mamba layer (bf16):

```
conv state  cache[0] : (B, conv_kernel-1=3, conv_dim=18432)
                       = 3 × 18432 × 2          = 110,592 B  = 108.0 KiB
ssm  state  cache[1] : (B, mamba_num_heads=256, mamba_head_dim=64, ssm_state_size=128)
                       = 256 × 64 × 128 × 2     = 4,194,304 B = 4096.0 KiB
                       -----------------------------------------------------
mamba state / layer                            = 4,304,896 B = 4204.0 KiB (FIXED)
```

Across 54 mamba layers:

```
Mamba state total = 4204.0 KiB × 54 = 221.7 MiB  (FIXED — does not grow)
```

### 2c. The crossover — why long context is an attention problem only

| Context | KV (grows) | Mamba state (fixed) | KV share |
|---|---|---|---|
| 1,024 | 6.0 MiB | 221.7 MiB | 3% |
| 32,768 | 192.0 MiB | 221.7 MiB | 46% |
| 131,072 | 768.0 MiB | 221.7 MiB | 78% |
| 262,144 | 1.50 GiB | 221.7 MiB | **87%** |

At short context the *fixed* mamba state dominates the runtime cache footprint;
KV only overtakes it past ~30k tokens. The mamba-heavy design is exactly what
keeps long-context serving cheap: 54 of 108 layers contribute **zero** growth.
A pure-attention 108-layer model at these dims would carry ~18 KiB/token →
~4.5 GiB KV at 262k. The hybrid pays ~1.5 GiB instead. **SSM-heavy layers
directly reduce long-context cache pressure** — see §4e.

Even at the full 262k window, total runtime state (KV 1.50 GiB + mamba
0.22 GiB ≈ **1.72 GiB**) is a rounding error against the weight budget below.
**For this model, runtime state is never the binding constraint — weights are.**

---

## 3. Byte budget table (weights on disk)

### 3a. Expert bytes (the bulk: ~298 GB of 323 GB)

```
expert bytes / MoE layer  = 298 GB / 48 layers        = 6.21 GB/layer
expert bytes / expert     = 6.21 GB / 512 experts     = 12.13 MB/expert
```

The 12.13 MB on-disk figure is the **affine-4bit packed weight + scales +
biases**. The bare fc1+fc2 latent-dim weights alone are:

```
per expert = (moe_latent×moe_inter + moe_inter×moe_latent) × 0.5
           = (2048×5120 + 5120×2048) × 0.5 = 20.97 M params × 0.5 = 10.49 MB
```

So ≈ **10.5 MB weights + ≈1.6 MB scales/biases ≈ 12.1 MB on disk** per expert.
This 13% group-quant overhead (group_size=32, 4-bit) is also the largest single
contributor to the 323 vs 317.8 GB reconciliation gap in §3d.

**Active expert bytes/token** (top-22 experts × 48 MoE layers):

```
weights-only : 22 × 10.49 MB × 48 = 11.07 GB/token
on-disk basis: 22 × 12.13 MB × 48 = 12.80 GB/token
```

### 3b. Non-expert base bytes (resident-everything floor)

These are the weights you cannot page out per-token without re-reading them
every token — the dense backbone.

| Component | Per-unit | Count | Total |
|---|---|---|---|
| Embeddings | 0.54 GB | 1 | 0.54 GB |
| LM head | 0.54 GB | 1 | 0.54 GB |
| MoE-layer base (gate + latent fc1/fc2 + shared expert + norms) | 0.145 GB | 48 | 6.9 GB |
| Mamba layer weights (in_proj + conv1d + out_proj) | 0.211 GB | 54 | 11.4 GB |
| Attention layer weights (q/k/v/o) | 69.2 MB | 6 | 0.42 GB |
| **Non-expert base TOTAL** | | | **≈ 19.8 GB** |

MoE-layer base breakdown (per layer): gate (router `[hidden×512]`) 2.1 MB,
latent_fc1 (8192→2048) 8.4 MB, latent_fc2 (2048→8192) 8.4 MB, shared expert
(gate+up+down on hidden↔10240) **0.126 GB** — the shared expert dominates the
MoE base.

> Vocab size is taken as 131072 (typical Nemotron) — **flag as approximate**;
> embed/lm_head scale linearly with it. ±20k vocab moves these two lines by
> ~±0.17 GB. Mamba/attention per-layer weights are estimated from the dim
> formulas; bias terms and norms are sub-1% and folded in.

### 3c. SSM state, KV growth, MTP, temp/packing

| Item | Bytes | Regime |
|---|---|---|
| SSM/mamba state | 221.7 MiB total (4204 KiB × 54) | **FIXED** |
| KV cache | 6144 B/token (6 attn layers) | **GROWS** — see §2a |
| MTP / nextn | **0** | absent from export |
| Temp expert table / packing | small; see note | transient |

**Temp / packing (transient, not resident floor):** dequant scratch and the
stacked `switch_mlp` gather buffers are bounded by *one* MoE layer's active
working set, not the whole model. A single MoE layer's full expert tensor is
6.21 GB; the per-token active slice is `22 × 12.13 MB ≈ 267 MB`. Double-buffered
streaming (read layer N+1 while computing layer N) therefore needs **≈ one
expert-layer of headroom (~6.2 GB)** above the resident floor, plus a small
dequant scratch (latent-dim, MB-scale). The `g2s` global→slot lookup tables are
`int32 × n_routed_experts × n_moe = 512 × 48 × 4 = ~98 KB` total — negligible.

### 3d. Reconciliation

```
non-expert base   19.8 GB
+ experts        298.0 GB
= 317.8 GB        vs 323 GB reported on disk
```

The ~5 GB gap is: affine-4bit group scales/biases beyond the 12.13 MB/expert
already counted vs the 10.49 MB weight core (the biggest piece), the
approximate vocab in embed/lm_head, per-layer norms/biases, and shard/container
padding. The budget ties out to within ~1.6%. **Honest:** the residual is not
re-derived tensor-by-tensor here (the checkpoint isn't on this host); it is
bounded and attributed, not zero.

---

## 4. Resident-floor estimates per serving mode

"Resident floor" = the minimum you must hold in unified memory to run that mode,
**before** KV/SSM state (which §2 shows is ≤1.72 GiB even at max context, i.e.
negligible against weights). M3 Ultra bandwidth ≈ 819 GB/s sets the streaming
floor; profiled decode is ~54 ms fixed per-forward launch overhead + ~9–15 ms/tok
marginal (launch-bound, not purely bandwidth-bound).

### 4a. Cold exact streaming (bit-exact, nothing pinned but base)

```
resident floor = base 19.8 GB + double-buffer headroom ~6.2 GB ≈ 26 GB
bytes READ per token = full active set = 27–31 GB/token   (see §5)
```

Every active expert is read from SSD every token. Bandwidth floor:
27–31 GB ÷ 819 GB/s ≈ **33–38 ms/token** if perfectly bandwidth-bound — but the
~54 ms fixed launch overhead dominates, so this is the **~2 tok/s bit-exact**
regime. The 24 GB Mac mini cannot hold even the 26 GB floor — cold exact
streaming is a Studio-class (256 GB) mode.

### 4b. Low-mem expert paging (Stage-2 page cache)

```
resident floor = base 19.8 GB + page-cache window (tunable) + ~6.2 GB working set
```

Pages experts on demand via `pread`/`drop_mmap`. RSS = base + whatever page-cache
window you grant + transient read buffers. This is the only mode that *could*
approach a small box, but at a steep tok/s cost (re-reading uncached experts).
On 24 GB you cannot even pin base + one MoE layer (19.8 + 6.2 ≈ 26 GB) — so this
host is for the tiny toy fixtures only; the real model is Studio-resident.

### 4c. Hot-expert residency (K=200, the fast near-exact mode)

`build_fixed_hotset` loads the warmup-frequency top-K experts **once** into the
stacked `switch_mlp` for native on-GPU gather.

```
hot experts resident = K × 12.13 MB × 48 layers
                     = 200 × 12.13 MB × 48 = 116 GB   (2.43 GB/layer × 48)
resident floor = base 19.8 GB + hot 116 GB ≈ 136 GB
K=200 / 512 = 39.1% of every layer's experts pinned
```

Cold experts (the other 61%) are handled by `cold_substitution` (g2s→buddy slot
via gate-row cosine) → near-exact (~0.1% drift, coherent on unseen prompts).
This is the genuine **~16–17 tok/s** number. 136 GB fits the 256 GB Studio with
~120 GB headroom for KV, OS, allocator, and a wider speculative union (§4d).

### 4d. Speculative verification (wider expert union)

Block-verify (`generate_greedy_speculative_deferred`) verifies a draft block in
one batched forward. The verify forward's active expert **union over the block**
is wider than a single token's top-22 — every distinct expert any draft token
routes to must be resident or paged for that forward.

```
per-token union   : 22 experts/layer
block of B tokens : up to min(22·B, 512) distinct experts/layer (typically far
                    below the cap due to routing overlap)
```

Sizing: hold the K=200 hot set (covers the high-frequency union for free) and
budget paging headroom for the cold tail of the block union. Practical floor =
**§4c's 136 GB + a modest paging window** for block-cold experts. The wider the
draft block, the wider the union, the more expert bytes the single verify
forward must touch — this trades resident headroom for accept-length.

> **Leakage warning:** the historical "41–57 tok/s" speculation figure used an
> **oracle drafter** (the draft tape was the generation's own greedy output →
> 100% accept by construction). It is meaningless. Real-drafter speculation
> throughput is **unmeasured**. Do not cite the oracle number. The verify
> *memory* model above is real; the *speedup* is not yet measured.

### 4e. Long-context serving (KV at 262k)

This is where §2 pays off. At the full 262,144-token window:

```
KV cache (6 attn layers)  : 1.50 GiB     (GROWS with context)
Mamba/SSM state (54 lyrs) : 0.22 GiB     (FIXED — unchanged from 1 token)
runtime state total       : ~1.72 GiB
```

Add this on top of whichever weight floor you chose (e.g. 136 GB hot-set →
~138 GiB at max context). **KV grows but never threatens the budget**: 1.5 GiB
at 262k is ~1% of the weight footprint. The mamba-heavy split is doing the work
— if all 108 layers were attention, KV at 262k would be ~4.5 GiB (3×) and every
SSM layer's 4 MiB fixed state would instead be a growing KV term. **SSM-heavy
layers materially reduce long-context cache pressure**; the binding constraint
at long context is still **expert weight residency**, not state.

Sensitivity: KV scales linearly with the (unconfirmed) attention-layer count.
At 8 attn layers, max-context KV = 2.0 GiB; at 4, it's 1.0 GiB. Either way,
sub-2 GiB and not binding.

---

## 5. Active read per token — the throughput driver (honest reconciliation)

The two figures in circulation reconcile as different scopes; this is the kind
of discrepancy to surface, not hide:

| Scope | Params | Bytes @4bit (W=0.5) |
|---|---|---|
| Active **experts only** (top-22 × 48) | 22.2 B | 11.1 GB (weights) / 12.8 GB (on-disk) |
| + MoE base (gate+latent+shared, all 48) | → 36.0 B | |
| + Mamba weights (all 54 layers, always active) | +22.8 B | |
| + Attention (6 layers) + embed + lm_head | +3.0 B | |
| **Total active params / token** | **≈ 62 B** | **≈ 31 GB/token** |

The advertised **"55B active / ~27.5 GB"** (A55B) sits just under this estimate.
The gap (62 B vs 55 B, 31 vs 27.5 GB) is attributable to: the exact
attention-layer count (we banded 6), whether the shared expert and full mamba
in_proj/out_proj are counted as "active" in the vendor's A-param definition, and
the vocab approximation. **Honest band: active read/token ≈ 27–31 GB**, and the
**mamba in_proj/out_proj (≈23 B params, always on) is as large as the 22 routed
experts** — a Mamba-side cost that pure-MoE accounting overlooks. Per-token
throughput is governed by this whole active set streamed at ≤819 GB/s **and** by
the ~54 ms fixed launch overhead, not by the experts alone.

---

## 6. Summary

- **Weights, not state, are the constraint.** Total disk 323 GB; experts 298 GB;
  base 19.8 GB. Runtime state (KV + SSM) ≤ ~1.72 GiB even at the 262k max window.
- **Two state regimes, budgeted apart:** KV grows at **1024 B/token/attn-layer**
  (6.0 KiB/token at 6 attn layers → 1.5 GiB at 262k); Mamba state is **FIXED at
  4204 KiB/layer → 221.7 MiB total**. KV only overtakes mamba past ~30k tokens.
- **Resident floors:** cold exact ≈ 26 GB + 27–31 GB/token read (~2 tok/s,
  bit-exact); hot K=200 ≈ **136 GB** (39% of experts pinned) → ~16–17 tok/s
  near-exact (cold→buddy substitution). Both are Studio-class (256 GB); the
  24 GB Mac mini runs only the toy fixtures.
- **MTP absent (0 bytes).** Speculation's verify-union memory model is real;
  its **speedup is unmeasured** — the "41–57 tok/s" oracle number is leakage.
- **Per expert ≈ 12.13 MB on disk** (10.49 MB weights + ~1.6 MB affine-4bit
  scales/biases). **Active read ≈ 27–31 GB/token**, with the Mamba projections
  contributing as much as the 22 routed experts.
- **Flags to carry forward:** max_position 262144 in export vs 1M in paper;
  affine-4bit export vs NVFP4-pretrain claim; exact attention-layer count and
  vocab size unconfirmed on this host (banded above).
