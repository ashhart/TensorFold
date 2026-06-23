# Nemotron-Ultra (exported checkpoint) architecture inventory

**Phase 1 of the runner spec.** This documents the **actual exported 4-bit MLX
checkpoint** of `Nemotron-3-Ultra-550B-A55B-4bit` (`model_type=nemotron_h`,
hybrid Mamba-2 / attention / LatentMoE) — *not* the paper. Where the paper and
the export disagree, the export is authoritative for this runner and the gap is
flagged.

Scope note: the 323 GB checkpoint lives on the M3 Ultra Studio and is **not
mounted on this machine** (24 GB Mac mini). Every claim below is therefore
sourced from one of:
- **CHECKPOINT FACTS** — config values inspected directly from the on-disk
  `config.json` in a prior session (carried in the runner brief), or
- **CODE** — the canonical `mlx_lm` NemotronH reference that loads this exact
  checkpoint (`mlx_lm/models/nemotron_h.py`), our runner / selective loader
  (`src/smarttensor/adapters/mlx.py`, `src/smarttensor/nemotron_layout.py`), and
  the tensor-name grammar asserted by the committed tests + tiny fixture
  (`tests/test_nemotron_*.py`, `tests/fixtures/tiny_nemotron.py`).

Each row/section tags its evidence. Anything not yet directly inspected is
called out as **NOT YET INSPECTED** rather than guessed.

---

## 1. Config inventory (`config.json`)

| Field | Value | Evidence |
|---|---|---|
| `model_type` | `nemotron_h` | CHECKPOINT FACTS; gated on by the runner (`mlx.py:4626`) |
| `architectures` | `["NemotronHForCausalLM"]` | CHECKPOINT FACTS |
| `num_hidden_layers` | **108** | CHECKPOINT FACTS (consistent with `len(layers_block_type)`) |
| `hidden_size` | **8192** | CHECKPOINT FACTS |
| `num_attention_heads` | **64** | CHECKPOINT FACTS |
| `num_key_value_heads` | **2** (GQA, 32:1) | CHECKPOINT FACTS |
| `head_dim` | **128** | CHECKPOINT FACTS |
| `layers_block_type` | per-layer list of `mamba`/`moe`/`attention` (108 entries) | CHECKPOINT FACTS; parsed by `plan_blocks` |
| — `moe` ("E") layers | **~48** | CHECKPOINT FACTS ("~48 'moe'") — exact count **NOT YET INSPECTED** (the full 108-entry list was not dumped) |
| — `mamba` ("M") layers | majority of the remainder | CHECKPOINT FACTS |
| — `attention` ("*") layers | "a few" | CHECKPOINT FACTS — exact count **NOT YET INSPECTED** |
| `n_routed_experts` | **512** | CHECKPOINT FACTS |
| `num_experts_per_tok` (top-k) | **22** | CHECKPOINT FACTS; read by runner as `top_k` (`mlx.py:3390`) |
| `n_shared_experts` | **1** | CHECKPOINT FACTS |
| `moe_latent_size` | **2048** | CHECKPOINT FACTS — defines the LatentMoE projection (see §3) |
| `moe_intermediate_size` | **5120** | CHECKPOINT FACTS (per-expert FFN width, in latent space) |
| `moe_shared_expert_intermediate_size` | **10240** | CHECKPOINT FACTS |
| `mamba_num_heads` | **256** | CHECKPOINT FACTS |
| `mamba_head_dim` | **64** | CHECKPOINT FACTS |
| `ssm_state_size` | **128** | CHECKPOINT FACTS |
| `conv_kernel` | **4** | CHECKPOINT FACTS (depthwise conv1d kernel width) |
| `n_groups` | **8** | CHECKPOINT FACTS (Mamba-2 group count) |
| `max_position_embeddings` | **262144** | CHECKPOINT FACTS — see REALITY CHECK §5 (paper claims 1M) |
| `quantization` | `{group_size: 32, bits: 4, mode: "affine"}` | CHECKPOINT FACTS — see REALITY CHECK §5 (paper says NVFP4) |

### Router / MoE-gate fields

| Field | Value / status | Evidence |
|---|---|---|
| `num_experts_per_tok` | 22 (top-k) | CHECKPOINT FACTS |
| `norm_topk_prob` | present in `ModelArgs` (bool); **value NOT YET INSPECTED** in the real config | CODE (`ModelArgs.norm_topk_prob`); tiny fixture sets `True` |
| `routed_scaling_factor` | present in `ModelArgs` (float); **value NOT YET INSPECTED** in the real config | CODE; tiny fixture sets `1.0` |
| `n_group` / `topk_group` | accepted by `ModelArgs` (group-limited routing, DeepSeek-style); **whether the real config sets them is NOT YET INSPECTED** | CODE (`ModelArgs.n_group`, `ModelArgs.topk_group`) |
| gate correction bias | `e_score_correction_bias`, a bare `(n_routed_experts,)` parameter (not a Linear); kept full-precision | CODE (`MoEGate`, `cast_predicate`) |

Note: the gate weight is `(n_routed_experts, hidden_size)` and emits **global**
router ids over all 512 experts; the runner's `NemotronHotsetGateAdapter`
remaps those globals to compact/hot-set slot positions (`mlx.py:4482+`).

### Mamba-2 / SSM time-step fields

`ModelArgs` carries `time_step_limit`, `time_step_min`, `time_step_max`
(default `time_step_limit=(0.0, inf)`). **Real-config values NOT YET
INSPECTED.** Evidence: CODE (`mlx_lm` `ModelArgs`, `__post_init__`).

### MTP / multi-token-prediction config fields

CHECKPOINT FACTS record that **config.json contains** `mtp_layers_block_type =
['attention','moe']` and `num_nextn_predict_layers = 1`. **These keys are
config-only and carry no weights in this export** (see §4 and §5). They are also
**not fields of `mlx_lm`'s `ModelArgs`** — the loader ignores them entirely.
Evidence: CHECKPOINT FACTS (config) + CODE (`ModelArgs` has no `mtp*`/`nextn`
field; `sanitize()` drops `mtp.` tensors, `nemotron_h.py:539`).

### Reasoning / thinking-budget fields

**NOT FOUND / NOT YET GREPPED in the real config.** A repo-wide grep for
`reasoning_budget|thinking_budget|max_thinking|budget_tokens|think_budget` over
`src/` and the fixtures returned **zero hits**, and `mlx_lm`'s `ModelArgs`
defines no such field. So either the export carries no reasoning-budget config
field, or it does and we have not inspected `config.json` for it. Treat as
**unconfirmed**. Evidence: CODE (grep, `ModelArgs`).

---

## 2. Derived shape / size facts

These are computed from §1, not separately measured, except where noted as
profiled/measured (which came from runs on the Studio, in the brief).

| Quantity | Value | Source |
|---|---|---|
| Total active params | **55B** (A55B) | CHECKPOINT FACTS / model name |
| On-disk total | **~323 GB**, 109/109 shards | CHECKPOINT FACTS |
| Expert weights (of total) | **~298 GB** | CHECKPOINT FACTS |
| Per-expert footprint | **~12 MB** (4-bit fc1+fc2 + scales + biases, latent-dim) | CHECKPOINT FACTS |
| Active weight read / token | **~27.5 GB** (4-bit) | CHECKPOINT FACTS |
| Base (non-expert) resident | **~25 GB** (embeddings + lm_head + norm + per-layer gate/latent/shared/mamba/attn) | CHECKPOINT FACTS |
| Fixed hot-set K=200 resident | **~110 GB** | CHECKPOINT FACTS |
| Bandwidth floor | **~34 ms/token** (27.5 GB ÷ ~819 GB/s, M3 Ultra) | CHECKPOINT FACTS (profiled HW) |
| Decode forward overhead | **~54 ms fixed** per-forward (kernel launches) + **~9–15 ms/token** marginal → launch-bound | CHECKPOINT FACTS (profiled) |

---

## 3. Tensor inventory by category

Naming is the **HF on-disk grammar** as consumed by `mlx_lm`'s
`sanitize()`/module tree and as asserted by our tests. Top-level prefixes are
`backbone.{embeddings,layers.N,norm_f}` and `lm_head`. The 4-bit export carries
**1791 tensors total** (CHECKPOINT FACTS), all under those prefixes.

> Per-layer tensors live under `backbone.layers.{N}.` and `…mixer.…`; **which
> sub-tree a layer has depends on its `layers_block_type[N]`** (mamba vs
> attention vs moe). Every block has a `backbone.layers.{N}.norm.weight`
> (RMSNorm). Evidence for names: CODE (`mlx_lm/models/nemotron_h.py` module
> tree) + tests (`a.mixer.…`, `backbone.layers.1.mixer.…`).

### 3.1 Embeddings & output head

| Tensor | Shape (logical) | Evidence |
|---|---|---|
| `backbone.embeddings.weight` | `(vocab_size, 8192)` | CODE (`NemotronHModel.embeddings`) |
| `backbone.norm_f.weight` | `(8192,)` final RMSNorm | CODE (`NemotronHModel.norm_f`) |
| `lm_head.weight` | `(vocab_size, 8192)`, `bias=False` | CODE (`Model.lm_head`) |

`vocab_size` is in config but its **exact value was NOT YET INSPECTED**. These
three are the "unlayered" tensors the runner loads explicitly outside the
per-layer loop (`_ensure_non_layer_weights`, `mlx.py:4795+`).

### 3.2 Mamba-2 / SSM mixer (on `mamba` "M" layers)

Module `NemotronHMamba2Mixer`. Per-layer prefix `backbone.layers.{N}.mixer.`:

| Tensor | Notes | Evidence |
|---|---|---|
| `…mixer.in_proj.weight` | input projection; `bias=mamba_proj_bias` | CODE |
| `…mixer.conv1d.weight` | depthwise conv, kernel=`conv_kernel`(4); **HF orientation on disk**, `moveaxis(2,1)` in `sanitize` | CODE (`sanitize`), fixture (inverse moveaxis on save) |
| `…mixer.conv1d.bias` | present iff `use_conv_bias` | CODE |
| `…mixer.A_log` | `(mamba_num_heads,)` = `(256,)`; **kept full-precision** (never quantized) | CODE (`cast_predicate` excludes `A_log`) |
| `…mixer.D` | `(256,)` skip/residual gain | CODE |
| `…mixer.dt_bias` | `(256,)` time-step bias | CODE |
| `…mixer.norm.weight` | `MambaRMSNormGated` (gated RMSNorm inside the mixer) | CODE |
| `…mixer.out_proj.weight` | output projection | CODE |

### 3.3 Attention mixer (on `attention` "*" layers)

Module `NemotronHAttention` (GQA, 64 q-heads / 2 kv-heads, `head_dim=128`):

| Tensor | Notes | Evidence |
|---|---|---|
| `…mixer.q_proj.weight` | `bias=attention_bias` | CODE |
| `…mixer.k_proj.weight` | kv has `num_key_value_heads=2` | CODE |
| `…mixer.v_proj.weight` | — | CODE |
| `…mixer.o_proj.weight` | output | CODE |

No rotary-embedding *weights* (RoPE is positional, not a stored tensor); no
separate q/k layernorm in the `mlx_lm` module. Evidence: CODE.

### 3.4 MoE / LatentMoE mixer (on `moe` "E" layers)

Module `NemotronHMoE`. This is the heavy category (~298 GB / ~298 GB experts).
Per-layer prefix `backbone.layers.{N}.mixer.`:

**Router (gate):**

| Tensor | Shape | Evidence |
|---|---|---|
| `…mixer.gate.weight` | `(512, 8192)` global router | CODE + tests |
| `…mixer.gate.e_score_correction_bias` | `(512,)`, full-precision | CODE (`MoEGate`) |

**Latent projections (the "Latent" in LatentMoE — present iff `moe_latent_size`):**

| Tensor | Shape | Evidence |
|---|---|---|
| `…mixer.fc1_latent_proj.weight` | `8192 → 2048` down to latent | CODE + tests (`backbone.layers.1.mixer.fc1_latent_proj.weight`) |
| `…mixer.fc2_latent_proj.weight` | `2048 → 8192` back to model dim | CODE |

Experts therefore operate **in the 2048-dim latent space**, not the 8192 model
dim — this is why per-expert footprint is small (~12 MB).

**Routed experts — STACKED `SwitchMLP` (the bytes that stream):**

| Tensor | Shape (stacked over experts on dim 0) | Evidence |
|---|---|---|
| `…mixer.switch_mlp.fc1.weight` | `(512, …)` — up/gate proj, latent→`moe_intermediate`(5120) | CODE + tests |
| `…mixer.switch_mlp.fc2.weight` | `(512, …)` — down proj, 5120→latent | CODE + tests |
| `…mixer.switch_mlp.fc1.scales` | `(512, …)` quant scales | tests (`…switch_mlp.fc1.scales`) |
| `…mixer.switch_mlp.fc1.biases` | `(512, …)` quant zero-points | tests |
| `…mixer.switch_mlp.fc2.scales` | `(512, …)` | tests (`…layers.19.…fc2.scales`) |
| `…mixer.switch_mlp.fc2.biases` | `(512, …)` | tests |

Key facts (CODE: `_expert_slice_names`/`_load_selected_experts`, `mlx.py:4825+`):
- Experts are stored **stacked**, not per-expert — the model's own `SwitchMLP`
  params. `sanitize()`'s per-expert `mx.stack` loop (`nemotron_h.py:545–553`)
  is a **no-op** for this export (it only fires for legacy `experts.{e}.*`
  names, which this checkpoint does not use).
- The selective loader slices the **same expert-row indices** out of
  `weight`, `scales`, **and** `biases` together (all stacked on dim 0).
- `biases` may be **optional** per layer (some affine exports omit all-zero
  biases); the loader drives off the manifest and only requests fields that
  actually exist on disk — so `weight`-only, `weight+scales`, and
  `weight+scales+biases` layouts all load.

**Shared expert (always-on, `n_shared_experts=1`):** module `NemotronHMLP`:

| Tensor | Notes | Evidence |
|---|---|---|
| `…mixer.shared_experts.up_proj.weight` | width `moe_shared_expert_intermediate_size`(10240) | CODE + tests (`a.mixer.shared_experts.up_proj.weight`) |
| `…mixer.shared_experts.down_proj.weight` | — | CODE |

MoE forward order (CODE, `NemotronHMoE.__call__`): gate → `fc1_latent_proj` →
`switch_mlp` (routed top-22) → `fc2_latent_proj`, **plus** `shared_experts`
applied to the residual and added back.

### 3.5 Quantization tensors (`.scales` / `.biases`)

- Format: **affine 4-bit, group_size 32** (CHECKPOINT FACTS).
- Every quantizable `Linear`/`SwitchLinear` carries `.scales` and (for affine)
  `.biases` alongside `.weight`. Confirmed present on `switch_mlp.fc1/fc2`
  (tests). The latent projs, shared-expert projs, attention projs, and
  mamba `in_proj`/`out_proj` are also `Linear`s and are quantized the same way
  (CODE: they are `nn.Linear`; `nn.quantize` covers them — mirrored by the
  quantized tiny fixture). **Exact per-tensor scales/biases presence in the
  real export was inspected only for `switch_mlp`; the rest is inferred from
  module type.**
- **Explicitly NOT quantized** (kept full-precision via `cast_predicate`,
  `nemotron_h.py:558–562`): `e_score_correction_bias` and `A_log`.

### 3.6 MTP / multi-token-prediction tensors — **ABSENT**

**Zero `mtp.*` / `nextn` tensors in the export.** See §4/§5 for the evidence
and mechanism.

---

## 4. Per-layer schedule (cache topology)

`plan_blocks(layers_block_type)` (`nemotron_layout.py`) maps each layer to a
single-char code and decides cache residency. This mirrors
`mlx_lm`'s `NemotronHModel`:

| `layers_block_type` | code | mixer | KV/SSM cache entry? |
|---|---|---|---|
| `mamba` | `M` | Mamba-2 SSM | **yes** (SSM state) |
| `attention` | `*` | GQA attention | **yes** (KV) |
| `moe` | `E` | LatentMoE | **no** |
| `mlp` | `-` | dense MLP (not used by Ultra; exists in fixture) | **no** |

`plan_blocks` also computes `fa_idx` (leading `M` count before the first `*`),
`ssm_idx`, and `n_cache` (= count of `M`+`*` layers, the number of cache
slots). Tensor partition for streaming: `partition_layer_tensors` splits each
layer into **base** (everything: gate, latent projs, shared experts, mamba/attn
weights, norms — stays resident) vs **routed_experts** (only
`.mixer.switch_mlp.*` — the streamed/hot-set bytes). Evidence: CODE
(`nemotron_layout.py`).

---

## 5. REALITY CHECK (paper vs export)

This section is the point of Phase 1: what the **export actually is**, against
common assumptions / the paper.

### MTP present? — **NO.** The export dropped it.

- **Config says yes, weights say no.** `config.json` *does* carry
  `mtp_layers_block_type=['attention','moe']` and `num_nextn_predict_layers=1`
  (CHECKPOINT FACTS), but the export's **1791 tensors are all under
  `backbone.{embeddings,layers,norm_f}` + `lm_head`, with ZERO `mtp`/`nextn`
  tensors** (CHECKPOINT FACTS).
- **Mechanism (decisive).** `mlx_lm`'s `sanitize()` **explicitly strips MTP on
  load**: `nemotron_h.py:539` —
  `weights = {k: v for (k, v) in weights.items() if not k.startswith("mtp.")}`.
  And `ModelArgs` has **no** `mtp*`/`nextn` field, so the config keys are
  inert. The `mlx_lm` conversion that produced this checkpoint therefore wrote
  no MTP weights, and even if any leaked, the loader would discard them.
- **Consequence for us:** speculative decoding cannot use a checkpoint MTP/
  draft head — there isn't one. (Matches the brief: real-drafter speculation is
  unmeasured; the historical "41–57 tok/s" used an oracle drafter = leakage.)

### LatentMoE? — **YES.**

- `moe_latent_size=2048` (CHECKPOINT FACTS) **and** the
  `fc1_latent_proj` (8192→2048) / `fc2_latent_proj` (2048→8192) tensors exist
  (CODE + tests). `NemotronHMoE` runs experts in the latent space (CODE,
  `__call__`). This is real and load-bearing — it is why experts are ~12 MB
  each.

### NVFP4? — **NO (in the export).** Affine 4-bit, group_size 32.

- The paper describes **NVFP4** pretraining, but the **export's `quantization`
  block is `{mode: "affine", bits: 4, group_size: 32}`** (CHECKPOINT FACTS),
  and the on-disk format is the standard MLX affine `{weight, scales, biases}`
  triple (tests confirm `.scales`/`.biases` on `switch_mlp`). So the **weights
  we stream are affine-int4, not FP4.** Flag: paper-precision ≠ export-precision.

### Context length: **262144 (export) vs 1M (paper).** Flag.

- `max_position_embeddings=262144` in the export's config (CHECKPOINT FACTS);
  the paper advertises ~1M. Discrepancy to flag — the runner should treat
  **262144** as the real positional ceiling for this checkpoint.

### Reasoning budget? — **unconfirmed (not found in our inspection).**

- No reasoning/thinking-budget config field was found in any inspected source
  (grep over `src/` + fixtures = 0 hits; not in `mlx_lm` `ModelArgs`). Either
  absent, or present in `config.json` and **NOT YET GREPPED there**. Do not
  assume one exists.

---

## 6. Open items (NOT YET INSPECTED)

To finish the inventory, dump from the real checkpoint on the Studio:
1. The **full 108-entry `layers_block_type`** → exact M / * / E counts and
   their positions (only "~48 E, a few *" is known).
2. `vocab_size` (drives embeddings + lm_head shapes / byte sizes).
3. Real-config values for `norm_topk_prob`, `routed_scaling_factor`, and
   whether `n_group`/`topk_group` are set (group-limited routing on/off).
4. Mamba time-step fields (`time_step_limit/min/max`) actual values.
5. Whether any reasoning-budget key exists in `config.json` (grep the real
   file, not just our repo).
6. Per-tensor confirmation that the **non-`switch_mlp`** Linears (latent,
   shared, attn, mamba projs) actually carry `.scales`/`.biases` in this export
   (currently inferred from module type; only `switch_mlp` was directly
   verified).
