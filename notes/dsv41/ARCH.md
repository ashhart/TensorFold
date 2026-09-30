# DeepSeek-V4.1-Flash: implementation spec for TensorFold (CUDA)

This spec covers the text model's forward pass (decode and prefill) and DSpark drafting, in the
order a token flows through them. It is derived from the vLLM port under `notes/ref/` (Apache-2.0),
the checkpoint inventory (`notes/dsv41/inspect.txt`, `notes/dsv41/tensors.json`) and the fixture
config (`tests/fixtures/deepseek_v41/config.json`).

Path abbreviations used in citations:

| tag | path |
| --- | --- |
| `M` | `notes/ref/models/deepseek_v4_1/nvidia/model.py` |
| `A` | `notes/ref/models/deepseek_v4_1/attention.py` |
| `C` | `notes/ref/models/deepseek_v4_1/compressor.py` |
| `FC` | `notes/ref/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py` |
| `IK` | `notes/ref/models/deepseek_v4_1/common/ops/indexer_k_store.py` |
| `CU` | `notes/ref/models/deepseek_v4_1/common/ops/cache_utils.py` |
| `E` | `notes/ref/models/deepseek_v4_1/common/engram.py` |
| `R` | `notes/ref/models/deepseek_v4_1/common/rope.py` |
| `FM` | `notes/ref/models/deepseek_v4_1/nvidia/flashmla.py` |
| `FI` | `notes/ref/models/deepseek_v4_1/nvidia/flashinfer_sparse.py` |
| `SM` | `notes/ref/models/deepseek_v4_1/sparse_mla.py` |
| `D` | `notes/ref/models/deepseek_v4_1/nvidia/dspark.py` |
| `MS` | `notes/ref/models/deepseek_v4_1/nvidia/model_state.py` |
| `AMD` | `notes/ref/models/deepseek_v4_1/amd/model.py` (the torch reference path) |
| `V4M` | `notes/ref/models/deepseek_v4/nvidia/model.py` |
| `IQ` | `notes/ref/models/deepseek_v4/common/ops/fused_indexer_q.py` |
| `IR` | `notes/ref/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py` |
| `OP` | `notes/ref/models/deepseek_v4/nvidia/ops/o_proj.py` |
| `SP` | `notes/ref/v1/worker/gpu/spec_decode/dspark/speculator.py` |
| `SU` | `notes/ref/v1/worker/gpu/spec_decode/dspark/utils.py` |
| `CFG` | `notes/ref/transformers_utils/configs/deepseek_v41.py` |
| `TF-*` | TensorFold MLX `src/tensorfold/families/deepseek_v4/*.py` and `glm5_next/model.py`, `kernels/glm/flash/v1/kernels.py` |

Code under `notes/ref` carries local patches tagged `[dsv41-...]` (EXL3 loading, SM12x block sizes,
file-backed Engram). Those tags affect deployment only; the math is upstream's.

Some vLLM kernels the model calls are not in `notes/ref`: the `mhc_*` hyper-connection kernels,
`SparseAttnIndexer` (DeepGEMM logits, top-k and candidate blocks), FlashMLA, the
`fused_deepseek_v4_qnorm_rope_kv_*` C ops, `GateLinear`/`fused_topk_bias`, `SiluAndMulWithClamp`,
`DFlashSpeculator`, and `DSparkMarkovHead`/`DSparkConfidenceHead`. Where the spec depends on one of
them, it says so and names the substitute source: the TensorFold V4 reference implementation,
DeepSeek-V3.2/V4 conventions, or a third-party re-implementation. §12 lists these as open questions.

---

## 0. Configuration (text_config) and derived constants

From `tests/fixtures/deepseek_v41/config.json`; `CFG:20-76` flattens `text_config` onto the top level.

| name | value | notes |
| --- | --- | --- |
| hidden_size D | 5120 | |
| num_hidden_layers | 40 | + 3 DSpark ("mtp") blocks, logical layer ids 40, 41, 42 |
| vocab_size | 129280 | bos 0, eos 1, pad 2 |
| num_attention_heads H | 64 | num_key_value_heads 1 (MQA, K = V) |
| head_dim | 512 | 448 NoPE + 64 RoPE (RoPE dims are the **last** 64) |
| q_lora_rank / o_lora_rank / o_groups | 1280 / 1024 / 8 | 8 heads per group |
| sliding_window | 128 | the window includes the current token |
| compress_ratios | `[0,0, 2×18, 1×20, 0,0,0]` | 43 entries: layers 0-39 then mtp 40-42 |
| rope_theta / compress_rope_theta | 10000 / 160000 | |
| rope_scaling | yarn, factor 16, beta_fast 32, beta_slow 1, orig 65536 | used only on compressed layers |
| max_position_embeddings | 1048576 | |
| kv_source_layer_ids | [2, 8, 14, 20] | |
| index_source_layer_ids | [2, 8, 14, 20, 24, 28, 32, 36] | |
| index_n_heads / index_head_dim / index_topk | 32 / 128 / 512 | |
| candidate_source_layer_id / _topk_blocks / _block_size | 20 / 2048 / 8 | |
| n_routed_experts / top-k / n_shared | 384 / 6 / 1 | moe_intermediate 2304 |
| scoring / topk_method / norm_topk / routed_scaling | sqrtsoftplus / noaux_tc / true / 1.5 | |
| swiglu_limit | 10.0 | |
| hc_mult / hc_sinkhorn_iters / hc_eps | 4 / 20 / 1e-6 | |
| rms_norm_eps | **1e-20** | every RMSNorm, including the HC and Engram norms |
| engram_layer_ids | [1, 14] | num_embeddings [384006168, 384016682] |
| engram max_ngram / n_heads / head_dim | 4 / 8 / 256 | 3 n-gram orders × 8 heads = 24 hash columns |
| engram_vocab_size / compressed_vocab / pad_token_id | 16000000 / 99092 / 2 | |
| num_nextn_predict_layers | 3 | DSpark blocks |
| dspark block / noise id / targets / markov rank | 5 / 128799 / [37, 38, 39] / 256 | |
| dspark experts / top-k | 128 / 3 | |

Checkpoint quantization: EXL3 v1.4.2 with the `mul1` codebook, `out_scales: always` and a 6-bit head.
It averages 2.93 bits over 207 GB (`inspect.txt:1-3`). The original release was FP8 with [32,32]
UE8M0 block scales on linears and MXFP4 experts (`quantization_config.original_quantization_config`).
Every EXL3 tensor is the quartet `{trellis I16 [in/16, out/16, 16·bits], suh F16 [in], svh F16 [out], mul1 I32 []}`.
From the trellis shape: `bits = trellis.shape[2] / 16`, `in = 16·shape[0]`, `out = 16·shape[1]`.

---

## 1. Global structure

### 1.1 The "CED 20+20" layout

`compress_ratios[L]` for layers 0-39 is `0,0,2,2,…,2 (layers 2-19), 1,…,1 (layers 20-39)`
(`A:246-294`). **Every layer is causal.** There is no bidirectional encoder anywhere in the text
path. The two halves differ in the kind of long-range KV they attend over:

* **Layers 0-19 ("compressed" half).** Layers 0 and 1 attend only over the sliding window. Layers
  2-19 attend over the window plus top-512 entries of a **ratio-2** pooled KV cache: one entry per
  two tokens, a learned per-dim softmax gate over the pair. Only layers 2, 8 and 14 own a compressor
  and an indexer. Each serves a group of six: 2→{2..7}, 8→{8..13}, 14→{14..19}.
* **Layers 20-39 ("YOCO-like" half).** These have compress ratio **1**. Layer 20 alone projects a
  full-length per-token latent KV (`compressor.wkv` + norm, no gate). Layers 21-39 produce no
  long-range KV of their own and attend over **layer 20's** cache: top-512 selected tokens plus their
  own 128-token window. The top-512 selection is recomputed by indexers at 20, 24, 28, 32 and 36,
  each shared by the next three layers. The indexers at 24/28/32/36 reuse layer 20's index keys
  and are restricted to the 2048 candidate blocks that layer 20 publishes.

So "encoder/decoder" is only an analogy. The second half behaves like a decoder cross-attending
into one KV produced at layer 20 (the You-Only-Cache-Once pattern). Every layer still keeps its own
128-token sliding-window KV computed from its own `wkv`.

### 1.2 Per-layer role table (derived from tensors.json and `A:265-294`, `A:371-422`, `A:464-494`)

| layer | ratio | RoPE | long-range KV read | top-k from | owns | extra |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | θ=1e4 plain | — | — | attn, MoE | wq_a/wkv are 6-bit |
| 1 | 0 | θ=1e4 plain | — | — | attn, MoE | **Engram** |
| 2 | 2 | θ=1.6e5 YaRN | own cmp cache | own indexer | compressor{wkv,wgate,norm}, indexer{wk,k_norm,wq_b,weights_proj} | |
| 3-7 | 2 | YaRN | L2 cache | L2 | attn, MoE | |
| 8 | 2 | YaRN | own | own | compressor + full indexer | |
| 9-13 | 2 | YaRN | L8 | L8 | | |
| 14 | 2 | YaRN | own | own | compressor + full indexer | **Engram** |
| 15-19 | 2 | YaRN | L14 | L14 | | experts 2-bit at 18, 19 |
| 20 | 1 | YaRN | own (full length) | own, **writes candidates** | compressor{wkv,norm} (no wgate), full indexer | experts 2-bit (18-22) |
| 21-23 | 1 | YaRN | L20 | L20 | | |
| 24 | 1 | YaRN | L20 | own (reads L20 index K, **candidate-masked**) | indexer{wq_b, weights_proj} only | |
| 25-27 | 1 | YaRN | L20 | L24 | | |
| 28 / 29-31 | 1 | YaRN | L20 | own / L28 | | |
| 32 / 33-35 | 1 | YaRN | L20 | own / L32 | | |
| 36 / 37-39 | 1 | YaRN | L20 | own / L36 | | DSpark taps at 37, 38, 39 |
| mtp 40-42 | 0 | θ=1e4 plain | — | — | attn, 128-expert MoE | DSpark |

Source rule (`A:286-291`): `kv_source = max(s ∈ kv_source_layer_ids, s ≤ L)`,
`index_source = max(s ∈ index_source_layer_ids, s ≤ L)`. Consumers do not recompute. They read the
most recent values in the shared `topk_indices_buffer [max_tokens, 512] int32` (`M:427-432`) and
the source's compressed cache (`A:966-973`).

### 1.3 Top-level forward (`M:543-698`)

```
e      = embed[input_ids]                                  # bf16 [T, 5120]
hashes = NgramHash(input_ids, positions, lookback ids)     # int32 [T, 2, 24]   (§6), once
prefetch Engram rows for layers 1 and 14                   # before layer 0 (M:611-620)
residual = broadcast(e) to [T, 4, 5120]; pre_mix = identity
for L in 0..39: DecoderLayer(L)                            # §2
residual = hc_post(ffn_out_39, residual, post, comb)       # final post
h = Σ_i pre_mix_39ffn[i] · residual[:, i, :]               # hc_collapse (M:691), no hc_head in v4.1
h = RMSNorm(h, norm.weight, 1e-20)
logits = head(h)                                           # EXL3 6-bit, [T, 129280]
```

When DSpark is enabled, the pre-collapse residual `[T, 4·5120]` bf16 is also stored
(`M:680-685`, `_mtp_hidden_buffer`). V4.1 DSpark uses the mean-pooled taps instead (§8).

### 1.4 Tokenizer and embedding

* Tokenizer: the HF fast tokenizer plus the V4.1 prompt encoder
  (`notes/ref/tokenizers/deepseek_v41.py`, `deepseek_v41_encoding.py`). BOS is
  `<｜begin▁of▁sentence｜>` (0) and EOS is `<｜end▁of▁sentence｜>` (1). Role tokens are
  `<｜User｜>`, `<｜Assistant｜>` and `<｜System｜>`, with `<think>`/`</think>` and `｜DSML｜` tool
  calls. Thinking is on by default with reasoning_effort `high`. The image placeholder
  `<｜deepseek_image｜>` has id 129264 (`IMAGE_SENTINEL_BASE_ID`) and the align pad has id 129265
  (`common/mm_preprocess.py:68-78`).
* `embed.weight` BF16 [129280, 5120] is a plain gather. It is a VocabParallelEmbedding under TP
  (`M:449-455`). The embedding is **not** scaled.
* Text-only serving drops `vision.*`, `aligner.*` and `image_*` (`M:733-736`, `M:969-974`).

---

## 2. Decoder layer and hyper-connections (mHC)

### 2.1 Tensors (every layer and mtp block)

| tensor | dtype | shape | meaning |
| --- | --- | --- | --- |
| `layers.L.hc_attn_fn` / `hc_ffn_fn` | F32 | [24, 20480] | mix projection; column index = `stream·5120 + d` (stream-major, `M:886-899`) |
| `layers.L.hc_attn_base` / `hc_ffn_base` | F32 | [24] | [0:4] pre, [4:8] post, [8:24] comb (4×4 row-major) |
| `layers.L.hc_attn_scale` / `hc_ffn_scale` | F32 | [3] | scale for pre, post, comb |
| `layers.L.attn_norm.weight` / `ffn_norm.weight` | BF16 | [5120] | RMSNorm applied to the collapsed input |

`mix_hc = (2 + hc)·hc = 24` and `hc_dim = 4·5120 = 20480` (`M:245-289`). V4.1 has **no**
`hc_head_{fn,base,scale}` (V4 had them, `V4M:1370-1390`).

### 2.2 Math of one HC "pre" (from `mhc_pre_delayed_*` arguments `M:312-389`; formula from the V4 reference `TF-kernels.py:516-530`, `TF-glm5_next/model.py:29-37`)

Inputs: residual `X ∈ bf16 [T, 4, 5120]`, fn, base, scale, the carried-in `pre_in [T,4] fp32`,
and norm_weight.

```
xf    = X.float().reshape(T, 20480)
mix   = (xf @ fn.T) * rsqrt(mean(xf², -1) + 1e-20)        # [T, 24] fp32 (RMS over all 20480)
pre   = sigmoid(mix[:, 0:4]  * scale[0] + base[0:4]) + hc_eps          # this sublayer's pre (for the NEXT sublayer)
post  = 2 · sigmoid(mix[:, 4:8] * scale[1] + base[4:8])                 # hc_post_alpha = 2.0 (M:219)
comb  = mix[:, 8:24].view(T,4,4) * scale[2] + base[8:24].view(4,4)
comb  = softmax(comb, dim=-1) + hc_eps
comb  = comb / (comb.sum(dim=-2, keepdim) + hc_eps)                     # column normalise
repeat 19×: comb /= (comb.sum(-1)+eps); comb /= (comb.sum(-2)+eps)      # sinkhorn_iters = 20 total
x_in  = Σ_i pre_in[:, i] · xf[:, i, :]          # DELAYED: collapse with the PREVIOUS sublayer's pre
x_in  = RMSNorm(x_in → bf16, norm_weight, 1e-20)                        # attn_norm or ffn_norm
return post, comb, x_in, pre
```

**The delayed pre-mix is new in V4.1** (`M:303-305`, `AMD:271-273`). Sublayer *k* collapses its
input with the pre-mix computed at sublayer *k−1* from sublayer *k−1*'s input stream. Attention of
layer L uses the FFN pre of layer L−1. The FFN of layer L uses the attention pre of layer L
(`M:376-389`, `pre_mix=attn_pre`). The final head collapse uses layer 39's FFN pre (`M:687-691`).
The pre computed at a sublayer is never used by that sublayer itself.

**First layer (`M:307-325`, `M:886-899`).** The stream is the embedding broadcast 4×. With the
identity pre-mix the collapsed input is `x = e`, passed explicitly. The mixes are computed with
`fn_broadcast = fn.view(24,4,5120).sum(1)` on `e`. This is exact because
`fn·[e,e,e,e] = (Σ_i fn_i)·e` and `rms([e,e,e,e]) = rms(e)`.

In the vLLM CUDA path the RMSNorm with `attn_norm` is fused into the kernel
(`norm_weight=…`, `M:323-324`). The AMD torch path applies `self.attn_norm(x)` afterwards
(`AMD:329`). The result is the same.

### 2.3 HC "post" (`mhc_post`, `M:343`, `M:375`; formula `TF-glm5_next/model.py:40-50`)

```
Y[:, j, :] = post[:, j] · b + Σ_i comb[:, i, j] · X[:, i, :]      # fp32 math, stored bf16 [T,4,5120]
```

`b` is the sublayer output (bf16 [T, 5120]). Because Sinkhorn ends on a column normalisation,
`Σ_i comb[i,j] = 1`: each new stream is a convex mix of the old ones plus `post_j·b`.

### 2.4 Layer sequence (`M:291-391`)

```
if L > 0:  residual = hc_post(prev_ffn_out, residual, post, comb)
if L ∈ {1, 14}: residual = Engram_L(residual, hashes[:, idx], mask)     # §6, on the full 4-stream state
post_a, comb_a, x, pre_a = hc_pre(residual, hc_attn_*, pre_in = pre_prev, attn_norm)
a = Attention_L(x, positions)                                            # §3
residual = hc_post(a, residual, post_a, comb_a)
post_f, comb_f, x, pre_f = hc_pre(residual, hc_ffn_*, pre_in = pre_a, ffn_norm)
f = MoE_L(x, input_ids)                                                  # §5
return f, residual, post_f, comb_f, pre_f                               # pre_f is carried into layer L+1
```

Pipeline boundaries must carry `residual [T,4,5120]` bf16 and `pre_mix [T,4]` fp32 (`M:516-541`).

---

## 3. Attention (every layer, `A:547-582`)

### 3.1 Tensors

| tensor | dtype | shape | bits |
| --- | --- | --- | --- |
| `attn.wq_a` | EXL3 | 5120 → 1280 | 5 (6 on L0) |
| `attn.wkv` | EXL3 | 5120 → 512 | 5 (6 on L0) |
| `attn.q_norm.weight` | BF16 | [1280] | |
| `attn.kv_norm.weight` | BF16 | [512] | |
| `attn.wq_b` | EXL3 | 1280 → 32768 (64×512) | 5 |
| `attn.wo_a.slice.{g}` g=0..7 | EXL3 | 4096 → 1024 each | 5 (4 in mtp) |
| `attn.wo_b` | EXL3 | 8192 → 5120 | 5 |
| `attn.attn_sink` | F32 | [64] | per-head sink logit |

vLLM fuses `wq_a|wkv` into one replicated GEMM with output [1280|512] (`A:307-314`, `M:705-706`).

### 3.2 RoPE (`R:9-53`)

* The rotary instance is per layer and shared by q, the window KV, the compressor latent, the
  indexer q and k, and the inverse RoPE on the output (`A:349-356`).
* **ratio > 0 (layers 2-39):** θ = `compress_rope_theta` = 160000 with DeepSeek-YaRN: factor 16,
  orig 65536, β_fast 32, β_slow 1, mscale = mscale_all_dim = 0, so the cos/sin amplitude is 1.
  The softmax scale is **not** modified (it stays 512^-0.5).
* **ratio 0 (layers 0, 1, mtp 40-42):** θ = `rope_theta` = 10000, factor 1 (plain RoPE).
* rope_dim = 64 is applied to the **last 64 dims** of each 512 (or 128) vector, in **GPT-J
  interleaved pairs** (`is_neox_style=False`): pairs `(x[448+2i], x[449+2i])`, i = 0..31.
* inv_freq: `f_i = θ^(-2i/64)`. With YaRN, `low = floor(64·ln(65536/(32·2π)) / (2 ln 160000)) = 15`
  and `high = ceil(64·ln(65536/(1·2π)) / (2 ln 160000)) = 25` (both computed). Then
  `ramp_i = clamp((i−15)/(25−15), 0, 1)` and `f'_i = f_i·(1−ramp_i) + (f_i/16)·ramp_i`.
  Pairs 0-15 are unscaled and pairs 25-31 are divided by 16.
* The cos/sin cache has layout `[max_pos, 64] = [cos(32) | sin(32)]` (`FC:329-334`, `IK:171-180`).
* Rotation: `e' = e·c − o·s`, `o' = o·c + e·s`. Inverse RoPE: `e' = e·c + o·s`, `o' = o·c − e·s`
  (`IR:95-110`).

### 3.3 Query path (`A:595-624`, `A:678-682`, `A:839-929`; per-head norm per `TF-attention.py:67-82`)

```
qr_kv = x @ [wq_a | wkv]                         # [T, 1792]
qr    = RMSNorm(qr_kv[:, :1280], q_norm, 1e-20)  # bf16; also feeds the indexer (§4.4)
kv    = RMSNorm(qr_kv[:, 1280:], kv_norm, 1e-20) # bf16 [T,512]
q     = (qr @ wq_b).view(T, 64, 512)
q_h   = q_h · rsqrt(mean(q_h²) + 1e-20)          # per-head RMS, NO weight ("qnorm" in the fused op)
q     = RoPE(q, pos)                             # dims 448..511
kv    = RoPE(kv, pos) → written to this layer's SWA cache at pos   # §7
```

vLLM pads q to 64 heads for FlashMLA (128 heads for local counts above 64) and zero-fills the
padding. Padded sink slots are −inf (`A:298-305`, `FM:77-85`).

### 3.4 Attention set per query token at position p (`CU:710-755`, `FM:180-259`)

```
topk_len = min((p+1) // r, 512)   if r > 0 else 0
keys     = [ Ccache_src[ topk_idx[t, 0:topk_len] ] ]   # compressed entries of this layer's kv-source (r>0)
         ++ [ SWA_L[ max(0, p−127) .. p ] ]            # this layer's own window, current token included
s_j      = (q_h · k_j) · 512^-0.5                     # k_j is the full 512-dim row (NoPE ++ rotated RoPE)
o_h      = Σ_j exp(s_j − m) · k_j / (Σ_j exp(s_j − m) + exp(sink_h − m))       # V = K
```

* MQA: one 512-wide latent serves as both K and V for all 64 heads, and V includes the rotated RoPE
  dims.
* The sink `attn_sink[h]` (F32) adds only to the denominator.
* Index entries equal to −1 are skipped. Nothing deduplicates across the two sets. On ratio-1
  layers a recent token can appear both as layer 20's latent and as this layer's window row. These
  are different vectors and both are kept.
* Compressed entry j covers tokens `[j·r, j·r + r − 1]` and becomes visible once `p ≥ j·r + r − 1`.

### 3.5 Output path (`A:579-582`, `OP:29-92`, `IR:95-160`)

```
o  = o[:, :64]                                         # drop padded heads
o  = inverse_RoPE(o, pos) on dims 448..511 per head    # fp32 (the fp8 quant is skipped for EXL3 wo_a)
o  = o.view(T, 8, 4096)                                # group g = heads 8g..8g+7, head-major within the group
z_g = o[:, g] @ wo_a.slice.g                           # [T,1024] per group, bf16
out = concat_g(z_g) @ wo_b                             # [T, 8192] → [T, 5120]; TP all-reduce here
```

### 3.6 Prefill vs decode

The math is identical. vLLM decode calls `flash_mla_with_kvcache` with the SWA cache as `k_cache`
plus the compressed cache as `extra_k_cache` and the top-k global slots (`FM:243-259`). Prefill
dequantizes and gathers compressed plus SWA rows into a bf16 workspace per chunk of at most 4
requests, builds combined indices (`CU:683-755`) and calls `flash_mla_sparse_fwd` (`FM:261-387`).
Causality comes entirely from the index lists: the window is `[max(0,p−127), p]` and the compressed
rows are those with index `< (p+1)//r` chosen by the indexer.

---

## 4. Compressor, KV sharing, indexer, candidates

### 4.1 Compressor tensors (layers 2, 8, 14, 20 only)

| tensor | dtype | shape | layers |
| --- | --- | --- | --- |
| `attn.compressor.wkv` | EXL3 5b | 5120 → 512 | 2, 8, 14, 20 |
| `attn.compressor.wgate` | EXL3 5b | 5120 → 512 | 2, 8, 14 only (ratio 1 has no gate, `C:204-218`) |
| `attn.compressor.norm.weight` | BF16 | [512] | 2, 8, 14, 20 |

V4.1 has **no** `ape` (V4's absolute positional score bias) and **no** overlapping windows (V4's
ratio-4 layers used 8 overlapping slots). The input is the layer's normed collapsed `x`, the same
input as `wq_a`.

### 4.2 Compressor math (`C:246-323`, `FC:115-219`, `FC:290-338`)

The projection runs in **fp32 output**: `kv_score = x @ [wkv | wgate]`, fp32 [T, 1024] for r=2 and
[T, 512] for r=1 (`A:776-787`, `FC:62`).

**Ratio 2 (layers 2, 8, 14).** Groups are `(2j, 2j+1)`, aligned to absolute position with no
overlap. The group closes at the odd position `p = 2j+1`:

```
kvA,scA = row(2j)[0:512], row(2j)[512:1024]     # raw fp32 (from the ring if 2j was in an earlier step/chunk)
kvB,scB = row(2j+1)
wA,wB   = softmax([scA, scB]) per dim           # elementwise over the 2 tokens, 512 independent softmaxes
pooled  = wA·kvA + wB·kvB                       # fp32 [512]
latent  = bf16( pooled · rsqrt(mean(pooled²) + 1e-20) · norm_w )     # _store_latent FC:213-219
```

**Ratio 1 (layer 20).** `latent_p = bf16(RMSNorm(kv_score_p, norm_w, 1e-20))` for every token.

**Main compressed-cache publish (`FC:290-338`).** At a group boundary `(p+1) % r == 0`:
`row = RoPE(latent, position = (p // r)·r)` on dims 448..511, using the **group's first token**
position and this layer's (YaRN) rotary. The row is stored at compressed index `p // r`. In the
fp8_ds_mla layout the 448 NoPE dims are quantized from the pre-RoPE latent and the RoPE dims are
stored as bf16 (§7).

**Ring state (r=2 only, `C:132-163`, `FC:38-47`, `FC:138-191`).** Per request the ring holds fp32
[capacity, 1024] raw `[kv | score]` rows. Position p lives in row `p % capacity`, with
`capacity = max(8, next_pow2(num_spec_tokens + 2))`, which is 8 with 5 drafts. Each step or chunk
stores its last `min(len, capacity)` raw rows. A group whose first token is in an earlier step
reads that token's row back. Because rows are addressed by position, spec-decode rollback needs no
fix-up as long as `capacity ≥ drafts + bonus + 1`.

### 4.3 KV-source sharing

The compressed cache is allocated **only** on kv sources (`A:938-964`). Consumers use
`compressed_cache_prefix = layers.{kv_source}` (`A:477-494`) together with the source's block table
and slot mapping (`FM:134-141`). Layers 3-7 therefore attend over entries layer 2 wrote earlier in
the same forward. Ratio-0 layers (0, 1, mtp) have none.

### 4.4 Indexer (`A:1023-1237`)

| tensor | dtype | shape | layers |
| --- | --- | --- | --- |
| `attn.indexer.wq_b` | EXL3 5b | 1280 → 4096 (32×128) | 2,8,14,20,24,28,32,36 |
| `attn.indexer.weights_proj.weight` | **F16** (unquantized) | [32, 5120] | same 8 |
| `attn.indexer.wk` | EXL3 **8b** | **512 → 128** (applied to the compressor latent, not to hidden) | 2, 8, 14, 20 |
| `attn.indexer.k_norm.weight` | BF16 | [128] | 2, 8, 14, 20 |

**Index keys (kv sources only, `A:1136-1168`, `IK:120-253`).** At each group boundary:

```
k = latent @ wk                                   # bf16 [128]; latent = the PRE-RoPE compressor output
k = bf16( k · rsqrt(mean(k²)+1e-20) · k_norm_w )
k = bf16( RoPE(k, (p//r)·r) on dims 64..127 )     # GPT-J pairs, layer's YaRN rotary
store: FP8 e4m3 with one per-row pow2 scale 2^ceil(log2(max(amax,1e-4)/448))  (132 B/row)
   or  MXFP4 (2 nibbles/B) + UE8M0 per 32 (68 B/row)  if the fp4 indexer cache is enabled
```

Layers 24, 28, 32 and 36 **share layer 20's index-K cache** (`A:377-399`). Their keys are layer 20's.

**Index queries and weights (every index source, `A:1206-1226`, `IQ:100-181`).**

```
iq  = (qr @ indexer.wq_b).view(T, 32, 128)      # qr = q_norm'd q-lora of THIS layer (no per-head norm)
iq  = RoPE(iq, p) on dims 64..127 → bf16 round trip
iq_fp8, qs = per-(token,head) fp8 with pow2 scale qs = 2^ceil(log2(max(amax,1e-4)/448))
w   = (x @ weights_proj.T)                      # [T, 32] (x = the layer's attn-normed input)
w'  = w · qs · 128^-0.5 · 32^-0.5               # q scale folded into the weight (FP8 path)
```

**Logits** (DeepGEMM `fp8_mqa_logits`, not in ref; formula per V3.2/V4 and `TF-compressor.py:137-141`):

```
I[t, s] = Σ_h w'[t,h] · ReLU( iq_fp8[t,h] · Kdeq[s] )      s ∈ [0, (p_t+1)//r)
```

**Selection.** `topk_idx[t] = top-512 of I[t, ·]` over the visible s, padded with −1. If
`max_seq_len // r ≤ 512` across the batch, every visible entry is taken in ascending order
(`A:1182-1204`, `_fill_short_context_topk_indices` `A:93-109`). Index keys are still produced in
that case.

### 4.5 Two-level candidate filtering (`M:434-447`, `A:400-421`, `A:1033-1037`)

`candidate_block_buffer [max_tokens, 2048] int32` is shared.

* Layer 20 (`candidate_write`) computes its own unmasked top-512 and also publishes candidate
  blocks. Block b = compressed indices `[8b, 8b+7]`. The block score is the **max** of `I` over the
  block's visible entries. The **newest block is pinned** to +∞. The top 2048 blocks are kept, with
  −1 for empty. The kernel is not in ref. These semantics come from the third-party
  re-implementation `…/coolbho-2x-dgx-spark/release/runtime/ds41/dcp_candidates.py:34-85`, whose
  docstring claims V4.1 parity.
* Layers 24, 28, 32, 36 set `I[t,s] = −∞` for s outside the chosen blocks, then take the top 512.
* The mask only matters beyond 2048·8 = **16384** tokens (ratio 1). Below that every block is a
  candidate.

### 4.6 What each layer attends over at decode (position p, 1 token)

| layers | compressed set | window |
| --- | --- | --- |
| 0, 1, mtp | none | own SWA, 128 |
| 2-7 | ≤512 of L2's ⌊(p+1)/2⌋ pooled rows, chosen by L2's indexer | own SWA |
| 8-13, 14-19 | same with L8 / L14 | own SWA |
| 20-23 | ≤512 of L20's p+1 per-token latents, chosen by L20 | own SWA |
| 24-27 … 36-39 | ≤512 of L20's latents, chosen by L24 … L36 over the candidate blocks | own SWA |

The top-k is computed **once per index source per step** and reused by the following consumer
layers in the same forward.

---

## 5. MoE (`M:92-117`, `V4M:773-1060`, `V4M:100-157`)

### 5.1 Tensors

| tensor | dtype | shape |
| --- | --- | --- |
| `ffn.gate.weight` | **F16** | [384, 5120] (mtp: [128, 5120]) |
| `ffn.gate.bias` | **F16** | [384] (mtp: [128]), loaded as `e_score_correction_bias` fp32 (`M:964`, `V4M:865-868`) |
| `ffn.experts.{e}.w1` / `w3` | EXL3 | 5120 → 2304 (gate / up); 3-bit, **2-bit on layers 18-22**, mtp 4-bit |
| `ffn.experts.{e}.w2` | EXL3 | 2304 → 5120 |
| `ffn.shared_experts.w1/w3` | EXL3 4-5b | 5120 → 2304 |
| `ffn.shared_experts.w2` | EXL3 4-5b | 2304 → 5120 |

### 5.2 Math

```
logits = x.float() @ gate.T                        # fp32 (router_logits_dtype fp32, V4M:829-835)
sc     = sqrt(softplus(logits))                    # "sqrtsoftplus"
idx    = topk_6( sc + bias )                       # noaux_tc; bias used only for selection; no groups (n_group = 1)
w      = sc[idx];  w = w / Σ w  (norm_topk_prob);  w *= 1.5
act(g,u) = silu(min(g, 10)) · clamp(u, −10, 10)    # SiluAndMulWithClamp(10); per TF-moe.py:184-188
y      = Σ_k w_k · W2_{idx_k}( act(W1 x, W3 x) )  +  Shared(x)           # shared uses the same clamp
```

* **No hash-routed layers in V4.1.** The model passes `num_hash_layers=0` (`M:115`) and the
  checkpoint has no `tid2eid`. V4 routed its first 3 layers by a token-id table.
* The DSpark blocks use 128 experts with top-3 (`M:102-108`). Everything else is identical,
  including the scaling of 1.5.
* `bias_vl` (image-token routing bias) exists only for vision and is not in this checkpoint. Ignore
  it for text-only serving.
* TensorFold V4 combines the routed outputs in ascending expert-id order in fp32
  (`TF-moe.py:236-274`). vLLM's fused MoE order is kernel-defined.

---

## 6. Engram (layers 1 and 14; `E`)

### 6.1 Tensors and storage

| tensor | dtype | shape | where |
| --- | --- | --- | --- |
| `layers.{1,14}.engram.embed.weight` | FP8 E4M3 | [384006168 / 384016682, 256] | **only in the original FP8 release** (shards 47/48). Absent from the EXL3 checkpoint (tensors.json has no `engram.embed`; `engram_table_dir` is null) |
| `layers.{1,14}.engram.embed.scale` | UE8M0 (`float8_e8m0fnu`) | [N, 8] | one scale per 32 consecutive dims of a row (`E:742-750`) |
| `layers.{1,14}.engram.wkv` | EXL3 (L1 5b, L14 4b) | 6144 → 25600 | |
| `layers.{1,14}.engram.q_weight` | F32 | [4, 5120] | vLLM stores it as **bf16** (`E:1044-1051`) |
| `layers.{1,14}.engram.k_weight` | F32 | [4, 5120] | same |

The mapper renames `engram.embed.scale` to `engram.embed_tokens.weight_scale_inv` (`M:929-945`).
Dequant: `val = float(fp8[r, c]) · 2^(scale[r, c//32] − 127)`, computed as
`(byte << 23) as f32` (`E:667-678`), then stored as bf16.

Size: 384M rows × (256 + 8) B ≈ **101.4 GB per layer**, 203 GB for both. vLLM default is
`cpu_offload=True` (pinned host memory, UVA, `notes/ref/config/engram.py:34-38`). Under TP=2 each
rank holds 12 of the 24 hash columns (≈ 50.7 GB, the "47 GiB/layer" in `M:1065-1066`). The local
fork reads rows from files instead (`E:1190-1204`).

### 6.2 Table layout: 24 prime-sized buckets per layer (`E:173-208`)

* Hash columns are indexed `col = (n−2)·8 + h`, with n-gram order n ∈ {2,3,4} and head h ∈ 0..7.
* Primes: for each layer (1, then 14), each n and each head, take the next prime **greater than**
  the running value that has not been used before. The running value restarts at
  `engram_vocab_size − 1 = 15999999` for each n. In effect these are consecutive primes above 16M:
  layer 1 uses primes #1-24 (16000057 … 16000463, sum 384006168) and layer 14 uses #25-48
  (16000477 … 16000889, sum 384016682). Both sums were computed and match `engram_num_embeddings`.
* `offset[layer][col] = Σ_{c' < col} prime[layer][c']`. The row in the layer's table is
  `hash % prime + offset`.

### 6.3 Token map (compressed vocab, `E:100-148`)

`token_map[id]` is built at load from the HF tokenizer. For each id, decode the single token raw
(`backend.decode([id], skip_special_tokens=False)`). If the text contains U+FFFD, the key is the
raw token string. Otherwise the key is NFKC → NFD → StripAccents → Lowercase →
collapse `[ \t\r\n]+` to a space → a lone " " becomes a sentinel → Strip → sentinel back to " ".
An empty result falls back to the text. Ids are assigned in order of first appearance. The map
**must** produce 99092 ids (asserted, `E:418-425`). `pad_id = token_map[2]`. Precompute it offline
and ship it as an int32 [129280] table.

### 6.4 Hash (`E:151-170`, `E:252-384`)

Multipliers: `m[ℓ][s] = 2·v + 1`, where `v = numpy.random.default_rng(10007·layer_id).integers(0, B, 4, int64)`
and `B = (int64_max // 99092) // 2 = 46539438283891`. Computed values:

```
layer 1 : [76632096046245, 4839876093313, 35959672319349, 73987337458391]
layer 14: [67716810739261, 51510806800915, 30921347202721, 82619226485591]
```

For a token at position p of a request (ℓ = engram layer 0/1):

```
rolling = 0; blocked = false
for s in 0..3:                                  # s = 0 is the current token
    q = p − s
    src = token_map[tok(q)]  (or DEAD=-1 if tok(q) is an image sentinel/pad 129264/129265)
    blocked |= (q < 0) | (src == DEAD)          # sticky
    val = pad_id if blocked else src
    rolling ^= val * m[ℓ][s]                    # int64; product < 2^63 by construction
    if s ≥ 1:  for h in 0..7:
        hash[ℓ][(s−1)·8 + h] = rolling % prime[ℓ][s−1][h] + offset[ℓ][(s−1)·8 + h]
```

`tok(q)` for q before the current chunk comes from `lookback_token_ids` (the 3 preceding ids of the
request, `MS:16-41`) or from the V1 slot cache. The result is int32 [T, 2, 24], computed **once per
forward** for both layers before layer 0. The hash depends only on token ids, never on activations,
so it can be computed ahead of the forward.

### 6.5 Engram forward (`E:1125-1188`, kernel `E:839-928`)

```
Erows = dequant(table_ℓ[hash[t, ℓ, 0:24]])        # bf16 [T, 24, 256]
kv    = flatten(Erows) @ engram.wkv               # [T, 6144] → [T, 25600] bf16
key_i = kv[:, 5120·i : 5120·(i+1)], i = 0..3      # one key per hc stream
val   = kv[:, 20480:25600]                        # one shared value
for each stream i:
    h   = residual[t, i, :].float()
    dot = Σ_d h_d · qw[i,d] · kw[i,d] · key_i,d
          · rsqrt(mean(h²)+1e-20) · rsqrt(mean(key_i²)+1e-20) / sqrt(5120)
    g   = sigmoid( sign(dot) · sqrt(max(|dot|, 1e-6)) )      # clamp_value 1e-6
    g   = 0 if token is an image/dead position (mask)
    residual'[t, i, :] = bf16( h + g · val )
```

This is an RMSNorm-weighted cosine gate. Placement (`M:343-352`): after the previous FFN's hc_post
and **before** this layer's attention hc_pre, on all 4 streams, so the mixes see the injected
stream. Layer 14's compressor and indexer see the Engram-modified input.

### 6.6 IO pattern per token

* Rows read per token per layer: 24 random rows of 256 B (fp8) plus 24 scale reads of 8 B from a
  separate array. That is 6336 B across 48 random accesses, or 12.7 KB per token for both layers.
* TP=2: 12 rows per rank, followed by an all-gather to [T, 24, 256] bf16 (12 KB/token) (`E:1122`).
* vLLM prefetches every layer's rows for the whole batch at the start of the forward
  (`M:611-620`), in a persistent-grid kernel sized to the SM count (`E:786-818`). It can run on a
  background stream at half the SMs.
* `wkv` (6144 × 25600) runs replicated on every rank.

---

## 7. KV cache and state

### 7.1 Per-layer state

| state | who | per unit | fp8_ds_mla | bf16 plain |
| --- | --- | --- | --- | --- |
| SWA KV (own `wkv` after RoPE) | **every** layer 0-39 and mtp 40-42 | per token, last 128 kept | 584 B | 1024 B |
| compressed KV (after RoPE) | 2, 8, 14 (r=2), 20 (r=1) | per compressed entry | 584 B | 1024 B |
| indexer K | 2, 8, 14, 20 | per compressed entry | FP8 132 B / MXFP4 68 B | (always quantized) |
| compressor ring | 2, 8, 14 | per request, 8 rows × 1024 fp32 | 32 KiB | 32 KiB |
| Engram hash history | model | none on the V2 runner (3 lookback ids) | — | — |

**fp8_ds_mla row (`FC:290-327`, `A:938-964`).** A page is segregated:
`[block × 576 value bytes][block × 8 scale bytes]`. Value bytes are 448 e4m3 NoPE values followed
by 64 **bf16** RoPE values (128 B). There are seven UE8M0 scale bytes, one per 64-dim NoPE group:
`exp = ceil(log2(max(amax,1e-4)/448))`, stored as `exp + 127`, clamped to [0, 255]. An eighth byte
is zero padding. Values are `clamp(x·2^-exp, ±448)` cast to e4m3. That is 584 B per token.

**Plain rows (FlashInfer).** A row is `[448 NoPE | 64 RoPE]` in bf16, or per-tensor fp8 with a
scale (`FC:341-383`).

### 7.2 Budget

* **Long-context growth per token (fp8_ds_mla, FP8 indexer):** L2/L8/L14 compressed
  3 × 584/2 = 876 B, L20 compressed 584 B, index K 3 × 132/2 + 132 = 330 B.
  Total **1790 B/token**, about 1.79 GB at 1M context. With bf16 plain rows: 1536 + 1024 + 330 =
  **2890 B/token**.
* **Fixed per request:** SWA of 43 layers × 128 × 584 B ≈ 3.1 MiB (fp8) or 5.4 MiB (bf16), plus
  the paging slack of one block per layer (32 or 64 tokens), plus 3 × 32 KiB of ring.
* Layers that store their own long-range cache: **only 2, 8, 14 and 20**. Every other layer stores
  just its SWA window.

### 7.3 Write order within one step (`A:654-736`)

For each layer: fused q path and SWA insert, overlapped with the compressor save/pool. Then the
main compressed-cache insert, overlapped with index-key production and index q. Then the indexer
logits and top-k. Then attention. Both cache writes finish before the attention reads.

---

## 8. Final norm, head, vocab

`norm.weight` BF16 [5120] with eps 1e-20. The head is EXL3 **6-bit**, trellis [320, 8080, 96],
5120 → 129280, and is not tied to `embed`. It is a ParallelLMHead under TP and logits are gathered
(`M:1050-1101`). The vocab has 129280 entries, including special, placeholder and image ids.

---

## 9. DSpark (`D`, `SP`, `SU`)

### 9.1 Tensors (`mtp.{0,1,2}.*`, remapped by `D:534-567`)

| tensor | shape | notes |
| --- | --- | --- |
| per block i=0..2: `attn.*` (wq_a, wkv, wq_b, wo_a.slice.0-7, wo_b, q/kv_norm, attn_sink), `hc_{attn,ffn}_{fn,base,scale}`, `attn_norm`, `ffn_norm`, `ffn.gate.{weight,bias}` [128,5120]/[128], 128 × `ffn.experts.*`, `ffn.shared_experts.*` | as in the backbone | EXL3 4-bit; logical layer ids 40+i, compress ratio 0 (SWA only, θ = 1e4) |
| `mtp.0.main_proj` | EXL3 4b 15360 → 5120 | input = concat(tap37, tap38, tap39) |
| `mtp.0.main_norm.weight` | BF16 [5120] | |
| `mtp.2.norm.weight` | BF16 [5120] | final draft norm |
| `mtp.2.markov_head.embed.weight` | BF16 [129280, 256] | `markov_w1` |
| `mtp.2.markov_head.head.weight` | F16 [129280, 256] | `markov_w2` |
| `mtp.2.confidence_head.proj.weight` | F16 [1, 5376] | 5376 = 5120 + 256 |

`embed` and `head` are shared with the target (`SU:356-382`). No Engram, no compressor and no
indexer run in the draft blocks.

### 9.2 Target taps

For L ∈ {37, 38, 39}: `tap_L = mean over the 4 streams of the residual after layer L's FFN
hc_post` (`M:654-663`, aux layers {38, 39, 40}). The `+1` convention is confirmed in
`…/mia-recipe/recipe/overlay/vllm/models/deepseek_v4/nvidia/model.py:1897-1918` (it captures after
layer `layer_id`). This is a plain mean, not the pre-mix collapse.

```
main_x = RMSNorm( concat(tap37, tap38, tap39) @ main_proj , main_norm, 1e-20 )   # D:148-153
```

### 9.3 Context KV (`D:155-186`, `D:232-296`)

For every target position whose taps exist, and for each draft block i, compute
`kv = kv_norm_i(wkv_i(main_x))` (the wq_a half is discarded). Apply RoPE at that token's position
with θ=1e4 and insert it into draft block i's SWA cache at that position. Draft blocks only ever
need the last 128 context positions.

### 9.4 One drafting round (`SP:1-24`, `SP:153-196`, `D:188-229`)

Let the verified prefix end at position P−1 and let `b` be the bonus token at position P (not yet
seen by the target):

```
ids   = [b, 128799, 128799, 128799, 128799]           # N = dspark_block_size = 5 queries (anchor + 4 noise)
pos   = [P, P+1, …, P+4]
x     = embed(ids) broadcast to [5, 4, 5120]          # D:205-206
for block i in 0..2:  DecoderLayer(40+i)              # HC + attention + 128-expert top-3 MoE
    attention: q/kv from this block's rows; block-row KV written to the SWA cache at pos;
               each query attends to the causal window of context KV AND to all 5 block rows (non-causal)
h     = hc_post(...);  h = Σ_i pre_mix_last_ffn[i] · h[:, i]        # collapse, no hc_head (D:219-228)
base  = head( RMSNorm(h, mtp.2.norm) )                             # [5, V]; row j predicts position P+1+j
prev  = b
for j in 0..4:                                                     # sequential Markov stage
    me   = markov_embed[prev]                                      # [256]
    lg   = base[j] + me @ markov_head.T                            # [V]
    d_j  = argmax(lg) (greedy) or gumbel-sample(lg / T, seed key = pos P+j)   # SP:120-151
    prev = d_j
conf_j = sigmoid( confidence_proj · concat(h_j, me_j) )            # adaptive verification only
```

* The draft's first layer receives the 3-D broadcast stream with `pre_mix = None`
  (`D:208-218`, taking the `residual = x` branch of `M:326-341`). What the kernel does with
  `pre_mix=None` is not visible. The natural reading, also used by TensorFold V4, is that the
  collapsed input equals the embedding.
* The non-causal mask is implemented by adding the future block tokens to each query's sparse SWA
  indices (`D:9-10`; the kernel `_COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL(window,
  num_spec_tokens)` is not in ref). TensorFold's MLX variant uses the context window
  `[P−128, P)` plus all 5 block rows, with no mask (`TF-dspark.py:30-45`). The two differ only in
  whether the window slides per query row.
* **Verification** runs in the target's next forward over `[b, d_0..d_4]` at positions P..P+5.
  Greedy mode accepts the longest prefix where `d_j == argmax target_j`. Random mode uses standard
  rejection sampling against the cached draft distribution, which includes the Markov bias and the
  temperature (`logits_cache=self.draft_logits`, `SP:140-151`). The next bonus comes from the
  target distribution at the first reject, or at the end if everything is accepted. With
  `enable_adaptive_verification`, a confidence head must exist (`SP:111-117`). How the
  confidences cut the verify length lives in DFlash/scheduler code that is not in ref.
* **State carried between rounds:** only the 3 draft SWA caches (context KV plus stale block rows,
  which the next round's context or block rows overwrite by position) and the sampling seeds/draft
  logits cache. There is **no** recurrent hidden state like MTP's. The context KV for the positions
  accepted in round r, `[P, P+m]`, is written from round r's verification taps before round r+1
  drafts.

---

## 10. Tensor parallelism (TP=2) in vLLM

| weight | split | collective |
| --- | --- | --- |
| `embed` | VocabParallelEmbedding (rows 0-64639 / 64640-129279) | all-reduce once per forward |
| HC fn/base/scale, all norms | replicated; HC runs redundantly on both ranks | — |
| `wq_a|wkv` (fused), `q_norm`, `kv_norm` | replicated (`disable_tp`, `A:307-314`) | — |
| `wq_b` | ColumnParallel: heads 0-31 / 32-63 (`A:316-323`) | — |
| `attn_sink` | head-sliced (`M:813-820`) | — |
| `wo_a` | ColumnParallel over groups: slices 0-3 / 4-7 (`A:326-335`) | — |
| `wo_b` | RowParallel over 8192 (`A:339-346`) | **all-reduce #1 / layer** |
| compressor `wkv|wgate`, `norm` | replicated (`C:219-228`) | — |
| indexer `wq_b`, `weights_proj`, `wk`, `k_norm` | replicated; indexer and top-k computed redundantly (`A:1075-1116`) | — |
| MoE `gate`, bias | replicated | — |
| routed experts | TP: each expert's intermediate split (w1/w3 1152 columns, w2 1152 rows). EP (`--enable-expert-parallel`): experts 0-191 / 192-383 (`V4M:969-1016`) | all-reduce (TP) or dispatch/combine (EP) |
| shared expert | MergedColumn/RowParallel with `reduce_results=False`, folded into the MoE reduction (`V4M:884-893`) | **all-reduce #2 / layer** (shared with routed) |
| Engram table | split by hash column: 12 per rank (`E:703-725`) | **all-gather** [T,12,256]→[T,24,256] at L1 and L14 |
| Engram `wkv`, q/k weights | replicated | — |
| `head` | ParallelLMHead vocab split | gather logits |
| DSpark `main_proj`, heads | replicated; draft blocks split like the backbone | same 2 per block |

Sequence parallel (tokens sharded; attention all-gather/reduce-scatter) is used only with EP and
(MegaMoE or DP>1) (`M:158-166`). Steady-state decode is **2 all-reduces of [T,5120] per layer**,
plus 2 Engram gathers per forward.

---

## 11. Numerics traps

1. **rms_norm_eps = 1e-20** everywhere: attn/ffn/q/kv/compressor/k_norm/final norms, the HC mix
   RMS, and both Engram RMS terms. In fp16 it flushes to 0 and in bf16 it barely registers, so
   compute every RMS in fp32. An all-zero padding row gives inf or NaN, so mask padding rows or
   skip them.
2. **hc_eps = 1e-6** is added to the sigmoid pre and inside Sinkhorn. The post multiplier is 2.0.
   Sinkhorn runs 20 iterations: 1 softmax + column pass, then 19 row+column pairs, ending on
   columns. The residual stream is stored in **bf16** between sublayers, while the HC math runs in
   fp32.
3. **The delayed pre-mix** (§2.2): the collapse uses the previous sublayer's pre. Getting this
   wrong still produces plausible text but the wrong logits.
4. **The compressor projection output is fp32** (`kv_score` is fp32, and the ring stores raw fp32
   kv and score). Pooling, softmax and norm run in fp32. The latent is rounded to **bf16 before
   RoPE**, and RoPE runs in fp32.
5. The indexer K path makes three bf16 round trips: after k_norm, after RoPE, and before the fp8
   absmax (`IK:156-187`, `IK:237-238`). The indexer q is rounded to bf16 after RoPE before its
   absmax (`IQ:128-131`).
6. **Power-of-two scales** everywhere (`2^ceil(log2(amax/448))`, UE8M0) with amax floors of 1e-4
   (fp8) or 6·2^-126 (fp4). For the index q, the FP8 scale is folded into the weights
   `w' = w·qs·128^-0.5·32^-0.5`. Do not apply it twice.
7. **Compressed RoPE position = group start `(p//r)·r`**, not p. The index key uses the same
   position. The query and window KV use p.
8. **RoPE on the LAST 64 dims, GPT-J interleaved.** Compressed layers (2-39) use θ=160000 with YaRN
   on pairs 15-25 and above. Layers 0-1 and the draft use θ=10000 plain. mscale is off: no cos/sin
   amplitude scaling and no softmax-scale change. vLLM probably keeps `cos_sin_cache` in the model
   dtype; build it in fp32 for exactness at 1M positions.
9. **V = K includes the RoPE dims.** Inverse RoPE on the output at the query position is
   mandatory before `wo_a`.
10. **attn_sink** is F32 and adds only to the softmax denominator. Padded heads use −inf.
11. **Dtype conversions vLLM performs:**
    * `indexer.weights_proj` F16 is loaded into a **bf16** param (ReplicatedLinear, `quant_config=None`).
    * `engram.q_weight`/`k_weight` F32 become **bf16** params.
    * `gate.weight` F16 goes to GateLinear, whose param dtype cannot be seen in ref.
    * `gate.bias` F16 becomes fp32, which is exact.

    Decide between parity (mimic the bf16 rounding) and exactness (keep F16/F32). Keeping
    F16/F32 is closer to the reference training kernels.
12. **Engram:**
    * All hash arithmetic is int64: `val·mult < 2^63`, XOR, then `% prime` on non-negative values.
    * `blocked` is sticky.
    * `pad_id` is `token_map[2]`, not 2.
    * UE8M0 byte 0 dequantizes to 0.0 and byte 255 to inf through the `<<23` trick.
    * The gate uses a signed sqrt with a clamp of 1e-6 before the sigmoid.
    * The multipliers come from NumPy PCG64 (`default_rng`). Precompute them; do not
      re-implement the generator.
13. **SwiGLU clamp:** the gate is clamped from above only (`min(g, 10)`) and the up projection is
    clamped on both sides. It applies to the shared expert too.
14. **Router:** sqrt(softplus) in fp32. The bias only affects **selection**. Weights are the
    unbiased scores, renormalised, then multiplied by 1.5. Ties in top-6 or top-3 are
    kernel-defined (torch.topk); TensorFold V4 sorts the selected ids ascending before summing.
15. **Top-k ties** in the indexer (top-512) and the candidate blocks (top-2048) are kernel-defined.
    The attention result depends only on the set, up to summation order. The newest candidate
    block is forced in with +∞. `−1` pads the index lists and must be skipped.
16. **Short-context path:** when the visible count is ≤ 512, all visible compressed entries are
    selected (ascending). Keep producing index keys and compressed rows anyway.
17. **Window duplication** on ratio-1 layers (the same token as both a compressed latent and a
    window row) is intended. Do not dedup.
18. **fp8 orientation:**
    * Engram scales are per row, per 32 contiguous dims.
    * The original release's linear scales are 2-D [32,32] blocks (`scale[out//32, in//32]`).
    * MXFP4 experts are [1,32] along the input dim.

    In the EXL3 checkpoint only Engram keeps fp8.
19. **Spec-decode rollback:**
    * Every cache is position-addressed: SWA slot, compressed index `p//r`, ring row `p % 8`.
    * The ring capacity must be at least drafts + 2.
    * Compressed rows and index keys for rejected positions must be overwritten on re-emit, never
      appended.
20. The first layer's HC may use `Σ_streams fn` on the un-broadcast embedding (exact). DSpark's
    first block sees a 3-D broadcast stream with `pre_mix=None`.

---

## 12. Delta versus the TensorFold MLX `deepseek_v4` family

| component | MLX V4 (V4-Flash) | V4.1 | reuse? |
| --- | --- | --- | --- |
| hidden / layers / experts | 4096 / 43 / 256 | 5120 / 40 (+3 draft) / 384 (draft 128 top-3) | shapes only |
| HC pre/post, Sinkhorn | `HC.split` + `hc_expand` (`TF-glm5_next/model.py:18-50`) | same math, but the collapse uses the **previous sublayer's pre** (delayed); first layer uses identity | **reuse the kernels**, rewire the pre flow |
| final collapse | `HeadHC` (learned hc_head, sigmoid, no Sinkhorn, `TF-model.py:47-66`) | no hc_head: collapse with layer 39's FFN pre | new (simpler) |
| attention q/kv/sink/inverse RoPE/grouped wo_a | `TF-attention.py:67-98` | identical structure | **reuse** |
| RoPE | YaRN on compressed layers, last 64 dims, interleaved | same; θ=1.6e5 YaRN on **all** ratio>0 layers (1 and 2), 1e4 plain on 0/1/draft | reuse, retabulate |
| compressor | ratio 4 (overlapping, 8 slots, ape) and ratio 128 | ratio 2 (non-overlapping pair softmax, no ape) and ratio 1 (norm only, no gate) | pool kernel reusable with r=2 and no ape; r=1 is trivial |
| compressed-KV ownership | every compressed layer owns its pool | **only 2, 8, 14, 20 own**; 36 of the 38 compressed layers read a source's cache | new cross-layer cache plumbing |
| indexer | ratio-4 layers, 64 heads, own compressor for index keys over hidden states | 8 index sources, 32 heads; **keys = k_norm(wk(compressor latent))** from the kv source; 24-36 share layer 20's K | score/select reusable; key path new |
| candidate blocks | none | layer 20 publishes 2048 blocks of 8; 24/28/32/36 mask to them | new |
| top-k reuse | each layer recomputes | consumers reuse the source's top-k within the forward | new (cheaper) |
| window | own 128-row ring per layer | same (every layer keeps its own SWA) | **reuse** |
| MoE | mxfp4 experts, hash routing on 3 layers | EXL3 2/3-bit experts, **no hash routing**, F16 gate/bias | router math reuse; expert GEMM new (EXL3) |
| Engram | none | layers 1 and 14; 203 GB of fp8 tables, n-gram hash, gate | **new** |
| DSpark | 3 blocks, taps after layers 40-42 (of 43), hc_head, Markov head | 3 blocks, taps after **37/38/39**, no hc_head (collapse by last FFN pre), 128-expert top-3 MoE, confidence head | mostly reuse (`TF-dspark.py`); change the collapse and the tap indices |
| quantization | MLX affine 4-bit + mxfp4 | EXL3 trellis (2-8 bit), fp32 compressor output | new GEMM path |
| KV per token (long ctx) | per-layer pools | ~1.8 KB/token total (fp8) | cache manager rewrite |

---

## 13. Open questions the source code does not answer

1. The exact `mhc_pre_delayed` kernel: Sinkhorn order and eps placement are taken from the V4
   reference, and what happens when `pre_mix=None` with a 3-D input (DSpark's first block) is not
   visible.
2. The `SparseAttnIndexer` internals: the exact candidate-block scoring (max assumed), newest-block
   pinning, the ReLU placement in the logits, and tie-breaking.
3. DSpark's non-causal SWA index construction (sliding versus fixed context window) and the
   adaptive-verification policy that uses `conf_j`.
4. `DSparkMarkovHead`/`DSparkConfidenceHead` internals: the concat order `[h, me]`, whether `h` is
   pre- or post-norm (the speculator passes the pre-norm `head_hidden`), and any scaling.
5. The GateLinear parameter dtype for the F16 router, and the `cos_sin_cache` dtype.
6. Whether any Hadamard rotation is applied to indexer q/k: `rotate=True` is stored but unused
   (`C:187,201`).
