# T01 — 0731 architecture vs ds4 donor: agreed serial contract (v1)

Decision file: `docs/evidence/deepseek-v4-t01-arch-contract-v1.json`
Companion to: `deepseek-v4-t01-evidence.json`, `docs/recipes/deepseek-v4-cuda-plan.md`

Sources compared:
- TensorFold serial reference: `families/deepseek_v4/{config.py, model.py, attention.py, caches.py, compressor.py, moe.py, weights.py, dense.py, runtime.py}`
- Installed donor: `/home/josh/code/ds4/{ds4.c, ds4.h, ds4_gpu.h, ds4_cuda.cu}`

## 1. Model structure — SAME (adopt donor reader)
Both sides derive the architecture from GGUF `deepseek4.*` keys: 43 layers, 4096 hidden,
64 heads, head_dim 512 (nope 448 + rope 64), shared kv_lora 512, sliding window 128,
per-layer compress ratios, 256 routed / 6 per-tok / 1 shared experts, 3 hash layers,
vocab 129280. Donor additionally reads an optional `mtp.*` head; the 0731 serial reference
does not use MTP. Contract: donor shape reader is authoritative; integers must match exactly.

## 2. Tensor layout — SAME (fixed by 0731 GGUF)
Per-layer `blk.{layer}.{attn_q|attn_kv|attn_output|mlp_gate|mlp_up|mlp_down|mlp_shared_experts|mlp_experts}`,
plus `token_embd.weight` and `output.weight`.
Quant: IQ2_XXS on expert gate/up, Q2_K on expert down, Q8_0 on dense projections, F16/F32 on
embeddings/output/norms. Names, shapes, quant must match the GGUF exactly.

## 3. KV cache — DIFFERENT (adopt donor slabs)
Reference is MLX compressed pools (bf16/fp32); donor is device-resident slabs
(raw_cache / comp_cache / index_comp_cache) with optional packed FP8 KV and FP4 activation.
Contract: adopt donor slab geometry; fp32 exact path is the release default. FP8/FP4 are
opt-in, lossy, gated on T19 operator evidence.

## 4. Attention — SAME (adopt donor CUDA kernel set)
Sliding-window 128 + per-layer ratio compressor + sparse indexer top-k over the compressed
pool, with hash-layer routing. Keep fp32 accumulation and exact masking in the serial path.

## 5. Serial execution path — DIFFERENT (adopt donor eager loop)
Reference is Python/MLX compiled per-row decode. Donor is a C/CUDA per-layer decode loop
over the slab cache with a token-stable scalar substrate, optional cudaGraph capture.
Contract: eager serial per-layer loop for R3. Drafting (MTP) and graph capture out of scope
(DSpark = R8 optional; capture gated on G11 equality evidence).

## Tolerances
- R6 exactness = bitwise fixture-oracle equality; no numeric tolerance substitutes.
- Cross-runtime (ds4 CUDA vs MLX) bitwise equality is NOT presumed.
- FP8/FP4 are declared-lossy; top-1 logit margin must hold against preregistered thresholds.

## Pending hardware-dependent checks
- T16: CUDA kernel bring-up on target (GB10) — PENDING; all kernel/residency claims unverified until then.
- T19: full-model operator evidence (incl. FP8/FP4 lossiness, 81 GiB-class residency) — PENDING.
- VMM: packed-KV demand-map residency — PENDING until T16/T19.
