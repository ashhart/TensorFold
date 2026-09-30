# DeepSeek-V4-Flash-0731 — Decision / Evidence Manifest (v1)

Versioned consolidation of the three T01 parent outputs for the serial CUDA port.
Machine-readable source: `deepseek-v4-t01-decision-manifest-v1.json`.

- Upstream audit: `deepseek-v4-t01-evidence.json/.yaml` (task t_79b60cb5, commit ebdbd50)
- 0731-vs-donor architecture contract: `deepseek-v4-t01-arch-contract-v1.json/.md` (audit t_36cc607c, commit 5d22c7a)
- G0 test-first plan: `deepseek-v4-g0-testfirst.md` (commit a4edf63)

## Scope

- Target: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` (~81 GiB, antirez/deepseek-v4-gguf).
- Hardware: one NVIDIA GB10 DGX Spark, 128 GB nominal unified memory, sharing the machine with loaded Hunyuan3D-2.1 and CUDA image moderation.
- Execution mode: serial only, one request, no drafting for the first release.
- Development rule: keep the live ds4 endpoint, Hunyuan, and moderation running; no full-model load without measured admission; no live backend stop/restart.

## Agreed decisions

- D1 model structure: adopt the donor shape reader as authoritative; keep the TensorFold config dataclass in sync. Architecture integers must match exactly (n_layer 43, n_embd 4096, n_head 64, head_dim 512, qk_rope 64, qk_nope 448, kv_lora_rank 512, sliding_window 128, n_routed_experts 256, experts_per_tok 6, n_shared_experts 1, num_hash_layers 3, vocab 129280; q/o lora ranks and compress ratios per-checkpoint/per-layer).
- D2 tensor layout: fixed by the 0731 GGUF layout (token_embd, attn_q/q_a/q_b, attn_kv/kv_a/kv_b, attn_output/output_a/output_b, mlp_gate/up/down, mlp_shared_experts, mlp_experts, norms, output). Quant scheme: experts IQ2_XXS, down Q2_K, dense Q8_0, embeddings/norms F16/F32.
- D3 KV cache: donor slab geometry (raw_cache / comp_cache / index_comp_cache); default fp32 exact path; FP8/FP4 opt-in only, disabled for release 1.
- D4 attention: donor kernel set (raw SWA read, compressor emit, indexer top-k) with reference math (fp32 accumulation, exact masking) in the serial path.
- D5 serial path: donor serial eager per-layer loop; drafting (MTP) and cudaGraph capture out of scope for R3.

## Unresolved upstream decisions (explicit)

1. U1 — Does the TensorFold maintainer have unpublished CUDA/DeepSeek work? Evidence: issue #14 closed 2026-09-29 by ac3de87c83bcb1018a7bb328a8b046a93eb6ed83 (Mac support shipped); comment says CUDA "still to come". Confirm before porting donor CUDA.
2. U2 — Adopt PR #119's numpy/mmap GGUF-reader pattern? PR #119 open; converter is glm5-specific Q8_0. Decide mmap reuse vs donor C GGUF reader.
3. U3 — Which donor kernel set (cuda/mmq vs gguf-tools/quants)? Donor ds4 ships a narrow quant API plus mmq kernels. Needs licensing/scope review.
4. U4 — License/attribution boundary for donor C modules under TensorFold's MIT tree. Donor ds4 LICENSE is MIT (ggml authors); mmq headers carry ggml lineage. Record notices before merge.

## Gate evidence

- G0 SATISFIED
  - G0a: GGUF discovery fixture fixed; historical revision pin preserved (commit 909221c).
  - G0b: serial-rejection test API pinned in tests/test_deepseek_v4_g0_discovery.py; RED->GREEN exit criteria documented.
  - G0c: focused CPU three-suite run passes (commit a4edf63).
  - Verified counts (three suites): 24 passed, 0 failed, 6 xfailed, 0 skipped. The 6 xfails are intentional RED pins (5 serial-contract rejections/signature + 1 MLX-gated check); documented as not qualification.
- R1-R3, R7 in force: R1 live services stay up; R2 read GGUF unchanged and validate from descriptors; R3 serial only (tp=1, one request, no drafting); R7 build/reproducibility recorded for release 1.

## Pending hardware checks

- T16: CUDA kernel bring-up on target (GB10) — PENDING; all kernel/residency claims unverified until T16.
- T19: full-model operator evidence (incl. FP8/FP4 lossiness) — PENDING.
- VMM: packed-KV demand-map residency — PENDING until T16/T19.

See the JSON manifest for the machine-readable version of every field above.
