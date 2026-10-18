//! CUDA image names and qualification flags.

/// Each .cu in zig/kernels/cuda (`src`, else `name`) with the flags its Python extension passes in `extra_cuda_cflags`.
pub const Kernel = struct { name: []const u8, flags: []const []const u8, src: ?[]const u8 = null, arch_specific: bool = false };

/// The torch-op replacements' qualification flags (runs/006): no contraction, no flush to zero.
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

/// IEEE division and square root, no contraction or flush: host glue references match bit for bit.
const glue = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true" };

pub const kernels = [_]Kernel{
    .{ .name = "attn_train", .flags = &.{"-O3"} },
    .{ .name = "qwen27_train", .flags = &.{"-O3"} },
    .{ .name = "gdn_train", .flags = &.{ "-O3", "--fmad=false" } },
    .{ .name = "train_q4", .flags = &.{"-O3"} },
    .{ .name = "train_attn", .flags = &.{"-O3"} },
    .{ .name = "flashnext_train", .flags = &.{"-O3"} },
    .{ .name = "flashnext_gdn", .flags = &.{ "-O3", "--fmad=false" } }, // qwen4_exp/cuda/gdn.cu chain, one row
    .{ .name = "flashnext_batch", .flags = &.{ "-O3", "--fmad=false" } }, // ours: Flash Next's batched prompt pass glue
    .{ .name = "flashnext_mtp", .flags = &.{"-O3"} }, // ours: Flash Next's MTP drafts, a row's best score and log-sum-exp
    .{ .name = "qmm", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py lane matmul, groups of 32
    .{ .name = "experts_rows", .flags = &.{"-O3"} }, // experts.cu's decode sums over a window or prompt chunk's rows
    .{ .name = "route_rows", .flags = glue }, // ours: router logits and softmax top-k rows (cuda/route_rows.zig)
    .{ .name = "flashnext_hc", .flags = glue }, // ours: Flash Next's hyper-connection and n-gram glue
    .{ .name = "flashnext_attn", .flags = glue }, // ours: Flash Next's attention glue and its indexer's pooling and scores
    .{ .name = "experts3", .flags = &.{"-O3"} }, // ours: 3-bit experts on experts.cu's steps, nibble words from planes
    .{ .name = "experts3_prefill", .flags = &.{"-O3"} },
    .{ .name = "moe", .flags = &.{"-O3"} }, // cuda/moe.py: softmax top-k routing with a sigmoid-gated shared expert
    .{ .name = "qwen36_mtp", .flags = &.{"-O3"} }, // ours: the Qwen3.6 MTP drafter's attention at the target's head width
    .{ .name = "ggml", .flags = &.{ "-O3", "--fmad=false" } },
    .{ .name = "gdn", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "probe", .flags = &.{"-O3"} },
    .{ .name = "qmm_group", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "qmm_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts", .flags = &.{"-O3"} }, // cuda/experts.py, tensorfold_experts_v7
    .{ .name = "experts_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts_pack", .flags = &.{"-O3"} },
    .{ .name = "prefill_attention", .flags = &.{ "-O3", "--fmad=false" } }, // tensorfold_prefill_attention_v1
    .{ .name = "scan_rows", .flags = &.{ "-O3", "--fmad=false" } }, // nemotron_h/cuda/mamba.py
    .{ .name = "nemotron_ops", .flags = &.{"-O3"} }, // ours: Nemotron's serial feed, routed plan and rests
    .{ .name = "affine4_pack", .flags = &.{"-O3"} }, // ours: the MLX affine-4 repack into qlinear's tiles
    .{ .name = "affine8", .flags = &.{"-O3"} }, // ours: MLX affine 2 to 8 bits on lane_gemv tiles of their own width
    .{ .name = "affine_prefill", .flags = &.{"-O3"} }, // ours: qmm_prefill's prompt GEMM over affine8's tiles
    .{ .name = "fp8_experts", .flags = &.{"-O3"} }, // fp8/experts.cu's device code, tensorfold_fp8_experts_v6
    .{ .name = "exlr8", .flags = &.{ "-O3", "--fmad=false" } }, // ours: EXLR8 lane projections, the Metal kernels' order
    .{ .name = "exlr8_dense", .flags = &.{ "-O3", "--fmad=false" } }, // ours: EXLR8 dense matrices by row, the Metal order
    .{ .name = "nvfp4_experts", .flags = &.{"-O3"} }, // nvfp4/experts.cu's device code, the NVFP4 expert kernels
    .{ .name = "rank_sum", .flags = &.{"-O3"} }, // ours: the rank-order sum of a reduce-scatter (rank_scatter.zig)
    .{ .name = "lane_gemv", .flags = &.{"-O3"} }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    .{ .name = "sample", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the Metal engine's keyed draws
    .{ .name = "train", .flags = &.{"-O3"} }, // ours: Sliding Weights' learning step (train.metal's kernels)
    .{ .name = "qwen35", .flags = &.{"-O3"} }, // ours: the Qwen3.5 hybrids' glue, the Metal kernels' rounding points
    .{ .name = "qwen35_draft", .flags = &.{"-O3"} }, // ours: the DFlash2 drafter's glue (draft_ops.metal's rounding)
    .{ .name = "qwen35_keep", .flags = &.{"-O3"} }, // ours: a verified window's kept path applied to its stream's state
    .{ .name = "qwen_vision", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the Qwen3.5 vision tower's glue and attention
    .{ .name = "qwen35_kv8", .flags = &.{"-O3"} }, // ours: --kv-cache fp8's e4m3 attention rows for the Qwen3.5 hybrids
    .{ .name = "nemotron_norms", .flags = glue }, // ours, each with a host reference (glue_ref.zig): Nemotron's glue
    .{ .name = "nemotron_route", .flags = glue },
    .{ .name = "nemotron_mamba", .flags = glue },
    .{ .name = "nemotron_attention", .flags = glue },
    .{ .name = "nemotron_keyed", .flags = glue },
    .{ .name = "nemotron_latent", .flags = glue }, // ours: a latent MoE's wide picks, share sums and shared add
    .{ .name = "ck_lane", .flags = &.{"-O3"} }, // nvfp4/lane4.cu's A8: FP8 rows times FP8 weights (checkpoint math)
    .{ .name = "ck_experts", .flags = &.{"-O3"}, .arch_specific = true }, // ours: NVFP4 x NVFP4 experts (sm_121a)
    .{ .name = "norms", .flags = &.{"-O3"} }, // ours: RMS norm rows for every family (cuda/norms.zig)
    .{ .name = "embed_rows", .flags = &.{"-O3"} }, // ours: 4-bit affine and bf16 token rows (cuda/embed_rows.zig)
    .{ .name = "lfm2", .flags = glue }, // ours: LFM2's short convolution, QK norm and rotary, gates and label rows
    .{ .name = "vit", .flags = glue }, // ours: vision towers' LayerNorm, Linear epilogues, activations, unshuffle
    .{ .name = "tree_round", .flags = &.{"-O3"} }, // ours: verified tree rounds for every family (cuda/tree_round.zig)
    .{ .name = "act_rows", .flags = &.{"-O3"} }, // ours: SwiGLU, output gates, group sums (cuda/act_rows.zig)
    .{ .name = "gdn_prep", .flags = &.{"-O3"} }, // ours: Gated DeltaNet conv, norms, g and beta (cuda/gdn_prep.zig)
    .{ .name = "qk_rows", .flags = &.{"-O3"} }, // ours: QK norm, RoPE and the cache write (cuda/qk_rows.zig)
    .{ .name = "qmmf", .flags = &.{"-O3"} }, // nvfp4/qmmf.cu's device code (FP8G, FP4), tensorfold_nvfp4_v3
    .{ .name = "prompt16", .flags = &.{"-O3"} }, // nvfp4/prompt.cu's FP4 and FP8 tile 4, tensorfold_nvfp4_prompt_v1
    .{ .name = "tree_attention", .flags = &.{"-O3"} }, // ours: committed keys and paths (cuda/tree_attention.zig)
    .{ .name = "block_attention", .flags = &.{"-O3"} }, // ours: a drafter's block over its context (cuda/block_attention.zig)
    .{ .name = "window_attention", .flags = &.{"-O3"} }, // ours: sliding-window attention (cuda/window_attention.zig)
    .{ .name = "moe_glue", .flags = &.{"-O3"} }, // ours: sigmoid top-k and the weighted combine (cuda/moe_glue.zig)
    .{ .name = "dense_rows", .flags = &.{"-O3"} }, // ours: bf16 rows times weight rows into fp32 (cuda/dense_rows.zig)
    .{ .name = "modelopt_pack", .flags = &.{"-O3"} }, // ours: ModelOpt tensors into the lane, prompt and expert layouts
    .{ .name = "glm_kda", .flags = &.{ "-O3", "--fmad=false" } }, // GLM: the Python family's glm5_next/cuda/kda.cu
    .{ .name = "glm_glue", .flags = &.{"-O3"} }, // ours: the Python family's Triton glue by hand, its PTX op for op
    .{ .name = "glm_ops", .flags = &.{"-O3"} }, // ours: GLM's integer work the Python family did in torch (the DSA token lists)
    .{ .name = "glm_quant", .flags = torch_ops }, // ours: GLM's draft-only weight quantizers, torch at load in the Python family
    .{ .name = "exl3_linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // exl3/linear.cu's device code, tensorfold_exl3_linear_v5
    .{ .name = "exl3_experts", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // exl3/experts.cu's device code, tensorfold_exl3_experts
    .{ .name = "exl3_candidate", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // #806's EXL3 tiles: Dense's candidate
    .{ .name = "glm_vision", .flags = &.{"-O3"} }, // ours: GLM's image tower steps around cuBLASLt (norms, rotary, softmax, merger)
    .{ .name = "glm_mm", .flags = &.{"-O3"} }, // ours: GLM's bf16 linears on tensor cores, weights tiled at load, row-stable
    .{ .name = "glm_dflash", .flags = &.{"-O3"} }, // the DFlash2 block drafter
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
    .{ .name = "roce_gather", .src = "roce_gather", .flags = &.{"-O3"} }, // b12x's one-shot RoCE all-gather, two ranks
    .{ .name = "drama3_staged_gemv", .flags = glue },
    .{ .name = "drama3", .flags = &.{"-O3"} },
    .{ .name = "drama3_fp8", .flags = &.{"-O3"} },
    .{ .name = "drama3_codec", .flags = &.{"-O3"} },
};

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
pub const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};
