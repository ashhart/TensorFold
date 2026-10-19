const cuda_build = @import("cuda_kernels.zig");
pub const units = [_]cuda_build.Kernel{
    .{ .name = "dsv41_exl3_experts", .src = "exl3_block/exl3_experts", .flags = &.{"-O3"} }, // tensorfold_exl3_experts_v1
    .{ .name = "dsv41_x3ld", .src = "exl3_block/x3ld", .flags = &.{"-O3"} }, // tf_dsv41_x3ld_v1
    .{ .name = "dsv41_x3pf", .src = "exl3_block/x3pf", .flags = &.{"-O3"} }, // tf_dsv41_x3pf_v1
    .{ .name = "dsv41_x3gm", .src = "exl3_block/x3gm", .flags = &.{"-O3"} }, // tf_dsv41_x3gm_v1
    .{ .name = "dsv41_linear", .src = "exl3_block/linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tensorfold_exl3_linear_v4
    .{ .name = "dsv41_x3seg", .src = "exl3_block/x3seg", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_x3seg_v3
    .{ .name = "dsv41_dense3", .src = "exl3_block/dense3", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_dense3_v1
    .{ .name = "dsv41_attn", .src = "deepseek_v41/attn_cuda", .min_sm = 90, .flags = &.{ "-O3", "--fmad=false" } }, // tf_dsv41_attn_cuda_v1
    .{ .name = "dsv41_topk", .src = "deepseek_v41/topk_cuda", .min_sm = 90, .flags = &.{ "-O3", "--fmad=false" } }, // tf_dsv41_attn_cuda_v1
    .{ .name = "dsv41_mhc", .src = "deepseek_v41/mhc_cuda", .flags = &.{ "-O3", "--fmad=false" } }, // tf_dsv41_mhc_cuda_v1
    .{ .name = "dsv41_mhc_pf", .src = "deepseek_v41/mhc_pf", .flags = &.{ "-O3", "--fmad=false" } }, // tf_dsv41_mhc_pf_v1
    .{ .name = "dsv41_router_gemv", .src = "deepseek_v41/router_gemv", .flags = &.{"-O3"} }, // tf_dsv41_router_gemv_v2
    .{ .name = "dsv41_pfdense_k8", .src = "exl3_block/pfdense_k8", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_pfdense_v1
    .{ .name = "dsv41_pfdense_k10", .src = "exl3_block/pfdense_k10", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_pfdense_k12", .src = "exl3_block/pfdense_k12", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_pfdense_k16", .src = "exl3_block/pfdense_k16", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_engram_gate", .src = "deepseek_v41/engram_gate", .flags = &.{"-O3"} }, // tf_dsv41_engram_gate_v1
    .{ .name = "dsv41_l2pace", .src = "deepseek_v41/l2pace", .flags = &.{"-O3"} }, // tf_dsv41_l2pace_v1
    .{ .name = "dsv41_l2pf", .src = "deepseek_v41/l2pf", .flags = &.{"-O3"} }, // tf_dsv41_l2pf_v1
    .{ .name = "dsv41_x3gm_plan", .src = "exl3_block/x3gm_plan", .flags = &.{"-O3"} }, // ours: x3gm.plan's torch ops
    .{ .name = "dsv41_topk_keys", .src = "deepseek_v41/topk_keys", .flags = &.{"-O3"} }, // ours: pick.top's torch.topk
    .{ .name = "dsv41_pointwise", .src = "deepseek_v41/pointwise", .flags = &.{ "-O3", "--fmad=false" } }, // ours: prefill MoE's torch pointwise ops
    .{ .name = "dsv41_kvsplit", .src = "deepseek_v41/kvsplit", .flags = &.{"-O3"} }, // ours: split KV's exchange copies
    .{ .name = "dsv41_glue", .src = "deepseek_v41/glue", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the window's torch glue
    .{ .name = "dsv41_x3gm3", .src = "exl3_block/x3gm3", .flags = &.{"-O3"} }, // ours: x3gm v3 (x3gm.cu's device code)
    .{ .name = "dsv41_pfglue", .src = "deepseek_v41/prefill_glue", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the prefill segment's torch glue
    .{ .name = "dsv41_router_glue", .src = "deepseek_v41/router_glue", .flags = &.{"-O3"} }, // histogram group + upstream rotation
    .{ .name = "dsv41_kv_glue", .src = "deepseek_v41/kv_glue", .flags = &.{"-O3"} }, // KV RMS in rounded FP64; store keeps Triton's FMA
    .{ .name = "dsv41_x3ld_epi", .src = "exl3_block/x3ld_epi", .flags = &.{"-O3"} }, // ours: x3ld + the expert epilogues
    .{ .name = "dsv41_topk_b", .src = "deepseek_v41/topk_b", .min_sm = 90, .flags = &.{ "-O3", "--fmad=false" } }, // ours: topk_cuda.cu row-bounded
};

/// Optional images: the keyed sampler (TF_DSV41_SAMPLING) and two-tile linears (TF_DSV41_LIN2); empty, they refuse.
pub const optional = [_]cuda_build.Kernel{
    .{ .name = "dsv41_sampling", .src = "deepseek_v41/sampling", .flags = &.{ "-O3", "--fmad=false" } },
    .{ .name = "dsv41_lin2", .src = "exl3_block/lin2", .min_sm = 90, .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
};
