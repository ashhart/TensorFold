//! registry.zig's record joins, one module a line, in order; add a module beside its family's lines, not at the end.
const registry = @import("registry.zig");

pub const cuda_after = @import("registry_media.zig").cuda ++
    @import("nemotron_prompt_records.zig").cuda ++
    @import("glm_cuda_records.zig").cuda ++
    @import("registry_shared_attention.zig").records(registry.CudaRecord) ++
    @import("gemma_cuda_records.zig").cuda ++
    @import("spark_cuda_records.zig").cuda ++
    @import("row_cuda_records.zig").cuda ++
    @import("flashnext_tile_records.zig").cuda ++
    @import("window_rows_records.zig").cuda ++
    @import("modelopt_lane_records.zig").cuda ++
    @import("nvfp4_fact_records.zig").cuda ++
    @import("nvfp4_experts32_records.zig").cuda ++
    @import("experts_bf16_records.zig").cuda ++
    @import("lane_half_records.zig").cuda ++
    @import("pocket_records.zig").cuda ++
    @import("minimax_cuda_records.zig").cuda ++
    @import("moss_image_cuda_records.zig").cuda ++
    @import("exl3_dense_records.zig").cuda ++
    @import("exl3_finish_records.zig").cuda ++
    @import("lfm2_moe_cuda_records.zig").cuda ++
    @import("dense_attention_records.zig").cuda ++
    @import("dots_mocr_cuda_records.zig").cuda;
pub const nvcc_132_after = @import("registry_learning.zig").cuda_132 ++
    @import("paging_pending.zig").cuda_132 ++
    @import("lfm2_moe_cuda_records.zig").cuda_132 ++
    @import("nemotron_prompt_records.zig").cuda_132 ++
    @import("glm_cuda_records.zig").cuda_132 ++
    @import("qwen_served_records.zig").cuda_132 ++
    @import("flashnext_tile_records.zig").cuda_132 ++
    @import("nvfp4_fact_records.zig").cuda_132 ++
    @import("experts_bf16_records.zig").cuda_132 ++
    @import("pocket_records.zig").cuda_132 ++
    @import("minimax_cuda_records.zig").cuda_132;
pub const nvcc_133_after = @import("flashnext_tile_records.zig").cuda_133 ++
    @import("glm_cuda_records.zig").cuda_133 ++
    @import("nvcc_133_records.zig").cuda_133 ++
    @import("minimax_cuda_records.zig").cuda_133 ++
    @import("nvfp4_experts32_records.zig").cuda_133 ++
    @import("paging_pending.zig").cuda_133 ++
    @import("experts_bf16_records.zig").cuda_133;
pub const cuda_pending = @import("qwen_whole_records.zig").records ++
    @import("flashnext_tile_records.zig").pending ++
    @import("exl3_prompt_pending.zig").records ++
    @import("nemotron_draw_records.zig").records ++
    @import("cuda_lane_pending.zig").records ++
    @import("weight_only_records.zig").cuda ++
    @import("hc_activation_records.zig").cuda ++
    @import("qwen4_records.zig").cuda;
pub const metal_before = @import("fsr_records.zig").metal ++
    @import("scale_index_records.zig").metal ++
    @import("lfm3_records.zig").metal;
pub const all_metal = @import("lane_q8_records.zig").records ++
    @import("attention_pv_records.zig").records ++
    registry.metal ++
    @import("delta_slots_records.zig").records ++
    @import("flashnext_pending.zig").records ++
    @import("simd_pairs_records.zig").records ++
    @import("simd_half_records.zig").records ++
    @import("metal_pair_records.zig").records ++
    @import("glm_rows_records.zig").records ++
    @import("weight_only_records.zig").metal ++
    @import("grouped_records.zig").metal ++
    @import("hc_activation_records.zig").metal ++
    @import("qwen4_records.zig").metal ++
    @import("lfm2_records.zig").records ++
    @import("nemotron_experts_records.zig").records ++
    @import("nemotron_pages_records.zig").records ++
    @import("nemotron_attention_records.zig").records ++
    @import("nemotron_prompt_mask_records.zig").records ++
    @import("nemotron_ssd_records.zig").records ++
    @import("flashnext_attention_records.zig").records ++
    @import("spark25_records.zig").records ++
    @import("gemma_attention_records.zig").records ++
    @import("moss_image_records.zig").records ++
    @import("h3_image_records.zig").records ++
    @import("dsv41_metal_records.zig").records ++
    @import("minimax_records.zig").records ++
    @import("ling_route_records.zig").records ++
    @import("qwen27_moe_records.zig").records ++
    @import("affine_gather_records.zig").records ++
    @import("flint_records.zig").records;
pub const landed_cuda = registry.cuda ++
    @import("dory_cuda_records.zig").cuda ++
    registry.gradient_pending ++
    registry.qwen_learning_pending ++
    @import("paging_pending.zig").extra ++
    @import("qwen_served_records.zig").records ++
    cuda_pending ++
    @import("qwen_decode_pending.zig").records ++
    @import("qwen_bf16_norm_records.zig").records ++
    @import("joint_cuda_records.zig").records ++
    registry.glm_pending.sampling ++
    @import("glm_kv_pending.zig").records ++
    @import("mla_cuda_records.zig").records ++
    @import("exl3_grouped_records.zig").records ++
    @import("proposal_sampling_records.zig").records ++
    @import("flashnext_pages_pending.zig").records ++
    @import("flashnext_heads_records.zig").records ++
    @import("glm_f32w_pending.zig").records ++
    @import("penalty_sampling_records.zig").records ++
    @import("flashnext_attn_lanes_pending.zig").records ++
    @import("glm_l2pf_pending.zig").records;
pub const all_cuda = @import("qmmf_prompt_records.zig").records ++
    @import("moetile_records.zig").records ++
    @import("gb10_many_records.zig").records ++
    @import("nvfp4_tile_pending.zig").records ++
    landed_cuda ++
    dsv41_pending ++
    @import("inc4_pending.zig").trees ++
    @import("inc4_pending.zig").rows ++
    @import("affine_verify_records.zig").records;
pub const dsv41_pending = @import("registry_union.zig").extra(
    landed_cuda,
    @import("dsv41_pending_1.zig").records ++
        @import("dsv41_pending_2.zig").records,
);

test {
    _ = @import("mla_cuda_records.zig");
}
