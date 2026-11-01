//! DeepSeek V4.1's Metal launches: the family's record modules chained, so registry.zig stays one screen per family.
const M = @import("limits.zig").MetalLaunch;

const head = @import("dsv41_fp8_records.zig").records ++ @import("dsv41_fp8_mma_records.zig").records ++
    @import("dsv41_topk512_records.zig").records ++ @import("dsv41_attn_records.zig").records ++
    @import("dsv41_experts_records.zig").records ++ @import("dsv41_index_records.zig").records ++
    @import("dsv41_engram_records.zig").records ++ @import("dsv41_glue_records.zig").records ++
    @import("dsv41_compress_records.zig").records ++ @import("dsv41_cand_records.zig").records ++
    @import("dsv41_xmma_records.zig").records ++ @import("dsv41_mv_records.zig").records ++
    @import("dsv41_vision_records.zig").records ++ @import("dsv41_hc_records.zig").records;

pub const records = head ++ @import("dsv41_fold_records.zig").records ++ @import("dsv41_glue_tmp_records.zig").records ++
    @import("dsv41_fused_records.zig").records ++ @import("dsv41_route_records.zig").records;
