//! DeepSeek V4.1's Metal launches: the family's record modules chained, so registry.zig stays one screen per family.
const M = @import("limits.zig").MetalLaunch;

const head = @import("dsv41_fp8_records.zig").records ++ @import("dsv41_fp8_mma_records.zig").records ++
    @import("dsv41_attn_records.zig").records ++ @import("dsv41_experts_records.zig").records ++
    @import("dsv41_xmma_records.zig").records;
pub const records = head;
