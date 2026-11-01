//! DeepSeek V4.1's Metal launches: the family's record modules chained, so registry.zig stays one screen per family.
pub const records = @import("dsv41_fp8_records.zig").records ++ @import("dsv41_fp8_mma_records.zig").records ++ @import("dsv41_experts_records.zig").records;
