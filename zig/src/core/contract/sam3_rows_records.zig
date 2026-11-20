const M = @import("limits.zig").MetalLaunch;

/// rows_attention.metal: one simdgroup a (row, head), 32 threads reserved in the source; and its planted control.
pub const metal = [_]M{
    .{ .name = "sam3 rows attention sam3_attention_rows", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "sam3 rows attention control sam3_attention_rows", .threads = 32, .tg_memory = 0, .reserved = 32 },
};
