//! Shared tree picks: exactly the two 32-thread, zero-threadgroup-memory launches in argmaxChunks.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "tree round picks tf_round_pick_chunks", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "tree round picks tf_round_pick_rows", .threads = 32, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};
