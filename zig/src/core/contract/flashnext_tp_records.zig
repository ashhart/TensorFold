//! Flash Next pair handoff launches; the arithmetic and threadgroup scratch match the existing pair kernels.
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "metal flashnext tp tp_argmax_part", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal flashnext tp tp_pick", .threads = 1, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal flashnext tp tp_ids", .threads = 1, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal flashnext tp tp_copy", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal flashnext tp tp_plain", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal flashnext tp tp_branch", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "metal flashnext tp tp_merge", .threads = 256, .tg_memory = 128, .m1 = 1024, .m2 = 1024 },
};
