//! DeepSeek V4.1 top-512 over fp32 score rows: one 1,024-thread group a row, which the source reserves.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "dsv41 topk512 tf_topk512", .threads = 1024, .tg_memory = 8592, .reserved = 1024, .m1 = 1024, .m2 = 1024 },
};

test "the DeepSeek V4.1 top-512 launch fits every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
