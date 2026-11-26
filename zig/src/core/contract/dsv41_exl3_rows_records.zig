//! DeepSeek V4.1's 2-16 row EXL3 projection and its gate/up SwiGLU: one 32-lane simdgroup a group, source-reserved.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

fn one(comptime name: []const u8) M {
    return .{ .name = "dsv41 exl3 rows " ++ name, .threads = 32, .tg_memory = 0, .reserved = 32 };
}

pub const records = [_]M{
    one("tf_exl3_matmul_rows1x2"), one("tf_exl3_matmul_rows1x4"), one("tf_exl3_matmul_rows1x8"),
    one("tf_exl3_matmul_rows2x8"), one("tf_exl3_swiglu"),
};

test "the DeepSeek V4.1 2-16 row EXL3 launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
