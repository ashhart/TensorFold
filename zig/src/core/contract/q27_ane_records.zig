//! Metal launch declarations for the ANE+GPU BF16 residual-add kernel.
const std = @import("std");
const limits = @import("limits.zig");

pub const records = [_]limits.MetalLaunch{
    .{ .name = "q27 ane awq q27_add_ane_bf16", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "ANE residual-add Metal launch fits every Apple GPU family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, false));
}
