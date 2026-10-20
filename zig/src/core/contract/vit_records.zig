//! RADIO's fused epilogue launch, measured with the release fatbins' nvcc.
const std = @import("std");
const registry = @import("registry.zig");

pub const record: registry.CudaRecord = .{
    .launch = .{ .name = "vit tf_vit_bias_act", .symbol = "tf_vit_bias_act", .image = "vit", .threads = 256 },
    .usage = .{ .{ .regs = 15, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 } },
};

test "RADIO fused epilogue is in the live registry and fits every SM" {
    var found: usize = 0;
    for (registry.all_cuda) |active| if (std.mem.eql(u8, active.launch.symbol, record.launch.symbol)) {
        try std.testing.expectEqualDeep(record, active);
        found += 1;
    };
    try std.testing.expectEqual(@as(usize, 1), found);
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{record}, true));
}
