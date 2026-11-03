//! HC vector mix resources from nvcc 13.0.88; later compiler sets remain pending.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "Flash Next vectorized BF16 HC mix", .symbol = "fn_hc_dense_mix_vec", .image = "flashnext_hc", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 2048, .bound = 256 }, .{ .regs = 31, .static = 2048, .bound = 256 }, .{ .regs = 31, .static = 2048, .bound = 256 }, .{ .regs = 29, .static = 2048, .bound = 256 }, .{ .regs = 29, .static = 2048, .bound = 256 } }, .later_pending = true },
};
test "HC vector mix holds its geometry and nvcc 13.0 usage, later sets pending" {
    const regs = [_]u32{ 32, 31, 31, 29, 29 };
    try std.testing.expectEqual(@as(u32, 256), records[0].launch.threads);
    try std.testing.expectEqualStrings("fn_hc_dense_mix_vec", records[0].launch.symbol);
    for (records) |r| {
        for (r.usage, regs) |usage, expected|
            try std.testing.expectEqual(limits.Usage{ .regs = expected, .static = 2048, .bound = 256 }, usage);
        try std.testing.expect(!r.geometry_pending and r.later_pending);
        try std.testing.expectEqual(r.usage, registry.usageFor(r, .{ .major = 13, .minor = 0 }));
        for ([_]limits.Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 }, .{ .major = 14, .minor = 0 } }) |version|
            try std.testing.expectEqual(registry.none, registry.usageFor(r, version));
    }
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records, true));
}
