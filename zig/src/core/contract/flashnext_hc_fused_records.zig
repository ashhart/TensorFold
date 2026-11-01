//! HC fusion resources from nvcc 13.0.88; later compiler sets remain pending.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

// Static bytes exclude Blackwell's 1024-byte system reservation, as the cubin reader reports them.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "Flash Next writeback and normalize", .symbol = "fn_hc_writeback_norm", .image = "flashnext_hc", .threads = 256 }, .usage = .{ .{ .regs = 40, .static = 976, .bound = 256 }, .{ .regs = 48, .static = 976, .bound = 256 }, .{ .regs = 48, .static = 976, .bound = 256 }, .{ .regs = 48, .static = 976, .bound = 256 }, .{ .regs = 48, .static = 976, .bound = 256 } }, .later_pending = true },
};

test "HC fusion nvcc 13.0 receipts cover all five SMs and preserve later compiler gaps" {
    const regs = [_]u32{ 40, 48, 48, 48, 48 };
    for (records) |record| {
        try std.testing.expectEqualStrings("fn_hc_writeback_norm", record.launch.symbol);
        try std.testing.expectEqual(@as(u32, 256), record.launch.threads);
        try std.testing.expect(!record.geometry_pending and record.later_pending);
        for (record.usage, regs) |usage, expected|
            try std.testing.expectEqual(limits.Usage{ .regs = expected, .static = 976, .bound = 256 }, usage);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{record}, true));
        try std.testing.expectEqual(record.usage, registry.usageFor(record, .{ .major = 13, .minor = 0 }));
        for ([_]limits.Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 }, .{ .major = 14, .minor = 0 } }) |version|
            try std.testing.expectEqual(registry.none, registry.usageFor(record, version));
    }
}
