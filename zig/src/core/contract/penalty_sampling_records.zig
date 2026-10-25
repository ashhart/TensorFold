//! tf_penalize_rows' resources: nvcc 13.0.88 (the baseline), 13.2 and 13.3 compile it the same on all five targets.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "sample tf_penalize_rows", .symbol = "tf_penalize_rows", .image = "sample", .threads = 256 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 } } },
};

test "penalty receipt: the 13.0 measurement fits every target, and the later releases read the same" {
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records, true));
    for (records[0].usage, [_]u32{ 14, 14, 14, 16, 16 }) |usage, regs|
        try std.testing.expectEqualDeep(limits.Usage{ .regs = regs, .static = 0, .bound = 256, .local = 0 }, usage);
    for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |rel|
        try std.testing.expectEqualDeep(records[0].usage, registry.usageFor(records[0], rel));
}

test "penalty receipt: the landed and full unions hold the measured entry once" {
    for ([_][]const registry.CudaRecord{ &registry.landed_cuda, &registry.all_cuda }) |list| {
        var found: usize = 0;
        for (list) |record| if (std.mem.eql(u8, record.launch.image, "sample") and std.mem.eql(u8, record.launch.symbol, "tf_penalize_rows")) {
            found += 1;
            try std.testing.expectEqualDeep(records[0], record);
        };
        try std.testing.expectEqual(@as(usize, 1), found);
    }
}
