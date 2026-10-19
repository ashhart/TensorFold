//! Prior pitched-draw resources from increment 164; refresh after the single-buffer source change.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

// The owner's receipt predates buffer sharing; new compiler resources are still owed.
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "sample tf_draw_pitched", .symbol = "tf_draw_pitched", .image = "sample", .threads = 1024 }, .usage = @splat(.{ .regs = 64, .static = 19504, .bound = 1024, .local = 0 }) },
};

test "pitched resource receipt: the last measured launch fits every recorded target" {
    const record = records[0];
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records, true));
    for (limits.sms, record.usage, [_]u32{ 80, 86, 89, 120, 121 }) |target, usage, sm| {
        try std.testing.expectEqual(sm, target.sm);
        try std.testing.expectEqualDeep(limits.Usage{ .regs = 64, .static = 19504, .bound = 1024, .local = 0 }, usage);
    }
    try std.testing.expectEqualDeep(record.usage, registry.usageFor(record, null));
    try std.testing.expectEqualDeep(record.usage, registry.usageFor(record, .{ .major = 13, .minor = 2 }));
}

test "pitched resource receipt: landed and collected unions contain one copy and pending contains none" {
    for ([_][]const registry.CudaRecord{ &registry.landed_cuda, &registry.all_cuda }) |list| {
        var found: usize = 0;
        for (list) |record| if (std.mem.eql(u8, record.launch.image, "sample") and
            std.mem.eql(u8, record.launch.symbol, "tf_draw_pitched"))
        {
            found += 1;
            try std.testing.expectEqualDeep(records[0], record);
        };
        try std.testing.expectEqual(@as(usize, 1), found);
    }
    try std.testing.expectEqual(@as(usize, 0), @import("proposal_sampling_pending.zig").records.len);
}
