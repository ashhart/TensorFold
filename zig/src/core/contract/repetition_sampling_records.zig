//! Repetition sampling launches; nvcc 13.0/13.2 remain pending until independently measured.
const std = @import("std");
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "sample tf_draw_repetition", .symbol = "tf_draw_repetition", .image = "sample", .threads = 1024 }, .usage = registry.none },
    .{ .launch = .{ .name = "sample tf_greedy_repetition", .symbol = "tf_greedy_repetition", .image = "sample", .threads = 1024 }, .usage = registry.none },
};

test "repetition launches preserve missing earlier compiler receipts" {
    for (records) |r| {
        try std.testing.expectEqualDeep(registry.none, registry.usageFor(r, null));
        try std.testing.expectEqualDeep(registry.none, registry.usageFor(r, .{ .major = 13, .minor = 2 }));
        try std.testing.expectEqual(@as(usize, 5), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}

test "repetition 13.3 measurements fit all five chips and persist as the newest compiler set" {
    for (records) |r| {
        const measured = registry.usageFor(r, .{ .major = 13, .minor = 3 });
        for (measured) |usage| try std.testing.expect(usage.regs > 0);
        var selected = r;
        selected.usage = measured;
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{selected}, true));
        try std.testing.expectEqualDeep(measured, registry.usageFor(r, .{ .major = 14, .minor = 0 }));
        var found: usize = 0;
        for (registry.all_cuda) |entry| if (std.mem.eql(u8, entry.launch.name, r.launch.name)) {
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
    }
}
