//! Repetition sampling launches; compiler resources remain pending until measured from the current source.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "sample tf_draw_repetition", .symbol = "tf_draw_repetition", .image = "sample", .threads = 1024 }, .usage = registry.none },
    .{ .launch = .{ .name = "sample tf_greedy_repetition", .symbol = "tf_greedy_repetition", .image = "sample", .threads = 1024 }, .usage = registry.none },
};

test "repetition launch resources stay pending until measured from the current source" {
    for (records) |r| {
        for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 }, .{ .major = 14, .minor = 0 } }) |release|
            try std.testing.expectEqualDeep(registry.none, registry.usageFor(r, release));
        try std.testing.expectEqual(@as(usize, 5), try registry.checkCuda(std.testing.allocator, &.{r}, true));
        var found: usize = 0;
        for (registry.all_cuda) |entry| if (std.mem.eql(u8, entry.launch.name, r.launch.name)) {
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
    }
}
