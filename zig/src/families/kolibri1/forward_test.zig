const std = @import("std");
const fwd = @import("forward.zig");
const config = @import("config.zig");

test "kolibri1_1 plan: the two dense layers count their slot ids and weights once (8,192 rows: 65,536 bytes)" {
    var why: config.Why = .{};
    var m: fwd.Model = undefined;
    m.c = try config.parse(std.testing.allocator, config.kolibri11.pinned, &why);
    m.dense = .{ 0, 0 };
    m.planned = 0;
    try m.denseSlots(undefined, 8192);
    try std.testing.expectEqual(@as(usize, 2 * 8192 * 4), m.planned.?);
}
