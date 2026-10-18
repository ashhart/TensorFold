const std = @import("std");
const t = std.testing;
const source = @import("copy_source.zig");
const budget = @import("copy_source_budget.zig");
const Fake = @import("copy_source_extra_fake.zig").Fake;
test "normal NotReady saturation refuses before a second allocation across contexts and reloads" {
    budget.setForTest(.{ .bytes = 32768, .nodes = 1 });
    defer budget.setForTest(.{});
    var d = Fake.init();
    var other = Fake.init();
    source.setCurrent(@ptrFromInt(1));
    defer {
        Fake.fail = false;
        source.setCurrent(@ptrFromInt(2));
        source.synchronizeContext(&other) catch unreachable;
        source.setCurrent(@ptrFromInt(1));
        source.synchronizeContext(&d) catch unreachable;
    }
    try source.enqueue(&d, 256, &.{7}, .{ .d = &d, .handle = null });
    // The event is NotReady without a failed wait. A second context and driver must share the limit.
    source.setCurrent(@ptrFromInt(2));
    const accepted = if (source.enqueue(&other, 256, &.{8}, .{ .d = &other, .handle = null })) |_| true else |e| blk: {
        try t.expectEqual(error.CopySourceLimit, e);
        break :blk false;
    };
    try t.expect(!accepted);
    try t.expectEqual(@as(usize, 1), source.retained());
}
test "active Stage and Buffer allocations share the byte limit before failed disposal" {
    budget.setForTest(.{ .bytes = 5000, .nodes = 16 });
    defer budget.setForTest(.{});
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(6));
    var s: source.Stage = .{};
    defer {
        Fake.fail = false;
        source.synchronizeContext(&d) catch unreachable;
        s.deinit();
    }
    _ = try s.take(&d, 16);
    const bytes: [8192]u8 = @splat(0);
    try t.expectError(error.CopySourceLimit, source.synchronousCopy(&d, 256, &bytes, t.allocator));
}
test "quarantinable CPU storage outlives a caller arena" {
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(7));
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    try t.expectError(error.NotReady, source.synchronousCopy(&d, 256, &.{0x69}, arena.allocator()));
    const borrowed = arena.queryCapacity() != 0;
    if (!borrowed) arena.deinit();
    Fake.fail = false;
    try source.synchronizeContext(&d);
    if (borrowed) arena.deinit();
    try t.expect(!borrowed);
    try t.expectEqual(@as(u8, 0x69), Fake.result[0]);
}
test "quarantinable Stage metadata outlives a caller arena" {
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(8));
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    var s: source.Stage = .{ .gpa = arena.allocator() };
    @memset(try s.take(&d, 16), 0x61);
    try s.copyTo(.{ .d = &d, .ptr = 256, .len = 16 }, 0, .{ .d = &d, .handle = null });
    s.deinit();
    const borrowed = arena.queryCapacity() != 0;
    if (!borrowed) arena.deinit();
    Fake.fail = false;
    try source.synchronizeContext(&d);
    if (borrowed) arena.deinit();
    try t.expect(!borrowed and Fake.early == 0);
    try t.expectEqual(@as(u8, 0x61), Fake.result[0]);
}

test "failed Stage disposal and active Buffer share the same finite reservation" {
    budget.setForTest(.{ .bytes = 5000, .nodes = 2 });
    defer budget.setForTest(.{});
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(60));
    var s: source.Stage = .{};
    defer {
        Fake.fail = false;
        source.synchronizeContext(&d) catch unreachable;
        s.deinit();
    }
    @memset(try s.take(&d, 16), 9);
    try s.copyTo(.{ .d = &d, .ptr = 256, .len = 16 }, 0, .{ .d = &d, .handle = null });
    s.deinit();
    try t.expectError(error.CopySourceLimit, source.Buffer.alloc(&d, 4096));
    try t.expectEqual(@as(usize, 1), source.retained());
    try t.expectEqual(@as(usize, 0), Fake.early);
    Fake.fail = false;
    try source.synchronizeContext(&d);
    var b = try source.Buffer.alloc(&d, 4096);
    defer b.free();
    var extra: source.Stage = .{};
    defer extra.deinit();
    try t.expectError(error.CopySourceLimit, extra.take(&d, 1));
}
test "ordinary NotReady event queries cannot bypass the node cap" {
    budget.setForTest(.{ .bytes = 32768, .nodes = 1 });
    defer budget.setForTest(.{});
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(61));
    defer {
        Fake.fail = false;
        source.synchronizeContext(&d) catch unreachable;
    }
    const stream: @import("stream.zig").Stream = .{ .d = &d, .handle = null };
    try source.enqueue(&d, 256, &.{1}, stream);
    try t.expectError(error.CopySourceLimit, source.enqueue(&d, 256, &.{2}, stream));
    try t.expectEqual(@as(usize, 1), source.retained());
    try t.expectEqual(@as(usize, 0), Fake.early);
}
