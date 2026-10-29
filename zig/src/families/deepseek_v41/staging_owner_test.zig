const std = @import("std");
const cuda = @import("cuda");
const mock = @import("load_mock.zig");
test "Draft state: row-table copy errors retain staging before a retry or close" {
    var d = mock.init();
    var table = try @import("batch_graph.zig").Table.init(&d, 2);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    defer {
        mock.fail_sync = false;
        table.drain() catch unreachable;
        table.deinit();
    }
    const mix = [_]@import("rowtab.zig").Seg{.{ .slot = 0, .start = 0, .rows = 1 }};
    mock.fail_copy = true;
    try std.testing.expectError(error.NotReady, table.stage(stream, &mix, &.{7}, 1, 1));
    const before = try std.testing.allocator.dupe(u8, mock.pinned.?);
    defer std.testing.allocator.free(before);
    mock.fail_sync = true;
    try std.testing.expectError(error.NotReady, table.stage(stream, &mix, &.{9}, 1, 1));
    try std.testing.expectEqualSlices(u8, before, mock.pinned.?);
    try std.testing.expectEqual(@as(usize, 0), mock.early_free);
}
test "Draft state: installed pack releases segments and removes its hook after all buffers close" {
    var d = mock.init();
    const a = std.testing.allocator;
    const previous = cuda.memory.currentHook();
    const x = try a.create(@import("dev_arena.zig").Installed);
    x.* = .{ .plain = .{ .d = &d }, .backend = .pack };
    x.pack = cuda.pack.Pack.init(a, x.plain.backend());
    x.pack.?.seg_min = 4096;
    x.pack.?.contiguous = false;
    x.pack.?.install();
    x.pack.?.begin();
    var buf = try cuda.DeviceBuffer.alloc(&d, 32);
    buf.free();
    x.deinit(a);
    try std.testing.expectEqual(previous, cuda.memory.currentHook());
    try std.testing.expectEqual(mock.allocated, mock.freed);
}
test "Draft state: DSpark statics hand out their own region without a context drain" {
    var d = mock.init();
    var bytes: [64]u8 align(16) = @splat(7);
    var dev: @import("dspark_dev.zig").Dev = undefined;
    dev.regions = 2;
    dev.region_bytes = 32;
    dev.stage = .{ .d = &d, .bytes = &bytes };
    dev.agreement = undefined;
    dev.agreement.stream = .{ .d = &d, .handle = null };
    // any synchronize fails: uploadAsync's copy_source custody, not a drain, frees the region for its next pass
    mock.fail_sync = true;
    defer mock.fail_sync = false;
    for (0..2) |g| {
        const sv = try dev.statics(g, 2, 3);
        try std.testing.expectEqual(@as([*]u8, @ptrCast(&bytes[32 * g])), @as([*]u8, @ptrCast(sv[0].ptr)));
        try std.testing.expectEqual(@as([*]u8, @ptrCast(&bytes[32 * g + 16])), @as([*]u8, @ptrCast(sv[1].ptr)));
        try std.testing.expectEqual(@as(usize, 3), sv[1].len);
    }
    try std.testing.expectError(error.Statics, dev.statics(2, 1, 1));
    try std.testing.expectError(error.Statics, dev.statics(0, 4, 1));
}
test "Draft state: prefill teardown releases the second microbatch and workspace owners" {
    var d = mock.init();
    var buffers: [6]cuda.DeviceBuffer = undefined;
    for (&buffers) |*b| b.* = try cuda.DeviceBuffer.alloc(&d, 32);
    defer for (&buffers) |*b| {
        if (mock.live[b.ptr / 256]) b.free();
    };
    var state: @import("forward_prefill.zig").State = .{};
    state.long_b.even = buffers[0];
    state.long_b.pos = buffers[1];
    state.long_b.lazy = .{ .bytes = 32, .buf = buffers[2] };
    state.k4ws.buf = buffers[3];
    state.k4ws.send = buffers[4];
    state.k4ws.gathered = buffers[5];
    state.deinit(std.testing.allocator);
    try std.testing.expectEqual(mock.allocated, mock.freed);
}
