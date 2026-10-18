const std = @import("std");
const fwd = @import("forward.zig");
const slots = @import("slots.zig");
const kvs = @import("cuda_kv.zig");
const kv = @import("kv");
const logical = @import("kv_state.zig");
fn copy(_: *anyopaque, dst: u64, src: u64, n: usize) !void {
    @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
}
test "Draft state: switching slots preserves independent pending transactions and rollback bytes" {
    const a = std.testing.allocator;
    var poolslots: [2]*kv.sessions.Slot = @splat(undefined);
    var k: kvs.Kv = undefined;
    k.slots = &poolslots;
    var f: fwd.Forward = undefined;
    f.kv = &k;
    f.transaction = .{};
    f.slot = .{};
    const ss = try slots.SlotSet.init(a, &f, 2);
    defer {
        ss.deinit();
        f.transaction.deinit(a);
    }
    var live = [_]u8{ 1, 2, 3, 4 };
    var backup: [4]u8 = undefined;
    const c: logical.Copy = .{ .ctx = &f, .run = copy };
    const regions = [_]logical.Region{.{ .live = @intFromPtr(&live), .backup = @intFromPtr(&backup), .row_bytes = 1, .rows = 4, .kind = .ring }};
    try ss.view(0);
    try ss.transaction(0).state.begin(&regions, c, 2);
    try ss.view(1);
    try ss.transaction(1).state.begin(&.{}, c, 1);
    live[0] = 9;
    live[1] = 8;
    try ss.view(0);
    try std.testing.expectEqual(@as(?u32, 2), ss.transaction(0).state.pending);
    try ss.transaction(0).state.rollback(&regions, c);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &live);
    try std.testing.expectEqual(@as(?u32, 1), ss.transaction(1).state.pending);
    try ss.transaction(1).state.keep(&.{}, c, 1);
    try ss.deactivate();
    try std.testing.expectEqual(@as(u64, 0), ss.transaction(0).state.position);
    try std.testing.expectEqual(@as(u64, 1), ss.transaction(1).state.position);
}
