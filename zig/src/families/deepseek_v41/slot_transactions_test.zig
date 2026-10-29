const std = @import("std");
const fwd = @import("forward.zig");
const slots = @import("slots.zig");
const kvs = @import("cuda_kv.zig");
const kv = @import("kv");
const logical = @import("kv_state.zig");
const mock = @import("load_mock.zig");
const run = @import("run.zig");
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
test "Draft state: a window's transaction backs up only the ratio-2 carries, never the SWA rings" {
    const a = std.testing.allocator;
    var d = mock.init();
    var runner: run.Runner = .{ .gpa = a, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer runner.deinit();
    var cfg: @import("config.zig").Config = .{};
    cfg.layers = 2;
    var widths: @import("block.zig").Widths = .{};
    var f = fwd.Forward.init(a, &cfg, &widths, .{}, &runner, undefined);
    defer {
        f.transaction.deinit(a);
        f.arena.deinit();
    }
    const roles = [_][]const u8{ "s.L0.swa.v", "s.L0.swa.s", "s.L1.swa.v", "s.L1.swa.s", "s.L1.carry" };
    for (roles, 1..) |role, i| try runner.external(role, 256 * i);
    f.slot = .{ .pos = 9 };
    try f.transaction.begin(&f, 4);
    try std.testing.expectEqual(@as(usize, 1), f.transaction.regions.items.len);
    const r = f.transaction.regions.items[0];
    try std.testing.expectEqual(@as(u64, 256 * 5), r.live);
    try std.testing.expect(r.kind == .carry);
    try std.testing.expectEqual(8 * @as(usize, cfg.head_dim), r.row_bytes);
    try f.transaction.keep(2);
    try std.testing.expectEqual(@as(u64, 11), f.transaction.state.position);
    try std.testing.expectEqual(@as(?u32, null), f.transaction.state.pending);
}
