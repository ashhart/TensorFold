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

const cuda = @import("cuda");
const ByteDriver = struct {
    var buffers: [4][4096]u8 align(16) = undefined;
    var used: [4]bool = @splat(false);
    var fail_destination: u64 = 0;
    fn alloc(out: *cuda.abi.DevicePtr, n: usize) callconv(.c) cuda.abi.Result {
        if (n > buffers[0].len) return cuda.abi.error_not_ready;
        for (&used, 0..) |*busy, i| if (!busy.*) {
            busy.* = true;
            out.* = @intFromPtr(&buffers[i]);
            return 0;
        };
        return cuda.abi.error_not_ready;
    }
    fn free(ptr: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        for (&buffers, &used) |*buf, *busy| if (@intFromPtr(buf) == ptr) {
            std.debug.assert(busy.*);
            busy.* = false;
            return 0;
        };
        unreachable;
    }
    fn move(dst: cuda.abi.DevicePtr, src: cuda.abi.DevicePtr, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (dst == fail_destination) return cuda.abi.error_not_ready;
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        return 0;
    }
};

// A recurrent host writer stands in for kernels; all transaction copies use the production CUDA adapter.
const WrapRows = struct {
    rings: [2][256][2]u64 = @splat(@splat(@splat(0))),
    carries: [2][1024]u32 = @splat(@splat(0)),
    ids: [2][640]u32 = undefined,
    lens: [2]usize = @splat(0),
    projected: [2][64][1024]u32 = undefined,

    fn write(w: *WrapRows, s: usize, pos: usize, token: u32) void {
        const previous = w.carries[s][0];
        w.rings[s][pos % 256] = .{ @as(u64, previous) + token, pos };
        const next = previous *% 3 +% token;
        for (&w.carries[s], 0..) |*v, col| v.* = next +% @as(u32, @intCast(col));
    }
    fn seed(w: *WrapRows, s: usize, n: usize, salt: u32) void {
        for (0..n) |i| {
            const token = salt + @as(u32, @intCast(i % 31));
            w.write(s, i, token);
            w.ids[s][i] = token;
        }
        w.lens[s] = n;
    }
    fn window(w: *WrapRows, f: *fwd.Forward, s: usize, ids: []const u32) !void {
        try std.testing.expectEqual(w.lens[s], f.slot.pos);
        try std.testing.expectEqual(@intFromPtr(&w.rings[s]), f.runner.addressOf("s.L2.swa.v").?);
        try std.testing.expectEqual(@intFromPtr(&w.carries[s]), f.runner.addressOf("s.L2.carry").?);
        try f.transaction.begin(f, @intCast(ids.len));
        f.slot.pending = .{ .start = f.slot.pos, .n = @intCast(ids.len) };
        for (ids, 0..) |token, j| {
            const pos = w.lens[s] + j;
            w.write(s, pos, token);
            w.projected[s][j] = w.carries[s];
            w.ids[s][pos] = token;
            try w.visible(s, pos + 1);
        }
    }
    fn accept(w: *WrapRows, f: *fwd.Forward, s: usize, n: u32) !void {
        // The production caller selects the accepted projection before Transaction.keep.
        w.carries[s] = w.projected[s][n - 1];
        try f.transaction.keep(n);
        w.lens[s] += n;
        f.slot.pos = f.transaction.state.position;
        f.slot.pending = null;
        try w.visible(s, w.lens[s]);
    }
    fn visible(w: *const WrapRows, s: usize, end: usize) !void {
        var reference: [640][2]u64 = undefined;
        var carry: u32 = 0;
        for (w.ids[s][0..end], 0..) |token, pos| {
            reference[pos] = .{ @as(u64, carry) + token, pos };
            carry = carry *% 3 +% token;
        }
        const lo = end -| 128;
        for (lo..end) |pos| try std.testing.expectEqualSlices(u64, &reference[pos], &w.rings[s][pos % 256]);
        for (w.carries[s], 0..) |v, col| {
            const expected = if (end == 0) 0 else carry +% @as(u32, @intCast(col));
            try std.testing.expectEqual(expected, v);
        }
    }
    fn selected(w: *const WrapRows, ss: *slots.SlotSet, s: u32) !void {
        try ss.view(s);
        try std.testing.expectEqual(@intFromPtr(&w.carries[s]), ss.f.runner.addressOf("s.L2.carry").?);
        try std.testing.expectEqual(@as(u64, w.lens[s]), ss.f.slot.pos);
        try w.visible(s, w.lens[s]);
    }
};

test "Draft state: production carry-only transactions preserve visible wrap rows through drop retry and slot reuse" {
    const a = std.testing.allocator;
    var d = mock.init();
    ByteDriver.used = @splat(false);
    ByteDriver.fail_destination = 0;
    d.api.cuMemAlloc_v2 = ByteDriver.alloc;
    d.api.cuMemFree_v2 = ByteDriver.free;
    d.api.cuMemcpyDtoDAsync_v2 = ByteDriver.move;
    var runner: run.Runner = .{ .gpa = a, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer runner.deinit();
    var cfg: @import("config.zig").Config = .{};
    cfg.layers = 3;
    const widths: @import("block.zig").Widths = .{};
    var f = fwd.Forward.init(a, &cfg, &widths, .{}, &runner, undefined);
    defer {
        f.transaction.deinit(a);
        f.arena.deinit();
    }
    const fams = [_]kv.sessions.Family{.{ .name = "kv", .ratio = 1, .row_bytes = 1 }};
    var pool = try kv.sessions.Pool.init(a, .{ .families = &fams, .page = 16 }, 1024, 1, 0);
    defer pool.deinit();
    var poolslots = [_]*kv.sessions.Slot{ try pool.newSlot(512), try pool.newSlot(512) };
    var k: kvs.Kv = undefined;
    k.slots = &poolslots;
    k.dev = undefined;
    k.dev.pool = &pool;
    k.dev.slots = 2;
    k.dev.compute = runner.stream;
    f.kv = &k;
    const ss = try slots.SlotSet.init(a, &f, 2);
    defer ss.deinit();
    var rows: WrapRows align(16) = .{};
    try ss.roles.put(a, "s.L2.carry", .{ .base = @intFromPtr(&rows.carries), .bytes = @sizeOf(@TypeOf(rows.carries[0])) });
    try ss.roles.put(a, "s.L2.swa.v", .{ .base = @intFromPtr(&rows.rings), .bytes = @sizeOf(@TypeOf(rows.rings[0])) });
    rows.seed(0, 253, 7);
    rows.seed(1, 249, 19);
    ss.states[0].pos = rows.lens[0];
    ss.states[1].pos = rows.lens[1];
    try rows.selected(ss, 0);
    const rejected: [64]u32 = @splat(91);
    for (0..2) |_| {
        try rows.window(&f, 0, &rejected);
        try f.drop();
        try rows.visible(0, rows.lens[0]);
        try std.testing.expectEqual(@as(u64, 253), f.slot.pos);
        try std.testing.expectEqual(@as(?u32, null), f.transaction.state.pending);
    }
    try rows.window(&f, 0, &.{ 31, 32, 33, 34, 35, 36 });
    try rows.accept(&f, 0, 3);
    try rows.selected(ss, 1);
    try rows.window(&f, 1, &.{ 61, 62, 63, 64 });
    try rows.selected(ss, 0);
    try rows.window(&f, 0, &.{ 41, 42, 43, 44 });
    ByteDriver.fail_destination = @intFromPtr(&rows.carries[0]);
    defer ByteDriver.fail_destination = 0;
    try std.testing.expectError(error.NotReady, f.drop());
    try std.testing.expectEqual(@as(u64, 256), f.slot.pos);
    try std.testing.expectEqual(@as(u64, 256), f.transaction.state.position);
    try std.testing.expectEqual(@as(?u32, 4), f.transaction.state.pending);
    try std.testing.expect(f.slot.pending != null);
    ByteDriver.fail_destination = 0;
    try f.drop();
    try rows.visible(0, rows.lens[0]);
    try rows.window(&f, 0, &.{ 51, 52, 53, 54 });
    try rows.accept(&f, 0, 2);
    try ss.view(1);
    try std.testing.expectEqual(@as(?u32, 4), f.transaction.state.pending);
    try rows.accept(&f, 1, 2);
    try rows.selected(ss, 0);
    try rows.window(&f, 0, &.{ 71, 72, 73 });
    try f.release();
    try std.testing.expectEqual(@as(u64, 0), f.slot.pos);
    try std.testing.expect(f.slot.pending == null);
    try std.testing.expectEqual(@as(?u32, null), f.transaction.state.pending);
    // A fresh prefill initializes carry; the cancelled request must have rolled back before slot reuse.
    try rows.visible(0, rows.lens[0]);
    rows.carries[0] = @splat(0);
    rows.seed(0, 255, 23);
    f.slot.pos = rows.lens[0];
    try rows.selected(ss, 1);
    try rows.selected(ss, 0);
    try rows.window(&f, 0, &.{ 81, 82, 83, 84 });
    try rows.accept(&f, 0, 2);
    try rows.selected(ss, 1);
    try rows.window(&f, 1, &.{ 101, 102, 103 });
    try rows.accept(&f, 1, 3);
    try rows.selected(ss, 0);
    try ss.deactivate();
}
