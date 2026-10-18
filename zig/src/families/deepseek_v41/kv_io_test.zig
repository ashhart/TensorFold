const std = @import("std");
const cuda = @import("cuda");
const kv = @import("kv");
const mock = @import("load_mock.zig");
const t = std.testing;
var source: ?[*]const u8 = null;
fn current(_: cuda.abi.Context) callconv(.c) cuda.abi.Result {
    return 0;
}
fn upload(_: cuda.abi.DevicePtr, src: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    source = @ptrCast(src.?);
    mock.pending = true;
    mock.fail_sync = true;
    return cuda.abi.error_not_ready;
}
fn download(dst: ?*anyopaque, _: cuda.abi.DevicePtr, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    @memset(@as([*]u8, @ptrCast(dst.?))[0..n], 99);
    mock.pending = true;
    mock.fail_sync = true;
    return cuda.abi.error_not_ready;
}
test "Snapshot state: failed page I/O keeps private transfer storage and never borrows the caller's bytes" {
    var d = mock.init();
    d.api.cuCtxSetCurrent = current;
    d.api.cuMemcpyHtoDAsync_v2 = upload;
    d.api.cuMemcpyDtoHAsync_v2 = download;
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    const fams = [_]kv.sessions.Family{.{ .name = "comp", .ratio = 1, .row_bytes = 2 }};
    var pool = try kv.sessions.Pool.init(t.allocator, .{ .families = &fams, .page = 4 }, 16, 1, 0);
    defer pool.deinit();
    var tensors = [_]cuda.DeviceBuffer{.{ .d = &d, .ptr = 256, .len = 64 }};
    var p: kv.DevicePool = undefined;
    p.pool = &pool;
    p.d = &d;
    p.ctx = &ctx;
    p.tensors = &tensors;
    p.staging = null;
    p.io = .{ .d = &d, .handle = null };
    defer {
        mock.fail_sync = false;
        p.io.synchronize() catch unreachable;
        if (p.staging) |*h| h.free();
    }
    var bytes: [8]u8 = @splat(7);
    try t.expectError(error.NotReady, p.store().write(0, &.{0}, &bytes));
    @memset(&bytes, 1);
    try t.expect(std.mem.allEqual(u8, source.?[0..8], 7));
    try t.expectError(error.NotReady, p.store().write(0, &.{0}, &bytes));
    try t.expect(std.mem.allEqual(u8, source.?[0..8], 7));
    mock.fail_sync = false;
    try t.expectError(error.NotReady, p.store().read(0, &.{0}, &bytes));
    try t.expect(std.mem.allEqual(u8, &bytes, 1));
    try t.expectEqual(@as(usize, 0), mock.early_free);
}
test "Snapshot state: failed page table publication restores the dirty range for retry" {
    var d = mock.init();
    const fams = [_]kv.sessions.Family{.{ .name = "comp", .ratio = 1, .row_bytes = 2 }};
    var pool = try kv.sessions.Pool.init(t.allocator, .{ .families = &fams, .page = 4 }, 16, 1, 0);
    defer pool.deinit();
    const slot = try pool.newSlot(8);
    try slot.ensure(4);
    var ht = [_]i32{ 0, 0 };
    var hl = [_]i32{ 0, 0 };
    var p: kv.DevicePool = undefined;
    p.pool = &pool;
    p.d = &d;
    p.pts = 2;
    p.slots = 1;
    p.host_table = &ht;
    p.host_local = &hl;
    p.compute = .{ .d = &d, .handle = null };
    p.table = try cuda.DeviceBuffer.alloc(&d, 8);
    p.local = try cuda.DeviceBuffer.alloc(&d, 8);
    defer {
        mock.fail_sync = false;
        p.compute.synchronize() catch unreachable;
        p.table.free();
        p.local.free();
    }
    mock.fail_copy = true;
    try t.expectError(error.NotReady, p.syncTables());
    try t.expectEqual(@as(u32, 0), slot.dirty_lo);
    try t.expectEqual(@as(u32, 1), slot.dirty_hi);
    mock.fail_copy = false;
    try p.syncTables();
    try t.expect(slot.takeDirty() == null);
    try t.expectEqual(@as(i32, @intCast(slot.tableAt(0))), ht[0]);
}
