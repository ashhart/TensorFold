//! The engine's uploads own their source: a fake driver reads it at the stream's sync, after the caller changed it.
const std = @import("std");
const cuda = @import("cuda");
const put = @import("model.zig").put;
const t = std.testing;

const Fake = struct {
    var pinned: ?[]align(16) u8 = null;
    var pending: ?[]const u8 = null;
    var result: [16]u8 = undefined;
    fn host(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        pinned = t.allocator.alignedAlloc(u8, .@"16", len) catch return cuda.abi.error_not_ready;
        out.* = pinned.?.ptr;
        return 0;
    }
    fn free(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(pending == null);
        t.allocator.free(pinned.?);
        pinned = null;
        return 0;
    }
    fn copy(_: u64, ptr: ?*const anyopaque, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending = @as([*]const u8, @ptrCast(ptr.?))[0..len];
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (pending) |bytes| @memcpy(result[0..bytes.len], bytes);
        pending = null;
        return 0;
    }
    fn event(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn query(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return if (pending != null) cuda.abi.error_not_ready else 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

test "an upload's source may change as soon as put returns: the device gets the bytes it was given" {
    var d: cuda.Driver = undefined;
    d.api.cuMemHostAlloc = Fake.host;
    d.api.cuMemFreeHost = Fake.free;
    d.api.cuMemcpyHtoDAsync_v2 = Fake.copy;
    d.api.cuStreamSynchronize = Fake.sync;
    d.api.cuEventCreate = Fake.event;
    d.api.cuEventRecord = Fake.record;
    d.api.cuEventQuery = Fake.query;
    d.api.cuEventDestroy_v2 = Fake.destroy;
    const s: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(0x1800) };
    Fake.pending = null;
    Fake.result = @splat(0);
    const host_before = cuda.memory.usage(false).host;
    defer {
        s.synchronize() catch unreachable;
        cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    }
    var meta = [4]i32{ 41, 0, 0, 0 };
    try put(&d, s, 0x1000, std.mem.asBytes(&meta));
    meta[0] = -7; // a draw's next meta, written into the same stack slot before the stream ran
    try s.synchronize();
    cuda.copy_source.dropSpares(&d, cuda.copy_source.current());
    try t.expectEqual(@as(i32, 41), std.mem.bytesToValue(i32, Fake.result[0..4]));
    try t.expect(Fake.pinned == null);
    try t.expectEqual(host_before, cuda.memory.usage(false).host);
}
