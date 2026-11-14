//! A paged bf16 cache's prompt-state save and restore with a mock driver: the kept side is the copy's own memory.
const std = @import("std");
const cuda = @import("cuda");
const reuse = @import("cuda_reuse.zig");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");
const t = std.testing;

const Mock = struct {
    const dev_at: u64 = 0x4000000;
    var dev_len: usize = 0;
    var host: []u8 = &.{};
    var pinned: usize = 0;
    var flat_host: usize = 0;
    var flat_dev: usize = 0;
    var bad: usize = 0;
    var pages: usize = 0;

    fn alloc(p: *cuda.abi.DevicePtr, len: usize) callconv(.c) cuda.abi.Result {
        dev_len = len;
        p.* = dev_at;
        return cuda.abi.success;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    fn hostAlloc(p: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        pinned += 1;
        host = std.heap.page_allocator.alloc(u8, len) catch return cuda.abi.error_not_ready;
        p.* = host.ptr;
        return cuda.abi.success;
    }
    fn freeHost(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        std.heap.page_allocator.free(host);
        host = &.{};
        return cuda.abi.success;
    }
    fn dtod(_: cuda.abi.DevicePtr, _: cuda.abi.DevicePtr, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        flat_dev += 1;
        return cuda.abi.success;
    }
    fn htod(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        flat_host += 1;
        return cuda.abi.success;
    }
    fn dtoh(_: ?*anyopaque, _: cuda.abi.DevicePtr, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        flat_host += 1;
        return cuda.abi.success;
    }
    /// A page copy's kept side must lie in the copy's own buffer, as its memory type says.
    fn copy2d(m: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pages += 1;
        const save = m.src_type == cuda.abi.memory_type_device and m.src_host == null and m.src < dev_at;
        const kind = if (save) m.dst_type else m.src_type;
        const hp: usize = if (save) (if (m.dst_host) |h| @intFromPtr(h) else 0) else (if (m.src_host) |h| @intFromPtr(h) else 0);
        const at: usize = if (kind == cuda.abi.memory_type_device) @intCast((if (save) m.dst else m.src) -% dev_at) else hp -% @intFromPtr(host.ptr);
        const room = if (kind == cuda.abi.memory_type_device) dev_len else host.len;
        if (at + (m.height - 1) * (if (save) m.dst_pitch else m.src_pitch) + m.width > room) bad += 1;
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
};

test "Qwen paged reuse: a bf16 save and restore copy a page each into the copy's own memory: device memory too, pinning nothing" {
    var api: cuda.abi.Api = undefined;
    api.cuMemAlloc_v2 = Mock.alloc;
    api.cuMemFree_v2 = Mock.free;
    api.cuMemHostAlloc = Mock.hostAlloc;
    api.cuMemFreeHost = Mock.freeHost;
    api.cuMemcpyDtoDAsync_v2 = Mock.dtod;
    api.cuMemcpyHtoDAsync_v2 = Mock.htod;
    api.cuMemcpyDtoHAsync_v2 = Mock.dtoh;
    api.cuMemcpy2DAsync_v2 = Mock.copy2d;
    api.cuStreamSynchronize = Mock.sync;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    var e: engine.Engine = undefined;
    e.kit = null;
    e.gpa = t.allocator;
    e.d = &d;
    e.g = undefined;
    e.g.hidden = 5120;
    e.g.k_heads = 16;
    e.g.v_heads = 48;
    e.g.q_heads = 32;
    e.g.kv_heads = 4;
    e.g.layers = 8;
    e.g.first = 0; // one engine, not a pipeline stage
    e.g.whole = 0;
    @memset(&e.g.linear, false);
    for (0..8) |i| e.g.linear[i] = i % 4 != 3;
    e.stream = .{ .d = &d, .handle = null };
    if (comptime @hasField(engine.Engine, "kv8")) e.kv8 = false;
    var ids = [_]u32{ 3, 0, 2, 1 };
    var cache: engine.Cache = undefined;
    cache.cap = 64;
    cache.len = 40; // two full pages of 16 and one of 8
    cache.last = null;
    cache.pend = 0;
    cache.delta = 3;
    cache.paged = .{ .base = 0x100000, .table = .{ .d = &d, .ptr = 0x90000, .len = 0 }, .ids = &ids, .tokens = 16, .own_ids = false };
    cache.kv = .{ .d = &d, .ptr = 0x10000, .len = 0 };
    cache.state = .{ .d = &d, .ptr = 0x20000, .len = 0 };
    for ([_]bool{ true, false }) |device| {
        Mock.pinned = 0;
        Mock.flat_host = 0;
        Mock.flat_dev = 0;
        Mock.bad = 0;
        Mock.pages = 0;
        const copy = try reuse.saveCache(&e, &cache, 40, &.{}, device);
        try t.expectEqual(@as(usize, 3), Mock.pages);
        try reuse.restoreCache(&e, &cache, copy, &.{});
        try t.expectEqual(@as(usize, 6), Mock.pages);
        try t.expectEqual(@as(usize, 0), Mock.bad);
        // the state piece goes flat: on the device for a device copy, else through the pinned host copy
        try t.expectEqual(@as(usize, if (device) 2 else 0), Mock.flat_dev);
        try t.expectEqual(@as(usize, if (device) 0 else 2), Mock.flat_host);
        try t.expectEqual(@as(usize, @intFromBool(!device)), Mock.pinned);
        reuse.dropCopy(&e, copy);
    }
}
