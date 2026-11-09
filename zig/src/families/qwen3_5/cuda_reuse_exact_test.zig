//! Actual Qwen save/restore calls against byte buffers and independent head/row witnesses, without a device.
const std = @import("std");
const cuda = @import("cuda");
const reuse = @import("cuda_reuse.zig");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const a = std.testing.allocator;
const M = struct {
    const base = [_]u64{ 1 << 32, 2 << 32, 3 << 32 };
    var buffers: [3][]u8 = @splat(&.{});
    var masks: [2][]bool = @splat(&.{});
    var host: []u8 = &.{};
    var restoring = false;

    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        buffers[2] = a.alloc(u8, n) catch return cuda.abi.error_not_ready;
        out.* = base[2];
        return cuda.abi.success;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        a.free(buffers[2]);
        buffers[2] = &.{};
        return cuda.abi.success;
    }
    fn pin(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        host = a.alloc(u8, n) catch return cuda.abi.error_not_ready;
        out.* = host.ptr;
        return cuda.abi.success;
    }
    fn unpin(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        a.free(host);
        host = &.{};
        return cuda.abi.success;
    }
    fn bytes(at: u64, n: usize, read: bool) []u8 {
        for (base, buffers, 0..) |b, data, i| if (at >= b and at - b <= data.len and n <= data.len - (at - b)) {
            const off: usize = @intCast(at - b);
            if (read and !restoring and i < 2) @memset(masks[i][off..][0..n], true);
            return data[off..][0..n];
        };
        const off: usize = @intCast(at - @intFromPtr(host.ptr));
        return host[off..][0..n];
    }
    fn move(dst: u64, src: u64, n: usize) void {
        const from = bytes(src, n, true);
        @memcpy(bytes(dst, n, false), from);
    }
    fn dd(dst: u64, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        move(dst, src, n);
        return cuda.abi.success;
    }
    fn dh(dst: ?*anyopaque, src: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        move(@intFromPtr(dst.?), src, n);
        return cuda.abi.success;
    }
    fn hd(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        move(dst, @intFromPtr(src.?), n);
        return cuda.abi.success;
    }
    fn two(c: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        const src = if (c.src_type == cuda.abi.memory_type_host) @intFromPtr(c.src_host.?) else c.src;
        const dst = if (c.dst_type == cuda.abi.memory_type_host) @intFromPtr(c.dst_host.?) else c.dst;
        for (0..c.height) |i| move(dst + i * c.dst_pitch, src + i * c.src_pitch, c.width);
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
};

fn roundtrip(at: usize, paged: bool, device: bool) !void {
    var api: cuda.abi.Api = undefined;
    api.cuMemAlloc_v2 = M.alloc;
    api.cuMemFree_v2 = M.free;
    api.cuMemHostAlloc = M.pin;
    api.cuMemFreeHost = M.unpin;
    api.cuMemcpyDtoDAsync_v2 = M.dd;
    api.cuMemcpyDtoHAsync_v2 = M.dh;
    api.cuMemcpyHtoDAsync_v2 = M.hd;
    api.cuMemcpy2DAsync_v2 = M.two;
    api.cuStreamSynchronize = M.sync;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    var e: engine.Engine = undefined;
    e.kit = null;
    e.gpa = a;
    e.d = &d;
    e.g = undefined;
    e.g.hidden = 5120;
    e.g.k_heads = 16;
    e.g.v_heads = 48;
    e.g.q_heads = 32;
    e.g.kv_heads = 4;
    e.g.layers = 8;
    e.g.first = 0; // a whole model, not a pipeline stage
    e.g.whole = 0;
    @memset(&e.g.linear, false);
    for (0..8) |i| e.g.linear[i] = i % 4 != 3;
    e.stream = .{ .d = &d, .handle = null };
    if (comptime @hasField(engine.Engine, "kv8")) e.kv8 = false;
    const heads = e.g.attentions() * 2 * e.g.kv_heads;
    const row = geometry.head_dim * 2;
    M.buffers[0] = try a.alloc(u8, heads * 64 * row);
    defer a.free(M.buffers[0]);
    M.buffers[1] = try a.alloc(u8, e.g.stateBytes());
    defer a.free(M.buffers[1]);
    M.masks[0] = try a.alloc(bool, M.buffers[0].len);
    defer a.free(M.masks[0]);
    M.masks[1] = try a.alloc(bool, M.buffers[1].len);
    defer a.free(M.masks[1]);
    var original: [2][]u8 = undefined;
    for (0..2) |i| {
        @memset(M.masks[i], false);
        for (M.buffers[i], 0..) |*v, j| v.* = @intCast((j * 13 + i * 31) % 251);
        original[i] = try a.dupe(u8, M.buffers[i]);
    }
    defer for (original) |b| a.free(b);
    var ids = [_]u32{ 3, 0, 2, 1 };
    var cache: engine.Cache = undefined;
    cache.cap = 64;
    cache.len = at;
    cache.last = null;
    cache.pend = 0;
    cache.delta = 3;
    cache.paged = if (paged) .{ .base = M.base[0], .table = .{ .d = &d, .ptr = 0x90000, .len = 0 }, .ids = &ids, .tokens = 16, .own_ids = false } else null;
    cache.kv = .{ .d = &d, .ptr = M.base[0], .len = M.buffers[0].len };
    cache.state = .{ .d = &d, .ptr = M.base[1], .len = M.buffers[1].len };
    M.restoring = false;
    const copy = try reuse.saveCache(&e, &cache, at, &.{}, device);
    defer reuse.dropCopy(&e, copy);
    for (0..heads) |h| for (0..64) |r| {
        const off = if (paged) ids[r / 16] * heads * 16 * row + h * 16 * row + r % 16 * row else (h * 64 + r) * row;
        for (M.masks[0][off..][0..row]) |read| try std.testing.expectEqual(r < at, read);
    };
    for (M.masks[1]) |read| try std.testing.expect(read);
    @memset(M.buffers[0], 0xa5);
    @memset(M.buffers[1], 0xa5);
    M.restoring = true;
    try reuse.restoreCache(&e, &cache, copy, &.{});
    try std.testing.expectEqual(at, cache.len);
    try std.testing.expectEqual(@as(i32, 3), cache.delta);
    for (0..2) |i| for (M.buffers[i], M.masks[i], original[i]) |got, selected, before|
        try std.testing.expectEqual(if (selected) before else @as(u8, 0xa5), got);
}

test "Qwen endpoint bytes: mid-chunk and partial-page states restore every GDN and KV byte exactly" {
    for ([_]usize{ 31, 40 }) |at| for ([_]bool{ false, true }) |paged|
        for ([_]bool{ false, true }) |device| try roundtrip(at, paged, device);
}
