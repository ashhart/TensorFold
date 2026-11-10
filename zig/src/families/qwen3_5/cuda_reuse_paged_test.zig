//! A paged fp8 cache's kept-state copies: each key, value and scale piece a run a page, from qwen35_kv8.cu's layout.
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");
const reuse = @import("cuda_reuse.zig");

const head_dim = geometry.head_dim;

/// A driver that keeps every device copy's (dst, src, bytes) and counts 2D copies; every call succeeds.
const Fake = struct {
    var copies: [64][3]u64 = undefined;
    var n: usize = 0;
    var two_d: usize = 0;
    fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        p.* = 0x7000_0000;
        return cuda.abi.success;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
    const Ptr = cuda.abi.DevicePtr;
    fn copy(dst: Ptr, src: Ptr, len: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (n == copies.len) return cuda.abi.error_not_ready;
        copies[n] = .{ dst, src, len };
        n += 1;
        return cuda.abi.success;
    }
    fn copy2d(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        two_d += 1;
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
};

test "a paged fp8 cache keeps each key, value and scale piece a run a page, the last page's partly, in a row" {
    var api: cuda.abi.Api = undefined;
    api.cuMemAlloc_v2 = Fake.alloc;
    api.cuMemFree_v2 = Fake.free;
    api.cuMemcpyDtoDAsync_v2 = Fake.copy;
    api.cuMemcpy2DAsync_v2 = Fake.copy2d;
    api.cuStreamSynchronize = Fake.sync;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    var e: engine.Engine = undefined;
    e.kit = null;
    e.gpa = std.testing.allocator;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.kv8 = true;
    var g: geometry.Geometry = undefined;
    g.hidden = 5120;
    g.k_heads = 16;
    g.v_heads = 48;
    g.q_heads = 32;
    g.kv_heads = 2;
    g.layers = 8;
    @memset(&g.linear, false);
    for (0..8) |i| g.linear[i] = i % 4 != 3; // two attention layers
    e.g = g;
    var ids = [_]u32{ 5, 0, 3, 1 }; // the stream's pages in row order, shuffled in the arena
    const tokens = 16;
    const at = 37; // two whole pages and 5 rows of a third
    var cache: engine.Cache = undefined;
    cache.cap = 64;
    cache.len = at;
    cache.last = null;
    cache.pend = 0;
    cache.delta = 0;
    cache.kv = .{ .d = &d, .ptr = 0x10000, .len = 0 };
    cache.state = .{ .d = &d, .ptr = 0x20000, .len = 0 };
    const base = 0x4000_0000;
    const table: cuda.DeviceBuffer = .{ .d = &d, .ptr = 0, .len = 0 };
    cache.paged = .{ .base = base, .table = table, .ids = &ids, .tokens = tokens, .own_ids = false };
    Fake.n = 0;
    Fake.two_d = 0;
    const copy = try reuse.saveCache(&e, &cache, at, &.{}, true);
    defer reuse.dropCopy(&e, copy);
    const kept = copy.dev.?.ptr;
    // the state first, whole; then per (layer, key/value block) its rows, then its scales, a run a page
    const kvh = g.kv_heads;
    const page_bytes = g.attentions() * 2 * kvh * tokens * (head_dim + 4);
    var want: [64][3]u64 = undefined;
    want[0] = .{ kept, cache.state.ptr, g.stateBytes() };
    var n: usize = 1;
    var off: u64 = g.stateBytes();
    for (0..g.attentions()) |layer| for (0..2 * kvh) |block| for ([_]bool{ false, true }) |scale| {
        const layer_off = layer * 2 * kvh * tokens * (head_dim + 4);
        const scales = layer_off + 2 * kvh * tokens * head_dim; // a layer's rows, then their scales
        const in_page = if (scale) scales + block * tokens * 4 else layer_off + block * tokens * head_dim;
        const row: usize = if (scale) 4 else head_dim;
        var r: usize = 0;
        while (r < at) : (r += tokens) {
            const rows = @min(tokens, at - r);
            want[n] = .{ kept + off, base + ids[r / tokens] * page_bytes + in_page, rows * row };
            n += 1;
            off += rows * row;
        }
    };
    try std.testing.expectEqual(copy.len, off);
    try std.testing.expectEqualSlices([3]u64, want[0..n], Fake.copies[0..Fake.n]);
    // a restore makes the same copies the other way
    Fake.n = 0;
    try reuse.restoreCache(&e, &cache, copy, &.{});
    try std.testing.expectEqual(n, Fake.n);
    for (want[0..n], Fake.copies[0..n]) |w, c| try std.testing.expectEqual([3]u64{ w[1], w[0], w[2] }, c);
    try std.testing.expectEqual(@as(usize, 0), Fake.two_d); // fp8 pages never take bf16's 2D copies
}
