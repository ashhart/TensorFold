//! Kept-image layout, copy ownership, device buffers and failed restore cleanup without a device.
const std = @import("std");
const cuda = @import("cuda");
const reuse = @import("cuda_reuse.zig");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const Engine = engine.Engine;
const Cache = engine.Cache;
const Geometry = geometry.Geometry;
const head_dim = geometry.head_dim;
const Format = reuse.Format;
const Seen = reuse.Seen;
const Copy = reuse.Copy;
const Layout = reuse.Layout;
const Piece = reuse.Piece;
const sameImages = reuse.sameImages;
const restorable = reuse.restorable;
const saveCache = reuse.saveCache;
const restoreCache = reuse.restoreCache;
const dropCopy = reuse.dropCopy;

fn testGeometry(layers: usize, attention_every: usize, kv_heads: usize) Geometry {
    var g: Geometry = undefined;
    g.hidden = 5120;
    g.k_heads = 16;
    g.v_heads = 48;
    g.q_heads = 32;
    g.kv_heads = kv_heads;
    g.layers = layers;
    @memset(&g.linear, false);
    for (0..layers) |i| g.linear[i] = i % attention_every != attention_every - 1;
    return g;
}

test "a kept state holds the recurrences and tails whole and each attention head's rows it read" {
    const g = testGeometry(8, 4, 4); // the 27B's pattern: three linear layers, then attention
    const l: Layout = .{ .g = g, .format = .bf16, .cap = 2048, .at = 1000 };
    try std.testing.expectEqual(@as(usize, 1 + 2 * 2 * 4), l.count());
    try std.testing.expectEqual(@as(u64, g.stateBytes()), (Layout{ .g = g, .format = .bf16, .cap = 2048, .at = 0 }).total());
    try std.testing.expectEqual(@as(u64, g.stateBytes() + 2 * 2 * 4 * 1000 * head_dim * 2), l.total());
    // fp8: e4m3 rows and a scale each, 260 bytes a position and head
    const l8: Layout = .{ .g = g, .format = .fp8, .cap = 2048, .at = 1000 };
    try std.testing.expectEqual(@as(usize, 1 + 2 * 2 * 2 * 4), l8.count());
    try std.testing.expectEqual(@as(u64, g.stateBytes() + 2 * 2 * 4 * 1000 * (head_dim + 4)), l8.total());
}

test "every geometry's pieces lie inside its allocation, apart, with no fixed limit on layers or heads" {
    const a = std.testing.allocator;
    // 33 attention layers of 32 key heads passed the old fixed array; the widest the engine reads is 128 layers
    for ([_][3]usize{ .{ 8, 4, 4 }, .{ 99, 3, 32 }, .{ 33, 1, 32 }, .{ 128, 1, 32 }, .{ 24, 4, 2 } }) |shape| {
        const g = testGeometry(shape[0], shape[1], shape[2]);
        for ([_]Format{ .bf16, .fp8 }) |f| for ([_]usize{ 1, 7, 64 }) |at| {
            const l: Layout = .{ .g = g, .format = f, .cap = 64, .at = at };
            const end = l.kvBytes();
            if (comptime @typeInfo(@TypeOf(Geometry.kvBytes)).@"fn".param_types.len == 3) try std.testing.expectEqual(g.kvBytes(64, f == .fp8), end);
            const spans = try a.alloc([2]usize, l.count() - 1);
            defer a.free(spans);
            try std.testing.expectEqual(@as(usize, g.stateBytes()), l.piece(0).len);
            for (spans, 1..) |*sp, i| {
                const p = l.piece(i);
                try std.testing.expect(p.kv and p.len > 0 and p.off + p.len <= end);
                sp.* = .{ p.off, p.off + p.len };
            }
            std.mem.sort([2]usize, spans, {}, struct {
                fn lt(_: void, x: [2]usize, y: [2]usize) bool {
                    return x[0] < y[0];
                }
            }.lt);
            for (spans[1..], spans[0 .. spans.len - 1]) |sp, prev| try std.testing.expect(sp[0] >= prev[1]);
        };
    }
}

test "fp8 pieces follow qwen35_kv8.cu: a layer's key then value rows, then their scales" {
    const g = testGeometry(4, 4, 2); // one attention layer
    const l: Layout = .{ .g = g, .format = .fp8, .cap = 100, .at = 10 };
    // (key, head 1): rows at 1 * cap * 256, its scales after both blocks' rows, at 2 * 2 * cap * 256 + 1 * cap * 4
    try std.testing.expectEqual(Piece{ .kv = true, .off = 100 * 256, .len = 10 * 256 }, l.piece(1 + 2 * 1));
    try std.testing.expectEqual(Piece{ .kv = true, .off = 4 * 100 * 256 + 100 * 4, .len = 10 * 4 }, l.piece(2 + 2 * 1));
    // (value, head 0): the third block's rows and scales
    try std.testing.expectEqual(Piece{ .kv = true, .off = 2 * 100 * 256, .len = 10 * 256 }, l.piece(1 + 2 * 2));
    try std.testing.expectEqual(Piece{ .kv = true, .off = 4 * 100 * 256 + 2 * 100 * 4, .len = 10 * 4 }, l.piece(2 + 2 * 2));
}

test "a state resumes only a request with the same images before its position" {
    const d = [_]u64{ 1, 2, 3 };
    const kept = [_]Seen{.{ .at = 5, .id = d[0] }};
    // a later image is the pass's
    try std.testing.expect(sameImages(&kept, &.{ .{ .at = 5, .id = d[0] }, .{ .at = 40, .id = d[2] } }, 20));
    try std.testing.expect(!sameImages(&kept, &.{.{ .at = 5, .id = d[1] }}, 20)); // another image under the same ids
    try std.testing.expect(!sameImages(&kept, &.{.{ .at = 6, .id = d[0] }}, 20));
    try std.testing.expect(!sameImages(&kept, &.{}, 20)); // a text prompt never resumes an image's state
    // nor an image prompt a text state past its image
    try std.testing.expect(!sameImages(&.{}, &.{.{ .at = 5, .id = d[0] }}, 20));
    try std.testing.expect(sameImages(&.{}, &.{.{ .at = 5, .id = d[0] }}, 5)); // a text prefix before it
}

test "a restore refuses another cache format, another length, a position past the cache and other images" {
    var e: Engine = undefined;
    e.n_taps = 0;
    e.kit = null;
    e.g = testGeometry(8, 4, 4);
    if (comptime @hasField(Engine, "kv8")) e.kv8 = false;
    var cache: Cache = undefined;
    cache.cap = 64;
    cache.paged = null; // contiguous: restoreCache reads it, and undefined bytes can read as a page table
    const d: u64 = 9;
    var images = [_]Seen{.{ .at = 2, .id = d }};
    const l: Layout = .{ .g = e.g, .format = .bf16, .cap = 64, .at = 10 };
    const host = try std.testing.allocator.alignedAlloc(u8, .@"16", @intCast(l.total()));
    defer std.testing.allocator.free(host);
    var copy: Copy = .{ .host = .{ .d = undefined, .bytes = host }, .len = l.total(), .at = 10, .delta = 0, .format = .bf16, .images = &images };
    const same = [_]Seen{.{ .at = 2, .id = d }};
    try std.testing.expect(restorable(&e, &cache, &copy, &same));
    try std.testing.expect(!restorable(&e, &cache, &copy, &.{.{ .at = 2, .id = 8 }}));
    copy.format = .fp8; // a state kept from fp8 rows never lands in bf16 ones, nor the other way
    try std.testing.expect(!restorable(&e, &cache, &copy, &same));
    copy.format = .bf16;
    copy.len -= 1;
    try std.testing.expect(!restorable(&e, &cache, &copy, &same));
    copy.len += 1;
    cache.cap = 10;
    try std.testing.expect(!restorable(&e, &cache, &copy, &same));
}

/// A driver whose later copies and stuck waits fail with CUDA_ERROR_NOT_READY (check logs nothing), counting frees.
const RestoreFake = struct {
    var copies: usize = 0;
    var fail_at: usize = 1;
    var stuck = true;
    var freed: usize = 0;

    fn htod(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copies += 1;
        return if (copies - 1 >= fail_at) cuda.abi.error_not_ready else cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (stuck) cuda.abi.error_not_ready else cuda.abi.success;
    }
    fn hostAlloc(p: *?*anyopaque, len: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        p.* = (std.testing.allocator.alignedAlloc(u8, .@"16", len) catch return 2).ptr;
        return cuda.abi.success;
    }
    fn freeHost(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        freed += 1;
        return cuda.abi.success;
    }
};

test "a failed restore's host copy stays through a drop until a wait, the restore's or the drop's, drains the stream" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    var api: cuda.abi.Api = undefined;
    api.cuMemcpyHtoDAsync_v2 = RestoreFake.htod;
    api.cuStreamSynchronize = RestoreFake.sync;
    api.cuMemHostAlloc = RestoreFake.hostAlloc;
    api.cuMemFreeHost = RestoreFake.freeHost;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    var e: Engine = undefined;
    e.n_taps = 0;
    e.kit = null;
    e.gpa = std.testing.allocator;
    e.d = &d;
    e.g = testGeometry(8, 4, 4);
    e.stream = .{ .d = &d, .handle = null };
    if (comptime @hasField(Engine, "kv8")) e.kv8 = false;
    var cache: Cache = undefined;
    cache.cap = 64;
    cache.paged = null; // contiguous: restoreCache reads it, and undefined bytes can read as a page table
    cache.kv = .{ .d = &d, .ptr = 0x10000, .len = 0 };
    cache.state = .{ .d = &d, .ptr = 0x20000, .len = 0 };
    const l: Layout = .{ .g = e.g, .format = .bf16, .cap = 64, .at = 10 };
    // the stream stuck through the restore and the drop, stuck only through the restore, or never
    for ([_][2]bool{ .{ true, true }, .{ true, false }, .{ false, false } }) |stuck| {
        RestoreFake.copies = 0;
        RestoreFake.freed = 0;
        RestoreFake.stuck = stuck[0];
        const copy = try std.testing.allocator.create(Copy);
        copy.* = .{ .host = try cuda.HostBuffer.alloc(&d, @intCast(l.total())), .len = l.total(), .at = 10, .delta = 0, .format = .bf16, .images = try std.testing.allocator.alloc(Seen, 0) };
        const pinned = copy.host.bytes; // the fake's free keeps them: the test gives them back
        defer std.testing.allocator.free(pinned);
        try std.testing.expectError(error.NotReady, restoreCache(&e, &cache, copy, &.{}));
        try std.testing.expectEqual(@as(usize, 2), RestoreFake.copies); // the state queued, the first head's rows refused
        RestoreFake.stuck = stuck[1];
        const before = cuda.copy_source.retained();
        dropCopy(&e, copy);
        // a stream that will not drain may still read the copy: quarantine holds it, counted; once drained, it is freed
        try std.testing.expectEqual(@as(usize, @intFromBool(!stuck[1])), RestoreFake.freed);
        try std.testing.expectEqual(before + @intFromBool(stuck[1]), cuda.copy_source.retained());
        RestoreFake.stuck = false;
        try e.stream.synchronize();
        try std.testing.expectEqual(@as(usize, 1), RestoreFake.freed);
        try std.testing.expectEqual(before, cuda.copy_source.retained());
    }
}

/// A driver that counts device and pinned allocations, frees, and copies by direction; every call succeeds.
const DeviceFake = struct {
    var allocs: usize = 0;
    var pinned: usize = 0;
    var frees: usize = 0;
    var dtod: usize = 0;
    var host_copies: usize = 0;

    fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        allocs += 1;
        p.* = 0x40000;
        return cuda.abi.success;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        frees += 1;
        return cuda.abi.success;
    }
    fn hostAlloc(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        pinned += 1;
        return cuda.abi.error_not_ready;
    }
    fn copy(_: cuda.abi.DevicePtr, _: cuda.abi.DevicePtr, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        dtod += 1;
        return cuda.abi.success;
    }
    fn htod(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        host_copies += 1;
        return cuda.abi.success;
    }
    fn dtoh(_: ?*anyopaque, _: cuda.abi.DevicePtr, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        host_copies += 1;
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }
};

test "a device save pins nothing: it copies on the device both ways, and its drop frees the device buffer" {
    var api: cuda.abi.Api = undefined;
    api.cuMemAlloc_v2 = DeviceFake.alloc;
    api.cuMemFree_v2 = DeviceFake.free;
    api.cuMemHostAlloc = DeviceFake.hostAlloc;
    api.cuMemcpyDtoDAsync_v2 = DeviceFake.copy;
    api.cuMemcpyHtoDAsync_v2 = DeviceFake.htod;
    api.cuMemcpyDtoHAsync_v2 = DeviceFake.dtoh;
    api.cuStreamSynchronize = DeviceFake.sync;
    const d: cuda.Driver = .{ .lib = undefined, .api = api };
    var e: Engine = undefined;
    e.n_taps = 0;
    e.kit = null;
    e.gpa = std.testing.allocator;
    e.d = &d;
    e.g = testGeometry(8, 4, 4);
    e.stream = .{ .d = &d, .handle = null };
    if (comptime @hasField(Engine, "kv8")) e.kv8 = false;
    var cache: Cache = undefined;
    cache.cap = 64;
    cache.len = 10;
    cache.last = null;
    cache.pend = 0;
    cache.delta = 3;
    cache.paged = null;
    cache.kv = .{ .d = &d, .ptr = 0x10000, .len = 0 };
    cache.state = .{ .d = &d, .ptr = 0x20000, .len = 0 };
    const l: Layout = .{ .g = e.g, .format = .bf16, .cap = 64, .at = 10 };
    const copy = try saveCache(&e, &cache, 10, &.{}, true);
    try std.testing.expectEqual(@as(usize, 1), DeviceFake.allocs);
    try std.testing.expectEqual(@as(usize, l.count()), DeviceFake.dtod); // a copy a piece, contiguous
    try std.testing.expectEqual(@as(usize, l.total()), copy.room());
    try restoreCache(&e, &cache, copy, &.{});
    try std.testing.expectEqual(@as(usize, 2 * l.count()), DeviceFake.dtod);
    try std.testing.expectEqual(@as(i32, 3), cache.delta);
    dropCopy(&e, copy);
    try std.testing.expectEqual(@as(usize, 1), DeviceFake.frees);
    try std.testing.expectEqual(@as(usize, 0), DeviceFake.pinned + DeviceFake.host_copies);
}
