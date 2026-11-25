//! Inkling 2's attention plan without a GPU: tiles, chunks, rings, scratch and the launches a fake driver records.
const std = @import("std");
const cuda = @import("cuda");
const k = @import("kernels.zig");
const t = std.testing;

test "inkling2 kernels: tiles cover each segment's (row, head) pairs, a tile never spans two streams" {
    var out: [16]k.Tile = undefined;
    const segs = [_]k.Seg{ .{ .row0 = 0, .n = 11, .p0 = 0, .off = 0 }, .{ .row0 = 11, .n = 1, .p0 = 900, .off = 1 << 20 } };
    const n = try k.tiles(&segs, 6, &out);
    try t.expectEqual(@as(usize, 3), n);
    try t.expectEqual(k.Tile{ .seg = 0, .m0 = 0 }, out[0]);
    try t.expectEqual(k.Tile{ .seg = 0, .m0 = 64 }, out[1]);
    try t.expectEqual(k.Tile{ .seg = 1, .m0 = 0 }, out[2]);
    try t.expectError(error.TooManyTiles, k.tiles(&segs, 6, out[0..2]));
    try t.expectEqual(@as(usize, 192), k.tileCount(2048, 6));
}

test "inkling2 kernels: chunks and scratch at the context targets (1M x 1 and 8 x 128K)" {
    try t.expectEqual(@as(u32, 64), k.chunks(1048575, k.global_chunk));
    try t.expectEqual(@as(u32, 8), k.chunks(131071, k.global_chunk));
    try t.expectEqual(@as(u32, 4), k.chunks(1023, k.sliding_chunk));
    // one decode row at 1M on a rank of 24 heads: 64 chunks of partials, 3.2 MiB
    try t.expectEqual(@as(usize, 24 * 64 * 520), k.partialBytes(1, 24, 64));
    // a 2,048-row prompt chunk at 1M depth: 1.6 GB of partials, 100 MiB of table
    try t.expectEqual(@as(usize, 1635778560), k.partialBytes(2048, 24, 64));
    try t.expectEqual(@as(usize, 100663296), k.tableBytes(2048, 24));
    const s = k.Layer.sliding(1024, 2048);
    try t.expectEqual(@as(u32, 3072), s.ring);
    try t.expect(s.ring % k.tile_keys == 0 and s.ring >= 1024 + 2048 - 1);
    try t.expectEqual(@as(u32, 1048576), k.Layer.global(1048576).ring);
    try t.expectEqual(@as(u32, 131136), k.Layer.global(131073).ring);
}

test "inkling2 kernels: the loader type-checks in a build with kernels" {
    _ = &k.Kernels.load;
    _ = &k.Kernels.unload;
}

const Fake = struct {
    var grids: [8][3]u32 = undefined;
    var blocks: [8]u32 = undefined;
    var shared: [8]u32 = undefined;
    var count: usize = 0;
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, sh: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const i = @intFromPtr(f) - 1;
        grids[i] = .{ gx, gy, gz };
        blocks[i] = bx;
        shared[i] = sh;
        count += 1;
        return 0;
    }
};

test "inkling2 kernels: one decode row at 1M splits 64 chunks into sub-chunks for each of a rank's 4 key heads" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const fun = struct {
        fn of(drv: *const cuda.Driver, i: usize) cuda.Function {
            return .{ .d = drv, .handle = @ptrFromInt(i) };
        }
    };
    const kern: k.Kernels = .{ .mod = .{ .d = &d, .handle = null }, .prep = fun.of(&d, 1), .rel_table = fun.of(&d, 2), .attn = fun.of(&d, 3), .merge = fun.of(&d, 4), .attn_fold = fun.of(&d, 3), .pmod = .{ .d = &d, .handle = null }, .attn_prompt = fun.of(&d, 3) };
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const r: k.Rows = .{ .rows = 1, .heads = 24, .kv_heads = 4, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 1, .last_pos = 1048575 };
    const l = k.Layer.global(1048576);
    try k.prep(&kern, s, r, l, .{ .qk_clip = 22.6, .rel_clip = 8, .floor = 128000, .alpha = 0.1, .eps = 1e-6 }, .{ 0, 0, 0, 0 }, .{ 0, 0 }, 0, 0, .{ .k = 0, .v = 0 });
    try k.relTable(&kern, s, r, 0, 0, 0);
    const all = 24 * 64 * k.subs;
    try k.attend(&kern, s, r, l, 64, all, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0);
    try t.expectError(error.Inkling2Partials, k.attend(&kern, s, r, l, 63, all, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0));
    try t.expectEqual(@as(usize, 4), Fake.count);
    try t.expectEqual([3]u32{ 1, 32, 1 }, Fake.grids[0]);
    try t.expectEqual([3]u32{ 24, 1, 1 }, Fake.grids[1]);
    try t.expectEqual([3]u32{ 1, 4, 64 * k.subs }, Fake.grids[2]);
    try t.expectEqual([3]u32{ 24, 1, 1 }, Fake.grids[3]);
    try t.expectEqual([4]u32{ 32, 256, 128, 128 }, Fake.blocks[0..4].*);
    try t.expectEqual([4]u32{ 0, 0, 0, 0 }, Fake.shared[0..4].*);
    // too few partials for every sub-chunk: one block a chunk, its sub-chunks folded inside
    try k.attend(&kern, s, r, l, 64, all - 1, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0);
    try t.expectEqual([3]u32{ 1, 4, 64 }, Fake.grids[2]);
    try t.expectEqual(@as(u32, k.prompt_threads), Fake.blocks[2]);
    try t.expectEqual(@as(u32, 98304), Fake.shared[2]);
    // two streams' rows keep the fold kernel; a 2,048-row prompt call takes 96 blocks of 128 pairs a key head
    var two = r;
    two.streams = 2;
    try k.attend(&kern, s, two, l, 64, all - 1, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0);
    try t.expectEqual(@as(u32, 128), Fake.blocks[2]);
    try t.expectEqual(@as(u32, 0), Fake.shared[2]);
    var prompt = r;
    prompt.rows = 2048;
    try k.attend(&kern, s, prompt, l, 64, 2048 * 24 * 64, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0);
    try t.expectEqual([3]u32{ 96, 4, 64 }, Fake.grids[2]);
    try t.expect(!k.split(.{ .rows = 65, .heads = 24, .kv_heads = 4, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 7, .last_pos = 100 }, 1, 1 << 30));
}

test "inkling2 kernels: a stream's caches are 256 bytes a key head and token" {
    try t.expectEqual(@as(usize, 1048576 * 4 * 256), k.Layer.global(1048576).cacheBytes(4));
    try t.expectEqual(k.Layer.sliding(1024, 2048).cacheBytes(4), 3072 * 4 * 256);
}

test "inkling2 kernels: sliding partials are a row's window at any depth, a tile's launch a fixed chunk count" {
    const sl = k.Layer.sliding(1024, 1);
    try t.expectEqual(@as(u32, 5), k.slotsFor(sl, 1048576)); // keys pos-1023..pos touch at most 5 chunks of 256
    try t.expectEqual(@as(u32, 6), k.tileChunks(6, sl)); // a tile's 12 rows and their windows: 6 chunks
    try t.expectEqual(@as(u32, 64), k.slotsFor(k.Layer.global(1048576), 1048576));
    try t.expectEqual(@as(u32, 8), k.slotsFor(k.Layer.global(131072), 131072));
}

/// A call's sliding stores as tf_ink2_attn makes them: each slot the merge reads once; returns the old strays.
fn slidingStores(a: std.mem.Allocator, p0: usize, n: usize) !usize {
    const group = 6;
    const l = k.Layer.sliding(1024, 2048);
    const stride: usize = k.slotsFor(l, 1); // 5 below 81,920: a stray store lands on a neighbour (row, head)'s slots
    const count = try a.alloc(u8, n * group * stride);
    defer a.free(count);
    @memset(count, 0);
    var tl: [256]k.Tile = undefined;
    const segs = [_]k.Seg{.{ .row0 = 0, .n = @intCast(n), .p0 = @intCast(p0), .off = 0 }};
    var stray: usize = 0;
    for (tl[0..try k.tiles(&segs, group, &tl)]) |tile| {
        const m0: usize = @intCast(tile.m0);
        const m_last = @min(m0 + 63, n * group - 1);
        const first = p0 + m0 / group;
        const hi = p0 + m_last / group;
        for (0..k.tileChunks(group, l)) |z| {
            const c = k.tileChunk(l, first, z);
            if (c * l.chunk > hi or c * l.chunk + l.chunk - 1 < (first + 1) -| l.window) continue;
            for (m0..m_last + 1) |mr| {
                const pos = p0 + mr / group;
                const raw = @as(isize, @intCast(c)) - @as(isize, @intCast(((pos + 1) -| l.window) / l.chunk));
                if (raw < 0 or raw >= stride) stray += 1;
                const s = k.partialSlot(l, c, pos) orelse continue;
                try t.expect(s < stride);
                count[mr * stride + s] += 1;
            }
        }
    }
    for (0..n * group) |mr| {
        const pos = p0 + mr / group;
        const read = pos / l.chunk - ((pos + 1) -| l.window) / l.chunk + 1; // the merge folds slots 0..read-1
        for (0..stride) |s| try t.expectEqual(@as(u8, @intFromBool(s < read)), count[mr * stride + s]);
    }
    return stray;
}

test "inkling2 kernels: sliding stores stay in their own slots for windows that straddle a chunk edge" {
    const a = t.allocator;
    // the drafted check's long text: 8-row windows at 2,296..2,302 (a window's first keys cross chunk 5's edge)
    try t.expectEqual(@as(usize, 0), try slidingStores(a, 2295, 8));
    try t.expect(try slidingStores(a, 2300, 8) > 0);
    try t.expect(try slidingStores(a, 2296, 8) > 0);
    var p0: usize = 0;
    while (p0 < 4400) : (p0 += 1) for (1..9) |n| {
        _ = try slidingStores(a, p0, n);
    };
    for ([_]usize{ 0, 2048, 4096 }) |at| _ = try slidingStores(a, at, 2048);
}
