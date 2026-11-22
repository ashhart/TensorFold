//! Inkling 2's HIP attention plan without a GPU: tiles, chunks, partial slots and the launches a mock runtime records.
const std = @import("std");
const hip = @import("hip");
const k = @import("hip_attention.zig");
const t = std.testing;

test "inkling2 hip: tiles hold 4 pairs, cover each segment's pairs, and never span two streams" {
    var out: [16]k.Tile = undefined;
    const segs = [_]k.Seg{ .{ .row0 = 0, .n = 3, .p0 = 0, .off = 0 }, .{ .row0 = 3, .n = 1, .p0 = 900, .off = 1 << 20 } };
    const n = try k.tiles(&segs, 3, &out);
    try t.expectEqual(@as(usize, 4), n);
    try t.expectEqualSlices(k.Tile, &.{ .{ .seg = 0, .m0 = 0 }, .{ .seg = 0, .m0 = 4 }, .{ .seg = 0, .m0 = 8 }, .{ .seg = 1, .m0 = 0 } }, out[0..4]);
    try t.expectError(error.TooManyTiles, k.tiles(&segs, 3, out[0..3]));
    try t.expectEqual(@as(usize, 1536), k.tileCount(2048, 3));
    try t.expectEqual(@as(usize, 24), @sizeOf(k.Seg));
}

test "inkling2 hip: rings hold whole tiles, scratch and chunks at Mini's 1M context" {
    try t.expectEqual(@as(u32, 64), k.chunks(1048575, k.global_chunk));
    try t.expectEqual(@as(usize, 24 * 64 * 520), k.partialBytes(1, 24, 64));
    try t.expectEqual(@as(usize, 100663296), k.tableBytes(2048, 24));
    const s = k.Layer.sliding(1024, 2048);
    try t.expectEqual(@as(u32, 3072), s.ring);
    try t.expectEqual(@as(usize, 3072 * 8 * 256), s.cacheBytes(8));
    try t.expectEqual(@as(u32, 131136), k.Layer.global(131073).ring);
}

test "inkling2 hip: every chunk a row reads has a partial slot, and a tile's blocks reach its last row's chunk" {
    const l = k.Layer.sliding(1024, 2048);
    for ([_]usize{ 0, 1, 255, 256, 1023, 1024, 1025, 5000, 70001 }) |pos| {
        const lo = (pos + 1) -| l.window;
        for (lo / l.chunk..pos / l.chunk + 1) |c| try t.expect(k.partialSlot(l, c, pos).? < k.slotsFor(l, pos + 1));
        try t.expectEqual(@as(?usize, null), k.partialSlot(l, pos / l.chunk + 1, pos));
        // a tile from this row onward: its last pair's row sits (tile_pairs - 1) / group rows later
        for ([_]u32{ 1, 3, 6 }) |group| {
            const last = pos + (k.tile_pairs - 1) / group;
            try t.expect(last / l.chunk < k.tileChunk(l, pos, k.tileChunks(group, l)));
        }
    }
    const g = k.Layer.global(1 << 20);
    try t.expectEqual(@as(u32, 64), k.slotsFor(g, 1 << 20));
    try t.expectEqual(@as(?usize, 63), k.partialSlot(g, 63, (1 << 20) - 1));
}

const Mock = struct {
    var grids: [8][3]u32 = undefined;
    var blocks: [8]u32 = undefined;
    var which: [8]usize = undefined;
    var count: usize = 0;
    fn launch(f: hip.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: hip.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) hip.abi.Result {
        grids[count] = .{ gx, gy, gz };
        blocks[count] = bx;
        which[count] = @intFromPtr(f);
        count += 1;
        return 0;
    }
};

fn mocked(r: *const hip.Runtime) k.Kernels {
    const fun = struct {
        fn of(rt: *const hip.Runtime, i: usize) hip.Function {
            return .{ .r = rt, .handle = @ptrFromInt(i) };
        }
    };
    return .{ .mod = .{ .r = r, .handle = null }, .prep = fun.of(r, 1), .rel_table = fun.of(r, 2), .attn = fun.of(r, 3), .attn_fold = fun.of(r, 4), .attn_prompt = fun.of(r, 5), .merge = fun.of(r, 6) };
}

test "inkling2 hip: the dispatcher picks split, fold or prompt by rows, streams and partial room" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    const no: k.Cache = .{ .k = 0, .v = 0 };
    var row: k.Rows = .{ .rows = 1, .heads = 24, .kv_heads = 8, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 1, .last_pos = 1048575 };
    const g = k.Layer.global(1048576);
    Mock.count = 0;
    try k.prep(&kern, s, row, g, .{ .qk_clip = 22.63, .rel_clip = 8, .floor = 8192, .alpha = 0.1, .eps = 1e-6 }, .{ 0, 0, 0, 0 }, .{ 0, 0 }, 0, 0, no);
    try k.relTable(&kern, s, row, 0, 0, 0);
    const all = 24 * 64 * k.subs;
    try k.attend(&kern, s, row, g, 64, all, 0, no, 0, 0, 0, 0, 0, 0); // split: a block a sub-chunk
    try t.expectError(error.Inkling2Partials, k.attend(&kern, s, row, g, 63, all, 0, no, 0, 0, 0, 0, 0, 0));
    try t.expectEqual(@as(usize, 4), Mock.count);
    try t.expectEqual([3]u32{ 1, 40, 1 }, Mock.grids[0]);
    try t.expectEqual([3]u32{ 24, 1, 1 }, Mock.grids[1]);
    try t.expectEqual([3]u32{ 1, 8, 64 * k.subs }, Mock.grids[2]);
    try t.expectEqual([4]u32{ 32, 256, 128, 128 }, Mock.blocks[0..4].*);
    try t.expectEqual([4]usize{ 1, 2, 3, 6 }, Mock.which[0..4].*);
    Mock.count = 0;
    try k.attend(&kern, s, row, g, 64, all - 1, 0, no, 0, 0, 0, 0, 0, 0); // one stream, no room: the prompt kernel
    row.rows = 130;
    row.streams = 2;
    row.n_tiles = 98;
    try k.attend(&kern, s, row, g, 64, all, 0, no, 0, 0, 0, 0, 0, 0); // two streams, too many rows: the fold
    try t.expectEqual([4]usize{ 5, 6, 4, 6 }, Mock.which[0..4].*);
    try t.expectEqual([3]u32{ 1, 8, 64 }, Mock.grids[0]);
    try t.expectEqual(@as(u32, 256), Mock.blocks[0]);
    try t.expectEqual([3]u32{ 98, 8, 64 }, Mock.grids[2]);
    Mock.count = 0;
    const sl = k.Layer.sliding(1024, 2048);
    try k.attend(&kern, s, row, sl, k.slotsFor(sl, 1), 0, 0, no, 0, 0, 0, 0, 0, 0);
    try t.expectEqual([3]u32{ 98, 8, 6 }, Mock.grids[0]);
    try t.expectEqual(@as(usize, 3), Mock.which[0]);
}
