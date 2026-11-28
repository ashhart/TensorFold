//! Inkling 2's HIP attention plan without a GPU: tiles, chunks, partial slots and the launches a mock runtime records.
const std = @import("std");
const hip = @import("hip");
const config = @import("config.zig");
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
    return .{ .mod = .{ .r = r, .handle = null }, .prep = fun.of(r, 1), .rel_tables = .{ fun.of(r, 2), fun.of(r, 7), fun.of(r, 8) }, .attn = fun.of(r, 3), .attn_fold = fun.of(r, 4), .attn_prompt = fun.of(r, 5), .merge = fun.of(r, 6) };
}

test "inkling2 hip: the dispatcher picks split, fold or prompt by rows, streams and partial room" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    const no: k.Cache = .{ .k = 0, .v = 0 };
    var row: k.Rows = .{ .rows = 1, .heads = 24, .kv_heads = 8, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 1, .last_pos = 1048575, .streams = 1 };
    const g = k.Layer.global(1048576);
    Mock.count = 0;
    try k.prep(&kern, s, row, g, .{ .qk_clip = 22.63, .rel_clip = 8, .floor = 8192, .alpha = 0.1, .eps = 1e-6 }, .{ 0, 0, 0, 0 }, .{ 0, 0 }, 0, 0, no);
    try k.relTable(&kern, s, row, 1024, 0, 0, 0); // Mini's extent: the table kernel it always launched
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
    try k.attend(&kern, s, row, g, 64, 130 * 24 * 64, 0, no, 0, 0, 0, 0, 0, 0); // two streams, too many rows: the fold
    try t.expectEqual([4]usize{ 5, 6, 4, 6 }, Mock.which[0..4].*);
    try t.expectEqual([3]u32{ 1, 8, 64 }, Mock.grids[0]);
    try t.expectEqual(@as(u32, 256), Mock.blocks[0]);
    try t.expectEqual([3]u32{ 98, 8, 64 }, Mock.grids[2]);
    Mock.count = 0;
    const sl = k.Layer.sliding(1024, 2048);
    try k.attend(&kern, s, row, sl, k.slotsFor(sl, 1), 130 * 24 * k.slotsFor(sl, 1), 0, no, 0, 0, 0, 0, 0, 0);
    try t.expectEqual([3]u32{ 98, 8, 6 }, Mock.grids[0]);
    try t.expectEqual(@as(usize, 3), Mock.which[0]);
}

const ring_rows: k.Rows = .{ .rows = 4, .heads = 24, .kv_heads = 8, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 1, .last_pos = 5000, .streams = 1 };

test "inkling2 hip: attend refuses a ring that is not whole 64-key tiles, and launches nothing" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    var g = k.Layer.global(8192);
    g.ring = 8191;
    Mock.count = 0;
    try t.expectError(error.Inkling2RingTiles, k.attend(&kern, s, ring_rows, g, 1, 0, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0));
    try t.expectEqual(@as(usize, 0), Mock.count);
    g.ring = 8192;
    try k.attend(&kern, s, ring_rows, g, 1, 4 * 24, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0);
    try t.expectEqual(@as(usize, 2), Mock.count);
}

test "inkling2 hip: attend refuses a sliding ring shorter than the window plus the call's rows less one" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    var l = k.Layer.sliding(1024, 4);
    try t.expectEqual(@as(u32, 1088), l.ring);
    l.ring = 1024; // whole tiles, but four rows' new keys overwrite keys the first row still reads
    Mock.count = 0;
    try t.expectError(error.Inkling2RingShort, k.attend(&kern, s, ring_rows, l, k.slotsFor(l, 1), 0, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0));
    try t.expectEqual(@as(usize, 0), Mock.count);
    l.ring = 1088;
    try k.attend(&kern, s, ring_rows, l, k.slotsFor(l, 1), 4 * 24 * k.slotsFor(l, 1), 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0);
    try t.expectEqual(@as(usize, 2), Mock.count);
}

test "inkling2 hip: a multi-stream call never takes the one-stream prompt kernel; zero streams is refused" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    const g = k.Layer.global(1048576);
    var row: k.Rows = .{ .rows = 70, .heads = 24, .kv_heads = 8, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 53, .last_pos = 1048575, .streams = 0 };
    const room = 70 * 24 * 64; // a partial a chunk: no room to split
    Mock.count = 0;
    try t.expectError(error.Inkling2Streams, k.attend(&kern, s, row, g, 64, room, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0));
    try t.expectEqual(@as(usize, 0), Mock.count);
    row.streams = 3;
    try k.attend(&kern, s, row, g, 64, room, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0);
    try t.expectEqual([2]usize{ 4, 6 }, Mock.which[0..2].*); // the fold over the tiles, then the merge
}

test "inkling2 hip: attend refuses partial buffers smaller than rows x heads x stride, before any launch" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    const sl = k.Layer.sliding(1024, 2048);
    const stride = k.slotsFor(sl, 1);
    const row: k.Rows = .{ .rows = 16, .heads = 24, .kv_heads = 8, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 12, .last_pos = 4095, .streams = 2 };
    Mock.count = 0;
    try t.expectError(error.Inkling2PartialRoom, k.attend(&kern, s, row, sl, stride, 16 * 24 * stride - 1, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0));
    const g = k.Layer.global(1048576);
    var one = row;
    one.streams = 1;
    try t.expectError(error.Inkling2PartialRoom, k.attend(&kern, s, one, g, 64, 16 * 24 * 64 - 1, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0));
    try t.expectEqual(@as(usize, 0), Mock.count);
    try k.attend(&kern, s, row, sl, stride, 16 * 24 * stride, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0);
    try k.attend(&kern, s, one, g, 64, 16 * 24 * 64, 0, .{ .k = 0, .v = 0 }, 0, 0, 0, 0, 0, 0);
    try t.expectEqual([4]usize{ 3, 6, 5, 6 }, Mock.which[0..4].*);
}

test "inkling2 hip: the bias table launches its extent's kernel, Mini's 1,024 or tiny-mini's 256 and 128, all at one row length" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    const row: k.Rows = .{ .rows = 3, .heads = 4, .kv_heads = 2, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 0, .last_pos = 2, .streams = 1 };
    Mock.count = 0;
    for ([_]u32{ 1024, 256, 128 }) |e| try k.relTable(&kern, s, row, e, 0, 0, 0);
    try t.expectEqual([3]usize{ 2, 7, 8 }, Mock.which[0..3].*);
    for (0..3) |i| {
        try t.expectEqual([3]u32{ 12, 1, 1 }, Mock.grids[i]);
        try t.expectEqual(@as(u32, 256), Mock.blocks[i]);
    }
    try t.expectEqual(@as(usize, 3 * 4 * 1024 * 2), k.tableBytes(3, 4));
}

test "inkling2 hip: the bias table refuses an extent it is not built at, and launches nothing" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    const row: k.Rows = .{ .rows = 1, .heads = 4, .kv_heads = 2, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 0, .last_pos = 0, .streams = 1 };
    Mock.count = 0;
    for ([_]u32{ 0, 64, 512, 2048, 1025 }) |e| try t.expectError(error.UnsupportedRelExtent, k.relTable(&kern, s, row, e, 0, 0, 0));
    try t.expectEqual(@as(usize, 0), Mock.count);
}

/// Inkling 2 Mini's parsed text config.
fn miniText() !config.Text {
    var why: config.Why = .{};
    return (try config.parse(t.allocator, @embedFile("fixtures/mini_config.json"), &why)).text;
}

fn tinyMiniText() !config.Text {
    var why: config.Why = .{};
    return (try config.parse(t.allocator, @embedFile("fixtures/tiny_mini_config.json"), &why)).text;
}

test "inkling2 hip: each layer's extent from the config, Mini's 1,024 and tiny-mini's 128 local and 256 global" {
    var text = try miniText();
    var why: config.Why = .{};
    for ([_]u32{ 0, 4, 5, 11, 23 }) |i| try t.expectEqual(@as(u32, 1024), try k.layerExtent(&text, i, &why));
    const tiny = try tinyMiniText();
    for ([_]u32{ 0, 4, 6, 16 }) |i| try t.expectEqual(@as(u32, 128), try k.layerExtent(&tiny, i, &why));
    for ([_]u32{ 5, 11, 17 }) |i| try t.expectEqual(@as(u32, 256), try k.layerExtent(&tiny, i, &why));
}

test "inkling2 hip: a layer extent with no table kernel is refused by name, naming it and the 1024, 256 and 128 built" {
    var text = try miniText();
    for ([_]u32{ 512, 2048, 64 }, [_]u32{ 5, 0, 11 }) |e, i| {
        if (i == 0) text.window = e else text.rel_extent = e;
        var why: config.Why = .{};
        try t.expectError(error.UnsupportedRelExtent, k.layerExtent(&text, i, &why));
        var want: [48]u8 = undefined;
        try t.expect(std.mem.indexOf(u8, why.text(), try std.fmt.bufPrint(&want, "layer {d}'s relative-bias extent {d} ", .{ i, e })) != null);
        try t.expect(std.mem.indexOf(u8, why.text(), "1024, 256 and 128") != null);
    }
}

test "inkling2 hip: relTable refuses every extent but 1024, 256 and 128, launching nothing, also between served ones" {
    var r: hip.Runtime = undefined;
    r.api.hipModuleLaunchKernel = Mock.launch;
    const kern = mocked(&r);
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(32) };
    const row: k.Rows = .{ .rows = 2, .heads = 4, .kv_heads = 2, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 0, .last_pos = 1, .streams = 1 };
    Mock.count = 0;
    for ([_]u32{ 0, 1, 127, 129, 255, 257, 1023, 1025, std.math.maxInt(u32) }) |e| {
        try t.expectError(error.UnsupportedRelExtent, k.relTable(&kern, s, row, e, 0, 0, 0));
        try t.expectEqual(@as(?usize, null), k.extentIndex(e));
    }
    try t.expectEqual(@as(usize, 0), Mock.count);
    for ([_]u32{ 1024, 512, 256, 0, 128, 129 }, [_]bool{ true, false, true, false, true, false }) |e, served| {
        if (served) try k.relTable(&kern, s, row, e, 0, 0, 0) else try t.expectError(error.UnsupportedRelExtent, k.relTable(&kern, s, row, e, 0, 0, 0));
    }
    try t.expectEqual(@as(usize, 3), Mock.count);
    try t.expectEqual([3]usize{ 2, 7, 8 }, Mock.which[0..3].*);
}

test "inkling2 hip: every layer of tiny-mini's text config takes 128 or 256; an extent with no table names layer, extent and the three built" {
    var text = try tinyMiniText();
    try t.expectEqual(@as(u32, 18), text.layers);
    for (0..18) |li| {
        const i: u32 = @intCast(li);
        var why: config.Why = .{};
        try t.expectEqual(@as(u32, if ((i + 1) % 6 == 0) 256 else 128), try k.layerExtent(&text, i, &why));
        try t.expectEqual(@as(usize, 0), why.len);
    }
    text.rel_extent = 512;
    var why: config.Why = .{};
    try t.expectError(error.UnsupportedRelExtent, k.layerExtent(&text, 17, &why));
    try t.expectEqualStrings("layer 17's relative-bias extent 512 has no HIP bias table: tf_ink2_rel_table is built at extents 1024, 256 and 128", why.text());
    text.window = 129;
    why = .{};
    try t.expectError(error.UnsupportedRelExtent, k.layerExtent(&text, 0, &why));
    try t.expectEqualStrings("layer 0's relative-bias extent 129 has no HIP bias table: tf_ink2_rel_table is built at extents 1024, 256 and 128", why.text());
}
