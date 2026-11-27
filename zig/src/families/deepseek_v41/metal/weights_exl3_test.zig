//! An EXL3 pack's admission on the host: each group's shape, width and codebook as placed, and the refusals.
const std = @import("std");
const host = @import("dsv41");
const weights = @import("weights.zig");
const x3w = @import("weights_exl3.zig");
const fx = @import("exl3_fixture.zig");

const testing = std.testing;
const Config = host.config.Config;
const Ids = host.config.Ids;

/// The tiny geometry (all linears multiples of 128); layer 1 a ratio-2 KV and index source.
fn config() Config {
    return .{
        .format = .exl3,
        .vocab = 256,
        .hidden = 256,
        .layers = 2,
        .heads = 4,
        .q_lora = 128,
        .o_lora = 128,
        .o_groups = 2,
        .experts = 16,
        .expert_width = 256,
        .compress_ratios = Ids.of(&.{ 0, 2 }),
        .kv_sources = Ids.of(&.{1}),
        .index_sources = Ids.of(&.{1}),
        .candidate_source = 1,
        .engram_layers = Ids.of(&.{}),
        .engram_rows = host.config.List(u64, 8).of(&.{}),
        .mtp_layers = 0,
        .dspark_targets = Ids.of(&.{}),
    };
}

/// A pack folder of `ts` (its image's last `cut` bytes dropped), opened.
const Folder = struct {
    tmp: testing.TmpDir,
    pack: host.pack.Pack,

    fn open(a: std.mem.Allocator, ts: []const fx.Tensor, cut: usize, poke: ?[]const u8) !Folder {
        const img = try fx.image(a, ts, 5);
        defer a.free(img);
        // a planted fault: the named marker's value changed
        if (poke) |name| {
            const head: usize = @intCast(std.mem.readInt(u64, img[0..8], .little));
            var at = 8 + head;
            for (ts) |t| {
                var n = host.pack.DType.parse(t.spec.dtype).?.size();
                for (t.spec.shape) |d| n *= d;
                if (std.mem.eql(u8, t.spec.name, name)) img[at] ^= 1;
                at += n;
            }
        }
        var f: Folder = .{ .tmp = testing.tmpDir(.{}), .pack = undefined };
        errdefer f.tmp.cleanup();
        try f.tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors", .data = img[0 .. img.len - cut] });
        const dir = try f.tmp.dir.realPathFileAlloc(testing.io, ".", a);
        defer a.free(dir);
        f.pack = try host.pack.Pack.open(a, testing.io, dir);
        return f;
    }

    fn close(f: *Folder) void {
        f.pack.deinit();
        f.tmp.cleanup();
    }
};

/// One group's tensors, as `fx.tensors` writes them.
fn one(a: std.mem.Allocator, g: fx.Group) !std.ArrayList(fx.Tensor) {
    var ts: std.ArrayList(fx.Tensor) = .empty;
    try fx.tensors(a, g, &ts);
    return ts;
}

test "an EXL3 layer's groups place at the config's shapes, widths and codebooks, markers checked after the fill" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = config();
    var gs: std.ArrayList(fx.Group) = .empty;
    var turn: usize = 0;
    var bb: [32]u8 = undefined;
    const block = try host.names.blockPrefix(&c, 1, &bb);
    try fx.groups(a, &c, block, 1, &gs, &turn);
    var ts: std.ArrayList(fx.Tensor) = .empty;
    for (gs.items) |g| try fx.tensors(a, g, &ts);
    var f = try Folder.open(a, ts.items, 0, null);
    defer f.close();
    var l: weights.Loader = .{ .a = a, .pack = &f.pack };
    const p = try x3w.place(&l, a, block, &c, 1);
    try testing.expectEqual(@as(usize, c.experts), p.experts.len);
    // every group a layer holds, re-read by name against the width and codebook it was written with
    for (gs.items) |g| {
        const got = try x3w.group(&l, g.name, @intCast(g.k), @intCast(g.n));
        try testing.expectEqual(g.halves, got.halves);
        try testing.expectEqual(g.book, got.book);
    }
    try testing.expectEqual(@as(?x3w.Placed, null), p.dense.engram_wkv);
    try testing.expect(p.dense.comp_wgate != null and p.dense.ix_wq_b != null and p.dense.wo_a[c.o_groups] == null);
    const bytes = try a.alloc(u8, l.size);
    try l.fill(testing.io, bytes);
    try x3w.checkMarkers(p, bytes);
}

test "an EXL3 group is the trellis where the pack also keeps a dense sibling" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g: fx.Group = .{ .name = "layers.0.attn.wq_a", .k = 256, .n = 128, .halves = 5, .book = .mul1 };
    var ts = try one(a, g);
    // a dense sibling of the same K x N, which the storage reader would take: the lane must still take the trellis
    try ts.append(a, .{ .spec = .{ .name = "layers.0.attn.wq_a.weight", .dtype = "BF16", .shape = &.{ 128, 256 } }, .fill = .words, .k = 256 });
    var f = try Folder.open(a, ts.items, 0, null);
    defer f.close();
    var l: weights.Loader = .{ .a = a, .pack = &f.pack };
    const stored = try host.projection.read(&f.pack, g.name);
    try testing.expect(stored == .dense);
    try testing.expectEqual(@as(usize, 256), stored.dense.k);
    try testing.expectEqual(@as(usize, 128), stored.dense.n);
    const got = try x3w.group(&l, g.name, 256, 128);
    try testing.expectEqual(g.halves, got.halves);
    try testing.expectEqual(g.book, got.book);
    try testing.expectEqual(@as(u32, 256), got.k);
    try testing.expectEqual(@as(u32, 128), got.n);
}

test "EXL3 admission refusals: wrong format either way, shape, width, marker, truncated shard" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const g: fx.Group = .{ .name = "layers.0.attn.wq_a", .k = 256, .n = 128, .halves = 5, .book = .mul1 };
    {
        // the release's FP8 tensors where an EXL3 pack's trellis is wanted
        var ts: std.ArrayList(fx.Tensor) = .empty;
        try ts.append(a, .{ .spec = .{ .name = "layers.0.attn.wq_a.weight", .dtype = "F8_E4M3", .shape = &.{ 128, 256 } }, .fill = .words, .k = 256 });
        try ts.append(a, .{ .spec = .{ .name = "layers.0.attn.wq_a.scale", .dtype = "F8_E8M0", .shape = &.{ 4, 8 } }, .fill = .words, .k = 256 });
        var f = try Folder.open(a, ts.items, 0, null);
        defer f.close();
        var l: weights.Loader = .{ .a = a, .pack = &f.pack };
        try testing.expectError(error.UnsupportedProjection, x3w.group(&l, g.name, 256, 128));
    }
    {
        // an EXL3 pack read as the release's: its trellis is not the FP8 projection asked for
        const ts = try one(a, g);
        var f = try Folder.open(a, ts.items, 0, null);
        defer f.close();
        var l: weights.Loader = .{ .a = a, .pack = &f.pack };
        try testing.expectError(error.UnsupportedProjection, l.fp8("layers.0", "attn.wq_a", 256, 128));
        // the group at another K x N than the config's
        try testing.expectError(error.BadProjectionShape, x3w.group(&l, g.name, 128, 256));
    }
    // widths the kernels do not decode: 4.5 bits, and 2.5 bits on a codebook other than mul1
    for ([_]struct { u16, fx.Book, anyerror }{ .{ 9, .mul1, error.UnsupportedExl3Width }, .{ 5, .mcg, error.BadMarker } }) |w| {
        var gw = g;
        gw.halves = w[0];
        gw.book = w[1];
        const ts = try one(a, gw);
        var f = try Folder.open(a, ts.items, 0, null);
        defer f.close();
        var l: weights.Loader = .{ .a = a, .pack = &f.pack };
        try testing.expectError(w[2], x3w.group(&l, g.name, 256, 128));
    }
    {
        var ts = try one(a, g);
        ts.items[0].spec.shape = &.{ 16, 8, 20 };
        var f = try Folder.open(a, ts.items, 0, null);
        defer f.close();
        var l: weights.Loader = .{ .a = a, .pack = &f.pack };
        try testing.expectError(error.BadTrellis, x3w.group(&l, g.name, 256, 128));
    }
    {
        // a marker whose value is not its codebook's
        const ts = try one(a, g);
        var f = try Folder.open(a, ts.items, 0, "layers.0.attn.wq_a.mul1");
        defer f.close();
        var l: weights.Loader = .{ .a = a, .pack = &f.pack };
        const p = try x3w.group(&l, g.name, 256, 128);
        const bytes = try a.alloc(u8, l.size);
        try l.fill(testing.io, bytes);
        try testing.expectError(error.BadMarker, x3w.checkMarker(p, bytes));
    }
    {
        // a shard cut short: its header promises bytes the file does not hold, refused when the pack opens
        const ts = try one(a, g);
        try testing.expectError(error.BadSafetensors, Folder.open(a, ts.items, 100, null));
    }
}
