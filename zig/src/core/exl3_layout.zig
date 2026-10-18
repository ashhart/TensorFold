//! EXL3 groups as the pack stores them: trellis geometry, codebooks, a rank's TP slice as byte spans, expert layouts.

const std = @import("std");

/// The codebook a group's trellis decodes with: the `.mul1` / `.mcg` marker's value, else the 3-instruction default.
pub const Codebook = @import("exl3_load.zig").Codebook;
pub const marker_mul1: u32 = 0x83DCD12D;
pub const marker_mcg: u32 = 0xCBAC1FED;
pub fn fromMarker(v: u32) !Codebook {
    return switch (v) {
        marker_mul1 => .mul1,
        marker_mcg => .mcg,
        else => error.UnknownCodebookMarker,
    };
}

/// How a rank holds a group under TP (every boundary a multiple of 128, so Hadamard blocks stay whole).
pub const Split = enum {
    /// all of it, on every rank
    whole,
    /// this rank's output columns: N tiles and `svh`
    col,
    /// this rank's input rows: K tiles and `suh` (a partial sum the ranks exchange)
    row,
};

/// One EXL3 matrix y = x @ W, W [K, N]: trellis int16 [K/16, N/16, words], suh fp16 [K], svh fp16 [N].
pub const Group = struct {
    k_tiles: u32,
    n_tiles: u32,
    /// int16 words a 16x16 tile: 16 x bits (the trellis's last dimension)
    words: u32,

    /// From the trellis entry's shape; a tile must hold whole K2 bytes and 1 to 8 bits a value.
    pub fn fromShape(shape: []const usize) !Group {
        if (shape.len != 3) return error.BadTrellis;
        if (shape[0] > std.math.maxInt(u32) / 16 or shape[1] > std.math.maxInt(u32) / 16) return error.BadTrellis;
        const w = shape[2];
        if (w < 16 or w % 8 != 0 or w > 128 or shape[0] == 0 or shape[1] == 0) return error.BadTrellis;
        return .{ .k_tiles = @intCast(shape[0]), .n_tiles = @intCast(shape[1]), .words = @intCast(w) };
    }

    pub fn k(g: Group) u32 {
        return 16 * g.k_tiles;
    }

    pub fn n(g: Group) u32 {
        return 16 * g.n_tiles;
    }

    /// Half-bits a value (the kernels' K2): 4 for 2 bits, 16 for 8 bits.
    pub fn k2(g: Group) u32 {
        return g.words / 8;
    }

    /// Bits a value, in halves exact (k2 / 2).
    pub fn bits(g: Group) f32 {
        return @as(f32, @floatFromInt(g.words)) / 16.0;
    }

    pub fn tileBytes(g: Group) u64 {
        return @as(u64, g.words) * 2;
    }

    pub fn trellisBytes(g: Group) u64 {
        return @as(u64, g.k_tiles) * g.n_tiles * g.tileBytes();
    }

    /// The rank's part of the group; a split that does not fall on 128-value blocks is refused.
    pub fn part(g: Group, split: Split, rank: u32, world: u32) !Part {
        if (world == 0 or rank >= world or world > std.math.maxInt(u32) / 128) return error.SplitNotOnBlocks;
        var p: Part = .{ .group = g, .trellis = .{ .offset = 0, .len = g.trellisBytes() }, .suh = .{ .offset = 0, .len = 2 * @as(u64, g.k()) }, .svh = .{ .offset = 0, .len = 2 * @as(u64, g.n()) } };
        switch (split) {
            .whole => {},
            .row => {
                if (g.k() % (world * 128) != 0) return error.SplitNotOnBlocks;
                const per = g.k_tiles / world;
                p.group.k_tiles = per;
                const row = @as(u64, g.n_tiles) * g.tileBytes();
                p.trellis = .{ .offset = rank * per * row, .len = per * row };
                p.suh = .{ .offset = 2 * 16 * @as(u64, rank * per), .len = 2 * 16 * @as(u64, per) };
            },
            .col => {
                if (g.n() % (world * 128) != 0) return error.SplitNotOnBlocks;
                const per = g.n_tiles / world;
                p.group.n_tiles = per;
                // one K-tile row at a time: this rank's N tiles are a run inside each row
                p.trellis = .{ .offset = rank * per * g.tileBytes(), .len = per * g.tileBytes(), .stride = @as(u64, g.n_tiles) * g.tileBytes(), .count = g.k_tiles };
                p.svh = .{ .offset = 2 * 16 * @as(u64, rank * per), .len = 2 * 16 * @as(u64, per) };
            },
        }
        return p;
    }
};

/// Bytes of a stored tensor to read: `count` runs of `len` bytes, `stride` apart, from `offset` into the tensor.
pub const Span = struct {
    offset: u64,
    len: u64,
    stride: u64 = 0,
    count: u64 = 1,

    pub fn bytes(s: Span) u64 {
        return s.len * s.count;
    }

    /// The `i`th run's start.
    pub fn at(s: Span, i: u64) u64 {
        return s.offset + i * s.stride;
    }
};

/// A rank's slice of a group: its geometry and where its trellis, suh and svh bytes sit in the stored tensors.
pub const Part = struct {
    group: Group,
    trellis: Span,
    suh: Span,
    svh: Span,

    pub fn bytes(p: Part) u64 {
        return p.trellis.bytes() + p.suh.bytes() + p.svh.bytes();
    }
};

/// Routed experts of one projection on a rank: one width (X3Stack) or a width per expert (X3Ragged, E2).
pub const Layout = struct {
    kind: enum { stack, ragged },
    k_tiles: u32,
    n_tiles: u32,
    /// each expert's int16 words a tile
    words: []u16,
    /// each expert's first int16 element in the projection's trellis buffer (back to back, as `ragged()` packs them)
    offsets: []u64,
    /// int16 elements of every expert's trellis
    elements: u64,

    /// The layout of the rank's expert parts; parts must share K and N (the kernels' tables index one shape).
    pub fn of(gpa: std.mem.Allocator, parts: []const Part) !Layout {
        if (parts.len == 0) return error.NoExperts;
        const words = try gpa.alloc(u16, parts.len);
        errdefer gpa.free(words);
        const offsets = try gpa.alloc(u64, parts.len);
        errdefer gpa.free(offsets);
        const kt = parts[0].group.k_tiles;
        const nt = parts[0].group.n_tiles;
        var at: u64 = 0;
        var same = true;
        for (parts, words, offsets) |p, *w, *o| {
            if (p.group.k_tiles != kt or p.group.n_tiles != nt) return error.ExpertShapesDiffer;
            w.* = @intCast(p.group.words);
            o.* = at;
            const elements_added = std.math.mul(u64, @as(u64, kt) * nt, p.group.words) catch return error.BadTrellis;
            at = std.math.add(u64, at, elements_added) catch return error.BadTrellis;
            if (at > std.math.maxInt(u64) / 2) return error.BadTrellis;
            same = same and p.group.words == parts[0].group.words;
        }
        return .{ .kind = if (same) .stack else .ragged, .k_tiles = kt, .n_tiles = nt, .words = words, .offsets = offsets, .elements = at };
    }

    pub fn deinit(l: *Layout, gpa: std.mem.Allocator) void {
        gpa.free(l.words);
        gpa.free(l.offsets);
        l.* = undefined;
    }

    pub fn count(l: *const Layout) usize {
        return l.words.len;
    }

    pub fn trellisBytes(l: *const Layout) u64 {
        return 2 * l.elements;
    }

    /// Byte offset of expert `e`'s trellis in the projection buffer: 16-byte aligned (a tile is 16 K2 bytes).
    pub fn byteOffset(l: *const Layout, e: usize) u64 {
        return 2 * l.offsets[e];
    }

    /// Experts at each K2 (index = K2, 1..16): the widths x3gm launches once each.
    pub fn histogram(l: *const Layout) [17]u32 {
        var h: [17]u32 = @splat(0);
        for (l.words) |w| h[w / 8] += 1;
        return h;
    }

    /// Distinct widths: x3gm's launches for this projection.
    pub fn widths(l: *const Layout) u32 {
        var n: u32 = 0;
        for (l.histogram()) |c| n += @intFromBool(c > 0);
        return n;
    }
};

const testing = std.testing;

test "group geometry and K2" {
    const g = try Group.fromShape(&.{ 320, 144, 32 });
    try testing.expectEqual(@as(u32, 5120), g.k());
    try testing.expectEqual(@as(u32, 2304), g.n());
    try testing.expectEqual(@as(u32, 4), g.k2());
    try testing.expectEqual(@as(f32, 2.0), g.bits());
    try testing.expectError(error.BadTrellis, Group.fromShape(&.{ 320, 144, 30 }));
    try testing.expectError(error.BadTrellis, Group.fromShape(&.{ 320, 144 }));
    try testing.expectEqual(Codebook.mcg, try Codebook.fromMarker(0xCBAC1FED));
    try testing.expectError(error.UnknownCodebookMarker, Codebook.fromMarker(7));
}

test "TP parts: rows contiguous, columns a run a K-tile row" {
    const w2 = try Group.fromShape(&.{ 144, 320, 48 }); // a 3-bit routed down projection: 2304 -> 5120
    const r1 = try w2.part(.row, 1, 2);
    try testing.expectEqual(@as(u32, 72), r1.group.k_tiles);
    try testing.expectEqual(@as(u64, 72 * 320 * 96), r1.trellis.len);
    try testing.expectEqual(r1.trellis.len, r1.trellis.offset);
    try testing.expectEqual(Span{ .offset = 2 * 1152, .len = 2 * 1152 }, r1.suh);
    try testing.expectEqual(@as(u64, 2 * 5120), r1.svh.len);
    const w1 = try Group.fromShape(&.{ 320, 144, 32 });
    const c1 = try w1.part(.col, 1, 2);
    try testing.expectEqual(@as(u32, 72), c1.group.n_tiles);
    try testing.expectEqual(@as(u64, 320), c1.trellis.count);
    try testing.expectEqual(@as(u64, 144 * 64), c1.trellis.stride);
    try testing.expectEqual(@as(u64, 72 * 64), c1.trellis.offset);
    try testing.expectEqual(c1.group.trellisBytes(), c1.trellis.bytes());
    // the 8-bit indexer key projection (512 -> 128) cannot split its 128 outputs over two ranks
    const wk = try Group.fromShape(&.{ 32, 8, 128 });
    try testing.expectError(error.SplitNotOnBlocks, wk.part(.col, 0, 2));
    try testing.expectEqual(wk.trellisBytes(), (try wk.part(.whole, 1, 2)).trellis.bytes());
}

test "expert layouts: one width stacks, mixed widths go ragged with aligned offsets" {
    const a = testing.allocator;
    const two = try (try Group.fromShape(&.{ 144, 320, 32 })).part(.row, 0, 2);
    const three = try (try Group.fromShape(&.{ 144, 320, 48 })).part(.row, 0, 2);
    var stack = try Layout.of(a, &.{ two, two, two });
    defer stack.deinit(a);
    try testing.expect(stack.kind == .stack);
    try testing.expectEqual(@as(u32, 1), stack.widths());
    // a mixed-bit pack's layer 2 down projection on rank 0: 383 experts at 2 bits and one at 3 bits
    var parts: [384]Part = @splat(two);
    parts[200] = three;
    var rag = try Layout.of(a, &parts);
    defer rag.deinit(a);
    try testing.expect(rag.kind == .ragged);
    try testing.expectEqual(@as(u64, 566968320), rag.trellisBytes());
    try testing.expectEqual(@as(u32, 383), rag.histogram()[4]);
    try testing.expectEqual(@as(u32, 1), rag.histogram()[6]);
    try testing.expectEqual(@as(u64, 0), rag.byteOffset(201) % 16);
    try testing.expectEqual(rag.byteOffset(200) + three.trellis.bytes(), rag.byteOffset(201));
    const odd = try (try Group.fromShape(&.{ 144, 160, 32 })).part(.row, 0, 2);
    try testing.expectError(error.ExpertShapesDiffer, Layout.of(a, &.{ two, odd }));
}

pub const source_bytes = @embedFile("exl3_layout.zig");
