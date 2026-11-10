//! The DFlash2 drafter's launches: tap gathers, its key ring, the dynamic conv and block attention (draft/draft.hip).

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const S = util.S;
const Args = util.Args;
const Error = util.Error;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;

/// The drafter's head width, the keys a partial scores, and the partials a row: a window of 2048 and a block.
pub const head = 128;
pub const chunk = 256;
pub const chunks = 9;
/// The most rows a gather takes by value, and the most block rows a mask's bits reach.
pub const most_rows = 64;
pub const mask_rows = 32;

/// draft.hip's DraftRows: rows of a window by value (n 0: in order).
pub const Rows = extern struct { n: c_int = 0, rows: [most_rows]c_int = @splat(0) };

/// draft.hip's DraftSegment: a stream's drafter ring, its context's first position and the block's first position.
pub const Segment = extern struct { ring: u64, begin: i64, position: i64 };

/// The fp32 partials of `rows` rows of `heads` heads: max and sum `chunks` a head, then the unnormalized outputs.
pub const Partials = struct {
    m: u64,
    l: u64,
    o: u64,

    pub fn bytes(rows: usize, heads: usize) usize {
        return rows * heads * chunks * (2 + head) * 4;
    }

    /// The three arrays packed in one buffer of `bytes(rows, heads)`.
    pub fn at(base: u64, rows: usize, heads: usize) Partials {
        const n = rows * heads * chunks * 4;
        return .{ .m = base, .l = base + n, .o = base + 2 * n };
    }
};

pub const Fns = struct {
    gather: Function,
    ring_put: Function,
    dyn_conv: Function,
    mask_rows: Function,
    attn_partial: Function,
    attn_merge: Function,

    pub fn load(m: Module) Error!Fns {
        return .{
            .gather = try m.function("tf_draft_gather"),
            .ring_put = try m.function("tf_draft_ring_put"),
            .dyn_conv = try m.function("tf_draft_dyn_conv"),
            .mask_rows = try m.function("tf_draft_mask_rows"),
            .attn_partial = try m.function("tf_draft_attn_partial"),
            .attn_merge = try m.function("tf_draft_attn_merge"),
        };
    }
};

/// out rows i = ring rows (start + i) % ring_rows, or (start + rows[i]) % ring_rows; `words` 32-bit words a row.
pub fn gather(l: *const Launcher, ring: u64, ring_rows: usize, start: u64, rows: ?[]const u32, n: usize, words: usize, out: u64, s: S) Error!void {
    if (n == 0) return;
    if (ring_rows == 0 or words == 0 or start > std.math.maxInt(i64)) return invalid("draft gather");
    var picked: Rows = .{};
    if (rows) |rs| {
        if (rs.len != n or n > most_rows) return invalid("draft gather rows");
        picked.n = @intCast(n);
        for (rs, 0..) |r, i| picked.rows[i] = @intCast(r);
    }
    var a: Args = .{};
    try a.add(ring);
    try a.add(@as(c_int, @intCast(ring_rows)));
    try a.add(@as(c_longlong, @intCast(start)));
    try a.add(picked);
    try a.add(@as(c_int, @intCast(words)));
    try a.add(out);
    try l.go(l.draft.gather, dim(n, 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// ringPut's shapes: no more rows than ring slots, a nonzero width, and a first position within i64.
pub fn ringPutFits(rows: usize, width: usize, pos0: u64, ring_rows: usize) bool {
    return ring_rows >= rows and width != 0 and pos0 <= std.math.maxInt(i64);
}

/// keys (fp32) and values (`kind`) of `rows` rows of `width` into a layer's rings at slots (pos0 + i) % ring_rows.
pub fn ringPut(l: *const Launcher, keys: u64, values: u64, kind: c_int, k_ring: u64, v_ring: u64, rows: usize, width: usize, pos0: u64, ring_rows: usize, s: S) Error!void {
    if (rows == 0) return;
    if (!ringPutFits(rows, width, pos0, ring_rows)) return invalid("draft ring put");
    var a: Args = .{};
    try a.add(keys);
    try a.add(values);
    try a.add(kind);
    try a.add(k_ring);
    try a.add(v_ring);
    try a.add(@as(c_int, @intCast(width)));
    try a.add(@as(c_longlong, @intCast(pos0)));
    try a.add(@as(c_int, @intCast(ring_rows)));
    try l.go(l.draft.ring_put, dim(rows, 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// dynConv's shapes: a width in whole groups, branch 0 or 1, and blocks of at least one row.
pub fn dynConvFits(d: usize, group: usize, branch: usize, block_rows: usize) bool {
    return group != 0 and d % group == 0 and branch <= 1 and block_rows != 0;
}

/// The dynamic conv's `branch` over `rows` rows of width `d`, rows grouped by `block_rows` a stream; `res` 0: none.
pub fn dynConv(l: *const Launcher, x: u64, dyn: u64, base: u64, res: u64, out: u64, kind: c_int, rows: usize, d: usize, group: usize, branch: usize, block_rows: usize, s: S) Error!void {
    if (rows == 0) return;
    if (!dynConvFits(d, group, branch, block_rows)) return invalid("draft dyn conv");
    var a: Args = .{};
    try a.add(x);
    try a.add(dyn);
    try a.add(base);
    try a.add(res);
    try a.add(out);
    try a.add(kind);
    try a.add(@as(c_int, @intCast(d)));
    try a.add(@as(c_int, @intCast(group)));
    try a.add(@as(c_int, @intCast(branch)));
    try a.add(@as(c_int, @intCast(block_rows)));
    try l.go(l.draft.dyn_conv, dim(rows, cdiv(d, 256), 1), dim(256, 1, 1), 0, s, &a);
}

/// maskRows' shapes: blocks of at least two rows, and a nonzero width.
pub fn maskRowsFits(d: usize, block_rows: usize) bool {
    return block_rows >= 2 and d != 0;
}

/// Each of `streams` streams' block rows but its first, packed: (streams * (block_rows - 1), d) 16-bit values.
pub fn maskRows(l: *const Launcher, x: u64, out: u64, streams: usize, d: usize, block_rows: usize, s: S) Error!void {
    if (streams == 0) return;
    if (!maskRowsFits(d, block_rows)) return invalid("draft mask rows");
    var a: Args = .{};
    try a.add(x);
    try a.add(out);
    try a.add(@as(c_int, @intCast(d)));
    try a.add(@as(c_int, @intCast(block_rows)));
    try l.go(l.draft.mask_rows, dim(streams * (block_rows - 1), 1, 1), dim(256, 1, 1), 0, s, &a);
}

/// The block attention's shape: rows grouped by `block_rows` a stream, GQA heads, a ring of `ring_rows`, `window`.
pub const Attention = struct {
    rows: usize,
    block_rows: usize,
    heads: usize,
    kv_heads: usize,
    ring_rows: usize,
    window: usize,
    layer: usize,

    /// Whether the kernels take the shape: rows in whole blocks, GQA heads, keys within the partials, mask bits.
    pub fn check(at: Attention, masked: bool) bool {
        if (at.rows == 0 or at.block_rows == 0 or at.rows % at.block_rows != 0 or at.window == 0) return false;
        if (at.kv_heads == 0 or at.heads % at.kv_heads != 0 or at.ring_rows < at.window) return false;
        if (masked and at.block_rows > mask_rows) return false;
        return at.window - 1 + at.block_rows <= chunk * chunks;
    }
};

/// Partials of every row's ring window and block rows, then their merge into out (rows, heads, 128) of `kind`.
pub fn attention(l: *const Launcher, q: u64, kb: u64, vb: u64, mask: u64, segments: u64, kind: c_int, at: Attention, p: Partials, out: u64, s: S) Error!void {
    if (!at.check(mask != 0)) return invalid("draft attention");
    var a: Args = .{};
    try a.add(q);
    try a.add(kb);
    try a.add(vb);
    try a.add(mask);
    try a.add(segments);
    try a.add(kind);
    try a.add(@as(c_int, @intCast(at.block_rows)));
    try a.add(@as(c_int, @intCast(at.heads)));
    try a.add(@as(c_int, @intCast(at.kv_heads)));
    try a.add(@as(c_int, @intCast(at.ring_rows)));
    try a.add(@as(c_int, @intCast(at.window)));
    try a.add(@as(c_int, @intCast(at.layer)));
    try a.add(p.m);
    try a.add(p.l);
    try a.add(p.o);
    try l.go(l.draft.attn_partial, dim(at.rows, at.heads, chunks), dim(128, 1, 1), 0, s, &a);
    var b: Args = .{};
    try b.add(p.m);
    try b.add(p.l);
    try b.add(p.o);
    try b.add(@as(c_int, @intCast(at.heads)));
    try b.add(kind);
    try b.add(out);
    try l.go(l.draft.attn_merge, dim(at.rows, at.heads, 1), dim(32, 1, 1), 0, s, &b);
}

test "the ring put refuses more rows than ring slots, a zero width and a first position past i64" {
    const top: u64 = std.math.maxInt(i64);
    try std.testing.expect(ringPutFits(16, 1024, 0, 2048));
    try std.testing.expect(ringPutFits(2048, 1024, top, 2048));
    try std.testing.expect(ringPutFits(1, 1, 0, 1));
    try std.testing.expect(!ringPutFits(2049, 1024, 0, 2048));
    try std.testing.expect(!ringPutFits(2, 1, 0, 1));
    try std.testing.expect(!ringPutFits(16, 0, 0, 2048));
    try std.testing.expect(!ringPutFits(16, 1024, top + 1, 2048));
}

test "the dyn conv refuses a width off its group, a zero group, a branch past 1 and zero block rows" {
    for ([_]usize{ 0, 1 }) |branch| try std.testing.expect(dynConvFits(512, 16, branch, 8));
    try std.testing.expect(dynConvFits(512, 512, 0, 1));
    for ([_]usize{ 511, 513, 512 + 8 }) |d| try std.testing.expect(!dynConvFits(d, 16, 0, 8));
    try std.testing.expect(!dynConvFits(512, 0, 0, 8));
    for ([_]usize{ 2, 3 }) |branch| try std.testing.expect(!dynConvFits(512, 16, branch, 8));
    try std.testing.expect(!dynConvFits(512, 16, 0, 0));
}

test "the mask rows refuse a block under 2 rows and a zero width" {
    for ([_]usize{ 2, 8, mask_rows }) |block| try std.testing.expect(maskRowsFits(256, block));
    try std.testing.expect(maskRowsFits(1, 2));
    for ([_]usize{ 0, 1 }) |block| try std.testing.expect(!maskRowsFits(256, block));
    try std.testing.expect(!maskRowsFits(0, 8));
}
