//! GLM-5.3-Flash's affine matrices at any MLX width: stored shapes, byte spans and the widths each device path reads.
const std = @import("std");
const mtl = @import("metal");
const quant = @import("quant.zig");

pub const Width = quant.Width;

pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn at(r: Ref, by: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + by };
    }

    pub fn addr(r: Ref) [*]u8 {
        return r.buf.contents() + r.off;
    }
};

/// The most runs a fused matrix splits into: its members.
pub const max_runs = 6;

/// A run of a split stack: one member (0.6.6 calls each apart), rows [row, row + n) of the fused output, bytes its own.
pub const Run = struct { w: Ref, s: Ref, b: Ref, row: u32, n: u32, bits: u8, group: u16 };

/// An MLX affine matrix: w [n, k bits / 32] u32, s and b [n, k / group] bf16 (stacked: [e, n, ...]).
pub const Affine = struct {
    w: Ref,
    s: Ref,
    b: Ref,
    n: u32,
    k: u32,
    bits: u8,
    group: u16,
    more: u8 = 0, // runs past the first, whose bytes and width are the matrix's own: 0 for one width
    runs: [max_runs - 1]Run = undefined,

    pub fn width(q: Affine) Width {
        return .{ .bits = q.bits, .group = q.group };
    }

    /// Run `i` (the first: the matrix's own bytes and width).
    pub fn run(q: Affine, i: usize) Run {
        if (i > 0) return q.runs[i - 1];
        return .{ .w = q.w, .s = q.s, .b = q.b, .row = 0, .n = if (q.more == 0) q.n else q.runs[0].row, .bits = q.bits, .group = q.group };
    }
};

/// Bytes of [n, k]'s packed codes at `w`.
pub fn wBytes(n: usize, k: usize, w: Width) usize {
    return n * k * w.bits / 8;
}

/// Bytes of [n, k]'s bf16 scale (or bias) table at `w`.
pub fn sBytes(n: usize, k: usize, w: Width) usize {
    return n * (k / w.group) * 2;
}

/// Bytes of [n, k]'s codes and both tables at `w`.
pub fn bytes(n: usize, k: usize, w: Width) usize {
    return wBytes(n, k, w) + 2 * sBytes(n, k, w);
}

/// [n, k]'s stored shapes at `w`: u32 words [n, k bits / 32], bf16 tables [n, k / group]; null when k does not pack.
pub fn shapes(n: usize, k: usize, w: Width) ?[2][2]usize {
    if (k == 0 or k % w.group != 0 or k * w.bits % 32 != 0) return null;
    return .{ .{ n, k * w.bits / 32 }, .{ n, k / w.group } };
}

/// Columns [lo, lo + k) as (offset, length) in a row's words and in its tables; null unless whole groups and words.
pub fn cut(lo: usize, k: usize, w: Width) ?[2][2]usize {
    if (lo % w.group != 0 or shapes(1, k, w) == null) return null;
    return .{ .{ wBytes(1, lo, w), wBytes(1, k, w) }, .{ sBytes(1, lo, w), sBytes(1, k, w) } };
}

/// Matrices (each run of a split stack) by width, for a plan's summary.
pub const Tally = struct {
    count: [9][3]u32 = @splat(@splat(0)), // by bits, then group 32, 64, 128

    pub fn add(t: *Tally, v: anytype) void {
        const T = @TypeOf(v);
        if (T == Affine) {
            for (0..@as(usize, v.more) + 1) |i| {
                const r = v.run(i);
                if (r.n > 0 and r.bits > 0) t.count[r.bits][std.math.log2_int(u16, r.group) - 5] += 1;
            }
            return;
        }
        switch (@typeInfo(T)) {
            .@"struct" => |s| inline for (s.field_names) |name| t.add(@field(v, name)),
            .@"union" => switch (v) {
                inline else => |x| t.add(x),
            },
            .optional => if (v) |x| t.add(x),
            .array => for (v) |x| t.add(x),
            else => {},
        }
    }

    /// "4-bit/64 x12, 8-bit/64 x3" in `buf`.
    pub fn summary(t: Tally, buf: []u8) []const u8 {
        var n: usize = 0;
        for (t.count, 0..) |g, bits| for (g, 0..) |x, gi| if (x > 0) {
            const sep = if (n == 0) "" else ", ";
            n += (std.fmt.bufPrint(buf[n..], "{s}{d}-bit/{d} x{d}", .{ sep, bits, @as(u32, 32) << @intCast(gi), x }) catch return buf[0..n]).len;
        };
        return buf[0..n];
    }
};

/// The device paths that read affine matrices, by projection class.
pub const Path = enum { embed, kda_in, kda_gate, kda_out, mla_in, mla_q, kv_b, mla_out, dense_mlp, shared_expert, routed_experts, head, eh_proj };

/// The code widths the row kernels (glm_rows.metal, after core/row_projection.metal) unpack, in groups of 32, 64, 128.
pub const row_bits = [_]u8{ 2, 3, 4, 5, 6, 8 };

/// Whether a path's kernels read `w`: kv_b 4/64, the embedding 4/6/8 in groups of 64, the rest the row kernels' widths.
pub fn reads(path: Path, w: Width) bool {
    return switch (path) {
        .kv_b => std.meta.eql(w, quant.uniform),
        .embed => w.group == 64 and (w.bits == 4 or w.bits == 6 or w.bits == 8),
        else => std.mem.indexOfScalar(u8, &row_bits, w.bits) != null and (w.group == 32 or w.group == 64 or w.group == 128),
    };
}

/// Refuses, by tensor name, a width the path's kernels do not read: a load fails before any weight is read.
pub fn admit(name: []const u8, path: Path, w: Width, rows: usize) error{ UnsupportedWidth, UnsupportedShape }!void {
    if (!reads(path, w)) {
        quant.say("glm: {s} is {d}-bit in groups of {d}, a width the {t} kernels do not read", .{ name, w.bits, w.group, path });
        return error.UnsupportedWidth;
    }
    if (rows % 4 == 0) return;
    quant.say("glm: {s} has {d} rows; the row kernels read a matrix's rows in fours", .{ name, rows });
    return error.UnsupportedShape;
}

/// Whether every matrix in `v` takes the generated kernels (one width, 4/64): no other width's pipelines needed.
pub fn allNative(v: anytype) bool {
    const T = @TypeOf(v);
    if (T == Affine) return v.n == 0 or (v.more == 0 and v.bits == 4 and v.group == 64);
    return switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names) |name| {
            if (!allNative(@field(v, name))) break false;
        } else true,
        .@"union" => switch (v) {
            inline else => |x| allNative(x),
        },
        .optional => if (v) |x| allNative(x) else true,
        .array => for (v) |x| {
            if (!allNative(x)) break false;
        } else true,
        else => true,
    };
}

const testing = std.testing;

test "stored shapes and bytes follow the width; 4/64 keeps the 4-bit matrix's" {
    try testing.expectEqual([2][2]usize{ .{ 4096, 512 }, .{ 4096, 64 } }, shapes(4096, 4096, quant.uniform).?);
    try testing.expectEqual(@as(usize, 4096 * 4096 / 2), wBytes(4096, 4096, quant.uniform));
    try testing.expectEqual(@as(usize, 4096 * (4096 / 64) * 2), sBytes(4096, 4096, quant.uniform));
    const cases = [_]struct { Width, [2]usize }{
        .{ .{ .bits = 2, .group = 64 }, .{ 16, 4 } },
        .{ .{ .bits = 3, .group = 32 }, .{ 24, 8 } },
        .{ .{ .bits = 5, .group = 64 }, .{ 40, 4 } },
        .{ .{ .bits = 6, .group = 128 }, .{ 48, 2 } },
        .{ .{ .bits = 8, .group = 64 }, .{ 64, 4 } },
    };
    for (cases) |c| {
        const s = shapes(7, 256, c[0]).?;
        try testing.expectEqual(c[1][0], s[0][1]);
        try testing.expectEqual(c[1][1], s[1][1]);
        try testing.expectEqual(7 * 4 * s[0][1], wBytes(7, 256, c[0]));
    }
    try testing.expect(shapes(1, 96, .{ .bits = 4, .group = 64 }) == null);
}

test "a column cut keeps whole groups and words at every width" {
    try testing.expectEqual([2][2]usize{ .{ 2048, 2048 }, .{ 128, 128 } }, cut(4096, 4096, quant.uniform).?);
    try testing.expectEqual([2][2]usize{ .{ 640, 640 }, .{ 32, 32 } }, cut(1024, 1024, .{ .bits = 5, .group = 64 }).?);
    try testing.expect(cut(32, 1024, quant.uniform) == null);
    try testing.expect(cut(64, 1024, .{ .bits = 6, .group = 128 }) == null);
}

test "every device path admits 4/64; the oQ widths where their kernels read them, others refused by name" {
    for (std.enums.values(Path)) |p| {
        try admit("x", p, quant.uniform, 64);
        try testing.expectError(error.UnsupportedWidth, admit("x", p, .{ .bits = 7, .group = 64 }, 64));
        try testing.expectError(error.UnsupportedWidth, admit("x", p, .{ .bits = 4, .group = 16 }, 64));
        try testing.expectError(error.UnsupportedShape, admit("x", p, quant.uniform, 66));
    }
    for ([_]Width{ .{ .bits = 2, .group = 64 }, .{ .bits = 5, .group = 64 }, .{ .bits = 6, .group = 64 }, .{ .bits = 8, .group = 64 }, .{ .bits = 3, .group = 32 }, .{ .bits = 5, .group = 128 } }) |w|
        for ([_]Path{ .kda_in, .kda_gate, .kda_out, .mla_in, .mla_q, .mla_out, .dense_mlp, .shared_expert, .routed_experts, .head, .eh_proj }) |p| try admit("x", p, w, 64);
    try admit("x", .embed, .{ .bits = 8, .group = 64 }, 64);
    try testing.expectError(error.UnsupportedWidth, admit("x", .embed, .{ .bits = 2, .group = 64 }, 64));
    try testing.expectError(error.UnsupportedWidth, admit("x", .kv_b, .{ .bits = 8, .group = 64 }, 64));
}

test "a matrix of another width or a split stack needs the other widths' pipelines; 4/64 and none do not" {
    const r: Ref = .{ .buf = undefined };
    const q: Affine = .{ .w = r, .s = r, .b = r, .n = 64, .k = 128, .bits = 4, .group = 64 };
    const none: Affine = .{ .w = r, .s = r, .b = r, .n = 0, .k = 128, .bits = 0, .group = 0 };
    try testing.expect(allNative(.{ q, none, @as(?Affine, null), [2]Affine{ q, q } }));
    var eight = q;
    eight.bits = 8;
    try testing.expect(!allNative(.{ q, [2]Affine{ q, eight } }));
    var split = q;
    split.more = 1;
    try testing.expect(!allNative(@as(?Affine, split)));
}
