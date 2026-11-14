//! One MLX affine module's .weight, .scales and .biases read by name and checked as a projection: views, no copies.
const std = @import("std");

pub const Error = error{UnexpectedAffineTensor};

/// What the family knows: `stack` projections [n, k] (1: one, rank 2), and the bits and group it fixes, if any.
pub const Want = struct { stack: usize = 1, n: usize, k: usize, bits: ?u8 = null, group: ?usize = null };

/// The stored dims: u32 words [.., n, k * bits / 32], tables [.., n, k / group]; `rank` 2, or 3 for a stack.
pub const Dims = struct { rank: usize, words: [3]usize, tables: [3]usize };

/// The three tensors as the checkpoint holds them, and the width their shapes carry.
pub fn Triple(comptime T: type) type {
    return struct { weight: T, scales: T, biases: T, n: usize, k: usize, bits: u8, group: usize };
}

/// The dims a projection of `bits` and `group` is stored in; refused unless K packs into whole words and groups.
pub fn dims(want: Want, bits: usize, group: usize) Error!Dims {
    if (want.stack == 0 or want.n == 0 or want.k == 0 or bits == 0 or group == 0) return error.UnexpectedAffineTensor;
    if (want.n > std.math.maxInt(i32) or want.k > std.math.maxInt(i32)) return error.UnexpectedAffineTensor;
    if (want.k % group != 0 or want.k * bits % 32 != 0) return error.UnexpectedAffineTensor;
    const words = want.k * bits / 32;
    const groups = want.k / group;
    if (want.stack == 1) return .{ .rank = 2, .words = .{ want.n, words, 0 }, .tables = .{ want.n, groups, 0 } };
    return .{ .rank = 3, .words = .{ want.stack, want.n, words }, .tables = .{ want.stack, want.n, groups } };
}

fn byteLen(t: anytype) usize {
    return if (@TypeOf(t.bytes) == usize) t.bytes else t.bytes.len;
}

/// `t` is `dtype` with exactly `want`, its bytes that extent.
pub fn require(t: anytype, dtype: @TypeOf(t.dtype), want: []const usize) Error!void {
    if (t.dtype != dtype or t.rank != want.len or t.rank > t.shape.len) return error.UnexpectedAffineTensor;
    if (!std.mem.eql(usize, t.shape[0..t.rank], want)) return error.UnexpectedAffineTensor;
    var bytes: usize = dtype.size();
    for (want) |n| {
        if (n == 0) return error.UnexpectedAffineTensor;
        bytes = std.math.mul(usize, bytes, n) catch return error.UnexpectedAffineTensor;
    }
    if (byteLen(t) != bytes) return error.UnexpectedAffineTensor;
}

/// The width the stored shapes carry (bits from the words per row, group from the tables), or the one `want` fixes.
fn width(w: anytype, s: anytype, want: Want) Error![2]usize {
    if (w.rank != s.rank or w.rank < 2 or w.rank > w.shape.len) return error.UnexpectedAffineTensor;
    const words = w.shape[w.rank - 1];
    const groups = s.shape[s.rank - 1];
    const packed_bits = std.math.mul(usize, words, 32) catch return error.UnexpectedAffineTensor;
    if (groups == 0 or want.k == 0 or want.k % groups != 0 or packed_bits % want.k != 0) return error.UnexpectedAffineTensor;
    if (packed_bits / want.k > std.math.maxInt(u8)) return error.UnexpectedAffineTensor;
    return .{ want.bits orelse packed_bits / want.k, want.group orelse want.k / groups };
}

/// Weight, scales and biases checked against each other and `want`: u32 words, bf16 tables, every byte accounted for.
pub fn check(w: anytype, s: @TypeOf(w), z: @TypeOf(w), want: Want) Error!Triple(@TypeOf(w)) {
    const wg = try width(w, s, want);
    const d = try dims(want, wg[0], wg[1]);
    try require(w, .u32, d.words[0..d.rank]);
    try require(s, .bf16, d.tables[0..d.rank]);
    try require(z, .bf16, d.tables[0..d.rank]);
    return .{ .weight = w, .scales = s, .biases = z, .n = want.n, .k = want.k, .bits = @intCast(wg[0]), .group = wg[1] };
}

/// The tensor type `get` returns on a checkpoint (or a pointer to one).
fn TensorOf(comptime C: type) type {
    const T = if (@typeInfo(C) == .pointer) @typeInfo(C).pointer.child else C;
    return @typeInfo(@typeInfo(@TypeOf(T.get)).@"fn".return_type.?).error_union.payload;
}

/// `module`'s three tensors from `c` (anything with `get(name)`), checked by `check`; the log names a refused one.
pub fn read(c: anytype, module: []const u8, want: Want) !Triple(TensorOf(@TypeOf(c))) {
    var buf: [320]u8 = undefined;
    const w = try c.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
    const s = try c.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{module}));
    const z = try c.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{module}));
    return check(w, s, z, want) catch |e| {
        std.log.err("{s}: not an MLX affine [{d}, {d}, {d}] projection", .{ module, want.stack, want.n, want.k });
        return e;
    };
}

test {
    _ = @import("mlx_affine_test.zig");
}
