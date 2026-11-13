//! EXL3 projections for decode rows: the input rotation and the trellis linear (exl3/exl3.hip), mul1 codebook.

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

/// The trellis widths with a linear (half-bits a weight, 2 to 8 bits), and the waves a block splits K over.
pub const widths = [_]u32{ 4, 6, 8, 10, 12, 16 };
pub const wave_counts = [_]u32{ 4, 8 };

pub const Fns = struct {
    rot_in: Function,
    linear: [widths.len][wave_counts.len]Function,

    pub fn load(m: Module) Error!Fns {
        var f: Fns = .{ .rot_in = try m.function("tf_exl3_rot_in"), .linear = undefined };
        inline for (widths, 0..) |k2, i| inline for (wave_counts, 0..) |wk, j| {
            f.linear[i][j] = try m.function(std.fmt.comptimePrint("tf_exl3_linear_k{d}_w{d}", .{ k2, wk }));
        };
        return f;
    }
};

/// The index of a trellis width in `widths`, null for one without a linear.
pub fn widthIndex(halves: u32) ?usize {
    return std.mem.indexOfScalar(u32, &widths, halves);
}

/// Whole 128-wide blocks of K and N and at least one row, each within the kernels' int arguments.
pub fn fits(rows: usize, k: usize, n: usize) bool {
    const most: usize = std.math.maxInt(c_int);
    if (rows == 0 or k == 0 or n == 0 or rows > most or k > most or n > most) return false;
    return k % 128 == 0 and n % 128 == 0;
}

/// CUDA's dense plan without a K split: 8 waves from 64 column blocks when each wave keeps 8 k tiles, else 4.
pub fn waves(k: usize, n: usize) u32 {
    return if (n / 128 >= 64 and k / 16 / 8 >= 8) 8 else 4;
}

/// The linear's LDS for its waves' sums: 128 floats a wave and row, at most 8 rows.
pub fn sharedBytes(wk: u32, rows: usize) u32 {
    return wk * @as(u32, @intCast(@min(rows, 8))) * 128 * 4;
}

/// xh [rows, k] fp16 = (x * suh) H / sqrt(128) by 128-wide blocks, of x [rows, k] (kind 0 fp32, 1 fp16, 2 bf16).
pub fn rotIn(l: *const Launcher, x: u64, x_kind: c_int, suh: u64, xh: u64, rows: usize, k: usize, s: S) Error!void {
    if (!fits(rows, k, 128) or x_kind < 0 or x_kind > 2) return invalid("exl3 rot_in");
    var a: Args = .{};
    try a.add(x);
    try a.add(x_kind);
    try a.add(suh);
    try a.add(xh);
    try a.add(@as(c_int, @intCast(k)));
    try l.go(l.exl3.rot_in, dim(cdiv(k / 128, 4), rows, 1), dim(128, 1, 1), 0, s, &a);
}

/// A matrix's trellis words: `halves` a weight, tile (kt, nt) at kt stride_k + nt / 8 stride_nb + nt % 8 tile words.
pub const Trellis = struct { words: u64, halves: u32, stride_k: u64, stride_nb: u64 };

/// y [rows, n] (kind 0 fp32, 2 bf16) = (xh @ W) H / sqrt(128) * svh by 128-column blocks, W decoded from `t`.
pub fn linear(
    l: *const Launcher,
    xh: u64,
    t: Trellis,
    svh: u64,
    y: u64,
    y_kind: c_int,
    rows: usize,
    k: usize,
    n: usize,
    s: S,
) Error!void {
    const wi = widthIndex(t.halves) orelse return invalid("exl3 linear width");
    if (!fits(rows, k, n) or (y_kind != 0 and y_kind != 2)) return invalid("exl3 linear");
    const most: u64 = std.math.maxInt(c_longlong);
    if (t.stride_k > most or t.stride_nb > most) return invalid("exl3 linear strides");
    const wk = waves(k, n);
    var a: Args = .{};
    try a.add(xh);
    try a.add(t.words);
    try a.add(@as(c_longlong, @intCast(t.stride_k)));
    try a.add(@as(c_longlong, @intCast(t.stride_nb)));
    try a.add(svh);
    try a.add(y);
    try a.add(y_kind);
    try a.add(@as(c_int, @intCast(rows)));
    try a.add(@as(c_int, @intCast(k)));
    try a.add(@as(c_int, @intCast(n)));
    const f = l.exl3.linear[wi][@intFromBool(wk == 8)];
    try l.go(f, dim(n / 128, 1, 1), dim(wk * 32, 1, 1), sharedBytes(wk, rows), s, &a);
}

test "exl3 launches: shapes in whole blocks, CUDA's wave counts, and LDS for at most 8 rows" {
    try std.testing.expect(fits(1, 5120, 129280) and fits(128, 512, 128));
    try std.testing.expect(!fits(0, 512, 128) and !fits(1, 640 - 64, 128) and !fits(1, 512, 192) and !fits(1, 0, 128));
    try std.testing.expect(!fits(@as(usize, std.math.maxInt(c_int)) + 1, 512, 128));
    try std.testing.expectEqual(@as(u32, 8), waves(5120, 129280));
    try std.testing.expectEqual(@as(u32, 8), waves(6144, 25600));
    try std.testing.expectEqual(@as(u32, 4), waves(512, 128));
    try std.testing.expectEqual(@as(u32, 4), waves(512, 129280));
    try std.testing.expectEqual(@as(u32, 4), waves(5120, 63 * 128));
    try std.testing.expectEqual(@as(u32, 8), waves(1024, 64 * 128));
    try std.testing.expectEqual(@as(u32, 4), waves(1024 - 128, 64 * 128));
    try std.testing.expectEqual(@as(u32, 4 * 128 * 4), sharedBytes(4, 1));
    try std.testing.expectEqual(@as(u32, 8 * 8 * 128 * 4), sharedBytes(8, 64));
    try std.testing.expectEqual(@as(?usize, 5), widthIndex(16));
    try std.testing.expectEqual(@as(?usize, null), widthIndex(14));
}
