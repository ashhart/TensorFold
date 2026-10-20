//! A norm tail's outputs against core/norms_ref.zig's float64 ranges, its 64-value sums against the kernel's fp32 tree.
const std = @import("std");
const core = @import("core");

const ref = core.norms_ref;
const bfr = core.bf16_round;
const c = core.contract.spec;

/// The vector norm's sum chain: 32 fma a thread over 16-byte loads, then 256 threads' tree (contract_rows' vecDepth).
pub const depth = 42;

/// An fp32 inverse RMS's relative spread from float64's: the sum's gamma halved by the root, fp32 steps, the products.
pub fn spread() f64 {
    return c.gamma(depth, 0x1p-24) / 2 + 8 * 0x1p-24;
}

/// One call's host inputs as bf16 bits: rows of `width`, optional residual, branch weight and scalar.
pub const Case = struct {
    rows: usize,
    width: usize,
    round: bool,
    x: []const u16,
    w: []const u16,
    r: ?[]const u16 = null,
    wb: ?[]const u16 = null,
    sc: ?u16 = null,

    /// The kernel writes h when the tail has a residual, a branch norm or a scalar.
    pub fn writesH(k: Case) bool {
        return k.r != null or k.wb != null or k.sc != null;
    }
};

/// What a check saw: outputs whose float64 range is one value, a range, and outputs non-finite in float64.
pub const Stats = struct { decided: usize = 0, ranged: usize = 0, nonfinite: usize = 0, sums: usize = 0 };

fn vals(a: std.mem.Allocator, b: []const u16) ![]f64 {
    const o = try a.alloc(f64, b.len);
    for (o, b) |*v, h| v.* = bfr.value(h);
    return o;
}

/// The float64 point reference of a row: h and y (spread 0).
pub fn point(a: std.mem.Allocator, k: Case, row: usize, h: []f64, y: []f64) !void {
    return rowRef(a, k, row, 0, h, try a.alloc(f64, k.width), y, try a.alloc(f64, k.width));
}

/// Row `row` of `k` as core/norms_ref.zig's tail and its float64 x and w, the inverse's spread `spread_`.
pub fn tailOf(a: std.mem.Allocator, k: Case, row: usize, spread_: f64) !struct { t: ref.Tail, x: []f64, w: []f64 } {
    const at = row * k.width;
    return .{
        .t = .{
            .round = k.round,
            .residual = if (k.r) |r| try vals(a, r[at..][0..k.width]) else null,
            .branch = if (k.wb) |wb| try vals(a, wb) else null,
            .scalar = if (k.sc) |s| bfr.value(s) else null,
            .spread = spread_,
        },
        .x = try vals(a, k.x[at..][0..k.width]),
        .w = try vals(a, k.w),
    };
}

fn rowRef(a: std.mem.Allocator, k: Case, row: usize, spread_: f64, h_lo: []f64, h_hi: []f64, y_lo: []f64, y_hi: []f64) !void {
    const t = try tailOf(a, k, row, spread_);
    ref.tail(t.t, t.x, t.w, ref.Exact{}, h_lo, h_hi, y_lo, y_hi);
}

/// got in [lo, hi]: by its bits where the ends are one value, else between ordered ends (-0 below +0); NaN for NaN.
fn inRange(got: u16, lo: f64, hi: f64, stats: *Stats) bool {
    const g: f64 = bfr.value(got);
    if (std.math.isFinite(lo) and std.math.isFinite(hi)) {
        if (sameBits(lo, hi)) stats.decided += 1 else stats.ranged += 1;
    } else stats.nonfinite += 1;
    if (std.math.isNan(lo) or std.math.isNan(hi)) return std.math.isNan(g);
    if (sameBits(lo, hi)) return bfr.value(bfr.bits(@floatCast(lo))) == lo and got == bfr.bits(@floatCast(lo));
    const a, const b = if (atMost(lo, hi)) .{ lo, hi } else .{ hi, lo };
    return atMost(a, g) and atMost(g, b);
}

fn sameBits(a: f64, b: f64) bool {
    return @as(u64, @bitCast(a)) == @as(u64, @bitCast(b));
}

/// a <= b with -0 below +0 (false for NaN).
fn atMost(a: f64, b: f64) bool {
    return a < b or (a == b and (std.math.signbit(a) or !std.math.signbit(b)));
}

/// norms_tail.cuh's group_sum8 over stored outputs: each thread's 8 in order from 0, then 8 lanes' xor tree 4, 2, 1.
pub fn groupSum(y: []const u16) f32 {
    var s: [8]f32 = undefined;
    for (&s, 0..) |*v, l| {
        v.* = 0;
        for (y[8 * l ..][0..8]) |h| v.* += bfr.value(h);
    }
    var m: usize = 4;
    while (m > 0) : (m >>= 1) {
        var t: [8]f32 = undefined;
        for (&t, 0..) |*v, l| v.* = s[l] + s[l ^ m];
        s = t;
    }
    return s[0];
}

/// Every h (when written), y and xs of `k`: the outputs in their float64 ranges, the sums bit-equal to the fp32 tree.
pub fn verify(a: std.mem.Allocator, k: Case, h: []const u16, y: []const u16, xs: ?[]const f32, stats: *Stats, msg: []u8) !?[]const u8 {
    const n = k.width;
    const b = try a.alloc(f64, 4 * n);
    defer a.free(b);
    const h_lo, const h_hi, const y_lo, const y_hi = .{ b[0..n], b[n .. 2 * n], b[2 * n .. 3 * n], b[3 * n ..] };
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    for (0..k.rows) |row| {
        _ = arena.reset(.retain_capacity);
        const ra = arena.allocator();
        try rowRef(ra, k, row, spread(), h_lo, h_hi, y_lo, y_hi);
        const at = row * n;
        for (0..n) |i| {
            if (k.writesH() and !inRange(h[at + i], h_lo[i], h_hi[i], stats))
                return try std.fmt.bufPrint(msg, "row {d} h[{d}] = {e} outside float64's [{e}, {e}]", .{ row, i, bfr.value(h[at + i]), h_lo[i], h_hi[i] });
            if (!inRange(y[at + i], y_lo[i], y_hi[i], stats))
                return try std.fmt.bufPrint(msg, "row {d} y[{d}] = {e} outside float64's [{e}, {e}]", .{ row, i, bfr.value(y[at + i]), y_lo[i], y_hi[i] });
        }
        if (xs) |sums| for (0..n / 64) |g| {
            const want = groupSum(y[at + 64 * g ..][0..64]);
            const got = sums[row * (n / 64) + g];
            stats.sums += 1;
            const same = @as(u32, @bitCast(got)) == @as(u32, @bitCast(want)) or (std.math.isNan(got) and std.math.isNan(want));
            if (!same) return try std.fmt.bufPrint(msg, "row {d} sum {d} = {e} where the fp32 tree of the stored y gives {e}", .{ row, g, got, want });
        };
    }
    return null;
}

test "the fp32 group tree: lanes then the xor tree, not one chain" {
    var y: [64]u16 = undefined;
    for (&y, 0..) |*v, i| v.* = bfr.bits(if (i == 0) 0x1p24 else 1);
    var chain: f32 = 0;
    for (y) |h| chain += bfr.value(h);
    try std.testing.expectEqual(@as(f32, 0x1p24), chain); // one chain loses all 63 ones beside 2^24
    try std.testing.expectEqual(@as(f32, 0x1p24 + 56), groupSum(&y)); // lanes 1 to 7 keep theirs
}

test "a range with one infinite end: a zero below its finite end is refused, the point and the end pass" {
    var x: [64]u16 = undefined;
    var w: [64]u16 = undefined;
    for (&x, &w) |*v, *u| {
        v.* = bfr.bits(1);
        u.* = bfr.bits(1);
    }
    x[62] = bfr.bits(0.625);
    x[63] = bfr.bits(1.0546875);
    w[0] = 0x7f7f; // the largest finite bf16: bf16(x0 inv) of 1 or 1 + 2^-7 within the spread, so y0 in [w0, +Inf]
    const k: Case = .{ .rows = 1, .width = 64, .round = true, .x = &x, .w = &w };
    const a = std.testing.allocator;
    var lo: [64]f64 = undefined;
    var hi: [64]f64 = undefined;
    var scratch: [2][64]f64 = undefined;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    try rowRef(arena.allocator(), k, 0, spread(), &scratch[0], &scratch[1], &lo, &hi);
    try std.testing.expectEqual(@as(f64, bfr.value(0x7f7f)), lo[0]);
    try std.testing.expect(std.math.isPositiveInf(hi[0]));
    var yp: [64]f64 = undefined;
    try point(arena.allocator(), k, 0, &scratch[0], &yp);
    try std.testing.expect(std.math.isFinite(yp[0]));
    var y: [64]u16 = undefined;
    for (&y, yp) |*o, v| o.* = bfr.bits(@floatCast(v));
    var msg: [200]u8 = undefined;
    var stats: Stats = .{};
    try std.testing.expectEqual(@as(?[]const u8, null), try verify(a, k, &y, &y, null, &stats, &msg));
    y[0] = 0x7f80; // +Inf, the range's upper end
    try std.testing.expectEqual(@as(?[]const u8, null), try verify(a, k, &y, &y, null, &stats, &msg));
    y[0] = 0; // the planted zero
    try std.testing.expect(try verify(a, k, &y, &y, null, &stats, &msg) != null);
}

test "range ends: the negative infinite side, signed zeros by their bits, NaN only for NaN" {
    var s: Stats = .{};
    const inf = std.math.inf(f64);
    const big = bfr.value(0x7f7f);
    try std.testing.expect(!inRange(0, -inf, -big, &s)); // a zero above the finite upper end
    try std.testing.expect(inRange(0xff7f, -inf, -big, &s) and inRange(0xff80, -inf, -big, &s));
    try std.testing.expect(!inRange(0x8000, 0, 0, &s)); // -0 where float64 decides +0
    try std.testing.expect(!inRange(0, -0.0, -0.0, &s));
    try std.testing.expect(!inRange(0x8000, 0, 1, &s) and inRange(0, 0, 1, &s)); // -0 below a +0 end
    try std.testing.expect(inRange(0x8000, -0.0, 0, &s) and inRange(0, 0, -0.0, &s));
    try std.testing.expect(inRange(0x7f80, inf, inf, &s) and !inRange(0x7f7f, inf, inf, &s));
    try std.testing.expect(inRange(0x3f80, -inf, inf, &s) and !inRange(0x7fc0, -inf, inf, &s));
    try std.testing.expect(!inRange(0x3f80, std.math.nan(f64), std.math.nan(f64), &s) and inRange(0x7fc0, std.math.nan(f64), std.math.nan(f64), &s));
}
