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

/// got within [lo, hi] (finite), or of the float64 point's class where the range is not finite.
fn inRange(got: u16, lo: f64, hi: f64, want: f64, stats: *Stats) bool {
    const g: f64 = bfr.value(got);
    if (std.math.isFinite(lo) and std.math.isFinite(hi)) {
        if (lo == hi) stats.decided += 1 else stats.ranged += 1;
        return std.math.isFinite(g) and lo <= g and g <= hi;
    }
    stats.nonfinite += 1;
    return c.classOf(g) == c.classOf(want);
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
    const b = try a.alloc(f64, 6 * n);
    defer a.free(b);
    const h_lo, const h_hi, const y_lo, const y_hi, const hp, const yp = .{ b[0..n], b[n .. 2 * n], b[2 * n .. 3 * n], b[3 * n .. 4 * n], b[4 * n .. 5 * n], b[5 * n ..] };
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    for (0..k.rows) |row| {
        _ = arena.reset(.retain_capacity);
        const ra = arena.allocator();
        try rowRef(ra, k, row, spread(), h_lo, h_hi, y_lo, y_hi);
        try point(ra, k, row, hp, yp);
        const at = row * n;
        for (0..n) |i| {
            if (k.writesH() and !inRange(h[at + i], h_lo[i], h_hi[i], hp[i], stats))
                return try std.fmt.bufPrint(msg, "row {d} h[{d}] = {e} outside float64's [{e}, {e}]", .{ row, i, bfr.value(h[at + i]), h_lo[i], h_hi[i] });
            if (!inRange(y[at + i], y_lo[i], y_hi[i], yp[i], stats))
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
