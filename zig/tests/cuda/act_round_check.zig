//! MLX's activation rows against core/act_ref.zig's float64 ends, their 64-value sums against the kernel's fp32 tree.
const std = @import("std");
const core = @import("core");

const aref = core.act_ref;
const bfr = core.bf16_round;
const c = core.contract.spec;

/// fp32 tanhf and expf within 2 ulps of their value (2^-22) and the division within half one: twice that, either way.
pub const spread = 0x1p-21;

/// One call's host rows as bf16 bits: gate and up rows at their strides, GeGLU or SwiGLU.
pub const Case = struct { geglu: bool, rows: usize, n: usize, gate: []const u16, g_ld: usize, up: []const u16, u_ld: usize };

/// What a check saw: outputs whose float64 range is one value, a range, non-finite in float64, and sums.
pub const Stats = struct { decided: usize = 0, ranged: usize = 0, nonfinite: usize = 0, sums: usize = 0 };

/// act_rows.cu's group_sum8 over stored outputs: each thread's 8 in order from 0, then 8 lanes' xor tree 4, 2, 1.
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

/// Element i of row `row`: its float64 ends and point.
pub fn want(k: Case, row: usize, i: usize) struct { lo: f64, hi: f64, point: f64 } {
    const x: f64 = bfr.value(k.gate[row * k.g_ld + i]);
    const u: f64 = bfr.value(k.up[row * k.u_ld + i]);
    const r = aref.mlxRange(k.geglu, x, u, spread);
    return .{ .lo = r[0], .hi = r[1], .point = if (k.geglu) aref.geglu(x, u, 0) else aref.swigluRound(x, u, 0, 0) };
}

/// Every output within float64's ends (NaN and infinities as float64's point), the sums bit-equal to the fp32 tree.
pub fn verify(k: Case, out: []const u16, xs: ?[]const f32, stats: *Stats, msg: []u8) !?[]const u8 {
    for (0..k.rows) |row| {
        for (0..k.n) |i| {
            const w = want(k, row, i);
            const g: f64 = bfr.value(out[row * k.n + i]);
            const ok = if (std.math.isFinite(w.lo) and std.math.isFinite(w.hi)) blk: {
                if (w.lo == w.hi) stats.decided += 1 else stats.ranged += 1;
                break :blk std.math.isFinite(g) and w.lo <= g and g <= w.hi;
            } else blk: {
                stats.nonfinite += 1;
                break :blk c.classOf(g) == c.classOf(w.point);
            };
            if (!ok) return try std.fmt.bufPrint(msg, "row {d} out[{d}] = {e} outside float64's [{e}, {e}]", .{ row, i, g, w.lo, w.hi });
        }
        if (xs) |sums| for (0..k.n / 64) |j| {
            const t = groupSum(out[row * k.n + 64 * j ..][0..64]);
            const got = sums[row * (k.n / 64) + j];
            stats.sums += 1;
            if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(t)) and !(std.math.isNan(got) and std.math.isNan(t)))
                return try std.fmt.bufPrint(msg, "row {d} sum {d} = {e} where the fp32 tree of the stored out gives {e}", .{ row, j, got, t });
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
