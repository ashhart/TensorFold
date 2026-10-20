//! qk_rows' tf_qk_rows_once and tf_qk_rows_round against core/qk_ref.zig's float64 ranges at their rounding points.
const std = @import("std");
const core = @import("core");

const ref = core.qk_ref;
const bfr = core.bf16_round;
const c = core.contract.spec;

/// cosf and sinf within 2 ulps of their value either way.
pub const trig = 0x1p-22;

/// The sum of squares' chain: a warp's xor tree (5), then the head's warps in order.
pub fn depth(dim: usize) usize {
    return 5 + dim / 32;
}

/// An fp32 inverse RMS's relative spread from float64's: the sum's gamma halved by the root, four fp32 steps.
pub fn spread(dim: usize) f64 {
    return c.gamma(@floatFromInt(depth(dim)), 0x1p-24) / 2 + 4 * 0x1p-24;
}

/// One call's host inputs: rows of h query, hk key and (v_from_k: none, k's columns read again) hk value heads.
pub const Case = struct {
    rows: usize,
    h: usize,
    hk: usize,
    dim: usize,
    rot: usize,
    /// bf16 bits [rows, ld]: q heads, then k heads, then (unless v_from_k) v heads.
    x: []const u16,
    ld: usize,
    v_from_k: bool = false,
    /// Norm weights' values (null: no norm), cos and sin [rows, rot] as served (null: no RoPE).
    wq: ?[]const f64 = null,
    wk: ?[]const f64 = null,
    cos: ?[]const f64 = null,
    sin: ?[]const f64 = null,
    /// cos and sin from cosf and sinf (inv_freq), not tables.
    computed: bool = false,
    points: ref.Points = .{},
    v_norm: bool = false,
    eps: f64 = 1e-6,

    pub fn heads(k: Case) usize {
        return k.h + 2 * k.hk;
    }

    /// Head hd's (all q, then k, then v) first input of row `row`.
    pub fn src(k: Case, row: usize, hd: usize) usize {
        const at = if (k.v_from_k and hd >= k.h + k.hk) hd - k.hk else hd;
        return row * k.ld + at * k.dim;
    }
};

/// What a check saw: outputs whose float64 range is one value, a range, and outputs non-finite in float64.
pub const Stats = struct { decided: usize = 0, ranged: usize = 0, nonfinite: usize = 0 };

/// Head hd of row `row` as float64 ends under `st` from its sum of squares `ss` (null: float64's).
pub fn head(k: Case, row: usize, hd: usize, st: ref.Stray, ss: ?f64, x: []f64, lo: []f64, hi: []f64) void {
    for (x, 0..) |*v, i| v.* = bfr.value(k.x[k.src(row, hd) + i]);
    const sq = ss orelse ref.squares(x);
    if (hd >= k.h + k.hk) {
        if (k.v_norm) ref.unitRange(x, sq, k.eps, st, lo, hi) else {
            @memcpy(lo, x);
            @memcpy(hi, x);
        }
        return;
    }
    const w = if (hd < k.h) k.wq else k.wk;
    const cs = if (k.cos) |t| t[row * k.rot ..][0..k.rot] else null;
    const sn = if (k.sin) |t| t[row * k.rot ..][0..k.rot] else null;
    ref.headRange(x, w, sq, k.eps, cs, sn, k.rot, k.points, st, lo, hi);
}

/// The output index of head hd's first value: q [rows, h, dim], then k and v [rows, hk, dim].
pub fn outAt(k: Case, row: usize, hd: usize) usize {
    if (hd < k.h) return (row * k.h + hd) * k.dim;
    const base = k.rows * k.h * k.dim;
    if (hd < k.h + k.hk) return base + (row * k.hk + hd - k.h) * k.dim;
    return base + k.rows * k.hk * k.dim + (row * k.hk + hd - k.h - k.hk) * k.dim;
}

/// The fp32 run's stray for the case's kernel.
pub fn stray(k: Case) ref.Stray {
    return .{ .spread = spread(k.dim), .u = 0x1p-24, .trig = if (k.computed) trig else 0 };
}

/// Every output (q, k, v as outAt lays them) within float64's ends, or of the float64 point's class where not finite.
pub fn verify(a: std.mem.Allocator, k: Case, out: []const u16, stats: *Stats, msg: []u8) !?[]const u8 {
    const buf = try a.alloc(f64, 5 * k.dim);
    defer a.free(buf);
    const x, const lo, const hi, const plo, const phi = .{ buf[0..k.dim], buf[k.dim .. 2 * k.dim], buf[2 * k.dim .. 3 * k.dim], buf[3 * k.dim .. 4 * k.dim], buf[4 * k.dim ..] };
    for (0..k.rows) |row| for (0..k.heads()) |hd| {
        head(k, row, hd, stray(k), null, x, lo, hi);
        head(k, row, hd, .{}, null, x, plo, phi);
        const at = outAt(k, row, hd);
        for (0..k.dim) |i| {
            const g: f64 = bfr.value(out[at + i]);
            const ok = if (std.math.isFinite(lo[i]) and std.math.isFinite(hi[i])) blk: {
                if (lo[i] == hi[i]) stats.decided += 1 else stats.ranged += 1;
                break :blk std.math.isFinite(g) and lo[i] <= g and g <= hi[i];
            } else blk: {
                stats.nonfinite += 1;
                break :blk c.classOf(g) == c.classOf(plo[i]);
            };
            if (!ok) return try std.fmt.bufPrint(msg, "row {d} head {d} out[{d}] = {e} outside float64's [{e}, {e}]", .{ row, hd, i, g, lo[i], hi[i] });
        }
    };
    return null;
}

test "the inverse's spread grows with the head's warps and stays far below a bf16 ulp" {
    try std.testing.expect(spread(512) > spread(256));
    try std.testing.expect(spread(512) < 0x1p-18);
}
