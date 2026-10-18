//! How deep a draft chain runs (cuda/draft_depth.py): a draft is verified while its chance of landing pays for its row.

const std = @import("std");

pub const max_rows = 32; // the widest verify window a cost table holds

/// Measured ms on this GPU: `verify[r]` for a window of r rows (index 0 unused), `level` for one chain level.
pub const Costs = struct {
    verify: [max_rows + 1]f64 = @splat(0),
    rows: usize = 0,
    level: f64 = 0,

    /// Timed costs, each row's added ms pooled so it never rises with rows inside one regime of `regime` widths.
    pub fn measured(verify: []const f64, level: f64, regime: usize) Costs {
        var sums: [max_rows]f64 = undefined;
        var counts: [max_rows]f64 = undefined;
        var n: usize = 0;
        var first: usize = 0; // the first pool a row may join: a regime's first row keeps its own ms
        for (verify[1 .. verify.len - 1], verify[2..], 2..) |a, b, r| {
            if ((r - 1) % regime == 0) first = n + 1;
            sums[n] = b - a;
            counts[n] = 1;
            n += 1;
            while (n > first + 1 and sums[n - 2] * counts[n - 1] < sums[n - 1] * counts[n - 2]) {
                sums[n - 2] += sums[n - 1];
                counts[n - 2] += counts[n - 1];
                n -= 1;
            }
        }
        var c: Costs = .{ .rows = verify.len - 1, .level = round4(@max(level, 0.0)) };
        c.verify[1] = verify[1];
        var at: usize = 2;
        for (sums[0..n], counts[0..n]) |s, k| {
            const base = c.verify[at - 1];
            var i: usize = 0;
            while (i < @as(usize, @intFromFloat(k))) : (i += 1) {
                c.verify[at] = base + s / k * @as(f64, @floatFromInt(i + 1));
                at += 1;
            }
        }
        for (c.verify[1..at]) |*v| v.* = round4(v.*);
        return c;
    }

    /// The ms the row carrying draft `k` adds to its window.
    pub fn row(c: Costs, k: usize) f64 {
        return @max(c.verify[k + 1] - c.verify[k], 0.0);
    }
};

fn round4(v: f64) f64 {
    return @round(v * 1e4) / 1e4;
}

/// The first draft, then draft k while P_k ** power >= tokens a ms earned x its row's ms (or P_k >= `floor`).
pub const DepthRule = struct {
    costs: Costs,
    most: usize,
    floor: ?f64 = null,
    power: f64 = 1.0,
    weight: f64 = 0.125,
    tokens: f64 = 2.0,
    ms: f64,

    pub fn init(costs: Costs, most: usize, floor: ?f64, power: f64) DepthRule {
        std.debug.assert(most >= 1 and most <= costs.rows - 1);
        return .{ .costs = costs, .most = most, .floor = floor, .power = power, .ms = costs.verify[2] + costs.level };
    }

    /// Tokens a ms the recent rounds earned, by the cost model.
    pub fn rate(r: DepthRule) f64 {
        return r.tokens / r.ms;
    }

    /// Verify draft `k` whose running confidence is `run` (every drafted window verifies its first draft).
    pub fn keep(r: DepthRule, k: usize, run: f64) bool {
        if (k == 1) return true;
        if (r.floor) |f| return run >= f;
        return std.math.pow(f64, run, r.power) >= r.rate() * r.costs.row(k);
    }

    /// Draft level k + 1 only if a certain draft there would pay for its row and its level.
    pub fn more(r: DepthRule, k: usize, run: f64) bool {
        if (k >= r.most) return false;
        if (r.floor) |f| return run >= f;
        return std.math.pow(f64, run, r.power) >= r.rate() * (r.costs.row(k + 1) + r.costs.level);
    }

    /// Follow a round from its modeled cost, never a clock, so every rank of a split model chooses alike.
    pub fn done(r: *DepthRule, tokens: usize, rows: usize, levels: usize) void {
        r.tokens += r.weight * (@as(f64, @floatFromInt(tokens)) - r.tokens);
        r.ms += r.weight * (r.costs.verify[rows] + @as(f64, @floatFromInt(levels)) * r.costs.level - r.ms);
    }
};

test "pooled costs never add more ms a row as rows grow" {
    const v = [_]f64{ 0, 9.85, 12.44, 14.72, 16.92, 18.51, 20.11, 21.7, 23.03, 24.36, 25.69, 26.95, 28.22, 29.33, 30.36, 31.22, 32.08 };
    const c = Costs.measured(&v, 0.6132, 16);
    try std.testing.expectEqual(@as(usize, 16), c.rows);
    for (2..16) |k| try std.testing.expect(c.row(k) <= c.row(k - 1) + 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 32.08), c.verify[16], 1e-9);
    var r = DepthRule.init(c, 15, null, 2.0);
    try std.testing.expect(r.keep(1, 0.0) and r.keep(2, 0.99) and !r.keep(2, 0.1));
    r.done(4, 4, 3);
    try std.testing.expect(r.tokens > 2.0);
}

test "pooling stops at a kernel regime: rows 1 to 16 cost what they cost alone when 17 rows step up" {
    var v: [33]f64 = @splat(0);
    const small = [_]f64{ 46.33, 54.42, 60.47, 66.52, 71.58, 76.64, 81.3, 85.95, 90.52, 94.65, 98.78, 102.91, 107.04, 111.18, 115.31, 117.58 };
    @memcpy(v[1..17], &small);
    v[17] = 122.58;
    for (18..33) |r| v[r] = v[r - 1] + 2.4 + 0.1 * @as(f64, @floatFromInt(r % 3));
    const c = Costs.measured(&v, 3.5, 16);
    const alone = Costs.measured(v[0..17], 3.5, 16);
    try std.testing.expectEqual(@as(usize, 32), c.rows);
    try std.testing.expectEqualSlices(f64, alone.verify[1..17], c.verify[1..17]);
    for ([_]usize{ 16, 17, 32 }) |r| try std.testing.expectApproxEqAbs(v[r], c.verify[r], 1e-9);
    for (19..33) |r| try std.testing.expect(c.row(r - 1) <= c.row(r - 2) + 1e-9);
}
