//! Draft depth by cost without tensor units (0.6.5's DraftDepth for one stream): the most expected tokens a ms.
const std = @import("std");

/// Drafts a round at most: windows up to 16 rows.
const top = 15;

/// Per-depth landing rates and measured round times; head rounds take at most `most` drafts, copied ones up to 15.
pub const CostDepth = struct {
    most: usize = 3, // 0.6.5's --mtp-drafts on a Mac
    p: [top]f64 = .{ 0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 },
    ms: [top + 1]f64 = @splat(0), // a round's milliseconds by drafts (0: not measured yet)
    rounds: usize = 0,

    const rate = 0.15; // the newest round's weight in a depth's landing rate
    const cost_rate = 0.2; // a measured round's weight in its depth's time
    const probe_every = 8; // rounds between probes one deeper (and, halfway, one shallower)

    /// A head round offered `drafts` and kept `landed`: draft j moves toward whether it was kept, while 1..j-1 were.
    pub fn observe(self: *CostDepth, drafts: usize, landed: usize) void {
        for (0..@min(drafts, top)) |j| {
            if (landed < j) break;
            const kept: f64 = if (landed > j) 1 else 0;
            self.p[j] += rate * (kept - self.p[j]);
        }
    }

    /// A round of `drafts` drafts took `ms` milliseconds.
    pub fn time(self: *CostDepth, drafts: usize, ms: f64) void {
        if (drafts > top or !(ms > 0)) return;
        const before = self.ms[drafts];
        self.ms[drafts] = if (before == 0) ms else before + cost_rate * (ms - before);
    }

    /// A round's time at `drafts` as 0.6.5's extend_costs: measured, interpolated, else at the widest's time a row.
    pub fn cost(self: *const CostDepth, drafts: usize) ?f64 {
        if (self.ms[drafts] > 0) return self.ms[drafts];
        var below: ?usize = null;
        var above: ?usize = null;
        for (0..top + 1) |d| if (self.ms[d] > 0) {
            if (d < drafts) below = d else if (above == null) above = d;
        };
        const hi = above orelse {
            const w = below orelse return null;
            return self.ms[w] * @as(f64, @floatFromInt(drafts + 1)) / @as(f64, @floatFromInt(w + 1));
        };
        const lo = below orelse return self.ms[hi];
        const f = @as(f64, @floatFromInt(drafts - lo)) / @as(f64, @floatFromInt(hi - lo));
        return self.ms[lo] + (self.ms[hi] - self.ms[lo]) * f;
    }

    /// The drafts with the most expected tokens a ms, landing as `chance(j)` for draft j, from 1 up to `most`.
    fn best(self: *const CostDepth, most: usize, ctx: anytype, comptime chance: fn (@TypeOf(ctx), usize) f64) usize {
        var top_d: usize = most;
        var pick_rate: f64 = -1;
        var expected: f64 = 1;
        var run: f64 = 1;
        for (1..most + 1) |d| {
            const c = self.cost(d) orelse return most;
            run *= chance(ctx, d - 1);
            expected += run;
            if (expected / c > pick_rate) {
                top_d = d;
                pick_rate = expected / c;
            }
        }
        return top_d;
    }

    fn headChance(self: *const CostDepth, j: usize) f64 {
        return self.p[j];
    }

    fn copyChance(q: f64, _: usize) f64 {
        return q;
    }

    /// The next head round's drafts: unmeasured depths first, then the best, now and then one deeper or shallower.
    pub fn pick(self: *CostDepth) usize {
        self.rounds += 1;
        var d = self.most;
        while (d >= 1) : (d -= 1) if (self.ms[d] == 0) return d;
        const b = self.best(self.most, self, headChance);
        if (b < self.most and self.rounds % probe_every == 0) return b + 1;
        if (b > 1 and self.rounds % probe_every == probe_every / 2) return b - 1;
        return b;
    }

    /// A copied window's drafts: copies land as `q` each; the widest window until any round has a time.
    pub fn copyDrafts(self: *const CostDepth, q: f64) usize {
        return self.best(top, @min(q, 0.99), copyChance);
    }
};

test "depth by cost: unmeasured depths first, then the most expected tokens a ms" {
    var c: CostDepth = .{};
    try std.testing.expectEqual(@as(usize, 3), c.pick());
    c.time(3, 30);
    try std.testing.expectEqual(@as(usize, 2), c.pick());
    c.time(2, 20);
    try std.testing.expectEqual(@as(usize, 1), c.pick());
    c.time(1, 10); // every draft as dear as a row: one draft wins (1.85 a 10 ms round)
    c.rounds = 0;
    try std.testing.expectEqual(@as(usize, 1), c.pick());
    for (1..4) |d| c.time(d, 20); // drafts nearly free: the deepest wins
    for (1..4) |d| c.ms[d] = 20 + @as(f64, @floatFromInt(d));
    c.rounds = 1;
    try std.testing.expectEqual(@as(usize, 3), c.pick());
}

test "landing rates move toward kept drafts; costs extend past the widest by rows" {
    var c: CostDepth = .{};
    c.observe(3, 1); // draft 1 kept, draft 2 not, draft 3 never tried
    try std.testing.expect(c.p[0] > 0.85 and c.p[1] < 0.75 and c.p[2] == 0.7);
    try std.testing.expectEqual(@as(?f64, null), c.cost(2));
    c.time(3, 40);
    try std.testing.expectEqual(@as(?f64, 40), c.cost(1)); // below the only measured: its time
    try std.testing.expectEqual(@as(?f64, 160), c.cost(15)); // 16 rows at 10 ms a row
    c.time(1, 20);
    try std.testing.expectEqual(@as(?f64, 30), c.cost(2));
    try std.testing.expectEqual(@as(usize, 1), c.copyDrafts(0.99)); // every row as dear: the cheapest round wins
    c.ms[3] = 24;
    c.time(15, 60); // wide windows nearly free: copies that land take the widest
    try std.testing.expectEqual(@as(usize, 15), c.copyDrafts(0.99));
    try std.testing.expectEqual(@as(usize, 3), c.copyDrafts(0.6));
}
