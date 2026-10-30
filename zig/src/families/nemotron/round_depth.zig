//! A GPU round's head depth, priced by the backend's window-cost table and head step time.
const std = @import("std");
const st = @import("state.zig");
const Costs = @import("copy_lanes.zig").Costs;

pub const Depth = struct {
    levels: usize,
    most: usize,
    fixed: bool,
    cost: [st.max_levels + 1]f64 = @splat(1),
    kept: [8]u32 = @splat(0), // the last rounds' kept tokens at the current depth
    seen: usize = 0,
    missed: [st.max_levels + 1]bool = @splat(false), // levels that landed no draft in a full observation window

    const prior = 0.75;

    pub fn init(most: usize, fixed: bool, costs: Costs, head_ms: f64) Depth {
        const cap = @min(@max(most, 1), st.max_levels);
        var d = Depth{ .levels = cap, .most = cap, .fixed = fixed };
        std.debug.assert(costs.rows.len > 0 and costs.widths.len == costs.rows.len);
        var best: usize = 1;
        var best_rate: f64 = 0;
        var expected: f64 = 1;
        var reach: f64 = 1;
        for (1..cap + 1) |s| {
            d.cost[s] = @max(costs.at(s + 1) + head_ms * @as(f64, @floatFromInt(s)), 1e-6);
            reach *= prior;
            expected += reach;
            const rate = expected / d.cost[s];
            if (rate > best_rate) {
                best = s;
                best_rate = rate;
            }
        }
        if (!fixed) d.levels = best;
        return d;
    }

    pub fn observe(d: *Depth, kept: u32) void {
        d.kept[d.seen % d.kept.len] = kept;
        d.seen += 1;
    }

    pub fn next(d: *Depth) usize {
        if (d.fixed or d.seen < d.kept.len or d.most <= 1) return d.levels;
        const cur = d.levels;
        var best: usize = 1;
        var best_rate: f64 = 0;
        var cur_all: f64 = 0;
        for (1..cur + 1) |s| {
            var all: u32 = 0;
            var landed = false;
            for (d.kept) |k| {
                all += @min(k, @as(u32, @intCast(s + 1)));
                landed = landed or k > s;
            }
            d.missed[s] = !landed;
            const tokens: f64 = @floatFromInt(all);
            if (s == cur) cur_all = tokens;
            const rate = tokens / d.cost[s];
            if (rate >= best_rate) {
                best = s;
                best_rate = rate;
            }
        }
        if (best == cur and cur < d.most and !d.missed[cur + 1]) {
            var long: u32 = 0;
            for (d.kept) |k| long += @intFromBool(k >= cur + 1);
            const p = @as(f64, @floatFromInt(long)) / @as(f64, @floatFromInt(d.kept.len));
            const extra = @as(f64, @floatFromInt(long)) * p;
            if ((cur_all + extra) / d.cost[cur + 1] > best_rate) best = cur + 1;
        }
        if (best != cur) {
            d.levels = best;
            d.seen = 0;
        }
        return d.levels;
    }
};

test "a steep per-row window table starts at 1 level and steps up only when windows saturate" {
    const widths = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const ms = [_]f64{ 19.16, 26.54, 35.00, 42.09, 50.85, 57.40, 66.36, 74.43, 83.78 };
    const costs = Costs{ .rows = &ms, .widths = &widths };
    var d = Depth.init(8, false, costs, 1.47);
    try std.testing.expectEqual(@as(usize, 1), d.levels);
    for ([_]u32{ 2, 2, 1, 2, 2, 2, 1, 2 }) |k| d.observe(k);
    try std.testing.expectEqual(@as(usize, 1), d.next());
    for (0..8) |_| d.observe(2);
    try std.testing.expectEqual(@as(usize, 2), d.next());
    for (0..8) |_| d.observe(2);
    try std.testing.expectEqual(@as(usize, 1), d.next());
}

test "a flat tile window table starts at 4 levels, steps up on full windows, and drops back when deep levels miss" {
    const widths = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const ms = [_]f64{ 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.4, 7.8, 8.3 };
    const costs = Costs{ .rows = &ms, .widths = &widths };
    var d = Depth.init(8, false, costs, 0.45);
    try std.testing.expectEqual(@as(usize, 4), d.levels);
    for (0..8) |_| d.observe(5);
    try std.testing.expectEqual(@as(usize, 5), d.next());
    for (0..8) |_| d.observe(4);
    try std.testing.expect(d.next() <= 4);
    const fixed = Depth.init(8, true, costs, 1.47);
    try std.testing.expectEqual(@as(usize, 8), fixed.levels);
}

test "a draft level that never lands is not repeatedly explored over 160 rounds" {
    const widths = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const ms = [_]f64{ 19.16, 26.54, 35.00, 42.09, 50.85, 57.40, 66.36, 74.43, 83.78 };
    var d = Depth.init(8, false, .{ .rows = &ms, .widths = &widths }, 1.47);
    var wasted: usize = 0;
    for (0..160) |_| {
        const levels = d.next();
        if (levels > 1) wasted += 1;
        d.observe(2);
    }
    try std.testing.expectEqual(@as(usize, 8), wasted);
}

test "productive deeper levels remain available after an earlier shorter window" {
    const widths = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const ms = [_]f64{ 19.16, 26.54, 35.00, 42.09, 50.85, 57.40, 66.36, 74.43, 83.78 };
    var d = Depth.init(8, false, .{ .rows = &ms, .widths = &widths }, 1.47);
    for (0..48) |_| {
        const levels = d.next();
        d.observe(@intCast(levels + 1));
    }
    try std.testing.expect(d.next() > 2);
}
