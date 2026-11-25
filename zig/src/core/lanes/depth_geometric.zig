//! Geometric draft depth (off unless a family sets Config.geometric_depth): one acceptance rate, one draft pass.
const std = @import("std");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const Rule = depth.Rule;
const State = depth.State;
const Who = depth.Who;

/// The stream's acceptance moves toward kept / tried: the drafts kept, and the first miss when one was tried.
pub fn observe(st: *State, proposed: i64, accepted: i64) void {
    const got = @min(@max(accepted, 0), proposed);
    const tried = got + @intFromBool(got < proposed);
    st.alpha = 0.7 * st.alpha + 0.3 * (@as(f64, @floatFromInt(got)) / @as(f64, @floatFromInt(tried)));
}

/// The depth (0 .. most) with the most expected tokens (1 + a + a^2 + ...) a ms; at 0, one draft every 8 rounds.
pub fn depthOf(r: *Rule, who: Who, alpha: f64, most: i64) !i64 {
    var best: i64 = 0;
    var best_rate: f64 = -1.0;
    var expected: f64 = 0.0;
    var run: f64 = 1.0;
    var d: i64 = 0;
    while (d <= most) : (d += 1) {
        expected += run;
        run *= alpha;
        const base = (try r.roundCost(d, who)) orelse break;
        if (base <= 0.0) continue;
        const past = @max(0.0, @as(f64, @floatFromInt(who.position)) - r.geo_ctx_base) / 1000.0;
        const ms = base + past * (r.geo_ctx[0] + r.geo_ctx[1] * @as(f64, @floatFromInt(d)));
        if (expected / ms > best_rate) {
            best = d;
            best_rate = expected / ms;
        }
    }
    const st = &who.state.*.?;
    st.rounds += 1;
    return if (best == 0 and st.rounds % 8 == 0) @min(1, most) else best;
}

/// The family's cost table at `rows`, past its end extended by its last step.
pub fn cost(tab: []const f64, rows: usize) f64 {
    if (rows <= tab.len) return tab[rows - 1];
    const step = if (tab.len > 1) tab[tab.len - 1] - tab[tab.len - 2] else 0.0;
    return tab[tab.len - 1] + @as(f64, @floatFromInt(rows - tab.len)) * step;
}

test "geometric depth: the best rate for one acceptance, plain when drafts miss, shallower deep in context" {
    const gpa = std.testing.allocator;
    var fc: Table = .{};
    defer fc.deinit(gpa);
    for (1..5) |w| try fc.put(gpa, @intCast(w), 8.0 + 2.0 * @as(f64, @floatFromInt(w)));
    const shared: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.9, 0.9, 0.9 }, .most_drafts = 3, .family_costs = &fc, .shared_costs = &shared, .mtp_step_ms = 2.0, .plain_guard = false, .node_probabilities = false, .batch_rows = 8, .geometric = true };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    var who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try std.testing.expectEqual(@as(i64, 2), try rule.depth(who)); // alpha 0.6: 1.96 tokens in 16 ms
    rule.geo_ctx = .{ 1.0, 1.0 };
    rule.geo_ctx_base = 0;
    who.position = 10000;
    try std.testing.expectEqual(@as(i64, 0), try rule.depth(who)); // 10 ms more a round and a draft at 10,000
    who.position = 0;
    for (0..10) |_| try rule.observeDepth(who, 2, 0);
    try std.testing.expect(st.?.alpha < 0.1);
    var depths: [8]i64 = undefined;
    for (&depths) |*d| d.* = try rule.depth(who);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 0, 0, 0, 1, 0, 0 }, &depths); // one probe every 8 rounds
}
