//! Price next-round head rows before proposing, retaining the last full node prices through zero-grant rounds.
const std = @import("std");
const win = @import("windows.zig");
const alloc = @import("allocate.zig");
const Engine = @import("engine.zig").Engine;
const depth = @import("depth.zig");

pub fn budgets(e: *Engine, plans: []const win.Plan, heads: []const usize, out: []i64) !void {
    const a = e.arena.allocator();
    const fixed = try a.alloc(u32, plans.len);
    const probs = try a.alloc([]const f64, plans.len);
    const neural = try a.alloc(bool, plans.len);
    const sampled = win.anySampled(plans);
    const limits = try costLimits(e, plans, heads, sampled, out);
    var live: usize = 0;
    for (plans, fixed, probs, neural, 0..) |p, *fx, *pr, *head, i| {
        const s = p.stream;
        fx.* = @intFromBool(!s.finished and !s.paused);
        pr.* = &.{};
        head.* = false;
        if (s.finished or s.paused) continue;
        live += 1;
        if (s.force.items.len > 0) {
            fx.* += if (s.drafts) @as(u32, @intCast(@min(e.cfg.base_width - 1, s.force.items.len))) else 0;
            continue;
        }
        const copy = try win.copyProposal(e, s);
        if (copy.tokens.len > 0) {
            const rate = if (e.cfg.own_copy_rates) win.copiedRate(e, s, copy.recalled) else e.cfg.copy_rate;
            pr.* = try alloc.chainProbabilities(a, &.{rate}, copy.tokens.len);
            continue;
        }
        if (!s.drafts) continue;
        const cap = if (e.backend.vtable.draft_capacity) |f| f(e.backend.ptr, s) else e.cfg.most_drafts;
        const most = @min(cap, e.cfg.most_drafts, if (limits) |l| l.depths[i] else e.cfg.most_drafts, @as(u32, @intCast(@max(0, s.draftRoom() - 1))));
        if (most == 0) continue;
        head.* = true;
        const cached = s.draft_chances.items;
        // a legacy cap prices by the rule's rates; granted depths keep the drafter's own chances under their bound
        if (cached.len > 0 and (limits == null or e.cfg.granted_depth)) {
            const prices = try a.alloc(f64, most);
            const known = @min(cached.len, most);
            @memcpy(prices[0..known], cached[0..known]);
            const rates = try e.rule.rates(win.who(s));
            var chance = cached[known - 1];
            for (known..most) |j| {
                chance *= rates[@min(j, rates.len - 1)];
                prices[j] = chance;
            }
            pr.* = prices;
        } else {
            pr.* = try alloc.chainProbabilities(a, try e.rule.rates(win.who(s)), most);
        }
    }
    if (live == 0) {
        @memset(out, 0);
        return;
    }
    // Copies add verify rows, not head work; granted depths already pay in payReserved.
    const rich = e.cfg.k.hold_costs.len > 0 and !e.cfg.granted_depth;
    const head_cost: alloc.HeadCost = if (rich) .{ .hold_costs = e.cfg.k.hold_costs, .hold_rows = e.cfg.k.hold_rows, .batched = e.cfg.draft_streams and depth.pricedBatched(), .charged = neural } else .{};
    const overhead = if (rich) e.rule.fixedMs(@intCast(live), sampled) else e.rule.overheadMs(@intCast(live), sampled);
    const counts = try alloc.allocateDrafts(a, fixed, probs, &e.cfg.shared_costs, overhead, @max(live, e.cfg.batch_rows), head_cost);
    for (heads, out) |i, *budget| {
        budget.* = if (neural[i]) counts[i] else 0;
    }
    if (limits) |l| if (e.cfg.granted_depth) {
        var base = @import("hold_price.zig").Base.pending(live);
        for (fixed, probs, neural, counts) |rows, chances, head, count| {
            if (rows == 0) continue;
            base.rows += rows - 1;
            base.tokens += @floatFromInt(rows - 1);
            if (head) continue;
            base.rows += count;
            for (chances[0..@min(count, chances.len)]) |chance| base.tokens += chance;
        }
        try payReserved(e, l.who, probs, heads, sampled, out, base);
    };
}

// Copy lookups may advance a proposer's backoff; the reservation loop, not this depth cap, asks it once.
fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

const Limits = struct {
    depths: []const u32, // by plan: the most drafts the rule lets a head stream reserve
    who: []depth.Who, // by head: its last verified grant, repriced by payReserved to the rows reserved now
};

/// What the depth rule lets each head stream reserve; `out` is its scratch, `budgets` overwrites it.
fn costLimits(e: *Engine, plans: []const win.Plan, heads: []const usize, sampled: bool, out: []i64) !?Limits {
    if (!e.cfg.granted_depth and !e.cfg.limit_shared_drafts) return null;
    const a = e.arena.allocator();
    const who = try a.alloc(depth.Who, heads.len);
    for (heads, who) |i, *w| w.* = win.who(plans[i].stream);
    const probe: depth.Probe = .{ .ptr = e, .copy = noCopy };
    if (e.cfg.granted_depth) {
        for (who, out) |*w, *d| d.* = try e.rule.reserveDepth(w, null, probe); // each stream's grant plus slack, once
    } else try e.rule.budgets(who, probe, sampled, out);
    const limits = try a.alloc(u32, plans.len);
    @memset(limits, e.cfg.most_drafts);
    for (heads, out) |i, d| limits[i] = @intCast(@max(0, d));
    return .{ .depths = limits, .who = who };
}

/// Price only reserved holds; a historical zero-grant probe survives only if it received fresh rows.
fn payReserved(e: *Engine, who: []depth.Who, probs: []const []const f64, heads: []const usize, sampled: bool, out: []i64, base: @import("hold_price.zig").Base) !void {
    depth.Rule.capPass(out, e.rule.batch_rows);
    if (e.rule.k.hold_costs.len == 0) return;
    for (who, heads, out) |*w, i, d| {
        if (d <= 0) continue;
        w.prices = probs[i][0..@min(@as(usize, @intCast(d)), probs[i].len)];
        if (w.granted == @as(?u32, 0)) continue; // only a freshly funded historical-zero probe survives
        w.granted = @intCast(d); // price every row reserved now, not only the previous verified grant
        var gain: f64 = 0;
        for (w.prices) |chance| gain += chance;
        w.gain = gain;
    }
    try e.rule.payHolds(who, sampled, out, base);
}

test "shared draft prices: a short cached tree can grow when allocation returns more rows" {
    const a = std.testing.allocator;
    const Cfg = @import("config.zig");
    var costs: [32]Cfg.Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 1 };
    var cfg = try Cfg.Config.init(a, .{ .exact_width = 32, .drafts = 15, .window_costs = &costs, .batch_rows = 32 }, 32, 15);
    defer cfg.deinit(a);
    var fake: @import("fake.zig").Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: @import("fake.zig").FixedClock = .{};
    var e = Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer e.deinit();
    var s = try @import("stream.zig").Stream.init(a, .{ .id = "growth", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer s.deinit(a);
    try s.draft_chances.appendSlice(a, &.{ 0.8, 0.5 });
    const p: win.Plan = .{ .stream = &s, .position = 0, .kind = .none };
    var out: [1]i64 = undefined;
    try budgets(&e, &.{p}, &.{0}, &out);
    try std.testing.expectEqual(@as(i64, 15), out[0]);
}

test "shared draft prices: plain forced streams cost one row and paused streams release their rows" {
    const a = std.testing.allocator;
    const Cfg = @import("config.zig");
    var costs: [8]Cfg.Cost = undefined;
    for (&costs, 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 1 };
    var cfg = try Cfg.Config.init(a, .{ .exact_width = 8, .drafts = 7, .window_costs = &costs, .batch_rows = 8 }, 8, 7);
    defer cfg.deinit(a);
    var fake: @import("fake.zig").Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: @import("fake.zig").FixedClock = .{};
    var e = Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer e.deinit();
    const S = @import("stream.zig").Stream;
    var head = try S.init(a, .{ .id = "head", .prompt = &.{ 7, 13, 29 }, .max_new = 64 });
    defer head.deinit(a);
    var plain = try S.init(a, .{ .id = "plain", .prompt = &.{ 7, 13, 29 }, .max_new = 64, .drafts = false });
    defer plain.deinit(a);
    try head.draft_chances.appendNTimes(a, 1, 7);
    try plain.force.appendNTimes(a, 11, 5);
    const plans = [_]win.Plan{ .{ .stream = &head, .position = 0, .kind = .none }, .{ .stream = &plain, .position = 0, .kind = .forced } };
    var out: [1]i64 = undefined;
    try budgets(&e, &plans, &.{0}, &out);
    try std.testing.expectEqual(@as(i64, 6), out[0]);
    plain.paused = true;
    try budgets(&e, &plans, &.{0}, &out);
    try std.testing.expectEqual(@as(i64, 7), out[0]);
}
