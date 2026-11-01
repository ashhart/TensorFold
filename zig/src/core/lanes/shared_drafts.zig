//! Price next-round head rows before proposing, retaining the last full node prices through zero-grant rounds.
const std = @import("std");
const win = @import("windows.zig");
const alloc = @import("allocate.zig");
const Engine = @import("engine.zig").Engine;

pub fn budgets(e: *Engine, plans: []const win.Plan, heads: []const usize, out: []i64) !void {
    const a = e.arena.allocator();
    const fixed = try a.alloc(u32, plans.len);
    const probs = try a.alloc([]const f64, plans.len);
    const neural = try a.alloc(bool, plans.len);
    const limits = try costLimits(e, plans, heads);
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
        const most = @min(cap, e.cfg.most_drafts, if (limits) |l| l[i] else e.cfg.most_drafts, @as(u32, @intCast(@max(0, s.draftRoom() - 1))));
        if (most == 0) continue;
        head.* = true;
        const cached = s.draft_chances.items;
        if (cached.len > 0 and limits == null) {
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
    const counts = try alloc.allocate(a, fixed, probs, &e.cfg.shared_costs, e.rule.overheadMs(@intCast(live), win.anySampled(plans)), @max(live, e.cfg.batch_rows));
    for (heads, out) |i, *budget| {
        budget.* = if (neural[i]) counts[i] else 0;
    }
}

// Copy lookups may advance a proposer's backoff; the reservation loop, not this depth cap, asks it once.
fn noCopy(_: *anyopaque, _: *const @import("depth.zig").Who) bool {
    return false;
}

fn costLimits(e: *Engine, plans: []const win.Plan, heads: []const usize) !?[]const u32 {
    if (!e.cfg.limit_shared_drafts) return null;
    const a = e.arena.allocator();
    const who = try a.alloc(@import("depth.zig").Who, heads.len);
    for (heads, who) |i, *w| w.* = win.who(plans[i].stream);
    const depths = try a.alloc(i64, heads.len);
    try e.rule.budgets(who, .{ .ptr = e, .copy = noCopy }, win.anySampled(plans), depths);
    const limits = try a.alloc(u32, plans.len);
    @memset(limits, e.cfg.most_drafts);
    for (heads, depths) |i, d| limits[i] = @intCast(@max(0, d));
    return limits;
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
