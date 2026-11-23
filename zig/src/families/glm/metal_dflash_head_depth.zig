//! The MTP head's depth for a GLM stream drafting with it beside DFlash2: main's own rule, so its rounds cost main's.
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");

/// A stream's head rounds: the rule's state, and what its last head round asked and its window verified.
pub const Stream = struct {
    state: ?lanes.depth.State = null,
    asked: ?u32 = null, // the drafts its last round asked of the head (null: that round was not the head's)
    first: bool = false, // that round was the prompt's first: its time carries the prefill-to-decode switch
    held: ?u32 = null, // the drafts the window after it verified
    cost: ?Cost = null, // that round's time, priced after the next depth as main's round loop prices it

    pub fn deinit(s: *Stream, gpa: std.mem.Allocator) void {
        if (s.state) |*x| x.deinit(gpa);
        s.* = .{};
    }
};

/// A head round's time waiting for the rule: main observes it after it has chosen the next round's depth.
pub const Cost = struct { held: u32, ms: f64, first: bool };

/// Main's depth rule, made from main's facts (no drafter loaded): its prior, window costs, head step and probes.
pub const Head = struct {
    gpa: std.mem.Allocator,
    cfg: lanes.Config,
    rule: lanes.depth.Rule,

    pub fn init(gpa: std.mem.Allocator, m: lanes.Model) !*Head {
        const h = try gpa.create(Head);
        errdefer gpa.destroy(h);
        h.gpa = gpa;
        h.cfg = try lanes.Config.init(gpa, m, st.max_rows, st.max_rows - 1);
        const c = &h.cfg;
        h.rule = .{ .gpa = gpa, .k = c.k, .prior = c.depth_prior, .most_drafts = c.most_drafts, .family_costs = &c.family_costs, .shared_costs = &c.shared_costs, .mtp_step_ms = c.mtp_step_ms, .draft_step_ms = c.draft_step_ms, .plain_guard = c.plain_guard, .node_probabilities = false, .batch_rows = c.batch_rows, .batched_heads = c.draft_streams };
        return h;
    }

    pub fn deinit(h: *Head) void {
        h.rule.deinit();
        h.cfg.deinit(h.gpa);
        h.gpa.destroy(h);
    }

    fn who(s: *Stream, room: i64) lanes.depth.Who {
        return .{ .state = &s.state, .draft_room = room, .finished = false, .forced = false };
    }

    /// The last head round's outcome: `accepted` of its window's drafts kept now, its `ms` once the next depth is set.
    pub fn observe(h: *Head, s: *Stream, accepted: u32, ms: f64) !void {
        try h.price(s);
        const asked = s.asked orelse return;
        const held = s.held orelse return;
        s.asked = null;
        s.held = null;
        if (held > asked or accepted > held) return; // a copy window: the core prices copies, not the head
        if (held > 0) try h.rule.observeDepth(who(s, st.max_rows), held, accepted);
        if (ms > 0) s.cost = .{ .held = held, .ms = ms, .first = s.first };
    }

    /// The waiting round time goes into the rule (main's order: conclude, next depth, then the round's cost).
    pub fn price(h: *Head, s: *Stream) !void {
        const c = s.cost orelse return;
        s.cost = null;
        try h.rule.observeCost(c.held, c.ms, c.first, who(s, st.max_rows));
    }

    /// The head's drafts this round: main's priced depth, within what the core asked (`asked` 0: none).
    pub fn depth(h: *Head, s: *Stream, asked: u32, first: bool) !u32 {
        if (asked == 0) return 0;
        // the core asks min(7, room - 1), or one with a copy ahead: only 2..6 says the cache's room binds
        const room: i64 = if (asked >= 2 and asked < 7) @as(i64, asked) + 1 else st.max_rows;
        const d: u32 = @intCast(@max(0, try h.rule.depth(who(s, room))));
        try h.price(s);
        const out = @min(asked, d);
        s.asked = out;
        s.first = first;
        s.held = null;
        return out;
    }

    pub fn observeShared(h: *Head, s: *Stream, accepted: u32) !void {
        try h.price(s);
        const asked = s.asked orelse return;
        const held = s.held orelse return;
        s.asked = null;
        s.held = null;
        if (held > asked or accepted > held) return;
        if (held > 0) try h.rule.observeDepth(who(s, st.max_rows), held, accepted);
    }

    fn noCopy(_: *anyopaque, _: *const lanes.depth.Who) bool {
        return false;
    }

    pub fn sharedDepths(h: *Head, streams: []const *Stream, asks: []const u32, firsts: []const bool, sampled: bool, total_streams: usize, out: []u32) !void {
        const n = streams.len;
        if (n == 0 or n > st.max_rows or asks.len != n or firsts.len != n or out.len != n) return error.BadDraftOptions;
        var whos: [st.max_rows]lanes.depth.Who = undefined;
        var raw: [st.max_rows]i64 = undefined;
        for (streams, asks, 0..) |s, asked, i| {
            const room: i64 = if (asked >= 2 and asked < 7) @as(i64, asked) + 1 else st.max_rows;
            whos[i] = .{ .state = &s.state, .draft_room = room, .finished = false, .forced = false, .sampled = sampled };
        }
        const probe: lanes.depth.Probe = .{ .ptr = @ptrCast(h), .copy = noCopy };
        try h.rule.budgetsFor(whos[0..n], probe, sampled, raw[0..n], total_streams);
        for (streams, asks, firsts, raw[0..n], out) |s, asked, first, d, *o| {
            try h.price(s);
            const picked: u32 = if (asked == 0) 0 else @min(asked, @as(u32, @intCast(@max(0, d))));
            s.asked = picked;
            s.first = first;
            s.held = null;
            o.* = picked;
        }
    }
};

fn testModel(costs: []const lanes.config.Cost) lanes.Model {
    return .{ .exact_width = st.max_rows, .mtp = true, .speculate = true, .plain_guard = true, .drafts = 4, .window_costs = costs, .mtp_step_ms = 1.3, .depth_rate = 0.08, .hidden_rows = true, .batch_rows = st.max_rows };
}

fn testCosts() [st.max_rows]lanes.config.Cost {
    var c: [st.max_rows]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = @intCast(i + 1), .ms = 13.0 + 4.1 * @as(f64, @floatFromInt(i)) };
    return c;
}

test "the head's depth is main's: the same rule, facts and order (a round priced after the next depth) round by round" {
    const gpa = std.testing.allocator;
    const costs = testCosts();
    const h = try Head.init(gpa, testModel(&costs));
    defer h.deinit();
    var main_cfg = try lanes.Config.init(gpa, testModel(&costs), st.max_rows, st.max_rows - 1);
    defer main_cfg.deinit(gpa);
    var main_rule: lanes.depth.Rule = .{ .gpa = gpa, .k = main_cfg.k, .prior = main_cfg.depth_prior, .most_drafts = main_cfg.most_drafts, .family_costs = &main_cfg.family_costs, .shared_costs = &main_cfg.shared_costs, .mtp_step_ms = main_cfg.mtp_step_ms, .plain_guard = true, .node_probabilities = false, .batch_rows = st.max_rows };
    defer main_rule.deinit();
    var s: Stream = .{};
    defer s.deinit(gpa);
    var main_state: ?lanes.depth.State = null;
    defer if (main_state) |*x| x.deinit(gpa);
    const mw: lanes.depth.Who = .{ .state = &main_state, .draft_room = 8, .finished = false, .forced = false };
    var prng = std.Random.DefaultPrng.init(7);
    var last: Cost = undefined;
    for (0..200) |i| {
        const d = try h.depth(&s, 7, i == 0);
        const want = try main_rule.depth(mw);
        try std.testing.expectEqual(@as(u32, @intCast(want)), d);
        try std.testing.expect(d <= 4); // main's facts hold four at most, not the drafter's seven
        const kept: u32 = @min(d, prng.random().uintLessThan(u32, 4));
        const ms = 13.0 + 5.4 * @as(f64, @floatFromInt(d)) + 8 * prng.random().float(f64); // jittered timing
        if (i > 0) try main_rule.observeCost(last.held, last.ms, last.first, mw); // main prices next depth
        s.held = d;
        try h.observe(&s, kept, ms);
        if (d > 0) try main_rule.observeDepth(mw, d, kept);
        last = .{ .held = d, .ms = ms, .first = i == 0 };
    }
}

test "the core's ask bounds the head: none when it asks none, one when a copy is ahead" {
    const gpa = std.testing.allocator;
    const costs = testCosts();
    const h = try Head.init(gpa, testModel(&costs));
    defer h.deinit();
    var s: Stream = .{};
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 0), try h.depth(&s, 0, false));
    try std.testing.expect(try h.depth(&s, 1, false) <= 1);
    try std.testing.expect(try h.depth(&s, 7, false) >= 1); // the prior pays for a draft at these costs
}

test "a window holding more than the head asked is a copy: it moves neither acceptance nor cost" {
    const gpa = std.testing.allocator;
    const costs = testCosts();
    const h = try Head.init(gpa, testModel(&costs));
    defer h.deinit();
    var s: Stream = .{};
    defer s.deinit(gpa);
    _ = try h.depth(&s, 7, false);
    const before = try gpa.dupe(f64, s.state.?.p);
    defer gpa.free(before);
    s.held = 9;
    try h.observe(&s, 9, 60);
    try std.testing.expectEqualSlices(f64, before, s.state.?.p);
    if (s.state.?.ms) |t| try std.testing.expect(t.get(9) == null);
    try std.testing.expectEqual(@as(?u32, null), s.asked);
}

test "sharedDepths budgets across streams and observeShared updates acceptance without stream cost" {
    const gpa = std.testing.allocator;
    const costs = testCosts();
    const h = try Head.init(gpa, testModel(&costs));
    defer h.deinit();
    var s0: Stream = .{};
    var s1: Stream = .{};
    defer s0.deinit(gpa);
    defer s1.deinit(gpa);
    var out: [2]u32 = undefined;
    try h.sharedDepths(&.{ &s0, &s1 }, &.{ 7, 3 }, &.{ true, false }, false, 3, &out);
    try std.testing.expect(out[0] <= 4 and out[1] <= 3);
    s0.held = out[0];
    s1.held = out[1];
    const p0 = try gpa.dupe(f64, s0.state.?.p);
    defer gpa.free(p0);
    try h.observeShared(&s0, @min(out[0], 1));
    try h.observeShared(&s1, out[1]);
    try std.testing.expect(s0.cost == null and s0.asked == null and s0.held == null);
    if (out[0] > 0) try std.testing.expect(!std.mem.eql(f64, p0, s0.state.?.p));
}
