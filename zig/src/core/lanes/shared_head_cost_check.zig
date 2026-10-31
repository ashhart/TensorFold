//! Host grants with real family facts and controlled head, copy and forward prices.
const std = @import("std");
const lanes = @import("lanes.zig");
const win = lanes.windows;
const a = std.testing.allocator;
const t = std.testing;
const code = [_]u32{ 112, 114, 105, 110, 116, 40, 105, 41, 10, 32, 32, 32, 32, 105, 43 };
const Copy = struct {
    fn propose(ptr: *anyopaque, _: []const u32, most: i64) ![]const u32 {
        const calls: *usize = @ptrCast(@alignCast(ptr));
        calls.* += 1;
        return code[0..@intCast(@min(most, code.len))];
    }
    fn match(_: *anyopaque) i64 {
        return 32;
    }
};
const Result = struct { calls: usize = 0, depths: [16]i64 = @splat(0), grants: [16]usize = @splat(0) };

pub fn attached(target: lanes.Model, facts: lanes.drafter.Facts) !lanes.Model {
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var head: @import("drafted_test.zig").FakeDrafter = .{ .facts_ = facts };
    defer head.deinit();
    var wrapper = try lanes.drafted.Drafted.init(a, fake.backend(), head.drafter());
    defer wrapper.deinit();
    return wrapper.facts(target);
}

fn replay(model: lanes.Model, shared: bool, stale: bool) !Result {
    var m = model;
    const n = @min(16, m.max_streams, m.batch_rows / 2);
    try t.expect(n >= 2);
    const rows = n + @min(8, m.batch_rows - n);
    const costs = try a.alloc(lanes.config.Cost, m.batch_rows);
    defer a.free(costs);
    for (costs, 1..) |*c, row| c.* = .{ .width = @intCast(row), .ms = if (row <= rows) 5 + 0.2 * @as(f64, @floatFromInt(row)) else 1000 };
    m.window_costs = costs;
    m.shared_costs = costs;
    var cfg = try lanes.Config.init(a, m, m.exact_width, m.drafts);
    defer cfg.deinit(a);
    cfg.timed = false;
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    e.alone = false;
    var streams: [16]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var current: [16]win.Plan = undefined;
    var who: [16]lanes.depth.Who = undefined;
    var heads: [16]usize = undefined;
    var got: Result = .{};
    for (streams[0..n], current[0..n], who[0..n], heads[0..n], 0..) |*s, *p, *w, *h, i| {
        s.* = try lanes.Stream.init(a, .{ .id = "code", .prompt = &.{ 1, 2, 3 }, .max_new = 128 });
        made += 1;
        @memset(try e.rule.rates(win.who(s)), if (i == 0) @as(f64, 0.99) else 0.01);
        if (stale and i == 0) try s.draft_chances.appendNTimes(a, 0.1, cfg.most_drafts);
        if (i == 1) s.proposer = .{ .ptr = &got.calls, .vtable = &.{ .propose = Copy.propose, .last_match = Copy.match, .priced = true } };
        p.* = .{ .stream = s, .position = 0, .kind = .none };
        w.* = win.who(s);
        h.* = i;
    }
    if (shared) try lanes.shared_drafts.budgets(&e, current[0..n], heads[0..n], got.depths[0..n]) else try e.rule.budgets(who[0..n], win.probe(&e), false, got.depths[0..n]);
    for (current[0..n], got.depths[0..n], 0..) |*p, depth, i| {
        p.kind = if (i == 1) .copy else if (depth > 0) .head else .none;
        p.held = if (i == 1) 0 else @intCast(depth);
        p.tokens = if (i == 1) &code else &.{};
    }
    try win.allocate(&e, current[0..n]);
    for (current[0..n], got.grants[0..n]) |p, *g| g.* = p.count();
    return got;
}

pub fn check(model: lanes.Model) !void {
    try t.expect(model.allocate_shared_drafts);
    try t.expect(!model.draft_probabilities and model.draft_confidence == null);
    for ([_]f64{ 2, 0 }) |ms| for ([_]bool{ false, true }) |stale| {
        var m = model;
        m.mtp_step_ms = ms;
        const old = try replay(m, false, stale);
        const next = try replay(m, true, stale);
        try t.expect(old.calls <= 1);
        try t.expectEqual(@as(usize, 1), next.calls);
        const hot: usize = if (ms == 0) @min(8, m.drafts) else @intFromBool(!m.plain_guard);
        try t.expectEqual(hot, old.grants[0]);
        try t.expectEqual(8 - hot, old.grants[1]);
        try t.expectEqualSlices(usize, &old.grants, &next.grants);
        for (old.depths, next.depths) |was, now| try t.expect(now <= was);
        if (ms > 0) {
            var uncapped = m;
            uncapped.limit_shared_drafts = false;
            const lost = try replay(uncapped, true, false);
            try t.expect(lost.grants[0] > old.grants[0]);
            try t.expect(lost.grants[1] < old.grants[1]);
        }
    };
}

test "shared head costs: attached node prices and confidence floors retain their existing rule" {
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var head: @import("drafted_test.zig").FakeDrafter = .{};
    defer head.deinit();
    var vtable = head.drafter().vtable.*;
    vtable.chances = struct {
        fn chances(_: *anyopaque, _: *lanes.Stream, _: []f64) !bool {
            return false;
        }
    }.chances;
    var wrapper = try lanes.drafted.Drafted.init(a, fake.backend(), .{ .ptr = &head, .vtable = &vtable });
    defer wrapper.deinit();
    const target: lanes.Model = .{ .exact_width = 16, .limit_shared_drafts = true };
    try t.expect(wrapper.facts(target).limit_shared_drafts);
    head.facts_.tree_nodes = 8;
    try t.expect(!wrapper.facts(target).limit_shared_drafts);
    head.facts_.tree_nodes = 0;
    var confidence = target;
    confidence.draft_confidence = 0.4;
    try t.expect(!wrapper.facts(confidence).limit_shared_drafts);
    try t.expect(!(try attached(.{ .exact_width = 16 }, head.facts_)).limit_shared_drafts);
}
