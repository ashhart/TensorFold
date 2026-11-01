//! Synthetic code-copy allocation with the CUDA adapter's real sixteen-row and eight-stream limits.
const std = @import("std");
const lanes = @import("lanes");
const win = lanes.windows;
const t = std.testing;
const a = t.allocator;
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
const Result = struct { copy_calls: usize = 0, budgets: [16]i64 = @splat(0), grants: [16]usize = @splat(0) };

fn replay(model: lanes.Model, policy: bool, stale: bool) !Result {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, row| c.* = .{ .width = @intCast(row), .ms = 5 + 0.2 * @as(f64, @floatFromInt(row)) };
    var m = model;
    m.allocate_shared_drafts = policy;
    m.window_costs = &costs;
    var cfg = try lanes.Config.init(a, m, m.exact_width, m.drafts);
    defer cfg.deinit(a);
    cfg.timed = false;
    try t.expectEqual(@as(u32, 8), cfg.batch_streams);
    try t.expectEqual(@as(u32, 16), cfg.batch_rows);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    e.alone = false;
    var streams: [16]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    var copy_calls: usize = 0;
    for (&streams, 0..) |*s, i| {
        s.* = try lanes.Stream.init(a, .{ .id = "code", .prompt = &.{ 1, 2, 3 }, .max_new = 128 });
        made += 1;
        const rates = try e.rule.rates(win.who(s));
        @memset(rates, if (i % 8 == 0) (if (stale) @as(f64, 0.983) else 0.99) else 0.3);
        // One accepted depth updates 0.98 to 0.983; the preceding window still cached its 0.98 prices.
        if (stale and i % 8 == 0) {
            var chance: f64 = 1;
            for (0..15) |_| {
                chance *= 0.98;
                try s.draft_chances.append(a, chance);
            }
        }
        if (i % 8 == 1) s.proposer = .{ .ptr = &copy_calls, .vtable = &.{ .propose = Copy.propose, .last_match = Copy.match } };
    }
    var got: Result = .{};
    for (0..2) |batch| {
        var current: [8]win.Plan = undefined;
        var who: [8]lanes.depth.Who = undefined;
        const start = batch * 8;
        for (&current, &who, streams[start..][0..8]) |*p, *w, *s| {
            p.* = .{ .stream = s, .position = s.cache_len, .kind = .none };
            w.* = win.who(s);
        }
        const budgets = got.budgets[start..][0..8];
        if (policy) try lanes.shared_drafts.budgets(&e, &current, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, budgets) else try e.rule.budgets(&who, win.probe(&e), false, budgets);
        var next: [8]win.Plan = undefined;
        for (&next, streams[start..][0..8], budgets, 0..) |*p, *s, depth, i| {
            p.* = .{ .stream = s, .position = s.cache_len, .kind = if (i == 1) .copy else if (depth > 0) .head else .none, .held = if (i == 1) 0 else @intCast(depth), .tokens = if (i == 1) &code else &.{} };
        }
        try win.allocate(&e, &next);
        for (next, got.grants[start..][0..8]) |p, *n| n.* = p.count();
    }
    got.copy_calls = copy_calls;
    return got;
}

pub fn check(model: lanes.Model) !void {
    try t.expect(model.allocate_shared_drafts);
    for ([_]f64{ 0.5, 2 }) |head_ms| for ([_]bool{ false, true }) |stale| {
        var m = model;
        m.mtp_step_ms = head_ms;
        const old = try replay(m, false, stale);
        const next = try replay(m, true, stale);
        try t.expectEqual(@as(usize, 2), old.copy_calls);
        try t.expectEqual(old.copy_calls, next.copy_calls);
        const expected = if (head_ms == 2) [_]usize{ 1, 7, 0, 0, 0, 0, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0 } else [_]usize{ 7, 1, 0, 0, 0, 0, 0, 0, 7, 1, 0, 0, 0, 0, 0, 0 };
        try t.expectEqualSlices(usize, &expected, &old.grants);
        try t.expectEqualSlices(usize, &old.grants, &next.grants);
        var before: i64 = 0;
        var after: i64 = 0;
        for (old.budgets, next.budgets) |was, now| {
            try t.expect(now <= was);
            before += was;
            after += now;
        }
        try t.expect(after < before);
        for (next.budgets, 0..) |count, i| if (i % 8 > 1) try t.expectEqual(@as(i64, 0), count);
    };
}

test "Nemotron shared prices: sixteen greedy streams retain the costed code-copy grants" {
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.c.latent = 0;
    var b: @import("cuda_lanes.zig").Cuda = undefined;
    b.e = &e;
    b.head = @ptrFromInt(16);
    b.copy_drafts = true;
    b.cost_count = 0;
    b.mtp_ms = 0;
    b.mtp_drafts = null;
    b.mtp_confidence = null;
    try check(b.facts());
    b.mtp_confidence = 0.4;
    if (comptime @hasField(lanes.Model, "limit_shared_drafts")) try std.testing.expect(!b.facts().limit_shared_drafts);
}

fn lifecycle(model: lanes.Model, policy: bool, seed: ?u64) ![]u32 {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, row| c.* = .{ .width = @intCast(row), .ms = 5 + 0.2 * @as(f64, @floatFromInt(row)) };
    var m = model;
    m.allocate_shared_drafts = policy;
    m.window_costs = &costs;
    m.mtp_step_ms = 2;
    var cfg = try lanes.Config.init(a, m, m.exact_width, m.drafts);
    defer cfg.deinit(a);
    cfg.timed = false;
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var streams: [16]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (&streams, 0..) |*s, i| {
        const prompt = [_]u32{ 7, 13, @intCast(29 + i) };
        s.* = try lanes.Stream.init(a, .{ .id = "code", .prompt = &prompt, .max_new = 48, .eos = &.{}, .sampling = if (seed) |v| .{ .seed = v, .temperature = 1, .top_p = 0.95 } else null });
        made += 1;
        try e.addStream(s);
    }
    while (e.activeCount() > 0) try e.step();
    const out = try a.alloc(u32, streams.len * 48);
    errdefer a.free(out);
    for (streams, 0..) |s, i| {
        try t.expectEqual(@as(usize, 48), s.emitted().len);
        @memcpy(out[i * 48 ..][0..48], s.emitted());
    }
    return out;
}

test "Nemotron shared prices: sixteen greedy and sampled streams retain their target tokens" {
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.c.latent = 0;
    var b: @import("cuda_lanes.zig").Cuda = undefined;
    b.e = &e;
    b.head = @ptrFromInt(16);
    b.copy_drafts = true;
    b.cost_count = 0;
    b.mtp_ms = 0;
    b.mtp_drafts = null;
    b.mtp_confidence = null;
    for ([_]?u64{ null, 1234 }) |seed| {
        const old = try lifecycle(b.facts(), false, seed);
        defer a.free(old);
        const next = try lifecycle(b.facts(), true, seed);
        defer a.free(next);
        var plain = b.facts();
        plain.mtp = false;
        plain.speculate = false;
        plain.copy_drafts = false;
        const serial = try lifecycle(plain, false, seed);
        defer a.free(serial);
        try t.expectEqualSlices(u32, serial, old);
        try t.expectEqualSlices(u32, serial, next);
    }
}
