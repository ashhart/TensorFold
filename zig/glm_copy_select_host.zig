//! Native TP=2 copy admission through real facts, depth probes and verification.
const std = @import("std");
const glm = @import("glm");
const lanes = @import("lanes");
const a = std.testing.allocator;
const policy = glm.native.copy_select;
const win = lanes.windows;
fn facts(on: bool) lanes.Model {
    var native: glm.engine.Engine = undefined;
    native.mtp = true;
    native.drafter = null;
    var backend: glm.lanes.Lanes = .{ .x = .{ .e = &native, .link = undefined } };
    backend.copy_selection = .{ .on = on, .world = 2 };
    return backend.factsWithCopies(false);
}
const Offer = struct {
    calls: u32 = 0,
    match: i64 = 8,
    good: bool = false,
    tokens: [15]u32 = undefined,
    fn propose(ptr: *anyopaque, context: []const u32, most: i64) ![]const u32 {
        const p: *Offer = @ptrCast(@alignCast(ptr));
        p.calls += 1;
        const n = @min(p.tokens.len, @as(usize, @intCast(@max(0, most))));
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(a);
        try history.appendSlice(a, context);
        for (p.tokens[0..n]) |*t| {
            t.* = if (p.good) lanes.fake.next(history.items, null, history.items.len) else 1001;
            try history.append(a, t.*);
        }
        return p.tokens[0..n];
    }
    fn matched(ptr: *anyopaque) i64 {
        return @as(*Offer, @ptrCast(@alignCast(ptr))).match;
    }
    fn proposer(p: *Offer) lanes.proposer.Proposer {
        return .{ .ptr = p, .vtable = &.{ .propose = propose, .last_match = matched } };
    }
};
fn teach(e: *lanes.Engine, s: *lanes.Stream, kept: usize) !void {
    for (0..4) |_| {
        _ = try win.copyProposal(e, s);
        win.observeCopied(e, s, false, kept, 3);
        s.rounds += 1;
    }
    s.copy_select.heads = 4;
}
fn setupCosts(m: *lanes.Model, costs: *[32]lanes.config.Cost) void {
    for (costs, 1..) |*c, n| c.* = .{ .width = @intCast(n), .ms = 10 };
    m.window_costs = costs;
    m.shared_costs = costs;
}
test "glm copy select: weak suffix frees the native head, productive copy and off keep old choice" {
    for ([_]bool{ false, true }) |on| {
        var m = facts(on);
        var costs: [32]lanes.config.Cost = undefined;
        setupCosts(&m, &costs);
        m.window_costs_lone = true;
        m.shared_costs = &.{};
        var cfg = try lanes.Config.init(a, m, 32, 15);
        defer cfg.deinit(a);
        var target: lanes.fake.Fake = .{ .gpa = a };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
        defer e.deinit();
        var offer: Offer = .{};
        var s = try lanes.Stream.init(a, .{ .id = "weak", .prompt = &.{ 7, 1, 2, 3, 4 }, .max_new = 64, .proposer = offer.proposer() });
        defer s.deinit(a);
        try teach(&e, &s, 0);
        var who = win.who(&s);
        const d = try e.rule.headDepth(&who, 3, win.probe(&e));
        try std.testing.expectEqual(@as(i64, if (on) 3 else 1), d);
        s.next = .{ .count = 3, .tokens = try a.dupe(u32, &.{ 4, 5, 6 }) };
        try std.testing.expectEqual(if (on) win.Kind.head else win.Kind.copy, (try win.plan(&e, &s, null)).kind);
        s.dropRoundState(a);
        offer.good = true;
        offer.match = 24;
        try teach(&e, &s, 3);
        try std.testing.expect((try win.copyProposal(&e, &s)).tokens.len > 0);
        try std.testing.expectEqual(@as(u64, 0), s.copy_select.bands[0].judgments);
    }
}
test "glm copy select: one lookup per context, bounded retries, silence and rollback" {
    var m = facts(true);
    var costs: [32]lanes.config.Cost = undefined;
    setupCosts(&m, &costs);
    var cfg = try lanes.Config.init(a, m, 32, 15);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var offer: Offer = .{};
    var s = try lanes.Stream.init(a, .{ .id = "one", .prompt = &.{ 1, 2, 3, 4 }, .max_new = 128, .proposer = offer.proposer() });
    defer s.deinit(a);
    try teach(&e, &s, 0);
    const before = offer.calls;
    for (0..4) |_| _ = try win.copyProposal(&e, &s);
    try std.testing.expectEqual(before + 1, offer.calls);
    try std.testing.expectEqual(@as(usize, 0), (try win.copyProposal(&e, &s)).tokens.len);
    for (0..3) |_| {
        s.rounds += 1;
        _ = try win.copyProposal(&e, &s);
    }
    try std.testing.expect((try win.copyProposal(&e, &s)).tokens.len > 0);
    s.dropRoundState(a);
    try std.testing.expectEqual(@as(?usize, null), s.copy_select.context);
    try std.testing.expectEqual(@as(u64, 0), s.copy_select.heads);
    var suffix = try lanes.proposer.SuffixLookup.init(a, .{ .ngram = 2, .min_match = 2, .silence_rounds = 3, .window = 1 });
    defer suffix.deinit();
    s.proposer = suffix.proposer();
    try suffix.observe(3, 0);
    for (0..3) |_| _ = try win.copyProposal(&e, &s);
    try std.testing.expectEqual(@as(i64, 2), suffix.silent_for);
    try std.testing.expectEqual(@as(i64, 1), suffix.silenced_rounds);
    try std.testing.expectEqual(@as(usize, 0), suffix.indexed);
    s.dropRoundState(a);
    suffix.silent_for = 0;
    try s.context.appendSlice(a, &.{ 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4 });
    const history = try a.dupe(u32, s.context.items);
    defer a.free(history);
    for (0..3) |_| _ = try win.copyProposal(&e, &s);
    try std.testing.expectEqual(s.context.items.len, suffix.indexed);
    try std.testing.expectEqualSlices(u32, history, s.context.items);
    var other = try lanes.Stream.init(a, .{ .id = "isolated", .prompt = history, .max_new = 128, .proposer = offer.proposer() });
    defer other.deinit(a);
    try std.testing.expectEqual(@as(u64, 0), other.copy_select.bands[0].judgments);
    try std.testing.expect((try win.copyProposal(&e, &other)).tokens.len > 0);
}
const Run = struct { on: bool = true, streams: usize = 2, seed: ?u64 = null, restore: bool = false, replace: bool = false, drafts: bool = true };
fn run(o: Run) !void {
    var m = facts(o.on);
    var costs: [32]lanes.config.Cost = undefined;
    setupCosts(&m, &costs);
    var cfg = try lanes.Config.init(a, m, 32, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var target: lanes.fake.Fake = .{ .gpa = a, .head_accepts = 1000 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    const prompts = [_][]const u32{ &.{ 7, 13, 29, 31 }, &.{ 3, 19, 41, 59 } };
    var offers: [2]Offer = .{ .{}, .{} };
    var streams: [2]lanes.Stream = undefined;
    var saved: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&saved) |*s| s.deinit(a);
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (streams[0..o.streams], 0..) |*s, i| {
        try saved[i].appendSlice(a, prompts[i][0..2]);
        s.* = try lanes.Stream.init(a, .{ .id = if (i == 0) "first" else "second", .prompt = prompts[i], .max_new = 192, .drafts = o.drafts, .proposer = offers[i].proposer(), .sampling = if (o.seed) |seed| .{ .seed = seed + i } else null, .reuse = if (o.restore) .{ .saved = &saved[i], .at = 2 } else .{} });
        made += 1;
        try e.addStream(s);
        try std.testing.expectEqual(@as(u32, if (o.restore) 2 else 0), s.cached);
    }
    var replaced = false;
    var heads: [2]u64 = .{ 0, 0 };
    var copies: [2]u64 = .{ 0, 0 };
    while (e.activeCount() > 0) {
        try std.testing.expect(e.steps < 1024);
        if (o.replace and !replaced and e.steps == 10) {
            e.discard(&streams[1]);
            streams[1].deinit(a);
            streams[1] = try lanes.Stream.init(a, .{ .id = "replacement", .prompt = prompts[1], .max_new = 192, .proposer = offers[1].proposer(), .sampling = if (o.seed) |seed| .{ .seed = seed + 1 } else null });
            try std.testing.expectEqual(@as(u64, 0), streams[1].copy_select.heads);
            try e.addStream(&streams[1]);
            replaced = true;
        }
        try e.step();
        for (streams[0..o.streams], 0..) |s, i| {
            heads[i] = @max(heads[i], s.copy_select.heads);
            copies[i] = @max(copies[i], s.copy_select.bands[0].judgments);
        }
        for (e.live.items) |s| {
            const cache = target.cache(s) orelse return error.NoCache;
            try std.testing.expect(cache.len == s.context.items.len or cache.len + 1 == s.context.items.len);
            try std.testing.expectEqualSlices(u32, s.context.items[0..cache.len], cache);
        }
    }
    try std.testing.expect(!o.replace or replaced);
    for (streams[0..o.streams], 0..) |s, i| {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(a);
        try history.appendSlice(a, prompts[i]);
        const sampling: ?lanes.sampling.Sampling = if (o.seed) |seed| .{ .seed = seed + i } else null;
        for (0..192) |_| try history.append(a, lanes.fake.next(history.items, sampling, history.items.len));
        try std.testing.expectEqualSlices(u32, history.items[prompts[i].len..], s.emitted());
        if (o.on and o.drafts) {
            try std.testing.expect(heads[i] >= 4);
            try std.testing.expect(copies[i] >= 4);
        }
    }
}
test "glm copy select: actual engine tokens, cached state, partial keeps and replacement" {
    for ([_]?u64{ null, 17 }) |seed| {
        try run(.{ .seed = seed });
        try run(.{ .seed = seed, .restore = true });
        try run(.{ .seed = seed, .replace = true });
        try run(.{ .seed = seed, .streams = 1 });
        try run(.{ .seed = seed, .on = false });
        try run(.{ .seed = seed, .drafts = false });
    }
}
test "glm copy select: registry forms and family mode gates" {
    const env = @import("sys").environment;
    for ([_]?[]const u8{ null, "", " \t" }) |raw| try std.testing.expectEqual(@as(?u64, null), try env.parse(policy.name, raw));
    for ([_][]const u8{ "0", "1" }, 0..) |raw, n| try std.testing.expectEqual(@as(?u64, n), try env.parse(policy.name, raw));
    for ([_][]const u8{ "2", "-1", "true", " 1", "1 " }) |raw| try std.testing.expectError(error.InvalidEnvironment, env.parse(policy.name, raw));
    try std.testing.expect(!facts(false).copy_select);
    for ([_]usize{ 1, 3 }) |world| try std.testing.expect(!policy.apply(facts(true), true, world, false, false).copy_select);
    try std.testing.expect(!policy.apply(facts(true), true, 2, true, false).copy_select);
    try std.testing.expect(!policy.apply(facts(true), true, 2, false, true).copy_select);
    var m = facts(false);
    m.root_copies = true;
    try std.testing.expect(!policy.apply(m, true, 2, false, false).copy_select);
}
