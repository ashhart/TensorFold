//! The round loop on the fake target: drafted == one-token rounds, shared rounds == solo, greedy and sampled.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;
const grammar = @import("grammar.zig");

const gpa = std.testing.allocator;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    sampling: ?Sampling = null,
    drafts: bool = true,
    think_budget: u32 = 0,
    logprobs: ?u8 = null,
    classes: ?Classes = null, // structured output under this test grammar
    eos: []const u32 = &.{96},
};

/// A test grammar over the fake's vocabulary: `length` tokens, the i-th one of class i % 3 (t % 3, never the stop
/// token 96), then 96; with `after` it starts after that token. Drafts it rejects are common, so windows get cut.
const Classes = struct {
    length: u32 = 12,
    after: ?u32 = null,
    at: u32 = 0,
    done: bool = false,

    fn rules(c: *Classes) grammar.Rules {
        return .{ .ptr = c, .vtable = &.{ .accept = acceptFn, .rollback = rollbackFn, .terminated = terminatedFn, .fill = fillFn, .free = freeFn } };
    }

    fn cast(ptr: *anyopaque) *Classes {
        return @ptrCast(@alignCast(ptr));
    }

    fn allows(c: *const Classes, t: u32) bool {
        if (c.done) return false;
        if (c.at == c.length) return t == 96;
        return t != 96 and t % 3 == c.at % 3;
    }

    fn acceptFn(ptr: *anyopaque, t: u32) anyerror!bool {
        const c = cast(ptr);
        if (!c.allows(t)) return false;
        if (c.at == c.length) c.done = true else c.at += 1;
        return true;
    }

    fn rollbackFn(ptr: *anyopaque, n: usize) anyerror!void {
        const c = cast(ptr);
        for (0..n) |_| {
            if (c.done) c.done = false else c.at -= 1;
        }
    }

    fn terminatedFn(ptr: *anyopaque) bool {
        return cast(ptr).done;
    }

    fn fillFn(ptr: *anyopaque, words: []u32) anyerror!void {
        const c = cast(ptr);
        @memset(words, 0);
        for (0..fake.vocab) |t| if (c.allows(@intCast(t))) {
            words[t / 32] |= @as(u32, 1) << @intCast(t % 32);
        };
    }

    fn freeFn(_: *anyopaque) void {}
};

fn model() !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

/// Every case's emitted tokens, the cases admitted together and stepped until done.
fn run(cases: []const Case) ![][]u32 {
    return runRanked(cases, false);
}

/// `run`, the fake head giving its held chains back when `ranked` (a grammar then walks them).
fn runRanked(cases: []const Case, ranked: bool) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .ranked = ranked };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    const classes = try gpa.alloc(Classes, cases.len);
    defer gpa.free(classes);
    const constraints = try gpa.alloc(grammar.Constraint, cases.len);
    defer gpa.free(constraints);
    for (cases, streams, proposers, classes, constraints) |c, *s, *p, *g, *k| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        if (c.classes) |given| {
            g.* = given;
            k.* = .init(g.rules(), (fake.vocab + 31) / 32, given.after);
        }
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = c.eos, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .logprobs = c.logprobs, .grammar = if (c.classes != null) k else null });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    for (cases, streams) |c, *s| if (c.logprobs) |k| try expectRows(c, s, k);
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

/// Each emitted token's row is the fake target's after the prompt and the tokens before it (forced: by `forToken`).
fn expectRows(c: Case, s: *const sm.Stream, k: u8) !void {
    try std.testing.expectEqual(s.emitted().len, s.rows.items.len);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, c.prompt);
    for (s.emitted(), s.rows.items) |t, got| {
        const want = fake.rowAt(history.items, fake.next(history.items, c.sampling, history.items.len), k).forToken(t);
        try std.testing.expectEqual(want.token, got.token);
        try std.testing.expectEqual(@as(u32, @bitCast(want.logprob)), @as(u32, @bitCast(got.logprob)));
        try std.testing.expectEqualSlices(u32, want.ids[0..k], got.ids[0..got.count]);
        try history.append(gpa, t);
    }
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const GuardCase = struct {
    prompt: []const u32,
    max_new: u32 = 400,
    drafts: bool = true,
    think_budget: u32 = 0,
    loop_guard: bool = true,
    think_open: ?bool = null,
    cycle_after: usize = 70,
    pattern: []const u32 = &.{ 11, 12, 13 },
    answer_cycles: bool = false,
};

const GuardOut = struct {
    emitted: []u32,
    reason: sm.Reason,
    period: ?u32,
    finished: bool,
};

fn runGuard(cases: []const GuardCase, step_limit: usize) ![]GuardOut {
    var cfg = try model();
    defer cfg.deinit(gpa);
    const c = cases[0];
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = c.cycle_after, .probe_prompt = c.prompt.len, .pattern = c.pattern, .answer_cycles = c.answer_cycles };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |item, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "guard", .prompt = item.prompt, .max_new = item.max_new, .eos = &.{96}, .drafts = item.drafts, .proposer = p.proposer(), .think_budget = item.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .think_open = item.think_open, .loop_guard = item.loop_guard });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < step_limit) : (steps += 1) try engine.step();
    const out = try gpa.alloc(GuardOut, cases.len);
    for (out, streams) |*result, *s| result.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .period = s.loop_period, .finished = s.finished };
    return out;
}

fn freeGuard(runs: []GuardOut) void {
    for (runs) |run_result| gpa.free(run_result.emitted);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };

test "drafted rounds commit the one-token decode, greedy and sampled" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }});
        defer free(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }});
        defer free(plain);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        // and the fake target's own decode
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &p1);
        for (drafted[0]) |t| {
            try std.testing.expectEqual(fake.next(history.items, s, history.items.len), t);
            try history.append(gpa, t);
        }
    }
}

test "shared rounds commit what each stream commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } });
    defer free(together);
    const one = try run(&.{.{ .prompt = &p1 }});
    defer free(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30 }});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], together[0]);
    try std.testing.expectEqualSlices(u32, two[0], together[1]);
}

test "the thinking budget's forced close is the same drafted, plain and shared" {
    const drafted = try run(&.{.{ .prompt = &p2, .think_budget = 9 }});
    defer free(drafted);
    const plain = try run(&.{.{ .prompt = &p2, .think_budget = 9, .drafts = false }});
    defer free(plain);
    const shared = try run(&.{ .{ .prompt = &p2, .think_budget = 9 }, .{ .prompt = &p1, .drafts = false } });
    defer free(shared);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, plain[0], shared[0]);
    try std.testing.expectEqual(@as(u32, 90), drafted[0][8]);
}

test "logprob rows follow the committed tokens drafted, plain, shared, sampled and through a forced close" {
    const sampled: Sampling = .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 };
    const cases = [_][]const Case{
        &.{.{ .prompt = &p1, .logprobs = 3 }},
        &.{.{ .prompt = &p1, .logprobs = 3, .drafts = false }},
        &.{.{ .prompt = &p1, .logprobs = 0, .sampling = sampled }},
        &.{ .{ .prompt = &p1, .logprobs = 20 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30, .logprobs = 2 } },
        &.{.{ .prompt = &p2, .think_budget = 9, .logprobs = 4 }},
    };
    for (cases) |with| {
        var without: [2]Case = undefined;
        for (with, without[0..with.len]) |c, *w| w.* = .{ .prompt = c.prompt, .max_new = c.max_new, .sampling = c.sampling, .drafts = c.drafts, .think_budget = c.think_budget };
        const a = try run(with); // checks every row
        defer free(a);
        const b = try run(without[0..with.len]);
        defer free(b);
        for (a, b) |x, y| try std.testing.expectEqualSlices(u32, y, x);
    }
}

test "the loop guard matches drafted and plain, reports its period, and answers after close" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2 }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), drafted[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .stop);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, plain[0].emitted[329..]);

    const off = try runGuard(&.{.{ .prompt = &p2, .loop_guard = false, .think_open = true }}, 5000);
    defer freeGuard(off);
    try std.testing.expect(off[0].finished and off[0].reason == .length and off[0].emitted.len == 400);
}

test "a cycle in the answer does not refire or reclose" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), plain[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .length);
    var close_count: usize = 0;
    for (plain[0].emitted) |token| close_count += @intFromBool(token == 90);
    try std.testing.expectEqual(@as(usize, 1), close_count);
}

test "a prompt pass a chunk a round between other streams' rounds gives the whole pass's tokens" {
    const cases = [_]Case{ .{ .prompt = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4 } }, .{ .prompt = &.{ 9, 10, 11, 12, 13, 14, 15, 9, 10, 11 }, .max_new = 30, .sampling = .{ .temperature = 0.8, .seed = 7 } } };
    const whole = try run(&cases);
    defer free(whole);
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer engine.deinit();
    try std.testing.expect(engine.fills());
    var proposers: [2]SuffixLookup = undefined;
    var streams: [2]sm.Stream = undefined;
    for (cases, &streams, &proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    }
    defer for (&streams, &proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    while (!try engine.fillStream(&streams[0], target.prefill_count == 0)) {}
    var first = true;
    while (!try engine.fillStream(&streams[1], first)) {
        first = false;
        try engine.step(); // the first stream decodes between the second's chunks
    }
    try std.testing.expectEqual(@as(usize, 8), target.prefill_count);
    while (engine.activeCount() > 0) try engine.step();
    for (streams, whole) |s, w| try std.testing.expectEqualSlices(u32, w, s.emitted());
}

/// Every emitted token from the grammar's start on is one the test grammar allows, and it ends at its stop token.
fn expectInGrammar(emitted: []const u32, classes: Classes) !void {
    var g = classes;
    var active = g.after == null;
    for (emitted) |t| {
        if (!active) {
            active = t == g.after.?;
            continue;
        }
        try std.testing.expect(g.allows(t));
        _ = try Classes.acceptFn(&g, t);
    }
    try std.testing.expect(g.done);
}

test "structured output: drafted rounds commit the one-token decode, every token inside the grammar" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        for ([_]bool{ false, true }) |ranked| {
            const drafted = try runRanked(&.{.{ .prompt = &p1, .sampling = s, .classes = .{} }}, ranked);
            defer free(drafted);
            const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false, .classes = .{} }});
            defer free(plain);
            try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
            try expectInGrammar(drafted[0], .{});
        }
    }
}

test "structured output: shared rounds with grammar and plain streams commit what each commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const cases = [_]Case{ .{ .prompt = &p1, .classes = .{ .length = 20 } }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 }, .{ .prompt = &p2, .sampling = sampled, .classes = .{ .length = 7 } } };
    const together = try runRanked(&cases, true);
    defer free(together);
    for (cases, together) |c, got| {
        const alone = try run(&.{c});
        defer free(alone);
        try std.testing.expectEqualSlices(u32, alone[0], got);
        if (c.classes) |g| try expectInGrammar(got, g);
    }
}

test "structured output: with thinking the grammar starts after the think end, through a forced close" {
    const c: Case = .{ .prompt = &p2, .think_budget = 9, .classes = .{ .after = 91 } };
    const drafted = try run(&.{c});
    defer free(drafted);
    var plain_case = c;
    plain_case.drafts = false;
    const plain = try run(&.{plain_case});
    defer free(plain);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91 }, drafted[0][8..10]); // the close ends at the think end
    try expectInGrammar(drafted[0], .{ .after = 91 });
}

test "structured output: the grammar's stop token ends a reply that ignores eos" {
    const out = try run(&.{.{ .prompt = &p1, .eos = &.{}, .max_new = 200, .classes = .{ .length = 5 } }});
    defer free(out);
    try std.testing.expectEqual(@as(usize, 6), out[0].len);
    try expectInGrammar(out[0], .{ .length = 5 });
}
