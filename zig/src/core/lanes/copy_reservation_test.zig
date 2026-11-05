//! A shared reservation and its next verify window use one owned copy decision, including an empty backoff answer.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const Stream = @import("stream.zig").Stream;
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const win = @import("windows.zig");
const shared = @import("shared_drafts.zig");
const fake = @import("fake.zig");
const t = std.testing;
const a = t.allocator;
const history = [_]u32{ 1, 2, 3, 4, 5, 9, 1, 2, 3 };

const Fixture = struct {
    cfg: Config = undefined,
    target: fake.Fake = undefined,
    clock: fake.FixedClock = .{},
    e: Engine = undefined,

    fn init(f: *Fixture, priced: bool) !void {
        var costs: [8]@import("config.zig").Cost = undefined;
        for (&costs, 1..) |*c, n| c.* = .{ .width = @intCast(n), .ms = if (priced) 10 + 6 * @as(f64, @floatFromInt(n - 1)) else 1 };
        f.cfg = try Config.init(a, .{ .exact_width = 4, .drafts = 3, .batch_rows = 8, .window_costs = &costs, .copy_match = 2, .own_copy_rates = priced, .matched_copy_rates = priced }, 4, 3);
        f.cfg.timed = false;
        f.target = .{ .gpa = a };
        f.e = Engine.init(a, &f.cfg, f.target.backend(), f.clock.clock());
        f.e.alone = false;
    }

    fn deinit(f: *Fixture) void {
        f.e.deinit();
        f.target.deinit();
        f.cfg.deinit(a);
    }

    fn reserve(f: *Fixture, s: *Stream) !void {
        const plans = [_]win.Plan{.{ .stream = s, .position = s.cache_len, .kind = .none }};
        var out: [1]i64 = undefined;
        try shared.budgets(&f.e, &plans, &.{0}, &out);
    }
};

fn lookup() !SuffixLookup {
    return SuffixLookup.init(a, .{ .ngram = 2, .min_match = 2, .silence_rounds = 1, .window = 1 });
}

test "copy reservation: a silent suffix lookup leaves the funded head for the actual verify" {
    var f: Fixture = .{};
    try f.init(false);
    defer f.deinit();
    var look = try lookup();
    defer look.deinit();
    var s = try Stream.init(a, .{ .id = "backoff", .prompt = history[0 .. history.len - 1], .max_new = 32, .proposer = look.proposer() });
    defer s.deinit(a);
    try f.e.backend.prefill(&s);
    s.cache_len = s.prompt_len;
    _ = try s.commit(a, &.{3}, &.{});
    s.pending = 3;
    // These are the target's next two tokens, unlike the repeated context's continuation 4, 5, 9.
    s.next = .{ .count = 2, .tokens = try a.dupe(u32, &.{ 43, 56 }) };
    try look.observe(3, 0);
    try f.reserve(&s); // the one remaining silent ask reserves a head, not a copy
    _ = f.e.arena.reset(.retain_capacity);
    const result = try f.e.familyRound(&s, null);
    try t.expectEqual(@as(u32, 3), result.rows);
    try t.expectEqual(@as(u32, 3), result.keep);
    try t.expectEqualSlices(u32, &.{ 43, 56, 63 }, result.got);
    // A duplicate ask at planning would wake the copy, reject 4 against 43, and commit only 43.
}

test "copy reservation: an empty decision is consumed once and unreserved asks retain backoff semantics" {
    var f: Fixture = .{};
    try f.init(false);
    defer f.deinit();
    var look = try lookup();
    defer look.deinit();
    var s = try Stream.init(a, .{ .id = "empty", .prompt = &history, .max_new = 32, .proposer = look.proposer() });
    defer s.deinit(a);
    try look.observe(3, 0);
    try f.reserve(&s);
    try f.reserve(&s); // repricing the same window cannot wake its copy either
    _ = f.e.arena.reset(.retain_capacity);
    try t.expectEqual(win.Kind.none, (try win.plan(&f.e, &s, null)).kind);
    try t.expectEqualSlices(u32, &.{ 4, 5, 9 }, (try win.plan(&f.e, &s, null)).tokens);
    try look.observe(3, 0);
    try t.expectEqual(win.Kind.none, (try win.plan(&f.e, &s, null)).kind);
    try t.expectEqualSlices(u32, &.{ 4, 5, 9 }, (try win.plan(&f.e, &s, null)).tokens);
}

test "copy reservation: shared borrowed tokens and match evidence survive another stream's query" {
    var f: Fixture = .{};
    try f.init(true);
    defer f.deinit();
    var look = try lookup();
    defer look.deinit();
    var strong_history: [45]u32 = undefined;
    for (0..21) |i| {
        strong_history[i] = @intCast(i + 1);
        strong_history[24 + i] = @intCast(i + 1);
    }
    strong_history[21..24].* = .{ 41, 42, 43 };
    var strong = try Stream.init(a, .{ .id = "strong", .prompt = &strong_history, .max_new = 64, .proposer = look.proposer() });
    defer strong.deinit(a);
    var weak = try Stream.init(a, .{ .id = "weak", .prompt = &.{90}, .max_new = 64, .proposer = look.proposer() });
    defer weak.deinit(a);
    _ = try weak.commit(a, &.{ 5, 6, 7, 80, 81, 82, 5, 6, 7 }, &.{});
    for (0..12) |_| {
        win.observePricedCopied(&f.e, &strong, false, false, 0, 3);
        win.observePricedCopied(&f.e, &weak, false, false, 0, 3);
    }
    // Observe above also silences the shared lookup; consume that ask before reserving these windows.
    _ = try look.propose(&history, 3);
    try f.reserve(&strong);
    try f.reserve(&weak); // overwrites the shared lookup's tokens and its trusted-match evidence
    _ = f.e.arena.reset(.retain_capacity);
    var plans = [_]win.Plan{ try win.plan(&f.e, &strong, null), try win.plan(&f.e, &weak, null) };
    try t.expectEqualSlices(u32, &.{ 41, 42, 43 }, plans[0].tokens);
    try t.expectEqualSlices(u32, &.{ 80, 81, 82 }, plans[1].tokens);
    _ = try look.propose(&.{ 91, 92, 93, 94 }, 3);
    try win.allocate(&f.e, &plans);
    try t.expect(plans[0].count() > 0); // strong copies keep their independent prior
    try t.expectEqual(@as(usize, 0), plans[1].count()); // this reply's weak copies have already lost their price
}

test "copy reservation: forced windows and changed history or lifecycle discard the old decision" {
    const Change = enum { commit, context, forced, close, release, position, pending, round, room, width, cohort, paused, proposer };
    inline for (std.meta.tags(Change)) |change| {
        var f: Fixture = .{};
        try f.init(false);
        defer f.deinit();
        var look = try lookup();
        defer look.deinit();
        var replacement = try lookup();
        defer replacement.deinit();
        var s = try Stream.init(a, .{ .id = "changed", .prompt = &history, .max_new = 32, .proposer = look.proposer(), .think_close = &.{ 77, 78 } });
        defer s.deinit(a);
        try f.reserve(&s);
        try look.observe(3, 0); // a fresh decision is empty; retaining the old one would wrongly copy 4, 5, 9
        switch (change) {
            .commit => _ = try s.commit(a, &.{77}, &.{}),
            .context => try s.context.append(a, 77),
            .forced => try s.force.append(a, 77),
            .close => _ = try s.startClose(a),
            .release => s.dropRoundState(a),
            .position => s.cache_len += 1,
            .pending => s.pending = 77,
            .round => s.rounds += 1,
            .room => s.max_new -= 1,
            .width => s.copy_width = 2,
            .cohort => f.e.alone = true,
            .paused => {
                s.paused = true;
                try f.reserve(&s);
                s.paused = false;
            },
            .proposer => {
                try replacement.observe(3, 0);
                s.proposer = replacement.proposer();
            },
        }
        if (change == .forced or change == .close) {
            const forced = try win.plan(&f.e, &s, null);
            try t.expectEqual(win.Kind.forced, forced.kind);
            try t.expectEqualSlices(u32, if (change == .forced) &.{77} else &.{78}, forced.tokens);
        }
        try t.expectEqual(win.Kind.none, (try win.plan(&f.e, &s, null)).kind);
    }
}

test "copy reservation: owned storage is reused after consuming a window" {
    var f: Fixture = .{};
    try f.init(false);
    defer f.deinit();
    var look = try lookup();
    defer look.deinit();
    var s = try Stream.init(a, .{ .id = "storage", .prompt = &history, .max_new = 32, .proposer = look.proposer() });
    defer s.deinit(a);
    try f.reserve(&s);
    _ = try win.plan(&f.e, &s, null);
    // A second window's ownership needs no allocator call once the largest copy has fit.
    var failing = t.FailingAllocator.init(a, .{ .fail_index = 0, .resize_fail_index = 0 });
    f.e.gpa = failing.allocator();
    defer f.e.gpa = a;
    _ = try win.reserveCopy(&f.e, &s);
    const next = try win.plan(&f.e, &s, null);
    try t.expectEqual(win.Kind.copy, next.kind);
    try t.expectEqualSlices(u32, &.{ 4, 5, 9 }, next.tokens);
}

test "copy reservation: Recall keeps its source beside another stream's lookup" {
    const recall = @import("recall.zig");
    var f: Fixture = .{};
    try f.init(false);
    defer f.deinit();
    var record: recall.Shared = .{ .io = t.io, .record = try recall.Record.init(a, 16) };
    defer record.record.deinit();
    var remembered: [23]u32 = undefined;
    for (remembered[0..20], 1..) |*token, i| token.* = @intCast(i);
    remembered[20..23].* = .{ 71, 72, 73 };
    try record.record.add(&remembered);
    var other: [67]u32 = undefined;
    for (other[0..64], 101..) |*token, i| token.* = @intCast(i);
    other[64..67].* = .{ 81, 82, 83 };
    try record.record.add(&other);
    var reader: recall.Reader = .{ .gpa = a, .shared = &record, .head = 0 };
    defer reader.deinit();
    var other_reader: recall.Reader = .{ .gpa = a, .shared = &record, .head = 100 };
    defer other_reader.deinit();
    var s = try Stream.init(a, .{ .id = "recall", .prompt = remembered[0..19], .max_new = 32, .recall = &reader });
    defer s.deinit(a);
    try f.e.backend.prefill(&s);
    s.cache_len = s.prompt_len;
    _ = try s.commit(a, &.{20}, &.{});
    s.pending = 20;
    try f.reserve(&s);
    // Readers are per stream; another stream can probe the shared record without changing this copy's evidence.
    record.rates.probe_every = 32;
    record.rates.since = 31;
    try t.expectEqualSlices(u32, &.{ 81, 82, 83 }, other_reader.propose(other[0..64], 3, 0));
    _ = f.e.arena.reset(.retain_capacity);
    const copied = try win.copyProposal(&f.e, &s);
    try t.expect(copied.recalled);
    try t.expectEqualSlices(u32, &.{ 71, 72, 73 }, copied.tokens);
    const missed_before = record.rates.missed;
    const result = try f.e.familyRound(&s, null);
    try t.expectEqual(@as(u32, 4), result.rows);
    try t.expectEqualSlices(u32, &.{75}, result.got);
    try t.expect(record.rates.missed[0] != missed_before[0]);
    try t.expectEqual(missed_before[2], record.rates.missed[2]);
    try t.expectEqual(@as(u32, 32), record.rates.probe_every); // this rejected copy was not the later lookup's probe
}
