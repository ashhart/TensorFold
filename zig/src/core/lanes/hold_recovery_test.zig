//! Real Engine rounds around a capped granted-depth hold: only real periodic head trials re-measure it (SYNTHETIC).
const std = @import("std");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const be = @import("backend.zig");
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const dr = @import("drafter.zig");
const Drafted = @import("drafted.zig").Drafted;
const Proposer = @import("proposer.zig").Proposer;
const dt = @import("drafted_test.zig");
const HoldCost = @import("hold_price.zig").HoldCost;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

/// SYNTHETIC TIMING, not a device or performance measurement: a shared forward of `w` rows costs 9 + w ms.
var scenario_costs: [16]Cost = undefined;

/// SYNTHETIC TIMING: a lone block's hold pays above a measured gain of about 0.86 (3.5 promised; first hold pays).
const holds_one = [_]HoldCost{.{ .streams = 1, .depth = 4, .ms = 11.5 }};
/// SYNTHETIC TIMING: beside a companion's paid hold, a second block pays above a measured gain of about 1.41.
const holds_two = [_]HoldCost{ .{ .streams = 1, .depth = 4, .ms = 2.0 }, .{ .streams = 2, .depth = 4, .ms = 4.4 } };

/// drafted_test's drafter, scripted: the target's own chains, first draft wrong for `wrong`, all priced 0.95.
const Script = struct {
    head: dt.FakeDrafter,
    vtable: dr.Drafter.VTable = undefined,
    inner_hold: *const fn (*anyopaque, []const dr.Hold) anyerror!void = undefined,
    wrong: ?*sm.Stream = null,
    watched: ?*sm.Stream = null,
    asked: u32 = 0, // holds the backend was asked for the watched stream

    fn of(ptr: *anyopaque) *Script {
        return @fieldParentPtr("head", @as(*dt.FakeDrafter, @ptrCast(@alignCast(ptr))));
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = of(ptr);
        try x.inner_hold(ptr, items);
        for (items) |it| {
            const lane = x.head.lanes.getPtr(it.stream) orelse return error.UnknownStream;
            // rounds without a hold absorbed the same rows, so the drafter's context still is the stream's
            try std.testing.expectEqualSlices(u32, it.stream.context.items[0..@intCast(it.position - 1)], lane.history.items);
            if (it.stream == x.watched) x.asked += 1;
            var guess: std.ArrayList(u32) = .empty;
            defer guess.deinit(gpa);
            try guess.appendSlice(gpa, lane.history.items);
            try guess.append(gpa, it.pending);
            for (lane.held.items, 0..) |*t, j| {
                t.* = fake.next(guess.items, it.stream.sampling, it.position + j);
                try guess.append(gpa, t.*);
            }
            if (it.stream == x.wrong) lane.held.items[0] = (lane.held.items[0] + 1) % fake.vocab;
        }
    }

    fn chances(ptr: *anyopaque, s: *sm.Stream, out: []f64) anyerror!bool {
        const lane = of(ptr).head.lanes.getPtr(s) orelse return false;
        if (out.len > lane.held.items.len) return error.TooFewDrafts;
        var reach: f64 = 1;
        for (out) |*o| {
            reach *= 0.95;
            o.* = reach;
        }
        return true;
    }

    fn drafter(x: *Script) dr.Drafter {
        const inner = x.head.drafter();
        x.vtable = inner.vtable.*;
        x.inner_hold = inner.vtable.hold;
        x.vtable.hold = hold;
        x.vtable.chances = chances;
        return .{ .ptr = &x.head, .vtable = &x.vtable };
    }
};

fn sampling(seed: ?u64) ?Sampling {
    return if (seed) |v| .{ .seed = v, .temperature = 1.0, .top_p = 0.95 } else null;
}

fn scenarioModel() Model {
    for (&scenario_costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 9.0 + @as(f64, @floatFromInt(w)) };
    return .{ .exact_width = 16, .window_costs = &scenario_costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 16, .lone_copies = false };
}

/// What the target alone decodes for `prompt`, one token a step, no drafter.
fn plainTokens(prompt: []const u32, max_new: u32, seed: ?u64) ![]u32 {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var cfg = try Config.init(gpa, scenarioModel(), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "plain", .prompt = prompt, .max_new = max_new, .drafts = false, .sampling = sampling(seed) });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return gpa.dupe(u32, s.emitted());
}

/// One step; how many drafts the subject stream had verified in it.
fn offered(engine: *Engine, s: *const sm.Stream) !u64 {
    const before = s.drafted;
    try engine.step();
    return s.drafted - before;
}

/// Subject's wrong guesses stand its hold aside; a wrong trial backs off, a later right one recovers it.
fn recovers(live: usize, seed: ?u64) !void {
    const subject_new = 260;
    const prompts = [2][]const u32{ &dt.p1, dt.long[0..11] };
    const plain_subject = try plainTokens(prompts[0], subject_new, seed);
    defer gpa.free(plain_subject);
    const plain_companion = try plainTokens(prompts[1], 520, seed);
    defer gpa.free(plain_companion);

    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    const holds: []const HoldCost = if (live == 1) &holds_one else &holds_two;
    var script: Script = .{ .head = .{ .facts_ = .{ .depth = 4, .step_ms = 0.5, .plain_guard = false, .batched = true, .chances = true, .hold_costs = holds } } };
    defer script.head.deinit();
    var wrapped = try Drafted.init(gpa, strict.backend(), script.drafter());
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, wrapped.facts(scenarioModel()), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false; // rounds are priced by the synthetic tables alone
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var streams: [2]sm.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    for (0..live) |i| {
        streams[i] = try sm.Stream.init(gpa, .{ .id = if (i == 0) "subject" else "companion", .prompt = prompts[i], .max_new = if (i == 0) subject_new else 520, .sampling = sampling(seed) });
        made += 1;
        try engine.addStream(&streams[i]);
    }
    const subject = &streams[0];
    script.watched = subject;
    const every: u32 = @intCast(engine.rule.k.depth_probe_every);

    // right guesses: every draft the subject's grants verify is kept, and its holds pay each round
    for (0..6) |_| try engine.step();
    try std.testing.expect(subject.drafted > 0 and subject.accepted == subject.drafted);

    // wrong guesses: each round's drafts all miss until the measured acceptance stands the hold aside
    script.wrong = subject;
    var suppressed = false;
    for (0..60) |_| {
        if (try offered(&engine, subject) == 0) {
            suppressed = true;
            break;
        }
    }
    try std.testing.expect(suppressed);
    try std.testing.expect(subject.granted.? > 0); // a positive verified grant, held aside
    try std.testing.expectEqual(live, engine.activeCount());
    const asked = script.asked;
    const kept = subject.accepted;
    const drafted = subject.drafted;
    const before = subject.depth.?.p[0];

    // the first real trial comes after the rounds a probe interval spans, one hold for the whole stretch
    var skipped: u32 = 1; // the round that found the hold aside
    var tried = false;
    for (0..64) |_| {
        if (try offered(&engine, subject) > 0) {
            tried = true;
            break;
        }
        skipped += 1;
    }
    if (!tried) return error.SuppressedHoldNeverTried;
    try std.testing.expect(skipped >= every / 2 and skipped <= 2 * every);
    try std.testing.expectEqual(asked + 1, script.asked);
    try std.testing.expect(subject.drafted > drafted and subject.accepted == kept); // a real trial, and it missed
    const missed = subject.depth.?.p[0];
    try std.testing.expect(missed < before);

    // right guesses again: the unprofitable trial backed the next off, and that one lands
    script.wrong = null;
    var later: u32 = 0;
    tried = false;
    for (0..160) |_| {
        if (try offered(&engine, subject) > 0) {
            tried = true;
            break;
        }
        later += 1;
    }
    if (!tried) return error.BackedOffHoldNeverTried;
    try std.testing.expect(later > skipped); // a longer wait than the first trial's
    try std.testing.expect(script.asked >= asked + 2); // the trial's own hold, and more once it pays
    try std.testing.expect(subject.accepted > kept); // the trial's drafts were kept
    try std.testing.expect(subject.depth.?.p[0] > missed); // the acceptance it measured

    // that measured acceptance pays the hold again, round after round
    var streak: u32 = 0;
    for (0..8) |_| {
        const was = subject.accepted;
        const got = try offered(&engine, subject);
        if (got == 0) break;
        try std.testing.expectEqual(got, subject.accepted - was);
        streak += 1;
    }
    try std.testing.expect(streak >= 6);
    try std.testing.expectEqual(live, engine.activeCount()); // not the only stream, and none finished

    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expectEqualSlices(u32, plain_subject, subject.emitted());
    if (live == 2) try std.testing.expectEqualSlices(u32, plain_companion, streams[1].emitted());
}

test "a hold an empirical cap stood aside is tried again by real periodic trials, and recovers by the acceptance they measure" {
    for ([_]usize{ 1, 2 }) |live| {
        for ([_]?u64{ null, 1234 }) |seed| try recovers(live, seed);
    }
}

/// What the subject's window carried in a round: a held head's drafts, a host copy's tokens, or neither.
const Kind = enum { plain, head, copy };

/// Reads each verify at the round loop's side of the drafter wrapper, where held drafts are still held.
const Tap = struct {
    var inner: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined;
    var subject: ?*sm.Stream = null;
    var seen: Kind = .plain;

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        for (windows) |w| if (w.stream == subject) {
            seen = if (w.held > 0) .head else if (w.tokens.len > 0) .copy else .plain;
        };
        return inner(ptr, windows, out);
    }
};

/// A priced proposer: while armed it offers the target's own next three tokens (they land), none gated.
const Copier = struct {
    armed: bool = false,
    sampling: ?Sampling = null,
    out: [3]u32 = undefined,

    fn propose(ptr: *anyopaque, context: []const u32, most: i64) anyerror![]const u32 {
        const c: *Copier = @ptrCast(@alignCast(ptr));
        if (!c.armed or most < 3) return &.{};
        var history: [64]u32 = undefined;
        @memcpy(history[0..context.len], context);
        for (&c.out, 0..) |*t, k| {
            t.* = fake.next(history[0 .. context.len + k], c.sampling, @intCast(context.len + k));
            history[context.len + k] = t.*;
        }
        return c.out[0..];
    }

    fn lastMatch(_: *anyopaque) i64 {
        return 0;
    }

    const vtable: Proposer.VTable = .{ .propose = propose, .last_match = lastMatch, .priced = true };
};

/// One step; the kind of window the subject verified in it.
fn stepped(engine: *Engine) !Kind {
    Tap.seen = .plain;
    try engine.step();
    return Tap.seen;
}

/// A real copy clears cached head chances after a first miss; real head probes must still resume from fallback rates.
fn copyBoundary(live: usize, seed: ?u64) !void {
    const subject_new = 260;
    const prompts = [2][]const u32{ &dt.p1, dt.long[0..11] };
    const plain_subject = try plainTokens(prompts[0], subject_new, seed);
    defer gpa.free(plain_subject);
    const plain_companion = try plainTokens(prompts[1], 520, seed);
    defer gpa.free(plain_companion);

    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    const holds: []const HoldCost = if (live == 1) &holds_one else &holds_two;
    var script: Script = .{ .head = .{ .facts_ = .{ .depth = 4, .step_ms = 0.5, .plain_guard = false, .batched = true, .chances = true, .hold_costs = holds } } };
    defer script.head.deinit();
    var wrapped = try Drafted.init(gpa, strict.backend(), script.drafter());
    defer wrapped.deinit();
    var model = scenarioModel();
    model.lone_copies = true; // a lone subject's copy window is verified too
    var cfg = try Config.init(gpa, wrapped.facts(model), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false; // rounds are priced by the synthetic tables alone
    var seen = wrapped.backend();
    var table = seen.vtable.*;
    Tap.inner = table.verify;
    table.verify = Tap.verify;
    seen.vtable = &table;
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, seen, clock.clock());
    defer engine.deinit();
    var copier: Copier = .{ .sampling = sampling(seed) };
    var streams: [2]sm.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    const subject = &streams[0];
    Tap.subject = subject;
    script.wrong = subject; // its first held draft is wrong: the first head trial misses
    for (0..live) |i| {
        const proposer: ?Proposer = if (i == 0) .{ .ptr = &copier, .vtable = &Copier.vtable } else null;
        streams[i] = try sm.Stream.init(gpa, .{ .id = if (i == 0) "subject" else "companion", .prompt = prompts[i], .max_new = if (i == 0) subject_new else 520, .sampling = sampling(seed), .proposer = proposer });
        made += 1;
        try engine.addStream(&streams[i]);
    }
    script.watched = subject;
    const every: u32 = @intCast(engine.rule.k.depth_probe_every);

    // the first conditional's trial misses: its measured acceptance stands the hold aside
    try std.testing.expectEqual(Kind.head, try stepped(&engine));
    try std.testing.expect(subject.drafted > 0 and subject.accepted == 0);
    try std.testing.expect(subject.granted.? > 0); // a positive verified grant, priced by the drafter's chances
    script.wrong = null; // later holds are right

    // a real copy window takes the next round: its plan drops the head chances the stream had cached
    copier.armed = true;
    try std.testing.expectEqual(Kind.copy, try stepped(&engine));
    copier.armed = false;

    // without them the hold is priced by the rule's rates: a probe interval of plain rounds, then a real head trial
    const asked = script.asked;
    var kept = subject.accepted;
    var waited: u32 = 0;
    var resumed = false;
    for (0..64) |_| {
        kept = subject.accepted;
        if (try stepped(&engine) == .head) {
            resumed = true;
            break;
        }
        waited += 1;
    }
    if (!resumed) return error.HeadNeverResumedAfterCopy;
    try std.testing.expect(waited >= every / 2 and waited <= 2 * every);
    try std.testing.expect(script.asked > asked); // the trial's own hold, asked after the copy
    try std.testing.expect(subject.accepted > kept); // and its drafts landed
    try std.testing.expectEqual(live, engine.activeCount());

    // the backed-off cadence keeps trying: more head trials follow while the cap binds; replies stay one-token decode
    var heads: u32 = 1;
    while (engine.activeCount() > 0) {
        if (try stepped(&engine) == .head) heads += 1;
    }
    try std.testing.expect(heads >= 2);
    try std.testing.expectEqualSlices(u32, plain_subject, subject.emitted());
    if (live == 2) try std.testing.expectEqualSlices(u32, plain_companion, streams[1].emitted());
}

test "a copy clearing a stream's cached head chances cannot starve its held-aside hold of real head trials" {
    for ([_]usize{ 1, 2 }) |live| {
        for ([_]?u64{ null, 1234 }) |seed| try copyBoundary(live, seed);
    }
}
