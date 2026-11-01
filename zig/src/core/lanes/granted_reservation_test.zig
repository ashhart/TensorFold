//! Shared head reservations under granted depths: bounds, zero rows, hold prices, probes and the proposer's one ask.
const std = @import("std");
const lanes = @import("lanes.zig");
const dt = @import("drafted_test.zig");
const win = lanes.windows;
const t = std.testing;
const a = t.allocator;
const Cost = lanes.config.Cost;

const Options = struct { batch_rows: u32 = 32, plain_guard: bool = false, hold_ms: ?f64 = null, legacy: bool = false };

/// A shared round's head streams with chosen last grants and cached chances; a row costs 9 + width ms, overhead 8 ms.
const Setup = struct {
    costs: [16]Cost = undefined,
    holds: [1]lanes.depth.HoldCost = undefined,
    cfg: lanes.Config = undefined,
    target: lanes.fake.Fake = undefined,
    clock: lanes.fake.FixedClock = .{},
    e: lanes.Engine = undefined,
    streams: [3]lanes.Stream = undefined,
    plans: [3]win.Plan = undefined,
    heads: [3]usize = .{ 0, 1, 2 },
    out: [3]i64 = undefined,
    n: usize = 0,

    fn init(s: *Setup, o: Options, n: usize) !void {
        for (&s.costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 9 + @as(f64, @floatFromInt(w)) };
        var m: lanes.Model = .{ .exact_width = 16, .drafts = 7, .mtp = true, .speculate = true, .speculate_early = false, .allocate_shared_drafts = true, .window_costs = &s.costs, .hidden_rows = true, .batch_rows = o.batch_rows, .max_streams = 4, .plain_guard = o.plain_guard, .draft_streams = true };
        if (o.legacy) {
            m.limit_shared_drafts = true;
            m.mtp_step_ms = 0.5;
        } else {
            m.draft_probabilities = true;
            m.granted_depth = true;
        }
        if (o.hold_ms) |ms| {
            s.holds = .{.{ .streams = 2, .depth = 1, .ms = ms }};
            m.hold_costs = &s.holds;
        }
        s.cfg = try lanes.Config.init(a, m, 16, 15);
        s.cfg.timed = false;
        s.cfg.k.grant_slack = 1;
        s.cfg.k.short_slack = 0;
        s.cfg.k.draft_slack = 2;
        s.target = .{ .gpa = a };
        s.e = lanes.Engine.init(a, &s.cfg, s.target.backend(), s.clock.clock());
        s.e.alone = false;
        try s.e.rule.observeOverhead(@intCast(n), @intCast(n), s.costs[n - 1].ms + 8, false, 0);
        for (0..n) |i| {
            s.streams[i] = try lanes.Stream.init(a, .{ .id = "reserve", .prompt = &.{ 1, 2, 3 }, .max_new = 128 });
            s.n += 1;
            s.plans[i] = .{ .stream = &s.streams[i], .position = 0, .kind = .none };
        }
    }

    fn deinit(s: *Setup) void {
        for (s.streams[0..s.n]) |*x| x.deinit(a);
        s.e.deinit();
        s.cfg.deinit(a);
        s.target.deinit();
    }

    /// Stream `i`'s last verified grant, whether it left held drafts unverified, and the drafts it expected to keep.
    fn history(s: *Setup, i: usize, granted: ?u32, short: bool, gain: ?f64) void {
        s.streams[i].granted = granted;
        s.streams[i].grant_short = short;
        s.streams[i].grant_gain = gain;
    }

    /// The chances the stream's last full head round priced, `head` then `tail` for every depth past it.
    fn chain(s: *Setup, i: usize, head: []const f64, tail: f64) !void {
        const c = &s.streams[i].draft_chances;
        try c.appendSlice(a, head);
        while (c.items.len < s.cfg.most_drafts) try c.append(a, tail);
    }

    fn budgets(s: *Setup) ![]const i64 {
        _ = s.e.arena.reset(.retain_capacity);
        try lanes.shared_drafts.budgets(&s.e, s.plans[0..s.n], s.heads[0..s.n], s.out[0..s.n]);
        return s.out[0..s.n];
    }
};

test "a reservation stays within the stream's last grant plus its slack, the deepest chain only before any grant" {
    var s: Setup = .{};
    try s.init(.{}, 3);
    defer s.deinit();
    s.history(0, 2, false, null); // every held draft was verified: one more
    try s.chain(0, &.{}, 0.9);
    s.history(1, 2, true, null); // drafts were left unverified: the grant
    try s.chain(1, &.{}, 0.8);
    try s.chain(2, &.{}, 0.7); // never granted
    try t.expectEqualSlices(i64, &.{ 3, 2, 7 }, try s.budgets());
}

test "a fresh reservation of none stays none: no slack, probe or hold price brings the hold back" {
    for ([_]?f64{ null, 1 }) |hold_ms| {
        var s: Setup = .{};
        try s.init(.{ .hold_ms = hold_ms }, 2);
        defer s.deinit();
        s.history(0, 0, false, null); // its probe may offer two drafts every round, but none pays now
        try s.chain(0, &.{}, 0.01);
        s.history(1, 3, false, 2.7);
        try s.chain(1, &.{ 0.95, 0.9, 0.85, 0.8 }, 0.75);
        for (0..12) |_| try t.expectEqualSlices(i64, &.{ 0, 4 }, try s.budgets());
    }
    var gone: Setup = .{};
    try gone.init(.{ .hold_ms = 1 }, 1);
    defer gone.deinit();
    gone.history(0, 3, false, 2.7); // a positive grant is no claim on rows that no longer pay
    try gone.chain(0, &.{}, 0.01);
    try t.expectEqualSlices(i64, &.{0}, try gone.budgets());
}

test "a hold is priced at the rows and drafts reserved now, not at its last grant's" {
    // one stream: its round costs 10 ms plain, 8 ms overhead and 20 ms for a hold pass
    const Case = struct { granted: u32, gain: f64, chain: []const f64, reserved: i64, kept: i64 };
    const cases = [_]Case{
        // a rich grant, but only one draft pays now: 1.5 expected tokens in 39 ms lose to 1 in 18
        .{ .granted = 4, .gain = 3.6, .chain = &.{ 0.5, 0.05, 0.05, 0.05, 0.05 }, .reserved = 1, .kept = 0 },
        // a poor grant, but its two drafts pay now: 2.85 expected tokens in 40 ms beat 1 in 18
        .{ .granted = 2, .gain = 0.1, .chain = &.{ 0.95, 0.9, 0.85 }, .reserved = 3, .kept = 3 },
    };
    for (cases) |c| {
        var plain: Setup = .{};
        try plain.init(.{}, 1);
        defer plain.deinit();
        plain.history(0, c.granted, false, c.gain);
        try plain.chain(0, c.chain, 0.05);
        try t.expectEqualSlices(i64, &.{c.reserved}, try plain.budgets()); // no hold price: the reservation itself
        var priced: Setup = .{};
        try priced.init(.{ .hold_ms = 20 }, 1);
        defer priced.deinit();
        priced.history(0, c.granted, false, c.gain);
        try priced.chain(0, c.chain, 0.05);
        try t.expectEqualSlices(i64, &.{c.kept}, try priced.budgets());
    }
}

test "a historical-zero trial needs a viable pass and a positive fresh reservation" {
    for ([_]f64{ 0.9, 0.01 }) |chance| {
        var s: Setup = .{};
        try s.init(.{ .hold_ms = 1000 }, 1);
        defer s.deinit();
        s.history(0, 0, false, null);
        try s.chain(0, &.{}, chance);
        try t.expectEqualSlices(i64, &.{0}, try s.budgets()); // even perfect acceptance cannot pay
        s.holds[0].ms = 1; // a changed live cost must release the saved trial immediately
        const probe: i64 = if (chance > 0.5) 2 else 0;
        try t.expectEqualSlices(i64, &.{probe}, try s.budgets());
    }
}

test "a historical zero grant consumes one probe tick per round at the configured interval" {
    var s: Setup = .{};
    try s.init(.{ .plain_guard = true, .hold_ms = 1 }, 1);
    defer s.deinit();
    s.e.rule.k.depth_probe_every = 5;
    s.history(0, 0, false, null);
    try s.chain(0, &.{}, 0.9);
    for (1..13) |round| {
        const probe: i64 = if (round == 5) 2 else 0;
        try t.expectEqualSlices(i64, &.{probe}, try s.budgets());
    }
}

test "the shared hold's cap is cut over the depths reserved now" {
    var s: Setup = .{};
    try s.init(.{ .batch_rows = 8 }, 3); // three blocks of the deepest stream's three drafts would not fit eight rows
    defer s.deinit();
    try s.chain(0, &.{ 0.95, 0.9, 0.85 }, 0.1);
    try s.chain(1, &.{0.8}, 0.1);
    try s.chain(2, &.{0.7}, 0.1);
    try t.expectEqualSlices(i64, &.{ 2, 1, 1 }, try s.budgets());
}

test "the drafter's own chances decide a bounded reservation, not the generic rates" {
    var s: Setup = .{};
    try s.init(.{}, 1);
    defer s.deinit();
    @memset(try s.e.rule.rates(win.who(&s.streams[0])), 0.95); // rates that would carry every draft
    s.history(0, 6, false, null);
    try s.chain(0, &.{ 0.9, 0.3, 0.05, 0.04, 0.03, 0.02 }, 0.01); // a second draft this seldom landing, then none
    try t.expectEqualSlices(i64, &.{2}, try s.budgets());
}

const code = [_]u32{ 11, 12, 13, 14 };
const Ask = struct { calls: usize = 0 };

/// Copies at the first ask only, as a proposer whose lookup advances its own backoff does.
fn propose(ptr: *anyopaque, _: []const u32, _: i64) ![]const u32 {
    const ask: *Ask = @ptrCast(@alignCast(ptr));
    ask.calls += 1;
    return if (ask.calls == 1) &code else &.{};
}

fn match(_: *anyopaque) i64 {
    return 32;
}

test "a stateful copy proposer is asked once a round, and its copy takes the round's hold" {
    for ([_]bool{ false, true }) |legacy| {
        var s: Setup = .{};
        try s.init(.{ .legacy = legacy }, 2);
        defer s.deinit();
        var ask: Ask = .{};
        s.streams[0].proposer = .{ .ptr = &ask, .vtable = &.{ .propose = propose, .last_match = match, .priced = true } };
        const out = try s.budgets();
        try t.expectEqual(@as(usize, 1), ask.calls);
        try t.expectEqual(@as(i64, 0), out[0]);
    }
}

/// The drafter of drafted_test, pricing every held chain by 0.9 a draft and holding each hold to the grant's bound.
const Scripted = struct {
    head: dt.FakeDrafter,
    vtable: lanes.drafter.Drafter.VTable = undefined,
    inner_hold: *const fn (*anyopaque, []const lanes.drafter.Hold) anyerror!void = undefined,
    watched: ?*lanes.Stream = null,
    log: std.ArrayList(u32) = .empty, // the watched stream's hold depths
    most: u32 = 4,

    fn of(ptr: *anyopaque) *Scripted {
        return @fieldParentPtr("head", @as(*dt.FakeDrafter, @ptrCast(@alignCast(ptr))));
    }

    /// A hold is at most the stream's last grant plus its slack (two after a grant of none), the deepest before any.
    fn hold(ptr: *anyopaque, items: []const lanes.drafter.Hold) anyerror!void {
        const x = of(ptr);
        for (items) |it| {
            const s = it.stream;
            const bound: u32 = if (s.granted) |g| (if (g == 0) @min(x.most, 2) else @min(x.most, g + @intFromBool(!s.grant_short))) else x.most;
            try t.expect(it.depth <= bound);
            const history = x.head.lanes.get(s).?.history.items;
            try t.expectEqualSlices(u32, s.context.items[0..@intCast(it.position - 1)], history);
            if (s == x.watched) try x.log.append(a, it.depth);
        }
        return x.inner_hold(ptr, items);
    }

    fn chances(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const lane = of(ptr).head.lanes.getPtr(s) orelse return false;
        if (out.len > lane.held.items.len) return error.TooFewDrafts;
        var reach: f64 = 1;
        for (out) |*o| {
            reach *= 0.9;
            o.* = reach;
        }
        return true;
    }

    fn drafter(x: *Scripted) lanes.drafter.Drafter {
        const inner = x.head.drafter();
        x.vtable = inner.vtable.*;
        x.inner_hold = inner.vtable.hold;
        x.vtable.hold = hold;
        x.vtable.chances = chances;
        return .{ .ptr = &x.head, .vtable = &x.vtable };
    }
};

/// Forwards for up to 16 rows cost 1 ms; a row past them 1000 ms.
fn wall(buf: *[32]Cost) void {
    for (buf, 1..) |*c, r| c.* = .{ .width = @intCast(r), .ms = if (r <= 16) 1 else 1000 };
}

fn spec(seed: ?u64) ?lanes.Sampling {
    return if (seed) |v| .{ .seed = v, .temperature = 1.0, .top_p = 0.95 } else null;
}

/// What a stream decodes one token a step, no drafter.
fn plainTokens(seed: ?u64) ![]u32 {
    var costs: [32]Cost = undefined;
    wall(&costs);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 16, .lone_copies = false }, 16, 15);
    defer cfg.deinit(a);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer e.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "plain", .prompt = &.{ 7, 13, 29 }, .max_new = 260, .drafts = false, .sampling = spec(seed) });
    defer s.deinit(a);
    try e.addStream(&s);
    while (e.activeCount() > 0) try e.step();
    return a.dupe(u32, s.emitted());
}

fn zeroThenPositive(seed: ?u64) !void {
    const plain = try plainTokens(seed);
    defer a.free(plain);
    var costs: [32]Cost = undefined;
    wall(&costs);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var head: Scripted = .{ .head = .{ .facts_ = .{ .depth = 4, .step_ms = 0, .plain_guard = false, .batched = true, .chances = true } } };
    defer head.head.deinit();
    defer head.log.deinit(a);
    var wrapped = try lanes.drafted.Drafted.init(a, strict.backend(), head.drafter());
    defer wrapped.deinit();
    const m = wrapped.facts(.{ .exact_width = 16, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 16, .lone_copies = false });
    var cfg = try lanes.Config.init(a, m, 16, 15);
    defer cfg.deinit(a);
    cfg.k.grant_slack = 1;
    cfg.k.short_slack = 0;
    cfg.k.draft_slack = 2;
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(a, &cfg, wrapped.backend(), clock.clock());
    defer e.deinit();
    var streams: [16]lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (&streams, 0..) |*s, i| {
        s.* = try lanes.Stream.init(a, .{ .id = "wall", .prompt = &.{ 7, 13, 29 }, .max_new = if (i == 0) 260 else 180, .sampling = spec(seed) });
        made += 1;
        try e.addStream(s);
    }
    head.watched = &streams[0];
    // sixteen streams fill the 16 rows that cost 1 ms: every held draft is cut, and no hold is asked for after it
    const holds = head.head.holds;
    try e.step();
    try t.expectEqual(@as(?u32, 0), streams[0].granted);
    try t.expectEqual(holds, head.head.holds);
    head.log.clearRetainingCapacity();
    var skipped: usize = 0;
    while (!streams[1].finished) {
        const before = head.head.holds;
        const scored = head.head.proposed_absorbs;
        try e.step();
        if (head.head.holds != before) continue;
        try t.expectEqual(scored, head.head.proposed_absorbs); // a round without a hold computes no proposal either
        skipped += 1;
    }
    try t.expect(skipped >= 128);
    while (e.activeCount() > 0) try e.step();
    // once alone the stream's holds grow from the probe's two by a draft a round, as its grants verify whole
    try t.expect(std.mem.startsWith(u32, head.log.items, &.{ 2, 3, 4 }));
    try t.expectEqualSlices(u32, plain, streams[0].emitted());
    for (streams[1..]) |*s| try t.expectEqualSlices(u32, plain[0..s.emitted().len], s.emitted());
}

test "granted depths over a zero-row interval: no hold or proposal while none fits, then a bounded one, tokens as plain" {
    for ([_]?u64{ null, 1234 }) |seed| try zeroThenPositive(seed);
}
