//! A drafter that prices its chains (Facts.chances): verify rows go by its chances and replies stay one-token decode.
const std = @import("std");
const Config = @import("config.zig").Config;
const Model = @import("config.zig").Model;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const be = @import("backend.zig");
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const dr = @import("drafter.zig");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const Drafted = @import("drafted.zig").Drafted;
const dt = @import("drafted_test.zig");

const gpa = std.testing.allocator;
const most_streams = 8;

/// drafted_test's drafter with a chance for each held draft: `sure` where its guess is right, `unsure` where wrong.
const Priced = struct {
    head: dt.FakeDrafter,
    sure: f64 = 0.7,
    unsure: f64 = 0.7,
    vtable: dr.Drafter.VTable = undefined,
    inner_hold: *const fn (*anyopaque, []const dr.Hold) anyerror!void = undefined,
    at: std.AutoHashMapUnmanaged(*sm.Stream, u64) = .empty, // each stream's last hold's position
    asked: u32 = 0,

    fn of(ptr: *anyopaque) *Priced {
        return @fieldParentPtr("head", @as(*dt.FakeDrafter, @ptrCast(@alignCast(ptr))));
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const p = of(ptr);
        for (items) |it| {
            try p.at.put(gpa, it.stream, it.position);
        }
        return p.inner_hold(ptr, items);
    }

    /// The running product along the chain, as the DSpark confidence head's survivals are.
    fn chances(ptr: *anyopaque, s: *sm.Stream, out: []f64) anyerror!bool {
        const p = of(ptr);
        const lane = p.head.lanes.getPtr(s) orelse return false;
        if (out.len > lane.held.items.len) return error.TooFewDrafts;
        const position = p.at.get(s) orelse return error.NoHold;
        p.asked += 1;
        var reach: f64 = 1;
        for (out, 0..) |*o, j| {
            const at = position + j;
            reach *= if ((at *% 2654435761 + j) % 5 == 0) p.unsure else p.sure; // drafted_test's wrong guesses
            o.* = reach;
        }
        return true;
    }

    fn drafter(p: *Priced) dr.Drafter {
        const base_ = p.head.drafter();
        p.vtable = base_.vtable.*;
        p.inner_hold = base_.vtable.hold;
        p.vtable.hold = hold;
        p.vtable.chances = chances;
        return .{ .ptr = &p.head, .vtable = &p.vtable };
    }

    fn deinit(p: *Priced) void {
        p.at.deinit(gpa);
        p.head.deinit();
    }
};

/// Every shared verify's window rows: whether one round gave its streams unequal rows (a ragged verify).
const Seen = struct {
    var inner: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined;
    var ragged: u32 = 0;
    var rows: u64 = 0;

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        for (windows) |w| {
            rows += w.rows();
            if (w.rows() != windows[0].rows()) {
                ragged += 1;
                break;
            }
        }
        return inner(ptr, windows, out);
    }
};

var costs: [16]Cost = undefined;

fn base() Model {
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return .{ .exact_width = 16, .gpu_tokens = true, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = most_streams };
}

const Run = struct { tokens: [most_streams][64]u32 = undefined, lens: [most_streams]usize = @splat(0), drafted: u64 = 0, holds: u32 = 0, asked: u32 = 0, ragged: u32 = 0, rows: u64 = 0 };

/// The drafter's Facts.slack for following runs; fresh reservations still cap each hold.
var slack: u32 = 0;

/// The drafter's chances: equal everywhere, or knowing its wrong guesses (a confidence head that ranks well).
const Head = enum { flat, knows };

/// Each prompt decoded together (one stream: alone); `floor` null: the chances price rows, else a fixed window.
fn run(prompts: []const []const u32, drafts: bool, floor: ?f64, head: Head) !Run {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var p: Priced = .{ .head = .{ .facts_ = .{ .depth = 9, .step_ms = 0.5, .batched = true, .chances = true, .confidence_floor = floor, .slack = slack } } };
    if (head == .knows) {
        p.sure = 0.95;
        p.unsure = 0.05;
    }
    defer p.deinit();
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var seen = strict.backend();
    var table = seen.vtable.*;
    Seen.inner = table.verify;
    Seen.ragged = 0;
    Seen.rows = 0;
    table.verify = Seen.verify;
    seen.vtable = &table;
    var wrapped = try Drafted.init(gpa, seen, p.drafter());
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, if (drafts) wrapped.facts(base()) else base(), 16, 15);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, if (drafts) wrapped.backend() else target.backend(), clock.clock());
    defer engine.deinit();
    var streams: [most_streams]sm.Stream = undefined;
    const n = prompts.len;
    for (prompts, streams[0..n]) |prompt, *s| s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = prompt, .max_new = 48, .eos = &.{96}, .drafts = drafts });
    defer for (streams[0..n]) |*s| s.deinit(gpa);
    for (streams[0..n]) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    var out: Run = .{ .holds = p.head.holds, .asked = p.asked, .ragged = Seen.ragged, .rows = Seen.rows };
    for (streams[0..n], 0..) |*s, i| {
        out.lens[i] = s.emitted().len;
        @memcpy(out.tokens[i][0..out.lens[i]], s.emitted());
        out.drafted += s.drafted;
    }
    return out;
}

fn expectSame(a: Run, b: Run, n: usize) !void {
    for (0..n) |i| try std.testing.expectEqualSlices(u32, a.tokens[i][0..a.lens[i]], b.tokens[i][0..b.lens[i]]);
}

/// `n` streams of unequal prompts, each a prefix of drafted_test's long prompt.
fn unequal(buf: *[most_streams][]const u32, n: usize) []const []const u32 {
    for (buf[0..n], 0..) |*p, i| p.* = dt.long[0 .. 11 + i];
    return buf[0..n];
}

test "a drafter's chances price its chain's rows, and drafted replies equal one-token decode alone and shared" {
    const sets = [_][]const []const u32{ &.{&dt.p1}, &.{ &dt.p1, &dt.long, &dt.p1 } };
    for (sets) |ps| {
        const plain = try run(ps, false, null, .flat);
        const priced = try run(ps, true, null, .flat);
        try expectSame(plain, priced, ps.len);
        try std.testing.expect(priced.asked > 0 and priced.drafted > 0); // the window allocator read the chances
        const fixed = try run(ps, true, 0, .flat);
        try expectSame(plain, fixed, ps.len);
        try std.testing.expect(fixed.drafted > priced.drafted); // a fixed window verifies every draft the rows fit
    }
}

test "at 1, 4 and 8 streams the scheduled and the fixed verify equal one-token decode; shared rounds are ragged" {
    var buf: [most_streams][]const u32 = undefined;
    for ([_]usize{ 1, 4, 8 }) |n| {
        const ps = unequal(&buf, n);
        const plain = try run(ps, false, null, .flat);
        const scheduled = try run(ps, true, null, .knows);
        try expectSame(plain, scheduled, n);
        const fixed = try run(ps, true, 0, .knows);
        try expectSame(plain, fixed, n);
        try std.testing.expect(scheduled.asked > 0 and scheduled.drafted > 0);
        // a confident head's survivals cut each stream's chain where its guess goes wrong: fewer rows verified
        try std.testing.expect(scheduled.rows < fixed.rows);
        if (n > 1) try std.testing.expect(scheduled.ragged > 0); // streams' verify lengths differ within a round
    }
}

test "the drafter's chances reach the lane facts: rows by chances and holds by grants, or a fixed window" {
    var p: Priced = .{ .head = .{ .facts_ = .{ .depth = 9, .step_ms = 0.5, .chances = true } } };
    defer p.deinit();
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var wrapped: Drafted = .{ .gpa = gpa, .target = target.backend(), .drafter = p.drafter() };
    defer wrapped.deinit();
    const m = wrapped.facts(base());
    try std.testing.expect(m.draft_probabilities and m.granted_depth and m.draft_confidence == null);
    var cfg = try Config.init(gpa, m, 16, 15);
    defer cfg.deinit(gpa);
    try std.testing.expectEqual(@as(i64, 2), cfg.k.draft_slack); // the depth rule's own
    try std.testing.expect(cfg.k.grant_slack == 1 and cfg.k.short_slack == 0); // a granted width's own
    p.head.facts_.slack = 1;
    var one = try Config.init(gpa, wrapped.facts(base()), 16, 15);
    defer one.deinit(gpa);
    try std.testing.expectEqual(@as(i64, 1), one.k.draft_slack); // a drafter's own slack past its grant
    try std.testing.expect(one.k.grant_slack == 1 and one.k.short_slack == 1); // short or not
    p.head.facts_.confidence_floor = 0;
    const f = wrapped.facts(base());
    try std.testing.expect(!f.draft_probabilities and !f.granted_depth and f.draft_confidence.? == 0);
    p.head.facts_.chances = false;
    const none = wrapped.facts(base());
    try std.testing.expect(!none.draft_probabilities and none.draft_confidence == null);
}

test "a granted depth: the deepest chain first, then the grant plus its slack, and after a grant of none a probe" {
    const empty: Table = .{};
    var rule: depth.Rule = .{ .gpa = gpa, .prior = &.{0.8}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = true, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
    defer rule.deinit();
    var st: ?depth.State = null;
    defer if (st) |*s| s.deinit(gpa);
    const No = struct {
        fn copy(_: *anyopaque, _: *const depth.Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &unused, .copy = No.copy };
    var who: depth.Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    try std.testing.expectEqual(@as(i64, 9), try rule.headDepth(&who, null, probe));
    who.granted = 3;
    try std.testing.expectEqual(@as(i64, 4), try rule.headDepth(&who, null, probe)); // every held draft granted: one more
    who.grant_short = true;
    try std.testing.expectEqual(@as(i64, 3), try rule.headDepth(&who, null, probe)); // drafts left unverified: the grant
    who.grant_short = false;
    who.granted = 8;
    try std.testing.expectEqual(@as(i64, 9), try rule.headDepth(&who, null, probe));
    rule.k.grant_slack = 2; // a drafter's slack sets both
    rule.k.short_slack = 2;
    who.grant_short = true;
    who.granted = 3;
    try std.testing.expectEqual(@as(i64, 5), try rule.headDepth(&who, null, probe));
    who.granted = 0;
    rule.k.depth_probe_every = 4;
    rule.k.plain_wait_most = 16;
    for ([_][]const depth.HoldCost{ &.{}, &.{.{ .streams = 1, .depth = 1, .ms = 1000 }} }) |holds| {
        rule.k.hold_costs = holds;
        st.?.plain = 0;
        st.?.wait = null;
        for (0..3) |_| try std.testing.expectEqual(@as(i64, 0), try rule.headDepth(&who, null, probe));
        try std.testing.expect(try rule.headDepth(&who, null, probe) > 0); // scalar callers own their probe clock
    }
}

test "chance-priced holds preserve one-token replies across stream counts and slack choices" {
    var buf: [most_streams][]const u32 = undefined;
    defer slack = 0;
    for ([_]usize{ 1, 4, 8 }) |n| {
        const ps = unequal(&buf, n);
        const plain = try run(ps, false, null, .flat);
        for ([_]Head{ .flat, .knows }) |head| {
            slack = 0;
            const narrow = try run(ps, true, null, head);
            slack = 2;
            const two = try run(ps, true, null, head);
            slack = 9;
            const whole = try run(ps, true, null, head);
            try expectSame(plain, narrow, n);
            try expectSame(plain, two, n);
            try expectSame(plain, whole, n);
            try std.testing.expect(narrow.drafted > 0);
        }
    }
}

test "hold budgeting queries a stateful copy proposer once for a granted depth" {
    const empty: Table = .{};
    var rule: depth.Rule = .{ .gpa = gpa, .prior = &.{0.8}, .most_drafts = 9, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 1, .plain_guard = true, .node_probabilities = true, .granted_depth = true, .batch_rows = 32 };
    defer rule.deinit();
    var state: ?depth.State = null;
    defer if (state) |*s| s.deinit(gpa);
    var whos = [_]depth.Who{.{ .state = &state, .draft_room = 64, .finished = false, .forced = false, .granted = 1 }};
    const Changing = struct {
        fn copy(ptr: *anyopaque, _: *const depth.Who) bool {
            const calls: *usize = @ptrCast(@alignCast(ptr));
            calls.* += 1;
            return calls.* > 1;
        }
    };
    var calls: usize = 0;
    var out: [1]i64 = undefined;
    try rule.budgets(&whos, .{ .ptr = &calls, .copy = Changing.copy }, false, &out);
    try std.testing.expectEqual(@as(usize, 1), calls);
    try std.testing.expectEqual(@as(i64, 2), out[0]);
}
