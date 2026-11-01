//! Window and hold pricing: sampled draws stay local, and paid holds stay out of fixed overhead.
const std = @import("std");
const Config = @import("config.zig").Config;
const Cost = @import("config.zig").Cost;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const be = @import("backend.zig");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;


const gpa = std.testing.allocator;

/// A round's ms: the model's window cost for its rows, plus 3 ms a row of host draws when a stream in it samples.
const RowClock = struct {
    target: *const fake.Fake,

    fn clock(c: *RowClock) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(_: *anyopaque) void {}

    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c: *RowClock = @ptrCast(@alignCast(ptr));
        const rows: f64 = @floatFromInt(c.target.last_rows);
        return 5.0 + 0.8 * rows + (if (c.target.last_sampled) 3.0 * rows else 0.0);
    }
};

/// The greedy stream's rounds and drafted rows after `sampled` sampled streams ran on the same engine.
fn greedyAfter(node_probabilities: bool, sampled: usize) ![2]u64 {
    var costs: [16]Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try Config.init(gpa, .{ .exact_width = 16, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 8, .window_costs = &costs, .mtp_step_ms = 0.5, .draft_probabilities = node_probabilities }, 16, 15);
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: RowClock = .{ .target = &target };
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const prompts = [_][]const u32{ &.{ 2, 7, 1, 8, 2, 8, 1, 8 }, &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5 } };
    for (0..sampled + 1) |i| {
        const greedy = i == sampled;
        var s = try sm.Stream.init(gpa, .{ .id = "s", .prompt = prompts[@intFromBool(greedy)], .max_new = 96, .sampling = if (greedy) null else .{ .seed = 5 + i, .temperature = 0.8 } });
        defer s.deinit(gpa);
        try engine.addStream(&s);
        while (engine.activeCount() > 0) try engine.step();
        if (greedy) return .{ s.rounds, s.drafted };
    }
    unreachable;
}

test "a greedy stream after sampled streams drafts as on a fresh engine, by depth or by node chances" {
    for ([_]bool{ false, true }) |node_probabilities| {
        const fresh = try greedyAfter(node_probabilities, 0);
        const after = try greedyAfter(node_probabilities, 2);
        try std.testing.expectEqual(fresh, after);
        try std.testing.expect(fresh[1] > fresh[0]); // it drafted
    }
}

test "observed padded holds do not make unpaid holds look profitable" {
    var shared: Table = .{};
    defer shared.deinit(gpa);
    for (1..7) |rows| try shared.put(gpa, @intCast(rows), 10);
    const holds = [_]depth.HoldCost{
        .{ .streams = 1, .depth = 1, .ms = 4 },
        .{ .streams = 2, .depth = 1, .ms = 4 },
        .{ .streams = 2, .depth = 3, .ms = 6 },
        .{ .streams = 3, .depth = 3, .ms = 24 },
    };
    const depths = [_]i64{ 1, 3, 0 };
    for ([_]bool{ false, true }) |sampled| for ([_]f64{ 0, 2.5 }) |host_ms| {
        var rule: depth.Rule = .{
            .gpa = gpa,
            .k = .{ .hold_costs = &holds },
            .prior = &.{0.75},
            .most_drafts = 3,
            .family_costs = &shared,
            .shared_costs = &shared,
            .mtp_step_ms = 1,
            .batched_heads = true,
            .plain_guard = true,
            .node_probabilities = true,
            .granted_depth = true,
            .batch_rows = 32,
        };
        defer rule.deinit();
        var states: [3]?depth.State = @splat(null);
        var who: [3]depth.Who = undefined;
        for (&who, &states, 0..) |*w, *state, i| w.* = .{
            .state = state,
            .draft_room = 32,
            .finished = false,
            .forced = false,
            .granted = if (i < 2) 1 else 0,
            .gain = if (i < 2) 0.75 else 0,
        };
        // The actual padded pass costs 6 ms; its linear head-step estimate is only 3 ms.
        try rule.observeOverhead(3, 5, 10 + host_ms + 6, sampled, rule.headMs(&depths));
        var out = depths;
        try rule.payHolds(&who, sampled, &out, price.Base.pending(who.len));
        const want: [3]i64 = if (host_ms == 0) .{ 0, 0, 0 } else depths;
        try std.testing.expectEqualSlices(i64, &want, &out);
    };
}

/// Synthetic timing fixture, not device or performance evidence: plain 10 ms, rows 11 ms, hold 5 ms (1 deep) / 20 (3).
const hold_costs = [_]depth.HoldCost{
    .{ .streams = 1, .depth = 1, .ms = 5 },
    .{ .streams = 1, .depth = 3, .ms = 20 },
};

fn noCopy(_: *anyopaque, _: *const depth.Who) bool {
    return false;
}

/// One stream (granted drafts, expected kept `gain`); `round` is the engine's reserve, pay, observe-if-offered step.
const HoldRun = struct {
    const Setup = struct { sampled: bool, granted: u32, gain: f64, prior: []const f64, k: depth.Constants };

    shared: Table = .{},
    state: ?depth.State = null,
    rule: depth.Rule = undefined,
    who: depth.Who = undefined,

    fn init(h: *HoldRun, s: Setup) !void {
        var k = s.k;
        k.hold_costs = &hold_costs;
        h.rule = .{
            .gpa = gpa,
            .k = k,
            .prior = s.prior,
            .most_drafts = 3,
            .family_costs = &h.shared,
            .shared_costs = &h.shared,
            .mtp_step_ms = 1,
            .batched_heads = true,
            .plain_guard = true,
            .node_probabilities = true,
            .granted_depth = true,
            .batch_rows = 32,
        };
        h.who = .{
            .state = &h.state,
            .draft_room = 32,
            .finished = false,
            .forced = false,
            .granted = s.granted,
            .gain = s.gain,
            .sampled = s.sampled,
        };
        try h.shared.put(gpa, 1, 10);
        for (2..5) |rows| try h.shared.put(gpa, @intCast(rows), 11);
        const granted: i64 = s.granted;
        const head = h.rule.headMs(&.{granted});
        try h.rule.observeOverhead(1, s.granted + 1, 11 + head, s.sampled, head);
    }

    fn deinit(h: *HoldRun) void {
        h.rule.deinit();
        if (h.state) |*s| s.deinit(gpa);
        h.shared.deinit(gpa);
    }

    /// One round's offered drafts, `accepted` of them kept; `fresh_zero`: the window reserved no rows.
    fn round(h: *HoldRun, accepted: i64, fresh_zero: bool) !i64 {
        const probe: depth.Probe = .{ .ptr = h, .copy = noCopy };
        var out = [_]i64{try h.rule.reserveDepth(&h.who, null, probe)};
        if (fresh_zero) out[0] = 0;
        try h.rule.payHolds(&.{h.who}, h.who.sampled, &out, price.Base.pending(1));
        if (out[0] > 0) try h.rule.observeDepth(h.who, out[0], @min(accepted, out[0]));
        return out[0];
    }

    /// The rounds (0-based) that offered drafts, the i-th kept `script[i]` of them (the last entry repeats).
    fn offered(h: *HoldRun, script: []const i64, at: []usize, limit: usize) !usize {
        var n: usize = 0;
        for (0..limit) |t| {
            if (n == at.len) break;
            if (try h.round(script[@min(n, script.len - 1)], false) > 0) {
                at[n] = t;
                n += 1;
            }
        }
        return n;
    }
};

test "a first successful conditional never lowers a profitable confident hold, and untouched suffix keeps its confidence" {
    // Synthetic: a 3-draft grant of 2.8 pays (3.8 tokens in 31 ms vs 1 in 10) at acceptance >= 0.7; default rate 0.15.
    const Case = struct { offered: i64, accepted: i64, prior: []const f64, kept: i64 };
    const cases = [_]Case{
        // one success, two untried
        .{ .offered = 1, .accepted = 1, .prior = &.{ 0.5, 0.5, 0.5 }, .kept = 3 },
        // every conditional succeeded once
        .{ .offered = 3, .accepted = 3, .prior = &.{ 0.5, 0.5, 0.5 }, .kept = 3 },
        // untried third keeps its confidence despite its low prior
        .{ .offered = 2, .accepted = 2, .prior = &.{ 0.95, 0.05, 0.05 }, .kept = 3 },
        // a real first miss stands the hold aside
        .{ .offered = 3, .accepted = 0, .prior = &.{ 0.5, 0.5, 0.5 }, .kept = 0 },
        // a real second miss does too
        .{ .offered = 2, .accepted = 1, .prior = &.{ 0.95, 0.05, 0.05 }, .kept = 0 },
    };
    for ([_]bool{ false, true }) |sampled| for (cases) |c| {
        var h: HoldRun = .{};
        defer h.deinit();
        try h.init(.{ .sampled = sampled, .granted = 3, .gain = 2.8, .prior = c.prior, .k = .{} });
        var out = [_]i64{3};
        try h.rule.payHolds(&.{h.who}, sampled, &out, price.Base.pending(1));
        try std.testing.expectEqualSlices(i64, &.{3}, &out); // before any head trial the grant's confidence holds
        try h.rule.observeDepth(h.who, c.offered, c.accepted);
        out = .{3};
        try h.rule.payHolds(&.{h.who}, sampled, &out, price.Base.pending(1));
        try std.testing.expectEqualSlices(i64, &.{c.kept}, &out);
    };
}

test "a suppressed hold recovers by real probe trials on the periodic cadence, then its backoff resets" {
    // Synthetic: a 1-draft grant pays at acceptance >= 0.6 (0.5 misses, 0.75 pays); probes every 4, wait doubles to 16.
    const every: usize = 4;
    for ([_]bool{ false, true }) |sampled| {
        var h: HoldRun = .{};
        defer h.deinit();
        try h.init(.{ .sampled = sampled, .granted = 1, .gain = 0.99, .prior = &.{0.5}, .k = .{ .depth_rate = 0.5, .depth_probe_every = every, .plain_wait_most = 16, .grant_slack = 0 } });
        var at: [6]usize = undefined;
        try std.testing.expectEqual(@as(usize, at.len), try h.offered(&.{ 0, 0, 1, 1, 0, 0 }, &at, 200));
        try std.testing.expectEqual(@as(usize, 0), at[0]); // the confident grant is first offered and misses
        const first = at[1] - at[0]; // then plain rounds until the first probe
        try std.testing.expect(first == every or first == every + 1);
        try std.testing.expectEqual(@as(usize, 8), at[2] - at[1]); // the miss doubled the wait
        try std.testing.expectEqual(@as(usize, 16), at[3] - at[2]); // 0.5 still does not pay: wait doubled
        try std.testing.expectEqual(@as(usize, 1), at[4] - at[3]); // 0.75 pays the very next round
        try std.testing.expectEqual(first, at[5] - at[4]); // the paid hold reset the cadence: no 32-round backoff
    }
}

test "probes after repeated misses back off by doubling to the cap and stay there" {
    for ([_]bool{ false, true }) |sampled| {
        var h: HoldRun = .{};
        defer h.deinit();
        try h.init(.{ .sampled = sampled, .granted = 1, .gain = 0.99, .prior = &.{0.5}, .k = .{ .depth_rate = 0.5, .depth_probe_every = 4, .plain_wait_most = 16, .grant_slack = 0 } });
        var at: [6]usize = undefined;
        try std.testing.expectEqual(@as(usize, at.len), try h.offered(&.{0}, &at, 200));
        const first = at[1] - at[0];
        try std.testing.expect(first == 4 or first == 5);
        const gaps = [_]usize{ 8, 16, 16, 16 };
        for (gaps, 2..) |gap, i| try std.testing.expectEqual(gap, at[i] - at[i - 1]);
    }
}

test "a due probe never resurrects a fresh reservation of zero" {
    for ([_]bool{ false, true }) |sampled| {
        const setup: HoldRun.Setup = .{ .sampled = sampled, .granted = 1, .gain = 0.99, .prior = &.{0.5}, .k = .{ .depth_rate = 0.5, .depth_probe_every = 4, .plain_wait_most = 16, .grant_slack = 0 } };
        var reference: HoldRun = .{};
        defer reference.deinit();
        try reference.init(setup);
        var at: [2]usize = undefined;
        try std.testing.expectEqual(@as(usize, at.len), try reference.offered(&.{0}, &at, 200));
        // Replay the same stream; at its first due probe the window reserves nothing, for several rounds.
        var h: HoldRun = .{};
        defer h.deinit();
        try h.init(setup);
        for (0..at[1]) |_| _ = try h.round(0, false);
        for (0..6) |_| try std.testing.expectEqual(@as(i64, 0), try h.round(1, true));
        // Once rows are reserved again the probe still happens: zero reservations did not starve or consume it.
        var probed = false;
        for (0..16) |_| {
            if (try h.round(1, false) > 0) {
                probed = true;
                break;
            }
        }
        try std.testing.expect(probed);
    }
}

const dr = @import("drafter.zig");
const Drafted = @import("drafted.zig").Drafted;
const Model = @import("config.zig").Model;
const Sampling = @import("sampling.zig").Sampling;
const dt = @import("drafted_test.zig");
const price = @import("hold_price.zig");

// SYNTHETIC, not performance data: forward 10 ms, modeled holds 7/9 ms; two .8-chance blocks pay above 1.25 ms.
const deferred_holds = [_]depth.HoldCost{ .{ .streams = 1, .depth = 1, .ms = 7 }, .{ .streams = 2, .depth = 1, .ms = 9 } };
// SYNTHETIC TIMING: every target forward width costs 10 ms.
var deferred_costs: [8]Cost = undefined;

fn deferredModel() Model {
    for (&deferred_costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 10 };
    return .{ .exact_width = 8, .window_costs = &deferred_costs, .hidden_rows = true };
}

fn seeded(seed: ?u64) ?Sampling {
    return if (seed) |v| .{ .seed = v, .temperature = 1.0, .top_p = 0.95 } else null;
}

/// What the target alone decodes for `prompt`, one token a step, no drafter.
fn deferredPlain(prompt: []const u32, max_new: u32, seed: ?u64) ![]u32 {
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var cfg = try Config.init(gpa, deferredModel(), 16, 15);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    var s = try sm.Stream.init(gpa, .{ .id = "plain", .prompt = prompt, .max_new = max_new, .drafts = false, .sampling = seeded(seed) });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return gpa.dupe(u32, s.emitted());
}

/// drafted_test's drafter whose holds settle late: a pass is charged to the round whose verify waits for it.
const Deferred = struct {
    head: dt.FakeDrafter,
    host_ms: f64, // SYNTHETIC TIMING: host work a round spends beyond its forward, the true fixed overhead
    vtable: dr.Drafter.VTable = undefined,
    inner_absorb: *const fn (*anyopaque, []const dr.Absorb) anyerror!void = undefined,
    inner_hold: *const fn (*anyopaque, []const dr.Hold) anyerror!void = undefined,
    inner_held: *const fn (*anyopaque, []const *sm.Stream, []const []u32) anyerror!void = undefined,
    pending: f64 = 0, // ms of launched hold passes no wait has settled yet
    window: f64 = 0, // ms settled since the round clock started
    settled: f64 = 0, // ms settled since the timing source was last taken
    blocks: u32 = 0, // hold blocks launched

    fn of(ptr: *anyopaque) *Deferred {
        return @fieldParentPtr("head", @as(*dt.FakeDrafter, @ptrCast(@alignCast(ptr))));
    }

    fn owner(ptr: *anyopaque) *Deferred {
        return @ptrCast(@alignCast(ptr));
    }

    fn settle(x: *Deferred) void {
        x.window += x.pending;
        x.settled += x.pending;
        x.pending = 0;
    }

    /// Settled GPU work outside any round (a load or calibration pass): the source holds it, no round window does.
    fn outside(x: *Deferred, ms: f64) void {
        x.settled += ms;
    }

    fn absorb(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const x = of(ptr);
        x.settle();
        try x.inner_absorb(ptr, items);
    }

    fn held(ptr: *anyopaque, streams: []const *sm.Stream, out: []const []u32) anyerror!void {
        const x = of(ptr);
        x.settle(); // the verify's wait on the previous round's hold
        try x.inner_held(ptr, streams, out);
    }

    fn hold(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = of(ptr);
        try x.inner_hold(ptr, items);
        var deepest: i64 = 0;
        for (items) |it| deepest = @max(deepest, @as(i64, it.depth));
        x.blocks += @intCast(items.len);
        x.pending += price.holdMs(&deferred_holds, items.len, deepest) + 4; // settled time differs from the model
    }

    fn chances(ptr: *anyopaque, s: *sm.Stream, out: []f64) anyerror!bool {
        const lane = of(ptr).head.lanes.getPtr(s) orelse return false;
        if (out.len > lane.held.items.len) return error.TooFewDrafts;
        var reach: f64 = 1;
        for (out) |*o| {
            reach *= 0.8;
            o.* = reach;
        }
        return true;
    }

    fn take(ptr: *anyopaque) f64 {
        const x = owner(ptr);
        const ms = x.settled;
        x.settled = 0;
        return ms;
    }

    /// Reports only settled work; the callback cannot wait for a pending hold.
    fn drafter(x: *Deferred) dr.Drafter {
        const inner = x.head.drafter();
        x.vtable = inner.vtable.*;
        x.inner_absorb = inner.vtable.absorb;
        x.inner_hold = inner.vtable.hold;
        x.inner_held = inner.vtable.held;
        x.vtable.absorb = absorb;
        x.vtable.hold = hold;
        x.vtable.held = held;
        x.vtable.chances = chances;
        return .{ .ptr = &x.head, .vtable = &x.vtable, .hold_timing = .{ .ptr = x, .take = take } };
    }

    fn clock(x: *Deferred) be.Clock {
        return .{ .ptr = x, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }

    fn start(ptr: *anyopaque) void {
        owner(ptr).window = 0;
    }

    /// A round's wall ms: the 10 ms forward, the host's fixed work, and the earlier holds it waited on.
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const x = owner(ptr);
        return 10.0 + x.host_ms + x.window;
    }
};

fn deferredLive(s: *const sm.Stream) bool {
    return !s.finished and s.emitted().len + 4 <= s.max_new;
}

/// Two drafting streams: assert real hold decisions, fixed-cost attribution and exact target-only tokens.
fn deferredRun(host_ms: f64, seed: ?u64) !void {
    const prompts = [2][]const u32{ &dt.p1, dt.long[0..11] };
    const ids = [2][]const u8{ "a", "b" };
    const max_new = [2]u32{ 48, 48 };
    var plain: [2][]u32 = undefined;
    var planned: usize = 0;
    defer for (plain[0..planned]) |p| gpa.free(p);
    for (&plain, prompts, max_new) |*p, prompt, n| {
        p.* = try deferredPlain(prompt, n, seed);
        planned += 1;
    }
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var strict: dt.Strict = .{ .inner = target.backend() };
    defer strict.deinit();
    var fix: Deferred = .{
        .head = .{ .facts_ = .{ .depth = 1, .step_ms = 0.5, .plain_guard = false, .batched = true, .chances = true, .hold_costs = &deferred_holds } },
        .host_ms = host_ms,
    };
    defer fix.head.deinit();
    var wrapped = try Drafted.init(gpa, strict.backend(), fix.drafter());
    defer wrapped.deinit();
    var cfg = try Config.init(gpa, wrapped.facts(deferredModel()), 16, 15);
    defer cfg.deinit(gpa);
    cfg.k.depth_rate = 0; // no empirical acceptance learning: this fixture isolates how round time is attributed
    var engine = Engine.init(gpa, &cfg, wrapped.backend(), fix.clock());
    defer engine.deinit();
    try engine.rule.observeOverhead(2, 2, 10 + host_ms, seed != null, 0); // synthetic baseline, no holds
    var streams: [2]sm.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    for (0..2) |i| {
        streams[i] = try sm.Stream.init(gpa, .{ .id = ids[i], .prompt = prompts[i], .max_new = max_new[i], .sampling = seeded(seed) });
        made += 1;
        try engine.addStream(&streams[i]);
    }

    var seen: usize = 0;
    while (engine.activeCount() > 0) {
        fix.outside(15); // calibration settled before this round starts: no part of its overhead, whatever the source
        const before = fix.blocks;
        try engine.step();
        if (!deferredLive(&streams[0]) or !deferredLive(&streams[1])) continue;
        seen += 1;
        // The measured host cost, not a previous hold's unsettled GPU debt, decides whether the two blocks pay.
        const want: u32 = if (host_ms > 1.25) 2 else 0;
        try std.testing.expectEqual(want, fix.blocks - before);
        try std.testing.expectApproxEqAbs(host_ms, engine.rule.fixedMs(2, seed != null), 1e-9);
    }
    try std.testing.expect(seen >= 12);
    for (plain, 0..) |p, i| try std.testing.expectEqualSlices(u32, p, streams[i].emitted());
}

test "a hold settled by a later verify is charged to that round: unpaid holds stop, paid ones stay, tokens are unchanged" {
    for ([_]f64{ 0, 2.5 }) |host_ms| for ([_]?u64{ null, 1234 }) |seed| try deferredRun(host_ms, seed);
}
