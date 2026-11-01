//! The rate guard (EXL3 with offloaded experts): a stream runs plain when its own plain rounds earn more tokens a ms.
const std = @import("std");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;
const Rule = depth.Rule;
const State = depth.State;
const Who = depth.Who;
const Probe = depth.Probe;
const min_samples = depth.min_samples;

test "with chances on the wire, a stream whose plain rounds earn more tokens a ms than its drafted ones runs plain, with probes" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.5, 0.5 }, .most_drafts = 2, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = true, .rate_guard = true, .node_probabilities = true, .batch_rows = 8 };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    const never = struct {
        fn copy(_: *anyopaque, _: *const Who) bool {
            return false;
        }
    };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = never.copy };
    // drafts are measured first, then plain
    try std.testing.expectEqual(@as(i64, 2), try rule.headDepth(&who, null, probe));
    for (0..min_samples) |i| {
        try rule.observeRate(who, 2, 60.0); // 2 tokens in 60 ms
        // then one plain round to measure
        try std.testing.expectEqual(@as(i64, if (i + 1 < min_samples) 2 else 0), try rule.headDepth(&who, null, probe));
    }
    for (0..min_samples) |_| try rule.observeRate(who, 1, 25.0); // a plain token in 25 ms beats 30 ms
    // the next decisions run plain until the probe comes
    var plain_rounds: usize = 0;
    while (try rule.headDepth(&who, null, probe) == 0) plain_rounds += 1;
    try std.testing.expectEqual(@as(usize, rule.k.depth_probe_every - 1), plain_rounds);
    // landing drafts that earn more flip it back
    st.?.rate.chain = true;
    for (0..8) |_| try rule.observeRate(who, 4, 60.0);
    try std.testing.expectEqual(@as(i64, 2), try rule.headDepth(&who, null, probe));
}

const never_copy = struct {
    fn copy(_: *anyopaque, _: *const Who) bool {
        return false;
    }
};

/// Each of `rounds` head depths for a stream whose plain rounds (1 token, 25 ms) beat its drafted (2 tokens, 60 ms).
fn plainBeatsDrafts(rule: *Rule, rounds: usize, out: []i64) !void {
    var st: ?State = null;
    defer if (st) |*s| s.deinit(rule.gpa);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = never_copy.copy };
    for (out[0..rounds]) |*o| {
        o.* = try rule.headDepth(&who, null, probe);
        // what the engine feeds rate_guard rounds, whether or not the rule uses it
        try rule.observeRate(who, if (o.* == 0) 1 else 2, if (o.* == 0) 25.0 else 60.0);
    }
}

test "without rate_guard every family class drafts as the node-probability rule always did, whatever the rounds measured" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    // plain_guard families (GLM-DSA, XPU, Qwen3.5 MTP and DFlash, HIP), and families without it (NVFP4 Flash Next)
    for ([_]bool{ true, false }) |plain_guard| {
        var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.5, 0.5 }, .most_drafts = 2, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = plain_guard, .node_probabilities = true, .batch_rows = 8 };
        defer rule.deinit();
        try std.testing.expect(!rule.rate_guard);
        var got: [64]i64 = undefined;
        try plainBeatsDrafts(&rule, got.len, &got);
        for (got) |d| try std.testing.expectEqual(@as(i64, 2), d);
    }
}

test "without rate_guard the probe wait stops at plain_wait_most, 128; rate_guard stops it at its own 64" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.5, 0.5 }, .most_drafts = 2, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = true, .node_probabilities = true, .batch_rows = 8 };
    defer rule.deinit();
    try std.testing.expectEqual(@as(u64, 128), rule.k.plain_wait_most);
    var st: State = .{ .p = &.{}, .rounds = 0 };
    for (0..1000) |_| _ = rule.probePlain(&st); // the wait doubles 8, 16, ... to the cap
    try std.testing.expectEqual(@as(?u64, 128), st.wait);
    rule.rate_guard = true;
    for (0..1000) |_| _ = rule.probePlain(&st);
    try std.testing.expectEqual(@as(?u64, 64), st.wait);
}

test "with rate_guard a stream is held on plain for 20% leave-plain hysteresis, not on a tie" {
    const gpa = std.testing.allocator;
    const empty: Table = .{};
    var rule: Rule = .{ .gpa = gpa, .prior = &.{ 0.5, 0.5 }, .most_drafts = 2, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = true, .rate_guard = true, .node_probabilities = true, .batch_rows = 8 };
    defer rule.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(gpa);
    const who: Who = .{ .state = &st, .draft_room = 100, .finished = false, .forced = false };
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = never_copy.copy };
    try std.testing.expectEqual(@as(i64, 2), try rule.headDepth(&who, null, probe));
    for (0..min_samples) |_| {
        try rule.observeRate(who, 2, 60.0); // 1/30 a ms
        _ = try rule.headDepth(&who, null, probe);
    }
    for (0..min_samples) |_| try rule.observeRate(who, 1, 25.0); // 1/25 a ms: plain wins
    try std.testing.expectEqual(@as(i64, 0), try rule.headDepth(&who, null, probe));
    // drafts 10% above plain beat the 5% plain margin but not the 20% leave margin: still plain
    st.?.rate.chain = true;
    st.?.rate.draft_tok = 1.1;
    st.?.rate.draft_ms = 25.0;
    try std.testing.expect(st.?.rate.draft_tok / st.?.rate.draft_ms > st.?.rate.plain_tok / st.?.rate.plain_ms * rule.k.plain_margin);
    st.?.rate.chain = false;
    try std.testing.expectEqual(@as(i64, 0), try rule.headDepth(&who, null, probe));
    // drafts 25% above plain leave it
    st.?.rate.draft_tok = 1.25;
    try std.testing.expectEqual(@as(i64, 2), try rule.headDepth(&who, null, probe));
}

/// A stream's state primed with 2-token drafted rounds of `draft_ms` and 1-token plain rounds of 25 ms.
fn primed(rule: *Rule, st: *?State, draft_ms: f64) !Who {
    const who: Who = .{ .state = st, .draft_room = 100, .finished = false, .forced = false };
    try rule.observeRate(who, 0, 1); // creates the state
    st.*.?.rate.chain = true;
    for (0..min_samples) |_| try rule.observeRate(who, 2, draft_ms);
    st.*.?.rate.chain = false;
    for (0..min_samples) |_| try rule.observeRate(who, 1, 25.0);
    return who;
}

fn sharedRule(empty: *const Table, rate_guard: bool) Rule {
    return .{ .gpa = std.testing.allocator, .prior = &.{ 0.5, 0.5 }, .most_drafts = 2, .family_costs = empty, .shared_costs = empty, .mtp_step_ms = 0.5, .plain_guard = true, .rate_guard = rate_guard, .node_probabilities = true, .batch_rows = 8 };
}

test "a shared round runs plain only when every stream's rates say so, else all draft" {
    const empty: Table = .{};
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = never_copy.copy };
    var out: [2]i64 = undefined;
    // both streams: plain (1 token in 25 ms) beats drafts (2 tokens in 60 ms)
    {
        var rule = sharedRule(&empty, true);
        defer rule.deinit();
        var sa: ?State = null;
        var sb: ?State = null;
        defer if (sa) |*s| s.deinit(std.testing.allocator);
        defer if (sb) |*s| s.deinit(std.testing.allocator);
        var whos = [_]Who{ try primed(&rule, &sa, 60.0), try primed(&rule, &sb, 60.0) };
        try rule.budgets(&whos, probe, false, &out);
        try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &out);
        try std.testing.expect(!sa.?.rate.chain and !sb.?.rate.chain);
    }
    // one stream's drafts land well (2 tokens in 30 ms beats 25 ms plain by more than the margin): both draft
    {
        var rule = sharedRule(&empty, true);
        defer rule.deinit();
        var sa: ?State = null;
        var sb: ?State = null;
        defer if (sa) |*s| s.deinit(std.testing.allocator);
        defer if (sb) |*s| s.deinit(std.testing.allocator);
        var whos = [_]Who{ try primed(&rule, &sa, 60.0), try primed(&rule, &sb, 30.0) };
        try rule.budgets(&whos, probe, false, &out);
        try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, &out);
        try std.testing.expect(sa.?.rate.chain and sb.?.rate.chain); // both launch chains, so their rounds' ms are filed as drafted
    }
}

test "rate guard and granted depths both bound lone and shared drafts" {
    const empty: Table = .{};
    var rule = sharedRule(&empty, true);
    rule.granted_depth = true;
    defer rule.deinit();
    var sa: ?State = null;
    var sb: ?State = null;
    defer if (sa) |*s| s.deinit(std.testing.allocator);
    defer if (sb) |*s| s.deinit(std.testing.allocator);
    var a = try primed(&rule, &sa, 60.0);
    var b = try primed(&rule, &sb, 30.0);
    a.granted = 1;
    a.grant_short = true;
    b.granted = 1;
    b.grant_short = true;
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = never_copy.copy };
    try std.testing.expectEqual(@as(i64, 0), try rule.headDepth(&a, null, probe));
    try std.testing.expectEqual(@as(i64, 1), try rule.headDepth(&b, null, probe));
    var whos = [_]Who{ a, b };
    var out: [2]i64 = undefined;
    try rule.budgets(&whos, probe, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 1, 1 }, &out);
    sb.?.rate.draft_ms = 60.0;
    try rule.budgets(&whos, probe, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &out);
}

test "rate decisions do not reset zero-grant probes" {
    const empty: Table = .{};
    var rule = sharedRule(&empty, true);
    rule.granted_depth = true;
    defer rule.deinit();
    var sa: ?State = null;
    var sb: ?State = null;
    defer if (sa) |*s| s.deinit(std.testing.allocator);
    defer if (sb) |*s| s.deinit(std.testing.allocator);
    var a = try primed(&rule, &sa, 30.0);
    var b = try primed(&rule, &sb, 30.0);
    a.granted = 0;
    b.granted = 0;
    var whos = [_]Who{b};
    var out: [1]i64 = undefined;
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = never_copy.copy };
    for (0..rule.k.depth_probe_every) |i| {
        const want: i64 = if (i + 1 == rule.k.depth_probe_every) 2 else 0;
        try std.testing.expectEqual(want, try rule.headDepth(&a, null, probe));
        try rule.budgets(&whos, probe, false, &out);
        try std.testing.expectEqual(want, out[0]);
    }
}

test "without rate_guard a shared round drafts every stream whatever the rates" {
    const empty: Table = .{};
    var unused: u8 = 0;
    const probe: Probe = .{ .ptr = &unused, .copy = never_copy.copy };
    var rule = sharedRule(&empty, false);
    defer rule.deinit();
    var sa: ?State = null;
    var sb: ?State = null;
    defer if (sa) |*s| s.deinit(std.testing.allocator);
    defer if (sb) |*s| s.deinit(std.testing.allocator);
    var whos = [_]Who{ try primed(&rule, &sa, 60.0), try primed(&rule, &sb, 60.0) };
    var out: [2]i64 = undefined;
    try rule.budgets(&whos, probe, false, &out);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, &out);
}

test "a plain stream's window taps hidden rows only when a draft may follow it" {
    const empty: Table = .{};
    var on = sharedRule(&empty, true);
    defer on.deinit();
    var off = sharedRule(&empty, false);
    defer off.deinit();
    var st: ?State = null;
    defer if (st) |*s| s.deinit(std.testing.allocator);
    const who = try primed(&on, &st, 60.0);
    var none: ?State = null;
    try std.testing.expect(on.rowsWanted(.{ .state = &none, .draft_room = 100, .finished = false, .forced = false })); // no state yet
    st.?.rate.chain = true;
    try std.testing.expect(on.rowsWanted(who)); // drafting
    st.?.rate.chain = false;
    st.?.plain = 0;
    st.?.wait = 8;
    try std.testing.expect(!on.rowsWanted(who)); // plain, no probe due
    st.?.plain = 7;
    try std.testing.expect(on.rowsWanted(who)); // the round ending now probes with a draft
    st.?.plain = 0;
    try std.testing.expect(off.rowsWanted(who)); // without rate_guard every window taps them
}
