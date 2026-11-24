//! Rank 0's calls for a server: caller-kept windows, queued opens, closes and cancels, keep-alives, and drain.
const std = @import("std");
const dt = @import("driver_test.zig");
const driver = @import("driver.zig");
const boot = @import("boot.zig");
const plan = @import("plan.zig");
const t = std.testing;
const Fake = dt.Fake;
const D = dt.D;
const s = std.time.ns_per_s;

/// Body `f` on one engine (no ring), the reference for the rings below.
fn wholeOf(gpa: std.mem.Allocator, f: *const fn (std.mem.Allocator, *D.Lead, *Fake.Engine) anyerror![]u32, width: u32) ![]u32 {
    var e: Fake.Engine = .{ .gpa = gpa, .first = 0, .end = dt.layers };
    defer e.out.deinit(gpa);
    var lead = try D.Lead.init(gpa, &e, null, width, dt.context, 1);
    defer lead.deinit(gpa);
    return f(gpa, &lead, &e);
}

/// Body `f` as rank 0 of `layout` with `plants` on the followers: rank 0's bytes, or the first rank's error.
fn ringOf(gpa: std.mem.Allocator, c: dt.Case) ![]u32 {
    var res: [dt.most]dt.Result = @splat(.{});
    try dt.ringRun(gpa, c, &res);
    for (res[0..c.layout.len], 0..) |x, i| if (x.err) |err| {
        if (i == 0) gpa.free(x.out);
        return err;
    };
    return res[0].out;
}

/// `f` on 2 and 3 stages gives `ref`'s bytes; a follower rolling back its position only changes them.
fn expectRings(ref: []const u32, f: *const fn (std.mem.Allocator, *D.Lead, *Fake.Engine) anyerror![]u32, width: u32) !void {
    const gpa = t.allocator;
    for ([_][]const boot.Range{ &dt.two, &dt.three }) |layout| {
        const got = try ringOf(gpa, .{ .layout = layout, .body = f, .width = width });
        defer gpa.free(got);
        try t.expectEqualSlices(u32, ref, got);
    }
    const bad = try ringOf(gpa, .{ .layout = &dt.two, .body = f, .width = width, .plants = .{ .none, .no_restore, .none } });
    defer gpa.free(bad);
    try t.expect(!std.mem.eql(u32, ref, bad));
}

fn idsOf(seed: u32, out: []u32) []u32 {
    for (out, 0..) |*x, i| x.* = 1000 + (seed *% 131 +% 37 * @as(u32, @intCast(i))) % 50000;
    return out;
}

/// The driver's scenario with every keep done on rank 0 first and handed over by `kept`: entries carry no counts.
fn keptBody(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    var ids: [dt.most_rows]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1000 + 37 * i);
    for (dt.scenario) |r| {
        var entries: [dt.parallel]driver.Entry = undefined;
        var n: u32 = 0;
        for (r.entries, entries[0..r.entries.len]) |en, *out| {
            out.* = en;
            n += en.rows;
            const c = lead.cache(en.id) orelse continue;
            if (en.op != .@"continue" or Fake.window(c) == null) continue;
            try Fake.rollback(e, c, en.accepted);
            try lead.kept(en.id, en.accepted);
            out.accepted = 0;
        }
        if (r.entries.len == 0) try lead.beat() else try lead.step(entries[0..r.entries.len], ids[0..n], r.mode);
    }
    return e.out.toOwnedSlice(gpa);
}

test "stage lead: windows kept on rank 0 first and sent as accepted give the scenario's bytes on 1, 2 and 3 stages" {
    const gpa = t.allocator;
    const ref = try dt.whole(gpa);
    defer gpa.free(ref);
    const got = try wholeOf(gpa, keptBody, dt.parallel);
    defer gpa.free(got);
    try t.expectEqualSlices(u32, ref, got);
    try expectRings(ref, keptBody, dt.parallel);
}

test "stage lead: a keep is refused unless rank 0 rolled back, once, for a known stream, with the plan's count" {
    const gpa = t.allocator;
    var e: Fake.Engine = .{ .gpa = gpa, .first = 0, .end = dt.layers };
    defer e.out.deinit(gpa);
    var lead = try D.Lead.init(gpa, &e, null, dt.parallel, dt.context, 1);
    defer lead.deinit(gpa);
    var ids: [4]u32 = undefined;
    try lead.step(&.{.{ .id = 1, .op = .open, .rows = 4 }}, idsOf(1, &ids), .prompt);
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 3 }}, idsOf(2, ids[0..3]), .window);
    try t.expectError(error.Unsettled, lead.kept(1, 2));
    try t.expectError(error.UnknownStream, lead.kept(9, 0));
    try Fake.rollback(&e, lead.cache(1).?, 2);
    try lead.kept(1, 2);
    try t.expectError(error.KeptTwice, lead.kept(1, 2));
    const rows = idsOf(3, ids[0..1]);
    try t.expectError(error.BadAccepted, lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 1, .accepted = 1 }}, rows, .window));
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 1, .accepted = 2 }}, rows, .window);
}

/// Rank 0 keeps a stream's window and steps it on, then ends it; `a` the count kept of its rows.
fn keepOn(lead: *D.Lead, e: *Fake.Engine, ids: []u32, id: u32, a: u32) !void {
    try Fake.rollback(e, lead.cache(id).?, a);
    try lead.kept(id, a);
    try lead.step(&.{.{ .id = id, .op = .@"continue", .rows = 2 }}, idsOf(id + 20, ids[0..2]), .window);
    try lead.end(id, .close);
    try lead.beat();
}

/// `kept` after a prompt round and on a stream whose open is still queued: refused by name, nothing sent.
fn noWindowBody(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    var ids: [dt.most_rows]u32 = undefined;
    try lead.step(&.{.{ .id = 1, .op = .open, .rows = 4 }}, idsOf(1, ids[0..4]), .prompt);
    const r0 = lead.round;
    try t.expectError(error.NoWindow, lead.kept(1, 1)); // a prompt round leaves no window
    _ = try lead.open(2);
    try t.expectError(error.NoWindow, lead.kept(2, 0)); // no round of it has run
    try t.expectEqual(r0, lead.round);
    try lead.step(&.{.{ .id = 2, .op = .@"continue", .rows = 3 }}, idsOf(2, ids[0..3]), .prompt);
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 3 }}, idsOf(3, ids[0..3]), .window);
    try lead.end(2, .close);
    try keepOn(lead, e, &ids, 1, 1);
    return e.out.toOwnedSlice(gpa);
}

test "stage lead: a keep with no window, after a prompt round or before a queued open's round, is refused" {
    const gpa = t.allocator;
    const ref = try wholeOf(gpa, noWindowBody, dt.parallel);
    defer gpa.free(ref);
    try expectRings(ref, noWindowBody, dt.parallel);
}

/// `kept` of more rows than the stream's last window ran: refused by name, nothing sent, then a count it holds.
fn aboveRowsBody(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    var ids: [dt.most_rows]u32 = undefined;
    try lead.step(&.{.{ .id = 1, .op = .open, .rows = 4 }}, idsOf(1, ids[0..4]), .prompt);
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 3 }}, idsOf(2, ids[0..3]), .window);
    try Fake.rollback(e, lead.cache(1).?, 3); // all 3 rows kept: the count may equal the rows
    try lead.kept(1, 3);
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 2 }}, idsOf(3, ids[0..2]), .window);
    const r0 = lead.round;
    try Fake.rollback(e, lead.cache(1).?, 1);
    try t.expectError(error.BadAccepted, lead.kept(1, 3));
    try t.expectEqual(r0, lead.round);
    try lead.kept(1, 1);
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 3 }}, idsOf(4, ids[0..3]), .window);
    try keepOn(lead, e, &ids, 1, 2);
    return e.out.toOwnedSlice(gpa);
}

test "stage lead: a keep of more rows than the last window ran is refused" {
    const gpa = t.allocator;
    const ref = try wholeOf(gpa, aboveRowsBody, dt.parallel);
    defer gpa.free(ref);
    try expectRings(ref, aboveRowsBody, dt.parallel);
}

/// Streams opened, ended and reopened over 2 slots: by `Lead.open` and `Lead.end` (queued) or by plan entries.
fn slotsRun(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine, queued: bool) ![]u32 {
    var ids: [dt.most_rows]u32 = undefined;
    const S = struct {
        fn prompt(l: *D.Lead, q: bool, id: u32, rows: u32, buf: []u32) !void {
            if (q) _ = try l.open(id);
            const op: plan.Op = if (q) .@"continue" else .open;
            try l.step(&.{.{ .id = id, .op = op, .rows = rows }}, idsOf(id *% rows, buf[0..rows]), .prompt);
        }
        fn end(l: *D.Lead, q: bool, id: u32, op: plan.Op) !void {
            if (q) return l.end(id, op);
            try l.step(&.{.{ .id = id, .op = op }}, &.{}, .window);
        }
    };
    try S.prompt(lead, queued, 1, 6, &ids);
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 3 }}, idsOf(7, ids[0..3]), .window);
    if (queued) _ = try lead.open(2); // its open rides stream 1's next round, its prompt the one after
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 2, .accepted = 1 }}, idsOf(8, ids[0..2]), .window);
    const op2: plan.Op = if (queued) .@"continue" else .open;
    try lead.step(&.{.{ .id = 2, .op = op2, .rows = 5 }}, idsOf(9, ids[0..5]), .prompt);
    const both = [_]driver.Entry{ .{ .id = 1, .op = .@"continue", .rows = 2, .accepted = 2 }, .{ .id = 2, .op = .@"continue", .rows = 3 } };
    try lead.step(&both, idsOf(10, ids[0..5]), .window);
    try S.end(lead, queued, 1, .close);
    try S.prompt(lead, queued, 3, 4, &ids); // no free slot: the queued close goes first
    try S.end(lead, queued, 2, .cancel);
    try lead.step(&.{.{ .id = 3, .op = .@"continue", .rows = 2 }}, idsOf(11, ids[0..2]), .window);
    if (queued) { // opened and ended before any plan: the followers never see it
        _ = try lead.open(4);
        try lead.end(4, .close);
    }
    try S.prompt(lead, queued, 4, 3, &ids);
    try S.end(lead, queued, 4, .close);
    try S.prompt(lead, queued, 4, 2, &ids); // the same id again: its close goes first
    try lead.step(&.{ .{ .id = 3, .op = .@"continue", .rows = 1, .accepted = 1 }, .{ .id = 4, .op = .@"continue", .rows = 1 } }, idsOf(12, ids[0..2]), .window);
    try S.end(lead, queued, 3, .close);
    try S.end(lead, queued, 4, .cancel);
    if (queued) try lead.beat();
    for ([_]u32{ 1, 2, 3, 4 }) |id| try t.expectEqual(@as(?*Fake.Cache, null), lead.cache(id));
    return e.out.toOwnedSlice(gpa);
}

fn queuedBody(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    return slotsRun(gpa, lead, e, true);
}

fn plainBody(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    return slotsRun(gpa, lead, e, false);
}

test "stage lead: opens, closes and cancels queued into the next plan give the bytes of plans that carry them" {
    const gpa = t.allocator;
    const ref = try wholeOf(gpa, plainBody, dt.parallel);
    defer gpa.free(ref);
    const got = try wholeOf(gpa, queuedBody, dt.parallel);
    defer gpa.free(got);
    try t.expectEqualSlices(u32, ref, got);
    try expectRings(ref, queuedBody, dt.parallel);
}

test "stage lead: a queued open or end is refused for an open, unknown or in-flight stream" {
    const gpa = t.allocator;
    var e: Fake.Engine = .{ .gpa = gpa, .first = 0, .end = dt.layers };
    defer e.out.deinit(gpa);
    var lead = try D.Lead.init(gpa, &e, null, dt.parallel, dt.context, 1);
    defer lead.deinit(gpa);
    try lead.pipe(t.io, 2);
    _ = try lead.open(1);
    try t.expectError(error.StreamOpen, lead.open(1));
    try t.expectError(error.UnknownStream, lead.end(2, .close));
    try t.expectError(error.BadOp, lead.end(1, .open));
    var ids: [3]u32 = undefined;
    try lead.step(&.{.{ .id = 1, .op = .@"continue", .rows = 3 }}, idsOf(1, &ids), .prompt);
    try lead.issue(&.{.{ .id = 1, .op = .@"continue", .rows = 1 }}, ids[0..1], .window);
    try t.expectError(error.StreamInFlight, lead.end(1, .cancel));
    try t.expectError(error.StreamInFlight, lead.kept(1, 0));
    try lead.drain();
    try lead.end(1, .cancel);
    try t.expectError(error.UnknownStream, lead.end(1, .close));
    try t.expectEqual(@as(?*Fake.Cache, null), lead.cache(1));
}

test "stage lead: a keep-alive goes out once 2 s pass after the last plan a poll saw, never with rounds in flight" {
    const gpa = t.allocator;
    var e: Fake.Engine = .{ .gpa = gpa, .first = 0, .end = dt.layers };
    defer e.out.deinit(gpa);
    var lead = try D.Lead.init(gpa, &e, null, dt.parallel, dt.context, 1);
    defer lead.deinit(gpa);
    try t.expect(!try lead.beatIfDue(100 * s)); // the first poll only starts the clock
    const r0 = lead.round;
    try t.expect(!try lead.beatIfDue(102 * s - 1));
    try t.expect(try lead.beatIfDue(102 * s));
    try t.expectEqual(r0 + 1, lead.round);
    try t.expect(!try lead.beatIfDue(103 * s));
    var ids: [3]u32 = undefined;
    try lead.step(&.{.{ .id = 1, .op = .open, .rows = 3 }}, idsOf(1, &ids), .prompt);
    try t.expect(!try lead.beatIfDue(104 * s)); // a round's plan resets the clock
    try t.expect(!try lead.beatIfDue(106 * s - 1));
    try t.expect(try lead.beatIfDue(106 * s));
    try lead.pipe(t.io, 2);
    try lead.issue(&.{.{ .id = 1, .op = .@"continue", .rows = 1 }}, ids[0..1], .window);
    const r1 = lead.round;
    try t.expect(!try lead.beatIfDue(110 * s));
    try t.expect(!try lead.beatIfDue(120 * s));
    try t.expectEqual(r1, lead.round);
    try lead.drain();
    try t.expect(try lead.beatIfDue(122 * s));
}

/// Two groups of two streams, rounds issued two at a time and drained (or one at a time by `step`), then ends.
fn drainRun(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine, piped: bool) ![]u32 {
    if (piped) try lead.pipe(t.io, 2);
    var ids: [dt.most_rows]u32 = undefined;
    for (1..5) |id| try lead.step(&.{.{ .id = @intCast(id), .op = .open, .rows = 4 }}, idsOf(@intCast(id), ids[0..4]), .prompt);
    for (0..4) |j| {
        var groups: [2][2]driver.Entry = undefined;
        for (&groups, 0..) |*g, k| for (g, 0..) |*en, i| {
            const id: u32 = @intCast(1 + 2 * k + i);
            if (j > 0) { // rank 0 keeps the last window first
                const a: u32 = @intCast((id + j) % 3);
                try Fake.rollback(e, lead.cache(id).?, a);
                try lead.kept(id, a);
            }
            en.* = .{ .id = id, .op = .@"continue", .rows = 2 };
        };
        for (&groups, 0..) |*g, k| {
            const rows = idsOf(@intCast(10 * j + k), ids[0..4]);
            if (piped) try lead.issue(g, rows, .window) else try lead.step(g, rows, .window);
        }
        if (piped) try t.expectEqual(@as(usize, 2), lead.inFlight());
        if (piped and j == 3) try t.expectError(error.StreamInFlight, lead.end(1, .close));
        try lead.drain();
        try t.expectEqual(@as(usize, 0), lead.inFlight());
    }
    for (1..5) |id| try lead.end(@intCast(id), if (id % 2 == 0) .cancel else .close);
    try lead.beat();
    return e.out.toOwnedSlice(gpa);
}

fn pipedBody(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    return drainRun(gpa, lead, e, true);
}

fn steppedBody(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    return drainRun(gpa, lead, e, false);
}

test "stage lead: draining two rounds in flight, then keeps and ends, gives one round at a time's bytes" {
    const gpa = t.allocator;
    const ref = try wholeOf(gpa, steppedBody, dt.pipe_streams);
    defer gpa.free(ref);
    const got = try wholeOf(gpa, pipedBody, dt.pipe_streams);
    defer gpa.free(got);
    try t.expectEqualSlices(u32, ref, got);
    try expectRings(ref, pipedBody, dt.pipe_streams);
}

fn oneAtATimeBody(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    try lead.pipe(t.io, 1);
    try dt.run(lead);
    return e.out.toOwnedSlice(gpa);
}

test "stage lead: one round in flight (k = 1) gives today's rounds, prompt chunks, keep-alive and closes alike" {
    const gpa = t.allocator;
    const ref = try dt.whole(gpa);
    defer gpa.free(ref);
    const got = try wholeOf(gpa, oneAtATimeBody, dt.parallel);
    defer gpa.free(got);
    try t.expectEqualSlices(u32, ref, got);
    try expectRings(ref, oneAtATimeBody, dt.parallel);
}

/// Two groups in flight while a cancel, a release and a prompt chunk wait: each refused, then run after a drain.
fn opsRun(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine, piped: bool) ![]u32 {
    if (piped) try lead.pipe(t.io, 2);
    var ids: [dt.most_rows]u32 = undefined;
    for (1..5) |id| try lead.step(&.{.{ .id = @intCast(id), .op = .open, .rows = 3 }}, idsOf(@intCast(id), ids[0..3]), .prompt);
    var groups = [2][2]u32{ .{ 1, 2 }, .{ 3, 4 } };
    var sizes = [2]usize{ 2, 2 };
    for (0..5) |j| {
        for (&groups, sizes, 0..) |*g, n, k| {
            var entries: [2]driver.Entry = undefined;
            for (g[0..n], entries[0..n]) |id, *en| {
                const c = lead.cache(id).?;
                if (Fake.window(c) != null) { // rank 0 keeps the last window first
                    const a: u32 = @intCast((id + j) % 3);
                    try Fake.rollback(e, c, a);
                    try lead.kept(id, a);
                }
                en.* = .{ .id = id, .op = .@"continue", .rows = 2 };
            }
            const rows = idsOf(@intCast(10 * j + k), ids[0 .. 2 * n]);
            if (piped) try lead.issue(entries[0..n], rows, .window) else try lead.step(entries[0..n], rows, .window);
        }
        if (piped and j == 1) { // a cancel of 4 and stream 5's prompt chunk wait for the rounds in flight
            try t.expectError(error.StreamInFlight, lead.end(4, .cancel));
            try t.expectError(error.RoundsInFlight, lead.step(&.{.{ .id = 5, .op = .open, .rows = 3 }}, ids[0..3], .prompt));
        }
        if (piped and j == 3) try t.expectError(error.StreamInFlight, lead.end(1, .close)); // a release waits too
        try lead.drain();
        if (j == 1) {
            try lead.end(4, .cancel);
            _ = try lead.open(5);
            try lead.step(&.{.{ .id = 5, .op = .@"continue", .rows = 3 }}, idsOf(5, ids[0..3]), .prompt);
            groups[1][1] = 5;
        }
        if (j == 3) {
            try lead.end(1, .close);
            groups[0][0] = 2;
            sizes[0] = 1;
        }
    }
    for ([_]u32{ 2, 3, 5 }) |id| try lead.end(id, .close);
    try lead.beat();
    return e.out.toOwnedSlice(gpa);
}

fn opsPiped(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    return opsRun(gpa, lead, e, true);
}

fn opsStepped(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    return opsRun(gpa, lead, e, false);
}

test "stage lead: a cancel, a release and a prompt chunk drain the rounds in flight first, and give one round's bytes" {
    const gpa = t.allocator;
    const ref = try wholeOf(gpa, opsStepped, dt.pipe_streams);
    defer gpa.free(ref);
    const got = try wholeOf(gpa, opsPiped, dt.pipe_streams);
    defer gpa.free(got);
    try t.expectEqualSlices(u32, ref, got);
    try expectRings(ref, opsPiped, dt.pipe_streams);
}

/// Rounds until the ring fails: that call fails by the ring's error, every later one is refused `StageLost`.
fn lostRun(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine, k: usize) ![]u32 {
    if (k > 0) try lead.pipe(t.io, k);
    var ids: [dt.most_rows]u32 = undefined;
    const first: anyerror = for (0..20) |j| {
        rounds(lead, &ids, j, k) catch |err| break err;
    } else return error.RingNeverFailed;
    try t.expect(first != error.StageLost);
    try t.expectEqual(@as(?anyerror, first), lead.lost);
    const one = [_]driver.Entry{.{ .id = 1, .op = .@"continue", .rows = 1 }};
    try t.expectError(error.StageLost, lead.open(9));
    try t.expectError(error.StageLost, lead.end(1, .cancel));
    try t.expectError(error.StageLost, lead.kept(1, 0));
    try t.expectError(error.StageLost, lead.step(&one, ids[0..1], .window));
    try t.expectError(error.StageLost, lead.beat());
    try t.expectError(error.StageLost, lead.beatIfDue(1 << 60));
    try t.expectError(error.StageLost, lead.drain());
    if (k > 0) try t.expectError(error.StageLost, lead.issue(&one, ids[0..1], .window));
    if (k > 0) try t.expectError(error.StageLost, lead.complete());
    return e.out.toOwnedSlice(gpa);
}

/// Round `j`: streams 1 and 2 opened by prompts first, then one window each, by `step` or in 2 groups in flight.
fn rounds(lead: *D.Lead, ids: []u32, j: usize, k: usize) !void {
    if (j < 2) return lead.step(&.{.{ .id = @intCast(j + 1), .op = .open, .rows = 3 }}, idsOf(@intCast(j), ids[0..3]), .prompt);
    const a: u32 = if (j == 2) 0 else 1;
    const both = [_]driver.Entry{ .{ .id = 1, .op = .@"continue", .rows = 1, .accepted = a }, .{ .id = 2, .op = .@"continue", .rows = 1, .accepted = a } };
    if (k == 0) return lead.step(&both, idsOf(@intCast(j), ids[0..2]), .window);
    try lead.issue(both[0..1], idsOf(@intCast(j), ids[0..1]), .window);
    try lead.issue(both[1..2], idsOf(@intCast(j + 50), ids[0..1]), .window);
    try lead.drain();
}

fn lostStepped(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    return lostRun(gpa, lead, e, 0);
}

fn lostPiped(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine) ![]u32 {
    return lostRun(gpa, lead, e, 2);
}

test "stage lead: a follower lost mid-round fails that call by the ring's error, then every call is StageLost" {
    const gpa = t.allocator;
    for ([_]*const fn (std.mem.Allocator, *D.Lead, *Fake.Engine) anyerror![]u32{ lostStepped, lostPiped }) |f| {
        for ([_][]const boot.Range{ &dt.two, &dt.three }) |layout| {
            var res: [dt.most]dt.Result = @splat(.{});
            try dt.ringRun(gpa, .{ .layout = layout, .body = f, .width = dt.parallel, .plants = .{ .none, .dies, .none } }, &res);
            defer gpa.free(res[0].out);
            try t.expectEqual(@as(?anyerror, null), res[0].err);
            try t.expectEqual(@as(?anyerror, error.PlantedDeath), res[1].err);
        }
    }
}

test "stage lead: one engine is never lost, an engine error is only that call's" {
    const gpa = t.allocator;
    var e: Fake.Engine = .{ .gpa = gpa, .first = 0, .end = dt.layers };
    defer e.out.deinit(gpa);
    var lead = try D.Lead.init(gpa, &e, null, dt.parallel, dt.context, 1);
    defer lead.deinit(gpa);
    var ids: [dt.most_rows + 1]u32 = undefined;
    try t.expectError(error.BadRows, lead.step(&.{.{ .id = 1, .op = .open, .rows = 3 }}, idsOf(1, ids[0..4]), .prompt));
    try t.expectEqual(@as(?anyerror, null), lead.lost);
    try lead.beat();
}
