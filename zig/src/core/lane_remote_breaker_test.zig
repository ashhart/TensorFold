//! The remote breaker: failures open it, a cool-down half-opens it for one trial, a good pull closes it.
const std = @import("std");
const remote = @import("lane_remote.zig");
const breaker = @import("lane_remote_breaker.zig");
const Route = remote.Route;
const t = std.testing;

const s: u64 = std.time.ns_per_s;

fn weigh(r: *Route, now: u64) remote.Verdict {
    return r.weigh(now, .{ .len = 40000, .cached = 0, .decoding = 1 }, 39999);
}

test "breaker: three failures in a row open it, the pair is not asked for the cool-down, one trial then decides" {
    var r: Route = .{};
    try t.expectEqual(remote.Verdict.remote, weigh(&r, 0));
    for (0..2) |i| try t.expectEqual(breaker.Change.none, r.finish(i * s, .down));
    try t.expectEqual(breaker.Change.none, r.finish(2 * s, .ok)); // an answer clears the count
    for (0..2) |i| try t.expectEqual(breaker.Change.none, r.finish((3 + i) * s, .down));
    try t.expectEqual(breaker.Change.none, r.finish(5 * s, .other)); // a refusal is neither
    try t.expectEqual(breaker.Change.opened, r.finish(6 * s, .down));
    try t.expectEqual(remote.Verdict.down, weigh(&r, 7 * s));
    try t.expectEqual(breaker.Door.local, r.door(30 * s)); // a pull that was already waiting at the gate
    try t.expectEqual(remote.Verdict.remote, weigh(&r, 36 * s)); // 30 s later a prompt may go
    try t.expectEqual(breaker.Door.probe, r.door(36 * s));
    try t.expectEqual(remote.Verdict.down, weigh(&r, 36 * s)); // the trial is under way: the rest stay here
    try t.expectEqual(breaker.Door.local, r.door(37 * s));
    try t.expectEqual(breaker.Change.opened, r.finish(38 * s, .down)); // the trial failed: shut again, whole cool-down
    try t.expectEqual(remote.Verdict.down, weigh(&r, 60 * s));
    try t.expectEqual(breaker.Door.probe, r.door(68 * s));
    try t.expectEqual(breaker.Change.closed, r.finish(69 * s, .ok));
    try t.expectEqual(breaker.Door.through, r.door(70 * s));
    try t.expectEqual(remote.Verdict.remote, weigh(&r, 70 * s));
}

test "breaker: a trial that ends in a refusal leaves the next prompt to try, and 0 never opens" {
    var r: Route = .{ .breaker = .{ .trip = 1, .cooldown_ns = 10 * s } };
    try t.expectEqual(breaker.Change.opened, r.finish(0, .down));
    try t.expectEqual(breaker.Door.probe, r.door(10 * s));
    try t.expectEqual(breaker.Change.none, r.finish(11 * s, .other));
    try t.expectEqual(remote.Verdict.remote, weigh(&r, 12 * s)); // not closed, not shut: the next prompt is the trial
    try t.expectEqual(breaker.Door.probe, r.door(12 * s));
    var never: Route = .{ .breaker = .{ .trip = 0 } };
    for (0..10) |i| try t.expectEqual(breaker.Change.none, never.finish(i * s, .down));
    try t.expectEqual(breaker.Door.through, never.door(11 * s));
    try t.expectEqual(remote.Verdict.remote, weigh(&never, 11 * s));
}

test "breaker: an open breaker keeps prompts here under route always too" {
    var r: Route = .{ .mode = .always, .breaker = .{ .trip = 1 } };
    try t.expectEqual(remote.Verdict.remote, weigh(&r, 0));
    _ = r.finish(0, .down);
    try t.expectEqual(remote.Verdict.down, weigh(&r, s));
}

test "breaker: the pair is down for a missing link or a silent server, up for an answer" {
    try t.expect(breaker.pairDown(error.LinkDown) and breaker.pairDown(error.Timeout) and breaker.pairDown(error.TriggerTimedOut));
    try t.expect(breaker.pairDown(error.NoMailbox) and breaker.pairDown(error.ConnectionRefused) and breaker.pairDown(error.LinkLost));
    for ([_]anyerror{ error.Refused, error.Mismatch, error.Protocol, error.PrefixTooLong, error.PartialPrefix, error.PartialState, error.OutOfMemory, error.NoEntropy, error.TriggerClosed, error.Aborted, error.TriggerUnauthorized, error.TriggerRefused, error.Corrupt, error.BadManifest, error.NotHeld, error.Unauthorized, error.WriteFailed }) |e|
        try t.expect(!breaker.pairDown(e));
}

test "deadline: TF_PREFILL_TIMEOUT_S as given, else 15 s and three times the predicted pull between 30 s and 30 min" {
    var r: Route = .{};
    try t.expectEqual(@as(u64, 30), r.deadlineS(100)); // 15 + 3 * 1.1 = 18.3: the floor
    try t.expectEqual(@as(u64, 43), r.deadlineS(8191)); // 15 + 3 * (8.191 + 1) = 42.6
    try t.expectEqual(@as(u64, 1800), r.deadlineS(900000));
    r.remote_tps.v = 2000; // a faster pair tightens it: 15 + 3 * (4.1 + 1) = 30.3
    try t.expectEqual(@as(u64, 31), r.deadlineS(8191));
    var set: Route = .{ .timeout_s = 600 };
    try t.expectEqual(@as(u64, 600), set.deadlineS(100));
}

/// A pull as `guarded` sees one: it counts its runs and liveness checks, and fails or not as told.
const Fake = struct {
    route: Route,
    io: std.Io = t.io,
    up: bool = true,
    fail: ?anyerror = null,
    runs: u32 = 0,
    probes: u32 = 0,

    pub fn alive(f: *Fake) bool {
        f.probes += 1;
        return f.up;
    }

    pub fn run(f: *Fake, _: []const u32, _: u64, _: u64, _: *const std.atomic.Value(bool), outcome: *breaker.Outcome) anyerror!?remote.Filled {
        f.runs += 1;
        if (f.fail) |err| return err;
        outcome.* = .ok;
        return null;
    }
};

test "guarded: failures open the breaker, a dead pair fails its liveness check, a live one closes it again" {
    var no: std.atomic.Value(bool) = .init(false);
    var f: Fake = .{ .route = .{ .breaker = .{ .trip = 2, .cooldown_ns = 30 * std.time.ns_per_ms } }, .fail = error.LinkDown };
    const prompt = [_]u32{1};
    try t.expectError(error.LinkDown, remote.guarded(&f, &prompt, 0, 1, &no));
    try t.expectError(error.LinkDown, remote.guarded(&f, &prompt, 0, 1, &no));
    try t.expectEqual(@as(?remote.Filled, null), try remote.guarded(&f, &prompt, 0, 1, &no)); // open: the pair is not asked
    try t.expectEqual(@as(u32, 2), f.runs);
    try t.expectEqual(remote.Verdict.down, weigh(&f.route, Route.nowNs(t.io)));
    try std.Io.sleep(t.io, .fromMilliseconds(40), .awake);
    f.up = false;
    try t.expectEqual(@as(?remote.Filled, null), try remote.guarded(&f, &prompt, 0, 1, &no)); // the trial: /health does not answer
    try t.expectEqual(@as(u32, 1), f.probes);
    try t.expectEqual(@as(u32, 2), f.runs);
    try std.Io.sleep(t.io, .fromMilliseconds(40), .awake);
    f.up = true;
    f.fail = error.Refused; // an answer, though a refusal: no failure, the breaker stays as it was
    try t.expectError(error.Refused, remote.guarded(&f, &prompt, 0, 1, &no));
    try t.expectEqual(@as(u32, 3), f.runs);
    f.fail = null;
    try t.expectEqual(@as(?remote.Filled, null), try remote.guarded(&f, &prompt, 0, 1, &no));
    try t.expectEqual(@as(u32, 4), f.runs);
    try t.expectEqual(remote.Verdict.remote, weigh(&f.route, Route.nowNs(t.io)));
    try t.expectEqual(breaker.Door.through, f.route.door(Route.nowNs(t.io)));
}
