//! Host-only pair liveness: READY enables pulses; an idle command wait still notices a silent peer.
const std = @import("std");
const fabric = @import("fabric");

pub const BEAT = 320;
pub const interval_ns = std.time.ns_per_s;
pub const silence_ns = 10 * std.time.ns_per_s;

pub const State = struct {
    started: bool = false,
    seen: u64 = 0,
    changed: u64 = 0,
    next: u64 = 0,
    sent: u64 = 0,

    pub fn check(s: *State, now: u64, received: u64) !void {
        if (s.started and now -| s.changed >= silence_ns) return error.PairPeerSilent;
        if (!s.started or received > s.seen) {
            s.started = true;
            s.seen = received;
            s.changed = now;
        }
    }

    fn due(s: *State, now: u64) bool {
        if (now < s.next) return false;
        s.next = now +| interval_ns;
        return true;
    }
};

/// The relay owns sends, so a blocked native heartbeat is covered by its watchdog and bounded close.
pub fn pulse(p: anytype, state: *State, word: *const u64, flag: usize) void {
    if (!p.heartbeat_ready.load(.acquire)) return;
    const seq = p.submitted.load(.acquire);
    const now = fabric.words.checkedNowNs() catch |err| return p.handoff.fail(seq, err);
    state.check(now, @atomicLoad(u64, word, .acquire)) catch |err| return p.handoff.fail(seq, err);
    if (!state.due(now)) return;
    const until = now +| interval_ns;
    if (!p.handoff.armNative(until)) return;
    state.sent += 1;
    p.ep.writeSignalFromUntil(p.peer, 0, 0, 0, flag, state.sent, until) catch |err| return p.handoff.fail(seq, err);
    p.ep.flushPeerUntil(p.peer, until) catch |err| return p.handoff.fail(seq, err);
    _ = p.handoff.finishNative();
}

const Clock = struct {
    fn now(_: Clock) !u64 {
        return fabric.words.checkedNowNs();
    }
    fn pause(_: Clock, spins: usize) void {
        if (spins > 200_000) fabric.words.pause(20_000) else std.atomic.spinLoopHint();
    }
};

pub fn reach(p: anytype, word: *const u64, beat: *const u64, value: u64, seconds: u64) !void {
    return reachWith(p, word, beat, value, seconds, Clock{});
}

fn active(p: anytype) !void {
    if (p.stop.load(.acquire)) return error.PairStopped;
    if (p.failed.load(.acquire)) return error.PairLinkFailed;
}

fn reachWith(p: anytype, word: *const u64, beat: *const u64, value: u64, seconds: u64, clock: anytype) !void {
    try active(p);
    const start = try clock.now();
    var idle: State = .{};
    var spins: usize = 0;
    while (@atomicLoad(u64, word, .acquire) < value) {
        try active(p);
        const now = try clock.now();
        if (seconds == 0) idle.check(now, @atomicLoad(u64, beat, .acquire)) catch |err| {
            p.handoff.fail(p.x, err);
            return err;
        };
        if (seconds != 0 and now -| start >= seconds * std.time.ns_per_s) {
            p.handoff.fail(p.x, error.PairPeerSilent);
            return error.PairPeerSilent;
        }
        if (spins % 10000 == 0) {
            var reason: [256]u8 = undefined;
            if (p.ep.peerStatus(p.peer, &reason) != null) {
                p.handoff.fail(p.x, error.PairLinkFailed);
                return error.PairLinkFailed;
            }
        }
        spins += 1;
        clock.pause(spins);
    }
    try active(p);
}

test "pair heartbeat: a changing peer stays healthy through long idle periods" {
    var s: State = .{};
    for (0..121) |second| try s.check(second * interval_ns, second);
    try std.testing.expectEqual(@as(u64, 120), s.seen);
}

test "pair heartbeat: a silent peer expires inclusively and a late beat cannot replace expiry" {
    for ([_]u64{ 1, 2 }) |late| {
        var s: State = .{};
        try s.check(0, 1);
        try s.check(silence_ns - 1, 1);
        try std.testing.expectError(error.PairPeerSilent, s.check(silence_ns, late));
        try std.testing.expectEqual(@as(u64, 0), s.changed);
    }
}

const Fake = struct {
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    x: u64 = 0,
    peer: u32 = 0,
    ep: Endpoint = .{},
    handoff: Failure = undefined,
    const Endpoint = struct {
        pub fn peerStatus(_: @This(), _: u32, _: []u8) ?[]const u8 {
            return null;
        }
    };
    const Failure = struct {
        failed: *std.atomic.Value(bool),
        pub fn fail(f: @This(), _: u64, _: anyerror) void {
            f.failed.store(true, .release);
        }
    };
};

const FakeClock = struct {
    at: u64 = 0,
    command: ?*u64 = null,
    pub fn now(c: *@This()) !u64 {
        const tick = c.at;
        c.at += interval_ns;
        if (tick == 500 * interval_ns) {
            if (c.command) |word| word.* = 1;
        }
        return tick;
    }
    pub fn pause(_: *@This(), _: usize) void {}
};

test "pair heartbeat: idle follower fails without optional peer status while long model readiness is preserved" {
    var p: Fake = .{};
    p.handoff = .{ .failed = &p.failed };
    var command: u64 = 0;
    var beat: u64 = 0;
    var clock: FakeClock = .{};
    try std.testing.expectError(error.PairPeerSilent, reachWith(&p, &command, &beat, 1, 0, &clock));
    try std.testing.expect(p.failed.load(.acquire));
    p.failed.store(false, .release);
    clock = .{ .command = &command };
    try reachWith(&p, &command, &beat, 1, 600, &clock);
    try std.testing.expect(!p.failed.load(.acquire));
}
