//! Idle request waits probe transport completion without limiting healthy idle time.
const std = @import("std");

pub const probe_period_ns = 250 * std.time.ns_per_ms;
pub const probe_budget_ns = std.time.ns_per_s;

const Clock = struct {
    fn now(_: Clock) !u64 {
        return @import("fabric").words.checkedNowNs();
    }

    fn pause(_: Clock) void {
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 50_000 };
        _ = std.c.nanosleep(&ts, null);
    }
};

pub fn wait(t: anytype, request_flag: usize, probe_flag: usize) !bool {
    return waitWith(t, request_flag, probe_flag, Clock{});
}

pub fn waitWith(t: anytype, request_flag: usize, probe_flag: usize, clock: anytype) !bool {
    var next_probe: u64 = 0;
    while (true) {
        if (!try running(t)) return false;
        if (@atomicLoad(u64, t.word64(request_flag), .acquire) >= t.req) return running(t);
        const now = try clock.now();
        if (now >= next_probe) {
            const until = now +| probe_budget_ns;
            try t.ep.writeSignalFromUntil(t.peer, 0, 0, 0, probe_flag, t.req, until);
            try t.ep.flushPeerUntil(t.peer, until);
            next_probe = now +| probe_period_ns;
        }
        clock.pause();
    }
}

fn running(t: anytype) !bool {
    if (t.failed.load(.acquire)) return error.TpLinkFailed;
    return !t.quitting.load(.acquire) and !t.stop.load(.acquire);
}

const FakeEndpoint = struct {
    probes: usize = 0,
    flushes: usize = 0,
    until: u64 = 0,
    failure: ?anyerror = null,

    pub fn writeSignalFromUntil(e: *FakeEndpoint, _: u32, _: usize, _: usize, _: usize, flag: usize, _: u64, until: u64) !void {
        try std.testing.expectEqual(@as(usize, 16), flag);
        e.probes += 1;
        e.until = until;
    }

    pub fn flushPeerUntil(e: *FakeEndpoint, _: u32, until: u64) !void {
        e.flushes += 1;
        try std.testing.expectEqual(e.until, until);
        if (e.failure) |err| return err;
    }
};

const Fake = struct {
    req: u64 = 0x1_0000_0001,
    peer_word: u64 = 1,
    peer: u32 = 0,
    ep: *FakeEndpoint,
    failed: std.atomic.Value(bool) = .init(false),
    quitting: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),

    pub fn word64(t: *Fake, _: usize) *u64 {
        return &t.peer_word;
    }
};

const FakeClock = struct {
    t: *Fake,
    at: usize = 0,
    times: []const u64,
    publish_at: ?u64 = null,

    pub fn now(c: *FakeClock) !u64 {
        if (c.at >= c.times.len) return error.ClockUnavailable;
        const value = c.times[c.at];
        c.at += 1;
        if (c.publish_at) |when| if (value >= when) @atomicStore(u64, &c.t.peer_word, c.t.req, .release);
        return value;
    }

    pub fn pause(_: *FakeClock) void {}
};

test "idle peer death is detected by a bounded transport probe" {
    var ep: FakeEndpoint = .{ .failure = error.PeerDown };
    var t: Fake = .{ .ep = &ep };
    var clock: FakeClock = .{ .t = &t, .times = &.{0} };
    try std.testing.expectError(error.PeerDown, waitWith(&t, 8, 16, &clock));
    try std.testing.expectEqual(@as(usize, 1), ep.probes);
    try std.testing.expectEqual(@as(usize, 1), ep.flushes);
    try std.testing.expectEqual(probe_budget_ns, ep.until);
}

test "a healthy peer may remain idle indefinitely and stale request high bits do not wake it" {
    var ep: FakeEndpoint = .{};
    var t: Fake = .{ .ep = &ep };
    const hours = 5 * 3600 * std.time.ns_per_s;
    var clock: FakeClock = .{ .t = &t, .times = &.{ 0, hours, hours + probe_period_ns }, .publish_at = hours + probe_period_ns };
    try std.testing.expect(try waitWith(&t, 8, 16, &clock));
    try std.testing.expectEqual(@as(usize, 3), ep.probes);
    try std.testing.expectEqual(ep.probes, ep.flushes);
}

test "failed and stopped links cannot accept even an already landed request" {
    var ep: FakeEndpoint = .{};
    var t: Fake = .{ .ep = &ep };
    t.peer_word = t.req;
    var clock: FakeClock = .{ .t = &t, .times = &.{} };
    t.failed.store(true, .release);
    try std.testing.expectError(error.TpLinkFailed, waitWith(&t, 8, 16, &clock));
    t.failed.store(false, .release);
    t.stop.store(true, .release);
    try std.testing.expect(!try waitWith(&t, 8, 16, &clock));
    t.stop.store(false, .release);
    t.quitting.store(true, .release);
    try std.testing.expect(!try waitWith(&t, 8, 16, &clock));
    try std.testing.expectEqual(@as(usize, 0), ep.probes);
}

test "probe timeout ends the idle wait without declaring a request arrived" {
    var ep: FakeEndpoint = .{ .failure = error.Timeout };
    var t: Fake = .{ .ep = &ep };
    var clock: FakeClock = .{ .t = &t, .times = &.{0} };
    try std.testing.expectError(error.Timeout, waitWith(&t, 8, 16, &clock));
    try std.testing.expectEqual(@as(u64, 1), t.peer_word);
}
