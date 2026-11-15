//! Both hosts finish encoding before either submits GPU work that can wait for the peer.
const std = @import("std");

pub const timeout_ns = 30 * std.time.ns_per_s;

pub fn ready(peer: u64, through: u64) bool {
    return peer >= through;
}

const Clock = struct {
    fn now(_: Clock) !u64 {
        return @import("fabric").words.checkedNowNs();
    }
};

pub fn prepare(t: anytype, flag: usize) !void {
    return prepareWith(t, flag, Clock{});
}

fn check(t: anytype) !void {
    if (t.failed.load(.acquire)) return error.TpPeerFailed;
    if (t.stop.load(.acquire) or t.quitting.load(.acquire)) return error.TpStopping;
}

pub fn prepareWith(t: anytype, flag: usize, clock: anytype) !void {
    const through = t.seq;
    if (through <= t.submitted.load(.acquire)) return;
    try check(t);
    if (t.local) {
        t.submitted.store(through, .release);
        return;
    }
    const until = (try clock.now()) +| timeout_ns;
    try t.ep.writeSignalFromUntil(t.peer, 0, 0, 0, flag, through, until);
    while (true) {
        try check(t);
        if (try clock.now() >= until) return error.TpSubmitPeerSilent;
        if (ready(@atomicLoad(u64, t.word64(flag), .acquire), through)) {
            try check(t);
            t.submitted.store(through, .release);
            return;
        }
        std.atomic.spinLoopHint();
    }
}

const FakeEndpoint = struct {
    sent: u64 = 0,
    calls: usize = 0,
    until: u64 = 0,
    broken: bool = false,

    pub fn writeSignalFromUntil(e: *FakeEndpoint, _: u32, _: usize, _: usize, _: usize, _: usize, value: u64, until: u64) !void {
        e.calls += 1;
        if (e.broken) return error.TransportFailed;
        e.sent = value;
        e.until = until;
    }
};

const Fake = struct {
    seq: u64 = 0x1_0000_0003,
    submitted: std.atomic.Value(u64) = .init(3),
    failed: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),
    quitting: std.atomic.Value(bool) = .init(false),
    peer: u32 = 1,
    local: bool = false,
    ep: *FakeEndpoint,
    peer_word: u64 = 3,
    gpu_deadline: @import("tp_deadline.zig").State = .{},

    pub fn word64(t: *Fake, _: usize) *u64 {
        return &t.peer_word;
    }
};

const FakeClock = struct {
    t: *Fake,
    times: []const u64,
    at: usize = 0,
    publish_at: ?u64 = null,

    pub fn now(c: *FakeClock) !u64 {
        if (c.at >= c.times.len) return error.ClockUnavailable;
        const value = c.times[c.at];
        c.at += 1;
        if (c.publish_at) |when| if (value >= when) @atomicStore(u64, &c.t.peer_word, c.t.seq, .release);
        return value;
    }
};

test "submission readiness compares the complete sequence, including its high bits" {
    try std.testing.expect(!ready(3, 0x1_0000_0003));
    try std.testing.expect(!ready(0x1_0000_0002, 0x1_0000_0003));
    try std.testing.expect(ready(0x1_0000_0003, 0x1_0000_0003));
    try std.testing.expect(ready(0x1_0000_0004, 0x1_0000_0003));
}

test "an old readiness signal cannot submit the current GPU batch" {
    var ep: FakeEndpoint = .{};
    var t: Fake = .{ .ep = &ep };
    var clock: FakeClock = .{ .t = &t, .times = &.{ 0, 1, timeout_ns } };
    try std.testing.expectError(error.TpSubmitPeerSilent, prepareWith(&t, 64, &clock));
    try std.testing.expectEqual(@as(u64, 3), t.submitted.load(.acquire));
    try std.testing.expectEqual(t.seq, ep.sent);
    try std.testing.expectEqual(timeout_ns, ep.until);
    try std.testing.expect(t.gpu_deadline.pending == null);
}

test "cold host encoding and peer readiness do not arm a GPU deadline" {
    var ep: FakeEndpoint = .{};
    var t: Fake = .{ .ep = &ep };
    const cold = 10 * std.time.ns_per_s;
    var clock: FakeClock = .{ .t = &t, .times = &.{ cold, cold + 1, cold + 6 * std.time.ns_per_s }, .publish_at = cold + 6 * std.time.ns_per_s };
    try std.testing.expect(t.gpu_deadline.pending == null);
    try prepareWith(&t, 64, &clock);
    try std.testing.expectEqual(t.seq, t.submitted.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), ep.calls);
    try std.testing.expect(t.gpu_deadline.pending == null);
}

test "a host command buffer without new pair jobs needs no peer barrier" {
    var ep: FakeEndpoint = .{};
    var t: Fake = .{ .ep = &ep };
    t.seq = t.submitted.load(.acquire);
    var clock: FakeClock = .{ .t = &t, .times = &.{} };
    try prepareWith(&t, 64, &clock);
    try std.testing.expectEqual(@as(usize, 0), ep.calls);
    try std.testing.expectEqual(@as(usize, 0), clock.at);
}

test "transport, clock and shutdown failures leave the GPU batch unsubmitted" {
    var ep: FakeEndpoint = .{ .broken = true };
    var t: Fake = .{ .ep = &ep };
    var clock: FakeClock = .{ .t = &t, .times = &.{0} };
    try std.testing.expectError(error.TransportFailed, prepareWith(&t, 64, &clock));
    ep.broken = false;
    try std.testing.expectError(error.ClockUnavailable, prepareWith(&t, 64, &clock));
    t.stop.store(true, .release);
    try std.testing.expectError(error.TpStopping, prepareWith(&t, 64, &clock));
    t.stop.store(false, .release);
    t.failed.store(true, .release);
    try std.testing.expectError(error.TpPeerFailed, prepareWith(&t, 64, &clock));
    try std.testing.expectEqual(@as(u64, 3), t.submitted.load(.acquire));
    try std.testing.expect(t.gpu_deadline.pending == null);
}

test "local submission completes without an endpoint call or peer readiness" {
    var ep: FakeEndpoint = .{ .broken = true };
    var t: Fake = .{ .ep = &ep, .local = true };
    var clock: FakeClock = .{ .t = &t, .times = &.{} };
    try prepareWith(&t, 64, &clock);
    try std.testing.expectEqual(t.seq, t.submitted.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), ep.calls);
    try std.testing.expectEqual(@as(usize, 0), clock.at);
    t.seq += 1;
    t.failed.store(true, .release);
    try std.testing.expectError(error.TpPeerFailed, prepareWith(&t, 64, &clock));
    try std.testing.expectEqual(t.seq - 1, t.submitted.load(.acquire));
    try std.testing.expectEqual(@as(usize, 0), ep.calls);
}
