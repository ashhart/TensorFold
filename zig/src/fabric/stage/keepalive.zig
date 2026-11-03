//! The ring's keep-alive as pure state: rank 0 sends an empty plan every 2 s when idle; a stage that misses 3 dies.
const std = @import("std");

pub const interval_ns: u64 = 2 * std.time.ns_per_s;
pub const most_missed: u64 = 3;

/// Rank 0's side: when the next keep-alive is due, from the last plan it sent (a round's or a keep-alive).
pub const Beat = struct {
    last_ns: u64,
    interval_ns: u64 = interval_ns,

    pub fn init(now_ns: u64) Beat {
        return .{ .last_ns = now_ns };
    }

    /// True once a whole interval has passed at `now_ns` without a plan sent.
    pub fn due(b: Beat, now_ns: u64) bool {
        return now_ns -| b.last_ns >= b.interval_ns;
    }

    /// A plan of any kind sent at `now_ns`.
    pub fn sent(b: *Beat, now_ns: u64) void {
        b.last_ns = @max(b.last_ns, now_ns);
    }
};

/// A stage's side: the ring is dead once the third keep-alive due since the last frame heard has not come.
pub const Watch = struct {
    last_ns: u64,
    interval_ns: u64 = interval_ns,
    most_missed: u64 = most_missed,

    pub fn init(now_ns: u64) Watch {
        return .{ .last_ns = now_ns };
    }

    /// A frame of any kind heard at `now_ns` (a plan, a keep-alive or rows).
    pub fn heard(w: *Watch, now_ns: u64) void {
        w.last_ns = @max(w.last_ns, now_ns);
    }

    /// The keep-alives due strictly before `now_ns` and not heard: one due exactly now is not missed yet.
    pub fn missed(w: Watch, now_ns: u64) u64 {
        const quiet = now_ns -| w.last_ns;
        return if (quiet == 0) 0 else (quiet - 1) / w.interval_ns;
    }

    /// `error.RingDead` once `most_missed` keep-alives are missed: the stage fails fast (RFC section 4).
    pub fn check(w: Watch, now_ns: u64) error{RingDead}!void {
        if (w.missed(now_ns) >= w.most_missed) return error.RingDead;
    }
};

const s = std.time.ns_per_s;

test "stage keepalive: rank 0 beats every 2 s when idle, and a round's plan resets the beat" {
    var b = Beat.init(10 * s);
    try std.testing.expect(!b.due(10 * s));
    try std.testing.expect(!b.due(12 * s - 1));
    try std.testing.expect(b.due(12 * s));
    b.sent(11 * s); // a round's plan
    try std.testing.expect(!b.due(12 * s));
    try std.testing.expect(b.due(13 * s));
    b.sent(5 * s); // an earlier time never moves the beat back
    try std.testing.expect(b.due(13 * s));
    try std.testing.expect(!b.due(0)); // a clock before the last send is not due
}

test "stage keepalive: a stage dies after three missed beats, not before, and any frame revives the count" {
    var w = Watch.init(100 * s);
    try std.testing.expectEqual(@as(u64, 0), w.missed(102 * s));
    try std.testing.expectEqual(@as(u64, 1), w.missed(102 * s + 1));
    try std.testing.expectEqual(@as(u64, 2), w.missed(106 * s));
    try w.check(106 * s);
    try std.testing.expectError(error.RingDead, w.check(106 * s + 1));
    w.heard(105 * s);
    try w.check(111 * s);
    try std.testing.expectError(error.RingDead, w.check(111 * s + 1));
    w.heard(1 * s); // a stale time is ignored
    try std.testing.expectError(error.RingDead, w.check(111 * s + 1));
    try w.check(0);
}

test "stage keepalive: a Beat and a Watch on one clock keep the ring alive, and a silent rank 0 ends it" {
    var b = Beat.init(0);
    var w = Watch.init(0);
    var now: u64 = 0;
    while (now < 60 * s) : (now += 100 * std.time.ns_per_ms) {
        if (b.due(now)) {
            b.sent(now);
            w.heard(now + 3 * std.time.ns_per_ms); // the wire's delay
        }
        try w.check(now);
    }
    const silent = now;
    while (now < silent + 7 * s) : (now += 100 * std.time.ns_per_ms) {
        if (w.check(now)) |_| {} else |_| break;
    }
    try std.testing.expectError(error.RingDead, w.check(now));
    try std.testing.expect(now > silent + 4 * s and now <= silent + 6 * s + 100 * std.time.ns_per_ms);
}
