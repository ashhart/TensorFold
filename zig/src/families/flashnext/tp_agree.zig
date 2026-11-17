//! The pair's shared steps: rank 0 publishes a step's decision only after rank 1 has taken the step before it.
const std = @import("std");
const words = @import("fabric").words;
const host_wait = @import("tp_host_wait.zig");

/// One step's decision, the same on both Macs: whether to stop, and the next round's draft depth.
pub const Decision = struct { quit: bool, depth: u32 };

/// Rank 0 waits until rank 1 has taken the previous step, then publishes this one; rank 1 takes it and says so.
pub fn agree(t: anytype, ctrl_flag: usize, ack_flag: usize, quit: bool, depth: u32, silent_ns: u64) !Decision {
    if (t.failed.load(.acquire)) return error.TpLinkFailed;
    t.ctrl += 1;
    if (t.rank == 0) {
        host_wait.wait(t.word64(ack_flag), t.ctrl - 1, .{ .failed = &t.failed, .quitting = &t.quitting, .stop = &t.stop }, silent_ns) catch |err| {
            if (err != error.TpStopping) t.handoff.fail(t.seq, err);
            return err;
        };
        try t.rd.signal(t.peer, ctrl_flag, (t.ctrl << 6) | (@as(u64, depth & 31) << 1) | @intFromBool(quit));
        return .{ .quit = quit, .depth = depth };
    }
    const began = try words.checkedNowNs();
    while (true) {
        const v = @atomicLoad(u64, t.word64(ctrl_flag), .acquire);
        if (v >> 6 >= t.ctrl) {
            try t.rd.signal(t.peer, ack_flag, t.ctrl);
            return .{ .quit = quit or (v >> 6 == t.ctrl and v & 1 != 0), .depth = @intCast((v >> 1) & 31) };
        }
        if ((try words.checkedNowNs()) -| began > silent_ns) return error.TpPeerSilent;
        std.atomic.spinLoopHint();
    }
}

const CTRL = 0;
const ACK = 8;

/// One Mac's side of the link: its window's words, and the peer's window its signals land in.
const Fake = struct {
    rank: u32,
    peer: u32,
    ctrl: u64 = 0,
    seq: u64 = 7,
    failed: std.atomic.Value(bool) = .init(false),
    quitting: std.atomic.Value(bool) = .init(false),
    stop: std.atomic.Value(bool) = .init(false),
    win: [2]u64 = .{ 0, 0 },
    rd: Link,
    handoff: Failures = .{},
    /// Rank 0's ack word at each publish, the steps it published and their words.
    acks: [64]u64 = undefined,
    sent: [64]u64 = undefined,
    published: usize = 0,

    const Link = struct {
        to: *Fake,
        from: *Fake,
        pub fn signal(l: Link, _: u32, off: usize, v: u64) !void {
            if (off == CTRL) {
                l.from.acks[l.from.published] = @atomicLoad(u64, l.from.word64(ACK), .acquire);
                l.from.sent[l.from.published] = v;
                l.from.published += 1;
            }
            @atomicStore(u64, l.to.word64(off), v, .release);
        }
    };
    const Failures = struct {
        failures: usize = 0,
        pub fn fail(f: *Failures, _: u64, _: anyerror) void {
            f.failures += 1;
        }
    };
    pub fn word64(f: *Fake, off: usize) *u64 {
        return &f.win[off / 8];
    }
};

fn link(macs: *[2]Fake) void {
    for (0..2) |k| {
        macs[k] = .{ .rank = @intCast(k), .peer = @intCast(1 - k), .rd = .{ .to = &macs[1 - k], .from = &macs[k] } };
    }
}

const Steps = 24;
fn depthAt(step: usize) u32 {
    return @intCast((step * 7) % 31);
}

fn rankZero(t: *Fake, out: *[Steps]Decision, err: *?anyerror) void {
    for (0..Steps) |s| out[s] = agree(t, CTRL, ACK, s + 1 == Steps, depthAt(s), std.time.ns_per_s) catch |e| {
        err.* = e;
        return;
    };
}

fn rankOne(t: *Fake, out: *[Steps]Decision, err: *?anyerror) void {
    for (0..Steps) |s| {
        @import("tp_handoff.zig").sleep(if (s % 3 == 0) 3_000_000 else 200_000); // a slower rank 1, by fits
        out[s] = agree(t, CTRL, ACK, false, 0, std.time.ns_per_s) catch |e| {
            err.* = e;
            return;
        };
    }
}

test "agree: a slow rank 1 takes every step's own depth and the quit, and rank 0 is never two steps ahead" {
    var macs: [2]Fake = undefined;
    link(&macs);
    var got: [2][Steps]Decision = undefined;
    var errs: [2]?anyerror = .{ null, null };
    const zero = try std.Thread.spawn(.{}, rankZero, .{ &macs[0], &got[0], &errs[0] });
    rankOne(&macs[1], &got[1], &errs[1]);
    zero.join();
    try std.testing.expectEqual(@as(?anyerror, null), errs[0]);
    try std.testing.expectEqual(@as(?anyerror, null), errs[1]);
    for (0..Steps) |s| {
        try std.testing.expectEqual(got[0][s], got[1][s]);
        try std.testing.expectEqual(Decision{ .quit = s + 1 == Steps, .depth = depthAt(s) }, got[1][s]);
        try std.testing.expectEqual(@as(u64, s), macs[0].acks[s]); // step s + 1 waited for rank 1 to take step s
        try std.testing.expectEqual(@as(u64, s + 1), macs[0].sent[s] >> 6);
    }
    try std.testing.expectEqual(@as(usize, Steps), macs[0].published);
    try std.testing.expectEqual(macs[0].ctrl, macs[1].ctrl);
    try std.testing.expectEqual(@as(u64, Steps), macs[0].win[ACK / 8]);
}

test "agree: rank 0 does not overwrite an unread step; a silent rank 1 fails the link with the peer-silence error" {
    var macs: [2]Fake = undefined;
    link(&macs);
    const first = try agree(&macs[0], CTRL, ACK, false, 5, std.time.ns_per_ms); // a request's first step waits for nothing
    try std.testing.expectEqual(Decision{ .quit = false, .depth = 5 }, first);
    try std.testing.expectError(error.TpPeerSilent, agree(&macs[0], CTRL, ACK, true, 9, 20 * std.time.ns_per_ms));
    try std.testing.expectEqual(@as(u64, (1 << 6) | (5 << 1)), macs[1].win[CTRL / 8]); // step 1 still there to read
    try std.testing.expectEqual(@as(usize, 1), macs[0].published);
    try std.testing.expectEqual(@as(usize, 1), macs[0].handoff.failures);
    try std.testing.expectEqual(Decision{ .quit = false, .depth = 5 }, try agree(&macs[1], CTRL, ACK, false, 0, std.time.ns_per_ms));
    try std.testing.expectEqual(@as(u64, 1), macs[0].win[ACK / 8]);
    try std.testing.expectError(error.TpPeerSilent, agree(&macs[1], CTRL, ACK, false, 0, 20 * std.time.ns_per_ms));
    try std.testing.expectEqual(@as(u64, 1), macs[0].win[ACK / 8]); // no acknowledgement of a step never published
    macs[0].failed.store(true, .release);
    try std.testing.expectError(error.TpLinkFailed, agree(&macs[0], CTRL, ACK, false, 0, std.time.ns_per_ms));
}

test "agree: rank 0's quit reaches rank 1 at that step, and the next request's first step waits for its acknowledgement" {
    var macs: [2]Fake = undefined;
    link(&macs);
    _ = try agree(&macs[0], CTRL, ACK, true, 3, std.time.ns_per_ms);
    try std.testing.expectEqual(Decision{ .quit = true, .depth = 3 }, try agree(&macs[1], CTRL, ACK, false, 0, std.time.ns_per_ms));
    try std.testing.expectEqual(Decision{ .quit = false, .depth = 4 }, try agree(&macs[0], CTRL, ACK, false, 4, std.time.ns_per_ms));
    try std.testing.expectEqual(Decision{ .quit = false, .depth = 4 }, try agree(&macs[1], CTRL, ACK, false, 0, std.time.ns_per_ms));
    try std.testing.expectEqual(@as(u64, 2), macs[0].win[ACK / 8]);
}
