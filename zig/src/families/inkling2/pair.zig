//! Two virtual ranks on one GPU (test tools only): a two-rank run's gathers as copies between two engines' buffers.
const std = @import("std");
const cuda = @import("cuda");

/// Bytes a rank copies from its peer's buffer into the same offsets of its own.
pub const Region = struct { off: u64, len: u64 };

/// The peer's part of an in-place all-gather over [2][bytes].
pub fn peerPart(rank: usize, bytes: u64) Region {
    return .{ .off = (1 - rank) * bytes, .len = bytes };
}

/// The two ranks' host threads meet here; a rank that fails marks it, so its peer stops instead of waiting forever.
pub const Meet = struct {
    arrived: std.atomic.Value(u32) = .init(0),
    round: std.atomic.Value(u32) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    pub fn wait(m: *Meet) !void {
        const r = m.round.load(.acquire);
        if (m.arrived.fetchAdd(1, .acq_rel) == 1) {
            m.arrived.store(0, .monotonic);
            _ = m.round.fetchAdd(1, .release);
            return;
        }
        while (m.round.load(.acquire) == r) {
            if (m.failed.load(.acquire)) return error.VirtualPeerFailed;
            std.Thread.yield() catch {};
        }
    }

    pub fn fail(m: *Meet) void {
        m.failed.store(true, .release);
    }
};

/// Rank 0's and rank 1's engines on one device, each on its own stream and host thread.
pub const Pair = struct {
    d: *const cuda.Driver,
    meet: Meet = .{},
    base: [2]u64 = .{ 0, 0 }, // each rank's buffer of the exchange in progress
    ready: [2]cuda.Event,
    done: [2]cuda.Event,

    pub fn init(d: *const cuda.Driver) !Pair {
        var ev: [4]cuda.Event = undefined;
        var made: usize = 0;
        errdefer for (ev[0..made]) |*e| e.deinit();
        for (&ev) |*e| {
            e.* = try cuda.Event.init(d, false);
            made += 1;
        }
        return .{ .d = d, .ready = .{ ev[0], ev[1] }, .done = .{ ev[2], ev[3] } };
    }

    pub fn deinit(p: *Pair) void {
        for ([_]*cuda.Event{ &p.ready[0], &p.ready[1], &p.done[0], &p.done[1] }) |e| e.deinit();
    }

    /// One exchange: each rank copies `regions` of the peer's buffer into its own; both streams wait for both copies.
    pub fn pull(p: *Pair, rank: usize, s: cuda.Stream, own: u64, regions: []const Region) !void {
        const peer = 1 - rank;
        try p.ready[rank].record(s);
        p.base[rank] = own;
        try p.meet.wait();
        try s.wait(p.ready[peer]);
        for (regions) |r| {
            const view: cuda.DeviceBuffer = .{ .d = p.d, .ptr = own + r.off, .len = r.len, .borrowed = true };
            try view.copyFrom(0, p.base[peer] + r.off, r.len, s.handle);
        }
        try p.done[rank].record(s);
        try p.meet.wait();
        try s.wait(p.done[peer]);
    }

    /// An in-place all-gather: `buf` holds [2][bytes], each rank's own part written.
    pub fn allGather(p: *Pair, rank: usize, s: cuda.Stream, buf: u64, bytes: u64) !void {
        try p.pull(rank, s, buf, &.{peerPart(rank, bytes)});
    }

    /// Rank 0's first `bytes` of `buf` into rank 1's.
    pub fn broadcast(p: *Pair, rank: usize, s: cuda.Stream, buf: u64, bytes: u64) !void {
        const all = [_]Region{.{ .off = 0, .len = bytes }};
        try p.pull(rank, s, buf, if (rank == 0) &.{} else &all);
    }
};

test "pair: each rank copies the other's part of a gather, and two threads meet round after round" {
    try std.testing.expectEqual(Region{ .off = 64, .len = 64 }, peerPart(0, 64));
    try std.testing.expectEqual(Region{ .off = 0, .len = 64 }, peerPart(1, 64));
    var m: Meet = .{};
    var seen: [2]std.atomic.Value(u32) = .{ .init(0), .init(0) };
    const Run = struct {
        fn go(mm: *Meet, me: usize, s: *[2]std.atomic.Value(u32), bad: *std.atomic.Value(bool)) void {
            for (0..1000) |i| {
                _ = s[me].fetchAdd(1, .acq_rel);
                mm.wait() catch return bad.store(true, .release);
                if (s[1 - me].load(.acquire) < i + 1) bad.store(true, .release); // the peer reached this round too
                mm.wait() catch return bad.store(true, .release);
            }
        }
    };
    var bad: std.atomic.Value(bool) = .init(false);
    const t = try std.Thread.spawn(.{}, Run.go, .{ &m, 1, &seen, &bad });
    Run.go(&m, 0, &seen, &bad);
    t.join();
    try std.testing.expect(!bad.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1000), seen[0].load(.acquire));
    try std.testing.expectEqual(@as(u32, 1000), seen[1].load(.acquire));
}

test "pair: a rank waiting on a peer that failed stops with an error instead of waiting forever" {
    var m: Meet = .{};
    m.fail();
    try std.testing.expectError(error.VirtualPeerFailed, m.wait());
}
