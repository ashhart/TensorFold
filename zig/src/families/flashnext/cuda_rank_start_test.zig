//! The first collective waits for the peer's load past the collective's own bound; a closed peer ends the wait.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("cuda_rank_engine.zig");
const wire = @import("cuda_rank_wire.zig");
const nccl = cuda.nccl;

// The first collective's first call records when it came and is refused, so the test stops there.
const Fake = struct {
    var entered: ?u64 = null;
    fn count(_: nccl.Comm, _: *c_int) callconv(.c) c_int {
        entered = wire.now();
        return 1;
    }
    fn abort(_: nccl.Comm) callconv(.c) c_int {
        return 0;
    }
};

/// The peer still loading: after `delay_ms` it sends its load-ready frame, or closes its link as a failed load does.
const Peer = struct {
    fd: c_int,
    delay_ms: u64,
    leaves: bool,
    sent: u64 = 0,
    fn run(p: *Peer) void {
        const pause: std.c.timespec = .{ .sec = @intCast(p.delay_ms / 1000), .nsec = @intCast(p.delay_ms % 1000 * std.time.ns_per_ms) };
        _ = std.c.nanosleep(&pause, null);
        p.sent = wire.now();
        if (p.leaves) {
            _ = std.c.shutdown(p.fd, std.posix.SHUT.RDWR);
        } else {
            const link: wire.Link = .{ .fd = p.fd };
            link.sendHeader(.{ .kind = .ready, .arg = 7 }, wire.deadline(1000)) catch {};
        }
    }
};

fn session(rank: usize, pair: *[2]c_int) !ranks.Session {
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, pair) != 0) return error.SocketPair;
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    Fake.entered = null;
    var s: ranks.Session = .{ .gpa = std.testing.allocator, .rank = rank, .link = .{ .fd = pair[0] }, .collective = undefined };
    s.collective.failed = false;
    s.collective.comm = @ptrFromInt(0x1000);
    s.collective.link = .{ .fd = pair[0] };
    s.collective.timeout_ms = 50;
    s.collective.lib.api.ncclCommCount = Fake.count;
    s.collective.api.ncclCommAbort = Fake.abort;
    return s;
}

fn afterPeer(rank: usize, delay_ms: u64, leaves: bool) !struct { s: ranks.Session, peer: Peer, result: anyerror!void, waited: u64 } {
    var pair: [2]c_int = undefined;
    var s = try session(rank, &pair);
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    var peer: Peer = .{ .fd = pair[1], .delay_ms = delay_ms, .leaves = leaves };
    const thread = try std.Thread.spawn(.{}, Peer.run, .{&peer});
    var ctx: cuda.Context = undefined;
    var d: cuda.Driver = undefined;
    const started = wire.now();
    const result = s.afterLoad(std.testing.io, &ctx, .{ .d = &d, .handle = null });
    const waited = wire.now() - started;
    thread.join();
    return .{ .s = s, .peer = peer, .result = result, .waited = waited };
}

test "rank start: a peer loading past the collective's bound is awaited before the first collective" {
    for (0..2) |rank| {
        const run = try afterPeer(rank, 400, false);
        try std.testing.expectError(error.NcclFailed, run.result);
        try std.testing.expect(run.waited > 4 * run.s.collective.timeout_ms * std.time.ns_per_ms);
        try std.testing.expect(Fake.entered.? >= run.peer.sent);
    }
}

test "rank start: a peer that closes its link during the wait ends it at once, before any collective" {
    for (0..2) |rank| {
        const run = try afterPeer(rank, 100, true);
        try std.testing.expectError(error.RankDisconnected, run.result);
        try std.testing.expect(run.waited < 1100 * std.time.ns_per_ms);
        try std.testing.expect(Fake.entered == null);
        try std.testing.expect(run.s.stopped and run.s.collective.failed and run.s.collective.comm == null);
    }
}

test "rank start: the loader's first collective waits for the peer's load" {
    const load = @embedFile("cuda_lanes_load.zig");
    try std.testing.expect(std.mem.indexOf(u8, load, "benchmark_group.") == null);
    try std.testing.expect(std.mem.indexOf(u8, load, "try r.afterLoad(io, ctx, own.stream);") != null);
}
