//! The `Collective` contract over the TCP Kind on loopback, plus late frames, lost peers, big frames and aborts.
const std = @import("std");
const collective = @import("collective.zig");
const reference = @import("reference.zig");
const sock = @import("sock.zig");
const tcp = @import("tcp.zig");
const host_test = @import("host_test.zig");
const Collective = collective.Collective;

const gpa = std.testing.allocator;
const opts: tcp.Options = .{ .timeout_ns = 5 * std.time.ns_per_s };

fn addr(b: []u8) collective.DevicePtr {
    return @intFromPtr(b.ptr);
}

/// A failure the contract names (a test that accepts any error could pass on a crash of the transport).
fn named(e: anyerror) !void {
    if (e != error.Aborted and e != error.Invalid and e != error.Timeout and e != error.PeerFailed) return e;
}

fn pair() ![2]tcp.Rank {
    const fds = try tcp.loopback(2);
    return .{ tcp.Rank.init(gpa, 0, 2, &.{ -1, fds[0][1] }, opts), tcp.Rank.init(gpa, 1, 2, &.{ fds[1][0], -1 }, opts) };
}

const Later = struct {
    fn abort(r: *tcp.Rank, ms: u64) void {
        sock.sleepNs(ms * std.time.ns_per_ms);
        r.iface().abort();
    }

    fn recvFrom(r: *tcp.Rank, peer: u32, out: *?anyerror) void {
        var buf: [8]u8 = undefined;
        r.iface().recv(addr(&buf), 8, .u8, peer, null) catch |e| {
            out.* = e;
            return;
        };
        out.* = error.ShouldHaveFailed;
    }

    fn gather(r: *tcp.Rank, n: usize, out: *?anyerror) void {
        out.* = error.OutOfMemory;
        const in = gpa.alloc(u8, n) catch return;
        defer gpa.free(in);
        const all = gpa.alloc(u8, 2 * n) catch return;
        defer gpa.free(all);
        @memset(in, 7);
        r.iface().allGather(addr(in), addr(all), n, .u8, null) catch |e| {
            out.* = e;
            return;
        };
        out.* = error.ShouldHaveFailed;
    }
};

test "every call on two, three and four ranks over TCP equals the reference" {
    try tcp.run(gpa, 2, opts, {}, host_test.Calls.go);
    try tcp.run(gpa, 3, opts, {}, host_test.Calls.go);
    try tcp.run(gpa, 4, opts, {}, host_test.Calls.go);
}

test "exchange, send and recv over TCP between two ranks, and in pairs of four" {
    try tcp.run(gpa, 2, opts, {}, host_test.Pairwise.go);
    try tcp.run(gpa, 4, opts, {}, host_test.Pairwise.go);
}

test "a desynchronised call over TCP is refused on every rank" {
    try tcp.run(gpa, 2, opts, {}, host_test.Desync.go);
    try tcp.run(gpa, 2, opts, {}, host_test.VarDesync.go);
}

test "abort over TCP frees a peer waiting in a collective" {
    try tcp.run(gpa, 2, opts, {}, host_test.Aborts.go);
}

test "a variable-length all-gather over TCP moves each rank's length" {
    try tcp.run(gpa, 2, opts, @as(u32, 0), host_test.VarGather.go);
    try tcp.run(gpa, 3, opts, @as(u32, 0), host_test.VarGather.go);
}

const KindOk = struct {
    fn go(c: Collective, _: u32, _: void) !void {
        try std.testing.expectEqual(collective.Kind.tcp, c.kind());
        try c.barrier();
    }
};

test "the TCP Kind names itself" {
    try tcp.run(gpa, 2, opts, {}, KindOk.go);
}

const Late = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        var a: [16]u8 = @splat(@intCast(rank));
        var b: [32]u8 = undefined;
        if (rank == 1) {
            // rank 1 comes after rank 0's deadline: its call may still finish with rank 0's frame, or fail by name
            sock.sleepNs(400 * std.time.ns_per_ms);
            if (c.allGather(addr(&a), addr(&b), 16, .u8, null)) |_| {
                try std.testing.expectEqualSlices(u8, &@as([16]u8, @splat(0)), b[0..16]);
                try std.testing.expectEqualSlices(u8, &@as([16]u8, @splat(1)), b[16..32]);
            } else |e| try named(e);
            if (c.allGather(addr(&a), addr(&b), 16, .u8, null)) |_| return error.ShouldHaveFailed else |e| try named(e);
            return;
        }
        const t0 = sock.nowNs();
        try std.testing.expectError(error.Timeout, c.allGather(addr(&a), addr(&b), 16, .u8, null));
        try std.testing.expect(sock.nowNs() - t0 < 300 * std.time.ns_per_ms);
        try std.testing.expectError(error.Timeout, c.check());
        try std.testing.expectError(error.Aborted, c.allGather(addr(&a), addr(&b), 16, .u8, null));
    }
};

test "a frame that does not come by the deadline is Timeout, and the communicator is closed" {
    try tcp.run(gpa, 2, .{ .timeout_ns = 150 * std.time.ns_per_ms }, {}, Late.go);
}

test "a peer that hangs up without an abort is PeerFailed" {
    const fds = try tcp.loopback(2);
    var r0 = tcp.Rank.init(gpa, 0, 2, &.{ -1, fds[0][1] }, opts);
    defer r0.deinit();
    sock.close(fds[1][0]); // rank 1's process is gone: its socket closes, no abort frame
    var a: [8]u8 = @splat(0);
    var b: [16]u8 = undefined;
    try std.testing.expectError(error.PeerFailed, r0.iface().allGather(addr(&a), addr(&b), 8, .u8, null));
    try std.testing.expectError(error.PeerFailed, r0.iface().check());
}

fn cpuNs() u64 {
    var ru: std.c.rusage = undefined;
    _ = std.c.getrusage(std.c.rusage.SELF, &ru);
    const us: u64 = @intCast(ru.utime.sec * 1_000_000 + ru.utime.usec + ru.stime.sec * 1_000_000 + ru.stime.usec);
    return us * std.time.ns_per_us;
}

test "a peer outside the call that hangs up partway through a header is not polled in a busy loop" {
    const fds = try tcp.loopback(3);
    var r0 = tcp.Rank.init(gpa, 0, 3, &.{ -1, fds[0][1], fds[0][2] }, .{ .timeout_ns = 300 * std.time.ns_per_ms });
    defer r0.deinit();
    defer for ([_]sock.Fd{ fds[1][0], fds[1][2], fds[2][1] }) |fd| sock.close(fd);
    try sock.sendAll(fds[2][0], "0123456789"); // rank 2 dies 10 bytes into a header, with a plain hang-up
    sock.close(fds[2][0]);
    var b: [8]u8 = undefined;
    const c0 = cpuNs();
    const t0 = sock.nowNs();
    try std.testing.expectError(error.Timeout, r0.iface().recv(addr(&b), 8, .u8, 1, null));
    // a watch that never stops spins for the whole wait; one that stops leaves the wait asleep in poll
    try std.testing.expect(cpuNs() - c0 < (sock.nowNs() - t0) / 4);
}

const Big = struct {
    fn go(c: Collective, rank: u32, _: void) !void {
        // 4 MiB a rank each way at once: far past the sockets' buffers, so a rank that only sent would wait forever
        const n = 4 << 20;
        const mine = try gpa.alloc(u8, n);
        defer gpa.free(mine);
        const out = try gpa.alloc(u8, n * c.world());
        defer gpa.free(out);
        reference.fill(mine, .u8, rank, 99);
        try c.allGather(addr(mine), addr(out), n, .u8, null);
        const want = try gpa.alloc(u8, n);
        defer gpa.free(want);
        for (0..c.world()) |q| {
            reference.fill(want, .u8, @intCast(q), 99);
            try std.testing.expectEqualSlices(u8, want, out[q * n ..][0..n]);
        }
        const fl = n / 4;
        reference.fill(mine, .f32, rank, 5);
        var views: [tcp.max_ranks][]const u8 = undefined;
        const ins = try gpa.alloc(u8, n * c.world());
        defer gpa.free(ins);
        for (0..c.world()) |q| {
            reference.fill(ins[q * n ..][0..n], .f32, @intCast(q), 5);
            views[q] = ins[q * n ..][0..n];
        }
        reference.reduce(want, views[0..c.world()], .f32, .sum);
        try c.allReduce(addr(mine), addr(mine), fl, .f32, .sum, null);
        try std.testing.expectEqualSlices(u8, want, mine);
    }
};

test "frames far past the socket buffers move both ways at once, bit for bit" {
    try tcp.run(gpa, 2, opts, {}, Big.go);
    try tcp.run(gpa, 3, opts, {}, Big.go);
}

test "abort from another thread wakes this rank's wait and the peer's, both as Aborted" {
    var rs = try pair();
    defer for (&rs) |*r| r.deinit();
    // both ranks wait on each other with nothing queued, so each can only learn of the abort from the abort frame
    var e0: ?anyerror = null;
    var e1: ?anyerror = null;
    const t1 = try std.Thread.spawn(.{}, Later.recvFrom, .{ &rs[1], @as(u32, 0), &e1 });
    const ta = try std.Thread.spawn(.{}, Later.abort, .{ &rs[0], @as(u64, 50) });
    const t0 = sock.nowNs();
    Later.recvFrom(&rs[0], 1, &e0);
    ta.join();
    t1.join();
    try std.testing.expect(sock.nowNs() - t0 < 2 * std.time.ns_per_s);
    try std.testing.expectEqual(@as(?anyerror, error.Aborted), e0);
    try std.testing.expectEqual(@as(?anyerror, error.Aborted), e1);
    try std.testing.expectError(error.Aborted, rs[1].iface().check());
}

test "an abort in the middle of a frame never lets the peer finish that call" {
    var rs = try pair();
    defer for (&rs) |*r| r.deinit();
    // rank 0's 32 MiB frame, on sockets with small buffers, is part sent when it aborts; rank 1 must fail
    for ([_]std.posix.fd_t{ rs[0].fds[1], rs[1].fds[0] }) |fd| for ([_]u32{ std.posix.SO.SNDBUF, std.posix.SO.RCVBUF }) |o| {
        const size: c_int = 64 << 10;
        _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, o, &size, @sizeOf(c_int));
    };
    const n = 32 << 20;
    var e0: ?anyerror = null;
    var e1: ?anyerror = null;
    const ta = try std.Thread.spawn(.{}, Later.abort, .{ &rs[0], @as(u64, 50) });
    Later.gather(&rs[0], n, &e0);
    ta.join();
    try std.testing.expectEqual(@as(?anyerror, error.Aborted), e0);
    Later.gather(&rs[1], n, &e1);
    try named(e1.?);
}

test "a frame that cannot be sent fails the call, even when the peer's frame arrives" {
    var rs = try pair();
    defer for (&rs) |*r| r.deinit();
    _ = std.c.shutdown(rs[0].fds[1], std.posix.SHUT.WR); // rank 0 can still read but no longer send
    var e1: ?anyerror = null;
    const t1 = try std.Thread.spawn(.{}, Later.gather, .{ &rs[1], @as(usize, 64), &e1 });
    var e0: ?anyerror = null;
    Later.gather(&rs[0], 64, &e0);
    t1.join();
    try std.testing.expectEqual(@as(?anyerror, error.PeerFailed), e0);
    try std.testing.expectError(error.PeerFailed, rs[0].iface().check());
    try named(e1.?);
}

test "with three ranks, an abort ends a wait that does not involve the rank that aborted" {
    const fds = try tcp.loopback(3);
    var rs: [3]tcp.Rank = undefined;
    for (0..3) |i| rs[i] = tcp.Rank.init(gpa, @intCast(i), 3, &.{ fds[i][0], fds[i][1], fds[i][2] }, opts);
    defer for (&rs) |*r| r.deinit();
    // rank 0 waits on rank 1, which never sends; rank 2 aborts and rank 0 must hear it, not wait for its deadline
    var e0: ?anyerror = null;
    const ta = try std.Thread.spawn(.{}, Later.abort, .{ &rs[2], @as(u64, 50) });
    const t0 = sock.nowNs();
    Later.recvFrom(&rs[0], 1, &e0);
    ta.join();
    try std.testing.expect(sock.nowNs() - t0 < 2 * std.time.ns_per_s);
    try std.testing.expectEqual(@as(?anyerror, error.Aborted), e0);
}
