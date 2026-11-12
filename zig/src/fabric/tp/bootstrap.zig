//! Rank discovery over TCP (a star on rank 0): small host blobs all-gathered, settings agreed, sockets kept for fate.
const std = @import("std");
const log = @import("log.zig");
const sock = @import("sock.zig");
const Config = @import("config.zig").Config;
const auth = @import("sys").rank_auth;

pub const max_ranks = 64;
/// Largest blob a rank contributes to one `allGather`.
pub const max_blob = 64 * 1024;

const magic: u32 = 0x31425054; // "TPB1"
const version: u32 = 2;

pub const Error = error{ Protocol, Mismatch, TooLarge, Unproven } || sock.Error;

/// `role`: 0 the control connection (bootstrap, then the fate channel), 1 the plan link.
const Hello = extern struct { magic: u32, version: u32, world: u32, rank: u32, role: u32 };

pub const Bootstrap = struct {
    rank: u32,
    world: u32,
    /// Rank 0: the socket of rank i at index i (index 0 unused); other ranks: rank 0's socket at index 0.
    fds: [max_ranks]sock.Fd = @splat(-1),
    /// The plan link's sockets, laid out as `fds`.
    plan_fds: [max_ranks]sock.Fd = @splat(-1),
    /// Each call's limit for the peers' bytes (the connect timeout).
    timeout_ns: u64,
    /// The group's key: every connection proves it before its hello (sys.rank_auth).
    secret: auth.Secret = .none,

    fn deadline(b: *const Bootstrap) u64 {
        return sock.nowNs() + b.timeout_ns;
    }

    /// Rank 0 listens on `link_port` and accepts every other rank; the others connect to `master` (connect timeout).
    pub fn open(cfg: Config) Error!Bootstrap {
        if (cfg.world > max_ranks or cfg.rank >= cfg.world) return error.Protocol;
        var b: Bootstrap = .{ .rank = cfg.rank, .world = cfg.world, .timeout_ns = cfg.connect_timeout_ns, .secret = cfg.secret };
        errdefer b.close();
        if (cfg.world == 1) return b;
        if (cfg.rank == 0) {
            const l = try sock.listen(cfg.link_port);
            defer sock.close(l);
            try b.acceptAll(l);
        } else {
            const until = b.deadline();
            for ([_]u32{ 0, 1 }) |role| {
                const fd = try sock.connect(cfg.master, cfg.link_port, until);
                if (role == 0) b.fds[0] = fd else b.plan_fds[0] = fd;
                _ = auth.connect(fd, &b.secret, purposeOf(role), cfg.rank, cfg.world, msUntil(until)) catch |e| {
                    log.warn("tp bootstrap: rank {d} and rank 0 did not prove the group's key: {s}", .{ cfg.rank, auth.explain(e) orelse @errorName(e) });
                    return error.Unproven;
                };
                const h: Hello = .{ .magic = magic, .version = version, .world = cfg.world, .rank = cfg.rank, .role = role };
                try sock.sendBefore(fd, std.mem.asBytes(&h), until);
            }
        }
        return b;
    }

    /// Rank 0 over an already listening socket (tests: an ephemeral port), with no key.
    pub fn acceptOn(l: sock.Fd, world: u32, timeout_ns: u64) Error!Bootstrap {
        return acceptWith(l, world, timeout_ns, .none);
    }

    /// `acceptOn` for a group whose ranks hold `secret`.
    pub fn acceptWith(l: sock.Fd, world: u32, timeout_ns: u64, secret: auth.Secret) Error!Bootstrap {
        if (world == 0 or world > max_ranks) return error.Protocol;
        var b: Bootstrap = .{ .rank = 0, .world = world, .timeout_ns = timeout_ns, .secret = secret };
        errdefer b.close();
        try b.acceptAll(l);
        return b;
    }

    fn acceptAll(b: *Bootstrap, l: sock.Fd) Error!void {
        // every other rank's control and plan connections, in any order
        var joined: u32 = 0;
        const until = b.deadline();
        while (joined < 2 * (b.world - 1)) {
            const fd = try sock.accept(l, until);
            // a connection that proves no key is closed and forgotten: it cannot take a rank's place or end the group
            const proven = auth.accept(fd, &b.secret, &.{ .tp_control, .tp_plan }, msUntil(until)) catch |e| {
                var at: [64]u8 = undefined;
                log.warn("tp bootstrap: a connection from {s} proved no key and is ignored: {s}", .{ auth.peerText(fd, &at), auth.explain(e) orelse @errorName(e) });
                sock.close(fd);
                continue;
            };
            var h: Hello = undefined;
            sock.recvExact(fd, std.mem.asBytes(&h), @min(until, sock.nowNs() + auth.budget_ms * std.time.ns_per_ms)) catch {
                sock.close(fd);
                continue;
            };
            if (h.magic != magic or h.version != version) {
                log.warn("tp bootstrap: a connection that is not a rank (magic {x}); ignored", .{h.magic});
                sock.close(fd);
                continue;
            }
            const table = if (h.role == 0) &b.fds else &b.plan_fds;
            const as_proven = h.role <= 1 and proven.purpose == purposeOf(h.role) and proven.rank == h.rank and proven.world == h.world;
            if (!as_proven or h.world != b.world or h.rank == 0 or h.rank >= b.world or table[h.rank] >= 0) {
                log.warn("tp bootstrap: rank {d} of {d} joined a world of {d}", .{ h.rank, h.world, b.world });
                sock.close(fd);
                return error.Mismatch;
            }
            table[h.rank] = fd;
            joined += 1;
        }
    }

    /// Takes the control sockets (the fate channel keeps them); `close` then leaves them open.
    pub fn release(b: *Bootstrap) [max_ranks]sock.Fd {
        const out = b.fds;
        b.fds = @splat(-1);
        return out;
    }

    /// Takes the plan link's sockets.
    pub fn releasePlan(b: *Bootstrap) [max_ranks]sock.Fd {
        const out = b.plan_fds;
        b.plan_fds = @splat(-1);
        return out;
    }

    pub fn close(b: *Bootstrap) void {
        for ([_]*[max_ranks]sock.Fd{ &b.fds, &b.plan_fds }) |table| for (table) |*fd| if (fd.* >= 0) {
            sock.close(fd.*);
            fd.* = -1;
        };
    }

    /// Every rank's `mine` in rank order into `out`, returned as slices of `out`; every rank calls it equally often.
    pub fn allGather(b: *Bootstrap, mine: []const u8, out: []u8, blobs: [][]const u8) Error![]const []const u8 {
        if (mine.len > max_blob) return error.TooLarge;
        if (blobs.len < b.world or mine.len > max_blob) return error.TooLarge;
        if (b.world == 1) {
            if (out.len < mine.len) return error.TooLarge;
            @memcpy(out[0..mine.len], mine);
            blobs[0] = out[0..mine.len];
            return blobs[0..1];
        }
        if (b.rank != 0) {
            try sendFrame(b.fds[0], mine, b.deadline());
            return b.readAll(b.fds[0], out, blobs);
        }
        // rank 0: everyone's blob, then the whole table to everyone
        var used: usize = 0;
        for (0..b.world) |i| {
            const len: u32 = if (i == 0) @intCast(mine.len) else blk: {
                var l: u32 = undefined;
                try sock.recvExact(b.fds[i], std.mem.asBytes(&l), b.deadline());
                break :blk l;
            };
            if (len > max_blob or used + len > out.len) return error.TooLarge;
            if (i == 0) @memcpy(out[used..][0..len], mine) else try sock.recvExact(b.fds[i], out[used..][0..len], b.deadline());
            blobs[i] = out[used..][0..len];
            used += len;
        }
        for (1..b.world) |i| {
            for (blobs[0..b.world]) |blob| try sendFrame(b.fds[i], blob, b.deadline());
        }
        return blobs[0..b.world];
    }

    fn readAll(b: *Bootstrap, fd: sock.Fd, out: []u8, blobs: [][]const u8) Error![]const []const u8 {
        var used: usize = 0;
        for (0..b.world) |i| {
            var l: u32 = undefined;
            try sock.recvExact(fd, std.mem.asBytes(&l), b.deadline());
            if (l > max_blob or used + l > out.len) return error.TooLarge;
            try sock.recvExact(fd, out[used..][0..l], b.deadline());
            blobs[i] = out[used..][0..l];
            used += l;
        }
        return blobs[0..b.world];
    }

    /// Rank 0's `bytes` on every rank (into `out` on the others; `bytes` is ignored there).
    pub fn broadcast(b: *Bootstrap, bytes: []const u8, out: []u8) Error!void {
        var blobs: [max_ranks][]const u8 = undefined;
        var scratch: [4096]u8 = undefined;
        if (out.len * b.world > scratch.len) return error.TooLarge;
        const all = try b.allGather(if (b.rank == 0) bytes else &.{}, &scratch, &blobs);
        if (all[0].len != out.len) return error.Protocol;
        @memcpy(out, all[0]);
    }

    /// Every rank's `bytes` must be equal (settings that change bytes on the wire); `what` names them in the log.
    pub fn agree(b: *Bootstrap, bytes: []const u8, what: []const u8) Error!void {
        var blobs: [max_ranks][]const u8 = undefined;
        var scratch: [8192]u8 = undefined;
        const all = try b.allGather(bytes, &scratch, &blobs);
        for (all, 0..) |blob, i| if (!std.mem.eql(u8, blob, all[0])) {
            log.warn("tp bootstrap: rank {d}'s {s} differs from rank 0's; both ranks refuse to start", .{ i, what });
            return error.Mismatch;
        };
    }
};

/// The key proof's link for a hello's `role`.
fn purposeOf(role: u32) auth.Purpose {
    return if (role == 0) .tp_control else .tp_plan;
}

/// Whole milliseconds from now to `deadline` (monotonic nanoseconds), none once it has passed.
fn msUntil(deadline: u64) u64 {
    return (deadline -| sock.nowNs()) / std.time.ns_per_ms;
}

fn sendFrame(fd: sock.Fd, bytes: []const u8, deadline: u64) sock.Error!void {
    const l: u32 = @intCast(bytes.len);
    try sock.sendBefore(fd, std.mem.asBytes(&l), deadline);
    try sock.sendBefore(fd, bytes, deadline);
}

/// An ephemeral listening port for tests.
pub fn testListener() !struct { fd: sock.Fd, port: u16 } {
    const l = try sock.listen(0);
    var addr: std.posix.sockaddr.in = undefined;
    var len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    _ = std.c.getsockname(l, @ptrCast(&addr), &len);
    return .{ .fd = l, .port = std.mem.bigToNative(u16, addr.port) };
}

fn testRank(cfg: Config, out: *[3][32]u8, err: *?anyerror) void {
    var b = Bootstrap.open(cfg) catch |e| {
        err.* = e;
        return;
    };
    defer b.close();
    var id: [32]u8 = undefined;
    b.broadcast(&@as([32]u8, @splat(0xAB)), &id) catch |e| {
        err.* = e;
        return;
    };
    var blobs: [max_ranks][]const u8 = undefined;
    var buf: [256]u8 = undefined;
    const mine: [3]u8 = @splat(@intCast(cfg.rank));
    const all = b.allGather(mine[0 .. cfg.rank + 1], &buf, &blobs) catch |e| {
        err.* = e;
        return;
    };
    for (all, 0..) |blob, i| @memcpy(out[i][0..blob.len], blob);
    b.agree("same", "settings") catch |e| {
        err.* = e;
    };
}

test "three ranks gather in rank order and agree" {
    const l = try testListener();
    defer sock.close(l.fd);
    var outs: [3][3][32]u8 = undefined;
    var errs: [3]?anyerror = @splat(null);
    var threads: [2]std.Thread = undefined;
    for (0..2) |i| threads[i] = try std.Thread.spawn(.{}, testRank, .{ Config{ .world = 3, .rank = @intCast(i + 1), .link_port = l.port, .connect_timeout_ns = 5 * std.time.ns_per_s }, &outs[i + 1], &errs[i + 1] });
    var b = try Bootstrap.acceptOn(l.fd, 3, 5 * std.time.ns_per_s);
    defer b.close();
    var id: [32]u8 = undefined;
    try b.broadcast(&@as([32]u8, @splat(0xAB)), &id);
    var blobs: [max_ranks][]const u8 = undefined;
    var buf: [256]u8 = undefined;
    const all = try b.allGather(&.{0}, &buf, &blobs);
    try b.agree("same", "settings");
    for (threads) |t| t.join();
    for (errs) |e| try std.testing.expect(e == null);
    try std.testing.expectEqual(@as(usize, 3), all.len);
    try std.testing.expectEqualSlices(u8, &.{ 2, 2, 2 }, all[2]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1 }, outs[2][1][0..2]);
}

test "settings that differ stop every rank" {
    const l = try testListener();
    defer sock.close(l.fd);
    const T = struct {
        fn go(port: u16, err: *?anyerror) void {
            var b = Bootstrap.open(.{ .world = 2, .rank = 1, .link_port = port, .connect_timeout_ns = 5 * std.time.ns_per_s }) catch |e| {
                err.* = e;
                return;
            };
            defer b.close();
            b.agree("knobs-b", "settings") catch |e| {
                err.* = e;
            };
        }
    };
    var err: ?anyerror = null;
    const t = try std.Thread.spawn(.{}, T.go, .{ l.port, &err });
    var b = try Bootstrap.acceptOn(l.fd, 2, 5 * std.time.ns_per_s);
    defer b.close();
    try std.testing.expectError(error.Mismatch, b.agree("knobs-a", "settings"));
    t.join();
    try std.testing.expectEqual(@as(?anyerror, error.Mismatch), err);
}

/// Rank 1 agrees on `theirs.digest()` against rank 0's `ours.digest()`; returns both ranks' results.
fn agreeDigests(ours: Config, theirs: Config) !struct { ?anyerror, ?anyerror } {
    const l = try testListener();
    defer sock.close(l.fd);
    const T = struct {
        fn go(port: u16, cfg: Config, err: *?anyerror) void {
            var c = cfg;
            c.world = 2;
            c.rank = 1;
            c.link_port = port;
            c.connect_timeout_ns = 5 * std.time.ns_per_s;
            var b = Bootstrap.open(c) catch |e| {
                err.* = e;
                return;
            };
            defer b.close();
            b.agree(&c.digest(), "settings") catch |e| {
                err.* = e;
            };
        }
    };
    var err1: ?anyerror = null;
    const t = try std.Thread.spawn(.{}, T.go, .{ l.port, theirs, &err1 });
    var b = try Bootstrap.acceptOn(l.fd, 2, 5 * std.time.ns_per_s);
    defer b.close();
    var err0: ?anyerror = null;
    b.agree(&ours.digest(), "settings") catch |e| {
        err0 = e;
    };
    t.join();
    return .{ err0, err1 };
}

test "a graph cache switched off on one rank stops both at start; 0 and OFF on the two ranks start" {
    const on: Config = .{ .graph_cache = .{} };
    const off: Config = .{ .graph_cache = .{}, .graph_cache_env = "off" };
    const r = try agreeDigests(on, off);
    try std.testing.expectEqual(@as(?anyerror, error.Mismatch), r[0]);
    try std.testing.expectEqual(@as(?anyerror, error.Mismatch), r[1]);
    const none = try agreeDigests(.{}, .{ .graph_cache_env = "off" }); // a family without the cache is not refused
    try std.testing.expectEqual(@as(?anyerror, null), none[0]);
    try std.testing.expectEqual(@as(?anyerror, null), none[1]);
    const same = try agreeDigests(.{ .graph_cache = .{}, .graph_cache_env = "0" }, .{ .graph_cache = .{}, .graph_cache_env = "OFF" });
    try std.testing.expectEqual(@as(?anyerror, null), same[0]);
    try std.testing.expectEqual(@as(?anyerror, null), same[1]);
    const family = try agreeDigests(.{ .graph_cache = .{ .max = 8 } }, .{ .graph_cache = .{ .max = 16 } });
    try std.testing.expectEqual(@as(?anyerror, error.Mismatch), family[0]);
}

/// A connection that is not a rank: it sends `bytes` and stops sending.
fn testStranger(port: u16, bytes: []const u8) !sock.Fd {
    const fd = try sock.connect("localhost", port, sock.nowNs() + 5 * std.time.ns_per_s);
    errdefer sock.close(fd);
    try sock.sendAll(fd, bytes);
    _ = std.c.shutdown(fd, std.posix.SHUT.WR);
    return fd;
}

test "security: a connection that proves no key cannot end the group or take a rank's place" {
    const key = try auth.Secret.fromBytes("the group's rank secret for the tests");
    const l = try testListener();
    defer sock.close(l.fd);
    // an older build's hello for another world: it ended the group for every rank
    const old: Hello = .{ .magic = magic, .version = version, .world = 9, .rank = 1, .role = 0 };
    const first = try testStranger(l.port, std.mem.asBytes(&old));
    defer sock.close(first);
    // rank 1's place claimed with a made-up proof
    const nonce: auth.key.Nonce = @splat(7);
    const guess: auth.key.Proof = @splat(9);
    const second = try testStranger(l.port, &(auth.hello(@backingInt(auth.Purpose.tp_control), 1, 2, true, &nonce) ++ guess));
    defer sock.close(second);
    var out: [3][32]u8 = undefined;
    var err: ?anyerror = null;
    const cfg: Config = .{ .world = 2, .rank = 1, .link_port = l.port, .connect_timeout_ns = 5 * std.time.ns_per_s, .secret = key };
    const t = try std.Thread.spawn(.{}, testRank, .{ cfg, &out, &err });
    var b = try Bootstrap.acceptWith(l.fd, 2, 5 * std.time.ns_per_s, key);
    defer b.close();
    var id: [32]u8 = undefined;
    try b.broadcast(&@as([32]u8, @splat(0xAB)), &id);
    var blobs: [max_ranks][]const u8 = undefined;
    var buf: [256]u8 = undefined;
    const all = try b.allGather(&.{0}, &buf, &blobs);
    try b.agree("same", "settings");
    t.join();
    try std.testing.expectEqual(@as(?anyerror, null), err);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1 }, all[1]);
    // the first got nothing; the second got rank 0's challenge and then nothing: no blob reached either
    const soon = sock.nowNs() + std.time.ns_per_s;
    var got: [auth.challenge_len]u8 = undefined;
    try std.testing.expectError(error.Closed, sock.recvExact(first, got[0..1], soon));
    try sock.recvExact(second, &got, soon);
    try std.testing.expectError(error.Closed, sock.recvExact(second, got[0..1], soon));
}

test "security: a rank with another key or with none joins no keyed group" {
    const key = try auth.Secret.fromBytes("the group's rank secret for the tests");
    const other = try auth.Secret.fromBytes("another group's secret for the tests");
    for ([_]auth.Secret{ other, auth.Secret.none }) |theirs| {
        const l = try testListener();
        defer sock.close(l.fd);
        var out: [3][32]u8 = undefined;
        var err: ?anyerror = null;
        const cfg: Config = .{ .world = 2, .rank = 1, .link_port = l.port, .connect_timeout_ns = 5 * std.time.ns_per_s, .secret = theirs };
        const t = try std.Thread.spawn(.{}, testRank, .{ cfg, &out, &err });
        try std.testing.expectError(error.Timeout, Bootstrap.acceptWith(l.fd, 2, std.time.ns_per_s, key));
        t.join();
        try std.testing.expectEqual(@as(?anyerror, error.Unproven), err);
    }
}

test "Failure boundaries: acceptOn validates the world before touching a descriptor" {
    try std.testing.expectError(error.Protocol, Bootstrap.acceptOn(-1, max_ranks + 1, 0));
}
