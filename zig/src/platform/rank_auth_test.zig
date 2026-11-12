//! Rank link authentication over loopback sockets: what a rank, a stranger and a recorded proof each get.
const std = @import("std");
const builtin = @import("builtin");
const auth = @import("rank_auth.zig");
const key = auth.key;
const c = std.c;
const posix = std.posix;
const Secret = auth.Secret;
const skip = builtin.os.tag == .windows;

/// A connected TCP pair on loopback: the connector's end, then the acceptor's.
pub fn pair() ![2]c.fd_t {
    const l = c.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (l < 0) return error.SocketFailed;
    defer _ = c.close(l);
    var addr: posix.sockaddr.in = .{ .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
    if (c.bind(l, @ptrCast(&addr), len) != 0 or c.listen(l, 4) != 0) return error.SocketFailed;
    if (c.getsockname(l, @ptrCast(&addr), &len) != 0) return error.SocketFailed;
    const a = c.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (a < 0) return error.SocketFailed;
    errdefer _ = c.close(a);
    if (c.connect(a, @ptrCast(&addr), len) != 0) return error.SocketFailed;
    const b = c.accept(l, null, null);
    if (b < 0) return error.SocketFailed;
    return .{ a, b };
}

pub fn close(fds: [2]c.fd_t) void {
    for (fds) |fd| _ = c.close(fd);
}

const Connector = struct {
    err: ?anyerror = null,
    session: auth.Key = undefined,

    /// A rank's connecting half; like a rank, it leaves when the proof fails.
    fn run(fd: c.fd_t, secret: Secret, purpose: auth.Purpose, rank: u32, world: u32, out: *Connector) void {
        out.session = auth.connect(fd, &secret, purpose, rank, world, 3000) catch |e| blk: {
            out.err = e;
            _ = c.shutdown(fd, posix.SHUT.WR);
            break :blk undefined;
        };
    }
};

const Met = struct { accepted: auth.Error!auth.Proven, connector: Connector };

/// Rank 0 with `mine` takes one connection from a rank with `theirs`.
fn meet(mine: Secret, theirs: Secret, allowed: []const auth.Purpose, purpose: auth.Purpose, rank: u32, world: u32) !Met {
    const fds = try pair();
    defer close(fds);
    var out: Met = .{ .accepted = undefined, .connector = .{} };
    const thread = try std.Thread.spawn(.{}, Connector.run, .{ fds[0], theirs, purpose, rank, world, &out.connector });
    out.accepted = auth.accept(fds[1], &mine, allowed, 3000);
    if (out.accepted) |_| {} else |_| _ = c.shutdown(fds[1], posix.SHUT.WR);
    thread.join();
    return out;
}

fn sendAll(fd: c.fd_t, bytes: []const u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = c.send(fd, bytes[at..].ptr, bytes.len - at, posix.MSG.NOSIGNAL);
        if (n <= 0) return error.Closed;
        at += @intCast(n);
    }
}

/// Reads up to `out.len` bytes until the peer closes; returns how many came.
fn readSome(fd: c.fd_t, out: []u8) usize {
    var at: usize = 0;
    while (at < out.len) {
        const n = c.recv(fd, out[at..].ptr, out.len - at, 0);
        if (n <= 0) break;
        at += @intCast(n);
    }
    return at;
}

fn nowMs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1000 + @as(u64, @intCast(ts.nsec)) / std.time.ns_per_ms;
}

const Acceptor = struct {
    result: auth.Error!auth.Proven = error.RankAuthTimeout,

    fn run(fd: c.fd_t, secret: Secret, ms: u64, out: *Acceptor) void {
        out.result = auth.accept(fd, &secret, &.{ .flashnext, .liveness }, ms);
        // an acceptor closes a failed connection; SHUT.WR, as macOS refuses RDWR once the peer's end has closed
        if (out.result) |_| {} else |_| _ = c.shutdown(fd, posix.SHUT.WR);
    }
};

test "rank auth: ranks with one key prove both ways, keep one session key, and rank 0 learns who came" {
    if (skip) return error.SkipZigTest;
    const k = try Secret.fromBytes("a shared rank secret\n");
    const m = try meet(k, k, &.{ .tp_control, .tp_plan }, .tp_plan, 3, 4);
    const proven = try m.accepted;
    try std.testing.expectEqual(@as(?anyerror, null), m.connector.err);
    try std.testing.expectEqual(auth.Purpose.tp_plan, proven.purpose);
    try std.testing.expectEqual(@as(u32, 3), proven.rank);
    try std.testing.expectEqual(@as(u32, 4), proven.world);
    try std.testing.expectEqualSlices(u8, &proven.session, &m.connector.session);
    const again = try meet(k, k, &.{.tp_plan}, .tp_plan, 3, 4);
    try std.testing.expect(!std.mem.eql(u8, &proven.session, &(try again.accepted).session)); // fresh nonces each time
}

test "rank auth: ranks with no key meet on one host" {
    if (skip) return error.SkipZigTest;
    const m = try meet(Secret.none, Secret.none, &.{.liveness}, .liveness, 1, 2);
    _ = try m.accepted;
    try std.testing.expectEqual(@as(?anyerror, null), m.connector.err);
}

test "rank auth: a rank with another key is refused, and so is a key at one end only" {
    if (skip) return error.SkipZigTest;
    const k = try Secret.fromBytes("a shared rank secret");
    const other = try Secret.fromBytes("b shared rank secret");
    const wrong = try meet(k, other, &.{.rocm}, .rocm, 1, 2);
    try std.testing.expectEqual(@as(?anyerror, error.RankAuthWrongKey), wrong.connector.err);
    try std.testing.expectError(error.RankAuthClosed, wrong.accepted); // the rank left at rank 0's proof
    const bare_peer = try meet(k, Secret.none, &.{.rocm}, .rocm, 1, 2);
    try std.testing.expectError(error.RankAuthPeerHasNoKey, bare_peer.accepted);
    try std.testing.expectEqual(@as(?anyerror, error.RankAuthPeerHasKey), bare_peer.connector.err);
    const bare_lead = try meet(Secret.none, k, &.{.rocm}, .rocm, 1, 2);
    try std.testing.expectError(error.RankAuthPeerHasKey, bare_lead.accepted);
    try std.testing.expectEqual(@as(?anyerror, error.RankAuthPeerHasNoKey), bare_lead.connector.err);
}

test "rank auth: a link, rank or world rank 0 does not take is refused before it answers" {
    if (skip) return error.SkipZigTest;
    const k = try Secret.fromBytes("a shared rank secret");
    for ([_]struct { auth.Purpose, u32, u32 }{ .{ .rocm, 1, 2 }, .{ .flashnext, 0, 2 }, .{ .flashnext, 2, 2 }, .{ .flashnext, 1, 1 }, .{ .flashnext, 1, auth.max_world + 1 } }) |claim| {
        const m = try meet(k, k, &.{.flashnext}, claim[0], claim[1], claim[2]);
        try std.testing.expectError(error.RankAuthBadClaim, m.accepted);
        try std.testing.expectEqual(@as(?anyerror, error.RankAuthClosed), m.connector.err);
    }
}

test "rank auth: a stranger without the key gets a challenge and nothing more" {
    if (skip) return error.SkipZigTest;
    const k = try Secret.fromBytes("a shared rank secret");
    const fds = try pair();
    defer close(fds);
    var acceptor: Acceptor = .{};
    const thread = try std.Thread.spawn(.{}, Acceptor.run, .{ fds[1], k, 3000, &acceptor });
    const nonce = try key.nonce();
    try sendAll(fds[0], &auth.hello(@backingInt(auth.Purpose.flashnext), 1, 2, true, &nonce));
    var ch: [auth.challenge_len]u8 = undefined;
    try std.testing.expectEqual(@as(usize, auth.challenge_len), readSome(fds[0], &ch));
    const guess: key.Proof = @splat(0x5a);
    try sendAll(fds[0], &guess);
    var more: [64]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 0), readSome(fds[0], &more)); // closed: no identity, no id, no frame
    thread.join();
    try std.testing.expectError(error.RankAuthWrongKey, acceptor.result);
}

test "rank auth: a recorded proof replayed, rank 0's proof sent back, and a proof for another rank are refused" {
    if (skip) return error.SkipZigTest;
    const k = try Secret.fromBytes("a shared rank secret");
    const nonce = try key.nonce();
    const hello = auth.hello(@backingInt(auth.Purpose.flashnext), 1, 2, true, &nonce);
    var recorded: key.Proof = undefined;
    const Case = enum { honest, replay, reflect, other_rank, other_link, other_end };
    for ([_]Case{ .honest, .replay, .reflect, .other_rank, .other_link, .other_end }) |case| {
        const fds = try pair();
        defer close(fds);
        var acceptor: Acceptor = .{};
        const thread = try std.Thread.spawn(.{}, Acceptor.run, .{ fds[1], k, 3000, &acceptor });
        try sendAll(fds[0], &hello);
        var ch: [auth.challenge_len]u8 = undefined;
        try std.testing.expectEqual(@as(usize, auth.challenge_len), readSome(fds[0], &ch));
        const theirs: *const key.Nonce = ch[12..44];
        const answer: key.Proof = switch (case) {
            .honest => auth.proof(&k.key, .connector, .flashnext, 1, 2, theirs, &nonce),
            .replay => recorded,
            .reflect => ch[44..76].*,
            .other_rank => auth.proof(&k.key, .connector, .flashnext, 2, 3, theirs, &nonce),
            .other_link => auth.proof(&k.key, .connector, .liveness, 1, 2, theirs, &nonce),
            .other_end => auth.proof(&k.key, .acceptor, .flashnext, 1, 2, theirs, &nonce),
        };
        if (case == .honest) recorded = answer;
        try sendAll(fds[0], &answer);
        thread.join();
        if (case == .honest) {
            _ = try acceptor.result;
        } else try std.testing.expectError(error.RankAuthWrongKey, acceptor.result);
    }
}

test "rank auth: other bytes, a short hello and a silent peer end inside the budget, with nothing sent" {
    if (skip) return error.SkipZigTest;
    const k = try Secret.fromBytes("a shared rank secret");
    const Case = enum { other_protocol, short, silent };
    for ([_]Case{ .other_protocol, .short, .silent }) |case| {
        const fds = try pair();
        defer close(fds);
        var acceptor: Acceptor = .{};
        const start = nowMs();
        const thread = try std.Thread.spawn(.{}, Acceptor.run, .{ fds[1], k, @as(u64, if (case == .silent) 200 else 3000), &acceptor });
        switch (case) {
            .other_protocol => {
                var old: [auth.hello_len]u8 = @splat(0x41);
                @memcpy(old[0..4], "TPB1"); // an older build's hello
                try sendAll(fds[0], &old);
            },
            .short => {
                try sendAll(fds[0], "TFRKAUTH\x01\x00");
                _ = c.shutdown(fds[0], posix.SHUT.WR);
            },
            .silent => {},
        }
        var got: [8]u8 = undefined;
        try std.testing.expectEqual(@as(usize, 0), readSome(fds[0], &got));
        thread.join();
        try std.testing.expectError(switch (case) {
            .other_protocol => error.RankAuthNotARank,
            .short => error.RankAuthClosed,
            .silent => error.RankAuthTimeout,
        }, acceptor.result);
        try std.testing.expect(nowMs() - start < 2000);
    }
}

test "rank auth: a connector tells an older build and a stranger from a rank, and sends them no proof" {
    if (skip) return error.SkipZigTest;
    const k = try Secret.fromBytes("a shared rank secret");
    const Peer = struct {
        /// Not a rank: reads the hello, answers with `bytes`, and reports what the connector sent after.
        fn run(fd: c.fd_t, bytes: []const u8, after: *usize) void {
            var hello: [auth.hello_len]u8 = undefined;
            _ = readSome(fd, &hello);
            sendAll(fd, bytes) catch {};
            var more: [64]u8 = undefined;
            after.* = readSome(fd, &more);
        }
    };
    var forged: [auth.challenge_len]u8 = @splat(0);
    @memcpy(forged[0..8], "TFRKCHAL");
    forged[8] = 1;
    const old: [auth.challenge_len]u8 = @splat(0x41);
    for ([_]struct { []const u8, anyerror }{ .{ &forged, error.RankAuthWrongKey }, .{ &old, error.RankAuthNotARank }, .{ "", error.RankAuthTimeout } }) |case| {
        const fds = try pair();
        defer close(fds);
        var after: usize = 99;
        const thread = try std.Thread.spawn(.{}, Peer.run, .{ fds[1], case[0], &after });
        const got = auth.connect(fds[0], &k, .flashnext, 1, 2, 300);
        _ = c.shutdown(fds[0], posix.SHUT.WR);
        thread.join();
        try std.testing.expectError(case[1], got);
        try std.testing.expectEqual(@as(usize, 0), after);
    }
}

test "rank auth: a proof binds the key, the end, the link, the rank, the world and both nonces" {
    const k = (try Secret.fromBytes("a shared rank secret")).key;
    const other = (try Secret.fromBytes("b shared rank secret")).key;
    const a = try key.nonce();
    const b = try key.nonce();
    const p = auth.proof(&k, .connector, .flashnext, 1, 2, &a, &b);
    try std.testing.expect(key.same(&p, &auth.proof(&k, .connector, .flashnext, 1, 2, &a, &b)));
    try std.testing.expect(!key.same(&p, &auth.proof(&other, .connector, .flashnext, 1, 2, &a, &b)));
    try std.testing.expect(!key.same(&p, &auth.proof(&Secret.none.key, .connector, .flashnext, 1, 2, &a, &b)));
    try std.testing.expect(!key.same(&p, &auth.proof(&k, .acceptor, .flashnext, 1, 2, &a, &b)));
    try std.testing.expect(!key.same(&p, &auth.proof(&k, .connector, .rocm, 1, 2, &a, &b)));
    try std.testing.expect(!key.same(&p, &auth.proof(&k, .connector, .flashnext, 2, 2, &a, &b)));
    try std.testing.expect(!key.same(&p, &auth.proof(&k, .connector, .flashnext, 1, 3, &a, &b)));
    try std.testing.expect(!key.same(&p, &auth.proof(&k, .connector, .flashnext, 1, 2, &b, &a)));
    try std.testing.expect(!key.same(&p, &auth.proof(&k, .connector, .flashnext, 1, 2, &a, &a)));
}

test "rank auth: the key comes from the file the setting names, and no setting means no key" {
    if (skip) return error.SkipZigTest;
    const Env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    _ = Env.unsetenv(auth.env_name);
    try std.testing.expect(!(try Secret.fromEnv()).keyed);
    for ([_][*:0]const u8{ "", " \t" }) |blank| {
        _ = Env.setenv(auth.env_name, blank, 1);
        try std.testing.expect(!(try Secret.fromEnv()).keyed);
    }
    _ = Env.setenv(auth.env_name, "/nonexistent/rank\nsecret", 1); // a control character: refused before any file is opened
    try std.testing.expectError(error.SecretUnreadable, Secret.fromEnv());
    _ = Env.setenv(auth.env_name, "/nonexistent/rank-secret", 1);
    defer _ = Env.unsetenv(auth.env_name);
    try std.testing.expectError(error.SecretUnreadable, Secret.fromEnv());
    try std.testing.expect(auth.explain(error.SecretUnreadable) != null);
    try std.testing.expect(auth.explain(error.OutOfMemory) == null);
}
