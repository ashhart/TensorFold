//! Bounded Linux TCP control channel. Tensor bytes travel over NCCL; this channel owns command order and peer EOF.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const p = std.posix;
const net = std.Io.net;
const auth = platform.rank_auth;
const Seal = auth.seal.Seal;

pub const max_payload = 1 << 20;
pub const Kind = enum(u32) { hello = 1, ready, create, forward, round, drop_seq, save, restore, drop_snap, stop, ack, verify, keep, media, verify_many };
pub const Header = struct {
    kind: Kind,
    sequence: u64 = 0,
    object: u64 = 0,
    arg: u64 = 0,
    bytes: u32 = 0,
    flags: u32 = 0,

    pub fn encode(h: Header) ![40]u8 {
        if (h.bytes > max_payload) return error.BadRankFrame;
        var out: [40]u8 = @splat(0);
        @memcpy(out[0..4], "TFR1");
        std.mem.writeInt(u32, out[4..8], @backingInt(h.kind), .little);
        std.mem.writeInt(u64, out[8..16], h.sequence, .little);
        std.mem.writeInt(u64, out[16..24], h.object, .little);
        std.mem.writeInt(u64, out[24..32], h.arg, .little);
        std.mem.writeInt(u32, out[32..36], h.bytes, .little);
        std.mem.writeInt(u32, out[36..40], h.flags, .little);
        return out;
    }

    pub fn decode(raw: *const [40]u8) !Header {
        if (!std.mem.eql(u8, raw[0..4], "TFR1")) return error.BadRankFrame;
        const h: Header = .{
            .kind = std.enums.fromInt(Kind, std.mem.readInt(u32, raw[4..8], .little)) orelse return error.BadRankFrame,
            .sequence = std.mem.readInt(u64, raw[8..16], .little),
            .object = std.mem.readInt(u64, raw[16..24], .little),
            .arg = std.mem.readInt(u64, raw[24..32], .little),
            .bytes = std.mem.readInt(u32, raw[32..36], .little),
            .flags = std.mem.readInt(u32, raw[36..40], .little),
        };
        if (h.bytes > max_payload) return error.BadRankFrame;
        return h;
    }
};

pub fn now() u64 {
    var ts: platform.c.timespec = undefined;
    if (platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts) != 0 or ts.sec < 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_ms * 1000 + @as(u64, @intCast(ts.nsec));
}

pub fn deadline(ms: u64) u64 {
    return now() +| (ms *| std.time.ns_per_ms);
}

pub fn pause() void {
    const ts: platform.c.timespec = .{ .sec = 0, .nsec = 50_000 };
    _ = platform.c.nanosleep(&ts, null);
}

fn wait(fd: p.socket_t, events: i16, until: ?u64) !void {
    while (true) {
        const current = now();
        if (until) |end| if (current >= end) return error.RankTimeout;
        const ms: i32 = if (until) |end| @intCast(@min(1000, (end - current + std.time.ns_per_ms - 1) / std.time.ns_per_ms)) else 1000;
        var fds = [_]p.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
        const count = try p.poll(&fds, ms);
        if (count == 0) continue;
        if (fds[0].revents & events != 0) return;
        if (fds[0].revents & (p.POLL.ERR | p.POLL.HUP | p.POLL.NVAL) != 0) return error.RankDisconnected;
    }
}

pub const Link = struct {
    fd: p.socket_t,
    /// The key proof's seal: every byte crosses in its records. Null with no key, and on a test's unproven socket pair.
    seal: ?*Seal = null,

    pub fn close(l: Link) void {
        _ = std.c.shutdown(l.fd, p.SHUT.RDWR);
        _ = std.c.close(l.fd);
        if (l.seal) |seal| std.heap.page_allocator.destroy(seal);
    }

    /// Exactly two rank processes, each proving `secret` first. Retry only the connect, never an accepted command.
    pub fn open(io: std.Io, host: []const u8, port: u16, rank: usize, timeout_ms: u64, secret: *const auth.Secret) !Link {
        if (builtin.os.tag != .linux) return error.RankPlatform;
        if (rank > 1 or port == 0) return error.BadRank;
        const address = net.IpAddress.parse(host, port) catch try net.IpAddress.resolve(io, host, port);
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&address, &storage);
        const family: u32 = if (address == .ip4) p.AF.INET else p.AF.INET6;
        const until = deadline(timeout_ms);
        if (rank == 0) {
            const fd = try socket(family);
            defer _ = std.c.close(fd);
            const one: c_int = 1;
            try p.setsockopt(fd, p.SOL.SOCKET, p.SO.REUSEADDR, std.mem.asBytes(&one));
            if (std.c.bind(fd, &storage.any, len) != 0) return error.RankBind;
            if (std.c.listen(fd, 1) != 0) return error.RankListen;
            while (true) {
                try wait(fd, p.POLL.IN, until);
                const peer = std.c.accept4(fd, null, null, p.SOCK.NONBLOCK | p.SOCK.CLOEXEC);
                switch (p.errno(peer)) {
                    .SUCCESS => {
                        if (proven(peer, secret, until)) |session| return sealed(peer, &session, false, secret.keyed);
                        _ = std.c.close(peer);
                    },
                    .INTR, .AGAIN => continue,
                    else => return error.RankAccept,
                }
            }
        }
        while (now() < until) {
            const fd = try socket(family);
            const rc = std.c.connect(fd, &storage.any, len);
            if (rc == 0) return prove(fd, secret, until);
            const e = p.errno(rc);
            if (e == .INPROGRESS or e == .INTR) {
                wait(fd, p.POLL.OUT, until) catch |err| {
                    _ = std.c.close(fd);
                    if (err == error.RankTimeout) return err;
                    pause();
                    continue;
                };
                var status: c_int = 0;
                var size: p.socklen_t = @sizeOf(c_int);
                if (std.c.getsockopt(fd, p.SOL.SOCKET, p.SO.ERROR, &status, &size) == 0 and status == 0) return prove(fd, secret, until);
            }
            _ = std.c.close(fd);
            pause();
        }
        return error.RankTimeout;
    }

    /// `bytes` to the peer, in the seal's records when the link has one.
    pub fn send(l: Link, bytes: []const u8, until: u64) !void {
        const seal = l.seal orelse return l.sendRaw(bytes, until);
        var wire: [auth.seal.max_record]u8 = undefined;
        var at: usize = 0;
        while (at < bytes.len) {
            const n = @min(bytes.len - at, auth.seal.max_plain);
            try l.sendRaw(wire[0..seal.seal(bytes[at..][0..n], &wire)], until);
            at += n;
        }
    }

    /// Fills `bytes` from the peer's records; with no `until` it waits for one to begin, and then not for long.
    pub fn receive(l: Link, bytes: []u8, until: ?u64) !void {
        const seal = l.seal orelse return l.receiveRaw(bytes, until);
        var at = seal.take(bytes);
        while (at < bytes.len) {
            var head: [auth.seal.head_len]u8 = undefined;
            try l.receiveRaw(head[0..1], until);
            const rest = until orelse deadline(30_000);
            try l.receiveRaw(head[1..], rest);
            var body: [auth.seal.max_plain + auth.seal.tag_len]u8 = undefined;
            const n = Seal.bodyLen(&head) catch return unsealed();
            try l.receiveRaw(body[0..n], rest);
            seal.open(&head, body[0..n]) catch return unsealed();
            at += seal.take(bytes[at..]);
        }
    }

    fn sendRaw(l: Link, bytes: []const u8, until: u64) !void {
        if (builtin.os.tag == .windows) return error.RankPlatform; // no MSG.NOSIGNAL or DONTWAIT in Winsock
        var at: usize = 0;
        while (at < bytes.len) {
            if (now() >= until) return error.RankTimeout;
            const rc = std.c.send(l.fd, bytes.ptr + at, bytes.len - at, p.MSG.NOSIGNAL);
            switch (p.errno(rc)) {
                .SUCCESS => {
                    if (rc == 0) return error.RankDisconnected;
                    at += @intCast(rc);
                },
                .INTR => continue,
                .AGAIN => try wait(l.fd, p.POLL.OUT, until),
                else => return error.RankDisconnected,
            }
        }
    }

    fn receiveRaw(l: Link, bytes: []u8, until: ?u64) !void {
        if (builtin.os.tag == .windows) return error.RankPlatform;
        var at: usize = 0;
        while (at < bytes.len) {
            if (until) |end| if (now() >= end) return error.RankTimeout;
            const rc = std.c.recv(l.fd, bytes.ptr + at, bytes.len - at, 0);
            switch (p.errno(rc)) {
                .SUCCESS => {
                    if (rc == 0) return error.RankDisconnected;
                    at += @intCast(rc);
                },
                .INTR => continue,
                .AGAIN => try wait(l.fd, p.POLL.IN, until),
                else => return error.RankDisconnected,
            }
        }
    }

    pub fn sendHeader(l: Link, h: Header, until: u64) !void {
        const raw = try h.encode();
        try l.send(&raw, until);
    }

    pub fn receiveHeader(l: Link, until: ?u64) !Header {
        var raw: [40]u8 = undefined;
        // An idle follower can wait indefinitely; a partial frame cannot.
        try l.receive(raw[0..1], until);
        try l.receive(raw[1..], until orelse deadline(30_000));
        return Header.decode(&raw);
    }

    /// A non-consuming EOF check while a GPU collective progresses. A future control frame stays queued.
    pub fn checkPeer(l: Link) !void {
        if (builtin.os.tag == .windows) return error.RankPlatform;
        var byte: u8 = 0;
        const rc = std.c.recv(l.fd, &byte, 1, p.MSG.PEEK | p.MSG.DONTWAIT);
        if (rc == 0) return error.RankDisconnected;
        if (rc < 0) switch (p.errno(rc)) {
            .AGAIN, .INTR => {},
            else => return error.RankDisconnected,
        };
    }
};

/// Whole milliseconds from now to `until`.
fn msLeft(until: u64) u64 {
    return (until -| now()) / std.time.ns_per_ms;
}

/// A record that does not open ends the link: something between the ranks changed it, or sent it.
fn unsealed() error{RankDisconnected} {
    std.log.warn("flashnext: a frame between the ranks did not open under the group's key, so the link is closed", .{});
    return error.RankDisconnected;
}

/// A proven connection, and its seal when the proof used a key: kept off the stack, as a link is copied by value.
fn sealed(fd: p.socket_t, session: *const auth.Key, connector: bool, keyed: bool) !Link {
    if (!keyed) return .{ .fd = fd };
    errdefer _ = std.c.close(fd);
    const seal = try std.heap.page_allocator.create(Seal);
    seal.* = .init(session, connector);
    return .{ .fd = fd, .seal = seal };
}

/// Rank 0: the session of a connection that proved the key as rank 1 of 2. One that did not is the caller's to close.
fn proven(peer: p.socket_t, secret: *const auth.Secret, until: u64) ?auth.Key {
    noDelay(peer) catch return null;
    const who = auth.accept(peer, secret, &.{.flashnext}, msLeft(until)) catch |e| {
        var at: [64]u8 = undefined;
        std.log.warn("flashnext: a connection from {s} proved no key and is ignored: {s}", .{ auth.peerText(peer, &at), auth.explain(e) orelse @errorName(e) });
        return null;
    };
    return if (who.rank == 1 and who.world == 2) who.session else null;
}

/// Rank 1: proves the key to rank 0 over the connected `fd`, which closes when the proof fails.
fn prove(fd: p.socket_t, secret: *const auth.Secret, until: u64) !Link {
    const session = auth.connect(fd, secret, .flashnext, 1, 2, msLeft(until)) catch |e| {
        std.log.warn("flashnext: rank 1 and rank 0 did not prove the group's key: {s}", .{auth.explain(e) orelse @errorName(e)});
        _ = std.c.close(fd);
        return error.RankUnproven;
    };
    return sealed(fd, &session, true, secret.keyed);
}

fn socket(family: u32) !p.socket_t {
    const fd = std.c.socket(family, p.SOCK.STREAM | p.SOCK.NONBLOCK | p.SOCK.CLOEXEC, 0);
    if (p.errno(fd) != .SUCCESS) return error.RankSocket;
    errdefer _ = std.c.close(fd);
    try noDelay(fd);
    return fd;
}

fn noDelay(fd: p.socket_t) !void {
    const one: c_int = 1;
    try p.setsockopt(fd, p.IPPROTO.TCP, p.TCP.NODELAY, std.mem.asBytes(&one));
}

test "rank frames are versioned, bounded and exact" {
    const h: Header = .{ .kind = .forward, .sequence = 700, .object = 19, .arg = 42, .bytes = max_payload, .flags = 3 };
    var raw = try h.encode();
    try std.testing.expectEqualDeep(h, try Header.decode(&raw));
    raw[0] = 0;
    try std.testing.expectError(error.BadRankFrame, Header.decode(&raw));
    raw = try h.encode();
    @memset(raw[4..8], 0xff);
    try std.testing.expectError(error.BadRankFrame, Header.decode(&raw));
    raw = try h.encode();
    @memset(raw[32..36], 0xff);
    try std.testing.expectError(error.BadRankFrame, Header.decode(&raw));
    try std.testing.expectError(error.BadRankFrame, (Header{ .kind = .forward, .bytes = max_payload + 1 }).encode());
}

/// A port nothing listens on yet: bound once at 0 and released.
fn testPort() !u16 {
    const fd = try socket(p.AF.INET);
    defer _ = std.c.close(fd);
    var addr: p.sockaddr.in = .{ .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    var len: p.socklen_t = @sizeOf(p.sockaddr.in);
    if (std.c.bind(fd, @ptrCast(&addr), len) != 0 or std.c.getsockname(fd, @ptrCast(&addr), &len) != 0) return error.RankBind;
    return std.mem.bigToNative(u16, addr.port);
}

const TestRank = struct {
    err: ?anyerror = null,
    got: ?Header = null,

    /// One rank of the pair: rank 0 sends its first frame once its peer is in, rank 1 reads it.
    fn run(port: u16, rank: usize, secret: auth.Secret, out: *TestRank) void {
        const link = Link.open(std.testing.io, "127.0.0.1", port, rank, 5000, &secret) catch |e| {
            out.err = e;
            return;
        };
        defer link.close();
        const until = deadline(5000);
        if (rank == 0) {
            link.sendHeader(.{ .kind = .hello, .arg = 0 }, until) catch |e| {
                out.err = e;
            };
            var end: [1]u8 = undefined;
            link.receive(&end, until) catch {}; // rank 1 leaves first
        } else out.got = link.receiveHeader(until) catch |e| blk: {
            out.err = e;
            break :blk null;
        };
    }
};

test "security: rank 0 takes only the peer that proves the key, and a stranger that came first gets no frame" {
    if (builtin.os.tag != .linux) return error.SkipZigTest; // Link.open is Linux's
    const secret = try auth.Secret.fromBytes("the pair's rank secret for the tests");
    const port = try testPort();
    var lead: TestRank = .{};
    const t0 = try std.Thread.spawn(.{}, TestRank.run, .{ port, 0, secret, &lead });
    // a stranger first, saying what an older build's rank 1 said: before the proof it was the peer and got the identity
    const stranger = blk: {
        const until = deadline(5000);
        while (now() < until) : (pause()) {
            const fd = std.c.socket(p.AF.INET, p.SOCK.STREAM | p.SOCK.CLOEXEC, 0);
            var addr: p.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
            if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(p.sockaddr.in)) == 0) break :blk fd;
            _ = std.c.close(fd);
        }
        return error.RankTimeout;
    };
    defer _ = std.c.close(stranger);
    var said: [60]u8 = @splat(0);
    said[0..40].* = try (Header{ .kind = .hello, .arg = 1, .bytes = 20 }).encode();
    try std.testing.expectEqual(@as(isize, said.len), std.c.send(stranger, &said, said.len, p.MSG.NOSIGNAL));
    var back: [1]u8 = undefined;
    try std.testing.expectEqual(@as(isize, 0), std.c.recv(stranger, &back, 1, 0)); // closed with nothing sent
    var follow: TestRank = .{};
    const t1 = try std.Thread.spawn(.{}, TestRank.run, .{ port, 1, secret, &follow });
    t1.join();
    t0.join();
    try std.testing.expectEqual(@as(?anyerror, null), follow.err);
    try std.testing.expectEqual(@as(?anyerror, null), lead.err);
    try std.testing.expectEqual(Kind.hello, follow.got.?.kind);
}

test "security: a rank 1 with another key is refused and rank 0 keeps waiting for its peer" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const secret = try auth.Secret.fromBytes("the pair's rank secret for the tests");
    const other = try auth.Secret.fromBytes("another pair's secret for the tests");
    const port = try testPort();
    var lead: TestRank = .{};
    const t0 = try std.Thread.spawn(.{}, TestRank.run, .{ port, 0, secret, &lead });
    var wrong: TestRank = .{};
    TestRank.run(port, 1, other, &wrong);
    try std.testing.expectEqual(@as(?anyerror, error.RankUnproven), wrong.err);
    var follow: TestRank = .{};
    TestRank.run(port, 1, secret, &follow);
    t0.join();
    try std.testing.expectEqual(@as(?anyerror, null), follow.err);
    try std.testing.expectEqual(@as(?anyerror, null), lead.err);
}

/// A host on the path between rank 1 and rank 0: it passes every byte, keeps what rank 0 sent, and can change a bit.
const TestRelay = struct {
    server: p.socket_t,
    to: u16,
    seen: [1 << 14]u8 = undefined,
    len: usize = 0,
    tamper: std.atomic.Value(bool) = .init(false),

    fn listen(port: u16) !p.socket_t {
        const fd = std.c.socket(p.AF.INET, p.SOCK.STREAM | p.SOCK.CLOEXEC, 0);
        if (p.errno(fd) != .SUCCESS) return error.RankSocket;
        var addr: p.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
        if (std.c.bind(fd, @ptrCast(&addr), @sizeOf(p.sockaddr.in)) != 0 or std.c.listen(fd, 1) != 0) return error.RankBind;
        return fd;
    }

    fn run(r: *TestRelay) void {
        const down = std.c.accept(r.server, null, null);
        defer _ = std.c.close(down);
        const up = std.c.socket(p.AF.INET, p.SOCK.STREAM | p.SOCK.CLOEXEC, 0);
        defer _ = std.c.close(up);
        var addr: p.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, r.to), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
        while (std.c.connect(up, @ptrCast(&addr), @sizeOf(p.sockaddr.in)) != 0) pause();
        var fds = [_]p.pollfd{ .{ .fd = down, .events = p.POLL.IN, .revents = 0 }, .{ .fd = up, .events = p.POLL.IN, .revents = 0 } };
        var buf: [4096]u8 = undefined;
        while (true) {
            _ = p.poll(&fds, -1) catch return;
            for (fds, 0..) |f, i| {
                if (f.revents == 0) continue;
                const rc = std.c.recv(f.fd, &buf, buf.len, 0);
                if (rc <= 0) return;
                const n: usize = @intCast(rc);
                if (i == 1) {
                    // the last byte it holds is the end of a record's tag
                    if (r.tamper.swap(false, .acq_rel)) buf[n - 1] ^= 1;
                    @memcpy(r.seen[r.len..][0..n], buf[0..n]);
                    r.len += n;
                }
                var at: usize = 0;
                while (at < n) {
                    const w = std.c.send(fds[1 - i].fd, buf[at..].ptr, n - at, p.MSG.NOSIGNAL);
                    if (w <= 0) return;
                    at += @intCast(w);
                }
            }
        }
    }
};

/// Rank 0 of a pair: it joins, waits to be told, and sends one forward frame.
const TestSender = struct {
    port: u16,
    secret: auth.Secret,
    go: std.atomic.Value(bool) = .init(false),
    err: ?anyerror = null,

    const text = "prompt ids 17 4021 88213";

    fn run(s: *TestSender) void {
        const link = Link.open(std.testing.io, "127.0.0.1", s.port, 0, 5000, &s.secret) catch |e| {
            s.err = e;
            return;
        };
        defer link.close();
        while (!s.go.load(.acquire)) pause();
        const until = deadline(5000);
        link.sendHeader(.{ .kind = .forward, .arg = 7, .bytes = text.len }, until) catch |e| {
            s.err = e;
            return;
        };
        link.send(text, until) catch |e| {
            s.err = e;
            return;
        };
        var end: [1]u8 = undefined;
        link.receive(&end, until) catch {}; // rank 1 leaves first
    }
};

fn testForward(link: Link, got: *[TestSender.text.len]u8) !void {
    const until = deadline(5000);
    const h = try link.receiveHeader(until);
    try std.testing.expectEqual(Kind.forward, h.kind);
    try std.testing.expectEqual(@as(u32, got.len), h.bytes);
    try link.receive(got, until);
}

/// Rank 1 joins rank 0 through a relay and reads one frame; `tamper` has the relay change a bit of it.
fn testThroughRelay(secret: auth.Secret, tamper: bool, relay: *TestRelay, got: *[TestSender.text.len]u8) !void {
    var sender: TestSender = .{ .port = try testPort(), .secret = secret };
    const relay_port = try testPort();
    relay.* = .{ .server = try TestRelay.listen(relay_port), .to = sender.port };
    defer _ = std.c.close(relay.server);
    const lead = try std.Thread.spawn(.{}, TestSender.run, .{&sender});
    const path = try std.Thread.spawn(.{}, TestRelay.run, .{relay});
    const link = try Link.open(std.testing.io, "127.0.0.1", relay_port, 1, 5000, &secret);
    if (tamper) relay.tamper.store(true, .release);
    sender.go.store(true, .release);
    const result = testForward(link, got);
    link.close();
    lead.join();
    path.join();
    try std.testing.expectEqual(@as(?anyerror, null), sender.err);
    return result;
}

test "security: the pair's frames cross sealed, and a bit changed on the way ends the link" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var got: [TestSender.text.len]u8 = undefined;
    var relay: TestRelay = undefined;
    const secret = try auth.Secret.fromBytes("the pair's rank secret for the tests");
    try testThroughRelay(secret, false, &relay, &got);
    try std.testing.expectEqualStrings(TestSender.text, &got);
    // a host on the path saw the key proof and then records: no frame magic, none of the payload
    const seen = relay.seen[0..relay.len];
    try std.testing.expect(seen.len > 40 + got.len);
    for ([_][]const u8{ "TFR1", "prompt", "4021" }) |clear| try std.testing.expect(std.mem.indexOf(u8, seen, clear) == null);
    try std.testing.expectError(error.RankDisconnected, testThroughRelay(secret, true, &relay, &got));
}

test "security: a pair with no key, on one host, is not sealed: after the proof its frames cross as they are" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    var got: [TestSender.text.len]u8 = undefined;
    var relay: TestRelay = undefined;
    try testThroughRelay(.none, false, &relay, &got);
    try std.testing.expectEqualStrings(TestSender.text, &got);
    // the proof's challenge, then the forward frame's header and payload: no record heads, no tags
    const header = try (Header{ .kind = .forward, .arg = 7, .bytes = TestSender.text.len }).encode();
    const seen = relay.seen[0..relay.len];
    try std.testing.expectEqual(auth.challenge_len + header.len + got.len, seen.len);
    try std.testing.expectEqualSlices(u8, &header, seen[auth.challenge_len..][0..header.len]);
    try std.testing.expectEqualStrings(TestSender.text, seen[auth.challenge_len + header.len ..]);
}
