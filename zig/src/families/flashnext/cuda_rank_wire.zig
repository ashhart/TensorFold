//! Bounded Linux TCP control channel. Tensor bytes travel over NCCL; this channel owns command order and peer EOF.
const std = @import("std");
const builtin = @import("builtin");
const p = std.posix;
const net = std.Io.net;

pub const max_payload = 1 << 20;
pub const Kind = enum(u32) { hello = 1, ready, create, forward, round, drop_seq, save, restore, drop_snap, stop, ack, verify, keep };
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
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0 or ts.sec < 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_ms * 1000 + @as(u64, @intCast(ts.nsec));
}

pub fn deadline(ms: u64) u64 {
    return now() +| (ms *| std.time.ns_per_ms);
}

pub fn pause() void {
    const ts: std.c.timespec = .{ .sec = 0, .nsec = 50_000 };
    _ = std.c.nanosleep(&ts, null);
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

    pub fn close(l: Link) void {
        _ = std.c.shutdown(l.fd, p.SHUT.RDWR);
        _ = std.c.close(l.fd);
    }

    /// Exactly two rank processes. Retry only connection establishment, never an accepted command.
    pub fn open(io: std.Io, host: []const u8, port: u16, rank: usize, timeout_ms: u64) !Link {
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
                        errdefer _ = std.c.close(peer);
                        try noDelay(peer);
                        return .{ .fd = peer };
                    },
                    .INTR, .AGAIN => continue,
                    else => return error.RankAccept,
                }
            }
        }
        while (now() < until) {
            const fd = try socket(family);
            const rc = std.c.connect(fd, &storage.any, len);
            if (rc == 0) return .{ .fd = fd };
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
                if (std.c.getsockopt(fd, p.SOL.SOCKET, p.SO.ERROR, &status, &size) == 0 and status == 0) return .{ .fd = fd };
            }
            _ = std.c.close(fd);
            pause();
        }
        return error.RankTimeout;
    }

    pub fn send(l: Link, bytes: []const u8, until: u64) !void {
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

    pub fn receive(l: Link, bytes: []u8, until: ?u64) !void {
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
        var byte: u8 = 0;
        const rc = std.c.recv(l.fd, &byte, 1, p.MSG.PEEK | p.MSG.DONTWAIT);
        if (rc == 0) return error.RankDisconnected;
        if (rc < 0) switch (p.errno(rc)) {
            .AGAIN, .INTR => {},
            else => return error.RankDisconnected,
        };
    }
};

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
