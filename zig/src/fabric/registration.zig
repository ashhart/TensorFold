//! A service's registration with its listen daemon ("MODE poll"), and the daemons' STATUS lines, as protocol 1 defines.
const std = @import("std");
const layout = @import("layout.zig");

pub const Error = error{ InvalidName, NoDaemon, Busy, Refused, Io };

/// The registration channel: the daemon sends nothing after OK until the link drops (BYE or a closed socket).
pub const Registration = struct {
    in_fd: std.c.fd_t,
    out_fd: std.c.fd_t,

    /// Register with the listen daemon at MCDMA_RPCD_SOCKET, else /tmp/mcdma-rpcd.NAME.sock.
    pub fn attach(name: []const u8) Error!Registration {
        if (!layout.validName(name)) return error.InvalidName;
        var buf: [104]u8 = undefined;
        const path = socketPath(&buf, name) catch return error.InvalidName;
        const fd = connectUnix(path) orelse return error.NoDaemon;
        var r: Registration = .{ .in_fd = fd, .out_fd = fd };
        r.handshake() catch |err| {
            r.close();
            return err;
        };
        return r;
    }

    /// Send MODE poll and read the one-line answer: OK, or ERR busy while another service holds the link.
    pub fn handshake(r: *const Registration) Error!void {
        try writeAll(r.out_fd, "MODE poll\n");
        var line: [64]u8 = undefined;
        const answer = try readLine(r.in_fd, &line);
        if (std.mem.eql(u8, answer, "OK")) return;
        if (std.mem.eql(u8, answer, "ERR busy")) return error.Busy;
        return error.Refused;
    }

    /// Whether the daemon still holds the registration: anything readable means BYE or a closed socket.
    pub fn alive(r: *const Registration) bool {
        var fds = [_]std.c.pollfd{.{ .fd = r.in_fd, .events = std.c.POLL.IN, .revents = 0 }};
        const n = std.c.poll(&fds, 1, 0);
        return n == 0;
    }

    pub fn close(r: *Registration) void {
        _ = std.c.close(r.in_fd);
        if (r.out_fd != r.in_fd) _ = std.c.close(r.out_fd);
        r.* = undefined;
    }
};

/// The listen daemon's socket for link `name`, honouring MCDMA_RPCD_SOCKET as mcdma-rpcd does.
pub fn socketPath(buf: []u8, name: []const u8) ![:0]const u8 {
    if (std.c.getenv("MCDMA_RPCD_SOCKET")) |env| {
        const s = std.mem.span(env);
        if (s.len > 0) return std.mem.printSentinel(buf, "{s}", .{s}, 0);
    }
    return std.mem.printSentinel(buf, "/tmp/mcdma-rpcd.{s}.sock", .{name}, 0);
}

/// Connect to the daemon's local Unix socket; the only socket the fabric opens, and never TCP.
fn connectUnix(path: [:0]const u8) ?std.c.fd_t {
    var addr: std.c.sockaddr.un = .{ .path = undefined };
    if (path.len >= addr.path.len) return null;
    @memset(&addr.path, 0);
    @memcpy(addr.path[0..path.len], path);
    const fd = std.c.socket(std.c.AF.UNIX, std.c.SOCK.STREAM, 0);
    if (fd < 0) return null;
    if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.un)) != 0) {
        _ = std.c.close(fd);
        return null;
    }
    return fd;
}

fn writeAll(fd: std.c.fd_t, bytes: []const u8) Error!void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (n <= 0) return error.Io;
        done += @intCast(n);
    }
}

/// Read one line a byte at a time, as the daemons write them; the newline is dropped.
fn readLine(fd: std.c.fd_t, buf: []u8) Error![]const u8 {
    var n: usize = 0;
    while (n < buf.len) {
        var c: [1]u8 = undefined;
        if (std.c.read(fd, &c, 1) != 1) return error.Io;
        if (c[0] == '\n') return buf[0..n];
        buf[n] = c[0];
        n += 1;
    }
    return error.Io;
}

/// One PEER line of a daemon's STATUS answer.
pub const Peer = struct {
    name: []const u8,
    up: bool,
    calls: u64,
    failures: u64,
    mib: u64,
    req_mib: u64 = 0,
    rep_mib: u64 = 0,
    since: i64 = 0,
    device: []const u8 = "",
    service: ?bool = null,
    /// From mcdma-rpcd 2.0.0: the exchange interface, its UDP port, and roce or thunderbolt.
    via: []const u8 = "",
    port: u16 = 0,
    link: []const u8 = "",
};

/// Parse "PEER NAME up|down calls N failures N MiB N key=value ..." from either daemon.
pub fn parsePeer(line: []const u8) ?Peer {
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    if (!std.mem.eql(u8, it.next() orelse return null, "PEER")) return null;
    var p: Peer = .{ .name = it.next() orelse return null, .up = false, .calls = 0, .failures = 0, .mib = 0 };
    const state = it.next() orelse return null;
    if (!std.mem.eql(u8, state, "up") and !std.mem.eql(u8, state, "down")) return null;
    p.up = std.mem.eql(u8, state, "up");
    inline for (.{ "calls", "failures", "MiB" }, .{ "calls", "failures", "mib" }) |key, field| {
        if (!std.mem.eql(u8, it.next() orelse return null, key)) return null;
        @field(p, field) = std.fmt.parseInt(u64, it.next() orelse return null, 10) catch return null;
    }
    while (it.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse return null;
        const key = kv[0..eq];
        const value = kv[eq + 1 ..];
        if (std.mem.eql(u8, key, "req_mib")) p.req_mib = std.fmt.parseInt(u64, value, 10) catch return null;
        if (std.mem.eql(u8, key, "rep_mib")) p.rep_mib = std.fmt.parseInt(u64, value, 10) catch return null;
        if (std.mem.eql(u8, key, "since")) p.since = std.fmt.parseInt(i64, value, 10) catch return null;
        if (std.mem.eql(u8, key, "device")) p.device = value;
        if (std.mem.eql(u8, key, "service")) p.service = std.mem.eql(u8, value, "attached");
        if (std.mem.eql(u8, key, "via")) p.via = value;
        if (std.mem.eql(u8, key, "link")) p.link = value;
        if (std.mem.eql(u8, key, "port")) p.port = std.fmt.parseInt(u16, value, 10) catch return null;
    }
    return p;
}

fn pipePair() ![2]std.c.fd_t {
    var fds: [2]std.c.fd_t = undefined;
    if (std.c.pipe(&fds) != 0) return error.Io;
    return fds;
}

test "registration answers OK, ERR busy or anything else, and BYE ends it" {
    for ([_][]const u8{ "OK\n", "ERR busy\n", "ERR unknown command\n" }, 0..) |answer, i| {
        const to_daemon = try pipePair();
        const from_daemon = try pipePair();
        defer for ([_]std.c.fd_t{ to_daemon[0], to_daemon[1], from_daemon[0], from_daemon[1] }) |fd| {
            _ = std.c.close(fd);
        };
        try writeAll(from_daemon[1], answer);
        const r: Registration = .{ .in_fd = from_daemon[0], .out_fd = to_daemon[1] };
        const got = r.handshake();
        var sent: [16]u8 = undefined;
        try std.testing.expectEqual(@as(isize, 10), std.c.read(to_daemon[0], &sent, sent.len));
        try std.testing.expectEqualStrings("MODE poll\n", sent[0..10]);
        switch (i) {
            0 => {
                try got;
                try std.testing.expect(r.alive());
                try writeAll(from_daemon[1], "BYE\n");
                try std.testing.expect(!r.alive());
            },
            1 => try std.testing.expectError(error.Busy, got),
            else => try std.testing.expectError(error.Refused, got),
        }
    }
}

test "STATUS lines from both daemons parse, 1.0.0 and 2.0.0" {
    const c = parsePeer("PEER worker-a up calls 12 failures 0 MiB 340 host=192.0.2.21 port=18620 device=rdma_mcrdma0 req_mib=4 rep_mib=64 since=1759500000").?;
    try std.testing.expectEqualStrings("worker-a", c.name);
    try std.testing.expect(c.up and c.calls == 12 and c.mib == 340 and c.rep_mib == 64 and c.since == 1759500000);
    try std.testing.expectEqualStrings("rdma_mcrdma0", c.device);
    try std.testing.expectEqual(@as(?bool, null), c.service);
    const l = parsePeer("PEER worker-a down calls 0 failures 1 MiB 0 device=rocep1s0f1 req_mib=4 rep_mib=64 since=0 service=none").?;
    try std.testing.expect(!l.up and l.failures == 1 and l.service.? == false);
    const v2 = parsePeer("PEER q2 up calls 5 failures 0 MiB 1 via=en9 port=18620 device=rdma_mcrdma0 link=roce req_mib=4 rep_mib=64 since=1759532400").?;
    try std.testing.expect(v2.up and v2.port == 18620 and v2.rep_mib == 64);
    try std.testing.expectEqualStrings("en9", v2.via);
    try std.testing.expectEqualStrings("roce", v2.link);
    try std.testing.expectEqual(@as(?Peer, null), parsePeer("VERSION mcdma-rpcd 1 2.0.0"));
    try std.testing.expectEqual(@as(?Peer, null), parsePeer("PEER x sideways calls 0 failures 0 MiB 0"));
}
