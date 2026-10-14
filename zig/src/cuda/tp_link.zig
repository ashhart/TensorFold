//! Two ranks over the hosts' network: a TCP control link (rank 0 listens, rank 1 connects; framed messages carry
//! requests and per-round decisions) and the NCCL communicator it brings up (rank 0 makes the unique id and sends it).

const std = @import("std");
const posix = std.posix;
const net = std.Io.net;
const Threaded = std.Io.Threaded;
const nccl = @import("nccl.zig");

pub const Error = error{ SocketFailed, BindFailed, ListenFailed, AcceptFailed, ConnectTimedOut, PeerClosed, IoFailed, FrameTooLarge };

pub const max_frame = 64 << 20;

/// A message: a tag the protocol defines and its bytes (owned by the caller's allocator).
pub const Message = struct { tag: u32, bytes: []u8 };

pub const Link = struct {
    fd: posix.socket_t,
    rank: u8,

    /// Rank 0: listen on ``port`` (all interfaces) and accept rank 1, waiting up to ``timeout_ms``.
    pub fn lead(port: u16, timeout_ms: i32) Error!Link {
        const address = net.IpAddress.parse("0.0.0.0", port) catch unreachable;
        var storage: Threaded.PosixAddress = undefined;
        const len = Threaded.addressToPosix(&address, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const lfd: posix.socket_t = @intCast(rc);
        defer _ = posix.system.close(lfd);
        const one: c_int = 1;
        posix.setsockopt(lfd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one)) catch return error.SocketFailed;
        if (posix.errno(posix.system.bind(lfd, &storage.any, len)) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(lfd, 1)) != .SUCCESS) return error.ListenFailed;
        var fds = [_]posix.pollfd{.{ .fd = lfd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, timeout_ms) catch return error.AcceptFailed;
        if (ready == 0) return error.ConnectTimedOut;
        const a = posix.system.accept(lfd, null, null);
        if (posix.errno(a) != .SUCCESS) return error.AcceptFailed;
        const fd: posix.socket_t = @intCast(a);
        noDelay(fd);
        return .{ .fd = fd, .rank = 0 };
    }

    /// Rank 1: connect to rank 0 at ``address``, retrying every 200 ms for up to ``timeout_ms``.
    pub fn follow(io: std.Io, address: net.IpAddress, timeout_ms: i64) Error!Link {
        var waited: i64 = 0;
        while (true) {
            var storage: Threaded.PosixAddress = undefined;
            const len = Threaded.addressToPosix(&address, &storage);
            const family: u32 = if (address == .ip4) posix.AF.INET else posix.AF.INET6;
            const rc = posix.system.socket(family, posix.SOCK.STREAM, 0);
            if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
            const fd: posix.socket_t = @intCast(rc);
            if (posix.errno(posix.system.connect(fd, &storage.any, len)) == .SUCCESS) {
                noDelay(fd);
                return .{ .fd = fd, .rank = 1 };
            }
            _ = posix.system.close(fd);
            if (waited >= timeout_ms) return error.ConnectTimedOut;
            std.Io.sleep(io, .fromMilliseconds(200), .awake) catch {};
            waited += 200;
        }
    }

    pub fn close(self: *Link) void {
        _ = posix.system.shutdown(self.fd, posix.SHUT.RDWR);
        _ = posix.system.close(self.fd);
        self.* = undefined;
    }

    /// One frame: tag and length (little-endian u32 each), then the bytes.
    pub fn send(self: Link, tag: u32, bytes: []const u8) Error!void {
        if (bytes.len > max_frame) return error.FrameTooLarge;
        var head: [8]u8 = undefined;
        std.mem.writeInt(u32, head[0..4], tag, .little);
        std.mem.writeInt(u32, head[4..8], @intCast(bytes.len), .little);
        try writeAll(self.fd, &head);
        try writeAll(self.fd, bytes);
    }

    /// The next frame; its bytes allocated with ``gpa``.
    pub fn recv(self: Link, gpa: std.mem.Allocator) (Error || std.mem.Allocator.Error)!Message {
        var head: [8]u8 = undefined;
        try readAll(self.fd, &head);
        const tag = std.mem.readInt(u32, head[0..4], .little);
        const n = std.mem.readInt(u32, head[4..8], .little);
        if (n > max_frame) return error.FrameTooLarge;
        const bytes = try gpa.alloc(u8, n);
        errdefer gpa.free(bytes);
        try readAll(self.fd, bytes);
        return .{ .tag = tag, .bytes = bytes };
    }

    /// Both ranks' ``mine`` exchanged (equal lengths): returns the peer's into ``theirs``.
    pub fn swap(self: Link, mine: []const u8, theirs: []u8) Error!void {
        if (self.rank == 0) {
            try writeAll(self.fd, mine);
            try readAll(self.fd, theirs);
        } else {
            try readAll(self.fd, theirs);
            try writeAll(self.fd, mine);
        }
    }
};

fn noDelay(fd: posix.socket_t) void {
    const one: c_int = 1;
    posix.setsockopt(fd, posix.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one)) catch {};
}

fn writeAll(fd: posix.socket_t, bytes: []const u8) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.write(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => at += @intCast(rc),
            .INTR, .AGAIN => continue,
            .PIPE, .CONNRESET => return error.PeerClosed,
            else => return error.IoFailed,
        }
    }
}

fn readAll(fd: posix.socket_t, bytes: []u8) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.read(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.PeerClosed;
                at += @intCast(rc);
            },
            .INTR, .AGAIN => continue,
            .CONNRESET => return error.PeerClosed,
            else => return error.IoFailed,
        }
    }
}

/// The two ranks' NCCL communicator: rank 0's unique id crosses the link, then both join.
pub fn communicator(lib: *nccl.Library, link: Link) !nccl.Comm {
    var uid: nccl.UniqueId = undefined;
    if (link.rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&uid), "ncclGetUniqueId");
        try link.send(1, &uid.internal);
    } else {
        var buf: [256]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        const m = try link.recv(fba.allocator());
        if (m.tag != 1 or m.bytes.len != uid.internal.len) return error.IoFailed;
        @memcpy(&uid.internal, m.bytes);
    }
    var comm: nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, 2, uid, link.rank), "ncclCommInitRank");
    return comm;
}

test "frames round-trip through a socket pair" {
    var fds: [2]posix.socket_t = undefined;
    if (std.c.socketpair(posix.AF.UNIX, posix.SOCK.STREAM, 0, &fds) != 0) return error.SkipZigTest;
    const a: Link = .{ .fd = fds[0], .rank = 0 };
    const b: Link = .{ .fd = fds[1], .rank = 1 };
    try a.send(7, "hello ranks");
    const m = try b.recv(std.testing.allocator);
    defer std.testing.allocator.free(m.bytes);
    try std.testing.expectEqual(@as(u32, 7), m.tag);
    try std.testing.expectEqualStrings("hello ranks", m.bytes);
    _ = posix.system.close(fds[0]);
    _ = posix.system.close(fds[1]);
}
