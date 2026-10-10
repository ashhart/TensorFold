//! The ranks' TCP link: rank 0 listens, the others connect once, get the communicator's id, take rank 0's commands.

const std = @import("std");
const posix = std.posix;
const rccl = @import("rccl.zig");
const Policy = @import("../policy.zig").Policy;

pub const Error = error{ SocketFailed, BindFailed, ConnectFailed, LinkClosed, BadRank, MixedGpus } || std.mem.Allocator.Error;

pub const max_world = 16;

/// What the ranks agree on at join: the kernels' GPU id (every rank's must match) and rank 0's policy.
pub const Hello = struct { caps: u32, policy: [Policy.word_count]u32 };

pub const Link = struct {
    rank: usize,
    world: usize,
    /// Rank 0: the follower of rank r + 1 at index r. A follower: rank 0's socket at index 0.
    fds: [max_world - 1]posix.socket_t = undefined,

    /// Rank 0 listens on `host:port`, sends each rank `id` and `hello`; a GPU that differs from rank 0's is refused.
    pub fn open(io: std.Io, rank: usize, world: usize, host: []const u8, port: u16, id: rccl.UniqueId, hello: Hello) Error!struct { Link, rccl.UniqueId, Hello } {
        if (world < 2 or world > max_world or rank >= world) return error.BadRank;
        const ip = std.Io.net.IpAddress.parse(host, port) catch return error.ConnectFailed;
        if (ip != .ip4) return error.ConnectFailed;
        var addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.readInt(u32, &ip.ip4.bytes, .little) };
        var link: Link = .{ .rank = rank, .world = world };
        if (rank == 0) {
            const server = try socket();
            defer _ = posix.system.close(server);
            const one: c_int = 1;
            posix.setsockopt(server, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one)) catch return error.SocketFailed;
            if (posix.errno(posix.system.bind(server, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
            if (posix.errno(posix.system.listen(server, 16)) != .SUCCESS) return error.BindFailed;
            var seen: usize = 0;
            while (seen < world - 1) : (seen += 1) {
                const rc = posix.system.accept(server, null, null);
                if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
                const fd: posix.socket_t = @intCast(rc);
                var peer: u32 = undefined;
                try readAll(fd, std.mem.asBytes(&peer));
                var caps: u32 = undefined;
                try readAll(fd, std.mem.asBytes(&caps));
                if (peer == 0 or peer >= world) return error.BadRank;
                link.fds[peer - 1] = fd;
                const status: u32 = if (caps == hello.caps) 0 else 1;
                try writeAll(fd, std.mem.asBytes(&status));
                if (status != 0) return error.MixedGpus;
                try writeAll(fd, std.mem.asBytes(&id));
                try writeAll(fd, std.mem.sliceAsBytes(&hello.policy));
            }
            return .{ link, id, hello };
        }
        var tries: usize = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) {
                link.fds[0] = fd;
                break;
            }
            _ = posix.system.close(fd);
            if (tries >= 6000) return error.ConnectFailed;
            std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
        }
        const mine: u32 = @intCast(rank);
        try writeAll(link.fds[0], std.mem.asBytes(&mine));
        try writeAll(link.fds[0], std.mem.asBytes(&hello.caps));
        var status: u32 = undefined;
        try readAll(link.fds[0], std.mem.asBytes(&status));
        if (status != 0) return error.MixedGpus;
        var got: rccl.UniqueId = undefined;
        try readAll(link.fds[0], std.mem.asBytes(&got));
        var theirs: Hello = .{ .caps = hello.caps, .policy = undefined };
        try readAll(link.fds[0], std.mem.sliceAsBytes(&theirs.policy));
        return .{ link, got, theirs };
    }

    pub fn close(l: *Link) void {
        const n = if (l.rank == 0) l.world - 1 else 1;
        for (l.fds[0..n]) |fd| _ = posix.system.close(fd);
    }

    /// Rank 0: `words` to every follower, behind their count.
    pub fn send(l: *const Link, words: []const u32) Error!void {
        const n: u32 = @intCast(words.len);
        for (l.fds[0 .. l.world - 1]) |fd| {
            try writeAll(fd, std.mem.asBytes(&n));
            try writeAll(fd, std.mem.sliceAsBytes(words));
        }
    }

    /// A follower: rank 0's next message, appended to `out` (cleared first).
    pub fn recv(l: *const Link, gpa: std.mem.Allocator, out: *std.ArrayList(u32)) Error!void {
        var n: u32 = undefined;
        try readAll(l.fds[0], std.mem.asBytes(&n));
        out.clearRetainingCapacity();
        try out.resize(gpa, n);
        try readAll(l.fds[0], std.mem.sliceAsBytes(out.items));
    }
};

fn socket() Error!posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    const one: c_int = 1;
    posix.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, std.mem.asBytes(&one)) catch {};
    return fd;
}

fn writeAll(fd: posix.socket_t, bytes: []const u8) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.write(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => at += @intCast(rc),
            .INTR => {},
            else => return error.LinkClosed,
        }
    }
}

fn readAll(fd: posix.socket_t, bytes: []u8) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.read(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.LinkClosed;
                at += @intCast(rc);
            },
            .INTR => {},
            else => return error.LinkClosed,
        }
    }
}
