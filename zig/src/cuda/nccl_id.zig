//! Two ranks' NCCL communicator: rank 0 serves its 128-byte unique id on host:port, rank 1 connects (started first).
const std = @import("std");
const posix = std.posix;
const nccl = @import("nccl.zig");

/// The pair's communicator for `rank` (0 or 1); the id crosses the cable over one TCP connection.
pub fn pair(lib: *const nccl.Library, io: std.Io, rank: u8, addr: []const u8) !nccl.Comm {
    if (rank > 1) return error.BadRank;
    var id: nccl.UniqueId = undefined;
    const bytes = std.mem.asBytes(&id);
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        try serve(addr, bytes);
    } else {
        try receive(io, addr, bytes);
    }
    var comm: nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, 2, id, @intCast(rank)), "ncclCommInitRank");
    return comm;
}

/// Rank 0 binds host:port and sends its unique id to the one peer (70 s for rank 1's retries).
fn serve(addr_text: []const u8, bytes: []const u8) !void {
    const a = try parseAddr(addr_text);
    var storage: std.Io.Threaded.PosixAddress = undefined;
    const len = try posixLen(a.host, a.port, &storage);
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    defer _ = posix.system.close(fd);
    const one: c_int = 1;
    try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
    switch (posix.errno(posix.system.bind(fd, &storage.any, len))) {
        .SUCCESS => {},
        .ADDRINUSE => return error.AddressInUse,
        else => return error.BindFailed,
    }
    if (posix.errno(posix.system.listen(fd, 1)) != .SUCCESS) return error.ListenFailed;
    var waited: u64 = 0;
    while (true) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, 500) catch return error.AcceptFailed;
        if (ready > 0) break;
        waited += 500;
        if (waited > 70_000) return error.PeerTimeout;
    }
    var peer: std.Io.Threaded.PosixAddress = undefined;
    var peer_len: posix.socklen_t = @sizeOf(std.Io.Threaded.PosixAddress);
    const rc2 = posix.system.accept(fd, &peer.any, &peer_len);
    if (posix.errno(rc2) != .SUCCESS) return error.AcceptFailed;
    const cfd: posix.socket_t = @intCast(rc2);
    defer _ = posix.system.close(cfd);
    var off: usize = 0;
    while (off < bytes.len) {
        const n = posix.system.write(cfd, bytes.ptr + off, bytes.len - off);
        if (posix.errno(n) != .SUCCESS) return error.WriteFailed;
        off += @intCast(n);
    }
}

/// Rank 1 connects to host:port, retrying for 60 s (rank 0 may not have bound yet), and reads the id.
fn receive(io: std.Io, addr_text: []const u8, bytes: []u8) !void {
    const a = try parseAddr(addr_text);
    const ip = try std.Io.net.IpAddress.parseIp4(a.host, a.port);
    var fd: posix.socket_t = -1;
    defer {
        if (fd >= 0) _ = posix.system.close(fd);
    }
    var waited: u64 = 0;
    while (fd < 0) {
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&ip, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const c: posix.socket_t = @intCast(rc);
        if (posix.errno(posix.system.connect(c, &storage.any, len)) == .SUCCESS) {
            fd = c;
        } else {
            _ = posix.system.close(c);
            waited += 200;
            if (waited > 60_000) return error.PeerTimeout;
            std.Io.sleep(io, .fromMilliseconds(200), .awake) catch {};
        }
    }
    var off: usize = 0;
    var read_waited: u64 = 0;
    while (off < bytes.len) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        const ready = posix.poll(&fds, 500) catch return error.ReadFailed;
        if (ready == 0) {
            read_waited += 500;
            if (read_waited > 60_000) return error.PeerTimeout;
            continue;
        }
        const n = posix.system.read(fd, bytes.ptr + off, bytes.len - off);
        switch (posix.errno(n)) {
            .SUCCESS => {
                if (n == 0) return error.PeerClosed; // the peer died before its id arrived
                off += @intCast(n);
            },
            .INTR => {},
            else => return error.ReadFailed,
        }
    }
}

fn posixLen(host: []const u8, port: u16, storage: *std.Io.Threaded.PosixAddress) !posix.socklen_t {
    const ip = try std.Io.net.IpAddress.parseIp4(host, port);
    return std.Io.Threaded.addressToPosix(&ip, storage);
}

fn parseAddr(text: []const u8) !struct { host: []const u8, port: u16 } {
    const at = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.BadAddress;
    return .{ .host = text[0..at], .port = try std.fmt.parseInt(u16, text[at + 1 ..], 10) };
}
