//! GLM's opt-in TP=2 step control: ordered host TCP frames beside the liveness socket.
const std = @import("std");
const platform = @import("sys");
const posix = platform.posix;

pub const Prefix = extern struct {
    magic: u32,
    bytes: u32,
    sequence: u64,

    comptime {
        std.debug.assert(@sizeOf(Prefix) == 16);
    }
};
pub const magic: u32 = 0x474c4d43;

pub const Channel = struct {
    fd: posix.socket_t,
    max_bytes: usize,
    next: u64 = 1,

    pub fn wanted() !bool {
        return @import("sys").environment.read(bool, "TF_GLM_HOST_CONTROL", false);
    }

    /// A connected rank 0/follower pair when TF_GLM_HOST_CONTROL is exactly 1.
    pub fn open(io: std.Io, rank: usize, world: usize, max_ids: usize) !?Channel {
        const enabled = try wanted();
        if (!enabled) return null;
        if (world != 2 or rank >= world) return error.GlmControlNeedsTwoRanks;
        const ids_bytes = std.math.mul(usize, max_ids, 4) catch return error.TooManyIds;
        const max_bytes = std.math.add(usize, 64, ids_bytes) catch return error.TooManyIds;
        const fd = try connect(io, rank);
        return .{ .fd = fd, .max_bytes = max_bytes };
    }

    pub fn deinit(c: *Channel) void {
        if (c.fd >= 0) _ = posix.system.close(c.fd);
        c.fd = -1;
    }

    /// A header and its ids cross in order; a partial send leaves this channel unusable.
    pub fn send(c: *Channel, header: []const u8, ids: []const u8) !void {
        if (header.len != 64 or ids.len % 4 != 0) return error.GlmControlFrame;
        const n = std.math.add(usize, header.len, ids.len) catch return error.GlmControlTooLarge;
        if (n > c.max_bytes or n > std.math.maxInt(u32)) return error.GlmControlTooLarge;
        errdefer c.deinit();
        const p: Prefix = .{ .magic = magic, .bytes = @intCast(n), .sequence = c.next };
        try writeAll(c.fd, std.mem.asBytes(&p));
        try writeAll(c.fd, header);
        if (ids.len > 0) try writeAll(c.fd, ids);
        c.next +%= 1;
        if (c.next == 0) return error.GlmControlOrder;
    }

    /// Nothing reaches `out` until one complete, bounded, next-sequence frame arrives.
    pub fn receive(c: *Channel, out: []u8) !usize {
        errdefer c.deinit();
        var p: Prefix = undefined;
        try readAll(c.fd, std.mem.asBytes(&p));
        if (p.magic != magic) return error.GlmControlFrame;
        if (p.sequence != c.next) return error.GlmControlOrder;
        const n: usize = p.bytes;
        if (n < 64 or n % 4 != 0) return error.GlmControlFrame;
        if (n > c.max_bytes or n > out.len) return error.GlmControlTooLarge;
        c.next +%= 1;
        if (c.next == 0) return error.GlmControlOrder;
        try readAll(c.fd, out[0..n]);
        return n;
    }
};

fn address() !posix.sockaddr.in {
    const id = std.mem.span(platform.c.getenv("NCCL_COMM_ID") orelse return error.GlmControlAddress);
    const colon = std.mem.lastIndexOfScalar(u8, id, ':') orelse return error.GlmControlAddress;
    const port = (std.fmt.parseInt(u16, id[colon + 1 ..], 10) catch return error.GlmControlAddress) +% 2;
    const ip = std.Io.net.IpAddress.parse(id[0..colon], port) catch return error.GlmControlAddress;
    if (ip != .ip4) return error.GlmControlAddress;
    return .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.readInt(u32, &ip.ip4.bytes, .little) };
}

fn socket() !posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM | posix.SOCK.CLOEXEC, 0);
    if (posix.errno(rc) != .SUCCESS) return error.GlmControlSocket;
    const fd: posix.socket_t = @intCast(rc);
    errdefer _ = posix.system.close(fd);
    try configure(fd);
    return fd;
}

fn configure(fd: posix.socket_t) !void {
    const one: c_int = 1;
    try posix.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, std.mem.asBytes(&one));
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.KEEPALIVE, std.mem.asBytes(&one)) catch {};
}

fn connect(io: std.Io, rank: usize) !posix.socket_t {
    var addr = try address();
    if (rank == 0) {
        const server = try socket();
        defer _ = posix.system.close(server);
        const one: c_int = 1;
        posix.setsockopt(server, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one)) catch return error.GlmControlSocket;
        if (posix.errno(posix.system.bind(server, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.GlmControlBind;
        if (posix.errno(posix.system.listen(server, 1)) != .SUCCESS) return error.GlmControlBind;
        var fds = [_]posix.pollfd{.{ .fd = server, .events = posix.POLL.IN, .revents = 0 }};
        if ((posix.poll(&fds, 300_000) catch 0) == 0) return error.GlmControlPeerLate;
        const rc = posix.system.accept(server, null, null);
        if (posix.errno(rc) != .SUCCESS) return error.GlmControlAccept;
        const fd: posix.socket_t = @intCast(rc);
        errdefer _ = posix.system.close(fd);
        try configure(fd);
        return fd;
    }
    var tries: usize = 0;
    while (true) : (tries += 1) {
        const fd = try socket();
        if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) return fd;
        _ = posix.system.close(fd);
        if (tries >= 3000) return error.GlmControlPeerLate;
        std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
    }
}

const nosignal: u32 = if (@hasDecl(posix.MSG, "NOSIGNAL")) posix.MSG.NOSIGNAL else 0;

fn writeAll(fd: posix.socket_t, bytes: []const u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = std.c.send(fd, bytes[at..].ptr, bytes.len - at, nosignal);
        if (rc < 0 and posix.errno(rc) == .INTR) continue;
        if (rc <= 0) return error.GlmControlSend;
        at += @intCast(rc);
    }
}

fn readAll(fd: posix.socket_t, bytes: []u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.read(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.GlmControlPeerClosed;
                at += @intCast(rc);
            },
            .INTR => {},
            else => return error.GlmControlReceive,
        }
    }
}
