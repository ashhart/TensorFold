//! Reserve the connection slot before creating its thread; an overload reply never waits for a slow peer.
const std = @import("std");
const platform = @import("sys");
const posix = platform.posix;
pub const default_limit: u32 = 128;
pub const response = "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 1\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";

pub fn acquire(count: *std.atomic.Value(u32), limit: u32) bool {
    var n = count.load(.acquire);
    while (n < limit) {
        n = count.cmpxchgWeak(n, n + 1, .acq_rel, .acquire) orelse return true;
    }
    return false;
}

/// How long a connection with no request in progress keeps its slot once the server is full.
pub const shed_after_ms: i32 = 5_000;

/// The open-connection count from which one address's share applies (96 of 128).
pub fn pressure(limit: u32) u32 {
    return limit - limit / 4;
}

/// One address's default share under pressure, so a single host cannot fill the server.
pub fn perPeer(limit: u32) u32 {
    return @max(1, limit / 4);
}

/// --max-connections-per-address and --trusted-proxy: one address's share under pressure, peers counted as local.
pub const Peering = struct {
    /// Slots one address may hold once `pressure` is reached; null: a quarter of the limit, 0: no share.
    per_address: ?u32 = null,
    /// Exact addresses never counted, as loopback is not: a reverse proxy, a load balancer, Docker's bridge gateway.
    trusted: []const std.Io.net.IpAddress = &.{},

    /// The share at `open` connections (this one counted) of `limit`; null below pressure or when it is off.
    pub fn cap(p: Peering, open: u32, limit: u32) ?u32 {
        const share = p.per_address orelse perPeer(limit);
        if (share == 0 or open < pressure(limit)) return null;
        return share;
    }

    pub fn trusts(p: Peering, peer: std.Io.net.IpAddress) bool {
        if (@import("auth.zig").loopbackAddress(peer)) return true;
        const k = exact(peer);
        for (p.trusted) |t| if (std.mem.eql(u8, &exact(t), &k)) return true;
        return false;
    }
};

/// What a connection with no request in progress asks: the open count and its limit.
pub const Shed = struct {
    open: *const std.atomic.Value(u32),
    limit: u32,

    /// True once it has waited `waited_ms` with every slot taken: below the limit a pooled client keeps its connection.
    pub fn due(s: Shed, waited_ms: i32) bool {
        return waited_ms >= shed_after_ms and s.open.load(.acquire) >= s.limit;
    }
};

/// A peer's slot at `open` connections (this one counted), or null when its address holds its share already.
pub fn admitPeer(peers: *Peers, io: std.Io, open: u32, limit: u32, rule: Peering, peer: std.Io.net.IpAddress) ?u16 {
    if (rule.trusts(peer)) return Peers.untracked;
    return peers.take(io, peer, rule.cap(open, limit));
}

/// The slot of a connection already counted in `open`, or null once it is refused for its address's share.
pub fn peerSlot(peers: *Peers, io: std.Io, open: *std.atomic.Value(u32), limit: u32, rule: Peering, fd: posix.socket_t, peer: std.Io.net.IpAddress) ?u16 {
    return admitPeer(peers, io, open.load(.acquire), limit, rule, peer) orelse {
        _ = open.fetchSub(1, .acq_rel);
        reject(fd);
        return null;
    };
}

/// Gives back a connection's slot and its place in the open count.
pub fn release(peers: *Peers, io: std.Io, open: *std.atomic.Value(u32), slot: u16) void {
    peers.give(io, slot);
    _ = open.fetchSub(1, .acq_rel);
}

/// Open connections by remote address (IPv6 by its /64), counted at any load so the share holds once pressure comes.
pub const Peers = struct {
    pub const capacity = 1024;
    pub const untracked: u16 = capacity;
    mutex: std.Io.Mutex = .init,
    used: [capacity]bool = @splat(false),
    keys: [capacity][16]u8 = undefined,

    /// A slot for `peer`, or null when its address holds `cap` already (null `cap`: counted, never refused).
    pub fn take(p: *Peers, io: std.Io, peer: std.Io.net.IpAddress, cap: ?u32) ?u16 {
        const k = key(peer);
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        var held: u32 = 0;
        var free: ?u16 = null;
        for (p.used, 0..) |used, i| {
            if (!used) {
                if (free == null) free = @intCast(i);
            } else if (std.mem.eql(u8, &p.keys[i], &k)) held += 1;
        }
        if (cap) |c| if (held >= c) return null;
        const slot = free orelse return untracked;
        p.used[slot] = true;
        p.keys[slot] = k;
        return slot;
    }

    pub fn give(p: *Peers, io: std.Io, slot: u16) void {
        if (slot == untracked) return;
        p.mutex.lockUncancelable(io);
        defer p.mutex.unlock(io);
        p.used[slot] = false;
    }
};

/// An address as 16 bytes: an IPv4 peer in its IPv4-mapped IPv6 form, so both forms of one address match.
fn exact(peer: std.Io.net.IpAddress) [16]u8 {
    return switch (peer) {
        .ip4 => |v4| .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, v4.bytes[0], v4.bytes[1], v4.bytes[2], v4.bytes[3] },
        .ip6 => |v6| v6.bytes,
    };
}

/// The counting key: IPv4 (mapped or not) by address, other IPv6 by its /64, the block one subscriber is given.
pub fn key(peer: std.Io.net.IpAddress) [16]u8 {
    var k = exact(peer);
    const mapped = std.mem.eql(u8, k[0..12], &[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff });
    if (!mapped) @memset(k[8..], 0);
    return k;
}

fn nonblocking(fd: posix.socket_t) bool {
    if (platform.windows) {
        const ws = struct {
            extern "ws2_32" fn ioctlsocket(socket: usize, cmd: i32, arg: *u32) callconv(.winapi) i32;
        };
        var one: u32 = 1;
        return ws.ioctlsocket(@bitCast(fd), @bitCast(@as(u32, 0x8004667e)), &one) == 0;
    }
    const c = platform.c;
    const flags = c.fcntl(fd, c.F.GETFL);
    const flag: c_int = @bitCast(@as(u32, @bitCast(c.O{ .NONBLOCK = true })));
    return flags >= 0 and c.fcntl(fd, c.F.SETFL, flags | flag) >= 0;
}

/// Best effort on an unshared accepted socket; close even if the peer cannot receive the overload reply.
pub fn reject(fd: posix.socket_t) void {
    defer _ = posix.system.close(fd);
    if (!nonblocking(fd)) return;
    var sent: usize = 0;
    while (sent < response.len) {
        const n = posix.system.write(fd, response[sent..].ptr, response.len - sent);
        if (posix.errno(n) != .SUCCESS or n == 0) return;
        sent += @intCast(n);
    }
}
