//! A ring of pipeline stages over TCP: each rank sends to the next and receives from the one before, in round order.
const std = @import("std");
const net = @import("net.zig");
const frame = @import("frame.zig");

pub const Peer = struct { host: []const u8, port: u16 };

pub const Options = struct {
    rank: u32,
    world: u32,
    /// Every rank's address, in rank order; a rank listens on its own entry only.
    peers: []const Peer,
    /// The checkpoint's and layout's fingerprint: a neighbour with another one is refused.
    fingerprint: [32]u8,
    /// How long the ring may take to form, and each later send or receive by default.
    timeout_ns: u64 = 60 * std.time.ns_per_s,
    /// The largest payload a frame may carry (a prompt chunk's rows, a plan).
    max_bytes: u64 = 256 << 20,
};

pub const Error = net.Error || frame.Error || error{ BadRing, NotNeighbour, OtherWorld, OtherFingerprint, OutOfOrder, WrongKind, Stopped, Broken };

const hello_bytes = 40; // rank, world, fingerprint

pub const Ring = struct {
    rank: u32,
    world: u32,
    max_bytes: u64,
    timeout_ns: u64,
    next: net.Fd,
    prev: net.Fd,
    /// The last round sent to the next rank and received from the one before.
    sent: u64 = 0,
    received: u64 = 0,
    /// Set by any failed send or receive: the stream may be mid-frame, so every later call is `error.Broken`.
    broken: bool = false,

    /// This rank's listener, on its own address only; `join` takes it.
    pub fn bind(o: Options) Error!net.Fd {
        try check(o);
        return net.listen(o.peers[o.rank].host, o.peers[o.rank].port);
    }

    /// Connects to the next rank, accepts the one before, and checks both ends agree on the ring (closes `listener`).
    pub fn join(o: Options, listener: net.Fd) Error!Ring {
        defer net.close(listener);
        try check(o);
        const deadline = net.nowNs() + o.timeout_ns;
        const nxt = (o.rank + 1) % o.world;
        const next = try net.connect(o.peers[nxt].host, o.peers[nxt].port, deadline);
        errdefer net.close(next);
        const prev = try net.accept(listener, deadline);
        errdefer net.close(prev);
        var r: Ring = .{ .rank = o.rank, .world = o.world, .max_bytes = o.max_bytes, .timeout_ns = o.timeout_ns, .next = next, .prev = prev };
        var mine: [hello_bytes]u8 = undefined;
        std.mem.writeInt(u32, mine[0..4], o.rank, .little);
        std.mem.writeInt(u32, mine[4..8], o.world, .little);
        mine[8..40].* = o.fingerprint;
        try r.put(.hello, 0, 0, &.{&mine}, deadline);
        var theirs: [hello_bytes]u8 = undefined;
        _ = try r.take(.hello, 0, &theirs, deadline);
        if (std.mem.readInt(u32, theirs[4..8], .little) != o.world) return error.OtherWorld;
        if (std.mem.readInt(u32, theirs[0..4], .little) != (o.rank + o.world - 1) % o.world) return error.NotNeighbour;
        if (!std.mem.eql(u8, theirs[8..40], &o.fingerprint)) return error.OtherFingerprint;
        return r;
    }

    pub fn open(o: Options) Error!Ring {
        return join(o, try bind(o));
    }

    /// Round `round`'s frame of `kind` to the next rank: a header, then `parts` back to back.
    pub fn send(r: *Ring, kind: frame.Kind, round: u64, rows: u32, parts: []const []const u8) Error!void {
        if (r.broken) return error.Broken;
        errdefer r.broken = true;
        if (kind == .hello or (kind != .stop and round != r.sent + 1 and round != r.sent)) return error.OutOfOrder;
        try r.put(kind, round, rows, parts, net.nowNs() + r.timeout_ns);
        r.sent = round;
    }

    /// Round `round`'s frame of `kind` from the rank before into `out`; its header, the payload at `out[0..bytes]`.
    pub fn recv(r: *Ring, kind: frame.Kind, round: u64, out: []u8) Error!frame.Header {
        if (r.broken) return error.Broken;
        errdefer r.broken = true;
        const h = try r.take(kind, round, out, net.nowNs() + r.timeout_ns);
        r.received = round;
        return h;
    }

    /// A clean stop to the next rank: it sees `error.Stopped` where it waited for a frame.
    pub fn stop(r: *Ring) void {
        r.put(.stop, r.sent, 0, &.{}, net.nowNs() + r.timeout_ns) catch {};
    }

    /// Both links closed: a neighbour waiting on either sees `error.Closed`, so a failed rank ends the ring.
    pub fn close(r: *Ring) void {
        net.close(r.next);
        net.close(r.prev);
    }

    fn put(r: *Ring, kind: frame.Kind, round: u64, rows: u32, parts: []const []const u8, deadline: u64) Error!void {
        var bytes: u64 = 0;
        for (parts) |p| bytes += p.len;
        if (bytes > r.max_bytes) return error.TooLarge;
        const head = frame.encode(.{ .kind = kind, .round = round, .rows = rows, .bytes = bytes });
        try net.sendAll(r.next, &head, deadline);
        for (parts) |p| try net.sendAll(r.next, p, deadline);
    }

    fn take(r: *Ring, kind: frame.Kind, round: u64, out: []u8, deadline: u64) Error!frame.Header {
        var raw: [frame.size]u8 = undefined;
        try net.recvExact(r.prev, &raw, deadline);
        const h = try frame.decode(&raw, @min(r.max_bytes, out.len));
        if (h.kind == .stop and kind != .stop) return error.Stopped;
        if (h.kind != kind) return error.WrongKind;
        if (h.round != round) return error.OutOfOrder;
        try net.recvExact(r.prev, out[0..@intCast(h.bytes)], deadline);
        return h;
    }
};

fn check(o: Options) Error!void {
    if (o.world < 2 or o.rank >= o.world or o.peers.len != o.world) return error.BadRing;
}

test {
    _ = @import("ring_test.zig");
}
