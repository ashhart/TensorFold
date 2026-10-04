//! The two-sided link over verbs: per peer a UC queue pair and a ring of one-packet receives that a message fills in order.
const std = @import("std");
const abi = @import("verbs_abi.zig");
const vqp = @import("verbs_qp.zig");
const sr = @import("sendrecv.zig");

pub const page = std.heap.page_size_min;
/// Thunderbolt RDMA cuts a SEND into packets of one MTU, each landing in the next posted receive of at most this size.
pub const packet = 4096;
/// A receive that completes with this status holds a full packet of a message that goes on in the next receive.
const more = 1;
/// Every link message carries one byte after its payload, so an empty one still travels (a zero-length SEND is lost).
const trailer = 1;

/// Memory registered as one MR; Thunderbolt RDMA registers 1 GiB in one though it reports 15.6 MiB as its limit.
pub const Region = struct {
    mem: []align(page) u8,
    mr: *abi.Mr,

    pub fn register(ep: *vqp.Endpoint, mem: []align(page) u8) !Region {
        return .{ .mem = mem, .mr = try ep.register(mem) };
    }

    pub fn deregister(r: *Region, ep: *vqp.Endpoint) void {
        ep.deregister(r.mr);
    }
};

/// One-packet receives posted in ring order; a message's packets fill consecutive slots and its last has status 0.
pub const Ring = struct {
    ep: *vqp.Endpoint,
    mem: []u8,
    lkey: u32,
    stage: []u8,
    slots: u64,
    filled: u64 = 0,
    start: u64 = 0,
    len: usize = 0,
    done: [cap]Msg = undefined,
    pushed: u64 = 0,
    returned: u64 = 0,
    freed: u64 = 0,

    const cap = 1024;
    const Msg = struct { first: u64, n: u64, len: usize };

    /// Post every slot of `mem` (registered under `lkey`); `stage` holds a copy of any message that wraps the ring.
    pub fn init(ep: *vqp.Endpoint, mem: []u8, lkey: u32, stage: []u8) !Ring {
        var r: Ring = .{ .ep = ep, .mem = mem, .lkey = lkey, .stage = stage, .slots = mem.len / packet };
        for (0..r.slots) |s| try r.post(s);
        return r;
    }

    fn post(r: *Ring, slot: u64) !void {
        const s: usize = @intCast(slot % r.slots);
        try r.ep.receive(r.mem[s * packet ..][0..packet], r.lkey, slot);
    }

    /// Fold landed packets into whole messages.
    fn pump(r: *Ring) !void {
        var wc: [32]abi.Wc = undefined;
        const n = r.ep.ctx.ops.poll_cq(r.ep.rcq, 32, &wc);
        if (n < 0) return error.Completion;
        for (wc[0..@intCast(n)]) |w| {
            if (w.status != 0 and !(w.status == more and w.byte_len == packet)) return error.Completion;
            r.len += w.byte_len;
            r.filled += 1;
            if (w.status == more) continue;
            if (r.pushed - r.freed == cap) return error.Protocol;
            r.done[r.pushed % cap] = .{ .first = r.start, .n = r.filled - r.start, .len = r.len };
            r.pushed += 1;
            r.start = r.filled;
            r.len = 0;
        }
    }

    /// The oldest whole message not yet returned, contiguous (a wrapped one is copied to the stage), or null.
    pub fn next(r: *Ring) !?[]const u8 {
        if (r.returned == r.pushed) try r.pump();
        if (r.returned == r.pushed) return null;
        const m = r.done[r.returned % cap];
        r.returned += 1;
        const s: usize = @intCast(m.first % r.slots);
        if (s + m.n <= r.slots) return r.mem[s * packet ..][0..m.len];
        if (m.len > r.stage.len) return error.Protocol;
        const head = (r.slots - s) * packet;
        @memcpy(r.stage[0..head], r.mem[s * packet ..][0..head]);
        @memcpy(r.stage[head..m.len], r.mem[0 .. m.len - head]);
        return r.stage[0..m.len];
    }

    /// Hand the oldest returned message's slots back, posting them again in ring order.
    pub fn release(r: *Ring) !void {
        if (r.freed == r.returned) return error.Protocol;
        const m = r.done[r.freed % cap];
        for (0..m.n) |i| try r.post(m.first + i);
        r.freed += 1;
    }
};

/// One peer: its queue pair, a receive ring, and a registered send buffer.
pub const Peer = struct {
    ep: vqp.Endpoint,
    rx: Region = undefined,
    tx: Region = undefined,
    ring: Ring = undefined,
    head: usize = 0,

    /// SEND `bytes` plus the trailer through the send buffer; a full buffer waits for its sends first.
    pub fn send(p: *Peer, bytes: []const u8) !void {
        const buf = p.tx.mem;
        const len = bytes.len + trailer;
        if (len > buf.len) return error.TooLarge;
        if (p.head + len > buf.len) {
            try p.ep.drain();
            p.head = 0;
        }
        @memcpy(buf[p.head..][0..bytes.len], bytes);
        buf[p.head + bytes.len] = 0xA5;
        try p.ep.send(buf[p.head..][0..len], p.tx.mr.lkey, 0);
        p.head = std.mem.alignForward(usize, p.head + len, 64);
    }
};

/// This rank's sendrecv.Link: peers[r] is null for itself; every peer is reached over its own queue pair.
pub const Fabric = struct {
    me: u32,
    peers: []?*Peer,

    pub fn link(f: *Fabric) sr.Link {
        return .{ .ptr = f, .vtable = &vtable };
    }

    const vtable: sr.Link.VTable = .{ .rank = rank, .size = size, .send = send, .next = next, .release = release, .flush = flush };

    fn self(ptr: *anyopaque) *Fabric {
        return @ptrCast(@alignCast(ptr));
    }
    fn rank(ptr: *anyopaque) u32 {
        return self(ptr).me;
    }
    fn size(ptr: *anyopaque) u32 {
        return @intCast(self(ptr).peers.len);
    }
    fn peer(f: *Fabric, r: u32) sr.Error!*Peer {
        if (r >= f.peers.len) return error.NoSuchRank;
        return f.peers[r] orelse error.NoSuchRank;
    }
    fn send(ptr: *anyopaque, r: u32, bytes: []const u8) sr.Error!void {
        const p = try self(ptr).peer(r);
        p.send(bytes) catch |err| return if (err == error.TooLarge) error.TooLarge else error.PeerDown;
    }
    fn next(ptr: *anyopaque, r: u32) sr.Error!?[]const u8 {
        const p = try self(ptr).peer(r);
        const m = (p.ring.next() catch return error.PeerDown) orelse return null;
        if (m.len < trailer) return error.Protocol;
        return m[0 .. m.len - trailer];
    }
    fn release(ptr: *anyopaque, r: u32) sr.Error!void {
        const p = try self(ptr).peer(r);
        p.ring.release() catch return error.PeerDown;
    }
    fn flush(ptr: *anyopaque) sr.Error!void {
        for (self(ptr).peers) |maybe| if (maybe) |p| p.ep.drain() catch return error.PeerDown;
    }
};
