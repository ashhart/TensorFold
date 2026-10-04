//! Lockstep collectives over one-sided writes: per peer one chunk write, then a flag word (step << 32 | bytes); no atomics.
const std = @import("std");
const rdma = @import("rdma.zig");
const words = @import("words.zig");
const Rdma = rdma.Rdma;

pub const Error = error{ TooLarge, Outstanding, PeerTimeout, Stale } || rdma.Error;

/// Most ranks a channel serves (stack scratch for chunk lists).
pub const max_ranks = 64;

fn flagWord(step: u64, len: usize) u64 {
    return (step & 0xFFFF_FFFF) << 32 | @as(u64, @intCast(len));
}

/// One exchange channel in every rank's window: per step parity, one flag word and one slot for each sender.
pub const Channel = struct {
    ep: Rdma,
    base: usize,
    slot_bytes: usize,
    step: u64 = 0,
    completed: u64 = 0,
    own: []const u8 = &.{},
    timeout_ns: u64 = 10 * std.time.ns_per_s,

    /// Window bytes a channel of `n` ranks with `slot_bytes` slots takes (a page multiple).
    pub fn bytes(n: u32, slot_bytes: usize) usize {
        return std.mem.alignForward(usize, 2 * n * (8 + std.mem.alignForward(usize, slot_bytes, 64)), rdma.page);
    }

    pub fn init(ep: Rdma, base: usize, slot_bytes: usize) Channel {
        std.debug.assert(ep.size() <= max_ranks and base % 64 == 0);
        return .{ .ep = ep, .base = base, .slot_bytes = std.mem.alignForward(usize, slot_bytes, 64) };
    }

    fn flagAt(ch: *const Channel, parity: u64, sender: u32) usize {
        return ch.base + (parity * ch.ep.size() + sender) * 8;
    }

    fn slotAt(ch: *const Channel, parity: u64, sender: u32) usize {
        const n = ch.ep.size();
        return ch.base + 2 * n * 8 + (parity * n + sender) * ch.slot_bytes;
    }

    /// Post `chunks[p]` to each rank p (empty allowed); chunks must stay unchanged until `complete` returns.
    pub fn post(ch: *Channel, chunks: []const []const u8) Error!u64 {
        if (ch.completed != ch.step) return error.Outstanding;
        const n = ch.ep.size();
        const me = ch.ep.rank();
        std.debug.assert(chunks.len == n);
        for (chunks) |c| if (c.len > ch.slot_bytes) return error.TooLarge;
        ch.step += 1;
        const parity = ch.step & 1;
        ch.own = chunks[me];
        for (0..n) |i| {
            const p: u32 = @intCast(i);
            if (p == me) continue;
            if (chunks[p].len > 0) try ch.ep.write(p, ch.slotAt(parity, me), chunks[p]);
            try ch.ep.signal(p, ch.flagAt(parity, me), flagWord(ch.step, chunks[p].len));
        }
        return ch.step;
    }

    /// Wait for every rank's chunk of `step`; out[p] views rank p's chunk (this rank's own is its posted chunk).
    pub fn complete(ch: *Channel, step: u64, out: [][]const u8) Error!void {
        std.debug.assert(step == ch.step and out.len == ch.ep.size());
        const me = ch.ep.rank();
        const parity = step & 1;
        const win = ch.ep.window();
        const deadline = words.nowNs() + ch.timeout_ns;
        for (out, 0..) |*o, i| {
            const p: u32 = @intCast(i);
            if (p == me) {
                o.* = ch.own;
                continue;
            }
            var w = ch.ep.local(ch.flagAt(parity, p));
            while (w >> 32 != step & 0xFFFF_FFFF) : (w = ch.ep.local(ch.flagAt(parity, p))) {
                if (words.nowNs() > deadline) return error.PeerTimeout;
                std.atomic.spinLoopHint();
            }
            const len: usize = @intCast(w & 0xFFFF_FFFF);
            if (len > ch.slot_bytes) return error.Stale;
            o.* = win[ch.slotAt(parity, p)..][0..len];
        }
        try ch.ep.flush();
        ch.completed = step;
    }

    /// Post then complete.
    pub fn exchange(ch: *Channel, chunks: []const []const u8, out: [][]const u8) Error!void {
        try ch.complete(try ch.post(chunks), out);
    }

    pub fn ranks(ch: *const Channel) u32 {
        return ch.ep.size();
    }

    pub fn myRank(ch: *const Channel) u32 {
        return ch.ep.rank();
    }
};

/// The collectives below take any step channel with exchange, ranks and myRank (Channel, or sendrecv.Exchange).

/// Every rank waits for every other: a step with empty chunks.
pub fn barrier(ch: anytype, scratch: [][]const u8) !void {
    const empty: [max_ranks][]const u8 = @splat(&.{});
    try ch.exchange(empty[0..ch.ranks()], scratch);
}

/// The leader's round plan to every rank, which is also the round barrier; returns the plan as this rank sees it.
pub fn broadcast(ch: anytype, root: u32, plan: []const u8, scratch: [][]const u8) ![]const u8 {
    var chunks: [max_ranks][]const u8 = @splat(&.{});
    if (ch.myRank() == root) {
        for (chunks[0..ch.ranks()]) |*c| c.* = plan;
    }
    try ch.exchange(chunks[0..ch.ranks()], scratch);
    return scratch[root];
}

/// Element types the reduction takes; sums are fp32 either way.
pub const Elem = enum {
    f32,
    bf16,

    pub fn size(e: Elem) usize {
        return if (e == .f32) 4 else 2;
    }
};

/// Sum every rank's partial rows in rank order 0..n-1, each element alone: the same bits on every rank, at any row count.
pub fn allReduce(ch: anytype, elem: Elem, partial: []const u8, out: []u8, scratch: [][]const u8) !void {
    std.debug.assert(partial.len == out.len and partial.len % elem.size() == 0);
    var chunks: [max_ranks][]const u8 = @splat(&.{});
    for (chunks[0..ch.ranks()]) |*c| c.* = partial;
    try ch.exchange(chunks[0..ch.ranks()], scratch);
    for (scratch) |s| if (s.len != partial.len) return error.Stale;
    reduce(elem, scratch, out);
}

/// out = ((p0 + p1) + p2) + ... in fp32, rounded once; the order never depends on timing, row count or batch.
pub fn reduce(elem: Elem, parts: []const []const u8, out: []u8) void {
    const count = out.len / elem.size();
    for (0..count) |i| {
        var acc: f32 = load(elem, parts[0], i);
        for (parts[1..]) |p| acc += load(elem, p, i);
        store(elem, out, i, acc);
    }
}

fn load(elem: Elem, b: []const u8, i: usize) f32 {
    return switch (elem) {
        .f32 => @bitCast(std.mem.readInt(u32, b[4 * i ..][0..4], .little)),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, b[2 * i ..][0..2], .little)) << 16),
    };
}

fn store(elem: Elem, b: []u8, i: usize, v: f32) void {
    switch (elem) {
        .f32 => std.mem.writeInt(u32, b[4 * i ..][0..4], @bitCast(v), .little),
        .bf16 => std.mem.writeInt(u16, b[2 * i ..][0..2], toBf16(v), .little),
    }
}

/// fp32 to bf16, round to nearest even; NaN stays a quiet NaN.
pub fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

/// Expert-parallel exchange: chunks[p] (any row count, maybe none) goes to rank p; out[p] is what p sent here.
pub fn allToAll(ch: anytype, chunks: []const []const u8, out: [][]const u8) !void {
    try ch.exchange(chunks, out);
}

/// Every rank's chunk to `root`; other ranks get only empty chunks.
pub fn gather(ch: anytype, root: u32, chunk: []const u8, out: [][]const u8) !void {
    var chunks: [max_ranks][]const u8 = @splat(&.{});
    chunks[root] = chunk;
    try ch.exchange(chunks[0..ch.ranks()], out);
}

/// One direction of a pipeline link: a flag per slot (step << 32 | bytes), a credit word the receiver returns, two slots.
pub const Pipe = struct {
    ep: Rdma,
    base: usize,
    slot_bytes: usize,
    peer: u32,
    sent: u64 = 0,
    received: u64 = 0,
    timeout_ns: u64 = 10 * std.time.ns_per_s,

    pub fn bytes(slot_bytes: usize) usize {
        return std.mem.alignForward(usize, 64 + 2 * std.mem.alignForward(usize, slot_bytes, 64), rdma.page);
    }

    /// Both ends pass the same `base`: the receiver's flags and slots and the sender's credit word live there.
    pub fn init(ep: Rdma, base: usize, slot_bytes: usize, peer: u32) Pipe {
        return .{ .ep = ep, .base = base, .slot_bytes = std.mem.alignForward(usize, slot_bytes, 64), .peer = peer };
    }

    fn flagAt(p: *const Pipe, step: u64) usize {
        return p.base + (step & 1) * 8;
    }

    fn creditAt(p: *const Pipe) usize {
        return p.base + 16;
    }

    fn slotAt(p: *const Pipe, step: u64) usize {
        return p.base + 64 + (step & 1) * p.slot_bytes;
    }

    /// Post one message; waits only while both slots hold messages the receiver has not released.
    pub fn send(p: *Pipe, msg: []const u8) Error!void {
        if (msg.len > p.slot_bytes) return error.TooLarge;
        const step = p.sent + 1;
        const deadline = words.nowNs() + p.timeout_ns;
        while (step > 2 and p.ep.local(p.creditAt()) < step - 2) {
            if (words.nowNs() > deadline) return error.PeerTimeout;
            std.atomic.spinLoopHint();
        }
        if (msg.len > 0) try p.ep.write(p.peer, p.slotAt(step), msg);
        try p.ep.signal(p.peer, p.flagAt(step), flagWord(step, msg.len));
        try p.ep.flush();
        p.sent = step;
    }

    /// The next message, valid until `release`.
    pub fn recv(p: *Pipe) Error![]const u8 {
        const step = p.received + 1;
        const deadline = words.nowNs() + p.timeout_ns;
        var w = p.ep.local(p.flagAt(step));
        while (w >> 32 != step & 0xFFFF_FFFF) : (w = p.ep.local(p.flagAt(step))) {
            if (words.nowNs() > deadline) return error.PeerTimeout;
            std.atomic.spinLoopHint();
        }
        const len: usize = @intCast(w & 0xFFFF_FFFF);
        if (len > p.slot_bytes) return error.Stale;
        return p.ep.window()[p.slotAt(step)..][0..len];
    }

    /// Hand the slot back to the sender.
    pub fn release(p: *Pipe) Error!void {
        p.received += 1;
        try p.ep.signal(p.peer, p.creditAt(), p.received);
    }
};

/// Copy `src` into the peer's window at `offset` in pieces of at most `piece` bytes, then store `done` at `flag`.
pub fn bulk(ep: Rdma, peer: u32, offset: usize, src: []const u8, piece: usize, flag: usize, done: u64) Error!void {
    var at: usize = 0;
    while (at < src.len) : (at += piece) {
        try ep.write(peer, offset + at, src[at..@min(src.len, at + piece)]);
    }
    try ep.signal(peer, flag, done);
    try ep.flush();
}
