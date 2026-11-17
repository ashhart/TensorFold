//! Flash Next's prompt states for the prompt cache: cache rows, DeltaNet states and the n-gram tail at a mark.
const std = @import("std");
const mtl = @import("metal");
const fz = @import("replay.zig");
const Engine = @import("engine.zig").Engine;

const LAYERS = fz.LAYERS;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const KEY_ROW = 256 * 2; // a key or value row of one head (bf16)
const RAW_ROW = 128 * 2; // an indexer key row (bf16)
const TAIL = fz.PLE_TAIL * fz.WIDE * 2; // the n-gram conv tail's rows

/// Where a state's DeltaNet rows live: the prompt pass's slot pair, the one layout snapshots are taken in today.
pub const Layout = enum { slots };

/// One kept state: its bytes, position, n-gram history, layout that wrote it, buffer size and pool.
pub const State = struct { buf: mtl.Buffer, at: usize, hist: [2]i64, bytes: usize, layout: Layout, cap: usize = 0, pool: ?*Pool = null };

/// Buffer size for a state of `n` bytes: an eighth more, in 32 MiB steps, so the next state of the conversation fits.
pub fn capacity(n: usize) usize {
    return std.mem.alignForward(usize, n + n / 8, 32 << 20);
}

/// Kept states' buffers: a dropped one waits for the next save; one is readied ahead (pages touched) while idle.
pub const Pool = struct {
    bufs: [2]?mtl.Buffer = .{ null, null },
    caps: [2]usize = .{ 0, 0 },
    last: usize = 0, // the bytes of the last state saved: the next is about this, a turn or so longer
    load_limit: u64 = 0,
    max: usize = std.math.maxInt(usize), // the prompt cache's budget: a buffer pads a state only up to it
    follow: ?mtl.ResidencySet.Follow = null, // the host's residency set: a new buffer joins it, a freed one leaves it

    /// A new buffer for states, in the host's residency set when it has one.
    fn made(p: *const Pool, device: mtl.Device, cap: usize) !mtl.Buffer {
        if (p.load_limit != 0 and !@import("flashnext_memory").growingFits(device.allocated(), cap, p.load_limit))
            return error.FlashNextMemoryShortfall;
        const buf = try device.buffer(cap, fz.opts);
        if (p.follow) |f| f.join(f.ctx, buf);
        return buf;
    }

    /// A buffer freed, out of the host's residency set first (the set's own reference would keep it).
    fn free(p: *const Pool, buf: mtl.Buffer) void {
        if (p.follow) |f| f.leave(f.ctx, buf);
        buf.deinit();
    }

    /// New buffer size for a state of `n` bytes: its capacity, capped at budget when the state fits.
    pub fn size(p: *const Pool, n: usize) usize {
        return if (n <= p.max) @min(capacity(n), p.max) else capacity(n);
    }

    /// The smallest free buffer that holds `n` bytes without wasting half again.
    fn fitting(p: *const Pool, n: usize) ?usize {
        var best: ?usize = null;
        for (p.bufs, p.caps, 0..) |b, cap, i| if (b != null and cap >= n and cap <= capacity(n) + capacity(n) / 2 and (best == null or cap < p.caps[best.?])) {
            best = i;
        };
        return best;
    }

    /// Whether a state of `n` bytes would take a free buffer (no new storage).
    pub fn fits(p: *const Pool, n: usize) bool {
        return p.fitting(n) != null;
    }

    /// The free buffers' bytes.
    pub fn spare(p: *const Pool) usize {
        var n: usize = 0;
        for (p.bufs, p.caps) |b, cap| if (b != null) {
            n += cap;
        };
        return n;
    }

    /// Free buffers go, the largest first, until at most `room` bytes stay free.
    pub fn trim(p: *Pool, room: usize) void {
        while (p.spare() > room) {
            const i: usize = if (p.bufs[0] == null) 1 else if (p.bufs[1] == null) 0 else if (p.caps[0] >= p.caps[1]) 0 else 1;
            p.free(p.bufs[i].?);
            p.bufs[i] = null;
            p.caps[i] = 0;
        }
    }

    /// A free buffer that holds `n` bytes, else a new one.
    fn take(p: *Pool, device: mtl.Device, n: usize) !struct { buf: mtl.Buffer, cap: usize } {
        if (p.fitting(n)) |i| {
            defer p.bufs[i] = null;
            return .{ .buf = p.bufs[i].?, .cap = p.caps[i] };
        }
        const cap = p.size(n);
        return .{ .buf = try p.made(device, cap), .cap = cap };
    }

    /// A buffer back: into a free slot, else in place of a smaller one, else freed.
    fn give(p: *Pool, buf: mtl.Buffer, cap: usize) void {
        const slot: usize = if (p.bufs[0] == null) 0 else if (p.bufs[1] == null) 1 else if (p.caps[0] <= p.caps[1]) 0 else 1;
        if (p.bufs[slot]) |old| {
            if (p.caps[slot] >= cap) return p.free(buf);
            p.free(old);
        }
        p.bufs[slot] = buf;
        p.caps[slot] = cap;
    }

    /// Idle: a free buffer for the next state (last one's and `more` bytes) in `room`, paged in till a request waits.
    pub fn ready(p: *Pool, device: mtl.Device, more: usize, waiting: Waiting, room: usize) Readied {
        if (p.last == 0) return .{};
        const n = p.last + more;
        for (p.bufs, p.caps) |b, cap| if (b != null and cap >= n) return .{};
        const cap = p.size(n);
        if (cap > room or waiting.check(waiting.ctx)) return .{};
        const buf = p.made(device, cap) catch return .{};
        const mem = buf.contents()[0..cap];
        var done: usize = 0;
        while (done < cap and !waiting.check(waiting.ctx)) {
            const end = @min(cap, done + (64 << 20));
            @memset(mem[done..end], 0);
            done = end;
        }
        p.give(buf, cap);
        return .{ .cap = cap, .touched = done };
    }

    /// What readying gives way to: a request waiting for the engine (rank 1: rank 0's next request).
    pub const Waiting = struct { ctx: *anyopaque, check: *const fn (*anyopaque) bool };
    pub const Readied = struct { cap: usize = 0, touched: usize = 0 };

    pub fn deinit(p: *Pool) void {
        for (&p.bufs) |*b| if (b.*) |x| {
            p.free(x);
            b.* = null;
        };
    }
};

/// The layout the engine's prompt pass keeps DeltaNet states in (a decode path with in-place states adds its own).
pub fn layoutOf(_: *const Engine) Layout {
    return .slots;
}

const Part = struct { live: fz.Buf, len: usize };
const MAX_PARTS = 36 * 2 + 13 * 5 + 1;

/// A state's buffer for the disk: DeltaNet states, each layer's two key and two value heads and indexer keys, a tail.
pub const shape = .{ .deltanet = 36 * (CS_ROW + SO_ROW), .layers = 13, .key_row = KEY_ROW, .raw_row = RAW_ROW, .tail = TAIL };

/// Bytes of a state after `at` tokens: 36 DeltaNet states, 13 layers of keys, values and indexer keys, the tail.
pub fn bytes(at: usize) usize {
    return 36 * (CS_ROW + SO_ROW) + 13 * at * (4 * KEY_ROW + RAW_ROW) + TAIL;
}

/// Pieces in order: DeltaNet states (from Engine.Passed past a mark), each attention layer's rows [0,at), head, tail.
fn parts(m: *fz.Model, at: usize, passed: ?Engine.Passed, out: *[MAX_PARTS]Part) usize {
    var n: usize = 0;
    var li: usize = 0;
    for (&m.layers) |*L| if (L.linear) {
        if (passed) |ps| {
            out[n] = .{ .live = m.marks.?.cs_at(li, ps.slot), .len = CS_ROW };
            out[n + 1] = .{ .live = m.marks.?.so_at(li, ps.slot), .len = SO_ROW };
        } else {
            out[n] = .{ .live = .{ .b = L.cs[m.state].b, .off = L.cs[m.state].off + m.state_row * CS_ROW }, .len = CS_ROW };
            out[n + 1] = .{ .live = .{ .b = L.so[m.state].b, .off = L.so[m.state].off + m.state_row * SO_ROW }, .len = SO_ROW };
        }
        n += 2;
        li += 1;
    };
    for (&m.layers) |*L| if (!L.linear) {
        n += rows(L.keys, L.vals, L.raw, m.cap, at, out[n..]);
    };
    n += rows(m.mtp.keys, m.mtp.vals, m.mtp.raw, m.cap, at, out[n..]);
    out[n] = .{ .live = if (passed) |ps| ps.tail else .{ .b = m.ple.cin.b, .off = m.ple.cin.off }, .len = TAIL };
    return n + 1;
}

fn rows(keys: fz.Buf, vals: fz.Buf, raw: fz.Buf, cap: usize, at: usize, out: []Part) usize {
    for (0..2) |hd| {
        out[hd] = .{ .live = .{ .b = keys.b, .off = keys.off + hd * cap * KEY_ROW }, .len = at * KEY_ROW };
        out[2 + hd] = .{ .live = .{ .b = vals.b, .off = vals.off + hd * cap * KEY_ROW }, .len = at * KEY_ROW };
    }
    out[4] = .{ .live = raw, .len = at * RAW_ROW };
    return 5;
}

/// Copies each part between the live buffers and `buf` (to_buf: save) on the engine's queue, and waits.
fn copy(e: *Engine, buf: mtl.Buffer, at: usize, to_buf: bool, passed: ?Engine.Passed) !void {
    const r = e.r;
    var list: [MAX_PARTS]Part = undefined;
    const n = parts(e.m, at, passed, &list);
    const cb = r.queue.commandBuffer();
    r.begin(cb, .serial);
    errdefer if (r.fsr != null) r.abort();
    var off: usize = 0;
    for (list[0..n]) |p| {
        if (p.len == 0) continue;
        const kept: fz.Buf = .{ .b = buf, .off = off };
        if (to_buf) e.pr.copyWords(p.live, kept, p.len / 4) else e.pr.copyWords(kept, p.live, p.len / 4);
        off += p.len;
    }
    try e.m.finish(cb);
    if (off != bytes(at)) return error.SnapshotSize;
}

/// The live state after `at` prompt tokens into a new buffer; the pass stands at `at`, or its last call passed it.
pub fn save(e: *Engine, gpa: std.mem.Allocator, at: usize) !*State {
    const passed: ?Engine.Passed = if (e.passed) |ps| (if (ps.at == at) ps else null) else null;
    if (at == 0 or (passed == null and e.m.pos != at)) return error.NotAtMark;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const n = bytes(at);
    const got = try e.snap_pool.take(e.r.device, n);
    errdefer e.snap_pool.give(got.buf, got.cap);
    try copy(e, got.buf, at, true, passed);
    const st = try gpa.create(State);
    st.* = .{ .buf = got.buf, .at = at, .hist = if (passed) |ps| ps.hist else e.m.ple.hist, .bytes = n, .layout = layoutOf(e), .cap = got.cap, .pool = &e.snap_pool };
    e.snap_pool.last = n;
    return st;
}

/// An empty state after `at` tokens, in a buffer from the pool: a learned state's file fills it (learned.zig).
pub fn blank(e: *Engine, gpa: std.mem.Allocator, at: usize) !*State {
    const n = bytes(at);
    const got = try e.snap_pool.take(e.r.device, n);
    errdefer e.snap_pool.give(got.buf, got.cap);
    const st = try gpa.create(State);
    st.* = .{ .buf = got.buf, .at = at, .hist = .{ 0, 0 }, .bytes = n, .layout = layoutOf(e), .cap = got.cap, .pool = &e.snap_pool };
    e.snap_pool.last = n;
    return st;
}

/// The live state becomes `st`'s: the next prompt chunk starts at its position (generateFrom with from = st.at).
pub fn restore(e: *Engine, st: *const State) !void {
    if (st.at > e.m.cap or st.bytes != bytes(st.at)) return error.SnapshotSize;
    if (st.layout != layoutOf(e)) return error.SnapshotLayout; // the store drops it and the pass starts at 0
    const m = e.m;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    e.hostMode();
    m.reset(); // a clean model if a copy fails; indexer blocks pool again from the restored keys (same kernel, same bits)
    try copy(e, st.buf, st.at, false, null);
    m.pos = st.at;
    m.ple.hist = st.hist;
}

pub fn drop(gpa: std.mem.Allocator, st: *State) void {
    if (st.pool) |p| p.give(st.buf, st.cap) else st.buf.deinit();
    gpa.destroy(st);
}

test "a state's parts add up to its bytes" {
    try std.testing.expectEqual(@as(usize, 36 * (CS_ROW + SO_ROW) + 184_320), bytes(0));
    try std.testing.expectEqual(@as(usize, 13 * 2304), bytes(1) - bytes(0));
}

test "a new buffer pads a state an eighth, in 32 MiB steps, but never past the budget a state fits" {
    const p: Pool = .{ .max = 1 << 30 };
    try std.testing.expectEqual(@as(usize, 704 << 20), p.size(600 << 20));
    try std.testing.expectEqual(@as(usize, 1 << 30), p.size(1000 << 20));
    try std.testing.expectEqual(capacity(1100 << 20), p.size(1100 << 20));
}

test "Flash Next context: packed snapshot rows are cap-independent while live head offsets follow each cap" {
    const at = 4096;
    const key: fz.Buf = .{ .b = undefined, .off = 64 };
    const val: fz.Buf = .{ .b = undefined, .off = 128 };
    const raw: fz.Buf = .{ .b = undefined, .off = 256 };
    var a: [5]Part = undefined;
    var b: [5]Part = undefined;
    try std.testing.expectEqual(@as(usize, 5), rows(key, val, raw, 8192, at, &a));
    try std.testing.expectEqual(@as(usize, 5), rows(key, val, raw, 65568, at, &b));
    for (a, b) |x, y| try std.testing.expectEqual(x.len, y.len);
    try std.testing.expectEqual(key.off + 8192 * KEY_ROW, a[1].live.off);
    try std.testing.expectEqual(key.off + 65568 * KEY_ROW, b[1].live.off);
    try std.testing.expectEqual(val.off + 8192 * KEY_ROW, a[3].live.off);
    try std.testing.expectEqual(val.off + 65568 * KEY_ROW, b[3].live.off);
}
