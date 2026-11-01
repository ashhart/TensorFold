//! MiMo-V2's kept prompt states: full-layer rows pinned in their slot, the sliding and drafter windows copied.

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const dflash = @import("dflash.zig");

/// A cache word on the plan link: [Op.cache, kind, id, slot, slot2, at].
pub const Kind = enum(i64) { save = 0, restore = 1, copy = 2, drop = 3 };

pub const WORDS = 6;

/// One ring of a slot: `base + slot * len * row`, `len` rows of `row` bytes; a state keeps its last `keep` rows.
const Plane = struct { base: u64, row: usize, len: usize, keep: usize };

/// The ring rows [lo, at) of a state as at most two runs of ring indices: (first row, rows) each.
pub fn runs(at: usize, keep: usize, len: usize) [2][2]usize {
    const lo = at -| keep;
    const n = at - lo;
    const first = lo % len;
    const a = @min(n, len - first);
    return .{ .{ first, a }, .{ 0, n - a } };
}

/// Every rank's copies of kept states' windows, by id.
pub const Copies = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *const fwd.Model,
    planes: []Plane = &.{}, // the sliding layers' K and V rings, then the drafter's
    bytes: usize = 0, // one state's copy
    map: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty,
    held: usize = 0,

    /// The planes a state copies: built after the drafter (when one is loaded) so its ring is in.
    pub fn init(gpa: std.mem.Allocator, m: *const fwd.Model, dr: ?*const dflash.Drafter) !Copies {
        const cfg = m.w.cfg;
        var list: std.ArrayList(Plane) = .empty;
        errdefer list.deinit(gpa);
        for (0..m.w.layers.len) |i| if (!cfg.full[i]) {
            const hk = cfg.kvHeads(i);
            try list.append(gpa, .{ .base = m.kc[i], .row = hk * 192 * 2, .len = m.ring, .keep = cfg.window });
            try list.append(gpa, .{ .base = m.vc[i], .row = hk * 128 * 2, .len = m.ring, .keep = cfg.window });
        };
        if (dr) |x| for (x.kc, x.vc) |k, v| {
            const p: Plane = .{ .base = k, .row = x.kv * 128 * 2, .len = x.ring, .keep = x.window + 2 * x.block };
            try list.append(gpa, p);
            try list.append(gpa, .{ .base = v, .row = p.row, .len = p.len, .keep = p.keep });
        };
        var c: Copies = .{ .gpa = gpa, .d = m.d, .m = m, .planes = try list.toOwnedSlice(gpa) };
        for (c.planes) |p| c.bytes += p.keep * p.row;
        return c;
    }

    pub fn deinit(c: *Copies) void {
        var it = c.map.valueIterator();
        while (it.next()) |b| b.free();
        c.map.deinit(c.gpa);
        c.gpa.free(c.planes);
        c.held = 0;
    }

    fn copy(c: *const Copies, s: cuda.Stream, dst: u64, src: u64, n: usize) !void {
        if (n > 0) try c.d.check(c.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, s.handle), "kept copy");
    }

    /// Each plane's rows of positions [at - keep, at) between `slot`'s rings and `buf` (save: rings -> buf).
    fn move(c: *const Copies, s: cuda.Stream, buf: u64, slot: u32, at: usize, save_: bool) !void {
        var off: usize = 0;
        for (c.planes) |p| {
            const ring = p.base + slot * p.len * p.row;
            for (runs(at, p.keep, p.len)) |r| {
                const dev = ring + r[0] * p.row;
                if (save_) try c.copy(s, buf + off, dev, r[1] * p.row) else try c.copy(s, dev, buf + off, r[1] * p.row);
                off += r[1] * p.row;
            }
            off += (p.keep - @min(p.keep, at)) * p.row; // a state under `keep` rows leaves the rest of its share unused
        }
    }

    /// Copy `slot`'s windows at `at` into a new buffer kept as `id` (on `s`, after the chunk that ends at the mark).
    pub fn save(c: *Copies, s: cuda.Stream, id: u64, slot: u32, at: usize) !void {
        if (slot >= c.m.slots) return error.BadSlot;
        if (c.map.contains(id)) return error.KeptTwice;
        var buf = try cuda.DeviceBuffer.alloc(c.d, c.bytes);
        errdefer buf.free();
        try c.move(s, buf.ptr, slot, at, true);
        try c.map.put(c.gpa, id, buf);
        c.held += c.bytes;
    }

    /// Kept `id`'s windows into `slot`'s rings (before the pass's first chunk), at the rows they came from.
    pub fn restore(c: *Copies, s: cuda.Stream, id: u64, slot: u32, at: usize) !void {
        if (slot >= c.m.slots) return error.BadSlot;
        const buf = c.map.get(id) orelse return error.NoSuchState;
        try c.move(s, buf.ptr, slot, at, false);
    }

    /// The full layers' first `at` rows of slot `from` into `to`, for a resume in another slot.
    pub fn copyPrefix(c: *const Copies, s: cuda.Stream, from: u32, to: u32, at: usize) !void {
        const m = c.m;
        const cfg = m.w.cfg;
        if (from >= m.slots or to >= m.slots or from == to) return error.BadSlot;
        if (at > m.context) return error.PastContext;
        for (0..m.w.layers.len) |i| if (cfg.full[i]) {
            const hk = cfg.kvHeads(i);
            for ([2]u64{ m.kc[i], m.vc[i] }, [2]usize{ hk * 192 * 2, hk * 128 * 2 }) |base, row|
                try c.copy(s, base + to * m.context * row, base + from * m.context * row, at * row);
        };
    }

    pub fn drop(c: *Copies, id: u64) void {
        var kv = c.map.fetchRemove(id) orelse return;
        kv.value.free();
        c.held -= c.bytes;
    }

    /// A follower's cache word (after Op.cache): the same copy rank 0 made.
    pub fn follow(c: *Copies, s: cuda.Stream, w: []const i64) !void {
        if (w.len != WORDS) return error.BadPlan;
        const kind: Kind = std.enums.fromInt(Kind, w[1]) orelse return error.BadPlan;
        const id: u64 = @intCast(w[2]);
        const slot: u32 = @intCast(w[3]);
        const at: usize = @intCast(w[5]);
        switch (kind) {
            .save => try c.save(s, id, slot, at),
            .restore => try c.restore(s, id, slot, at),
            .copy => try c.copyPrefix(s, slot, @intCast(w[4]), at),
            .drop => c.drop(id),
        }
    }
};

/// The words of a cache step (Op.cache is the first word's value, passed in so this file needs no engine).
pub fn words(op: i64, kind: Kind, id: u64, slot: u32, slot2: u32, at: usize) [WORDS]i64 {
    return .{ op, @backingInt(kind), @intCast(id), slot, slot2, @intCast(at) };
}

/// One kept state on rank 0: the slot whose full-layer rows it pins, its position, its window copy's id.
pub const State = struct {
    id: u64,
    slot: u32,
    at: u32,
    live: bool = true, // false once a pass rewrote rows under `at` (the cache drops it at its next lookup or eviction)
    freed: bool = false, // its window copy is gone already (on every rank)
    used: u64 = 0,
};

/// Rank 0's kept states and the slot choice that keeps them alive.
pub const Pins = struct {
    gpa: std.mem.Allocator,
    states: std.ArrayList(*State) = .empty,
    next: u64 = 1,
    tick: u64 = 0,

    pub fn deinit(p: *Pins) void {
        for (p.states.items) |st| p.gpa.destroy(st);
        p.states.deinit(p.gpa);
    }

    pub fn add(p: *Pins, slot: u32, at: u32) !*State {
        const st = try p.gpa.create(State);
        errdefer p.gpa.destroy(st);
        p.tick += 1;
        st.* = .{ .id = p.next, .slot = slot, .at = at, .used = p.tick };
        try p.states.append(p.gpa, st);
        p.next += 1;
        return st;
    }

    pub fn remove(p: *Pins, st: *State) void {
        if (std.mem.indexOfScalar(*State, p.states.items, st)) |i| _ = p.states.swapRemove(i);
        p.gpa.destroy(st);
    }

    pub fn touch(p: *Pins, st: *State) void {
        p.tick += 1;
        st.used = p.tick;
    }

    /// The lowest slot of `free` (slot bits) that no live state pins.
    pub fn unpinned(p: *const Pins, free: u32) ?u32 {
        var bits = free;
        while (bits != 0) : (bits &= bits - 1) {
            const slot: u32 = @ctz(bits);
            const pinned = for (p.states.items) |st| {
                if (st.live and st.slot == slot) break true;
            } else false;
            if (!pinned) return slot;
        }
        return null;
    }

    /// Whether a live state of `slot` sits past `at` (a pass resumed at `at` there would end it).
    pub fn later(p: *const Pins, slot: u32, at: u32) bool {
        for (p.states.items) |st| if (st.live and st.slot == slot and st.at > at) return true;
        return false;
    }

    /// A stream's slot from `free`: the lowest no live state pins, else the one whose newest state is oldest.
    pub fn pick(p: *const Pins, free: u32) ?u32 {
        if (free == 0) return null;
        var best: ?u32 = null;
        var best_used: u64 = std.math.maxInt(u64);
        var bits = free;
        while (bits != 0) : (bits &= bits - 1) {
            const slot: u32 = @ctz(bits);
            var newest: u64 = 0;
            for (p.states.items) |st| if (st.live and st.slot == slot) {
                newest = @max(newest, st.used);
            };
            if (newest == 0) return slot;
            if (newest < best_used) {
                best_used = newest;
                best = slot;
            }
        }
        return best;
    }

    /// A resume's slot: `st`'s own when free and safe, else a free unpinned one (prefix copied), else LRU (or null).
    pub fn resumeSlot(p: *const Pins, free: u32, st: *const State) ?u32 {
        const own = free & (@as(u32, 1) << @intCast(st.slot)) != 0;
        if (own and !p.later(st.slot, st.at)) return st.slot;
        if (p.unpinned(free)) |u| return u;
        if (own) return st.slot;
        return p.pick(free);
    }

    /// A pass on `slot` from `from` ends its live states past it; the first `out.len` go to `out` to free now.
    pub fn overwrite(p: *Pins, slot: u32, from: u32, out: []*State) []*State {
        var n: usize = 0;
        for (p.states.items) |st| if (st.live and st.slot == slot and st.at > from) {
            st.live = false;
            if (n < out.len) {
                out[n] = st;
                n += 1;
            }
        };
        return out[0..n];
    }
};

/// The rows of a pass's next chunk from `at` of `len`: at most `cap`, ending at the next mark past `at` if sooner.
pub fn chunkRows(at: usize, len: usize, cap: usize, marks: []const u32) usize {
    var n = @min(cap, len - at);
    for (marks) |mk| if (mk > at and mk < at + n) {
        n = mk - at;
        break;
    };
    return n;
}

test "a state's window as ring runs" {
    const t = std.testing;
    try t.expectEqual([2][2]usize{ .{ 72, 128 }, .{ 0, 0 } }, runs(200, 128, 16512)); // rows 72 .. 199
    try t.expectEqual([2][2]usize{ .{ 0, 50 }, .{ 0, 0 } }, runs(50, 128, 16512)); // a short prompt keeps all it has
    try t.expectEqual([2][2]usize{ .{ 16484, 28 }, .{ 0, 100 } }, runs(16612, 128, 16512)); // the window wraps
    try t.expectEqual([2][2]usize{ .{ 0, 128 }, .{ 0, 0 } }, runs(16640, 128, 16512)); // ends at the ring's edge
}

test "chunks end at marks, the prompt's end or the cap" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 1024), chunkRows(0, 5000, 1024, &.{}));
    try t.expectEqual(@as(usize, 300), chunkRows(0, 5000, 1024, &.{ 300, 4000 }));
    try t.expectEqual(@as(usize, 1024), chunkRows(300, 5000, 1024, &.{ 300, 4000 }));
    try t.expectEqual(@as(usize, 952), chunkRows(3048, 5000, 1024, &.{ 300, 4000 }));
    try t.expectEqual(@as(usize, 5), chunkRows(4091, 4096, 128, &.{5000})); // a mark past the prompt cuts nothing
}

test "slot choice keeps live states: free unpinned slots first, then the least recently used" {
    const t = std.testing;
    var p: Pins = .{ .gpa = t.allocator };
    defer p.deinit();
    try t.expectEqual(@as(?u32, 0), p.pick(0b1111));
    const a = try p.add(0, 4096);
    const b = try p.add(1, 9000);
    try t.expectEqual(@as(?u32, 2), p.pick(0b1111));
    try t.expectEqual(@as(?u32, 0), p.pick(0b0011));
    p.touch(a);
    try t.expectEqual(@as(?u32, 1), p.pick(0b0011));
    var out: [4]*State = undefined;
    try t.expectEqual(@as(usize, 0), p.overwrite(1, 9000, &out).len); // a pass resumed at B's own mark keeps B
    const ended = p.overwrite(1, 0, &out); // a fresh pass on slot 1 ends B
    try t.expectEqual(@as(usize, 1), ended.len);
    try t.expect(!b.live and a.live);
    const sys = try p.add(2, 1500);
    try t.expectEqual(@as(?u32, 2), p.resumeSlot(0b1111, sys)); // nothing later on slot 2: resume in place
    _ = try p.add(2, 8000);
    try t.expectEqual(@as(?u32, 1), p.resumeSlot(0b1110, sys)); // later state on 2: an unpinned free slot instead
}

test "cache words round-trip the plan link" {
    const w = words(5, .copy, 7, 2, 3, 70000);
    try std.testing.expectEqual(Kind.copy, std.enums.fromInt(Kind, w[1]).?);
    try std.testing.expectEqual(@as(i64, 70000), w[5]);
}
