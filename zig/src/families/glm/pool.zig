//! One Mac's streams share the engine's MLA cache, a contiguous region each; kept prefixes hold blocks until needed.
const std = @import("std");
const platform = @import("sys");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");
const Engine = @import("engine.zig").Engine;
const Slots = @import("slots.zig").Slots;
const Slot = @import("slots.zig").Slot;
const kept_mod = @import("kept.zig");

/// A stream's span of the pool, in blocks of `Pool.block` tokens.
pub const Region = struct { start: u32, blocks: u32 };

pub const Pool = struct {
    base: [st.max_mla]st.Mla, // the engine's caches, `cap` tokens
    blocks: u32,
    kpool: u32,

    pub const block: u32 = 256; // tokens; a multiple of the indexer's pooling width

    /// The tokens a region of `n` blocks holds: its last pooling block keeps the cache's extra pooled entry.
    fn tokens(p: *const Pool, n: u32) u32 {
        return n * block - p.kpool;
    }

    /// The blocks `tokens_` take.
    fn need(p: *const Pool, tokens_: u32) u32 {
        return (tokens_ + p.kpool + block - 1) / block;
    }

    /// The caches of a region: each part offset into the pool's.
    fn caches(p: *const Pool, c: anytype, r: Region, out: []st.Mla) void {
        const t0: usize = @as(usize, r.start) * block;
        for (out, p.base[0..out.len]) |*m, b| m.* = .{
            .keys = b.keys.at(t0 * c.kv_lora * 2),
            .ik = b.ik.at(t0 * c.i_dim * 2),
            .ig = b.ig.at(t0 * c.i_dim * 2),
            .pool = b.pool.at(t0 / p.kpool * c.i_dim * 2),
        };
    }
};

/// Whether the slots share one pool: on one Mac (a pair's ranks mirror slot commands), unless GLM_POOL=0.
pub fn on(e: *const Engine) bool {
    if (e.ep != null or e.followsPeer()) return false;
    const v = platform.c.getenv("GLM_POOL") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// The pool over the engine's caches, for slots made with KDA states only past the first.
pub fn setUp(sl: *Slots) void {
    const e = sl.e;
    sl.pool = .{ .base = e.s.mla, .blocks = e.s.cap / Pool.block, .kpool = e.c.kpool };
    std.log.info("glm: {d} streams share one {d}-token MLA pool", .{ sl.slots.len, capacity(sl) });
}

/// The most tokens one stream's caches hold: the pool's, or a slot's own window.
pub fn capacity(sl: *const Slots) u32 {
    const p = sl.pool orelse return sl.e.s.cap;
    return p.tokens(p.blocks);
}

/// The first free run of `n` blocks, or null; `held`: blocks kept prompt states hold count as taken.
fn findRun(sl: *const Slots, n: u32, held: bool) ?u32 {
    return findExcept(sl, n, held, null);
}

fn findExcept(sl: *const Slots, n: u32, held: bool, own: ?*const snapshot.Snap) ?u32 {
    const p = sl.pool.?;
    var at: u32 = 0;
    while (at + n <= p.blocks) {
        var next: ?u32 = null; // the end of the first region overlapping [at, at + n)
        for (sl.slots) |s| if (s.region) |r| if (r.start < at + n and at < r.start + r.blocks) {
            next = @max(next orelse 0, r.start + r.blocks);
        };
        if (held) {
            var it = sl.snaps.valueIterator();
            while (it.next()) |snap| if (snap.* != own) if (snap.*.held) |h| if (h[0] < at + n and at < h[0] + h[1]) {
                next = @max(next orelse 0, h[0] + h[1]);
            };
        }
        if (next) |end| at = end else return at;
    }
    return null;
}

/// Whether a stream of `tokens` can take its region now (true without a pool or if it never fits: prefill refuses it).
pub fn fits(sl: *const Slots, tokens: u32) bool {
    const p = sl.pool orelse return true;
    if (tokens > capacity(sl)) return true;
    return findRun(sl, p.need(tokens), false) != null;
}

/// Slot `i`'s region for `tokens` of caches before its stream begins (`from`: a held prefix it restores).
pub fn reserve(sl: *Slots, i: u32, tokens: u32, from: ?*const snapshot.Snap) !void {
    return reserveWith(sl, i, tokens, from, Mover{});
}

/// The real allocator with only the drained shared-memory copy supplied by the caller for host tests.
pub fn reserveWith(sl: *Slots, i: u32, tokens: u32, from: ?*const snapshot.Snap, mover: anytype) !void {
    const p = sl.pool orelse return;
    if (i >= sl.slots.len or sl.slots[i].used or sl.slots[i].region != null) return error.SlotOutOfStep;
    if (tokens > capacity(sl)) return error.ContextFull;
    const n = p.need(tokens);
    const start = try placeWith(sl, n, from, mover);
    const slot = &sl.slots[i];
    slot.region = .{ .start = start, .blocks = n };
    p.caches(&sl.e.c, slot.region.?, slot.s.mla[0..mlaCount(sl.e)]);
    slot.s.cap = p.tokens(n);
}

/// Where `n` blocks go: on `from`'s held prefix (restored in place) if nothing else is there, else the first free run.
fn placeWith(sl: *Slots, n: u32, from: ?*const snapshot.Snap, mover: anytype) !u32 {
    while (true) {
        if (from) |f| if (f.held) |h| if (clearAt(sl, h[0], n, f)) return h[0];
        if (findRun(sl, n, true)) |at| return at;
        if (evictHeld(sl, from, false, false)) continue;
        if (from) |f| if (!f.stale and f.held != null) if (findExcept(sl, n, true, f)) |at| {
            mover.move(sl, f, at) catch |err| {
                std.log.warn("glm: restoring {d} tokens failed ({s}); pool relocation aborted", .{ f.at, @errorName(err) });
                return err;
            };
            @constCast(f).held.?[0] = at;
            return at;
        };
        if (evictHeld(sl, from, false, true)) continue;
        if (!evictHeld(sl, from, true, true)) return error.PoolFull;
    }
}

const Mover = struct {
    pub fn move(_: Mover, sl: *Slots, snap: *const snapshot.Snap, start: u32) !void {
        try sl.flush();
        for (sl.ring_cb) |pending| if (pending) |cb| {
            cb.wait();
            if (cb.failure() != null) return error.PoolMoveNotDrained;
        };
        const e = sl.e;
        if (e.event.value() < e.ev and !e.event.wait(e.ev, 120_000)) return error.PoolMoveNotDrained;
        @import("pool_move.zig").prefix(&e.c, sl.pool.?.base[0..mlaCount(e)], snap.held.?[0], start, snap.at);
    }
};

/// Whether blocks [at, at + n) lie in the pool clear of every stream's region and of every held prefix but `own`.
fn clearAt(sl: *const Slots, at: u32, n: u32, own: *const snapshot.Snap) bool {
    if (at + n > sl.pool.?.blocks) return false;
    for (sl.slots) |s| if (s.region) |r| if (r.start < at + n and at < r.start + r.blocks) return false;
    var it = sl.snaps.valueIterator();
    while (it.next()) |snap| if (snap.* != own) if (snap.*.held) |h| if (h[0] < at + n and at < h[0] + h[1]) return false;
    return true;
}

/// Evict only an eligible held state; selected-prefix relocation precedes borrowed-state eviction.
fn evictHeld(sl: *Slots, spare: ?*const snapshot.Snap, allow_spare: bool, allow_borrowed: bool) bool {
    const snap = oldest(sl, spare, allow_spare, allow_borrowed) orelse return false;
    snap.held = null;
    snap.stale = true;
    kept_mod.wentStale(sl, snap);
    return true;
}

/// The oldest eligible held snap, with the selected prefix last when it is eligible.
fn oldest(sl: *Slots, spare: ?*const snapshot.Snap, allow_spare: bool, allow_borrowed: bool) ?*snapshot.Snap {
    var best: ?*snapshot.Snap = null;
    var it = sl.snaps.valueIterator();
    while (it.next()) |snap| if (snap.*.held != null and (allow_spare or snap.* != spare) and (allow_borrowed or !borrowed(sl, snap.*))) if (best == null or first(snap.*, best.?, spare)) {
        best = snap.*;
    };
    return best;
}

fn borrowed(sl: *Slots, snap: *snapshot.Snap) bool {
    const q = sl.borrowed orelse return false;
    return q.f(q.ctx, @ptrCast(snap));
}

/// Eviction order: every held prefix before `spare`, then the older id first.
fn first(a: *const snapshot.Snap, b: *const snapshot.Snap, spare: ?*const snapshot.Snap) bool {
    const a_last = if (spare) |x| a == x else false;
    const b_last = if (spare) |x| b == x else false;
    return if (a_last != b_last) b_last else a.id < b.id;
}

/// Slot `i`'s stream left: its region goes back, except the blocks under its resident prompt states' prefixes.
pub fn leave(sl: *Slots, i: u32) void {
    const slot = &sl.slots[i];
    const r = slot.region orelse return;
    slot.region = null; // the GPU's reads of it end before the next writer's
    const p = sl.pool.?;
    var top: u32 = 0;
    var it = sl.snaps.valueIterator();
    while (it.next()) |snap| if (snap.*.home == i and !snap.*.stale) {
        top = @max(top, snap.*.at);
    };
    if (top == 0) return;
    const kept = [2]u32{ r.start, @min(r.blocks, p.need(top)) };
    it = sl.snaps.valueIterator();
    while (it.next()) |snap| if (snap.*.home == i and !snap.*.stale) {
        snap.*.home = null;
        snap.*.held = kept;
    };
}

/// The caches a resident state's prefix sits in: its slot's, or (held in the pool) a view of its blocks in `view`.
pub fn prefixOf(sl: *Slots, snap: *const snapshot.Snap, view: *st.State) ?*st.State {
    if (snap.home) |h| return &sl.slots[h].s;
    const h = snap.held orelse return null;
    view.* = sl.slots[0].s;
    sl.pool.?.caches(&sl.e.c, .{ .start = h[0], .blocks = h[1] }, view.mla[0..mlaCount(sl.e)]);
    return view;
}

fn mlaCount(e: *const Engine) usize {
    return e.c.countKind(.mla) + @as(u32, if (e.c.mtp > 0) 1 else 0);
}

test "GLM expiry: a region of n blocks holds n blocks less the pooled entry" {
    const p: Pool = .{ .base = undefined, .blocks = 4096, .kpool = 4 };
    try std.testing.expectEqual(@as(u32, 1_048_572), p.tokens(p.blocks));
    for ([_]u32{ 1, 252, 253, 508, 509, 6376, 1_048_572 }) |t| {
        const n = p.need(t);
        try std.testing.expect(p.tokens(n) >= t);
        try std.testing.expect(n == 1 or p.tokens(n - 1) < t);
    }
}

/// The tests' borrow query: a state whose id the test names is still restoring.
fn ofBorrowed(ctx: *anyopaque, saved: *anyopaque) bool {
    const sl: *Slots = @ptrCast(@alignCast(ctx));
    const snap: *const snapshot.Snap = @ptrCast(@alignCast(saved));
    return sl.snaps.get(snap.id) == snap and snap.id == 1;
}

/// Slots over a pool of `blocks` blocks with no GPU behind them, for tests that read only regions and kept states.
fn testSlots(slots: []Slot, blocks: u32) Slots {
    return .{ .gpa = std.testing.allocator, .e = undefined, .slots = slots, .rows = undefined, .ids = undefined, .lasts = undefined, .picks = undefined, .ring = undefined, .pool = .{ .base = undefined, .blocks = blocks, .kpool = 4 } };
}

const MetadataMover = struct {
    pub fn move(_: MetadataMover, _: *Slots, _: *const snapshot.Snap, _: u32) !void {}
};

test "GLM expiry: a held prefix extends in place after other prefixes go first" {
    const gpa = std.testing.allocator;
    var slots: [2]Slot = undefined;
    for (&slots) |*x| x.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var sl = testSlots(&slots, 16);
    defer sl.snaps.deinit(gpa);
    var older: snapshot.Snap = .{ .id = 1, .at = 700, .buf = undefined, .bytes = 0, .held = .{ 9, 3 } };
    var own: snapshot.Snap = .{ .id = 2, .at = 1500, .buf = undefined, .bytes = 0, .held = .{ 2, 6 } };
    try sl.snaps.put(gpa, older.id, &older);
    try sl.snaps.put(gpa, own.id, &own);
    try std.testing.expectEqual(@as(u32, 2), try placeWith(&sl, 9, &own, MetadataMover{}));
    try std.testing.expect(older.stale and older.held == null);
    try std.testing.expect(!own.stale and own.held != null); // its prefix rows stay where they are: still valid
}

test "GLM expiry: a prefix relocates beside a running stream without overwriting it" {
    const gpa = std.testing.allocator;
    var slots: [2]Slot = undefined;
    for (&slots) |*x| x.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var sl = testSlots(&slots, 16);
    defer sl.snaps.deinit(gpa);
    slots[1].region = .{ .start = 12, .blocks = 4 }; // a running stream
    var own: snapshot.Snap = .{ .id = 2, .at = 1500, .buf = undefined, .bytes = 0, .held = .{ 2, 6 } };
    try sl.snaps.put(gpa, own.id, &own);
    try std.testing.expectEqual(@as(u32, 0), try placeWith(&sl, 12, &own, MetadataMover{}));
    try std.testing.expect(!own.stale and own.held.?[0] == 0);
    try std.testing.expectEqual(@as(u32, 12), slots[1].region.?.start);
    try std.testing.expectError(error.PoolFull, placeWith(&sl, 13, null, MetadataMover{}));
}

test "GLM expiry: with no restore, held prefixes give their blocks up oldest first" {
    const gpa = std.testing.allocator;
    var slots: [2]Slot = undefined;
    for (&slots) |*x| x.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var sl = testSlots(&slots, 16);
    defer sl.snaps.deinit(gpa);
    var older: snapshot.Snap = .{ .id = 1, .at = 700, .buf = undefined, .bytes = 0, .held = .{ 8, 4 } };
    var newer: snapshot.Snap = .{ .id = 2, .at = 700, .buf = undefined, .bytes = 0, .held = .{ 0, 4 } };
    try sl.snaps.put(gpa, older.id, &older);
    try sl.snaps.put(gpa, newer.id, &newer);
    try std.testing.expectEqual(@as(u32, 4), try placeWith(&sl, 10, null, MetadataMover{}));
    try std.testing.expect(older.stale and !newer.stale);
}

test "GLM expiry: a held state an outstanding lookup still restores gives its blocks up last" {
    const gpa = std.testing.allocator;
    var slots: [2]Slot = undefined;
    for (&slots) |*x| x.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var sl = testSlots(&slots, 16);
    defer sl.snaps.deinit(gpa);
    var borrowed_snap: snapshot.Snap = .{ .id = 1, .at = 700, .buf = undefined, .bytes = 0, .held = .{ 0, 4 } };
    var other: snapshot.Snap = .{ .id = 2, .at = 700, .buf = undefined, .bytes = 0, .held = .{ 8, 4 } };
    try sl.snaps.put(gpa, borrowed_snap.id, &borrowed_snap);
    try sl.snaps.put(gpa, other.id, &other);
    sl.borrowed = .{ .ctx = &sl, .f = ofBorrowed };
    defer sl.borrowed = null;
    try std.testing.expectEqual(@as(u32, 4), try placeWith(&sl, 10, null, MetadataMover{})); // [4, 14): the other prefix went first
    try std.testing.expect(other.stale and !borrowed_snap.stale);
    try std.testing.expectEqual(@as(u32, 4), try placeWith(&sl, 4, null, MetadataMover{})); // a run that fits past the borrowed prefix leaves it alone
    try std.testing.expect(!borrowed_snap.stale);
}

test "GLM expiry: a borrowed state's blocks still free when nothing else can serve the pass" {
    const gpa = std.testing.allocator;
    var slots: [1]Slot = undefined;
    for (&slots) |*x| x.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var sl = testSlots(&slots, 16);
    defer sl.snaps.deinit(gpa);
    var borrowed_snap: snapshot.Snap = .{ .id = 1, .at = 700, .buf = undefined, .bytes = 0, .held = .{ 0, 12 } };
    try sl.snaps.put(gpa, borrowed_snap.id, &borrowed_snap);
    sl.borrowed = .{ .ctx = &sl, .f = ofBorrowed };
    defer sl.borrowed = null;
    try std.testing.expectEqual(@as(u32, 0), try placeWith(&sl, 12, null, MetadataMover{})); // liveness: the borrowed state went last, not never
    try std.testing.expect(borrowed_snap.stale);
}

test "GLM expiry: selected prefix relocates before evicting another outstanding restore" {
    const gpa = std.testing.allocator;
    var slots: [1]Slot = undefined;
    for (&slots) |*x| x.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var sl = testSlots(&slots, 16);
    defer sl.snaps.deinit(gpa);
    var other: snapshot.Snap = .{ .id = 1, .at = 700, .buf = undefined, .bytes = 0, .held = .{ 0, 4 } };
    var own: snapshot.Snap = .{ .id = 2, .at = 700, .buf = undefined, .bytes = 0, .held = .{ 9, 3 } };
    try sl.snaps.put(gpa, other.id, &other);
    try sl.snaps.put(gpa, own.id, &own);
    sl.borrowed = .{ .ctx = &sl, .f = ofBorrowed };
    try std.testing.expectEqual(@as(u32, 4), try placeWith(&sl, 10, &own, MetadataMover{}));
    try std.testing.expectEqual(@as(u32, 4), own.held.?[0]);
    try std.testing.expect(!own.stale and !other.stale);
}
