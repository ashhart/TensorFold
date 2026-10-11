//! A slot's kept prompt states: saved, restored, and made whole or stale before their slot writes below them.
const std = @import("std");
const mtl = @import("metal");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");
const pool_mod = @import("pool.zig");
const slots_mod = @import("slots.zig");
const Slots = slots_mod.Slots;
const Slot = slots_mod.Slot;

/// The prompt cache's leave for a kept state to take `extra` more bytes (Store.grow).
pub const Grow = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, saved: *anyopaque, extra: u64) bool,
};

/// Keep slot `i`'s prompt state at `at` tokens (its pass stands there) as snapshot `id` (0: the next); `home`: resident.
pub fn save(sl: *Slots, i: u32, at: u32, id: u32, home: bool) !*snapshot.Snap {
    const slot = try sl.slotAt(i);
    if (at == 0 or at != slot.s.pos or slot.rows != 0 or sl.snaps.contains(id)) return error.SnapshotOutOfStep;
    const n = snapshot.bytes(&sl.e.c, at, home);
    const buf = try sl.e.device.buffer(n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    const snap = try sl.gpa.create(snapshot.Snap);
    errdefer sl.gpa.destroy(snap);
    snap.* = .{ .id = if (id != 0) id else sl.next_snap, .at = at, .buf = buf, .bytes = n, .home = if (home) i else null };
    try copySnap(sl, slot, snap, true);
    try sl.snaps.put(sl.gpa, snap.id, snap);
    sl.next_snap = @max(sl.next_snap, snap.id + 1);
    return snap;
}

/// Slot `i`, just begun, takes snapshot `id`'s state: its prompt pass goes on from the snapshot's position.
pub fn restore(sl: *Slots, i: u32, id: u32) !void {
    const slot = try sl.slotAt(i);
    const snap = sl.snaps.get(id) orelse return error.SnapshotOutOfStep;
    if (slot.s.pos != 0 or snap.at >= slot.prompt_len or snap.stale) return error.SnapshotOutOfStep;
    if (snap.home != i) try overwrite(sl, i, 0); // another slot's prefixes, the pool's held blocks or a whole state's, copied over every one of this slot's
    try copySnap(sl, slot, snap, false);
}

/// Whether `snap` is still kept and usable (a resident state whose slot has not written below it).
pub fn usable(sl: *const Slots, snap: *const snapshot.Snap) bool {
    return sl.snaps.get(snap.id) == snap and !snap.stale;
}

/// Slot `i` writes its MLA rows from `at` on: its resident states past `at` are made whole if the cache has room, else go stale.
pub fn overwrite(sl: *Slots, i: u32, at: u32) !void {
    var it = sl.snaps.valueIterator();
    while (it.next()) |snap| if (snap.*.home == i and snap.*.at > at and !snap.*.stale) {
        if (!try spill(sl, snap.*)) snap.*.stale = true;
    };
}

/// A resident state made whole from its slot and charged to the prompt cache; never on a pair, whose rank 1 keeps its half resident.
fn spill(sl: *Slots, snap: *snapshot.Snap) !bool {
    const home = snap.home orelse return false;
    const g = sl.grow orelse return false;
    if (sl.e.ep != null) return false;
    const whole = snapshot.bytes(&sl.e.c, snap.at, false);
    if (!g.f(g.ctx, snap, whole - snap.bytes)) return false;
    const buf = try sl.e.device.buffer(whole, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    try sl.flush();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const x = sl.ctx(&sl.slots[home]);
    const b = sl.e.begin();
    snapshot.spill(&x, b.enc, .{ .buf = snap.buf }, .{ .buf = buf }, snap.at, &sl.slots[home].s);
    try sl.e.finish(b.cb, b.enc);
    snap.buf.deinit();
    snap.buf = buf;
    snap.bytes = whole;
    snap.home = null;
    std.log.info("glm: a kept state of {d} tokens copied its prefixes out of slot {d} before the slot wrote below it ({d} MiB more)", .{ snap.at, home, (whole - snapshot.bytes(&sl.e.c, snap.at, true)) >> 20 });
    return true;
}

/// A fixed prompt's state through slot 0, hashed: two builds agree only when their prompt passes give equal bits.
pub fn probe(sl: *Slots) !u64 {
    var toks: [4096 + 40 + 7]u32 = undefined; // a full chunk past the sparse index's reach, a middle one, a short one
    for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i * 7919 % 50000);
    const len: u32 = @intCast(@min(toks.len, pool_mod.capacity(sl) -| (st.max_rows + 2)));
    try sl.begin(0, toks[0..len], true);
    defer sl.release(0);
    var at: u32 = 0;
    for ([_]u32{ 4096, 4136, len }) |end| while (at < @min(end, len)) {
        const n = sl.chunkRows(at, @min(end, len));
        try sl.chunk(0, at, n);
        at += n;
    };
    const snap = try save(sl, 0, at, 0, false);
    defer drop(sl, snap.id);
    return std.hash.Wyhash.hash(0x70, snap.buf.contents()[0..snap.bytes]);
}

/// Snapshot `id` to its learned-state file, once the copy into it has finished.
pub fn writeSnap(sl: *Slots, id: u32, file: [:0]const u8) !void {
    const snap = sl.snaps.get(id) orelse return error.SnapshotOutOfStep;
    try sl.flush();
    try snapshot.writeFile(snap, file);
}

/// A learned state of `at` tokens read from `file` as snapshot `id`.
pub fn readSnap(sl: *Slots, id: u32, at: u32, file: [:0]const u8) !*snapshot.Snap {
    if (id == 0 or sl.snaps.contains(id)) return error.SnapshotOutOfStep;
    const n = snapshot.bytes(&sl.e.c, at, false);
    const buf = try sl.e.device.buffer(n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    const snap = try sl.gpa.create(snapshot.Snap);
    errdefer sl.gpa.destroy(snap);
    snap.* = .{ .id = id, .at = at, .buf = buf, .bytes = n };
    try snapshot.readFile(snap, file);
    try sl.snaps.put(sl.gpa, id, snap);
    sl.next_snap = @max(sl.next_snap, id + 1);
    return snap;
}

/// Forget snapshot `id`.
pub fn drop(sl: *Slots, id: u32) void {
    const kv = sl.snaps.fetchRemove(id) orelse return;
    freeSnap(sl.gpa, kv.value);
}

pub fn freeSnap(gpa: std.mem.Allocator, snap: *snapshot.Snap) void {
    snap.buf.deinit();
    gpa.destroy(snap);
}

fn copySnap(sl: *Slots, slot: *Slot, snap: *snapshot.Snap, into: bool) !void {
    try sl.flush();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const x = sl.ctx(slot);
    const b = sl.e.begin();
    var view: st.State = undefined;
    snapshot.copy(&x, b.enc, &slot.s, .{ .buf = snap.buf }, snap.at, into, pool_mod.prefixOf(sl, snap, &view));
    try sl.e.finish(b.cb, b.enc);
}

/// A new stream's free slot: `reuse`'s home when free, else the one holding the fewest usable resident states.
pub fn pick(sl: *const Slots, reuse: ?*const snapshot.Snap) ?u32 {
    if (reuse) |r| if (r.home) |h| if (h < sl.slots.len and !sl.slots[h].used and usable(sl, r)) return h;
    var best: ?u32 = null;
    var fewest: usize = std.math.maxInt(usize);
    for (sl.slots, 0..) |s, i| {
        if (s.used) continue;
        var held: usize = 0;
        var it = sl.snaps.valueIterator();
        while (it.next()) |snap| held += @intFromBool(snap.*.home == @as(u32, @intCast(i)) and !snap.*.stale);
        if (held < fewest) {
            fewest = held;
            best = @intCast(i);
        }
    }
    return best;
}
