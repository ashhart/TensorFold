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
    snap.* = .{ .id = if (id != 0) id else sl.next_snap, .at = at, .buf = buf, .bytes = n, .home = if (home) i else null, .used = stamp(sl) };
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
    if (snap.home != i) overwrite(sl, i, 0); // another slot's prefixes, the pool's held blocks or a whole state's, copied over every one of this slot's
    try copySnap(sl, slot, snap, false);
    snap.used = stamp(sl);
}

/// The next use stamp: kept and restored states take one, so a slot's newest says when it last served.
fn stamp(sl: *Slots) u64 {
    sl.uses += 1;
    return sl.uses;
}

/// Whether `snap` is still kept and usable (a resident state whose slot has not written below it).
pub fn usable(sl: *const Slots, snap: *const snapshot.Snap) bool {
    return sl.snaps.get(snap.id) == snap and !snap.stale;
}

/// Slot `i` writes its MLA rows from `at` on: its resident states past `at` that were not copied out go stale.
pub fn overwrite(sl: *Slots, i: u32, at: u32) void {
    var it = sl.snaps.valueIterator();
    while (it.next()) |snap| if (snap.*.home == i and snap.*.at > at) {
        snap.*.stale = true;
    };
}

/// Before slot `i` writes from `at`: its resident states past `at` made whole where the budget has room, each id into `ids`.
pub fn copyOut(sl: *Slots, i: u32, at: u32, ids: *std.ArrayList(u32), gpa: std.mem.Allocator) !void {
    const g = sl.grow orelse return;
    var n: usize = 0;
    var it = sl.snaps.valueIterator();
    while (it.next()) |snap| n += @intFromBool(snap.*.home == i and snap.*.at > at and !snap.*.stale);
    try ids.ensureUnusedCapacity(gpa, n); // an id is recorded once its copy is made, so a pair's ranks never disagree
    it = sl.snaps.valueIterator();
    while (it.next()) |snap| if (snap.*.home == i and snap.*.at > at and !snap.*.stale) {
        if (!g.f(g.ctx, snap.*, snapshot.bytes(&sl.e.c, snap.*.at, false) - snap.*.bytes)) continue;
        try whole(sl, snap.*);
        ids.appendAssumeCapacity(snap.*.id);
    };
}

/// A pair's rank 1: the states rank 0 made whole before slot `i` writes, made whole here from this Mac's half.
pub fn copyIds(sl: *Slots, i: u32, ids: []const u32) !void {
    for (ids) |id| {
        const snap = sl.snaps.get(id) orelse return error.SnapshotOutOfStep;
        if (snap.home != i or snap.stale) return error.SnapshotOutOfStep;
        try whole(sl, snap);
    }
}

/// A resident state made whole: a buffer with its MLA prefixes beside its KDA states, filled from its slot before the slot writes.
fn whole(sl: *Slots, snap: *snapshot.Snap) !void {
    const home = snap.home orelse return error.SnapshotOutOfStep;
    const n = snapshot.bytes(&sl.e.c, snap.at, false);
    const buf = try sl.e.device.buffer(n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
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
    snap.bytes = n;
    snap.home = null;
    std.log.info("glm: a kept state of {d} tokens copied its prefixes out of slot {d} before the slot wrote below it ({d} MiB more)", .{ snap.at, home, (n - snapshot.bytes(&sl.e.c, snap.at, true)) >> 20 });
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
        try sl.chunk(0, at, n, null);
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

/// A new stream's free slot: `reuse`'s home when free, else the least recently used one (leastRecent).
pub fn pick(sl: *const Slots, reuse: ?*const snapshot.Snap) ?u32 {
    if (reuse) |r| if (r.home) |h| if (h < sl.slots.len and !sl.slots[h].used and usable(sl, r)) return h;
    return leastRecent(sl.slots, &sl.snaps);
}

/// The free slot holding no usable resident state, else the one whose newest such state was used longest ago; null: none free.
pub fn leastRecent(slots: []const Slot, snaps: *const std.AutoHashMapUnmanaged(u32, *snapshot.Snap)) ?u32 {
    var best: ?u32 = null;
    var oldest: u64 = std.math.maxInt(u64);
    for (slots, 0..) |s, i| {
        if (s.used) continue;
        var newest: u64 = 0; // holds none: before any stamp
        var it = snaps.valueIterator();
        while (it.next()) |snap| if (snap.*.home == @as(u32, @intCast(i)) and !snap.*.stale) {
            newest = @max(newest, snap.*.used);
        };
        if (newest < oldest) {
            oldest = newest;
            best = @intCast(i);
        }
    }
    return best;
}

/// Slots with no GPU behind them, for tests that read only slots' `used`, states' homes, stamps and staleness.
fn testSlots(slots: []Slot) Slots {
    return .{ .gpa = std.testing.allocator, .e = undefined, .slots = slots, .rows = undefined, .ids = undefined, .lasts = undefined, .picks = undefined, .ring = undefined };
}

test "a new stream takes its state's free home, else a slot holding none, else the one whose newest state was used longest ago" {
    const gpa = std.testing.allocator;
    var slots: [4]Slot = undefined;
    for (&slots) |*s| s.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var sl = testSlots(&slots);
    defer sl.snaps.deinit(gpa);
    var states: [5]snapshot.Snap = undefined;
    for (&states, [_]u32{ 0, 0, 1, 2, 3 }, [_]u64{ 3, 9, 5, 7, 1 }, 0..) |*x, home, used, k| {
        x.* = .{ .id = @intCast(k + 1), .at = 64, .buf = undefined, .bytes = 0, .home = home, .used = used };
        try sl.snaps.put(gpa, x.id, x);
    }
    states[4].stale = true; // slot 3's only state: its slot has written below it
    try std.testing.expectEqual(@as(?u32, 3), pick(&sl, null));
    try std.testing.expectEqual(@as(?u32, 0), pick(&sl, &states[1]));
    slots[3].used = true;
    try std.testing.expectEqual(@as(?u32, 1), pick(&sl, null)); // newest 5, before slot 2's 7 and slot 0's 9
    states[2].used = 10; // slot 1's state restored since
    try std.testing.expectEqual(@as(?u32, 2), pick(&sl, null));
    slots[0].used = true;
    try std.testing.expectEqual(@as(?u32, 2), pick(&sl, &states[1])); // its home is busy
    states[1].stale = true;
    slots[0].used = false;
    try std.testing.expectEqual(@as(?u32, 0), pick(&sl, &states[1])); // stale: no home rule, and slot 0's newest usable is 3
    for (&slots) |*s| s.used = true;
    try std.testing.expectEqual(@as(?u32, null), pick(&sl, null));
}

test "after a busy session an unrelated request leaves the newest conversation's slot alone (1.0.6's known issue)" {
    const gpa = std.testing.allocator;
    var slots: [3]Slot = undefined;
    for (&slots) |*s| s.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var sl = testSlots(&slots);
    defer sl.snaps.deinit(gpa);
    var states: [5]snapshot.Snap = undefined;
    for (&states, [_]u32{ 0, 0, 1, 1, 2 }, 1..) |*x, home, k| {
        x.* = .{ .id = @intCast(k), .at = 64, .buf = undefined, .bytes = 0, .home = home, .used = k };
        try sl.snaps.put(gpa, x.id, x);
    }
    // slot 2 holds only A1's state, the newest: the count-based pick took it (fewest held) and A1 went stale
    try std.testing.expectEqual(@as(?u32, 0), pick(&sl, null));
}

test "a slot copies out only what the budget grants, and rank 1 copies only ids resident in that slot" {
    const gpa = std.testing.allocator;
    var slots: [2]Slot = undefined;
    for (&slots) |*s| s.* = .{ .s = undefined, .held = undefined, .last = undefined };
    var e: @import("engine.zig").Engine = undefined;
    e.c = .{};
    var sl = testSlots(&slots);
    sl.e = &e;
    defer sl.snaps.deinit(gpa);
    var states: [4]snapshot.Snap = undefined;
    for (&states, [_]u32{ 0, 0, 0, 1 }, [_]u32{ 64, 128, 32, 64 }, 1..) |*x, home, at, k| {
        x.* = .{ .id = @intCast(k), .at = at, .buf = undefined, .bytes = 0, .home = home };
        try sl.snaps.put(gpa, x.id, x);
    }
    states[2].stale = true;
    const Refuse = struct {
        var asked: u32 = 0;
        fn f(_: *anyopaque, _: *anyopaque, _: u64) bool {
            asked += 1;
            return false;
        }
    };
    var unused: u8 = 0;
    sl.grow = .{ .ctx = &unused, .f = Refuse.f };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try copyOut(&sl, 0, 48, &ids, gpa); // ids 1 and 2 lie past 48 in slot 0; 3 is stale, 4 is slot 1's
    try std.testing.expectEqual(@as(u32, 2), Refuse.asked);
    try std.testing.expectEqual(@as(usize, 0), ids.items.len); // refused: nothing copied, nothing sent to rank 1
    overwrite(&sl, 0, 48);
    try std.testing.expect(states[0].stale and states[1].stale and !states[3].stale and states[3].home.? == 1);
    try std.testing.expectError(error.SnapshotOutOfStep, copyIds(&sl, 0, &.{9})); // not held
    try std.testing.expectError(error.SnapshotOutOfStep, copyIds(&sl, 0, &.{4})); // resident in slot 1
    try std.testing.expectError(error.SnapshotOutOfStep, copyIds(&sl, 0, &.{1})); // stale
    try copyIds(&sl, 0, &.{});
}
