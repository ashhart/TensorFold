//! Flash Next EXL3 offload: the VRAM line table, its LRU scheduling, and the line addressing the device pool reads.

const std = @import("std");
const pol = @import("exl3_offload_policy.zig");

const Cache = pol.Cache;
const Decision = pol.Decision;
const Policy = pol.Policy;
const shared_id = pol.shared_id;
const indexOf = pol.indexOf;
const pushUnique = pol.pushUnique;
const hotterFirst = pol.hotterFirst;
const copyCount = pol.copyCount;

pub const Line = enum { empty, inflight, resident };

/// VRAM working set. Host RAM still holds every routed expert. `stamp` is the last reap or touch time. A higher stamp.
pub const Vram = struct {
    slots: []u16,
    line: []Line,
    stamp: []u32,
    /// Layer of `slots[i]`. Parallel to the other three slices.
    layer: []u8,

    pub fn init(slots: []u16, line: []Line, stamp: []u32, layer: []u8) Vram {
        std.debug.assert(slots.len == line.len and slots.len == stamp.len and slots.len == layer.len);
        @memset(slots, Cache.empty_id);
        @memset(line, .empty);
        @memset(stamp, 0);
        @memset(layer, 0);
        return .{ .slots = slots, .line = line, .stamp = stamp, .layer = layer };
    }
};

/// True once `id` has been reaped into VRAM. Id-only: the first matching line wins.
pub fn onDevice(v: Vram, id: u16) bool {
    const i = vramIndex(v, id) orelse return false;
    return v.line[i] == .resident;
}

/// True when this layer's copy of `id` is resident. Another layer's copy does not count.
pub fn residentAt(v: Vram, layer: u8, id: u16) bool {
    const i = vramIndexAt(v, layer, id) orelse return false;
    return v.line[i] == .resident;
}

/// A resident hit. Moves that line to `now` so LRU keeps it.
pub fn touch(v: *Vram, id: u16, now: u32) bool {
    const i = vramIndex(v.*, id) orelse return false;
    if (v.line[i] != .resident) return false;
    v.stamp[i] = now;
    return true;
}

/// Start a host-to-VRAM copy of `id`.
pub fn stage(v: *Vram, id: u16) error{ Busy, Shared }!usize {
    return stageKeeping(v, id, &.{});
}

/// Same as `stage`, except a resident whose id is in `keep` is not a victim.
pub fn stageKeeping(v: *Vram, id: u16, keep: []const u16) error{ Busy, Shared }!usize {
    if (id == shared_id or id == Cache.empty_id) return error.Shared;
    if (vramIndex(v.*, id)) |i| {
        if (v.line[i] == .inflight) return i;
        v.line[i] = .inflight;
        return i;
    }
    const slot = vramEmpty(v.*) orelse vramLruKeep(v.*, keep) orelse return error.Busy;
    v.slots[slot] = id;
    v.layer[slot] = 0;
    v.line[slot] = .inflight;
    v.stamp[slot] = 0;
    return slot;
}

/// One step of the VRAM cache.
pub fn schedule(
    v: *Vram,
    ids: []const u16,
    policy: Policy,
    now: u32,
    decisions: []Decision,
    scratch: []u16,
) error{ Length, Scratch }!void {
    if (decisions.len != ids.len) return error.Length;

    var n: usize = 0;
    for (ids, decisions) |id, *decision| {
        if (id == shared_id) {
            decision.* = .hit;
            continue;
        }
        if (onDevice(v.*, id)) {
            _ = touch(v, id, now);
            decision.* = .hit;
            continue;
        }
        decision.* = .cpu;
        if (vramIndex(v.*, id) != null) continue;
        try pushUnique(scratch, &n, id);
    }
    std.mem.sort(u16, scratch[0..n], ids, hotterFirst);

    const copies = copyCount(policy, n);
    var staged: usize = 0;
    for (scratch[0..n]) |id| {
        if (staged >= copies) break;
        _ = stageKeeping(v, id, ids) catch break;
        staged += 1;
        for (ids, decisions) |asked, *decision| {
            if (asked == id) decision.* = .copy;
        }
    }
}

/// The copy of `(layer, id)` has landed.
pub fn reapAt(v: *Vram, layer: u8, id: u16, now: u32) error{NotInflight}!void {
    const i = vramIndexAt(v.*, layer, id) orelse return error.NotInflight;
    if (v.line[i] != .inflight) return error.NotInflight;
    v.line[i] = .resident;
    v.stamp[i] = now;
}

/// Layer-0 wrapper. Does not promote another layer's inflight copy. Old tests are layer 0. Generate calls `reapAt`.
pub fn reap(v: *Vram, id: u16, now: u32) error{NotInflight}!void {
    return reapAt(v, 0, id, now);
}

/// First occupied slot with this id, on any layer.
pub fn vramIndex(v: Vram, id: u16) ?usize {
    if (id == Cache.empty_id) return null;
    for (v.slots, 0..) |slot, i| {
        if (slot == id and v.line[i] != .empty) return i;
    }
    return null;
}

/// Slot whose layer and id both match. The id-only search aliases two layers.
pub fn vramIndexAt(v: Vram, layer: u8, id: u16) ?usize {
    if (id == Cache.empty_id) return null;
    std.debug.assert(v.layer.len == v.slots.len and v.line.len == v.slots.len);
    for (v.slots, v.layer, 0..) |slot, lay, i| {
        if (slot == id and lay == layer and v.line[i] != .empty) return i;
    }
    return null;
}

pub fn touchAt(v: *Vram, layer: u8, id: u16, now: u32) bool {
    const i = vramIndexAt(v.*, layer, id) orelse return false;
    if (v.line[i] != .resident) return false;
    v.stamp[i] = now;
    return true;
}

pub fn vramEmpty(v: Vram) ?usize {
    for (v.line, 0..) |state, i| if (state == .empty) return i;
    return null;
}

/// Oldest resident stamp. Equal stamps take the higher slot index.
fn vramLru(v: Vram) ?usize {
    return vramLruKeep(v, &.{});
}

/// Oldest resident whose id is not in `keep`. Equal stamps take the higher slot index.
fn vramLruKeep(v: Vram, keep: []const u16) ?usize {
    var best: ?usize = null;
    for (v.line, 0..) |state, i| {
        if (state != .resident) continue;
        if (indexOf(keep, v.slots[i]) != null) continue;
        if (best) |b| {
            if (v.stamp[i] < v.stamp[b] or (v.stamp[i] == v.stamp[b] and i > b)) best = i;
        } else best = i;
    }
    return best;
}

/// Line index of `(layer, id)` in any state, or null. The device pool addresses its buffers by this index.
pub fn lineOf(v: Vram, layer: u8, id: u16) ?usize {
    return vramIndexAt(v, layer, id);
}

/// A host-to-device copy the policy asked for: expert `id` into pool line `line`.
pub const Copy = struct { id: u16, line: u32 };

/// Per pick, the pool line the hit kernel reads, or -1 when the kernel must skip the slot (a `.copy` or `.cpu`
pub fn lineIds(v: Vram, layer: u8, ids: []const u16, decisions: []const Decision, out: []i32) error{Length}!void {
    if (decisions.len != ids.len or out.len != ids.len) return error.Length;
    for (ids, decisions, out) |id, d, *o| {
        o.* = -1;
        if (d != .hit or id == shared_id) continue;
        const i = vramIndexAt(v, layer, id) orelse continue;
        if (v.line[i] != .resident) continue;
        o.* = @intCast(i);
    }
}

/// Unique `(id, line)` pairs the round staged: a `.copy` decision whose line is in flight.
pub fn stagedCopies(v: Vram, layer: u8, ids: []const u16, decisions: []const Decision, out: []Copy) error{Length}!usize {
    if (decisions.len != ids.len) return error.Length;
    var n: usize = 0;
    for (ids, decisions) |id, d| {
        if (d != .copy) continue;
        const i = vramIndexAt(v, layer, id) orelse continue;
        if (v.line[i] != .inflight) continue;
        var seen = false;
        for (out[0..n]) |c| {
            if (c.id == id) seen = true;
        }
        if (seen) continue;
        if (n >= out.len) return error.Length;
        out[n] = .{ .id = id, .line = @intCast(i) };
        n += 1;
    }
    return n;
}
