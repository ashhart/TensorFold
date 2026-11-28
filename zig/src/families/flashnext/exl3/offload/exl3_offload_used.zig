//! Flash Next EXL3 offload: VRAM scheduling ordered by lifetime counts, per layer and per rank.

const std = @import("std");
const pol = @import("exl3_offload_policy.zig");
const vram_ = @import("exl3_offload_vram.zig");
const counts_ = @import("exl3_offload_counts.zig");

const shape = pol.shape;
const shared_id = pol.shared_id;
const Cache = pol.Cache;
const Decision = pol.Decision;
const Policy = pol.Policy;
const indexOf = pol.indexOf;
const pushUnique = pol.pushUnique;
const copyCount = pol.copyCount;
const Vram = vram_.Vram;
const vramEmpty = vram_.vramEmpty;
const vramIndexAt = vram_.vramIndexAt;
const residentAt = vram_.residentAt;
const touchAt = vram_.touchAt;
const Owned = counts_.Owned;
const inOwned = counts_.inOwned;
const countAt = counts_.countAt;
const UsedOrder = counts_.UsedOrder;
const usedFirst = counts_.usedFirst;

/// Lowest `counts[layer][id]`, then oldest stamp, then the higher slot index.
fn vramCold(
    v: Vram,
    counts: *const [shape.layers][shape.experts]u32,
    keep_layer: u8,
    keep_ids: []const u16,
) ?usize {
    std.debug.assert(v.layer.len == v.slots.len);
    var best: ?usize = null;
    for (v.line, 0..) |state, i| {
        if (state != .resident) continue;
        if (v.layer[i] == keep_layer and indexOf(keep_ids, v.slots[i]) != null) continue;
        if (best) |b| {
            const ci = countAt(counts, v.layer[i], v.slots[i]);
            const cb = countAt(counts, v.layer[b], v.slots[b]);
            const colder = ci < cb or
                (ci == cb and v.stamp[i] < v.stamp[b]) or
                (ci == cb and v.stamp[i] == v.stamp[b] and i > b);
            if (colder) best = i;
        } else best = i;
    }
    return best;
}

/// `stageKeeping`, but the victim is the coldest lifetime count.
pub fn stageUsed(v: *Vram, id: u16, keep: []const u16, counts: []const u32) error{ Busy, Shared, Length }!usize {
    if (counts.len != shape.experts) return error.Length;
    var table: [shape.layers][shape.experts]u32 = std.mem.zeroes([shape.layers][shape.experts]u32);
    @memcpy(table[0][0..], counts);
    return stageUsedAt(v, 0, id, keep, &table);
}

pub fn stageUsedAt(
    v: *Vram,
    layer: u8,
    id: u16,
    keep_ids: []const u16,
    counts: *const [shape.layers][shape.experts]u32,
) error{ Busy, Shared, Length }!usize {
    if (layer >= shape.layers) return error.Length;
    if (id == shared_id or id == Cache.empty_id) return error.Shared;
    if (vramIndexAt(v.*, layer, id)) |i| {
        if (v.line[i] == .inflight) return i;
        v.line[i] = .inflight;
        return i;
    }
    const slot = vramEmpty(v.*) orelse vramCold(v.*, counts, layer, keep_ids) orelse return error.Busy;
    v.slots[slot] = id;
    v.layer[slot] = layer;
    v.line[slot] = .inflight;
    v.stamp[slot] = 0;
    return slot;
}

/// `schedule`, ordered by lifetime counts from `observe`.
pub fn scheduleUsed(
    v: *Vram,
    ids: []const u16,
    counts: []const u32,
    policy: Policy,
    now: u32,
    decisions: []Decision,
    scratch: []u16,
    owned: Owned,
) error{ Length, Scratch }!void {
    if (decisions.len != ids.len) return error.Length;
    if (counts.len != shape.experts) return error.Length;
    var table: [shape.layers][shape.experts]u32 = std.mem.zeroes([shape.layers][shape.experts]u32);
    @memcpy(table[0][0..], counts);
    return scheduleUsedAt(v, 0, ids, &table, policy, now, decisions, scratch, owned);
}

/// Generate path. Misses sort by `counts[layer]`. A victim scores `counts[victim.layer][victim.id]`. Keep is `(layer).
pub fn scheduleUsedAt(
    v: *Vram,
    layer: u8,
    ids: []const u16,
    counts: *const [shape.layers][shape.experts]u32,
    policy: Policy,
    now: u32,
    decisions: []Decision,
    scratch: []u16,
    owned: Owned,
) error{ Length, Scratch }!void {
    if (decisions.len != ids.len) return error.Length;
    if (layer >= shape.layers) return error.Length;

    var n: usize = 0;
    for (ids, decisions) |id, *decision| {
        if (id == shared_id) {
            decision.* = .hit;
            continue;
        }
        if (residentAt(v.*, layer, id)) {
            _ = touchAt(v, layer, id, now);
            decision.* = .hit;
            continue;
        }
        decision.* = .cpu;
        if (!inOwned(owned, id)) continue;
        if (vramIndexAt(v.*, layer, id) != null) continue;
        try pushUnique(scratch, &n, id);
    }
    const row: []const u32 = &counts[layer];
    std.mem.sort(u16, scratch[0..n], UsedOrder{ .counts = row, .ids = ids }, usedFirst);

    const copies = copyCount(policy, n);
    var staged: usize = 0;
    for (scratch[0..n]) |id| {
        if (staged >= copies) break;
        _ = stageUsedAt(v, layer, id, ids, counts) catch |err| switch (err) {
            error.Length => return error.Length,
            error.Busy, error.Shared => break,
        };
        staged += 1;
        for (ids, decisions) |asked, *decision| {
            if (asked == id) decision.* = .copy;
        }
    }
}
