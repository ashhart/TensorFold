//! Host expert-cache policy for Flash Next EXL3 offload (ported from exl3-flashnext-window-wip, unchanged rules).
//! Routed experts stay in host RAM. A VRAM line is a copy that is in flight until
//! `reap`, then resident. Eviction is LRU among resident lines; an in-flight line
//! is not reusable until it is reaped. `planRound` does not write either table.
//! Shared expert 512 is always a hit and never occupies a routed slot.
//! Generate addresses a line as `(layer, id)` through `scheduleUsedAt` and `reapAt`.
//! `schedule` and `onDevice` stay id-only and are not used on a layer-tagged table.
//! On upstream's `Experts` a miss is always copied (the device pool, `cuda_exl3_pool.zig`, has no CPU path): the round
//! planner (`exl3_round.zig`) stages every miss with `stageUsedAt` and keeps the copy budget for optional prefetch.

const std = @import("std");
/// Flash Next's routed geometry (was exl3_shape.zig): 48 decoder layers, 512 routed experts, the shared expert is id 512.
pub const shape = struct {
    pub const layers: usize = 48;
    pub const experts: usize = 512;
    pub const shared_expert: usize = 512;
};

pub const shared_id: u16 = @intCast(shape.shared_expert);

pub const Decision = enum { hit, copy, cpu };

pub const Policy = struct {
    slots: u16,
    expert_bytes: u64,
    /// Bytes the copy engine may move this round. copy_count = expert_bytes==0 ? 0 : copy_bytes / expert_bytes.
    copy_bytes: u64,
    swap_every: u16 = 4,
    swap_cap: u16 = 96,
};

pub const Cache = struct {
    slots: []u16, // expert id, or empty_id
    freq: []u32, // frequency of the resident id, parallel to slots
    pub const empty_id: u16 = 0xffff;
};

/// Clear every slot to `empty_id` and every frequency to 0. Lengths must match.
pub fn init(slots: []u16, freq: []u32) Cache {
    std.debug.assert(slots.len == freq.len);
    @memset(slots, Cache.empty_id);
    @memset(freq, 0);
    return .{ .slots = slots, .freq = freq };
}

/// Slot index of a resident expert, or null. `empty_id` is a vacant line, not an expert.
pub fn find(c: Cache, id: u16) ?usize {
    if (id == Cache.empty_id) return null;
    return indexOf(c.slots, id);
}

/// One decision per id. Does not write the slot table.
/// Unique misses sort by count descending, then id ascending. The first
/// `copy_bytes / expert_bytes` of those (0 when `expert_bytes` is 0) are `copy`;
/// the rest are `cpu`. Duplicate ids in `ids` share one decision.
pub fn planRound(
    c: Cache,
    ids: []const u16,
    policy: Policy,
    decisions: []Decision,
    scratch: []u16,
) error{ Length, Scratch }!void {
    if (decisions.len != ids.len) return error.Length;
    std.debug.assert(c.slots.len == c.freq.len);

    var n: usize = 0;
    for (ids) |id| {
        if (id == shared_id) continue;
        if (find(c, id) != null) continue;
        try pushUnique(scratch, &n, id);
    }
    std.mem.sort(u16, scratch[0..n], ids, hotterFirst);

    const copies = copyCount(policy, n);
    for (ids, decisions) |id, *decision| {
        if (id == shared_id or find(c, id) != null) {
            decision.* = .hit;
            continue;
        }
        const rank = indexOf(scratch[0..n], id).?;
        decision.* = if (rank < copies) .copy else .cpu;
    }
}

/// Install on swap rounds only (`round > 0` and `round % swap_every == 0`).
/// Residents named in `requested` are heated before victims are chosen.
/// Candidates are unique non-resident, non-shared ids, hottest first, at most
/// `swap_cap`. Empty lines are filled before any eviction. A line placed earlier
/// in this call is not a victim of a later candidate. Returns how many experts
/// were installed.
pub fn swap(
    c: *Cache,
    requested: []const u16,
    round: u32,
    policy: Policy,
    scratch: []u16,
) error{Scratch}!u16 {
    if (policy.swap_every == 0 or round == 0 or round % @as(u32, policy.swap_every) != 0) return 0;
    std.debug.assert(c.slots.len == c.freq.len);

    var n: usize = 0;
    for (requested) |id| {
        if (id == shared_id) continue;
        if (find(c.*, id) != null) continue;
        try pushUnique(scratch, &n, id);
    }
    std.mem.sort(u16, scratch[0..n], requested, hotterFirst);
    heatResidents(c, requested);

    const cap = @min(n, @as(usize, policy.swap_cap));
    var installed: usize = 0;
    for (scratch[0..cap]) |id| {
        const slot = indexOf(c.slots, Cache.empty_id) orelse victim(c.*, scratch[0..installed]) orelse break;
        c.slots[slot] = id;
        c.freq[slot] = appearances(requested, id);
        installed += 1;
    }
    return @intCast(installed);
}

/// Fill slots from a routed histogram of length 512. Hottest count, then lower id.
/// `freq[slot] = histogram[id]`. Id 512 is not a histogram entry and is not stored.
pub fn installProfile(c: *Cache, histogram: []const u32) void {
    std.debug.assert(histogram.len == shape.experts);
    std.debug.assert(c.slots.len == c.freq.len);
    @memset(c.slots, Cache.empty_id);
    @memset(c.freq, 0);

    const n = @min(c.slots.len, histogram.len);
    var filled: usize = 0;
    while (filled < n) : (filled += 1) {
        var best: u16 = 0;
        var best_n: u32 = 0;
        var any = false;
        for (histogram, 0..) |count, i| {
            const id: u16 = @intCast(i);
            if (indexOf(c.slots[0..filled], id) != null) continue;
            // Scan is id-ascending, so the first id that reaches the max count wins ties.
            if (!any or count > best_n) {
                best = id;
                best_n = count;
                any = true;
            }
        }
        if (!any) break;
        c.slots[filled] = best;
        c.freq[filled] = best_n;
    }
}

fn copyCount(policy: Policy, unique_misses: usize) usize {
    if (policy.expert_bytes == 0 or unique_misses == 0) return 0;
    const raw = policy.copy_bytes / policy.expert_bytes;
    if (raw >= unique_misses) return unique_misses;
    return @intCast(raw);
}

fn appearances(ids: []const u16, id: u16) u32 {
    var n: u32 = 0;
    for (ids) |x| {
        if (x == id) n +|= 1;
    }
    return n;
}

/// Count descending, id ascending. `ctx` is the multiset the counts are taken from.
fn hotterFirst(ctx: []const u16, a: u16, b: u16) bool {
    const ca = appearances(ctx, a);
    const cb = appearances(ctx, b);
    if (ca != cb) return ca > cb;
    return a < b;
}

fn indexOf(ids: []const u16, id: u16) ?usize {
    for (ids, 0..) |x, i| {
        if (x == id) return i;
    }
    return null;
}

fn pushUnique(dst: []u16, n: *usize, id: u16) error{Scratch}!void {
    if (indexOf(dst[0..n.*], id) != null) return;
    if (n.* >= dst.len) return error.Scratch;
    dst[n.*] = id;
    n.* += 1;
}

fn heatResidents(c: *Cache, requested: []const u16) void {
    for (c.slots, 0..) |id, i| {
        if (id == Cache.empty_id) continue;
        const n = appearances(requested, id);
        if (n == 0) continue;
        c.freq[i] = c.freq[i] +| n;
    }
}

/// Lowest frequency, skipping lines already placed in this call.
/// Equal frequency evicts the higher slot index.
fn victim(c: Cache, skip: []const u16) ?usize {
    var best: ?usize = null;
    for (c.slots, 0..) |id, i| {
        if (id == Cache.empty_id) continue;
        if (indexOf(skip, id) != null) continue;
        if (best) |b| {
            if (c.freq[i] < c.freq[b] or (c.freq[i] == c.freq[b] and i > b)) best = i;
        } else best = i;
    }
    return best;
}

pub const Line = enum { empty, inflight, resident };

/// VRAM working set. Host RAM still holds every routed expert.
/// `stamp` is the last reap or touch time. A higher stamp is newer.
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

/// True once `id` has been reaped into VRAM.
/// Id-only: the first matching line wins. Generate calls `residentAt`.
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

/// Start a host-to-VRAM copy of `id`. An empty line is used first.
/// Otherwise the resident line with the oldest stamp is replaced.
/// An in-flight line is left alone. `error.Busy` means every line is still in flight.
pub fn stage(v: *Vram, id: u16) error{ Busy, Shared }!usize {
    return stageKeeping(v, id, &.{});
}

/// Same as `stage`, except a resident whose id is in `keep` is not a victim.
/// `error.Busy` means every line is in flight or held by `keep`.
/// Id-only. A newly assigned line is layer 0. Do not call this on a multi-layer table.
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

/// One step of the VRAM cache. Does not reap.
/// A resident id is touched and decided `hit`. An in-flight id is `cpu` until reap.
/// Unique misses, hottest first, are staged until the copy budget runs out or every
/// free line is gone. A staged miss is `copy` and stays off-device until `reap`.
/// Residents named in `ids` are not eviction victims. Shared id 512 is `hit`.
/// A zero `expert_bytes` stages nothing: every miss stays `cpu`.
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

/// The copy of `(layer, id)` has landed. Another layer's resident or inflight
/// line is not this copy. Generate calls this, not `reap`.
pub fn reapAt(v: *Vram, layer: u8, id: u16, now: u32) error{NotInflight}!void {
    const i = vramIndexAt(v.*, layer, id) orelse return error.NotInflight;
    if (v.line[i] != .inflight) return error.NotInflight;
    v.line[i] = .resident;
    v.stamp[i] = now;
}

/// Layer-0 wrapper. Does not promote another layer's inflight copy.
/// Old tests are layer 0. Generate calls `reapAt`.
pub fn reap(v: *Vram, id: u16, now: u32) error{NotInflight}!void {
    return reapAt(v, 0, id, now);
}

/// First occupied slot with this id, on any layer. That alias is what
/// `schedule` and `onDevice` still call. Generate uses `vramIndexAt`.
fn vramIndex(v: Vram, id: u16) ?usize {
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

fn vramEmpty(v: Vram) ?usize {
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

fn countOf(counts: []const u32, id: u16) u32 {
    if (id >= counts.len) return 0;
    return counts[id];
}

pub fn countAt(counts: *const [shape.layers][shape.experts]u32, layer: u8, id: u16) u32 {
    if (layer >= shape.layers or id >= shape.experts) return 0;
    return counts[layer][id];
}

/// Lowest `counts[layer][id]`, then oldest stamp, then the higher slot index.
/// A keep entry is `(keep_layer, id)`. The same id on another layer is not protected.
/// Equal counts are the stamp LRU, so a zeroed table matches `vramLruKeep`.
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

const UsedOrder = struct {
    counts: []const u32,
    ids: []const u16,
};

/// Lifetime count descending, then this round's appearances, then lower id.
fn usedFirst(ctx: UsedOrder, a: u16, b: u16) bool {
    const ca = countOf(ctx.counts, a);
    const cb = countOf(ctx.counts, b);
    if (ca != cb) return ca > cb;
    const aa = appearances(ctx.ids, a);
    const ab = appearances(ctx.ids, b);
    if (aa != ab) return aa > ab;
    return a < b;
}

/// Add one to each routed id in `ids`. Shared id 512 is not a histogram entry.
/// `counts` is one cell per routed expert (`shape.experts`).
pub fn observe(counts: []u32, ids: []const u16) error{Length}!void {
    if (counts.len != shape.experts) return error.Length;
    for (ids) |id| {
        if (id >= shape.experts) continue;
        counts[id] +|= 1;
    }
}

/// Expert ids this rank stores. Rank 0 is `0..256`, rank 1 is `256..512`.
pub const Owned = struct { lo: u16, hi: u16 };

pub fn allOwned() Owned {
    return .{ .lo = 0, .hi = @intCast(shape.experts) };
}

fn inOwned(owned: Owned, id: u16) bool {
    return id >= owned.lo and id < owned.hi;
}

/// Fill empty VRAM lines with the hottest counted ids. No eviction.
/// Ties take the lower id. Count 0 and ids already in the table are skipped.
/// Staged lines stay in flight until `reap`.
pub fn seedHottest(v: *Vram, counts: []const u32, limit: u16) error{Length}!u16 {
    return seedSpan(v, counts, limit, allOwned());
}

fn seedSpan(v: *Vram, counts: []const u32, limit: u16, owned: Owned) error{Length}!u16 {
    if (counts.len != shape.experts) return error.Length;
    var seeded: u16 = 0;
    while (seeded < limit) {
        if (vramEmpty(v.*) == null) break;
        var best: ?u16 = null;
        var best_n: u32 = 0;
        var id: u16 = owned.lo;
        while (id < owned.hi) : (id += 1) {
            const count = counts[id];
            if (count == 0) continue;
            if (vramIndex(v.*, id) != null) continue;
            if (best == null or count > best_n) {
                best = id;
                best_n = count;
            }
        }
        if (best == null) break;
        _ = stage(v, best.?) catch break;
        seeded += 1;
    }
    return seeded;
}

/// Use every empty line. Hottest owned ids go in first. Lines that are still
/// empty take the remaining owned ids, low id first, so the card is not left
/// idle while the histogram warms up. No eviction. The other rank's ids are skipped.
pub fn fillOwned(v: *Vram, counts: []const u32, owned: Owned) error{Length}!u16 {
    if (owned.hi > shape.experts or owned.lo > owned.hi) return error.Length;
    var n: u16 = try seedSpan(v, counts, std.math.maxInt(u16), owned);
    var id: u16 = owned.lo;
    while (id < owned.hi) : (id += 1) {
        if (vramEmpty(v.*) == null) break;
        if (vramIndex(v.*, id) != null) continue;
        _ = stage(v, id) catch break;
        n += 1;
    }
    return n;
}

fn quotaAt(lines: u16, layer: usize) u16 {
    const n_layers: u16 = @intCast(shape.layers);
    const base = lines / n_layers;
    if (layer < @as(usize, lines % n_layers)) return base + 1;
    return base;
}

fn placeOwned(v: *Vram, layer: u8, id: u16) bool {
    // Shared 512 is not a routed line. Callers only pass an owned id.
    std.debug.assert(id != shared_id and id != Cache.empty_id);
    const slot = vramEmpty(v.*) orelse return false;
    v.slots[slot] = id;
    v.layer[slot] = layer;
    v.line[slot] = .inflight;
    v.stamp[slot] = 0;
    return true;
}

/// Hottest owned `(layer, id)` that still has stripe room, one line at a time.
fn placeHottest(
    v: *Vram,
    lines: u16,
    counts: []const [shape.experts]u32,
    owned: Owned,
    used: *[shape.layers]u16,
) u16 {
    var placed: u16 = 0;
    while (placed < lines and vramEmpty(v.*) != null) {
        var found = false;
        var best_layer: u8 = 0;
        var best_id: u16 = 0;
        var best_c: u32 = 0;
        for (counts, 0..) |row, layer_i| {
            if (used[layer_i] >= quotaAt(lines, layer_i)) continue;
            var id = owned.lo;
            while (id < owned.hi) : (id += 1) {
                const c = row[id];
                if (c == 0) continue;
                if (vramIndexAt(v.*, @intCast(layer_i), id) != null) continue;
                const layer: u8 = @intCast(layer_i);
                const better = !found or c > best_c or
                    (c == best_c and id < best_id) or
                    (c == best_c and id == best_id and layer < best_layer);
                if (!better) continue;
                found = true;
                best_layer = layer;
                best_id = id;
                best_c = c;
            }
        }
        if (!found) break;
        if (!placeOwned(v, best_layer, best_id)) break;
        used[best_layer] += 1;
        placed += 1;
    }
    return placed;
}

/// Id, then layer. An uncapped layer-major walk would park every line on layer 0.
fn fillStripe(v: *Vram, lines: u16, owned: Owned, used: *[shape.layers]u16, placed: u16) u16 {
    var n = placed;
    var id = owned.lo;
    while (id < owned.hi and n < lines) : (id += 1) {
        var layer: usize = 0;
        while (layer < shape.layers and n < lines) : (layer += 1) {
            if (used[layer] >= quotaAt(lines, layer)) continue;
            if (vramIndexAt(v.*, @intCast(layer), id) != null) continue;
            if (!placeOwned(v, @intCast(layer), id)) return n;
            used[layer] += 1;
            n += 1;
        }
    }
    return n;
}

/// Stage up to `lines` owned experts. `lines` is the caller's live count.
/// Each layer gets `lines / shape.layers`, and the remainder goes one each to the low layers.
/// A zero histogram fills the lowest owned id on each of those layers.
/// A warm histogram takes the hottest owned pair that still fits that stripe, then
/// fills any line still empty. The other rank's ids are not staged.
pub fn fillOwnedLayers(v: *Vram, lines: u16, counts: []const [shape.experts]u32, owned: Owned) error{Length}!u16 {
    if (counts.len != shape.layers) return error.Length;
    if (owned.hi > shape.experts or owned.lo > owned.hi) return error.Length;
    std.debug.assert(v.layer.len == v.slots.len and v.line.len == v.slots.len);
    var used: [shape.layers]u16 = @splat(0);
    const hot = placeHottest(v, lines, counts, owned, &used);
    return fillStripe(v, lines, owned, &used, hot);
}

/// `stageKeeping`, but the victim is the coldest lifetime count.
/// Layer 0 only: a flat histogram does not score another layer's copy.
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
/// A high count keeps its line even when its stamp is older.
/// Residents named in `ids` are not victims. `schedule` itself is unchanged.
/// This is the layer-0 wrapper. It is not the multi-layer path: the flat
/// histogram is copied into layer 0, so another layer's copy scores 0.
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

/// Generate path. Misses sort by `counts[layer]`. A victim scores
/// `counts[victim.layer][victim.id]`. Keep is `(layer, id)` for this call's ids.
/// An id outside `owned` stays `.cpu` and is not staged. Shared 512 is a hit
/// and does not occupy a line. A landed copy is promoted with `reapAt`.
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

/// Line index of `(layer, id)` in any state, or null. The device pool addresses its buffers by this index.
pub fn lineOf(v: Vram, layer: u8, id: u16) ?usize {
    return vramIndexAt(v, layer, id);
}

/// A host-to-device copy the policy asked for: expert `id` into pool line `line`.
pub const Copy = struct { id: u16, line: u32 };

/// Per pick, the pool line the hit kernel reads, or -1 when the kernel must skip the slot
/// (a `.copy` or `.cpu` decision, the shared expert, or a line that is not resident).
/// The kernel ignores a negative id, so the host writes that slot's row into `y` itself.
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
/// Returns how many were written. `out` shorter than the unique count is `error.Length`.
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

fn benchPolicy(slot_n: u16, expert_bytes: u64, copy_bytes: u64) Policy {
    return .{
        .slots = slot_n,
        .expert_bytes = expert_bytes,
        .copy_bytes = copy_bytes,
    };
}

test "a staged copy is not on device until reap, and LRU drops the older resident" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);

    try std.testing.expectEqual(@as(usize, 0), try stage(&vram, 4));
    try std.testing.expectEqual(@as(usize, 1), try stage(&vram, 7));
    try std.testing.expect(!onDevice(vram, 4));
    try std.testing.expectError(error.NotInflight, reap(&vram, 9, 1));
    try reap(&vram, 4, 1);
    try reap(&vram, 7, 2);
    try std.testing.expect(onDevice(vram, 4));
    try std.testing.expect(touch(&vram, 7, 5));
    try std.testing.expect(onDevice(vram, 4));

    // 4 was reaped at 1 and not touched again. 7 was touched at 5. 4 is the victim.
    try std.testing.expectEqual(@as(usize, 0), try stage(&vram, 8));
    try std.testing.expect(!onDevice(vram, 8));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expectEqual(@as(u16, 7), slots[1]);
    try reap(&vram, 8, 6);
    try std.testing.expect(onDevice(vram, 8));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 4));
}

test "an in-flight line is not reused, and a full in-flight set is busy" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    _ = try stage(&vram, 1);
    _ = try stage(&vram, 2);
    try std.testing.expectError(error.Busy, stage(&vram, 3));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u16, 2), slots[1]);
    try std.testing.expectError(error.Shared, stage(&vram, shared_id));
}

test "schedule touches residents, stages the hottest miss, and will not evict a hit" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var decisions: [4]Decision = undefined;
    var scratch: [4]u16 = undefined;
    const policy = benchPolicy(2, 100, 100);

    const first = [_]u16{ 7, shared_id, 4, 7 };
    try schedule(&vram, &first, policy, 3, &decisions, &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[3]);
    try std.testing.expectEqual(Decision.cpu, decisions[2]);
    try std.testing.expect(!onDevice(vram, 7));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);
    try std.testing.expectEqual(Line.empty, line[1]);

    try reap(&vram, 7, 4);
    const held = [_]u16{7};
    try schedule(&vram, &held, policy, 9, decisions[0..1], &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(@as(u32, 9), stamp[0]);

    // Both lines are residents this step asked for, so the miss cannot take either.
    _ = try stage(&vram, 3);
    try reap(&vram, 3, 10);
    const protected = [_]u16{ 7, 3, 8 };
    try schedule(&vram, &protected, benchPolicy(2, 100, 10_000), 11, decisions[0..3], &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.cpu, decisions[2]);
    try std.testing.expect(onDevice(vram, 7));
    try std.testing.expect(onDevice(vram, 3));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 8));
}

test "a zero copy budget stages nothing, and an older resident is the one that moves" {
    var slots = [_]u16{ 4, 7 };
    var line = [_]Line{ .resident, .resident };
    var stamp = [_]u32{ 1, 5 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 4, 7 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 5 };
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;

    const miss = [_]u16{9};
    try schedule(&vram, &miss, benchPolicy(2, 0, 10_000), 6, &decisions, &scratch);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(@as(u16, 4), slots[0]);
    try std.testing.expectEqual(Line.resident, line[0]);

    try schedule(&vram, &miss, benchPolicy(2, 100, 100), 6, &decisions, &scratch);
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(onDevice(vram, 7));
    try std.testing.expect(!onDevice(vram, 9));
}

test "resident ids are hits and planRound does not change slots" {
    var slots = [_]u16{ 4, 1, 9 };
    var freq = [_]u32{ 3, 8, 1 };
    const cache = init(&slots, &freq);
    slots = .{ 4, 1, 9 };
    freq = .{ 3, 8, 1 };
    const before_slots = slots;
    const before_freq = freq;

    const ids = [_]u16{ 4, shared_id, 1, 9, 4 };
    var decisions: [5]Decision = undefined;
    try planRound(cache, &ids, benchPolicy(3, 100, 10_000), &decisions, &[_]u16{});

    for (decisions) |d| try std.testing.expectEqual(Decision.hit, d);
    try std.testing.expectEqual(before_slots, slots);
    try std.testing.expectEqual(before_freq, freq);
    try std.testing.expectEqual(@as(?usize, 1), find(cache, 1));
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));
}

test "misses past the copy budget are cpu and slots stay put" {
    var slots = [_]u16{42};
    var freq = [_]u32{6};
    const cache = init(&slots, &freq);
    slots[0] = 42;
    freq[0] = 6;
    const before_slots = slots;
    const before_freq = freq;

    // Counts: 5 x4, 1 x3, 8 x1, 3 x1. Budget 250/100 = 2, so 5 and 1 are copied.
    const ids = [_]u16{ 8, 5, 1, 5, 3, 5, 1, 42, 5, 1 };
    var decisions: [ids.len]Decision = undefined;
    var scratch: [8]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(1, 100, 250), &decisions, &scratch);

    const want = [_]Decision{ .cpu, .copy, .copy, .copy, .cpu, .copy, .copy, .hit, .copy, .copy };
    try std.testing.expectEqualSlices(Decision, &want, &decisions);
    try std.testing.expectEqual(before_slots, slots);
    try std.testing.expectEqual(before_freq, freq);
}

test "duplicate miss ids share one decision" {
    var slots: [1]u16 = undefined;
    var freq: [1]u32 = undefined;
    const cache = init(&slots, &freq);
    const ids = [_]u16{ 6, 2, 6 };
    var decisions: [3]Decision = undefined;
    var scratch: [4]u16 = undefined;
    // One copy. 6 appears twice, 2 once, so both 6s are copy and 2 is cpu.
    try planRound(cache, &ids, benchPolicy(1, 100, 100), &decisions, &scratch);
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[2]);
}

test "equal miss counts break ties toward the lower id" {
    var slots: [0]u16 = .{};
    var freq: [0]u32 = .{};
    const cache = init(&slots, &freq);
    // 10 x5, 6 x4, 2 x4. One copy would be 10; two copies are 10 and 2, not 6.
    const ids = [_]u16{ 6, 10, 2, 6, 10, 2, 10, 6, 2, 10, 6, 10, 2 };
    var decisions: [ids.len]Decision = undefined;
    var scratch: [4]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(0, 50, 100), &decisions, &scratch);
    for (ids, decisions) |id, d| {
        const want: Decision = if (id == 10 or id == 2) .copy else .cpu;
        try std.testing.expectEqual(want, d);
    }
}

test "shared id is a hit and is never stored" {
    var slots: [4]u16 = @splat(Cache.empty_id);
    var freq: [4]u32 = @splat(0);
    var cache = init(&slots, &freq);
    slots[0] = 7;
    freq[0] = 2;

    const ids = [_]u16{ shared_id, 4, shared_id };
    var decisions: [3]Decision = undefined;
    var scratch: [4]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(4, 100, 0), &decisions, &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
    try std.testing.expectEqual(Decision.hit, decisions[2]);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);

    const requested = [_]u16{ shared_id, 4, shared_id, 9 };
    const installed = try swap(&cache, &requested, 4, benchPolicy(4, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 2), installed);
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));
    for (slots) |id| try std.testing.expect(id != shared_id);

    var hist: [shape.experts]u32 = @splat(0);
    hist[4] = 3;
    installProfile(&cache, &hist);
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));
    for (slots) |id| try std.testing.expect(id != shared_id);
}

test "round 2 installs nothing and round 4 installs min of cap and missing" {
    var slots: [8]u16 = @splat(Cache.empty_id);
    var freq: [8]u32 = @splat(0);
    var cache = init(&slots, &freq);
    const requested = [_]u16{ 3, 1, 3, 8 };
    var scratch: [8]u16 = undefined;
    var policy = benchPolicy(8, 1, 0);
    policy.swap_every = 4;

    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &requested, 0, policy, &scratch));
    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &requested, 2, policy, &scratch));
    try std.testing.expectEqual(@as(u16, Cache.empty_id), slots[0]);
    try std.testing.expectEqual(@as(u32, 0), freq[0]);

    // Three unique misses, under the cap.
    const n = try swap(&cache, &requested, 4, policy, &scratch);
    try std.testing.expectEqual(@as(u16, 3), n);
    try std.testing.expectEqual(@as(u16, 3), slots[0]);
    try std.testing.expectEqual(@as(u32, 2), freq[0]);
    try std.testing.expectEqual(@as(u16, 1), slots[1]);
    try std.testing.expectEqual(@as(u16, 8), slots[2]);
    try std.testing.expectEqual(@as(u16, Cache.empty_id), slots[3]);
}

test "a cold line is replaced by the missing id" {
    var slots = [_]u16{41};
    var freq = [_]u32{0};
    var cache = init(&slots, &freq);
    slots[0] = 41;
    freq[0] = 0;
    var scratch: [2]u16 = undefined;
    const requested = [_]u16{9};
    const n = try swap(&cache, &requested, 4, benchPolicy(1, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(@as(u32, 1), freq[0]);
    try std.testing.expectEqual(@as(?usize, null), find(cache, 41));
}

test "eviction ties take the later slot" {
    var slots = [_]u16{ 30, 10, 20 };
    var freq = [_]u32{ 4, 4, 4 };
    var cache = init(&slots, &freq);
    slots = .{ 30, 10, 20 };
    freq = .{ 4, 4, 4 };
    var scratch: [2]u16 = undefined;
    const n = try swap(&cache, &[_]u16{7}, 4, benchPolicy(3, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    // Higher id would have dropped 30. Higher slot index drops 20.
    try std.testing.expectEqual([_]u16{ 30, 10, 7 }, slots);
    try std.testing.expectEqual(@as(u32, 1), freq[2]);
}

test "a resident named this round is heated before victims are chosen" {
    var slots = [_]u16{ 2, 1 };
    var freq = [_]u32{ 0, 0 };
    var cache = init(&slots, &freq);
    slots = .{ 2, 1 };
    freq = .{ 0, 0 };
    var scratch: [2]u16 = undefined;
    // Without the heat, both freqs stay 0 and the later slot (id 1) would be the victim.
    const n = try swap(&cache, &[_]u16{ 1, 99 }, 4, benchPolicy(2, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    try std.testing.expectEqual([_]u16{ 99, 1 }, slots);
    try std.testing.expectEqual(@as(u32, 1), freq[0]);
    try std.testing.expectEqual(@as(u32, 1), freq[1]);
}

test "resident frequency adds this round's request count" {
    var slots = [_]u16{ 5, Cache.empty_id };
    var freq = [_]u32{ 4, 0 };
    var cache = init(&slots, &freq);
    slots = .{ 5, Cache.empty_id };
    freq = .{ 4, 0 };
    var scratch: [2]u16 = undefined;
    const requested = [_]u16{ 5, 5, 5, 4 };
    const n = try swap(&cache, &requested, 4, benchPolicy(2, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 1), n);
    try std.testing.expectEqual(@as(u16, 5), slots[0]);
    try std.testing.expectEqual(@as(u32, 7), freq[0]);
    try std.testing.expectEqual(@as(u16, 4), slots[1]);
    try std.testing.expectEqual(@as(u32, 1), freq[1]);
}

test "empty lines are filled before a cold resident is evicted" {
    var slots = [_]u16{ 30, Cache.empty_id, 40 };
    var freq = [_]u32{ 5, 0, 1 };
    var cache = init(&slots, &freq);
    slots = .{ 30, Cache.empty_id, 40 };
    freq = .{ 5, 0, 1 };
    var scratch: [4]u16 = undefined;
    const n = try swap(&cache, &[_]u16{ 7, 8 }, 4, benchPolicy(3, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 2), n);
    try std.testing.expectEqual([_]u16{ 30, 7, 8 }, slots);
    try std.testing.expectEqual(@as(u32, 5), freq[0]);
}

test "a candidate placed this call is not evicted by a later one" {
    var slots = [_]u16{ 1, 2, 3, 4 };
    var freq = [_]u32{ 1, 50, 2, 40 };
    var cache = init(&slots, &freq);
    slots = .{ 1, 2, 3, 4 };
    freq = .{ 1, 50, 2, 40 };
    var scratch: [4]u16 = undefined;
    const n = try swap(&cache, &[_]u16{ 10, 11, 12 }, 4, benchPolicy(4, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 3), n);
    try std.testing.expectEqual([_]u16{ 10, 2, 11, 12 }, slots);
    try std.testing.expectEqual(@as(u32, 50), freq[1]);
}

test "installProfile keeps the hottest routed ids" {
    var slots: [3]u16 = @splat(shared_id);
    var freq = [_]u32{ 9, 9, 9 };
    var cache = init(&slots, &freq);
    var hist: [shape.experts]u32 = @splat(0);
    hist[10] = 1;
    hist[4] = 7;
    hist[8] = 7;
    hist[2] = 3;
    installProfile(&cache, &hist);
    try std.testing.expectEqual([_]u16{ 4, 8, 2 }, slots);
    try std.testing.expectEqual([_]u32{ 7, 7, 3 }, freq);
    try std.testing.expectEqual(@as(?usize, null), find(cache, shared_id));

    var wide_slots: [513]u16 = undefined;
    var wide_freq: [513]u32 = undefined;
    var wide = init(&wide_slots, &wide_freq);
    var flat: [shape.experts]u32 = @splat(0);
    flat[0] = 1;
    installProfile(&wide, &flat);
    for (wide_slots[0..shape.experts], 0..) |id, i| {
        try std.testing.expectEqual(@as(u16, @intCast(i)), id);
    }
    try std.testing.expectEqual(Cache.empty_id, wide_slots[512]);
    try std.testing.expectEqual(@as(u32, 1), wide_freq[0]);
    try std.testing.expectEqual(@as(u32, 0), wide_freq[1]);
    try std.testing.expectEqual(@as(?usize, null), find(wide, shared_id));
}

test "swap installs at most the cap even when slots and misses are larger" {
    var slots: [200]u16 = @splat(Cache.empty_id);
    var freq: [200]u32 = @splat(0);
    var cache = init(&slots, &freq);
    var requested: [200]u16 = undefined;
    for (&requested, 0..) |*id, i| id.* = @intCast(i);
    var scratch: [200]u16 = undefined;
    const n = try swap(&cache, &requested, 4, benchPolicy(200, 1, 0), &scratch);
    try std.testing.expectEqual(@as(u16, 96), n);
    for (slots[0..96], 0..) |id, i| try std.testing.expectEqual(@as(u16, @intCast(i)), id);
    try std.testing.expectEqual(Cache.empty_id, slots[96]);
    try std.testing.expectEqual(@as(u32, 1), freq[0]);
    try std.testing.expectEqual(@as(u32, 0), freq[96]);
}

test "a high count keeps an old stamp and a one-off is the victim" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 100 };
    var counts: [shape.experts]u32 = @splat(0);
    counts[1] = 50;
    counts[4] = 1;
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;

    const miss = [_]u16{9};
    try scheduleUsed(&vram, &miss, &counts, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(Line.resident, line[0]);
    try std.testing.expectEqual(@as(u16, 9), slots[1]);
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expect(onDevice(vram, 1));
    try std.testing.expect(!onDevice(vram, 9));
}

test "zero lifetime counts fall back to stamp LRU" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 4, 7 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 5 };
    var counts: [shape.experts]u32 = @splat(0);
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;

    const miss = [_]u16{9};
    try scheduleUsed(&vram, &miss, &counts, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(onDevice(vram, 7));
}

test "lifetime count beats this round's appearances" {
    var slots = [_]u16{Cache.empty_id};
    var line = [_]Line{.empty};
    var stamp = [_]u32{0};
    var layer = [_]u8{0};
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.experts]u32 = @splat(0);
    counts[8] = 5;
    const ids = [_]u16{ 3, 3, 3, 3, 8 };
    var decisions: [ids.len]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try scheduleUsed(&vram, &ids, &counts, benchPolicy(1, 100, 100), 1, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[4]);
    try std.testing.expectEqual(@as(u16, 8), slots[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
}

test "a resident named this step is kept even when its count is low" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 100 };
    var counts: [shape.experts]u32 = @splat(0);
    counts[4] = 1;
    const ids = [_]u16{ 1, 4, 9 };
    var decisions: [3]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try scheduleUsed(&vram, &ids, &counts, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.cpu, decisions[2]);
    try std.testing.expect(onDevice(vram, 1));
    try std.testing.expect(onDevice(vram, 4));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 9));
}

test "observed counts seed empty lines and a later round keeps the hot id" {
    var counts: [shape.experts]u32 = @splat(0);
    try observe(&counts, &.{ 1, shared_id, 1, 3 });
    try std.testing.expectEqual(@as(u32, 2), counts[1]);
    try std.testing.expectEqual(@as(u32, 1), counts[3]);
    try observe(&counts, &.{1});
    try std.testing.expectEqual(@as(u32, 3), counts[1]);
    var short: [1]u32 = .{9};
    try std.testing.expectError(error.Length, observe(&short, &.{1}));
    try std.testing.expectEqual(@as(u32, 9), short[0]);

    var seeded_counts: [shape.experts]u32 = @splat(0);
    seeded_counts[3] = 10;
    seeded_counts[1] = 10;
    seeded_counts[8] = 2;
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    const n = try seedHottest(&vram, &seeded_counts, 2);
    try std.testing.expectEqual(@as(u16, 2), n);
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u16, 3), slots[1]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expect(!onDevice(vram, 1));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 8));
    try reap(&vram, 1, 1);
    try std.testing.expect(onDevice(vram, 1));

    var wide: [3]u16 = .{ Cache.empty_id, Cache.empty_id, Cache.empty_id };
    var wide_line: [3]Line = .{ .empty, .empty, .empty };
    var wide_stamp: [3]u32 = .{ 0, 0, 0 };
    var wide_layer: [3]u8 = .{ 0, 0, 0 };
    var wide_vram = Vram.init(&wide, &wide_line, &wide_stamp, &wide_layer);
    try std.testing.expectError(error.Length, seedHottest(&wide_vram, counts[0..3], 1));
    try std.testing.expectEqual(Line.empty, wide_line[0]);
}

test "a rank fills its own lines and does not copy the other rank's expert" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.experts]u32 = @splat(0);
    counts[400] = 100;
    counts[7] = 3;
    const rank0 = Owned{ .lo = 0, .hi = 256 };
    const n = try fillOwned(&vram, &counts, rank0);
    try std.testing.expectEqual(@as(u16, 2), n);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);
    try std.testing.expectEqual(@as(u16, 0), slots[1]);
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 400));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try reap(&vram, 7, 1);
    try reap(&vram, 0, 1);

    var decisions: [2]Decision = undefined;
    var scratch: [2]u16 = undefined;
    const asked = [_]u16{ 400, 9 };
    try scheduleUsed(&vram, &asked, &counts, benchPolicy(2, 100, 10_000), 4, &decisions, &scratch, rank0);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[1]);
    try std.testing.expect(onDevice(vram, 7));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 400));
    try std.testing.expect(vramIndex(vram, 9) != null);
}

test "an empty slot table misses every routed id and installs nothing" {
    var slots: [0]u16 = .{};
    var freq: [0]u32 = .{};
    var cache = init(&slots, &freq);
    const ids = [_]u16{ shared_id, 4, 4, 8 };
    var decisions: [4]Decision = undefined;
    var scratch: [2]u16 = undefined;
    const policy = benchPolicy(0, 10, 10);
    try planRound(cache, &ids, policy, &decisions, &scratch);
    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[2]);
    try std.testing.expectEqual(Decision.cpu, decisions[3]);

    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &ids, 4, policy, &scratch));
    try std.testing.expectEqual(@as(usize, 0), slots.len);
}

test "zero expert bytes copies nothing" {
    var slots: [0]u16 = .{};
    var freq: [0]u32 = .{};
    const cache = init(&slots, &freq);
    const ids = [_]u16{ 1, 2 };
    var decisions: [2]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try planRound(cache, &ids, benchPolicy(0, 0, 1000), &decisions, &scratch);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.cpu, decisions[1]);
}

test "a short scratch or a short decision buffer is refused without writes" {
    var slots = [_]u16{1};
    var freq = [_]u32{5};
    var cache = init(&slots, &freq);
    slots[0] = 1;
    freq[0] = 5;

    var decisions: [1]Decision = undefined;
    try std.testing.expectError(error.Length, planRound(cache, &[_]u16{ 1, 2 }, benchPolicy(1, 1, 1), &decisions, &[_]u16{}));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u32, 5), freq[0]);

    var one: [2]Decision = undefined;
    try std.testing.expectError(error.Scratch, planRound(cache, &[_]u16{ 2, 3 }, benchPolicy(1, 1, 1), &one, &[_]u16{}));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);

    var tiny: [1]u16 = undefined;
    try std.testing.expectError(error.Scratch, swap(&cache, &[_]u16{ 2, 3 }, 4, benchPolicy(1, 1, 0), &tiny));
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u32, 5), freq[0]);

    var off = benchPolicy(1, 1, 0);
    off.swap_every = 0;
    var scratch: [2]u16 = undefined;
    try std.testing.expectEqual(@as(u16, 0), try swap(&cache, &[_]u16{2}, 4, off, &scratch));
    try std.testing.expectEqual(@as(u32, 5), freq[0]);
}

test "residentAt distinguishes the same expert id on two layers" {
    var slots = [_]u16{ 7, 7 };
    var line = [_]Line{ .resident, .resident };
    var stamp = [_]u32{ 1, 2 };
    var layer = [_]u8{ 0, 40 };
    const vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 7, 7 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 2 };
    layer = .{ 0, 40 };

    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expect(residentAt(vram, 40, 7));
    try std.testing.expect(!residentAt(vram, 1, 7));
    try std.testing.expect(!residentAt(vram, 0, 8));
    // Id-only lookup still aliases. Generate does not call it on this table.
    try std.testing.expect(onDevice(vram, 7));

    line[1] = .inflight;
    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expect(!residentAt(vram, 40, 7));
}

test "reapAt promotes one inflight layer and leaves the other inflight" {
    var slots = [_]u16{ 7, 7 };
    var line = [_]Line{ .inflight, .inflight };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 40 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 7, 7 };
    line = .{ .inflight, .inflight };
    stamp = .{ 0, 0 };
    layer = .{ 0, 40 };

    try reapAt(&vram, 0, 7, 3);
    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expect(!residentAt(vram, 40, 7));
    try std.testing.expectEqual(@as(u32, 3), stamp[0]);

    // The other layer's copy is still inflight, then promotable on its own reap.
    try reapAt(&vram, 40, 7, 4);
    try std.testing.expect(residentAt(vram, 40, 7));
    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expectEqual(@as(u32, 4), stamp[1]);
    try std.testing.expectEqual(@as(u32, 3), stamp[0]);

    // Layer 0 is already resident. The wrapper must not treat that as the
    // layer-40 inflight copy, and must not move layer 40.
    line[1] = .inflight;
    stamp[1] = 0;
    try std.testing.expectError(error.NotInflight, reap(&vram, 7, 9));
    try std.testing.expectEqual(Line.resident, line[0]);
    try std.testing.expectEqual(Line.inflight, line[1]);
    try std.testing.expectEqual(@as(u32, 3), stamp[0]);
    try reapAt(&vram, 40, 7, 9);
    try std.testing.expect(residentAt(vram, 40, 7));
    try std.testing.expectEqual(@as(u32, 9), stamp[1]);

    // An inflight copy that exists only on layer 40 is not promoted by `reap`.
    slots = .{ 7, Cache.empty_id };
    line = .{ .inflight, .empty };
    stamp = .{ 0, 0 };
    layer = .{ 40, 0 };
    try std.testing.expectError(error.NotInflight, reap(&vram, 7, 1));
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(!residentAt(vram, 40, 7));
    try reapAt(&vram, 40, 7, 2);
    try std.testing.expect(residentAt(vram, 40, 7));
    try std.testing.expectEqual(@as(u8, 40), layer[0]);
}

test "stageKeeping tags a newly assigned line as layer 0" {
    var slots = [_]u16{4};
    var line = [_]Line{.resident};
    var stamp = [_]u32{1};
    var layer = [_]u8{40};
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{4};
    line = .{.resident};
    stamp = .{1};
    layer = .{40};

    try std.testing.expectEqual(@as(usize, 0), try stageKeeping(&vram, 9, &.{}));
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(@as(u8, 0), layer[0]);
    try std.testing.expectEqual(Line.inflight, line[0]);
    try std.testing.expect(vramIndexAt(vram, 0, 9) != null);
    try std.testing.expectEqual(@as(?usize, null), vramIndexAt(vram, 40, 9));
    try std.testing.expect(slots[0] != shared_id);
}

test "a keep on layer 0 does not protect that id on layer 40" {
    var slots = [_]u16{ 7, 7, 3 };
    var line = [_]Line{ .resident, .resident, .resident };
    var stamp = [_]u32{ 4, 4, 4 };
    var layer = [_]u8{ 0, 40, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 7, 7, 3 };
    line = .{ .resident, .resident, .resident };
    stamp = .{ 4, 4, 4 };
    layer = .{ 0, 40, 0 };

    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    counts[0][7] = 10;
    counts[40][7] = 1;
    counts[0][3] = 8;
    const ids = [_]u16{ 7, 9 };
    var decisions: [2]Decision = undefined;
    var scratch: [2]u16 = undefined;
    try scheduleUsedAt(&vram, 0, &ids, &counts, benchPolicy(3, 100, 100), 8, &decisions, &scratch, allOwned());

    try std.testing.expectEqual(Decision.hit, decisions[0]);
    try std.testing.expectEqual(Decision.copy, decisions[1]);
    try std.testing.expect(residentAt(vram, 0, 7));
    try std.testing.expect(residentAt(vram, 0, 3));
    try std.testing.expect(!residentAt(vram, 40, 7));
    try std.testing.expect(vramIndexAt(vram, 0, 9) != null);
    try std.testing.expectEqual(@as(u32, 8), stamp[0]);
    try std.testing.expectEqual(@as(u32, 4), stamp[2]);
    try std.testing.expectEqual(@as(u16, 7), slots[0]);
    try std.testing.expectEqual(@as(u16, 3), slots[2]);
}

test "scheduleUsed is the layer-0 wrapper and is not the multi-layer path" {
    var slots = [_]u16{ 1, 4 };
    var line = [_]Line{ .resident, .resident };
    var stamp = [_]u32{ 1, 50 };
    var layer = [_]u8{ 0, 40 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 50 };
    layer = .{ 0, 40 };

    // The flat vector says id 4 is hot. The wrapper stores that on layer 0 only,
    // so the layer-40 copy scores 0 and is the victim.
    var flat: [shape.experts]u32 = @splat(0);
    flat[1] = 1;
    flat[4] = 100;
    var decisions: [1]Decision = undefined;
    var scratch: [2]u16 = undefined;
    const miss = [_]u16{9};
    try scheduleUsed(&vram, &miss, &flat, benchPolicy(2, 100, 100), 6, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(Decision.copy, decisions[0]);
    try std.testing.expectEqual(@as(u16, 1), slots[0]);
    try std.testing.expectEqual(@as(u8, 0), layer[0]);
    try std.testing.expectEqual(@as(u16, 9), slots[1]);
    try std.testing.expectEqual(@as(u8, 0), layer[1]);
    try std.testing.expect(!residentAt(vram, 40, 4));

    slots = .{ 1, 4 };
    line = .{ .resident, .resident };
    stamp = .{ 1, 50 };
    layer = .{ 0, 40 };
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    counts[0][1] = 1;
    counts[40][4] = 100;
    try scheduleUsedAt(&vram, 0, &miss, &counts, benchPolicy(2, 100, 100), 7, &decisions, &scratch, allOwned());
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(@as(u8, 0), layer[0]);
    try std.testing.expectEqual(@as(u16, 4), slots[1]);
    try std.testing.expectEqual(@as(u8, 40), layer[1]);
    try std.testing.expect(residentAt(vram, 40, 4));
    try std.testing.expect(!residentAt(vram, 0, 1));
}

test "scheduleUsedAt does not stage another rank or the shared expert" {
    var slots = [_]u16{ Cache.empty_id, Cache.empty_id };
    var line = [_]Line{ .empty, .empty };
    var stamp = [_]u32{ 0, 0 };
    var layer = [_]u8{ 0, 0 };
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    counts[3][400] = 50;
    counts[3][9] = 1;
    const ids = [_]u16{ 400, shared_id, 9 };
    var decisions: [3]Decision = undefined;
    var scratch: [4]u16 = undefined;
    const rank0 = Owned{ .lo = 0, .hi = 256 };
    const policy = benchPolicy(2, 100, 10_000);
    try std.testing.expectError(error.Length, scheduleUsedAt(&vram, @intCast(shape.layers), &ids, &counts, policy, 4, &decisions, &scratch, rank0));
    try std.testing.expectEqual(Line.empty, line[0]);

    try scheduleUsedAt(&vram, 3, &ids, &counts, policy, 4, &decisions, &scratch, rank0);
    try std.testing.expectEqual(Decision.cpu, decisions[0]);
    try std.testing.expectEqual(Decision.hit, decisions[1]);
    try std.testing.expectEqual(Decision.copy, decisions[2]);
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, 400));
    try std.testing.expectEqual(@as(?usize, null), vramIndex(vram, shared_id));
    try std.testing.expect(!residentAt(vram, 3, 9));
    try std.testing.expect(vramIndexAt(vram, 3, 9) != null);
    try std.testing.expectEqual(@as(u16, 9), slots[0]);
    try std.testing.expectEqual(@as(u8, 3), layer[0]);
    try std.testing.expectEqual(Line.empty, line[1]);
    try std.testing.expect(slots[0] != shared_id and slots[1] != shared_id);
}

test "fillOwnedLayers stripes an explicit line count and skips the other rank" {
    const wide: u16 = @intCast(shape.layers + 2);
    var slots: [shape.layers + 2]u16 = @splat(Cache.empty_id);
    var line: [shape.layers + 2]Line = @splat(.empty);
    var stamp: [shape.layers + 2]u32 = @splat(0);
    var layer: [shape.layers + 2]u8 = @splat(0);
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    const rank0 = Owned{ .lo = 0, .hi = 256 };
    var short: [2][shape.experts]u32 = @splat(@splat(0));
    try std.testing.expectError(error.Length, fillOwnedLayers(&vram, wide, &short, rank0));
    try std.testing.expectError(error.Length, fillOwnedLayers(&vram, wide, &counts, .{ .lo = 4, .hi = 2 }));
    try std.testing.expectEqual(Line.empty, line[0]);

    const n = try fillOwnedLayers(&vram, wide, &counts, rank0);
    try std.testing.expectEqual(wide, n);
    // Id, then layer: slot 1 is layer 1's lowest id, not layer 0's second id.
    try std.testing.expectEqual(@as(u16, 0), slots[0]);
    try std.testing.expectEqual(@as(u8, 0), layer[0]);
    try std.testing.expectEqual(@as(u16, 0), slots[1]);
    try std.testing.expectEqual(@as(u8, 1), layer[1]);
    try std.testing.expectEqual(@as(u16, 0), slots[shape.layers - 1]);
    try std.testing.expectEqual(@as(u8, @intCast(shape.layers - 1)), layer[shape.layers - 1]);
    try std.testing.expectEqual(@as(u16, 1), slots[shape.layers]);
    try std.testing.expectEqual(@as(u8, 0), layer[shape.layers]);
    try std.testing.expectEqual(@as(u16, 1), slots[shape.layers + 1]);
    try std.testing.expectEqual(@as(u8, 1), layer[shape.layers + 1]);
    var on_layer0: usize = 0;
    for (slots, layer) |id, lay| {
        try std.testing.expect(id < 256);
        try std.testing.expect(id != shared_id);
        if (lay == 0) on_layer0 += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), on_layer0);

    // Fewer lines than layers: the remainder is layers 0..3, one lowest id each.
    var few_slots: [4]u16 = @splat(Cache.empty_id);
    var few_line: [4]Line = @splat(.empty);
    var few_stamp: [4]u32 = @splat(0);
    var few_layer: [4]u8 = @splat(0);
    var few = Vram.init(&few_slots, &few_line, &few_stamp, &few_layer);
    try std.testing.expectEqual(@as(u16, 4), try fillOwnedLayers(&few, 4, &counts, rank0));
    try std.testing.expectEqual([_]u16{ 0, 0, 0, 0 }, few_slots);
    try std.testing.expectEqual([_]u8{ 0, 1, 2, 3 }, few_layer);

    // Rank 1. A hotter id outside Owned is not staged. Layer 0 keeps its hottest owned id.
    var rank_slots: [4]u16 = @splat(Cache.empty_id);
    var rank_line: [4]Line = @splat(.empty);
    var rank_stamp: [4]u32 = @splat(0);
    var rank_layer: [4]u8 = @splat(0);
    var rank = Vram.init(&rank_slots, &rank_line, &rank_stamp, &rank_layer);
    counts[0][1] = 999;
    counts[0][300] = 5;
    const rank1 = Owned{ .lo = 256, .hi = 512 };
    try std.testing.expectEqual(@as(u16, 4), try fillOwnedLayers(&rank, 4, &counts, rank1));
    try std.testing.expect(vramIndexAt(rank, 0, 300) != null);
    try std.testing.expect(vramIndexAt(rank, 1, 256) != null);
    try std.testing.expect(vramIndexAt(rank, 2, 256) != null);
    try std.testing.expect(vramIndexAt(rank, 3, 256) != null);
    try std.testing.expectEqual(@as(?usize, null), vramIndex(rank, 1));
    for (rank_slots) |id| {
        try std.testing.expect(id >= 256 and id < 512);
        try std.testing.expect(id != shared_id);
    }
}

test "fillOwnedLayers keeps a hot id inside the stripe and does not give its layer a second line" {
    const lines: u16 = @intCast(shape.layers);
    var slots: [shape.layers]u16 = @splat(Cache.empty_id);
    var line: [shape.layers]Line = @splat(.empty);
    var stamp: [shape.layers]u32 = @splat(0);
    var layer: [shape.layers]u8 = @splat(0);
    var vram = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    counts[0][9] = 10;
    counts[0][8] = 9;
    const n = try fillOwnedLayers(&vram, lines, &counts, .{ .lo = 0, .hi = 256 });
    try std.testing.expectEqual(lines, n);
    try std.testing.expect(vramIndexAt(vram, 0, 9) != null);
    try std.testing.expectEqual(@as(?usize, null), vramIndexAt(vram, 0, 8));
    try std.testing.expect(vramIndexAt(vram, 1, 0) != null);
    try std.testing.expect(vramIndexAt(vram, 40, 0) != null);
    var on_layer0: usize = 0;
    for (layer) |lay| {
        if (lay == 0) on_layer0 += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), on_layer0);
}

test "lineIds maps hits to their line and everything else to -1, stagedCopies lists each staged line once" {
    var slots: [3]u16 = undefined;
    var line: [3]Line = undefined;
    var stamp: [3]u32 = undefined;
    var layer: [3]u8 = undefined;
    var v = Vram.init(&slots, &line, &stamp, &layer);
    var counts: [shape.layers][shape.experts]u32 = @splat(@splat(0));
    const pol = Policy{ .slots = 3, .expert_bytes = 10, .copy_bytes = 20 };
    var ids = [_]u16{ 7, 9, 9, 11, 512 };
    var dec: [5]Decision = undefined;
    var scratch: [8]u16 = undefined;
    try observe(&counts[2], &ids);
    try scheduleUsedAt(&v, 2, &ids, &counts, pol, 1, &dec, &scratch, allOwned());
    // 3 unique misses (9 twice is hottest), budget for two copies: 9 and 7 copy, 11 cpu, 512 hit.
    try std.testing.expectEqual(Decision.copy, dec[0]);
    try std.testing.expectEqual(Decision.copy, dec[1]);
    try std.testing.expectEqual(Decision.copy, dec[2]);
    try std.testing.expectEqual(Decision.cpu, dec[3]);
    try std.testing.expectEqual(Decision.hit, dec[4]);
    var copies: [4]Copy = undefined;
    const n = try stagedCopies(v, 2, &ids, &dec, &copies);
    try std.testing.expectEqual(@as(usize, 2), n);
    var ln: [5]i32 = undefined;
    try lineIds(v, 2, &ids, &dec, &ln);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, -1, -1, -1 }, &ln);
    try std.testing.expectError(error.Length, stagedCopies(v, 2, &ids, &dec, copies[0..1]));
    for (copies[0..n]) |c| try reapAt(&v, 2, c.id, 2);
    try scheduleUsedAt(&v, 2, &ids, &counts, pol, 3, &dec, &scratch, allOwned());
    try lineIds(v, 2, &ids, &dec, &ln);
    try std.testing.expectEqual(@as(i32, @intCast(lineOf(v, 2, 7).?)), ln[0]);
    try std.testing.expectEqual(@as(i32, @intCast(lineOf(v, 2, 9).?)), ln[1]);
    try std.testing.expectEqual(ln[1], ln[2]);
    try std.testing.expect(ln[3] == -1 and ln[4] == -1);
    try std.testing.expectEqual(@as(?usize, null), lineOf(v, 3, 7));
}
