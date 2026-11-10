//! Flash Next EXL3 offload: the host slot cache and its per-round copy/cpu/hit planner.

const std = @import("std");
/// Flash Next's routed geometry (was exl3_shape.zig): 48 decoder layers, 512 routed experts.
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

/// One decision per id. Does not write the slot table. Unique misses sort by count descending.
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

/// Fill slots from a routed histogram of length 512.
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

pub fn copyCount(policy: Policy, unique_misses: usize) usize {
    if (policy.expert_bytes == 0 or unique_misses == 0) return 0;
    const raw = policy.copy_bytes / policy.expert_bytes;
    if (raw >= unique_misses) return unique_misses;
    return @intCast(raw);
}

pub fn appearances(ids: []const u16, id: u16) u32 {
    var n: u32 = 0;
    for (ids) |x| {
        if (x == id) n +|= 1;
    }
    return n;
}

/// Count descending, id ascending. `ctx` is the multiset the counts are taken from.
pub fn hotterFirst(ctx: []const u16, a: u16, b: u16) bool {
    const ca = appearances(ctx, a);
    const cb = appearances(ctx, b);
    if (ca != cb) return ca > cb;
    return a < b;
}

pub fn indexOf(ids: []const u16, id: u16) ?usize {
    for (ids, 0..) |x, i| {
        if (x == id) return i;
    }
    return null;
}

pub fn pushUnique(dst: []u16, n: *usize, id: u16) error{Scratch}!void {
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
