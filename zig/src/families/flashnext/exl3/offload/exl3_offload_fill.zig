//! Flash Next EXL3 offload: first fill of the VRAM lines, hottest-first and striped across layers.

const std = @import("std");
const pol = @import("exl3_offload_policy.zig");
const vram_ = @import("exl3_offload_vram.zig");
const counts_ = @import("exl3_offload_counts.zig");

const shape = pol.shape;
const shared_id = pol.shared_id;
const Cache = pol.Cache;
const Vram = vram_.Vram;
const stage = vram_.stage;
const vramEmpty = vram_.vramEmpty;
const vramIndex = vram_.vramIndex;
const vramIndexAt = vram_.vramIndexAt;
const Owned = counts_.Owned;
const allOwned = counts_.allOwned;

/// Fill empty VRAM lines with the hottest counted ids.
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

/// Use every empty line. Hottest owned ids go in first. Lines that are still empty take the remaining owned ids.
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

/// Stage up to `lines` owned experts.
pub fn fillOwnedLayers(v: *Vram, lines: u16, counts: []const [shape.experts]u32, owned: Owned) error{Length}!u16 {
    if (counts.len != shape.layers) return error.Length;
    if (owned.hi > shape.experts or owned.lo > owned.hi) return error.Length;
    std.debug.assert(v.layer.len == v.slots.len and v.line.len == v.slots.len);
    var used: [shape.layers]u16 = @splat(0);
    const hot = placeHottest(v, lines, counts, owned, &used);
    return fillStripe(v, lines, owned, &used, hot);
}
