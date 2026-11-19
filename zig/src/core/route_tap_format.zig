//! Route-tap wire format and its bounded SPSC storage, independent of a live session.
const std = @import("std");

/// `TF_ROUTE_TAP=/path` enables the tap; empty or unset means off.
pub const env_name = "TF_ROUTE_TAP";
/// File magic and format version (little-endian header, then records).
pub const magic = "TFR1";
pub const version: u16 = 1;
pub const flag_weights: u16 = 1;
pub const header_bytes = 16;
/// Slots in the host ring; wrap drops the newest record when the writer lags.
pub const ring_slots = 128;
/// Expert ids (and weights) one record may hold; a longer layer is truncated.
pub const max_ids = 4096;

pub const Header = extern struct {
    magic: [4]u8 = magic.*,
    version: u16 = version,
    flags: u16 = flag_weights,
    top_k: u16 = 0, // 0: each record's n is rows × k (variable); not a fixed k
    reserved: u16 = 0,
};

pub const Rec = struct {
    step: u32 = 0,
    layer: u16 = 0,
    n_ids: u16 = 0,
};

pub const Slot = struct {
    index: u32 = 0,
    rec: Rec = .{},
    ids: []i32 = &.{},
    weights: []f32 = &.{},
};

/// SPSC ring: one producer reserves, publishes after the step wait; the writer pops published slots.
pub const Ring = struct {
    recs: []Rec,
    ids: []i32,
    weights: []f32,
    cap: u32,
    max_ids: u16,
    reserved: std.atomic.Value(u32) = .init(0),
    published: std.atomic.Value(u32) = .init(0),
    read: std.atomic.Value(u32) = .init(0),
    drops: std.atomic.Value(u32) = .init(0),
    /// False when `ids`/`weights` are a backend's pinned host buffers (CUDA/HIP), not this allocator's.
    owned: bool = true,

    pub fn init(a: std.mem.Allocator, cap: u32, ids_each: u16) !Ring {
        const recs = try a.alloc(Rec, cap);
        errdefer a.free(recs);
        const ids = try a.alloc(i32, @as(usize, cap) * ids_each);
        errdefer a.free(ids);
        const weights = try a.alloc(f32, @as(usize, cap) * ids_each);
        @memset(recs, .{});
        @memset(ids, 0);
        @memset(weights, 0);
        return .{ .recs = recs, .ids = ids, .weights = weights, .cap = cap, .max_ids = ids_each, .owned = true };
    }

    pub fn deinit(r: *Ring, a: std.mem.Allocator) void {
        if (r.owned) {
            a.free(r.weights);
            a.free(r.ids);
        }
        a.free(r.recs);
        r.* = undefined;
    }

    pub fn slot(r: *Ring, i: u32) Slot {
        const n: usize = r.max_ids;
        const at = @as(usize, i) * n;
        return .{ .index = i, .rec = r.recs[i], .ids = r.ids[at .. at + n], .weights = r.weights[at .. at + n] };
    }

    /// Next write index, or null when `reserved - read == cap` (wrap would overwrite unread).
    pub fn reserve(r: *Ring) ?u32 {
        const rd = r.read.load(.acquire);
        const rs = r.reserved.load(.monotonic);
        if (rs - rd >= r.cap) {
            _ = r.drops.fetchAdd(1, .monotonic);
            return null;
        }
        r.reserved.store(rs + 1, .release);
        return rs % r.cap;
    }

    pub fn put(r: *Ring, i: u32, rec: Rec, ids: []const i32, wts: []const f32) void {
        r.recs[i] = rec;
        const n = @min(@as(usize, rec.n_ids), r.max_ids);
        const at = @as(usize, i) * r.max_ids;
        @memcpy(r.ids[at .. at + n], ids[0..n]);
        if (wts.len >= n) @memcpy(r.weights[at .. at + n], wts[0..n]);
    }

    /// Make every reserved slot visible to the writer (call after the step's existing wait).
    pub fn publish(r: *Ring) void {
        r.published.store(r.reserved.load(.monotonic), .release);
    }

    /// Next published slot, or null. The slot stays reserved until `consume`.
    pub fn pop(r: *Ring) ?u32 {
        const pubd = r.published.load(.acquire);
        const rd = r.read.load(.monotonic);
        if (rd == pubd) return null;
        return rd % r.cap;
    }

    pub fn consume(r: *Ring) void {
        _ = r.read.fetchAdd(1, .release);
    }
};

pub fn encodeHeader(out: *[header_bytes]u8, flags: u16) void {
    const h = Header{ .flags = flags };
    @memcpy(out[0..4], &h.magic);
    std.mem.writeInt(u16, out[4..6], h.version, .little);
    std.mem.writeInt(u16, out[6..8], h.flags, .little);
    std.mem.writeInt(u16, out[8..10], h.top_k, .little);
    std.mem.writeInt(u16, out[10..12], h.reserved, .little);
    @memset(out[12..16], 0);
}

pub fn encodeRec(out: []u8, rec: Rec, ids: []const i32, wts: []const f32) usize {
    std.debug.assert(out.len >= 8 + ids.len * 4 + wts.len * 4);
    std.mem.writeInt(u32, out[0..4], rec.step, .little);
    std.mem.writeInt(u16, out[4..6], rec.layer, .little);
    std.mem.writeInt(u16, out[6..8], rec.n_ids, .little);
    var at: usize = 8;
    for (ids[0..rec.n_ids]) |id| {
        std.mem.writeInt(i32, out[at..][0..4], id, .little);
        at += 4;
    }
    for (wts[0..rec.n_ids]) |w| {
        std.mem.writeInt(u32, out[at..][0..4], @bitCast(w), .little);
        at += 4;
    }
    return at;
}

