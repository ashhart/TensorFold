//! Which layer owns what in a sparse-attention model: KV, selection, pool, cache slot, index keys; from explicit lists.
const std = @import("std");
const sparse_index = @import("sparse_index.zig");

/// One layer's owners (layer numbers); null where the layer has none of that kind.
pub const Entry = struct {
    sparse: bool = false, // scores and selects keys (else dense or sliding)
    kv_owner: u32, // whose KV cache it reads: itself or its source
    cache_slot: ?u32 = null, // its own cache's index among the KV owners; null for a borrower
    select_owner: ?u32 = null, // whose selection it uses: itself (an indexer) or the owner it follows
    pool_owner: ?u32 = null, // whose candidate pool it rescores (pool reuse), or itself
    index_key_owner: ?u32 = null, // whose cache holds the keys its scores read
};

pub const Source = struct { layer: u32, source: u32 };
pub const Follow = struct { layer: u32, owner: u32 };

/// A config's explicit lists; select_from names each sparse non-indexer's owner (none inferred: UnsettledSelectOwner).
pub const Lists = struct {
    layers: usize,
    sparse: []const u32,
    kv_source: []const Source = &.{},
    index_owners: []const u32 = &.{},
    select_from: []const Follow = &.{},
    index_cache: []const Source = &.{}, // an indexer whose index keys live in another layer's index cache (else its KV's)
    reuse: sparse_index.Reuse = .final,
};

pub const Error = error{ LayerOutOfRange, Repeated, LaterSource, ChainedSource, OwnerNotSparse, FollowerOfNonOwner, LaterOwner, FollowerIsOwner, FollowerNotSparse, UnsettledSelectOwner, OutOfMemory };

fn has(xs: []const u32, x: u32) bool {
    return std.mem.indexOfScalar(u32, xs, x) != null;
}

/// The table, validated once: sources and owners earlier than their users, no chains, every sparse layer settled.
pub fn build(gpa: std.mem.Allocator, l: Lists) Error![]Entry {
    const t = try gpa.alloc(Entry, l.layers);
    errdefer gpa.free(t);
    for (t, 0..) |*e, i| e.* = .{ .kv_owner = @intCast(i) };
    for (l.sparse, 0..) |s, i| {
        if (s >= l.layers) return error.LayerOutOfRange;
        if (has(l.sparse[0..i], s)) return error.Repeated;
        t[s].sparse = true;
    }
    for (l.kv_source, 0..) |s, i| {
        if (s.layer >= l.layers or s.source >= l.layers) return error.LayerOutOfRange;
        for (l.kv_source[0..i]) |o| if (o.layer == s.layer) return error.Repeated;
        if (s.source >= s.layer) return error.LaterSource;
        t[s.layer].kv_owner = s.source;
    }
    for (l.kv_source) |s| if (t[s.source].kv_owner != s.source) return error.ChainedSource;
    var slot: u32 = 0;
    for (t, 0..) |*e, i| if (e.kv_owner == i) {
        e.cache_slot = slot;
        slot += 1;
    };
    for (l.index_owners, 0..) |o, i| {
        if (o >= l.layers) return error.LayerOutOfRange;
        if (has(l.index_owners[0..i], o)) return error.Repeated;
        if (!t[o].sparse) return error.OwnerNotSparse;
        t[o].select_owner = o;
        t[o].pool_owner = o;
        t[o].index_key_owner = t[o].kv_owner;
    }
    for (l.index_cache, 0..) |x, i| { // an index cache apart from the KV cache: named, never inferred
        if (x.layer >= l.layers or x.source >= l.layers) return error.LayerOutOfRange;
        for (l.index_cache[0..i]) |o| if (o.layer == x.layer) return error.Repeated;
        if (!has(l.index_owners, x.layer)) return error.OwnerNotSparse;
        if (x.source > x.layer) return error.LaterSource;
        t[x.layer].index_key_owner = x.source;
    }
    for (l.select_from, 0..) |f, i| {
        if (f.layer >= l.layers or f.owner >= l.layers) return error.LayerOutOfRange;
        for (l.select_from[0..i]) |o| if (o.layer == f.layer) return error.Repeated;
        if (!t[f.layer].sparse) return error.FollowerNotSparse;
        if (has(l.index_owners, f.layer)) return error.FollowerIsOwner;
        if (!has(l.index_owners, f.owner)) return error.FollowerOfNonOwner;
        if (f.owner >= f.layer) return error.LaterOwner;
        const e = &t[f.layer];
        e.select_owner = f.owner;
        // final reuse takes the owner's picks (scored on its keys); pool reuse rescores the pool on its own keys
        e.pool_owner = if (l.reuse == .pool) f.owner else null;
        e.index_key_owner = if (l.reuse == .pool) e.kv_owner else t[f.owner].index_key_owner;
    }
    for (t) |e| if (e.sparse and e.select_owner == null) return error.UnsettledSelectOwner;
    return t;
}

/// A candidate follow rule, for tests and oracle runs only: each sparse non-owner follows the latest owner before it.
pub fn followLatest(gpa: std.mem.Allocator, sparse: []const u32, owners: []const u32) ![]Follow {
    var out: std.ArrayList(Follow) = .empty;
    errdefer out.deinit(gpa);
    for (sparse) |s| if (!has(owners, s)) {
        var best: ?u32 = null;
        for (owners) |o| if (o < s and (best == null or o > best.?)) {
            best = o;
        };
        if (best) |b| try out.append(gpa, .{ .layer = s, .owner = b });
    };
    return out.toOwnedSlice(gpa);
}

/// Bytes a selecting layer keeps for `rows` queries: its picks (budget each) and, when followers rescore it, its pool.
pub fn keptBytes(rows: usize, budget: usize, pool: usize) usize {
    return rows * (budget + pool) * @sizeOf(u32);
}

/// Query rows a scores launch may take over `keys` fp32 scores a row within `max_bytes` (at least one).
pub fn tileRows(keys: usize, max_bytes: usize) usize {
    return @max(1, max_bytes / @max(1, keys * @sizeOf(f32)));
}

test {
    _ = @import("sparse_topology_test.zig");
}
