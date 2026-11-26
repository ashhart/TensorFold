//! The latent caches' shape (State.Shape): a cache a DSA layer and the MTP layer, its rows and its pooled index keys.
const std = @import("std");
const config = @import("config.zig");
const latent = @import("cuda_latent.zig");

pub const page_rows = latent.chunk;

pub const Shape = struct {
    caches: usize,
    row: usize, // kv_lora + qk_rope: 512 on GLM-5.3-Flash
    i_dim: usize,
    pool: usize, // rows a pooled key sums: 4 on GLM-5.3-Flash
    keys: usize, // index key and gate rows a cache holds: a ring of index_ring, or maxInt for a whole sequence's
    shared: std.StaticBitSet(config.max_layers) = .empty, // caches whose layer keeps no index keys (full GLM-5.3's)

    /// `caches` caches: the loaded layers' (full GLM-5.3: all DSA, cache j is layer j's), then the MTP layer's.
    pub fn of(c: anytype, caches: usize) Shape {
        var s: Shape = .{ .caches = caches, .row = c.latentRow(), .i_dim = c.i_dim, .pool = c.kpool, .keys = @import("cuda_state.zig").index_ring };
        if (c.dsa) for (0..caches -| 1) |j| if (c.shared.isSet(j)) s.shared.set(j);
        return s;
    }

    /// Cache `j` keeps index keys: its ring rows and pooled keys (the MTP layer's always does).
    pub fn pooled(s: Shape, j: usize) bool {
        return !s.shared.isSet(j);
    }

    /// The caches before `j` that keep index keys.
    pub fn pooledBefore(s: Shape, j: usize) usize {
        var n: usize = 0;
        for (0..j) |i| n += @intFromBool(s.pooled(i));
        return n;
    }

    /// One cache's latent rows on a page, and their pooled keys.
    pub fn latentBytes(s: Shape) usize {
        return page_rows * s.row * 2;
    }

    pub fn pkBytes(s: Shape) usize {
        return page_rows / s.pool * s.i_dim * 2;
    }

    /// One keyed cache's part of a page: its latent rows, then their pools' keys.
    pub fn cacheBytes(s: Shape) usize {
        return s.latentBytes() + s.pkBytes();
    }

    /// Cache `j`'s part of a page from its start: the parts before it (a shared layer's has no keys).
    pub fn cacheAt(s: Shape, j: usize) usize {
        return j * s.latentBytes() + s.pooledBefore(j) * s.pkBytes();
    }

    /// A page's bytes: every cache's part, padded to whole latent rows (page_rows a page) and whole token bytes.
    pub fn pageBytes(s: Shape) u64 {
        const unit = s.row * 2 / std.math.gcd(s.row * 2, page_rows) * page_rows;
        return (s.cacheAt(s.caches) + unit - 1) / unit * unit;
    }

    /// TF_GLM_KV_HOST: every keyed cache's pooled keys on a device page of their own.
    pub fn pkPageBytes(s: Shape) u64 {
        return s.pooledBefore(s.caches) * s.pkBytes();
    }
};
