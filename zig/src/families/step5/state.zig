//! A stream's caches (sliding layers: a ring; full layers: every position, region summaries), and a window's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

/// Rows a forward call takes at most: the router's launch limit.
pub const window_rows = 16;
/// Token rows of the indexer's k and z kept for regions still filling: a region (8) plus a window, rounded up.
pub const index_ring = 32;

/// Ring slots of a sliding layer: its window and one more window of rows.
pub fn slidingRing(c: *const cfg.Config) usize {
    return c.window + window_rows;
}

/// Bytes a stream of `capacity` positions holds in caches.
pub fn cacheBytes(c: *const cfg.Config, capacity: usize) usize {
    var total: usize = @as(usize, c.vocab) * 2;
    const kv: usize = c.kvWidth() * 2 * 2;
    for (c.layer[0..c.layers]) |l| total += if (l.kind == .sliding) slidingRing(c) * kv else capacity * kv + regions(c, capacity) * c.proxy * 2 + 2 * index_ring * c.proxy * 2;
    return total;
}

pub fn regions(c: *const cfg.Config, capacity: usize) usize {
    return capacity / c.region + 1;
}

pub const Cache = struct {
    kv: mtl.Buffer, // keys then values of each layer, at `layer_at`
    index: mtl.Buffer, // full layers: summaries, then the k and z rings, at `index_at`
    logits: mtl.Buffer,
    layer_at: [cfg.max_layers]usize = @splat(0),
    index_at: [cfg.max_layers]usize = @splat(0),
    capacity: usize,
    len: usize = 0,

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Cache {
        var out = Cache{ .kv = undefined, .index = undefined, .logits = undefined, .capacity = capacity };
        var kv: usize = 0;
        var index: usize = 0;
        for (c.layer[0..c.layers], 0..) |l, i| {
            out.layer_at[i] = kv;
            kv += (if (l.kind == .sliding) slidingRing(c) else capacity) * c.kvWidth() * 2 * 2;
            if (l.kind == .full) {
                out.index_at[i] = index;
                index += (regions(c, capacity) + 2 * index_ring) * c.proxy * 2;
            }
        }
        out.kv = try device.buffer(kv, mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.index = try device.buffer(@max(index, 16), mtl.ResourceOptions.private);
        errdefer out.index.deinit();
        out.logits = try device.buffer(@as(usize, c.vocab) * 2, mtl.ResourceOptions.shared);
        return out;
    }

    /// Slots of layer `l`'s keys: a ring's size, or 0 when slots are positions.
    pub fn ring(_: *const Cache, c: *const cfg.Config, l: usize) usize {
        return if (c.layer[l].kind == .sliding) slidingRing(c) else 0;
    }

    fn slots(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        const r = self.ring(c, l);
        return if (r > 0) r else self.capacity;
    }

    pub fn keys(self: *const Cache, l: usize) usize {
        return self.layer_at[l];
    }

    pub fn values(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        return self.layer_at[l] + self.slots(c, l) * c.kvWidth() * 2;
    }

    pub fn summaries(self: *const Cache, l: usize) usize {
        return self.index_at[l];
    }

    /// The indexer's k ring of full layer `l` (its z ring follows).
    pub fn indexKeys(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        return self.index_at[l] + regions(c, self.capacity) * c.proxy * 2;
    }

    pub fn indexZ(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        return self.indexKeys(c, l) + index_ring * c.proxy * 2;
    }

    pub fn deinit(self: *Cache) void {
        self.kv.deinit();
        self.index.deinit();
        self.logits.deinit();
    }
};

const fields = [_][]const u8{ "ids", "resid", "x", "q", "qn", "k", "v", "g", "o", "y", "shared", "gate", "up", "act", "eg", "eu", "ea", "ey", "logits32", "pick", "wts", "iq", "iqn", "ik", "iz", "iw", "scores", "sel" };

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer, // fp32 [rows, hidden]
    x: mtl.Buffer, // bf16 normed rows
    q: mtl.Buffer,
    qn: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    g: mtl.Buffer,
    o: mtl.Buffer,
    y: mtl.Buffer, // a sublayer's bf16 output [rows, hidden]
    shared: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    eg: mtl.Buffer, // [rows * topk, moe_intermediate] each
    eu: mtl.Buffer,
    ea: mtl.Buffer,
    ey: mtl.Buffer, // [rows * topk, hidden]
    logits32: mtl.Buffer,
    pick: mtl.Buffer,
    wts: mtl.Buffer,
    iq: mtl.Buffer, // the indexer: raw and normed q, raw k and z, head weights, scores, selections
    iqn: mtl.Buffer,
    ik: mtl.Buffer,
    iz: mtl.Buffer,
    iw: mtl.Buffer,
    scores: mtl.Buffer,
    sel: mtl.Buffer,
    route: [7]mtl.Buffer, // the router's per-Mac groups, unused on one Mac

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Scratch {
        const r: usize = window_rows;
        const h: usize = c.hidden;
        const slots = r * c.topk;
        const qw = r * c.heads * c.head_dim * 2;
        const ix = r * c.index_heads * c.proxy * 2;
        const sizes = [fields.len]usize{
            r * 4,         r * h * 4,      r * h * 2,               qw,                      qw,
            r * c.kvWidth() * 2, r * c.kvWidth() * 2, r * c.heads * 2, qw,               r * h * 2,
            r * h * 2,     r * c.intermediate * 2, r * c.intermediate * 2, r * c.intermediate * 2, slots * c.moe_intermediate * 2,
            slots * c.moe_intermediate * 2, slots * c.moe_intermediate * 2, slots * h * 2, r * c.experts * 4, slots * 4,
            slots * 4,     ix,             ix,                      r * c.proxy * 2,         r * c.proxy * 2,
            r * c.index_heads * 4, r * c.kv_heads * regions(c, capacity) * 4, r * c.kv_heads * c.index_topk * 4,
        };
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(sizes[i], mtl.ResourceOptions.shared);
            made += 1;
        }
        var routed: usize = 0;
        errdefer for (s.route[0..routed]) |b| b.deinit();
        for (&s.route) |*b| {
            b.* = try device.buffer(@as(usize, c.experts) * window_rows * 4 + 4096, mtl.ResourceOptions.shared);
            routed += 1;
        }
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
        for (s.route) |b| b.deinit();
    }
};
