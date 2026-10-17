//! A stream's caches (sliding layers: a ring; full layers: every position, region summaries), and a window's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

/// Rows a decode forward takes at most: the router's launch limit; more take the prompt path (tensor-unit matmuls).
pub const window_rows = 16;
/// Rows a prompt chunk takes at most: each expert's weights are read once a chunk.
pub const prompt_rows = 2048;
/// Rows the indexer scores at once: its scores buffer is these rows by every region.
pub const score_rows = 64;
/// Token rows of the indexer's k and z kept for regions still filling: a region (8) plus a prompt chunk, and spare.
pub const index_ring = prompt_rows + 16;

/// Ring slots of a sliding layer: its window and one more prompt chunk.
pub fn slidingRing(c: *const cfg.Config) usize {
    return c.window + prompt_rows;
}

/// Bytes a stream of `capacity` positions holds in caches.
pub fn cacheBytes(c: *const cfg.Config, capacity: usize) usize {
    var total: usize = @as(usize, c.vocab) * 2;
    const kv: usize = c.kvWidth() * 2 * 2;
    total += window_rows * @as(usize, c.hidden) * 4;
    for (c.layer[0 .. c.layers + c.mtp]) |l| total += if (l.kind == .sliding) slidingRing(c) * kv else capacity * kv + regions(c, capacity) * c.proxy * 2 + 2 * index_ring * c.proxy * 2;
    return total;
}

pub fn regions(c: *const cfg.Config, capacity: usize) usize {
    return capacity / c.region + 1;
}

pub const Cache = struct {
    kv: mtl.Buffer, // keys then values of each layer (MTP layers after the main stack), at `layer_at`
    index: mtl.Buffer, // full layers: summaries, then the k and z rings, at `index_at`
    logits: mtl.Buffer,
    hidden: mtl.Buffer, // fp32 residual rows of the last window, before the final norm: the draft head's input
    held: [8]u32 = undefined, // the draft head's tokens for the next window
    nheld: usize = 0,
    layer_at: [cfg.max_layers]usize = @splat(0),
    index_at: [cfg.max_layers]usize = @splat(0),
    capacity: usize,
    len: usize = 0,

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Cache {
        var out = Cache{ .kv = undefined, .index = undefined, .logits = undefined, .hidden = undefined, .capacity = capacity };
        var kv: usize = 0;
        var index: usize = 0;
        for (c.layer[0 .. c.layers + c.mtp], 0..) |l, i| {
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
        errdefer out.logits.deinit();
        out.hidden = try device.buffer(window_rows * @as(usize, c.hidden) * 4, mtl.ResourceOptions.shared);
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
        self.hidden.deinit();
    }
};

pub const fields = [_][]const u8{ "ids", "resid", "x", "q", "qn", "k", "v", "g", "o", "y", "shared", "gate", "up", "act", "eg", "eu", "ea", "ey", "logits32", "pick", "wts", "iq", "iqn", "ik", "iz", "iw", "scores", "sel", "xs", "sums", "offsets", "order", "cat", "hid", "chain", "logits_all" };

/// Bytes of each Scratch buffer for `rows` rows at once, in `fields` order.
fn sizes(c: *const cfg.Config, capacity: usize, rows: usize) [fields.len]usize {
    const r: usize = rows;
    const prompt = rows > window_rows;
    const h: usize = c.hidden;
    const slots = r * c.topk;
    const qw = r * c.heads * c.head_dim * 2;
    const ix = r * c.index_heads * c.proxy * 2;
    return .{
        r * 4,                                          r * h * 4,                                                   r * h * 2,                         qw,                                qw,
        r * c.kvWidth() * 2,                            r * c.kvWidth() * 2,                                         r * c.heads * 2,                   qw,                                r * h * 2,
        r * h * 2,                                      r * c.intermediate * 2,                                      r * c.intermediate * 2,            r * c.intermediate * 2,            slots * c.moe_intermediate * 2,
        slots * c.moe_intermediate * 2,                 slots * c.moe_intermediate * 2,                              slots * h * 2,                     r * c.experts * 4,                 slots * 4,
        slots * 4,                                      ix,                                                          ix,                                r * c.proxy * 2,                   r * c.proxy * 2,
        r * c.index_heads * 4,                          @min(r, score_rows) * c.kv_heads * regions(c, capacity) * 4, r * c.kv_heads * c.index_topk * 4, if (prompt) slots * h * 2 else 16, if (prompt) @max(r * c.intermediate / 64, slots * h / 128) * 4 else 16,
        c.experts * 4,                                  slots * 4,                                                   r * h * 2 * 2,                     r * h * 4,                         h * 4,
        @min(r, window_rows) * @as(usize, c.vocab) * 2,
    };
}

/// Bytes a Scratch for `rows` rows holds, its router buffers included.
pub fn scratchBytes(c: *const cfg.Config, capacity: usize, rows: usize) usize {
    var total: usize = 7 * (@as(usize, c.experts) * window_rows * 4 + 4096);
    for (sizes(c, capacity, rows)) |n| total += n;
    return total;
}

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
    xs: mtl.Buffer, // the prompt path: picks' input rows by expert, group sums, each expert's first pair, the pair order
    sums: mtl.Buffer,
    offsets: mtl.Buffer,
    order: mtl.Buffer,
    cat: mtl.Buffer, // the MTP head: [enorm(embed) | hnorm(hidden)] rows, kept residual rows, a chain's hidden, every row's logits
    hid: mtl.Buffer,
    chain: mtl.Buffer,
    logits_all: mtl.Buffer,
    route: [7]mtl.Buffer, // the router's per-Mac groups, unused on one Mac
    rows: usize,

    /// Buffers for `rows` rows at once (window_rows for decode, prompt_rows for prompt chunks).
    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize, rows: usize) !Scratch {
        @setEvalBranchQuota(10000);
        const sizes_ = sizes(c, capacity, rows);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(sizes_[i], mtl.ResourceOptions.shared);
            made += 1;
        }
        var routed: usize = 0;
        errdefer for (s.route[0..routed]) |b| b.deinit();
        for (&s.route) |*b| {
            b.* = try device.buffer(@as(usize, c.experts) * window_rows * 4 + 4096, mtl.ResourceOptions.shared);
            routed += 1;
        }
        s.rows = rows;
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
        for (s.route) |b| b.deinit();
    }
};
