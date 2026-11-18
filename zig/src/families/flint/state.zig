//! A stream's caches (sliding layers: a ring; full layers: every position), and a window's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

/// Rows one stream's decode window takes at most (a verify window: the pending token and its drafts).
pub const window_rows = 16;
/// Rows a shared decode forward takes at most (several streams' windows, row kernels); more take the prompt path.
pub const batch_rows = 32;
/// Rows a prompt chunk takes at most: each expert's weights are read once a chunk.
pub const prompt_rows = 2048;

/// Ring slots of a sliding layer: its window and one more prompt chunk, to a multiple of 64 (no key block wraps).
pub fn slidingRing(c: *const cfg.Config) usize {
    return std.mem.alignForward(usize, c.window + prompt_rows, 64);
}

/// Slots of a full layer for `capacity` positions: whole 64-key blocks, so a block's reads stay in the layer.
pub fn fullSlots(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, 64);
}

/// Keys a decode attention block takes: a run of positions, at the same absolute place for every row.
pub const sliding_chunk = 16;
/// A full layer's block for a stream of `capacity`: about 512 blocks at capacity, 32 to 512 keys each.
pub fn fullChunk(capacity: usize) usize {
    return std.math.clamp(std.math.ceilPowerOfTwo(usize, std.math.divCeil(usize, capacity, 512) catch unreachable) catch 512, 32, 512);
}
/// Partials a decode row's head holds at most: a full layer's blocks at capacity (or a sliding window's), and more.
pub const max_blocks = 544;

/// Bytes a stream of `capacity` positions holds in caches.
pub fn cacheBytes(c: *const cfg.Config, capacity: usize) usize {
    var total: usize = @as(usize, c.vocab) * 2;
    const kv: usize = c.kvWidth() * 2 * 2;
    for (c.layer[0..c.layers]) |l| total += (if (l.kind == .sliding) slidingRing(c) else fullSlots(capacity)) * kv;
    return total;
}

pub const Cache = struct {
    kv: mtl.Buffer, // keys then values of each layer, at `layer_at`
    logits: mtl.Buffer,
    layer_at: [cfg.max_layers]usize = @splat(0),
    capacity: usize,
    len: usize = 0,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn setResident(self: *Cache, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| mtl.resident.fields.apply(self, f, false);
        self.follow = hook;
        if (hook) |f| mtl.resident.fields.apply(self, f, true);
    }

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Cache {
        var out = Cache{ .kv = undefined, .logits = undefined, .capacity = capacity };
        var kv: usize = 0;
        for (c.layer[0..c.layers], 0..) |l, i| {
            out.layer_at[i] = kv;
            kv += (if (l.kind == .sliding) slidingRing(c) else fullSlots(capacity)) * c.kvWidth() * 2 * 2;
        }
        out.kv = try device.buffer(@max(kv, 16), mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.logits = try device.buffer(@as(usize, c.vocab) * 2, mtl.ResourceOptions.shared);
        return out;
    }

    /// Slots of layer `l`'s keys: a ring's size, or 0 when slots are positions.
    pub fn ring(_: *const Cache, c: *const cfg.Config, l: usize) usize {
        return if (c.layer[l].kind == .sliding) slidingRing(c) else 0;
    }

    fn slots(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        const r = self.ring(c, l);
        return if (r > 0) r else fullSlots(self.capacity);
    }

    pub fn keys(self: *const Cache, l: usize) usize {
        return self.layer_at[l];
    }

    pub fn values(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        return self.layer_at[l] + self.slots(c, l) * c.kvWidth() * 2;
    }

    pub fn deinit(self: *Cache) void {
        self.setResident(null);
        self.kv.deinit();
        self.logits.deinit();
    }
};

pub const fields = [_][]const u8{ "ids", "resid", "x", "q", "qn", "o", "y", "shared", "gate", "act", "eg", "eu", "ea", "ey", "logits32", "pick", "wts", "xs", "sums", "offsets", "order", "logits_all", "pm", "pl", "po" };

/// Bytes of each Scratch buffer for `rows` rows at once, in `fields` order.
fn sizes(c: *const cfg.Config, rows: usize) [fields.len]usize {
    const r: usize = rows;
    const prompt = rows > batch_rows;
    const h: usize = c.hidden;
    const slots = r * c.topk;
    const qw = r * c.heads * c.head_dim * 2;
    const qkv: usize = (@as(usize, c.heads) + 2 * c.kv_heads) * c.head_dim; // a row of the stacked q|k|v
    // a prompt chunk's fp32 group sums: groups of 32 at the finest, over the widest input a projection reads
    const widest: usize = @max(@max(h, c.intermediate), @as(usize, c.heads) * c.head_dim);
    const partials = @min(r, window_rows) * c.heads * max_blocks * 4; // decode attention's per-chunk maxima, sums and outputs (fp32)
    const sums = if (prompt) @max(r * widest, slots * @max(h, @as(usize, c.moe_intermediate))) / 32 * 4 else 16;
    return .{
        r * 4,                          r * h * 4,                                     r * h * 2,                         r * qkv * 2,                  qw,
        qw,                             r * h * 2,                                     r * h * 2,                         r * 2 * c.intermediate * 2,   r * c.intermediate * 2,
        slots * c.moe_intermediate * 2, slots * c.moe_intermediate * 2,                slots * c.moe_intermediate * 2,    slots * h * 2,                r * c.experts * 4,
        slots * 4,                      slots * 4,                                     if (prompt) slots * h * 2 else 16, sums,                         c.experts * 4,
        slots * 4,                      @min(r, batch_rows) * @as(usize, c.vocab) * 2, if (prompt) 16 else partials,      if (prompt) 16 else partials, if (prompt) 16 else partials * c.head_dim,
    };
}

/// Bytes a Scratch for `rows` rows holds, its router buffers included.
pub fn scratchBytes(c: *const cfg.Config, rows: usize) usize {
    var total: usize = 0;
    for (sizes(c, rows)) |n| total += n;
    return total;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer, // fp32 [rows, hidden], bf16 values: the reference's residual
    x: mtl.Buffer, // bf16 normed rows
    q: mtl.Buffer, // q|k|v rows of the stacked projection
    qn: mtl.Buffer,
    o: mtl.Buffer,
    y: mtl.Buffer, // a sublayer's bf16 output [rows, hidden]
    shared: mtl.Buffer,
    gate: mtl.Buffer, // gate|up rows of the stacked projection
    act: mtl.Buffer,
    eg: mtl.Buffer, // [rows * topk, moe_intermediate] each
    eu: mtl.Buffer,
    ea: mtl.Buffer,
    ey: mtl.Buffer, // [rows * topk, hidden]
    logits32: mtl.Buffer,
    pick: mtl.Buffer,
    wts: mtl.Buffer,
    xs: mtl.Buffer, // the prompt path: picks' input rows by expert, group sums, each expert's first pair, the pair order
    sums: mtl.Buffer,
    offsets: mtl.Buffer,
    order: mtl.Buffer,
    logits_all: mtl.Buffer, // every row's logits of a window
    pm: mtl.Buffer, // decode attention's partials by chunk: maxima, sums, outputs
    pl: mtl.Buffer,
    po: mtl.Buffer,
    rows: usize,

    /// Buffers for `rows` rows at once (batch_rows for decode, prompt_rows for prompt chunks).
    pub fn init(device: mtl.Device, c: *const cfg.Config, rows: usize) !Scratch {
        @setEvalBranchQuota(10000);
        const sizes_ = sizes(c, rows);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(sizes_[i], mtl.ResourceOptions.shared);
            made += 1;
        }
        s.rows = rows;
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
    }
};
