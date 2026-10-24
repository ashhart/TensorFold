//! A stream's caches (every layer's keys, values and index keys by position), and a window's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

/// Rows one stream's decode window takes at most (a verify window: the pending token and its drafts).
pub const window_rows = 16;
/// Rows a shared decode forward takes at most (several streams' windows, row kernels); more take the prompt path.
pub const batch_rows = 32;
/// Rows a prompt chunk takes at most: each expert's weights are read once a chunk.
pub const prompt_rows = 2048;

/// Slots of a layer for `capacity` positions: whole index blocks, so a block's reads stay in the layer.
pub fn slots(c: *const cfg.Config, capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, c.block);
}

/// Bytes of one position in one layer: key and value heads (int8 ones with an fp32 scale a head), and the index key.
pub fn rowBytes(c: *const cfg.Config) usize {
    const scales: usize = if (c.kv8) @as(usize, c.kv_heads) * 4 * 2 else 0;
    return @as(usize, c.kvWidth()) * 2 * c.kvBytes() + scales + @as(usize, c.index_dim) * 2;
}

/// Bytes a stream of `capacity` positions holds in caches.
pub fn cacheBytes(c: *const cfg.Config, capacity: usize) usize {
    return @as(usize, c.vocab) * 2 + @as(usize, c.layers) * slots(c, capacity) * rowBytes(c);
}

/// A stream's latest tapped rows (positions [start, start + rows) at `off` of `buf`), for its drafter.
pub const Tap = struct { buf: ?mtl.Buffer = null, off: usize = 0, start: u64 = 0, rows: usize = 0 };

pub const Cache = struct {
    kv: mtl.Buffer, // each layer's keys, values, then index keys, at `layer_at`
    logits: mtl.Buffer,
    tap: Tap = .{},
    /// The prompt's last tapped row, kept for the first draft (the prompt scratch serves the next pass).
    last_tap: ?mtl.Buffer = null,
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
        const per = slots(c, capacity) * rowBytes(c);
        for (0..c.layers) |i| out.layer_at[i] = i * per;
        out.kv = try device.buffer(@max(per * c.layers, 16), mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.logits = try device.buffer(@as(usize, c.vocab) * 2, mtl.ResourceOptions.shared);
        return out;
    }

    pub fn keys(self: *const Cache, l: usize) usize {
        return self.layer_at[l];
    }

    pub fn values(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        return self.layer_at[l] + slots(c, self.capacity) * c.kvWidth() * c.kvBytes();
    }

    /// An int8 cache's key scales [slot][kv head] fp32, then the value scales.
    pub fn keyScales(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        return self.layer_at[l] + slots(c, self.capacity) * c.kvWidth() * 2;
    }

    pub fn valueScales(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        return self.keyScales(c, l) + slots(c, self.capacity) * c.kv_heads * 4;
    }

    pub fn indexKeys(self: *const Cache, c: *const cfg.Config, l: usize) usize {
        const scales: usize = if (c.kv8) slots(c, self.capacity) * c.kv_heads * 4 * 2 else 0;
        return self.layer_at[l] + slots(c, self.capacity) * c.kvWidth() * c.kvBytes() * 2 + scales;
    }

    pub fn deinit(self: *Cache) void {
        self.setResident(null);
        if (self.last_tap) |x| x.deinit();
        self.kv.deinit();
        self.logits.deinit();
    }
};

pub const fields = [_][]const u8{ "ids", "resid", "x", "q", "qn", "iq", "o", "y", "shared", "gate", "act", "egu", "ea", "ey", "logits32", "pick", "wts", "xs", "sums", "offsets", "order", "logits_all", "pm", "pl", "po", "scores", "sel", "bits", "groups", "members" };

/// Bitmap words of a row's selected blocks: the model's whole window (the prompt attention is compiled for it).
pub fn selWords(c: *const cfg.Config) usize {
    return std.math.divCeil(usize, c.blocks(c.context), 32) catch unreachable;
}

/// Bytes of each Scratch buffer for `rows` rows at once over at most `context` positions, in `fields` order.
fn sizes(c: *const cfg.Config, rows: usize, context: usize) [fields.len]usize {
    const r: usize = rows;
    const prompt = rows > batch_rows;
    const h: usize = c.hidden;
    const slots_ = r * c.topk;
    const qw = r * c.heads * c.head_dim * 2;
    const inter: usize = @max(c.dense_intermediate, c.shared_intermediate);
    // a prompt chunk's fp32 group sums: groups of 32 at the finest, over the widest input a projection reads
    const widest: usize = @max(@max(h, inter), @as(usize, c.heads) * c.head_dim);
    const decode = @min(r, batch_rows);
    const partials = decode * c.heads * c.topk_blocks * 4 * 4; // decode attention's per-quarter maxima and sums (fp32)
    const sums = if (prompt) @max(r * widest, slots_ * @max(h, @as(usize, c.moe_intermediate))) / 32 * 4 else 16;
    const nb = c.blocks(context);
    const sel_rows = r * c.kv_heads;
    return .{
        r * 4,                                      r * h * 4,                           r * h * 2,                                               r * c.qkvRows() * 2,          qw,
        (r + 32) * c.index_heads * c.index_dim * 2, qw,                                  r * h * 2,                                               r * h * 2,                    r * 2 * inter * 2,
        r * inter * 2,                              slots_ * 2 * c.moe_intermediate * 2, slots_ * c.moe_intermediate * 2,                         slots_ * h * 2,               r * c.experts * 4,
        slots_ * 4,                                 slots_ * 4,                          if (prompt) slots_ * h * 2 else 16,                      sums,                         c.experts * 4,
        slots_ * 4,                                 decode * @as(usize, c.vocab) * 2,    if (prompt) 16 else partials,                            if (prompt) 16 else partials, if (prompt) 16 else partials * c.head_dim,
        sel_rows * nb * 4,                          sel_rows * c.topk_blocks * 4,        if (r > window_rows) sel_rows * selWords(c) * 4 else 16, (1 + 2 * slots_) * 4,         slots_ * 4,
    };
}

/// Bytes a Scratch for `rows` rows over `context` positions holds.
pub fn scratchBytes(c: *const cfg.Config, rows: usize, context: usize) usize {
    var total: usize = 0;
    for (sizes(c, rows, context)) |n| total += n;
    return total;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer, // fp32 [rows, hidden], bf16 values: the reference's residual
    x: mtl.Buffer, // bf16 normed rows
    q: mtl.Buffer, // the stacked projection's rows: q | k | v | index q | index k
    qn: mtl.Buffer, // q normed and rotated
    iq: mtl.Buffer, // index q normed and rotated
    o: mtl.Buffer,
    y: mtl.Buffer, // a sublayer's bf16 output [rows, hidden]
    shared: mtl.Buffer,
    gate: mtl.Buffer, // gate|up rows of a dense or shared MLP
    act: mtl.Buffer,
    egu: mtl.Buffer, // [rows * topk, 2 * moe_intermediate]: each pick's gate|up
    ea: mtl.Buffer,
    ey: mtl.Buffer, // [rows * topk, hidden]
    logits32: mtl.Buffer, // the router's fp32 logits [rows, experts]
    pick: mtl.Buffer,
    wts: mtl.Buffer,
    xs: mtl.Buffer, // the prompt path: picks' input rows by expert, group sums, each expert's first pair, the pair order
    sums: mtl.Buffer,
    offsets: mtl.Buffer,
    order: mtl.Buffer,
    logits_all: mtl.Buffer, // every row's logits of a window
    pm: mtl.Buffer, // decode attention's partials by selected block: maxima, sums, outputs
    pl: mtl.Buffer,
    po: mtl.Buffer,
    scores: mtl.Buffer, // the indexer's fp32 block scores [rows, kv_heads, blocks]
    sel: mtl.Buffer, // each (row, key head)'s selected blocks ascending, -1 past the last [rows, kv_heads, topk_blocks]
    bits: mtl.Buffer, // a prompt chunk's selections as bitmaps [rows, kv_heads, words]
    groups: mtl.Buffer, // a decode window's experts picked and each one's first member (experts.metal's G)
    members: mtl.Buffer, // the slots by expert (M)
    rows: usize,
    context: usize,
    /// The drafter's tapped rows [rows, taps x hidden] bf16, once a drafter asked for them.
    taps: ?mtl.Buffer = null,

    /// Buffers for `rows` rows at once (batch_rows for decode, prompt_rows for prompt chunks) over `context` positions.
    pub fn init(device: mtl.Device, c: *const cfg.Config, rows: usize, context: usize) !Scratch {
        @setEvalBranchQuota(10000);
        const sizes_ = sizes(c, rows, context);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(sizes_[i], mtl.ResourceOptions.shared);
            made += 1;
        }
        s.rows = rows;
        s.context = context;
        s.taps = null;
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
        if (s.taps) |x| x.deinit();
    }
};

test "minimax_m3 state: a position's bytes in every layer, slots by whole blocks" {
    const c = cfg.Config{};
    try std.testing.expectEqual(@as(usize, 2304), rowBytes(&c));
    try std.testing.expectEqual(@as(usize, 256), slots(&c, 129));
    try std.testing.expectEqual(@as(usize, 200064 * 2 + 60 * 128 * 2304), cacheBytes(&c, 100));
}
