//! A rank's MiniMax-M3.1-Flash CUDA memory: a sequence's caches and a forward's buffers.

const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const weights = @import("cuda_weights.zig");

/// Rows one stream's verify window takes at most (the pending token and its drafts).
pub const window_rows = 16;
/// Rows a shared forward takes at most (several streams' windows).
pub const batch_rows = 32;
/// Rows a prompt chunk takes at most: each expert's weights are read once a chunk.
pub const prompt_rows = 2048;

/// Sub-allocations of one device allocation, 256-byte aligned like the driver's own.
const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.buf.len);
        return a.buf.ptr + at;
    }
};

fn arenaBytes(sizes: []const usize) usize {
    var total: usize = 0;
    for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
    return @max(total, 256);
}

/// Allocates `sizes` (bytes) in one zeroed device buffer and hands out each field's address in order.
fn arena(d: *const cuda.Driver, sizes: []const usize, out: []u64) !Arena {
    var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, arenaBytes(sizes)) };
    errdefer a.buf.free();
    try a.buf.fill8(0, null);
    for (sizes, out) |n, *p| p.* = a.take(n);
    return a;
}

/// Slots of a layer's caches for `capacity` positions: whole index blocks.
pub fn slots(c: *const cfg.Config, capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, c.block);
}

/// Bytes one position takes in one layer on this rank: its key and value heads and the shared index key, bf16.
pub fn rowBytes(c: *const cfg.Config, kv_heads: usize) usize {
    return (2 * kv_heads * c.head_dim + c.index_dim) * 2;
}

/// Bytes a stream of `capacity` positions holds on this rank.
pub fn cacheBytes(c: *const cfg.Config, kv_heads: usize, capacity: usize) usize {
    return arenaBytes(&.{@as(usize, c.layers) * slots(c, capacity) * rowBytes(c, kv_heads)});
}

/// One stream's caches: each layer's keys [slots][kv][128], values, then index keys [slots][128], at layer_at.
pub const State = struct {
    arena: Arena,
    layer_at: []u64,
    kv_heads: usize,
    capacity: usize,
    pos: usize = 0,
    pending: usize = 0, // rows the last forward wrote past pos (committed by keep)
    gpa: std.mem.Allocator,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, c: *const cfg.Config, kv_heads: usize, capacity: usize) !State {
        const per = slots(c, capacity) * rowBytes(c, kv_heads);
        var at: [1]u64 = undefined;
        var a = try arena(d, &.{@as(usize, c.layers) * per}, &at);
        errdefer a.buf.free();
        const layer_at = try gpa.alloc(u64, c.layers);
        for (layer_at, 0..) |*l, i| l.* = at[0] + i * per;
        return .{ .arena = a, .layer_at = layer_at, .kv_heads = kv_heads, .capacity = capacity, .gpa = gpa };
    }

    pub fn deinit(st: *State) void {
        st.gpa.free(st.layer_at);
        st.arena.buf.free();
        st.* = undefined;
    }

    pub fn keys(st: *const State, l: usize) u64 {
        return st.layer_at[l];
    }

    pub fn values(st: *const State, c: *const cfg.Config, l: usize) u64 {
        return st.layer_at[l] + slots(c, st.capacity) * st.kv_heads * c.head_dim * 2;
    }

    pub fn indexKeys(st: *const State, c: *const cfg.Config, l: usize) u64 {
        return st.layer_at[l] + slots(c, st.capacity) * st.kv_heads * c.head_dim * 2 * 2;
    }

    /// A fresh sequence: the caches need no clearing (reads stop at pos), only the length.
    pub fn reset(st: *State) void {
        st.pos = 0;
        st.pending = 0;
    }
};

/// The names of a Buffers' fields that live in its arena, in allocation order.
const fields = [_][]const u8{ "ids", "resid", "x", "qkv", "q", "iq", "o", "gathered", "gu", "act", "sy", "logits32", "pick", "wts", "members", "items", "counts", "rank", "hist", "eact", "ey", "part", "scores", "sel", "bits", "fnormed", "logits", "top_v", "top_i", "taps" };

/// What a forward of up to `rows` rows writes; `prefill` buffers take prompt chunks (the last row's logits only).
pub const Buffers = struct {
    arena: Arena,
    rows: usize,
    prefill: bool,
    context: usize,
    ids: u64 = 0, // i32 [rows]
    resid: u64 = 0, // fp32 [rows][hidden], bf16 values: the residual
    x: u64 = 0, // bf16 [rows][hidden]: the normed rows
    qkv: u64 = 0, // bf16 [rows][q] then [rows][k], [rows][v], [rows][index q | index k] of this rank (Forward.projections)
    q: u64 = 0, // bf16 [rows][hq][128] normed and rotated
    iq: u64 = 0, // bf16 [rows][hi][128]
    o: u64 = 0, // bf16 [rows][hq * 128]: the attention output
    gathered: u64 = 0, // fp32 [world][rows][2][hidden]: every rank's partial of a site (attention: one slot a row)
    gu: u64 = 0, // bf16 [rows][widest intermediate share] gate then up (dspark/'s: [rows][gate | up])
    act: u64 = 0, // bf16 [rows][widest intermediate share]
    sy: u64 = 0, // fp32 [rows][hidden]: the shared expert's partial
    logits32: u64 = 0, // fp32 [rows][experts]: the router's
    pick: u64 = 0, // i32 [rows][topk]
    wts: u64 = 0, // fp32 [rows][topk]
    members: u64 = 0, // the routed plan (grouped.zig)
    items: u64 = 0,
    counts: u64 = 0,
    rank: u64 = 0,
    hist: u64 = 0,
    eact: u64 = 0, // bf16 [rows * topk][expert share]: each pick's activation
    ey: u64 = 0, // fp32 [rows * topk][hidden]: each pick's down rows
    part: u64 = 0, // the lane matmul's split-K partials
    scores: u64 = 0, // fp32 [rows][hi][blocks of the context]
    sel: u64 = 0, // i32 [rows][hi][topk_blocks]
    bits: u64 = 0, // u32 [rows][hi][words]: a prompt chunk's selections
    fnormed: u64 = 0, // bf16 [rows][hidden]: the final norm's rows
    logits: u64 = 0, // bf16 [rows][this rank's vocabulary]
    top_v: u64 = 0, // fp32 [rows]: each row's greedy pick on this rank, value and id
    top_i: u64 = 0,
    taps: u64 = 0, // bf16 [rows][5][hidden]: a drafter's tapped residual rows (16 bytes without one)
    max_items: usize = 0,

    /// Buffers for `rows` rows over `context` positions on `world` ranks; `extra`: a drafter's linears, for partials.
    pub fn init(d: *const cuda.Driver, w: *const weights.Weights, rows: usize, prefill: bool, context: usize, world: usize, lane: *const cuda.qmmf.Lane, extra: []const weights.Mx, taps: bool) !Buffers {
        const c = &w.config;
        const s = w.share;
        const h: usize = c.hidden;
        const hq = s.queries(c).len();
        const hk = s.kv.len();
        const hi = hk;
        const topk: usize = c.topk;
        const pairs = rows * topk;
        const tile: usize = if (prefill) 64 else 16;
        const max_items = cuda.grouped.maxItems(pairs, c.experts, tile);
        const inter = @max(s.dense.len(), s.shared.len());
        const nb = c.blocks(context);
        const words = std.math.divCeil(usize, nb, 32) catch unreachable;
        const qkv_rows = (hq + 2 * hk) * c.head_dim + (hi + 1) * c.index_dim;
        var part: usize = 16; // the lane's weights: each layer's indexer, dspark/'s projections
        for (w.layers) |l| part = @max(part, lane.partBytes(rows, l.attn.index));
        for (extra) |x| part = @max(part, lane.partBytes(rows, x));
        const sizes = [fields.len]usize{
            rows * 4,                    rows * h * 4,               rows * h * 2,             rows * qkv_rows * 2,                               rows * hq * c.head_dim * 2,
            rows * hi * c.index_dim * 2, rows * hq * c.head_dim * 2, world * rows * 2 * h * 4, rows * 2 * inter * 2,                              rows * inter * 2,
            rows * h * 4,                rows * c.experts * 4,       pairs * 4,                pairs * 4,                                         pairs * 4,
            max_items * 3 * 4 + 16,      16,                         pairs * 4 + 16,           (c.experts + 1) * 4 * ((pairs + 1023) / 1024 + 1), pairs * s.moe.len() * 2,
            pairs * h * 4,               part,                       rows * hi * nb * 4,       rows * hi * c.topk_blocks * 4,                     rows * hi * words * 4,
            rows * h * 2,                rows * s.vocab.len() * 2,   rows * 4,                 rows * 4,                                          if (taps) rows * 5 * h * 2 else 16,
        };
        var ptrs: [fields.len]u64 = undefined;
        var a = try arena(d, &sizes, &ptrs);
        errdefer a.buf.free();
        var b: Buffers = .{ .arena = a, .rows = rows, .prefill = prefill, .context = context, .max_items = max_items };
        inline for (fields, 0..) |f, i| @field(b, f) = ptrs[i];
        return b;
    }

    pub fn deinit(b: *Buffers) void {
        b.arena.buf.free();
        b.* = undefined;
    }

    /// The routed plan the grouped kernels read.
    pub fn plan(b: *const Buffers) cuda.grouped.Plan {
        return .{ .members = b.members, .items = b.items, .counts = b.counts, .rank = b.rank, .hist = b.hist };
    }

    /// Bitmap words of a row's selected blocks over `context`.
    pub fn selWords(b: *const Buffers, c: *const cfg.Config) usize {
        return std.math.divCeil(usize, c.blocks(b.context), 32) catch unreachable;
    }
};

test "a position's bytes in a layer on a rank with two key heads, slots by whole blocks, the cache's bytes" {
    const c = cfg.Config{};
    try std.testing.expectEqual(@as(usize, (2 * 2 * 128 + 128) * 2), rowBytes(&c, 2));
    try std.testing.expectEqual(@as(usize, 256), slots(&c, 129));
    try std.testing.expectEqual(@as(usize, 60 * 128 * 1280), cacheBytes(&c, 2, 100));
    try std.testing.expectEqual(fields.len, @typeInfo(Buffers).@"struct".field_names.len - 5);
}
