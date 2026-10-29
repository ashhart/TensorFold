//! Spark2.5 on Metal: a stream's caches (sliding layers: a ring; full layers: every position), and a window's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

const geo = cfg.spark4b;

/// Rows one stream's decode window takes at most (a verify window: the pending token and its drafts).
pub const window_rows = 16;
/// Rows a shared decode forward takes at most (several streams' windows, row kernels); more take the prompt path.
pub const batch_rows = 32;
/// Rows a prompt chunk takes at most.
pub const prompt_rows = 2048;

/// The fused projection's rows: q, k, v, the gate a head, and zero rows to a whole 64 (the prompt matmul's tile).
pub const qkvg_rows = std.mem.alignForward(usize, geo.qkvgWidth(), 64);
/// The gate's first column in a fused projection row.
pub const gate_col = geo.queryWidth() + 2 * geo.kvWidth();

/// Ring slots of a sliding layer: its window and one more prompt chunk, to a multiple of 64 (no key block wraps).
pub const sliding_ring = std.mem.alignForward(usize, cfg.window + prompt_rows, 64);

/// Slots of a full layer for `capacity` positions: whole 64-key blocks, so a block's reads stay in the layer.
pub fn fullSlots(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, 64);
}

/// Keys a decode attention block takes: a run of positions, at the same absolute place for every row.
pub const sliding_chunk = 16;
/// A full layer's block for a stream of `capacity`: at most 512 blocks at capacity, 32 to 2,048 keys each.
pub fn fullChunk(capacity: usize) usize {
    return std.math.clamp(std.math.ceilPowerOfTwo(usize, std.math.divCeil(usize, capacity, 512) catch unreachable) catch 2048, 32, 2048);
}
/// Partials a decode row's head holds at most: a full layer's blocks at capacity (or a sliding window's), and more.
pub const max_blocks = 544;
/// The longest stream the full layers' blocks reach (512 of 2,048 keys).
pub const max_capacity = 512 * 2048;

/// Bytes one position takes in one layer: its keys and values.
const kv_bytes: usize = geo.kvWidth() * 2 * 2;

/// Bytes a stream of `capacity` positions holds in caches.
pub fn cacheBytes(capacity: usize) usize {
    var total: usize = cfg.vocab * 2;
    for (0..geo.layers) |i| total += (if (cfg.full(i)) fullSlots(capacity) else sliding_ring) * kv_bytes;
    return total;
}

pub const Cache = struct {
    kv: mtl.Buffer, // keys then values of each layer, at `layer_at`
    logits: mtl.Buffer,
    layer_at: [geo.layers]usize = @splat(0),
    capacity: usize,
    len: usize = 0,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn setResident(self: *Cache, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| mtl.resident.fields.apply(self, f, false);
        self.follow = hook;
        if (hook) |f| mtl.resident.fields.apply(self, f, true);
    }

    pub fn init(device: mtl.Device, capacity: usize) !Cache {
        var out = Cache{ .kv = undefined, .logits = undefined, .capacity = capacity };
        var kv: usize = 0;
        for (0..geo.layers) |i| {
            out.layer_at[i] = kv;
            kv += out.slots(i) * kv_bytes;
        }
        out.kv = try device.buffer(@max(kv, 16), mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.logits = try device.buffer(cfg.vocab * 2, mtl.ResourceOptions.shared);
        return out;
    }

    /// Slots of layer `l`'s keys: a ring's size, or 0 when slots are positions.
    pub fn ring(_: *const Cache, l: usize) usize {
        return if (cfg.full(l)) 0 else sliding_ring;
    }

    fn slots(self: *const Cache, l: usize) usize {
        return if (cfg.full(l)) fullSlots(self.capacity) else sliding_ring;
    }

    pub fn keys(self: *const Cache, l: usize) usize {
        return self.layer_at[l];
    }

    pub fn values(self: *const Cache, l: usize) usize {
        return self.layer_at[l] + self.slots(l) * geo.kvWidth() * 2;
    }

    pub fn deinit(self: *Cache) void {
        self.setResident(null);
        self.kv.deinit();
        self.logits.deinit();
    }
};

pub const fields = [_][]const u8{ "ids", "resid", "x", "q", "qn", "att", "o", "y", "gu", "act", "sums", "logits_all", "pm", "pl", "po" };

/// Bytes of each Scratch buffer for `rows` rows at once, in `fields` order.
fn sizes(rows: usize) [fields.len]usize {
    const r = rows;
    const prompt = rows > batch_rows;
    const h = geo.hidden;
    const qw = r * geo.queryWidth() * 2;
    // a prompt chunk's fp32 group sums: groups of 32 at the finest, over the widest input a projection reads
    const widest = @max(@max(h, geo.intermediate), geo.queryWidth());
    const partials = @as(usize, @min(r, window_rows)) * geo.query_heads * max_blocks * 4; // decode attention's maxima, sums, outputs (fp32)
    return .{
        r * 4,                        r * h * 4,                    r * h * 2,                                   r * qkvg_rows * 2,
        qw,                           qw,                           qw,                                          r * h * 2,
        r * 2 * geo.intermediate * 2, r * geo.intermediate * 2,     if (prompt) r * widest / 32 * 4 else 16,     @as(usize, @min(r, batch_rows)) * cfg.vocab * 2,
        if (prompt) 16 else partials, if (prompt) 16 else partials, if (prompt) 16 else partials * cfg.head_dim,
    };
}

/// Bytes a Scratch for `rows` rows holds.
pub fn scratchBytes(rows: usize) usize {
    var total: usize = 0;
    for (sizes(rows)) |n| total += n;
    return total;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer, // fp32 [rows, hidden]: the reference's residual
    x: mtl.Buffer, // bf16 normed rows
    q: mtl.Buffer, // the fused projection's rows: q | k | v | gates
    qn: mtl.Buffer, // q rotated
    att: mtl.Buffer, // attention's output, before the gate
    o: mtl.Buffer, // gated
    y: mtl.Buffer, // a sublayer's bf16 output [rows, hidden]
    gu: mtl.Buffer, // gate|up rows of the stacked projection
    act: mtl.Buffer,
    sums: mtl.Buffer, // a prompt chunk's group sums
    logits_all: mtl.Buffer, // every row's logits of a window
    pm: mtl.Buffer, // decode attention's partials by block: maxima, sums, outputs
    pl: mtl.Buffer,
    po: mtl.Buffer,
    rows: usize,

    /// Buffers for `rows` rows at once (batch_rows for decode, prompt_rows for prompt chunks).
    pub fn init(device: mtl.Device, rows: usize) !Scratch {
        const sizes_ = sizes(rows);
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

test "spark metal state: the fused rows, the ring and a stream's caches" {
    try std.testing.expectEqual(@as(usize, 6208), qkvg_rows);
    try std.testing.expectEqual(@as(usize, 6144), gate_col);
    try std.testing.expectEqual(@as(usize, 2560), sliding_ring);
    // 27 sliding rings and 9 full layers of 4,096 positions, 4 KiB a position a layer, and a logits row
    try std.testing.expectEqual(@as(usize, cfg.vocab * 2 + (27 * 2560 + 9 * 4096) * 4096), cacheBytes(4096));
    try std.testing.expect(std.math.divCeil(usize, max_capacity, fullChunk(max_capacity)) catch unreachable <= 512);
    try std.testing.expectEqual(@as(usize, 32), fullChunk(4096));
}
