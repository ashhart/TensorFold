//! Ripple on Metal: a stream's caches (global layers hold every position, sliding layers a ring) and the scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

/// Rows one stream's decode window takes at most (a verify window: the pending token and its drafts).
pub const window_rows = 16;
/// Rows a shared decode forward takes at most (several streams' windows, row kernels); more take the prompt path.
pub const batch_rows = 32;
/// Rows a prompt chunk takes at most.
pub const prompt_rows = 2048;
/// Keys a sliding layer's decode attention block takes: its window spans a handful of blocks.
pub const ring_chunk = 128;

/// Slots of a global layer for `capacity` positions: whole 64-key blocks, so a prompt tile's reads stay in the layer.
pub fn slots(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, 64);
}

/// Slots of a sliding layer: a chunk's rows and the window behind its first row, in whole 128-key blocks.
pub fn ring(c: *const cfg.Config, capacity: usize) usize {
    return @min(slots(capacity), std.mem.alignForward(usize, c.window - 1 + prompt_rows, ring_chunk));
}

/// Keys a global layer's decode attention block takes for a stream of `capacity`: at most 512 blocks, 32 to 2,048 keys.
pub fn chunk(capacity: usize) usize {
    return std.math.clamp(std.math.ceilPowerOfTwo(usize, std.math.divCeil(usize, capacity, 512) catch unreachable) catch 2048, 32, 2048);
}
/// Partials a decode row's head holds at most: a layer's blocks at capacity, and more.
pub const max_blocks = 544;
/// The longest stream the blocks reach (512 of 2,048 keys).
pub const max_capacity = 512 * 2048;

/// Bytes one position takes in one layer: its keys and values.
pub fn kvBytes(c: *const cfg.Config) usize {
    return c.kvWidth() * 2 * 2;
}

/// Slots layer `l` holds for a stream of `capacity`.
pub fn layerSlots(c: *const cfg.Config, l: usize, capacity: usize) usize {
    return if (c.global[l]) slots(capacity) else ring(c, capacity);
}

/// Bytes a stream of `capacity` positions holds in caches, its logits row included.
pub fn cacheBytes(c: *const cfg.Config, capacity: usize) usize {
    var total: usize = c.vocab * 2;
    for (0..c.layers) |l| total += layerSlots(c, l, capacity) * kvBytes(c);
    return total;
}

pub const Cache = struct {
    kv: mtl.Buffer, // keys then values of each layer, at `layer_at`
    logits: mtl.Buffer,
    layer_at: [cfg.max_layers]usize = @splat(0),
    layer_slots: [cfg.max_layers]usize = @splat(0),
    layers: usize,
    kv_width: usize,
    capacity: usize,
    len: usize = 0,
    follow: ?mtl.ResidencySet.Follow = null,
    /// The latest tapped rows a drafter may absorb: positions [start, start + rows), in a scratch's or last_tap.
    tap: Tap = .{},
    last_tap: ?mtl.Buffer = null, // the prompt's last row's taps, kept for the first draft

    pub fn setResident(self: *Cache, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| mtl.resident.fields.apply(self, f, false);
        self.follow = hook;
        if (hook) |f| mtl.resident.fields.apply(self, f, true);
    }

    pub fn init(device: mtl.Device, c: *const cfg.Config, capacity: usize) !Cache {
        var out = Cache{ .kv = undefined, .logits = undefined, .layers = c.layers, .kv_width = c.kvWidth(), .capacity = capacity };
        var kv: usize = 0;
        for (0..c.layers) |i| {
            out.layer_at[i] = kv;
            out.layer_slots[i] = layerSlots(c, i, capacity);
            kv += out.layer_slots[i] * kvBytes(c);
        }
        out.kv = try device.buffer(@max(kv, 16), mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.logits = try device.buffer(c.vocab * 2, mtl.ResourceOptions.shared);
        return out;
    }

    pub fn keys(self: *const Cache, l: usize) usize {
        return self.layer_at[l];
    }

    pub fn values(self: *const Cache, l: usize) usize {
        return self.layer_at[l] + self.layer_slots[l] * self.kv_width * 2;
    }

    pub fn deinit(self: *Cache) void {
        self.setResident(null);
        self.kv.deinit();
        self.logits.deinit();
        if (self.last_tap) |x| x.deinit();
    }
};

pub const Tap = struct { buf: ?mtl.Buffer = null, off: usize = 0, start: usize = 0, rows: usize = 0 };

pub const fields = [_][]const u8{ "ids", "resid", "v0", "x", "vx", "q", "g", "qn", "att", "y", "gu", "act", "logits_all", "pm", "pl", "po", "wq" };

/// Bytes of each Scratch buffer for `rows` rows at once, in `fields` order.
fn sizes(c: *const cfg.Config, rows: usize) [fields.len]usize {
    const r = rows;
    const prompt = rows > batch_rows;
    const hb = r * c.hidden * 2;
    const qw = r * c.queryWidth() * 2;
    const partials = @as(usize, @min(r, window_rows)) * c.heads * max_blocks * 4; // a segment's decode attention partials
    return .{
        r * 4,                                       hb,                                                              hb,                                            hb,                           hb,
        r * c.qkvWidth() * 2,                        qw,                                                              qw,                                            qw,                           hb,
        r * 2 * c.intermediate * 2,                  r * c.intermediate * 2,                                          @as(usize, @min(r, batch_rows)) * c.vocab * 2, if (prompt) 16 else partials, if (prompt) 16 else partials,
        if (prompt) 16 else partials * cfg.head_dim, if (prompt and c.fp4_mlp) c.intermediate * c.hidden * 2 else 16,
    };
}

/// Bytes a Scratch for `rows` rows holds.
pub fn scratchBytes(c: *const cfg.Config, rows: usize) usize {
    var total: usize = 0;
    for (sizes(c, rows)) |n| total += n;
    return total;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer, // bf16 [rows, hidden]: the release's residual
    v0: mtl.Buffer, // the rows' scaled embeddings: the value residual's source in every layer
    x: mtl.Buffer, // bf16 normed rows
    vx: mtl.Buffer, // v's input: lerp(v0, x, the layer's blend)
    q: mtl.Buffer, // the projections' rows: q | k | v
    g: mtl.Buffer, // the gate projection's rows
    qn: mtl.Buffer, // q normed (and rotated on sliding layers)
    att: mtl.Buffer, // attention's output, then gated
    y: mtl.Buffer, // a sublayer's bf16 output [rows, hidden]
    gu: mtl.Buffer, // gate | up rows
    act: mtl.Buffer,
    logits_all: mtl.Buffer, // every row's logits of a window
    pm: mtl.Buffer, // decode attention's partials by block: maxima, sums, outputs
    pl: mtl.Buffer,
    po: mtl.Buffer,
    wq: mtl.Buffer, // a prompt chunk's NVFP4 projection as bf16 (its global left to the GEMM)
    taps: ?mtl.Buffer = null, // with a drafter: every row's tapped residuals (bf16 [rows, taps, hidden])
    rows: usize,

    /// Buffers for `rows` rows at once (batch_rows for decode, prompt_rows for prompt chunks).
    pub fn init(device: mtl.Device, c: *const cfg.Config, rows: usize) !Scratch {
        const sizes_ = sizes(c, rows);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(sizes_[i], mtl.ResourceOptions.shared);
            made += 1;
        }
        s.rows = rows;
        s.taps = null;
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
        if (s.taps) |x| x.deinit();
    }
};

test "ricursive state: global layers hold every position, sliding layers a ring of 3,072" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    try std.testing.expectEqual(@as(usize, 3072), ring(&c, 262144));
    try std.testing.expectEqual(@as(usize, 128), ring(&c, 100)); // a short stream: its own whole blocks
    try std.testing.expectEqual(@as(usize, 4096), kvBytes(&c));
    // 16 global layers of 262,144 positions, 64 sliding layers of 3,072, 4 KiB a position a layer, and a logits row
    try std.testing.expectEqual(@as(usize, 100352 * 2 + (16 * 262144 + 64 * 3072) * 4096), cacheBytes(&c, 262144));
    try std.testing.expect(ring(&c, max_capacity) % ring_chunk == 0 and ring(&c, max_capacity) >= c.window - 1 + prompt_rows);
    try std.testing.expect(std.math.divCeil(usize, max_capacity, chunk(max_capacity)) catch unreachable <= 512);
    try std.testing.expect(scratchBytes(&c, prompt_rows) > scratchBytes(&c, batch_rows));
}
