//! Llama on Metal: a stream's caches (every layer holds every position), and a window's or a chunk's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");

/// Rows one stream's decode window takes at most (a verify window: the pending token and its drafts).
pub const window_rows = 16;
/// Rows a shared decode forward takes at most (several streams' windows, row kernels); more take the prompt path.
pub const batch_rows = 32;
/// Rows a prompt chunk takes at most.
pub const prompt_rows = 2048;

/// Slots of a layer for `capacity` positions: whole 64-key blocks, so a prompt tile's reads stay in the layer.
pub fn slots(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, 64);
}

/// Keys a decode attention block takes for a stream of `capacity`: at most 512 blocks, 32 to 2,048 keys each.
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

/// Bytes a stream of `capacity` positions holds in caches, its logits row included.
pub fn cacheBytes(c: *const cfg.Config, capacity: usize) usize {
    return c.vocab * 2 + c.layers * slots(capacity) * kvBytes(c);
}

pub const Cache = struct {
    kv: mtl.Buffer, // keys then values of each layer, at `layer_at`
    logits: mtl.Buffer,
    layer_at: [cfg.max_layers]usize = @splat(0),
    layers: usize,
    kv_width: usize,
    capacity: usize,
    len: usize = 0,
    follow: ?mtl.ResidencySet.Follow = null,

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
            kv += slots(capacity) * kvBytes(c);
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
        return self.layer_at[l] + slots(self.capacity) * self.kv_width * 2;
    }

    pub fn deinit(self: *Cache) void {
        self.setResident(null);
        self.kv.deinit();
        self.logits.deinit();
    }
};

pub const fields = [_][]const u8{ "ids", "resid", "x", "q", "qn", "att", "y", "gu", "act", "logits_all", "pm", "pl", "po", "wq" };

/// Bytes of each Scratch buffer for `rows` rows at once, in `fields` order.
fn sizes(c: *const cfg.Config, rows: usize) [fields.len]usize {
    const r = rows;
    const prompt = rows > batch_rows;
    const h = c.hidden;
    const qw = r * c.queryWidth() * 2;
    const partials = @as(usize, @min(r, window_rows)) * c.heads * max_blocks * 4; // decode attention's maxima, sums, outputs (fp32)
    return .{
        r * 4,                                       r * h * 2,                                                      r * h * 2,                    r * c.qkvWidth() * 2,
        qw,                                          qw,                                                             r * h * 2,                    r * 2 * c.intermediate * 2,
        r * c.intermediate * 2,                      @as(usize, @min(r, batch_rows)) * c.vocab * 2,                  if (prompt) 16 else partials, if (prompt) 16 else partials,
        if (prompt) 16 else partials * cfg.head_dim, if (prompt and c.weights == .int4_g32) dequantBytes(c) else 16,
    };
}

/// Bytes of the widest int4 projection as bf16: a prompt chunk's GEMM reads each one from there.
fn dequantBytes(c: *const cfg.Config) usize {
    return @max(c.queryWidth(), c.intermediate) * c.hidden * 2;
}

/// Bytes a Scratch for `rows` rows holds.
pub fn scratchBytes(c: *const cfg.Config, rows: usize) usize {
    var total: usize = 0;
    for (sizes(c, rows)) |n| total += n;
    return total;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    resid: mtl.Buffer, // bf16 [rows, hidden]: the reference's residual
    x: mtl.Buffer, // bf16 normed rows
    q: mtl.Buffer, // the projections' rows: q | k | v
    qn: mtl.Buffer, // q rotated
    att: mtl.Buffer, // attention's output
    y: mtl.Buffer, // a sublayer's bf16 output [rows, hidden]
    gu: mtl.Buffer, // gate | up rows
    act: mtl.Buffer,
    logits_all: mtl.Buffer, // every row's logits of a window
    pm: mtl.Buffer, // decode attention's partials by block: maxima, sums, outputs
    pl: mtl.Buffer,
    po: mtl.Buffer,
    wq: mtl.Buffer, // an int4 projection as bf16 for a prompt chunk's GEMM
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
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
    }
};

test "llama state: a stream's caches on Nanbeige4.3-33B's geometry, and the decode blocks" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    // 52 layers of 4,096 positions, 4 KiB a position a layer, and a logits row
    try std.testing.expectEqual(@as(usize, 166912 * 2 + 52 * 4096 * 4096), cacheBytes(&c, 4096));
    try std.testing.expectEqual(@as(usize, 208 << 10), c.layers * kvBytes(&c));
    try std.testing.expect(std.math.divCeil(usize, max_capacity, chunk(max_capacity)) catch unreachable <= 512);
    try std.testing.expectEqual(@as(usize, 32), chunk(4096));
    try std.testing.expectEqual(@as(usize, 1024), chunk(524288));
    try std.testing.expect(scratchBytes(&c, prompt_rows) > scratchBytes(&c, batch_rows));
}
