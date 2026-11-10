//! The DFlash2 drafter's weights on the HIP device: projections quantized to MLX affine-4 g64 on the host.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const config = @import("config.zig");

const quant = core.quant;
const Projection = hip.quant.Projection;
/// The drafter's safetensors as `load` reads them.
pub const Checkpoint = core.Checkpoint;

pub const Layer = struct {
    input_norm: u64, // fp32 rows of the norms, as the target's
    post_norm: u64,
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: u64,
    k_norm: u64,
    gate: Projection,
    up: Projection,
    down: Projection,
    attn_conv: Projection, // kernel_projection: hidden -> 4 * groups dynamic coefficients
    attn_base: u64, // base_kernel (2 branches, 2 lags, hidden) fp32
    mlp_conv: Projection,
    mlp_base: u64,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    buffers: std.ArrayList(hip.DeviceBuffer) = .empty,
    fc: Projection = undefined, // the taps' concatenation -> hidden
    hidden_norm: u64 = 0,
    final_norm: u64 = 0,
    selector: Projection = undefined, // hidden_projection: hidden -> rank
    predecessor: []u16 = &.{}, // codebooks (vocab, rank) bf16, on the host
    successor: []u16 = &.{},
    layers: []Layer = &.{},

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.predecessor);
        w.gpa.free(w.successor);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    /// Device bytes of every uploaded tensor.
    pub fn deviceBytes(w: *const Weights) usize {
        var n: usize = 0;
        for (w.buffers.items) |b| n += b.len;
        return n;
    }
};

/// The drafter's widths the loader needs.
pub const Shape = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    q_width: usize,
    kv_width: usize,
    head_dim: usize,
    dynamic: usize,
    rank: usize,
    taps: usize,

    pub fn of(c: config.Config) Shape {
        return .{
            .hidden = c.hidden,
            .intermediate = c.intermediate,
            .layers = c.layers,
            .vocab = c.vocab,
            .q_width = c.qWidth(),
            .kv_width = c.kvWidth(),
            .head_dim = c.head_dim,
            .dynamic = c.dynamic(),
            .rank = c.rank,
            .taps = c.n_taps,
        };
    }
};

fn bf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn toBf16(x: f32) u16 {
    // round to nearest even, as the device does
    const u: u32 = @bitCast(x);
    const rounding: u32 = 0x7fff + ((u >> 16) & 1);
    return @intCast((u +% rounding) >> 16);
}

/// One (n, k) projection in MLX's affine layout: words (n, k / 8) low nibble first, tables (n, k / 64) bf16.
pub const Quantized = struct {
    words: []u32,
    scales: []u16,
    biases: []u16,
    n: usize,
    k: usize,

    pub fn deinit(q: *Quantized, gpa: std.mem.Allocator) void {
        gpa.free(q.words);
        gpa.free(q.scales);
        gpa.free(q.biases);
        q.* = undefined;
    }

    /// The tensors as an MLX checkpoint stores them, which the HIP MLX format uploads as they are.
    pub fn host(q: *const Quantized) quant.mlx.Host {
        const g = q.k / 64;
        const sb = std.mem.sliceAsBytes;
        return .{
            .words = .{ .dtype = .i32, .rank = 2, .shape = .{ q.n, q.k / 8, 1, 1, 1 }, .bytes = sb(q.words) },
            .scales = .{ .dtype = .bf16, .rank = 2, .shape = .{ q.n, g, 1, 1, 1 }, .bytes = sb(q.scales) },
            .biases = .{ .dtype = .bf16, .rank = 2, .shape = .{ q.n, g, 1, 1, 1 }, .bytes = sb(q.biases) },
            .bits = 4,
            .group = 64,
        };
    }
};

/// MLX affine-4, group 64: scale bf16((max - min) / 15), bias bf16(min), nibbles round((w - bias) / scale), low first.
pub fn quantize(gpa: std.mem.Allocator, w: []align(1) const u16, n: usize, k: usize) !Quantized {
    if (n == 0 or k == 0 or k % 64 != 0 or w.len != n * k) return error.BadDraftShape;
    const groups = k / 64;
    const words = try gpa.alloc(u32, n * k / 8);
    errdefer gpa.free(words);
    const scales = try gpa.alloc(u16, n * groups);
    errdefer gpa.free(scales);
    const biases = try gpa.alloc(u16, n * groups);
    errdefer gpa.free(biases);
    for (0..n) |r| {
        for (0..groups) |g| {
            const row = w[r * k + g * 64 ..][0..64];
            var lo: f32 = std.math.inf(f32);
            var hi: f32 = -std.math.inf(f32);
            for (row) |h| {
                const x = bf16(h);
                if (!std.math.isFinite(x)) return error.BadDraftWeights;
                lo = @min(lo, x);
                hi = @max(hi, x);
            }
            const scale = toBf16((hi - lo) / 15.0);
            const bias = toBf16(lo);
            scales[r * groups + g] = scale;
            biases[r * groups + g] = bias;
            const s = bf16(scale);
            const b = bf16(bias);
            const inv = if (s == 0) 0.0 else 1.0 / s;
            for (0..8) |j| {
                var word: u32 = 0;
                for (0..8) |i| {
                    const q = std.math.clamp(@round((bf16(row[j * 8 + i]) - b) * inv), 0.0, 15.0);
                    word |= @as(u32, @intFromFloat(q)) << @intCast(4 * i);
                }
                words[(r * k + g * 64) / 8 + j] = word;
            }
        }
    }
    return .{ .words = words, .scales = scales, .biases = biases, .n = n, .k = k };
}

const Loader = struct {
    gpa: std.mem.Allocator,
    up: hip.Upload,
    ck: *core.Checkpoint,

    /// A bf16 (n, k) tensor quantized on the host and uploaded as an MLX projection.
    fn linear(l: Loader, name: []const u8, n: usize, k: usize) !Projection {
        const t = try l.ck.expect(name, .bf16, &.{ n, k });
        var q = try quantize(l.gpa, std.mem.bytesAsSlice(u16, t.bytes), n, k);
        defer q.deinit(l.gpa);
        return hip.quant.view(try hip.quant.upload(l.up.uploader(), .{ .mlx = q.host() }), false);
    }

    /// A bf16 vector widened to fp32 on the device, as the target keeps its norms.
    fn vector(l: Loader, name: []const u8, shape: []const usize) !u64 {
        const t = try l.ck.expect(name, .bf16, shape);
        const wide = try quant.convert.float32(l.gpa, t);
        defer l.gpa.free(wide.bytes);
        return (try l.up.uploader().tensor(wide)).ptr;
    }

    fn hostBf16(l: Loader, name: []const u8, shape: []const usize) ![]u16 {
        const t = try l.ck.expect(name, .bf16, shape);
        const out = try l.gpa.alloc(u16, t.bytes.len / 2);
        @memcpy(out, std.mem.bytesAsSlice(u16, t.bytes));
        return out;
    }
};

/// The drafter checkpoint in `dir` (model.safetensors) on the device `d`, shaped as `s`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const hip.Runtime, dir: []const u8, s: Shape) !Weights {
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var w: Weights = .{ .gpa = gpa };
    errdefer w.deinit();
    const l: Loader = .{ .gpa = gpa, .up = .{ .r = d, .buffers = &w.buffers, .gpa = gpa }, .ck = &ck };
    w.fc = try l.linear("fc.weight", s.hidden, s.taps * s.hidden);
    w.hidden_norm = try l.vector("hidden_norm.weight", &.{s.hidden});
    w.final_norm = try l.vector("norm.weight", &.{s.hidden});
    w.selector = try l.linear("candidate_selector.hidden_projection.weight", s.rank, s.hidden);
    w.predecessor = try l.hostBf16("candidate_selector.predecessor_codebook", &.{ s.vocab, s.rank });
    w.successor = try l.hostBf16("candidate_selector.successor_codebook", &.{ s.vocab, s.rank });
    w.layers = try gpa.alloc(Layer, s.layers);
    for (w.layers, 0..) |*layer, i| {
        var buf: [96]u8 = undefined;
        const Name = struct {
            buf: *[96]u8,
            i: usize,
            fn of(self: @This(), suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(self.buf, "layers.{d}.{s}", .{ self.i, suffix });
            }
        };
        const nm: Name = .{ .buf = &buf, .i = i };
        layer.* = .{
            .input_norm = try l.vector(try nm.of("input_layernorm.weight"), &.{s.hidden}),
            .post_norm = try l.vector(try nm.of("post_attention_layernorm.weight"), &.{s.hidden}),
            .q = try l.linear(try nm.of("self_attn.q_proj.weight"), s.q_width, s.hidden),
            .k = try l.linear(try nm.of("self_attn.k_proj.weight"), s.kv_width, s.hidden),
            .v = try l.linear(try nm.of("self_attn.v_proj.weight"), s.kv_width, s.hidden),
            .o = try l.linear(try nm.of("self_attn.o_proj.weight"), s.hidden, s.q_width),
            .q_norm = try l.vector(try nm.of("self_attn.q_norm.weight"), &.{s.head_dim}),
            .k_norm = try l.vector(try nm.of("self_attn.k_norm.weight"), &.{s.head_dim}),
            .gate = try l.linear(try nm.of("mlp.gate_proj.weight"), s.intermediate, s.hidden),
            .up = try l.linear(try nm.of("mlp.up_proj.weight"), s.intermediate, s.hidden),
            .down = try l.linear(try nm.of("mlp.down_proj.weight"), s.hidden, s.intermediate),
            .attn_conv = try l.linear(try nm.of("attention_conv.kernel_projection.weight"), s.dynamic, s.hidden),
            .attn_base = try l.vector(try nm.of("attention_conv.base_kernel"), &.{ 2, 2, s.hidden }),
            .mlp_conv = try l.linear(try nm.of("mlp_conv.kernel_projection.weight"), s.dynamic, s.hidden),
            .mlp_base = try l.vector(try nm.of("mlp_conv.base_kernel"), &.{ 2, 2, s.hidden }),
        };
    }
    return w;
}
