//! LFM3's weights on CUDA: MLX affine-4, or weight-only NVFP4 (FFN) or block FP8 (FFN, q/k/v/o) with bf16 the rest.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const c = @import("config.zig");
const admit = @import("weights.zig");
const kern = @import("cuda_kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;
pub const Proj = cuda.qlinear.Weight;

/// Rows of the stacked attention projection: q (16 x 128), k and v (4 x 128 each), the 16 head gates.
pub const qkvg_rows = c.query_heads * c.head_dim + 2 * c.kv_heads * c.head_dim + c.query_heads;

pub const Conv = struct { in: Proj, out: Proj, weight: u64 };
/// q, k, v and the head gates as one projection (or q|k|v and `gate`); the q and k norms bf16 [128] each, q's first.
pub const Attention = struct { qkvg: Proj, gate: ?Proj = null, out: Proj, norms: u64, sliding: bool };

pub const Layer = struct {
    operator_norm: u64,
    post_operator_norm: u64,
    ffn_norm: u64,
    post_ffn_norm: u64,
    mixer: union(enum) { conv: Conv, attention: Attention },
    /// w1 | w3 stacked (gate rows, then up rows), and w2.
    gate_up: Proj,
    down: Proj,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: c.Config,
    /// The embedding as stored, bf16 [V, H] (the embed rows' table); the head is its lane copy.
    embed: u64 = 0,
    /// MLX: the embedding's words, scales and biases as stored (the embed rows' table); the head is their tiles.
    embed_affine: ?[3]u64 = null,
    head: Proj = undefined,
    final_norm: u64 = 0,
    layers: []Layer = &.{},
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.config.deinit();
        w.* = undefined;
    }

    fn alloc(w: *Weights, d: *const cuda.Driver, bytes: usize) !u64 {
        const b = try DeviceBuffer.alloc(d, @max(bytes, 256));
        errdefer {
            var x = b;
            x.free();
        }
        try w.buffers.append(w.gpa, b);
        return b.ptr;
    }

    fn upload(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        const b = try DeviceBuffer.fromHost(d, bytes);
        errdefer {
            var x = b;
            x.free();
        }
        try w.buffers.append(w.gpa, b);
        return b.ptr;
    }
};

/// The largest staged projection: conv in (3 x 2048 rows) in bf16, or the FFN's w1 and w3 as FP8 codes.
pub const stage_bytes = @max(3 * c.hidden * c.hidden * 2, 2 * c.intermediate * c.hidden) + 4 * 256;

const Ctx = struct {
    w: *Weights,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const kern.Kernels,
    ck: *core.checkpoint.Checkpoint,
    stage: DeviceBuffer,

    fn tensor(x: *Ctx, name: []const u8, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
        return x.w.upload(x.d, (try x.ck.expect(name, dtype, shape)).bytes);
    }

    /// bf16 projections [n_i, k] stacked along n, packed as one `.bf16` lane weight.
    fn bf16(x: *Ctx, names: []const []const u8, ns: []const usize, k: usize) !Proj {
        var n: usize = 0;
        var buf: [192]u8 = undefined;
        for (names, ns) |name, rows| {
            const t = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), .bf16, &.{ rows, k });
            if ((n + rows) * k * 2 > x.stage.len) return error.StageTooSmall;
            try x.stage.upload(n * k * 2, t.bytes);
            n += rows;
        }
        return x.pack(x.stage.ptr, n, k);
    }

    /// bf16 rows [n, k] on the device as a `.bf16` lane weight.
    fn pack(x: *Ctx, src: u64, n: usize, k: usize) !Proj {
        const out = try x.w.alloc(x.d, cuda.modelopt.padded(n) * k * 2);
        try x.k.pack.bf16Lane(x.s, src, out, n, k);
        try x.s.synchronize(); // the next projection's staging overwrites this one's
        return .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) };
    }

    /// MLX affine-4 projections (g64) [n_i, k] stacked along n, packed as one into the affine tiles.
    fn affine(x: *Ctx, names: []const []const u8, ns: []const usize, k: usize) !Proj {
        var buf: [192]u8 = undefined;
        var n: usize = 0;
        for (ns) |rows| n += rows;
        const specs = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k / 8 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
        var raw: [3]u64 = undefined;
        var at: usize = 0;
        for (specs, &raw) |sp, *r| {
            r.* = x.stage.ptr + at;
            for (names, ns) |name, rows| {
                const t = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, sp[0] }), sp[1], &.{ rows, sp[2] });
                if (at + t.bytes.len > x.stage.len) return error.StageTooSmall;
                try x.stage.upload(at, t.bytes);
                at += t.bytes.len;
            }
            at = (at + 255) / 256 * 256;
        }
        const q = try x.affineFrom(raw, n, k);
        try x.s.synchronize(); // the next projection's staging overwrites this one's
        return q;
    }

    /// Affine-4 tiles of words, scales and biases [n, k] already on the device.
    fn affineFrom(x: *Ctx, raw: [3]u64, n: usize, k: usize) !Proj {
        const l = cuda.qlinear.Affine4.layout(n, k);
        const q: cuda.qlinear.Affine4 = .{ .w = try x.w.alloc(x.d, l.words), .s = try x.w.alloc(x.d, l.scales), .b = try x.w.alloc(x.d, l.scales), .n = n, .k = k, .npad = l.npad };
        try x.k.affine.pack(x.s, raw[0], raw[1], raw[2], q);
        return .{ .affine4 = q };
    }

    /// NVFP4 projections [n_i, k] sharing one tensor scale, stacked along n as one `.nvfp4` lane weight.
    fn nvfp4(x: *Ctx, names: []const []const u8, ns: []const usize, k: usize) !Proj {
        var buf: [192]u8 = undefined;
        var n: usize = 0;
        for (ns) |rows| n += rows;
        const codes_bytes = n * k / 2;
        if (codes_bytes + n * k / 16 > x.stage.len) return error.StageTooSmall;
        var global: ?f32 = null;
        var at: usize = 0;
        for (names, ns) |name, rows| {
            const codes = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), .u8, &.{ rows, k / 2 });
            const blocks = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{name}), .f8_e4m3, &.{ rows, k / 16 });
            const g = try scalar(x.ck, try std.fmt.bufPrint(&buf, "{s}.weight_scale_2", .{name}));
            // weight-only: the static FP4 input scale is read (admission requires it), not applied; rows stay bf16
            _ = try scalar(x.ck, try std.fmt.bufPrint(&buf, "{s}.input_scale", .{name}));
            if (global) |v| if (v != g) return error.UnsupportedLfm3Quantization; // one scale a stacked weight
            global = g;
            try x.stage.upload(at * k / 2, codes.bytes);
            try x.stage.upload(codes_bytes + at * k / 16, blocks.bytes);
            at += rows;
        }
        const words = try x.w.alloc(x.d, cuda.modelopt.padded(n) * k / 2);
        const tiles = try x.w.alloc(x.d, cuda.modelopt.padded(n) * k / 16);
        try x.k.pack.fp4Lane(x.s, x.stage.ptr, x.stage.ptr + codes_bytes, words, tiles, n, k);
        try x.s.synchronize();
        return .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, global.?, n, k) };
    }

    /// Block FP8 projections [n_i, k] stacked along n as one `.fp8g` lane weight, a block's scale each 64 inputs.
    fn fp8(x: *Ctx, names: []const []const u8, ns: []const usize, k: usize) !Proj {
        const b = c.fp8_block;
        var buf: [192]u8 = undefined;
        var n: usize = 0;
        for (ns) |rows| n += rows;
        if (n * k > x.stage.len) return error.StageTooSmall;
        const blocks = try x.w.gpa.alloc(f32, n / b * (k / b));
        defer x.w.gpa.free(blocks);
        var at: usize = 0;
        for (names, ns) |name, rows| {
            const codes = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), .f8_e4m3, &.{ rows, k });
            const scales = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{name}), .bf16, &.{ rows / b, k / b });
            try x.stage.upload(at * k, codes.bytes);
            for (std.mem.bytesAsSlice(u16, scales.bytes), blocks[at / b * (k / b) ..][0 .. rows / b * (k / b)]) |v, *o|
                o.* = @bitCast(@as(u32, v) << 16);
            at += rows;
        }
        const words = try x.w.alloc(x.d, cuda.modelopt.padded(n) * k);
        try x.k.pack.fp8Lane(x.s, x.stage.ptr, words, n, k);
        const tiles = try x.w.gpa.alloc(f32, cuda.fp8.padded(n) * (k / cuda.fp8.group));
        defer x.w.gpa.free(tiles);
        cuda.fp8.packScales(tiles, blocks, n, k);
        const sd = try x.w.upload(x.d, std.mem.sliceAsBytes(tiles));
        try x.s.synchronize();
        return .{ .fp8g = cuda.fp8.weight(words, sd, n, k) };
    }
};

/// An fp32 tensor of one element.
fn scalar(ck: *core.checkpoint.Checkpoint, name: []const u8) !f32 {
    const t = try ck.get(name);
    if (t.dtype != .f32 or t.bytes.len != 4) return error.UnexpectedTensor;
    return @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
}

/// The text decoder's tensors against the shared admission (weights.zig): names, dtypes, shapes, bytes, nothing extra.
fn admitted(gpa: std.mem.Allocator, config: *const c.Config, ck: *const core.checkpoint.Checkpoint) !admit.Namespace {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var all: core.safetensors.Header = .empty;
    for (ck.files.items) |*f| for (f.names.keys(), f.names.values()) |name, e| try all.put(arena.allocator(), name, e);
    return (try admit.validate(gpa, config, &all)).namespace;
}

/// The MLX widths CUDA serves: affine 4 bits in groups of 64 everywhere (6 and 8 bits, g32 and VL are refused).
fn mlxServed(config: *const c.Config) bool {
    return config.quant.global.bits == 4 and config.quant.global.group == 64 and config.quant.modules.count() == 0;
}

fn named(b: []u8, prefix: []const u8, suffix: []const u8) ![]const u8 {
    return std.fmt.bufPrint(b, "{s}{s}", .{ prefix, suffix });
}

/// Every tensor checked against the admitted geometry and packed for the lane; `layers` caps them (tests).
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const kern.Kernels, dir: []const u8, layers: ?usize) !Weights {
    var config = try c.Config.read(gpa, io, dir);
    if (config.kind != .text or (config.scheme == .mlx_affine and !mlxServed(&config))) {
        config.deinit();
        return error.UnsupportedLfm3Quantization;
    }
    var w: Weights = .{ .gpa = gpa, .config = config };
    errdefer w.deinit();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const pre = (try admitted(gpa, &w.config, &ck)).prefix();
    const fp8 = w.config.scheme == .fp8_block;
    const mlx = w.config.scheme == .mlx_affine;
    if (!@import("builtin").is_test) {
        if (fp8) std.log.info("LFM3 FP8 on CUDA serves weight-only: FP8 weights, bf16 activations; the export's dynamic FP8 input quantization is not applied", .{});
        if (mlx) std.log.info("LFM3 MLX 4-bit on CUDA serves the export as stored: affine-4 weights, bf16 activations", .{});
        if (!fp8 and !mlx) std.log.info("LFM3 NVFP4 on CUDA serves weight-only: FP4 weights, bf16 activations; the export's static FP4 input scales are read and not applied", .{});
    }
    const h = c.hidden;
    var stage = try DeviceBuffer.alloc(d, stage_bytes);
    defer stage.free();
    var x: Ctx = .{ .w = &w, .d = d, .s = s, .k = k, .ck = &ck, .stage = stage };
    var b1: [128]u8 = undefined;
    var b2: [128]u8 = undefined;
    var b3: [128]u8 = undefined;
    var b4: [128]u8 = undefined;
    if (mlx) {
        var e: [3]u64 = undefined;
        e[0] = try x.tensor(try named(&b1, pre, "embed_tokens.weight"), .u32, &.{ c.vocab, h / 8 });
        e[1] = try x.tensor(try named(&b1, pre, "embed_tokens.scales"), .bf16, &.{ c.vocab, h / 64 });
        e[2] = try x.tensor(try named(&b1, pre, "embed_tokens.biases"), .bf16, &.{ c.vocab, h / 64 });
        w.embed_affine = e;
        w.head = try x.affineFrom(e, c.vocab, h); // tied: the embedding's own codes as tiles
        try s.synchronize();
    } else {
        w.embed = try x.tensor(try named(&b1, pre, "embed_tokens.weight"), .bf16, &.{ c.vocab, h });
        w.head = try x.pack(w.embed, c.vocab, h); // tied: the lane's copy of the embedding
    }
    w.final_norm = try x.tensor(try named(&b1, pre, "embedding_norm.weight"), .bf16, &.{h});
    const n = @min(layers orelse c.layers, c.layers);
    w.layers = try gpa.alloc(Layer, n);
    for (w.layers, w.config.layer_types[0..n], 0..) |*l, kind, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "{s}layers.{d}.", .{ pre, i });
        l.operator_norm = try x.tensor(try named(&b1, p, "operator_norm.weight"), .bf16, &.{h});
        l.post_operator_norm = try x.tensor(try named(&b1, p, "post_operator_norm.weight"), .bf16, &.{h});
        l.ffn_norm = try x.tensor(try named(&b1, p, "ffn_norm.weight"), .bf16, &.{h});
        l.post_ffn_norm = try x.tensor(try named(&b1, p, "post_ffn_norm.weight"), .bf16, &.{h});
        if (kind == .conv) {
            const in = try named(&b1, p, "conv.in_proj");
            const out = try named(&b2, p, "conv.out_proj");
            // MLX keeps the depthwise kernel as [channels, taps, 1], torch as [channels, 1, taps]: the same values
            const taps: []const usize = if (mlx) &.{ h, c.conv_taps, 1 } else &.{ h, 1, c.conv_taps };
            l.mixer = .{ .conv = .{
                .in = if (mlx) try x.affine(&.{in}, &.{3 * h}, h) else try x.bf16(&.{in}, &.{3 * h}, h),
                .out = if (mlx) try x.affine(&.{out}, &.{h}, h) else try x.bf16(&.{out}, &.{h}, h),
                .weight = try x.tensor(try named(&b3, p, "conv.conv.weight"), .bf16, taps),
            } };
        } else {
            const qw = c.query_heads * c.head_dim;
            const kw = c.kv_heads * c.head_dim;
            const norms = try w.alloc(d, 2 * c.head_dim * 2);
            const qn = try ck.expect(try named(&b1, p, "self_attn.q_norm.weight"), .bf16, &.{c.head_dim});
            const kn = try ck.expect(try named(&b1, p, "self_attn.k_norm.weight"), .bf16, &.{c.head_dim});
            const nb: DeviceBuffer = .{ .d = d, .ptr = norms, .len = 4 * c.head_dim, .borrowed = true };
            try nb.upload(0, qn.bytes);
            try nb.upload(2 * c.head_dim, kn.bytes);
            const names = [_][]const u8{ try named(&b1, p, "self_attn.q_proj"), try named(&b2, p, "self_attn.k_proj"), try named(&b3, p, "self_attn.v_proj"), try named(&b4, p, "self_attn.gate_proj") };
            const sliding = kind == .sliding_attention;
            var b5: [128]u8 = undefined; // names holds b1 to b4
            const o_proj = try named(&b5, p, "self_attn.o_proj");
            l.mixer = .{ .attention = if (mlx) .{
                .qkvg = try x.affine(&names, &.{ qw, kw, kw, c.query_heads }, h),
                .out = try x.affine(&.{o_proj}, &.{h}, qw),
                .norms = norms,
                .sliding = sliding,
            } else if (fp8) .{
                .qkvg = try x.fp8(names[0..3], &.{ qw, kw, kw }, h),
                .gate = try x.bf16(names[3..], &.{c.query_heads}, h),
                .out = try x.fp8(&.{o_proj}, &.{h}, qw),
                .norms = norms,
                .sliding = sliding,
            } else .{
                .qkvg = try x.bf16(&names, &.{ qw, kw, kw, c.query_heads }, h),
                .out = try x.bf16(&.{o_proj}, &.{h}, qw),
                .norms = norms,
                .sliding = sliding,
            } };
        }
        const ffn = [_][]const u8{ try named(&b1, p, "feed_forward.w1"), try named(&b2, p, "feed_forward.w3"), try named(&b3, p, "feed_forward.w2") };
        const ii = [_]usize{ c.intermediate, c.intermediate };
        l.gate_up = switch (w.config.scheme) {
            .mlx_affine => try x.affine(ffn[0..2], &ii, h),
            .fp8_block => try x.fp8(ffn[0..2], &ii, h),
            .modelopt_nvfp4 => try x.nvfp4(ffn[0..2], &ii, h),
        };
        l.down = switch (w.config.scheme) {
            .mlx_affine => try x.affine(ffn[2..], &.{h}, c.intermediate),
            .fp8_block => try x.fp8(ffn[2..], &.{h}, c.intermediate),
            .modelopt_nvfp4 => try x.nvfp4(ffn[2..], &.{h}, c.intermediate),
        };
    }
    return w;
}

test "the stage holds conv in's bf16 rows and the FFN's stacked codes and scales" {
    try std.testing.expect(stage_bytes >= 3 * c.hidden * c.hidden * 2);
    try std.testing.expect(stage_bytes >= 2 * c.intermediate * c.hidden);
    try std.testing.expect(stage_bytes >= 2 * c.intermediate * c.hidden / 2 + 2 * c.intermediate * c.hidden / 16);
    try std.testing.expect(stage_bytes >= qkvg_rows * c.hidden * 2);
    try std.testing.expectEqual(@as(usize, 3088), qkvg_rows);
}
