//! Gemma 4's weights on CUDA: MLX affine-4 projections packed into qlinear's tiles, norms as stored, the tied head.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cfg = @import("cuda_config.zig");
const kern = @import("cuda_kernels.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Affine4 = cuda.qlinear.Affine4;
const AffineN = cuda.qlinear.AffineN;

/// A projection at its MLX width: affine-4 tiles, or the tiles of every other width.
pub const Proj = union(enum) {
    a4: Affine4,
    an: AffineN,

    pub fn n(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.n,
        };
    }

    pub fn k(p: Proj) usize {
        return switch (p) {
            inline else => |q| q.k,
        };
    }
};

/// A layer's tensors: each projection as qlinear tiles at its width, bf16 vectors as stored (norm weights with no 1 +).
pub const Layer = struct {
    q: Proj,
    k: Proj,
    v: ?Proj, // global layers have none: v is k_proj's output (attention_k_eq_v)
    o: Proj,
    gate: Proj,
    up: Proj,
    down: Proj,
    q_norm: u64, // bf16 [head dim]
    k_norm: u64,
    input_norm: u64, // bf16 [hidden]
    post_attention: u64,
    pre_feedforward: u64,
    post_feedforward: u64,
    scalar: u64, // bf16 [1]: the residual's layer scalar
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    /// The embedding table as MLX stores it: words u32 [V, H bits / 32], scales and biases [V, H / 64].
    embed_w: u64 = 0,
    embed_bits: u8 = 4,
    embed_s: u64 = 0,
    embed_b: u64 = 0,
    /// The same table as the tied head's tiles.
    head: Proj = undefined,
    final_norm: u64 = 0,
    layers: []Layer = &.{},
    /// The 12B's image embedder (--vision), null without.
    vision: ?Vision = null,
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
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

/// The 12B's image embedder: LayerNorms, patch_dense bf16, the position table, the projection at its width (or bf16).
pub const Vision = struct {
    ln1_w: u64,
    ln1_b: u64,
    dense_w: u64,
    dense_b: u64,
    ln2_w: u64,
    ln2_b: u64,
    table: u64,
    pos_w: u64,
    pos_b: u64,
    proj: union(enum) { q: Proj, bf16: u64 },
};

/// The embedder's sizes: a merged 48-px patch's values, the output width, each axis's position rows.
pub const merged_width = 6912;
pub const vision_width = 3840;
pub const table_rows = 1120;

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

    /// An n-by-k MLX affine projection of `bits` codes (words, scales, biases) staged and packed into qlinear's tiles.
    fn linear(x: *Ctx, prefix: []const u8, n: usize, k: usize, bits: u8) !Proj {
        var buf: [192]u8 = undefined;
        const specs = [_]struct { []const u8, core.checkpoint.DType, usize }{ .{ "weight", .u32, k * bits / 32 }, .{ "scales", .bf16, k / 64 }, .{ "biases", .bf16, k / 64 } };
        var raw: [3]u64 = undefined;
        var at: usize = 0;
        for (specs, &raw) |sp, *r| {
            const t = try x.ck.expect(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, sp[0] }), sp[1], &.{ n, sp[2] });
            if (at + t.bytes.len > x.stage.len) return error.StageTooSmall;
            try x.stage.upload(at, t.bytes);
            r.* = x.stage.ptr + at;
            at += t.bytes.len;
        }
        const served = cuda.qlinear.servedBits(bits);
        const out: Proj = if (served == 4) blk: {
            const l = Affine4.layout(n, k);
            const q: Affine4 = .{ .w = try x.w.alloc(x.d, l.words), .s = try x.w.alloc(x.d, l.scales), .b = try x.w.alloc(x.d, l.scales), .n = n, .k = k, .npad = l.npad };
            try x.k.affine.pack(x.s, raw[0], raw[1], raw[2], q);
            break :blk .{ .a4 = q };
        } else blk: {
            const l = AffineN.layout(n, k, served);
            const q: AffineN = .{ .w = try x.w.alloc(x.d, l.words), .s = try x.w.alloc(x.d, l.scales), .b = try x.w.alloc(x.d, l.scales), .n = n, .k = k, .npad = l.npad, .bits = served };
            try x.k.wide.packWords(x.s, raw[0], bits, n, k, q.w);
            try x.k.wide.packTable(x.s, raw[1], n, k, q.npad, q.s);
            try x.k.wide.packTable(x.s, raw[2], n, k, q.npad, q.b);
            break :blk .{ .an = q };
        };
        try x.s.synchronize(); // the next projection's staging overwrites this one's
        return out;
    }
};

const lm = "language_model.model.";

/// Admission as Metal's (MLX affine 2-6 and 8 bits, groups of 64), every tensor checked. `layers` caps them (checks).
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const kern.Kernels, dir: []const u8, layers: ?usize, images: bool) !Weights {
    const c = try cfg.Config.read(gpa, io, dir);
    const geo = c.geo;
    var w: Weights = .{ .gpa = gpa, .config = c };
    errdefer w.deinit();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const h = geo.hidden;
    // the largest projection as stored: the tied head's table
    var stage = try DeviceBuffer.alloc(d, cfg.vocab * h * c.widths.embed.bits / 8 + 2 * cfg.vocab * (h / 64) * 2);
    defer stage.free();
    var x: Ctx = .{ .w = &w, .d = d, .s = s, .k = k, .ck = &ck, .stage = stage };
    w.embed_bits = c.widths.embed.bits;
    w.embed_w = try x.tensor(lm ++ "embed_tokens.weight", .u32, &.{ cfg.vocab, h * w.embed_bits / 32 });
    w.embed_s = try x.tensor(lm ++ "embed_tokens.scales", .bf16, &.{ cfg.vocab, h / 64 });
    w.embed_b = try x.tensor(lm ++ "embed_tokens.biases", .bf16, &.{ cfg.vocab, h / 64 });
    w.head = try x.linear(lm ++ "embed_tokens", cfg.vocab, h, w.embed_bits);
    w.final_norm = try x.tensor(lm ++ "norm.weight", .bf16, &.{h});
    const n = @min(layers orelse geo.layers, geo.layers);
    w.layers = try gpa.alloc(Layer, n);
    var buf: [192]u8 = undefined;
    for (w.layers, 0..) |*l, i| {
        const dh = cfg.headDim(i);
        const kvh = geo.kvHeads(i);
        const q_out = geo.query_heads * dh;
        const kv_out = kvh * dh;
        const p = try std.fmt.allocPrint(gpa, lm ++ "layers.{d}.", .{i});
        defer gpa.free(p);
        const name = struct {
            fn of(b: []u8, pre: []const u8, suffix: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "{s}{s}", .{ pre, suffix });
            }
        }.of;
        l.q = try x.linear(try name(&buf, p, "self_attn.q_proj"), q_out, h, c.widths.of(i, .q).bits);
        l.k = try x.linear(try name(&buf, p, "self_attn.k_proj"), kv_out, h, c.widths.of(i, .k).bits);
        l.v = if (cfg.global(i)) null else try x.linear(try name(&buf, p, "self_attn.v_proj"), kv_out, h, c.widths.of(i, .v).bits);
        l.o = try x.linear(try name(&buf, p, "self_attn.o_proj"), h, q_out, c.widths.of(i, .o).bits);
        l.gate = try x.linear(try name(&buf, p, "mlp.gate_proj"), geo.intermediate, h, c.widths.of(i, .gate).bits);
        l.up = try x.linear(try name(&buf, p, "mlp.up_proj"), geo.intermediate, h, c.widths.of(i, .up).bits);
        l.down = try x.linear(try name(&buf, p, "mlp.down_proj"), h, geo.intermediate, c.widths.of(i, .down).bits);
        l.q_norm = try x.tensor(try name(&buf, p, "self_attn.q_norm.weight"), .bf16, &.{dh});
        l.k_norm = try x.tensor(try name(&buf, p, "self_attn.k_norm.weight"), .bf16, &.{dh});
        l.input_norm = try x.tensor(try name(&buf, p, "input_layernorm.weight"), .bf16, &.{h});
        l.post_attention = try x.tensor(try name(&buf, p, "post_attention_layernorm.weight"), .bf16, &.{h});
        l.pre_feedforward = try x.tensor(try name(&buf, p, "pre_feedforward_layernorm.weight"), .bf16, &.{h});
        l.post_feedforward = try x.tensor(try name(&buf, p, "post_feedforward_layernorm.weight"), .bf16, &.{h});
        l.scalar = try x.tensor(try name(&buf, p, "layer_scalar"), .bf16, &.{1});
    }
    if (images) w.vision = try vision(&x, c);
    return w;
}

/// The image embedder's tensors; a quantized patch_dense (the bf16 GEMM's input) is refused by name.
fn vision(x: *Ctx, c: cfg.Config) !Vision {
    if (!c.geo.unified()) return error.GemmaVisionTowerNotServed;
    if (!x.ck.has("vision_embedder.patch_dense.weight")) return error.NoGemmaVision;
    if (x.ck.has("vision_embedder.patch_dense.scales")) return error.GemmaVisionDenseQuantized;
    const e = "vision_embedder.";
    const mw = merged_width;
    const vw = vision_width;
    var v: Vision = .{
        .ln1_w = try x.tensor(e ++ "patch_ln1.weight", .bf16, &.{mw}),
        .ln1_b = try x.tensor(e ++ "patch_ln1.bias", .bf16, &.{mw}),
        .dense_w = try x.tensor(e ++ "patch_dense.weight", .bf16, &.{ vw, mw }),
        .dense_b = try x.tensor(e ++ "patch_dense.bias", .bf16, &.{vw}),
        .ln2_w = try x.tensor(e ++ "patch_ln2.weight", .bf16, &.{vw}),
        .ln2_b = try x.tensor(e ++ "patch_ln2.bias", .bf16, &.{vw}),
        .table = try x.tensor(e ++ "pos_embedding", .bf16, &.{ table_rows, 2, vw }),
        .pos_w = try x.tensor(e ++ "pos_norm.weight", .bf16, &.{vw}),
        .pos_b = try x.tensor(e ++ "pos_norm.bias", .bf16, &.{vw}),
        .proj = undefined,
    };
    const pp = "embed_vision.embedding_projection";
    v.proj = if (x.ck.has(pp ++ ".scales")) .{ .q = try x.linear(pp, c.geo.hidden, vw, c.widths.vision.bits) } else .{ .bf16 = try x.tensor(pp ++ ".weight", .bf16, &.{ c.geo.hidden, vw }) };
    return v;
}
