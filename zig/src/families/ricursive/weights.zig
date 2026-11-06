//! Ripple's checkpoint on Metal: tensors as stored (bf16, or the NVFP4 release's MLPs), nothing converted.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");

pub const Tensor = ck.Tensor;

/// A bf16 [n, k] matrix as the release stores it, at `off` in `buf`.
pub const Dense = struct { buf: mtl.Buffer, off: usize, n: usize, k: usize };

pub const Attention = struct {
    q: Dense,
    k: Dense,
    v: Dense,
    g: Dense, // the output gate's projection
    o: Dense,
    q_norm: Tensor,
    k_norm: Tensor,
    /// sigmoid(value_res_lambda) rounded to bf16, as the release blends the layer-0 embeddings into v's input.
    blend: f32,
};
/// An NVFP4 [n, k] matrix as ModelOpt stores it: nibbles [n, k / 2], E4M3 scales [n, k / 16] and the fp32 global.
pub const Fp4 = struct { w: mtl.Buffer, w_off: usize, s: mtl.Buffer, s_off: usize, n: usize, k: usize, global: f32 };
/// An MLP projection: the release's bf16 rows, or the publisher's NVFP4.
pub const Proj = union(enum) { bf16: Dense, fp4: Fp4 };
pub const Mlp = struct { gate: Proj, up: Proj, down: Proj };
pub const Layer = struct {
    input_norm: Tensor,
    post_attn_norm: Tensor,
    pre_ffn_norm: Tensor,
    post_ffn_norm: Tensor,
    attn: Attention,
    mlp: Mlp,
};

pub const Weights = struct {
    embed: Dense,
    head: Dense,
    norm: Tensor,
    layers: []Layer,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.layers);
    }
};

/// `t` is `dtype` with exactly `dims`, its bytes those dims' extent.
fn require(t: Tensor, dtype: @TypeOf(t.dtype), dims: []const usize) error{UnexpectedRicursiveTensor}!void {
    if (t.dtype != dtype or t.rank != dims.len or t.rank > t.shape.len) return error.UnexpectedRicursiveTensor;
    if (!std.mem.eql(usize, t.shape[0..t.rank], dims)) return error.UnexpectedRicursiveTensor;
    var bytes: usize = switch (dtype) {
        .bf16 => 2,
        .f32 => 4,
        else => 1,
    };
    for (dims) |n| bytes = std.math.mul(usize, bytes, n) catch return error.UnexpectedRicursiveTensor;
    if (t.bytes != bytes) return error.UnexpectedRicursiveTensor;
}

/// The bf16 tensor `prefix ++ suffix` of exactly `dims`.
pub fn get(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dims: []const usize) !Tensor {
    return getAs(c, prefix, suffix, .bf16, dims);
}

/// The `dtype` tensor `prefix ++ suffix` of exactly `dims`.
fn getAs(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: anytype, dims: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    const full = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
    const t = try c.get(full);
    require(t, dtype, dims) catch |err| {
        std.log.err("ricursive: {s} is not {s}{any}", .{ full, @tagName(dtype), dims });
        return err;
    };
    return t;
}

/// The MLP projection `prefix ++ name`: bf16 rows, or NVFP4 (the row kernels read 8-byte blocks of 16 nibbles).
fn proj(c: *const ck.Checkpoint, f: *const cfg.Config, prefix: []const u8, comptime name: []const u8, n: usize, k: usize) !Proj {
    if (!f.fp4_mlp) return .{ .bf16 = try dense(c, prefix, name ++ ".weight", n, k) };
    if (k % 256 != 0) return error.UnexpectedRicursiveTensor;
    const w = try getAs(c, prefix, name ++ ".weight", .u8, &.{ n, k / 2 });
    const s = try getAs(c, prefix, name ++ ".weight_scale", .f8_e4m3, &.{ n, k / 16 });
    const g = try getAs(c, prefix, name ++ ".weight_scale_2", .f32, &.{});
    if (w.offset % 8 != 0) return error.UnexpectedRicursiveTensor;
    const global = g.host(f32)[0];
    if (!std.math.isFinite(global) or global <= 0) return error.UnexpectedRicursiveTensor;
    return .{ .fp4 = .{ .w = w.buffer, .w_off = w.offset, .s = s.buffer, .s_off = s.offset, .n = n, .k = k, .global = global } };
}

/// The release's bf16 [n, k] matrix in place: k whole 256s for the GEMM's steps, 8-byte aligned for the row kernel.
pub fn dense(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, n: usize, k: usize) !Dense {
    const w = try get(c, prefix, suffix, &.{ n, k });
    if (k % 256 != 0 or w.offset % 8 != 0) {
        std.log.err("ricursive: {s}{s} at byte {d}: the row kernels need 8-byte rows of whole 256s", .{ prefix, suffix, w.offset });
        return error.UnexpectedRicursiveTensor;
    }
    return .{ .buf = w.buffer, .off = w.offset, .n = n, .k = k };
}

/// torch's bf16 sigmoid of a bf16 scalar: fp32 1 / (1 + exp(-x)), rounded to bf16 (to nearest, ties to even).
pub fn blendOf(lambda: u16) f32 {
    const x: f32 = @bitCast(@as(u32, lambda) << 16);
    const s: f32 = 1.0 / (1.0 + @exp(-x));
    const bits: u32 = @bitCast(s);
    const rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
    return @bitCast(rounded);
}

pub fn load(gpa: std.mem.Allocator, c: *const ck.Checkpoint, f: *const cfg.Config) !Weights {
    const h = f.hidden;
    var w = Weights{
        .embed = try dense(c, "model.", "embed_tokens.weight", f.vocab, h),
        .head = try dense(c, "", "lm_head.weight", f.vocab, h),
        .norm = try get(c, "model.", "norm.weight", &.{h}),
        .layers = try gpa.alloc(Layer, f.layers),
    };
    errdefer w.deinit(gpa);
    for (w.layers, 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        var ab: [96]u8 = undefined;
        const a = try std.fmt.bufPrint(&ab, "{s}self_attn.", .{p});
        var mb: [96]u8 = undefined;
        const m = try std.fmt.bufPrint(&mb, "{s}mlp.", .{p});
        const lambda = try get(c, a, "value_res_lambda", &.{1});
        l.* = .{
            .input_norm = try get(c, p, "input_layernorm.weight", &.{h}),
            .post_attn_norm = try get(c, p, "post_attention_layernorm.weight", &.{h}),
            .pre_ffn_norm = try get(c, p, "pre_feedforward_layernorm.weight", &.{h}),
            .post_ffn_norm = try get(c, p, "post_feedforward_layernorm.weight", &.{h}),
            .attn = .{
                .q = try dense(c, a, "q_proj.weight", f.queryWidth(), h),
                .k = try dense(c, a, "k_proj.weight", f.kvWidth(), h),
                .v = try dense(c, a, "v_proj.weight", f.kvWidth(), h),
                .g = try dense(c, a, "g_proj.weight", f.queryWidth(), h),
                .o = try dense(c, a, "o_proj.weight", h, f.queryWidth()),
                .q_norm = try get(c, a, "q_norm.weight", &.{cfg.head_dim}),
                .k_norm = try get(c, a, "k_norm.weight", &.{cfg.head_dim}),
                .blend = blendOf(lambda.host(u16)[0]),
            },
            .mlp = .{
                .gate = try proj(c, f, m, "gate_proj", f.intermediate, h),
                .up = try proj(c, f, m, "up_proj", f.intermediate, h),
                .down = try proj(c, f, m, "down_proj", h, f.intermediate),
            },
        };
    }
    return w;
}

test "ricursive weights: the blend is torch's bf16 sigmoid of the stored lambda" {
    try std.testing.expectEqual(@as(f32, 0.5), blendOf(0x0000)); // lambda 0: one half
    try std.testing.expectEqual(@as(f32, 0.73046875), blendOf(0x3f80)); // lambda 1: 0.7310586 rounds to 0.73046875
    try std.testing.expectEqual(@as(f32, 0.26953125), blendOf(0xbf80)); // lambda -1: 0.2689414 rounds to 0.26953125
}
