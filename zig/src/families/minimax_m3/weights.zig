//! The converted MiniMax-M3.1-Flash checkpoint as row-projection matrices: stacked projections, fp32 routers.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const rows = @import("../../core/row_projection.zig");
const cfg = @import("config.zig");

pub const Tensor = ck.Tensor;
pub const Linear = rows.Weights;

/// q | k | v | index q | index k stacked by rows (one launch), o; the four head norms (Gemma's: scaled by 1 + w).
pub const Attention = struct { qkv: Linear, o: Linear, q_norm: Tensor, k_norm: Tensor, iq_norm: Tensor, ik_norm: Tensor };
/// gate | up stacked by rows.
pub const Mlp = struct { gu: Linear, down: Linear };
/// The fp32 router [experts, hidden] and its score bias, the experts (gate | up rows each) and the shared expert.
pub const Moe = struct { router: Tensor, bias: Tensor, gu: rows.Experts, down: rows.Experts, shared: Mlp };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, attn: Attention, ffn: union(enum) { dense: Mlp, moe: Moe } };

pub const Weights = struct {
    embed: Linear,
    head: Linear,
    norm: Tensor,
    layers: []Layer,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.layers);
    }
};

fn name(buf: []u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

/// `t` is `dtype` with exactly `dims`, its bytes those dims' extent.
fn require(t: Tensor, dtype: ck.DType, dims: []const usize) error{UnexpectedMiniMaxTensor}!void {
    if (t.dtype != dtype or t.rank != dims.len or t.rank > t.shape.len) return error.UnexpectedMiniMaxTensor;
    if (!std.mem.eql(usize, t.shape[0..t.rank], dims)) return error.UnexpectedMiniMaxTensor;
    var bytes: usize = dtype.size();
    for (dims) |n| {
        if (n == 0) return error.UnexpectedMiniMaxTensor;
        bytes = std.math.mul(usize, bytes, n) catch return error.UnexpectedMiniMaxTensor;
    }
    if (t.bytes != bytes) return error.UnexpectedMiniMaxTensor;
}

fn tensor(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: ck.DType, dims: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    const n = try name(&buf, prefix, suffix);
    const t = try c.get(n);
    require(t, dtype, dims) catch |e| {
        std.log.err("minimax_m3: {s} is not {s}{any}", .{ n, @tagName(dtype), dims });
        return e;
    };
    return t;
}

/// An affine [.., n, k] matrix: bits and group read from the stored shapes, which must agree with each other.
fn affine(c: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, stack: usize, n: usize, k: usize) !Linear {
    var buf: [256]u8 = undefined;
    const base = try name(&buf, prefix, suffix);
    var b2: [256]u8 = undefined;
    const w = try c.get(try name(&b2, base, ".weight"));
    const s = try c.get(try name(&b2, base, ".scales"));
    const z = try c.get(try name(&b2, base, ".biases"));
    return checkedAffine(w, s, z, stack, n, k) catch |e| {
        std.log.err("minimax_m3: {s} is not an affine [{d}, {d}, {d}] matrix", .{ base, stack, n, k });
        return e;
    };
}

/// MLX's affine layout: u32 words [.., n, k * bits / 32], bf16 scales and biases [.., n, k / group].
pub fn checkedAffine(w: Tensor, s: Tensor, z: Tensor, stack: usize, n: usize, k: usize) !Linear {
    if (stack == 0 or n == 0 or k == 0 or n > std.math.maxInt(i32) or k > std.math.maxInt(i32)) return error.UnexpectedMiniMaxTensor;
    const rank: usize = if (stack == 1) 2 else 3;
    if (w.rank != rank or s.rank != rank or z.rank != rank or w.dtype != .u32) return error.UnexpectedMiniMaxTensor;
    const words = w.shape[rank - 1];
    const groups = s.shape[rank - 1];
    const wd = [3]usize{ stack, n, words };
    const sd = [3]usize{ stack, n, groups };
    try require(w, .u32, wd[3 - rank ..]);
    try require(s, .bf16, sd[3 - rank ..]);
    try require(z, .bf16, sd[3 - rank ..]);
    const packed_bits = std.math.mul(usize, words, 32) catch return error.UnexpectedMiniMaxTensor;
    if (groups == 0 or k % groups != 0 or packed_bits % k != 0 or packed_bits / k > std.math.maxInt(u8)) return error.UnexpectedMiniMaxTensor;
    const out = Linear{ .w = w.buffer, .w_off = w.offset, .scales = s.buffer, .s_off = s.offset, .biases = z.buffer, .b_off = z.offset, .n = n, .k = k, .group = k / groups, .bits = @intCast(packed_bits / k) };
    try out.validate();
    return out;
}

pub fn load(gpa: std.mem.Allocator, c: *const ck.Checkpoint, f: *const cfg.Config) !Weights {
    try cfg.validate(f);
    const h: usize = f.hidden;
    const w = Weights{
        .embed = try affine(c, "model.", "embed_tokens", 1, f.vocab, h),
        .head = try affine(c, "", "lm_head", 1, f.vocab, h),
        .norm = try tensor(c, "model.", "norm.weight", .bf16, &.{h}),
        .layers = try gpa.alloc(Layer, f.layers),
    };
    errdefer gpa.free(w.layers);
    const qn: usize = f.heads * f.head_dim;
    const d: usize = f.head_dim;
    for (w.layers, 0..) |*l, i| {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        var ab: [96]u8 = undefined;
        const a = try name(&ab, p, "self_attn");
        l.input_norm = try tensor(c, p, "input_layernorm.weight", .bf16, &.{h});
        l.post_norm = try tensor(c, p, "post_attention_layernorm.weight", .bf16, &.{h});
        l.attn = .{
            .qkv = try affine(c, a, ".qkv_proj", 1, f.qkvRows(), h),
            .o = try affine(c, a, ".o_proj", 1, h, qn),
            .q_norm = try tensor(c, a, ".q_norm.weight", .bf16, &.{d}),
            .k_norm = try tensor(c, a, ".k_norm.weight", .bf16, &.{d}),
            .iq_norm = try tensor(c, a, ".index_q_norm.weight", .bf16, &.{f.index_dim}),
            .ik_norm = try tensor(c, a, ".index_k_norm.weight", .bf16, &.{f.index_dim}),
        };
        var mb: [96]u8 = undefined;
        const m = try name(&mb, p, "mlp");
        if (!f.layer[i].moe) {
            l.ffn = .{ .dense = .{
                .gu = try affine(c, m, ".gate_up_proj", 1, 2 * @as(usize, f.dense_intermediate), h),
                .down = try affine(c, m, ".down_proj", 1, h, f.dense_intermediate),
            } };
            continue;
        }
        const e: usize = f.experts;
        var sb: [128]u8 = undefined;
        const sh = try name(&sb, m, ".shared_experts");
        var xb: [128]u8 = undefined;
        const sw = try name(&xb, m, ".switch_mlp");
        l.ffn = .{ .moe = .{
            .router = try tensor(c, m, ".gate.weight", .f32, &.{ e, h }),
            .bias = try tensor(c, m, ".e_score_correction_bias", .f32, &.{e}),
            .gu = .{ .one = try affine(c, sw, ".gate_up_proj", e, 2 * @as(usize, f.moe_intermediate), h), .experts = e, .repeat = 1 },
            .down = .{ .one = try affine(c, sw, ".down_proj", e, h, f.moe_intermediate), .experts = e, .repeat = 1 },
            .shared = .{
                .gu = try affine(c, sh, ".gate_up_proj", 1, 2 * @as(usize, f.shared_intermediate), h),
                .down = try affine(c, sh, ".down_proj", 1, h, f.shared_intermediate),
            },
        } };
    }
    return w;
}
