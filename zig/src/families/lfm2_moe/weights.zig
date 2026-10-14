//! Views into the checkpoint's buffers: row-kernel projections, experts back to back, the tied 6- or 8-bit embedding.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const row = @import("../../core/row_projection.zig");
const c = @import("config.zig");

pub const Tensor = ck.Tensor;
pub const Linear = row.Weights;
pub const Conv = struct { in: Linear, out: Linear, weight: Tensor };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Dense = struct { gate: Linear, up: Linear, down: Linear };
pub const Moe = struct { router: Tensor, bias: Tensor, gate: row.Experts, up: row.Experts, down: row.Experts };
pub const Layer = struct {
    operator_norm: Tensor,
    ffn_norm: Tensor,
    mixer: union(enum) { conv: Conv, attention: Attention },
    ffn: union(enum) { dense: Dense, moe: Moe },
};

pub const Weights = struct {
    /// The embedding is also the head (tie_word_embeddings).
    embedding: Linear,
    norm: Tensor,
    blocks: [c.layers]Layer,
};

fn expect(ckpt: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    const t = try ckpt.get(name);
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape) or t.offset % 4 != 0) {
        std.log.err("{s}: unexpected LFM2 tensor dtype, shape or alignment", .{name});
        return error.UnexpectedLfmTensor;
    }
    return t;
}

fn tensor(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    return expect(ckpt, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }), dtype, shape);
}

/// `count` matrices [n, k] (count 0: one matrix of rank 2) of `bits`-wide codes as the row kernels read them.
fn affine(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, count: usize, n: usize, k: usize, bits: u8) !Linear {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
    const lead: []const usize = if (count == 0) &.{} else &.{count};
    var shape: [3]usize = undefined;
    @memcpy(shape[0..lead.len], lead);
    const r = lead.len + 2;
    shape[lead.len] = n;
    shape[lead.len + 1] = k * bits / 32;
    const w = try tensor(ckpt, name, ".weight", .u32, shape[0..r]);
    shape[lead.len + 1] = k / c.group;
    const s = try tensor(ckpt, name, ".scales", .bf16, shape[0..r]);
    const b = try tensor(ckpt, name, ".biases", .bf16, shape[0..r]);
    const out: Linear = .{ .w = w.buffer, .w_off = w.offset, .scales = s.buffer, .s_off = s.offset, .biases = b.buffer, .b_off = b.offset, .n = n, .k = k, .group = c.group, .bits = bits };
    try out.validate();
    return out;
}

fn experts(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, n: usize, k: usize, bits: u8, repeat: usize) !row.Experts {
    const out: row.Experts = .{ .one = try affine(ckpt, prefix, suffix, c.experts, n, k, bits), .experts = c.experts, .repeat = repeat };
    try out.validate();
    return out;
}

pub fn load(ckpt: *const ck.Checkpoint, config: *const c.Config) !Weights {
    const bits = config.bits;
    var w = Weights{
        .embedding = try affine(ckpt, "model.", "embed_tokens", 0, c.vocab, c.hidden, config.embed_bits),
        .norm = try tensor(ckpt, "model.", "embedding_norm.weight", .bf16, &.{c.hidden}),
        .blocks = undefined,
    };
    for (&w.blocks, 0..) |*block, i| {
        var buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "model.layers.{d}.", .{i});
        block.operator_norm = try tensor(ckpt, p, "operator_norm.weight", .bf16, &.{c.hidden});
        block.ffn_norm = try tensor(ckpt, p, "ffn_norm.weight", .bf16, &.{c.hidden});
        block.mixer = if (config.attention[i]) .{ .attention = .{
            .q = try affine(ckpt, p, "self_attn.q_proj", 0, c.query_heads * c.head_dim, c.hidden, bits),
            .k = try affine(ckpt, p, "self_attn.k_proj", 0, c.kv_heads * c.head_dim, c.hidden, bits),
            .v = try affine(ckpt, p, "self_attn.v_proj", 0, c.kv_heads * c.head_dim, c.hidden, bits),
            .out = try affine(ckpt, p, "self_attn.out_proj", 0, c.hidden, c.query_heads * c.head_dim, bits),
            .q_norm = try tensor(ckpt, p, "self_attn.q_layernorm.weight", .bf16, &.{c.head_dim}),
            .k_norm = try tensor(ckpt, p, "self_attn.k_layernorm.weight", .bf16, &.{c.head_dim}),
        } } else .{ .conv = .{
            .in = try affine(ckpt, p, "conv.in_proj", 0, 3 * c.hidden, c.hidden, bits),
            .out = try affine(ckpt, p, "conv.out_proj", 0, c.hidden, c.hidden, bits),
            .weight = try tensor(ckpt, p, "conv.conv.weight", .bf16, &.{ c.hidden, c.conv_taps, 1 }),
        } };
        block.ffn = if (config.moe(i)) .{ .moe = .{
            .router = try tensor(ckpt, p, "feed_forward.gate.weight", .f32, &.{ c.experts, c.hidden }),
            .bias = try tensor(ckpt, p, "feed_forward.expert_bias", .f32, &.{c.experts}),
            .gate = try experts(ckpt, p, "feed_forward.switch_mlp.gate_proj", c.moe_intermediate, c.hidden, bits, c.topk),
            .up = try experts(ckpt, p, "feed_forward.switch_mlp.up_proj", c.moe_intermediate, c.hidden, bits, c.topk),
            .down = try experts(ckpt, p, "feed_forward.switch_mlp.down_proj", c.hidden, c.moe_intermediate, bits, 1),
        } } else .{ .dense = .{
            .gate = try affine(ckpt, p, "feed_forward.gate_proj", 0, c.intermediate, c.hidden, bits),
            .up = try affine(ckpt, p, "feed_forward.up_proj", 0, c.intermediate, c.hidden, bits),
            .down = try affine(ckpt, p, "feed_forward.down_proj", 0, c.hidden, c.intermediate, bits),
        } };
    }
    return w;
}
