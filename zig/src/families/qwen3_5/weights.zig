//! Views into one shared checkpoint buffer: the output projection aliases the packed input embedding.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const c = @import("config.zig");

pub const Tensor = ck.Tensor;
pub const Linear = struct { weight: Tensor, scales: Tensor, biases: Tensor, inputs: usize, outputs: usize };
pub const Delta = struct { qkv: Linear, z: Linear, a: Linear, b: Linear, out: Linear, conv: Tensor, norm: Tensor, a_log: Tensor, dt_bias: Tensor };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

pub const Weights = struct {
    embedding: Linear,
    norm: Tensor,
    blocks: [c.layers]Layer,

    pub fn head(self: *const Weights) Linear {
        return self.embedding;
    }
};

fn expect(ckpt: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    const t = try ckpt.get(name);
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
        std.log.err("{s}: unexpected Qwen tensor dtype or shape", .{name});
        return error.UnexpectedQwenTensor;
    }
    return t;
}

fn tensor(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    return expect(ckpt, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }), dtype, shape);
}

fn projection(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, outputs: usize, inputs: usize) !Linear {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
    return .{
        .weight = try tensor(ckpt, name, ".weight", .u32, &.{ outputs, inputs / 8 }),
        .scales = try tensor(ckpt, name, ".scales", .bf16, &.{ outputs, inputs / c.group }),
        .biases = try tensor(ckpt, name, ".biases", .bf16, &.{ outputs, inputs / c.group }),
        .inputs = inputs,
        .outputs = outputs,
    };
}

pub fn load(ckpt: *const ck.Checkpoint) !Weights {
    const prefix = "language_model.model.";
    var w = Weights{
        .embedding = try projection(ckpt, prefix, "embed_tokens", c.vocab, c.hidden),
        .norm = try tensor(ckpt, prefix, "norm.weight", .bf16, &.{c.hidden}),
        .blocks = undefined,
    };
    if (ckpt.has("language_model.lm_head.weight")) return error.UnexpectedUntiedQwenHead;
    for (&w.blocks, 0..) |*block, i| {
        var buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "{s}layers.{d}.", .{ prefix, i });
        block.* = .{
            .input_norm = try tensor(ckpt, p, "input_layernorm.weight", .bf16, &.{c.hidden}),
            .post_norm = try tensor(ckpt, p, "post_attention_layernorm.weight", .bf16, &.{c.hidden}),
            .gate = try projection(ckpt, p, "mlp.gate_proj", c.intermediate, c.hidden),
            .up = try projection(ckpt, p, "mlp.up_proj", c.intermediate, c.hidden),
            .down = try projection(ckpt, p, "mlp.down_proj", c.hidden, c.intermediate),
            .mixer = undefined,
        };
        if (c.linear(i)) {
            block.mixer = .{ .delta = .{
                .qkv = try projection(ckpt, p, "linear_attn.in_proj_qkv", c.conv_dim, c.hidden),
                .z = try projection(ckpt, p, "linear_attn.in_proj_z", c.hidden, c.hidden),
                .a = try projection(ckpt, p, "linear_attn.in_proj_a", c.linear_heads, c.hidden),
                .b = try projection(ckpt, p, "linear_attn.in_proj_b", c.linear_heads, c.hidden),
                .out = try projection(ckpt, p, "linear_attn.out_proj", c.hidden, c.hidden),
                .conv = try tensor(ckpt, p, "linear_attn.conv1d.weight", .bf16, &.{ c.conv_dim, c.conv_taps, 1 }),
                .norm = try tensor(ckpt, p, "linear_attn.norm.weight", .bf16, &.{c.linear_dim}),
                .a_log = try tensor(ckpt, p, "linear_attn.A_log", .f32, &.{c.linear_heads}),
                .dt_bias = try tensor(ckpt, p, "linear_attn.dt_bias", .bf16, &.{c.linear_heads}),
            } };
        } else {
            block.mixer = .{ .attention = .{
                .q = try projection(ckpt, p, "self_attn.q_proj", 2 * c.hidden, c.hidden),
                .k = try projection(ckpt, p, "self_attn.k_proj", c.kv_heads * c.head_dim, c.hidden),
                .v = try projection(ckpt, p, "self_attn.v_proj", c.kv_heads * c.head_dim, c.hidden),
                .out = try projection(ckpt, p, "self_attn.o_proj", c.hidden, c.hidden),
                .q_norm = try tensor(ckpt, p, "self_attn.q_norm.weight", .bf16, &.{c.head_dim}),
                .k_norm = try tensor(ckpt, p, "self_attn.k_norm.weight", .bf16, &.{c.head_dim}),
            } };
        }
    }
    return w;
}
