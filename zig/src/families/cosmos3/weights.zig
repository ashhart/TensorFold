//! Views into the checkpoint's buffers: the text model's projections and norms, the image tower's and the projector's.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const row = @import("../../core/row_projection.zig");
const c = @import("config.zig");

pub const Tensor = ck.Tensor;
pub const Linear = row.Weights;
/// A projection with a bias (the tower's and the projector's).
pub const Biased = struct { w: Linear, bias: Tensor };
pub const Norm = struct { w: Tensor, b: Tensor };

pub const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    fc1: Linear,
    fc2: Linear,
};

pub const VisionLayer = struct {
    ln1: Norm,
    ln2: Norm,
    q: Biased,
    k: Biased,
    v: Biased,
    out: Biased,
    fc1: Biased,
    fc2: Biased,
};

pub const Vision = struct {
    patch: Biased,
    position: Tensor, // [16 * 16, 1152] bf16
    layers: [c.v_layers]VisionLayer,
    post: Norm,
    merge_norm: Norm,
    fc1: Biased,
    fc2: Biased,
};

pub const Weights = struct {
    embedding: Linear,
    head: Linear,
    norm: Tensor,
    layers: [c.layers]Layer,
    vision: Vision,
};

fn expect(ckpt: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    const t = try ckpt.get(name);
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape) or t.offset % 4 != 0) {
        std.log.err("{s}: unexpected Cosmos3 tensor dtype, shape or alignment", .{name});
        return error.UnexpectedCosmosTensor;
    }
    return t;
}

fn tensor(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    return expect(ckpt, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }), dtype, shape);
}

/// A matrix [n, k] of `bits`-wide codes in groups of 64.
fn affine(ckpt: *const ck.Checkpoint, prefix: []const u8, name: []const u8, n: usize, k: usize, bits: u8) !Linear {
    var buf: [256]u8 = undefined;
    const base = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name });
    const w = try tensor(ckpt, base, ".weight", .u32, &.{ n, k * bits / 32 });
    const s = try tensor(ckpt, base, ".scales", .bf16, &.{ n, k / c.group });
    const b = try tensor(ckpt, base, ".biases", .bf16, &.{ n, k / c.group });
    const out: Linear = .{ .w = w.buffer, .w_off = w.offset, .scales = s.buffer, .s_off = s.offset, .biases = b.buffer, .b_off = b.offset, .n = n, .k = k, .group = c.group, .bits = bits };
    try out.validate();
    return out;
}

fn biased(ckpt: *const ck.Checkpoint, prefix: []const u8, name: []const u8, n: usize, k: usize, bits: u8) !Biased {
    var buf: [256]u8 = undefined;
    const base = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name });
    return .{ .w = try affine(ckpt, prefix, name, n, k, bits), .bias = try tensor(ckpt, base, ".bias", .bf16, &.{n}) };
}

fn norm(ckpt: *const ck.Checkpoint, prefix: []const u8, name: []const u8, width: usize) !Norm {
    var buf: [256]u8 = undefined;
    const base = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name });
    return .{ .w = try tensor(ckpt, base, ".weight", .bf16, &.{width}), .b = try tensor(ckpt, base, ".bias", .bf16, &.{width}) };
}

pub fn load(ckpt: *const ck.Checkpoint, config: *const c.Config) !Weights {
    const bits = config.bits;
    var w: Weights = undefined;
    w.embedding = try affine(ckpt, "model.", "embed_tokens", c.vocab, c.hidden, bits);
    if (w.embedding.bits != 8) return error.UnsupportedCosmosEmbedding; // cos3_embed_b8 reads 8-bit codes
    w.head = try affine(ckpt, "", "lm_head", c.vocab, c.hidden, bits);
    w.norm = try tensor(ckpt, "model.", "norm.weight", .bf16, &.{c.hidden});
    const qd = c.query_heads * c.head_dim;
    const kvd = c.kv_heads * c.head_dim;
    for (&w.layers, 0..) |*l, i| {
        var buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "model.layers.{d}.", .{i});
        l.input_norm = try tensor(ckpt, p, "input_layernorm.weight", .bf16, &.{c.hidden});
        l.post_norm = try tensor(ckpt, p, "post_attention_layernorm.weight", .bf16, &.{c.hidden});
        l.q = try affine(ckpt, p, "self_attn.q_proj", qd, c.hidden, bits);
        l.k = try affine(ckpt, p, "self_attn.k_proj", kvd, c.hidden, bits);
        l.v = try affine(ckpt, p, "self_attn.v_proj", kvd, c.hidden, bits);
        l.o = try affine(ckpt, p, "self_attn.o_proj", c.hidden, qd, bits);
        l.fc1 = try affine(ckpt, p, "mlp.fc1", c.intermediate, c.hidden, bits);
        l.fc2 = try affine(ckpt, p, "mlp.fc2", c.hidden, c.intermediate, bits);
    }
    const v = &w.vision;
    v.patch = try biased(ckpt, "vision.", "patch_embed", c.v_hidden, c.patch_width, bits);
    v.position = try tensor(ckpt, "vision.", "position", .bf16, &.{ c.grid * c.grid, c.v_hidden });
    for (&v.layers, 0..) |*l, i| {
        var buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "vision.layers.{d}.", .{i});
        l.ln1 = try norm(ckpt, p, "layer_norm1", c.v_hidden);
        l.ln2 = try norm(ckpt, p, "layer_norm2", c.v_hidden);
        l.q = try biased(ckpt, p, "attn.q_proj", c.v_hidden, c.v_hidden, bits);
        l.k = try biased(ckpt, p, "attn.k_proj", c.v_hidden, c.v_hidden, bits);
        l.v = try biased(ckpt, p, "attn.v_proj", c.v_hidden, c.v_hidden, bits);
        l.out = try biased(ckpt, p, "attn.out_proj", c.v_hidden, c.v_hidden, bits);
        l.fc1 = try biased(ckpt, p, "mlp.fc1", c.v_intermediate, c.v_hidden, bits);
        l.fc2 = try biased(ckpt, p, "mlp.fc2", c.v_hidden, c.v_intermediate, bits);
    }
    v.post = try norm(ckpt, "vision.", "post_layernorm", c.v_hidden);
    v.merge_norm = try norm(ckpt, "projector.", "norm", c.v_hidden);
    v.fc1 = try biased(ckpt, "projector.", "fc1", c.projector_hidden, c.v_hidden * c.merge * c.merge, bits);
    v.fc2 = try biased(ckpt, "projector.", "fc2", c.hidden, c.projector_hidden, bits);
    return w;
}
