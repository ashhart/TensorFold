//! Validated LFM3 text views preserve checkpoint buffers, offsets, BF16 tensors and each module's affine width.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const row = @import("../../core/row_projection.zig");
const c = @import("config.zig");
const mlx_affine = @import("../../core/quant/mlx_affine.zig");

pub const Tensor = ck.Tensor;
pub const Linear = row.Weights;
pub const Conv = struct { in: Linear, out: Linear, weight: Tensor };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, out: Linear, gate: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Dense = struct { gate: Linear, up: Linear, down: Linear };
pub const Layer = struct {
    operator_norm: Tensor,
    ffn_norm: Tensor,
    post_operator_norm: Tensor,
    post_ffn_norm: Tensor,
    mixer: union(enum) { conv: Conv, attention: Attention },
    ffn: Dense,
};
pub const Weights = struct { embedding: Linear, norm: Tensor, blocks: [c.layers]Layer };

const Loader = struct {
    checkpoint: *const ck.Checkpoint,
    config: *const c.Config,
    allocator: std.mem.Allocator,
    seen: std.StringHashMapUnmanaged(void) = .empty,

    fn tensor(self: *Loader, prefix: []const u8, suffix: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
        const t = try self.checkpoint.get(name);
        if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape))
            return error.UnexpectedLfm3Tensor;
        var bytes: usize = dtype.size();
        for (shape) |dim| bytes = std.math.mul(usize, bytes, dim) catch return error.TensorBytes;
        const end = std.math.add(usize, t.offset, t.bytes) catch return error.TensorBytes;
        if (t.bytes != bytes or end > t.buffer.length()) return error.TensorBytes;
        if (t.offset % 4 != 0) return error.UnalignedLfm3Tensor;
        try self.seen.put(self.allocator, try self.allocator.dupe(u8, name), {});
        return t;
    }

    fn affine(self: *Loader, prefix: []const u8, suffix: []const u8, n: usize, k: usize) !Linear {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
        const q = try self.config.width(name);
        const d = mlx_affine.dims(.{ .n = n, .k = k }, q.bits, q.group) catch return error.UnsupportedQuantization;
        const w = try self.tensor(name, ".weight", .u32, d.words[0..d.rank]);
        const s = try self.tensor(name, ".scales", .bf16, d.tables[0..d.rank]);
        const b = try self.tensor(name, ".biases", .bf16, d.tables[0..d.rank]);
        return row.affine(.{ .weight = w, .scales = s, .biases = b, .n = n, .k = k, .bits = q.bits, .group = q.group });
    }
};

pub fn load(ckpt: *const ck.Checkpoint, cfg: *const c.Config) !Weights {
    const bare = ckpt.has("model.embed_tokens.weight");
    const wrapped = ckpt.has("language_model.model.embed_tokens.weight");
    if (bare and wrapped) return error.MixedNamespaces;
    if (!bare and !wrapped) return error.MissingTensor;
    const prefix: []const u8 = if (bare) "model." else "language_model.model.";
    var arena = std.heap.ArenaAllocator.init(ckpt.allocator);
    defer arena.deinit();
    var loader: Loader = .{ .checkpoint = ckpt, .config = cfg, .allocator = arena.allocator() };
    var weights: Weights = .{
        .embedding = try loader.affine(prefix, "embed_tokens", c.vocab, c.hidden),
        .norm = try loader.tensor(prefix, "embedding_norm.weight", .bf16, &.{c.hidden}),
        .blocks = undefined,
    };
    for (&weights.blocks, cfg.layer_types, 0..) |*block, kind, i| {
        var buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "{s}layers.{d}.", .{ prefix, i });
        block.operator_norm = try loader.tensor(p, "operator_norm.weight", .bf16, &.{c.hidden});
        block.ffn_norm = try loader.tensor(p, "ffn_norm.weight", .bf16, &.{c.hidden});
        block.post_operator_norm = try loader.tensor(p, "post_operator_norm.weight", .bf16, &.{c.hidden});
        block.post_ffn_norm = try loader.tensor(p, "post_ffn_norm.weight", .bf16, &.{c.hidden});
        block.ffn = .{
            .gate = try loader.affine(p, "feed_forward.w1", c.intermediate, c.hidden),
            .up = try loader.affine(p, "feed_forward.w3", c.intermediate, c.hidden),
            .down = try loader.affine(p, "feed_forward.w2", c.hidden, c.intermediate),
        };
        block.mixer = if (kind == .conv) .{ .conv = .{
            .in = try loader.affine(p, "conv.in_proj", 3 * c.hidden, c.hidden),
            .out = try loader.affine(p, "conv.out_proj", c.hidden, c.hidden),
            .weight = try loader.tensor(p, "conv.conv.weight", .bf16, &.{ c.hidden, c.conv_taps, 1 }),
        } } else .{ .attention = .{
            .q = try loader.affine(p, "self_attn.q_proj", c.query_heads * c.head_dim, c.hidden),
            .k = try loader.affine(p, "self_attn.k_proj", c.kv_heads * c.head_dim, c.hidden),
            .v = try loader.affine(p, "self_attn.v_proj", c.kv_heads * c.head_dim, c.hidden),
            .out = try loader.affine(p, "self_attn.o_proj", c.hidden, c.query_heads * c.head_dim),
            .gate = try loader.affine(p, "self_attn.gate_proj", c.query_heads, c.hidden),
            .q_norm = try loader.tensor(p, "self_attn.q_norm.weight", .bf16, &.{c.head_dim}),
            .k_norm = try loader.tensor(p, "self_attn.k_norm.weight", .bf16, &.{c.head_dim}),
        } };
    }
    var iterator = ckpt.tensors.keyIterator();
    while (iterator.next()) |name| {
        if (std.mem.startsWith(u8, name.*, "model.") or std.mem.startsWith(u8, name.*, "language_model.")) {
            if (!loader.seen.contains(name.*)) return error.UnexpectedTensor;
        }
    }
    return weights;
}
