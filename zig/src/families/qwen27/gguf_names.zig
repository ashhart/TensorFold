//! Header-checked text tensor mapping; GGUF value heads are restored before the native loader owns fusions.
const std = @import("std");
const gg = @import("../../core/gguf.zig");
const cfg = @import("config.zig");
pub const Transform = enum { affine, bf16, ssm_a, f32 };
pub const Entry = struct { tensor: gg.Tensor, name: []const u8, shape: []const usize, transform: Transform, layer: ?usize, heads: ?@import("gguf_heads.zig").Order = null };
/// The caller supplies an arena that owns the entries, their names and their shapes.
pub fn plan(a: std.mem.Allocator, d: *const gg.Document, c: cfg.Config) ![]Entry {
    var b = Builder{ .a = a, .d = d, .c = c };
    try b.add("token_embd.weight", "language_model.model.embed_tokens", &.{ c.vocab, c.hidden }, .affine, null);
    try b.add("output_norm.weight", "language_model.model.norm.weight", &.{c.hidden}, .f32, null);
    try b.add("output.weight", "language_model.lm_head", &.{ c.vocab, c.hidden }, .affine, null);
    for (0..c.layers) |i| {
        try b.layer(i, "attn_norm.weight", "input_layernorm.weight", &.{c.hidden}, .f32);
        try b.layer(i, "post_attention_norm.weight", "post_attention_layernorm.weight", &.{c.hidden}, .f32);
        try b.layer(i, "ffn_gate.weight", "mlp.gate_proj", &.{ c.intermediate, c.hidden }, .affine);
        try b.layer(i, "ffn_up.weight", "mlp.up_proj", &.{ c.intermediate, c.hidden }, .affine);
        try b.layer(i, "ffn_down.weight", "mlp.down_proj", &.{ c.hidden, c.intermediate }, .affine);
        if (c.kind(i) == .linear) {
            try b.layer(i, "attn_qkv.weight", "linear_attn.in_proj_qkv", &.{ c.gdnQkvDim(), c.hidden }, .affine);
            try b.layer(i, "attn_gate.weight", "linear_attn.in_proj_z", &.{ c.gdnValueDim(), c.hidden }, .affine);
            try b.layer(i, "ssm_beta.weight", "linear_attn.in_proj_b", &.{ c.v_heads, c.hidden }, .affine);
            try b.layer(i, "ssm_alpha.weight", "linear_attn.in_proj_a", &.{ c.v_heads, c.hidden }, .affine);
            try b.layer(i, "ssm_out.weight", "linear_attn.out_proj", &.{ c.hidden, c.gdnValueDim() }, .affine);
            try b.layer(i, "ssm_conv1d.weight", "linear_attn.conv1d.weight", &.{ c.gdnQkvDim(), c.conv_kernel, 1 }, .bf16);
            try b.layer(i, "ssm_a", "linear_attn.ssm_a", &.{c.v_heads}, .ssm_a);
            try b.layer(i, "ssm_dt.bias", "linear_attn.dt_bias", &.{c.v_heads}, .bf16);
            try b.layer(i, "ssm_norm.weight", "linear_attn.norm.weight", &.{c.dv}, .bf16);
        } else {
            try b.layer(i, "attn_q.weight", "self_attn.q_proj", &.{ c.qDim(), c.hidden }, .affine);
            try b.layer(i, "attn_k.weight", "self_attn.k_proj", &.{ c.kvDim(), c.hidden }, .affine);
            try b.layer(i, "attn_v.weight", "self_attn.v_proj", &.{ c.kvDim(), c.hidden }, .affine);
            try b.layer(i, "attn_output.weight", "self_attn.o_proj", &.{ c.hidden, c.heads * c.head_dim }, .affine);
            try b.layer(i, "attn_q_norm.weight", "self_attn.q_norm.weight", &.{c.head_dim}, .f32);
            try b.layer(i, "attn_k_norm.weight", "self_attn.k_norm.weight", &.{c.head_dim}, .f32);
        }
    }
    if (d.tensors.len != b.entries.items.len) return error.UnmappedGgufTensor;
    return b.entries.toOwnedSlice(a);
}
const Builder = struct {
    a: std.mem.Allocator,
    d: *const gg.Document,
    c: cfg.Config,
    entries: std.ArrayList(Entry) = .empty,
    fn layer(b: *Builder, i: usize, src: []const u8, dst: []const u8, shape: []const usize, transform: Transform) !void {
        try b.add(try std.fmt.allocPrint(b.a, "blk.{d}.{s}", .{ i, src }), try std.fmt.allocPrint(b.a, "language_model.model.layers.{d}.{s}", .{ i, dst }), shape, transform, i);
        const Kind = enum { scalar, value, qkv, out };
        const kind = std.StaticStringMap(Kind).initComptime(.{
            .{ "ssm_a", .scalar },            .{ "ssm_dt.bias", .scalar },
            .{ "ssm_alpha.weight", .scalar }, .{ "ssm_beta.weight", .scalar },
            .{ "attn_gate.weight", .value },  .{ "attn_qkv.weight", .qkv },
            .{ "ssm_conv1d.weight", .qkv },   .{ "ssm_out.weight", .out },
        }).get(src) orelse return;
        b.entries.items[b.entries.items.len - 1].heads = .{
            .axis = if (kind == .out) .columns else .rows,
            .key_heads = b.c.k_heads,
            .value_heads = b.c.v_heads,
            .width = if (kind == .scalar) 1 else b.c.dv,
            .offset = if (kind == .qkv) 2 * b.c.k_heads * b.c.dk else 0,
        };
    }
    fn add(b: *Builder, src: []const u8, dst: []const u8, shape: []const usize, transform: Transform, layer_index: ?usize) !void {
        const t = b.d.getTensor(src) orelse return error.MissingGgufTensor;
        const source_shape = if (shape.len == 3) shape[0..2] else shape;
        if (t.rank != source_shape.len or (transform != .affine and t.kind != .f32)) return error.BadGgufTensor;
        for (source_shape, 0..) |n, i| if (t.shape[t.rank - 1 - i] != n) return error.BadGgufTensor;
        try b.entries.append(b.a, .{ .tensor = t, .name = try b.a.dupe(u8, dst), .shape = try b.a.dupe(usize, shape), .transform = transform, .layer = layer_index });
    }
};
