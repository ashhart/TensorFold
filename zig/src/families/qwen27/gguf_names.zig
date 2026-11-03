//! Destination identities and shapes; GGUF decay and head order remain explicit source conventions, not native weights.
const std = @import("std");
const cfg = @import("config.zig");
pub const Role = enum { matrix, vector, convolution, negative_exp_decay };
pub const Name = struct { gguf: []const u8, native: []const u8, rank: u8, dims: [3]usize, role: Role };
const Builder = struct {
    a: std.mem.Allocator,
    names: std.ArrayList(Name) = .empty,
    fn add(b: *Builder, layer: ?usize, source: []const u8, dest: []const u8, dims: []const usize, role: Role) !void {
        const src = if (layer) |l| try std.fmt.allocPrint(b.a, "blk.{d}.{s}", .{ l, source }) else source;
        const dst = if (layer) |l| try std.fmt.allocPrint(b.a, "model.layers.{d}.{s}", .{ l, dest }) else dest;
        var shape: [3]usize = @splat(1);
        @memcpy(shape[0..dims.len], dims);
        try b.names.append(b.a, .{ .gguf = src, .native = dst, .rank = @intCast(dims.len), .dims = shape, .role = role });
    }
};
/// All allocations belong to the caller's arena; no name borrows a formatting scratch buffer.
pub fn plan(a: std.mem.Allocator, c: cfg.Config) ![]Name {
    var b = Builder{ .a = a };
    try b.add(null, "token_embd.weight", "model.embed_tokens.weight", &.{ c.vocab, c.hidden }, .matrix);
    try b.add(null, "output.weight", "lm_head.weight", &.{ c.vocab, c.hidden }, .matrix);
    try b.add(null, "output_norm.weight", "model.norm.weight", &.{c.hidden}, .vector);
    for (0..c.layers) |l| {
        try b.add(l, "attn_norm.weight", "input_layernorm.weight", &.{c.hidden}, .vector);
        try b.add(l, "post_attention_norm.weight", "post_attention_layernorm.weight", &.{c.hidden}, .vector);
        try b.add(l, "ffn_gate.weight", "mlp.gate_proj.weight", &.{ c.intermediate, c.hidden }, .matrix);
        try b.add(l, "ffn_up.weight", "mlp.up_proj.weight", &.{ c.intermediate, c.hidden }, .matrix);
        try b.add(l, "ffn_down.weight", "mlp.down_proj.weight", &.{ c.hidden, c.intermediate }, .matrix);
        if (c.kind(l) == .linear) {
            try b.add(l, "attn_qkv.weight", "linear_attn.in_proj_qkv.weight", &.{ c.gdnQkvDim(), c.hidden }, .matrix);
            try b.add(l, "attn_gate.weight", "linear_attn.in_proj_z.weight", &.{ c.gdnValueDim(), c.hidden }, .matrix);
            try b.add(l, "ssm_alpha.weight", "linear_attn.in_proj_a.weight", &.{ c.v_heads, c.hidden }, .matrix);
            try b.add(l, "ssm_beta.weight", "linear_attn.in_proj_b.weight", &.{ c.v_heads, c.hidden }, .matrix);
            try b.add(l, "ssm_out.weight", "linear_attn.out_proj.weight", &.{ c.hidden, c.gdnValueDim() }, .matrix);
            try b.add(l, "ssm_conv1d.weight", "linear_attn.conv1d.weight", &.{ c.gdnQkvDim(), c.conv_kernel, 1 }, .convolution);
            try b.add(l, "ssm_a", "linear_attn.A_log", &.{c.v_heads}, .negative_exp_decay);
            try b.add(l, "ssm_dt.bias", "linear_attn.dt_bias", &.{c.v_heads}, .vector);
            try b.add(l, "ssm_norm.weight", "linear_attn.norm.weight", &.{c.dv}, .vector);
        } else {
            try b.add(l, "attn_q.weight", "self_attn.q_proj.weight", &.{ c.qDim(), c.hidden }, .matrix);
            try b.add(l, "attn_k.weight", "self_attn.k_proj.weight", &.{ c.kvDim(), c.hidden }, .matrix);
            try b.add(l, "attn_v.weight", "self_attn.v_proj.weight", &.{ c.kvDim(), c.hidden }, .matrix);
            try b.add(l, "attn_output.weight", "self_attn.o_proj.weight", &.{ c.hidden, c.heads * c.head_dim }, .matrix);
            try b.add(l, "attn_q_norm.weight", "self_attn.q_norm.weight", &.{c.head_dim}, .vector);
            try b.add(l, "attn_k_norm.weight", "self_attn.k_norm.weight", &.{c.head_dim}, .vector);
        }
    }
    return b.names.toOwnedSlice(a);
}
