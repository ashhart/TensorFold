//! Source layouts for the tiny bailingmoe3 GGUF; transformed roles must not masquerade as native tensors.
const std = @import("std");
pub const Role = enum { matrix, vector, convolution, positive_exp_decay, experts, mla_key_transposed, mla_value };
pub const Name = struct { source: []const u8, native: []const u8, role: Role, rank: u8, dims: [3]u64 };
const Builder = struct {
    a: std.mem.Allocator,
    entries: std.ArrayList(Name) = .empty,
    fn add(b: *Builder, layer: ?usize, src: []const u8, dst: []const u8, dims: []const u64, role: Role) !void {
        var shape: [3]u64 = @splat(1);
        @memcpy(shape[0..dims.len], dims);
        try b.entries.append(b.a, .{ .source = if (layer) |l| try std.fmt.allocPrint(b.a, "blk.{d}.{s}", .{ l, src }) else src, .native = if (layer) |l| try std.fmt.allocPrint(b.a, "model.layers.{d}.{s}", .{ l, dst }) else dst, .role = role, .rank = @intCast(dims.len), .dims = shape });
    }
};
pub fn tiny(a: std.mem.Allocator) ![]Name {
    var b = Builder{ .a = a };
    try b.add(null, "token_embd.weight", "model.word_embeddings.weight", &.{ 1536, 157184 }, .matrix);
    try b.add(null, "output.weight", "lm_head.weight", &.{ 1536, 157184 }, .matrix);
    try b.add(null, "output_norm.weight", "model.norm.weight", &.{1536}, .vector);
    for (0..24) |l| {
        try b.add(l, "attn_norm.weight", "input_layernorm.weight", &.{1536}, .vector);
        try b.add(l, "ffn_norm.weight", "post_attention_layernorm.weight", &.{1536}, .vector);
        if (l % 4 != 3) {
            inline for (.{ "q", "k", "v" }) |part| {
                try b.add(l, "attn_" ++ part ++ ".weight", "attention." ++ part ++ "_proj.weight", &.{ 1536, 2048 }, .matrix);
                try b.add(l, "ssm_conv1d_" ++ part ++ ".weight", "attention." ++ part ++ "_conv1d.weight", &.{ 4, 1, 2048 }, .convolution);
            }
            try b.add(l, "attn_output.weight", "attention.o_proj.weight", &.{ 2048, 1536 }, .matrix);
            try b.add(l, "ssm_f_a.weight", "attention.f_proj.weight", &.{ 1536, 2048 }, .matrix);
            try b.add(l, "ssm_g_a.weight", "attention.g_proj.weight", &.{ 1536, 2048 }, .matrix);
            try b.add(l, "ssm_beta.weight", "attention.b_proj.weight", &.{ 1536, 16 }, .matrix);
            try b.add(l, "ssm_a", "attention.A_log", &.{ 1, 16 }, .positive_exp_decay);
            try b.add(l, "ssm_dt.bias", "attention.dt_bias", &.{2048}, .vector);
            try b.add(l, "ssm_norm.weight", "attention.o_norm.weight", &.{128}, .vector);
        } else {
            try b.add(l, "attn_q_a.weight", "attention.q_a_proj.weight", &.{ 1536, 256 }, .matrix);
            try b.add(l, "attn_q_a_norm.weight", "attention.q_a_layernorm.weight", &.{256}, .vector);
            try b.add(l, "attn_q_b.weight", "attention.q_b_proj.weight", &.{ 256, 3072 }, .matrix);
            try b.add(l, "attn_kv_a_mqa.weight", "attention.kv_a_proj_with_mqa.weight", &.{ 1536, 576 }, .matrix);
            try b.add(l, "attn_kv_a_norm.weight", "attention.kv_a_layernorm.weight", &.{512}, .vector);
            try b.add(l, "attn_k_b.weight", "attention.kv_b_proj.weight", &.{ 128, 512, 16 }, .mla_key_transposed);
            try b.add(l, "attn_v_b.weight", "attention.kv_b_proj.weight", &.{ 512, 128, 16 }, .mla_value);
            try b.add(l, "attn_gate.weight", "attention.g_proj.weight", &.{ 1536, 16 }, .matrix);
            try b.add(l, "attn_output.weight", "attention.dense.weight", &.{ 2048, 1536 }, .matrix);
        }
        if (l == 0) {
            inline for (.{ "gate", "up" }) |part| try b.add(l, "ffn_" ++ part ++ ".weight", "mlp." ++ part ++ "_proj.weight", &.{ 1536, 4608 }, .matrix);
            try b.add(l, "ffn_down.weight", "mlp.down_proj.weight", &.{ 4608, 1536 }, .matrix);
        } else {
            try b.add(l, "ffn_gate_inp.weight", "mlp.gate.weight", &.{ 1536, 128 }, .matrix);
            try b.add(l, "exp_probs_b.bias", "mlp.gate.expert_bias", &.{128}, .vector);
            inline for (.{ "gate", "up", "down" }) |part| {
                const down = std.mem.eql(u8, part, "down");
                const k: u64 = if (down) 512 else 1536;
                const n: u64 = if (down) 1536 else 512;
                try b.add(l, "ffn_" ++ part ++ "_exps.weight", "mlp.experts.{id}." ++ part ++ "_proj.weight", &.{ k, n, 128 }, .experts);
                try b.add(l, "ffn_" ++ part ++ "_shexp.weight", "mlp.shared_experts." ++ part ++ "_proj.weight", &.{ k, n }, .matrix);
            }
        }
    }
    return b.entries.toOwnedSlice(a);
}

test "tiny GGUF plan covers 526 unique source identities and explicit transformed roles" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const entries = try tiny(arena.allocator());
    try std.testing.expectEqual(@as(usize, 526), entries.len);
    var roles: [std.enums.values(Role).len]usize = @splat(0);
    for (entries, 0..) |n, i| {
        roles[@backingInt(n.role)] += 1;
        for (entries[0..i]) |p| try std.testing.expect(!std.mem.eql(u8, p.source, n.source));
    }
    try std.testing.expectEqual(@as(usize, 69), roles[@backingInt(Role.experts)]);
    try std.testing.expectEqual(@as(usize, 18), roles[@backingInt(Role.positive_exp_decay)]);
    try std.testing.expectEqual(@as(usize, 6), roles[@backingInt(Role.mla_key_transposed)]);
}
