//! SenseNova router ownership and text tensor selection.
const std = @import("std");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");

pub const Routers = struct { attn_weight: u64, attn_bias: u64, ffn_weight: u64, ffn_bias: u64 };

pub fn active(g: geometry.Geometry, i: usize) bool {
    const c = g.loop orelse return false;
    const m = c.mhc orelse return false;
    return i >= m.start and i < m.end;
}

pub fn bytes(g: geometry.Geometry, i: usize) usize {
    if (!active(g, i)) return 0;
    const aligned = @import("cuda_engine.zig").aligned;
    return 2 * (aligned(8 * 2 * g.hidden * 4) + aligned(8 * 4));
}

pub fn load(e: anytype, ck: anytype, prefix: []const u8) !Routers {
    const shape = .{ 8, 2 * e.g.hidden };
    return .{
        .attn_weight = try weights.tensor(e, ck, "{s}mhc_attn.weight", .{prefix}, .f32, &shape),
        .attn_bias = try weights.tensor(e, ck, "{s}mhc_attn.bias", .{prefix}, .f32, &.{8}),
        .ffn_weight = try weights.tensor(e, ck, "{s}mhc_ffn.weight", .{prefix}, .f32, &shape),
        .ffn_bias = try weights.tensor(e, ck, "{s}mhc_ffn.bias", .{prefix}, .f32, &.{8}),
    };
}

pub fn loaded(g: geometry.Geometry, name: []const u8) bool {
    if (std.mem.eql(u8, name, "lm_head.weight") or std.mem.eql(u8, name, "model.language_model.embed_tokens.weight") or std.mem.eql(u8, name, "model.language_model.norm.weight")) return true;
    const prefix = "model.language_model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return false;
    const rest = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
    const i = std.fmt.parseInt(usize, rest[0..dot], 10) catch return false;
    if (i >= g.layers) return false;
    const suffix = rest[dot + 1 ..];
    const common = [_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight" };
    const delta = [_][]const u8{ "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d.weight", "linear_attn.norm.weight", "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight", "linear_attn.out_proj.weight" };
    const attn = [_][]const u8{ "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight" };
    const mhc = [_][]const u8{ "mhc_attn.weight", "mhc_attn.bias", "mhc_ffn.weight", "mhc_ffn.bias" };
    for (common) |n| if (std.mem.eql(u8, suffix, n)) return true;
    const mixer: []const []const u8 = if (g.linear[i]) &delta else &attn;
    for (mixer) |n| if (std.mem.eql(u8, suffix, n)) return true;
    if (active(g, i)) for (mhc) |n| if (std.mem.eql(u8, suffix, n)) return true;
    return false;
}

pub fn checkDtBias(tensor_value: @import("core").checkpoint.Tensor, heads: usize) !void {
    if (tensor_value.is(.f32, &.{heads})) return error.SenseNovaFp32DtBias;
    if (!tensor_value.is(.bf16, &.{heads})) return error.UnexpectedTensor;
}

pub fn preflight(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, g: geometry.Geometry) !void {
    var ck = try @import("core").Checkpoint.openModelPrefix(gpa, io, dir, "model.language_model.layers.");
    defer ck.close();
    var buf: [128]u8 = undefined;
    for (0..g.layers) |i| if (g.linear[i]) {
        const name = try std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.linear_attn.dt_bias", .{i});
        try checkDtBias(try ck.get(name), g.v_heads);
    };
}
