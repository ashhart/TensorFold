const std = @import("std");

/// Modules the graph reads only through repacked copies: their tensors stay in the files until load stages them.
const modules = [_][]const u8{ ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.in_proj_b", ".linear_attn.in_proj_a", ".linear_attn.out_proj", ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj" };

/// Whether `name` is a weight, scales or biases tensor of a repacked module or the LM head.
pub fn repacked(name: []const u8, _: usize) bool {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return false;
    const leaf = name[dot + 1 ..];
    if (!std.mem.eql(u8, leaf, "weight") and !std.mem.eql(u8, leaf, "scales") and !std.mem.eql(u8, leaf, "biases")) return false;
    const module = name[0..dot];
    if (std.mem.eql(u8, module, "lm_head") or std.mem.endsWith(u8, module, ".lm_head")) return true;
    for (modules) |m| if (std.mem.endsWith(u8, module, m)) return true;
    return false;
}
