//! Qwen3 embedding weights, BF16 in place, with no output head required or copied.
const std = @import("std");
const ck = @import("../../../core/checkpoint_metal.zig");
const cfg = @import("../../llama/config.zig");
const w = @import("../../llama/weights.zig");
pub fn load(gpa: std.mem.Allocator, c: *const ck.Checkpoint, f: *const cfg.Config) !w.Weights {
    const scope: []const u8 = if (c.has("model.embed_tokens.weight")) "model." else "";
    const embed = try w.dense(c, scope, "embed_tokens.weight", f.vocab, f.hidden);
    var out: w.Weights = .{ .embed = embed, .head = embed, .norm = try w.norm(c, f.hidden, scope, "norm.weight"), .layers = try gpa.alloc(w.Layer, f.layers) };
    errdefer out.deinit(gpa);
    for (out.layers, 0..) |*l, i| {
        var pb: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "{s}layers.{d}.", .{ scope, i });
        var ab: [128]u8 = undefined;
        const a = try std.fmt.bufPrint(&ab, "{s}self_attn.", .{p});
        var mb: [128]u8 = undefined;
        const m = try std.fmt.bufPrint(&mb, "{s}mlp.", .{p});
        l.* = .{ .input_norm = try w.norm(c, f.hidden, p, "input_layernorm.weight"), .post_norm = try w.norm(c, f.hidden, p, "post_attention_layernorm.weight"), .attn = .{ .q = try w.dense(c, a, "q_proj.weight", f.queryWidth(), f.hidden), .k = try w.dense(c, a, "k_proj.weight", f.kvWidth(), f.hidden), .v = try w.dense(c, a, "v_proj.weight", f.kvWidth(), f.hidden), .o = try w.dense(c, a, "o_proj.weight", f.hidden, f.queryWidth()), .q_norm = try w.norm(c, 128, a, "q_norm.weight"), .k_norm = try w.norm(c, 128, a, "k_norm.weight") }, .mlp = .{ .gate = try w.dense(c, m, "gate_proj.weight", f.intermediate, f.hidden), .up = try w.dense(c, m, "up_proj.weight", f.intermediate, f.hidden), .down = try w.dense(c, m, "down_proj.weight", f.hidden, f.intermediate) } };
    }
    return out;
}
