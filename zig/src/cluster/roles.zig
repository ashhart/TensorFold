//! What each tensor is for, from its name: the role decides how the planner splits, replicates or spreads it.
const std = @import("std");

pub const Role = enum(u8) {
    embed,
    head,
    final_norm,
    norm,
    /// Attention weights split by head: output rows (q, k, v, gates, per-head convs and biases).
    attn_col,
    /// Attention output projections, split by input (head) columns; their partial sums need a reduction.
    attn_row,
    /// Low-rank down projections and norms every head shares.
    attn_rep,
    router,
    shared_col,
    shared_row,
    dense_col,
    dense_row,
    latent_in,
    latent_out,
    latent_norm,
    expert,
    vision,
    draft,
    other,
};

pub const Class = struct { role: Role, layer: ?u32 = null, expert: ?u32 = null };

/// How a role's bytes divide over a tensor-parallel group.
pub const Split = enum { column, row, replicate, expert };

pub fn split(r: Role) Split {
    return switch (r) {
        .attn_col, .shared_col, .dense_col, .head => .column,
        .attn_row, .shared_row, .dense_row => .row,
        .expert => .expert,
        else => .replicate,
    };
}

/// Classify a checkpoint tensor by its name (DeepSeek, Kimi and GLM naming, with or without a language_model prefix).
pub fn classify(name: []const u8) Class {
    const layer_at = std.mem.indexOf(u8, name, "layers.") orelse return top(name);
    if (has(name[0..layer_at], "vision") or has(name[0..layer_at], "mtp")) return top(name);
    const after = name[layer_at + "layers.".len ..];
    const dot = std.mem.indexOfScalar(u8, after, '.') orelse return .{ .role = .other };
    const layer = std.fmt.parseInt(u32, after[0..dot], 10) catch return .{ .role = .other };
    const rest = after[dot + 1 ..];
    return .{ .role = inLayer(rest), .layer = layer, .expert = expertOf(rest) };
}

fn top(name: []const u8) Class {
    if (has(name, "vision") or has(name, "mm_projector") or has(name, "multi_modal_projector")) return .{ .role = .vision };
    if (has(name, "mtp") or has(name, "nextn")) return .{ .role = .draft };
    if (has(name, "embed_tokens") or has(name, "wte")) return .{ .role = .embed };
    if (has(name, "lm_head")) return .{ .role = .head };
    if (has(name, "norm")) return .{ .role = .final_norm };
    return .{ .role = .other };
}

fn inLayer(rest: []const u8) Role {
    for ([_][]const u8{ "self_attn.", "attn.", "linear_attn.", "mixer." }) |p| {
        if (!std.mem.startsWith(u8, rest, p)) continue;
        const seg = segment(rest[p.len..]);
        if (eql(seg, "o_proj") or eql(seg, "out_proj")) return .attn_row;
        if (eql(seg, "indexer")) return .attn_rep;
        if (std.mem.endsWith(u8, seg, "_a_proj") or eql(seg, "kv_a_proj_with_mqa") or has(seg, "norm")) return .attn_rep;
        return .attn_col;
    }
    if (std.mem.startsWith(u8, rest, "mlp.experts.") or std.mem.startsWith(u8, rest, "block_sparse_moe.experts.")) return .expert;
    if (std.mem.startsWith(u8, rest, "mlp.shared_expert")) return if (has(rest, "down_proj")) .shared_row else .shared_col;
    if (std.mem.startsWith(u8, rest, "mlp.gate.") or std.mem.startsWith(u8, rest, "mlp.router")) return .router;
    if (std.mem.startsWith(u8, rest, "mlp.")) {
        const seg = segment(rest["mlp.".len..]);
        if (eql(seg, "gate_proj") or eql(seg, "up_proj") or eql(seg, "gate_up_proj")) return .dense_col;
        if (eql(seg, "down_proj")) return .dense_row;
        if (has(seg, "latent") or has(seg, "_in") or has(seg, "_out")) {
            if (has(seg, "norm")) return .latent_norm;
            return if (has(seg, "out")) .latent_out else .latent_in;
        }
        if (has(seg, "norm")) return .norm;
        return .other;
    }
    if (has(rest, "norm")) return .norm;
    return .other;
}

fn expertOf(rest: []const u8) ?u32 {
    const key = "experts.";
    const at = std.mem.indexOf(u8, rest, key) orelse return null;
    const tail = rest[at + key.len ..];
    const end = std.mem.indexOfScalar(u8, tail, '.') orelse tail.len;
    return std.fmt.parseInt(u32, tail[0..end], 10) catch null;
}

fn segment(s: []const u8) []const u8 {
    return s[0 .. std.mem.indexOfScalar(u8, s, '.') orelse s.len];
}

fn has(s: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, s, needle) != null;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

test "Kimi, DeepSeek and GLM names land in their roles" {
    const cases = [_]struct { []const u8, Role, ?u32, ?u32 }{
        .{ "language_model.model.embed_tokens.weight", .embed, null, null },
        .{ "language_model.lm_head.weight", .head, null, null },
        .{ "language_model.model.norm.weight", .final_norm, null, null },
        .{ "language_model.model.layers.12.self_attn.q_proj.weight", .attn_col, 12, null },
        .{ "language_model.model.layers.12.self_attn.f_b_proj.weight", .attn_col, 12, null },
        .{ "language_model.model.layers.12.self_attn.g_a_proj.weight", .attn_rep, 12, null },
        .{ "language_model.model.layers.12.self_attn.o_norm.weight", .attn_rep, 12, null },
        .{ "language_model.model.layers.12.self_attn.o_proj.weight", .attn_row, 12, null },
        .{ "model.layers.3.self_attn.kv_a_proj_with_mqa.weight", .attn_rep, 3, null },
        .{ "model.layers.3.self_attn.kv_a_layernorm.weight", .attn_rep, 3, null },
        .{ "model.layers.3.self_attn.kv_b_proj.weight", .attn_col, 3, null },
        .{ "model.layers.3.self_attn.indexer.wq_b.weight", .attn_rep, 3, null },
        .{ "model.layers.3.self_attn.o_proj.weight_scale_inv", .attn_row, 3, null },
        .{ "model.layers.5.mlp.experts.895.down_proj.weight_scale", .expert, 5, 895 },
        .{ "model.layers.5.mlp.shared_experts.down_proj.weight", .shared_row, 5, null },
        .{ "model.layers.5.mlp.shared_experts.up_proj.weight", .shared_col, 5, null },
        .{ "model.layers.5.mlp.gate.weight", .router, 5, null },
        .{ "model.layers.5.mlp.gate.e_score_correction_bias", .router, 5, null },
        .{ "model.layers.0.mlp.down_proj.weight", .dense_row, 0, null },
        .{ "model.layers.0.mlp.gate_proj.weight", .dense_col, 0, null },
        .{ "model.layers.5.mlp.latent_in_proj.weight", .latent_in, 5, null },
        .{ "model.layers.5.mlp.latent_out_proj.weight", .latent_out, 5, null },
        .{ "model.layers.5.mlp.latent_norm.weight", .latent_norm, 5, null },
        .{ "model.layers.5.input_layernorm.weight", .norm, 5, null },
        .{ "vision_tower.encoder.layers.3.attn.qkv.weight", .vision, null, null },
        .{ "model.mtp.layers.0.eh_proj.weight", .draft, null, null },
    };
    for (cases) |c| {
        const got = classify(c[0]);
        std.testing.expectEqual(c[1], got.role) catch |err| {
            std.debug.print("{s}: {s}\n", .{ c[0], @tagName(got.role) });
            return err;
        };
        try std.testing.expectEqual(c[2], got.layer);
        try std.testing.expectEqual(c[3], got.expert);
    }
    try std.testing.expectEqual(Split.row, split(.attn_row));
    try std.testing.expectEqual(Split.column, split(.head));
    try std.testing.expectEqual(Split.replicate, split(.router));
}
