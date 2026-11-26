//! Drama 3's checkpoint tensors by name and shape, in load order, for both backends; experts, lm_head per backend.
const std = @import("std");
const cfg = @import("config.zig");

pub const embed = "model.language_model.embed_tokens.weight";
pub const codebook_embeddings = "audio_decoder.codebook_embeddings.weight";
pub const final_norm = "model.language_model.norm.weight";
pub const lm_head = "lm_head.weight";
/// A routed expert's three projections, in the order both backends pack them.
pub const projections = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };

pub const Gdn = struct { proj: u64, conv: u64, a_log: u64, dt_bias: u64, norm: u64, out: u64 };
pub const Attn = struct { qkv: u64, q_norm: u64, k_norm: u64, o: u64 };

/// A backbone layer's dense tensors; the backend's Layer has these fields by the same names.
pub const Dense = struct {
    in_norm: u64,
    post_norm: u64,
    gdn: Gdn = undefined,
    attn: Attn = undefined,
    moe_in: u64, // [1288, 2048]
    shared_down: u64, // [2048, 512]
};

pub const TowerLayer = struct {
    table: u64, // [4096, 2048] the code embedding (semantic_embedding / codebook_embedding)
    enorm: u64,
    hnorm: u64,
    eh: u64, // [2048, 4096]
    attn_norm: u64,
    qkv: u64, // [9216, 2048]: q 4096 | gate 4096 | k 512 | v 512
    q_norm: u64,
    k_norm: u64,
    wo: u64, // [2048, 4096]
    ffn_norm: u64,
    w13: u64, // [12288, 2048]
    w2: u64, // [2048, 6144]
    final_norm: u64,
};

pub const Top = struct { embed: u64, cb_emb: u64, final_norm: u64 };

fn fmt(buf: []u8, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, f, args) catch unreachable;
}

/// Expert `e`'s projection `p` in layer `li`, with `suffix` after ".weight" ("", "_scale", "_scale_2", "_scale_inv").
pub fn expert(buf: []u8, li: usize, e: usize, p: []const u8, suffix: []const u8) []const u8 {
    return fmt(buf, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.weight{s}", .{ li, e, p, suffix });
}

/// The same projection's module name (no ".weight"), as the shared block-FP8 reader takes it.
pub fn expertModule(buf: []u8, li: usize, e: usize, p: []const u8) []const u8 {
    return fmt(buf, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ li, e, p });
}

/// The token and codebook embeddings and the final norm, before lm_head.
pub fn top(l: anytype) !Top {
    const H = cfg.hidden;
    return .{
        .embed = try l.one(embed, &.{ cfg.vocab, H }),
        .cb_emb = try l.one(codebook_embeddings, &.{ cfg.codebooks * cfg.codebook_size, H }),
        .final_norm = try l.one(final_norm, &.{H}),
    };
}

/// Backbone layer `li`'s dense tensors: norms, full attention or GDN, the shared expert and router.
pub fn dense(l: anytype, li: usize, full: bool) !Dense {
    const H = cfg.hidden;
    const pre = "model.language_model.layers.{d}.";
    var nb: [4][192]u8 = undefined;
    var d: Dense = .{
        .in_norm = try l.one(fmt(&nb[0], pre ++ "input_layernorm.weight", .{li}), &.{H}),
        .post_norm = try l.one(fmt(&nb[0], pre ++ "post_attention_layernorm.weight", .{li}), &.{H}),
        .moe_in = undefined,
        .shared_down = undefined,
    };
    if (full) {
        const q = fmt(&nb[0], pre ++ "self_attn.q_proj.weight", .{li});
        const k = fmt(&nb[1], pre ++ "self_attn.k_proj.weight", .{li});
        const v = fmt(&nb[2], pre ++ "self_attn.v_proj.weight", .{li});
        const kv = cfg.kv_heads * cfg.head_dim;
        d.attn = .{
            .qkv = try l.stack(&.{ q, k, v }, &.{ 2 * cfg.heads * cfg.head_dim, kv, kv }, H, cfg.attn_proj),
            .q_norm = try l.one(fmt(&nb[0], pre ++ "self_attn.q_norm.weight", .{li}), &.{cfg.head_dim}),
            .k_norm = try l.one(fmt(&nb[0], pre ++ "self_attn.k_norm.weight", .{li}), &.{cfg.head_dim}),
            .o = try l.one(fmt(&nb[0], pre ++ "self_attn.o_proj.weight", .{li}), &.{ H, cfg.heads * cfg.head_dim }),
        };
    } else {
        const qkv = fmt(&nb[0], pre ++ "linear_attn.in_proj_qkv.weight", .{li});
        const z = fmt(&nb[1], pre ++ "linear_attn.in_proj_z.weight", .{li});
        const b = fmt(&nb[2], pre ++ "linear_attn.in_proj_b.weight", .{li});
        const a = fmt(&nb[3], pre ++ "linear_attn.in_proj_a.weight", .{li});
        d.gdn = .{
            .proj = try l.stack(&.{ qkv, z, b, a }, &.{ cfg.conv_channels, 4096, 32, 32 }, H, cfg.gdn_proj),
            .conv = try l.one(fmt(&nb[0], pre ++ "linear_attn.conv1d.weight", .{li}), &.{ cfg.conv_channels, 1, cfg.conv_kernel }),
            .a_log = try l.one(fmt(&nb[0], pre ++ "linear_attn.A_log", .{li}), &.{cfg.lin_v_heads}),
            .dt_bias = try l.one(fmt(&nb[0], pre ++ "linear_attn.dt_bias", .{li}), &.{cfg.lin_v_heads}),
            .norm = try l.one(fmt(&nb[0], pre ++ "linear_attn.norm.weight", .{li}), &.{cfg.lin_dim}),
            .out = try l.one(fmt(&nb[0], pre ++ "linear_attn.out_proj.weight", .{li}), &.{ H, cfg.lin_v_heads * cfg.lin_dim }),
        };
    }
    const sg = fmt(&nb[0], pre ++ "mlp.shared_expert.gate_proj.weight", .{li});
    const su = fmt(&nb[1], pre ++ "mlp.shared_expert.up_proj.weight", .{li});
    const router = fmt(&nb[2], pre ++ "mlp.gate.weight", .{li});
    const sgate = fmt(&nb[3], pre ++ "mlp.shared_expert_gate.weight", .{li});
    const moe_rows = [_]usize{ cfg.shared_width, cfg.shared_width, cfg.experts, 1 };
    d.moe_in = try l.stack(&.{ sg, su, router, sgate }, &moe_rows, H, cfg.moe_proj);
    const down = fmt(&nb[0], pre ++ "mlp.shared_expert.down_proj.weight", .{li});
    d.shared_down = try l.one(down, &.{ H, cfg.shared_width });
    return d;
}

/// Copies a layer's dense tensors into the backend's Layer, field by field.
pub fn setDense(layer: anytype, d: Dense) void {
    inline for (comptime std.meta.fieldNames(Dense)) |f| @field(layer, f) = @field(d, f);
}

/// Tower layer `ti`: 4 shared layers (semantic embedding), then 9 residual ones (codebook embedding).
pub fn tower(l: anytype, ti: usize) !TowerLayer {
    const H = cfg.hidden;
    const shared = ti < cfg.shared_layers;
    const idx = if (shared) ti else ti - cfg.shared_layers;
    const p: []const u8 = if (shared) "audio_decoder_mtp_shared.layers." else "audio_decoder_mtp.layers.";
    var nb: [2][192]u8 = undefined;
    const tn = struct {
        fn f(buf: []u8, pfx: []const u8, i: usize, suffix: []const u8) []const u8 {
            return fmt(buf, "{s}{d}.{s}", .{ pfx, i, suffix });
        }
    }.f;
    const table = if (shared) "semantic_embedding.weight" else "codebook_embedding.weight";
    const ff = "transformer_layer.feed_forward.";
    return .{
        .table = try l.one(tn(&nb[0], p, idx, table), &.{ cfg.codebook_size, H }),
        .enorm = try l.one(tn(&nb[0], p, idx, "enorm.weight"), &.{H}),
        .hnorm = try l.one(tn(&nb[0], p, idx, "hnorm.weight"), &.{H}),
        .eh = try l.one(tn(&nb[0], p, idx, "eh_proj.weight"), &.{ H, 2 * H }),
        .attn_norm = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention_norm.weight"), &.{H}),
        .qkv = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention.wqkv.weight"), &.{ cfg.attn_proj, H }),
        .q_norm = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention.q_norm.weight"), &.{cfg.head_dim}),
        .k_norm = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention.k_norm.weight"), &.{cfg.head_dim}),
        .wo = try l.one(tn(&nb[0], p, idx, "transformer_layer.attention.wo.weight"), &.{ H, cfg.heads * cfg.head_dim }),
        .ffn_norm = try l.one(tn(&nb[0], p, idx, "transformer_layer.ffn_norm.weight"), &.{H}),
        .w13 = try l.stack(&.{ tn(&nb[0], p, idx, ff ++ "w1.weight"), tn(&nb[1], p, idx, ff ++ "w3.weight") }, &.{ cfg.tower_ffn, cfg.tower_ffn }, H, 2 * cfg.tower_ffn),
        .w2 = try l.one(tn(&nb[0], p, idx, ff ++ "w2.weight"), &.{ H, cfg.tower_ffn }),
        .final_norm = try l.one(tn(&nb[0], p, idx, "final_layernorm.weight"), &.{H}),
    };
}

/// Codebook head `hi` (codebooks 1..9).
pub fn head(l: anytype, hi: usize) !u64 {
    var nb: [64]u8 = undefined;
    return l.one(fmt(&nb, "audio_decoder_mtp.heads.{d}.weight", .{hi}), &.{ cfg.codebook_size, cfg.hidden });
}
