//! Full GLM-5.3 (glm_moe_dsa) as the CUDA pieces' Config: glm_dsa's shape and indexer pattern in Flash's fields.
const std = @import("std");
const dsa = @import("glm_dsa_config");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");

pub const model_type = "glm_moe_dsa";

/// The CUDA pieces' Config of either checkpoint: full GLM-5.3's by its model_type, GLM-5.3-Flash's otherwise.
pub fn of(gpa: std.mem.Allocator, json: []const u8) !cfg.Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const t = if (parsed.value == .object) parsed.value.object.get("model_type") else null;
    if (t != null and t.? == .string and std.mem.eql(u8, t.?.string, model_type)) return parse(gpa, json);
    return cfg.parse(gpa, json);
}

fn flag(o: std.json.ObjectMap, key: []const u8, default: bool) bool {
    const v = o.get(key) orelse return default;
    return v != .bool or v.bool;
}

/// config.json of a glm_moe_dsa checkpoint; glm_dsa refuses another shape, smaller counts pass (synthetic checkpoints).
pub fn parse(gpa: std.mem.Allocator, json: []const u8) !cfg.Config {
    const d = try dsa.parse(gpa, json);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const t = parsed.value.object;
    try cfg.activation(t, t);
    // interleaved rope pairs in the latent's and the indexer's roped dims are what the kernels take
    if (!flag(t, "rope_interleave", true) or !flag(t, "indexer_rope_interleave", true)) return error.UnsupportedRope;
    if (flag(t, "tie_word_embeddings", false)) return error.UnsupportedGlmShape;
    var c: cfg.Config = .{
        .dsa = true,
        .shares = 3, // TP=1 runs the three shares TP=3 runs, in one process (cuda_engine_shares)
        .hidden = d.hidden,
        .layers = d.layers,
        .run = d.layers,
        .vocab = d.vocab,
        .eps = d.eps,
        .dense_layers = d.dense_layers,
        .dense_inter = d.dense_inter,
        .experts = d.experts,
        .own = .{ 0, d.experts },
        .inter = .{ 0, d.moe_inter },
        .topk = d.topk,
        .moe_inter = d.moe_inter,
        .shared_inter = d.moe_inter * @as(u32, @intCast(if (t.get("n_shared_experts")) |v| v.integer else 1)),
        .routed_scale = d.routed_scale,
        .norm_topk = flag(t, "norm_topk_prob", true),
        .swiglu_limit = std.math.inf(f32), // no clamp: min(g, inf) and [-inf, inf] keep every finite value
        .kda_heads = 0,
        .mla_heads = d.heads,
        .nope = d.nope,
        .v_dim = d.v_dim,
        .q_lora = d.q_lora,
        .kv_lora = d.kv_lora,
        .rope = d.rope,
        .rope_theta = d.rope_theta,
        .i_heads = d.i_heads,
        .i_dim = d.i_dim,
        .i_topk = d.i_topk,
        .kpool = 1,
        .hc = 1, // one residual stream: no hyper-connections
        .sinkhorn = 0,
        .mtp = d.mtp,
        .eos = d.eos,
        .eos_n = d.eos_n,
        .mla = .empty,
    };
    if (cfg.quantBlock(t)) |q| if (q.get("quant_method")) |m| if (m == .string and std.mem.eql(u8, m.string, "exl3")) {
        c.head_unit = 128;
        c.exl3 = true;
    };
    for (0..c.layers + c.mtp) |i| c.mla.set(i);
    for (0..c.layers) |i| if (!d.full.isSet(i)) c.shared.set(i);
    var eos_in = true;
    for (c.eos[0..c.eos_n]) |id| eos_in = eos_in and id < c.vocab;
    if (c.dense_layers > c.layers or c.experts < c.topk or c.vocab % 128 != 0 or !eos_in or c.mtp > 1)
        return error.UnsupportedGlmShape;
    return c;
}

/// Full GLM-5.3's config.json with the counts given: its widths, and its indexers at 0-2 then every fourth from 6.
pub fn dsaConfig(gpa: std.mem.Allocator, layers: u32, dense: u32, experts: u32, vocab: u32) ![]u8 {
    var types: std.ArrayList(u8) = .empty;
    defer types.deinit(gpa);
    for (0..layers) |i| try types.print(gpa, "{s}\"{s}\"", .{ if (i > 0) ", " else "", if (i < 3 or (i >= 6 and (i - 6) % 4 == 0)) "full" else "shared" });
    return std.fmt.allocPrint(gpa,
        \\{{"model_type": "glm_moe_dsa", "hidden_size": 6144, "num_hidden_layers": {d}, "vocab_size": {d},
        \\"rms_norm_eps": 1e-05, "first_k_dense_replace": {d}, "intermediate_size": 12288, "n_routed_experts": {d},
        \\"num_experts_per_tok": 8, "moe_intermediate_size": 2048, "n_shared_experts": 1, "norm_topk_prob": true,
        \\"routed_scaling_factor": 2.5, "num_attention_heads": 64, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
        \\"v_head_dim": 256, "q_lora_rank": 2048, "kv_lora_rank": 512, "index_n_heads": 32, "index_head_dim": 128,
        \\"index_topk": 2048, "num_nextn_predict_layers": 1, "n_group": 1, "topk_group": 1, "scoring_func": "sigmoid",
        \\"rope_interleave": true, "indexer_rope_interleave": true, "tie_word_embeddings": false,
        \\"rope_parameters": {{"rope_theta": 8000000, "rope_type": "default"}}, "eos_token_id": [{d}],
        \\"indexer_types": [{s}], "quantization_config": {{"quant_method": "exl3", "codebook": "mcg", "head_bits": 16}}}}
    , .{ layers, vocab, dense, experts, vocab - 2, types.items });
}

const testing = std.testing;

test "full GLM-5.3's config.json: every layer MLA, 576-wide roped rows, pools of 1, 21 indexed layers, MTP" {
    const json = try dsaConfig(testing.allocator, 78, 3, 256, 154880);
    defer testing.allocator.free(json);
    const c = try parse(testing.allocator, json);
    try testing.expect(c.dsa and c.exl3 and c.head_unit == 128 and c.hc == 1 and c.kda_heads == 0);
    try testing.expect((try of(testing.allocator, json)).dsa);
    try testing.expectEqual(@as(u32, 576), c.latentRow());
    try testing.expectEqual(@as(u32, 2048), c.keyWidth());
    try testing.expectEqual(@as(u32, 79), c.countKind(.mla) + 1); // countKind counts the backbone; the MTP layer is MLA too
    try testing.expectEqual(cfg.Kind.mla, c.kind(78));
    try testing.expectEqual(@as(u32, 0), c.countKind(.kda));
    try testing.expectEqual(@as(f64, 8e6), c.rope_theta);
    try testing.expectEqual(@as(u32, 2048), c.shared_inter);
    try testing.expectEqual(@as(u32, 64 * 192 + 32 * 128), c.qrProj());
    var indexed: u32 = 0;
    for (0..c.layers) |i| indexed += @intFromBool(c.indexFrom(@intCast(i)) == i);
    try testing.expectEqual(@as(u32, 21), indexed); // 0, 1, 2 and 6, 10, .., 74
    try testing.expectEqual(@as(u32, 2), c.indexFrom(5));
    try testing.expectEqual(@as(u32, 74), c.indexFrom(77));
    try testing.expectEqual(@as(u32, 78), c.indexFrom(78)); // the MTP layer runs its own
    try testing.expectEqual(@as(u32, 154878), c.eos[0]); // dsaConfig's: vocab - 2
}

test "a synthetic count parses; GLM-5.3-Flash's config, other widths and tied embeddings do not" {
    const gpa = testing.allocator;
    const small = try dsaConfig(gpa, 4, 1, 16, 8192);
    defer gpa.free(small);
    const s = try parse(gpa, small);
    try testing.expectEqual(@as(u32, 4), s.layers);
    try testing.expectEqual(@as(u32, 2), s.indexFrom(3));
    try testing.expectEqual(cfg.Kind.mla, s.kind(4));
    const wide = try std.mem.replaceOwned(u8, gpa, small, "\"hidden_size\": 6144", "\"hidden_size\": 4096");
    defer gpa.free(wide);
    try testing.expectError(error.UnsupportedGlmShape, parse(gpa, wide));
    const tied = try std.mem.replaceOwned(u8, gpa, small, "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true");
    defer gpa.free(tied);
    try testing.expectError(error.UnsupportedGlmShape, parse(gpa, tied));
    const halves = try std.mem.replaceOwned(u8, gpa, small, "\"rope_interleave\": true", "\"rope_interleave\": false");
    defer gpa.free(halves);
    try testing.expectError(error.UnsupportedRope, parse(gpa, halves));
    const few = try dsaConfig(gpa, 4, 1, 4, 8192);
    defer gpa.free(few);
    try testing.expectError(error.UnsupportedGlmShape, parse(gpa, few));
    const flash = try cfg.textConfig(gpa, 2, 1, &.{1}, 16, 8192, 4096);
    defer gpa.free(flash);
    try testing.expectError(error.NotGlmMoeDsa, parse(gpa, flash));
    try testing.expect(!(try of(gpa, flash)).dsa);
}

test "tensorfold_activation_dtype: bfloat16 parses, float32 refuses by name" {
    const gpa = testing.allocator;
    const small = try dsaConfig(gpa, 4, 1, 16, 8192);
    defer gpa.free(small);
    const bf16 = try std.mem.replaceOwned(u8, gpa, small, "{\"model_type\"", "{\"tensorfold_activation_dtype\": \"bfloat16\", \"model_type\"");
    defer gpa.free(bf16);
    _ = try parse(gpa, bf16);
    const f32s = try std.mem.replaceOwned(u8, gpa, small, "{\"model_type\"", "{\"tensorfold_activation_dtype\": \"float32\", \"model_type\"");
    defer gpa.free(f32s);
    try testing.expectError(error.UnsupportedActivationDtype, parse(gpa, f32s));
}

test "full GLM-5.3 at TP=3: no KDA heads, 22/21/21 DSA heads, 768/640/640 EXL3 expert rows, 704/704/640 shared rows" {
    const json = try dsaConfig(testing.allocator, 78, 3, 256, 154880);
    defer testing.allocator.free(json);
    const c = try parse(testing.allocator, json);
    var heads: usize = 0;
    for (0..3) |r| {
        const s = try shard.Share.of(&c, r, 3);
        try testing.expectEqual(@as(usize, 0), s.kda.len());
        try testing.expectEqual(@as(usize, 4096), s.dense.len());
        try testing.expectEqual(@as(usize, if (r == 0) 768 else 640), s.moe.len());
        try testing.expectEqual(@as(usize, if (r == 2) 640 else 704), s.shared.len());
        try testing.expectEqual(@as(usize, if (r == 0) 404 else 403) * 128, s.vocab.len());
        heads += s.mla.len();
    }
    try testing.expectEqual(@as(usize, 64), heads);
    try testing.expectEqual(shard.Span{ .lo = 0, .hi = 22 }, (try shard.Share.of(&c, 0, 3)).mla);
    try testing.expectEqual([2]u64{ 768, 2048 }, (try shard.Share.of(&c, 0, 3)).part(&c));
}
