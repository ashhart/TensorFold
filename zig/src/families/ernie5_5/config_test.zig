//! config.json: the release parses to its shape, every pinned key refuses another value by name, small shapes pass.
const std = @import("std");
const config = @import("config.zig");

const t = std.testing;
pub const release_json = @embedFile("fixtures/config.json");
/// A key set to JSON text, or removed when the text is null.
pub const Edit = struct { []const u8, ?[]const u8 };

/// The release config with `edits` applied.
pub fn edited(a: std.mem.Allocator, edits: []const Edit) ![]const u8 {
    var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, release_json, .{});
    for (edits) |e| {
        if (e[1]) |v| {
            try doc.object.put(a, e[0], try std.json.parseFromSliceLeaky(std.json.Value, a, v, .{}));
        } else _ = doc.object.orderedRemove(e[0]);
    }
    return std.json.Stringify.valueAlloc(a, doc, .{});
}

test "the release config parses to the published shape" {
    var p: config.Problem = .{};
    const c = try config.parse(t.allocator, release_json, &p);
    try t.expectEqual(@as(u32, 4096), c.hidden);
    try t.expectEqual(@as(u32, 43), c.layers);
    try t.expectEqual(@as(u32, 201216), c.vocab);
    try t.expectEqual(@as(u32, 1048576), c.max_context);
    var dsa: [6]u32 = undefined;
    var n: usize = 0;
    for (0..c.layers) |i| if (c.isDsa(@intCast(i))) {
        dsa[n] = @intCast(i);
        n += 1;
    };
    try t.expectEqualSlices(u32, &.{ 7, 14, 21, 28, 35, 42 }, dsa[0..n]);
    try t.expectEqual(@as(i32, 128), c.compress[0]);
    try t.expectEqual(config.Hca{ .heads = 64, .head_dim = 512, .nope = 448, .rope = 64, .q_lora = 1024, .o_groups = 8, .o_lora = 1024, .window = 128 }, c.hca);
    try t.expectEqual(config.Mla{ .heads = 64, .q_lora = 1024, .kv_lora = 512, .nope = 192, .rope = 64, .value = 256 }, c.mla);
    try t.expectEqual(config.Indexer{ .heads = 64, .dim = 128, .topk = 2048 }, c.index);
    try t.expectEqual(@as(u32, 512), c.moe.experts);
    try t.expectEqual(@as(u32, 10), c.moe.topk);
    try t.expectEqual(@as(u32, 2048), c.moe.latent);
    try t.expectEqual(@as(u32, 1), c.moe.dense_layers);
    try t.expectEqual(@as(u32, 14336), c.moe.dense_inter);
    try t.expect(c.isDense(0) and !c.isDense(1));
    try t.expectEqual(config.HyperConnections{ .streams = 4, .sinkhorn = 20, .eps = 1e-6 }, c.hc);
    try t.expectEqual(config.Rope{ .theta = 8e6, .compress_theta = 160000, .factor = 16, .original = 65536, .beta_fast = 32, .beta_slow = 1 }, c.rope);
    try t.expectEqualSlices(u32, &.{ 40, 41, 42 }, c.dspark.taps[0..c.dspark.taps_n]);
    try t.expectEqual(@as(u32, 7), c.dspark.block);
    try t.expectEqual(@as(u32, 256), c.dspark.rank);
}

test "a pinned key with another value is refused by name, and so is a missing one" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, ?[]const u8, anyerror, []const u8 }{
        .{ "use_vha_attention", "false", error.Unsupported, "use_vha_attention must be true: the weight map follows it" },
        .{ "n_shared_experts", "2", error.Unsupported, "n_shared_experts must be 1" },
        .{ "multimax", "null", error.Unsupported, "multimax must be \"lm_head\"" },
        .{ "scoring_func", "\"softmax\"", error.Unsupported, "scoring_func must be \"sigmoid\": the vendor's plugin runs no other value" },
        .{ "n_group", "8", error.Unsupported, "n_group must be 1" },
        .{ "csa_dense_mode", "false", error.Unsupported, "csa_dense_mode must be true: its meaning is not read yet" },
        .{ "use_attn_merger", null, error.BadConfig, "config.json: use_attn_merger is missing" },
        .{ "compress_ratios", "[128, -1]", error.Layout, "compress_ratios must hold one entry for each of the 43 layers" },
        .{ "num_key_value_heads", "8", error.Layout, "HCA layers need one key/value head" },
        .{ "hybrid_mla_num_key_value_heads", "1", error.Layout, "DSA layers need as many key/value heads" },
        .{ "architectures", "[\"DeepseekV4ForCausalLM\"]", error.NotErnie55, "not an ERNIE 5.5 checkpoint" },
        .{ "quantization_config", "{\"quant_method\": \"fp8\", \"fmt\": \"e4m3\", \"weight_block_size\": [128, 128]}", error.Quant, "MXFP4 routed experts" },
        .{ "dspark_target_layer_ids", "[40, 43]", error.Layout, "names a layer past the 43 layers" },
        .{ "first_k_dense_replace", "44", error.BadConfig, "first_k_dense_replace is 44, outside 0..43" },
        .{ "rope_scaling", "{\"type\": \"linear\", \"factor\": 4}", error.Unsupported, "type must be \"yarn\"" },
    };
    for (cases) |c| {
        var p: config.Problem = .{};
        const json = try edited(a, &.{.{ c[0], c[1] }});
        try t.expectError(c[2], config.parse(t.allocator, json, &p));
        if (std.mem.indexOf(u8, p.text(), c[3]) == null) {
            std.debug.print("{s}: got \"{s}\"\n", .{ c[0], p.text() });
            return error.TestUnexpectedRefusal;
        }
    }
}

test "every compress ratio entry is a positive ratio or -2, and a small shape passes" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const small = [_]Edit{
        .{ "hidden_size", "64" },                   .{ "num_hidden_layers", "4" },
        .{ "compress_ratios", "[4, -2, 4, 4]" },    .{ "vocab_size", "300" },
        .{ "num_attention_heads", "4" },            .{ "head_dim", "32" },
        .{ "qk_nope_head_dim", "24" },              .{ "qk_rope_head_dim", "8" },
        .{ "v_head_dim", "32" },                    .{ "q_lora_rank", "16" },
        .{ "o_groups", "2" },                       .{ "o_lora_rank", "16" },
        .{ "sliding_window", "4" },                 .{ "hybrid_mla_num_attention_heads", "4" },
        .{ "hybrid_mla_num_key_value_heads", "4" }, .{ "hybrid_mla_q_lora_rank", "16" },
        .{ "hybrid_mla_kv_lora_rank", "16" },       .{ "hybrid_mla_qk_nope_head_dim", "8" },
        .{ "hybrid_mla_qk_rope_head_dim", "8" },    .{ "hybrid_mla_v_head_dim", "8" },
        .{ "index_n_heads", "2" },                  .{ "index_head_dim", "8" },
        .{ "index_topk", "8" },                     .{ "n_routed_experts", "8" },
        .{ "num_experts_per_tok", "2" },            .{ "moe_intermediate_size", "32" },
        .{ "moe_latent_size", "32" },               .{ "intermediate_size", "96" },
        .{ "dspark_target_layer_ids", "[2, 3]" },   .{ "dspark_noise_token_id", "10" },
        .{ "dspark_markov_rank", "8" },
    };
    var p: config.Problem = .{};
    const c = try config.parse(t.allocator, try edited(a, &small), &p);
    try t.expect(!c.isDsa(0) and c.isDsa(1) and c.compress[2] == 4);
    try t.expectEqual(@as(u32, 2), c.dspark.taps_n);
    for ([_][]const u8{ "[4, 0, 4, 4]", "[4, -1, 4, 4]", "[4, -3, 4, 4]", "[4, 2.5, 4, 4]" }) |bad| {
        var edits = small;
        edits[2] = .{ "compress_ratios", bad };
        try t.expectError(error.Layout, config.parse(t.allocator, try edited(a, &edits), &p));
    }
}
