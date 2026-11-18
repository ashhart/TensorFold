//! Hy4-flash's own config.json (fixtures/config.json) parsed, and each refusal on a one-edit copy.
const std = @import("std");
const config = @import("config.zig");

const real = @embedFile("fixtures/config.json");
const t = std.testing;

fn load() !config.Config {
    var why: config.Why = .{};
    return config.parse(t.allocator, real, &why);
}

/// The fixture with `from` (which must occur exactly once) replaced by `to`.
fn edit(from: []const u8, to: []const u8) ![]u8 {
    if (std.mem.count(u8, real, from) != 1) return error.EditAnchor;
    return std.mem.replaceOwned(u8, t.allocator, real, from, to);
}

fn refused(from: []const u8, to: []const u8, want: anyerror, says: []const u8) !void {
    const json = try edit(from, to);
    defer t.allocator.free(json);
    var why: config.Why = .{};
    try t.expectError(want, config.parse(t.allocator, json, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), says) != null);
}

test "the checkpoint's config.json parses to Hy4-flash's shape" {
    const c = try load();
    const x = c.text;
    try t.expectEqual(@as(u32, 3072), x.hidden);
    try t.expectEqual(@as(u32, 42), x.layers);
    try t.expectEqual(@as(u32, 120832), x.vocab);
    try t.expectEqual(@as(u32, 1048576), x.max_context);
    try t.expectEqual(@as(u32, 4), x.hc);
    try t.expectEqual(config.Heads{ .q = 32, .nope = 192, .rope = 64, .v = 256 }, x.sparse);
    try t.expectEqual(config.Heads{ .q = 64, .nope = 192, .rope = 64, .v = 256 }, x.sliding);
    try t.expectEqual(@as(u32, 1024), x.q_rank);
    try t.expectEqual(@as(u32, 512), x.kv_rank);
    try t.expectEqual(@as(u32, 656), x.kvRowBytes());
    try t.expectEqual(@as(u32, 512), x.window);
    try t.expectEqual(@as(u32, 2048), x.index_topk);
    try t.expectEqual(@as(u32, 16), x.index_heads);
    try t.expectEqual(@as(u32, 128), x.index_dim);
    try t.expectEqual(@as(u32, 4), x.compress);
    try t.expectEqual(@as(u32, 512), x.group_topk);
    try t.expectEqual(@as(u32, 13824), x.dense_inter);
    try t.expectEqual(@as(u32, 512), x.experts);
    try t.expectEqual(@as(u32, 16), x.topk);
    try t.expectEqual(@as(u32, 1536), x.latent);
    try t.expectEqual(@as(u32, 1536), x.expert_inter);
    try t.expectEqual(@as(u32, 1), x.shared);
    try t.expectEqual(@as(f32, 3.998), x.route_scale);
    try t.expectEqual(@as(f32, 10.0), x.swiglu_limit);
    try t.expectEqual(@as(u32, 120025), c.eos);
    try t.expectEqual(@as(u32, 120120), c.image_token);
    try t.expect(c.has_vision);
}

test "the layer tables: 12 sparse and 30 sliding layers, one dense MLP, four Engram layers" {
    const c = try load();
    const x = c.text;
    try t.expectEqual(@as(u32, 12), x.count(.sparse));
    try t.expectEqual(@as(u32, 30), x.count(.sliding));
    try t.expectEqual(config.Kind.sparse, x.kinds[0]);
    try t.expectEqual(config.Kind.sparse, x.kinds[1]);
    try t.expectEqual(config.Kind.sliding, x.kinds[2]);
    try t.expectEqual(config.Kind.sparse, x.kinds[5]);
    try t.expectEqual(@as(u32, 32), x.heads(0).q);
    try t.expectEqual(@as(u32, 64), x.heads(2).q);
    try t.expect(!x.moe[0]);
    for (x.moe[1..x.layers]) |m| try t.expect(m);
    // engram_layer_ids are [2, 12, 22, 32] and count from 1
    for (0..x.layers) |i| try t.expectEqual(i == 1 or i == 11 or i == 21 or i == 31, x.engram[i]);
    try t.expectEqual(config.Engram{ .layers = 4, .max_ngram = 3, .heads = 8, .dim = 96, .pad = 120002, .fp8 = true }, c.engram);
}

test "the drafter: DFly with five layers, five tokens and its five backbone taps" {
    const d = (try load()).drafter.?;
    try t.expectEqual(@as(u32, 5), d.layers);
    try t.expectEqual(@as(u32, 5), d.tokens);
    try t.expectEqualSlices(u32, &.{ 25, 29, 33, 37, 41 }, d.taps[0..d.taps_n]);
}

test "another model, a missing key and broken JSON are refused by name" {
    try refused("\"model_type\": \"hy_v4_flash\",\n  \"processor_class\"", "\"model_type\": \"hy_v4\",\n  \"processor_class\"", error.NotHy4Flash, "not a Hy4-flash checkpoint");
    try refused("\"hidden_size\": 3072", "\"hidden_width\": 3072", error.BadConfig, "has no hidden_size");
    var why: config.Why = .{};
    try t.expectError(error.BadConfig, config.parse(t.allocator, "{", &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "not JSON") != null);
}

test "an MLA the engine does not run is refused" {
    try refused("\"gated_mla\": true", "\"gated_mla\": false", error.Mla, "gated_mla must be true");
    try refused("\"learnable_sink\": true", "\"learnable_sink\": false", error.Mla, "learnable_sink must be true");
    try refused("\"gating_type\": \"elementwise\"", "\"gating_type\": \"headwise\"", error.Mla, "gating_type");
    try refused("\"gate_lora_rank\": null", "\"gate_lora_rank\": 64", error.Mla, "gate_lora_rank must be null");
    try refused("\"kv_lora_rank\": 512", "\"kv_lora_rank\": 384", error.Mla, "a KV row holds");
    try refused("\"full_nope\": false", "\"full_nope\": true", error.Mla, "full_nope must be false");
}

test "an indexer whose groups do not fill the top-k is refused" {
    try refused("\"index_topk\": 2048", "\"index_topk\": 2000", error.Indexer, "128-row tiles");
    try refused("\"compressed_indexer_group_topk\": 512", "\"compressed_indexer_group_topk\": 256", error.Indexer, "must fill it");
    try refused("\"use_compressed_indexer\": true", "\"use_compressed_indexer\": false", error.Indexer, "use_compressed_indexer must be true");
}

test "a router, activation or norm the engine does not run is refused" {
    try refused("\"num_experts_per_tok\": 16", "\"num_experts_per_tok\": 600", error.Router, "not an expert shape");
    try refused("\"n_group\": 1", "\"n_group\": 8", error.Router, "no groups");
    try refused("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"", error.Activation, "hidden_act");
    try refused("\"sandwich_norm\": true", "\"sandwich_norm\": false", error.Norm, "sandwich_norm must be true");
}

test "layer tables that do not match the layer count or each other are refused" {
    try refused("\"num_hidden_layers\": 42,\n    \"num_key_value_heads\": 32", "\"num_hidden_layers\": 41,\n    \"num_key_value_heads\": 32", error.LayerList, "must list 41 layers");
    try refused("\"num_hidden_layers\": 42,\n    \"num_key_value_heads\": 32", "\"num_hidden_layers\": 99,\n    \"num_key_value_heads\": 32", error.TooManyLayers, "the loader holds");
}

test "Engram tables and a drafter the engine cannot place are refused" {
    try refused("\"engram_layer_ids\": [\n      2,", "\"engram_layer_ids\": [\n      43,", error.Engram, "outside 1 to 42");
    try refused("\"engram_n_head_per_ngram\": 8", "\"engram_n_head_per_ngram\": 7", error.Engram, "must divide the width");
    try refused("\"method\": \"dfly\"", "\"method\": \"mtp\"", error.Drafter, "no other drafter");
    try refused("\"dfly_num_hidden_layers\": 5", "\"dfly_num_hidden_layers\": 4", error.Drafter, "one backbone layer for each");
}
