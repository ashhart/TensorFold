//! Admission from the supplied checkpoint, including malformed semantic fields and conflicting aliases.
const std = @import("std");
const config = @import("config.zig");
const generation = @import("generation.zig");
const t = std.testing;
const real = @embedFile("fixtures/config.json");

pub fn load() !config.Config {
    var why: config.Why = .{};
    return config.parse(t.allocator, real, &why);
}

test "gpt-oss real config preserves explicit heads, layers, routing and YaRN fields" {
    const c = try load();
    try t.expectEqual(@as(u32, 2880), c.hidden);
    try t.expectEqual(@as(u32, 4096), c.qWidth());
    try t.expectEqual(@as(u32, 512), c.kvWidth());
    try t.expectEqual(@as(u32, 32), c.experts);
    try t.expectEqual(@as(u32, 4), c.topk);
    try t.expectEqual(@as(u32, 131072), c.context);
    try t.expectEqual(@as(u32, 128), c.window);
    try t.expectEqual(@as(f64, 7), c.swiglu_limit);
    try t.expectEqual(@as(f64, 150000), c.rope_theta);
    try t.expectEqual(@as(f64, 32), c.yarn.factor);
    try t.expectEqual(@as(f64, 32), c.yarn.beta_fast);
    try t.expectEqual(@as(f64, 1), c.yarn.beta_slow);
    try t.expect(!c.yarn.truncate);
    for (0..24) |layer| try t.expectEqual(if (layer % 2 == 0) config.Kind.sliding else .full, try c.kind(layer));
    try t.expectError(error.BadLayer, c.kind(24));
}

fn refused(before: []const u8, after: []const u8, field: []const u8) !void {
    const changed = try std.mem.replaceOwned(u8, t.allocator, real, before, after);
    defer t.allocator.free(changed);
    try t.expect(!std.mem.eql(u8, changed, real));
    var why: config.Why = .{};
    if (config.parse(t.allocator, changed, &why)) |_| return error.AcceptedBadConfig else |_| {}
    try t.expect(std.mem.indexOf(u8, why.text(), field) != null);
}

test "gpt-oss refusals name the field before geometry or unsupported math can be used" {
    const Case = struct { []const u8, []const u8, []const u8 };
    for ([_]Case{
        .{ "\"gpt_oss\"", "\"other\"", "model_type" },
        .{ "GptOssForCausalLM", "OtherForCausalLM", "architectures" },
        .{ "\"hidden_size\": 2880", "\"hidden_size\": 0", "hidden_size" },
        .{ "\"hidden_size\": 2880", "\"hidden_size\": 2881", "hidden_size" },
        .{ "\"num_hidden_layers\": 24", "\"num_hidden_layers\": 129", "num_hidden_layers" },
        .{ "\"num_attention_heads\": 64", "\"num_attention_heads\": 63", "num_attention_heads" },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 0", "num_key_value_heads" },
        .{ "\"head_dim\": 64", "\"head_dim\": 0", "head_dim" },
        .{ "\"experts_per_token\": 4", "\"experts_per_token\": 33", "experts_per_token" },
        .{ "\"num_experts_per_tok\": 4", "\"num_experts_per_tok\": 3", "num_experts_per_tok" },
        .{ "\"num_local_experts\": 32", "\"num_local_experts\": -1", "num_local_experts" },
        .{ "\"intermediate_size\": 2880", "\"intermediate_size\": 31", "intermediate_size" },
        .{ "sliding_attention", "unknown_attention", "layer_types" },
        .{ "\"layer_types\":", "\"other_layer_types\":", "layer_types" },
        .{ "\"sliding_window\": 128", "\"sliding_window\": 0", "sliding_window" },
        .{ "\"attention_bias\": true", "\"attention_bias\": false", "attention_bias" },
        .{ "\"attention_dropout\": 0.0", "\"attention_dropout\": 0.1", "attention_dropout" },
        .{ "\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"", "hidden_act" },
        .{ "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true", "tie_word_embeddings" },
        .{ "\"rms_norm_eps\": 1e-05", "\"rms_norm_eps\": -1", "rms_norm_eps" },
        .{ "\"rope_theta\": 150000", "\"rope_theta\": 0", "rope_theta" },
        .{ "\"rope_type\": \"yarn\"", "\"rope_type\": \"linear\"", "rope_scaling.rope_type" },
        .{ "\"factor\": 32.0", "\"factor\": 0", "rope_scaling.factor" },
        .{ "\"beta_slow\": 1.0", "\"beta_slow\": 33", "rope_scaling.beta_slow" },
        .{ "\"truncate\": false", "\"truncate\": true", "rope_scaling.truncate" },
        .{ "\"original_max_position_embeddings\": 4096", "\"original_max_position_embeddings\": 2048", "rope_scaling.original_max_position_embeddings" },
        .{ "\"quant_method\": \"mxfp4\"", "\"quant_method\": \"nvfp4\"", "quantization_config.quant_method" },
        .{ "model.layers.*.mlp.router", "model.layers.*.mlp.other", "quantization_config.modules_to_not_convert" },
        .{ "\"eos_token_id\": 200002", "\"eos_token_id\": 201088", "eos_token_id" },
        .{ "\"use_cache\": true", "\"use_cache\": false", "use_cache" },
        .{ "\"output_router_logits\": false", "\"output_router_logits\": true", "output_router_logits" },
        .{ "\"swiglu_limit\": 7.0", "\"swiglu_limit\": 0", "swiglu_limit" },
        .{ "\"vocab_size\": 201088", "\"vocab_size\": 201088, \"unknown_attention\": true", "unknown_attention" },
    }) |c| try refused(c[0], c[1], c[2]);
}

test "gpt-oss layer schedule is read rather than regenerated from parity" {
    const changed = try std.mem.replaceOwned(u8, t.allocator, real, "sliding_attention", "full_attention");
    defer t.allocator.free(changed);
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, changed, &why);
    for (0..c.layers) |layer| try t.expectEqual(config.Kind.full, try c.kind(layer));
}

test "gpt-oss generation stop IDs retain all three supplied endings" {
    const c = try load();
    var why: config.Why = .{};
    const g = try generation.parse(t.allocator, @embedFile("fixtures/generation_config.json"), c.vocab, &why);
    try t.expect(g.sample);
    try t.expectEqual(@as(u32, 199998), g.bos);
    try t.expectEqualSlices(u32, &.{ 200002, 199999, 200012 }, g.eos[0..g.eos_n]);
    try t.expect(g.stops(200012) and !g.stops(200007));
    try t.expectError(error.BadGptOssConfig, generation.parse(t.allocator, "{\"bos_token_id\":-1}", c.vocab, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "bos_token_id") != null);
}
