//! Config parsing: the real checkpoint's config.json, the tiny-random one, the tiny Mini, and what parse refuses.
const std = @import("std");
const config = @import("config.zig");

const t = std.testing;
pub const real = @embedFile("fixtures/config.json");
pub const tiny = @embedFile("fixtures/tiny_config.json");
pub const mini = @embedFile("fixtures/mini_config.json");
pub const tiny_mini = @embedFile("fixtures/tiny_mini_config.json");

fn refused(json: []const u8, from: []const u8, to: []const u8, want: anyerror, says: []const u8) !void {
    if (std.mem.count(u8, json, from) != 1) return error.EditAnchor;
    const edited = try std.mem.replaceOwned(u8, t.allocator, json, from, to);
    defer t.allocator.free(edited);
    var why: config.Why = .{};
    try t.expectError(want, config.parse(t.allocator, edited, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), says) != null);
}

test "Inkling 2 Small's config.json parses: 48 layers, 8 global, 384 experts, 8 per token" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, real, &why);
    try t.expectEqual(@as(u32, 4608), c.text.hidden);
    try t.expectEqual(@as(u32, 48), c.text.layers);
    try t.expectEqual(@as(u32, 201024), c.text.vocab);
    try t.expectEqual(@as(u32, 200058), c.text.unpadded_vocab);
    try t.expectEqual(config.Heads{ .q = 48, .kv = 8, .dim = 128 }, c.text.global);
    try t.expectEqual(config.Heads{ .q = 48, .kv = 8, .dim = 128 }, c.text.local);
    try t.expectEqual(@as(u32, 1024), c.text.window);
    try t.expectEqual(@as(u32, 16), c.text.d_rel);
    try t.expectEqual(@as(u32, 6), c.text.block);
    try t.expectEqual(@as(u32, 384), c.text.experts);
    try t.expectEqual(@as(u32, 8), c.text.topk);
    try t.expectEqual(@as(u32, 2304), c.text.expert_inter);
    try t.expectEqual(@as(f32, 8), c.text.route_scale);
    try t.expectEqual(@as(f32, 30), c.text.logit_cap);
    try t.expectEqual(@as(u32, 200006), c.eos);
    try t.expectEqual(@as(u32, 200054), c.image_token);
    var global: u32 = 0;
    for (0..c.text.layers) |i| global += @intFromBool(config.isGlobal(@intCast(i)));
    try t.expectEqual(@as(u32, 8), global);
    try t.expectEqual(@as(u32, 23), c.vision.layers);
    try t.expectEqual(@as(u32, 1024), c.vision.hidden);
    const m = c.mtp.?;
    try t.expectEqual(@as(u32, 5), m.layers);
    try t.expectEqual(@as(u32, 256), m.rank);
    try t.expectEqualSlices(u32, &.{ 5, 11, 17, 23, 29, 35, 41, 48 }, m.taps[0..m.taps_n]);
    try t.expectEqualSlices(bool, &.{ true, true, false, true, true }, m.local[0..5]);
}

test "Inkling 2 Mini's config.json parses: 24 weight layers, 48 visits and four drafter taps" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, mini, &why);
    try t.expectEqual(@as(u32, 2304), c.text.hidden);
    try t.expectEqual(@as(u32, 24), c.text.layers);
    try t.expectEqual(@as(u32, 48), c.text.visitCount());
    try t.expectEqual(config.Heads{ .q = 24, .kv = 8, .dim = 128 }, c.text.global);
    try t.expectEqual(@as(u32, 8192), c.text.log_floor);
    try t.expectEqual(@as(u32, 128), c.text.experts);
    try t.expectEqual(@as(u32, 1152), c.text.expert_inter);
    try t.expectEqual(@as(f32, 8), c.text.route_scale);
    const loop = c.text.loop.?;
    try t.expectEqual(@as(u32, 6), loop.core_start);
    try t.expectEqual(@as(u32, 18), loop.core_end);
    try t.expectEqual(@as(u32, 3), loop.passes);
    try t.expectEqual(@as(u32, 64), loop.gate_rank);
    try t.expectEqual(@as(u32, 2304), c.vision.text_hidden);
    try t.expectEqual(@as(f32, 1e-5), c.vision.eps);
    const m = c.mtp.?;
    try t.expectEqual(@as(u32, 3), m.layers);
    try t.expectEqual(@as(u32, 9216), m.inter);
    try t.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, m.taps[0..m.taps_n]);
    try t.expectEqualSlices(bool, &.{ true, false, true }, m.local[0..3]);
}

test "loop layouts the engine does not run are refused by their config key" {
    try refused(mini, "\"is_varloop\": false", "\"is_varloop\": true", error.Unsupported, "is_varloop must be false");
    try refused(mini, "\"core_start\": 6", "\"core_start\": 0", error.Layout, "must start after layer 0");
    try refused(mini, "\"core_start\": 6", "\"core_start\": 7", error.Layout, "multiples of the 6-layer");
    try refused(mini, "\"core_end\": 18", "\"core_end\": 24", error.Layout, "multiples of the 6-layer");
    try refused(mini, "\"core_end\": 18,", "\"core_end\": 30,", error.Layout, "fit inside the 24 physical layers");
    try refused(mini, "Inkling2LoopForConditionalGeneration", "Inkling2ForConditionalGeneration", error.NotInkling2, "model_type and architectures must match");
    try refused(mini, "\"model_type\": \"inkling2_loop\"", "\"model_type\": \"inkling2\"", error.NotInkling2, "model_type and architectures must match");
}

test "tiny-random's config.json parses: 12 layers, 96 experts, width 128, no quantization_config" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, tiny, &why);
    try t.expectEqual(@as(u32, 128), c.text.hidden);
    try t.expectEqual(@as(u32, 12), c.text.layers);
    try t.expectEqual(@as(u32, 96), c.text.experts);
    try t.expectEqual(@as(u32, 4), c.vision.layers);
    try t.expectEqualSlices(u32, &.{ 5, 12 }, c.mtp.?.taps[0..c.mtp.?.taps_n]);
    try t.expectEqual(@as(u32, 1024), c.text.extent(0));
    try t.expectEqual(@as(u32, 1024), c.text.extent(5));
}

test "tiny-inkling2-mini's config.json parses: 18 weight layers, a 6-layer core, 30 visits and no drafter" {
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, tiny_mini, &why);
    try t.expectEqual(@as(u32, 256), c.text.hidden);
    try t.expectEqual(@as(u32, 18), c.text.layers);
    try t.expectEqual(@as(u32, 30), c.text.visitCount());
    try t.expectEqual(config.Heads{ .q = 4, .kv = 2, .dim = 128 }, c.text.global);
    try t.expectEqual(config.Heads{ .q = 4, .kv = 2, .dim = 128 }, c.text.local);
    try t.expectEqual(@as(u32, 8192), c.text.log_floor);
    try t.expectEqual(@as(u32, 16), c.text.experts);
    try t.expectEqual(@as(u32, 4), c.text.topk);
    try t.expectEqual(@as(u32, 256), c.text.expert_inter);
    try t.expectEqual(@as(u32, 128), c.text.extent(0));
    try t.expectEqual(@as(u32, 256), c.text.extent(5));
    const loop = c.text.loop.?;
    try t.expectEqual(@as(u32, 6), loop.core_start);
    try t.expectEqual(@as(u32, 12), loop.core_end);
    try t.expectEqual(@as(u32, 3), loop.passes);
    try t.expectEqual(@as(u32, 64), loop.gate_rank);
    try t.expectEqual(@as(u32, 256), c.vision.text_hidden);
    try t.expectEqual(@as(u32, 2), c.vision.layers);
    try t.expect(c.mtp == null);
}

test "a config that is not Inkling 2 is refused by name" {
    try refused(tiny, "\"model_type\": \"inkling2\"", "\"model_type\": \"llama\"", error.NotInkling2, "not an Inkling 2 checkpoint");
    try refused(tiny, "Inkling2ForConditionalGeneration", "Inkling2ForCausalLM", error.NotInkling2, "not an Inkling 2 checkpoint");
}

test "layouts the engine does not run are refused with the key that differs" {
    try refused(tiny, "\"rel_extent\": 1024,\n    \"q_bias\": false", "\"rel_extent\": 1024,\n    \"q_bias\": true", error.Unsupported, "q_bias must be false");
    try refused(tiny, "\"use_value_norm_no_gamma\": true", "\"use_value_norm_no_gamma\": false", error.Unsupported, "use_value_norm_no_gamma must be true");
    try refused(tiny, "\"n_shared_experts\": 0", "\"n_shared_experts\": 1", error.Unsupported, "n_shared_experts must be 0");
    try refused(tiny, "\"sconv_kernel_size\": 4", "\"sconv_kernel_size\": 3", error.Unsupported, "sconv_kernel_size must be 4");
    try refused(tiny, "\"use_gate_bias\": true,\n    \"tie_word_embeddings\": false", "\"use_gate_bias\": true,\n    \"tie_word_embeddings\": true", error.Unsupported, "tie_word_embeddings");
}

test "a layer list that is not every sixth layer global is refused, and a missing key is named" {
    try refused(tiny, "4,\n      6,", "4,\n      5,", error.Layout, "global, repeated or missing layer");
    try refused(real, "\"n_layers_per_attn_res_block\":", "\"n_layers_per_block\":", error.BadConfig, "n_layers_per_attn_res_block");
}

test "geometry the experts cannot hold and a non-NVFP4 quantization are refused, each by its key" {
    try refused(tiny, "\"hidden_size\": 128,\n    \"num_hidden_layers\": 12", "\"hidden_size\": 100,\n    \"num_hidden_layers\": 12", error.Layout, "hidden_size 100 is not a multiple of 32");
    try refused(tiny, "\"intermediate_size\": 128,\n    \"mlp_gate_softcap\"", "\"intermediate_size\": 100,\n    \"mlp_gate_softcap\"", error.Layout, "intermediate_size 100 is not a multiple of 32");
    try refused(tiny, "\"num_experts_per_tok\": 8", "\"num_experts_per_tok\": 97", error.Layout, "num_experts_per_tok 97 is past n_routed_experts 96");
    try refused(tiny, "\"unpadded_vocab_size\": 200058", "\"unpadded_vocab_size\": 201025", error.Layout, "unpadded_vocab_size 201025 is past vocab_size 201024");
    try refused(tiny, "\"num_attention_heads\": 8,\n    \"num_key_value_heads\": 4,\n    \"head_dim\": 128,\n    \"d_rel\"", "\"num_attention_heads\": 9,\n    \"num_key_value_heads\": 4,\n    \"head_dim\": 128,\n    \"d_rel\"", error.Layout, "do not split");
    try refused(real, "\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"FP8\"", error.Quant, "ModelOpt NVFP4");
    try refused(tiny, "\"eos_token_id\": 200006", "\"eos_token_id\": 300000", error.BadConfig, "eos_token_id");
}

test "a config without a vision tower is refused rather than read as text only" {
    try refused(tiny, "\"vision_config\"", "\"vision_cfg\"", error.BadConfig, "vision_config");
}

test "the draft head is refused by key when it is not the one the engine runs" {
    try refused(tiny, "\"markov_head_type\": \"rnn_hidden\"", "\"markov_head_type\": \"gru\"", error.Layout, "rnn_hidden Markov head");
    try refused(tiny, "\"hidden_size\": 128,\n    \"num_hidden_layers\": 5", "\"hidden_size\": 64,\n    \"num_hidden_layers\": 5", error.Layout, "the backbone's width");
    try refused(tiny, "      3,\n      4\n    ],\n    \"sliding_window_size\"", "      3,\n      9\n    ],\n    \"sliding_window_size\"", error.Layout, "names a layer the head does not have");
    try refused(tiny, "      5,\n      12\n    ]", "      5,\n      13\n    ]", error.Layout, "names a layer the backbone does not have");
}

test "the vision tower is refused by key when it is not the one the engine reads" {
    try refused(tiny, "\"projector_bias\": false", "\"projector_bias\": true", error.Unsupported, "projector_bias must be false");
    try refused(tiny, "\"mlp_ratio\": 4.0", "\"mlp_ratio\": 4.5", error.Layout, "whole mlp_ratio");
    try refused(tiny, "\"num_attention_heads\": 4,\n    \"mlp_ratio\"", "\"num_attention_heads\": 5,\n    \"mlp_ratio\"", error.Layout, "heads that divide the width");
    try refused(tiny, "\"text_hidden_size\": 128", "\"text_hidden_size\": 256", error.Layout, "text_hidden_size equal to hidden_size");
}

test "an mlp_ratio past 1024 is refused instead of overflowing, as is a block longer than the model" {
    try refused(tiny, "\"mlp_ratio\": 4.0", "\"mlp_ratio\": 1e30", error.Layout, "at most 1024");
    try refused(tiny, "\"mlp_ratio\": 4.0", "\"mlp_ratio\": 1e39", error.Layout, "at most 1024");
    try refused(tiny, "\"n_layers_per_attn_res_block\": 6", "\"n_layers_per_attn_res_block\": 13", error.Layout, "is past num_hidden_layers");
    try refused(tiny, "\"n_layers_per_attn_res_block\": 6", "\"n_layers_per_attn_res_block\": 4294967295", error.Layout, "is past num_hidden_layers");
}

test "attention-residual blocks past the mixes' eight and widths the attention kernels do not run are refused" {
    var why: config.Why = .{};
    try t.expectEqual(@as(u32, config.max_blocks), (try config.parse(t.allocator, real, &why)).text.layers / 6);
    try refused(real, "\"n_layers_per_attn_res_block\": 6", "\"n_layers_per_attn_res_block\": 5", error.Layout, "into more than the residual mixes' 8 blocks");
    try refused(tiny, "\"n_layers_per_attn_res_block\": 6", "\"n_layers_per_attn_res_block\": 1", error.Layout, "splits 12 layers into more than");
    try refused(tiny, "\"head_dim\": 128,\n    \"d_rel\"", "\"head_dim\": 64,\n    \"d_rel\"", error.Unsupported, "head_dim 64 is not 128");
    try refused(tiny, "\"swa_head_dim\": 128", "\"swa_head_dim\": 256", error.Unsupported, "swa_head_dim 256 is not 128");
    try refused(tiny, "\"d_rel\": 16", "\"d_rel\": 32", error.Unsupported, "d_rel 32 is not 16");
}

test "a scoring function, a layer count, a quantization_config and a root that the engine cannot read are refused" {
    try refused(tiny, "\"use_gate_bias\": true,\n    \"tie_word_embeddings\"", "\"use_gate_bias\": true,\n    \"scoring_func\": \"softmax\",\n    \"tie_word_embeddings\"", error.Unsupported, "scoring_func must be");
    try refused(tiny, "\"hidden_size\": 128,\n    \"num_hidden_layers\": 12", "\"hidden_size\": 128,\n    \"num_hidden_layers\": 129", error.Layout, "129 layers: the plan tables hold 128");
    try refused(tiny, "\"model_type\": \"inkling2\",", "\"model_type\": \"inkling2\",\n  \"quantization_config\": 5,", error.Quant, "quantization_config is not an object");
    var why: config.Why = .{};
    try t.expectError(error.BadConfig, config.parse(t.allocator, "[]", &why));
    try t.expectEqualStrings("config.json is not an object", why.text());
    try t.expectError(error.BadConfig, config.parse(t.allocator, "{", &why));
    try t.expectEqualStrings("config.json is not JSON", why.text());
}

test "kortexa-inkling2-exl3-v1 quantization_config is accepted on Small and refused on Mini or other formats" {
    const from =
        \\  "quantization_config": {
        \\    "quant_method": "modelopt",
        \\    "quant_algo": "NVFP4"
        \\  }
    ;
    const to =
        \\  "quantization_config": {
        \\    "quant_method": "exl3",
        \\    "format": "kortexa-inkling2-exl3-v1"
        \\  }
    ;
    const edited = try std.mem.replaceOwned(u8, t.allocator, real, from, to);
    defer t.allocator.free(edited);
    var why: config.Why = .{};
    const c = try config.parse(t.allocator, edited, &why);
    try t.expectEqual(config.Quant.exl3_kortexa_v1, c.quant);
    try refused(edited, "kortexa-inkling2-exl3-v1", "unknown-exl3-v2", error.Quant, "kortexa-inkling2-exl3-v1");
    try refused(mini, from, to, error.Quant, "Inkling 2 Small");
}
