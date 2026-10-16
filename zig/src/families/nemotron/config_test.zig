//! config.zig's checks: the Lightning config, ModelOpt and omni configs, per-module widths and their refusals.
const std = @import("std");
const cfg = @import("config.zig");

const Why = cfg.Why;
const Width = cfg.Width;
const parse = cfg.parse;
const checkShapes = cfg.checkShapes;
const lightning = cfg.lightning;
const Format = cfg.Format;
const Kind = cfg.Kind;

/// The Lightning config with its quantization block's text replaced by `block`.
fn withQuantization(block: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, std.testing.allocator, lightning, "\"quantization\": {\"group_size\": 64, \"bits\": 4}", block);
}

/// parse() must refuse the Lightning config with quantization `block`, its reason naming each of `words`.
fn expectRefused(block: []const u8, err: anyerror, words: []const []const u8) !void {
    const text = try withQuantization(block);
    defer std.testing.allocator.free(text);
    var why: Why = .{};
    try std.testing.expectError(err, parse(std.testing.allocator, text, &why));
    for (words) |word| if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    };
}

/// The Lightning config parsed with quantization `block`.
fn parsed(block: []const u8) !cfg.Config {
    const text = try withQuantization(block);
    defer std.testing.allocator.free(text);
    var why: Why = .{};
    return parse(std.testing.allocator, text, &why);
}

test "per-module entries give each matrix its own width as MLX reads them; the rest keep the checkpoint's" {
    const fc1 = "backbone.layers.1.mixer.switch_mlp.fc1";
    for ([_][]const u8{
        "\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\", \"lm_head\": true}",
        "\"quantization\": {\"bits\": 4, \"" ++ fc1 ++ "\": {\"group_size\": 64, \"bits\": 4}, \"lm_head\": {\"mode\": \"affine\"}}",
        "\"quantization_config\": {\"group_size\": 64, \"bits\": 4, \"quant_method\": \"mlx\", \"oq_note\": \"metadata\"}",
    }) |block| {
        const c = try parsed(block);
        try checkShapes(c);
        try std.testing.expectEqual(@as(u64, 0), c.widthsTag());
    }
    // oQ4's mix: experts at the checkpoint's 4/64, an 8-bit embedding and head, 5- and 6-bit mixer matrices
    const c = try parsed("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"backbone.embeddings\": {\"group_size\": 64, \"bits\": 8}, " ++
        "\"lm_head\": {\"bits\": 8}, \"backbone.layers.0.mixer.in_proj\": {\"group_size\": 64, \"bits\": 5}, " ++
        "\"language_model.backbone.layers.2.mixer.k_proj\": {\"group_size\": 32, \"bits\": 6}, \"backbone.layers.1.mixer.gate\": {\"bits\": 8}}");
    try checkShapes(c);
    try std.testing.expectEqual(Width{ .bits = 8 }, c.embed_width);
    try std.testing.expectEqual(Width{ .bits = 8 }, c.head_width);
    try std.testing.expectEqual(Width{ .bits = 5 }, c.widthOf(0, "in_proj"));
    try std.testing.expectEqual(Width{}, c.widthOf(0, "out_proj"));
    try std.testing.expectEqual(Width{ .bits = 6, .group = 32 }, c.widthOf(2, "k_proj"));
    try std.testing.expectEqual(Width{}, c.widthOf(1, "switch_mlp.fc1"));
    var buf: [96]u8 = undefined;
    try std.testing.expectEqualStrings("backbone.embeddings", c.other(&buf).?.name);
    try std.testing.expect(c.widthsTag() != 0);
    // oQ8's: one width for every matrix and no entries, which the shape check no longer ties to 4 bits
    const eight = try parsed("\"quantization\": {\"group_size\": 64, \"bits\": 8}");
    try checkShapes(eight);
    try std.testing.expectEqual(Width{ .bits = 8 }, eight.widthOf(1, "switch_mlp.fc2"));
    try std.testing.expect(eight.widthsTag() != 0 and eight.widthsTag() != c.widthsTag());
    try std.testing.expectEqualStrings("backbone.layers.0.mixer.in_proj", (try parsed("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"backbone.layers.0.mixer.in_proj\": {\"bits\": 6}}")).other(&buf).?.name);
}

test "a width no kernel reads, an unquantized matrix and another mode are refused by the module's name" {
    const fc1 = "backbone.layers.1.mixer.switch_mlp.fc1";
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"" ++ fc1 ++ "\": {\"group_size\": 64, \"bits\": 7}}", error.UnsupportedQuantization, &.{ fc1, "7-bit in groups of 64" });
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"backbone.embeddings\": {\"bits\": 6, \"group_size\": 16}}", error.UnsupportedQuantization, &.{ "backbone.embeddings", "6-bit in groups of 16" });
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": false}", error.UnsupportedQuantization, &.{ "lm_head", "unquantized" });
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": {}}", error.UnsupportedQuantization, &.{ "lm_head", "unquantized" });
    try expectRefused("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": {\"mode\": \"mxfp8\"}}", error.UnsupportedQuantization, &.{ "lm_head", "mxfp8" });
    try expectRefused("\"quantization\": {\"group_size\": 16, \"bits\": 4}", error.UnsupportedQuantization, &.{"4-bit in groups of 16"});
    try expectRefused("\"quantization\": {\"group_size\": 32, \"bits\": 4, \"mode\": \"mxfp4\"}", error.UnsupportedQuantization, &.{"mxfp4"});
    try expectRefused("\"quantization_config\": {\"quant_method\": \"fp8\", \"weight_block_size\": [128, 128]}", error.UnsupportedQuantization, &.{"fp8"});
    try expectRefused("\"quantization\": {\"group_size\": 64}", error.UnsupportedQuantization, &.{"bits"});
    try expectRefused("\"tie_word_embeddings\": false", error.UnsupportedQuantization, &.{"no quantization"});
}

test "a lesson's bf16 output projection (false) takes the 4-bit codes the loader makes; the rest the checkpoint's" {
    const c = try parsed("\"quantization\": {\"group_size\": 64, \"bits\": 6, \"backbone.layers.0.mixer.out_proj\": false}");
    try std.testing.expectEqual(Width{}, c.widthOf(0, "out_proj"));
    try std.testing.expectEqual(Width{ .bits = 6 }, c.widthOf(0, "in_proj"));
}

test "a ModelOpt checkpoint parses for the CUDA loader, which reads its formats from the tensors" {
    const text = try withQuantization("\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\"}");
    defer std.testing.allocator.free(text);
    var why: Why = .{};
    const c = try parse(std.testing.allocator, text, &why);
    try std.testing.expectEqual(Format.modelopt, c.format);
    try std.testing.expectEqual(@as(usize, 8), c.slots());
}

test "parse the Lightning config" {
    var why: Why = .{};
    const c = try parse(std.testing.allocator, lightning, &why);
    try std.testing.expectEqual(@as(usize, 3), c.layers);
    try std.testing.expectEqual(Kind.attention, c.kinds[2]);
    try std.testing.expectEqual(@as(usize, 10304), c.projDim());
    try std.testing.expectEqual(@as(usize, 6144), c.convDim());
    try std.testing.expectEqual(@as(usize, 4608), c.qkvDim());
    try std.testing.expectEqual(@as(usize, 8), c.slots());
    try std.testing.expect(c.isEos(11) and !c.isEos(3));
    try std.testing.expectEqual(std.math.inf(f32), c.dt_max);
    try checkShapes(c);
}

const super_text =
    \\{"model_type": "nemotron_h_omni", "quantization_config": {"quant_method": "modelopt_mixed"},
    \\ "tensorfold_tp": {"rank": 1, "world": 2, "routed_width": 1344, "shared_width": 2688, "scheme": "moe_width"},
    \\ "llm_config": {"model_type": "nemotron_h", "hidden_size": 4096, "vocab_size": 131072, "mamba_num_heads": 128,
    \\ "mamba_head_dim": 64, "n_groups": 8, "ssm_state_size": 128, "conv_kernel": 4, "num_attention_heads": 32,
    \\ "num_key_value_heads": 2, "head_dim": 128, "n_routed_experts": 512, "num_experts_per_tok": 22,
    \\ "moe_intermediate_size": 2688, "moe_shared_expert_intermediate_size": 5376, "moe_latent_size": 1024,
    \\ "routed_scaling_factor": 5.0, "layer_norm_epsilon": 1e-05, "eos_token_id": 11,
    \\ "time_step_limit": [0.0, {"__float__": "Infinity"}], "layers_block_type": ["mamba", "moe", "attention"]}}
;

test "nemotron_h_omni: the text config under llm_config, latent shares, a split checkpoint's share" {
    var why: Why = .{};
    const c = try parse(std.testing.allocator, super_text, &why);
    try std.testing.expect(c.omni and c.format == .modelopt and c.latent == 1024 and c.shares == 2 and c.held.? == 1);
    try std.testing.expectEqual(@as(usize, 22), c.slots());
    try std.testing.expectEqualStrings("language_model.", c.prefix());
    try std.testing.expectEqual(std.math.inf(f32), c.dt_max);
    try std.testing.expectEqual(@as(usize, 18560), c.projDim());
    const bad = try std.mem.replaceOwned(u8, std.testing.allocator, super_text, "\"routed_width\": 1344", "\"routed_width\": 1024");
    defer std.testing.allocator.free(bad);
    try std.testing.expectError(error.BadConfig, parse(std.testing.allocator, bad, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "tensorfold_tp") != null);
}

test "the latent router's bounds: 33 picks, 1,025 experts, zero picks and expert groups are refused by name" {
    const a = std.testing.allocator;
    const cases = [_][2][]const u8{
        .{ "\"num_experts_per_tok\": 22", "\"num_experts_per_tok\": 33" },
        .{ "\"num_experts_per_tok\": 22", "\"num_experts_per_tok\": 0" },
        .{ "\"n_routed_experts\": 512", "\"n_routed_experts\": 1025" },
        .{ "\"moe_latent_size\": 1024", "\"moe_latent_size\": 1024, \"n_group\": 2" },
        .{ "\"moe_latent_size\": 1024", "\"moe_latent_size\": 1024, \"mlp_hidden_act\": \"silu\"" },
    };
    for (cases) |k| {
        var why: Why = .{};
        const text = try std.mem.replaceOwned(u8, a, super_text, k[0], k[1]);
        defer a.free(text);
        try std.testing.expectError(error.BadConfig, parse(a, text, &why));
        try std.testing.expect(why.text().len > 0);
    }
    var why: Why = .{};
    const ok = try std.mem.replaceOwned(u8, a, super_text, "\"moe_latent_size\": 1024", "\"moe_latent_size\": 1024, \"n_group\": 1, \"topk_group\": 1");
    defer a.free(ok);
    _ = try parse(a, ok, &why);
}

test "CUDA and the Intel GPU engine refuse any width but 4/64 by name; other configs go on to their own checks" {
    const a = std.testing.allocator;
    const oq = try withQuantization("\"quantization\": {\"group_size\": 64, \"bits\": 4, \"lm_head\": {\"bits\": 8}}");
    defer a.free(oq);
    try std.testing.expectError(error.UnsupportedQuantization, cfg.fourOnlyText(a, oq, "CUDA"));
    const eight = try withQuantization("\"quantization\": {\"group_size\": 64, \"bits\": 8}");
    defer a.free(eight);
    try std.testing.expectError(error.UnsupportedQuantization, cfg.fourOnlyText(a, eight, "the Intel GPU engine"));
    try cfg.fourOnlyText(a, lightning, "CUDA");
    const modelopt = try withQuantization("\"quantization_config\": {\"quant_method\": \"modelopt\"}");
    defer a.free(modelopt);
    try cfg.fourOnlyText(a, modelopt, "CUDA");
    try cfg.fourOnlyText(a, "{\"model_type\": \"llama\"}", "CUDA");
}
