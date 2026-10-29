//! Checkpoint-derived config fixtures and refusal cases for the first LFM3 admission slice.
const std = @import("std");
const c = @import("config.zig");

test {
    _ = @import("config_fp8_test.zig");
}

test "text and VL preserve their distinct context and actual mixed embedding width" {
    var t = try c.parse(std.testing.allocator, text);
    defer t.deinit();
    var v = try c.parse(std.testing.allocator, vl);
    defer v.deinit();
    try std.testing.expectEqual(c.Kind.text, t.kind);
    try std.testing.expectEqual(c.Kind.vision_language, v.kind);
    try std.testing.expectEqual(@as(usize, 262144), t.context);
    try std.testing.expectEqual(@as(usize, 128000), v.context);
    try std.testing.expectEqual(c.Width{ .bits = 4, .group = 64 }, try t.width("model.embed_tokens"));
    try std.testing.expectEqual(c.Width{ .bits = 8, .group = 32 }, try v.width("model.embed_tokens.weight"));
    try std.testing.expectEqual(c.Width{ .bits = 4, .group = 32 }, try v.width("layers.0.conv.in_proj"));
    try std.testing.expectEqual(c.LayerType.sliding_attention, t.layer_types[4]);
    try std.testing.expectEqual(@as(f32, 0.5), t.full_rotary.partial);
    try std.testing.expectEqual(@as(f32, 1), t.sliding_rotary.partial);
}

test "6 and 8 bit affine variants use the same config parser" {
    inline for (.{ "6", "8" }) |bits| {
        const replacement = "\"bits\": " ++ bits;
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, text, "\"bits\": 4", replacement);
        defer std.testing.allocator.free(bytes);
        var config = try c.parse(std.testing.allocator, bytes);
        defer config.deinit();
        try std.testing.expectEqual(try std.fmt.parseInt(u8, bits, 10), config.quant.global.bits);
    }
}

test "attention output and gate projections honor module widths and reject out_proj" {
    inline for (.{ "o_proj", "gate_proj" }) |name| {
        const replacement = "\"language_model.model.layers.2.self_attn." ++ name ++ "\": {";
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, vl, "\"language_model.model.embed_tokens\": {", replacement);
        defer std.testing.allocator.free(bytes);
        var config = try c.parse(std.testing.allocator, bytes);
        defer config.deinit();
        try std.testing.expectEqual(c.Width{ .bits = 8, .group = 32 }, try config.width("model.layers.2.self_attn." ++ name));
        try std.testing.expectEqual(c.Width{ .bits = 4, .group = 32 }, try config.width("model.layers.6.self_attn." ++ name));
    }
    const bad = try std.mem.replaceOwned(u8, std.testing.allocator, vl, "\"language_model.model.embed_tokens\": {", "\"language_model.model.layers.2.self_attn.out_proj\": {");
    defer std.testing.allocator.free(bad);
    try std.testing.expectError(error.UnsupportedLfm3Quantization, c.parse(std.testing.allocator, bad));
}

test "layer override indices must use canonical decimal spelling" {
    inline for (.{ "02", "+2", "00", "-2", "" }) |index| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, vl, "\"language_model.model.embed_tokens\": {", "\"language_model.model.layers." ++ index ++ ".self_attn.gate_proj\": {");
        defer std.testing.allocator.free(bytes);
        try std.testing.expectError(error.UnsupportedLfm3Quantization, c.parse(std.testing.allocator, bytes));
    }
    const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, vl, "\"language_model.model.embed_tokens\": {", "\"language_model.model.layers.0.conv.in_proj\": {");
    defer std.testing.allocator.free(bytes);
    var config = try c.parse(std.testing.allocator, bytes);
    defer config.deinit();
    try std.testing.expectEqual(c.Width{ .bits = 8, .group = 32 }, try config.width("layers.0.conv.in_proj"));
}

test "geometry, mixer schedule, rotary, sandwich and unsupported quantization fail admission" {
    const cases = .{
        .{ "\"hidden_size\": 2048", "\"hidden_size\": 1024", error.UnsupportedLfm3Geometry },
        .{ "\"conv_L_cache\": 7", "\"conv_L_cache\": 3", error.UnsupportedLfm3Geometry },
        .{ "\"sliding_attention\",", "\"full_attention\",", error.UnsupportedLfm3LayerSchedule },
        .{ "\"partial_rotary_factor\": 0.5", "\"partial_rotary_factor\": 1.0", error.UnsupportedLfm3Config },
        .{ "\"use_sandwich_norm\": true", "\"use_sandwich_norm\": false", error.UnsupportedLfm3Config },
        .{ "\"bits\": 4", "\"bits\": 5", error.UnsupportedLfm3Quantization },
        .{ "\"group_size\": 64", "\"group_size\": 128", error.UnsupportedLfm3Quantization },
        .{ "\"max_position_embeddings\": 262144", "\"max_position_embeddings\": 128000", error.UnsupportedLfm3Context },
    };
    inline for (cases) |case| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, text, case[0], case[1]);
        defer std.testing.allocator.free(bytes);
        try std.testing.expectError(case[2], c.parse(std.testing.allocator, bytes));
    }
}

test "quantization declarations must agree and reject malformed or unknown overrides" {
    const mismatch = try std.mem.replaceOwned(u8, std.testing.allocator, vl, "\"quantization_config\": {\n    \"group_size\": 32", "\"quantization_config\": {\n    \"group_size\": 64");
    defer std.testing.allocator.free(mismatch);
    try std.testing.expectError(error.UnsupportedLfm3Quantization, c.parse(std.testing.allocator, mismatch));
    const cases = .{
        .{ "\"language_model.model.embed_tokens\": {", "\"unknown_module\": {" },
        .{ "\"language_model.model.embed_tokens\": {", "\"language_model.model.layers.99.conv.in_proj\": {" },
        .{ "\"mode\": \"affine\"", "\"mode\": \"mxfp4\"" },
    };
    inline for (cases) |case| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, vl, case[0], case[1]);
        defer std.testing.allocator.free(bytes);
        const result = c.parse(std.testing.allocator, bytes);
        if (result) |value| {
            var config = value;
            config.deinit();
            return error.TestExpectedError;
        } else |_| {}
    }
}

test "VL projector and vision geometry are not silently accepted as the known model" {
    const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, vl, "\"patch_size\": 16", "\"patch_size\": 14");
    defer std.testing.allocator.free(bytes);
    try std.testing.expectError(error.UnsupportedLfm3Geometry, c.parse(std.testing.allocator, bytes));
}

test "malformed JSON fails without retaining allocations" {
    try std.testing.expectError(error.UnexpectedEndOfInput, c.parse(std.testing.allocator, "{"));
}

test "the ModelOpt NVFP4 export parses as its own scheme: FP4 g16 FFN, bf16 mixers" {
    var config = try c.parse(std.testing.allocator, nvfp4);
    defer config.deinit();
    try std.testing.expectEqual(c.Scheme.modelopt_nvfp4, config.scheme);
    try std.testing.expectEqual(c.nvfp4_width, try config.width("model.layers.0.feed_forward.w1"));
    try std.testing.expectEqual(@as(usize, 262144), config.context);
    var mlx = try c.parse(std.testing.allocator, text);
    defer mlx.deinit();
    try std.testing.expectEqual(c.Scheme.mlx_affine, mlx.scheme);
}

test "Metal refuses the ModelOpt NVFP4 export by name; CUDA serves it, and both serve the MLX layouts" {
    var config = try c.parse(std.testing.allocator, nvfp4);
    defer config.deinit();
    try std.testing.expectError(error.Lfm3ModeloptNvfp4NotOnMetal, c.servedOn(&config, .metal));
    try c.servedOn(&config, .cuda);
    var mlx = try c.parse(std.testing.allocator, text);
    defer mlx.deinit();
    try c.servedOn(&mlx, .metal);
    try c.servedOn(&mlx, .cuda);
}

test "an NVFP4 export with another recipe, a missing or extra exclusion, or dynamic inputs is refused" {
    const edits = [_][2][]const u8{
        .{ "\"quant_algo\": \"NVFP4\"", "\"quant_algo\": \"FP8\"" },
        .{ "\"model.layers.9.conv*\"", "\"model.layers.9.self_attn*\"" },
        .{ "      \"model.layers.9.conv*\"\n", "      \"model.layers.9.conv*\",\n      \"model.layers.9.feed_forward*\"\n" },
        .{ "      \"model.layers.8.self_attn*\",\n", "" },
        .{ "\"model.layers.10.self_attn*\"", "\"model.layers.2.self_attn*\"" },
        .{ "\"dynamic\": false", "\"dynamic\": true" },
        .{ "\"group_size\": 16", "\"group_size\": 32" },
        .{ "\"Linear\"", "\"Conv1d\"" },
    };
    for (edits) |e| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, nvfp4, e[0], e[1]);
        defer std.testing.allocator.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, nvfp4));
        try std.testing.expectError(error.UnsupportedLfm3Quantization, c.parse(std.testing.allocator, bytes));
    }
}

pub const text =
    \\{
    \\  "model_type": "lfm3",
    \\  "conv_L_cache": 7,
    \\  "conv_bias": false,
    \\  "dtype": "bfloat16",
    \\  "head_dim": 128,
    \\  "hidden_size": 2048,
    \\  "intermediate_size": 7680,
    \\  "layer_types": [
    \\    "conv",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv"
    \\  ],
    \\  "max_position_embeddings": 262144,
    \\  "norm_eps": 1e-05,
    \\  "num_attention_heads": 16,
    \\  "num_hidden_layers": 24,
    \\  "num_key_value_heads": 4,
    \\  "quantization": {
    \\    "group_size": 64,
    \\    "bits": 4,
    \\    "mode": "affine"
    \\  },
    \\  "quantization_config": {
    \\    "group_size": 64,
    \\    "bits": 4,
    \\    "mode": "affine"
    \\  },
    \\  "rope_parameters": {
    \\    "full_attention": {
    \\      "partial_rotary_factor": 0.5,
    \\      "rope_theta": 1000000.0,
    \\      "rope_type": "default"
    \\    },
    \\    "sliding_attention": {
    \\      "rope_theta": 10000,
    \\      "rope_type": "default"
    \\    }
    \\  },
    \\  "sliding_window": 256,
    \\  "tie_word_embeddings": true,
    \\  "use_sandwich_norm": true,
    \\  "vocab_size": 128000
    \\}
;

pub const vl =
    \\{
    \\  "downsample_factor": 2,
    \\  "dtype": "bfloat16",
    \\  "image_token_id": 125030,
    \\  "model_type": "lfm3_vl",
    \\  "projector_bias": true,
    \\  "projector_hidden_act": "gelu",
    \\  "projector_hidden_size": 4096,
    \\  "projector_layer_norm_eps": 1e-06,
    \\  "text_config": {
    \\    "conv_L_cache": 7,
    \\    "conv_bias": false,
    \\    "dtype": "bfloat16",
    \\    "head_dim": 128,
    \\    "hidden_size": 2048,
    \\    "intermediate_size": 7680,
    \\    "layer_types": [
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "sliding_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "sliding_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "sliding_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "sliding_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "sliding_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv"
    \\    ],
    \\    "max_position_embeddings": 128000,
    \\    "model_type": "lfm3",
    \\    "norm_eps": 1e-05,
    \\    "num_attention_heads": 16,
    \\    "num_hidden_layers": 24,
    \\    "num_key_value_heads": 4,
    \\    "rope_parameters": {
    \\      "full_attention": {
    \\        "partial_rotary_factor": 0.5,
    \\        "rope_theta": 1000000.0,
    \\        "rope_type": "default"
    \\      },
    \\      "sliding_attention": {
    \\        "rope_theta": 10000,
    \\        "rope_type": "default"
    \\      }
    \\    },
    \\    "sliding_window": 256,
    \\    "tie_word_embeddings": true,
    \\    "use_sandwich_norm": true,
    \\    "vocab_size": 128000
    \\  },
    \\  "tie_word_embeddings": true,
    \\  "video_token_id": 125029,
    \\  "vision_config": {
    \\    "conv_kernel_size": 3,
    \\    "dtype": "bfloat16",
    \\    "hidden_size": 1024,
    \\    "in_channels": 3,
    \\    "initializer_range": 0.02,
    \\    "intermediate_size": 2560,
    \\    "layer_types": [
    \\      "conv",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv",
    \\      "full_attention",
    \\      "conv"
    \\    ],
    \\    "model_type": "lfm3_vl_vision",
    \\    "norm_eps": 1e-05,
    \\    "num_attention_heads": 16,
    \\    "num_hidden_layers": 14,
    \\    "num_key_value_heads": 8,
    \\    "patch_size": 16,
    \\    "rope_parameters": {
    \\      "rope_theta": 1000000.0,
    \\      "rope_type": "axial"
    \\    },
    \\    "spatial_merge_size": 2,
    \\    "temporal_patch_size": 2
    \\  },
    \\  "vision_end_token_id": 125010,
    \\  "vision_start_token_id": 125009,
    \\  "quantization": {
    \\    "group_size": 32,
    \\    "bits": 4,
    \\    "mode": "affine",
    \\    "language_model.model.embed_tokens": {
    \\      "group_size": 32,
    \\      "bits": 8,
    \\      "mode": "affine"
    \\    }
    \\  },
    \\  "quantization_config": {
    \\    "group_size": 32,
    \\    "bits": 4,
    \\    "mode": "affine",
    \\    "language_model.model.embed_tokens": {
    \\      "group_size": 32,
    \\      "bits": 8,
    \\      "mode": "affine"
    \\    }
    \\  }
    \\}
;

/// LiquidAI/LFM3-1.7B-NVFP4 (b30025bb)'s config.json as published.
pub const nvfp4 =
    \\{
    \\  "architectures": [
    \\    "Lfm3ForCausalLM"
    \\  ],
    \\  "auto_map": {
    \\    "AutoConfig": "configuration_lfm3.Lfm3Config",
    \\    "AutoModel": "modeling_lfm3.Lfm3Model",
    \\    "AutoModelForCausalLM": "modeling_lfm3.Lfm3ForCausalLM"
    \\  },
    \\  "bos_token_id": 124894,
    \\  "conv_L_cache": 7,
    \\  "conv_bias": false,
    \\  "dtype": "bfloat16",
    \\  "eos_token_id": 124900,
    \\  "head_dim": 128,
    \\  "hidden_size": 2048,
    \\  "initializer_range": 0.02,
    \\  "intermediate_size": 7680,
    \\  "layer_types": [
    \\    "conv",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv"
    \\  ],
    \\  "max_position_embeddings": 262144,
    \\  "model_type": "lfm3",
    \\  "norm_eps": 1e-05,
    \\  "num_attention_heads": 16,
    \\  "num_hidden_layers": 24,
    \\  "num_key_value_heads": 4,
    \\  "pad_token_id": 124893,
    \\  "rope_parameters": {
    \\    "full_attention": {
    \\      "partial_rotary_factor": 0.5,
    \\      "rope_theta": 1000000.0,
    \\      "rope_type": "default"
    \\    },
    \\    "sliding_attention": {
    \\      "rope_theta": 10000,
    \\      "rope_type": "default"
    \\    }
    \\  },
    \\  "sliding_window": 256,
    \\  "tie_word_embeddings": true,
    \\  "transformers_version": "5.12.1",
    \\  "use_cache": true,
    \\  "use_sandwich_norm": true,
    \\  "vocab_size": 128000,
    \\  "quantization_config": {
    \\    "config_groups": {
    \\      "group_0": {
    \\        "input_activations": {
    \\          "dynamic": false,
    \\          "num_bits": 4,
    \\          "type": "float",
    \\          "group_size": 16
    \\        },
    \\        "weights": {
    \\          "dynamic": false,
    \\          "num_bits": 4,
    \\          "type": "float",
    \\          "group_size": 16
    \\        },
    \\        "targets": [
    \\          "Linear"
    \\        ]
    \\      }
    \\    },
    \\    "ignore": [
    \\      "lm_head",
    \\      "model.embed_tokens",
    \\      "model.layers.0.conv*",
    \\      "model.layers.1.conv*",
    \\      "model.layers.10.self_attn*",
    \\      "model.layers.11.conv*",
    \\      "model.layers.12.self_attn*",
    \\      "model.layers.13.conv*",
    \\      "model.layers.14.self_attn*",
    \\      "model.layers.15.conv*",
    \\      "model.layers.16.self_attn*",
    \\      "model.layers.17.conv*",
    \\      "model.layers.18.self_attn*",
    \\      "model.layers.19.conv*",
    \\      "model.layers.2.self_attn*",
    \\      "model.layers.20.self_attn*",
    \\      "model.layers.21.conv*",
    \\      "model.layers.22.self_attn*",
    \\      "model.layers.23.conv*",
    \\      "model.layers.3.conv*",
    \\      "model.layers.4.self_attn*",
    \\      "model.layers.5.conv*",
    \\      "model.layers.6.self_attn*",
    \\      "model.layers.7.conv*",
    \\      "model.layers.8.self_attn*",
    \\      "model.layers.9.conv*"
    \\    ],
    \\    "quant_algo": "NVFP4",
    \\    "producer": {
    \\      "name": "modelopt",
    \\      "version": "0.0.1.dev1+g4c1f81382"
    \\    },
    \\    "quant_method": "modelopt"
    \\  }
    \\}
;
