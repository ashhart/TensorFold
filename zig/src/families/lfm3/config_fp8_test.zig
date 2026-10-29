//! The block FP8 export's admission, against its published config.json.
const std = @import("std");
const c = @import("config.zig");

test "the block FP8 export parses as its own scheme: FP8 FFN and q/k/v/o in 128 x 128 blocks" {
    var config = try c.parse(std.testing.allocator, fp8);
    defer config.deinit();
    try std.testing.expectEqual(c.Scheme.fp8_block, config.scheme);
    try std.testing.expectEqual(c.fp8_width, try config.width("model.layers.2.self_attn.q_proj"));
    try std.testing.expectEqual(@as(usize, 262144), config.context);
    try std.testing.expectError(error.Lfm3Fp8BlockNotOnMetal, c.servedOn(&config, .metal));
    try c.servedOn(&config, .cuda);
}

test "an FP8 export with another block, static inputs, an extra ignore or a kv-cache scheme is refused" {
    const edits = [_][2][]const u8{
        .{ "128,\n            128\n", "64,\n            128\n" },
        .{ "\"dynamic\": true", "\"dynamic\": false" },
        .{ "\"strategy\": \"block\"", "\"strategy\": \"channel\"" },
        .{ "\"group_size\": 128", "\"group_size\": 64" },
        .{ "\"lm_head\",", "\"lm_head\",\n      \"re:.*mlp.*\"," },
        .{ "\"kv_cache_scheme\": null", "\"kv_cache_scheme\": {}" },
        .{ "\"format\": \"float-quantized\",\n    \"global", "\"format\": \"pack-quantized\",\n    \"global" },
        .{ "\"Linear\"", "\"Conv1d\"" },
    };
    for (edits) |e| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fp8, e[0], e[1]);
        defer std.testing.allocator.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fp8));
        try std.testing.expectError(error.UnsupportedLfm3Quantization, c.parse(std.testing.allocator, bytes));
    }
}

/// LiquidAI/LFM3-1.7B-FP8 (243955da)'s config.json as published.
pub const fp8 =
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
    \\  "quantization_config": {
    \\    "config_groups": {
    \\      "FP8_BLOCK": {
    \\        "format": "float-quantized",
    \\        "input_activations": {
    \\          "actorder": null,
    \\          "block_structure": null,
    \\          "dynamic": true,
    \\          "group_size": 128,
    \\          "num_bits": 8,
    \\          "observer": null,
    \\          "observer_kwargs": {},
    \\          "scale_dtype": null,
    \\          "strategy": "group",
    \\          "symmetric": true,
    \\          "type": "float",
    \\          "zp_dtype": null
    \\        },
    \\        "output_activations": null,
    \\        "targets": [
    \\          "Linear"
    \\        ],
    \\        "weights": {
    \\          "actorder": null,
    \\          "block_structure": [
    \\            128,
    \\            128
    \\          ],
    \\          "dynamic": false,
    \\          "group_size": null,
    \\          "num_bits": 8,
    \\          "observer": null,
    \\          "observer_kwargs": {},
    \\          "scale_dtype": null,
    \\          "strategy": "block",
    \\          "symmetric": true,
    \\          "type": "float",
    \\          "zp_dtype": null
    \\        }
    \\      }
    \\    },
    \\    "format": "float-quantized",
    \\    "global_compression_ratio": null,
    \\    "ignore": [
    \\      "lm_head",
    \\      "model.embed_tokens",
    \\      "re:.*\\.gate$",
    \\      "re:.*\\.(?:conv|short_conv)\\..*",
    \\      "re:.*\\.self_attn\\.gate_proj$"
    \\    ],
    \\    "kv_cache_scheme": null,
    \\    "quant_method": "compressed-tensors",
    \\    "quantization_status": "compressed",
    \\    "transform_config": {},
    \\    "version": "0.19.0"
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
    \\  "transformers_version": "5.12.1",
    \\  "use_cache": true,
    \\  "use_sandwich_norm": true,
    \\  "vocab_size": 128000
    \\}
;
