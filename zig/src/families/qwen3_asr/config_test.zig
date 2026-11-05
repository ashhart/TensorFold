//! Config admission: the 0.6B's fields as released, and the refusals for what the tower and LM do not run.
const std = @import("std");
const c = @import("config.zig");

const base =
    \\{"model_type": "qwen3_asr", "thinker_config": {"audio_token_id": 151676, "audio_start_token_id": 151669,
    \\ "audio_end_token_id": 151670, "audio_config": {"model_type": "qwen3_asr_audio_encoder",
    \\ "activation_function": "gelu", "scale_embedding": false, "d_model": 896, "encoder_layers": 18,
    \\ "encoder_attention_heads": 14, "encoder_ffn_dim": 3584, "num_mel_bins": 128, "downsample_hidden_size": 480,
    \\ "output_dim": 1024, "n_window": 50, "n_window_infer": 800, "max_source_positions": 1500, "conv_chunksize": 500},
    \\ "text_config": {"model_type": "qwen3", "hidden_act": "silu", "attention_bias": false, "tie_word_embeddings": true,
    \\ "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
    \\ "intermediate_size": 3072, "vocab_size": 151936, "rms_norm_eps": 1e-06, "rope_theta": 1000000,
    \\ "max_position_embeddings": 65536, "rope_scaling": {"rope_type": "default", "mrope_section": [24, 20, 20]}}}}
;

fn swapped(a: std.mem.Allocator, from: []const u8, to: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, a, base, from, to);
}

test "the 0.6B: tower and LM geometry, chunk frames, mel bins after the convolutions" {
    const cfg = try c.parse(std.testing.allocator, base);
    try std.testing.expectEqual(@as(usize, 64), cfg.audio.headDim());
    try std.testing.expectEqual(@as(usize, 16), cfg.audio.convBins());
    try std.testing.expectEqual(@as(usize, 100), cfg.audio.chunkFrames());
    try std.testing.expectEqual(@as(usize, 8), cfg.text.kv_heads);
    try std.testing.expectEqual(@as(f64, 1e6), cfg.text.theta);
    try std.testing.expectEqual(@as(u32, 151676), cfg.audio_token);
}

test "refusals: another activation, M-RoPE sections not spanning the head, biased attention, other mel bins" {
    const a = std.testing.allocator;
    const cases = [_][3][]const u8{
        .{ "\"activation_function\": \"gelu\"", "\"activation_function\": \"relu\"", "UnsupportedQwen3AsrConfig" },
        .{ "[24, 20, 20]", "[24, 20, 16]", "UnsupportedQwen3AsrRotary" },
        .{ "\"attention_bias\": false", "\"attention_bias\": true", "UnsupportedQwen3AsrConfig" },
        .{ "\"num_mel_bins\": 128", "\"num_mel_bins\": 80", "UnsupportedQwen3AsrGeometry" },
        .{ "\"n_window\": 50", "\"n_window\": 40", "UnsupportedQwen3AsrGeometry" },
        .{ "\"model_type\": \"qwen3_asr\"", "\"model_type\": \"qwen3\"", "UnsupportedQwen3AsrConfig" },
        .{ "\"output_dim\": 1024", "\"output_dim\": 2048", "UnsupportedQwen3AsrGeometry" },
    };
    for (cases) |x| {
        const text = try swapped(a, x[0], x[1]);
        defer a.free(text);
        if (c.parse(a, text)) |_| return error.TestExpectedError else |err| try std.testing.expectEqualStrings(x[2], @errorName(err));
    }
}

/// The 1.7B's geometry as #2105 states it, until its release config.json is downloaded and replaces this.
const big_swaps = [_][2][]const u8{
    .{ "\"d_model\": 896", "\"d_model\": 1024" },                              .{ "\"encoder_layers\": 18", "\"encoder_layers\": 24" },
    .{ "\"encoder_attention_heads\": 14", "\"encoder_attention_heads\": 16" }, .{ "\"encoder_ffn_dim\": 3584", "\"encoder_ffn_dim\": 4096" },
    .{ "\"output_dim\": 1024", "\"output_dim\": 2048" },                       .{ "\"hidden_size\": 1024", "\"hidden_size\": 2048" },
    .{ "\"intermediate_size\": 3072", "\"intermediate_size\": 6144" },
};

test "the 1.7B: admitted, with the tower and LM shapes the CUDA binding takes" {
    const a = std.testing.allocator;
    var text = try a.dupe(u8, base);
    defer a.free(text);
    for (big_swaps) |s| {
        const next = try std.mem.replaceOwned(u8, a, text, s[0], s[1]);
        a.free(text);
        text = next;
    }
    const cfg = try c.parse(a, text);
    try std.testing.expectEqual(@as(usize, 64), cfg.audio.headDim()); // vision_kernel<64, 8>, as the 0.6B
    try std.testing.expectEqual(@as(usize, 2048), cfg.audio.out);
    try std.testing.expectEqual(cfg.audio.out, cfg.text.hidden); // the audio rows splice in at the LM's width
    try std.testing.expectEqual(@as(usize, 128), cfg.text.head_dim); // Nemotron's chunk attention at 128
    try std.testing.expect(cfg.text.heads / cfg.text.kv_heads <= 16);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const names = @import("names.zig");
    try std.testing.expectEqual(@as(usize, 24 * 16 + 28 * 11 + 16), (try names.all(arena.allocator(), cfg)).len);
}
