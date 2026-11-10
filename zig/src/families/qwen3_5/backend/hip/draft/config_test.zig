//! The HIP drafter's config against the 27B target: the stock DFlash2 file parses, each malformed one is refused.

const std = @import("std");
const config = @import("config.zig");

const Config = config.Config;
const gpa = std.testing.allocator;

/// Qwen3.8-27B's geometry: what the stock drafter is validated against.
const q27: config.Target = .{ .hidden = 5120, .layers = 64, .vocab = 248320 };

/// The stock Qwen3.8-27B-DFlash2 config.json, its fields the parser reads.
const stock =
    \\{"dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248070,
    \\  "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 19, 33, 47, 61]},
    \\ "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408, "num_attention_heads": 32,
    \\ "num_hidden_layers": 5, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
    \\ "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "sliding_window": 2048,
    \\ "vocab_size": 248320}
;

/// `stock` with the first `from` replaced by `to`.
fn edited(from: []const u8, to: []const u8) ![]u8 {
    const at = std.mem.indexOf(u8, stock, from) orelse return error.TestFixture;
    return std.mem.concat(gpa, u8, &.{ stock[0..at], to, stock[at + from.len ..] });
}

fn refused(want: anyerror, from: []const u8, to: []const u8) !void {
    const bytes = try edited(from, to);
    defer gpa.free(bytes);
    try std.testing.expectError(want, Config.parse(gpa, bytes, q27));
}

test "the stock DFlash2 config parses against the 27B target with the HIP tap window" {
    const c = try Config.parse(gpa, stock, q27);
    try std.testing.expectEqual(@as(usize, 5120), c.hidden);
    try std.testing.expectEqual(@as(usize, 17408), c.intermediate);
    try std.testing.expectEqual(@as(usize, 5), c.layers);
    try std.testing.expectEqual(@as(usize, 248320), c.vocab);
    try std.testing.expectEqual(@as(usize, 32), c.heads);
    try std.testing.expectEqual(@as(usize, 8), c.kv_heads);
    try std.testing.expectEqual(@as(usize, 8), c.block);
    try std.testing.expectEqual(@as(u32, 248070), c.mask);
    try std.testing.expectEqual(@as(usize, 2048), c.window);
    try std.testing.expectEqual(@as(usize, 16), c.group);
    try std.testing.expectEqual(@as(usize, 256), c.rank);
    try std.testing.expectEqual(@as(usize, 16), c.topk);
    try std.testing.expectEqual(@as(f32, 1e-6), c.eps);
    try std.testing.expectEqual(@as(f32, 1e7), c.theta);
    try std.testing.expectEqualSlices(u32, &.{ 5, 19, 33, 47, 61 }, c.tapLayers());
    try std.testing.expectEqual(@as(usize, 4096), c.qWidth());
    try std.testing.expectEqual(@as(usize, 1024), c.kvWidth());
    try std.testing.expectEqual(@as(usize, 1280), c.dynamic());
    try std.testing.expectEqual(config.tap_window, c.window);
}

test "a drafter for another target's width or vocabulary is refused" {
    try refused(error.UnsupportedDraftConfig, "\"hidden_size\": 5120", "\"hidden_size\": 4096");
    try refused(error.UnsupportedDraftConfig, "\"vocab_size\": 248320", "\"vocab_size\": 151936");
}

test "a tap past the target's layers is refused, and so is no tap or a negative one" {
    try refused(error.BadDraftConfig, "47, 61]", "47, 64]");
    try refused(error.BadDraftConfig, "[5, 19, 33, 47, 61]", "[]");
    try refused(error.BadDraftConfig, "[5, 19", "[-1, 19");
    try refused(error.BadDraftConfig, "[5, 19, 33, 47, 61]", "[1, 2, 3, 4, 5, 6, 7, 8, 9]");
    const last = try edited("47, 61]", "47, 63]");
    defer gpa.free(last);
    _ = try Config.parse(gpa, last, q27);
}

test "a tap layer named twice is refused, as the ring would hold one residual in two taps" {
    try refused(error.BadDraftConfig, "[5, 19, 33, 47, 61]", "[5, 19, 33, 19, 61]");
    try refused(error.BadDraftConfig, "[5, 19, 33, 47, 61]", "[5, 5]");
    const one = try edited("[5, 19, 33, 47, 61]", "[61]");
    defer gpa.free(one);
    try std.testing.expectEqualSlices(u32, &.{61}, (try Config.parse(gpa, one, q27)).tapLayers());
}

test "a window past the HIP tap ring is refused, the ring's own width accepted" {
    try refused(error.UnsupportedDraftConfig, "\"sliding_window\": 2048", "\"sliding_window\": 2049");
    try refused(error.UnsupportedDraftConfig, "\"sliding_window\": 2048", "\"sliding_window\": 15");
}

test "shapes the kernels do not run are refused" {
    try refused(error.UnsupportedDraftConfig, "\"head_dim\": 128", "\"head_dim\": 64");
    try refused(error.UnsupportedDraftConfig, "\"conv_kernel_size\": 2", "\"conv_kernel_size\": 3");
    try refused(error.UnsupportedDraftConfig, "\"block_size\": 8", "\"block_size\": 16");
    try refused(error.UnsupportedDraftConfig, "\"selector_top_k\": 16", "\"selector_top_k\": 21");
    try refused(error.UnsupportedDraftConfig, "\"mask_token_id\": 248070", "\"mask_token_id\": 248320");
    try refused(error.UnsupportedDraftConfig, "\"conv_group_size\": 16", "\"conv_group_size\": 3");
    try refused(error.UnsupportedDraftConfig, "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 5");
    try refused(error.UnsupportedDraftConfig, "\"num_hidden_layers\": 5", "\"num_hidden_layers\": 9");
    try refused(error.UnsupportedDraftConfig, "\"intermediate_size\": 17408", "\"intermediate_size\": 17400");
}

test "malformed JSON, a missing field and a zero size are refused as a bad config" {
    try refused(error.BadDraftConfig, "\"vocab_size\": 248320}", "\"vocab_size\": 248320");
    try refused(error.BadDraftConfig, "\"rms_norm_eps\": 1e-06,", "");
    try refused(error.BadDraftConfig, "\"num_attention_heads\": 32", "\"num_attention_heads\": 0");
    try refused(error.BadDraftConfig, "\"hidden_size\": 5120", "\"hidden_size\": \"5120\"");
    try std.testing.expectError(error.BadDraftConfig, Config.parse(gpa, "[1, 2]", q27));
    try std.testing.expectError(error.BadDraftConfig, Config.parse(gpa, "{}", q27));
}

test "a norm epsilon or RoPE base that is zero, negative or past f32 is refused" {
    try refused(error.BadDraftConfig, "\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 0");
    try refused(error.BadDraftConfig, "\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": -1e-06");
    try refused(error.BadDraftConfig, "\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 1e-50");
    try refused(error.BadDraftConfig, "\"rope_theta\": 10000000", "\"rope_theta\": 0");
    try refused(error.BadDraftConfig, "\"rope_theta\": 10000000", "\"rope_theta\": -10000000");
    try refused(error.BadDraftConfig, "\"rope_theta\": 10000000", "\"rope_theta\": 1e300");
}

test "the target geometry comes from the HIP model spec" {
    var s: config.Spec = undefined;
    s.hidden = 5120;
    s.n_layers = 64;
    s.vocab = 248320;
    try std.testing.expectEqual(q27, config.Target.of(s));
}
