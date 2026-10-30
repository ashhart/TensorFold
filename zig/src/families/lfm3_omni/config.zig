//! LFM3-Omni-2B's shapes and token ids (the released config), checked against the checkpoint's config.json at load.
const std = @import("std");
const stack = @import("spec.zig");

pub const hidden = 2048;
pub const vocab = 128000;
pub const codebooks = 12;
pub const codebook_size = 2017; // 2016 codes and the end-of-speech code
pub const audio_eos_code = 2016;
pub const maskgit_schedule = [_]usize{ 4, 4, 4 };
pub const interleaved_n_text = 6;
pub const interleaved_n_audio = 12;

pub const bos_token_id = 124894;
pub const eos_token_id = 124900; // <|im_end|>
pub const text_end_token_id = 125014;
pub const audio_start_token_id = 125012;
pub const mixed_start_token_id = 125015;
pub const audio_token_id = 125031;
pub const audio_out_token_id = 125032;
pub const image_token_id = 125030;
pub const video_token_id = 125029;
pub const vision_start_token_id = 125009;
pub const vision_end_token_id = 125010;

// speech encoder
pub const mel_bins = 128;
pub const conv_channels = 256;
pub const enc_d = 768;
// depth decoder
pub const depth_d = 1024;
// codec
pub const codec_d = 512;
pub const quantizer_dim = 256;
pub const frame_stack = 6;
pub const n_fft = 1280;
pub const hop = 320;
pub const spec_bins = n_fft / 2 + 1;
pub const istft_trim = (n_fft - hop) / 2;
pub const sample_rate = 24000;
pub const frame_samples = hop * frame_stack;

const C = stack.Kind.conv;
const F = stack.Kind.full;
const S = stack.Kind.sliding;

pub const encoder_kinds = [_]stack.Kind{ C, C, F, C, C, F, C, C, F, C, C, F, C, C, F, F, F };
pub const backbone_kinds = [_]stack.Kind{ C, C, F, C, S, C, F, C, S, C, F, C, S, C, F, C, S, C, F, C, S, C, F, C };
pub const depth_kinds = [_]stack.Kind{ F, F, F };
pub const codec_kinds = [_]stack.Kind{ C, C, S, C, S, C, S, C };

/// The speech encoder: LFM2 layers (3-tap conv, no output gate), causal attention over the last 32 tokens.
pub fn encoderSpec(max_tokens: usize) stack.Spec {
    return .{ .name = "speech encoder", .prefix = "model.audio_encoder.lfm.", .d = enc_d, .ffn = 1792, .heads = 8, .kv_heads = 4, .head_dim = 96, .conv_k = 3, .kinds = &encoder_kinds, .window = 32, .window_all = true, .sandwich = false, .lfm3 = false, .rope_full = .{ .theta = 1e6 }, .rope_mode = 2, .max_rows = max_tokens, .max_ctx = max_tokens };
}

pub fn backboneSpec(rows: usize, ctx: usize) stack.Spec {
    return .{ .name = "backbone", .prefix = "model.language_model.", .d = hidden, .ffn = 7680, .heads = 16, .kv_heads = 4, .head_dim = 128, .conv_k = 7, .kinds = &backbone_kinds, .window = 256, .sandwich = true, .lfm3 = true, .rope_full = .{ .theta = 1e6, .fraction = 0.5 }, .rope_sliding = .{ .theta = 10000 }, .rope_mode = 1, .max_rows = rows, .max_ctx = ctx };
}

/// The depth decoder: three bidirectional layers over a frame's 12 codebook slots.
pub fn depthSpec() stack.Spec {
    return .{ .name = "depth decoder", .prefix = "model.depth_decoder.model.", .d = depth_d, .ffn = 2816, .heads = 32, .kv_heads = 8, .head_dim = 32, .conv_k = 7, .kinds = &depth_kinds, .causal = false, .sandwich = false, .lfm3 = true, .rope_full = .{ .theta = 1e6 }, .rope_mode = 1, .max_rows = codebooks, .max_ctx = codebooks };
}

/// The Liquid codec's decoder at 75 rows a second: sliding attention over the last 30 rows, RoPE in fp32.
pub fn codecSpec(rows: usize, ctx: usize) stack.Spec {
    return .{ .name = "codec decoder", .prefix = "decoder.lfm.", .d = codec_d, .ffn = 2304, .heads = 8, .kv_heads = 2, .head_dim = 64, .conv_k = 7, .kinds = &codec_kinds, .window = 30, .sandwich = true, .lfm3 = true, .rope_full = .{ .theta = 1e6 }, .rope_sliding = .{ .theta = 1e6 }, .rope_mode = 0, .max_rows = rows, .max_ctx = ctx };
}

fn get(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse {
        std.log.err("lfm3_omni: config.json has no {s}", .{key});
        return error.BadConfig;
    };
}

fn int(o: std.json.ObjectMap, key: []const u8, want: i64) !void {
    const v = try get(o, key);
    if (v != .integer or v.integer != want) {
        std.log.err("lfm3_omni: config {s} is not {d}; only LFM3-Omni-2B's shapes are supported", .{ key, want });
        return error.UnsupportedConfig;
    }
}

fn kinds(o: std.json.ObjectMap, want: []const stack.Kind) !void {
    const v = try get(o, "layer_types");
    if (v != .array or v.array.items.len != want.len) return error.UnsupportedConfig;
    for (v.array.items, want) |x, k| {
        if (x != .string) return error.UnsupportedConfig;
        const name = switch (k) {
            .conv => "conv",
            .full => "full_attention",
            .sliding => "sliding_attention",
        };
        if (!std.mem.eql(u8, x.string, name)) return error.UnsupportedConfig;
    }
}

fn object(v: std.json.Value) !std.json.ObjectMap {
    if (v != .object) return error.BadConfig;
    return v.object;
}

/// Refuses a checkpoint whose config differs from LFM3-Omni-2B's in anything this port hard-codes.
pub fn check(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(text);
    var tree = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer tree.deinit();
    const root = try object(tree.value);
    const mt = try get(root, "model_type");
    if (mt != .string or !std.mem.eql(u8, mt.string, "lfm3_omni")) return error.NotLfm3Omni;
    try int(root, "num_codebooks", codebooks);
    try int(root, "codebook_size", codebook_size);
    try int(root, "audio_eos_code", audio_eos_code);
    try int(root, "audio_token_id", audio_token_id);
    try int(root, "audio_out_token_id", audio_out_token_id);
    try int(root, "mixed_start_token_id", mixed_start_token_id);
    try int(root, "text_end_token_id", text_end_token_id);
    try int(root, "interleaved_n_text", interleaved_n_text);
    try int(root, "interleaved_n_audio", interleaved_n_audio);
    const tc = try object(try get(root, "text_config"));
    try int(tc, "hidden_size", hidden);
    try int(tc, "num_hidden_layers", backbone_kinds.len);
    try int(tc, "intermediate_size", 7680);
    try int(tc, "num_attention_heads", 16);
    try int(tc, "num_key_value_heads", 4);
    try int(tc, "sliding_window", 256);
    try int(tc, "vocab_size", vocab);
    try kinds(tc, &backbone_kinds);
    const ac = try object(try get(root, "audio_encoder_config"));
    try int(ac, "hidden_size", enc_d);
    try int(ac, "num_mel_bins", mel_bins);
    try int(ac, "sliding_window", 32);
    try int(ac, "conv_channels", conv_channels);
    try kinds(try object(try get(ac, "text_config")), &encoder_kinds);
    const dc = try object(try get(root, "depth_config"));
    try int(dc, "hidden_size", depth_d);
    try int(dc, "num_hidden_layers", depth_kinds.len);
    try int(dc, "head_dim", 32);
    const ms = try get(root, "maskgit_schedule");
    if (ms != .array or ms.array.items.len != maskgit_schedule.len) return error.UnsupportedConfig;
    for (ms.array.items, maskgit_schedule) |x, w| if (x != .integer or x.integer != w) return error.UnsupportedConfig;
}
