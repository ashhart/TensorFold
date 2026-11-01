//! Transcribe's audio config: the perception config's ASR encoder, NeMo's defaults for the Sortformer and windows.
const std = @import("std");

pub const Encoder = struct {
    d: usize,
    heads: usize,
    layers: usize,
    rot_frac: f64, // RoPE on this fraction of each head's channels
    base: f64 = 10000.0,
    qk_norm: bool,
    eps: f32 = 1e-5,

    pub fn dim(e: Encoder) usize {
        return e.d / e.heads;
    }

    pub fn rot(e: Encoder) usize {
        return @intFromFloat(@as(f64, @floatFromInt(e.dim())) * e.rot_frac);
    }
};

// AudioToMelSpectrogramPreprocessor as configured
pub const sample_rate = 16000;
pub const n_fft = 512;
pub const hop = 160;
pub const win = 400;
pub const mels = 128;
pub const bins = n_fft / 2 + 1;
pub const preemph: f32 = 0.97;
pub const log_guard: f32 = 0x1p-24;
pub const norm_eps: f32 = 1e-5;
/// FeatureStacking: mel frames an encoder frame stacks (both encoders)
pub const sub = 8;

// ParallelExpertEncoder's online inference: core windows of 500 encoder frames, 50 frames of context a side
pub const window_frames = 500 * sub;
pub const context_frames = 50 * sub;

// Sortformer: its head's width, its speakers, the streaming caches and the speaker-cache compression constants
pub const tf_d = 192;
pub const speakers = 8;
pub const fifo_len = 40;
pub const spkcache_update_period = 300;
pub const spkcache_len = 188;
pub const sil_frames_per_spk = 1;
pub const pred_score_threshold: f32 = 0.25;
pub const scores_boost_latest: f32 = 0.05;
pub const strong_boost_rate = 0.75;
pub const weak_boost_rate = 1.5;
pub const min_pos_scores_rate = 0.5;
/// speaker_feature_mode thresholded: a speaker is active above this probability
pub const speaker_threshold: f32 = 0.5;

asr: Encoder,
sf: Encoder = .{ .d = 512, .heads = 8, .layers = 31, .rot_frac = 1.0, .qk_norm = false },
text_hidden: usize,

/// config.json's perception block: its encoder must be the TransformerEncoder these kernels run.
pub fn parse(root: std.json.ObjectMap) !@This() {
    const p = (root.get("perception") orelse return error.NoPerception).object;
    const e = (p.get("encoder") orelse return error.NoPerception).object;
    const pre = (p.get("preprocessor") orelse return error.NoPerception).object;
    if (int(pre, "features") != mels or int(pre, "n_fft") != n_fft or int(pre, "sample_rate") != sample_rate) return error.UnsupportedAudioConfig;
    if (int(e, "subsampling_factor") != sub or int(e, "feat_in") != mels) return error.UnsupportedAudioConfig;
    if (!flag(e, "qk_norm") or flag(e, "qkv_bias") or !std.mem.eql(u8, str(e, "self_attention_model"), "rope")) return error.UnsupportedAudioConfig;
    const c: @This() = .{
        .asr = .{ .d = int(e, "d_model"), .heads = int(e, "n_heads"), .layers = int(e, "n_layers"), .rot_frac = num(e, "rotary_fraction"), .base = num(e, "rope_base"), .qk_norm = true },
        .text_hidden = int(p, "output_dim"),
    };
    if (c.asr.heads == 0 or c.asr.d % c.asr.heads != 0 or int(e, "ff_expansion") != 4) return error.UnsupportedAudioConfig;
    return c;
}

fn int(o: std.json.ObjectMap, k: []const u8) usize {
    const v = o.get(k) orelse return 0;
    return if (v == .integer and v.integer > 0) @intCast(v.integer) else 0;
}

fn num(o: std.json.ObjectMap, k: []const u8) f64 {
    const v = o.get(k) orelse return 0;
    return switch (v) {
        .float => v.float,
        .integer => @floatFromInt(v.integer),
        else => 0,
    };
}

fn flag(o: std.json.ObjectMap, k: []const u8) bool {
    const v = o.get(k) orelse return false;
    return v == .bool and v.bool;
}

fn str(o: std.json.ObjectMap, k: []const u8) []const u8 {
    const v = o.get(k) orelse return "";
    return if (v == .string) v.string else "";
}

test "the published perception block parses" {
    const text =
        \\{"perception":{"output_dim":2688,"preprocessor":{"sample_rate":16000,"features":128,"n_fft":512},
        \\"encoder":{"feat_in":128,"d_model":1280,"n_heads":10,"n_layers":32,"qkv_bias":false,"qk_norm":true,"ff_expansion":4,
        \\"subsampling_factor":8,"self_attention_model":"rope","rope_base":10000.0,"rotary_fraction":0.5}}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
    defer parsed.deinit();
    const c = try parse(parsed.value.object);
    try std.testing.expectEqual(@as(usize, 128), c.asr.dim());
    try std.testing.expectEqual(@as(usize, 64), c.asr.rot());
    try std.testing.expectEqual(@as(usize, 64), c.sf.rot());
    try std.testing.expectEqual(@as(usize, 2688), c.text_hidden);
}
