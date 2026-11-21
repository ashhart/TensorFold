//! Nemotron 3.5 Transcribe's audio side from config.json, and the geometry a clip's samples take through it.
const std = @import("std");
const Value = std.json.Value;
const Object = std.json.ObjectMap;

/// One transformer stack: the ASR branch or the speaker branch's encoder.
pub const Stack = struct {
    width: usize,
    heads: usize,
    layers: usize,
    ffn: usize,
    rot: usize, // leading dims of each head that turn with RoPE (rotary_fraction of the head, all of it when unset)
    qk_norm: bool,
    base: f32,

    pub fn dim(s: Stack) usize {
        return s.width / s.heads;
    }
};

pub const Config = struct {
    asr: Stack,
    spk: Stack,
    speakers: usize, // the speaker branch's slots (num_spks)
    head_width: usize, // the speaker head's hidden width (tf_d_model)
    output: usize, // the language model's width (perception.output_dim)
    threshold: f32, // a slot is active past it (speaker_activity_threshold)
    kernel_scale: f32, // spk_kernel_scale
    boost_latest: f32, // the speaker cache's scores: frames new to it (scores_boost_latest)
    score_clamp: f32, // a slot's probability counts at least this much in its own frame scores (pred_score_threshold)
    min_pos_rate: f32, // a slot with floor(this x its share of the cache) positive scores keeps only those (min_pos_scores_rate)
    strong_rate: f32,
    weak_rate: f32,
    sil_per_speaker: usize, // silence rows a slot keeps in the speaker cache (spkcache_sil_frames_per_spk)

    pub const rate = 16000;
    pub const hop = 160; // window_stride 0.01 s
    pub const window = 400; // window_size 0.025 s
    pub const fft = 512;
    pub const bins = fft / 2 + 1;
    pub const features = 128;
    pub const stack = 8; // frames to a row (feature stacking)
    /// What the dumps show beyond config.json: pre-emphasis, the log's guard, the per-feature normalisation's epsilon.
    pub const preemph: f32 = 0.97;
    pub const log_guard: f32 = 1.0 / 16777216.0;
    pub const norm_eps: f32 = 1e-5;
    pub const eps: f32 = 1e-5; // every LayerNorm
    /// The windows the encoder takes a long clip in (dumps): 4000-frame cores with 400 frames of context each side.
    pub const core = 4000;
    pub const context = 400;
    /// The speaker branch's streaming state between windows (dumps): its cache and queue rows.
    pub const cache_rows = 188;
    pub const fifo_rows = 40;
    /// A slot's frames under this probability never score in the speaker cache (arXiv 2507.18446, step 3).
    pub const speech = 0.5;
    /// The audio a request may hold: the model card's one hour.
    pub const max_seconds = 3600;

    /// The audio side, or error.NoAudioEncoder for a config without one; error.UnsupportedAudio names a setting off it.
    pub fn parse(a: std.mem.Allocator, text: []const u8) !Config {
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const doc = try std.json.parseFromSliceLeaky(Value, arena.allocator(), text, .{});
        const root = object(doc) orelse return error.BadConfig;
        const per = object(root.get("perception")) orelse return error.NoAudioEncoder;
        const pe = object(root.get("pe_encoder_config")) orelse return error.NoAudioEncoder;
        try preprocessor(object(per.get("preprocessor")) orelse return error.BadConfig);
        const diar = object(pe.get("diarization_model_cfg")) orelse return error.BadConfig;
        const mods = object(diar.get("sortformer_modules")) orelse return error.BadConfig;
        try preprocessor(object(diar.get("preprocessor")) orelse return error.BadConfig);
        if (!eql(pe.get("asr_normalize_type"), "per_feature") or !eql(pe.get("diar_normalize_type"), "NA")) return error.UnsupportedAudio;
        if (!eql(pe.get("speaker_feature_mode"), "thresholded")) return error.UnsupportedAudio;
        const c: Config = .{
            .asr = try readStack(object(pe.get("asr_encoder_cfg")) orelse return error.BadConfig),
            .spk = try readStack(object(diar.get("encoder")) orelse return error.BadConfig),
            .speakers = try size(mods.get("num_spks")),
            .head_width = try size(mods.get("tf_d_model")),
            .output = try size(per.get("output_dim")),
            .threshold = @floatCast(try float(pe.get("speaker_activity_threshold"))),
            .kernel_scale = @floatCast(try float(pe.get("spk_kernel_scale"))),
            .boost_latest = @floatCast(try float(mods.get("scores_boost_latest"))),
            .score_clamp = @floatCast(try float(mods.get("pred_score_threshold"))),
            .min_pos_rate = @floatCast(try float(mods.get("min_pos_scores_rate"))),
            .strong_rate = @floatCast(try float(mods.get("strong_boost_rate"))),
            .weak_rate = @floatCast(try float(mods.get("weak_boost_rate"))),
            .sil_per_speaker = try size(mods.get("spkcache_sil_frames_per_spk")),
        };
        if (!(c.min_pos_rate >= 0)) return error.BadConfig;
        if (c.speakers != 8 or c.head_width == 0 or c.output == 0 or c.speakers * (c.sil_per_speaker + 1) > cache_rows) return error.UnsupportedAudio;
        return c;
    }

    /// Mel frames the preprocessor gives n samples, the zero frame past the end included.
    pub fn frames(n: usize) usize {
        return n / hop + 1;
    }

    /// The rows a clip of n samples gives the language model: its valid frames, eight to a row.
    pub fn rows(n: usize) usize {
        return std.math.divCeil(usize, n / hop, stack) catch unreachable;
    }

    /// The most rows a request's audio takes (one hour).
    pub fn limit() u32 {
        return @intCast(rows(max_seconds * rate));
    }
};

/// Window i of a clip of `f` frames (`valid` of them valid): the frames it reads and the rows of its output kept.
pub const Window = struct { start: usize, end: usize, rows: usize, lo: usize, hi: usize };

/// One window a 4000-frame core of the `valid` frames (4200 valid frames take two, as the frames_4200 dump shows).
pub fn windowCount(valid: usize) usize {
    return @max(1, std.math.divCeil(usize, valid, Config.core) catch unreachable);
}

pub fn window(f: usize, valid: usize, i: usize) Window {
    const core0 = i * Config.core;
    const start = core0 -| Config.context;
    const end = @min(f, core0 + Config.core + Config.context);
    const total = std.math.divCeil(usize, valid, Config.stack) catch unreachable;
    const lo = (core0 - start) / Config.stack;
    const kept = @min(std.math.divCeil(usize, @min(f, core0 + Config.core) - core0, Config.stack) catch unreachable, total - core0 / Config.stack);
    // rows of nothing but frames past the valid end are left out, which a key mask would do
    const rows = std.math.divCeil(usize, @min(end, valid) - start, Config.stack) catch unreachable;
    return .{ .start = start, .end = end, .rows = rows, .lo = lo, .hi = lo + kept };
}

fn preprocessor(p: Object) !void {
    if (!eql(p.get("window"), "hann") or (try size(p.get("features"))) != Config.features or (try size(p.get("n_fft"))) != Config.fft or (try size(p.get("sample_rate"))) != Config.rate) return error.UnsupportedAudio;
    if (@abs(try float(p.get("window_size")) - 0.025) > 1e-9 or @abs(try float(p.get("window_stride")) - 0.01) > 1e-9) return error.UnsupportedAudio;
    if (p.get("frame_splicing")) |v| if ((try size(v)) != 1) return error.UnsupportedAudio;
}

fn readStack(o: Object) !Stack {
    const width = try size(o.get("d_model"));
    const heads = try size(o.get("n_heads"));
    if (width == 0 or heads == 0 or width % heads != 0) return error.BadConfig;
    const dim = width / heads;
    const fraction = if (o.get("rotary_fraction")) |v| try float(v) else 1.0;
    const rot: usize = @intFromFloat(@as(f64, @floatFromInt(dim)) * fraction);
    if (!eql(o.get("self_attention_model"), "rope") or !eql(o.get("attn_mode"), "full") or !truthy(o.get("pre_block_norm"))) return error.UnsupportedAudio;
    if ((try size(o.get("subsampling_factor"))) != Config.stack or truthy(o.get("qkv_bias")) or truthy(o.get("xscaling"))) return error.UnsupportedAudio;
    if (rot != 64 or rot > dim or (dim != 64 and dim != 128)) return error.UnsupportedAudio; // nemo_audio.cu turns 64 dims
    const base = if (o.get("rope_base")) |v| try float(v) else 10000.0;
    return .{ .width = width, .heads = heads, .layers = try size(o.get("n_layers")), .ffn = @intFromFloat(@as(f64, @floatFromInt(width)) * try float(o.get("ff_expansion"))), .rot = rot, .qk_norm = truthy(o.get("qk_norm")), .base = @floatCast(base) };
}

fn object(v: ?Value) ?Object {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

fn eql(v: ?Value, s: []const u8) bool {
    const x = v orelse return false;
    return x == .string and std.mem.eql(u8, x.string, s);
}

fn truthy(v: ?Value) bool {
    const x = v orelse return false;
    return x == .bool and x.bool;
}

fn size(v: ?Value) !usize {
    const x = v orelse return error.BadConfig;
    if (x != .integer or x.integer < 0) return error.BadConfig;
    return @intCast(x.integer);
}

fn float(v: ?Value) !f64 {
    const x = v orelse return error.BadConfig;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadConfig,
    };
}

test "nemo audio rows: the dumps' clips" {
    const clips = [_][2]usize{ .{ 30560, 24 }, .{ 56080, 44 }, .{ 93680, 74 }, .{ 470400, 368 }, .{ 312160, 244 }, .{ 547440, 428 }, .{ 2087520, 1631 } };
    for (clips) |c| try std.testing.expectEqual(c[1], Config.rows(c[0]));
    try std.testing.expectEqual(@as(u32, 45000), Config.limit());
}

test "nemo audio windows: the long clip's four, and one for a clip that fits" {
    const f = Config.frames(2087520);
    try std.testing.expectEqual(@as(usize, 13048), f);
    try std.testing.expectEqual(@as(usize, 4), windowCount(f - 1));
    const want = [_]Window{
        .{ .start = 0, .end = 4400, .rows = 550, .lo = 0, .hi = 500 },
        .{ .start = 3600, .end = 8400, .rows = 600, .lo = 50, .hi = 550 },
        .{ .start = 7600, .end = 12400, .rows = 600, .lo = 50, .hi = 550 },
        .{ .start = 11600, .end = 13048, .rows = 181, .lo = 50, .hi = 181 },
    };
    for (want, 0..) |w, i| try std.testing.expectEqual(w, window(f, f - 1, i));
    try std.testing.expectEqual(@as(usize, 1), windowCount(547440 / Config.hop));
    try std.testing.expectEqual(Window{ .start = 0, .end = 3422, .rows = 428, .lo = 0, .hi = 428 }, window(3422, 3421, 0));
}

test "nemo audio windows: a row of nothing but the zero frame is not encoded or kept" {
    // 8 * 160 * 30 samples: 240 valid frames, the 241st (zero) frame alone in row 30
    const n = 8 * 160 * 30;
    try std.testing.expectEqual(@as(usize, 30), Config.rows(n));
    const w = window(Config.frames(n), n / Config.hop, 0);
    try std.testing.expectEqual(@as(usize, 30), w.rows);
    try std.testing.expectEqual(@as(usize, 30), w.hi);
}

test "nemo audio windows: 4200 valid frames take two windows, 500 rows and 25 kept" {
    try std.testing.expectEqual(@as(usize, 2), windowCount(4200));
    const w0 = window(4201, 4200, 0);
    const w1 = window(4201, 4200, 1);
    try std.testing.expectEqual(@as(usize, 525), w0.rows);
    try std.testing.expectEqual(@as(usize, 500), w0.hi - w0.lo);
    try std.testing.expectEqual(@as(usize, 75), w1.rows);
    try std.testing.expectEqual(@as(usize, 25), w1.hi - w1.lo);
}

test "nemo audio windows: the kept rows cover every row once" {
    for ([_]usize{ 2, 4001, 4401, 8000, 8401, 8801, 12345, 40001 }) |f| {
        const valid = f - 1;
        var next: usize = 0;
        for (0..windowCount(valid)) |i| {
            const w = window(f, valid, i);
            try std.testing.expectEqual(next, w.start / Config.stack + w.lo);
            try std.testing.expect(w.hi <= w.rows);
            next += w.hi - w.lo;
        }
        try std.testing.expectEqual(std.math.divCeil(usize, valid, 8) catch unreachable, next);
    }
}
