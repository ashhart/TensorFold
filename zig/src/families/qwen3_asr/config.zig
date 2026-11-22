//! Qwen3-ASR admission (0.6B and 1.7B): the audio tower's and the Qwen3 language model's geometry from config.json.
const std = @import("std");

/// The tower's audio as the package reads it: 16 kHz mono, 25 ms frames every 10 ms, 128 mel bins.
pub const sample_rate = 16000;
pub const n_fft = 400;
pub const hop = 160;
/// The three stride-2 convolutions' kernel: 3 x 3, padding 1.
pub const conv_taps = 3;

pub const Audio = struct {
    d: usize, // d_model
    layers: usize,
    heads: usize,
    ffn: usize,
    mels: usize,
    conv: usize, // downsample_hidden_size
    out: usize, // output_dim: the language model's width
    window: usize, // n_window: half a chunk, in frames
    window_infer: usize, // n_window_infer: frames an attention window spans
    positions: usize, // max_source_positions
    conv_chunks: usize, // conv_chunksize

    pub fn headDim(a: Audio) usize {
        return a.d / a.heads;
    }

    /// Mel bins after the three convolutions: ((m + 1) / 2 + 1) / 2 + 1) / 2.
    pub fn convBins(a: Audio) usize {
        return (((a.mels + 1) / 2 + 1) / 2 + 1) / 2;
    }

    /// Frames a chunk holds: 2 n_window.
    pub fn chunkFrames(a: Audio) usize {
        return 2 * a.window;
    }
};

pub const Text = struct {
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    vocab: usize,
    eps: f32,
    theta: f64,
    context: usize,
};

pub const Config = struct {
    audio: Audio,
    text: Text,
    audio_token: u32,
    audio_start: u32,
    audio_end: u32,
};

const Object = std.json.ObjectMap;

fn object(v: ?std.json.Value) !Object {
    const x = v orelse return error.BadQwen3AsrConfig;
    return if (x == .object) x.object else error.BadQwen3AsrConfig;
}

fn field(o: Object, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadQwen3AsrConfig;
}

fn integer(o: Object, key: []const u8) !usize {
    const v = try field(o, key);
    if (v != .integer or v.integer < 1) return error.BadQwen3AsrConfig;
    return std.math.cast(usize, v.integer) orelse error.BadQwen3AsrConfig;
}

fn id(o: Object, key: []const u8) !u32 {
    return std.math.cast(u32, try integer(o, key)) orelse error.BadQwen3AsrConfig;
}

fn float(o: Object, key: []const u8) !f64 {
    return switch (try field(o, key)) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadQwen3AsrConfig,
    };
}

fn expect(o: Object, key: []const u8, want: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, want)) return error.UnsupportedQwen3AsrConfig;
}

fn flag(o: Object, key: []const u8, want: bool) !void {
    const v = o.get(key) orelse return;
    if (v != .bool or v.bool != want) return error.UnsupportedQwen3AsrConfig;
}

fn audio(o: Object) !Audio {
    try expect(o, "model_type", "qwen3_asr_audio_encoder");
    try expect(o, "activation_function", "gelu");
    try flag(o, "scale_embedding", false);
    const a: Audio = .{
        .d = try integer(o, "d_model"),
        .layers = try integer(o, "encoder_layers"),
        .heads = try integer(o, "encoder_attention_heads"),
        .ffn = try integer(o, "encoder_ffn_dim"),
        .mels = try integer(o, "num_mel_bins"),
        .conv = try integer(o, "downsample_hidden_size"),
        .out = try integer(o, "output_dim"),
        .window = try integer(o, "n_window"),
        .window_infer = try integer(o, "n_window_infer"),
        .positions = try integer(o, "max_source_positions"),
        .conv_chunks = try integer(o, "conv_chunksize"),
    };
    if (a.d % a.heads != 0 or a.d % 8 != 0 or a.ffn % 8 != 0 or a.conv % 8 != 0) return error.UnsupportedQwen3AsrGeometry;
    // the package's per-chunk lengths assume a chunk of 100 frames (13 rows) and windows of one or more whole chunks
    if (a.chunkFrames() != 100 or a.window_infer < a.chunkFrames() or a.window_infer % a.chunkFrames() != 0) return error.UnsupportedQwen3AsrGeometry;
    if (a.mels != 128) return error.UnsupportedQwen3AsrGeometry;
    return a;
}

fn text(o: Object) !Text {
    try expect(o, "model_type", "qwen3");
    try expect(o, "hidden_act", "silu");
    try flag(o, "attention_bias", false);
    try flag(o, "tie_word_embeddings", true);
    const t: Text = .{
        .hidden = try integer(o, "hidden_size"),
        .layers = try integer(o, "num_hidden_layers"),
        .heads = try integer(o, "num_attention_heads"),
        .kv_heads = try integer(o, "num_key_value_heads"),
        .head_dim = try integer(o, "head_dim"),
        .intermediate = try integer(o, "intermediate_size"),
        .vocab = try integer(o, "vocab_size"),
        .eps = @floatCast(try float(o, "rms_norm_eps")),
        .theta = try float(o, "rope_theta"),
        .context = try integer(o, "max_position_embeddings"),
    };
    if (t.heads % t.kv_heads != 0 or t.head_dim % 2 != 0) return error.UnsupportedQwen3AsrGeometry;
    try rope(try object(o.get("rope_scaling")), t.head_dim);
    return t;
}

/// The interleaved M-RoPE the config names: positions are equal on all three axes for audio, so it is plain RoPE.
fn rope(o: Object, head_dim: usize) !void {
    try expect(o, "rope_type", "default");
    const v = try field(o, "mrope_section");
    if (v != .array) return error.BadQwen3AsrConfig;
    var sum: usize = 0;
    for (v.array.items) |x| {
        if (x != .integer or x.integer < 0) return error.BadQwen3AsrConfig;
        sum += @intCast(x.integer);
    }
    if (sum != head_dim / 2) return error.UnsupportedQwen3AsrRotary;
}

/// The config of a Qwen3-ASR checkpoint folder's config.json text.
pub fn parse(gpa: std.mem.Allocator, json: []const u8) !Config {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer doc.deinit();
    const root = try object(doc.value);
    try expect(root, "model_type", "qwen3_asr");
    const thinker = try object(root.get("thinker_config"));
    const c: Config = .{
        .audio = try audio(try object(thinker.get("audio_config"))),
        .text = try text(try object(thinker.get("text_config"))),
        .audio_token = try id(thinker, "audio_token_id"),
        .audio_start = try id(thinker, "audio_start_token_id"),
        .audio_end = try id(thinker, "audio_end_token_id"),
    };
    if (c.audio.out != c.text.hidden) return error.UnsupportedQwen3AsrGeometry; // the audio rows splice in at the LM's width
    return c;
}

/// `dir`/config.json, parsed and admitted.
pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    return parse(gpa, bytes);
}
