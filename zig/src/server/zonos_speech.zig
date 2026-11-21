//! ZONOS3 Mini behind /v1/audio/speech: the release's default voices and a request's prompt rows.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const pyrepr = @import("pyrepr.zig");
const text_mod = @import("model_text.zig");
const req = @import("zonos_request");
const textnorm = @import("zonos_textnorm.zig");
const Allocator = std.mem.Allocator;

pub const error_prefix = "ZONOS3 request:";
pub const temperature = req.temperature;
pub const top_p = req.top_p;
pub const top_k = req.top_k;
pub const max_top_k = req.config.codebook_size;

/// The request's own fields: a voice as codes or as a clip, the normalizer's, the condition rows and guidance.
pub const fields = struct {
    pub const codes = "speaker_reference_codes";
    pub const clip = "ref_audio";
    pub const language = "language";
    pub const normalize = "text_normalization";
    pub const guidance = "speaker_cfg_scale";
    pub const buckets = "condition_buckets";
    pub const values = "condition_values";
    pub const all = [_][]const u8{ codes, clip, language, normalize, guidance, buckets, values };
};

pub const State = struct {
    config: req.config.Config,
    voices: []const req.voices.Voice,
    silence: [req.config.codebooks]i32,
    /// The voice "default" names: --default-voice, else savannah, else the first voice.
    default: ?*const req.voices.Voice,
};

pub const Made = struct { prompt: []const u32, frames: []const [11]i32, reference_audio: []const f32 = &.{}, conditions: []const u32 = &.{}, guidance: f32 = 1 };

const rate = 44100;
/// A reference clip: at least half a second, and the first 60 s of it in whole frames of 2940 samples.
const min_clip = rate / 2;
const max_clip = req.reference_frames * 2940;

/// The state for a ZONOS3 Mini release at `dir` (from the arena `a`), or null for any other checkpoint.
pub fn load(a: Allocator, io: std.Io, dir: []const u8, default_voice: []const u8) !?*State {
    if (dir.len == 0 or !req.release.present(io, dir)) return null;
    var rel = try req.release.open(a, io, dir, null);
    defer rel.deinit();
    const st = try a.create(State);
    st.* = .{ .config = try req.config.Config.load(a, io, rel.lm), .voices = try req.voices.load(a, io, dir, req.reference_frames), .silence = rel.silence, .default = null };
    const named = !std.mem.eql(u8, default_voice, "default");
    st.default = req.voices.find(st.voices, if (named) default_voice else "savannah");
    if (st.default == null) {
        if (named) return error.UnknownDefaultVoice;
        if (st.voices.len > 0) st.default = &st.voices[0];
    }
    return st;
}

/// A request's own voice: 1 to 900 frames of ten codec codes, each below 4096.
fn referenceCodes(cx: *errors.Cx, value: json.Value) errors.Refused![]const [req.config.codebooks]i32 {
    const message = error_prefix ++ " speaker_reference_codes must be 1 to 900 frames of ten integers from 0 through 4095";
    const frames = value.items() orelse return cx.refuse(message);
    if (frames.len == 0 or frames.len > req.reference_frames) return cx.refuse(message);
    const out = try cx.a.alloc([req.config.codebooks]i32, frames.len);
    for (frames, out) |frame, *row| {
        const codes = frame.items() orelse return cx.refuse(message);
        if (codes.len != row.len) return cx.refuse(message);
        for (codes, row) |code, *slot| {
            if (code != .int) return cx.refuse(message);
            slot.* = std.fmt.parseInt(i32, code.int, 10) catch return cx.refuse(message);
            if (slot.* < 0 or slot.* >= req.config.codebook_size) return cx.refuse(message);
        }
    }
    return out;
}

/// `ref_audio` (a base64 WAV, or a data: URL of one) to mono 44.1 kHz as the release prepares a clip, cut to 60 s.
fn referenceAudio(cx: *errors.Cx, value: json.Value) errors.Refused![]const f32 {
    const text = value.str() orelse return cx.refuse(error_prefix ++ " ref_audio must be a base64 string holding a WAV file");
    const payload = if (std.mem.startsWith(u8, text, "data:")) text[(std.mem.indexOfScalar(u8, text, ',') orelse return cx.refuse(error_prefix ++ " ref_audio is a data URL without a payload")) + 1 ..] else text;
    const decoder = std.base64.standard.Decoder;
    const size = decoder.calcSizeForSlice(payload) catch return cx.refuse(error_prefix ++ " ref_audio is not valid base64");
    const bytes = try cx.a.alloc(u8, size);
    decoder.decode(bytes, payload) catch return cx.refuse(error_prefix ++ " ref_audio is not valid base64");
    const wav = @import("core").wav;
    const clip = wav.parse(cx.a, bytes) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.refuse(error_prefix ++ " ref_audio must be a WAV file: PCM of 8, 16, 24 or 32 bits, or 32-bit float"),
    };
    if (clip.rate < 8000 or clip.rate > 192000 or clip.rate % 50 != 0) return cx.refuse(error_prefix ++ " ref_audio's sample rate must be a multiple of 50 Hz from 8000 to 192000");
    const audio = try wav.resample(cx.a, clip.samples, clip.rate, rate);
    if (audio.len < min_clip) return cx.refuse(error_prefix ++ " ref_audio must be at least half a second long");
    var peak: f32 = 0;
    for (audio) |x| {
        if (!std.math.isFinite(x)) return cx.refuse(error_prefix ++ " ref_audio holds samples that are not finite");
        peak = @max(peak, @abs(x));
    }
    if (peak > 1) for (audio) |*x| {
        x.* /= peak;
    };
    return audio[0..@min(audio.len, max_clip)];
}

fn number(v: json.Value) ?f64 {
    return switch (v) {
        .float => |f| f,
        .int => |t| std.fmt.parseFloat(f64, t) catch null,
        else => null,
    };
}

/// `condition_buckets` or `condition_values` as an entry a condition row: an object by name, or a list in prompt order.
fn conditionEntries(cx: *errors.Cx, what: []const u8, raw: ?json.Value) errors.Refused![req.config.conditions]?json.Value {
    const names = req.config.condition_names;
    var out: [names.len]?json.Value = @splat(null);
    const value = raw orelse return out;
    switch (value) {
        .object => |o| for (o.keys(), o.values()) |key, v| {
            const i = for (names, 0..) |name, j| {
                if (std.mem.eql(u8, key, name)) break j;
            } else return cx.fail(.request, error_prefix ++ " {s}: unknown condition name {s}; expected loudness, fmax, speaking_rate", .{ what, try pyrepr.repr(cx.a, .{ .string = key }) });
            if (v != .null) out[i] = v;
        },
        .array => |list| {
            if (list.len != names.len) return cx.fail(.request, error_prefix ++ " {s}: expected 3 entries in the order loudness, fmax, speaking_rate", .{what});
            for (list, &out) |v, *slot| {
                if (v != .null) slot.* = v;
            }
        },
        else => return cx.fail(.request, error_prefix ++ " {s} must be an object by condition name or a list of 3", .{what}),
    }
    return out;
}

/// plan.resolve_condition_buckets: an id wins over a raw value, a row left out is unknown; empty: every row is.
fn conditions(st: *const State, cx: *errors.Cx, raw: json.Value) errors.Refused![]const u32 {
    const ids = try conditionEntries(cx, fields.buckets, raw.field(fields.buckets));
    const values = try conditionEntries(cx, fields.values, raw.field(fields.values));
    const out = try cx.a.alloc(u32, ids.len);
    var named = false;
    for (out, ids, values, req.config.condition_names, 0..) |*bucket, id, value, name, i| {
        const count = st.config.buckets[i];
        bucket.* = count;
        if (id) |v| {
            const message = error_prefix ++ " condition_buckets[{s}] must be an integer from 0 through {d}, or {d} (unknown)";
            if (v != .int) return cx.fail(.request, message, .{ name, count - 1, count });
            bucket.* = std.fmt.parseInt(u32, v.int, 10) catch return cx.fail(.request, message, .{ name, count - 1, count });
            if (bucket.* > count) return cx.fail(.request, message, .{ name, count - 1, count });
        } else if (value) |v| {
            const x = number(v) orelse return cx.fail(.request, error_prefix ++ " condition_values[{s}] must be a number", .{name});
            if (!std.math.isFinite(x)) return cx.fail(.request, error_prefix ++ " condition_values[{s}] must be finite", .{name});
            bucket.* = st.config.bucketOf(i, x);
        } else continue;
        named = true;
    }
    return if (named) out else &.{};
}

/// OpenAI's own voice names, which its clients send unless told otherwise: each speaks in the default voice.
const stock_voices = [_][]const u8{ "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin", "nova", "onyx", "sage", "shimmer", "verse" };

pub fn stockVoice(name: []const u8) bool {
    for (stock_voices) |v| if (std.ascii.eqlIgnoreCase(name, v)) return true;
    return false;
}

/// The text the tokenizer gets: line breaks collapsed, then its spoken form only when the request sends `true`.
pub fn spokenText(a: Allocator, input: []const u8, lang: textnorm.Language, normalize: ?bool) ![]u8 {
    const typed = try req.prompt.collapseLineBreaks(a, input);
    if (normalize != true) return typed;
    defer a.free(typed);
    return textnorm.normalize(a, typed, lang);
}

/// The request's prompt: the voice's reference codes as a delayed prefix, the condition rows, the text, the silence.
pub fn plan(st: *const State, cx: *errors.Cx, text: text_mod.Text, sp: api.SpeechInfo, input: []const u8, voice: []const u8, raw: json.Value) errors.Refused!Made {
    const own = raw.field(fields.codes);
    const clip = raw.field(fields.clip);
    const language = raw.strField(fields.language);
    const normalize: ?bool = if (raw.field(fields.normalize)) |v| (if (v == .bool) v.bool else return cx.refuse("text_normalization must be a boolean")) else null;
    // speaker_cfg.guided_logits: 1 is unguided, 0 the prompt without its voice alone
    const guidance: f32 = if (raw.field(fields.guidance)) |v| @floatCast(number(v) orelse return cx.refuse(error_prefix ++ " speaker_cfg_scale must be a number")) else 1;
    if (!std.math.isFinite(guidance) or guidance < 0) return cx.refuse(error_prefix ++ " speaker_cfg_scale must be finite and not negative");
    const buckets = try conditions(st, cx, raw);
    const lang = textnorm.language(language orelse textnorm.default_language) orelse
        return cx.fail(.request, error_prefix ++ " unsupported language code {s}; supported: " ++ textnorm.codes, .{try pyrepr.repr(cx.a, .{ .string = language.? })});
    if (normalize == true and !textnorm.available(cx.a)) return cx.refuse(textnorm.missing());
    if (own != null and clip != null) return cx.refuse(error_prefix ++ " send ref_audio or speaker_reference_codes, not both");
    const audio: []const f32 = if (clip) |value| try referenceAudio(cx, value) else &.{};
    const reference: []const [req.config.codebooks]i32 = if (clip != null) &.{} else if (own) |value| try referenceCodes(cx, value) else blk: {
        const chosen = if (voice.len == 0 or std.ascii.eqlIgnoreCase(voice, "default")) st.default else req.voices.find(st.voices, voice) orelse if (stockVoice(voice)) st.default else null;
        const v = chosen orelse {
            if (st.voices.len == 0) return cx.refuse(error_prefix ++ " this release folder has no voices under gguf/voices");
            return cx.fail(.request, error_prefix ++ " unknown voice {s}; see /v1/voices", .{try pyrepr.repr(cx.a, .{ .string = voice })});
        };
        break :blk v.codes;
    };
    const spoken = spokenText(cx.a, input, lang, normalize) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.NoLibrary => cx.refuse(textnorm.missing()),
        else => cx.refuse("input must be valid UTF-8 text"),
    };
    const ids = text.encode(cx.a, spoken, false) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.other("the tokenizer cannot encode this prompt"),
    };
    const speaker: req.prompt.Speaker = if (clip != null) .none else .{ .reference = reference };
    const built = req.prompt.build(cx.a, ids, speaker, st.silence, req.silence_frames) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.refuse(error_prefix ++ " the text has nothing to speak"),
    };
    // a clip's prefix is made by the engine: a row a frame and nine more
    const rows = built.rows.len + if (audio.len > 0) (audio.len + 2939) / 2940 + req.config.codebooks - 1 else 0;
    if (rows > sp.max_prompt_tokens)
        return cx.fail(.request, error_prefix ++ " a prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ rows, sp.max_prompt_tokens });
    return .{ .prompt = ids, .frames = built.rows, .reference_audio = audio, .conditions = buckets, .guidance = guidance };
}

pub fn listing(a: Allocator, st: *const State) !json.Value {
    const data = try a.alloc(json.Value, st.voices.len + 1);
    data[0] = .{ .string = "default" };
    for (st.voices, data[1..]) |voice, *slot| slot.* = .{ .string = voice.name };
    const details = try a.alloc(json.Value, st.voices.len);
    for (st.voices, details) |voice, *slot| {
        const entry = try json.newObject(a);
        try entry.put(a, "id", .{ .string = voice.name });
        try entry.put(a, "description", .{ .string = try std.fmt.allocPrint(a, "{s} ({s})", .{ voice.clip, voice.language }) });
        slot.* = .{ .object = entry };
    }
    const object = try json.newObject(a);
    try object.put(a, "voices", .{ .array = data });
    try object.put(a, "details", .{ .array = details });
    if (st.default) |d| try object.put(a, "default", .{ .string = d.name });
    return .{ .object = object };
}

test "OpenAI's stock voice names are known, whatever their case; other names are not" {
    try std.testing.expect(stockVoice("alloy"));
    try std.testing.expect(stockVoice("Nova"));
    try std.testing.expect(!stockVoice("nobody"));
    try std.testing.expect(!stockVoice(""));
}
