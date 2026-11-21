//! ZONOS3 Mini behind /v1/audio/speech: the release's default voices and a request's prompt rows.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const pyrepr = @import("pyrepr.zig");
const text_mod = @import("model_text.zig");
const req = @import("zonos_request");
const textnorm = @import("zonos_textnorm.zig");
const steer = @import("zonos_steer.zig");
const laci = @import("zonos_laci.zig");
const tokens = @import("zonos_tokens.zig");
const short = @import("zonos_short.zig");
const reading = @import("zonos_reading.zig");
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
    pub const mode = "speaker_mode";
    pub const embedding = "speaker_embedding";
    pub const start = "start_immediately";
    pub const decoder = "decoder";
    pub const all = [_][]const u8{ codes, clip, language, normalize, guidance, buckets, values, mode, embedding, laci.field, short.field, start, decoder } ++ steer.fields.all;
};

pub const State = struct {
    config: req.config.Config,
    voices: []const req.voices.Voice,
    silence: [req.config.codebooks]i32,
    /// The voice "default" names: --default-voice, else savannah, else the first voice.
    default: ?*const req.voices.Voice,
    /// --steer-bundle: the directions a request's controls name; null: controls are refused.
    bundle: ?*const req.steer_bundle.Bundle = null,
    /// --laci-heads: each language's alignment heads; null: `laci` is refused.
    laci: ?*const laci.Table = null,
    /// The release's phoneme, start and pause tokens; null: it has none, and their markers are refused.
    tokens: ?tokens.Spec = null,
    /// --short-text: what makes a text short and where its take is cut; null: no text is treated as short.
    short: ?*const short.Config = null,
    /// The release folder holds the speech-edit adapter: /v1/audio/edit is served.
    editing: bool = false,
    /// The release folder holds the forced aligner (web/aligner): edits may read a clip's words from its transcript.
    aligning: bool = false,
    /// The release's speaker encoder, when its checkpoint is in the folder: a clip's embedding for the speaker row.
    speaker: ?*anyopaque = null,

    /// A clip's speaker embedding from its samples at their own rate, on this thread's cores; null without the encoder.
    pub fn embed(st: *const State, clip: Clip) !?[req.config.speaker_dim]f32 {
        if (!@hasDecl(req, "speaker")) return null;
        const encoder: *const req.speaker.Encoder = @ptrCast(@alignCast(st.speaker orelse return null));
        return try encoder.embed(std.heap.c_allocator, clip.source, clip.rate, null);
    }
};

pub const Made = struct { prompt: []const u32, frames: []const [11]i32, reference_audio: []const f32 = &.{}, conditions: []const u32 = &.{}, guidance: f32 = 1, speaker: []const f32 = &.{}, steering: ?*const api.SpeechSteering = null, laci: ?*const api.SpeechLaci = null, decoder: u8 = 0 };

/// speech.resolve_decoder: the request's `decoder` among those served, as an index (0, the default, when null).
pub fn decoderOf(cx: *errors.Cx, sp: api.SpeechInfo, raw: json.Value) errors.Refused!u8 {
    const v = raw.field(fields.decoder) orelse return 0;
    if (v == .null) return 0;
    const id = v.str() orelse return cx.refuse(error_prefix ++ " decoder must be a string");
    for (sp.decoders, 0..) |d, i| if (std.mem.eql(u8, d, id)) return @intCast(i);
    return cx.fail(.request, "Unknown decoder '{s}'; this server serves: {s}.", .{ id, try std.mem.join(cx.a, ", ", sp.decoders) });
}

/// GET /v1/audio/decoders: the decoders a request may name, the default first.
pub fn decoders(a: Allocator, sp: api.SpeechInfo) !json.Value {
    const list = try a.alloc(json.Value, sp.decoders.len);
    for (list, sp.decoders) |*v, id| v.* = .{ .string = id };
    const object = try json.newObject(a);
    try object.put(a, "decoders", .{ .array = list });
    if (sp.decoders.len > 0) try object.put(a, "default", .{ .string = sp.decoders[0] });
    return .{ .object = object };
}

const rate = 44100;
/// A reference clip: at least half a second, and the first 60 s of it in whole frames of 2940 samples.
const min_clip = rate / 2;
const max_clip = req.reference_frames * 2940;

/// The state for a ZONOS3 Mini release at `dir` (from the arena `a`), or null for any other checkpoint.
pub fn load(a: Allocator, io: std.Io, dir: []const u8, default_voice: []const u8, steer_bundle: ?[]const u8, laci_heads: ?[]const u8, short_text: ?[]const u8) !?*State {
    if (dir.len == 0 or !req.release.present(io, dir)) return if (steer_bundle != null) error.SteerBundleNeedsZonos else if (laci_heads != null or short_text != null) error.LaciHeadsNeedZonos else null;
    var rel = try req.release.open(a, io, dir, null);
    defer rel.deinit();
    const st = try a.create(State);
    st.* = .{ .config = try req.config.Config.load(a, io, rel.lm), .voices = try req.voices.load(a, io, dir, req.reference_frames), .silence = rel.silence, .default = null, .editing = rel.edit_adapter.len != 0, .aligning = rel.aligner.len != 0 };
    const named = !std.mem.eql(u8, default_voice, "default");
    st.default = req.voices.find(st.voices, if (named) default_voice else "savannah");
    if (st.default == null) {
        if (named) return error.UnknownDefaultVoice;
        if (st.voices.len > 0) st.default = &st.voices[0];
    }
    if (steer_bundle) |path| {
        const bundle = try a.create(req.steer_bundle.Bundle);
        bundle.* = try req.steer_bundle.load(a, io, path);
        st.bundle = bundle;
    }
    if (@hasDecl(req, "speaker") and rel.speaker_encoder.len != 0) st.speaker = try req.speaker.load(std.heap.c_allocator, io, rel.speaker_encoder);
    if (laci_heads) |path| st.laci = try laci.load(a, io, path);
    st.tokens = try tokens.Spec.load(a, io, rel.lm);
    if (short_text) |path| st.short = try short.load(a, io, path);
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

/// A clip field (a base64 WAV, or a data: URL of one) to mono 44.1 kHz as the release prepares one; `cut`: to 60 s.
pub fn clipAudio(cx: *errors.Cx, value: json.Value, comptime name: []const u8, cut: bool) errors.Refused![]const f32 {
    return (try clipSourced(cx, value, name, cut)).audio;
}

/// A clip at 44.1 kHz for the codec, and as sent (`source` at `rate`) for the speaker encoder and aligner to resample.
pub const Clip = struct { audio: []const f32, source: []const f32, rate: u32 };

pub fn clipSourced(cx: *errors.Cx, value: json.Value, comptime name: []const u8, cut: bool) errors.Refused!Clip {
    const what = error_prefix ++ " " ++ name;
    const text = value.str() orelse return cx.refuse(what ++ " must be a base64 string holding a WAV file");
    const payload = if (std.mem.startsWith(u8, text, "data:")) text[(std.mem.indexOfScalar(u8, text, ',') orelse return cx.refuse(what ++ " is a data URL without a payload")) + 1 ..] else text;
    const decoder = std.base64.standard.Decoder;
    const size = decoder.calcSizeForSlice(payload) catch return cx.refuse(what ++ " is not valid base64");
    const bytes = try cx.a.alloc(u8, size);
    decoder.decode(bytes, payload) catch return cx.refuse(what ++ " is not valid base64");
    const wav = @import("core").wav;
    const clip = wav.parse(cx.a, bytes) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.refuse(what ++ " must be a WAV file: PCM of 8, 16, 24 or 32 bits, or 32-bit float"),
    };
    if (clip.rate < 8000 or clip.rate > 192000 or clip.rate % 50 != 0) return cx.refuse(what ++ "'s sample rate must be a multiple of 50 Hz from 8000 to 192000");
    const audio = try wav.resample(cx.a, clip.samples, clip.rate, rate);
    if (audio.len < min_clip) return cx.refuse(what ++ " must be at least half a second long");
    var peak: f32 = 0;
    for (audio) |x| {
        if (!std.math.isFinite(x)) return cx.refuse(what ++ " holds samples that are not finite");
        peak = @max(peak, @abs(x));
    }
    if (peak > 1) for (audio) |*x| {
        x.* /= peak;
    };
    const kept = if (cut) audio[0..@min(audio.len, max_clip)] else audio;
    if (clip.rate == rate) return .{ .audio = kept, .source = kept, .rate = rate };
    for (clip.samples) |x| if (!std.math.isFinite(x)) return cx.refuse(what ++ " holds samples that are not finite");
    const seconds = if (cut) max_clip / rate else clip.samples.len;
    return .{ .audio = kept, .source = clip.samples[0..@min(clip.samples.len, seconds * clip.rate)], .rate = clip.rate };
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
pub fn conditions(st: *const State, cx: *errors.Cx, raw: json.Value) errors.Refused![]const u32 {
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
    const decoder = try decoderOf(cx, sp, raw);
    const lang = textnorm.language(language orelse textnorm.default_language) orelse
        return cx.fail(.request, error_prefix ++ " unsupported language code {s}; supported: " ++ textnorm.codes, .{try pyrepr.repr(cx.a, .{ .string = language.? })});
    if (normalize == true and !textnorm.available(cx.a)) return cx.refuse(textnorm.missing());
    if (own != null and clip != null) return cx.refuse(error_prefix ++ " send ref_audio or speaker_reference_codes, not both");
    // a voice by its speaker embedding (the speaker row) in place of reference codes (a prefix)
    const sent = raw.field(fields.embedding);
    const by_embedding = if (raw.field(fields.mode)) |v| blk: {
        const name = v.str() orelse return cx.refuse(error_prefix ++ " speaker_mode must be \"reference\" or \"embedding\"");
        if (std.mem.eql(u8, name, "embedding")) break :blk true;
        if (!std.mem.eql(u8, name, "reference")) return cx.refuse(error_prefix ++ " speaker_mode must be \"reference\" or \"embedding\"");
        if (sent != null) return cx.refuse(error_prefix ++ " speaker_embedding needs speaker_mode \"embedding\"");
        break :blk false;
    } else sent != null;
    if (by_embedding and own != null) return cx.refuse(error_prefix ++ " speaker_mode \"embedding\" takes a voice name, speaker_embedding or ref_audio; codes carry no embedding");
    if (by_embedding and clip != null and sent != null) return cx.refuse(error_prefix ++ " send ref_audio or speaker_embedding, not both");
    var vector: [req.config.speaker_dim]f32 = undefined;
    if (sent) |value| {
        const message = error_prefix ++ " speaker_embedding must be 192 finite numbers";
        const list = value.items() orelse return cx.refuse(message);
        if (list.len != vector.len) return cx.refuse(message);
        for (&vector, list) |*x, v| {
            x.* = @floatCast(number(v) orelse return cx.refuse(message));
            if (!std.math.isFinite(x.*)) return cx.refuse(message);
        }
    }
    const uploaded: ?Clip = if (clip) |value| try clipSourced(cx, value, fields.clip, true) else null;
    var audio: []const f32 = if (uploaded) |u| u.audio else &.{};
    if (by_embedding and clip != null) {
        // the clip's embedding fills the speaker row; its codes are not a prefix
        vector = (st.embed(uploaded.?) catch return cx.other("the speaker encoder failed on this clip")) orelse
            return cx.refuse(error_prefix ++ " a clip's embedding needs the release's speaker encoder (speaker_encoders/), which this folder lacks");
        audio = &.{};
    }
    const reference: []const [req.config.codebooks]i32 = if (clip != null or sent != null) &.{} else if (own) |value| try referenceCodes(cx, value) else blk: {
        const chosen = if (voice.len == 0 or std.ascii.eqlIgnoreCase(voice, "default")) st.default else req.voices.find(st.voices, voice) orelse if (stockVoice(voice)) st.default else null;
        const v = chosen orelse {
            if (st.voices.len == 0) return cx.refuse(error_prefix ++ " this release folder has no voices under gguf/voices");
            return cx.fail(.request, error_prefix ++ " unknown voice {s}; see /v1/voices", .{try pyrepr.repr(cx.a, .{ .string = voice })});
        };
        vector = v.embedding;
        break :blk v.codes;
    };
    const chars = std.unicode.utf8CountCodepoints(input) catch return cx.refuse("input must be valid UTF-8 text");
    const steered = try steer.plan(st.bundle, cx, raw, chars, if (by_embedding) &vector else null, guidance != 1);
    if (steered.embedding) |moved| vector = moved;
    const controls_guided = if (raw.field(steer.fields.guidance)) |v| (number(v) orelse 1) != 1 else false;
    const spoken = spokenText(cx.a, input, lang, normalize) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.NoLibrary => cx.refuse(textnorm.missing()),
        else => cx.refuse("input must be valid UTF-8 text"),
    };
    // start_immediately opens the text with the start marker, which asks for speech from the first frame
    const wants_start = if (raw.field(fields.start)) |v| (if (v == .bool) v.bool else return cx.refuse(error_prefix ++ " start_immediately must be a boolean")) else false;
    const opened = if (wants_start and !std.mem.startsWith(u8, spoken, tokens.start_marker)) try std.mem.concat(cx.a, u8, &.{ tokens.start_marker, spoken }) else spoken;
    // a short text runs under LACI with a pause and a filler sentence after it, cut where the filler starts
    const guided = guidance != 1 or controls_guided;
    const brief = try short.plan(st.short, cx, raw, text, st.tokens, opened, lang.code, st.laci != null and !guided and raw.field(steer.fields.spans) == null);
    var watched = try laci.plan(st.laci, cx, raw, lang, guided);
    if (brief) |sh| {
        const with = try cx.a.create(api.SpeechLaci);
        with.* = laci.of(st.laci.?, lang).*;
        with.short = sh.short;
        watched = with;
    }
    const said = if (brief) |sh| sh.text else opened;
    // spans that follow the reading: their tokens in the text the prompt spells, read by the language's heads
    if (steered.follow) |f| if (st.laci) |table| if (steered.steering) |s| {
        s.follow = .{ .ranges = try reading.ranges(cx, text, input, said, f.chars), .lead = f.lead, .lag = f.lag, .heads = laci.of(table, lang).heads };
    };
    const ids = try tokens.payload(cx, text, st.tokens, said);
    const silence_frames: usize = if (std.mem.startsWith(u8, said, tokens.start_marker)) 0 else req.silence_frames;
    const speaker: req.prompt.Speaker = if (by_embedding) .embedding else if (clip != null) .none else .{ .reference = reference };
    const built = req.prompt.build(cx.a, ids, speaker, st.silence, silence_frames) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.refuse(error_prefix ++ " the text has nothing to speak"),
    };
    // a clip's prefix is made by the engine: a row a frame and nine more
    const rows = built.rows.len + if (audio.len > 0) (audio.len + 2939) / 2940 + req.config.codebooks - 1 else 0;
    if (rows > sp.max_prompt_tokens)
        return cx.fail(.request, error_prefix ++ " a prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ rows, sp.max_prompt_tokens });
    return .{ .prompt = ids, .frames = built.rows, .reference_audio = audio, .conditions = buckets, .guidance = guidance, .speaker = if (by_embedding) try cx.a.dupe(f32, &vector) else &.{}, .steering = steered.steering, .laci = watched, .decoder = decoder };
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
