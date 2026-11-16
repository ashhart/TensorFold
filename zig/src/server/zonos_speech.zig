//! ZONOS3 Mini behind /v1/audio/speech: the release's default voices and a request's prompt rows.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const pyrepr = @import("pyrepr.zig");
const text_mod = @import("model_text.zig");
const req = @import("zonos_request");
const Allocator = std.mem.Allocator;

pub const error_prefix = "ZONOS3 request:";
pub const temperature = req.temperature;
pub const top_p = req.top_p;
pub const top_k = req.top_k;
pub const max_top_k = req.config.codebook_size;

pub const State = struct {
    voices: []const req.voices.Voice,
    silence: [req.config.codebooks]i32,
    /// The voice "default" names: --default-voice, else savannah, else the first voice.
    default: ?*const req.voices.Voice,
};

pub const Made = struct { prompt: []const u32, frames: []const [11]i32 };

/// The state for a ZONOS3 Mini release at `dir` (from the arena `a`), or null for any other checkpoint.
pub fn load(a: Allocator, io: std.Io, dir: []const u8, default_voice: []const u8) !?*State {
    if (dir.len == 0 or !req.release.present(io, dir)) return null;
    var rel = try req.release.open(a, io, dir, null);
    defer rel.deinit();
    const st = try a.create(State);
    st.* = .{ .voices = try req.voices.load(a, io, dir, req.reference_frames), .silence = rel.silence, .default = null };
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

/// The request's prompt: the voice's reference codes as a delayed prefix, the condition rows, the text, the silence.
pub fn plan(st: *const State, cx: *errors.Cx, text: text_mod.Text, sp: api.SpeechInfo, input: []const u8, voice: []const u8, own: ?json.Value) errors.Refused!Made {
    const reference = if (own) |value| try referenceCodes(cx, value) else blk: {
        const chosen = if (voice.len == 0 or std.ascii.eqlIgnoreCase(voice, "default")) st.default else req.voices.find(st.voices, voice);
        const v = chosen orelse {
            if (st.voices.len == 0) return cx.refuse(error_prefix ++ " this release folder has no voices under gguf/voices");
            return cx.fail(.request, error_prefix ++ " unknown voice {s}; see /v1/voices", .{try pyrepr.repr(cx.a, .{ .string = voice })});
        };
        break :blk v.codes;
    };
    const spoken = req.prompt.collapseLineBreaks(cx.a, input) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.refuse("input must be valid UTF-8 text"),
    };
    const ids = text.encode(cx.a, spoken, false) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.other("the tokenizer cannot encode this prompt"),
    };
    const built = req.prompt.build(cx.a, ids, .{ .reference = reference }, st.silence, req.silence_frames) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => cx.refuse(error_prefix ++ " the text has nothing to speak"),
    };
    if (built.rows.len > sp.max_prompt_tokens)
        return cx.fail(.request, error_prefix ++ " a prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ built.rows.len, sp.max_prompt_tokens });
    return .{ .prompt = ids, .frames = built.rows };
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
