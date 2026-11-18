//! POST /v1/audio/edit: recorded speech re-said on ZONOS3 Mini's edit adapter, as the release's /v2/edit plans it.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const ids = @import("ids.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const speech = @import("speech.zig");
const streaming = @import("speech_stream.zig");
const speech_sampling = @import("speech_sampling.zig");
const req = @import("zonos_request");
const zonos = @import("zonos_speech.zig");
const steer = @import("zonos_steer.zig");
const textnorm = @import("zonos_textnorm.zig");
const tokens = @import("zonos_tokens.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;
const Cx = errors.Cx;
const Value = json.Value;

pub const route = "/v1/audio/edit";
/// POST {audio_base64, text}: the text's words placed in the clip, as an edit's original_words take them.
pub const align_route = "/v1/audio/edit/align";
const aw = req.align_words;
/// The languages the release edits in; Japanese and Mandarin are cut into words by ICU there, which is not served.
const edit_languages = [_][]const u8{ "en_us", "en_gb", "fr_fr", "de", "es", "it", "pt_br", "ko" };
const prefix = "ZONOS3 edit: ";
const C = req.config.codebooks;
const W = req.config.frame_width;
const rate = 15.0;
const ep = req.edit_plan;

const own = [_][]const u8{ "model", "audio_base64", "target_text", "original_text", "original_words", "spans", "margin_seconds", "max_span_seconds", "language", "text_normalization", "seed", "num_takes", "speaker_embedding", "condition_buckets", "condition_values", "controls", "region_controls", "steer_hold", "steer_onset_seconds", "steer_ramp_seconds", "max_context_seconds", "context_seconds", "min_fill_ratio", "verify", "decoder" };

fn known(key: []const u8) bool {
    for (own) |name| if (std.mem.eql(u8, key, name)) return true;
    return speech_sampling.isField(key);
}

/// A number field inside [lo, hi] (`open`: above lo), or its default.
fn ranged(cx: *Cx, raw: Value, comptime name: []const u8, default: f64, lo: f64, hi: f64, open: bool) errors.Refused!f64 {
    const v = raw.field(name) orelse return default;
    const x = v.float64() orelse return cx.refuse(prefix ++ name ++ " must be a number");
    if (v == .bool or !std.math.isFinite(x) or x > hi or x < lo or (open and x == lo)) return cx.fail(.request, prefix ++ name ++ " must be {s} {d} and at most {d}", .{ if (open) "above" else "at least", lo, hi });
    return x;
}

/// One engine step of the edit, waited for; false when the client left (nothing more is sent).
fn step(srv: *Server, c: *Conn, cx: *Cx, request: *const api.SpeechRequest) errors.Refused!bool {
    var box: Mailbox = .{ .io = srv.io, .gpa = srv.gpa };
    defer box.deinit();
    const engine_id = srv.next_id.fetchAdd(1, .monotonic);
    srv.engine.submitSpeech(engine_id, request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => cx.fail(.capacity, "the engine is busy; retry shortly", .{}),
        error.Closed => cx.other("the scheduler is closed"),
        error.InvalidRequest => cx.refuse(prefix ++ "the engine refused a step of this edit (its prompt, clip or frame limit)"),
        error.Unsupported => cx.refuse("this model does not support speech editing"),
    };
    if (streaming.settle(srv, c, &box, engine_id, null) == null) return false;
    if (streaming.why(&box, request.result.samples, request.pcm.len)) |message| return cx.fail(.server, "{s}", .{try cx.a.dupe(u8, message)});
    return true;
}

fn words(cx: *Cx, value: Value, original: []const u8, duration: f64) errors.Refused![]const ep.Word {
    const message = prefix ++ "original_words must be a list of {word, start, end} in order";
    const list = value.items() orelse return cx.refuse(message);
    const out = try cx.a.alloc(ep.Word, list.len);
    for (out, list) |*w, entry| {
        const word = entry.strField("word") orelse return cx.refuse(message);
        const start = (entry.field("start") orelse return cx.refuse(message)).float64() orelse return cx.refuse(message);
        const end = (entry.field("end") orelse return cx.refuse(message)).float64() orelse return cx.refuse(message);
        w.* = .{ .word = word, .start = start, .end = end, .confidence = if (entry.field("confidence")) |v| v.float64() else null };
        if (entry.field("tokens")) |sent| {
            const given = sent.items() orelse return cx.refuse(message);
            const list_tokens = try cx.a.alloc([]const u8, given.len);
            for (list_tokens, given) |*t, v| t.* = v.str() orelse return cx.refuse(message);
            w.tokens = list_tokens;
        } else w.tokens = ep.tokens(cx.a, word) catch |e| return tokenFault(cx, e);
    }
    var at: usize = 0;
    const spelled = ep.tokens(cx.a, original) catch |e| return tokenFault(cx, e);
    ep.checkWords(out, spelled, duration, &at) catch |e| return cx.fail(.request, prefix ++ "original_words[{d}] ({s}): {s}", .{ at, out[@min(at, out.len -| 1)].word, switch (e) {
        error.NoOriginalWords => "the list is empty",
        error.WordTimeNotFinite => "a time is not finite",
        error.WordBeforeClip => "it starts before the clip",
        error.WordEndsBeforeItStarts => "it must start before it ends",
        error.WordsOverlap => "it starts before the previous word ends",
        error.WordAfterClip => "it ends after the clip",
        error.WordsDoNotSpellText => "the words do not spell original_text",
    } });
    return out;
}

fn tokenFault(cx: *Cx, e: anyerror) errors.Refused {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.EditTextNeedsWordBreaks => cx.refuse(prefix ++ "Han and kana text is cut into words by ICU in the release, which is not served here"),
        else => cx.refuse(prefix ++ "the text must be valid UTF-8"),
    };
}

fn object(a: Allocator) !*json.Object {
    return json.newObject(a);
}

fn pairs(a: Allocator, list: []const [2]f64) !Value {
    const out = try a.alloc(Value, list.len);
    for (out, list) |*o, p| {
        const two = try a.alloc(Value, 2);
        two[0] = .{ .float = p[0] };
        two[1] = .{ .float = p[1] };
        o.* = .{ .array = two };
    }
    return .{ .array = out };
}

fn floats(a: Allocator, list: []const f64) !Value {
    const out = try a.alloc(Value, list.len);
    for (out, list) |*o, x| o.* = .{ .float = x };
    return .{ .array = out };
}

/// Region k's steering: `controls` plus `region_controls[k]`, doses summed by id and zero ones dropped.
fn regionSteering(st: *const zonos.State, cx: *Cx, raw: Value, k: ?usize, chars: usize, embedding: ?*const [req.config.speaker_dim]f32) errors.Refused!steer.Made {
    const merged = try object(cx.a);
    const sources = [_]?Value{ raw.field("controls"), if (k) |i| (if (raw.field("region_controls")) |list| (if (list.array[i] == .null) null else list.array[i]) else null) else null };
    for (sources) |source| if (source) |doses| {
        if (doses != .object) return cx.refuse(prefix ++ "controls and each region_controls entry must be an object of control names and doses");
        for (doses.object.keys(), doses.object.values()) |id, v| {
            const dose = v.float64() orelse return cx.refuse(prefix ++ "every dose must be a number");
            const before = if (merged.get(id)) |had| had.float else 0;
            try merged.put(cx.a, id, .{ .float = before + dose });
        }
    };
    const kept = try object(cx.a);
    for (merged.keys(), merged.values()) |id, v| if (v.float != 0) try kept.put(cx.a, id, v);
    const sent = try object(cx.a);
    if (kept.count() != 0) try sent.put(cx.a, steer.fields.controls, .{ .object = kept });
    inline for (.{ steer.fields.hold, steer.fields.onset, steer.fields.ramp }) |name| if (raw.field(name)) |v| try sent.put(cx.a, name, v);
    const made = try steer.plan(st.bundle, cx, .{ .object = sent }, chars, embedding, false);
    if (made.embedding != null and embedding == null) return cx.refuse(prefix ++ "a control that moves the speaker embedding needs the speaker row: leave speaker_embedding on");
    return made;
}

fn editLanguage(cx: *Cx, raw: Value) errors.Refused!textnorm.Language {
    const lang = textnorm.language(raw.strField("language") orelse textnorm.default_language) orelse return cx.refuse(prefix ++ "unsupported language code; supported: " ++ textnorm.codes);
    for (edit_languages) |code| if (std.mem.eql(u8, code, lang.code)) return lang;
    return cx.refuse(prefix ++ "speech editing serves en_us, en_gb, fr_fr, de, es, it, pt_br and ko (ja and cmn need ICU word breaks, which are not served)");
}

fn hold(srv: *Server, cx: *Cx) errors.Refused!void {
    var held = srv.speech_inflight.load(.acquire);
    while (true) {
        if (held >= @max(1, srv.info.lanes)) return cx.fail(.capacity, "speech output capacity is full", .{});
        if (srv.speech_inflight.cmpxchgWeak(held, held + 1, .acq_rel, .acquire)) |now| held = now else break;
    }
}

/// edit_aligned_words: the text's words placed by the aligner, with edit tokens; null: the client left.
fn alignText(srv: *Server, c: *Conn, cx: *Cx, st: *const zonos.State, audio: []const f32, transcript: []const u8) errors.Refused!?[]ep.Word {
    const a = cx.a;
    if (!st.aligning) return cx.refuse(prefix ++ "this release folder has no aligner (web/aligner): send original_words with their timings");
    if (audio.len > 30 * 44100) return cx.refuse(prefix ++ "the aligner places words in clips of at most 30 s: send original_words with their timings");
    const spoken = ep.words(a, transcript) catch return cx.refuse(prefix ++ "the text must be valid UTF-8");
    var wordy: std.ArrayList([]const u8) = .empty;
    var bytes: usize = 0;
    for (spoken) |t| if (aw.isWordy(t)) {
        bytes += t.len + @intFromBool(wordy.items.len != 0);
        try wordy.append(a, t);
    };
    if (bytes > aw.max_bytes) return cx.fail(.request, prefix ++ "the transcript is {d} bytes; the aligner takes at most {d}: send original_words with their timings", .{ bytes, aw.max_bytes });
    const placed = try a.alloc(?[2]f64, wordy.items.len);
    const presence = try a.alloc(f32, wordy.items.len);
    if (wordy.items.len != 0) {
        const spans = try a.alloc([2]u32, wordy.items.len);
        var result: api.SpeechResult = .{};
        var out: api.SpeechEdit.Out = .{};
        const ask: api.SpeechEdit = .{ .op = .place, .words = wordy.items, .spans_ms = spans, .presence = presence, .out = &out };
        if (!try step(srv, c, cx, &.{ .prompt = &.{}, .pcm = &.{}, .result = &result, .seed = 0, .reference_audio = audio, .edit = &ask })) return null;
        for (placed, presence, spans) |*p, *logit, span| {
            const prob = 1.0 / (1.0 + @exp(-logit.*));
            logit.* = if (std.math.isNan(prob)) 0 else prob;
            p.* = if (logit.* >= aw.presence_threshold) .{ @as(f64, @floatFromInt(span[0])) / 1000.0, @as(f64, @floatFromInt(span[1])) / 1000.0 } else null;
        }
    }
    // the clip's length as the aligner hears it: at 16 kHz
    const duration = @as(f64, @floatFromInt((audio.len * 160 + 440) / 441)) / 16000.0;
    const found = try aw.assemble(a, spoken, placed, presence, duration);
    var out: std.ArrayList(ep.Word) = .empty;
    for (spoken, found) |token, w| {
        const edit_tokens = ep.tokens(a, token) catch |e| return tokenFault(cx, e);
        if (edit_tokens.len == 0) continue;
        try out.append(a, .{ .word = token, .start = w.start, .end = w.end, .confidence = w.presence, .tokens = edit_tokens });
    }
    for (out.items[0..out.items.len -| 1], out.items[@min(1, out.items.len)..]) |*previous, *word| {
        if (word.start < previous.end) {
            const middle = (previous.end + word.start) / 2;
            previous.end = middle;
            word.start = middle;
        }
    }
    return out.items;
}

fn wordList(a: Allocator, list: []const ep.Word) !Value {
    const out = try a.alloc(Value, list.len);
    for (out, list) |*slot, w| {
        const entry = try object(a);
        try entry.put(a, "word", .{ .string = w.word });
        try entry.put(a, "start", .{ .float = w.start });
        try entry.put(a, "end", .{ .float = w.end });
        try entry.put(a, "confidence", if (w.confidence) |x| .{ .float = x } else .null);
        const toks = try a.alloc(Value, w.tokens.len);
        for (toks, w.tokens) |*t, s| t.* = .{ .string = s };
        try entry.put(a, "tokens", .{ .array = toks });
        slot.* = .{ .object = entry };
    }
    return .{ .array = out };
}

/// /v2/edit/align: the text's words in the clip.
fn alignOnly(srv: *Server, c: *Conn, cx: *Cx, st: *const zonos.State, raw: Value) errors.Refused!?Value {
    const a = cx.a;
    if (raw != .object) return cx.refuse("align request body must be a JSON object");
    for (raw.object.keys()) |key| {
        if (!std.mem.eql(u8, key, "audio_base64") and !std.mem.eql(u8, key, "text") and !std.mem.eql(u8, key, "language") and !std.mem.eql(u8, key, "model"))
            return cx.fail(.request, prefix ++ "unknown field {s}", .{try @import("pyrepr.zig").repr(a, .{ .string = key })});
    }
    _ = try editLanguage(cx, raw);
    const transcript = std.mem.trim(u8, raw.strField("text") orelse return cx.refuse(prefix ++ "text is required"), " \t\r\n");
    if (transcript.len == 0) return cx.refuse(prefix ++ "text is empty");
    const audio = try zonos.clipAudio(cx, raw.field("audio_base64") orelse return cx.refuse(prefix ++ "audio_base64 is required"), "audio_base64", false);
    try hold(srv, cx);
    defer _ = srv.speech_inflight.fetchSub(1, .acq_rel);
    const found = try alignText(srv, c, cx, st, audio, transcript) orelse return null;
    const o = try object(a);
    try o.put(a, "text", .{ .string = transcript });
    try o.put(a, "duration", .{ .float = @as(f64, @floatFromInt(audio.len)) / 44100.0 });
    try o.put(a, "words", try wordList(a, found));
    return .{ .object = o };
}

const Take = struct { fills: [][]const [C]i32, stopped: []bool, pcm: []f32 = &.{}, seed: u64 };

pub fn post(srv: *Server, c: *Conn, a: Allocator, align_only: bool) !void {
    const st = srv.zonos orelse {
        _ = try http_body.read(c, a, http_body.limit);
        return @import("routes.zig").unknown(c, a);
    };
    const sp = srv.info.speech.?;
    var cx: Cx = .{ .a = a };
    const id = try ids.make(a, "edit-", 32);
    const received = clock.nowNs(srv.io);
    const raw = http_body.readJson(c, &cx) catch |e| return refuse(c, &cx, id, e);
    const done = (if (align_only) alignOnly(srv, c, &cx, st, raw) else run(srv, c, &cx, st, sp, raw)) catch |e| return refuse(c, &cx, id, e);
    const body = done orelse return streaming.left(srv, c, id, 0, received);
    const text = try json.stringify(a, body, .{});
    c.startResponse(200, null) catch return;
    c.addHeader("Content-Type", "application/json") catch return;
    var len: [24]u8 = undefined;
    c.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{text.len}) catch unreachable) catch return;
    if (c.close) c.addHeader("Connection", "close") catch return;
    c.finish(text) catch {
        c.close = true;
    };
    log.line("edit {s} bytes={d} seconds={d:.2}", .{ id, text.len, clock.seconds(clock.nowNs(srv.io) - received) });
}

fn refuse(c: *Conn, cx: *Cx, id: []const u8, e: errors.Refused) void {
    if (e == error.OutOfMemory) {
        cx.kind = .server;
        cx.message = "out of memory";
    }
    speech.fail(c, cx, id, .{ .enabled = false }, cx.kind, cx.message);
}

/// The whole edit; null when the client left.
fn run(srv: *Server, c: *Conn, cx: *Cx, st: *const zonos.State, sp: api.SpeechInfo, raw: Value) errors.Refused!?Value {
    const a = cx.a;
    if (raw != .object) return cx.refuse("edit request body must be a JSON object");
    for (raw.object.keys()) |key| if (!known(key)) return cx.fail(.request, prefix ++ "unknown field {s}", .{try @import("pyrepr.zig").repr(a, .{ .string = key })});
    if (!st.editing) return cx.refuse(prefix ++ "this release folder has no speech-edit adapter (lm/zonos3_mini/edit_lora): speech editing is not served");
    const target = std.mem.trim(u8, raw.strField("target_text") orelse return cx.refuse(prefix ++ "target_text is required"), " \t\r\n");
    if (target.len == 0) return cx.refuse(prefix ++ "target_text is empty");
    for ([_][]const u8{ tokens.start_marker, "\xee\x80\x83", "\xee\x80\x84", "\xee\x80\x85", "\xee\x80\x86", "\xee\x80\x87", "\xee\x80\x88", "\xee\x80\x89", "\xee\x80\x8a" }) |marker| {
        if (std.mem.indexOf(u8, target, marker) != null) return cx.refuse(prefix ++ "target_text takes no start or pause markers: remove U+E002 and U+E003 to U+E00A");
    }
    if (raw.field("verify")) |v| if (v.truthy()) return cx.refuse(prefix ++ "verify needs speech recognition, which is not served: set verify to false");
    if (raw.field("decoder")) |v| if (!std.mem.eql(u8, v.str() orelse "", req.release.default_decoder)) return cx.refuse(prefix ++ "the decoder served is " ++ req.release.default_decoder);
    const sampling = try speech_sampling.parseFrom(cx, raw, .{ .temperature = zonos.temperature, .top_p = zonos.top_p, .top_k = zonos.top_k }, zonos.max_top_k);
    const takes_n: usize = @intFromFloat(try ranged(cx, raw, "num_takes", 1, 1, 4, false));
    const lang = try editLanguage(cx, raw);
    const normalize: ?bool = if (raw.field("text_normalization")) |v| (if (v == .bool) v.bool else return cx.refuse(prefix ++ "text_normalization must be a boolean")) else null;
    if (normalize == true and !textnorm.available(a)) return cx.refuse(textnorm.missing());
    const buckets = try zonos.conditions(st, cx, raw);
    const audio = try zonos.clipAudio(cx, raw.field("audio_base64") orelse return cx.refuse(prefix ++ "audio_base64 is required"), "audio_base64", false);
    const duration = @as(f64, @floatFromInt(audio.len)) / 44100.0;
    const clip_frames = (audio.len + 2939) / 2940;
    if (clip_frames > req.reference_frames) return cx.fail(.request, prefix ++ "the clip is {d:.1} s long; at most 60 s are edited here", .{duration});

    // the speaker row: the clip's embedding from the speaker encoder, or one the request brings
    var embedding: ?[req.config.speaker_dim]f32 = null;
    var encode_voice = true;
    if (raw.field("speaker_embedding")) |v| switch (v) {
        .bool => |on| encode_voice = on,
        .array => |list| {
            const message = prefix ++ "speaker_embedding must be a boolean or 192 finite numbers";
            if (list.len != req.config.speaker_dim) return cx.refuse(message);
            var sent: [req.config.speaker_dim]f32 = undefined;
            for (&sent, list) |*x, n| {
                x.* = @floatCast(n.float64() orelse return cx.refuse(message));
                if (!std.math.isFinite(x.*)) return cx.refuse(message);
            }
            embedding = sent;
            encode_voice = false;
        },
        else => return cx.refuse(prefix ++ "speaker_embedding must be a boolean or 192 finite numbers"),
    };

    try hold(srv, cx);
    defer _ = srv.speech_inflight.fetchSub(1, .acq_rel);
    if (srv.keepalive) |k| k.begin();
    defer if (srv.keepalive) |k| k.end();

    // the clip's codes, and its voice
    var result: api.SpeechResult = .{};
    var out: api.SpeechEdit.Out = .{};
    const clip = try a.alloc([C]i32, clip_frames);
    if (encode_voice) embedding = (st.embed(audio) catch return cx.other("the speaker encoder failed on this clip")) orelse
        return cx.refuse(prefix ++ "the clip's speaker row needs the release's speaker encoder (speaker_encoders/), which this folder lacks: send speaker_embedding false, or 192 numbers");
    const encode: api.SpeechEdit = .{ .op = .encode, .codes = clip, .out = &out };
    if (!try step(srv, c, cx, &.{ .prompt = &.{}, .pcm = &.{}, .result = &result, .seed = 0, .reference_audio = audio, .edit = &encode })) return null;
    const codes = clip[0..out.frames];

    var request: ep.Request = .{ .target_text = target, .frames = codes.len, .max_spans = req.edit.marker_rows };
    request.margin_seconds = try ranged(cx, raw, "margin_seconds", 0.08, 0, 1, false);
    request.max_span_seconds = try ranged(cx, raw, "max_span_seconds", 30, 0, 120, true);
    request.max_context_seconds = try ranged(cx, raw, "max_context_seconds", 16, 0, 20, true);
    request.context_seconds = try ranged(cx, raw, "context_seconds", 4, 0, 10, false);
    request.min_fill_ratio = try ranged(cx, raw, "min_fill_ratio", 0.7, 0, 1, false);
    const original_text = raw.strField("original_text");
    if (raw.field("original_words")) |v| {
        request.original = try words(cx, v, original_text orelse return cx.refuse(prefix ++ "original_words needs the original_text they spell"), duration);
    }
    // the clip's words, when the edit reads them and the request brings only their text: the aligner places them
    if (request.original == null and (raw.field("spans") == null or request.windowed() or (request.min_fill_ratio > 0 and original_text != null))) {
        const said = original_text orelse return cx.refuse(prefix ++ "this edit reads the clip's words: send the clip's transcript as original_text (no speech recognition is served)");
        request.original = try alignText(srv, c, cx, st, audio, said) orelse return null;
    }
    if (raw.field("spans")) |v| {
        const message = prefix ++ "spans must be a list of [start, end] in seconds";
        const list = v.items() orelse return cx.refuse(message);
        const sent = try a.alloc([2]f64, list.len);
        for (sent, list) |*p, entry| {
            const two = entry.items() orelse return cx.refuse(message);
            if (two.len != 2) return cx.refuse(message);
            p.* = .{ two[0].float64() orelse return cx.refuse(message), two[1].float64() orelse return cx.refuse(message) };
        }
        request.spans = sent;
    }
    if (raw.field("region_controls")) |v| {
        const list = v.items() orelse return cx.refuse(prefix ++ "region_controls must be a list, one entry a span (null for none)");
        if (request.spans == null) return cx.refuse(prefix ++ "region_controls needs explicit spans (one entry per span)");
        if (list.len != request.spans.?.len) return cx.fail(.request, prefix ++ "region_controls has {d} entries for {d} spans; send one per span (null for none)", .{ list.len, request.spans.?.len });
    }
    const plan = ep.plan(a, request) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.EditNeedsOriginalWords => cx.refuse(prefix ++ "this edit reads the clip's words: send the clip's transcript as original_text"),
        error.BadEditSpans => cx.fail(.request, prefix ++ "edit spans must be one to {d}, ascending, separated and within the audio", .{req.edit.marker_rows}),
        error.EditChangesNothing => cx.refuse(prefix ++ "target text equals the original transcript"),
        error.EditTouchesTooManyRegions => cx.fail(.request, prefix ++ "the edit touches more than {d} regions", .{req.edit.marker_rows}),
        error.NoWordsToWindowBy => cx.refuse(prefix ++ "original_words is empty, so the edit cannot be windowed"),
        else => tokenFault(cx, e),
    };

    // every region's controls, resolved before anything is generated
    const chars = std.unicode.utf8CountCodepoints(target) catch return cx.refuse(prefix ++ "target_text must be valid UTF-8");
    const steered = try a.alloc(steer.Made, plan.spans.len);
    for (steered, 0..) |*made, k| made.* = try regionSteering(st, cx, raw, if (request.spans != null) k else null, chars, if (embedding) |*e| e else null);

    const marker: i32 = @intCast(if (st.tokens) |spec| spec.pause_ids[spec.pause_ids.len - 1] + 1 else req.config.text_vocab + 1);
    const rows_cap = @min(plan.max_rows, sp.max_frames);
    const fill_rows = try a.alloc([W]i32, rows_cap);
    const takes = try a.alloc(Take, takes_n);
    // unseeded, an edit's takes follow from its clip and target as a spoken reply's follow from its prompt
    const base_seed = sampling.seedFor(&.{ @intCast(codes.len), @intCast(target.len), std.hash.Crc32.hash(target) }, srv.config.seed_salt, 0);
    const pcm_len = sp.max_frames * sp.frame_length;
    for (takes, 0..) |*take, k| {
        take.* = .{ .fills = try a.alloc([]const [C]i32, plan.spans.len), .stopped = try a.alloc(bool, plan.spans.len), .seed = base_seed + k };
        for (plan.windows) |window| {
            const typed = req.prompt.collapseLineBreaks(a, window.text) catch return cx.refuse(prefix ++ "target_text must be valid UTF-8");
            const spoken = if (normalize orelse textnorm.available(a)) textnorm.normalize(a, typed, lang) catch |e| return switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                error.NoLibrary => cx.refuse(textnorm.missing()),
            } else typed;
            const text_ids = try tokens.payload(cx, srv.text, st.tokens, spoken);
            const spans = try a.alloc(req.edit.Span, window.regions.len);
            for (spans, window.regions) |*s, region| s.* = .{ .start = plan.spans[region].start - window.start, .len = plan.spans[region].len };
            var priors: std.ArrayList([]const [C]i32) = .empty;
            for (window.regions) |region| {
                const made = steered[region];
                const built = req.prompt.build(a, text_ids, if (embedding != null) .embedding else .none, st.silence, 0) catch return cx.refuse(prefix ++ "the target text has nothing to speak");
                const tail = req.edit.layout(a, codes[window.start..window.end], spans, priors.items, marker) catch return cx.other("the edit prompt could not be laid out");
                const prompt = try std.mem.concat(a, [W]i32, &.{ built.rows, tail });
                if (prompt.len > sp.max_prompt_tokens) return cx.fail(.request, prefix ++ "a fill prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ prompt.len, sp.max_prompt_tokens });
                const moved: ?[req.config.speaker_dim]f32 = made.embedding orelse embedding;
                out = .{};
                const fill: api.SpeechEdit = .{ .op = .fill, .rows = fill_rows, .min_frames = @intCast(plan.min_frames[region]), .out = &out };
                const ask: api.SpeechRequest = .{ .prompt = text_ids, .frames = prompt, .pcm = &.{}, .result = &result, .seed = take.seed, .max_frames = @intCast(rows_cap), .temperature = sampling.temperature, .top_p = sampling.top_p, .top_k = sampling.top_k, .speaker = if (moved) |*m| m else &.{}, .conditions = buckets, .steering = made.steering, .edit = &fill };
                if (!try step(srv, c, cx, &ask)) return null;
                const got = req.edit.extract(a, fill_rows[0..out.rows], if (out.end_frame >= 0) @intCast(out.end_frame) else null) catch return error.OutOfMemory;
                take.fills[region] = got.frames;
                take.stopped[region] = got.stopped;
                try priors.append(a, got.frames);
            }
        }
        const spliced = req.edit.splice(a, codes, plan.spans, take.fills) catch return cx.other("the fills could not be spliced");
        if (spliced.len == 0 or spliced.len > sp.max_frames) return cx.fail(.request, prefix ++ "the edited clip is {d} frames long; at most {d} are decoded here", .{ spliced.len, sp.max_frames });
        const pcm = try srv.gpa.alloc(f32, pcm_len);
        errdefer srv.gpa.free(pcm);
        const decode: api.SpeechEdit = .{ .op = .decode, .codes = @constCast(spliced), .out = &out };
        result = .{};
        if (!try step(srv, c, cx, &.{ .prompt = &.{}, .pcm = pcm, .result = &result, .seed = 0, .edit = &decode })) {
            srv.gpa.free(pcm);
            return null;
        }
        take.pcm = try a.dupe(f32, pcm[0..result.samples]);
        srv.gpa.free(pcm);
    }
    return try report(a, sp, plan, original_text, target, request, takes);
}

/// The release's /v2/edit response: the regions, the windows, the expected fill lengths and each take's audio.
fn report(a: Allocator, sp: api.SpeechInfo, plan: ep.Plan, original_text: ?[]const u8, target: []const u8, request: ep.Request, takes: []Take) !Value {
    const o = try object(a);
    try o.put(a, "sample_rate", .{ .int = try std.fmt.allocPrint(a, "{d}", .{sp.sample_rate}) });
    try o.put(a, "original_text", if (original_text) |t| .{ .string = t } else .null);
    try o.put(a, "target_text", .{ .string = target });
    if (request.original) |original| {
        try o.put(a, "words", try wordList(a, original));
    } else try o.put(a, "words", .null);
    const spans = try a.alloc([2]f64, plan.spans.len);
    for (spans, plan.spans) |*p, s| p.* = .{ @as(f64, @floatFromInt(s.start)) / rate, @as(f64, @floatFromInt(s.start + s.len)) / rate };
    try o.put(a, "spans", try pairs(a, spans));
    if (plan.windowed) {
        const contexts = try a.alloc([2]f64, plan.windows.len);
        for (contexts, plan.windows) |*p, w| p.* = .{ @as(f64, @floatFromInt(w.start)) / rate, @as(f64, @floatFromInt(w.end)) / rate };
        try o.put(a, "contexts", try pairs(a, contexts));
    } else try o.put(a, "contexts", .null);
    try o.put(a, "expected_seconds", try floats(a, plan.expected));
    const list = try a.alloc(Value, takes.len);
    for (list, takes) |*slot, take| {
        const entry = try object(a);
        try entry.put(a, "seed", .{ .int = try std.fmt.allocPrint(a, "{d}", .{take.seed}) });
        const samples = take.pcm.len;
        const data = speech.quantizeInPlace(take.pcm) catch return error.OutOfMemory;
        const wav = try a.alloc(u8, speech.wav_header_len + data.len);
        @memcpy(wav[0..speech.wav_header_len], &speech.wavHeader(sp.sample_rate, @intCast(data.len)));
        @memcpy(wav[speech.wav_header_len..], data);
        const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(wav.len));
        try entry.put(a, "audio_base64", .{ .string = std.base64.standard.Encoder.encode(encoded, wav) });
        try entry.put(a, "duration", .{ .float = @as(f64, @floatFromInt(samples)) / @as(f64, @floatFromInt(sp.sample_rate)) });
        const regions = try a.alloc([2]f64, plan.spans.len);
        const seconds = try a.alloc(f64, plan.spans.len);
        const stopped = try a.alloc(Value, plan.spans.len);
        var shift: f64 = 0;
        for (regions, seconds, stopped, plan.spans, take.fills, take.stopped) |*p, *s, *flag, span, fill, ended| {
            const start = @as(f64, @floatFromInt(span.start)) + shift;
            p.* = .{ start / rate, (start + @as(f64, @floatFromInt(fill.len))) / rate };
            s.* = @as(f64, @floatFromInt(fill.len)) / rate;
            flag.* = .{ .bool = ended };
            shift += @as(f64, @floatFromInt(fill.len)) - @as(f64, @floatFromInt(span.len));
        }
        try entry.put(a, "edited_regions", try pairs(a, regions));
        try entry.put(a, "fill_seconds", try floats(a, seconds));
        try entry.put(a, "stopped", .{ .array = stopped });
        try entry.put(a, "verify", .null);
        try entry.put(a, "words", .null);
        slot.* = .{ .object = entry };
    }
    try o.put(a, "takes", .{ .array = list });
    return .{ .object = o };
}
