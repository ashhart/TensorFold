//! POST /v1/audio/speech: synthesized whole and sent as WAV or raw PCM; anything but complete audio is an error.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const openai = @import("openai.zig");
const pyrepr = @import("pyrepr.zig");
const reply_text = @import("reply_text.zig");
const drama_prompt = @import("drama_prompt.zig");
const ids = @import("ids.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// ``MAX_SPEECH_INPUT_CHARS``: the most characters (code points) one input may hold.
pub const max_input_chars = 4096;

/// The request fields served: the OpenAI ones (``speaker`` is ``voice``'s alias).
const served_fields = [_][]const u8{ "model", "input", "voice", "speaker", "response_format" };

/// The rest of the speech protocol's fields: synthesis modifiers, references and streaming this route does not honor.
const modifier_fields = [_][]const u8{ "speed", "stream", "stream_format", "task_type", "language", "instructions", "ref_audio", "ref_text", "references", "x_vector_only_mode", "stream_codec_output", "suppress_bootstrap_silence", "token_count", "duration_tokens", "initial_codec_chunk_frames", "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "seed", "stage_params" };

pub const Format = enum {
    wav,
    pcm,

    fn contentType(f: Format) []const u8 {
        return switch (f) {
            .wav => "audio/wav",
            .pcm => "audio/pcm",
        };
    }
};

const Plan = struct { prompt: []const u32, seed: u64, format: Format };

pub fn post(srv: *Server, c: *Conn, a: Allocator) !void {
    const sp = srv.info.speech orelse {
        _ = try http_body.read(c, a, http_body.limit); // dropped, so it cannot reach the next request on this connection
        return routes.unknown(c, a);
    };
    var cx: Cx = .{ .a = a };
    const id = try ids.make(a, "speech-", 32);
    const received = clock.nowNs(srv.io);
    const raw = http_body.readJson(c, &cx) catch |e| return refuse(c, &cx, id, e);
    const plan = prepare(srv, &cx, sp, raw) catch |e| return refuse(c, &cx, id, e);
    synthesize(srv, c, &cx, id, sp, plan, received);
}

fn listed(names: []const []const u8, key: []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, name, key)) return true;
    return false;
}

/// Refuses what the route would drop: unknown fields even when null, set modifiers, non-string served fields.
fn checkFields(cx: *Cx, body: Value) errors.Refused!void {
    for (body.object.keys(), body.object.values()) |key, value| {
        if (listed(&served_fields, key)) {
            if (value != .string) return cx.fail(.request, "{s} must be a string", .{key});
        } else if (listed(&modifier_fields, key)) {
            if (value != .null) return cx.fail(.request, "{s} is not supported by this model", .{key});
        } else return cx.fail(.request, "unknown speech request field: {s}", .{key});
    }
}

/// ``response_format.strip().lower()``, which must name WAV or raw PCM.
fn responseFormat(cx: *Cx, value: []const u8) errors.Refused!Format {
    const name = reply_text.pyStrip(value);
    if (std.ascii.eqlIgnoreCase(name, "wav")) return .wav;
    if (std.ascii.eqlIgnoreCase(name, "pcm")) return .pcm;
    return cx.refuse("response_format must be one of: pcm, wav");
}

/// Every check that can refuse the request, then its prompt ids, before anything reaches the engine.
fn prepare(srv: *Server, cx: *Cx, sp: api.SpeechInfo, raw: Value) errors.Refused!Plan {
    if (raw != .object) return cx.refuse("speech request body must be a JSON object");
    try checkFields(cx, raw);
    const model = raw.strField("model") orelse return cx.fail(.request, "model is required: send {s}", .{srv.config.served_name});
    if (!listed(srv.config.model_ids, model))
        return cx.fail(.request, "model {s} is not served here: send {s}", .{ try pyrepr.repr(cx.a, .{ .string = model }), srv.config.served_name });
    const input = raw.strField("input") orelse return cx.refuse("input must be a non-empty string");
    if (!std.unicode.utf8ValidateSlice(input)) return cx.refuse("input must be valid UTF-8 text");
    if (reply_text.pyStrip(input).len == 0) return cx.refuse("input must be a non-empty string");
    if ((std.unicode.utf8CountCodepoints(input) catch unreachable) > max_input_chars)
        return cx.fail(.request, "input must be at most {d} characters", .{max_input_chars});
    const format = try responseFormat(cx, raw.strField("response_format") orelse "wav");
    const voice = raw.strField("voice") orelse raw.strField("speaker") orelse "default";
    if (voice.len > 0 and !std.ascii.eqlIgnoreCase(voice, "default"))
        return cx.fail(.request, drama_prompt.error_prefix ++ " unknown voice {s}; only the default voice is supported", .{try pyrepr.repr(cx.a, .{ .string = voice })});
    if (try drama_prompt.problem(cx.a, input)) |message| return cx.refuse(message);
    const prompt = drama_prompt.encode(cx.a, srv.text, input) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => cx.other("the tokenizer cannot encode this prompt"),
    };
    if (prompt.len > sp.max_prompt_tokens)
        return cx.fail(.request, drama_prompt.error_prefix ++ " a prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ prompt.len, sp.max_prompt_tokens });
    return .{ .prompt = prompt, .seed = api.seedFor(prompt, srv.config.seed_salt), .format = format };
}

/// One synthesis: submit, wait for the engine's finished event, then send the whole audio or the reason there is none.
fn synthesize(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, sp: api.SpeechInfo, plan: Plan, received: i96) void {
    const pcm_len = std.math.mul(usize, sp.max_frames, sp.frame_length) catch 0;
    if (pcm_len == 0 or sp.sample_rate == 0 or sp.sample_rate > std.math.maxInt(u32) / 2)
        return fail(c, cx, id, .server, "the engine advertises no usable speech output");
    const pcm = srv.gpa.alloc(f32, pcm_len) catch return fail(c, cx, id, .server, "out of memory");
    defer srv.gpa.free(pcm);
    var result: api.SpeechResult = .{};
    const request: api.SpeechRequest = .{
        .prompt = plan.prompt,
        .pcm = pcm,
        .result = &result,
        .seed = plan.seed,
        .max_frames = sp.max_frames,
        .temperature = drama_prompt.temperature,
        .top_p = drama_prompt.top_p,
        .top_k = drama_prompt.top_k,
    };
    var box: Mailbox = .{ .io = srv.io, .gpa = srv.gpa };
    defer box.deinit();
    const engine_id = srv.next_id.fetchAdd(1, .monotonic);
    if (srv.keepalive) |k| k.begin();
    defer if (srv.keepalive) |k| k.end();
    srv.engine.submitSpeech(engine_id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => fail(c, cx, id, .capacity, "the engine is busy; retry shortly"),
        error.Closed => fail(c, cx, id, .other, "the scheduler is closed"),
        error.InvalidRequest => fail(c, cx, id, .request, "the engine refused this speech request's prompt, frame limit or sampling"),
        error.Unsupported => fail(c, cx, id, .request, "this model does not support speech synthesis"),
    };
    // The engine holds request, prompt, pcm and result until finished; no path below returns before that.
    if (!settle(srv, c, &box, engine_id)) {
        c.close = true;
        srv.metrics.disconnected(srv.io);
        var buf: [1024]u8 = undefined;
        log.line("{s}", .{log.ended(&buf, id, null, plan.prompt.len, 0, clock.seconds(clock.nowNs(srv.io) - received))});
        return;
    }
    const samples = result.samples;
    const why: ?[]const u8 = switch (box.reason) {
        .stop => if (samples == 0 or samples > pcm.len) "the engine finished without complete audio" else null,
        .length => "speech reached the frame limit without ending; truncated audio is not returned",
        .cancelled => "speech synthesis was cancelled",
        .failed => if (box.message.len > 0) box.message else "speech synthesis failed",
    };
    if (why) |message| return fail(c, cx, id, .server, message);
    const data = quantizeInPlace(pcm[0..samples]) catch return fail(c, cx, id, .server, "the synthesized audio has non-finite samples");
    if (data.len > std.math.maxInt(u32) - 36) return fail(c, cx, id, .server, "the synthesized audio is too long for one reply");
    sendAudio(c, plan.format, sp.sample_rate, data) catch {
        c.close = true; // a head already started or a client gone: nothing more can be sent on this connection
        return;
    };
    log.line("speech {s} prompt={d} samples={d} format={s} seconds={d:.2}", .{ id, plan.prompt.len, samples, @tagName(plan.format), clock.seconds(clock.nowNs(srv.io) - received) });
}

/// Waits for the engine's finished event; a client that leaves cancels the synthesis first. False: the client left.
fn settle(srv: *Server, c: *Conn, box: *Mailbox, engine_id: api.Id) bool {
    while (true) {
        box.mutex.lockUncancelable(box.io);
        if (!box.finished) box.cond.waitTimeout(box.io, &box.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
        const done = box.finished;
        box.mutex.unlock(box.io);
        if (done) return true;
        if (c.peerGone()) {
            srv.engine.cancel(engine_id);
            box.mutex.lockUncancelable(box.io);
            defer box.mutex.unlock(box.io);
            while (!box.finished) box.cond.waitUncancelable(box.io, &box.mutex);
            return false;
        }
    }
}

fn refuse(c: *Conn, cx: *Cx, id: []const u8, e: errors.Refused) void {
    if (e == error.OutOfMemory) {
        cx.kind = .server;
        cx.message = "out of memory";
    }
    send(c, cx, id);
}

fn fail(c: *Conn, cx: *Cx, id: []const u8, kind: errors.Kind, message: []const u8) void {
    cx.kind = kind;
    cx.message = message;
    send(c, cx, id);
}

/// OpenAI's error object, logged first: the access line shows only the status.
fn send(c: *Conn, cx: *Cx, id: []const u8) void {
    var buf: [1400]u8 = undefined;
    log.line("{s}", .{log.refused(&buf, id, cx.message)});
    const body = openai.errorBody(cx.a, cx, null) catch return;
    routes.sendValue(c, cx.a, cx.status(), openai.wrapError(cx.a, body) catch return);
}

/// The head, then the WAV header and the samples where they lie: the audio is never copied to prepend a header.
fn sendAudio(c: *Conn, format: Format, sample_rate: u32, data: []const u8) error{ Closed, OutOfMemory }!void {
    const header: ?[wav_header_len]u8 = if (format == .wav) wavHeader(sample_rate, @intCast(data.len)) else null;
    const header_len: usize = if (header != null) wav_header_len else 0;
    try c.startResponse(200, null);
    try c.addHeader("Content-Type", format.contentType());
    var len: [24]u8 = undefined;
    try c.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{header_len + data.len}) catch unreachable);
    if (c.close) try c.addHeader("Connection", "close");
    defer c.out.clearRetainingCapacity();
    if (!std.mem.eql(u8, c.version, "HTTP/0.9")) try c.out.appendSlice(c.gpa, "\r\n");
    if (header) |h| try c.out.appendSlice(c.gpa, &h);
    try c.writeAll(c.out.items);
    try c.writeAll(data);
}

pub const wav_header_len = 44;

/// A canonical 44-byte RIFF/WAVE header for mono 16-bit PCM of ``data_bytes`` bytes.
pub fn wavHeader(sample_rate: u32, data_bytes: u32) [wav_header_len]u8 {
    var h: [wav_header_len]u8 = undefined;
    @memcpy(h[0..4], "RIFF");
    std.mem.writeInt(u32, h[4..8], 36 + data_bytes, .little);
    @memcpy(h[8..12], "WAVE");
    @memcpy(h[12..16], "fmt ");
    std.mem.writeInt(u32, h[16..20], 16, .little); // fmt chunk size
    std.mem.writeInt(u16, h[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, h[22..24], 1, .little); // mono
    std.mem.writeInt(u32, h[24..28], sample_rate, .little);
    std.mem.writeInt(u32, h[28..32], sample_rate * 2, .little); // byte rate
    std.mem.writeInt(u16, h[32..34], 2, .little); // block align
    std.mem.writeInt(u16, h[34..36], 16, .little); // bits a sample
    @memcpy(h[36..40], "data");
    std.mem.writeInt(u32, h[40..44], data_bytes, .little);
    return h;
}

/// The reference's (clip(x, -1, 1) * 32767).astype(int16) as PCM16LE in place; each sample is read before written.
pub fn quantizeInPlace(pcm: []f32) error{NonFinite}![]u8 {
    const bytes = std.mem.sliceAsBytes(pcm);
    for (0..pcm.len) |i| {
        const x = pcm[i];
        if (!std.math.isFinite(x)) return error.NonFinite;
        const q: i16 = @intFromFloat(std.math.clamp(x, -1.0, 1.0) * 32767.0);
        std.mem.writeInt(i16, bytes[2 * i ..][0..2], q, .little);
    }
    return bytes[0 .. 2 * pcm.len];
}

test "quantization clips, truncates toward zero and stays in place" {
    var pcm = [_]f32{ 0, 1, -1, 1.5, -7, 0.5, -0.5, 0.25, 1e-6, -1e-6 };
    const want = [_]i16{ 0, 32767, -32767, 32767, -32767, 16383, -16383, 8191, 0, 0 };
    const bytes = try quantizeInPlace(&pcm);
    try std.testing.expectEqual(@as(usize, 2 * want.len), bytes.len);
    try std.testing.expectEqual(@intFromPtr(&pcm), @intFromPtr(bytes.ptr));
    for (want, 0..) |w, i| try std.testing.expectEqual(w, std.mem.readInt(i16, bytes[2 * i ..][0..2], .little));
}

test "non-finite audio is refused" {
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        var pcm = [_]f32{ 0.1, bad, 0.2 };
        try std.testing.expectError(error.NonFinite, quantizeInPlace(&pcm));
    }
}

test "the WAV header describes mono 16-bit PCM at the advertised rate" {
    const h = wavHeader(44100, 4096);
    try std.testing.expectEqualStrings("RIFF", h[0..4]);
    try std.testing.expectEqual(@as(u32, 36 + 4096), std.mem.readInt(u32, h[4..8], .little));
    try std.testing.expectEqualStrings("WAVEfmt ", h[8..16]);
    try std.testing.expectEqual(@as(u32, 16), std.mem.readInt(u32, h[16..20], .little));
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, h[20..22], .little));
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, h[22..24], .little));
    try std.testing.expectEqual(@as(u32, 44100), std.mem.readInt(u32, h[24..28], .little));
    try std.testing.expectEqual(@as(u32, 88200), std.mem.readInt(u32, h[28..32], .little));
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, h[32..34], .little));
    try std.testing.expectEqual(@as(u16, 16), std.mem.readInt(u16, h[34..36], .little));
    try std.testing.expectEqualStrings("data", h[36..40]);
    try std.testing.expectEqual(@as(u32, 4096), std.mem.readInt(u32, h[40..44], .little));
}

test "fields: unknown ones are refused even when null, modifiers when set" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"voice\": \"default\", \"response_format\": \"wav\", \"speed\": null, \"references\": null}", "" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream_audio\": null}", "unknown speech request field: stream_audio" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"speed\": 1.0}", "speed is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream\": false}", "stream is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"temperature\": 0.7}", "temperature is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"ref_audio\": \"x.wav\"}", "ref_audio is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": 3}", "input must be a string" },
        .{ "{\"model\": null, \"input\": \"hi\"}", "model must be a string" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"voice\": null}", "voice must be a string" },
    };
    for (cases) |case| {
        var cx: Cx = .{ .a = a };
        const body = (try json.parse(a, case[0])).ok;
        if (case[1].len == 0) {
            try checkFields(&cx, body);
        } else {
            try std.testing.expectError(error.Refused, checkFields(&cx, body));
            try std.testing.expectEqualStrings(case[1], cx.message);
        }
    }
}

test "response_format is WAV or PCM, stripped and case-folded as Python does" {
    var cx: Cx = .{ .a = std.testing.allocator };
    try std.testing.expectEqual(Format.wav, try responseFormat(&cx, " WAV\n"));
    try std.testing.expectEqual(Format.pcm, try responseFormat(&cx, "\u{3000}Pcm\x1c"));
    for ([_][]const u8{ "mp3", "", "wave", "pcm16" }) |bad| try std.testing.expectError(error.Refused, responseFormat(&cx, bad));
}
