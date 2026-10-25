//! POST /v1/audio/speech: WAV or raw PCM whole, or raw PCM chunks with stream; only complete synthesis is audio.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const http_body = @import("http_body.zig");
const routes = @import("routes.zig");
const openai = @import("openai.zig");
const pyrepr = @import("pyrepr.zig");
const reply_text = @import("reply_text.zig");
const drama_prompt = @import("drama_prompt.zig");
const speech_redraw = @import("speech_redraw.zig");
const speech_register = @import("speech_register_log.zig");
const drama_cfg = @import("drama_cfg.zig");
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

/// The request fields served: the OpenAI ones (``speaker`` is ``voice``'s alias); ``stream*`` are checked apart.
const served_fields = [_][]const u8{ "model", "input", "voice", "speaker", "response_format" };

/// The rest of the speech protocol's fields: synthesis modifiers and references this route does not honor.
const modifier_fields = [_][]const u8{ "speed", "task_type", "language", "instructions", "ref_audio", "ref_text", "references", "x_vector_only_mode", "stream_codec_output", "suppress_bootstrap_silence", "token_count", "duration_tokens", "initial_codec_chunk_frames", "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "seed", "stage_params" };

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

const Plan = struct { input: []const u8, prompt: []const u32, frames: ?[]const [11]i32 = null, refs: []const @import("drama_prompt").Reference = &.{}, reference_rows: u32 = 0, seed: u64, format: Format, stream: bool, budget_s: ?f64 = null };

/// One whole take: its audio, why it ended, and whether it ended in a mid-utterance silence.
const Take = struct { pcm: []f32, reason: api.Reason, silence_mid: bool = false, trip: bool = false, jump: ?f64 = null };

const Metadata = @import("speech_metadata.zig").Metadata;

pub fn post(srv: *Server, c: *Conn, a: Allocator) !void {
    const sp = srv.info.speech orelse {
        // dropped, so it cannot reach the next request on this connection
        _ = try http_body.read(c, a, http_body.limit);
        return routes.unknown(c, a);
    };
    var cx: Cx = .{ .a = a };
    const id = try ids.make(a, "speech-", 32);
    const received = clock.nowNs(srv.io);
    var metadata: Metadata = .{ .enabled = srv.config.redraw.count() > 0, .register_on = srv.config.redraw.register.on() };
    const raw = http_body.readJson(c, &cx) catch |e| return refuse(c, &cx, id, metadata, e);
    var plan = prepare(srv, &cx, sp, raw) catch |e| return refuse(c, &cx, id, metadata, e);
    metadata.cfg_gamma = if (srv.config.cfg) |options| drama_cfg.gamma(options, c.header("X-Cfg-Gamma")) orelse
        return fail(c, &cx, id, metadata, .request, "X-Cfg-Gamma must be a number from 1 (unguided) to 5") else null;
    // only chunked framing tells a client that a stream which ends early is incomplete
    if (plan.stream and !std.mem.eql(u8, c.version, "HTTP/1.1")) return fail(c, &cx, id, metadata, .request, "stream=true requires HTTP/1.1");
    if (srv.config.redraw.count() > 0 or srv.config.redraw.register.on())
        plan.budget_s = speech_redraw.budget(srv.config.redraw.budget_s, c.header("X-Deadline-S")) catch
            return fail(c, &cx, id, metadata, .request, "X-Deadline-S must be a positive number of seconds");
    synthesize(srv, c, &cx, id, sp, plan, received, &metadata);
}

fn listed(names: []const []const u8, key: []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, name, key)) return true;
    return false;
}

/// Refuses what this route would drop: unknown fields, set modifiers, non-strings, a bad ``stream``/``stream_format``.
fn checkFields(cx: *Cx, body: Value) errors.Refused!void {
    for (body.object.keys(), body.object.values()) |key, value| {
        if (listed(&served_fields, key)) {
            if (value != .string) return cx.fail(.request, "{s} must be a string", .{key});
        } else if (std.mem.eql(u8, key, "stream")) {
            if (value != .bool and value != .null) return cx.refuse("stream must be a boolean");
        } else if (std.mem.eql(u8, key, "stream_format")) {
            if (value == .null or (value == .string and std.mem.eql(u8, value.string, "audio"))) continue;
            if (value == .string and std.mem.eql(u8, value.string, "sse")) return cx.refuse("stream_format sse is not supported by this model");
            return cx.refuse("stream_format must be one of: audio, sse");
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
    const input = try @import("drama_allow.zig").filter(cx.a, srv.config.allowed_controls, raw.strField("input") orelse return cx.refuse("input must be a non-empty string"));
    if (!std.unicode.utf8ValidateSlice(input)) return cx.refuse("input must be valid UTF-8 text");
    if (reply_text.pyStrip(input).len == 0) return cx.refuse("input must be a non-empty string");
    if ((std.unicode.utf8CountCodepoints(input) catch unreachable) > max_input_chars)
        return cx.fail(.request, "input must be at most {d} characters", .{max_input_chars});
    const format = try responseFormat(cx, raw.strField("response_format") orelse "wav");
    const stream = if (raw.get("stream")) |s| s == .bool and s.bool else false;
    if (stream and format != .pcm) return cx.refuse("stream=true requires response_format='pcm'");
    const voice = raw.strField("voice") orelse raw.strField("speaker") orelse "default";
    const refs = @import("drama_voice.zig").select(srv.voices, srv.config.default_voice, voice) catch {
        const name = try pyrepr.repr(cx.a, .{ .string = voice });
        if (srv.config.voices_dir != null)
            return cx.fail(.request, drama_prompt.error_prefix ++ " unknown voice {s}; see /v1/voices", .{name});
        return cx.fail(.request, drama_prompt.error_prefix ++ " unknown voice {s}; only the default voice is supported", .{name});
    };
    if (try @import("drama_voice.zig").requestPrompt(cx, srv.text, sp, input, refs)) |made| return .{ .input = input, .prompt = made.prompt, .frames = made.frames, .refs = refs, .reference_rows = made.reference_rows, .seed = api.seedFor(made.prompt, srv.config.seed_salt), .format = format, .stream = stream };
    const prompt = try promptOf(srv, cx, sp, input);
    return .{ .input = input, .prompt = prompt, .seed = api.seedFor(prompt, srv.config.seed_salt), .format = format, .stream = stream };
}

/// The prompt ids of a valid UTF-8 input, refused when it cannot be spoken or is too long.
fn promptOf(srv: *Server, cx: *Cx, sp: api.SpeechInfo, input: []const u8) errors.Refused![]const u32 {
    if (try drama_prompt.problem(cx.a, input)) |message| return cx.refuse(message);
    const prompt = drama_prompt.encode(cx.a, srv.text, input) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Template => cx.other("the tokenizer cannot encode this prompt"),
    };
    if (prompt.len > sp.max_prompt_tokens)
        return cx.fail(.request, drama_prompt.error_prefix ++ " a prompt of {d} rows exceeds the maximum allowed length of {d} rows", .{ prompt.len, sp.max_prompt_tokens });
    return prompt;
}

/// One synthesis: submit, then send the whole audio after the finished event, or stream it as the engine publishes it.
fn synthesize(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, sp: api.SpeechInfo, plan: Plan, received: i96, metadata: *Metadata) void {
    var held = srv.speech_inflight.load(.acquire);
    while (true) {
        if (held >= @max(1, srv.info.lanes)) return fail(c, cx, id, metadata.*, .capacity, "speech output capacity is full");
        if (srv.speech_inflight.cmpxchgWeak(held, held + 1, .acq_rel, .acquire)) |now| held = now else break;
    }
    defer _ = srv.speech_inflight.fetchSub(1, .acq_rel);
    const pcm_len = std.math.mul(usize, sp.max_frames, sp.frame_length) catch 0;
    if (pcm_len == 0 or sp.sample_rate == 0 or sp.sample_rate > std.math.maxInt(u32) / 2)
        return fail(c, cx, id, metadata.*, .server, "the engine advertises no usable speech output");
    const pcm = srv.gpa.alloc(f32, pcm_len) catch return fail(c, cx, id, metadata.*, .server, "out of memory");
    defer srv.gpa.free(pcm);
    var result: api.SpeechResult = .{};
    const frames = plan.frames orelse drama_prompt.rows(cx.a, plan.prompt) catch |e| return switch (e) {
        error.OutOfMemory => fail(c, cx, id, metadata.*, .server, "out of memory"),
        error.InvalidToken => fail(c, cx, id, metadata.*, .request, "invalid speech token"),
    };
    const guided = drama_cfg.guide(cx.a, srv.text, srv.config.cfg, metadata.cfg_gamma orelse 1, plan.input, plan.refs.len > 0) catch
        return fail(c, cx, id, metadata.*, .server, "the tokenizer cannot encode the unconditional prompt");
    const request: api.SpeechRequest = .{
        .prompt = plan.prompt,
        .frames = frames,
        .pcm = pcm,
        .result = &result,
        .seed = plan.seed,
        .max_frames = sp.max_frames,
        .temperature = drama_prompt.temperature,
        .top_p = drama_prompt.top_p,
        .top_k = drama_prompt.top_k,
        .stream = plan.stream,
        .reference_rows = plan.reference_rows,
        .unconditioned = guided.rows,
        .cfg_gamma = guided.gamma,
    };
    var box: Mailbox = .{ .io = srv.io, .gpa = srv.gpa };
    defer box.deinit();
    const engine_id = srv.next_id.fetchAdd(1, .monotonic);
    if (srv.keepalive) |k| k.begin();
    defer if (srv.keepalive) |k| k.end();
    const started = clock.nowNs(srv.io);
    srv.engine.submitSpeech(engine_id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => fail(c, cx, id, metadata.*, .capacity, "the engine is busy; retry shortly"),
        error.Closed => fail(c, cx, id, metadata.*, .other, "the scheduler is closed"),
        error.InvalidRequest => fail(c, cx, id, metadata.*, .request, "the engine refused this speech request's prompt, frame limit or sampling"),
        error.Unsupported => fail(c, cx, id, metadata.*, .request, "this model does not support speech synthesis"),
    };
    // The engine holds request, prompt, pcm and result until finished; no path below returns before that.
    if (plan.stream) return streamReply(srv, c, cx, id, sp, plan, received, &box, engine_id, pcm, &result, metadata.*);
    if (settle(srv, c, &box, engine_id, null) == null) return left(srv, c, id, plan.prompt.len, received);
    if (why(&box, result.samples, pcm.len)) |message| return fail(c, cx, id, metadata.*, .server, message);
    const take: Take = .{ .pcm = pcm[0..result.samples], .reason = box.reason, .silence_mid = result.silence_mid };
    var spare: []f32 = &.{};
    defer srv.gpa.free(spare);
    const sent = redraw(srv, c, cx, id, sp, plan, received, request, started, take, &spare, metadata) orelse return;
    reply(srv, c, cx, id, sp, plan, received, sent, metadata.*);
}

fn redraw(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, sp: api.SpeechInfo, plan: Plan, received: i96, first: api.SpeechRequest, started: i96, drawn: Take, spare: *[]f32, metadata: *Metadata) ?Take {
    const policy = srv.config.redraw;
    var next = first;
    var input = plan.input;
    var take = drawn;
    var held: ?Take = null;
    var start = started;
    var k: usize = 1;
    while (true) : (k += 1) {
        take.trip = policy.count() > 0 and policy.trips(take.reason, take.pcm.len, sp.sample_rate, input);
        const fallback = keep(take, held, srv.config.silence_guard);
        if (take.trip) {
            if (metadata.redraws == policy.count()) break;
            if (metadata.redraws == 0) if (policy.controls) |controls| {
                input = @import("speech_plain.zig").filter(cx.a, controls, plan.input) catch return unused(id, k, fallback, "out of memory");
                if (input.ptr != plan.input.ptr) {
                    const made = @import("drama_voice.zig").requestPrompt(cx, srv.text, sp, input, plan.refs) catch |e| return unused(id, k, fallback, if (e == error.OutOfMemory) "out of memory" else cx.message);
                    if (made) |voice| {
                        next.prompt = voice.prompt;
                        next.frames = voice.frames;
                        next.reference_rows = voice.reference_rows;
                    } else {
                        next.prompt = promptOf(srv, cx, sp, input) catch |e| return unused(id, k, fallback, if (e == error.OutOfMemory) "out of memory" else cx.message);
                        next.frames = drama_prompt.rows(cx.a, next.prompt) catch |e| return unused(id, k, fallback, @errorName(e));
                    }
                    const guided = drama_cfg.guide(cx.a, srv.text, srv.config.cfg, metadata.cfg_gamma orelse 1, input, plan.refs.len > 0) catch
                        return unused(id, k, fallback, "the tokenizer cannot encode the unconditional prompt");
                    next.unconditioned = guided.rows;
                    next.cfg_gamma = guided.gamma;
                }
            };
        } else {
            if (!policy.register.on()) break;
            take.jump = speech_register.jump(cx.a, id, take.pcm, sp.sample_rate, policy.register);
            if ((take.jump orelse 0) == 0) break;
            metadata.register_tripped = true;
            log.line("speech {s} register jump={d:.2}st", .{ id, take.jump.? });
            if (metadata.register_redrawn) break;
            held = take;
        }
        const now = clock.nowNs(srv.io);
        if (!speech_redraw.fits(plan.budget_s, received, start, now)) return unused(id, k, fallback, "t + d exceeds the budget");
        const start_by = if (plan.budget_s) |s| s - clock.seconds(now - start) else null;
        start = now;
        if (spare.len == 0) spare.* = srv.gpa.alloc(f32, first.pcm.len) catch return unused(id, k, fallback, "out of memory");
        var result: api.SpeechResult = .{};
        // Once register holds a take, subsequent silence retries must never overwrite it.
        const preserved = if (held) |h| h.pcm.ptr else take.pcm.ptr;
        next.pcm = if (preserved == first.pcm.ptr) spare.* else first.pcm;
        next.result = &result;
        next.seed = api.seedFor(next.prompt, srv.config.seed_salt +% @as(i64, @intCast(k)));
        var box: Mailbox = .{ .io = srv.io, .gpa = srv.gpa };
        defer box.deinit();
        const engine_id = srv.next_id.fetchAdd(1, .monotonic);
        submitRedraw(srv, c, engine_id, &next, &box, received, start_by) catch |e| return unused(id, k, fallback, @errorName(e));
        if (take.trip) metadata.redraws += 1 else metadata.register_redrawn = true;
        if (settle(srv, c, &box, engine_id, null) == null) {
            left(srv, c, id, plan.prompt.len, received);
            return null;
        }
        if (why(&box, result.samples, next.pcm.len)) |message| return unused(id, k, fallback, message);
        take = .{ .pcm = next.pcm[0..result.samples], .reason = box.reason, .silence_mid = result.silence_mid };
        log.line("speech {s} redraw {d} finish={s}", .{ id, k, @tagName(take.reason) });
    }
    const sent = keep(take, held, srv.config.silence_guard);
    if (held != null and sent.pcm.ptr == held.?.pcm.ptr)
        return unused(id, k - 1, sent, if (take.jump == null) "unmeasured" else "it ends worse or jumps no less");
    return sent;
}

fn keep(later: Take, held: ?Take, guard: bool) Take {
    const first = held orelse return later;
    if (later.pcm.ptr == first.pcm.ptr) return first;
    const better = !later.trip and later.reason != .silence and
        (if (later.jump) |value| value < first.jump.? else guard);
    return if (better) later else first;
}

fn submitRedraw(srv: *Server, c: *Conn, id: api.Id, request: *const api.SpeechRequest, box: *Mailbox, received: i96, start_by: ?f64) api.SpeechSubmitError!void {
    while (true) {
        if (start_by) |by| if (clock.seconds(clock.nowNs(srv.io) - received) > by) return error.Busy;
        srv.engine.submitSpeech(id, request, .{ .ctx = box, .event = Mailbox.onEvent }) catch |e| {
            if (e != error.Busy or start_by == null or c.peerGone()) return e;
            std.Io.sleep(srv.io, .fromMilliseconds(20), .awake) catch {};
            continue;
        };
        return;
    }
}

/// Logs why redraw `k` gave no take; `take` stays the reply.
fn unused(id: []const u8, k: usize, take: Take, why_not: []const u8) Take {
    log.line("speech {s} redraw {d} not used: {s}", .{ id, k, why_not });
    return take;
}

/// Sends a whole take under its finish reason.
fn reply(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, sp: api.SpeechInfo, plan: Plan, received: i96, take: Take, metadata: Metadata) void {
    const data = quantizeInPlace(take.pcm) catch return fail(c, cx, id, metadata, .server, "the synthesized audio has non-finite samples");
    if (data.len > std.math.maxInt(u32) - 36) return fail(c, cx, id, metadata, .server, "the synthesized audio is too long for one reply");
    sendAudio(c, plan.format, sp.sample_rate, data, take.reason, if (srv.config.silence_mid) @intFromBool(take.silence_mid) else null, metadata) catch {
        c.close = true; // a head already started or a client gone: nothing more can be sent on this connection
        return;
    };
    log.line("speech {s} prompt={d} samples={d} format={s} finish={s} seconds={d:.2}", .{ id, plan.prompt.len, take.pcm.len, @tagName(plan.format), @tagName(take.reason), clock.seconds(clock.nowNs(srv.io) - received) });
}

/// The client left before the reply ended; the synthesis was cancelled and waited out.
fn left(srv: *Server, c: *Conn, id: []const u8, prompt_len: usize, received: i96) void {
    c.close = true;
    srv.metrics.disconnected(srv.io);
    var buf: [1024]u8 = undefined;
    log.line("{s}", .{log.ended(&buf, id, null, prompt_len, 0, clock.seconds(clock.nowNs(srv.io) - received))});
}

/// Why a finished synthesis has no complete audio, or null; an empty reply is no fault, length and silence clip it.
fn why(box: *const Mailbox, samples: usize, capacity: usize) ?[]const u8 {
    return switch (box.reason) {
        .stop, .length, .silence => if (samples > capacity) "the engine wrote more audio than the reply holds" else null,
        .cancelled => "speech synthesis was cancelled",
        .failed => if (box.message.len > 0) box.message else "speech synthesis failed",
    };
}

/// An error reply when synthesis ends before any audio; else the chunks, and the terminator only after a complete one.
fn streamReply(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, sp: api.SpeechInfo, plan: Plan, received: i96, box: *Mailbox, engine_id: api.Id, pcm: []f32, result: *const api.SpeechResult, metadata: Metadata) void {
    const sent = streamAudio(srv, c, sp, box, engine_id, pcm, metadata) orelse return left(srv, c, id, plan.prompt.len, received);
    const mismatch: ?[]const u8 = if (result.samples != sent.samples) "the finished synthesis does not match the audio it published" else null;
    const problem = why(box, result.samples, pcm.len) orelse mismatch;
    if (sent.samples == 0) {
        if (problem) |message| return fail(c, cx, id, metadata, .server, message);
        // the model closed without a frame: an empty stream, not a fault
        startStream(c, sp.sample_rate, metadata) catch {
            c.close = true;
            return;
        };
        c.writeAll(if (box.reason == .length) length_chunk else last_chunk) catch {
            c.close = true;
            return;
        };
        c.response_complete = true;
        log.line("speech {s} prompt={d} samples=0 format=pcm stream=true finish={s} seconds={d:.2}", .{ id, plan.prompt.len, @tagName(box.reason), clock.seconds(clock.nowNs(srv.io) - received) });
        return;
    }
    if (problem) |message| {
        c.close = true; // no terminating chunk: the client's HTTP/1.1 parser reports an incomplete body
        log.line("speech {s} stream ended incomplete after samples={d}: {s}", .{ id, sent.samples, message });
        return;
    }
    c.writeAll(if (box.reason == .silence) silence_chunk else if (box.reason == .length) length_chunk else last_chunk) catch {
        c.close = true;
        return;
    };
    c.response_complete = true;
    log.line("speech {s} prompt={d} samples={d} format=pcm stream=true finish={s} first_audio_ready_seconds={d:.3} first_frame_written_seconds={d:.3} seconds={d:.2}", .{
        id, plan.prompt.len, sent.samples, @tagName(box.reason), clock.seconds(sent.first_ready - received), clock.seconds(sent.first_written - received), clock.seconds(clock.nowNs(srv.io) - received),
    });
}

/// What a stream sent, when this thread first saw published audio, and when frame 0's own bytes reached the socket.
const Sent = struct { samples: usize = 0, first_ready: i96 = 0, first_written: i96 = 0 };

/// Sends everything published since the last send as one chunk, converted in place; null: the client left.
fn streamAudio(srv: *Server, c: *Conn, sp: api.SpeechInfo, box: *Mailbox, engine_id: api.Id, pcm: []f32, metadata: Metadata) ?Sent {
    var sent: Sent = .{};
    var size: [20]u8 = undefined;
    while (true) {
        const now = settle(srv, c, box, engine_id, sent.samples) orelse return null;
        if (now == sent.samples) return sent; // finished, and everything published is sent
        const first = sent.samples == 0;
        if (first) {
            sent.first_ready = clock.nowNs(srv.io);
            startStream(c, sp.sample_rate, metadata) catch return abandon(srv, box, engine_id);
        }
        // the audio event made pcm[..now] final: this thread converts it in place
        const bytes = quantizeInPlace(pcm[sent.samples..now]) catch return abandon(srv, box, engine_id);
        // frame 0's bytes go apart, so their hand-off time does not depend on the chunk's size
        const split: usize = if (first) @min(2 * sp.frame_length, bytes.len) else 0;
        c.writeAll(std.fmt.bufPrint(&size, "{x}\r\n", .{bytes.len}) catch unreachable) catch return abandon(srv, box, engine_id);
        c.writeAll(bytes[0..split]) catch return abandon(srv, box, engine_id);
        if (first) sent.first_written = clock.nowNs(srv.io);
        c.writeAll(bytes[split..]) catch return abandon(srv, box, engine_id);
        c.writeAll("\r\n") catch return abandon(srv, box, engine_id);
        sent.samples = now;
    }
}

/// Cancels a synthesis whose client is gone, then waits for its finished event; null, for the callers' `return`.
fn abandon(srv: *Server, box: *Mailbox, engine_id: api.Id) ?Sent {
    srv.engine.cancel(engine_id);
    box.mutex.lockUncancelable(box.io);
    defer box.mutex.unlock(box.io);
    while (!box.finished) box.cond.waitUncancelable(box.io, &box.mutex);
    return null;
}

/// Waits for `finished` or (with `sent`) samples past it, cancelling if the client leaves; samples so far, or null.
fn settle(srv: *Server, c: *Conn, box: *Mailbox, engine_id: api.Id, sent: ?usize) ?usize {
    while (true) {
        box.mutex.lockUncancelable(box.io);
        const idle = if (sent) |n| box.audio == n else true;
        if (!box.finished and idle) box.cond.waitTimeout(box.io, &box.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
        const finished = box.finished;
        const published = box.audio;
        box.mutex.unlock(box.io);
        if (finished or (if (sent) |n| published > n else false)) return published;
        if (c.peerGone()) {
            _ = abandon(srv, box, engine_id);
            return null;
        }
    }
}

/// 200 with raw PCM16LE in HTTP/1.1 chunks, under the reference's sample-format headers.
fn startStream(c: *Conn, sample_rate: u32, metadata: Metadata) error{Closed}!void {
    c.startResponse(200, null) catch return error.Closed;
    c.addHeader("Content-Type", Format.pcm.contentType()) catch return error.Closed;
    c.addHeader("Transfer-Encoding", "chunked") catch return error.Closed;
    var rate: [16]u8 = undefined;
    c.addHeader("X-Sample-Rate", std.fmt.bufPrint(&rate, "{d}", .{sample_rate}) catch unreachable) catch return error.Closed;
    c.addHeader("X-Channels", "1") catch return error.Closed;
    c.addHeader("X-Bit-Depth", "16") catch return error.Closed;
    c.addHeader("Trailer", "X-Finish-Reason") catch return error.Closed;
    metadata.add(c) catch return error.Closed;
    if (c.close) c.addHeader("Connection", "close") catch return error.Closed;
    // each chunk leaves at once rather than waiting on the client's acknowledgement of the last (no effect off TCP)
    const one: c_int = 1;
    if (platform.windows) platform.posix.setsockopt(c.fd, platform.posix.IPPROTO.TCP, platform.posix.TCP.NODELAY, std.mem.asBytes(&one)) catch {} else _ = std.c.setsockopt(c.fd, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, &one, @sizeOf(c_int));
    // a blocking write into a full socket returns after the stall limit, so writeAll's poll can end a stalled stream
    const stall: platform.posix.timeval = .{ .sec = @divTrunc(c.timeouts.write_ms, 1000), .usec = @rem(c.timeouts.write_ms, 1000) * 1000 };
    platform.posix.setsockopt(c.fd, platform.posix.SOL.SOCKET, platform.posix.SO.SNDTIMEO, std.mem.asBytes(&stall)) catch {};
    try c.finish("");
    c.response_complete = false;
}

/// Ends a complete stream; length and silence use an X-Finish-Reason trailer instead.
const last_chunk = "0\r\n\r\n";
const length_chunk = "0\r\nX-Finish-Reason: length\r\n\r\n";
const silence_chunk = "0\r\nX-Finish-Reason: silence\r\n\r\n";

fn refuse(c: *Conn, cx: *Cx, id: []const u8, metadata: Metadata, e: errors.Refused) void {
    if (e == error.OutOfMemory) {
        cx.kind = .server;
        cx.message = "out of memory";
    }
    send(c, cx, id, metadata);
}

fn fail(c: *Conn, cx: *Cx, id: []const u8, metadata: Metadata, kind: errors.Kind, message: []const u8) void {
    cx.kind = kind;
    cx.message = message;
    send(c, cx, id, metadata);
}

/// OpenAI's error object, logged first: the access line shows only the status.
fn send(c: *Conn, cx: *Cx, id: []const u8, metadata: Metadata) void {
    var buf: [1400]u8 = undefined;
    log.line("{s}", .{log.refused(&buf, id, cx.message)});
    const body = openai.errorBody(cx.a, cx, null) catch return;
    const text = json.stringify(cx.a, openai.wrapError(cx.a, body) catch return, .{}) catch return;
    c.startResponse(cx.status(), null) catch return;
    c.addHeader("Content-Type", "application/json") catch return;
    var len: [24]u8 = undefined;
    c.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{text.len}) catch unreachable) catch return;
    metadata.add(c) catch return;
    if (c.close) c.addHeader("Connection", "close") catch return;
    c.finish(text) catch {
        c.close = true;
    };
}

/// The head, then the WAV header and the samples where they lie: the audio is never copied to prepend a header.
fn sendAudio(c: *Conn, format: Format, sample_rate: u32, data: []const u8, reason: api.Reason, silence_mid: ?u8, metadata: Metadata) error{ Closed, OutOfMemory }!void {
    const header: ?[wav_header_len]u8 = if (format == .wav) wavHeader(sample_rate, @intCast(data.len)) else null;
    const header_len: usize = if (header != null) wav_header_len else 0;
    try c.startResponse(200, null);
    try c.addHeader("Content-Type", format.contentType());
    var len: [24]u8 = undefined;
    try c.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{header_len + data.len}) catch unreachable);
    try c.addHeader("X-Finish-Reason", @tagName(reason));
    if (silence_mid) |trips| {
        var count: [3]u8 = undefined;
        try c.addHeader("X-Silence-Mid-Trips", std.fmt.bufPrint(&count, "{d}", .{trips}) catch unreachable);
    }
    try metadata.addWhole(c);
    if (c.close) try c.addHeader("Connection", "close");
    defer c.out.clearRetainingCapacity();
    if (!std.mem.eql(u8, c.version, "HTTP/0.9")) try c.out.appendSlice(c.gpa, "\r\n");
    if (header) |h| try c.out.appendSlice(c.gpa, &h);
    try c.writeAll(c.out.items);
    try c.writeAll(data);
    c.response_complete = true;
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

test "fields: unknown ones are refused even when null, modifiers when set" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"voice\": \"default\", \"response_format\": \"wav\", \"speed\": null, \"references\": null}", "" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream_audio\": null}", "unknown speech request field: stream_audio" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"speed\": 1.0}", "speed is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream\": false, \"stream_format\": \"audio\"}", "" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream\": null, \"stream_format\": null}", "" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream\": \"yes\"}", "stream must be a boolean" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream_format\": \"sse\"}", "stream_format sse is not supported by this model" },
        .{ "{\"model\": \"m\", \"input\": \"hi\", \"stream_format\": \"mp3\"}", "stream_format must be one of: audio, sse" },
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

test "allowed controls filters before speech input validation" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv: Server = undefined;
    srv.config = .{ .served_name = "m", .model_ids = &.{"m"}, .allowed_controls = try @import("drama_allow.zig").parse("") };
    const sp: api.SpeechInfo = .{ .sample_rate = 44100, .frame_length = 2048, .max_frames = 1024, .max_prompt_tokens = 4096 };
    const long = try a.alloc(u8, max_input_chars + 1);
    @memset(long, 'x');
    const body = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":\"[{s}]\"}}", .{long}))).ok;
    var cx: Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, prepare(&srv, &cx, sp, body));
    try std.testing.expectEqualStrings("input must be at most 4096 characters", cx.message);
    const controls = (try json.parse(a, "{\"model\":\"m\",\"input\":\"[sighs]\"}")).ok;
    try std.testing.expectError(error.Refused, prepare(&srv, &cx, sp, controls));
    try std.testing.expectEqualStrings("input must be a non-empty string", cx.message);
    srv.config.allowed_controls = null;
    try std.testing.expectError(error.Refused, prepare(&srv, &cx, sp, body));
    try std.testing.expectEqualStrings("input must be at most 4096 characters", cx.message);
}

pub const Test = if (@import("builtin").is_test) struct {
    pub const redrawTake = redraw;
} else struct {};

test {
    _ = @import("speech_audio_test.zig");
}
