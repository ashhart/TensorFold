//! HTTP/1.1 for one LFM3-Omni engine, one request at a time: POST /v1/audio/chat (SSE or whole) and GET /health.
const std = @import("std");
const core = @import("core");
const engine = @import("common.zig");
const cfg = @import("config.zig");
const mel = @import("mel.zig");
const prompt = @import("prompt.zig");
const wav = @import("wav");

extern "c" fn socket(domain: c_int, ty: c_int, proto: c_int) c_int;
extern "c" fn setsockopt(fd: c_int, level: c_int, name: c_int, val: *const anyopaque, len: u32) c_int;
extern "c" fn bind(fd: c_int, addr: *const anyopaque, len: u32) c_int;
extern "c" fn listen(fd: c_int, n: c_int) c_int;
extern "c" fn accept(fd: c_int, addr: ?*anyopaque, len: ?*u32) c_int;
extern "c" fn recv(fd: c_int, buf: [*]u8, n: usize, flags: c_int) isize;
extern "c" fn send(fd: c_int, buf: [*]const u8, n: usize, flags: c_int) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn inet_pton(af: c_int, src: [*:0]const u8, dst: *anyopaque) c_int;

const SockAddrIn = extern struct { family: u16, port: u16, addr: u32, zero: [8]u8 = @splat(0) };
const msg_nosignal = 0x4000;
const max_body = 64 << 20;

pub const Options = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8441,
    /// The system text a request without one gets.
    system: []const u8 = "Respond with interleaved text and audio.",
    smp: engine.Sampling = .{},
    stream: engine.Streaming = .{},
};

const Conn = struct {
    fd: c_int,
    closed: bool = false,

    fn writeAll(c: *Conn, b: []const u8) !void {
        var at: usize = 0;
        while (at < b.len) {
            const n = send(c.fd, b.ptr + at, b.len - at, msg_nosignal);
            if (n <= 0) {
                c.closed = true;
                return error.ClientGone;
            }
            at += @intCast(n);
        }
    }

    /// One chunk of a chunked body.
    fn chunk(c: *Conn, b: []const u8) !void {
        var h: [32]u8 = undefined;
        try c.writeAll(try std.fmt.bufPrint(&h, "{x}\r\n", .{b.len}));
        try c.writeAll(b);
        try c.writeAll("\r\n");
    }
};

pub fn run(gpa: std.mem.Allocator, io: std.Io, e: anytype, tok: *const core.tokenizer.Tokenizer, o: Options) !void {
    const fd = socket(2, 1, 0);
    if (fd < 0) return error.Socket;
    defer _ = close(fd);
    const one: c_int = 1;
    _ = setsockopt(fd, 1, 2, &one, @sizeOf(c_int));
    var addr: SockAddrIn = .{ .family = 2, .port = std.mem.nativeToBig(u16, o.port), .addr = 0 };
    const hz = try gpa.dupeSentinel(u8, o.host, 0);
    defer gpa.free(hz);
    if (inet_pton(2, hz.ptr, &addr.addr) != 1) return error.BadHost;
    if (bind(fd, &addr, @sizeOf(SockAddrIn)) != 0) return error.Bind;
    if (listen(fd, 16) != 0) return error.Listen;
    std.log.info("lfm3-omni: serving POST /v1/audio/chat on {s}:{d}", .{ o.host, o.port });
    while (true) {
        const cfd = accept(fd, null, null);
        if (cfd < 0) continue;
        var c: Conn = .{ .fd = cfd };
        handle(gpa, io, e, tok, o, &c) catch |err| {
            std.log.warn("lfm3-omni: request failed: {s}", .{@errorName(err)});
            if (!c.closed) respond(&c, "500 Internal Server Error", "application/json", "{\"error\":\"internal error\"}") catch {};
        };
        _ = close(cfd);
    }
}

fn respond(c: *Conn, status: []const u8, ctype: []const u8, body: []const u8) !void {
    var h: [256]u8 = undefined;
    try c.writeAll(try std.fmt.bufPrint(&h, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ status, ctype, body.len }));
    try c.writeAll(body);
}

fn badRequest(c: *Conn, a: std.mem.Allocator, msg: []const u8) !void {
    try respond(c, "400 Bad Request", "application/json", try std.fmt.allocPrint(a, "{{\"error\":{s}}}", .{try jstr(a, msg)}));
}

/// A JSON string literal.
fn jstr(a: std.mem.Allocator, s: []const u8) ![]const u8 {
    return std.json.Stringify.valueAlloc(a, s, .{});
}

/// The request head and body (Content-Length only).
fn readRequest(a: std.mem.Allocator, c: *Conn) !struct { method: []const u8, path: []const u8, body: []const u8 } {
    var buf: std.ArrayList(u8) = .empty;
    var head_end: ?usize = null;
    var tmp: [65536]u8 = undefined;
    while (head_end == null) {
        const n = recv(c.fd, &tmp, tmp.len, 0);
        if (n <= 0) return error.ClientGone;
        try buf.appendSlice(a, tmp[0..@intCast(n)]);
        head_end = std.mem.indexOf(u8, buf.items, "\r\n\r\n");
        if (head_end == null and buf.items.len > 1 << 16) return error.HeadTooLong;
    }
    const head = buf.items[0..head_end.?];
    var lines = std.mem.splitSequence(u8, head, "\r\n");
    const first = lines.next() orelse return error.BadRequest;
    var parts = std.mem.splitScalar(u8, first, ' ');
    const method = parts.next() orelse return error.BadRequest;
    const path = parts.next() orelse return error.BadRequest;
    var len: usize = 0;
    while (lines.next()) |l| {
        const colon = std.mem.indexOfScalar(u8, l, ':') orelse continue;
        if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, l[0..colon], " "), "content-length"))
            len = try std.fmt.parseInt(usize, std.mem.trim(u8, l[colon + 1 ..], " "), 10);
    }
    if (len > max_body) return error.BodyTooLarge;
    const start = head_end.? + 4;
    while (buf.items.len < start + len) {
        const n = recv(c.fd, &tmp, tmp.len, 0);
        if (n <= 0) return error.ClientGone;
        try buf.appendSlice(a, tmp[0..@intCast(n)]);
    }
    return .{ .method = method, .path = path, .body = buf.items[start .. start + len] };
}

const Turn = struct {
    a: std.mem.Allocator,
    c: *Conn,
    tok: *const core.tokenizer.Tokenizer,
    stream: bool,
    text_ids: std.ArrayList(u32) = .empty,
    sent_text: usize = 0,
    pcm: std.ArrayList(f32) = .empty,
    buf: std.ArrayList(u8) = .empty,
    first_audio_ms: f64 = 0,
    t0: i128,
    io: std.Io,

    fn event(t: *Turn, json_text: []const u8) !void {
        t.buf.clearRetainingCapacity();
        try t.buf.appendSlice(t.a, "data: ");
        try t.buf.appendSlice(t.a, json_text);
        try t.buf.appendSlice(t.a, "\n\n");
        try t.c.chunk(t.buf.items);
    }

    /// The reply's words so far (special tokens left out) past what was sent, whole UTF-8 sequences only.
    fn onText(ctx: *anyopaque, id: u32) anyerror!void {
        const t: *Turn = @ptrCast(@alignCast(ctx));
        if (id >= cfg.bos_token_id) return;
        try t.text_ids.append(t.a, id);
        if (!t.stream) return;
        const all = try t.tok.decode(t.a, t.text_ids.items, true);
        var end = all.len;
        while (end > t.sent_text and !std.unicode.utf8ValidateSlice(all[t.sent_text..end])) end -= 1;
        if (end <= t.sent_text) return;
        try t.event(try std.fmt.allocPrint(t.a, "{{\"type\":\"text\",\"delta\":{s}}}", .{try jstr(t.a, all[t.sent_text..end])}));
        t.sent_text = end;
    }

    fn onPcm(ctx: *anyopaque, x: []const f32) anyerror!void {
        const t: *Turn = @ptrCast(@alignCast(ctx));
        if (t.pcm.items.len == 0 and t.first_audio_ms == 0) t.first_audio_ms = @as(f64, @floatFromInt(engine.nowNs(t.io) - t.t0)) / 1e6;
        if (!t.stream) return t.pcm.appendSlice(t.a, x);
        const raw = try t.a.alloc(u8, x.len * 2);
        wav.pcm16(x, raw);
        const b64 = try t.a.alloc(u8, std.base64.standard.Encoder.calcSize(raw.len));
        _ = std.base64.standard.Encoder.encode(b64, raw);
        const j = try std.fmt.allocPrint(t.a, "{{\"type\":\"audio\",\"format\":\"pcm16\",\"sample_rate\":24000,\"data\":\"{s}\"}}", .{b64});
        try t.event(j);
    }
};

fn num(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn handle(gpa: std.mem.Allocator, io: std.Io, e: anytype, tok: *const core.tokenizer.Tokenizer, o: Options, c: *Conn) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const req = try readRequest(a, c);
    if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/health")) return respond(c, "200 OK", "application/json", "{\"ok\":true,\"model\":\"lfm3-omni\"}");
    if (!std.mem.eql(u8, req.method, "POST") or !std.mem.eql(u8, req.path, "/v1/audio/chat")) return respond(c, "404 Not Found", "application/json", "{\"error\":\"POST /v1/audio/chat or GET /health\"}");
    const t0 = engine.nowNs(io);
    const parsed = std.json.parseFromSlice(std.json.Value, a, req.body, .{}) catch return badRequest(c, a, "the body is not JSON");
    if (parsed.value != .object) return badRequest(c, a, "the body is not a JSON object");
    const obj = parsed.value.object;
    const audio_b64 = if (obj.get("audio")) |v| (if (v == .string) v.string else return badRequest(c, a, "audio: a base64 WAV")) else return badRequest(c, a, "audio is required (a base64 WAV)");
    const wav_bytes = try a.alloc(u8, std.base64.standard.Decoder.calcSizeForSlice(audio_b64) catch return badRequest(c, a, "audio is not base64"));
    std.base64.standard.Decoder.decode(wav_bytes, audio_b64) catch return badRequest(c, a, "audio is not base64");
    const au = wav.parse(a, wav_bytes) catch return badRequest(c, a, "audio is not a PCM or float WAV");
    const pcm = try wav.resample(a, au.samples, au.rate, mel.sample_rate);
    if (pcm.len == 0) return badRequest(c, a, "the audio is empty");
    if (pcm.len > e.o.max_input_s * mel.sample_rate) return badRequest(c, a, "the audio is longer than the server takes");
    const system = if (obj.get("system")) |v| (if (v == .string) v.string else o.system) else o.system;
    const stream_out = if (obj.get("stream")) |v| (v == .bool and v.bool) else true;
    var smp = o.smp;
    if (num(obj.get("audio_temperature"))) |x| smp.audio_temperature = @floatCast(x);
    if (num(obj.get("audio_repetition_penalty"))) |x| smp.audio_repetition_penalty = @floatCast(x);
    if (num(obj.get("seed"))) |x| smp.seed = @intFromFloat(x);
    if (num(obj.get("max_new"))) |x| smp.max_new = @min(@as(usize, @intFromFloat(x)), e.o.max_frames);
    smp.audio_top_k = 0;

    var st: engine.Stats = .{};
    var turn: Turn = .{ .a = a, .c = c, .tok = tok, .stream = stream_out, .t0 = t0, .io = io };
    const nf = try e.featuresDevice(pcm);
    const n_audio = try e.encode(null, nf);
    const p = try prompt.build(a, tok, system, n_audio);
    if (p.ids.len + smp.max_new + 1 > e.o.max_ctx) return badRequest(c, a, "the prompt and reply do not fit the window");
    try e.prefill(p.ids, p.audio_at, null);
    if (stream_out) try c.writeAll("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
    try e.reply(smp, o.stream, .{ .ctx = &turn, .text = Turn.onText, .pcm = Turn.onPcm }, null, &st, t0);
    const text = try tok.decode(a, turn.text_ids.items, true);
    const stats = try std.fmt.allocPrint(a, "\"text\":{s},\"frames\":{d},\"speech_s\":{d:.3},\"audio_tokens\":{d},\"prompt_rows\":{d},\"first_text_ms\":{d:.1},\"first_audio_ms\":{d:.1},\"total_ms\":{d:.1}", .{ try jstr(a, text), st.frames, @as(f64, @floatFromInt(st.samples)) / cfg.sample_rate, n_audio, p.ids.len, st.first_text_ms, turn.first_audio_ms, st.total_ms });
    if (!stream_out) {
        const wb = try wav.encode16(a, turn.pcm.items, cfg.sample_rate);
        const b64 = try a.alloc(u8, std.base64.standard.Encoder.calcSize(wb.len));
        _ = std.base64.standard.Encoder.encode(b64, wb);
        return respond(c, "200 OK", "application/json", try std.fmt.allocPrint(a, "{{{s},\"audio\":\"{s}\"}}", .{ stats, b64 }));
    }
    try turn.event(try std.fmt.allocPrint(a, "{{\"type\":\"done\",{s}}}", .{stats}));
    try turn.event("[DONE]");
    try c.chunk("");
    std.log.info("lfm3-omni: {d} audio tokens, {d} frames, first audio {d:.1} ms, total {d:.1} ms", .{ n_audio, st.frames, turn.first_audio_ms, st.total_ms });
}
