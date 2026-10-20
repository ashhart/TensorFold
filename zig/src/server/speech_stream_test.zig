//! The streamed speech reply on the wire, against a scripted engine over a socketpair with the route on its own thread.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

const frame = 2048;
const gpa = std.testing.allocator;

/// Publishes 0.5-filled steps then finishes: inside submit (`inline_`), else on a thread that may `hold` for `release`.
const Script = struct {
    steps: []const usize,
    reason: api.Reason = .stop,
    message: []const u8 = "",
    hold: usize = 0,
    inline_: bool = false,
    max_frames: u32 = 4,
    output_full: bool = false,
    release: std.atomic.Value(bool) = .init(false),
    cancelled: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    fn engine(e: *Script) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .submit_speech = speech } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const e: *Script = @ptrCast(@alignCast(ctx));
        return .{ .text_generation = false, .speech = .{ .sample_rate = 44100, .frame_length = frame, .max_frames = e.max_frames, .max_prompt_tokens = 4096 } };
    }
    fn speech(ctx: *anyopaque, id: api.Id, request: *const api.SpeechRequest, sink: api.Sink) api.SpeechSubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        if (e.inline_) return run(e, id, request, sink);
        e.thread = std.Thread.spawn(.{}, run, .{ e, id, request, sink }) catch return error.Busy;
    }
    fn run(e: *Script, id: api.Id, request: *const api.SpeechRequest, sink: api.Sink) void {
        var done: usize = 0;
        for (e.steps, 0..) |n, i| {
            if (e.hold > 0 and i == e.hold) {
                while (!e.release.load(.acquire)) std.Thread.yield() catch {};
                if (e.cancelled.load(.acquire)) return sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
            }
            @memset(request.pcm[done..n], 0.5);
            if (request.stream) sink.event(sink.ctx, id, &.{ .audio = n });
            done = n;
        }
        if (e.reason == .stop or e.reason == .length) request.result.samples = done;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = e.reason, .message = e.message } });
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        e.cancelled.store(true, .release);
        e.release.store(true, .release);
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// Every byte is its own token: the route's prompt encoding needs no vocabulary.
const Bytes = struct {
    fn text(t: *Bytes) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, _: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return error.Template;
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// One request served on its own thread; the test reads the reply from the client end as it arrives.
const Served = struct {
    srv: *server.Server,
    text: Bytes = .{},
    client: std.c.fd_t,
    route_fd: std.c.fd_t,
    route: std.Thread,
    write_ms: i32,
    got: std.ArrayList(u8) = .empty,
    read_at: usize = 0,
    eof: bool = false,
    client_open: bool = true,

    fn start(s: *Served, script: *Script, version: []const u8, body: []const u8, write_ms: i32) !void {
        var pair: [2]std.c.fd_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        s.* = .{ .srv = undefined, .client = pair[0], .route_fd = pair[1], .route = undefined, .write_ms = write_ms };
        s.srv = try server.Server.init(gpa, std.testing.io, script.engine(), s.text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        if (script.output_full) s.srv.speech_inflight.store(s.srv.info.lanes, .release);
        var head: [160]u8 = undefined;
        try writeAll(s.client, try std.fmt.bufPrint(&head, "POST /v1/audio/speech {s}\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ version, body.len }));
        try writeAll(s.client, body);
        s.route = try std.Thread.spawn(.{}, serve, .{s});
    }

    fn serve(s: *Served) void {
        var conn = Conn.init(gpa, s.route_fd, "test") catch return;
        defer conn.deinit();
        conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = s.write_ms };
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const outcome = conn.readRequest(arena.allocator(), false) catch return;
        if (outcome == .ready) routes.dispatch(s.srv, &conn, arena.allocator());
        if (conn.close) _ = std.c.shutdown(s.route_fd, 1); // what the server loop does to a closing connection: EOF
    }

    /// Waits for the route, then for the engine's own thread.
    fn join(s: *Served, script: *Script) void {
        s.route.join();
        if (script.thread) |t| t.join();
        script.thread = null;
    }

    fn deinit(s: *Served) void {
        if (s.client_open) _ = std.c.close(s.client);
        _ = std.c.close(s.route_fd);
        s.got.deinit(gpa);
        s.srv.deinit();
    }

    /// Reads more of the reply; false at EOF or after `wait_ms` of silence.
    fn more(s: *Served, wait_ms: i32) !bool {
        if (s.eof) return false;
        var fds = [_]std.posix.pollfd{.{ .fd = s.client, .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&fds, wait_ms) == 0) return false;
        var tmp: [65536]u8 = undefined;
        const n = std.posix.read(s.client, &tmp) catch 0;
        if (n == 0) {
            s.eof = true;
            return false;
        }
        try s.got.appendSlice(gpa, tmp[0..n]);
        return true;
    }

    fn readHead(s: *Served) ![]const u8 {
        while (std.mem.indexOf(u8, s.got.items, "\r\n\r\n") == null) if (!try s.more(2000)) return error.NoHead;
        const end = std.mem.indexOf(u8, s.got.items, "\r\n\r\n").? + 4;
        s.read_at = end;
        return s.got.items[0..end];
    }

    /// The next chunk's bytes, or null after the terminating chunk; NoChunk when the reply ends first.
    fn readChunk(s: *Served) !?[]const u8 {
        while (true) {
            const rest = s.got.items[s.read_at..];
            if (std.mem.indexOf(u8, rest, "\r\n")) |line| {
                const size = try std.fmt.parseInt(usize, rest[0..line], 16);
                if (rest.len >= line + 2 + size + 2) {
                    if (!std.mem.eql(u8, rest[line + 2 + size ..][0..2], "\r\n")) return error.BadChunk;
                    s.read_at += line + 2 + size + 2;
                    return if (size == 0) null else rest[line + 2 ..][0..size];
                }
            }
            if (!try s.more(2000)) return error.NoChunk;
        }
    }

    /// Every chunk's bytes until the reply ends; `terminated`: the zero chunk came.
    fn payload(s: *Served, terminated: *bool) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        while (s.readChunk()) |chunk| {
            const bytes = chunk orelse {
                terminated.* = true;
                return out.toOwnedSlice(gpa);
            };
            try out.appendSlice(gpa, bytes);
        } else |e| if (e != error.NoChunk) return e;
        terminated.* = false;
        return out.toOwnedSlice(gpa);
    }

    /// The rest of a reply the route has finished writing.
    fn readRest(s: *Served) ![]const u8 {
        while (try s.more(200)) {}
        return s.got.items[s.read_at..];
    }
};

fn writeAll(fd: std.c.fd_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.write(fd, data[sent..].ptr, data.len - sent);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

const stream_body = "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"stream\":true}";
/// `n` frames of 0.5, which quantizes to 16383 (0x3fff), as PCM16LE.
fn pcm(comptime n: usize) [2 * frame * n]u8 {
    var bytes: [2 * frame * n]u8 = undefined;
    for (0..frame * n) |i| bytes[2 * i ..][0..2].* = .{ 0xff, 0x3f };
    return bytes;
}

fn expectHeader(head: []const u8, line: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, head, line) != null);
}

test "a complete stream carries every frame in order, converted, then the terminating chunk" {
    var script: Script = .{ .steps = &.{ frame, 3 * frame } };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", stream_body, 2000);
    defer s.deinit();
    const head = try s.readHead();
    try expectHeader(head, "HTTP/1.1 200");
    try expectHeader(head, "Content-Type: audio/pcm\r\n");
    try expectHeader(head, "Transfer-Encoding: chunked\r\n");
    try expectHeader(head, "X-Sample-Rate: 44100\r\nX-Channels: 1\r\nX-Bit-Depth: 16\r\n");
    try std.testing.expect(std.mem.indexOf(u8, head, "Content-Length") == null);
    var terminated = false;
    const audio = try s.payload(&terminated); // where the route caught each publication is scheduling, not the bytes
    defer gpa.free(audio);
    s.join(&script);
    try std.testing.expectEqualStrings(&pcm(3), audio);
    try std.testing.expect(terminated);
}

test "audio published before the route reads it goes out as one chunk" {
    var script: Script = .{ .steps = &.{ frame, 3 * frame }, .inline_ = true };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", stream_body, 2000);
    defer s.deinit();
    _ = try s.readHead();
    const rest = try s.readRest();
    s.join(&script);
    try std.testing.expect(std.mem.startsWith(u8, rest, "3000\r\n") and std.mem.endsWith(u8, rest, "\r\n0\r\n\r\n"));
    try std.testing.expectEqualStrings(&pcm(3), rest["3000\r\n".len .. rest.len - "\r\n0\r\n\r\n".len]);
}

test "a full speech output budget refuses before starting another synthesis" {
    var script: Script = .{ .steps = &.{frame}, .output_full = true };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", stream_body, 2000);
    defer s.deinit();
    const head = try s.readHead();
    try std.testing.expect(std.mem.indexOf(u8, head, "503 Service Unavailable") != null);
    s.join(&script);
    try std.testing.expect(script.thread == null);
    try std.testing.expectEqual(@as(u32, 1), s.srv.speech_inflight.load(.acquire));
}

test "audio is on the wire before the synthesis ends; frame 0 alone when sent before more is published" {
    var script: Script = .{ .steps = &.{ frame, 3 * frame }, .hold = 1 };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", stream_body, 2000);
    defer s.deinit();
    _ = try s.readHead();
    try std.testing.expectEqualStrings(&pcm(1), (try s.readChunk()).?); // the engine still holds frames 1 and 2
    script.release.store(true, .release);
    try std.testing.expectEqualStrings(&pcm(2), (try s.readChunk()).?);
    try std.testing.expect(try s.readChunk() == null);
    s.join(&script);
}

test "a failure after audio ends the stream without its terminating chunk" {
    var script: Script = .{ .steps = &.{frame}, .reason = .failed, .message = "the codec produced non-finite audio" };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", stream_body, 2000);
    defer s.deinit();
    _ = try s.readHead();
    var terminated = true;
    const audio = try s.payload(&terminated);
    defer gpa.free(audio);
    s.join(&script);
    try std.testing.expectEqualStrings(&pcm(1), audio);
    try std.testing.expect(!terminated and s.eof);
}

test "audio clipped at the frame limit ends the stream with the length trailer" {
    var script: Script = .{ .steps = &.{ frame, 2 * frame }, .reason = .length, .inline_ = true };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", stream_body, 2000);
    defer s.deinit();
    try expectHeader(try s.readHead(), "Trailer: X-Finish-Reason\r\n");
    const rest = try s.readRest();
    s.join(&script);
    const end = "\r\n0\r\nX-Finish-Reason: length\r\n\r\n";
    try std.testing.expect(std.mem.startsWith(u8, rest, "2000\r\n") and std.mem.endsWith(u8, rest, end));
    try std.testing.expectEqualStrings(&pcm(2), rest["2000\r\n".len .. rest.len - end.len]);
}

test "a whole reply returns audio clipped at the frame limit and names its finish reason" {
    for ([_]api.Reason{ .stop, .length }) |reason| {
        var script: Script = .{ .steps = &.{frame}, .reason = reason };
        var s: Served = undefined;
        try s.start(&script, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\"}", 2000);
        defer s.deinit();
        const head = try s.readHead();
        try expectHeader(head, "HTTP/1.1 200");
        try expectHeader(head, if (reason == .length) "X-Finish-Reason: length\r\n" else "X-Finish-Reason: stop\r\n");
        s.join(&script);
        try std.testing.expectEqualStrings(&pcm(1), try s.readRest());
    }
}

test "a failure before any audio is an error reply" {
    var script: Script = .{ .steps = &.{}, .reason = .failed, .message = "the model ended speech before its first audio frame" };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", stream_body, 2000);
    defer s.deinit();
    try expectHeader(try s.readHead(), "HTTP/1.1 500");
    s.join(&script);
    try std.testing.expect(std.mem.indexOf(u8, try s.readRest(), "the model ended speech before its first audio frame") != null);
}

test "a client that leaves after the first chunk cancels the synthesis, which is waited out" {
    var script: Script = .{ .steps = &.{ frame, 3 * frame }, .hold = 1 };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", stream_body, 2000);
    defer s.deinit();
    _ = try s.readHead();
    _ = (try s.readChunk()).?;
    _ = std.c.close(s.client); // the route's peerGone poll cancels; the engine then finishes cancelled
    s.client_open = false;
    s.join(&script); // returns only after the finished event: the request storage outlived the synthesis
    try std.testing.expect(script.cancelled.load(.acquire));
}

test "a client that stops reading is cut off at the write stall limit and the synthesis cancelled" {
    var steps: [255]usize = undefined;
    for (&steps, 1..) |*n, i| n.* = i * frame;
    var script: Script = .{ .steps = &steps, .max_frames = 256, .hold = 1 };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", stream_body, 200);
    defer s.deinit();
    script.release.store(true, .release); // 1 MiB of PCM16 published while nobody reads overflows the socket buffers
    s.join(&script);
    try std.testing.expect(script.cancelled.load(.acquire));
    _ = try s.readHead();
    var terminated = true;
    gpa.free(try s.payload(&terminated));
    try std.testing.expect(!terminated);
}

test "stream needs pcm and HTTP/1.1; stream false is the whole reply" {
    const cases = [_]struct { version: []const u8, body: []const u8, want: []const u8 }{
        .{ .version = "HTTP/1.1", .body = "{\"model\":\"m\",\"input\":\"hi\",\"stream\":true}", .want = "stream=true requires response_format='pcm'" },
        .{ .version = "HTTP/1.0", .body = stream_body, .want = "stream=true requires HTTP/1.1" },
    };
    for (cases) |case| {
        var script: Script = .{ .steps = &.{frame} };
        var s: Served = undefined;
        try s.start(&script, case.version, case.body, 2000);
        defer s.deinit();
        try expectHeader(try s.readHead(), " 400 ");
        s.join(&script);
        try std.testing.expect(std.mem.indexOf(u8, try s.readRest(), case.want) != null);
    }
    var script: Script = .{ .steps = &.{frame} };
    var s: Served = undefined;
    try s.start(&script, "HTTP/1.1", "{\"model\":\"m\",\"input\":\"hi\",\"response_format\":\"pcm\",\"stream\":false}", 2000);
    defer s.deinit();
    try expectHeader(try s.readHead(), "Content-Length: 4096\r\n");
    s.join(&script);
    try std.testing.expectEqualStrings(&pcm(1), try s.readRest());
}
