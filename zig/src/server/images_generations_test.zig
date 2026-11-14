//! POST /v1/images/generations on the wire: the fields the engine sees, the base64 PNG reply, and every refusal.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;
const Bytes = @import("speech_stream_test.zig").Bytes;

const t = std.testing;
const gpa = t.allocator;
const images_route = "POST /v1/images/generations";
const chat_route = "POST /v1/chat/completions";

/// The PNG the stub appends: the signature itself, base64 `iVBORw0KGgo=`.
const png_bytes = "\x89PNG\r\n\x1a\n";
/// The host's stated limits, the same the API's own caps allow, so only the route's checks ever bind.
const host_caps: api.ImageInfo = .{ .max_side = 1536, .max_pixels = 1572864, .max_steps = 200, .max_prompt_tokens = 4096 };

/// Records one image request, appends the fixed PNG with the request's own allocator, and ends it.
const Stub = struct {
    caps: ?api.ImageInfo = host_caps,
    submit_error: ?api.ImageSubmitError = null,
    inline_: bool = false,
    hold: bool = false,
    reason: api.Reason = .stop,
    message: []const u8 = "",
    release: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    calls: usize = 0,
    prompt_ids: [16]u32 = undefined,
    prompt_len: usize = 0,
    negative_ids: [16]u32 = undefined,
    negative_len: usize = 0,
    width: u16 = 0,
    height: u16 = 0,
    steps: u32 = 0,
    guidance: f32 = 0,
    seed: u64 = 0,

    fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .submit_image = image } };
    }
    fn cast(ctx: *anyopaque) *Stub {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        return .{ .text_generation = false, .image = cast(ctx).caps };
    }
    /// The image route is the only one this engine serves; routes.zig refuses the text ones before it is reached.
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        cast(ctx).release.store(true, .release);
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn image(ctx: *anyopaque, id: api.Id, request: *const api.ImageRequest, sink: api.Sink) api.ImageSubmitError!void {
        const s = cast(ctx);
        if (s.submit_error) |err| return err;
        s.record(request);
        s.calls += 1;
        if (s.inline_) return finish(s, id, request, sink);
        s.thread = std.Thread.spawn(.{}, finish, .{ s, id, request, sink }) catch return error.Busy;
    }
    /// The ids by value: the route's arena frees its prompt arrays as soon as the reply is written.
    fn record(s: *Stub, request: *const api.ImageRequest) void {
        s.prompt_len = @min(request.prompt.len, s.prompt_ids.len);
        @memcpy(s.prompt_ids[0..s.prompt_len], request.prompt[0..s.prompt_len]);
        s.negative_len = @min(request.negative.len, s.negative_ids.len);
        @memcpy(s.negative_ids[0..s.negative_len], request.negative[0..s.negative_len]);
        s.width = request.width;
        s.height = request.height;
        s.steps = request.steps;
        s.guidance = request.guidance;
        s.seed = request.seed;
    }
    fn finish(s: *Stub, id: api.Id, request: *const api.ImageRequest, sink: api.Sink) void {
        if (s.hold) while (!s.release.load(.acquire)) std.Thread.yield() catch {};
        request.png.appendSlice(request.alloc, png_bytes) catch {};
        request.result.* = .{ .bytes = png_bytes.len, .reason = s.reason };
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = s.reason, .message = s.message } });
    }
};

/// One request served on its own thread; the test reads the reply from the client end as it arrives.
const Served = struct {
    srv: *server.Server,
    text: Bytes = .{},
    client: std.c.fd_t,
    route_fd: std.c.fd_t,
    route: std.Thread,
    got: std.ArrayList(u8) = .empty,
    read_at: usize = 0,

    pub fn open(s: *Served, stub: *Stub, request_line: []const u8, body: []const u8) !void {
        var pair: [2]std.c.fd_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        s.* = .{ .srv = undefined, .client = pair[0], .route_fd = pair[1], .route = undefined };
        errdefer _ = std.c.close(s.client);
        bound(s.client);
        s.srv = server.Server.init(gpa, t.io, stub.engine(), s.text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null) catch |e| {
            _ = std.c.close(s.route_fd);
            return e;
        };
        errdefer s.srv.deinit();
        // from here the route thread owns route_fd and closes it on every exit (`serve`)
        s.route = std.Thread.spawn(.{}, serve, .{s}) catch |e| {
            _ = std.c.close(s.route_fd);
            return e;
        };
        errdefer {
            _ = std.c.shutdown(s.client, 1);
            s.route.join();
        }
        var head: [256]u8 = undefined;
        try writeAll(s.client, try std.fmt.bufPrint(&head, "{s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ request_line, body.len }));
        try writeAll(s.client, body);
    }

    fn serve(s: *Served) void {
        // every exit closes this end, so a client still writing the body gets EPIPE, not a full buffer for ever
        defer _ = std.c.close(s.route_fd);
        var conn = Conn.init(gpa, s.route_fd, "test") catch return;
        defer conn.deinit();
        conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const outcome = conn.readRequest(arena.allocator(), false) catch return;
        if (outcome == .ready) routes.dispatch(s.srv, &conn, arena.allocator());
        if (conn.close) _ = std.c.shutdown(s.route_fd, 1); // what the server loop does to a closing connection: EOF
    }

    /// Waits for the route, then for the engine's own thread.
    pub fn join(s: *Served, stub: *Stub) void {
        s.route.join();
        if (stub.thread) |thread| {
            thread.join();
            stub.thread = null;
        }
    }

    /// After `join`: the route thread has closed its own end.
    pub fn deinit(s: *Served) void {
        _ = std.c.close(s.client);
        s.got.deinit(gpa);
        s.srv.deinit();
    }

    /// Reads more of the reply; false at EOF or after `wait_ms` of silence.
    pub fn more(s: *Served, wait_ms: i32) !bool {
        var fds = [_]std.posix.pollfd{.{ .fd = s.client, .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&fds, wait_ms) == 0) return false;
        var tmp: [8192]u8 = undefined;
        const n = std.posix.read(s.client, &tmp) catch 0;
        if (n == 0) return false;
        try s.got.appendSlice(gpa, tmp[0..n]);
        return true;
    }

    pub fn readHead(s: *Served) ![]const u8 {
        while (std.mem.indexOf(u8, s.got.items, "\r\n\r\n") == null) if (!try s.more(2000)) return error.NoHead;
        const end = std.mem.indexOf(u8, s.got.items, "\r\n\r\n").? + 4;
        s.read_at = end;
        return s.got.items[0..end];
    }

    /// The rest of a reply the route has finished writing.
    pub fn readRest(s: *Served) ![]const u8 {
        while (try s.more(200)) {}
        return s.got.items[s.read_at..];
    }
};

/// How long one client write may wait for the route to read: far past the route's own 2 s read timeout.
const client_send_s = 10;

/// The client end's writes end: a send timeout, and a closed route an error (EPIPE) rather than SIGPIPE.
fn bound(fd: std.c.fd_t) void {
    const tv: std.c.timeval = .{ .sec = client_send_s, .usec = 0 };
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.SNDTIMEO, &tv, @sizeOf(std.c.timeval));
    if (@hasDecl(std.posix.SO, "NOSIGPIPE")) {
        const one: c_int = 1;
        _ = std.c.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.NOSIGPIPE, &one, @sizeOf(c_int));
    }
}

fn writeAll(fd: std.c.fd_t, data: []const u8) !void {
    const flags: u32 = if (@hasDecl(std.posix.MSG, "NOSIGNAL")) std.posix.MSG.NOSIGNAL else 0;
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.send(fd, data[sent..].ptr, data.len - sent, flags);
        if (n <= 0) return error.Closed;
        sent += @intCast(n);
    }
}

/// The reply one request got: its head and body, each owned by `gpa` and freed by the test.
const Reply = struct { head: []const u8, body: []const u8 };

/// One request through the real route, its reply read as it arrives; joins the route and the engine's thread.
fn ask(stub: *Stub, request_line: []const u8, body: []const u8) !Reply {
    var s: Served = undefined;
    try s.open(stub, request_line, body);
    // a reply that never comes still joins the route and frees the server, then fails the test
    const head, const rest = read: {
        errdefer {
            s.join(stub);
            s.deinit();
        }
        break :read .{ try s.readHead(), try s.readRest() };
    };
    s.join(stub);
    const reply = Reply{ .head = try gpa.dupe(u8, head), .body = try gpa.dupe(u8, rest) };
    s.deinit();
    return reply;
}

/// A 400 in the API's error shape: `invalid_request_error`, and a message naming `want`.
fn expectRefusal(reply: *const Reply, want: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expect(std.mem.indexOf(u8, reply.head, " 400 ") != null);
    const err = (try json.parse(a, reply.body)).ok.get("error").?;
    try t.expectEqualStrings("invalid_request_error", err.get("type").?.str().?);
    const got = err.get("message").?.str().?;
    if (std.mem.indexOf(u8, got, want) == null) {
        std.debug.print("refusal want={s} got={s}\n", .{ want, got });
        return error.TestUnexpectedResult;
    }
}

/// The reply's one picture, decoded from `data[0].b64_json`.
fn pngOf(body: []const u8) ![png_bytes.len]u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const data = (try json.parse(arena.allocator(), body)).ok.get("data").?.items().?;
    const encoded = data[0].get("b64_json").?.str().?;
    var out: [png_bytes.len]u8 = undefined;
    try std.base64.standard.Decoder.decode(&out, encoded);
    return out;
}

test "an accepted request reaches the engine once and the reply carries its PNG as base64" {
    var stub: Stub = .{ .inline_ = true };
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":\"fog\",\"size\":\"1024x1024\",\"steps\":8,\"guidance\":2,\"seed\":7}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try t.expect(std.mem.indexOf(u8, reply.head, " 200 ") != null);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = (try json.parse(arena.allocator(), reply.body)).ok;
    const data = root.get("data").?.items().?;
    try t.expectEqual(@as(usize, 1), data.len);
    var png: [png_bytes.len]u8 = undefined;
    try std.base64.standard.Decoder.decode(&png, data[0].get("b64_json").?.str().?);
    try t.expectEqualStrings(png_bytes, &png);
    const created = root.get("created").?;
    try t.expect(created == .int or created == .float);
    try t.expectEqual(@as(usize, 1), stub.calls);
    try t.expectEqual(@as(usize, 2), stub.prompt_len);
    try t.expectEqual(@as(u32, 'h'), stub.prompt_ids[0]);
    try t.expectEqual(@as(u32, 'i'), stub.prompt_ids[1]);
    try t.expectEqual(@as(usize, 3), stub.negative_len);
    try t.expectEqual(@as(u32, 'f'), stub.negative_ids[0]);
    try t.expectEqual(@as(u32, 'o'), stub.negative_ids[1]);
    try t.expectEqual(@as(u32, 'g'), stub.negative_ids[2]);
    try t.expectEqual(@as(u16, 1024), stub.width);
    try t.expectEqual(@as(u16, 1024), stub.height);
    try t.expectEqual(@as(u32, 8), stub.steps);
    try t.expectEqual(@as(f32, 2), stub.guidance);
    try t.expectEqual(@as(u64, 7), stub.seed);
}

test "omitted seed, steps, guidance, negative_prompt and size reach the engine as the defaults" {
    var stub: Stub = .{ .inline_ = true };
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\"}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try t.expect(std.mem.indexOf(u8, reply.head, " 200 ") != null);
    try t.expectEqual(@as(usize, 1), stub.calls);
    try t.expectEqual(@as(u64, 42), stub.seed);
    try t.expectEqual(@as(u32, 50), stub.steps);
    try t.expectEqual(@as(f32, 3), stub.guidance);
    try t.expectEqual(@as(usize, 0), stub.negative_len);
    try t.expectEqual(@as(u16, 1024), stub.width);
    try t.expectEqual(@as(u16, 1024), stub.height);
}

test "an omitted size past a smaller host's limits is the route's own refusal naming them" {
    var stub: Stub = .{ .caps = .{ .max_side = 768, .max_pixels = 589824, .max_steps = 200, .max_prompt_tokens = 4096 } };
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\"}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try expectRefusal(&reply, "the default size 1024x1024 does not fit this model's limits of 768 a side and 589824 pixels");
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "an omitted size within a smaller host's side but past its pixels is the route's own refusal" {
    var stub: Stub = .{ .caps = .{ .max_side = 1536, .max_pixels = 1000000, .max_steps = 200, .max_prompt_tokens = 4096 } };
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\"}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try expectRefusal(&reply, "the default size 1024x1024 does not fit this model's limits of 1536 a side and 1000000 pixels");
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "a sent size within a smaller host's limits still reaches the engine" {
    var stub: Stub = .{ .inline_ = true, .caps = .{ .max_side = 768, .max_pixels = 589824, .max_steps = 200, .max_prompt_tokens = 4096 } };
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\",\"size\":\"768x768\"}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try t.expect(std.mem.indexOf(u8, reply.head, " 200 ") != null);
    try t.expectEqual(@as(u16, 768), stub.width);
}

/// One refusal: the request body, and the text its message must name.
const Case = struct { body: []const u8, want: []const u8 };

const refusals = [_]Case{
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"n\":2}", .want = "n must be 1" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"n\":0}", .want = "n must be 1" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"size\":\"1000x1024\"}", .want = "size 1000 must be a positive multiple of 16" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"size\":\"2048x2048\"}", .want = "size 2048 is past the 1536 limit a side" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"size\":\"1552x1024\"}", .want = "size 1552 is past the 1536 limit a side" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"size\":\"1536x1040\"}", .want = "size 1536x1040 is past the 1572864 pixel limit" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"response_format\":\"url\"}", .want = "response_format must be one of: b64_json" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"steps\":0}", .want = "steps must be from 1 to 200" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"steps\":201}", .want = "steps must be from 1 to 200" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"guidance\":0.5}", .want = "guidance must be a finite number from 1" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"guidance\":1e300}", .want = "guidance must be a finite number from 1" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"quality\":\"high\"}", .want = "quality is not supported by this model" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"style\":\"none\"}", .want = "style is not supported by this model" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"user\":\"u1\"}", .want = "user is not supported by this model" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"hi\",\"top_p\":0.9}", .want = "unknown images request field: top_p" },
    .{ .body = "{\"model\":\"m\",\"prompt\":\"\"}", .want = "prompt must be a non-empty string" },
    .{ .body = "{\"model\":\"other\",\"prompt\":\"hi\"}", .want = "model 'other' is not served here: send m" },
};

test "each refusal is a 400 invalid_request_error that names the field or the limit" {
    for (refusals) |case| {
        var stub: Stub = .{};
        const reply = try ask(&stub, images_route, case.body);
        defer gpa.free(reply.head);
        defer gpa.free(reply.body);
        try expectRefusal(&reply, case.want);
        try t.expectEqual(@as(usize, 0), stub.calls); // settled before the engine, so nothing was queued
    }
}

test "a prompt past 16 KiB is refused by that limit" {
    var big: [16385]u8 = undefined;
    @memset(&big, 'a');
    const body = try std.fmt.allocPrint(gpa, "{{\"model\":\"m\",\"prompt\":\"{s}\"}}", .{&big});
    defer gpa.free(body);
    var stub: Stub = .{};
    const reply = try ask(&stub, images_route, body);
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try expectRefusal(&reply, "prompt must be at most 16384 bytes");
}

test "a negative_prompt past 16 KiB or past the token limit is refused by the prompt's own limits" {
    var big: [16385]u8 = undefined;
    @memset(&big, 'a');
    const body = try std.fmt.allocPrint(gpa, "{{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":\"{s}\"}}", .{&big});
    defer gpa.free(body);
    var stub: Stub = .{};
    const reply = try ask(&stub, images_route, body);
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try expectRefusal(&reply, "prompt must be at most 16384 bytes");
    var long: [5000]u8 = undefined;
    @memset(&long, 'a');
    const tokens = try std.fmt.allocPrint(gpa, "{{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":\"{s}\"}}", .{&long});
    defer gpa.free(tokens);
    var second: Stub = .{};
    const too_long = try ask(&second, images_route, tokens);
    defer gpa.free(too_long.head);
    defer gpa.free(too_long.body);
    try expectRefusal(&too_long, "a prompt of 5000 tokens exceeds the maximum allowed length of 4096 tokens");
}

test "a negative_prompt that is not valid UTF-8 is refused by its name" {
    var stub: Stub = .{};
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":\"\\ud800\"}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try expectRefusal(&reply, "negative_prompt must be valid UTF-8 text");
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "a null negative_prompt is refused like the other optional fields" {
    var stub: Stub = .{};
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":null}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try expectRefusal(&reply, "negative_prompt must be a string");
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "a failed generation is a 500 whose message is the engine's own" {
    var stub: Stub = .{ .inline_ = true, .reason = .failed, .message = "the pass did not converge" };
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\"}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try t.expect(std.mem.indexOf(u8, reply.head, " 500 ") != null);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const err = (try json.parse(arena.allocator(), reply.body)).ok.get("error").?;
    try t.expectEqualStrings("the pass did not converge", err.get("message").?.str().?);
}

test "a text route on an image engine is a 400 that names the images route" {
    var stub: Stub = .{ .inline_ = true };
    const reply = try ask(&stub, chat_route, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try t.expect(std.mem.indexOf(u8, reply.head, " 400 ") != null);
    try t.expect(std.mem.indexOf(u8, reply.body, "POST /v1/images/generations") != null);
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "POST /v1/ocr on an image engine is the same 400, before the engine" {
    var stub: Stub = .{ .inline_ = true };
    const reply = try ask(&stub, "POST /v1/ocr", "{\"model\":\"m\",\"document\":{\"type\":\"image_url\",\"image_url\":\"data:image/png;base64,iVBORw0KGgo=\"}}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try t.expect(std.mem.indexOf(u8, reply.head, " 400 ") != null);
    try t.expect(std.mem.indexOf(u8, reply.body, "POST /v1/images/generations") != null);
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "an images request to a text-only engine is the unknown-path 404" {
    var stub: Stub = .{ .caps = null };
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\"}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try t.expect(std.mem.indexOf(u8, reply.head, " 404 ") != null);
    try t.expect(std.mem.indexOf(u8, reply.body, "unknown path") != null);
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "error.Busy from the engine is a 503 with Retry-After" {
    var stub: Stub = .{ .inline_ = true, .submit_error = error.Busy };
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\"}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try t.expect(std.mem.indexOf(u8, reply.head, " 503 ") != null);
    try t.expect(std.mem.indexOf(u8, reply.head, "Retry-After:") != null);
    try t.expect(std.mem.indexOf(u8, reply.body, "the engine is busy") != null);
}

test "wait reason ledger: an images engine's Busy counts as an engine_busy refusal beside the same 503" {
    var stub: Stub = .{ .inline_ = true, .submit_error = error.Busy };
    var s: Served = undefined;
    try s.open(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\"}");
    defer s.deinit();
    const head = try s.readHead();
    try t.expect(std.mem.indexOf(u8, head, " 503 ") != null);
    try t.expect(std.mem.indexOf(u8, try s.readRest(), "the engine is busy") != null);
    s.join(&stub);
    const counted = s.srv.load.read().refusals;
    try t.expectEqual(@as(u64, 1), counted[@backingInt(api.load_ledger.reasons.Refusal.engine_busy)]);
    var total: u64 = 0;
    for (counted) |n| total += n;
    try t.expectEqual(@as(u64, 1), total);
}

test "error.InvalidRequest from the engine is a 400 naming the refusal" {
    var stub: Stub = .{ .inline_ = true, .submit_error = error.InvalidRequest };
    const reply = try ask(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\"}");
    defer gpa.free(reply.head);
    defer gpa.free(reply.body);
    try t.expect(std.mem.indexOf(u8, reply.head, " 400 ") != null);
    try t.expect(std.mem.indexOf(u8, reply.body, "invalid_request_error") != null);
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "nothing reaches the client before the engine's finished event" {
    var stub: Stub = .{ .hold = true };
    var s: Served = undefined;
    try s.open(&stub, images_route, "{\"model\":\"m\",\"prompt\":\"hi\"}");
    try t.expect(!try s.more(20)); // nothing on the wire: the held engine has not sent finished yet
    stub.release.store(true, .release);
    try t.expect(std.mem.indexOf(u8, try s.readHead(), " 200 ") != null);
    try t.expectEqualStrings(png_bytes, &(try pngOf(try s.readRest())));
    s.join(&stub);
    try t.expectEqual(@as(usize, 1), stub.calls); // the one submit, ended by the finished event the reply waited for
    s.deinit();
}

test "a route that returns before reading the body ends the client's write at once, not when the buffer fills" {
    // a request line of four words: readRequest answers 400 from the head alone and the route returns unread
    const body = try gpa.alloc(u8, 4 << 20); // past any socket buffer, Linux's or macOS's
    defer gpa.free(body);
    @memset(body, 'a');
    var stub: Stub = .{};
    var s: Served = undefined;
    const start = std.Io.Clock.awake.now(t.io).toNanoseconds();
    try t.expectError(error.Closed, s.open(&stub, "POST /v1/images/generations extra", body));
    const took_ms = @divTrunc(std.Io.Clock.awake.now(t.io).toNanoseconds() - start, std.time.ns_per_ms);
    // the route closed its end, so the write failed at once; a route that kept it open leaves only the send timeout
    try t.expect(took_ms < client_send_s * std.time.ms_per_s / 2);
    try t.expectEqual(@as(usize, 0), stub.calls);
}
