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
        s.srv = try server.Server.init(gpa, t.io, stub.engine(), s.text.text(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        var head: [256]u8 = undefined;
        try writeAll(s.client, try std.fmt.bufPrint(&head, "{s} HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{ request_line, body.len }));
        try writeAll(s.client, body);
        s.route = try std.Thread.spawn(.{}, serve, .{s});
    }

    fn serve(s: *Served) void {
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

    pub fn deinit(s: *Served) void {
        _ = std.c.close(s.client);
        _ = std.c.close(s.route_fd);
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

fn writeAll(fd: std.c.fd_t, data: []const u8) !void {
    var sent: usize = 0;
    while (sent < data.len) {
        const n = std.c.write(fd, data[sent..].ptr, data.len - sent);
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
    const head = try s.readHead();
    const rest = try s.readRest();
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
