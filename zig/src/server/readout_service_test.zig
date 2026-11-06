//! A readout engine's server: its capabilities checked at start, every other route refused, the prompt-only wait.
const std = @import("std");
const api = @import("engine_api");
const wait = @import("prompt_only_wait.zig");
const Server = @import("server.zig").Server;
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const t = std.testing;
const readout = @import("lanes").readout;
const cap: readout.Info = .{ .max_rows = 64, .max_questions = 4, .max_options = 8 };

const Provider = struct {
    mode: enum { normal, tokens, stop, delayed } = .normal,
    text_generation: bool = false,
    embeddings: ?api.EmbeddingInfo = null,
    calls: usize = 0,
    cancels: usize = 0,
    borrowed_alive: bool = false,
    pending: ?*const api.Request = null,
    sink: api.Sink = undefined,
    thread: ?std.Thread = null,
    fn engine(p: *Provider) api.Engine {
        return .{ .ctx = p, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn cast(ptr: *anyopaque) *Provider {
        return @ptrCast(@alignCast(ptr));
    }
    fn info(ptr: *anyopaque) api.Info {
        const p = cast(ptr);
        return .{ .text_generation = p.text_generation, .context_window = 64, .embeddings = p.embeddings, .readouts = cap };
    }
    fn submit(ptr: *anyopaque, id: api.Id, ask: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const p = cast(ptr);
        p.calls += 1;
        const spec = ask.readout orelse return error.InvalidRequest;
        if (p.mode == .delayed) {
            p.pending = ask;
            p.sink = sink;
            return;
        }
        for (spec.logits, 0..) |*l, i| l.* = @intCast(0x3f80 + i);
        if (p.mode == .tokens) sink.event(sink.ctx, id, &.{ .tokens = &.{120} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (p.mode == .stop) .stop else .length } });
    }
    fn cancel(ptr: *anyopaque, id: api.Id) void {
        const p = cast(ptr);
        p.cancels += 1;
        if (p.cancels > 1) return;
        p.thread = std.Thread.spawn(.{}, finishLater, .{ p, id }) catch {
            p.pending = null;
            p.sink.event(p.sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
            return;
        };
    }
    fn finishLater(p: *Provider, id: api.Id) void {
        std.Io.sleep(t.io, .fromMilliseconds(10), .awake) catch {};
        const ask = p.pending.?;
        p.borrowed_alive = ask.prompt.len == 4 and ask.readout.?.logits.len == 2;
        ask.readout.?.logits[0] = 7;
        p.pending = null;
        p.sink.event(p.sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn deinit(p: *Provider) void {
        if (p.thread) |thread| thread.join();
    }
};

fn server(p: *Provider) !*Server {
    return Server.init(t.allocator, t.io, p.engine(), ByteText.model(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
}

const Present = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};

const prompt = [_]u32{ 1, 2, 3, 4 };

fn specFor(logits: []u16) readout.Spec {
    return .{ .question_spans = &.{.{ 0, 1 }}, .question_types = &.{1}, .option_spans = &.{ .{ 1, 2 }, .{ 2, 4 } }, .option_question = &.{ 0, 0 }, .logits = logits };
}

/// The server refuses to start; one that started anyway is closed, so the failure cannot leave its threads behind.
fn refusedStart(p: *Provider) !void {
    const srv = server(p) catch |err| return t.expectEqual(error.InvalidReadoutCapabilities, err);
    srv.deinit();
    return error.TestUnexpectedResult;
}

test "readout server: readouts beside text generation or embeddings refuse to start" {
    var p: Provider = .{ .text_generation = true };
    try refusedStart(&p);
    p = .{ .embeddings = .{ .dimensions = 2, .vocabulary = 256, .max_batch_tokens = 8 } };
    try refusedStart(&p);
    p = .{};
    const srv = try server(&p);
    defer srv.deinit();
    try t.expect(srv.info.readouts != null and !srv.info.text_generation);
}

test "readout server: every engine route but System One refuses with 400 naming it, reading only its own body" {
    const http = @import("http_conn.zig");
    const Writer = struct {
        bytes: std.ArrayList(u8) = .empty,
        fn write(ptr: *anyopaque, bytes: []const u8) error{Closed}!void {
            const w: *@This() = @ptrCast(@alignCast(ptr));
            w.bytes.appendSlice(t.allocator, bytes) catch return error.Closed;
        }
    };
    var p: Provider = .{};
    const srv = try server(&p);
    defer srv.deinit();
    const paths = [_][]const u8{ "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/v1/audio/speech", "/v1/decisions", "/v1/embeddings", "/v1/slide/learn", "/v1/imprint/load", "/v1/systemone", "/systemone", "/v1/tokenize" };
    for (paths) |path| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var c = try http.Conn.init(t.allocator, -1, "fixture");
        defer c.deinit();
        var writer: Writer = .{};
        defer writer.bytes.deinit(t.allocator);
        c.write_hooks = .{ .ctx = &writer, .write = Writer.write };
        c.method = "POST";
        c.path = path;
        c.version = "HTTP/1.1";
        c.close = false;
        c.quiet_log = true;
        var headers = [_]http.Header{.{ .name = "Content-Length", .value = "2" }};
        c.headers = &headers;
        @memcpy(c.buf[0..6], "{}NEXT");
        c.end = 6;
        srv.dispatch(&c, arena.allocator());
        try t.expect(c.response_complete);
        try t.expectEqualStrings("NEXT", c.buf[c.start..c.end]);
        const refused = std.mem.indexOf(u8, writer.bytes.items, "use POST /v1/systemone") != null;
        const passes = std.mem.endsWith(u8, path, "/systemone") or std.mem.endsWith(u8, path, "/tokenize");
        try t.expectEqual(!passes, refused);
        if (refused) try t.expect(std.mem.indexOf(u8, writer.bytes.items, "400 Bad Request") != null);
    }
    try t.expectEqual(@as(usize, 0), p.calls);
    try t.expectEqual(@as(u32, 0), srv.generating.load(.seq_cst));
}

test "readout server: the prompt-only wait gives the answer, and fails on tokens or another finish" {
    for ([_]@TypeOf(@as(Provider, .{}).mode){ .normal, .tokens, .stop }) |mode| {
        var p: Provider = .{ .mode = mode };
        var logits: [2]u16 = @splat(0);
        const s = specFor(&logits);
        const ask: api.Request = .{ .prompt = &prompt, .max_tokens = 0, .drafts = false, .readout = &s };
        const ended = try wait.run(p.engine(), t.io, 1, &ask, Present{});
        try t.expectEqual(@as(wait.Ended, if (mode == .normal) .done else .failed), ended);
        try t.expectEqualSlices(u16, &.{ 0x3f80, 0x3f81 }, &logits);
    }
    var p: Provider = .{};
    var logits: [2]u16 = undefined;
    const s = specFor(logits[0..1]);
    const invalid: api.Request = .{ .prompt = &prompt, .max_tokens = 0, .drafts = false, .readout = &s };
    try t.expectError(error.InvalidRequest, wait.run(p.engine(), t.io, 1, &invalid, Present{}));
    try t.expectEqual(@as(usize, 0), p.calls);
}

test "readout server: a gone client cancels once and the wait outlives the engine's last write" {
    var p: Provider = .{ .mode = .delayed };
    defer p.deinit();
    const Gone = struct {
        pub fn check(_: @This()) bool {
            return true;
        }
    };
    var logits: [2]u16 = @splat(0);
    const s = specFor(&logits);
    const ask: api.Request = .{ .prompt = &prompt, .max_tokens = 0, .drafts = false, .readout = &s };
    try t.expectEqual(wait.Ended.cancelled, try wait.run(p.engine(), t.io, 1, &ask, Gone{}));
    try t.expect(p.borrowed_alive and p.pending == null);
    try t.expectEqual(@as(usize, 1), p.cancels);
    try t.expectEqual(@as(u16, 7), logits[0]);
}
