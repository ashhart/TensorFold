//! The HTTP embedding service refuses generators and retains borrowed output through delayed cancellation.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const embeddings = @import("embeddings.zig");
const Server = @import("server.zig").Server;
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const t = std.testing;
const cap: api.EmbeddingInfo = .{ .dimensions = 3, .vocabulary = 256, .max_batch_tokens = 8, .max_inputs = 4, .mrl_dimensions = &.{ 1, 2 } };
const Provider = struct {
    mode: enum { normal, generator, partial, tokens, delayed } = .normal,
    calls: usize = 0,
    cancelled: bool = false,
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
        return .{ .text_generation = cast(ptr).mode == .generator, .context_window = 4, .embeddings = if (cast(ptr).mode == .generator) null else cap };
    }
    fn submit(ptr: *anyopaque, id: api.Id, ask: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const p = cast(ptr);
        p.calls += 1;
        if (p.mode == .delayed) {
            p.pending = ask;
            p.sink = sink;
            return;
        }
        const batch = ask.embedding orelse return error.InvalidRequest;
        if (p.mode == .partial) {
            batch.output[0] = 1;
        } else for (batch.ends, 0..) |_, i| {
            const row: [3]f32 = if (i == 0) .{ 3, 4, 12 } else .{ 5, 12, 0 };
            @memcpy(batch.output[i * 3 ..][0..3], &row);
        }
        if (p.mode == .tokens) sink.event(sink.ctx, id, &.{ .tokens = &.{120} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
    }
    fn cancel(ptr: *anyopaque, id: api.Id) void {
        const p = cast(ptr);
        p.cancelled = true;
        p.thread = std.Thread.spawn(.{}, finishLater, .{ p, id }) catch {
            p.pending = null;
            p.sink.event(p.sink.ctx, id, &.{ .finished = .{ .reason = .cancelled } });
            return;
        };
    }
    fn finishLater(p: *Provider, id: api.Id) void {
        std.Io.sleep(t.io, .fromMilliseconds(10), .awake) catch {};
        const ask = p.pending.?;
        p.borrowed_alive = std.mem.eql(u32, ask.prompt, &.{ 1, 2 }) and ask.embedding.?.output.len == 3;
        ask.embedding.?.output[0] = 7;
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
const Present = struct {
    pub fn check(_: @This()) bool {
        return false;
    }
};
fn server(p: *Provider) !*Server {
    return Server.init(t.allocator, t.io, p.engine(), ByteText.model(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
}

test "embeddings HTTP: capability guards refuse generators before submission and serialize valid results" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    var p: Provider = .{ .mode = .generator };
    const gen = try server(&p);
    defer gen.deinit();
    const body = (try json.parse(cx.a, "{\"model\":\"m\",\"input\":[[3,4],[5,12]],\"dimensions\":2}")).ok;
    try t.expectError(error.Refused, embeddings.reply(gen, &cx, body, Present{}));
    try t.expectEqual(@as(usize, 0), p.calls);
    p.mode = .normal;
    const embed = try server(&p);
    defer embed.deinit();
    const result = try embeddings.reply(embed, &cx, body, Present{});
    try t.expectEqual(@as(usize, 2), result.get("data").?.array.len);
    try t.expectEqualStrings("4", result.get("usage").?.get("prompt_tokens").?.int);
    try t.expectEqual(@as(usize, 1), p.calls);
    try t.expect(!embed.info.text_generation);
}

test "embeddings HTTP: missing vector writes and generated tokens cannot become successful embeddings" {
    for ([_]@TypeOf(@as(Provider, .{}).mode){ .partial, .tokens }) |mode| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var cx: errors.Cx = .{ .a = arena.allocator() };
        var p: Provider = .{ .mode = mode };
        const srv = try server(&p);
        defer srv.deinit();
        const body = (try json.parse(cx.a, "{\"input\":[1,2]}")).ok;
        try t.expectError(error.Refused, embeddings.reply(srv, &cx, body, Present{}));
        try t.expectEqual(errors.Kind.server, cx.kind);
    }
}

test "embeddings HTTP: disconnect waits for terminal completion before borrowed request storage ends" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    var p: Provider = .{ .mode = .delayed };
    defer p.deinit();
    const srv = try server(&p);
    defer srv.deinit();
    const Gone = struct {
        calls: usize = 0,
        pub fn check(g: *@This()) bool {
            g.calls += 1;
            return g.calls > 1;
        }
    };
    var gone: Gone = .{};
    const body = (try json.parse(cx.a, "{\"input\":[1,2]}")).ok;
    try t.expectError(error.Cancelled, embeddings.reply(srv, &cx, body, &gone));
    try t.expect(p.cancelled and p.borrowed_alive and p.pending == null);
}

test "embeddings HTTP: generation routes refuse the encoder and consume only their own body" {
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
    for ([_][]const u8{ "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/v1/audio/speech", "/v1/decisions", "/v1/systemone" }) |path| {
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
        try t.expect(c.response_complete and !c.close);
        try t.expectEqualStrings("NEXT", c.buf[c.start..c.end]);
        try t.expect(std.mem.indexOf(u8, writer.bytes.items, "400 Bad Request") != null);
        try t.expect(std.mem.indexOf(u8, writer.bytes.items, "POST /v1/embeddings") != null);
    }
    try t.expectEqual(@as(usize, 0), p.calls);
    try t.expectEqual(@as(u32, 0), srv.generating.load(.seq_cst));
}
