//! Ollama routes over a real loopback socket with the CPU fake engine: replies end, NDJSON is whole, tokens agree.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const server = @import("server.zig");
const auth = @import("auth.zig");
const listener = @import("listener.zig");
const model_text = @import("model_text.zig");
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const Text = struct {
    const table = blk: {
        var base = ByteText.model().vtable.*;
        base.render = render;
        base.decode = decode;
        break :blk base;
    };
    fn model() model_text.Text {
        return .{ .ctx = undefined, .vtable = &table };
    }
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) ![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (out, ids) |*byte, id| byte.* = @intCast(id);
        return out;
    }
};
const KeyFile = @import("auth_scope_test.zig").KeyFile;
const posix = @import("sys").posix;
const t = std.testing;
const Allocator = std.mem.Allocator;

const Engine = struct {
    embed: bool = false,
    fail_after_tokens: bool = false,
    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn cast(ptr: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ptr));
    }
    fn info(ptr: *anyopaque) api.Info {
        const e = cast(ptr);
        if (e.embed) return .{
            .context_window = 4096,
            .name = "fake-model",
            .text_generation = false,
            .embeddings = .{ .dimensions = 2, .vocabulary = 256, .max_batch_tokens = 128 },
        };
        return .{ .context_window = 4096, .name = "fake-model" };
    }
    fn submit(ptr: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = cast(ptr);
        if (e.embed) {
            if (request.embedding) |batch| {
                for (batch.output) |*v| v.* = 0.5;
            }
            sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
            return;
        }
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 2, 7, 1 } });
        if (e.fail_after_tokens) return error.Closed;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const Harness = struct {
    srv: *server.Server = undefined,
    listener: listener.Listener = undefined,
    accept: std.Thread = undefined,
    stop: std.atomic.Value(bool) = .init(false),
    log_path: ?[]const u8 = null,

    fn start(h: *Harness, engine: *Engine, keys: ?*auth.Store) !void {
        h.srv = try server.Server.init(t.allocator, t.io, engine.engine(), Text.model(), .{
            .served_name = "fake-model",
            .model_ids = &.{"fake-model"},
            .request_log = h.log_path,
            .enable_thinking = false,
            .use_drafts = false,
            .keep_warm_s = 0,
        }, keys);
        errdefer h.srv.deinit();
        h.listener = try listener.Listener.open(.{ .ip4 = .loopback(0) });
        errdefer h.listener.close();
        h.accept = try std.Thread.spawn(.{}, server.Server.serve, .{ h.srv, h.listener, &h.stop });
    }
    fn deinit(h: *Harness) void {
        h.stop.store(true, .release);
        h.accept.join();
        var waited: usize = 0;
        while (h.srv.open_connections.load(.acquire) > 0) : (waited += 1) {
            if (waited == 30000) @panic("ollama fixture did not quiesce");
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
        h.listener.close();
        h.srv.deinit();
    }
    fn exchange(h: *Harness, a: Allocator, raw: []const u8) ![]const u8 {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.listener.port()) };
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&address, &storage);
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.Socket;
        const fd: posix.socket_t = @intCast(rc);
        defer _ = posix.system.close(fd);
        if (posix.errno(posix.system.connect(fd, &storage.any, len)) != .SUCCESS) return error.Connect;
        var sent: usize = 0;
        while (sent < raw.len) {
            const n = posix.system.write(fd, raw[sent..].ptr, raw.len - sent);
            if (posix.errno(n) != .SUCCESS or n == 0) return error.Write;
            sent += @intCast(n);
        }
        var out: std.ArrayList(u8) = .empty;
        var buffer: [4096]u8 = undefined;
        while (true) {
            var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
            if (try posix.poll(&fds, 30000) == 0) return error.Timeout;
            const n = posix.system.read(fd, &buffer, buffer.len);
            if (posix.errno(n) != .SUCCESS) return error.Read;
            if (n == 0) return out.items;
            try out.appendSlice(a, buffer[0..@intCast(n)]);
        }
    }
    fn ask(h: *Harness, a: Allocator, path: []const u8, key: ?[]const u8, body: []const u8) ![]const u8 {
        const auth_header = if (key) |k| try std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{k}) else "";
        const raw = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: localhost\r\n{s}Content-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ path, auth_header, body.len, body });
        return h.exchange(a, raw);
    }

    fn get(h: *Harness, a: Allocator, path: []const u8, key: ?[]const u8) ![]const u8 {
        const auth_header = if (key) |k| try std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{k}) else "";
        const raw = try std.fmt.allocPrint(a, "GET {s} HTTP/1.1\r\nHost: localhost\r\n{s}Connection: close\r\n\r\n", .{ path, auth_header });
        return h.exchange(a, raw);
    }
};

fn bodyOf(reply: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return "";
    return reply[at + 4 ..];
}

fn statusIs(reply: []const u8, code: u16) bool {
    var buf: [32]u8 = undefined;
    const want = std.fmt.bufPrint(&buf, "HTTP/1.1 {d} ", .{code}) catch return false;
    return std.mem.startsWith(u8, reply, want);
}

fn lines(a: Allocator, body: []const u8) ![]json.Value {
    var out: std.ArrayList(json.Value) = .empty;
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const parsed = json.parse(a, line) catch return error.BadNdjson;
        if (parsed != .ok) return error.BadNdjson;
        try out.append(a, parsed.ok);
    }
    return out.items;
}

const chat_body = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"stream\":false,\"options\":{\"temperature\":0,\"num_predict\":8,\"seed\":1234}}";
const generate_body = "{\"model\":\"fake-model\",\"prompt\":\"H\",\"stream\":false,\"options\":{\"temperature\":0,\"num_predict\":8,\"seed\":1234}}";

test "ollama HTTP: non-stream chat and generate replies end with one complete done line" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();
    for ([_][]const u8{ "/api/chat", "/api/generate" }) |path| {
        const body = if (std.mem.eql(u8, path, "/api/chat")) chat_body else generate_body;
        const reply = try h.ask(a, path, null, body);
        try t.expect(statusIs(reply, 200));
        const rows = try lines(a, bodyOf(reply));
        try t.expectEqual(@as(usize, 1), rows.len); // the socket closed: no hang, one final line
        const row = rows[0];
        try t.expect(row.get("done").?.bool);
        for ([_][]const u8{ "done_reason", "prompt_eval_count", "eval_count", "prompt_eval_duration", "eval_duration" }) |key|
            try t.expect(row.get(key) != null);
    }
}

test "ollama HTTP: a streaming chat sends false deltas then one done line" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();
    const raw = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"stream\":true,\"options\":{\"temperature\":0}}";
    const reply = try h.ask(a, "/api/chat", null, raw);
    try t.expect(statusIs(reply, 200));
    const rows = try lines(a, bodyOf(reply));
    try t.expect(rows.len >= 2);
    var done: usize = 0;
    for (rows, 0..) |row, i| {
        if (row.get("done").?.bool) {
            done += 1;
            try t.expectEqual(rows.len - 1, i);
            try t.expect(row.get("done_reason") != null);
        } else try t.expect(row.get("message") != null);
    }
    try t.expectEqual(@as(usize, 1), done);
}

test "ollama HTTP: seeded chat and completions return the same token ids" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();
    const ollama = try h.ask(a, "/api/chat", null, chat_body);
    try t.expect(statusIs(ollama, 200));
    const rows = try lines(a, bodyOf(ollama));
    const ollama_ids = rows[0].get("token_ids").?.array;
    const openai_body = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"max_tokens\":8,\"temperature\":0,\"seed\":1234,\"return_token_ids\":true}";
    const openai = try h.ask(a, "/v1/chat/completions", null, openai_body);
    try t.expect(statusIs(openai, 200));
    const parsed = json.parse(a, bodyOf(openai)) catch return error.BadJson;
    const openai_ids = parsed.ok.get("tensorfold").?.get("token_ids").?.array;
    try t.expectEqual(ollama_ids.len, openai_ids.len);
    for (ollama_ids, openai_ids) |x, y| try t.expectEqualStrings(x.int, y.int);
}

test "ollama HTTP: show answers over POST, tags has no empty fields, a read key gets tags but not chat" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var keys_file = try KeyFile.init("monitor[read]: read-secret\nops: ops-secret\n");
    defer keys_file.deinit();
    var keys: auth.Store = undefined;
    var problem: []const u8 = "";
    keys = try auth.Store.init(t.allocator, &.{}, "", keys_file.path, false, &problem);
    defer keys.deinit();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, &keys);
    defer h.deinit();

    try t.expect(statusIs(try h.ask(a, "/api/show", "ops-secret", "{\"model\":\"fake-model\"}"), 200));
    try t.expect(statusIs(try h.ask(a, "/api/chat", "read-secret", chat_body), 403));
    const tags = try h.get(a, "/api/tags", "read-secret");
    try t.expect(statusIs(tags, 200));
    const parsed = json.parse(a, bodyOf(tags)) catch return error.BadJson;
    const models = parsed.ok.get("models").?.array;
    try t.expect(models.len > 0);
    try t.expect(models[0].get("modified_at") == null);
    try t.expect(models[0].get("size") == null);
}

fn openaiWhole(a: Allocator, h: *Harness, body: []const u8) ![]const u8 {
    const reply = try h.ask(a, "/v1/chat/completions", null, body);
    const parsed = json.parse(a, bodyOf(reply)) catch return error.BadJson;
    const choices = parsed.ok.get("choices").?.array;
    if (choices.len == 0) return error.BadJson;
    if (choices[0].get("message")) |m| return m.strField("content") orelse "";
    return choices[0].strField("text") orelse "";
}

fn rfc3339Shape(text: []const u8) bool {
    return text.len == 20 and text[4] == '-' and text[7] == '-' and text[10] == 'T' and
        text[13] == ':' and text[16] == ':' and text[19] == 'Z';
}

test "ollama HTTP: non-stream content, generate response, streamed join and created_at match" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();

    const openai_body = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"max_tokens\":8,\"temperature\":0,\"seed\":1234,\"return_token_ids\":true}";
    const whole = try openaiWhole(a, &h, openai_body);
    try t.expect(whole.len > 0);

    const chat = try h.ask(a, "/api/chat", null, chat_body);
    try t.expect(statusIs(chat, 200));
    const rows = try lines(a, bodyOf(chat));
    const content = rows[0].get("message").?.strField("content") orelse "";
    try t.expectEqualStrings(whole, content);
    try t.expect(rfc3339Shape(rows[0].strField("created_at") orelse ""));

    const generated = try h.ask(a, "/api/generate", null, generate_body);
    try t.expect(statusIs(generated, 200));
    const generated_rows = try lines(a, bodyOf(generated));
    try t.expectEqualStrings(whole, generated_rows[0].strField("response") orelse "");
    try t.expect(generated_rows[0].get("message") == null);

    const stream_raw = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"stream\":true,\"options\":{\"temperature\":0,\"num_predict\":8,\"seed\":1234}}";
    const stream = try h.ask(a, "/api/chat", null, stream_raw);
    const stream_rows = try lines(a, bodyOf(stream));
    var joined: std.ArrayList(u8) = .empty;
    for (stream_rows) |row| {
        const message = row.get("message") orelse continue;
        try joined.appendSlice(a, message.strField("content") orelse "");
    }
    try t.expectEqualStrings(whole, joined.items); // the final done line adds no text
    const final = stream_rows[stream_rows.len - 1];
    const final_text: []const u8 = final.get("message").?.strField("content") orelse "x";
    try t.expect(final_text.len == 0);
}

test "ollama HTTP: an invalid value and an unknown option keep their status and name the cause" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();

    const hot = try h.ask(a, "/api/chat", null, "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"options\":{\"temperature\":\"hot\"}}");
    try t.expect(!statusIs(hot, 200));
    try t.expect(std.mem.indexOf(u8, bodyOf(hot), "error") != null);

    const unknown = try h.ask(a, "/api/chat", null, "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"options\":{\"num_gpu\":1}}");
    try t.expect(!statusIs(unknown, 200));
    try t.expect(std.mem.indexOf(u8, bodyOf(unknown), "num_gpu") != null);
}

test "ollama HTTP: two requests on one kept-alive connection both answer" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();
    const show_body = "{\"model\":\"fake-model\"}";
    const raw = try std.fmt.allocPrint(
        a,
        "POST /api/show HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}GET /api/version HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        .{ show_body.len, show_body },
    );
    const reply = try h.exchange(a, raw);
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, reply, "HTTP/1.1 200"));
    try t.expect(std.mem.indexOf(u8, reply, "501") == null);
}

test "ollama HTTP: embed answers in Ollama's shape with an embedding fake" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{ .embed = true };
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();
    const reply = try h.ask(a, "/api/embed", null, "{\"model\":\"fake-model\",\"input\":\"hello\"}");
    try t.expect(statusIs(reply, 200));
    const parsed = json.parse(a, bodyOf(reply)) catch return error.BadJson;
    const embeddings = parsed.ok.get("embeddings").?.array;
    try t.expectEqual(@as(usize, 1), embeddings.len);
    try t.expectEqual(@as(usize, 2), embeddings[0].array.len);
    try t.expect(parsed.ok.get("object") == null);
}

test "ollama HTTP: a stream that fails after its tokens ends with an error line, not a success line" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{ .fail_after_tokens = true };
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();
    const raw = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}],\"stream\":true}";
    const reply = try h.ask(a, "/api/chat", null, raw);
    try t.expect(statusIs(reply, 200));
    const rows = try lines(a, bodyOf(reply));
    try t.expect(rows.len >= 1);
    var saw_error = false;
    for (rows, 0..) |row, i| {
        if (row.get("done")) |done| try t.expect(!done.bool);
        if (row.get("error") != null) {
            saw_error = true;
            try t.expectEqual(rows.len - 1, i);
        }
    }
    try t.expect(saw_error);
}

test "ollama HTTP: format, tools and images are mapped or refused by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();
    const prefix = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}]";
    const json_format = try std.fmt.allocPrint(a, "{s},\"format\":\"json\"}}", .{prefix});
    try t.expect(statusIs(try h.ask(a, "/api/chat", null, json_format), 200));
    const schema = try std.fmt.allocPrint(a, "{s},\"format\":{{\"type\":\"object\",\"properties\":{{}}}}}}", .{prefix});
    try t.expect(statusIs(try h.ask(a, "/api/chat", null, schema), 200));
    const tools = try std.fmt.allocPrint(a, "{s},\"tools\":[{{\"type\":\"function\",\"function\":{{\"name\":\"lookup\",\"description\":\"d\",\"parameters\":{{\"type\":\"object\",\"properties\":{{}}}}}}}}]}}", .{prefix});
    try t.expect(statusIs(try h.ask(a, "/api/chat", null, tools), 200));
    for ([_][]const u8{ "format\":\"xml", "tools\":{}", "images\":[\"AQ==\"" }) |field| {
        const bad = try std.fmt.allocPrint(a, "{s},{s}}}", .{ prefix, field });
        const reply = try h.ask(a, "/api/chat", null, bad);
        try t.expect(statusIs(reply, 400));
        try t.expect(std.mem.indexOf(u8, bodyOf(reply), field[0..std.mem.indexOfScalar(u8, field, '\"').?]) != null);
    }
}

test "ollama HTTP: stream takes only true, false or nothing" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var engine: Engine = .{};
    var h: Harness = .{};
    try h.start(&engine, null);
    defer h.deinit();
    const prefix = "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"H\"}]";
    for ([_][]const u8{ "\"stream\":\"yes\"", "\"stream\":1", "\"stream\":null" }) |value| {
        const raw = try std.fmt.allocPrint(a, "{s},{s}}}", .{ prefix, value });
        const reply = try h.ask(a, "/api/chat", null, raw);
        try t.expect(statusIs(reply, 400));
        try t.expect(std.mem.indexOf(u8, bodyOf(reply), "stream") != null);
    }
    const absent = try std.fmt.allocPrint(a, "{s}}}", .{prefix});
    try t.expect(statusIs(try h.ask(a, "/api/chat", null, absent), 200));
}

test "ollama HTTP: one request writes one request-log line" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    var engine: Engine = .{};
    var h: Harness = .{ .log_path = path };
    try h.start(&engine, null);
    defer h.deinit();
    try t.expect(statusIs(try h.ask(a, "/api/chat", null, chat_body), 200));
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 16));
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| if (line.len > 0) {
        count += 1;
    };
    try t.expectEqual(@as(usize, 1), count);
}
