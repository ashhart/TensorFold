//! Tool-call ids on every route, whole and streamed: a replay under one key names its calls again, others apart.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const anthropic = @import("anthropic.zig");
const responses = @import("responses.zig");
const preflight = @import("stream_preflight_test.zig");
const Conn = @import("http_conn.zig").Conn;
const call_ids = @import("ids.zig");
const tool_id_key = @import("tool_id_key.zig");
const A = std.mem.Allocator;
const t = std.testing;

/// Bytes as tokens, like the preflight text; the prompt renders as "prompt" and the messages it got are kept.
const Text = struct {
    rendered: std.ArrayList(u8) = .empty,
    gpa: A,

    fn text(x: *Text) model_text.Text {
        return .{ .ctx = x, .vtable = &.{ .encode = preflight.TestText.encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn decode(_: *anyopaque, a: A, ids: []const u32) A.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *b| b.* = @intCast(id);
        return out;
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: A, id: u32) A.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(ctx: *anyopaque, a: A, messages: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        const x: *Text = @ptrCast(@alignCast(ctx));
        try x.rendered.appendSlice(x.gpa, try json.stringify(a, messages, .{ .ascii = false }));
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

fn bytesOf(comptime s: []const u8) [s.len]u32 {
    var out: [s.len]u32 = undefined;
    for (s, &out) |b, *id| id.* = b;
    return out;
}

/// Two Qwen XML calls, which stream as they are written (``Streamer.begin``).
const xml_reply = bytesOf("<tool_call>\n<function=lookup>\n<parameter=q>\na\n</parameter>\n</function>\n</tool_call>\n" ++
    "<tool_call>\n<function=lookup>\n<parameter=q>\nb\n</parameter>\n</function>\n</tool_call>");

/// The preflight's call engine on a ``warm_turns`` engine, so each reply also renders the next turn's prompt.
const Warm = struct {
    inner: preflight.CallEngine = .{},

    fn engine(w: *Warm) api.Engine {
        return .{ .ctx = w, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096, .warm_turns = true };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const w: *Warm = @ptrCast(@alignCast(ctx));
        const e = w.inner.engine();
        return e.vtable.submit(e.ctx, id, r, sink);
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const e = @as(*Warm, @ptrCast(@alignCast(ctx))).inner.engine();
        e.vtable.cancel(e.ctx, id);
    }
    fn status(ctx: *anyopaque, out: *api.Status, ids: []u32) void {
        const e = @as(*Warm, @ptrCast(@alignCast(ctx))).inner.engine();
        e.vtable.status(e.ctx, out, ids);
    }
    fn memory(ctx: *anyopaque, device: bool) ?api.Memory {
        const e = @as(*Warm, @ptrCast(@alignCast(ctx))).inner.engine();
        return e.vtable.memory(e.ctx, device);
    }
};

const Route = enum { chat, messages, responses };
const Engine = enum { json, xml }; // JSON envelopes arrive whole at the end of a stream; XML ones stream

const Wire = struct {
    a: A,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

/// What the client read for one request, and the messages its template got for it.
const Sent = struct { bytes: []const u8, rendered: []const u8 };

/// The fixed key most tests share, as one operator key file gives every server that reads it.
fn shared() call_ids.Key {
    return .of(call_ids.test_key_text);
}

/// One server over the call engine, keyed by ``key`` as a key file gives it, or by a key of its own when null.
const Fixture = struct {
    backend: Warm = .{},
    text: Text,
    key: ?call_ids.Key,
    srv: *server.Server = undefined,

    fn init(f: *Fixture, a: A, engine: Engine, key: ?call_ids.Key) !void {
        f.* = .{ .text = .{ .gpa = a }, .key = key };
        if (engine == .xml) f.backend.inner.tokens = &xml_reply;
        f.srv = try server.Server.init(t.allocator, t.io, f.backend.engine(), f.text.text(), .{ .served_name = "test-model", .model_ids = &.{"test-model"}, .enable_thinking = false, .use_drafts = false, .tool_id_key = if (f.key) |*k| k else null }, null);
        if (f.key) |*k| try t.expect(std.mem.allEqual(u8, std.mem.asBytes(k), 0)); // the server took the one copy
        try t.expect(f.srv.config.tool_id_key == null);
    }

    fn deinit(f: *Fixture) void {
        f.srv.deinit();
    }

    fn post(f: *Fixture, a: A, route: Route, body: []const u8) !Sent {
        const before = f.text.rendered.items.len;
        var wire: Wire = .{ .a = a };
        var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
        var c: Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = if (route == .messages) "/v1/messages" else "", .headers = &headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
        switch (route) {
            .chat => openai.post(f.srv, &c, a, true),
            .messages => anthropic.post(f.srv, &c, a),
            .responses => responses.post(f.srv, &c, a),
        }
        try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 200 "));
        return .{ .bytes = wire.bytes.items, .rendered = try a.dupe(u8, f.text.rendered.items[before..]) };
    }
};

/// What the client read for ``body`` on ``route`` from a fresh server under the ``shared`` key.
fn post(a: A, route: Route, engine: Engine, body: []const u8) !Sent {
    var f: Fixture = undefined;
    try f.init(a, engine, shared());
    defer f.deinit();
    return f.post(a, route, body);
}

/// Each JSON value ``"prefix`` + ``n`` lowercase hex + ``"`` in ``bytes`` once, first seen first; other values fail.
fn idsIn(a: A, bytes: []const u8, prefix: []const u8, n: usize) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    const quoted = try std.mem.concat(a, u8, &.{ "\"", prefix });
    while (std.mem.indexOfPos(u8, bytes, at, quoted)) |q| : (at = q + 1) {
        const before = std.mem.trimEnd(u8, bytes[0..q], " ");
        if (before.len == 0 or before[before.len - 1] != ':') continue; // a key such as "call_id"
        const id = bytes[q + 1 ..][0 .. prefix.len + n];
        for (id[prefix.len..]) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return error.NotHex;
        if (bytes[q + 1 + id.len] != '"') return error.WrongLength;
        for (out.items) |seen| {
            if (std.mem.eql(u8, seen, id)) break;
        } else try out.append(a, id);
    }
    return out.items;
}

const Ids = struct { calls: []const []const u8, items: []const []const u8 };

/// The reply's call ids (``call_`` + 24 hex) and Responses' function_call item ids (``fc_`` + 32 hex).
fn idsOf(a: A, sent: Sent) !Ids {
    return .{ .calls = try idsIn(a, sent.bytes, "call_", 24), .items = try idsIn(a, sent.bytes, "fc_", 32) };
}

const tool_head = "\"model\":\"test-model\",\"max_tokens\":200,";
const chat_tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
const messages_tools = ",\"tools\":[{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\"}}]";
const responses_tools = ",\"tools\":[{\"type\":\"function\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}]";

/// A request on ``route`` with ``conversation`` as its messages (Responses' input) and the lookup tool.
fn requestWith(a: A, route: Route, conversation: []const u8, stream: bool) ![]const u8 {
    const tools = switch (route) {
        .chat => chat_tools,
        .messages => messages_tools,
        .responses => responses_tools,
    };
    const field = if (route == .responses) "\"input\":" else "\"messages\":";
    return std.mem.concat(a, u8, &.{ "{" ++ tool_head, field, conversation, tools, ",\"stream\":", if (stream) "true" else "false", "}" });
}

/// A request on ``route`` whose user says ``word``, streamed or not.
fn request(a: A, route: Route, word: []const u8, stream: bool) ![]const u8 {
    const user = try std.mem.concat(a, u8, &.{ "{\"role\":\"user\",\"content\":\"", word, "\"}" });
    return requestWith(a, route, if (route == .responses) try std.mem.concat(a, u8, &.{ "\"", word, "\"" }) else try std.mem.concat(a, u8, &.{ "[", user, "]" }), stream);
}

fn expectSame(x: []const []const u8, y: []const []const u8) !void {
    try t.expectEqual(x.len, y.len);
    for (x, y) |p, q| try t.expectEqualStrings(p, q);
}

fn expectApart(x: []const []const u8, y: []const []const u8) !void {
    for (x) |p| for (y) |q| try t.expect(!std.mem.eql(u8, p, q));
}

test "a replayed request names its tool calls as before, and its stream names them as its whole reply does" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Route{ .chat, .messages, .responses }) |route| for ([_]Engine{ .json, .xml }) |engine| {
        const whole = try idsOf(a, try post(a, route, engine, try request(a, route, "x", false)));
        try t.expectEqual(@as(usize, 2), whole.calls.len); // two calls, two ids
        try t.expectEqual(@as(usize, if (route == .responses) 2 else 0), whole.items.len);
        try expectSame(whole.calls, (try idsOf(a, try post(a, route, engine, try request(a, route, "x", false)))).calls);
        const stream_sent = try post(a, route, engine, try request(a, route, "x", true));
        const streamed = try idsOf(a, stream_sent);
        try expectSame(whole.calls, streamed.calls);
        for (streamed.calls) |id| try t.expect(std.mem.indexOf(u8, stream_sent.rendered, id) != null); // the warm next turn
        try expectSame(whole.items, streamed.items);
        try expectSame(streamed.calls, (try idsOf(a, try post(a, route, engine, try request(a, route, "x", true)))).calls);
    };
}

test "two calls in one reply and the calls of two different requests have different ids" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Route{ .chat, .messages, .responses }) |route| for ([_]bool{ false, true }) |stream| {
        const x = try idsOf(a, try post(a, route, .xml, try request(a, route, "x", stream)));
        const y = try idsOf(a, try post(a, route, .xml, try request(a, route, "y", stream)));
        try t.expectEqual(@as(usize, 2), x.calls.len);
        try t.expectEqual(@as(usize, 2), y.calls.len);
        try expectApart(x.calls[0..1], x.calls[1..]);
        try expectApart(x.calls, y.calls);
        try expectApart(x.items, y.items);
    };
}

test "a chat reply's call ids are the request's own derivation, call_ and 24 hex digits" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sent = try request(a, .chat, "x", false);
    const got = try idsOf(a, try post(a, .chat, .json, sent));
    const key = shared();
    var calls: call_ids.Calls = .{ .key = &key, .body = (try json.parse(a, sent)).ok };
    for (got.calls, 0..) |id, i| try t.expectEqualStrings(try calls.id(a, 0, i), id);
}

/// The client's own ids in a conversation's history, as each route takes them.
fn withHistory(a: A, route: Route, stream: bool) ![]const u8 {
    return requestWith(a, route, switch (route) {
        .chat => "[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"client-kept-1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"client-kept-1\",\"content\":\"found\"}]",
        .messages => "[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"client-kept-1\",\"name\":\"lookup\",\"input\":{}}]},{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"client-kept-1\",\"content\":\"found\"}]}]",
        .responses => "[{\"role\":\"user\",\"content\":\"x\"},{\"type\":\"function_call\",\"id\":\"fc_client\",\"call_id\":\"client-kept-1\",\"name\":\"lookup\",\"arguments\":\"{}\"},{\"type\":\"function_call_output\",\"call_id\":\"client-kept-1\",\"output\":\"found\"}]",
    }, stream);
}

test "ids a client sends back reach the template as it sent them, and the reply's new calls get their own" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Route{ .chat, .messages, .responses }) |route| for ([_]bool{ false, true }) |stream| {
        const sent = try post(a, route, .xml, try withHistory(a, route, stream));
        const kept = std.mem.count(u8, sent.rendered, "\"client-kept-1\"");
        try t.expect(kept > 0 and kept % 2 == 0); // the call and its result, in each render (the prompt, the warm next turn)
        const got = try idsOf(a, sent);
        try t.expectEqual(@as(usize, 2), got.calls.len);
        try t.expect(std.mem.indexOf(u8, sent.bytes, "client-kept-1") == null); // a reply never renames history
        try expectApart(got.calls, (try idsOf(a, try post(a, route, .xml, try request(a, route, "x", stream)))).calls);
    };
}

test "without a key file each server keys its own ids: a replay to it names its calls again, another server apart" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]Route{ .chat, .messages, .responses }) |route| for ([_]Engine{ .json, .xml }) |engine| {
        var one: Fixture = undefined;
        try one.init(a, engine, null);
        defer one.deinit();
        var two: Fixture = undefined;
        try two.init(a, engine, null);
        defer two.deinit();
        const whole = try request(a, route, "x", false);
        const first = try idsOf(a, try one.post(a, route, whole));
        try t.expectEqual(@as(usize, 2), first.calls.len);
        try expectSame(first.calls, (try idsOf(a, try one.post(a, route, whole))).calls);
        const streamed = try idsOf(a, try one.post(a, route, try request(a, route, "x", true)));
        try expectSame(first.calls, streamed.calls);
        try expectSame(first.items, streamed.items);
        const other = try idsOf(a, try two.post(a, route, whole));
        try t.expectEqual(@as(usize, 2), other.calls.len);
        try expectApart(first.calls, other.calls);
        try expectApart(first.items, other.items);
        try expectApart(first.calls, (try idsOf(a, try post(a, route, engine, whole))).calls); // nor the shared key's
    };
}

/// ``data`` in an owner-only file under ``dir``; its path as the setting would name it.
fn keyFile(a: A, dir: std.testing.TmpDir, file: []const u8, data: []const u8) ![]const u8 {
    try dir.dir.writeFile(t.io, .{ .sub_path = file, .data = data });
    const path = try std.fmt.allocPrintSentinel(a, ".zig-cache/tmp/{s}/{s}", .{ dir.sub_path, file }, 0);
    try t.expectEqual(@as(c_int, 0), @import("sys").c.fchmodat(@import("sys").posix.AT.FDCWD, path, 0o600, 0));
    return path;
}

fn keyed(a: A, path: []const u8) !call_ids.Key {
    var problem: []const u8 = "";
    return (try tool_id_key.load(a, path, &problem)).?;
}

const secret = "SECRET-served-tool-id-key-0123456789abcdef";

test "servers reading the same key file name a request's calls the same way, and another key file's apart" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // the key files are owner-only by mode bits
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const same = try keyFile(a, tmp, "same", secret ++ "\n");
    const other = try keyFile(a, tmp, "other", secret ++ "-other\n");
    for ([_]Route{ .chat, .messages, .responses }) |route| for ([_]Engine{ .json, .xml }) |engine| for ([_]bool{ false, true }) |stream| {
        var got: [3]Ids = undefined;
        for (&got, [_][]const u8{ same, same, other }) |*out, path| {
            var f: Fixture = undefined;
            try f.init(a, engine, try keyed(a, path));
            defer f.deinit();
            out.* = try idsOf(a, try f.post(a, route, try request(a, route, "x", stream)));
        }
        try t.expectEqual(@as(usize, 2), got[0].calls.len);
        try expectSame(got[0].calls, got[1].calls);
        try expectSame(got[0].items, got[1].items);
        try expectApart(got[0].calls, got[2].calls);
        try expectApart(got[0].items, got[2].items);
    };
}

const Capture = struct {
    bytes: std.ArrayList(u8) = .empty,
    fn line(ctx: *anyopaque, text: []const u8) void {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        self.bytes.appendSlice(t.allocator, text) catch @panic("log capture");
    }
};

test "no reply, log line or refusal carries a byte of the key file, in its text or as hex" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // the key files are owner-only by mode bits
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var capture: Capture = .{};
    defer capture.bytes.deinit(t.allocator);
    @import("log.zig").Test.sink = .{ .ctx = &capture, .call = Capture.line };
    defer @import("log.zig").Test.sink = null;
    var said: std.ArrayList(u8) = .empty;
    var problem: []const u8 = "";
    for ([_][]const u8{ "SECRET-short\n", secret }) |data| {
        const path = try keyFile(a, tmp, "refused", data);
        _ = @import("sys").c.fchmodat(@import("sys").posix.AT.FDCWD, try a.dupeSentinel(u8, path, 0), 0o644, 0);
        try t.expectError(error.Refused, tool_id_key.load(a, path, &problem));
        try said.appendSlice(a, problem);
    }
    try t.expectError(error.Refused, tool_id_key.load(a, secret, &problem)); // the key itself in the setting
    try said.appendSlice(a, problem);
    var f: Fixture = undefined;
    try f.init(a, .xml, try keyed(a, try keyFile(a, tmp, "key", secret ++ "\n")));
    defer f.deinit();
    for ([_]Route{ .chat, .messages, .responses }) |route| for ([_]bool{ false, true }) |stream| {
        const sent = try f.post(a, route, try request(a, route, "x", stream));
        try t.expectEqual(@as(usize, 2), (try idsOf(a, sent)).calls.len);
        try said.appendSlice(a, sent.bytes);
        try said.appendSlice(a, sent.rendered);
    };
    try said.appendSlice(a, capture.bytes.items);
    const hex = std.fmt.bytesToHex(secret[0..16].*, .lower);
    for ([_][]const u8{ "SECRET", secret[8..24], &hex, hex[0..16] }) |part| try t.expect(std.mem.indexOf(u8, said.items, part) == null);
    try t.expect(std.mem.indexOf(u8, said.items, tool_id_key.name) != null); // the refusals were read
}

/// One way a reply's calls reach the client, the reply that takes it, and how many calls the client gets.
const Path = struct {
    name: []const u8,
    reply: []const u8 = "", // bytes as tokens; Kimi's come from its golden reply
    after: []const u8 = "", // Cortheon's next round, once the prompt holds its results
    fields: []const u8 = "", // the request's fields after its messages and tools
    max_tokens: usize = 2000,
    calls: usize,
    kind: enum { markup, cortheon, kimi } = .markup,
};

const xml_a = "<tool_call>\n<function=lookup>\n<parameter=q>\na\n</parameter>\n</function>\n</tool_call>\n";
const xml_cut = "<tool_call>\n<function=lookup>\n<parameter=q>\nThe history of";
const glm_a = "<tool_call>lookup<arg_key>q</arg_key><arg_value>a</arg_value></tool_call>\n";
const glm_cut = "<tool_call>lookup<arg_key>q</arg_key><arg_value>The history of";
const dsml_call = "<｜DSML｜invoke name=\"lookup\"><｜DSML｜parameter name=\"q\" string=\"true\">a</｜DSML｜parameter></｜DSML｜invoke>";
const search = "<tool_call>\n<function=cortheon_search>\n<parameter=query>\nzig\n</parameter>\n</function>\n</tool_call>\n";

const paths = [_]Path{
    .{ .name = "a call max_tokens cuts", .reply = xml_a ++ xml_cut ++ " the bicycle\n</parameter>\n</function>\n</tool_call>", .max_tokens = (xml_a ++ xml_cut).len, .calls = 2 },
    .{ .name = "the one-call policy", .reply = xml_a ++ xml_a, .fields = ",\"parallel_tool_calls\":false", .calls = 1 },
    .{ .name = "GLM", .reply = glm_a ++ glm_a, .calls = 2 },
    .{ .name = "a GLM call max_tokens cuts", .reply = glm_a ++ glm_cut ++ " the bicycle</arg_value></tool_call>", .max_tokens = (glm_a ++ glm_cut).len, .calls = 2 },
    .{ .name = "DSML", .reply = "<｜DSML｜tool_calls>" ++ dsml_call ++ dsml_call ++ "</｜DSML｜tool_calls>", .calls = 2 },
    .{ .name = "Cortheon's client call beside its own", .reply = search ++ search ++ search ++ search ++ search ++ search, .after = search ++ xml_a, .calls = 1, .kind = .cortheon },
    .{ .name = "Kimi's call_N", .calls = 2, .kind = .kimi },
};

/// ``first``, or ``after`` once the prompt holds Cortheon's results: one character a step, or all at once.
const Scripted = struct {
    first: []const u32,
    after: []const u32,
    batched: bool,
    kimi: bool,

    fn engine(s: *Scripted) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        const s: *Scripted = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 32768, .thinking_budget = !s.kimi };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *Scripted = @ptrCast(@alignCast(ctx));
        const tokens = if (std.mem.indexOf(u32, r.prompt, &bytesOf("Web results")) != null) s.after else s.first;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var at: usize = 0;
        while (at < tokens.len) {
            const n = if (s.batched) tokens.len else if (tokens[at] > 0xff) 1 else std.unicode.utf8ByteSequenceLength(@intCast(tokens[at])) catch 1;
            sink.event(sink.ctx, id, &.{ .tokens = tokens[at..][0..n] });
            at += n;
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The call ids a chat client read, in order: the whole reply's, or those of the stream's deltas that open a call.
const Read = struct {
    a: A,
    ids: std.ArrayList([]const u8) = .empty,
    status: u16 = 0,

    fn out(r: *Read) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = open, .event = event, .reply = whole } };
    }
    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        @as(*Read, @ptrCast(@alignCast(ctx))).status = 200;
    }
    fn keep(r: *Read, calls: ?json.Value) error{Closed}!void {
        for ((calls orelse return).items() orelse return) |c| if (c.strField("id")) |id| r.ids.append(r.a, r.a.dupe(u8, id) catch return error.Closed) catch return error.Closed;
    }
    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const r: *Read = @ptrCast(@alignCast(ctx));
        const choices = ((payload orelse return).get("choices") orelse return).items() orelse return;
        if (choices.len > 0) try r.keep((choices[0].get("delta") orelse return).get("tool_calls"));
    }
    fn whole(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const r: *Read = @ptrCast(@alignCast(ctx));
        r.status = status;
        const choices = (payload.get("choices") orelse return).items() orelse return;
        if (choices.len > 0) r.keep((choices[0].get("message") orelse return).get("tool_calls")) catch {};
    }
};

/// ``p``'s request to one server under the shared key, whole and streamed, each twice: all four name the same calls.
fn expectPath(p: Path, batched: bool) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const kimi = if (p.kind == .kimi) try @import("kimi_text_test.zig").synthetic(t.allocator) else null;
    defer if (kimi) |k| k.deinit();
    var plain: preflight.TestText = .{};
    var rendered: @import("cortheon/cortheon_test.zig").Text = .{};
    const text = switch (p.kind) {
        .markup => plain.text(),
        .cortheon => rendered.text(),
        .kimi => kimi.?.text(),
    };
    var engine: Scripted = .{ .first = try preflight.TestText.encode(undefined, a, p.reply, false), .after = try preflight.TestText.encode(undefined, a, p.after, false), .batched = batched, .kimi = kimi != null };
    if (kimi != null) for ((try json.parse(a, @embedFile("fixtures/kimi/golden.json"))).ok.get("cases").?.array) |c| {
        if (std.mem.eql(u8, c.strField("id").?, "tool-roundtrip")) engine.first = try @import("kimi_reply_test.zig").assistant(a, text, c);
    };
    var key = shared();
    const srv = try server.Server.init(t.allocator, t.io, engine.engine(), text, .{ .served_name = "test-model", .model_ids = &.{"test-model"}, .enable_thinking = p.kind == .kimi, .use_drafts = false, .tool_id_key = &key }, null);
    defer srv.deinit();
    var web: @import("cortheon/cortheon_test.zig").Web = .{};
    if (p.kind == .cortheon) srv.cortheon = try @import("cortheon/cortheon.zig").State.initWith(t.allocator, t.io, web.web());
    const head = try std.fmt.allocPrint(a, "{{\"model\":\"test-model\",\"max_tokens\":{d},\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}]{s}{s},\"stream\":", .{ p.max_tokens, chat_tools, p.fields });
    var got: [4][]const []const u8 = undefined;
    for (&got, [_][]const u8{ "false}", "false}", "true}", "true}" }) |*ids, stream| {
        var read: Read = .{ .a = a };
        var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
        openai.run(srv, a, read.out(), .{ .conn = &conn }, true, (try json.parse(a, try std.mem.concat(a, u8, &.{ head, stream }))).ok);
        try t.expectEqual(@as(u16, 200), read.status);
        ids.* = read.ids.items;
    }
    try t.expectEqual(p.calls, got[0].len);
    for (got[1..]) |again| try expectSame(got[0], again); // a replay, the stream and the stream's replay
    const same = shared();
    var calls: call_ids.Calls = .{ .key = &same, .body = (try json.parse(a, try std.mem.concat(a, u8, &.{ head, "false}" }))).ok };
    for (got[0], 0..) |id, i| if (p.kind == .kimi) {
        try t.expect(std.mem.startsWith(u8, id, "call_") and id.len < 8); // the index Kimi wrote, not one of ours
        for (id[5..]) |ch| try t.expect(std.ascii.isDigit(ch));
    } else try t.expectEqualStrings(try calls.id(a, 0, i), id);
}

test "on every path a call takes, a replay names its calls again and the stream names them as the whole reply does" {
    var failed: usize = 0;
    for (paths) |p| for ([_]bool{ false, true }) |batched| expectPath(p, batched) catch |e| {
        std.debug.print("tool-call ids on {s} ({s}): {s}\n", .{ p.name, if (batched) "batched" else "per character", @errorName(e) });
        failed += 1;
    };
    try t.expectEqual(@as(usize, 0), failed);
}
