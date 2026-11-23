//! The chat command against the benchmark's fake server: piped, a conversation, errors, Ctrl-C and its own server.
const std = @import("std");
const platform = @import("sys");
const json = @import("json");
const chat = @import("chat_cli.zig");
const history = @import("chat_cli_history.zig");
const stop = @import("benchmark_stop.zig");
const fake = @import("benchmark_fake.zig");
const stream = @import("benchmark_stream.zig");
const Allocator = std.mem.Allocator;

pub const Run = struct {
    a: Allocator,
    env: std.process.Environ.Map,
    dir: []const u8,
    out: std.Io.Writer.Allocating,
    err: std.Io.Writer.Allocating,
    stdin: std.Io.Reader = .fixed(""),
    /// What each command reads from a fixed stdin, unless ``reader`` names another.
    input: []const u8 = "",
    reader: ?*std.Io.Reader = null,
    interactive: bool = false,
    self_exe: ?[]const u8 = null,

    pub fn init(a: Allocator, tmp: *std.testing.TmpDir) !Run {
        var r: Run = .{ .a = a, .env = .init(a), .dir = try fake.tmpPath(a, tmp), .out = .init(a), .err = .init(a) };
        try r.env.put("PATH", "/usr/bin:/bin");
        for ([_][]const u8{ "TMPDIR", "HOME" }) |name| try r.env.put(name, r.dir);
        return r;
    }

    pub fn command(r: *Run, argv: []const []const u8) !u8 {
        r.out.clearRetainingCapacity();
        r.err.clearRetainingCapacity();
        r.stdin = .fixed(r.input);
        const ctx: chat.Context = .{
            .a = r.a,
            .io = std.testing.io,
            .environ = &r.env,
            .out = &r.out.writer,
            .err = &r.err.writer,
            .input = r.reader orelse &r.stdin,
            .interactive = r.interactive,
            .terminal = false,
            .self_exe = r.self_exe,
        };
        return chat.command(&ctx, argv);
    }

    fn path(r: *const Run, name: []const u8) ![]const u8 {
        return std.fs.path.join(r.a, &.{ r.dir, name });
    }
};

pub const models_doc = "{\"object\": \"list\", \"data\": [{\"id\": \"example-model\", \"capabilities\": {\"thinking_switch\": true}}, {\"id\": \"other-model\"}]}";

fn chunk(a: Allocator, delta: []const u8) ![]const u8 {
    return fake.event(a, try std.fmt.allocPrint(a, "{{\"choices\": [{{\"index\": 0, \"delta\": {s}, \"finish_reason\": null}}]}}", .{delta}));
}

/// Reasoning, then "Hello there!" in two events, the second cut across two writes, a finish and [DONE].
fn streamed(a: Allocator) ![]const fake.Write {
    const second = try chunk(a, "{\"content\": \"there!\"}");
    const w = try a.alloc(fake.Write, 3);
    const head = try fake.headOf(a, "200 OK", "text/event-stream");
    w[0] = .{ .bytes = try std.mem.concat(a, u8, &.{ head, try chunk(a, "{\"role\": \"assistant\", \"reasoning_content\": \"The user greets me.\"}"), try chunk(a, "{\"content\": \"Hello \"}"), second[0 .. second.len / 2] }), .pause_ms = 30 };
    w[1] = .{ .bytes = second[second.len / 2 ..] };
    w[2] = .{ .bytes = try std.mem.concat(a, u8, &.{ try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"stop\"}]}"), try fake.event(a, "[DONE]") }) };
    return w;
}

/// A server that lists ``models`` and answers the n-th chat request with ``replies[n]`` (the last one after).
pub fn world(a: Allocator, models: []const u8, replies: []const []const fake.Write) !*fake.Fake {
    return worldKeyed(a, models, replies, null);
}

/// ``world`` as a keyed server, wanting the key ``key_file`` holds on every route but GET /health.
fn worldKeyed(a: Allocator, models: []const u8, replies: []const []const fake.Write, key_file: ?[]const u8) !*fake.Fake {
    return fake.Fake.startKeyed(std.testing.allocator, try a.dupe(fake.Route, &.{
        .{ .method = "GET", .path = "/health", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"status\": \"ok\", \"warming\": false}")}) },
        .{ .method = "GET", .path = "/v1/models", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", models)}) },
        .{ .method = "POST", .path = "/v1/chat/completions", .replies = replies },
    }), key_file);
}

fn posts(f: *fake.Fake) usize {
    var n: usize = 0;
    for (0..f.count()) |i| n += @intFromBool(std.mem.eql(u8, f.request(i).method, "POST"));
    return n;
}

/// The i-th chat request's messages, written back as JSON.
fn sent(a: Allocator, f: *fake.Fake, i: usize) ![]const u8 {
    var seen: usize = 0;
    for (0..f.count()) |k| if (std.mem.eql(u8, f.request(k).method, "POST")) {
        if (seen == i) return json.stringify(a, (try json.parse(a, f.request(k).body)).ok.get("messages").?, .{});
        seen += 1;
    };
    return error.TestUnexpectedResult;
}

/// Polls ``ready`` every 5 ms for up to 5 s, so a broken path fails its test instead of hanging it.
fn waitFor(comptime ready: anytype, arg: anytype) bool {
    var waited: u32 = 0;
    while (waited < 5000) : (waited += 5) {
        if (ready(arg)) return true;
        fake.nap(5);
    }
    return false;
}

fn hasPost(f: *fake.Fake) bool {
    return posts(f) > 0;
}

fn handled(_: void) bool {
    return stop.requested() == null;
}

test "chat CLI: a piped message to --url asks for the server's first model and prints only its answer" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, models_doc, try a.dupe([]const fake.Write, &.{try streamed(a)}));
    defer f.stop();
    r.input = "  Say hello.\n";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "/v1") }));
    try std.testing.expectEqualStrings("Hello there!\n", r.out.written());
    try std.testing.expectEqualStrings("", r.err.written());
    try std.testing.expectEqual(@as(usize, 2), f.count());
    try std.testing.expectEqualStrings("/v1/models", f.request(0).path);
    try std.testing.expectEqualStrings("{\"model\": \"example-model\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}], \"stream\": true}", f.request(1).body);
    try std.testing.expectEqualStrings("TensorFold-chat/1", f.header(1, "user-agent").?);
    try std.testing.expect(f.header(0, "authorization") == null and f.header(1, "authorization") == null);
    try std.testing.expectEqualStrings("text/event-stream", f.header(1, "accept").?);
    r.input = "Say hello.";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, ""), "other-model", "--show-thinking", "--system", "Be brief.", "--temperature", "0", "--max-tokens", "32" }));
    try std.testing.expectEqualStrings("The user greets me.\n\nHello there!\n", r.out.written());
    try std.testing.expectEqualStrings("{\"model\": \"other-model\", \"messages\": [{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\": \"Say hello.\"}], " ++
        "\"stream\": true, \"max_tokens\": 32, \"temperature\": 0.0}", f.request(3).body);
}

test "chat CLI: a reply of hidden reasoning alone, or one cut at its token limit, says so on stderr" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const head = try fake.headOf(a, "200 OK", "text/event-stream");
    const cut = try fake.event(a, "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"length\"}]}");
    const thought = try a.dupe(fake.Write, &.{.{ .bytes = try std.mem.concat(a, u8, &.{ head, try chunk(a, "{\"reasoning_content\": \"Let me think\"}"), cut, try fake.event(a, "[DONE]") }) }});
    const f = try world(a, models_doc, try a.dupe([]const fake.Write, &.{thought}));
    defer f.stop();
    r.input = "hi";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
    try std.testing.expectEqualStrings("", r.out.written());
    try std.testing.expectEqualStrings("The reply stopped at its token limit; --max-tokens raises it.\n" ++
        "The reply held only reasoning, which is hidden; --show-thinking shows it.\n", r.err.written());
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, ""), "--show-thinking" }));
    try std.testing.expectEqualStrings("Let me think\n", r.out.written());
    try std.testing.expectEqualStrings("The reply stopped at its token limit; --max-tokens raises it.\n", r.err.written());
}

test "chat CLI: --url sends TENSORFOLD_API_KEY's first key as the bearer, and a 401 names the variable" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const key_file = try r.path("server-key");
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = key_file, .data = "example-key-1\n" });
    const f = try worldKeyed(a, models_doc, try a.dupe([]const fake.Write, &.{try streamed(a)}), key_file);
    defer f.stop();
    const url = try f.url(a, "");
    r.input = "hi";
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", url }));
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "tensorfold chat: no model to chat with: {s}/v1/models answered HTTP 401; " ++
        "set TENSORFOLD_API_KEY to a key the server accepts\n", .{url}), r.err.written());
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", url, "example-model" }));
    try std.testing.expectEqualStrings("tensorfold chat: HTTP 401: A valid API key is required; set TENSORFOLD_API_KEY to a key the server accepts\n", r.err.written());
    try r.env.put("TENSORFOLD_API_KEY", "wrong-key");
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", url, "example-model" }));
    try std.testing.expectEqualStrings("tensorfold chat: HTTP 401: A valid API key is required; the server refused the key in TENSORFOLD_API_KEY\n", r.err.written());
    try std.testing.expectEqual(@as(usize, 5), f.refusals());
    try std.testing.expectEqual(@as(usize, 0), f.count());
    try r.env.put("TENSORFOLD_API_KEY", " ,example-key-1, example-key-2");
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", url }));
    try std.testing.expectEqualStrings("Hello there!\n", r.out.written());
    try std.testing.expect(f.refusals() == 5 and f.count() == 2);
    for (0..f.count()) |i| try std.testing.expectEqualStrings("Bearer example-key-1", f.header(i, "authorization").?);
    try r.env.put("TENSORFOLD_API_KEY", "two words");
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", url }));
    try std.testing.expectEqualStrings("tensorfold chat: TENSORFOLD_API_KEY: expected Latin-1 keys without spaces or control characters, at most 256 bytes each\n", r.err.written());
    try std.testing.expect(f.refusals() == 5 and f.count() == 2);
}

test "chat CLI: attached responses redact credentials before every output boundary" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const first = "fixture";
    const second = "fixture-secret";
    const third = "fixture-\u{00c3}\u{00a9}";
    const latin1 = "fixture-\xc3\xa9";
    const json_reflection = try std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"{s} and {s}\"}}}}", .{ first, second });
    const event_head = try fake.headOf(a, "200 OK", "text/event-stream");
    const event_error = try fake.event(a, try std.fmt.allocPrint(a, "{{\"error\": {{\"message\": \"{s}\"}}}}", .{second}));
    const event_content = try std.mem.concat(a, u8, &.{ try chunk(a, "{\"content\": \"split-\"}"), try chunk(a, "{\"content\": \"secret\"}") });
    var filler: [299]u8 = undefined;
    @memset(&filler, 'x');
    const truncated = try std.mem.concat(a, u8, &.{ &filler, second });
    const quoted = try std.fmt.allocPrint(a, "\"{s}", .{filler[0..255]});
    @memset(filler[0..100], 'y');
    const escaped = try std.mem.concat(a, u8, &.{ "{\"error\":{\"message\":\"", filler[0..100], "\\\"", quoted[1..], "\"}}}" });
    const replies = try a.dupe([]const fake.Write, &.{
        try fake.jsonReply(a, "401 Unauthorized", json_reflection),
        try fake.jsonReply(a, "500 Internal Server Error", second),
        &.{.{ .bytes = try std.mem.concat(a, u8, &.{ event_head, event_error, try fake.event(a, "[DONE]") }) }},
        try fake.jsonReply(a, "500 Internal Server Error", truncated),
        try fake.jsonReply(a, "500 Internal Server Error", escaped),
        try fake.jsonReply(a, "500 Internal Server Error", "{\"error\":{\"message\":\"fixture-\\u00c3\\u00a9\"}}"),
        &.{.{ .bytes = try std.mem.concat(a, u8, &.{ event_head, event_content, try fake.event(a, "[DONE]") }) }},
    });
    const models = try std.fmt.allocPrint(a, "{{\"data\": [{{\"id\": \"{s}\"}}]}}", .{second});
    const f = try world(a, models, replies);
    defer f.stop();
    try r.env.put("TENSORFOLD_API_KEY", try std.mem.concat(a, u8, &.{ first, ",", second, ",", third, ",split-secret", ",", quoted }));
    r.input = "hi";
    var shown: std.ArrayList(u8) = .empty;
    for (0..7) |i| {
        try std.testing.expectEqual(@as(u8, if (i < 6) 1 else 0), try r.command(&.{ "--url", try f.url(a, "") }));
        try shown.appendSlice(a, r.out.written());
        try shown.appendSlice(a, r.err.written());
    }
    r.interactive = true;
    r.input = "/bye\n";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
    try shown.appendSlice(a, r.out.written());
    for ([_][]const u8{ first, second, third, latin1, "split-secret", quoted, quoted[0..100] }) |secret|
        try std.testing.expect(std.mem.indexOf(u8, shown.items, secret) == null);
    try std.testing.expect(std.mem.indexOf(u8, shown.items, "[redacted]") != null);
}

test "chat CLI: C1 controls in any key entry are refused before any request" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const replies = try a.dupe([]const fake.Write, &.{try streamed(a)});
    const f = try world(a, models_doc, replies);
    defer f.stop();
    r.input = "hi";
    var ok = true;
    for ([_][]const u8{ "fixture-\u{009b}", "good-key,fixture-\u{009b}" }) |key| {
        try r.env.put("TENSORFOLD_API_KEY", key);
        const code = try r.command(&.{ "--url", try f.url(a, "") });
        ok = ok and code == 1;
        ok = ok and f.count() == 0;
        ok = ok and std.mem.indexOf(u8, r.err.written(), "expected Latin-1 keys without spaces or control characters") != null;
        ok = ok and std.mem.indexOf(u8, r.err.written(), key) == null;
    }
    try std.testing.expect(ok);
}

test "chat CLI: --no-thinking sends the documented switch, and is refused before any turn when the model has none" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, models_doc, try a.dupe([]const fake.Write, &.{try streamed(a)}));
    defer f.stop();
    r.input = "hi\n";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, ""), "--no-thinking" }));
    const body = (try json.parse(a, f.request(1).body)).ok;
    try std.testing.expect(body.get("chat_template_kwargs").?.get("enable_thinking").?.bool == false);
    const g = try world(a, "{\"data\": [{\"id\": \"example-model\", \"capabilities\": {\"thinking_switch\": false}}]}", try a.dupe([]const fake.Write, &.{try streamed(a)}));
    defer g.stop();
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", try g.url(a, ""), "--no-thinking" }));
    try std.testing.expectEqualStrings("tensorfold chat: --no-thinking: this model's chat template cannot turn thinking off (its /v1/models thinking_switch is false)\n", r.err.written());
    try std.testing.expectEqual(@as(usize, 0), posts(g));
}

test "chat CLI: a refusal, an error event, an unreachable server and an empty message are printed plainly and exit 1" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const refusal = try fake.jsonReply(a, "400 Bad Request", "{\"error\": {\"message\": \"This server's maximum context length is 8 tokens\", \"type\": \"invalid_request_error\"}}");
    const failing = try a.dupe(fake.Write, &.{.{ .bytes = try std.mem.concat(a, u8, &.{ try fake.headOf(a, "200 OK", "text/event-stream"), try chunk(a, "{\"content\": \"Par\"}"), try fake.event(a, "{\"error\": {\"message\": \"generation failed\", \"type\": \"server_error\"}}"), try fake.event(a, "[DONE]") }) }});
    const f = try world(a, models_doc, try a.dupe([]const fake.Write, &.{ refusal, failing }));
    defer f.stop();
    r.input = "hi";
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", try f.url(a, "") }));
    try std.testing.expectEqualStrings("tensorfold chat: HTTP 400: This server's maximum context length is 8 tokens\n", r.err.written());
    try std.testing.expectEqualStrings("", r.out.written());
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", try f.url(a, "") }));
    try std.testing.expectEqualStrings("Par\n", r.out.written());
    try std.testing.expectEqualStrings("tensorfold chat: generation failed\n", r.err.written());
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", "http://127.0.0.1:1", "example-model" }));
    try std.testing.expectEqualStrings("tensorfold chat: could not reach the server at http://127.0.0.1:1/v1/chat/completions\n", r.err.written());
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", "http://127.0.0.1:1" }));
    try std.testing.expectEqualStrings("tensorfold chat: no model to chat with: http://127.0.0.1:1/v1/models could not connect; name one after the URL\n", r.err.written());
    const before = f.count();
    r.input = " \n\t\n";
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ "--url", try f.url(a, "") }));
    try std.testing.expectEqualStrings("tensorfold chat: stdin held no message to send\n", r.err.written());
    try std.testing.expectEqual(before, f.count());
    try std.testing.expectEqual(@as(u8, 2), try r.command(&.{ "--url", "http://127.0.0.1:1", "--", "--parallel", "1" }));
    try std.testing.expect(std.mem.endsWith(u8, r.err.written(), "tensorfold chat: error: flags after -- go to the server tensorfold chat starts; --url names one already running\n"));
}

test "chat CLI: a conversation keeps its turns, /system and /clear shape the next request, and a failed turn is not kept" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const ok = try streamed(a);
    const refusal = try fake.jsonReply(a, "503 Service Unavailable", "{\"error\": {\"message\": \"draining\"}}");
    const f = try world(a, models_doc, try a.dupe([]const fake.Write, &.{ ok, ok, refusal, ok }));
    defer f.stop();
    r.interactive = true;
    r.input = "first\n/system Be brief.\nsecond\n/clear\nlost\n/nope\n/help\n\n /etc holds what?\n";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
    try std.testing.expectEqual(@as(usize, 4), posts(f));
    try std.testing.expectEqualStrings("[{\"role\": \"user\", \"content\": \"first\"}]", try sent(a, f, 0));
    try std.testing.expectEqualStrings("[{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\": \"first\"}, " ++
        "{\"role\": \"assistant\", \"content\": \"Hello there!\"}, {\"role\": \"user\", \"content\": \"second\"}]", try sent(a, f, 1));
    try std.testing.expectEqualStrings("[{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\": \"lost\"}]", try sent(a, f, 2));
    try std.testing.expectEqualStrings("[{\"role\": \"system\", \"content\": \"Be brief.\"}, {\"role\": \"user\", \"content\": \"/etc holds what?\"}]", try sent(a, f, 3));
    const out = r.out.written();
    try std.testing.expect(std.mem.startsWith(u8, out, "Chatting with example-model. /help lists the commands; /bye or Ctrl-D ends the chat.\n> Hello there!\n> System prompt set.\n> "));
    for ([_][]const u8{ "Conversation cleared.\n", "Unknown command /nope; /help lists the commands.\n", history.help_text }) |want|
        try std.testing.expect(std.mem.indexOf(u8, out, want) != null);
    try std.testing.expect(std.mem.endsWith(u8, out, "> Hello there!\n> \n"));
    try std.testing.expectEqualStrings("tensorfold chat: HTTP 503: draining\n", r.err.written());
    r.input = "/bye\nnever sent\n";
    try std.testing.expectEqual(@as(u8, 0), try r.command(&.{ "--url", try f.url(a, "") }));
    try std.testing.expectEqual(@as(usize, 4), posts(f));
}

/// A conversation read from a pipe that a test thread writes into, as a terminal's lines arrive.
const Pipe = struct {
    fds: [2]std.posix.fd_t = .{ -1, -1 },
    buf: [256]u8 = undefined,
    file: std.Io.File.Reader = undefined,

    fn open(p: *Pipe) bool {
        if (platform.c.pipe(&p.fds) != 0) return false;
        p.file = (std.Io.File{ .handle = p.fds[0], .flags = .{ .nonblocking = false } }).readerStreaming(std.testing.io, &p.buf);
        return true;
    }

    fn close(p: *Pipe) void {
        for (p.fds) |fd| _ = platform.c.close(fd);
    }

    fn send(p: *const Pipe, text: []const u8) void {
        _ = platform.c.write(p.fds[1], text.ptr, text.len);
    }
};

test "chat CLI: Ctrl-C during a reply stops it at once, keeps what it printed, and the conversation goes on" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    stop.reset();
    defer stop.reset();
    var gate: std.Io.Event = .unset;
    const trickle = try a.dupe(fake.Write, &.{ .{ .bytes = try std.mem.concat(a, u8, &.{ try fake.headOf(a, "200 OK", "text/event-stream"), try chunk(a, "{\"content\": \"Once upon\"}") }), .after = &gate }, .{ .bytes = try chunk(a, "{\"content\": \" a time\"}") } });
    const f = try world(a, models_doc, try a.dupe([]const fake.Write, &.{ trickle, try streamed(a) }));
    defer {
        gate.set(io);
        f.stop();
    }
    var pipe: Pipe = .{};
    if (!pipe.open()) return error.SkipZigTest;
    defer pipe.close();
    r.interactive = true;
    r.reader = &pipe.file.interface;
    const Typist = struct {
        fn run(p: *const Pipe, fk: *fake.Fake, g: *std.Io.Event) void {
            p.send("Tell me a long story.\n");
            if (waitFor(hasPost, fk)) fake.nap(300);
            stop.request(.INT);
            if (!waitFor(handled, {})) g.set(std.testing.io); // a reply that never stopped finishes, and the test fails
            p.send("Shorter, please.\n/bye\n");
        }
    };
    const started = std.Io.Clock.awake.now(io).toMilliseconds();
    const code = blk: {
        const t = try std.Thread.spawn(.{}, Typist.run, .{ &pipe, f, &gate });
        defer t.join();
        break :blk try r.command(&.{ "--url", try f.url(a, "") });
    };
    try std.testing.expectEqual(@as(u8, 0), code);
    try std.testing.expect(!gate.isSet() and std.Io.Clock.awake.now(io).toMilliseconds() - started < 5000);
    try std.testing.expectEqualStrings("[{\"role\": \"user\", \"content\": \"Tell me a long story.\"}, {\"role\": \"assistant\", \"content\": \"Once upon\"}, " ++
        "{\"role\": \"user\", \"content\": \"Shorter, please.\"}]", try sent(a, f, 1));
    try std.testing.expect(std.mem.indexOf(u8, r.out.written(), "> Once upon\n> Hello there!\n> ") != null);
}

test "chat CLI: Ctrl-C at the prompt says how to end the chat and reads on" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    stop.reset();
    defer stop.reset();
    const f = try world(a, models_doc, try a.dupe([]const fake.Write, &.{try streamed(a)}));
    defer f.stop();
    var pipe: Pipe = .{};
    if (!pipe.open()) return error.SkipZigTest;
    defer pipe.close();
    r.interactive = true;
    r.reader = &pipe.file.interface;
    const Typist = struct {
        fn listed(fk: *fake.Fake) bool {
            return fk.count() > 0;
        }
        fn run(p: *const Pipe, fk: *fake.Fake) void {
            if (waitFor(listed, fk)) fake.nap(300);
            stop.request(.INT);
            _ = waitFor(handled, {});
            p.send("/bye\n");
        }
    };
    const code = blk: {
        const t = try std.Thread.spawn(.{}, Typist.run, .{ &pipe, f });
        defer t.join();
        break :blk try r.command(&.{ "--url", try f.url(a, "") });
    };
    try std.testing.expectEqual(@as(u8, 0), code);
    try std.testing.expect(std.mem.endsWith(u8, r.out.written(), "> \n(Type /bye or press Ctrl-D to end the chat.)\n> "));
    try std.testing.expectEqual(@as(usize, 0), posts(f));
}

/// A serve stand-in that records its pid and arguments, says ``serving`` on ``port`` if given, then runs ``then``.
fn standIn(r: *Run, port: ?u16, then: []const u8) !void {
    const script = try r.path("serve-stand-in");
    const serving = if (port) |p| try std.fmt.allocPrint(r.a, "printf '[tensorfold] serving example-checkpoint at http://127.0.0.1:%s/v1 (x)\\n' {d}\n", .{p}) else "";
    const key = try std.fmt.allocPrint(r.a, "printf '%s' \"$TENSORFOLD_API_KEY\" > {s}\n", .{try r.path("key")});
    const text = try std.fmt.allocPrint(r.a, "#!/bin/sh\necho $$ > {s}\nprintf '%s\\n' \"$@\" > {s}\n{s}printf 'loading\\n'\n{s}{s}\n", .{ try r.path("pid"), try r.path("args"), key, serving, then });
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = script, .data = text, .flags = .{ .permissions = .fromMode(0o755) } });
    try std.Io.Dir.cwd().createDirPath(std.testing.io, try r.path("example-checkpoint"));
    r.self_exe = script;
}

fn childPid(r: *Run) ?std.posix.pid_t {
    const text = std.Io.Dir.cwd().readFileAlloc(std.testing.io, r.path("pid") catch return null, r.a, .limited(64)) catch return null;
    return std.fmt.parseInt(std.posix.pid_t, std.mem.trim(u8, text, " \n"), 10) catch null;
}

fn gone(pid: std.posix.pid_t) bool {
    std.posix.kill(pid, @fromBackingInt(0)) catch |e| return e == error.ProcessNotFound;
    return false;
}

extern "c" fn getpgid(pid: std.posix.pid_t) std.posix.pid_t;

const sh_models = "{\"data\": [{\"id\": \"example-checkpoint\"}]}";

test "chat CLI: MODEL starts its own serve child in a process group of its own, keyed, chats through it and stops it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    try r.env.put("TENSORFOLD_API_KEY", "example-user-key");
    var gate: std.Io.Event = .unset;
    const held = try a.dupe(fake.Write, try streamed(a));
    held[0].after = &gate;
    const f = try worldKeyed(a, sh_models, try a.dupe([]const fake.Write, &.{held}), try r.path("key"));
    defer {
        gate.set(io);
        f.stop();
    }
    try standIn(&r, f.port, "exec sleep 30");
    r.input = "hi\n";
    const Checker = struct {
        group: std.posix.pid_t = -1,
        pid: std.posix.pid_t = -1,
        fn run(c: *@This(), fk: *fake.Fake, run_: *Run, g: *std.Io.Event) void {
            if (waitFor(hasPost, fk)) if (childPid(run_)) |pid| {
                c.pid = pid;
                c.group = getpgid(pid);
            };
            g.set(std.testing.io);
        }
    };
    var checker: Checker = .{};
    const code = blk: {
        const t = try std.Thread.spawn(.{}, Checker.run, .{ &checker, f, &r, &gate });
        defer t.join();
        break :blk try r.command(&.{ try r.path("example-checkpoint"), "--startup-timeout", "20", "--", "--parallel", "1" });
    };
    try std.testing.expectEqual(@as(u8, 0), code);
    try std.testing.expectEqualStrings("Hello there!\n", r.out.written());
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "Loading {s}; Ctrl-C cancels.\n", .{try r.path("example-checkpoint")}), r.err.written());
    try std.testing.expect(checker.pid > 0 and checker.group == checker.pid and gone(checker.pid));
    const args = try std.Io.Dir.cwd().readFileAlloc(io, try r.path("args"), a, .limited(4096));
    const want = try std.fmt.allocPrint(a, "serve\n{s}\n--host\n127.0.0.1\n--port\n0\n--name\nexample-checkpoint\n--parallel\n1\n", .{try r.path("example-checkpoint")});
    try std.testing.expectEqualStrings(want, args);
    // the child's key came in its environment alone, fresh, never the user's, and the chat printed it nowhere
    const key = try std.Io.Dir.cwd().readFileAlloc(io, try r.path("key"), a, .limited(256));
    try std.testing.expect(key.len == 64 and !std.mem.eql(u8, key, "example-user-key"));
    for ([_][]const u8{ r.out.written(), r.err.written(), args }) |text| try std.testing.expect(std.mem.indexOf(u8, text, key) == null);
    const bearer = try std.mem.concat(a, u8, &.{ "Bearer ", key });
    try std.testing.expect(f.refusals() == 0 and f.count() >= 4);
    for (0..f.count()) |i| try std.testing.expectEqualStrings(bearer, f.header(i, "authorization").?);
    var client: std.http.Client = .{ .allocator = std.heap.smp_allocator, .io = io };
    defer client.deinit();
    const anonymous = stream.fetch(a, io, &client, .{ .url = try f.url(a, "/v1/models"), .limit = 4096, .timeout_s = 5 });
    try std.testing.expect(anonymous.failed == .none and anonymous.status == 401 and f.refusals() == 1);
}

test "chat CLI: its own server's startup failure shows the server's last line; refused flags and an uncached model start nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const f = try world(a, sh_models, try a.dupe([]const fake.Write, &.{try streamed(a)}));
    defer f.stop();
    try standIn(&r, null, "printf '\\033[31merror: cannot read tokenizer.json\\033[0m\\n'; exit 3");
    r.input = "hi\n";
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ try r.path("example-checkpoint"), "--startup-timeout", "20" }));
    try std.testing.expect(std.mem.endsWith(u8, r.err.written(), "tensorfold chat: Model server exited during startup; check the checkpoint and backend toolchain\n" ++
        "Last server log line: error: cannot read tokenizer.json\n"));
    try std.testing.expectEqual(@as(usize, 0), posts(f));
    try std.Io.Dir.cwd().deleteFile(std.testing.io, try r.path("pid"));
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{ try r.path("example-checkpoint"), "--", "--port", "8080" }));
    try std.testing.expectEqualStrings("tensorfold chat: --host and --port: tensorfold chat gives the server it starts a free port on 127.0.0.1\n", r.err.written());
    try std.testing.expectEqual(@as(u8, 1), try r.command(&.{"example-org/not-cached"}));
    try std.testing.expect(std.mem.indexOf(u8, r.err.written(), "run tensorfold pull example-org/not-cached first") != null);
    try std.testing.expect(childPid(&r) == null);
}

test "chat CLI: when its own server exits mid-conversation the chat ends with the server's last lines" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var r = try Run.init(a, &tmp);
    const closes = try a.dupe(fake.Write, &.{});
    const f = try world(a, sh_models, try a.dupe([]const fake.Write, &.{closes}));
    defer f.stop();
    try standIn(&r, f.port, "printf 'error: the GPU was lost\\n'; exec sleep 30");
    var pipe: Pipe = .{};
    if (!pipe.open()) return error.SkipZigTest;
    defer pipe.close();
    r.interactive = true;
    r.reader = &pipe.file.interface;
    const Typist = struct {
        fn listed(fk: *fake.Fake) bool {
            var n: usize = 0;
            for (0..fk.count()) |i| n += @intFromBool(std.mem.eql(u8, fk.request(i).path, "/v1/models"));
            return n >= 2; // the readiness probe's, then the chat's own look
        }
        fn run(p: *const Pipe, fk: *fake.Fake, run_: *Run) void {
            if (waitFor(listed, fk)) if (childPid(run_)) |pid| {
                _ = platform.c.kill(pid, .KILL);
                fake.nap(200);
            };
            p.send("hello\n/bye\n");
        }
    };
    const code = blk: {
        const t = try std.Thread.spawn(.{}, Typist.run, .{ &pipe, f, &r });
        defer t.join();
        break :blk try r.command(&.{ try r.path("example-checkpoint"), "--startup-timeout", "20" });
    };
    try std.testing.expectEqual(@as(u8, 1), code);
    try std.testing.expect(std.mem.endsWith(u8, r.err.written(), "/v1/chat/completions\nThe model server has exited.\nLast server log line: error: the GPU was lost\n"));
}
