//! The DeepSeek serving side end to end over HTTP: real sockets, the family's text side, a stub engine's scripted ids.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const serve_mod = @import("dsv41_serve");
const deepseek = @import("deepseek.zig");
const server_mod = @import("server.zig");
const listener_mod = @import("listener.zig");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const D = "｜DSML｜";

/// Replies by the marker the last user message holds.
const scripts = [_]struct { []const u8, []const u8 }{
    .{ "@think", "Let me think about the greeting.</think>Hello there!" },
    .{ "@tool", "I should look up the weather.</think>Checking.\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"city\" string=\"true\">Paris</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"days\" string=\"false\">3</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"city\" string=\"true\">東京</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>" },
    .{ "@stop", "ok</think>abc END def" },
    .{ "@chat", "Plain answer." },
};

const Stub = struct {
    gpa: Allocator,
    io: std.Io,
    tok: *serve_mod.tokenizer.Tokenizer,
    eos: u32,
    window: u32,
    cached: u32 = 3,
    lock: std.Io.Mutex = .init,
    last_prompt: std.ArrayList(u32) = .empty,
    last_max: u32 = 0,

    fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn self(ctx: *anyopaque) *Stub {
        return @ptrCast(@alignCast(ctx));
    }

    fn info(ctx: *anyopaque) api.Info {
        return .{ .context_window = self(ctx).window, .name = "stub" };
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s = self(ctx);
        s.lock.lockUncancelable(s.io);
        s.last_prompt.clearRetainingCapacity();
        s.last_prompt.appendSlice(s.gpa, r.prompt) catch {};
        s.last_max = r.max_tokens;
        s.lock.unlock(s.io);
        const t = std.Thread.spawn(.{}, run, .{ s, id, r, sink }) catch return error.Busy;
        t.detach();
    }

    fn run(s: *Stub, id: api.Id, r: *const api.Request, sink: api.Sink) void {
        var arena = std.heap.ArenaAllocator.init(s.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const prompt = s.tok.decodeAlloc(a, r.prompt) catch "";
        const last_user = if (std.mem.lastIndexOf(u8, prompt, "<｜User｜>")) |at| prompt[at..] else prompt;
        var reply: []const u8 = "?";
        for (scripts) |sc| if (std.mem.indexOf(u8, last_user, sc[0]) != null) {
            reply = sc[1];
        };
        var ids = s.tok.encodeAlloc(a, reply) catch &.{};
        ids = std.mem.concat(a, u32, &.{ ids, &.{s.eos} }) catch ids;
        sink.event(sink.ctx, id, &.{ .prefilled = s.cached });
        var i: usize = 0;
        var n: usize = 0;
        while (i < ids.len and n < r.max_tokens) {
            const step = @min(1 + i % 3, ids.len - i, r.max_tokens - n);
            sink.event(sink.ctx, id, &.{ .tokens = ids[i .. i + step] });
            i += step;
            n += step;
        }
        const reason: api.Reason = if (i == ids.len) .stop else .length;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = reason } });
    }
};

const Harness = struct {
    gpa: Allocator,
    io: std.Io,
    stub: Stub,
    ds: *deepseek.DeepSeek,
    srv: *server_mod.Server,
    stop: std.atomic.Value(bool) = .init(false),
    thread: std.Thread = undefined,
    port: u16 = 0,
    lis: listener_mod.Listener = undefined,

    fn start(h: *Harness, tok: *serve_mod.tokenizer.Tokenizer, eos: u32, window: u32, options: deepseek.Options) !void {
        var o = options;
        o.eos = &.{eos};
        h.ds = try deepseek.DeepSeek.init(h.gpa, tok, o);
        h.stub = .{ .gpa = h.gpa, .io = h.io, .tok = tok, .eos = eos, .window = window };
        h.srv = try server_mod.Server.init(h.gpa, h.io, h.stub.engine(), h.ds.text(), .{
            .served_name = "deepseek-v41",
            .model_ids = &.{"deepseek-v41"},
            .default_sampling = null,
            .family = h.ds.family_(),
        }, null);
        h.lis = try listener_mod.Listener.open(.{ .ip4 = .loopback(0) });
        h.port = h.lis.port();
        h.thread = try std.Thread.spawn(.{}, server_mod.Server.serve, .{ h.srv, h.lis, &h.stop });
    }

    fn finish(h: *Harness) void {
        h.stop.store(true, .release);
        h.thread.join();
        h.lis.close();
        while (h.srv.open_connections.load(.acquire) > 0) std.Io.sleep(h.io, .fromMilliseconds(5), .awake) catch {};
        h.srv.deinit();
        h.ds.deinit(false);
        h.stub.last_prompt.deinit(h.gpa);
    }

    /// POST ``body`` to ``path``: the status and the response body (the whole stream for SSE).
    fn post(h: *Harness, a: Allocator, path: []const u8, body: []const u8) !struct { status: u16, body: []const u8 } {
        const addr: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.port) };
        const stream = try addr.connect(h.io, .{ .mode = .stream });
        defer stream.close(h.io);
        const head = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ path, body.len, body });
        var wbuf: [4096]u8 = undefined;
        var w = std.Io.net.Stream.Writer.init(stream, h.io, &wbuf);
        try w.interface.writeAll(head);
        try w.interface.flush();
        var rbuf: [16384]u8 = undefined;
        var r = std.Io.net.Stream.Reader.init(stream, h.io, &rbuf);
        var out: std.ArrayList(u8) = .empty;
        try out.appendSlice(a, try r.interface.allocRemaining(a, .unlimited));
        const text = out.items;
        const status = try std.fmt.parseInt(u16, text[9..12], 10);
        const split = std.mem.indexOf(u8, text, "\r\n\r\n") orelse return error.Head;
        var payload: []const u8 = text[split + 4 ..];
        if (std.mem.indexOf(u8, text[0..split], "chunked") != null) payload = try unchunk(a, payload);
        return .{ .status = status, .body = payload };
    }

    fn unchunk(a: Allocator, s: []const u8) ![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        var i: usize = 0;
        while (i < s.len) {
            const eol = std.mem.indexOfPos(u8, s, i, "\r\n") orelse break;
            const n = try std.fmt.parseInt(usize, std.mem.trim(u8, s[i..eol], " "), 16);
            if (n == 0) break;
            try out.appendSlice(a, s[eol + 2 .. eol + 2 + n]);
            i = eol + 2 + n + 2;
        }
        return out.items;
    }

    /// The prompt the engine got for the last request, as text.
    fn lastPrompt(h: *Harness, a: Allocator) ![]const u8 {
        return h.stub.tok.decodeAlloc(a, h.stub.last_prompt.items);
    }
};

/// The ``data:`` events of an SSE body, parsed (``[DONE]`` as null).
fn events(a: Allocator, body: []const u8) ![]?json.Value {
    var out: std.ArrayList(?json.Value) = .empty;
    var it = std.mem.splitSequence(u8, body, "\n\n");
    while (it.next()) |ev| {
        const line = std.mem.trim(u8, ev, "\r\n");
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const data = line[6..];
        if (std.mem.eql(u8, data, "[DONE]")) {
            try out.append(a, null);
            continue;
        }
        try out.append(a, (try json.parseText(a, data)).ok);
    }
    return out.items;
}

fn delta(ev: json.Value) json.Value {
    return ev.get("choices").?.array[0].get("delta").?;
}

fn check(h: *Harness, a: Allocator) !void {
    const tok = h.stub.tok;
    // 1. a whole reply: reasoning in both fields, content, usage with the engine's cached tokens
    {
        const r = try h.post(a, "/v1/chat/completions", "{\"model\":\"deepseek-v41\",\"messages\":[{\"role\":\"user\",\"content\":\"hi @think\"}],\"max_tokens\":200}");
        try testing.expectEqual(@as(u16, 200), r.status);
        const v = (try json.parseText(a, r.body)).ok;
        const msg = v.get("choices").?.array[0].get("message").?;
        try testing.expectEqualStrings("Hello there!", msg.get("content").?.string);
        try testing.expectEqualStrings("Let me think about the greeting.", msg.get("reasoning_content").?.string);
        try testing.expectEqualStrings("Let me think about the greeting.", msg.get("reasoning").?.string);
        try testing.expectEqualStrings("stop", v.get("choices").?.array[0].get("finish_reason").?.string);
        const usage = v.get("usage").?;
        try testing.expectEqual(@as(i64, @intCast(h.stub.last_prompt.items.len)), usage.get("prompt_tokens").?.int64().?);
        try testing.expectEqual(@as(i64, 3), usage.get("prompt_tokens_details").?.get("cached_tokens").?.int64().?);
        const reply_ids = try tok.encodeAlloc(a, scripts[0][1]);
        try testing.expectEqual(@as(i64, @intCast(reply_ids.len + 1)), usage.get("completion_tokens").?.int64().?);
        // the prompt is the encoding's, with the default effort
        var p: serve_mod.template.Problem = .{};
        const want = try serve_mod.template.render(a, (try json.parseText(a, "[{\"role\":\"user\",\"content\":\"hi @think\"}]")).ok, .{ .budget = 75 }, &p);
        try testing.expectEqualStrings(want, try h.lastPrompt(a));
        try testing.expectEqualSlices(u32, try tok.encodeAlloc(a, want), h.stub.last_prompt.items);
    }
    // 2. a streamed reply with two calls: role, reasoning, calls as they complete, usage, [DONE]
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}}}}]";
    {
        const body = try std.fmt.allocPrint(a, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"weather @tool\"}}],\"tools\":{s},\"stream\":true,\"stream_options\":{{\"include_usage\":true}},\"reasoning_effort\":\"medium\"}}", .{tools});
        const r = try h.post(a, "/v1/chat/completions", body);
        try testing.expectEqual(@as(u16, 200), r.status);
        const evs = try events(a, r.body);
        try testing.expect(evs.len > 4 and evs[evs.len - 1] == null);
        try testing.expectEqualStrings("assistant", delta(evs[0].?).get("role").?.string);
        var reasoning: std.ArrayList(u8) = .empty;
        var content: std.ArrayList(u8) = .empty;
        var args: [2]std.ArrayList(u8) = .{ .empty, .empty };
        var names: [2][]const u8 = .{ "", "" };
        var ids: [2][]const u8 = .{ "", "" };
        var finish: []const u8 = "";
        var usage: ?json.Value = null;
        for (evs) |ev| {
            const e = ev orelse continue;
            if (e.get("usage")) |u| usage = u;
            const choices = e.get("choices").?.array;
            if (choices.len == 0) continue;
            if (choices[0].get("finish_reason")) |f| if (f == .string) {
                finish = f.string;
            };
            const d = choices[0].get("delta").?;
            if (d.get("reasoning_content")) |t| try reasoning.appendSlice(a, t.string);
            if (d.get("content")) |t| if (t == .string) try content.appendSlice(a, t.string);
            if (d.get("tool_calls")) |tc| for (tc.array) |c| {
                const k: usize = @intCast(c.get("index").?.int64().?);
                if (c.get("id")) |id| ids[k] = id.string;
                const f = c.get("function").?;
                if (f.get("name")) |n| names[k] = n.string;
                try args[k].appendSlice(a, f.get("arguments").?.string);
            };
        }
        try testing.expectEqualStrings("I should look up the weather.", reasoning.items);
        try testing.expectEqualStrings("Checking.", content.items);
        try testing.expectEqualStrings("get_weather", names[0]);
        try testing.expectEqualStrings("get_weather", names[1]);
        try testing.expect(std.mem.startsWith(u8, ids[0], "call_") and ids[0].len == 29 and !std.mem.eql(u8, ids[0], ids[1]));
        try testing.expectEqualStrings("{\"city\":\"Paris\",\"days\":3}", args[0].items);
        try testing.expectEqualStrings("{\"city\":\"東京\"}", args[1].items);
        try testing.expectEqualStrings("tool_calls", finish);
        try testing.expect(usage != null and usage.?.get("prompt_tokens_details").?.get("cached_tokens").?.int64().? == 3);
        try testing.expect(std.mem.indexOf(u8, try h.lastPrompt(a), "Reasoning Effort: 75 (range") != null);
        try testing.expect(std.mem.indexOf(u8, try h.lastPrompt(a), "\"name\": \"get_weather\"") != null);
    }
    // 3. the same reply whole, one call kept (parallel_tool_calls false); a later turn gets its reasoning back
    var call_id: []const u8 = "";
    {
        const body = try std.fmt.allocPrint(a, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"weather @tool\"}}],\"tools\":{s},\"parallel_tool_calls\":false}}", .{tools});
        const r = try h.post(a, "/v1/chat/completions", body);
        const v = (try json.parseText(a, r.body)).ok;
        const choice = v.get("choices").?.array[0];
        try testing.expectEqualStrings("tool_calls", choice.get("finish_reason").?.string);
        const msg = choice.get("message").?;
        try testing.expect(msg.get("content").? == .null);
        const calls = msg.get("tool_calls").?.array;
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("{\"city\":\"Paris\",\"days\":3}", calls[0].get("function").?.get("arguments").?.string);
        call_id = calls[0].get("id").?.string;
    }
    {
        const body = try std.fmt.allocPrint(a, "{{\"messages\":[{{\"role\":\"user\",\"content\":\"weather\"}},{{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{{\"id\":\"{s}\",\"type\":\"function\",\"function\":{{\"name\":\"get_weather\",\"arguments\":\"{{\\\"city\\\": \\\"Paris\\\"}}\"}}}}]}},{{\"role\":\"tool\",\"tool_call_id\":\"{s}\",\"content\":\"sunny\"}},{{\"role\":\"user\",\"content\":\"thanks @chat\"}}],\"tools\":{s}}}", .{ call_id, call_id, tools });
        const r = try h.post(a, "/v1/chat/completions", body);
        try testing.expectEqual(@as(u16, 200), r.status);
        const prompt = try h.lastPrompt(a);
        try testing.expect(std.mem.indexOf(u8, prompt, "I should look up the weather.</think>\n\n<" ++ D ++ " calls>") != null);
        try testing.expect(std.mem.indexOf(u8, prompt, "<tool_result>sunny</tool_result>") != null);
    }
    // 4. a stop string ends the answer (reasoning is not matched) and the reply
    {
        const r = try h.post(a, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"x @stop\"}],\"stop\":[\"END\"],\"stream\":true}");
        var content: std.ArrayList(u8) = .empty;
        var finish: []const u8 = "";
        for (try events(a, r.body)) |ev| {
            const e = ev orelse continue;
            const c = e.get("choices").?.array;
            if (c.len == 0) continue;
            if (c[0].get("finish_reason")) |f| if (f == .string) {
                finish = f.string;
            };
            if (c[0].get("delta").?.get("content")) |t| if (t == .string) try content.appendSlice(a, t.string);
        }
        try testing.expectEqualStrings("abc ", content.items);
        try testing.expectEqualStrings("stop", finish);
    }
    // 5. thinking off by reasoning_effort none: the chat-mode prompt; an empty answer is null
    {
        const r = try h.post(a, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"q @chat\"}],\"reasoning_effort\":\"none\"}");
        const v = (try json.parseText(a, r.body)).ok;
        try testing.expectEqualStrings("Plain answer.", v.get("choices").?.array[0].get("message").?.get("content").?.string);
        try testing.expect(std.mem.endsWith(u8, try h.lastPrompt(a), "<｜Assistant｜></think>"));
        const bad = try h.post(a, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"}],\"reasoning_effort\":\"huge\"}");
        try testing.expectEqual(@as(u16, 400), bad.status);
    }
    // 6. an image part reads as the placeholder notice
    {
        const r = try h.post(a, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"what is this @chat\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}]}]}");
        try testing.expectEqual(@as(u16, 200), r.status);
        try testing.expect(std.mem.indexOf(u8, try h.lastPrompt(a), "what is this @chat\n\n[image omitted:") != null);
        const v = (try json.parseText(a, r.body)).ok;
        try testing.expectEqual(@as(i64, 1), v.get("tensorfold").?.get("images_omitted").?.int64().?);
    }
    // 7. /tokenize renders messages as the chat route does
    {
        const r = try h.post(a, "/tokenize", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi @think\"}]}");
        const v = (try json.parseText(a, r.body)).ok;
        var p: serve_mod.template.Problem = .{};
        const want = try tok.encodeAlloc(a, try serve_mod.template.render(a, (try json.parseText(a, "[{\"role\":\"user\",\"content\":\"hi @think\"}]")).ok, .{}, &p));
        try testing.expectEqual(@as(i64, @intCast(want.len)), v.get("count").?.int64().?);
    }
    // 8. past the context window: OpenAI's code inside the stream
    {
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"max_tokens\":100000}";
        const r = try h.post(a, "/v1/chat/completions", body);
        try testing.expectEqual(@as(u16, 200), r.status);
        try testing.expect(std.mem.indexOf(u8, r.body, "context_length_exceeded") != null);
    }
}

test "DeepSeek-V4.1 over HTTP: whole and streamed replies, reasoning, DSML calls, stops, images, limits (mini tokenizer)" {
    const gpa = testing.allocator;
    const io = testing.io;
    const tok = try serve_mod.tokenizer.Tokenizer.parse(gpa, serve_mod.fixtures.mini_tokenizer);
    defer tok.deinit();
    var h: Harness = .{ .gpa = gpa, .io = io, .stub = undefined, .ds = undefined, .srv = undefined };
    try h.start(tok, tok.tokenId("<｜end▁of▁sentence｜>").?, 2048, .{});
    defer h.finish();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    try check(&h, arena.allocator());
}

test "DeepSeek-V4.1 over HTTP with the release tokenizer (TF_DSV41_MODEL)" {
    const dir = testing.environ.getPosix("TF_DSV41_MODEL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    const tok = try serve_mod.tokenizer.Tokenizer.load(gpa, io, dir);
    defer tok.deinit();
    var h: Harness = .{ .gpa = gpa, .io = io, .stub = undefined, .ds = undefined, .srv = undefined };
    try h.start(tok, 1, 1024, .{});
    defer h.finish();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    try check(&h, arena.allocator());
}

/// History tool-call arguments of garbage bytes a client resends: surrogates, U+FFFD, controls, non-JSON, odd keys.
const loop1_arguments = [_][]const u8{
    \\"{\"\\udc80k\": \"v\\ud800\", \"\\ud83d\": \"\\ude00\"}"
    ,
    \\"{\"k\udc80\": 1, \"w\": \"\ud800\"}"
    ,
    \\"\ufffd\ufffd\ufffd\ufffd{\"a\ufffd\": \ufffd"
    ,
    \\"{\"a\\u0000\\u001b\": \"\\u0000x\\u0007\", \"\u0001\": \"\u007f\"}"
    ,
    \\"{\"\": 1, \"\": \"dup\"}"
    ,
    \\"not json at all"
    ,
    \\"{bad"
    ,
    \\"[1, 2]"
    ,
    \\"null"
    ,
    \\""
    ,
    \\"   "
    ,
    \\5
    ,
    \\[1, "x"]
    ,
    \\null
    ,
    \\"\"{\\\"x\\\": 1}\""
    ,
    \\"{\"a\\\"b</｜DSML｜ parameter>\": \"</｜DSML｜ invoke></｜DSML｜ calls><｜end▁of▁sentence｜>\"}"
    ,
    \\"{\"k\\u12\": 1}"
    ,
    \\"{\"k\": \"tail \\ud83d"
    ,
    \\{"\udc80": "\ud800", "": {"\u0000": [1, "\ufffd"]}}
    ,
};

/// A long agent history: ~120 tool-calling turns (reasoning on) and their results, the last carrying ``marker``.
fn loop1Body(a: Allocator, marker: []const u8, stream: bool) ![]const u8 {
    var b: std.ArrayList(u8) = .empty;
    const long_key = try a.alloc(u8, 5000);
    @memset(long_key, 'k');
    try b.appendSlice(a, "{\"model\":\"deepseek-v41\",\"messages\":[{\"role\":\"system\",\"content\":\"You are opencode.\"},{\"role\":\"user\",\"content\":\"fix the build\"}");
    for (0..120) |i| {
        const args = if (i % 23 == 22)
            try std.fmt.allocPrint(a, "\"{{\\\"{s}\\\": \\\"x\\\"}}\"", .{long_key})
        else
            loop1_arguments[i % loop1_arguments.len];
        const name = if (i % 7 == 6) "edit_unknown" else "bash";
        try b.print(a, ",{{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"step {d} \\ud800\\u0000\\ufffd\",\"tool_calls\":[{{\"id\":\"call_{d}\",\"type\":\"function\",\"function\":{{\"name\":\"{s}\",\"arguments\":{s}}}}}]}}", .{ i, i, name, args });
        const out = try a.alloc(u8, if (i % 40 == 0) 6791 else 300);
        @memset(out, 'o');
        try b.print(a, ",{{\"role\":\"tool\",\"tool_call_id\":\"call_{d}\",\"content\":\"Error: invalid arguments \\udc80\\u001b[31m {s}{s}\"}}", .{ i, out, if (i == 119) marker else "" });
    }
    try b.print(a, "],\"tools\":[{{\"type\":\"function\",\"function\":{{\"name\":\"bash\",\"parameters\":{{\"type\":\"object\",\"properties\":{{\"command\":{{\"type\":\"string\"}}}}}}}}}},{{\"type\":\"function\",\"function\":{{\"name\":\"get_weather\",\"parameters\":{{}}}}}}],\"stream\":{},\"reasoning_effort\":\"high\",\"max_tokens\":4096}}", .{stream});
    return b.items;
}

/// Send ``body`` streamed and hang up after the first bytes of the reply (a client abort mid-stream).
fn abortMidStream(h: *Harness, a: Allocator, body: []const u8) !void {
    const addr: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.port) };
    const stream = try addr.connect(h.io, .{ .mode = .stream });
    defer stream.close(h.io);
    const head = try std.fmt.allocPrint(a, "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}", .{ body.len, body });
    var wbuf: [4096]u8 = undefined;
    var w = std.Io.net.Stream.Writer.init(stream, h.io, &wbuf);
    try w.interface.writeAll(head);
    try w.interface.flush();
    var rbuf: [64]u8 = undefined;
    var r = std.Io.net.Stream.Reader.init(stream, h.io, &rbuf);
    _ = try r.interface.peek(1);
}

fn checkLoop1(h: *Harness, a: Allocator) !void {
    // streamed and whole, a reply with calls and a plain one: the server renders the history and answers
    for ([_][]const u8{ "@tool", "@chat", "@think" }) |marker| for ([_]bool{ true, false }) |stream| {
        errdefer std.debug.print("long session shape, {s}, stream {}\n", .{ marker, stream });
        const r = try h.post(a, "/v1/chat/completions", try loop1Body(a, marker, stream));
        errdefer std.debug.print("{d}: {s}\n", .{ r.status, r.body[0..@min(r.body.len, 400)] });
        try testing.expectEqual(@as(u16, 200), r.status);
        const prompt = try h.lastPrompt(a);
        try testing.expect(std.mem.indexOf(u8, prompt, "parameter name=\"arguments\" string=\"true\">not json at all<") != null);
        try testing.expect(std.mem.indexOf(u8, prompt, "step 119 ") != null);
        try testing.expect(std.mem.indexOf(u8, prompt, "<tool_result>Error: invalid arguments") != null);
        if (stream) {
            const evs = try events(a, r.body);
            try testing.expect(evs.len > 2 and evs[evs.len - 1] == null);
        } else {
            const v = (try json.parseText(a, r.body)).ok;
            try testing.expect(v.get("choices") != null);
        }
    };
    // invalid UTF-8 bytes in the body itself, in a history key and value
    {
        const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"k\xff\xfe\\\": \\\"\xc3\\\"}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"c\",\"content\":\"\xed\xa0\x80 \xe2\x82 @chat\"}],\"stream\":true}";
        const r = try h.post(a, "/v1/chat/completions", body);
        try testing.expect(r.status == 200 or r.status == 400);
    }
    // client aborts mid-stream, then the server still answers
    for (0..4) |_| try abortMidStream(h, a, try loop1Body(a, "@tool", true));
    const r = try h.post(a, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi @think\"}]}");
    try testing.expectEqual(@as(u16, 200), r.status);
}

test "a long session's request shape: 120 history calls with garbage arguments, a mid-stream abort (mini tokenizer)" {
    const gpa = testing.allocator;
    const io = testing.io;
    const tok = try serve_mod.tokenizer.Tokenizer.parse(gpa, serve_mod.fixtures.mini_tokenizer);
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = .{ .gpa = gpa, .io = io, .stub = undefined, .ds = undefined, .srv = undefined };
    try h.start(tok, tok.tokenId("<｜end▁of▁sentence｜>").?, 1 << 22, .{});
    defer h.finish();
    try checkLoop1(&h, a);
}
