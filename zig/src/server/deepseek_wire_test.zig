//! The DeepSeek family's request fields over HTTP on the default wire: what reaches the engine, what the reply says.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const serve_mod = @import("dsv41_serve");
const deepseek = @import("deepseek.zig");
const server_mod = @import("server.zig");
const listener_mod = @import("listener.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const testing = std.testing;
const D = "｜DSML｜";

/// Replies by the last marker in the prompt's last user turn.
const scripts = [_]struct { []const u8, []const u8 }{
    .{ "@think", "Let me think about the greeting.</think>Hello there!" },
    .{ "@tool", "I should look up the weather.</think>Checking.\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"city\" string=\"true\">Paris</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n<" ++ D ++ " invoke name=\"get_weather\">\n<" ++ D ++ " parameter name=\"city\" string=\"true\">Rome</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>" },
    .{ "@stop", "ok</think>abc END def" },
    .{ "@long", "Plenty to say here, more than the limit lets through." },
    .{ "@plain", "Plain answer." },
};

/// What the engine was asked for, the last request.
const Seen = struct { prompt: []u32 = &.{}, max_tokens: u32 = 0, sampling: ?api.Sampling = null, drafts: bool = true, stop_eos: bool = true, background: bool = false };

const Stub = struct {
    gpa: Allocator,
    io: std.Io,
    tok: *serve_mod.tokenizer.Tokenizer,
    eos: u32,
    lock: std.Io.Mutex = .init,
    seen: ?Seen = null,

    fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn self(ctx: *anyopaque) *Stub {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096, .name = "stub" };
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
        if (s.seen) |x| s.gpa.free(x.prompt);
        s.seen = .{ .prompt = s.gpa.dupe(u32, r.prompt) catch &.{}, .max_tokens = r.max_tokens, .sampling = r.sampling, .drafts = r.drafts, .stop_eos = r.eos.len > 0, .background = r.background };
        s.lock.unlock(s.io);
        const t = std.Thread.spawn(.{}, run, .{ s, id, r.prompt, r.max_tokens, sink }) catch return error.Busy;
        t.detach();
    }

    fn run(s: *Stub, id: api.Id, prompt_ids: []const u32, max_tokens: u32, sink: api.Sink) void {
        var arena = std.heap.ArenaAllocator.init(s.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const prompt = s.tok.decodeAlloc(a, prompt_ids) catch "";
        const last = if (std.mem.lastIndexOf(u8, prompt, "<｜User｜>")) |at| prompt[at..] else prompt;
        var reply: []const u8 = "?";
        for (scripts) |sc| if (std.mem.indexOf(u8, last, sc[0]) != null) {
            reply = sc[1];
        };
        var ids = s.tok.encodeAlloc(a, reply) catch &.{};
        ids = std.mem.concat(a, u32, &.{ ids, &.{s.eos} }) catch ids;
        sink.event(sink.ctx, id, &.{ .prefilled = 3 });
        var i: usize = 0;
        while (i < ids.len and i < max_tokens) {
            const step = @min(1 + i % 3, ids.len - i, max_tokens - i);
            sink.event(sink.ctx, id, &.{ .tokens = ids[i .. i + step] });
            i += step;
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (i == ids.len) .stop else .length } });
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
    lis: listener_mod.Listener = undefined,

    fn start(h: *Harness, tok: *serve_mod.tokenizer.Tokenizer, a: Allocator) !void {
        const eos = tok.tokenId("<｜end▁of▁sentence｜>").?;
        h.ds = try deepseek.DeepSeek.init(h.gpa, tok, .{ .eos = &.{eos} });
        h.stub = .{ .gpa = h.gpa, .io = h.io, .tok = tok, .eos = eos };
        const sampling = (try json.parseText(a, "{\"temperature\": 1.0, \"top_k\": 20, \"top_p\": 0.95, \"min_p\": 0.05}")).ok;
        h.srv = try server_mod.Server.init(h.gpa, h.io, h.stub.engine(), h.ds.text(), .{
            .served_name = "deepseek-v41",
            .model_ids = &.{ "deepseek-v41", "alias-a" },
            .default_sampling = sampling,
            .family = h.ds.family_(),
        }, null);
        h.lis = try listener_mod.Listener.open(.{ .ip4 = .loopback(0) });
        h.thread = try std.Thread.spawn(.{}, server_mod.Server.serve, .{ h.srv, h.lis, &h.stop });
    }

    fn finish(h: *Harness) void {
        h.stop.store(true, .release);
        h.thread.join();
        h.lis.close();
        while (h.srv.open_connections.load(.acquire) > 0) std.Io.sleep(h.io, .fromMilliseconds(5), .awake) catch {};
        h.srv.deinit();
        h.ds.deinit(false);
        if (h.stub.seen) |x| h.gpa.free(x.prompt);
    }

    const Got = struct { status: u16, body: []const u8, seen: ?Seen };

    /// POST ``body``: the status, the body (an SSE stream whole) and what the engine got (null: not asked).
    fn post(h: *Harness, a: Allocator, path: []const u8, body: []const u8) !Got {
        if (h.stub.seen) |x| h.gpa.free(x.prompt);
        h.stub.seen = null;
        const addr: std.Io.net.IpAddress = .{ .ip4 = .loopback(h.lis.port()) };
        const stream = try addr.connect(h.io, .{ .mode = .stream });
        defer stream.close(h.io);
        const head = try std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ path, body.len, body });
        var wbuf: [4096]u8 = undefined;
        var w = std.Io.net.Stream.Writer.init(stream, h.io, &wbuf);
        try w.interface.writeAll(head);
        try w.interface.flush();
        var rbuf: [16384]u8 = undefined;
        var r = std.Io.net.Stream.Reader.init(stream, h.io, &rbuf);
        const text = try r.interface.allocRemaining(a, .unlimited);
        const split = std.mem.indexOf(u8, text, "\r\n\r\n") orelse return error.Head;
        var seen = h.stub.seen;
        if (seen) |*x| x.prompt = try a.dupe(u32, x.prompt);
        return .{ .status = try std.fmt.parseInt(u16, text[9..12], 10), .body = text[split + 4 ..], .seen = seen };
    }

    /// A chat request's body: ``extra`` fields after the user message ``content``.
    fn chat(h: *Harness, a: Allocator, content: []const u8, extra: []const u8) !Got {
        return h.post(a, "/v1/chat/completions", try std.fmt.allocPrint(a, "{{\"model\":\"deepseek-v41\",\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}],\"max_tokens\":200{s}}}", .{ content, extra }));
    }
};

fn message(a: Allocator, body: []const u8) !Value {
    const v = (try json.parseText(a, body)).ok;
    return v.get("choices").?.array[0];
}

fn tokens(a: Allocator, body: []const u8) ![]u32 {
    const list = (try json.parseText(a, body)).ok.get("tokens").?.array;
    const out = try a.alloc(u32, list.len);
    for (list, out) |t, *slot| slot.* = @intCast(t.int64().?);
    return out;
}

const tool = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}}}]";

test "the family's request fields reach the engine on the default wire (mini tokenizer)" {
    const gpa = testing.allocator;
    const tok = try serve_mod.tokenizer.Tokenizer.parse(gpa, serve_mod.fixtures.mini_tokenizer);
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var h: Harness = .{ .gpa = gpa, .io = testing.io, .stub = undefined, .ds = undefined, .srv = undefined };
    try h.start(tok, a);
    defer h.finish();

    // 1. efforts other than none / high reach the prompt as /tokenize renders the same body
    const high = (try h.chat(a, "hi @think", ",\"temperature\":0")).seen.?.prompt;
    const low = (try h.chat(a, "hi @think", ",\"temperature\":0,\"reasoning_effort\":\"low\"")).seen.?.prompt;
    const n37 = (try h.chat(a, "hi @think", ",\"temperature\":0,\"reasoning_effort\":37")).seen.?.prompt;
    try testing.expect(!std.mem.eql(u32, high, low) and !std.mem.eql(u32, high, n37) and !std.mem.eql(u32, low, n37));
    const tz = try h.post(a, "/tokenize", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi @think\"}],\"reasoning_effort\":\"low\"}");
    try testing.expectEqualSlices(u32, low, try tokens(a, tz.body));
    const bad = try h.chat(a, "hi @think", ",\"reasoning_effort\":\"huge\"");
    try testing.expect(bad.status == 400 and bad.seen == null);

    // 2. chat_template_kwargs: thinking off renders as effort none; a non-object refuses before the engine
    const none = (try h.chat(a, "hi @plain", ",\"temperature\":0,\"reasoning_effort\":\"none\"")).seen.?.prompt;
    const off = (try h.chat(a, "hi @plain", ",\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}")).seen.?.prompt;
    const plain_on = (try h.chat(a, "hi @plain", ",\"temperature\":0")).seen.?.prompt;
    try testing.expectEqualSlices(u32, none, off);
    try testing.expect(!std.mem.eql(u32, off, plain_on));
    const kw = try h.chat(a, "hi @plain", ",\"chat_template_kwargs\":[]");
    try testing.expect(kw.status == 400 and kw.seen == null);

    // 3. sampling: the defaults, then the request's own fields, except min_p (the family keeps the default's)
    const dflt = (try h.chat(a, "hi @plain", "")).seen.?.sampling.?;
    try testing.expect(dflt.temperature == 1.0 and dflt.top_k == 20 and dflt.top_p == 0.95 and dflt.min_p == 0.05);
    const own = (try h.chat(a, "hi @plain", ",\"temperature\":0.7,\"top_p\":0.9,\"seed\":7,\"min_p\":0.5")).seen.?.sampling.?;
    try testing.expect(own.temperature == 0.7 and own.top_k == 20 and own.top_p == 0.9 and own.seed == 7 and own.min_p == 0.05);
    try testing.expect((try h.chat(a, "hi @plain", ",\"temperature\":0")).seen.?.sampling == null);
    try testing.expect(!(try h.chat(a, "hi @plain", ",\"temperature\":0,\"draft\":false")).seen.?.drafts);
    try testing.expect((try h.chat(a, "hi @plain", ",\"temperature\":0,\"priority\":\"background\"")).seen.?.background);
    try testing.expect(!(try h.chat(a, "hi @plain", ",\"temperature\":0,\"ignore_eos\":true")).seen.?.stop_eos);

    // 4. tools: both calls whole, one with parallel_tool_calls false, none offered with tool_choice none
    const two = try message(a, (try h.chat(a, "hi @tool", ",\"temperature\":0" ++ tool)).body);
    try testing.expectEqual(@as(usize, 2), two.get("message").?.get("tool_calls").?.array.len);
    try testing.expectEqualStrings("tool_calls", two.strField("finish_reason").?);
    try testing.expect(two.get("message").?.get("content").? == .null);
    const one = try message(a, (try h.chat(a, "hi @tool", ",\"temperature\":0,\"parallel_tool_calls\":false" ++ tool)).body);
    try testing.expectEqual(@as(usize, 1), one.get("message").?.get("tool_calls").?.array.len);
    const no_tools = try h.chat(a, "hi @tool", ",\"temperature\":0,\"tool_choice\":\"none\"" ++ tool);
    try testing.expectEqualSlices(u32, (try h.chat(a, "hi @tool", ",\"temperature\":0")).seen.?.prompt, no_tools.seen.?.prompt);
    try testing.expect((try message(a, no_tools.body)).get("message").?.get("tool_calls") == null);

    // 5. a stop string, the length limit (whole and streamed), max_completion_tokens
    const stop = try message(a, (try h.chat(a, "hi @stop", ",\"temperature\":0,\"stop\":\"END\"")).body);
    try testing.expectEqualStrings("abc ", stop.get("message").?.strField("content").?);
    try testing.expectEqualStrings("stop", stop.strField("finish_reason").?);
    const long = try h.post(a, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi @long\"}],\"max_tokens\":12,\"temperature\":0,\"reasoning_effort\":\"none\"}");
    try testing.expectEqualStrings("length", (try message(a, long.body)).strField("finish_reason").?);
    const long_stream = try h.post(a, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi @long\"}],\"max_tokens\":12,\"temperature\":0,\"reasoning_effort\":\"none\",\"stream\":true}");
    try testing.expect(std.mem.indexOf(u8, long_stream.body, "\"length\"") != null);
    const mct = try h.post(a, "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi @plain\"}],\"max_completion_tokens\":5,\"temperature\":0}");
    try testing.expectEqual(@as(u32, 5), mct.seen.?.max_tokens);

    // 6. an alias the server answers to is echoed
    const alias = try h.post(a, "/v1/chat/completions", "{\"model\":\"alias-a\",\"messages\":[{\"role\":\"user\",\"content\":\"hi @plain\"}],\"temperature\":0}");
    try testing.expectEqualStrings("alias-a", (try json.parseText(a, alias.body)).ok.strField("model").?);

    // 7. /v1/completions beside the family (text and ids, whole and streamed) and /tokenize of a prompt
    const enc = try tok.encodeAlloc(a, "hi @plain");
    const done = try h.post(a, "/v1/completions", "{\"prompt\":\"hi @plain\",\"max_tokens\":50,\"temperature\":0}");
    try testing.expectEqualSlices(u32, enc, done.seen.?.prompt);
    try testing.expectEqualStrings("Plain answer.", (try message(a, done.body)).strField("text").?);
    const streamed = try h.post(a, "/v1/completions", "{\"prompt\":\"hi @plain\",\"max_tokens\":50,\"temperature\":0,\"stream\":true}");
    try testing.expect(std.mem.indexOf(u8, streamed.body, "data: [DONE]") != null);
    const listed = try std.fmt.allocPrint(a, "{{\"prompt\":[{d},{d}],\"max_tokens\":5,\"temperature\":0}}", .{ enc[0], enc[enc.len - 1] });
    try testing.expectEqualSlices(u32, &.{ enc[0], enc[enc.len - 1] }, (try h.post(a, "/v1/completions", listed)).seen.?.prompt);
    try testing.expectEqualSlices(u32, enc, try tokens(a, (try h.post(a, "/tokenize", "{\"prompt\":\"hi @plain\"}")).body));
}
