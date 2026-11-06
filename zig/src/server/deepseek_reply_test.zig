//! DeepSeek-V4.1 through HfText and the OpenAI route: the thinking switch, and a reply's reasoning, content and tokens.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const openai = @import("openai.zig");
const t = std.testing;

const bos = "<｜begin▁of▁sentence｜>";
const eos = "<｜end▁of▁sentence｜>";
const think_open = "<think>";
const think_close = "</think>";

/// The committed DeepSeek mini tokenizer, with the release's BOS and EOS named as its config names them.
const Fixture = struct {
    tmp: std.testing.TmpDir,
    path: [:0]u8,

    fn init(a: std.mem.Allocator) !Fixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, "zig/src/families/deepseek_v41/fixtures/serve/mini_tokenizer.json", a, .limited(1 << 20));
        // the release's byte-level decoder, so pieces join as the real tokenizer joins them
        const with_decoder = try std.mem.concat(a, u8, &.{ "{\"decoder\":{\"type\":\"ByteLevel\"},", bytes[1..] });
        try tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = with_decoder });
        try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"deepseek_v41\"}" });
        try tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer_config.json", .data = "{\"bos_token\":\"<｜begin▁of▁sentence｜>\",\"eos_token\":\"<｜end▁of▁sentence｜>\"}" });
        return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(t.io, ".", a) };
    }

    fn deinit(f: *Fixture) void {
        f.tmp.cleanup();
    }
};

/// Answers `reply` in chunks of `stride` tokens and keeps the prompt it was given.
const Script = struct {
    reply: []const u32,
    stride: usize,
    prompt: std.ArrayList(u32) = .empty,

    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        s.prompt.appendSlice(t.allocator, request.prompt) catch {};
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var start: usize = 0;
        while (start < s.reply.len) {
            const end = @min(start + s.stride, s.reply.len);
            sink.event(sink.ctx, id, &.{ .tokens = s.reply[start..end] });
            start = end;
        }
        const ended = s.reply.len > 0 and std.mem.indexOfScalar(u32, request.eos, s.reply[s.reply.len - 1]) != null;
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = if (ended) .stop else .length } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The route's whole reply or its stream's events, as JSON text.
const Got = struct {
    a: std.mem.Allocator,
    status: u16 = 0,
    body: []const u8 = "",
    events: std.ArrayList([]const u8) = .empty,

    fn out(g: *Got) openai.Out {
        return .{ .ctx = g, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }
    fn open(_: *anyopaque, _: usize) error{Closed}!void {}
    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        const text = json.stringify(g.a, payload orelse return, .{ .ascii = false }) catch return error.Closed;
        g.events.append(g.a, text) catch return error.Closed;
    }
    fn reply(ctx: *anyopaque, status_code: u16, payload: json.Value) void {
        const g: *Got = @ptrCast(@alignCast(ctx));
        g.status = status_code;
        g.body = json.stringify(g.a, payload, .{ .ascii = false }) catch "";
    }
};

/// What a client is left with: reasoning_content and content, joined across a stream's deltas.
const Seen = struct { reasoning: []const u8, content: []const u8, prompt_end: []const u8 };

fn ask(a: std.mem.Allocator, reply: []const u32, server_thinking: bool, extra: []const u8, stream: bool, stride: usize) !Seen {
    var fx = try Fixture.init(a);
    defer fx.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, fx.path, null, a, &problem);
    defer hf.deinit();
    var script: Script = .{ .reply = reply, .stride = stride };
    defer script.prompt.deinit(t.allocator);
    const srv = try Server.init(a, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = server_thinking, .use_drafts = false }, null);
    defer srv.deinit();
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":64,\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],\"stream\":{s}{s}}}", .{ if (stream) "true" else "false", extra });
    var got: Got = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    openai.run(srv, a, got.out(), .{ .conn = &conn }, true, (try json.parseText(a, body)).ok);
    const prompt_end = if (script.prompt.items.len > 0) try hf.text().tokenString(a, script.prompt.items[script.prompt.items.len - 1]) else "";
    var reasoning: std.ArrayList(u8) = .empty;
    var content: std.ArrayList(u8) = .empty;
    if (!stream) {
        try t.expectEqual(@as(u16, 200), got.status);
        const v = (try json.parseText(a, got.body)).ok;
        const m = v.get("choices").?.array[0].get("message").?;
        if (m.get("reasoning_content")) |r| if (r == .string) try reasoning.appendSlice(a, r.string);
        if (m.get("content")) |c| if (c == .string) try content.appendSlice(a, c.string);
    } else for (got.events.items) |e| {
        const v = (try json.parseText(a, e)).ok;
        const choices = v.get("choices") orelse continue;
        if (choices != .array or choices.array.len == 0) continue;
        const d = choices.array[0].get("delta") orelse continue;
        if (d.get("reasoning_content")) |r| if (r == .string) try reasoning.appendSlice(a, r.string);
        if (d.get("content")) |c| if (c == .string) try content.appendSlice(a, c.string);
    }
    return .{ .reasoning = reasoning.items, .content = content.items, .prompt_end = prompt_end };
}

fn ids(a: std.mem.Allocator, parts: []const Part) ![]const u32 {
    var fx = try Fixture.init(a);
    defer fx.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, fx.path, null, a, &problem);
    defer hf.deinit();
    var out: std.ArrayList(u32) = .empty;
    for (parts) |p| switch (p) {
        .id => |piece| try out.append(a, hf.text().tokenId(piece).?),
        .text => |s| try out.appendSlice(a, try hf.text().encode(a, s, false)),
    };
    return out.items;
}

const Part = union(enum) { id: []const u8, text: []const u8 };

/// Prints every path's split, and checks the stream ends where the whole reply does.
fn probe(a: std.mem.Allocator, name: []const u8, parts: []const Part, server_thinking: bool, extra: []const u8) !Seen {
    const reply = try ids(a, parts);
    const whole = try ask(a, reply, server_thinking, extra, false, 1);
    for ([_]usize{ 1, 2, 256 }) |stride| {
        const s = try ask(a, reply, server_thinking, extra, true, stride);
        if (!std.mem.eql(u8, s.content, whole.content) or !std.mem.eql(u8, std.mem.trim(u8, s.reasoning, " \n"), whole.reasoning))
            std.debug.print("DSREPLY {s} stream/{d} DIFFERS: reasoning={f} content={f}\n", .{ name, stride, std.json.fmt(s.reasoning, .{}), std.json.fmt(s.content, .{}) });
        try t.expectEqualStrings(whole.content, s.content);
        try t.expectEqualStrings(whole.reasoning, std.mem.trim(u8, s.reasoning, " \n"));
    }
    return whole;
}

test "DeepSeek chat: reply splits, whole and streamed, thinking on and off" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // thinking on: the prompt opens the block, the reply closes it with its own token
    var s = try probe(a, "on/closed", &.{ .{ .text = "Plan" }, .{ .id = think_close }, .{ .text = "Hi" }, .{ .id = eos } }, true, "");
    try t.expectEqualStrings(think_open, s.prompt_end);
    try t.expectEqualStrings("Plan", s.reasoning);
    try t.expectEqualStrings("Hi", s.content);
    // cut by the length while still reasoning: no content
    s = try probe(a, "on/cut", &.{ .{ .text = "Plan" }, .{ .text = "Hi" } }, true, "");
    try t.expectEqualStrings("", s.content);
    // ended (EOS) before any close: all reasoning
    s = try probe(a, "on/no-close", &.{ .{ .text = "Plan" }, .{ .id = eos } }, true, "");
    try t.expectEqualStrings("", s.content);
    // a reply that writes its own opener after the prompt's
    s = try probe(a, "on/own-opener", &.{ .{ .id = think_open }, .{ .text = "Plan" }, .{ .id = think_close }, .{ .text = "Hi" }, .{ .id = eos } }, true, "");
    try t.expectEqualStrings("Hi", s.content);
    // thinking off: the prompt closes the block; reasoning the model writes anyway is content
    s = try probe(a, "off/closes-anyway", &.{ .{ .text = "Plan" }, .{ .id = think_close }, .{ .text = "Hi" }, .{ .id = eos } }, false, "");
    try t.expectEqualStrings(think_close, s.prompt_end);
    // a BOS the model emits mid-reply, and text after the EOS (which the engine would not send)
    try t.expectEqualStrings("Plan</think>Hi", s.content);
    // a BOS the model emits mid-reply is decoded as its spelling (special tokens are not skipped), whole and streamed
    s = try probe(a, "on/bos-mid", &.{ .{ .text = "Plan" }, .{ .id = think_close }, .{ .text = "Hi" }, .{ .id = bos }, .{ .text = "Hi" }, .{ .id = eos } }, true, "");
    try t.expectEqualStrings("Hi" ++ bos ++ "Hi", s.content);
    s = try probe(a, "off/bos-mid", &.{ .{ .text = "Hi" }, .{ .id = bos }, .{ .text = "Hi" }, .{ .id = eos } }, false, "");
    s = try probe(a, "on/bos-in-reasoning", &.{ .{ .text = "Plan" }, .{ .id = bos }, .{ .id = think_close }, .{ .text = "Hi" }, .{ .id = eos } }, true, "");
    try t.expectEqualStrings("Plan" ++ bos, s.reasoning);
    s = try probe(a, "off/after-eos", &.{ .{ .text = "Hi" }, .{ .id = eos }, .{ .text = "Plan" } }, false, "");
    try t.expectEqualStrings("Hi", s.content);
}

test "DeepSeek chat: each request shape for thinking off reaches the prompt" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try ids(a, &.{ .{ .text = "Hi" }, .{ .id = eos } });
    const shapes = [_][]const u8{
        ",\"chat_template_kwargs\":{\"enable_thinking\":false}",
        ",\"chat_template_kwargs\":{\"thinking\":false}",
        ",\"chat_template_kwargs\":{\"thinking\":{\"type\":\"disabled\"}}",
        ",\"reasoning_effort\":\"none\"",
        // DeepSeek's own API (and Hermes Agent's DeepSeek profile): extra_body {"thinking": {"type": "disabled"}}
        ",\"thinking\":{\"type\":\"disabled\"}",
    };
    for (shapes) |shape| for ([_]bool{ false, true }) |stream| {
        const s = try ask(a, reply, true, shape, stream, 1);
        try t.expectEqualStrings(think_close, s.prompt_end);
        try t.expectEqualStrings("Hi", s.content);
    };
    // and on, against a server started with --no-thinking
    for ([_][]const u8{ ",\"chat_template_kwargs\":{\"enable_thinking\":true}", ",\"thinking\":{\"type\":\"enabled\"}" }) |shape| {
        const s = try ask(a, reply, false, shape, false, 1);
        try t.expectEqualStrings(think_open, s.prompt_end);
    }
}
