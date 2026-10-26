//! Recipient replies (Muse Spark's template) split into reasoning, answer and calls, while Harmony replies keep theirs.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const recipient_reply = @import("recipient_reply.zig");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const Conn = @import("http_conn.zig").Conn;

const muse_markers = [_][]const u8{ "<|start|>", "<|message|>", "<|eom|>", "<|eot|>" };
const harmony_markers = [_][]const u8{ "<|start|>", "<|message|>", "<|channel|>", "<|end|>", "<|return|>", "<|call|>" };
const muse_eos = [_]u32{303}; // <|eot|>
const harmony_eos = [_]u32{304}; // <|return|>
const muse_template = "{%- set reasoning_effort_map = {'low': 8, 'xhigh': 512} -%}{{- '<|start|>assistant to=self<|message|>' -}}";
const harmony_template = "{{- '<|start|>assistant<|channel|>commentary to=functions.' ~ name ~ '<|message|>' -}}";
const all_bytes = blk: {
    var all: [256]u8 = undefined;
    for (&all, 0..) |*b, i| b.* = i;
    break :blk all;
};

/// Bytes as tokens and each marker of the format as one token from 300; every prompt renders as `prompt`.
pub const Text = struct {
    harmony: bool = false,
    prompt: []const u8 = "prompt<|start|>assistant", // rendered after `effort=E;`, the effort the template heard

    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn markers(t: *const Text) []const []const u8 {
        return if (t.harmony) &harmony_markers else &muse_markers;
    }

    fn piece(t: *const Text, id: u32) []const u8 {
        return if (id >= 300) t.markers()[id - 300] else all_bytes[id..][0..1];
    }

    /// The reply's tokens: each marker the text spells is its token, every other byte its own.
    pub fn tokens(t: *const Text, a: std.mem.Allocator, s: []const u8) ![]const u32 {
        var out: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        next: while (i < s.len) {
            for (t.markers(), 0..) |m, k| if (std.mem.startsWith(u8, s[i..], m)) {
                try out.append(a, @intCast(300 + k));
                i += m.len;
                continue :next;
            };
            try out.append(a, s[i]);
            i += 1;
        }
        return out.items;
    }

    fn encode(ctx: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        return @constCast(try t.tokens(a, input));
    }

    fn decode(ctx: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| try out.appendSlice(a, t.piece(id));
        return out.items;
    }

    fn tokenId(ctx: *anyopaque, p: []const u8) ?u32 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        for (t.markers(), 0..) |m, k| if (std.mem.eql(u8, m, p)) return @as(u32, @intCast(300 + k));
        return null;
    }

    fn tokenString(ctx: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return decode(ctx, a, &.{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 310;
    }

    fn eosIds(ctx: *anyopaque) []const u32 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        return if (t.harmony) &harmony_eos else &muse_eos;
    }

    fn render(ctx: *anyopaque, a: std.mem.Allocator, _: json.Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        const t: *Text = @ptrCast(@alignCast(ctx));
        return .{ .text = try std.mem.concat(a, u8, &.{ "effort=", o.reasoning_effort orelse "none", ";", t.prompt }) };
    }

    fn templateSource(ctx: *anyopaque) []const u8 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        return if (t.harmony) harmony_template else muse_template;
    }
};

/// Answers `reply` in chunks of `stride` tokens.
pub const Script = struct {
    reply: []const u32,
    stride: usize,
    eos: u32,
    effort: [8]u8 = undefined, // the effort the submitted prompt was rendered with
    effort_len: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        const named = request.prompt["effort=".len..];
        e.effort_len = std.mem.indexOfScalar(u32, named, ';').?;
        for (named[0..e.effort_len], e.effort[0..e.effort_len]) |t, *b| b.* = @intCast(t);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var at: usize = 0;
        while (at < e.reply.len) : (at = @min(at + e.stride, e.reply.len)) {
            sink.event(sink.ctx, id, &.{ .tokens = e.reply[at..@min(at + e.stride, e.reply.len)] });
        }
        const ended = e.reply.len > 0 and e.reply[e.reply.len - 1] == e.eos;
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

/// What a reply showed its client, whole or streamed: reasoning, content, the calls begun and their arguments.
pub const Shown = struct {
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    arguments: std.ArrayList(u8) = .empty,
    calls: usize = 0,
    status: u16 = 0,
    length: bool = false, // finish_reason "length"
    effort: [8]u8 = undefined,
    effort_len: usize = 0,

    fn out(s: *Shown) openai.Out {
        return .{ .ctx = s, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    pub fn deinit(s: *Shown) void {
        for ([_]*std.ArrayList(u8){ &s.reasoning, &s.content, &s.arguments }) |l| l.deinit(std.testing.allocator);
    }

    /// One message or delta object: its text fields appended, each call with an id counted, arguments appended.
    fn take(s: *Shown, choice: json.Value, key: []const u8) !void {
        if (choice.strField("finish_reason")) |f| s.length = s.length or std.mem.eql(u8, f, "length");
        const m = choice.get(key) orelse return;
        const g = std.testing.allocator;
        if (m.get("reasoning_content")) |v| if (v == .string) try s.reasoning.appendSlice(g, v.string);
        if (m.get("content")) |v| if (v == .string) try s.content.appendSlice(g, v.string);
        const list = m.get("tool_calls") orelse return;
        if (list != .array) return;
        for (list.array) |call| {
            if (call.get("id") != null) s.calls += 1;
            const f = call.get("function") orelse continue;
            if (f.strField("arguments")) |args| try s.arguments.appendSlice(g, args);
        }
    }

    fn open(_: *anyopaque, _: usize) error{Closed}!void {}

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const s: *Shown = @ptrCast(@alignCast(ctx));
        const p = payload orelse return;
        const choices = p.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        s.take(choices.array[0], "delta") catch return error.Closed;
    }

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const s: *Shown = @ptrCast(@alignCast(ctx));
        s.status = status;
        const choices = payload.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        s.take(choices.array[0], "message") catch return;
    }
};

/// `body` answered with `reply`'s tokens, `stride` at a time; the engine stops at the end token or after the reply.
pub fn send(settings: Text, reply: []const u8, stride: usize, body: []const u8) !Shown {
    var text = settings;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var backend: Script = .{ .reply = try text.tokens(a, reply), .stride = stride, .eos = Text.eosIds(&text)[0] };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = true,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const raw = (try json.parse(a, body)).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var shown: Shown = .{};
    openai.run(srv, a, shown.out(), .{ .conn = &conn }, true, raw);
    @memcpy(shown.effort[0..backend.effort_len], backend.effort[0..backend.effort_len]);
    shown.effort_len = backend.effort_len;
    return shown;
}

const ask = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":1024";
const no_thinking = ",\"chat_template_kwargs\":{\"enable_thinking\":false}";
const weather = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}}}}]";

fn atem(comptime city: []const u8, comptime days: []const u8) []const u8 {
    return "<atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">" ++ city ++
        "</atem:parameter>\n<atem:parameter name=\"days\">" ++ days ++ "</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
}

const chat_reply = " to=self<|message|>The sky question is simple.<|eom|><|start|>assistant to=user<|message|>Blue sky.<|eot|>";
const call_reply = " to=self<|message|>Two forecasts are needed.<|eom|><|start|>assistant to=commentary<|message|>Checking both." ++
    "<|eom|><|start|>assistant to=get_weather<|message|>" ++ atem("Paris", "3") ++ "<|eom|><|start|>assistant to=get_weather" ++
    "<|message|>" ++ atem("Oslo", "1") ++ "<|eot|>";
const harmony_reply = "<|channel|>analysis<|message|>Think.<|end|><|start|>assistant<|channel|>final<|message|>Blue sky.<|return|>";

test "recipient replies: detected from the markers and the template, never for Harmony's tokens" {
    var muse: Text = .{};
    var harmony: Text = .{ .harmony = true };
    try std.testing.expect(recipient_reply.detect(muse.text()));
    try std.testing.expect(!recipient_reply.detect(harmony.text()));
}

test "recipient replies: a split while streaming only grows, holds calls back, and the finished split extends it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = call_reply[0 .. call_reply.len - "<|eot|>".len];
    var reading: recipient_reply.Reading = .{ .lead = "assistant", .tools = (try json.parse(a, weather[",\"tools\":".len..])).ok.array };
    const done = try reading.split(a, reply, .ended);
    try std.testing.expectEqualStrings("Two forecasts are needed.", done.reasoning);
    const xml = "<tool_call>\n<function=get_weather>\n<parameter=city>\n{s}\n</parameter>\n<parameter=days>\n{s}\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const want = try std.fmt.allocPrint(a, "Checking both.\n\n" ++ xml ++ "\n\n" ++ xml, .{ "Paris", "3", "Oslo", "1" });
    try std.testing.expectEqualStrings(want, done.answer);
    var last: reply_text.Split = .{ .reasoning = "", .answer = "" };
    for (1..reply.len + 1) |n| {
        const s = try reading.split(a, reply[0..n], .streaming);
        try std.testing.expect(std.mem.startsWith(u8, s.reasoning, last.reasoning));
        try std.testing.expect(std.mem.startsWith(u8, s.answer, last.answer));
        try std.testing.expect(std.mem.indexOf(u8, s.answer, "<|") == null and std.mem.indexOf(u8, s.answer, "atem") == null);
        try std.testing.expect(std.mem.startsWith(u8, done.answer, s.answer));
        last = s;
    }
    try std.testing.expect(std.mem.indexOf(u8, last.answer, "Oslo") == null); // the open call waits for the reply's end
    try std.testing.expectEqualStrings("plain", (try reading.split(a, " plain ", .ended)).answer);
}

test "recipient replies: atem calls become OpenAI calls with schema-typed arguments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, weather[",\"tools\":".len..])).ok.array;
    const body = atem("Rio de Janeiro", "4");
    const parsed = try tool_parse.parse(a, (try recipient_reply.calls(a, body, "get_weather", tools, .ended)).?, tools, null);
    try std.testing.expectEqualStrings("", parsed.content);
    const f = parsed.calls.?[0].get("function").?;
    try std.testing.expectEqualStrings("get_weather", f.strField("name").?);
    try std.testing.expectEqualStrings("{\"city\":\"Rio de Janeiro\",\"days\":4}", f.strField("arguments").?);
    const amp = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"a&b\"}}]")).ok.array;
    const named = (try recipient_reply.calls(a, "<atem:function_calls>\n<atem:invoke name=\"a&amp;b\">\n</atem:invoke>", "a&b", amp, .ended)).?;
    try std.testing.expectEqualStrings("<tool_call>\n<function=a&b>\n</function>\n</tool_call>", named);
    try std.testing.expect((try recipient_reply.calls(a, "no calls here", "get_weather", tools, .ended)) == null);
}

test "recipient replies: whole and streamed chats show the same reasoning, answer and calls" {
    const cases = [_]struct { reply: []const u8, body: []const u8, content: []const u8, reasoning: []const u8, arguments: []const u8, calls: usize }{
        .{ .reply = chat_reply, .body = ask ++ "}", .content = "Blue sky.", .reasoning = "The sky question is simple.", .arguments = "", .calls = 0 },
        .{ .reply = call_reply, .body = ask ++ weather ++ "}", .content = "Checking both.", .reasoning = "Two forecasts are needed.", .arguments = "{\"city\":\"Paris\",\"days\":3}{\"city\":\"Oslo\",\"days\":1}", .calls = 2 },
    };
    for (cases) |c| {
        var whole = try send(.{}, c.reply, 4096, c.body);
        defer whole.deinit();
        try std.testing.expectEqual(@as(u16, 200), whole.status);
        for ([_]usize{ 1, 7, 4096 }) |stride| {
            const streamed_body = try std.mem.concat(std.testing.allocator, u8, &.{ c.body[0 .. c.body.len - 1], ",\"stream\":true}" });
            defer std.testing.allocator.free(streamed_body);
            var streamed = try send(.{}, c.reply, stride, streamed_body);
            defer streamed.deinit();
            for ([_]*const Shown{ &whole, &streamed }) |s| {
                try std.testing.expectEqualStrings(c.content, s.content.items);
                try std.testing.expectEqualStrings(c.reasoning, s.reasoning.items);
                try std.testing.expectEqualStrings(c.arguments, s.arguments.items);
                try std.testing.expectEqual(c.calls, s.calls);
            }
        }
    }
}

test "recipient replies: a Harmony reply keeps its final channel as content and analysis as reasoning" {
    var whole = try send(.{ .harmony = true }, harmony_reply, 4096, ask ++ no_thinking ++ "}");
    defer whole.deinit();
    var streamed = try send(.{ .harmony = true }, harmony_reply, 1, ask ++ no_thinking ++ ",\"stream\":true}");
    defer streamed.deinit();
    for ([_]*const Shown{ &whole, &streamed }) |s| {
        try std.testing.expectEqualStrings("Blue sky.", s.content.items);
        try std.testing.expectEqualStrings("Think.", s.reasoning.items);
    }
}

test "recipient replies: a prompt that opened the self message keeps the reasoning that follows it" {
    const opened: Text = .{ .prompt = "prompt<|start|>assistant to=self<|message|>" };
    const reply = "The sky question is simple.<|eom|><|start|>assistant to=user<|message|>Blue sky.<|eot|>";
    var whole = try send(opened, reply, 4096, ask ++ "}");
    defer whole.deinit();
    for ([_]usize{ 1, 7, 4096 }) |stride| {
        var streamed = try send(opened, reply, stride, ask ++ ",\"stream\":true}");
        defer streamed.deinit();
        for ([_]*const Shown{ &whole, &streamed }) |s| {
            try std.testing.expectEqualStrings("Blue sky.", s.content.items);
            try std.testing.expectEqualStrings("The sky question is simple.", s.reasoning.items);
        }
        // cut while it still reasons: the reasoning so far, no answer
        var cut = try send(opened, "Still weighing the sky", stride, ask ++ ",\"stream\":true}");
        defer cut.deinit();
        try std.testing.expectEqualStrings("Still weighing the sky", cut.reasoning.items);
        try std.testing.expectEqualStrings("", cut.content.items);
    }
}

test "recipient replies: a call to no offered tool stays as the model wrote it, never as other markup" {
    const time = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"parameters\":{\"type\":\"object\"}}}]";
    const want = comptime "Checking both.\n\n" ++ atem("Paris", "3") ++ "\n\n" ++ atem("Oslo", "1");
    for ([_][]const u8{ "", time }) |tools| {
        const body = try std.mem.concat(std.testing.allocator, u8, &.{ ask, tools, "}" });
        defer std.testing.allocator.free(body);
        var whole = try send(.{}, call_reply, 4096, body);
        defer whole.deinit();
        for ([_]usize{ 1, 7, 4096 }) |stride| {
            const streamed_body = try std.mem.concat(std.testing.allocator, u8, &.{ ask, tools, ",\"stream\":true}" });
            defer std.testing.allocator.free(streamed_body);
            var streamed = try send(.{}, call_reply, stride, streamed_body);
            defer streamed.deinit();
            for ([_]*const Shown{ &whole, &streamed }) |s| {
                try std.testing.expectEqualStrings(want, s.content.items);
                try std.testing.expectEqual(@as(usize, 0), s.calls);
            }
        }
    }
}

test "recipient replies: thinking off keeps self messages out of the reply and the template at its lowest effort" {
    for ([_][]const u8{ no_thinking, ",\"reasoning_effort\":\"none\"" }) |off| {
        const body = try std.mem.concat(std.testing.allocator, u8, &.{ ask, off, "}" });
        defer std.testing.allocator.free(body);
        var whole = try send(.{}, chat_reply, 4096, body);
        defer whole.deinit();
        const streamed_body = try std.mem.concat(std.testing.allocator, u8, &.{ ask, off, ",\"stream\":true}" });
        defer std.testing.allocator.free(streamed_body);
        for ([_]usize{ 1, 7 }) |stride| {
            var streamed = try send(.{}, chat_reply, stride, streamed_body);
            defer streamed.deinit();
            for ([_]*const Shown{ &whole, &streamed }) |s| {
                try std.testing.expectEqualStrings("Blue sky.", s.content.items);
                try std.testing.expectEqualStrings("", s.reasoning.items);
                try std.testing.expectEqualStrings("low", s.effort[0..s.effort_len]);
            }
        }
    }
    // thinking on: the template's default effort, or the named level nearest the request's
    var on = try send(.{}, chat_reply, 4096, ask ++ "}");
    defer on.deinit();
    try std.testing.expectEqualStrings("none", on.effort[0..on.effort_len]);
    try std.testing.expectEqualStrings("The sky question is simple.", on.reasoning.items);
    var high = try send(.{}, chat_reply, 4096, ask ++ ",\"reasoning_effort\":\"high\"}");
    defer high.deinit();
    try std.testing.expectEqualStrings("xhigh", high.effort[0..high.effort_len]);
}

test "recipient replies: a streamed read goes on from the last call and equals a fresh read at every length" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, weather[",\"tools\":".len..])).ok.array;
    for ([_][]const u8{ chat_reply, call_reply, " to=self<|message|>x<|eo" }) |reply| for ([_][]const u8{ "assistant", "assistant to=self<|message|>" }) |lead| {
        var going: recipient_reply.Reading = .{ .lead = lead, .tools = tools };
        for (1..reply.len + 1) |n| {
            var fresh: recipient_reply.Reading = .{ .lead = lead, .tools = tools };
            const want = try fresh.split(a, reply[0..n], .streaming);
            const got = try going.split(a, reply[0..n], .streaming);
            try std.testing.expectEqualStrings(want.reasoning, got.reasoning);
            try std.testing.expectEqualStrings(want.answer, got.answer);
        }
        for ([_]recipient_reply.End{ .ended, .cut }) |end| {
            var fresh: recipient_reply.Reading = .{ .lead = lead, .tools = tools };
            var done = going;
            const want = try fresh.split(a, reply, end);
            const got = try done.split(a, reply, end);
            try std.testing.expectEqualStrings(want.reasoning, got.reasoning);
            try std.testing.expectEqualStrings(want.answer, got.answer);
        }
    };
}

test "recipient replies: a reply with no markup streams as it comes; one that may still be a header waits" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var r: recipient_reply.Reading = .{ .lead = "assistant" };
    const reply = " Blue sky, mostly. ";
    var last: []const u8 = "";
    for (1..reply.len + 1) |n| {
        const s = (try r.split(a, reply[0..n], .streaming)).answer;
        try std.testing.expect(std.mem.startsWith(u8, s, last) and std.mem.startsWith(u8, "Blue sky, mostly.", s));
        last = s;
    }
    try std.testing.expectEqualStrings("Blue sky, mostly", (try r.split(a, reply[0 .. reply.len - 2], .streaming)).answer);
    try std.testing.expectEqualStrings("Blue sky, mostly.", (try r.split(a, reply, .ended)).answer);
    var h: recipient_reply.Reading = .{ .lead = "assistant" };
    for ([_][]const u8{ " t", " to", " to=se", " <" }) |p| try std.testing.expectEqualStrings("", (try h.split(a, p, .streaming)).answer);
    try std.testing.expectEqualStrings("hi", (try h.split(a, " to=user<|message|>hi", .streaming)).answer);
}
