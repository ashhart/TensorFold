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
const muse_template = "{{- '<|start|>assistant to=self<|message|>' -}}";
const harmony_template = "{{- '<|start|>assistant<|channel|>commentary to=functions.' ~ name ~ '<|message|>' -}}";
const all_bytes = blk: {
    var all: [256]u8 = undefined;
    for (&all, 0..) |*b, i| b.* = i;
    break :blk all;
};

/// Bytes as tokens and each marker of the format as one token from 300; every prompt renders as "prompt".
const Text = struct {
    harmony: bool = false,

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
    fn tokens(t: *const Text, a: std.mem.Allocator, s: []const u8) ![]const u32 {
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

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
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

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try (a.dupe(u8, "prompt")) };
    }

    fn templateSource(ctx: *anyopaque) []const u8 {
        const t: *Text = @ptrCast(@alignCast(ctx));
        return if (t.harmony) harmony_template else muse_template;
    }
};

/// Answers `reply` in chunks of `stride` tokens.
const Script = struct {
    reply: []const u32,
    stride: usize,
    eos: u32,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
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
const Shown = struct {
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    arguments: std.ArrayList(u8) = .empty,
    calls: usize = 0,
    status: u16 = 0,

    fn out(s: *Shown) openai.Out {
        return .{ .ctx = s, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn deinit(s: *Shown) void {
        for ([_]*std.ArrayList(u8){ &s.reasoning, &s.content, &s.arguments }) |l| l.deinit(std.testing.allocator);
    }

    /// One message or delta object: its text fields appended, each call with an id counted, arguments appended.
    fn take(s: *Shown, m: json.Value) !void {
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
        const delta = choices.array[0].get("delta") orelse return;
        s.take(delta) catch return error.Closed;
    }

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const s: *Shown = @ptrCast(@alignCast(ctx));
        s.status = status;
        const choices = payload.get("choices") orelse return;
        if (choices != .array or choices.array.len == 0) return;
        s.take(choices.array[0].get("message") orelse return) catch return;
    }
};

fn send(settings: Text, reply: []const u8, stride: usize, body: []const u8) !Shown {
    var text = settings;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var backend: Script = .{ .reply = try text.tokens(a, reply), .stride = stride, .eos = Text.eosIds(&text)[0] };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const raw = (try json.parse(a, body)).ok;
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var shown: Shown = .{};
    openai.run(srv, a, shown.out(), .{ .conn = &conn }, true, raw);
    return shown;
}

const ask = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":1024";
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
    const done = try recipient_reply.split(a, reply, true);
    try std.testing.expectEqualStrings("Two forecasts are needed.", done.reasoning);
    const xml = "<tool_call>\n<function=get_weather>\n<parameter=city>\n{s}\n</parameter>\n<parameter=days>\n{s}\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const want = try std.fmt.allocPrint(a, "Checking both.\n\n" ++ xml ++ "\n\n" ++ xml, .{ "Paris", "3", "Oslo", "1" });
    try std.testing.expectEqualStrings(want, done.answer);
    var last: reply_text.Split = .{ .reasoning = "", .answer = "" };
    for (1..reply.len + 1) |n| {
        const s = try recipient_reply.split(a, reply[0..n], false);
        try std.testing.expect(std.mem.startsWith(u8, s.reasoning, last.reasoning));
        try std.testing.expect(std.mem.startsWith(u8, s.answer, last.answer));
        try std.testing.expect(std.mem.indexOf(u8, s.answer, "<|") == null and std.mem.indexOf(u8, s.answer, "atem") == null);
        try std.testing.expect(std.mem.startsWith(u8, done.answer, s.answer));
        last = s;
    }
    try std.testing.expect(std.mem.indexOf(u8, last.answer, "Oslo") == null); // the open call waits for the reply's end
    try std.testing.expectEqualStrings("plain", (try recipient_reply.split(a, " plain ", true)).answer);
}

test "recipient replies: atem calls become OpenAI calls with schema-typed arguments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, weather[",\"tools\":".len..])).ok.array;
    const body = atem("Rio de Janeiro", "4");
    const parsed = try tool_parse.parse(a, (try recipient_reply.calls(a, body)).?, tools, null);
    try std.testing.expectEqualStrings("", parsed.content);
    const f = parsed.calls.?[0].get("function").?;
    try std.testing.expectEqualStrings("get_weather", f.strField("name").?);
    try std.testing.expectEqualStrings("{\"city\":\"Rio de Janeiro\",\"days\":4}", f.strField("arguments").?);
    const named = (try recipient_reply.calls(a, "<atem:invoke name=\"a&amp;b\">\n</atem:invoke>")).?;
    try std.testing.expectEqualStrings("<tool_call>\n<function=a&b>\n</function>\n</tool_call>", named);
    try std.testing.expect((try recipient_reply.calls(a, "no calls here")) == null);
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
    var whole = try send(.{ .harmony = true }, harmony_reply, 4096, ask ++ "}");
    defer whole.deinit();
    var streamed = try send(.{ .harmony = true }, harmony_reply, 1, ask ++ ",\"stream\":true}");
    defer streamed.deinit();
    for ([_]*const Shown{ &whole, &streamed }) |s| {
        try std.testing.expectEqualStrings("Blue sky.", s.content.items);
        try std.testing.expectEqualStrings("Think.", s.reasoning.items);
    }
}
