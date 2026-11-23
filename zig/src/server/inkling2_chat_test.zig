//! Inkling 2's release reply grammar through HfText and the actual OpenAI route.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const openai = @import("openai.zig");
const t = std.testing;

const sample = "<|message_model|><|content_thinking|>The user asks: ... 340 + 51 = 391. ... <|end_message|><|message_model|><|content_text|>391<|end_message|>";
const thought = "The user asks: ... 340 + 51 = 391. ... ";

/// The committed release tokenizer and template, with only the served model type changed by a case.
const Fixture = struct {
    tmp: std.testing.TmpDir,
    path: [:0]u8,

    fn init(a: std.mem.Allocator, kind: []const u8) !Fixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        for ([_][2][]const u8{
            .{ "zig/src/families/inkling2/fixtures/mini_tokenizer.json", "tokenizer.json" },
            .{ "zig/src/families/inkling2/fixtures/chat_template.jinja", "chat_template.jinja" },
        }) |file| {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(t.io, file[0], a, .limited(1 << 20));
            try tmp.dir.writeFile(t.io, .{ .sub_path = file[1], .data = bytes });
        }
        try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = try std.fmt.allocPrint(a, "{{\"model_type\":\"{s}\"}}", .{kind}) });
        // The cut tokenizer's base vocabulary ends below the release EOS id; id 2 only drives the fake engine's finish.
        try tmp.dir.writeFile(t.io, .{ .sub_path = "generation_config.json", .data = "{\"eos_token_id\":2}" });
        return .{ .tmp = tmp, .path = try tmp.dir.realPathFileAlloc(t.io, ".", a) };
    }

    fn deinit(f: *Fixture) void {
        f.tmp.cleanup();
    }
};

/// Answers `reply` in chunks of `stride` tokens.
const Script = struct {
    reply: []const u32,
    stride: usize,

    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
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

/// What a client receives, joined across whole replies or stream deltas.
const Seen = struct {
    reasoning: []const u8,
    content: []const u8,
    ids: []const u8,
    names: []const u8,
    arguments: []const u8,
    rejected: i64 = 0, // the tensorfold block's rejected_tool_calls
};

fn rejectedOf(v: json.Value) i64 {
    const tf = v.get("tensorfold") orelse return 0;
    const n = tf.get("rejected_tool_calls") orelse return 0;
    return if (n == .int) std.fmt.parseInt(i64, n.int, 10) catch 0 else 0;
}

fn appendCall(a: std.mem.Allocator, call: json.Value, ids: *std.ArrayList(u8), names: *std.ArrayList(u8), arguments: *std.ArrayList(u8)) !void {
    if (call.get("id")) |v| if (v == .string) try ids.appendSlice(a, v.string);
    const f = call.get("function") orelse return;
    if (f.get("name")) |v| if (v == .string) try names.appendSlice(a, v.string);
    if (f.get("arguments")) |v| if (v == .string) try arguments.appendSlice(a, v.string);
}

fn run(a: std.mem.Allocator, kind: []const u8, reply: []const u32, extra: []const u8, stream: bool, stride: usize, request: []const u8, chat_route: bool) !Seen {
    var fx = try Fixture.init(a, kind);
    defer fx.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, fx.path, null, a, &problem);
    defer hf.deinit();
    var script: Script = .{ .reply = reply, .stride = stride };
    const srv = try Server.init(a, t.io, script.engine(), hf.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":256,{s},\"stream\":{s}{s}}}", .{ request, if (stream) "true" else "false", extra });
    var got: Got = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    openai.run(srv, a, got.out(), .{ .conn = &conn }, chat_route, (try json.parseText(a, body)).ok);
    var reasoning: std.ArrayList(u8) = .empty;
    var content: std.ArrayList(u8) = .empty;
    var ids: std.ArrayList(u8) = .empty;
    var names: std.ArrayList(u8) = .empty;
    var arguments: std.ArrayList(u8) = .empty;
    var rejected: i64 = 0;
    if (!stream) {
        try t.expectEqual(@as(u16, 200), got.status);
        const v = (try json.parseText(a, got.body)).ok;
        rejected = rejectedOf(v);
        const first = v.get("choices").?.array[0];
        const m = if (first.get("message")) |message| message else first;
        if (m.get("reasoning_content")) |r| if (r == .string) try reasoning.appendSlice(a, r.string);
        if (m.get("content")) |c| if (c == .string) try content.appendSlice(a, c.string);
        if (m.get("text")) |c| if (c == .string) try content.appendSlice(a, c.string);
        if (m.get("tool_calls")) |calls| if (calls == .array) for (calls.array) |call| try appendCall(a, call, &ids, &names, &arguments);
    } else for (got.events.items) |e| {
        const v = (try json.parseText(a, e)).ok;
        rejected += rejectedOf(v);
        const choices = v.get("choices") orelse continue;
        if (choices != .array or choices.array.len == 0) continue;
        const first = choices.array[0];
        const d = if (first.get("delta")) |delta| delta else first;
        if (d.get("reasoning_content")) |r| if (r == .string) try reasoning.appendSlice(a, r.string);
        if (d.get("content")) |c| if (c == .string) try content.appendSlice(a, c.string);
        if (d.get("text")) |c| if (c == .string) try content.appendSlice(a, c.string);
        if (d.get("tool_calls")) |calls| if (calls == .array) for (calls.array) |call| try appendCall(a, call, &ids, &names, &arguments);
    }
    return .{ .reasoning = reasoning.items, .content = content.items, .ids = ids.items, .names = names.items, .arguments = arguments.items, .rejected = rejected };
}

fn ask(a: std.mem.Allocator, kind: []const u8, reply: []const u32, extra: []const u8, stream: bool, stride: usize) !Seen {
    return run(a, kind, reply, extra, stream, stride, "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]", true);
}

fn askRaw(a: std.mem.Allocator, reply: []const u32, stream: bool, stride: usize) !Seen {
    return run(a, "inkling2", reply, "", stream, stride, "\"prompt\":\"Hi\"", false);
}

fn encodeReply(a: std.mem.Allocator, raw: []const u8) ![]const u32 {
    var fx = try Fixture.init(a, "inkling2");
    defer fx.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, fx.path, null, a, &problem);
    defer hf.deinit();
    return hf.text().encode(a, raw, false);
}

/// Compare the whole body with streams at several engine chunk widths.
fn probe(a: std.mem.Allocator, raw: []const u8, extra: []const u8) !Seen {
    const reply = try encodeReply(a, raw);
    const whole = try ask(a, "inkling2", reply, extra, false, 1);
    for ([_]usize{ 1, 2, 3, 256 }) |stride| {
        const s = try ask(a, "inkling2", reply, extra, true, stride);
        try t.expectEqualStrings(whole.content, s.content);
        try t.expectEqualStrings(whole.reasoning, s.reasoning);
        try t.expectEqualStrings(whole.ids, s.ids);
        try t.expectEqualStrings(whole.names, s.names);
        try t.expectEqualStrings(whole.arguments, s.arguments);
    }
    return whole;
}

test "Inkling 2 reply: the supplied raw reply splits without exposing controls or requiring EOS" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const got = try probe(arena.allocator(), sample, "");
    try t.expectEqualStrings(thought, got.reasoning);
    try t.expectEqualStrings("391", got.content);
    try t.expectEqualStrings("", got.ids);
    try t.expectEqualStrings("", got.names);
    try t.expectEqualStrings("", got.arguments);
}

test "Inkling 2 reply: repeated blocks and a native JSON tool call agree whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const raw = "<|message_model|><|content_thinking|>Check <|end_message|>" ++
        "<|message_model|><|content_text|>Before<|end_message|>" ++
        "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"q\":\"x\",\"nested\":{\"ok\":true}}}<|end_message|>" ++
        "<|content_text|> After<|end_message|><|content_model_end_sampling|>";
    const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    const got = try probe(arena.allocator(), raw, tools);
    try t.expectEqualStrings("Check ", got.reasoning);
    try t.expectEqualStrings("Before After", got.content);
    try t.expectEqualStrings("call_0", got.ids);
    try t.expectEqualStrings("lookup", got.names);
    try t.expectEqualStrings("{\"q\":\"x\",\"nested\":{\"ok\":true}}", got.arguments);
}

test "Inkling 2 reply: another model keeps the same token bytes on the previous generic path" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try encodeReply(a, sample);
    const whole = try ask(a, "not_inkling2", reply, "", false, 1);
    try t.expectEqualStrings(sample, whole.content);
    try t.expectEqualStrings("", whole.reasoning);
    for ([_]usize{ 1, 2, 256 }) |stride| {
        const streamed = try ask(a, "not_inkling2", reply, "", true, stride);
        try t.expectEqualStrings(sample, streamed.content);
        try t.expectEqualStrings("", streamed.reasoning);
    }
}

test "Inkling 2 reply: raw completions keep control spellings literal whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const reply = try encodeReply(a, sample);
    const whole = try askRaw(a, reply, false, 1);
    try t.expectEqualStrings(sample, whole.content);
    try t.expectEqualStrings("", whole.reasoning);
    for ([_]usize{ 1, 2, 256 }) |stride| {
        const streamed = try askRaw(a, reply, true, stride);
        try t.expectEqualStrings(sample, streamed.content);
        try t.expectEqualStrings("", streamed.reasoning);
    }
}

test "Inkling 2 reply: the tokenizer's other special tokens inside blocks never reach the client" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const raw = "<|message_model|><|content_thinking|>a<|message_system|>b<|end_message|>" ++
        "<|message_model|><|content_text|>A<|message_user|>B<|endoftext|>C<|content_xml|>D<|end_message|>";
    const got = try probe(arena.allocator(), raw, "");
    try t.expectEqualStrings("ab", got.reasoning);
    try t.expectEqualStrings("ABCD", got.content);
}

test "Inkling 2 reply: a refused call is counted for the client, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = "<|message_model|>other<|content_invoke_tool_json|>{\"name\":\"other\",\"args\":{}}<|end_message|>";
    const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    const reply = try encodeReply(a, raw);
    const whole = try ask(a, "inkling2", reply, tools, false, 1);
    try t.expectEqual(@as(i64, 1), whole.rejected);
    try t.expectEqualStrings("", whole.content);
    const streamed = try ask(a, "inkling2", reply, tools, true, 2);
    try t.expectEqual(@as(i64, 1), streamed.rejected);
}
