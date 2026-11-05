//! Native Kimi text reaches the real OpenAI handler; a scripted engine supplies release-generated reply fixtures.
const std = @import("std");
const fieldIs = @import("kimi_values.zig").fieldIs;
const api = @import("engine_api");
const json = @import("json");
const helper = @import("kimi_text_test.zig");
const replies = @import("kimi_reply_test.zig");
const openai = @import("openai.zig");
const http = @import("http_conn.zig");
const Server = @import("server.zig").Server;
const V = json.Value;

const Script = struct {
    ids: []const u32,
    fn engine(s: *@This()) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 32768, .thinking_budget = false, .reasoning = true };
    }
    fn submit(ptr: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s: *@This() = @ptrCast(@alignCast(ptr));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (s.ids) |token| sink.event(sink.ctx, id, &.{ .tokens = &.{token} });
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
const Receipt = struct {
    a: std.mem.Allocator,
    code: u16 = 0,
    body: V = .null,
    events: std.ArrayList(V) = .empty,
    fn out(r: *@This()) openai.Out {
        return .{ .ctx = r, .vt = &.{ .open = begin, .event = event, .reply = whole } };
    }
    fn begin(ptr: *anyopaque, _: usize) error{Closed}!void {
        const r: *@This() = @ptrCast(@alignCast(ptr));
        r.code = 200;
    }
    fn event(ptr: *anyopaque, value: ?V) error{Closed}!void {
        const r: *@This() = @ptrCast(@alignCast(ptr));
        if (value) |v| r.events.append(r.a, v) catch return error.Closed;
    }
    fn whole(ptr: *anyopaque, code: u16, value: V) void {
        const r: *@This() = @ptrCast(@alignCast(ptr));
        r.code = code;
        r.body = value;
    }
};

test "Kimi HTTP reasoning and tools agree for whole replies and one-token streaming cuts" {
    const t = try helper.synthetic(std.testing.allocator);
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fixture = (try json.parse(a, @embedFile("fixtures/kimi/golden.json"))).ok;
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]")).ok;
    for (fixture.get("cases").?.array) |c| {
        const thought = fieldIs(c, "id", "history-reasoning");
        if (!thought and !fieldIs(c, "id", "tool-roundtrip")) continue;
        var script: Script = .{ .ids = try replies.assistant(a, t.text(), c) };
        const srv = try Server.init(a, std.testing.io, script.engine(), t.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false }, null);
        defer srv.deinit();
        var conn = try http.Conn.init(a, -1, "fixture");
        defer conn.deinit();
        var body = (try json.parse(a, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4096}")).ok;
        try body.object.put(a, "tools", tools);
        var whole: Receipt = .{ .a = a };
        openai.run(srv, a, whole.out(), .{ .conn = &conn }, true, body);
        try std.testing.expectEqual(@as(u16, 200), whole.code);
        const choice = whole.body.get("choices").?.array[0];
        const message = choice.get("message").?;
        if (thought) {
            try std.testing.expectEqualStrings("Because.", message.strField("content").?);
            try std.testing.expectEqualStrings("Think carefully.", message.strField("reasoning_content").?);
            try std.testing.expect(!message.has("tool_calls"));
        } else {
            try std.testing.expect(fieldIs(choice, "finish_reason", "tool_calls"));
            try std.testing.expectEqual(@as(usize, 2), message.get("tool_calls").?.array.len);
        }
        try body.object.put(a, "stream", .{ .bool = true });
        var stream: Receipt = .{ .a = a };
        openai.run(srv, a, stream.out(), .{ .conn = &conn }, true, body);
        try std.testing.expectEqual(@as(u16, 200), stream.code);
        var content: std.ArrayList(u8) = .empty;
        var reason: std.ArrayList(u8) = .empty;
        var calls: usize = 0;
        for (stream.events.items) |event| {
            const choices = event.get("choices") orelse continue;
            if (choices.array.len == 0) continue;
            const delta = choices.array[0].get("delta") orelse continue;
            if (delta.strField("content")) |s| try content.appendSlice(a, s);
            if (delta.strField("reasoning_content")) |s| try reason.appendSlice(a, s);
            if (delta.get("tool_calls")) |list| for (list.array) |call| {
                if (call.get("function").?.has("name")) calls += 1;
            };
        }
        try std.testing.expectEqualStrings(if (thought) "Because." else "", content.items);
        try std.testing.expectEqualStrings(if (thought) "Think carefully." else "", reason.items);
        try std.testing.expectEqual(@as(usize, if (thought) 0 else 2), calls);
    }
}

test "Kimi HTTP leaves ordinary XTML spellings and generic tool markup as literal content" {
    const t = try helper.synthetic(std.testing.allocator);
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const literal = "<|close|>response<|sep|><tool_call>{\"name\":\"lookup\",\"arguments\":{}}</tool_call>";
    var script: Script = .{ .ids = try t.text().encodePlain(a, literal) };
    const srv = try Server.init(a, std.testing.io, script.engine(), t.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    const body = (try json.parse(a, "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}],\"max_tokens\":4096}")).ok;
    var conn = try http.Conn.init(a, -1, "fixture");
    defer conn.deinit();
    var whole: Receipt = .{ .a = a };
    openai.run(srv, a, whole.out(), .{ .conn = &conn }, true, body);
    try std.testing.expectEqual(@as(u16, 200), whole.code);
    const choice = whole.body.get("choices").?.array[0];
    const message = choice.get("message").?;
    try std.testing.expectEqualStrings(literal, message.strField("content").?);
    try std.testing.expect(!message.has("tool_calls"));
    try std.testing.expect(!fieldIs(choice, "finish_reason", "tool_calls"));
}

test "Kimi raw completions retain generated XTML as text instead of interpreting chat channels" {
    const t = try helper.synthetic(std.testing.allocator);
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const literal = "<|close|>think<|sep|><|open|>response<|sep|>Hello";
    var script: Script = .{ .ids = try t.text().encode(a, literal, false) };
    const srv = try Server.init(a, std.testing.io, script.engine(), t.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = true, .use_drafts = false }, null);
    defer srv.deinit();
    const body = (try json.parse(a, "{\"model\":\"m\",\"prompt\":\"Hello\",\"max_tokens\":4096}")).ok;
    var conn = try http.Conn.init(a, -1, "fixture");
    defer conn.deinit();
    var whole: Receipt = .{ .a = a };
    openai.run(srv, a, whole.out(), .{ .conn = &conn }, false, body);
    try std.testing.expectEqual(@as(u16, 200), whole.code);
    const choice = whole.body.get("choices").?.array[0];
    try std.testing.expectEqualStrings(literal, choice.strField("text").?);
}
