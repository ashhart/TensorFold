//! A reply with a complete call keeps its deadline, cancel or loop end on chat, Messages and Responses alike.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const http = @import("http_conn.zig");
const Server = @import("server.zig").Server;
const calls = @import("openai_calls.zig");
const Allocator = std.mem.Allocator;
const t = std.testing;

const eos_id: u32 = 0x7f;
const call = "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": \"a\"}}\n</tool_call>";
const tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"q\":{\"type\":\"string\"}}}}}]";
const tools_messages = "\"tools\":[{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"q\":{\"type\":\"string\"}}}}]";
const tools_responses = "\"tools\":[{\"type\":\"function\",\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"," ++
    "\"properties\":{\"q\":{\"type\":\"string\"}}}}]";

/// One byte a token; 0x7f is EOS.
const Bytes = struct {
    fn text(b: *Bytes) model_text.Text {
        return .{ .ctx = b, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocab, .eos_ids = eos, .render = render, .template_source = template } };
    }
    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }
    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id < 256) try out.append(a, @intCast(id));
        return out.toOwnedSlice(a);
    }
    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }
    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocab(_: *anyopaque) u32 {
        return 256;
    }
    fn eos(_: *anyopaque) []const u32 {
        return &.{eos_id};
    }
    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn template(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Gives `reply` and then ends as `end` says; `hang`: waits for the server's cancel (a deadline) and ends cancelled.
const Called = struct {
    reply: []const u8 = call,
    end: enum { stop, cancelled, hang } = .stop,
    sink: ?api.Sink = null,
    id: api.Id = 0,
    done: bool = false,
    fn engine(e: *Called) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 8192 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Called = @ptrCast(@alignCast(ctx));
        e.* = .{ .reply = e.reply, .end = e.end, .sink = sink, .id = id };
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.reply) |byte| sink.event(sink.ctx, id, &.{ .tokens = &.{byte} });
        switch (e.end) {
            .stop => e.finish(.stop),
            .cancelled => e.finish(.cancelled),
            .hang => {},
        }
    }
    fn finish(e: *Called, reason: api.Reason) void {
        if (e.done) return;
        e.done = true;
        e.sink.?.event(e.sink.?.ctx, e.id, &.{ .finished = .{ .reason = reason } });
    }
    fn cancel(ctx: *anyopaque, _: api.Id) void {
        const e: *Called = @ptrCast(@alignCast(ctx));
        if (e.sink != null) e.finish(.cancelled);
    }
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The bytes a route writes, in place of a socket.
const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

/// `body` posted to `path` through the dispatcher, the whole HTTP answer (a JSON body or an event stream) as text.
fn post(a: Allocator, e: *Called, path: []const u8, body: []const u8) ![]const u8 {
    var text: Bytes = .{};
    const srv = try Server.init(t.allocator, t.io, e.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var wire: Wire = .{ .a = a };
    const headers = try a.dupe(http.Header, &.{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }});
    var c: http.Conn = .{ .fd = -1, .peer = "127.0.0.1", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = path, .headers = headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    routes.dispatch(srv, &c, a);
    return wire.bytes.items;
}

fn has(out: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, out, needle) != null;
}

/// The chat reply, whole and streamed, ends `finish` (as JSON text, spaced or not) with its one call.
fn expectChat(a: Allocator, e: *Called, extra: []const u8, finish: []const u8) !void {
    for ([_][]const u8{ "false", "true" }) |stream| {
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"max_tokens\":1000,{s},\"stream\":{s}{s}}}", .{ tools, stream, extra });
        const out = try post(a, e, "/v1/chat/completions", body);
        const spaced = try std.fmt.allocPrint(a, "\"finish_reason\": \"{s}\"", .{finish});
        const tight = try std.fmt.allocPrint(a, "\"finish_reason\":\"{s}\"", .{finish});
        try t.expect(has(out, spaced) or has(out, tight));
        try t.expect(has(out, "lookup"));
    }
}

test "tool finish: the finish helper maps only a natural stop to tool_calls" {
    try t.expectEqualStrings("tool_calls", calls.finished("stop"));
    for ([_][]const u8{ "length", "deadline", "cancelled", @import("loop_stop.zig").finish_reason }) |kept| try t.expectEqualStrings(kept, calls.finished(kept));
}

test "tool finish: a complete call when deadline_s ends the reply keeps finish deadline, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var e: Called = .{ .end = .hang };
    try expectChat(arena.allocator(), &e, ",\"deadline_s\":0.05", "deadline");
}

test "tool finish: a complete call in a cancelled reply stays cancelled on chat, Messages and Responses" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Called = .{ .end = .cancelled };
    try expectChat(a, &e, "", "cancelled");
    for ([_][]const u8{ "false", "true" }) |stream| {
        const m = try post(a, &e, "/v1/messages", try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"max_tokens\":1000,\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],{s},\"stream\":{s}}}", .{ tools_messages, stream }));
        try t.expect(has(m, "lookup"));
        try t.expect(!has(m, "tool_use\"}") and !has(m, "\"stop_reason\":\"tool_use\"") and !has(m, "\"stop_reason\": \"tool_use\""));
        const r = try post(a, &e, "/v1/responses", try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"input\":\"x\",\"max_output_tokens\":1000,\"store\":false,{s},\"stream\":{s}}}", .{ tools_responses, stream }));
        try t.expect(has(r, "lookup"));
        try t.expect(has(r, "\"status\":\"cancelled\"") or has(r, "\"status\": \"cancelled\""));
        try t.expect(!has(r, "response.completed"));
    }
}

test "tool finish: an EOS landing exactly at max_tokens is a natural end: tool_calls, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Called = .{ .reply = call ++ [_]u8{eos_id}, .end = .stop };
    for ([_][]const u8{ "false", "true" }) |stream| {
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"max_tokens\":{d},{s},\"stream\":{s}}}", .{ call.len + 1, tools, stream });
        const out = try post(a, &e, "/v1/chat/completions", body);
        try t.expect(has(out, "\"finish_reason\": \"tool_calls\"") or has(out, "\"finish_reason\":\"tool_calls\""));
    }
}

test "tool finish: capped replies identify complete calls, not merely valid JSON arguments" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const open = "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"q\": \"a\"}}";
    const cases = [_]struct { reply: []const u8, complete: []const u8 }{
        .{ .reply = call, .complete = "[0]" },
        .{ .reply = open, .complete = "[]" },
        .{ .reply = call ++ "\n" ++ open, .complete = "[0]" },
    };
    for (cases) |case| for ([_][]const u8{ "false", "true" }) |stream| {
        var e: Called = .{ .reply = case.reply };
        const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"messages\":[{{\"role\":\"user\",\"content\":\"x\"}}],\"max_tokens\":{d},{s},\"stream\":{s}}}", .{ case.reply.len, tools, stream });
        const out = try post(a, &e, "/v1/chat/completions", body);
        const tight = try std.fmt.allocPrint(a, "\"complete_tool_calls\":{s}", .{case.complete});
        const spaced = try std.fmt.allocPrint(a, "\"complete_tool_calls\": {s}", .{case.complete});
        try t.expect(has(out, tight) or has(out, spaced));
    };
}
