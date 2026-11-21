//! Inkling 2 tool calls through /v1/chat/completions, whole and streamed: one keyed call, no header shown as text.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const preflight = @import("stream_preflight_test.zig");
const Conn = @import("http_conn.zig").Conn;
const call_ids = @import("ids.zig");
const markers = @import("inkling_reply.zig").markers;
const A = std.mem.Allocator;
const t = std.testing;

/// Ids 256.. in this order; below 256 an id is its byte.
const specials = [_][]const u8{ markers.model, markers.text, markers.thinking, markers.end, markers.end_sampling, markers.tool_json };

/// Bytes as tokens plus Inkling 2's markers as whole tokens, read as the .inkling format.
const Text = struct {
    fn text(x: *Text) model_text.Text {
        return .{ .ctx = x, .vtable = &.{ .format = .inkling, .encode = preflight.TestText.encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }
    fn decode(_: *anyopaque, a: A, ids: []const u32) A.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| try out.appendSlice(a, if (id < 256) &[_]u8{@intCast(id)} else specials[id - 256]);
        return out.toOwnedSlice(a);
    }
    fn tokenId(_: *anyopaque, piece: []const u8) ?u32 {
        for (specials, 0..) |s, i| if (std.mem.eql(u8, s, piece)) return @intCast(256 + i);
        return null;
    }
    fn tokenString(_: *anyopaque, a: A, id: u32) A.Error![]u8 {
        return if (id >= 256) a.dupe(u8, specials[id - 256]) else std.fmt.allocPrint(a, "{d}", .{id});
    }
    fn vocabSize(_: *anyopaque) u32 {
        return 256 + specials.len;
    }
    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn render(_: *anyopaque, a: A, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }
    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

const payload = "{\"name\":\"run_shell_command\",\"args\":{\"command\":\"ls /tmp\"}}";
const reply_parts = [_][]const u8{ markers.thinking, "Need the files.", markers.end, markers.model, "run_shell_command", markers.tool_json, payload, markers.end, markers.end_sampling };

/// A scripted reply as ids: each marker one token, every other byte its own.
fn script(a: A, parts: []const []const u8) ![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    for (parts) |p| if (Text.tokenId(undefined, p)) |id| try out.append(a, id) else for (p) |b| try out.append(a, b);
    return out.items;
}

/// Sends its tokens one per event, then stops.
const OneByOne = struct {
    tokens: []const u32,

    fn engine(e: *OneByOne) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *OneByOne = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (0..e.tokens.len) |i| sink.event(sink.ctx, id, &.{ .tokens = e.tokens[i..][0..1] });
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

const Wire = struct {
    a: A,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

/// What a chat client read: its calls (id, name, arguments), reasoning, content and last finish_reason.
const Got = struct {
    ids: std.ArrayList([]const u8) = .empty,
    names: std.ArrayList(u8) = .empty,
    arguments: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    content: std.ArrayList(u8) = .empty,
    finish: []const u8 = "",

    fn keep(g: *Got, a: A, choice: json.Value, part: []const u8) !void {
        if (choice.strField("finish_reason")) |f| g.finish = f;
        if (choice.strField("text")) |x| try g.content.appendSlice(a, x);
        const m = choice.get(part) orelse return;
        if (m.strField("reasoning_content")) |r| try g.reasoning.appendSlice(a, r);
        if (m.strField("content")) |c| try g.content.appendSlice(a, c);
        for ((m.get("tool_calls") orelse return).items() orelse return) |c| {
            if (c.strField("id")) |id| try g.ids.append(a, id);
            const f = c.get("function") orelse continue;
            if (f.strField("name")) |n| try g.names.appendSlice(a, n);
            if (f.strField("arguments")) |x| try g.arguments.appendSlice(a, x);
        }
    }
};

const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"run_shell_command\",\"parameters\":{\"type\":\"object\"}}}]";

fn request(a: A, word: []const u8, stream: bool) ![]const u8 {
    return std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"max_tokens\":400,\"messages\":[{\"role\":\"user\",\"content\":\"", word, "\"}]", tools, ",\"stream\":", if (stream) "true" else "false", "}" });
}

/// ``body`` to the chat route of a fresh server under the fixed test key, read as a whole reply or a stream.
fn post(a: A, body: []const u8, stream: bool) !Got {
    return postTo(a, body, stream, true);
}

/// ``post`` to the chat route or, with ``is_chat`` false, the raw completions route.
fn postTo(a: A, body: []const u8, stream: bool, is_chat: bool) !Got {
    return postReply(a, &reply_parts, body, stream, is_chat);
}

/// ``postTo`` with the model writing `parts` instead of the scripted call.
fn postReply(a: A, parts: []const []const u8, body: []const u8, stream: bool, is_chat: bool) !Got {
    var backend: OneByOne = .{ .tokens = try script(a, parts) };
    var text: Text = .{};
    var key: call_ids.Key = .of(call_ids.test_key_text);
    const srv = try server.Server.init(t.allocator, t.io, backend.engine(), text.text(), .{ .served_name = "test-model", .model_ids = &.{"test-model"}, .enable_thinking = false, .use_drafts = false, .tool_id_key = &key }, null);
    defer srv.deinit();
    var wire: Wire = .{ .a = a };
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var c: Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = "", .headers = &headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    openai.post(srv, &c, a, is_chat);
    const bytes = wire.bytes.items;
    try t.expect(std.mem.startsWith(u8, bytes, "HTTP/1.1 200 "));
    var got: Got = .{};
    var frames: std.ArrayList([]const u8) = .empty;
    if (stream) {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, bytes, at, "data: ")) |d| {
            const end = std.mem.indexOfPos(u8, bytes, d, "\n\n") orelse bytes.len;
            if (!std.mem.eql(u8, bytes[d + 6 .. end], "[DONE]")) try frames.append(a, bytes[d + 6 .. end]);
            at = end;
        }
    } else try frames.append(a, bytes[(std.mem.indexOf(u8, bytes, "\r\n\r\n") orelse return error.NoBody) + 4 ..]);
    for (frames.items) |f| {
        const choices = ((try json.parse(a, f)).ok.get("choices") orelse continue).items() orelse continue;
        if (choices.len > 0) try got.keep(a, choices[0], if (stream) "delta" else "message");
    }
    return got;
}

fn expectCallId(id: []const u8) !void {
    try t.expect(std.mem.startsWith(u8, id, "call_"));
    try t.expectEqual(@as(usize, 5 + 24), id.len);
    for (id[5..]) |ch| try t.expect(std.ascii.isDigit(ch) or (ch >= 'a' and ch <= 'f'));
}

test "an Inkling 2 tool call reaches the chat route as one call with its arguments and reasoning, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |stream| {
        const got = try post(a, try request(a, "x", stream), stream);
        try t.expectEqual(@as(usize, 1), got.ids.items.len);
        try t.expectEqualStrings("run_shell_command", got.names.items);
        try t.expectEqualStrings("{\"command\":\"ls /tmp\"}", got.arguments.items);
        try t.expectEqualStrings("tool_calls", got.finish);
        try t.expectEqualStrings("Need the files.", got.reasoning.items);
    }
}

test "an Inkling 2 reply never shows a tool block's header as content, whole or streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |stream| {
        const got = try post(a, try request(a, "x", stream), stream);
        try t.expect(std.mem.indexOf(u8, got.content.items, "run_shell") == null);
    }
}

test "an Inkling 2 call id is the request's own: call_ and 24 hex, whole as streamed, apart for another request" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var ids: [3][]const u8 = undefined;
    for (&ids, [_][]const u8{ "x", "x", "y" }, [_]bool{ false, true, false }) |*id, word, stream| {
        const got = try post(a, try request(a, word, stream), stream);
        try t.expectEqual(@as(usize, 1), got.ids.items.len);
        try expectCallId(got.ids.items[0]);
        id.* = got.ids.items[0];
    }
    try t.expectEqualStrings(ids[0], ids[1]);
    try t.expect(!std.mem.eql(u8, ids[0], ids[2]));
}

test "a raw Inkling 2 completion comes back unsplit, markers and all, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const want = try std.mem.concat(a, u8, &reply_parts);
    for ([_]bool{ false, true }) |stream| {
        const body = try std.mem.concat(a, u8, &.{ "{\"model\":\"test-model\",\"max_tokens\":400,\"prompt\":\"x\",\"stream\":", if (stream) "true" else "false", "}" });
        const got = try postTo(a, body, stream, false);
        try t.expectEqualStrings(want, got.content.items);
        try t.expectEqualStrings("", got.reasoning.items);
        try t.expectEqual(@as(usize, 0), got.ids.items.len);
    }
}

test "an Inkling 2 reply a stop string ended keeps no tool calls, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |stream| {
        const plain = try request(a, "x", stream);
        const body = try std.mem.concat(a, u8, &.{ plain[0 .. plain.len - 1], ",\"stop\":[\"", markers.end_sampling, "\"]}" });
        const got = try post(a, body, stream);
        try t.expectEqual(@as(usize, 0), got.ids.items.len);
        try t.expectEqualStrings("Need the files.", got.reasoning.items);
    }
}

/// A chat request offering `list` (the JSON after `"tools":`) with `extra` members after it.
fn offering(a: A, list: []const u8, extra: []const u8, stream: bool) ![]const u8 {
    const head = "{\"model\":\"test-model\",\"max_tokens\":400,\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":";
    return std.mem.concat(a, u8, &.{ head, list, extra, ",\"stream\":", if (stream) "true" else "false", "}" });
}

/// A tools list offering the one function `name`.
fn offer(comptime name: []const u8) []const u8 {
    return "[{\"type\":\"function\",\"function\":{\"name\":\"" ++ name ++ "\",\"parameters\":{\"type\":\"object\"}}}]";
}

test "an Inkling 2 call takes the offered tool's spelling, and one to a tool not offered is content, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |stream| {
        const spelled = try post(a, try offering(a, comptime offer("Run_Shell_Command"), "", stream), stream);
        try t.expectEqualStrings("Run_Shell_Command", spelled.names.items);
        const none = ",\"tool_choice\":\"none\"";
        for ([_][]const u8{ try offering(a, comptime offer("read_file"), "", stream), try offering(a, comptime offer("run_shell_command"), none, stream) }) |body| {
            const got = try post(a, body, stream);
            try t.expectEqual(@as(usize, 0), got.ids.items.len);
            try t.expectEqualStrings(payload, got.content.items);
            try t.expectEqualStrings("stop", got.finish);
        }
    }
}

test "an Inkling 2 tool block that is not JSON is content, not a call, whole and streamed" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const broken = payload[0 .. payload.len - 1];
    const parts = [_][]const u8{ markers.model, "run_shell_command", markers.tool_json, broken, markers.end, markers.end_sampling };
    for ([_]bool{ false, true }) |stream| {
        const got = try postReply(a, &parts, try request(a, "x", stream), stream, true);
        try t.expectEqual(@as(usize, 0), got.ids.items.len);
        try t.expectEqualStrings(broken, got.content.items);
        try t.expectEqualStrings("stop", got.finish);
    }
}
