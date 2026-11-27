//! Inkling 2 tool-call ids follow the server's request-keyed ids over two client turns.
const std = @import("std");
const json = @import("json");
const ids = @import("ids.zig");
const openai = @import("openai.zig");
const anthropic = @import("anthropic.zig");
const HfText = @import("hf_text.zig").HfText;
const Server = @import("server.zig").Server;
const h = @import("inkling2_chat_test.zig");
const t = std.testing;

const Capture = struct {
    a: std.mem.Allocator,
    whole: ?json.Value = null,
    events: std.ArrayList(json.Value) = .empty,

    fn out(c: *Capture) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque, _: usize) error{Closed}!void {}

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.events.append(c.a, payload orelse return) catch return error.Closed;
    }

    fn reply(ctx: *anyopaque, _: u16, payload: json.Value) void {
        const c: *Capture = @ptrCast(@alignCast(ctx));
        c.whole = payload;
    }
};

const Wire = struct {
    a: std.mem.Allocator,
    bytes: std.ArrayList(u8) = .empty,

    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(ctx));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

const Seen = struct {
    ids: []const []const u8,
    tokens: []const u8,
};

fn read(a: std.mem.Allocator, value: json.Value, call_ids: *std.ArrayList([]const u8), tokens: *[]const u8) !void {
    if (value.get("tensorfold")) |tf| if (tf.get("token_ids")) |v| {
        tokens.* = try json.stringify(a, v, .{ .compact = true });
    };
    const choices = value.get("choices") orelse return;
    if (choices != .array or choices.array.len == 0) return;
    const first = choices.array[0];
    const message = first.get("message") orelse first.get("delta") orelse return;
    const calls = message.get("tool_calls") orelse return;
    if (calls != .array) return;
    for (calls.array) |call| {
        const id = call.get("id") orelse continue;
        if (id == .string) try call_ids.append(a, id.string);
    }
}

fn post(a: std.mem.Allocator, srv: *Server, body: []const u8) !Seen {
    var capture: Capture = .{ .a = a };
    var conn = try @import("http_conn.zig").Conn.init(a, -1, "fixture");
    defer conn.deinit();
    openai.run(srv, a, capture.out(), .{ .conn = &conn }, true, (try json.parseText(a, body)).ok);
    var call_ids: std.ArrayList([]const u8) = .empty;
    var tokens: []const u8 = "";
    if (capture.whole) |whole| try read(a, whole, &call_ids, &tokens);
    for (capture.events.items) |event| try read(a, event, &call_ids, &tokens);
    return .{ .ids = call_ids.items, .tokens = tokens };
}

fn request(a: std.mem.Allocator, messages: []const u8, stream: bool) ![]const u8 {
    return std.mem.concat(a, u8, &.{
        "{\"model\":\"m\",\"max_tokens\":256,\"return_token_ids\":true,\"stream\":",
        if (stream) "true" else "false",
        ",\"messages\":",
        messages,
        h.lookup,
        "}",
    });
}

fn anthropicRequest(a: std.mem.Allocator, messages: []const u8, stream: bool) ![]const u8 {
    return std.mem.concat(a, u8, &.{
        "{\"model\":\"m\",\"max_tokens\":256,\"stream\":",
        if (stream) "true" else "false",
        ",\"messages\":",
        messages,
        ",\"tools\":[{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\"}}]}",
    });
}

fn anthropicPost(a: std.mem.Allocator, srv: *Server, body: []const u8) !Seen {
    var wire: Wire = .{ .a = a };
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var conn: @import("http_conn.zig").Conn = .{
        .fd = -1,
        .peer = "",
        .buf = try a.dupe(u8, body),
        .end = body.len,
        .gpa = a,
        .version = "HTTP/1.1",
        .method = "POST",
        .path = "/v1/messages",
        .headers = &headers,
        .write_hooks = .{ .ctx = &wire, .write = Wire.write },
    };
    anthropic.post(srv, &conn, a);
    try t.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 200 "));
    var found: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, wire.bytes.items, at, "\"call_")) |q| : (at = q + 1) {
        const end = std.mem.indexOfScalarPos(u8, wire.bytes.items, q + 1, '"') orelse return error.BadCallId;
        const id = wire.bytes.items[q + 1 .. end];
        for (found.items) |seen| {
            if (std.mem.eql(u8, seen, id)) break;
        } else try found.append(a, id);
    }
    return .{ .ids = found.items, .tokens = "" };
}

fn expectDerived(a: std.mem.Allocator, body: []const u8, got: Seen) !void {
    const key: ids.Key = .of(ids.test_key_text);
    var expected: ids.Calls = .{ .key = &key, .body = (try json.parseText(a, body)).ok };
    try t.expectEqual(@as(usize, 2), got.ids.len);
    for (got.ids, 0..) |id, i| {
        try t.expectEqualStrings(try expected.id(a, 0, i), id);
        try t.expectEqual(@as(usize, 29), id.len);
        try t.expect(std.mem.startsWith(u8, id, "call_"));
    }
}

fn expectFormat(got: Seen) !void {
    try t.expectEqual(@as(usize, 2), got.ids.len);
    for (got.ids) |id| {
        try t.expectEqual(@as(usize, 29), id.len);
        try t.expect(std.mem.startsWith(u8, id, "call_"));
        for (id["call_".len..]) |c| try t.expect(std.ascii.isDigit(c) or (c >= 'a' and c <= 'f'));
    }
}

fn expectTokens(a: std.mem.Allocator, got: Seen, expected: []const u32) !void {
    const parsed = try json.parseText(a, got.tokens);
    const value = parsed.ok;
    try t.expect(value == .array);
    try t.expectEqual(expected.len, value.array.len);
    for (expected, value.array) |id, row| try t.expectEqual(@as(i64, id), row.int64().?);
}

fn expectApart(all: []const Seen) !void {
    for (all, 0..) |one, i| for (all[i + 1 ..]) |two| {
        for (one.ids) |x| for (two.ids) |y| try t.expect(!std.mem.eql(u8, x, y));
    };
}

test "Inkling 2 reply: OpenAI turns use server-keyed unique ids whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"q\":\"x\"}}<|end_message|>" ++
        "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"q\":\"y\"}}<|end_message|>";
    const tokens = try std.mem.concat(a, u32, &.{ try h.encodeReply(a, raw), &.{2} });
    var fixture = try h.Fixture.init(a, "inkling2");
    defer fixture.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    var script: h.Script = .{ .a = a, .reply = tokens, .stride = 1 };
    var key: ids.Key = .of(ids.test_key_text);
    const srv = try Server.init(a, t.io, script.engine(), hf.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
        .tool_id_key = &key,
    }, null);
    defer srv.deinit();

    const first_messages = "[{\"role\":\"user\",\"content\":\"first\"}]";
    const first_body = try request(a, first_messages, false);
    const first_stream_body = try request(a, first_messages, true);
    const first = try post(a, srv, first_body);
    const first_stream = try post(a, srv, first_stream_body);

    const history_template =
        \\[{"role":"user","content":"first"},{"role":"assistant","content":"","tool_calls":[
        \\{"id":"ID0","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}},
        \\{"id":"ID1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"y\"}"}}]},
        \\{"role":"tool","tool_call_id":"ID0","content":"x result"},
        \\{"role":"tool","tool_call_id":"ID1","content":"y result"},
        \\{"role":"user","content":"second"}]
    ;
    const with_first = try std.mem.replaceOwned(u8, a, history_template, "ID0", first.ids[0]);
    const second_messages = try std.mem.replaceOwned(u8, a, with_first, "ID1", first.ids[1]);
    const second_body = try request(a, second_messages, false);
    const second_stream_body = try request(a, second_messages, true);
    const second = try post(a, srv, second_body);
    const second_stream = try post(a, srv, second_stream_body);
    try expectApart(&.{ first, second });
    try expectDerived(a, first_body, first);
    try expectDerived(a, first_stream_body, first_stream);
    for (first.ids, first_stream.ids) |x, y| try t.expectEqualStrings(x, y);
    try expectTokens(a, first, tokens);
    try expectTokens(a, first_stream, tokens);
    try expectDerived(a, second_body, second);
    try expectDerived(a, second_stream_body, second_stream);
    for (second.ids, second_stream.ids) |x, y| try t.expectEqualStrings(x, y);
    try expectTokens(a, second, tokens);
    try expectTokens(a, second_stream, tokens);
    try expectApart(&.{ first, second });
}

test "Inkling 2 reply: Anthropic turns use server-keyed unique tool_use ids whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"q\":\"x\"}}<|end_message|>" ++
        "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"q\":\"y\"}}<|end_message|>";
    const tokens = try std.mem.concat(a, u32, &.{ try h.encodeReply(a, raw), &.{2} });
    var fixture = try h.Fixture.init(a, "inkling2");
    defer fixture.deinit();
    var problem: []const u8 = "";
    const hf = try HfText.load(a, t.io, fixture.path, null, a, &problem);
    defer hf.deinit();
    var script: h.Script = .{ .a = a, .reply = tokens, .stride = 1 };
    var key: ids.Key = .of(ids.test_key_text);
    const srv = try Server.init(a, t.io, script.engine(), hf.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
        .tool_id_key = &key,
    }, null);
    const first_messages = "[{\"role\":\"user\",\"content\":\"first\"}]";

    const anthropic_first_body = try anthropicRequest(a, first_messages, false);
    const anthropic_first_stream_body = try anthropicRequest(a, first_messages, true);
    const anthropic_first = try anthropicPost(a, srv, anthropic_first_body);
    const anthropic_first_stream = try anthropicPost(a, srv, anthropic_first_stream_body);
    for (anthropic_first.ids, anthropic_first_stream.ids) |x, y| try t.expectEqualStrings(x, y);
    const anthropic_history_template =
        \\[{"role":"user","content":"first"},{"role":"assistant","content":[
        \\{"type":"tool_use","id":"ID0","name":"lookup","input":{"q":"x"}},
        \\{"type":"tool_use","id":"ID1","name":"lookup","input":{"q":"y"}}]},
        \\{"role":"user","content":[
        \\{"type":"tool_result","tool_use_id":"ID0","content":"x result"},
        \\{"type":"tool_result","tool_use_id":"ID1","content":"y result"}]},
        \\{"role":"user","content":"second"}]
    ;
    const anthropic_with_first = try std.mem.replaceOwned(u8, a, anthropic_history_template, "ID0", anthropic_first.ids[0]);
    const anthropic_second_messages = try std.mem.replaceOwned(u8, a, anthropic_with_first, "ID1", anthropic_first.ids[1]);
    const anthropic_second_body = try anthropicRequest(a, anthropic_second_messages, false);
    const anthropic_second_stream_body = try anthropicRequest(a, anthropic_second_messages, true);
    const anthropic_second = try anthropicPost(a, srv, anthropic_second_body);
    const anthropic_second_stream = try anthropicPost(a, srv, anthropic_second_stream_body);
    try expectApart(&.{ anthropic_first, anthropic_second });
    try expectFormat(anthropic_first);
    try expectFormat(anthropic_first_stream);
    for (anthropic_first.ids, anthropic_first_stream.ids) |x, y| try t.expectEqualStrings(x, y);
    try expectFormat(anthropic_second);
    try expectFormat(anthropic_second_stream);
    for (anthropic_second.ids, anthropic_second_stream.ids) |x, y| try t.expectEqualStrings(x, y);
}
