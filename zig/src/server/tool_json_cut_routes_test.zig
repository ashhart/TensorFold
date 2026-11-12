//! A JSON call max_tokens cuts on /v1/messages and /v1/responses, whole and streamed (the chat run's cut call).
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const Conn = @import("http_conn.zig").Conn;
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Bytes as tokens, no end token; every prompt renders as "prompt".
const ByteText = struct {
    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{
            .encode = encode,
            .decode = decode,
            .token_id = tokenId,
            .token_string = tokenString,
            .vocab_size = vocabSize,
            .eos_ids = eosIds,
            .render = render,
            .template_source = templateSource,
        } };
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const out = try a.alloc(u32, input.len);
        for (input, out) |byte, *id| id.* = byte;
        return out;
    }

    fn decode(_: *anyopaque, a: Allocator, tokens: []const u32) Allocator.Error![]u8 {
        const out = try a.alloc(u8, tokens.len);
        for (tokens, out) |id, *byte| byte.* = @intCast(id);
        return out;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return decode(undefined, a, &.{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    const Rendered = model_text.Error!model_text.Rendered;

    fn render(_: *anyopaque, a: Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) Rendered {
        return .{ .text = try a.dupe(u8, "prompt") };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Writes `reply` a byte a token, then stops at max_tokens.
const Script = struct {
    reply: []const u8,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{
            .info = info,
            .submit = submit,
            .cancel = cancel,
            .status = status,
            .memory = memory,
        } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }

    fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.reply) |byte| sink.event(sink.ctx, id, &.{ .tokens = &.{byte} });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .length } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

const schema = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}," ++
    "\"content\":{\"type\":\"string\"}}}";
const named = "Writing it.\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": ";
const cut_args = "{\"path\": \"notes.txt\", \"content\": \"The history of";
const whole_args = "{\"path\": \"notes.txt\"}";
const compact = "{\"path\":\"notes.txt\"}";
const cut_closed = "{\"path\":\"notes.txt\",\"content\":\"The history of\"}";
const xml_cut = "Writing it.\n<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.txt\n</parameter>\n" ++
    "<parameter=content>\nThe history of";

const messages_tool = "{\"name\":\"write_file\",\"input_schema\":" ++ schema ++ "}";
const responses_tool = "{\"type\":\"function\",\"name\":\"write_file\",\"parameters\":" ++ schema ++ "}";

/// What a route's reply says: its status, text, the call's arguments, and why it ended.
const Seen = struct { status: u16, text: []const u8 = "", args: []const u8 = "", finish: []const u8 = "" };

/// The whole reply a route wrote, held in memory (a socket pair would cap a stream at the OS buffer, 8 KiB on macOS).
const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

/// `body` posted to `path` (/v1/messages or /v1/responses), its status and payload as written.
fn capture(a: Allocator, srv: *server.Server, path: []const u8, body: []const u8) !struct { status: u16, body: []const u8 } {
    var wire: Wire = .{ .a = a };
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var c: Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = path, .headers = &headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    if (std.mem.eql(u8, path, "/v1/messages")) @import("anthropic.zig").post(srv, &c, a) else @import("responses.zig").post(srv, &c, a);
    const raw = wire.bytes.items;
    const split = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadReply;
    const status = std.fmt.parseInt(u16, raw["HTTP/1.1 ".len..][0..3], 10) catch return error.BadReply;
    return .{ .status = status, .body = raw[split + 4 ..] };
}

/// `reply` cut at its end on `path`, whole or streamed, read back as `Seen`.
fn send(a: Allocator, path: []const u8, reply: []const u8, stream: bool) !Seen {
    var text: ByteText = .{};
    var backend: Script = .{ .reply = reply };
    const srv = try server.Server.init(testing.allocator, testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    const messages = std.mem.eql(u8, path, "/v1/messages");
    const body = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"{s}\":{d},\"stream\":{s},{s},\"tools\":[{s}]}}", .{
        if (messages) "max_tokens" else "max_output_tokens",
        reply.len,
        if (stream) "true" else "false",
        if (messages) "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]" else "\"input\":\"x\"",
        if (messages) messages_tool else responses_tool,
    });
    const got = try capture(a, srv, path, body);
    const raw = got.body;
    var seen: Seen = .{ .status = got.status };
    if (got.status != 200) return seen;
    if (!stream) {
        const doc = (try json.parseText(a, raw)).ok;
        if (messages) {
            for (doc.get("content").?.array) |b| {
                if (b.typeIs("text")) seen.text = b.strField("text").?;
                const compacted: json.Options = .{ .ascii = false, .compact = true };
                if (b.typeIs("tool_use")) seen.args = try json.stringify(a, b.get("input").?, compacted);
            }
            seen.finish = doc.strField("stop_reason").?;
            return seen;
        }
        for (doc.get("output").?.array) |item| {
            if (item.typeIs("message")) seen.text = item.get("content").?.array[0].strField("text").?;
            if (item.typeIs("function_call")) seen.args = item.strField("arguments").?;
        }
        seen.finish = doc.get("incomplete_details").?.strField("reason").?;
        return seen;
    }
    var text_out: std.ArrayList(u8) = .empty;
    var args_out: std.ArrayList(u8) = .empty;
    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const ev = switch (try json.parseText(a, std.mem.trim(u8, line[6..], "\r"))) {
            .ok => |v| v,
            .err => continue,
        };
        if (messages) {
            if (ev.get("delta")) |d| {
                if (d.typeIs("text_delta")) try text_out.appendSlice(a, d.strField("text").?);
                if (d.typeIs("input_json_delta")) try args_out.appendSlice(a, d.strField("partial_json").?);
                if (d.strField("stop_reason")) |r| seen.finish = r;
            }
            continue;
        }
        if (ev.typeIs("response.output_text.delta")) try text_out.appendSlice(a, ev.strField("delta").?);
        if (ev.typeIs("response.function_call_arguments.delta")) try args_out.appendSlice(a, ev.strField("delta").?);
        // the stream only grows: its pieces join to the arguments it ends with
        if (ev.typeIs("response.function_call_arguments.done"))
            try testing.expectEqualStrings(args_out.items, ev.strField("arguments").?);
        const details = if (ev.typeIs("response.incomplete")) ev.get("response").?.get("incomplete_details") else null;
        if (details) |d| seen.finish = d.strField("reason").?;
    }
    seen.text = text_out.items;
    seen.args = args_out.items;
    return seen;
}

fn expectSeen(want: Seen, got: Seen) !void {
    try testing.expectEqual(want.status, got.status);
    try testing.expectEqualStrings(want.text, got.text);
    try testing.expectEqualStrings(want.args, got.args);
    try testing.expectEqualStrings(want.finish, got.finish);
}

test "a cut JSON call on /v1/responses is the cut function_call, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ false, true }) |stream| {
        const cut: Seen = .{ .status = 200, .text = "Writing it.", .args = cut_args, .finish = "max_output_tokens" };
        try expectSeen(cut, try send(a, "/v1/responses", named ++ cut_args, stream));
        const whole: Seen = .{ .status = 200, .text = "Writing it.", .args = compact, .finish = "max_output_tokens" };
        try expectSeen(whole, try send(a, "/v1/responses", named ++ whole_args, stream));
    }
}

test "a cut JSON or XML call on /v1/messages is a tool_use with its input as far as written, whole and streamed" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cut: Seen = .{ .status = 200, .text = "Writing it.", .args = cut_args, .finish = "max_tokens" };
    try expectSeen(cut, try send(a, "/v1/messages", named ++ cut_args, true));
    const whole: Seen = .{ .status = 200, .text = "Writing it.", .args = compact, .finish = "max_tokens" };
    for ([_]bool{ false, true }) |stream|
        try expectSeen(whole, try send(a, "/v1/messages", named ++ whole_args, stream));
    // a whole Messages reply needs an input object: the cut arguments closed as far as written, the XML cut's alike
    const closed: Seen = .{ .status = 200, .text = "Writing it.", .args = cut_closed, .finish = "max_tokens" };
    try expectSeen(closed, try send(a, "/v1/messages", named ++ cut_args, false));
    try expectSeen(closed, try send(a, "/v1/messages", xml_cut, false));
}
