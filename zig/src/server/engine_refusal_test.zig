//! An engine's own words for a request it cannot serve: one optional vtable entry, asked before anything is submitted.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const openai = @import("openai.zig");
const anthropic_translate = @import("anthropic_translate.zig");
const responses_translate = @import("responses_translate.zig");
const errors = @import("errors.zig");
const Conn = @import("http_conn.zig").Conn;
const pre = @import("stream_preflight_test.zig");
const TestText = pre.TestText;
const NoHistory = pre.NoHistory;

/// What the fake engine says when asked; `off` builds a vtable with no `refusal` entry at all.
const Mode = enum { off, always, tools_only, long_message, empty };

/// An engine that answers in its own words and counts what the server asked it.
const Refuser = struct {
    mode: Mode,
    submissions: usize = 0,
    asked: usize = 0,
    prompt_len: usize = 0, // the whole request's fields, as the engine saw them
    sampled: bool = false,
    tools_len: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = if (e.mode == .off) &plain else &asked };
    }
};

const plain: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory };
const asked: api.Engine.VTable = .{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .refusal = refusal };

fn info(_: *anyopaque) api.Info {
    return .{ .context_window = 4096 };
}

fn submit(ctx: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
    const e: *Refuser = @ptrCast(@alignCast(ctx));
    e.submissions += 1;
    sink.event(sink.ctx, id, &.{ .prefilled = 0 });
    sink.event(sink.ctx, id, &.{ .tokens = &.{ 'h', 'i' } });
    sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
}

fn cancel(_: *anyopaque, _: api.Id) void {}

fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
    out.* = .{};
}

fn memory(_: *anyopaque, _: bool) ?api.Memory {
    return null;
}

/// 86 three-byte characters (258 bytes): cut into 256, the tail lands inside the last character.
const long_pattern: [258]u8 = blk: {
    var bytes: [258]u8 = undefined;
    var i: usize = 0;
    while (i < 258) : (i += 1) bytes[i] = if (i % 3 == 0) 0xE6 else if (i % 3 == 1) 0x97 else 0xA5;
    break :blk bytes;
};

/// The engine's own words, chosen by `mode`; null: it accepts the request.
fn refusal(ctx: *anyopaque, request: *const api.Request, out: []u8) ?[]const u8 {
    const e: *Refuser = @ptrCast(@alignCast(ctx));
    e.asked += 1;
    e.prompt_len = request.prompt.len;
    e.sampled = request.sampling != null;
    e.tools_len = request.tools_json.len;
    return switch (e.mode) {
        .off => null, // never read: this engine's vtable carries no entry
        .always => put(out, "the engine cannot serve this request as asked"),
        .tools_only => if (request.tools_json.len > 0) put(out, "this engine serves tools that name one parameter") else null,
        .long_message => put(out, &long_pattern),
        .empty => out[0..0],
    };
}

/// The engine writes into the caller's buffer and keeps back the part it wrote.
fn put(out: []u8, message: []const u8) []const u8 {
    const n = @min(message.len, out.len);
    @memcpy(out[0..n], message[0..n]);
    return out[0..n];
}

/// What the route's output saw: its status, whether a stream opened, and the error's message and type.
const Seen = struct {
    opened: bool = false,
    events: usize = 0,
    status: ?u16 = null,
    preparing: i64 = 0,
    message: [api.refusal_bytes]u8 = undefined,
    message_len: usize = 0,
    kind: [32]u8 = undefined,
    kind_len: usize = 0,

    fn text(s: *const Seen) []const u8 {
        return s.message[0..s.message_len];
    }

    fn errorType(s: *const Seen) []const u8 {
        return s.kind[0..s.kind_len];
    }

    fn out(c: *Seen) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque, _: usize) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.opened = true;
    }

    fn event(ctx: *anyopaque, payload: ?json.Value) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.events += 1;
        if (payload) |p| c.keepError(p);
    }

    fn reply(ctx: *anyopaque, code: u16, payload: json.Value) void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.status = code;
        c.keepError(payload);
    }

    fn keepError(c: *Seen, payload: json.Value) void {
        const err = payload.get("error") orelse return;
        if (err.strField("message")) |m| {
            c.message_len = @min(m.len, c.message.len);
            @memcpy(c.message[0..c.message_len], m[0..c.message_len]);
        }
        if (err.strField("type")) |t| {
            c.kind_len = @min(t.len, c.kind.len);
            @memcpy(c.kind[0..c.kind_len], t[0..c.kind_len]);
        }
    }
};

const Route = enum { chat, completions, responses, messages };

/// `body` through `route`'s translation and the chat handler, as the route runs it; the preparing count after it.
fn send(backend: *Refuser, route: Route, body: []const u8) !Seen {
    var text: TestText = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = (try json.parse(a, body)).ok;
    var cx: errors.Cx = .{ .a = a };
    const chat = switch (route) {
        .chat, .completions => raw,
        .messages => try anthropic_translate.translate(&cx, raw, false),
        .responses => (try responses_translate.translate(&cx, raw, NoHistory{})).chat,
    };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{};
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, route != .completions, chat);
    seen.preparing = srv.preparing.load(.acquire);
    return seen;
}

const tool = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}]";
const no_tools = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2";
const with_tools = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2," ++ tool;

test "engine refusal: no entry keeps today's path" {
    var backend: Refuser = .{ .mode = .off };
    const seen = try send(&backend, .chat, no_tools ++ "}");
    try std.testing.expectEqual(@as(?u16, 200), seen.status);
    try std.testing.expect(!seen.opened);
    try std.testing.expectEqual(@as(usize, 1), backend.submissions);
    try std.testing.expectEqual(@as(usize, 0), backend.asked);
    try std.testing.expectEqual(@as(i64, 0), seen.preparing);
}

test "engine refusal: a whole request is asked once and never submitted" {
    var backend: Refuser = .{ .mode = .always };
    const streamed = try send(&backend, .chat, with_tools ++ ",\"stream\":true,\"temperature\":0.5}");
    try std.testing.expectEqual(@as(?u16, 400), streamed.status);
    try std.testing.expect(!streamed.opened);
    try std.testing.expectEqual(@as(usize, 0), streamed.events);
    try std.testing.expectEqualStrings("the engine cannot serve this request as asked", streamed.text());
    try std.testing.expectEqualStrings("invalid_request_error", streamed.errorType());
    try std.testing.expectEqual(@as(usize, 0), backend.submissions);
    try std.testing.expectEqual(@as(usize, 1), backend.asked);
    try std.testing.expect(backend.prompt_len > 0); // the request was whole when the engine saw it
    try std.testing.expect(backend.sampled);
    try std.testing.expect(backend.tools_len > 0);
    try std.testing.expectEqual(@as(i64, 0), streamed.preparing);
    var again: Refuser = .{ .mode = .always };
    const whole = try send(&again, .chat, with_tools ++ ",\"temperature\":0.5}");
    try std.testing.expectEqual(@as(?u16, 400), whole.status);
    try std.testing.expectEqualStrings(streamed.text(), whole.text());
    try std.testing.expectEqual(@as(usize, 0), again.submissions);
    try std.testing.expectEqual(@as(usize, 1), again.asked);
}

test "engine refusal: all four routes answer 400 before a stream" {
    const cases = [_]struct { route: Route, body: []const u8 }{
        .{ .route = .chat, .body = no_tools ++ ",\"stream\":true}" },
        .{ .route = .completions, .body = "{\"model\":\"test-model\",\"prompt\":\"hello\",\"max_tokens\":2}" },
        .{ .route = .responses, .body = "{\"model\":\"test-model\",\"input\":\"x\",\"max_output_tokens\":2,\"stream\":true}" },
        .{ .route = .messages, .body = no_tools ++ ",\"stream\":false}" },
    };
    for (cases) |c| {
        var backend: Refuser = .{ .mode = .always };
        const seen = try send(&backend, c.route, c.body);
        try std.testing.expectEqual(@as(?u16, 400), seen.status);
        try std.testing.expect(!seen.opened);
        try std.testing.expectEqual(@as(usize, 0), seen.events);
        try std.testing.expectEqualStrings("the engine cannot serve this request as asked", seen.text());
        try std.testing.expectEqual(@as(usize, 0), backend.submissions);
        try std.testing.expectEqual(@as(usize, 1), backend.asked);
        try std.testing.expectEqual(@as(i64, 0), seen.preparing);
    }
}

test "engine refusal: a message for one request and none for the next does not leak" {
    var backend: Refuser = .{ .mode = .tools_only };
    const refused = try send(&backend, .chat, with_tools ++ "}");
    try std.testing.expectEqual(@as(?u16, 400), refused.status);
    try std.testing.expectEqualStrings("this engine serves tools that name one parameter", refused.text());
    const served = try send(&backend, .chat, no_tools ++ "}");
    try std.testing.expectEqual(@as(?u16, 200), served.status);
    try std.testing.expectEqual(@as(usize, 1), backend.submissions);
    const again = try send(&backend, .chat, with_tools ++ "}");
    try std.testing.expectEqual(@as(?u16, 400), again.status);
    try std.testing.expectEqualStrings(refused.text(), again.text());
    try std.testing.expectEqual(@as(usize, 1), backend.submissions);
    try std.testing.expectEqual(@as(usize, 3), backend.asked);
}

test "engine refusal: a long message is cut at 256 bytes on a UTF-8 boundary" {
    var backend: Refuser = .{ .mode = .long_message };
    const seen = try send(&backend, .chat, no_tools ++ "}");
    try std.testing.expectEqual(@as(?u16, 400), seen.status);
    try std.testing.expect(std.unicode.utf8ValidateSlice(seen.text()));
    try std.testing.expect(seen.text().len <= api.refusal_bytes);
    try std.testing.expect(seen.text().len > 0);
    try std.testing.expectEqual(@as(usize, 255), seen.text().len); // the engine's 256 cut the last character in two
    try std.testing.expectEqual(@as(usize, 0), backend.submissions);
}

test "engine refusal: an empty message still refuses" {
    var backend: Refuser = .{ .mode = .empty };
    const seen = try send(&backend, .chat, no_tools ++ "}");
    try std.testing.expectEqual(@as(?u16, 400), seen.status);
    try std.testing.expect(!seen.opened);
    try std.testing.expectEqualStrings("this engine refused the request without a reason", seen.text());
    try std.testing.expectEqual(@as(usize, 0), backend.submissions);
}
