//! What a Responses socket sends: turn events as text frames, ping keepalives, error and steer refusal events.
const std = @import("std");
const json = @import("json");
const ws_frame = @import("ws_frame.zig");
const Sink = @import("responses_sink.zig").Sink;
const Session = @import("responses_ws.zig").Session;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// One turn's events on the socket, each tagged with the turn's ``stream_id`` when it named one.
pub const Lane = struct {
    session: *Session,
    stream_id: ?[]const u8,
    /// An event reached the socket: the turn counts as a 200, as an SSE stream that opened does, even if cut short.
    sent: bool = false,

    pub fn sink(l: *Lane) Sink {
        return .{ .ctx = l, .vt = &vt };
    }

    const vt: Sink.VTable = .{ .open = open, .event = event, .keepalive = keepalive };

    fn of(ctx: *anyopaque) *Lane {
        return @ptrCast(@alignCast(ctx));
    }

    fn open(_: *anyopaque) error{Closed}!void {}

    fn event(ctx: *anyopaque, a: Allocator, _: []const u8, payload: *json.Object) error{Closed}!void {
        const l = of(ctx);
        if (l.stream_id) |id| payload.put(a, "stream_id", .{ .string = id }) catch return error.Closed;
        try frame(l.session, .{ .object = payload });
        l.sent = true;
    }

    fn keepalive(ctx: *anyopaque) error{Closed}!void {
        const s = of(ctx).session;
        if (s.close_sent) return error.Closed;
        return ws_frame.writeFrame(s.conn, .ping, "");
    }
};

/// One event as a text frame; its JSON text lives only while it is written.
pub fn frame(s: *Session, value: Value) error{Closed}!void {
    if (s.close_sent) return error.Closed; // nothing follows a close frame
    var arena = std.heap.ArenaAllocator.init(s.gpa);
    defer arena.deinit();
    const text = json.stringify(arena.allocator(), value, .{}) catch return error.Closed;
    ws_frame.writeFrame(s.conn, .text, text) catch {
        s.gone = true;
        return error.Closed;
    };
}

/// An ``error`` event's fields: the HTTP status the same refusal gets over POST, and OpenAI's error object.
pub const Problem = struct {
    status: u16,
    kind: ?[]const u8 = null,
    code: ?[]const u8 = null,
    message: []const u8,
    param: ?[]const u8 = null,

    /// The fields of an HTTP error reply: ``{"error": {...}}`` (or a bare error object).
    pub fn fromPayload(status: u16, payload: Value) Problem {
        const e = payload.get("error") orelse payload;
        return .{
            .status = status,
            .kind = e.strField("type"),
            .code = e.strField("code"),
            .message = e.strField("message") orelse "the request failed",
            .param = e.strField("param"),
        };
    }

    /// The fields of an HTTP error body's JSON text.
    pub fn fromBody(a: Allocator, status: u16, body: []const u8) Problem {
        const parsed = json.parse(a, body) catch return .{ .status = status, .message = "the request failed" };
        return fromPayload(status, if (parsed == .ok) parsed.ok else .null);
    }
};

/// A request-scoped ``error`` event (``stream_id`` when the request named one); the socket stays open.
pub fn requestError(s: *Session, stream_id: ?[]const u8, p: Problem) void {
    var arena = std.heap.ArenaAllocator.init(s.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const value = errorEvent(a, stream_id, p) catch return;
    frame(s, value) catch {};
}

/// A connection-level ``error`` event: it names no lane.
pub fn connectionError(s: *Session, p: Problem) void {
    requestError(s, null, p);
}

fn orNull(text: ?[]const u8) Value {
    return if (text) |t| .{ .string = t } else .null;
}

fn errorEvent(a: Allocator, stream_id: ?[]const u8, p: Problem) Allocator.Error!Value {
    const e = try json.newObject(a);
    try e.put(a, "type", orNull(p.kind));
    try e.put(a, "code", orNull(p.code));
    try e.put(a, "message", .{ .string = p.message });
    try e.put(a, "param", orNull(p.param));
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "error" });
    try o.put(a, "status", try json.intValue(a, p.status));
    if (stream_id) |id| try o.put(a, "stream_id", .{ .string = id });
    try o.put(a, "error", .{ .object = e });
    return .{ .object = o };
}

/// ``response.steer`` is not supported: ``response.steer.failed`` with the steer's own fields.
pub fn steerFailed(s: *Session, a: Allocator, body: Value) void {
    const value = steerEvent(a, body) catch return;
    frame(s, value) catch {};
}

fn steerEvent(a: Allocator, body: Value) Allocator.Error!Value {
    const e = try json.newObject(a);
    try e.put(a, "type", .{ .string = "invalid_request_error" });
    try e.put(a, "code", .{ .string = "steering_not_supported" });
    try e.put(a, "message", .{ .string = "this server does not support response.steer: send a new response.create" });
    try e.put(a, "param", .null);
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "response.steer.failed" });
    if (body.strField("stream_id")) |id| try o.put(a, "stream_id", .{ .string = id });
    try o.put(a, "previous_response_id", body.get("previous_response_id") orelse .null);
    try o.put(a, "input", body.get("input") orelse .null);
    try o.put(a, "error", .{ .object = e });
    return .{ .object = o };
}

const testing = std.testing;

test "an error event carries its status, its lane when named, and OpenAI's four error fields" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = "{\"error\": {\"message\": \"m\", \"type\": \"rate_limit_error\", \"param\": null, \"code\": \"max_inflight_exceeded\"}}";
    const p = Problem.fromBody(a, 429, body);
    const text = try json.stringify(a, try errorEvent(a, "main", p), .{});
    try testing.expectEqualStrings("{\"type\": \"error\", \"status\": 429, \"stream_id\": \"main\", \"error\": {\"type\": \"rate_limit_error\", \"code\": \"max_inflight_exceeded\", \"message\": \"m\", \"param\": null}}", text);
    const bare = try json.stringify(a, try errorEvent(a, null, Problem.fromBody(a, 400, "{\"error\":{\"message\":\"x\"}}")), .{});
    try testing.expectEqualStrings("{\"type\": \"error\", \"status\": 400, \"error\": {\"type\": null, \"code\": null, \"message\": \"x\", \"param\": null}}", bare);
}
