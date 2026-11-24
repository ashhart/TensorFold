//! The router's own error replies, in the shape of the API the request came in on.
const std = @import("std");
const json = @import("json");
const log = @import("../log.zig");
const Conn = @import("../http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// The API a route belongs to, which sets its error shape.
pub const Shape = enum { openai, anthropic, ollama };

/// One refusal: its status, OpenAI's error type and code, the parameter it names and the words a client reads.
pub const Refusal = struct {
    status: u16,
    message: []const u8,
    kind: []const u8 = "invalid_request_error",
    code: ?[]const u8 = null,
    param: ?[]const u8 = null,
    retry_after_s: ?u32 = null,
};

/// The request's model is none the router serves.
pub fn unknownModel(a: Allocator, name: []const u8) Refusal {
    const message = std.fmt.allocPrint(a, "model '{s}' is not served here; GET /v1/models lists the models this router serves", .{name}) catch "model not found";
    return .{ .status = 404, .message = message, .code = "model_not_found", .param = "model" };
}

/// A route or method the router refuses by name in this version.
pub fn unsupported(a: Allocator, what: []const u8, why: []const u8) Refusal {
    const message = std.fmt.allocPrint(a, "{s} is not served through the router in this version: {s}", .{ what, why }) catch "not served through the router";
    return .{ .status = 404, .message = message, .code = "unsupported_route" };
}

/// Sends ``r`` in ``shape``: OpenAI's error object, Anthropic's typed error, or Ollama's ``{"error": "..."}``.
pub fn send(conn: *Conn, a: Allocator, shape: Shape, r: Refusal) void {
    const body = encode(a, shape, r) catch return conn.sendJson(500, "{\"error\": {\"message\": \"out of memory\"}}");
    var retry: [16]u8 = undefined;
    var header: [1][2][]const u8 = undefined;
    var extra: []const [2][]const u8 = &.{};
    if (r.retry_after_s) |s| {
        header[0] = .{ "Retry-After", std.fmt.bufPrint(&retry, "{d}", .{s}) catch "30" };
        extra = &header;
    }
    conn.sendJsonWith(r.status, body, extra);
}

pub fn encode(a: Allocator, shape: Shape, r: Refusal) ![]const u8 {
    const o = try json.newObject(a);
    switch (shape) {
        .openai => {
            const e = try json.newObject(a);
            try e.put(a, "message", .{ .string = r.message });
            try e.put(a, "type", .{ .string = r.kind });
            try e.put(a, "param", if (r.param) |p| .{ .string = p } else .null);
            try e.put(a, "code", if (r.code) |c| .{ .string = c } else .null);
            try o.put(a, "error", .{ .object = e });
        },
        .anthropic => {
            const e = try json.newObject(a);
            try e.put(a, "type", .{ .string = anthropicType(r.status) });
            try e.put(a, "message", .{ .string = r.message });
            try o.put(a, "type", .{ .string = "error" });
            try o.put(a, "error", .{ .object = e });
        },
        .ollama => try o.put(a, "error", .{ .string = r.message }),
    }
    return json.stringify(a, .{ .object = o }, .{});
}

/// Anthropic's error type for a status, as the server's Messages routes name them.
fn anthropicType(status: u16) []const u8 {
    return switch (status) {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        else => "api_error",
    };
}

/// The access line for a relayed reply: the server's line, then the model that answered.
pub fn relayed(conn: *const Conn, status: u16, id: []const u8) void {
    log.line("{s} \"{s}\" {d} -{s} model={s}", .{ conn.peer, conn.requestline, status, log.keySuffix(conn), id });
}

test "serve --router: a refusal takes the shape of its route's API" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = unknownModel(a, "x\"y");
    try std.testing.expectEqualStrings("{\"error\": {\"message\": \"model 'x\\\"y' is not served here; GET /v1/models lists the models this router serves\", \"type\": \"invalid_request_error\", \"param\": \"model\", \"code\": \"model_not_found\"}}", try encode(a, .openai, r));
    try std.testing.expectEqualStrings("{\"type\": \"error\", \"error\": {\"type\": \"not_found_error\", \"message\": \"model 'x\\\"y' is not served here; GET /v1/models lists the models this router serves\"}}", try encode(a, .anthropic, r));
    try std.testing.expectEqualStrings("{\"error\": \"model 'x\\\"y' is not served here; GET /v1/models lists the models this router serves\"}", try encode(a, .ollama, r));
}
