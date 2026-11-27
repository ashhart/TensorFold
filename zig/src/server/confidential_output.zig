//! --confidential's output rule: an error the client or the log sees has a fixed message, allowed code, request id.
const std = @import("std");
const json = @import("json");
const wait = @import("wait_reason.zig");
const Value = json.Value;
const Conn = @import("http_conn.zig").Conn;
const log = @import("log.zig");
const Allocator = std.mem.Allocator;

var on: std.atomic.Value(bool) = .init(false);

/// From here on errors and log lines follow this file's rule; `confidential.enforce` calls it once a start passes.
pub fn confine() void {
    on.store(true, .release);
}

/// Tests only: the rule off again, so one test's confinement doesn't reach the next test.
pub fn release() void {
    on.store(false, .release);
}

pub fn active() bool {
    return on.load(.acquire);
}

var serials: std.atomic.Value(u64) = .init(0);

/// Numbers each request in the process: a request is named by it until it has a reply id.
pub fn nextSerial() u64 {
    return serials.fetchAdd(1, .monotonic) + 1;
}

/// The request id an error names: the reply id once there is one, else the request's number.
pub fn requestId(c: *const Conn, buf: []u8) []const u8 {
    return c.id orelse std.fmt.bufPrint(buf, "req-{d}", .{c.serial}) catch "req";
}

/// The error types an error object keeps: every other `type` value is written as null.
const types = [_][]const u8{
    "invalid_request_error", "server_error",     "error",            "authentication_error", "permission_error",
    "not_found_error",       "rate_limit_error", "overloaded_error", "api_error",            "request_too_large",
};

/// The error codes an error object keeps: fixed words clients key on. Any other `code` value is written as null.
const codes = [_][]const u8{
    "context_length_exceeded",            "invalid_prompt",                 "server_error",          "timeout",
    "model_unloaded",                     "model_not_found",                "request_not_found",     "steering_not_supported",
    "loopback_only",                      "invalid_api_key",                "invalid_event",         "invalid_stream_id",
    "previous_response_not_found",        "request_quarantined",            wait.restarting.code,    wait.loading.code,
    wait.reload_failed.code,              wait.engine_busy,                 wait.engine_unavailable, wait.memory_pressure,
    "websocket_connection_limit_reached", "websocket_stream_limit_reached", "max_inflight_exceeded",
};

fn listed(list: []const []const u8, text: ?[]const u8) ?[]const u8 {
    const s = text orelse return null;
    for (list) |item| if (std.mem.eql(u8, s, item)) return item;
    return null;
}

pub fn allowedType(text: ?[]const u8) ?[]const u8 {
    return listed(&types, text);
}

pub fn allowedCode(text: ?[]const u8) ?[]const u8 {
    return listed(&codes, text);
}

pub const refused_message = "the request was refused; --confidential gives no detail";
pub const failed_message = "the server could not complete the request; --confidential gives no detail";
pub const context_message = @import("errors.zig").context_limit ++ " too short for this request";

/// The message an error gets: a wait's own words, the context-limit wording clients match, or one per status class.
pub fn message(status: u16, code: ?[]const u8) []const u8 {
    if (code) |c| {
        if (std.mem.eql(u8, c, "context_length_exceeded")) return context_message;
        for ([_]wait.Wait{ wait.restarting, wait.loading, wait.reload_failed }) |w| if (std.mem.eql(u8, c, w.code)) return w.message;
    }
    return if (status >= 500) failed_message else refused_message;
}

/// An error object in OpenAI's shape built only from fixed parts: message, allowed type and code, the request id.
pub fn errorObject(a: Allocator, status: u16, kind: ?[]const u8, code: ?[]const u8, request_id: []const u8) Allocator.Error!Value {
    const c = allowedCode(code);
    const o = try json.newObject(a);
    try o.put(a, "message", .{ .string = message(status, c) });
    try o.put(a, "type", if (allowedType(kind)) |k| .{ .string = k } else .null);
    try o.put(a, "code", if (c) |x| .{ .string = x } else .null);
    try o.put(a, "param", .null);
    try o.put(a, "request_id", .{ .string = request_id });
    return .{ .object = o };
}

/// An HTTP error body (status 400 and up) in place of whatever the route wrote: `{"error": {fixed parts}}`.
pub fn httpBody(a: Allocator, status: u16, body: []const u8, request_id: []const u8) Allocator.Error![]const u8 {
    var kind: ?[]const u8 = null;
    var code: ?[]const u8 = null;
    const parsed = try json.parse(a, body);
    if (parsed == .ok) {
        const e = parsed.ok.get("error") orelse parsed.ok;
        kind = fieldOf(e, "type");
        code = fieldOf(e, "code");
    }
    const o = try json.newObject(a);
    try o.put(a, "error", try errorObject(a, status, kind, code, request_id));
    return json.stringify(a, .{ .object = o }, .{}) catch return error.OutOfMemory;
}

/// A stream or socket event with its error object replaced by fixed parts; any other event as given.
pub fn event(a: Allocator, value: Value, request_id: []const u8) Allocator.Error!Value {
    if (!active() or value != .object) return value;
    const top = value.object;
    const status: u16 = @intCast(std.math.clamp(if (top.get("status")) |s| s.int64() orelse 500 else 500, 400, 599));
    if (top.get("error")) |e| return replaced(a, value, "error", try errorObject(a, status, fieldOf(e, "type"), fieldOf(e, "code"), request_id));
    if (top.get("response")) |r| if (r == .object) if (r.object.get("error")) |e| if (e != .null) {
        const fixed = try errorObject(a, status, fieldOf(e, "type"), fieldOf(e, "code"), request_id);
        return replaced(a, value, "response", try replaced(a, r, "error", fixed));
    };
    // a bare `{"type": "error", "message": ...}` event, as Responses streams write one
    const kind = value.strField("type") orelse return value;
    if (!std.mem.eql(u8, kind, "error") or top.get("message") == null) return value;
    const fixed = try errorObject(a, status, null, value.strField("code"), request_id);
    var copy = try replaced(a, value, "message", .{ .string = message(status, allowedCode(value.strField("code"))) });
    try copy.object.put(a, "code", fixed.object.get("code").?);
    if (copy.object.get("param") != null) try copy.object.put(a, "param", .null);
    try copy.object.put(a, "request_id", .{ .string = request_id });
    return copy;
}

fn fieldOf(e: Value, key: []const u8) ?[]const u8 {
    return if (e == .object) e.strField(key) else null;
}

/// A shallow copy of an object with one field set, so a value the caller still holds is left as it was.
fn replaced(a: Allocator, value: Value, key: []const u8, field: Value) Allocator.Error!Value {
    const o = try json.copyObject(a, value.object);
    try o.put(a, key, field);
    return .{ .object = o };
}

/// Under --confidential: the access line, and for an error status a fixed JSON body in place of the route's.
pub fn head(c: *Conn, body: []const u8) Allocator.Error!?[]const u8 {
    var id_buf: [32]u8 = undefined;
    const id = requestId(c, &id_buf);
    if (c.status < 400 or std.mem.eql(u8, c.version, "HTTP/0.9")) {
        if (c.access_due) log.confidentialAccess(id, c.status, null, null);
        c.access_due = false;
        return null;
    }
    var arena = std.heap.ArenaAllocator.init(c.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try httpBody(a, c.status, body, id);
    if (c.access_due) {
        const e = (json.parse(a, text) catch unreachable).ok.get("error").?;
        log.confidentialAccess(id, c.status, e.strField("type"), e.strField("code"));
    }
    c.access_due = false;
    // the route's Content-Type and Content-Length named its own body: this head names the fixed one
    var lines_out: std.ArrayList(u8) = .empty;
    defer lines_out.deinit(c.gpa);
    var lines = std.mem.splitSequence(u8, c.out.items, "\r\n");
    while (lines.next()) |l| {
        if (l.len == 0) continue;
        if (std.ascii.startsWithIgnoreCase(l, "content-type:") or std.ascii.startsWithIgnoreCase(l, "content-length:")) continue;
        try lines_out.appendSlice(c.gpa, l);
        try lines_out.appendSlice(c.gpa, "\r\n");
    }
    const sent = if (std.mem.eql(u8, c.method, "HEAD")) "" else text;
    try lines_out.print(c.gpa, "Content-Type: application/json\r\nContent-Length: {d}\r\n", .{sent.len});
    c.out.clearRetainingCapacity();
    try c.out.appendSlice(c.gpa, lines_out.items);
    return try c.gpa.dupe(u8, sent);
}
