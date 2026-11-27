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

/// The access line a response still owes, for a route that writes its head itself (whole audio).
pub fn flushAccess(c: *Conn) void {
    if (!c.access_due) return;
    c.access_due = false;
    var buf: [32]u8 = undefined;
    log.confidentialAccess(requestId(c, &buf), c.status, null, null);
}

/// The connection's own request number: a persistent socket's error names this, as a turn's reply id dies with it.
pub fn serialId(c: *const Conn, buf: []u8) []const u8 {
    return std.fmt.bufPrint(buf, "req-{d}", .{c.serial}) catch "req";
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

/// A stream or socket error event rebuilt from fixed parts only; any other event as given.
pub fn event(a: Allocator, value: Value, request_id: []const u8) Allocator.Error!Value {
    if (!active() or value != .object) return value;
    const top = value.object;
    const nested = top.get("error");
    const failed: ?Value = if (value.get("response")) |r| (if (r.get("error")) |e| (if (e != .null) e else null) else null) else null;
    const bare = value.typeIs("error") and top.get("message") != null;
    if (nested == null and failed == null and !bare) return value;
    const e = nested orelse failed orelse value;
    const kind = if (bare and nested == null) null else fieldOf(e, "type");
    const code = fieldOf(e, "code");
    const fixed = try errorObject(a, statusOf(top.get("status"), kind, code), kind, code, request_id);
    // nothing else of the event is kept: its other fields may echo the request
    const o = try json.newObject(a);
    if (value.strField("type")) |t| if (listed(&.{ "error", "response.failed" }, t)) |k| try o.put(a, "type", .{ .string = k });
    for ([_][]const u8{ "sequence_number", "status" }) |k| if (top.get(k)) |n| if (n == .int) try o.put(a, k, n);
    if (failed != null) {
        const r = try json.newObject(a);
        if (value.get("response").?.strField("id")) |id| if (std.mem.startsWith(u8, id, "resp_")) try r.put(a, "id", .{ .string = id });
        try r.put(a, "object", .{ .string = "response" });
        try r.put(a, "status", .{ .string = "failed" });
        try r.put(a, "error", fixed);
        try o.put(a, "response", .{ .object = r });
    } else if (nested != null) {
        try o.put(a, "error", fixed);
    } else for (fixed.object.keys(), fixed.object.values()) |k, v| if (!std.mem.eql(u8, k, "type")) try o.put(a, k, v);
    return .{ .object = o };
}

/// An error's status: the one it names, else its class from its type and code (a refusal is 4xx, a failure 5xx).
fn statusOf(named: ?Value, kind: ?[]const u8, code: ?[]const u8) u16 {
    if (named) |n| if (n.int64()) |v| return @intCast(std.math.clamp(v, 400, 599));
    if (allowedCode(code)) |c| for ([_][]const u8{ wait.restarting.code, wait.loading.code, wait.reload_failed.code, wait.engine_busy, wait.engine_unavailable, wait.memory_pressure }) |w|
        if (std.mem.eql(u8, c, w)) return 503;
    if (listed(&.{"overloaded_error"}, kind) != null) return 503;
    if (listed(&.{ "server_error", "api_error" }, kind) != null or listed(&.{"server_error"}, code) != null) return 500;
    return 400;
}

fn fieldOf(e: Value, key: []const u8) ?[]const u8 {
    return if (e == .object) e.strField(key) else null;
}

/// Under --confidential: the access line, and for an error status a fixed JSON body in place of the route's.
pub fn head(c: *Conn, body: []const u8) Allocator.Error!?[]const u8 {
    var id_buf: [32]u8 = undefined;
    const id = requestId(c, &id_buf);
    if (c.status < 400) {
        flushAccess(c);
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
    if (std.mem.eql(u8, c.version, "HTTP/0.9")) return try c.gpa.dupe(u8, text); // no head: the body alone
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
