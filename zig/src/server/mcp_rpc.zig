//! MCP's wire: one JSON-RPC 2.0 message a line, the protocol revisions served and how a request picks one.
const std = @import("std");
const json = @import("json");
const Allocator = std.mem.Allocator;
const Value = json.Value;
const none: Value = .null;

pub const parse_error: i32 = -32700;
pub const invalid_request: i32 = -32600;
pub const method_not_found: i32 = -32601;
pub const invalid_params: i32 = -32602;
pub const internal_error: i32 = -32603;
/// 2026-07-28's UnsupportedProtocolVersionError: the request's revision is not one this server speaks.
pub const unsupported_version: i32 = -32022;

/// The longest message line read; a longer one is refused and skipped to its newline.
pub const max_line = 4 << 20;

pub const meta_version = "io.modelcontextprotocol/protocolVersion";
pub const meta_capabilities = "io.modelcontextprotocol/clientCapabilities";
pub const meta_server = "io.modelcontextprotocol/serverInfo";

/// A revision served: 2026-07-28 names itself in every request's _meta; the others open with an initialize handshake.
pub const Version = enum {
    @"2026-07-28",
    @"2025-11-25",
    @"2025-06-18",

    /// True for the stateless revision, whose results carry resultType and the server's identity.
    pub fn stateless(v: Version) bool {
        return v == .@"2026-07-28";
    }
};

/// Every revision served, newest first, as an UnsupportedProtocolVersionError and server/discover list them.
pub const supported = [_][]const u8{ "2026-07-28", "2025-11-25", "2025-06-18" };

/// initialize's answer: the requested handshake revision when served, else the newest one.
pub fn negotiate(requested: ?[]const u8) Version {
    const v = std.meta.stringToEnum(Version, requested orelse "") orelse return .@"2025-11-25";
    return if (v.stateless()) .@"2025-11-25" else v;
}

pub const Request = struct { id: Value, method: []const u8, params: Value };
pub const Notification = struct { method: []const u8, params: Value };
/// A message answered with an error: ``id`` is null when the message's own could not be read.
pub const Refusal = struct { id: Value = .null, code: i32, message: []const u8, data: ?Value = null };

pub const Message = union(enum) {
    request: Request,
    notification: Notification,
    refused: Refusal,
    /// Nothing to answer: a blank line, a client's reply (this server sends no requests) or a bad notification.
    dropped: []const u8,
};

/// One line as a message; params is an object or null.
pub fn read(a: Allocator, line: []const u8) !Message {
    const text = std.mem.trim(u8, line, " \t\r");
    if (text.len == 0) return .{ .dropped = "" };
    const v = switch (try json.parse(a, text)) {
        .ok => |v| v,
        .err => |why| return refuse(.null, parse_error, try std.fmt.allocPrint(a, "Parse error: {s}", .{why})),
    };
    if (v == .array) return refuse(.null, invalid_request, "Invalid Request: batches are not supported; send one message a line");
    if (v != .object) return refuse(.null, invalid_request, "Invalid Request: a message is a JSON object");
    const id = v.get("id");
    const id_ok = if (id) |x| x == .string or x == .int else false;
    const echo: Value = if (id_ok) id.? else .null;
    if (!std.mem.eql(u8, v.strField("jsonrpc") orelse "", "2.0")) return refuse(echo, invalid_request, "Invalid Request: jsonrpc must be \"2.0\"");
    const method = v.get("method") orelse {
        if (id != null and (v.has("result") or v.has("error"))) return .{ .dropped = "a reply to no request this server sent" };
        return refuse(echo, invalid_request, "Invalid Request: no method");
    };
    if (method != .string) return refuse(echo, invalid_request, "Invalid Request: method must be a string");
    const params = v.get("params") orelse .null;
    if (id == null) {
        if (params != .null and params != .object) return .{ .dropped = "a notification whose params are not an object" };
        return .{ .notification = .{ .method = method.string, .params = params } };
    }
    if (!id_ok) return refuse(.null, invalid_request, "Invalid Request: id must be a string or an integer");
    if (params != .null and params != .object) return refuse(echo, invalid_params, "Invalid params: params must be an object");
    return .{ .request = .{ .id = id.?, .method = method.string, .params = params } };
}

fn refuse(id: Value, code: i32, message: []const u8) Message {
    return .{ .refused = .{ .id = id, .code = code, .message = message } };
}

pub const Picked = union(enum) { ok: Version, refused: Refusal };

/// The revision a request is served under: its _meta's, else the handshake's; a refusal when neither names one.
pub fn version(a: Allocator, r: Request, handshake: ?Version) !Picked {
    const meta = r.params.get("_meta") orelse none;
    if (meta.get(meta_version)) |named| {
        const v: ?Version = if (named == .string) std.meta.stringToEnum(Version, named.string) else null;
        if (v == null or !v.?.stateless()) {
            const data = try json.newObject(a);
            try data.put(a, "supported", try strings(a, &supported));
            try data.put(a, "requested", named);
            return .{ .refused = .{ .id = r.id, .code = unsupported_version, .message = "Unsupported protocol version", .data = .{ .object = data } } };
        }
        if (meta.get(meta_capabilities) == null or meta.get(meta_capabilities).? != .object)
            return .{ .refused = .{ .id = r.id, .code = invalid_params, .message = "Invalid params: _meta names no " ++ meta_capabilities ++ " object" } };
        return .{ .ok = v.? };
    }
    if (handshake) |h| return .{ .ok = h };
    return .{ .refused = .{ .id = r.id, .code = invalid_params, .message = "Invalid params: no protocol version; send initialize first, or name " ++ meta_version ++ " in _meta" } };
}

pub fn strings(a: Allocator, items: []const []const u8) !Value {
    const out = try a.alloc(Value, items.len);
    for (out, items) |*o, s| o.* = .{ .string = s };
    return .{ .array = out };
}

/// ``{"jsonrpc": "2.0", "id": id, "result": result}``.
pub fn result(a: Allocator, id: Value, body: *json.Object) !Value {
    const o = try json.newObject(a);
    try o.put(a, "jsonrpc", .{ .string = "2.0" });
    try o.put(a, "id", id);
    try o.put(a, "result", .{ .object = body });
    return .{ .object = o };
}

/// ``{"jsonrpc": "2.0", "id": id, "error": {"code", "message", "data"}}``.
pub fn failure(a: Allocator, r: Refusal) !Value {
    const e = try json.newObject(a);
    try e.put(a, "code", try json.intValue(a, r.code));
    try e.put(a, "message", .{ .string = r.message });
    if (r.data) |d| try e.put(a, "data", d);
    const o = try json.newObject(a);
    try o.put(a, "jsonrpc", .{ .string = "2.0" });
    try o.put(a, "id", r.id);
    try o.put(a, "error", .{ .object = e });
    return .{ .object = o };
}

/// ``{"jsonrpc": "2.0", "method": method, "params": params}``, a notification needing no answer.
pub fn notification(a: Allocator, method: []const u8, params: *json.Object) !Value {
    const o = try json.newObject(a);
    try o.put(a, "jsonrpc", .{ .string = "2.0" });
    try o.put(a, "method", .{ .string = method });
    try o.put(a, "params", .{ .object = params });
    return .{ .object = o };
}

/// The stateless revision's result fields: ``resultType`` and the server's identity in ``_meta``.
pub fn complete(a: Allocator, body: *json.Object, v: Version, server: Value) !void {
    if (!v.stateless()) return;
    try body.put(a, "resultType", .{ .string = "complete" });
    const meta = try json.newObject(a);
    try meta.put(a, meta_server, server);
    try body.put(a, "_meta", .{ .object = meta });
}

/// stdout's messages, one whole line at a time under a lock, so call threads and the reader never interleave.
pub const Wire = struct {
    out: *std.Io.Writer,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    broken: bool = false,

    /// ``message`` as one compact ASCII line; a write error is told once on stderr, as the client may be gone.
    pub fn send(w: *Wire, message: Value) void {
        var arena: std.heap.ArenaAllocator = .init(std.heap.smp_allocator);
        defer arena.deinit();
        var line: std.Io.Writer.Allocating = .init(arena.allocator());
        json.write(&line.writer, message, .{ .compact = true }) catch return;
        line.writer.writeByte('\n') catch return;
        w.mutex.lockUncancelable(w.io);
        defer w.mutex.unlock(w.io);
        w.out.writeAll(line.written()) catch return w.fail();
        w.out.flush() catch return w.fail();
    }

    fn fail(w: *Wire) void {
        if (!w.broken) std.debug.print("tensorfold mcp: stdout is closed; replies are dropped\n", .{});
        w.broken = true;
    }
};
