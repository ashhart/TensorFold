//! POST /v1/requests/{id}/cancel: a running or queued request cancelled by the id its client got.
const std = @import("std");
const json = @import("json");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const auth = @import("auth.zig");
const routes = @import("routes.zig");
const Allocator = std.mem.Allocator;

/// A client request's ask: set by the route, read by each of its generation loops (chat_generation.zig).
pub const Entry = struct { asked: std.atomic.Value(bool) = .init(false) };

/// The process's client requests by public id (chatcmpl-, cmpl-, resp_), held by openai.runNamed; ids are random.
pub var registry: Registry = .{};

/// The Responses id of the turn this thread runs: responses.turn sets it around openai.runNamed, which holds it too.
pub threadlocal var alias: ?[]const u8 = null;

pub const Registry = struct {
    mutex: std.Io.Mutex = .init,
    map: std.StringHashMapUnmanaged(*Entry) = .empty,

    const gpa = std.heap.smp_allocator;

    /// `e` under each nonempty name in `names` (a name already held keeps its entry); out of memory: not held.
    pub fn add(r: *Registry, io: std.Io, names: []const []const u8, e: *Entry) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        for (names) |n| if (n.len > 0) {
            const got = r.map.getOrPut(gpa, n) catch continue;
            if (got.found_existing) continue;
            got.key_ptr.* = gpa.dupe(u8, n) catch {
                r.map.removeByPtr(got.key_ptr);
                continue;
            };
            got.value_ptr.* = e;
        };
    }

    /// The names `e` holds, dropped (a name another entry holds stays).
    pub fn remove(r: *Registry, io: std.Io, names: []const []const u8, e: *const Entry) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        for (names) |n| if (r.map.getEntry(n)) |kv| if (kv.value_ptr.* == e) {
            const key = kv.key_ptr.*;
            r.map.removeByPtr(kv.key_ptr);
            gpa.free(key);
        };
    }

    /// Asks the request named `id` to stop; false when no running or queued request has it.
    pub fn cancel(r: *Registry, io: std.Io, id: []const u8) bool {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        const e = r.map.get(id) orelse return false;
        e.asked.store(true, .release);
        return true;
    }

    /// The names held now (for tests), copied into `a`.
    pub fn held(r: *Registry, io: std.Io, a: Allocator) ![][]const u8 {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        const out = try a.alloc([]const u8, r.map.count());
        var it = r.map.keyIterator();
        var i: usize = 0;
        while (it.next()) |k| : (i += 1) out[i] = try a.dupe(u8, k.*);
        return out;
    }
};

/// The id in `route` when it is /v1/requests/{id}/cancel (or without /v1), else null.
pub fn match(route: []const u8) ?[]const u8 {
    for ([_][]const u8{ "/v1/requests/", "/requests/" }) |head| if (std.mem.startsWith(u8, route, head)) {
        const rest = route[head.len..];
        if (!std.mem.endsWith(u8, rest, "/cancel")) return null;
        const id = rest[0 .. rest.len - "/cancel".len];
        if (id.len == 0 or std.mem.indexOfScalar(u8, id, '/') != null) return null;
        return id;
    };
    return null;
}

/// The route's answer: 200 with the id when a request had it, else 404 `request_not_found`.
pub fn answer(io: std.Io, a: Allocator, id: []const u8) !struct { status: u16, body: json.Value } {
    const o = try json.newObject(a);
    if (registry.cancel(io, id)) {
        try o.put(a, "id", .{ .string = id });
        try o.put(a, "object", .{ .string = "request.cancel" });
        try o.put(a, "cancelled", .{ .bool = true });
        return .{ .status = 200, .body = .{ .object = o } };
    }
    const err = try json.newObject(a);
    try err.put(a, "message", .{ .string = try std.fmt.allocPrint(a, "no running or queued request has the id {s}", .{id}) });
    try err.put(a, "type", .{ .string = "invalid_request_error" });
    try err.put(a, "code", .{ .string = "request_not_found" });
    try o.put(a, "error", .{ .object = err });
    return .{ .status = 404, .body = .{ .object = o } };
}

/// Headers a proxy adds: a keyless server's loopback peer may be a proxy relaying anyone, so they refuse.
const relayed = [_][]const u8{ "Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Real-IP", "Via", "CF-Connecting-IP", "True-Client-IP" };

/// Who may cancel or flush the prompt cache: with keys, an admin key; without, a loopback peer no proxy relayed.
pub fn allowed(keyed: bool, conn: *const Conn) bool {
    if (keyed) return conn.key_label != null and conn.key_scopes.admin; // authenticate matched it before routing
    for (relayed) |name| if (conn.headerCount(name) > 0) return false;
    return auth.loopbackAddress(peerAddress(conn.peer) orelse return false); // the socket's address, never a header
}

/// conn.peer as listener.peerText writes it; IpAddress.parse refuses its IPv4-mapped form, ::ffff: and two hex groups.
fn peerAddress(text: []const u8) ?std.Io.net.IpAddress {
    if (std.Io.net.IpAddress.parse(text, 0)) |ip| return ip else |_| {}
    if (!std.mem.startsWith(u8, text, "::ffff:")) return null;
    var groups = std.mem.splitScalar(u8, text["::ffff:".len..], ':');
    const hi = std.fmt.parseInt(u16, groups.next().?, 16) catch return null;
    const lo = std.fmt.parseInt(u16, groups.next() orelse return null, 16) catch return null;
    if (groups.next() != null) return null;
    return .{ .ip4 = .{ .bytes = .{ @truncate(hi >> 8), @truncate(hi), @truncate(lo >> 8), @truncate(lo) }, .port = 0 } };
}

/// The 403 a keyless server gives a caller off the loopback address or behind a proxy.
pub fn refusal(a: Allocator) !json.Value {
    const err = try json.newObject(a);
    try err.put(a, "message", .{ .string = "without API keys, only a direct loopback caller may cancel a request" });
    try err.put(a, "type", .{ .string = "permission_error" });
    try err.put(a, "code", .{ .string = "loopback_only" });
    const o = try json.newObject(a);
    try o.put(a, "error", .{ .object = err });
    return .{ .object = o };
}

/// POST /v1/requests/{id}/cancel: an admin key's route, or a direct loopback caller's on a server without keys.
pub fn post(srv: *Server, conn: *Conn, a: Allocator, id: []const u8) void {
    const keyed = if (srv.keys) |k| k.enabled() else false;
    if (!allowed(keyed, conn)) {
        if (keyed) return @import("auth_scope_reply.zig").deny(conn, false, .route);
        return routes.sendValue(conn, a, 403, refusal(a) catch return);
    }
    const got = answer(srv.io, a, id) catch return conn.sendError(a, 500, "out of memory", null) catch {};
    routes.sendValue(conn, a, got.status, got.body);
}

test "the cancel route's path names one id" {
    try std.testing.expectEqualStrings("chatcmpl-abc", match("/v1/requests/chatcmpl-abc/cancel").?);
    try std.testing.expectEqualStrings("resp_x", match("/requests/resp_x/cancel").?);
    for ([_][]const u8{ "/v1/requests//cancel", "/v1/requests/a/b/cancel", "/v1/requests/a", "/v1/requests/a/cancelled", "/v1/chat/completions" }) |r|
        try std.testing.expect(match(r) == null);
}

test "a held name is cancelled once asked, dropped when its request ends, and an unknown one is not found" {
    const io = std.testing.io;
    var e: Entry = .{};
    const names_ = [_][]const u8{ "cmpl-test-held", "", "resp_test-held" };
    registry.add(io, &names_, &e);
    try std.testing.expect(registry.cancel(io, "resp_test-held"));
    try std.testing.expect(e.asked.load(.acquire));
    registry.remove(io, &names_, &e);
    try std.testing.expect(!registry.cancel(io, "cmpl-test-held"));
    try std.testing.expect(!registry.cancel(io, "cmpl-never"));
}

test "with API keys only an admin key may cancel; without keys only a direct loopback caller may" {
    const scope = @import("auth_scope.zig");
    const path = "/v1/requests/chatcmpl-x/cancel";
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c: Conn = .{ .fd = -1, .peer = "10.0.0.5", .buf = &.{}, .gpa = a };
    // keys: the gate refuses scoped keys before the route (403 insufficient_scope), and the route checks again
    for ([_]scope.Scopes{ .{ .inference = true }, .{ .read = true }, .{ .inference = true, .read = true } }) |s| {
        try std.testing.expect(!scope.allows(s, "POST", path));
        c.key_label = "app";
        c.key_scopes = s;
        try std.testing.expect(!allowed(true, &c));
    }
    try std.testing.expect(scope.allows(scope.full, "POST", path));
    c.key_scopes = scope.full;
    try std.testing.expect(allowed(true, &c));
    c.key_label = null; // no key matched: never the route's, even from loopback
    c.peer = "127.0.0.1";
    try std.testing.expect(!allowed(true, &c));
    // no keys: loopback peers only, in conn.peer's forms (listener.peerText writes IPv4-mapped as hex groups)
    var buf: [64]u8 = undefined;
    const mapped = @import("listener.zig").peerText(&buf, try std.Io.net.IpAddress.parse("::ffff:127.0.0.1", 0));
    try std.testing.expectEqualStrings("::ffff:7f00:1", mapped);
    for ([_][]const u8{ "127.0.0.1", "127.8.9.10", "::1", mapped, "::ffff:7f12:3456" }) |p| {
        c.peer = p;
        try std.testing.expect(allowed(false, &c));
    }
    const refused = [_][]const u8{ "10.1.2.3", "::ffff:a00:1", "::ffff:7f00", "::ffff:7f00:1:2", "2001:db8::1", "", "localhost", "0.0.0.0" };
    for (refused) |p| {
        c.peer = p;
        try std.testing.expect(!allowed(false, &c));
    }
    // a proxy on the same machine: its loopback peer is refused, whatever the forwarded address claims
    c.peer = "127.0.0.1";
    for (relayed) |name| {
        var headers = [_]@import("http_conn.zig").Header{.{ .name = name, .value = "127.0.0.1" }};
        c.headers = &headers;
        try std.testing.expect(!allowed(false, &c));
    }
    var lower = [_]@import("http_conn.zig").Header{.{ .name = "x-forwarded-for", .value = "127.0.0.1" }};
    c.headers = &lower;
    try std.testing.expect(!allowed(false, &c));
    try std.testing.expectEqualStrings("loopback_only", (try refusal(a)).get("error").?.strField("code").?);
}
