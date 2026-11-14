//! POST /v1/prompt_cache/flush through the dispatcher: who may flush, the engine's answer, 501 and a withdrawn 503.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const auth = @import("auth.zig");
const scope = @import("auth_scope.zig");
const flush = @import("cache_flush.zig");
const routes = @import("routes.zig");
const http = @import("http_conn.zig");
const Server = @import("server.zig").Server;
const Allocator = std.mem.Allocator;
const Ask = api.ImprintAsk;
const t = std.testing;

/// An engine whose flush answers at once with fixed counts, or (`answers` false) holds the ask as long requests would.
const Engine = struct {
    asks: u32 = 0,
    answers: bool = true,
    takes: bool = false, // with `answers` false: taken by the engine thread, so its caller cannot withdraw it
    held: ?*Ask = null,
    fn engine(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .imprint = imprint } };
    }
    fn bare(e: *Engine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn imprint(ctx: *anyopaque, ask: *Ask) bool {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        e.asks += 1;
        if (!e.answers) {
            if (e.takes) ask.claim.store(.taken, .release);
            e.held = ask;
            return true;
        }
        ask.dropped = 3;
        ask.ram_freed = 4096;
        ask.freed = 512;
        ask.disk = true;
        ask.done.store(true, .release);
        return true;
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Busy;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn serve(engine: api.Engine, keys: ?*auth.Store) !*Server {
    const text = @import("embedding_protocol_test.zig").ByteText.model();
    return Server.init(t.allocator, t.io, engine, text, .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, keys);
}

/// The bytes a route writes, in place of a socket.
const Wire = struct {
    a: Allocator,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Wire = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

const Caller = struct { peer: []const u8 = "127.0.0.1", headers: []const http.Header = &.{}, key: ?[]const u8 = null, scopes: scope.Scopes = scope.full };

/// One flush through the dispatcher, as the connection thread hands it over after the key gate; status and body.
fn call(srv: *Server, a: Allocator, c: Caller) !struct { status: u16, body: json.Value } {
    var wire: Wire = .{ .a = a };
    const headers = try std.mem.concat(a, http.Header, &.{ &.{.{ .name = "Content-Length", .value = "2" }}, c.headers });
    var conn: http.Conn = .{ .fd = -1, .peer = c.peer, .buf = try a.dupe(u8, "{}"), .end = 2, .gpa = a, .version = "HTTP/1.1", .method = "POST", .path = flush.path, .headers = headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write }, .key_label = c.key, .key_scopes = c.scopes };
    routes.dispatch(srv, &conn, a);
    const out = wire.bytes.items;
    const at = std.mem.indexOf(u8, out, "\r\n\r\n") orelse return error.MissingHttpHead;
    return .{ .status = try std.fmt.parseInt(u16, out["HTTP/1.1 ".len..][0..3], 10), .body = (try json.parse(a, out[at + 4 ..])).ok };
}

fn code(body: json.Value) []const u8 {
    return body.get("error").?.strField("code").?;
}

test "security: prompt cache flush: a keyless server answers only a direct loopback caller" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Engine = .{};
    const srv = try serve(e.engine(), null);
    defer srv.deinit();
    for ([_][]const u8{ "10.1.2.3", "172.16.0.7", "2001:db8::1", "::ffff:a00:1", "localhost", "0.0.0.0", "" }) |peer| {
        const got = try call(srv, a, .{ .peer = peer });
        try t.expectEqual(@as(u16, 403), got.status);
        try t.expectEqualStrings("loopback_only", code(got.body));
    }
    // a reverse proxy on the same machine: its loopback socket is refused, whatever address it forwards
    for ([_][]const u8{ "Forwarded", "X-Forwarded-For", "x-real-ip", "Via", "CF-Connecting-IP", "True-Client-IP", "X-Forwarded-Host" }) |name| {
        const got = try call(srv, a, .{ .headers = &.{.{ .name = name, .value = "127.0.0.1" }} });
        try t.expectEqual(@as(u16, 403), got.status);
        try t.expectEqualStrings("loopback_only", code(got.body));
    }
    try t.expectEqual(@as(u32, 0), e.asks); // no refused caller reached the engine
    for ([_][]const u8{ "127.0.0.1", "127.8.9.10", "::1", "::ffff:7f00:1" }) |peer| {
        const got = try call(srv, a, .{ .peer = peer });
        try t.expectEqual(@as(u16, 200), got.status);
        try t.expectEqualStrings("prompt_cache.flush", got.body.strField("object").?);
        try t.expectEqualStrings("3", got.body.get("states").?.int);
        try t.expectEqualStrings("4096", got.body.get("ram_bytes").?.int);
        try t.expectEqualStrings("512", got.body.get("disk_bytes").?.int);
        try t.expect(got.body.get("disk").?.bool);
        try t.expect(got.body.get("waited_seconds") != null);
    }
    try t.expectEqual(@as(u32, 4), e.asks);
}

test "security: prompt cache flush: with API keys the route checks the matched key's admin scope itself" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var keys = try auth.Store.init(t.allocator, &.{"sk-flush-test"}, "", null, false, &problem);
    defer keys.deinit();
    var e: Engine = .{};
    const srv = try serve(e.engine(), &keys);
    defer srv.deinit();
    // the key gate refuses these before routing; if one got past it, the route refuses it too, from loopback as well
    const refused = [_]Caller{ .{}, .{ .key = "app", .scopes = .{ .inference = true } }, .{ .key = "grafana", .scopes = .{ .read = true } }, .{ .key = "both", .scopes = .{ .inference = true, .read = true } } };
    for (refused) |c| {
        const got = try call(srv, a, c);
        try t.expectEqual(@as(u16, 403), got.status);
        try t.expectEqualStrings("insufficient_scope", code(got.body));
    }
    try t.expectEqual(@as(u32, 0), e.asks);
    const admin = try call(srv, a, .{ .peer = "10.1.2.3", .key = "ops", .headers = &.{.{ .name = "X-Forwarded-For", .value = "10.9.9.9" }} });
    try t.expectEqual(@as(u16, 200), admin.status); // an admin key may flush from any address, through a proxy too
    try t.expectEqual(@as(u32, 1), e.asks);
}

test "prompt cache flush: 501 where no store keeps states, and a 503 whose flush was withdrawn" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Engine = .{};
    const bare = try flush.answer(t.allocator, t.io, a, e.bare(), 20);
    try t.expectEqual(@as(u16, 501), bare.status);
    try t.expectEqualStrings("not_supported", code(bare.body));
    // requests that do not end within the wait: the flush is withdrawn untaken, so the engine never runs it
    e.answers = false;
    const late = try flush.answer(t.allocator, t.io, a, e.engine(), 20);
    try t.expectEqual(@as(u16, 503), late.status);
    try t.expectEqualStrings("timeout", code(late.body));
    const withdrawn = e.held.?;
    defer t.allocator.destroy(withdrawn); // the engine's to drop: the route leaves it, never frees it under the engine
    try t.expectEqual(api.ImprintClaim.withdrawn, withdrawn.claim.load(.acquire));
    try t.expect(!withdrawn.done.load(.acquire));
    // one the engine took cannot be withdrawn: the route waits once more for it, then answers 503 and leaves it
    e.takes = true;
    const taken = try flush.answer(t.allocator, t.io, a, e.engine(), 20);
    try t.expectEqual(@as(u16, 503), taken.status);
    defer t.allocator.destroy(e.held.?);
    try t.expectEqual(api.ImprintClaim.taken, e.held.?.claim.load(.acquire));
    try t.expect(std.mem.indexOf(u8, taken.body.get("error").?.strField("message").?, "withdrawn") == null);
}

test "prompt cache flush: a model --idle-unload parked answers 503 unloaded, neither asked nor woken" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var e: Engine = .{};
    const srv = try serve(e.engine(), null);
    defer srv.deinit();
    var manager: @import("idle_unload.zig").Manager = .{ .gpa = t.allocator, .io = t.io, .args = .{}, .re = undefined, .dir = "m", .model_type = "m", .idle_s = 60, .parked = true, .state = .unloaded };
    srv.idle_manager = &manager;
    defer srv.idle_manager = null;
    const got = try call(srv, a, .{});
    try t.expectEqual(@as(u16, 503), got.status);
    try t.expectEqualStrings("model_unloaded", code(got.body));
    try t.expectEqual(@as(u32, 0), e.asks);
    try t.expectEqual(@as(u32, 0), manager.active); // the flush's lease on the engine ended with it
    manager.parked = false;
    manager.state = .resident;
    try t.expectEqual(@as(u16, 200), (try call(srv, a, .{})).status);
    try t.expectEqual(@as(u32, 1), e.asks);
    try t.expectEqual(@as(u32, 0), manager.active);
}
