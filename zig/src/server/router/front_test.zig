//! The router over loopback sockets with fake children: routing, refusals, keys, byte-for-byte streams and crashes.
const std = @import("std");
const auth = @import("../auth.zig");
const listener_mod = @import("../listener.zig");
const tc = @import("test_child.zig");
const fake = @import("../benchmark_fake.zig");
const catalog = @import("catalog.zig");
const pool_mod = @import("pool.zig");
const front = @import("front.zig");
const t = std.testing;
const posix = std.posix;
const exchangeWith = tc.exchangeWith;
const dial = tc.dial;
const sendAll = tc.sendAll;
const readUntil = tc.readUntil;
const post = tc.post;
const get = tc.get;
const status = tc.status;
const bodyOf = tc.bodyOf;

const Spec = struct { id: []const u8, aliases: []const []const u8 = &.{}, routes: []const fake.Route = &.{} };

const Harness = struct {
    arena: std.heap.ArenaAllocator = .init(t.allocator),
    tmp: std.testing.TmpDir = undefined,
    children: std.ArrayList(tc.Child) = .empty,
    pool: *pool_mod.Pool = undefined,
    front: *front.Front = undefined,
    store: ?auth.Store = null,
    lis: listener_mod.Listener = undefined,
    stop: std.atomic.Value(bool) = .init(false),
    accept: ?std.Thread = null,

    fn start(h: *Harness, specs: []const Spec, max_loaded: u32, keys: []const []const u8) !void {
        const a = h.arena.allocator();
        h.tmp = std.testing.tmpDir(.{});
        var entries: std.ArrayList(catalog.Entry) = .empty;
        for (specs) |s| {
            const c = try tc.start(a, s.id, s.aliases, s.routes);
            try h.children.append(a, c);
            try entries.append(a, c.entry);
        }
        h.pool = try tc.pool(a, &h.tmp, entries.items, .{ .max_loaded = max_loaded, .busy_wait_s = 5, .log_dir = "" });
        var problem: []const u8 = "";
        if (keys.len > 0) h.store = try auth.Store.init(t.allocator, keys, "", null, false, &problem);
        h.front = try a.create(front.Front);
        h.front.* = .{ .gpa = t.allocator, .io = t.io, .pool = h.pool, .entries = h.pool.entries, .keys = if (h.store) |*s| s else null, .version = "0.0.0-test" };
        h.lis = try listener_mod.Listener.open(.{ .ip4 = .loopback(0) });
        h.accept = try std.Thread.spawn(.{}, front.Front.serve, .{ h.front, h.lis, &h.stop });
    }

    fn deinit(h: *Harness) void {
        h.stop.store(true, .release);
        if (h.accept) |th| th.join();
        var waited: u32 = 0;
        while (h.front.open.load(.acquire) > 0 and waited < 500) : (waited += 1) fake.nap(10);
        h.pool.stopAll();
        h.pool.deinit();
        h.lis.close();
        if (h.store) |*s| s.deinit();
        for (h.children.items) |c| c.server.stop();
        h.tmp.cleanup();
        h.arena.deinit();
    }

    /// One whole exchange: ``request`` sent on a new connection, the reply read until the router closes it.
    fn exchange(h: *Harness, request: []const u8) ![]const u8 {
        return exchangeWith(h.arena.allocator(), h.lis.port(), request);
    }

    fn childKey(h: *Harness, i: usize) [64]u8 {
        h.pool.mutex.lockUncancelable(t.io);
        defer h.pool.mutex.unlock(t.io);
        return h.pool.slots[i].key;
    }
};

fn has(text: []const u8, needle: []const u8) bool {
    if (std.mem.indexOf(u8, text, needle) != null) return true;
    std.debug.print("missing \"{s}\" in:\n{s}\n", .{ needle, text });
    return false;
}

fn route(a: std.mem.Allocator, path: []const u8, replies: []const []const fake.Write) !fake.Route {
    return .{ .method = "POST", .path = path, .replies = try a.dupe([]const fake.Write, replies) };
}

test "serve --router: a request reaches the child its model names, by id or alias, with its body unchanged" {
    var h: Harness = .{};
    const a = h.arena.allocator();
    try h.start(&.{
        .{ .id = "alpha", .aliases = &.{"a1"}, .routes = &.{try route(a, "/v1/chat/completions", &.{try fake.jsonReply(a, "200 OK", "{\"id\": \"from-alpha\"}")})} },
        .{ .id = "beta", .routes = &.{try route(a, "/v1/chat/completions", &.{try fake.jsonReply(a, "200 OK", "{\"id\": \"from-beta\"}")})} },
    }, 2, &.{});
    defer h.deinit();
    const body = "{\"model\": \"a1\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}";
    const first = try h.exchange(try post(a, "/v1/chat/completions", body, ""));
    try t.expectEqual(@as(u16, 200), status(first));
    try t.expectEqualStrings("{\"id\": \"from-alpha\"}", bodyOf(first));
    const alpha = h.children.items[0].server;
    const seen = alpha.request(alpha.count() - 1);
    try t.expectEqualStrings("/v1/chat/completions", seen.path);
    try t.expectEqualStrings(body, seen.body);
    const second = try h.exchange(try post(a, "/v1/chat/completions", "{\"model\": \"beta\"}", ""));
    try t.expectEqualStrings("{\"id\": \"from-beta\"}", bodyOf(second));
    try t.expectEqual(@as(usize, 2), (try h.pool.loaded(a)).len);
}

test "serve --router: an unknown or missing model is refused in its route's shape and loads nothing" {
    var h: Harness = .{};
    const a = h.arena.allocator();
    try h.start(&.{.{ .id = "alpha" }}, 1, &.{});
    defer h.deinit();
    const openai = try h.exchange(try post(a, "/v1/chat/completions", "{\"model\": \"nope\"}", ""));
    try t.expectEqual(@as(u16, 404), status(openai));
    try t.expectEqualStrings("{\"error\": {\"message\": \"model 'nope' is not served here; GET /v1/models lists the models this router serves\", \"type\": \"invalid_request_error\", \"param\": \"model\", \"code\": \"model_not_found\"}}", bodyOf(openai));
    const ollama = try h.exchange(try post(a, "/api/chat", "{\"model\": \"nope\"}", ""));
    try t.expectEqual(@as(u16, 404), status(ollama));
    try t.expectEqualStrings("{\"error\": \"model 'nope' is not served here; GET /v1/models lists the models this router serves\"}", bodyOf(ollama));
    const messages = try h.exchange(try post(a, "/v1/messages", "{\"model\": \"nope\"}", ""));
    try t.expect(status(messages) == 404 and has(bodyOf(messages), "\"type\": \"not_found_error\""));
    const missing = try h.exchange(try post(a, "/v1/completions", "{\"prompt\": \"x\"}", ""));
    try t.expect(status(missing) == 400 and has(bodyOf(missing), "\"code\": \"model_required\""));
    const broken = try h.exchange(try post(a, "/v1/embeddings", "{not json", ""));
    try t.expectEqual(@as(u16, 400), status(broken));
    try t.expectEqual(@as(usize, 0), (try h.pool.loaded(a)).len);
}

test "serve --router: the child gets its own key in place of the client's, and no reply carries it" {
    var h: Harness = .{};
    const a = h.arena.allocator();
    try h.start(&.{.{ .id = "alpha", .routes = &.{try route(a, "/v1/completions", &.{try fake.jsonReply(a, "200 OK", "{\"ok\": true}")})} }}, 1, &.{"sk-router-example"});
    defer h.deinit();
    const keyed = "Authorization: Bearer sk-router-example\r\nX-Api-Key: sk-router-example\r\nOrigin: http://127.0.0.1\r\n";
    const reply = try h.exchange(try post(a, "/v1/completions", "{\"model\": \"alpha\", \"prompt\": \"x\"}", keyed));
    try t.expectEqual(@as(u16, 200), status(reply));
    const key = h.childKey(0);
    const alpha = h.children.items[0].server;
    const last = alpha.count() - 1;
    try t.expectEqualStrings(try std.fmt.allocPrint(a, "Bearer {s}", .{&key}), alpha.header(last, "authorization").?);
    try t.expect(alpha.header(last, "x-api-key") == null and alpha.header(last, "origin") == null);
    try t.expect(std.mem.indexOf(u8, alpha.request(last).head, "sk-router-example") == null);
    for ([_][]const u8{ "/v1/models", "/health", "/api/ps", "/api/tags" }) |path| {
        const listed = try h.exchange(try get(a, path, "Authorization: Bearer sk-router-example\r\n"));
        try t.expectEqual(@as(u16, 200), status(listed));
        try t.expect(std.mem.indexOf(u8, listed, &key) == null);
    }
    try t.expect(std.mem.indexOf(u8, reply, &key) == null);
    const open = try h.exchange(try post(a, "/v1/completions", "{\"model\": \"alpha\"}", ""));
    try t.expectEqual(@as(u16, 401), status(open));
    try t.expectEqual(@as(u16, 200), status(try h.exchange(try get(a, "/health", ""))));
}

test "serve --router: a streamed reply passes byte for byte as it arrives, an event split across reads included" {
    var h: Harness = .{};
    const a = h.arena.allocator();
    var gate: std.Io.Event = .unset;
    const head = try fake.headOf(a, "200 OK", "text/event-stream");
    const part = try std.mem.concat(a, u8, &.{ head, ": keepalive\n\ndata: {\"choices\": [{\"delta\": {\"content\": \"he" });
    const rest = "llo\"}}]}\n\n";
    const done = "data: [DONE]\n\n";
    const writes = try a.dupe(fake.Write, &.{ .{ .bytes = part, .after = &gate }, .{ .bytes = rest, .pause_ms = 30 }, .{ .bytes = done } });
    try h.start(&.{.{ .id = "alpha", .routes = &.{try route(a, "/v1/chat/completions", &.{writes})} }}, 1, &.{});
    defer h.deinit();
    defer gate.set(t.io); // before the fake stops, so a test that fails early never leaves its writer waiting
    const fd = try dial(h.lis.port());
    defer _ = posix.system.close(fd);
    try sendAll(fd, try post(a, "/v1/chat/completions", "{\"model\": \"alpha\", \"stream\": true}", ""));
    var got: std.ArrayList(u8) = .empty;
    // the first write ends inside an event and the child then waits: the router must pass it on unfinished
    try readUntil(a, fd, &got, part.len, 10_000);
    try t.expectEqualStrings(part, got.items);
    gate.set(t.io);
    try readUntil(a, fd, &got, null, 10_000);
    try t.expectEqualStrings(try std.mem.concat(a, u8, &.{ part, rest, done }), got.items);
}

const Background = struct {
    arena: std.heap.ArenaAllocator = .init(t.allocator),
    port: u16,
    request: []const u8,
    reply: ?[]const u8 = null,

    fn run(b: *Background) void {
        b.reply = exchangeWith(b.arena.allocator(), b.port, b.request) catch null;
    }
};

/// Waits until ``server`` has seen ``n`` POSTs to ``path``.
fn posted(server: *fake.Fake, path: []const u8, n: usize) bool {
    var waited: u32 = 0;
    while (waited < 10_000) : (waited += 10) {
        var seen: usize = 0;
        for (0..server.count()) |i| {
            const r = server.request(i);
            seen += @intFromBool(std.mem.eql(u8, r.method, "POST") and std.mem.eql(u8, r.path, path));
        }
        if (seen >= n) return true;
        fake.nap(10);
    }
    return false;
}

test "serve --router: a child that dies answering gives 502 or ends its stream, and the next request starts it again" {
    var h: Harness = .{};
    const a = h.arena.allocator();
    var before_head: std.Io.Event = .unset;
    var mid_stream: std.Io.Event = .unset;
    const partial = try std.mem.concat(a, u8, &.{ try fake.headOf(a, "200 OK", "text/event-stream"), "data: {\"a\"" });
    try h.start(&.{.{ .id = "alpha", .routes = &.{try route(a, "/v1/completions", &.{
        try a.dupe(fake.Write, &.{.{ .bytes = "", .after = &before_head }}),
        try a.dupe(fake.Write, &.{.{ .bytes = partial, .after = &mid_stream }}),
        try fake.jsonReply(a, "200 OK", "{\"ok\": 3}"),
    })} }}, 1, &.{});
    defer h.deinit();
    defer before_head.set(t.io);
    defer mid_stream.set(t.io);
    const alpha = h.children.items[0].server;
    const request = try post(a, "/v1/completions", "{\"model\": \"alpha\"}", "");
    var bg: Background = .{ .port = h.lis.port(), .request = request };
    defer bg.arena.deinit();
    const thread = try std.Thread.spawn(.{}, Background.run, .{&bg});
    try t.expect(posted(alpha, "/v1/completions", 1));
    const first = tc.pid(h.pool, 0).?;
    try posix.kill(first, .KILL);
    before_head.set(t.io);
    thread.join();
    try t.expectEqual(@as(u16, 502), status(bg.reply.?));
    try t.expect(has(bodyOf(bg.reply.?), "the server for model 'alpha' exited while answering; the next request starts it again"));
    try t.expect(tc.pid(h.pool, 0) == null);
    var bg2: Background = .{ .port = h.lis.port(), .request = request };
    defer bg2.arena.deinit();
    const again = try std.Thread.spawn(.{}, Background.run, .{&bg2});
    try t.expect(posted(alpha, "/v1/completions", 2));
    const second = tc.pid(h.pool, 0).?;
    try t.expect(second != first);
    try posix.kill(second, .KILL);
    mid_stream.set(t.io);
    again.join();
    try t.expectEqualStrings(partial, bg2.reply.?); // the stream ends where the child stopped, as a lone server's would
    const third = try h.exchange(request);
    try t.expectEqualStrings("{\"ok\": 3}", bodyOf(third));
    try t.expect(tc.pid(h.pool, 0).? != second);
}

test "serve --router: routes it does not serve are refused by name and load nothing" {
    var h: Harness = .{};
    const a = h.arena.allocator();
    try h.start(&.{.{ .id = "alpha" }}, 1, &.{});
    defer h.deinit();
    const cases = [_]struct { request: []const u8, says: []const u8 }{
        .{ .request = try get(a, "/v1/responses/resp_1", ""), .says = "GET /v1/responses/{id} is not served through the router in this version" },
        .{ .request = try get(a, "/v1/responses", "Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"), .says = "The Responses WebSocket mode is not served through the router" },
        .{ .request = try post(a, "/v1/audio/speech", "{\"model\": \"alpha\"}", ""), .says = "POST /v1/audio/speech is not served through the router" },
        .{ .request = try get(a, "/metrics", ""), .says = "metrics are per model" },
        .{ .request = try get(a, "/v1/models/alpha", ""), .says = "GET /v1/models/alpha is not served through the router" },
        .{ .request = try get(a, "/api/show", ""), .says = "GET /api/show is not served through the router" },
    };
    for (cases) |c| {
        const reply = try h.exchange(c.request);
        try t.expectEqual(@as(u16, 404), status(reply));
        try t.expect(has(bodyOf(reply), c.says));
    }
    try t.expect(has(bodyOf(try h.exchange(try get(a, "/api/show", ""))), "{\"error\": \""));
    try t.expectEqual(@as(usize, 0), (try h.pool.loaded(a)).len);
}

test "serve --router: /v1/models lists every model with loaded, a ready child's entries as it gives them" {
    var h: Harness = .{};
    const a = h.arena.allocator();
    try h.start(&.{
        .{ .id = "alpha", .aliases = &.{"a1"}, .routes = &.{try route(a, "/v1/completions", &.{try fake.jsonReply(a, "200 OK", "{}")})} },
        .{ .id = "beta" },
    }, 1, &.{});
    defer h.deinit();
    const cold = bodyOf(try h.exchange(try get(a, "/v1/models", "")));
    try t.expect(has(cold, "\"id\": \"alpha\"") and has(cold, "\"id\": \"a1\"") and has(cold, "\"id\": \"beta\""));
    try t.expect(std.mem.indexOf(u8, cold, "\"loaded\": true") == null and std.mem.indexOf(u8, cold, "context_length") == null);
    _ = try h.exchange(try post(a, "/v1/completions", "{\"model\": \"alpha\"}", ""));
    const warm = bodyOf(try h.exchange(try get(a, "/v1/models", "")));
    try t.expect(has(warm, "{\"id\": \"alpha\", \"object\": \"model\", \"created\": 1, \"owned_by\": \"tensorfold\", \"context_length\": 4096, \"loaded\": true}"));
    try t.expect(has(warm, "{\"id\": \"a1\", \"object\": \"model\", \"created\": 1, \"owned_by\": \"tensorfold\", \"context_length\": 4096, \"loaded\": true}"));
    try t.expect(has(warm, "\"owned_by\": \"tensorfold\", \"loaded\": false}"));
    try t.expectEqualStrings("{\"models\": [{\"name\": \"alpha\", \"model\": \"alpha\"}, {\"name\": \"a1\", \"model\": \"a1\"}, {\"name\": \"beta\", \"model\": \"beta\"}]}", bodyOf(try h.exchange(try get(a, "/api/tags", ""))));
    try t.expectEqualStrings("{\"models\": [{\"name\": \"alpha\", \"model\": \"alpha\"}]}", bodyOf(try h.exchange(try get(a, "/api/ps", ""))));
    try t.expectEqualStrings("{\"status\": \"ok\", \"router\": true, \"models\": 2, \"max_loaded\": 1, \"loaded\": [{\"id\": \"alpha\", \"state\": \"ready\", \"requests_in_progress\": 0}], \"version\": \"0.0.0-test\"}", bodyOf(try h.exchange(try get(a, "/health", ""))));
    try t.expectEqualStrings("{\"version\": \"0.0.0-test\"}", bodyOf(try h.exchange(try get(a, "/api/version", ""))));
}
