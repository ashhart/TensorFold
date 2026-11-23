//! Responses ownership through production handlers and key matching, without a listening socket.
const std = @import("std");
const json = @import("json");
const api = @import("engine_api");
const server = @import("server.zig");
const auth = @import("auth.zig");
const harness = @import("ocr/harness_test.zig");
const cancels = @import("cancel_ids.zig");
const t = std.testing;
const A = std.mem.Allocator;
const key_a = "owner-test-a";
const key_b = "owner-test-b";
const Reply = struct { status: u16, body: []const u8 };

const Fixture = struct {
    srv: *server.Server,
    keys: auth.Store,
    text: @import("stream_preflight_test.zig").TestText = .{},
    probe: bool = false,
    foreign: u16 = 0,
    own: u16 = 0,
    asked_by_other: bool = false,
    foreign_body: ?[]u8 = null,
    submitted: usize = 0,

    fn init(f: *Fixture, keys: []const []const u8) !void {
        var why: []const u8 = "";
        f.* = .{ .srv = undefined, .keys = try auth.Store.init(t.allocator, keys, "", null, false, &why) };
        errdefer f.keys.deinit();
        f.srv = try server.Server.init(t.allocator, t.io, f.engine(), f.text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .enable_thinking = false, .use_drafts = false }, &f.keys);
    }
    fn deinit(f: *Fixture) void {
        f.srv.deinit();
        f.keys.deinit();
        if (f.foreign_body) |body| t.allocator.free(body);
    }
    fn http(f: *Fixture, a: A, method: []const u8, route: []const u8, key: ?[]const u8, body: []const u8) !Reply {
        var wire: harness.Wire = .{ .a = a };
        const conn = try harness.request(a, &wire, route, "application/json", body);
        conn.method = method;
        conn.peer = "127.0.0.1";
        conn.auth_enabled = f.keys.enabled();
        if (key) |value| {
            const bearer = try std.fmt.allocPrint(a, "Bearer {s}", .{value});
            const matched = f.keys.matchIdentity(a, bearer, null) orelse return error.TestKeyDidNotMatch;
            conn.key_digest = matched.digest;
            conn.key_label = matched.label;
            conn.key_scopes = matched.scopes;
        }
        f.srv.dispatch(conn, a);
        return .{ .status = wire.status(), .body = wire.bytes.items[wire.head().len..] };
    }
    fn engine(f: *Fixture) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .context_window = 4096 };
    }
    fn submit(ptr: *anyopaque, rid: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const f: *Fixture = @ptrCast(@alignCast(ptr));
        f.submitted += 1;
        sink.event(sink.ctx, rid, &.{ .prefilled = 0 });
        sink.event(sink.ctx, rid, &.{ .tokens = &.{ 'o', 'k' } });
        if (f.probe) {
            var arena = std.heap.ArenaAllocator.init(t.allocator);
            defer arena.deinit();
            const a = arena.allocator();
            const name = cancels.alias orelse return error.Busy;
            const route = std.fmt.allocPrint(a, "/v1/requests/{s}/cancel", .{name}) catch return error.Busy;
            const other = f.http(a, "POST", route, key_b, "") catch return error.Busy;
            f.foreign = other.status;
            f.foreign_body = t.allocator.dupe(u8, other.body) catch return error.Busy;
            cancels.registry.mutex.lockUncancelable(t.io);
            f.asked_by_other = cancels.registry.map.get(name).?.asked.load(.acquire);
            cancels.registry.mutex.unlock(t.io);
            f.own = (f.http(a, "POST", route, key_a, "") catch return error.Busy).status;
        }
        sink.event(sink.ctx, rid, &.{ .finished = .{ .reason = if (f.probe) .cancelled else .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

fn id(a: A, reply: Reply) ![]const u8 {
    try t.expectEqual(@as(u16, 200), reply.status);
    return (try json.parse(a, reply.body)).ok.strField("id").?;
}
fn path(a: A, rid: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "/v1/responses/{s}", .{rid});
}
fn next(a: A, rid: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"input\":\"next\",\"previous_response_id\":\"{s}\"}}", .{rid});
}
fn sameMiss(a: A, foreign: Reply, missing: Reply, rid: []const u8) !void {
    try t.expectEqual(missing.status, foreign.status);
    const one = try std.mem.replaceOwned(u8, a, foreign.body, rid, "<id>");
    const two = try std.mem.replaceOwned(u8, a, missing.body, "resp_missing", "<id>");
    try t.expectEqualStrings(two, one);
}

test "Responses owner: another full key cannot read, continue or delete a stored response" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.init(&.{ key_a, key_b });
    defer f.deinit();
    const rid = try id(a, try f.http(a, "POST", "/v1/responses", key_a, "{\"input\":\"private A turn\"}"));
    const stored = try path(a, rid);
    const missing = "/v1/responses/resp_missing";
    try sameMiss(a, try f.http(a, "GET", stored, key_b, ""), try f.http(a, "GET", missing, key_b, ""), rid);
    try sameMiss(a, try f.http(a, "POST", "/v1/responses", key_b, try next(a, rid)), try f.http(a, "POST", "/v1/responses", key_b, try next(a, "resp_missing")), rid);
    try sameMiss(a, try f.http(a, "DELETE", stored, key_b, ""), try f.http(a, "DELETE", missing, key_b, ""), rid);
    _ = try id(a, try f.http(a, "GET", stored, key_a, ""));
    const child = try id(a, try f.http(a, "POST", "/v1/responses", key_a, try next(a, rid)));
    try t.expectEqual(@as(u16, 404), (try f.http(a, "GET", try path(a, child), key_b, "")).status);
    try t.expectEqual(@as(u16, 200), (try f.http(a, "DELETE", stored, key_a, "")).status);
}

test "Responses owner: another full key cannot signal a running Responses cancellation" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.init(&.{ key_a, key_b });
    defer f.deinit();
    f.probe = true;
    const made = try f.http(a, "POST", "/v1/responses", key_a, "{\"input\":\"cancel this A turn\"}");
    const rid = try id(a, made);
    try t.expectEqual(@as(u16, 404), f.foreign);
    try t.expect(!f.asked_by_other);
    try t.expectEqual(@as(u16, 200), f.own);
    const unknown = try f.http(a, "POST", try std.fmt.allocPrint(a, "/v1/requests/{s}/cancel", .{rid}), key_b, "");
    try t.expectEqual(@as(u16, 404), unknown.status);
    try t.expectEqualStrings(unknown.body, f.foreign_body.?);
    try t.expectEqualStrings("cancelled", (try json.parse(a, made.body)).ok.strField("status").?);
    _ = try id(a, try f.http(a, "GET", try path(a, rid), key_a, ""));
    try t.expectEqual(@as(u16, 404), (try f.http(a, "GET", try path(a, rid), key_b, "")).status);
}

test "Responses owner: single-key and keyless read, continuation and deletion stay unchanged" {
    for ([_]?[]const u8{ key_a, null }) |key| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var f: Fixture = undefined;
        try f.init(if (key != null) &.{key_a} else &.{});
        defer f.deinit();
        const rid = try id(a, try f.http(a, "POST", "/v1/responses", key, "{\"input\":\"owned\"}"));
        _ = try id(a, try f.http(a, "GET", try path(a, rid), key, ""));
        _ = try id(a, try f.http(a, "POST", "/v1/responses", key, try next(a, rid)));
        try t.expectEqual(@as(u16, 200), (try f.http(a, "DELETE", try path(a, rid), key, "")).status);
    }
}

const Events = struct {
    count: usize = 0,
    fn sink(e: *Events) @import("responses_sink.zig").Sink {
        return .{ .ctx = e, .vt = &.{ .open = open, .event = event, .keepalive = keepalive } };
    }
    fn open(_: *anyopaque) error{Closed}!void {}
    fn keepalive(_: *anyopaque) error{Closed}!void {}
    fn event(ptr: *anyopaque, _: A, _: []const u8, _: *json.Object) error{Closed}!void {
        const e: *Events = @ptrCast(@alignCast(ptr));
        e.count += 1;
    }
};

test "Responses owner: a stored warmup records its creator without a generation request" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.init(&.{ key_a, key_b });
    defer f.deinit();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    const body = (try json.parse(a, "{\"input\":\"warmup\"}")).ok;
    const NoHistory = struct {
        pub fn conversation(_: @This(), c: *@import("errors.zig").Cx, _: json.Value) @import("errors.zig").Refused![]json.Value {
            return c.refuse("no earlier turn");
        }
    };
    const translated = try @import("responses_translate.zig").translate(&cx, body, NoHistory{});
    const matched = f.keys.matchIdentity(a, "Bearer " ++ key_a, null).?;
    var events: Events = .{};
    const made = @import("responses.zig").warmup(f.srv, a, &cx, body, translated, events.sink(), 6, .{}, matched.digest) orelse return error.WarmupFailed;
    try t.expect(events.count >= 3);
    try t.expectEqual(@as(usize, 0), f.submitted);
    const rid = made.strField("id").?;
    _ = try id(a, try f.http(a, "GET", try path(a, rid), key_a, ""));
    try t.expectEqual(@as(u16, 404), (try f.http(a, "GET", try path(a, rid), key_b, "")).status);
}

test "Responses owner: every chain ancestor and keyless entry keeps its own identity" {
    const store = @import("responses_store.zig");
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: @import("errors.zig").Cx = .{ .a = a };
    var s: store.Store = .{ .gpa = t.allocator };
    defer s.deinit();
    const one: [32]u8 = @splat(1);
    const two: [32]u8 = @splat(2);
    var messages = [_]json.Value{(try json.parse(a, "{\"role\":\"user\",\"content\":\"private\"}")).ok};
    try s.put(t.io, &cx, (try json.parse(a, "{\"id\":\"resp_b\",\"output\":[]}")).ok, &messages, two);
    try s.put(t.io, &cx, (try json.parse(a, "{\"id\":\"resp_a\",\"previous_response_id\":\"resp_b\",\"output\":[]}")).ok, &messages, one);
    try s.put(t.io, &cx, (try json.parse(a, "{\"id\":\"resp_plain\",\"output\":[]}")).ok, &messages, null);
    const ha: store.History = .{ .store = &s, .io = t.io, .owner = one };
    const hb: store.History = .{ .store = &s, .io = t.io, .owner = two };
    const hp: store.History = .{ .store = &s, .io = t.io, .owner = null };
    try t.expect(ha.lookup(&cx, "resp_a") == null);
    try t.expect(hb.lookup(&cx, "resp_a") == null);
    try t.expect(hp.lookup(&cx, "resp_a") == null);
    try t.expect(ha.lookup(&cx, "resp_plain") == null);
    try t.expectEqual(@as(usize, 1), hp.lookup(&cx, "resp_plain").?.len);
    try s.putWhole(t.io, &cx, (try json.parse(a, "{\"id\":\"resp_whole\",\"previous_response_id\":\"resp_b\",\"output\":[]}")).ok, &messages, one);
    try t.expectEqual(@as(usize, 1), ha.lookup(&cx, "resp_whole").?.len);
    try t.expect(hb.lookup(&cx, "resp_whole") == null);
    try t.expect((try s.get(t.io, a, "resp_whole", null)) == null);
    try t.expect(!(try s.get(t.io, a, "resp_whole", one)).?.has("owner"));
}
