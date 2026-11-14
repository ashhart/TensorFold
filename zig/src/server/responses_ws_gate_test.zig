//! Responses WebSocket mode behind key scopes and admission: each response.create meets the POST route's gates.
const std = @import("std");
const json = @import("json");
const auth = @import("auth.zig");
const d = @import("drain_test.zig");
const client = @import("responses_ws_client_test.zig");
const Fixture = @import("responses_ws_test.zig").Fixture;
const KeyFile = @import("auth_scope_test.zig").KeyFile;
const Client = client.Client;
const Value = json.Value;
const Allocator = std.mem.Allocator;
const testing = std.testing;

fn create(a: Allocator, fields: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"type\":\"response.create\",{s}}}", .{fields});
}

fn expectError(events: []Value, status: i64, code: []const u8) !void {
    const e = client.last(events);
    try testing.expectEqualStrings("error", e.strField("type").?);
    try testing.expectEqual(status, e.get("status").?.int64().?);
    try testing.expectEqualStrings(code, client.code(events));
}

fn expectCompleted(events: []Value) !void {
    try testing.expectEqualStrings("response.completed", client.last(events).strField("type").?);
}

fn param(events: []Value) ?[]const u8 {
    return client.last(events).get("error").?.strField("param");
}

test "Responses socket: a restricted key opens it, stays stateless but for its cache, and is checked on each create" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var keys = try KeyFile.init("app[inference]: infer-secret\nops[admin]: admin-secret\nmonitor[read]: read-secret\n");
    defer keys.deinit();
    var f: Fixture = undefined;
    try f.start(testing.allocator, testing.io, .{ .api_key_file = keys.path });
    errdefer f.stop() catch {};
    const io = f.io;

    // 9. the upgrade takes an inference or admin key; a read key and no key are refused before it
    var reader: Client = undefined;
    try reader.open(a, io, f.port, "read-secret");
    try testing.expectEqual(@as(u16, 403), reader.status);
    reader.close();
    var anonymous: Client = undefined;
    try anonymous.open(a, io, f.port, null);
    try testing.expectEqual(@as(u16, 401), anonymous.status);
    anonymous.close();
    var c: Client = undefined;
    try c.open(a, io, f.port, "infer-secret");
    try testing.expectEqual(@as(u16, 101), c.status);
    // store:true is denied; store:false runs, and continues from this socket's cache
    const stored = try c.turn(a, try create(a, "\"input\":\"x\""));
    try expectError(stored, 403, "insufficient_scope");
    try testing.expectEqualStrings("store", param(stored).?);
    const first = try c.turn(a, try create(a, "\"input\":\"x\",\"store\":false"));
    try expectCompleted(first);
    const cached = client.final(first).strField("id").?;
    try expectCompleted(try c.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"y\",\"store\":false,\"previous_response_id\":\"{s}\"", .{cached}))));
    // a stored parent stays the server's: the HTTP scope denial
    const admin = try client.http(a, io, f.port, "POST", "/v1/responses", "admin-secret", "{\"input\":\"s\",\"store\":true}");
    try testing.expectEqual(@as(u16, 200), admin.status);
    const stored_id = (try json.parse(a, admin.body)).ok.strField("id").?;
    const history = try c.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"y\",\"store\":false,\"previous_response_id\":\"{s}\"", .{stored_id})));
    try expectError(history, 403, "insufficient_scope");
    try testing.expectEqualStrings("previous_response_id", param(history).?);
    // the key loses its inference scope: the next create is denied and the socket stays open
    try keys.write("app[read]: infer-secret\nops[admin]: admin-secret\n");
    auth.reload_requested.store(true, .release);
    const narrowed = try c.turn(a, try create(a, "\"input\":\"z\",\"store\":false"));
    try expectError(narrowed, 403, "insufficient_scope");
    try testing.expect(param(narrowed) == null);
    // the key is removed: a connection-level 401 and close 1008
    try keys.write("ops[admin]: admin-secret\n");
    auth.reload_requested.store(true, .release);
    const removed = try c.turn(a, try create(a, "\"input\":\"z\",\"store\":false"));
    try expectError(removed, 401, "invalid_api_key");
    try testing.expectEqual(@as(?u16, 1008), try c.closeCode(a));
    c.close();
    try f.stop();
}

test "Responses socket: an admission refusal is a 429 error event and the socket stays open" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(testing.allocator, testing.io, .{ .admission = .{ .max_inflight = 1 } });
    errdefer f.stop() catch {};
    const io = f.io;

    // 10. one reply holds the only admission slot
    var held: d.Background = .{ .a = a, .io = io, .port = f.port, .body = try d.chatBody(a, "@hold", true) };
    const bt = try std.Thread.spawn(.{}, d.Background.run, .{&held});
    d.waitActive(io, &f.slow, 1);
    var c: Client = undefined;
    try c.open(a, io, f.port, null);
    const refused = try c.turn(a, try create(a, "\"stream_id\":\"main\",\"input\":\"x\",\"store\":false"));
    try expectError(refused, 429, "max_inflight_exceeded");
    try testing.expectEqualStrings("rate_limit_error", client.last(refused).get("error").?.strField("type").?);
    try testing.expectEqualStrings("main", client.last(refused).strField("stream_id").?);
    f.slow.release.store(true, .release);
    bt.join();
    try testing.expect(!held.failed);
    try expectCompleted(try c.turn(a, try create(a, "\"stream_id\":\"main\",\"input\":\"x\",\"store\":false")));
    c.close();
    try f.stop();
}
