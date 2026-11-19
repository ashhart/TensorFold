//! Stored and running Responses stay with their creating key over HTTP and WebSocket.
const std = @import("std");
const json = @import("json");
const client = @import("responses_ws_client_test.zig");
const Fixture = @import("responses_ws_test.zig").Fixture;
const cancels = @import("cancel_ids.zig");
const t = std.testing;
const A = std.mem.Allocator;
const key_a = "owner-test-a";
const key_b = "owner-test-b";

fn id(a: A, reply: client.Reply) ![]const u8 {
    try t.expectEqual(@as(u16, 200), reply.status);
    return (try json.parse(a, reply.body)).ok.strField("id").?;
}

fn path(a: A, rid: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "/v1/responses/{s}", .{rid});
}

fn next(a: A, rid: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"input\":\"next\",\"previous_response_id\":\"{s}\"}}", .{rid});
}

fn sameMiss(a: A, foreign: client.Reply, missing: client.Reply, rid: []const u8) !void {
    try t.expectEqual(missing.status, foreign.status);
    const one = try std.mem.replaceOwned(u8, a, foreign.body, rid, "<id>");
    const two = try std.mem.replaceOwned(u8, a, missing.body, "resp_missing", "<id>");
    try t.expectEqualStrings(two, one);
}

test "Responses owner socket: another full key cannot read, continue or delete a stored response" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(t.allocator, t.io, .{ .api_key = &.{ key_a, key_b } });
    errdefer f.stop() catch {};
    const rid = try id(a, try client.http(a, f.io, f.port, "POST", "/v1/responses", key_a, "{\"input\":\"private A turn\"}"));
    const stored = try path(a, rid);
    const missing = "/v1/responses/resp_missing";
    try sameMiss(a, try client.http(a, f.io, f.port, "GET", stored, key_b, ""), try client.http(a, f.io, f.port, "GET", missing, key_b, ""), rid);
    try sameMiss(a, try client.http(a, f.io, f.port, "POST", "/v1/responses", key_b, try next(a, rid)), try client.http(a, f.io, f.port, "POST", "/v1/responses", key_b, try next(a, "resp_missing")), rid);
    try sameMiss(a, try client.http(a, f.io, f.port, "DELETE", stored, key_b, ""), try client.http(a, f.io, f.port, "DELETE", missing, key_b, ""), rid);
    _ = try id(a, try client.http(a, f.io, f.port, "GET", stored, key_a, ""));
    const child = try id(a, try client.http(a, f.io, f.port, "POST", "/v1/responses", key_a, try next(a, rid)));
    try t.expectEqual(@as(u16, 404), (try client.http(a, f.io, f.port, "GET", try path(a, child), key_b, "")).status);
    try t.expectEqual(@as(u16, 200), (try client.http(a, f.io, f.port, "DELETE", stored, key_a, "")).status);
    try t.expectEqual(@as(u16, 404), (try client.http(a, f.io, f.port, "GET", stored, key_a, "")).status);
    try f.stop();
}

test "Responses owner socket: WebSocket stored history and warmup use the authenticated key" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(t.allocator, t.io, .{ .api_key = &.{ key_a, key_b } });
    errdefer f.stop() catch {};
    const rid = try id(a, try client.http(a, f.io, f.port, "POST", "/v1/responses", key_a, "{\"input\":\"A stored turn\"}"));
    var b: client.Client = undefined;
    try b.open(a, f.io, f.port, key_b);
    var b_open = true;
    defer if (b_open) b.close();
    const wrong = try b.turn(a, try std.fmt.allocPrint(a, "{{\"type\":\"response.create\",\"input\":\"x\",\"previous_response_id\":\"{s}\"}}", .{rid}));
    const unknown = try b.turn(a, "{\"type\":\"response.create\",\"input\":\"x\",\"previous_response_id\":\"resp_missing\"}");
    try t.expectEqualStrings("error", client.last(wrong).strField("type").?);
    try t.expectEqualStrings(client.code(unknown), client.code(wrong));
    try t.expectEqualStrings("previous_response_not_found", client.code(wrong));
    var own: client.Client = undefined;
    try own.open(a, f.io, f.port, key_a);
    var own_open = true;
    defer if (own_open) own.close();
    const warm = try own.turn(a, "{\"type\":\"response.create\",\"input\":\"owned warmup\",\"generate\":false}");
    const warm_id = client.final(warm).strField("id").?;
    _ = try id(a, try client.http(a, f.io, f.port, "GET", try path(a, warm_id), key_a, ""));
    try t.expectEqual(@as(u16, 404), (try client.http(a, f.io, f.port, "GET", try path(a, warm_id), key_b, "")).status);
    var again: client.Client = undefined;
    try again.open(a, f.io, f.port, key_a);
    var again_open = true;
    defer if (again_open) again.close();
    const continued = try again.turn(a, try std.fmt.allocPrint(a, "{{\"type\":\"response.create\",\"input\":\"next\",\"previous_response_id\":\"{s}\"}}", .{warm_id}));
    try t.expectEqualStrings("response.completed", client.last(continued).strField("type").?);
    again.close();
    again_open = false;
    own.close();
    own_open = false;
    b.close();
    b_open = false;
    try f.stop();
}

fn running(a: A, io: std.Io) ![]const u8 {
    const until = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (std.Io.Clock.awake.now(io).toNanoseconds() < until) {
        for (try cancels.registry.held(io, a)) |name| if (std.mem.startsWith(u8, name, "resp_")) return name;
        try std.Io.sleep(io, .fromMilliseconds(1), .awake);
    }
    return error.ResponseDidNotStart;
}

test "Responses owner socket: another full key cannot cancel a running response or read its cancelled result" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(t.allocator, t.io, .{ .api_key = &.{ key_a, key_b } });
    errdefer f.stop() catch {};
    var c: client.Client = undefined;
    try c.open(a, f.io, f.port, key_a);
    var c_open = true;
    defer if (c_open) c.close();
    try c.send(a, "{\"type\":\"response.create\",\"input\":\"hold @hang\"}");
    const rid = try running(a, f.io);
    const route = try std.fmt.allocPrint(a, "/v1/requests/{s}/cancel", .{rid});
    const missing = "/v1/requests/resp_missing/cancel";
    try sameMiss(a, try client.http(a, f.io, f.port, "POST", route, key_b, ""), try client.http(a, f.io, f.port, "POST", missing, key_b, ""), rid);
    try t.expectEqual(@as(u16, 200), (try client.http(a, f.io, f.port, "POST", route, key_a, "")).status);
    const done = try c.until(a);
    try t.expectEqualStrings("cancelled", client.final(done).strField("status").?);
    _ = try id(a, try client.http(a, f.io, f.port, "GET", try path(a, rid), key_a, ""));
    try t.expectEqual(@as(u16, 404), (try client.http(a, f.io, f.port, "GET", try path(a, rid), key_b, "")).status);
    c.close();
    c_open = false;
    try f.stop();
}

test "Responses owner socket: a single key and a keyless server retain read, continue and delete" {
    for ([_]?[]const u8{ key_a, null }) |key| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var f: Fixture = undefined;
        try f.start(t.allocator, t.io, .{ .api_key = if (key != null) &.{key_a} else &.{} });
        errdefer f.stop() catch {};
        const rid = try id(a, try client.http(a, f.io, f.port, "POST", "/v1/responses", key, "{\"input\":\"owned\"}"));
        _ = try id(a, try client.http(a, f.io, f.port, "GET", try path(a, rid), key, ""));
        _ = try id(a, try client.http(a, f.io, f.port, "POST", "/v1/responses", key, try next(a, rid)));
        try t.expectEqual(@as(u16, 200), (try client.http(a, f.io, f.port, "DELETE", try path(a, rid), key, "")).status);
        try f.stop();
    }
}
