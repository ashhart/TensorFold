//! Responses WebSocket mode's bounds: browser origins meet the HTTP guard, and the socket cache stays in memory.
const std = @import("std");
const json = @import("json");
const ws = @import("responses_ws.zig");
const client = @import("responses_ws_client_test.zig");
const Fixture = @import("responses_ws_test.zig").Fixture;
const Client = client.Client;
const Allocator = std.mem.Allocator;
const testing = std.testing;

fn create(a: Allocator, fields: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"type\":\"response.create\",{s}}}", .{fields});
}

/// The upgrade's status with these header lines; the socket is closed again.
fn upgrade(a: Allocator, io: std.Io, port: u16, headers: []const u8) !u16 {
    var c: Client = undefined;
    try c.openWith(a, io, port, headers);
    defer c.close();
    return c.status;
}

/// Entries and bytes of the server's stored responses.
fn stored(srv: anytype, io: std.Io) [2]usize {
    srv.store.mutex.lockUncancelable(io);
    defer srv.store.mutex.unlock(io);
    return .{ srv.store.entries.items.len, srv.store.bytes };
}

test "Responses socket: a cross-origin or foreign-host upgrade is refused before the 101, as HTTP is" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(testing.allocator, testing.io, .{});
    errdefer f.stop() catch {};
    const io = f.io;
    const here = try std.fmt.allocPrint(a, "127.0.0.1:{d}", .{f.port});

    ws.test_server.store(null, .release);
    // a page on another site, a page on this host over another scheme or port, and a rebound foreign Host
    try testing.expectEqual(@as(u16, 403), try upgrade(a, io, f.port, "Host: localhost\r\nOrigin: http://evil.example\r\n"));
    try testing.expectEqual(@as(u16, 403), try upgrade(a, io, f.port, try std.fmt.allocPrint(a, "Host: {s}\r\nOrigin: https://{s}\r\n", .{ here, here })));
    try testing.expectEqual(@as(u16, 403), try upgrade(a, io, f.port, try std.fmt.allocPrint(a, "Host: {s}\r\nOrigin: http://127.0.0.1:1\r\n", .{here})));
    try testing.expectEqual(@as(u16, 403), try upgrade(a, io, f.port, "Host: evil.example\r\n"));
    try testing.expectEqual(@as(u16, 403), try upgrade(a, io, f.port, "Host: localhost\r\nOrigin: http://evil.example\r\nOrigin: http://localhost\r\n"));
    // none of them reached the socket session
    try testing.expect(ws.test_server.load(.acquire) == null);
    // the same origin, and no Origin at all (a non-browser client), open the socket
    try testing.expectEqual(@as(u16, 101), try upgrade(a, io, f.port, try std.fmt.allocPrint(a, "Host: {s}\r\nOrigin: http://{s}\r\n", .{ here, here })));
    try testing.expectEqual(@as(u16, 101), try upgrade(a, io, f.port, "Host: localhost\r\n"));
    try testing.expect(ws.test_server.load(.acquire) != null);
    try f.stop();
}

test "Responses socket: a store:false chain never reaches the stored responses, and a close frees its cache bytes" {
    const log_level = testing.log_level;
    testing.log_level = .err;
    defer testing.log_level = log_level;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fixture = undefined;
    try f.start(testing.allocator, testing.io, .{});
    errdefer f.stop() catch {};
    const io = f.io;

    ws.test_server.store(null, .release);
    var c: Client = undefined;
    try c.open(a, io, f.port, null);
    try testing.expectEqual(@as(u16, 101), c.status);
    const srv = ws.test_server.load(.acquire).?;
    const before = stored(srv, io);
    try testing.expectEqual(@as(usize, 0), srv.ws_cache_bytes.load(.acquire));
    // two turns on one lane and a fork of the second on another, all store:false, continued from the socket cache
    const first = client.final(try c.turn(a, try create(a, "\"input\":\"one\",\"store\":false"))).strField("id").?;
    const second = client.final(try c.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"two\",\"store\":false,\"previous_response_id\":\"{s}\"", .{first})))).strField("id").?;
    const fork = client.final(try c.turn(a, try create(a, try std.fmt.allocPrint(a, "\"stream_id\":\"side\",\"input\":\"f\",\"store\":false,\"previous_response_id\":\"{s}\"", .{second})))).strField("id").?;
    try testing.expectEqualDeep(before, stored(srv, io));
    for ([_][]const u8{ first, second, fork }) |id| {
        const got = try client.http(a, io, f.port, "GET", try std.fmt.allocPrint(a, "/v1/responses/{s}", .{id}), null, "");
        try testing.expectEqual(@as(u16, 404), got.status);
    }
    try testing.expect(srv.ws_cache_bytes.load(.acquire) > 0);
    // the close frees both lanes' entries, and their bytes leave the server-wide budget
    c.close();
    var waited: u32 = 0;
    while (srv.ws_cache_bytes.load(.acquire) != 0 and waited < 500) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
    try testing.expectEqual(@as(usize, 0), srv.ws_cache_bytes.load(.acquire));
    // the cache died with its socket: a new socket cannot continue from it
    var next: Client = undefined;
    try next.open(a, io, f.port, null);
    const miss = try next.turn(a, try create(a, try std.fmt.allocPrint(a, "\"input\":\"x\",\"store\":false,\"previous_response_id\":\"{s}\"", .{second})));
    try testing.expectEqualStrings("previous_response_not_found", client.code(miss));
    next.close();
    try testing.expectEqualDeep(before, stored(srv, io));
    try f.stop();
}
