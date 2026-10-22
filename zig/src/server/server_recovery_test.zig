//! Handler failures, transient SSE ownership and both fetchers' address admission without a network.
const std = @import("std");
const http = @import("http_conn.zig");
const Server = @import("server.zig").Server;
const sse = @import("sse.zig");
const a = std.testing.allocator;

const Writer = struct {
    calls: usize = 0,
    invalid: bool = false,
    closed: bool = false,
    fn write(ptr: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *Writer = @ptrCast(@alignCast(ptr));
        if (w.closed) return error.Closed;
        w.calls += 1;
        if (!std.mem.eql(u8, bytes, "data: \"delta\"\n\n") and
            !std.mem.eql(u8, bytes, "event: token\ndata: \"delta\"\n\n")) w.invalid = true;
    }
};

const Head = struct {
    fn now(_: *anyopaque) i64 {
        return 0;
    }
    fn read(_: *anyopaque, out: []u8, _: i32) http.IoError!usize {
        const request = "GET /missing HTTP/1.1\r\nHost: localhost\r\n\r\n";
        @memcpy(out[0..request.len], request);
        return request.len;
    }
};

test "Server recovery: a complete reply reuses the connection but does not complete the next request" {
    var srv: Server = undefined;
    var c = try http.Conn.init(a, -1, "fixture");
    defer c.deinit();
    c.method = "GET";
    c.path = "/missing";
    c.version = "HTTP/1.1";
    c.close = false;
    c.quiet_log = true;
    var writer: Writer = .{};
    c.write_hooks = .{ .ctx = &writer, .write = Writer.write };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    srv.dispatch(&c, arena.allocator());
    try std.testing.expect(c.response_complete and !c.close);
    c.read_hooks = .{ .ctx = &writer, .now = Head.now, .read = Head.read };
    try std.testing.expectEqual(http.Outcome.ready, try c.readRequest(arena.allocator(), true));
    try std.testing.expect(!c.response_complete);
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    srv.dispatch(&c, failing.allocator());
    try std.testing.expect(c.close);
}

test "Server recovery: a failed response head clears earlier completion before closing" {
    var counted = std.testing.FailingAllocator.init(a, .{});
    var c = try http.Conn.init(counted.allocator(), -1, "fixture");
    defer c.deinit();
    c.method = "GET";
    c.path = "/missing";
    c.version = "HTTP/1.1";
    c.close = false;
    c.response_complete = true;
    c.quiet_log = true;
    counted.fail_index = counted.alloc_index;
    try std.testing.expectError(error.OutOfMemory, c.startResponse(200, null));
    try std.testing.expect(!c.response_complete);
    var srv: Server = undefined;
    srv.dispatch(&c, counted.allocator());
    try std.testing.expect(c.close);
}

test "Server recovery: an allocation failure before a reply closes the keep-alive connection" {
    var srv: Server = undefined;
    var c = try http.Conn.init(a, -1, "fixture");
    defer c.deinit();
    c.method = "GET";
    c.path = "/missing";
    c.version = "HTTP/1.1";
    c.close = false;
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    srv.dispatch(&c, failing.allocator());
    try std.testing.expect(failing.has_induced_failure);
    try std.testing.expect(c.close);
}

test "Server recovery: SSE data and typed frames retain no per-event allocations" {
    var counted = std.testing.FailingAllocator.init(a, .{});
    var c = try http.Conn.init(counted.allocator(), -1, "fixture");
    defer c.deinit();
    var writer: Writer = .{};
    c.write_hooks = .{ .ctx = &writer, .write = Writer.write };
    var arena = std.heap.ArenaAllocator.init(counted.allocator());
    defer arena.deinit();
    const live = counted.allocated_bytes - counted.freed_bytes;
    for (0..1000) |_| {
        try sse.data(&c, arena.allocator(), .{ .string = "delta" });
        try sse.event(&c, arena.allocator(), "token", .{ .string = "delta" });
    }
    try std.testing.expectEqual(@as(usize, 2000), writer.calls);
    try std.testing.expect(!writer.invalid);
    try std.testing.expectEqual(live, counted.allocated_bytes - counted.freed_bytes);
}

test "Server recovery: failed SSE writes and allocation failures release frame storage" {
    var counted = std.testing.FailingAllocator.init(a, .{});
    var c = try http.Conn.init(counted.allocator(), -1, "fixture");
    defer c.deinit();
    var writer: Writer = .{ .closed = true };
    c.write_hooks = .{ .ctx = &writer, .write = Writer.write };
    const live = counted.allocated_bytes - counted.freed_bytes;
    try std.testing.expectError(error.Closed, sse.data(&c, a, .{ .string = "delta" }));
    try std.testing.expect(c.broken);
    try std.testing.expectEqual(live, counted.allocated_bytes - counted.freed_bytes);
    c.broken = false;
    counted.fail_index = counted.alloc_index;
    try std.testing.expectError(error.Closed, sse.event(&c, a, "token", .{ .string = "delta" }));
    try std.testing.expectEqual(live, counted.allocated_bytes - counted.freed_bytes);
}

test "Server recovery: both fetchers refuse cloud metadata and retain public address controls" {
    const media = @import("media_fetch.zig");
    const web = @import("cortheon/web.zig");
    for ([_][]const u8{ "169.254.169.254", "168.63.129.16", "100.100.100.200", "fd00:ec2::254", "::ffff:168.63.129.16", "64:ff9b::a83f:8110", "2002:a83f:8110::", "::ffff:8.8.8.8", "64:ff9b::808:808", "2001::1" }) |text| {
        const ip = try std.Io.net.IpAddress.parse(text, 443);
        try std.testing.expect(!media.publicIp(ip));
        try std.testing.expect(if (ip == .ip4) !web.publicIp4(ip.ip4.bytes) else !web.publicIp6(ip.ip6.bytes));
    }
    for ([_][]const u8{ "1.1.1.1", "8.8.8.8", "2606:4700:4700::1111" }) |text| {
        const ip = try std.Io.net.IpAddress.parse(text, 443);
        try std.testing.expect(media.publicIp(ip));
        try std.testing.expect(if (ip == .ip4) web.publicIp4(ip.ip4.bytes) else web.publicIp6(ip.ip6.bytes));
    }
}
