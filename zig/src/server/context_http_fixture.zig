//! Drive a buffered decision route and capture its HTTP400 body without sockets.
const std = @import("std");
const json = @import("json");
const Conn = @import("http_conn.zig").Conn;
const Server = @import("server.zig").Server;
const A = std.mem.Allocator;
const Wire = struct {
    a: A,
    bytes: std.ArrayList(u8) = .empty,
    fn write(p: *anyopaque, bytes: []const u8) error{Closed}!void {
        const w: *@This() = @ptrCast(@alignCast(p));
        w.bytes.appendSlice(w.a, bytes) catch return error.Closed;
    }
};

pub fn call(srv: *Server, a: A, handler: *const fn (*Server, *Conn, A) void, body: []const u8) !json.Value {
    var wire: Wire = .{ .a = a };
    var headers = [_]@import("http_conn.zig").Header{.{ .name = "Content-Length", .value = try std.fmt.allocPrint(a, "{d}", .{body.len}) }};
    var c: Conn = .{ .fd = -1, .peer = "", .buf = try a.dupe(u8, body), .end = body.len, .gpa = a, .version = "HTTP/1.1", .method = "POST", .headers = &headers, .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    handler(srv, &c, a);
    try std.testing.expect(std.mem.startsWith(u8, wire.bytes.items, "HTTP/1.1 400 "));
    const at = std.mem.indexOf(u8, wire.bytes.items, "\r\n\r\n") orelse return error.MissingHttpHead;
    return (try json.parse(a, wire.bytes.items[at + 4 ..])).ok;
}
