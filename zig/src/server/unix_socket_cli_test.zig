//! ``--unix-socket`` replaces TCP: the TCP flags are refused beside it, and Windows refuses it at parse time.
const std = @import("std");
const builtin = @import("builtin");
const cli = @import("cli.zig");

test "--unix-socket is refused with --host or --port, and on Windows" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    if (builtin.os.tag == .windows) {
        try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--unix-socket", "s.sock" }, &u));
        try std.testing.expectEqualStrings("argument --unix-socket: not served on Windows", u.message);
        return;
    }
    try std.testing.expectEqualStrings("/tmp/s.sock", (try cli.parse(a, &.{ "m", "--unix-socket", "/tmp/s.sock" }, &u)).unix_socket.?);
    try std.testing.expect((try cli.parse(a, &.{"m"}, &u)).unix_socket == null);
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--unix-socket", "s.sock", "--host", "localhost" }, &u));
    try std.testing.expectEqualStrings("argument --unix-socket: not allowed with --host", u.message);
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--port=8080", "--unix-socket=s.sock" }, &u));
    try std.testing.expectEqualStrings("argument --unix-socket: not allowed with --port", u.message);
}
