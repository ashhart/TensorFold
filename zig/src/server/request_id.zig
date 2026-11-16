//! X-Request-Id: the client's own id when it sent one valid value, else the id the server holds the request by.
const std = @import("std");
const ids = @import("ids.zig");
const Conn = @import("http_conn.zig").Conn;

pub const header = "X-Request-Id";

/// The most characters a client's id may have.
pub const max_len = 128;

/// A client id the server echoes: 1 to 128 printable ASCII characters, none of them a space.
pub fn valid(value: []const u8) bool {
    if (value.len == 0 or value.len > max_len) return false;
    for (value) |c| if (c <= ' ' or c > '~') return false;
    return true;
}

/// The client's id: exactly one X-Request-Id header holding a valid value; none, several or an invalid one are none.
pub fn client(c: *const Conn) ?[]const u8 {
    if (c.headerCount(header) != 1) return null;
    const value = c.header(header).?;
    return if (valid(value)) value else null;
}

/// An id for a request the server names no other way: ``req_`` then 32 random lowercase hex digits, kept in ``buf``.
pub fn fresh(buf: *[36]u8) []const u8 {
    return ids.fill(buf, "req_");
}

test "request id: a client's value is 1 to 128 printable ASCII characters without a space" {
    const longest: [max_len]u8 = @splat('z');
    for ([_][]const u8{ "a", "client-trace-1", "req_0123", "!~\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}", &longest }) |ok|
        try std.testing.expect(valid(ok));
    const over: [max_len + 1]u8 = @splat('z');
    for ([_][]const u8{ "", &over, "a b", "a\tb", "\x7f", "caf\xe9", "caf\xc3\xa9", "a\x00" }) |bad|
        try std.testing.expect(!valid(bad));
    var buf: [36]u8 = undefined;
    const id = fresh(&buf);
    try std.testing.expectEqualStrings("req_", id[0..4]);
    try std.testing.expectEqual(@as(usize, 36), id.len);
    for (id[4..]) |c| try std.testing.expect(std.ascii.isDigit(c) or (c >= 'a' and c <= 'f'));
}
