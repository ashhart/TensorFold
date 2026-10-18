//! Browser policy and the real repeatable CLI inputs, without networking or a model.
const std = @import("std");
const guard = @import("request_guard.zig");
const Header = @import("http_conn.zig").Header;
const t = std.testing;
fn expect(p: guard.Policy, host: ?[]const u8, origin: ?[]const u8, wanted: ?guard.Reason) !void {
    var headers: [2]Header = undefined;
    var n: usize = 0;
    if (host) |v| {
        headers[n] = .{ .name = "Host", .value = v };
        n += 1;
    }
    if (origin) |v| {
        headers[n] = .{ .name = "Origin", .value = v };
        n += 1;
    }
    try t.expectEqual(wanted, p.blocked(headers[0..n]));
}
test "browser policy: loopback names accept any valid port and mapped loopback binds stay protected" {
    for ([_][]const u8{ "localhost", "LOCALHOST:8000", "127.0.0.1:1", "127.0.0.2:65535", "[::1]", "[0:0:0:0:0:0:0:1]:0", "[::ffff:127.0.0.1]:80" }) |host| try expect(.{}, host, null, null);
    const ip = try std.Io.net.IpAddress.parse("::ffff:127.0.0.1", 0);
    try t.expect(@import("auth.zig").loopbackAddress(ip));
}
test "browser policy: foreign missing duplicate and malformed hosts refuse on loopback" {
    try expect(.{}, null, null, .host);
    for ([_][]const u8{ "", "rebind.invalid", "localhost.evil.invalid", "127.0.0.1.evil", "0.0.0.0", "localhost:65536", "localhost:abc", "localhost@evil", "localhost:80/", "http://localhost", "[::1]@evil", "[]", "localhost,evil" }) |host| try expect(.{}, host, null, .host);
    try t.expectEqual(@as(?guard.Reason, .host), (guard.Policy{}).blocked(&.{ .{ .name = "Host", .value = "localhost" }, .{ .name = "hOsT", .value = "localhost" } }));
}
test "browser policy: same HTTP origins normalize case default ports and IPv6 spellings" {
    try expect(.{}, "localhost", "HTTP://LOCALHOST:80", null);
    try expect(.{}, "localhost:8000", "http://localhost:8000", null);
    try expect(.{}, "[::1]:80", "http://[0:0:0:0:0:0:0:1]", null);
    try expect(.{}, "127.0.0.1", null, null);
    for ([_][]const u8{ "null", "", "https://localhost", "http://localhost:81", "http://evil.invalid", "http://localhost/", "http://localhost@evil", "http://localhost http://evil.invalid" }) |origin| try expect(.{}, "localhost", origin, .origin);
}
test "browser policy: nonloopback binds skip host admission but keep origin validation" {
    const p: guard.Policy = .{ .loopback_bind = false };
    try expect(p, "rebind.invalid", null, null);
    try expect(p, "public.invalid:80", "http://public.invalid", null);
    try expect(p, "public.invalid", "http://evil.invalid", .origin);
    try expect(p, "public.invalid", "null", .origin);
    try expect(p, null, "http://public.invalid", .host);
}
test "browser policy: host and origin exceptions are independent and exact" {
    const p: guard.Policy = .{ .allowed_hosts = &.{ "trusted.one", "trusted.two", "[2001:db8::1]" }, .allowed_origins = &.{ "https://ui.one", "http://ui.two:3000" } };
    try expect(p, "TRUSTED.ONE:9000", null, null);
    try expect(p, "trusted.two", "http://trusted.two", null);
    try expect(p, "[2001:db8::1]", "http://[2001:0db8::1]", null);
    try expect(p, "localhost", "https://ui.one:443", null);
    try expect(p, "localhost", "http://ui.two:3000", null);
    try expect(p, "evil.invalid", "https://ui.one", .host);
    try expect(p, "trusted.one.evil", null, .host);
    try expect(p, "trusted.one", "https://ui.one.evil", .origin);
    try t.expectEqual(@as(?guard.Reason, .origin), p.blocked(&.{ .{ .name = "Host", .value = "localhost" }, .{ .name = "Origin", .value = "http://localhost" }, .{ .name = "oRiGiN", .value = "http://localhost" } }));
}
test "browser policy: repeatable CLI flags preserve both entries and existing aliases and keys" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const cli = @import("cli.zig");
    var usage: cli.Usage = .{};
    const args = try cli.parse(arena.allocator(), &.{ "m", "--alias", "alias", "--api-key", "fixture-key", "--allowed-host", "one.invalid", "--allowed-host=two.invalid", "--allowed-origin", "https://one.invalid", "--allowed-origin=http://two.invalid:8000" }, &usage);
    try t.expectEqualStrings("alias", args.alias[0]);
    try t.expectEqualStrings("fixture-key", args.api_key[0]);
    try t.expectEqual(@as(usize, 2), args.allowed_host.len);
    try t.expectEqualStrings("one.invalid", args.allowed_host[0]);
    try t.expectEqualStrings("two.invalid", args.allowed_host[1]);
    try t.expectEqual(@as(usize, 2), args.allowed_origin.len);
    for ([_][]const u8{ "null", "*", "https://user@one.invalid", "https://one.invalid/path" }) |origin| try t.expectError(error.Usage, cli.parse(arena.allocator(), &.{ "m", "--allowed-origin", origin }, &usage));
    for ([_][]const u8{ "", "*", "one.invalid:80", "http://one.invalid" }) |host| try t.expectError(error.Usage, cli.parse(arena.allocator(), &.{ "m", "--allowed-host", host }, &usage));
}

test "browser policy: malformed authorities and origin tuples never gain trust" {
    for ([_][]const u8{ "localhost.", "localhost..", "[::1%0]", "[127.0.0.1]", "::1", "localhost:", "localhost:+80", "localhost:-1", "localhost:80@evil", "localhost/", "localhost\\evil", "localhost?x", "localhost#x", "localhost%2e", "local host", "-localhost", "localhost-", "localhost..evil" }) |host| try expect(.{}, host, null, .host);
    for ([_][]const u8{ "file://localhost", "data:text/plain,x", "http://localhost.evil", "http://localhost#x", "http://localhost?x", "http://localhost:80/", "http://localhost,evil", "http://localhost\r\nX: y" }) |origin| try expect(.{}, "localhost", origin, .origin);
    try expect(.{}, "localhost", "http://127.0.0.1", .origin);
    const p: guard.Policy = .{ .allowed_hosts = &.{"trusted.invalid"}, .allowed_origins = &.{"https://ui.invalid:8443"} };
    try expect(p, "trusted.invalid", "https://ui.invalid:8444", .origin);
    try expect(p, "trusted.invalid", "http://ui.invalid:8443", .origin);
}
