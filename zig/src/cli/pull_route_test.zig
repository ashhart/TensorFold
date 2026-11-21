//! Pull's proxy variables, NO_PROXY matching and messages, without a network.
const std = @import("std");
const route = @import("pull_route.zig");

fn env(a: std.mem.Allocator, pairs: []const [2][]const u8) !std.process.Environ.Map {
    var m: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    for (pairs) |p| try m.put(p[0], p[1]);
    return m;
}

test "pull proxy: each scheme takes its own variable, lower case first, ALL_PROXY last" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try env(a, &.{ .{ "HTTPS_PROXY", "http://upper:1" }, .{ "https_proxy", "lower:2" }, .{ "ALL_PROXY", "all.example:3" } });
    const s = try route.fromEnv(a, &m);
    try std.testing.expectEqualStrings("lower", s.https.?.host);
    try std.testing.expectEqual(@as(u16, 2), s.https.?.port);
    try std.testing.expectEqualStrings("https_proxy", s.https.?.from);
    try std.testing.expectEqualStrings("all.example", s.http.?.host);
    try std.testing.expectEqualStrings(if (@import("builtin").os.tag == .windows) "all_proxy" else "ALL_PROXY", s.http.?.from);
    try std.testing.expectEqual(@as(u32, 30), s.connect_s);
    try std.testing.expectEqual(@as(u32, 60), s.stall_s);
    var none = try env(a, &.{});
    const plain = try route.fromEnv(a, &none);
    try std.testing.expect(plain.http == null and plain.https == null);
}

test "pull proxy: an https request picks HTTPS_PROXY and an http one HTTP_PROXY, hop by hop" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try env(a, &.{ .{ "HTTPS_PROXY", "http://secure.example:3128" }, .{ "HTTP_PROXY", "http://plain.example:8080" }, .{ "NO_PROXY", "localhost" } });
    const s = try route.fromEnv(a, &m);
    try std.testing.expectEqualStrings("secure.example", route.pick(s, true, "huggingface.co", 443).?.host);
    try std.testing.expectEqualStrings("secure.example", route.pick(s, true, "cas-bridge.xethub.hf.co", 443).?.host);
    try std.testing.expectEqualStrings("plain.example", route.pick(s, false, "huggingface.co", 80).?.host);
    try std.testing.expect(route.pick(s, true, "localhost", 443) == null);
}

test "pull proxy: credentials go to the proxy as Basic and never into what pull prints" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try env(a, &.{.{ "HTTPS_PROXY", "http://ana:s3cret@proxy.example:3128/" }});
    route.settings = try route.fromEnv(a, &m);
    defer route.settings = .{};
    const p = route.settings.https.?;
    try std.testing.expectEqualStrings("Basic YW5hOnMzY3JldA==", p.auth.?);
    var out: std.Io.Writer.Allocating = .init(a);
    try route.proxyInUse(&out.writer);
    const name = if (@import("builtin").os.tag == .windows) "https_proxy" else "HTTPS_PROXY";
    try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "https through http://proxy.example:3128 (from {s})", .{name}), out.written());
    const io = std.testing.io;
    route.fail(io, .{ .proxy_status = .{ .proxy = p, .status = 407 } });
    defer route.reset(io);
    try route.explain(io, &out.writer, error.AccessDenied);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "s3cret") == null and std.mem.indexOf(u8, out.written(), "ana") == null);
}

test "pull proxy: an unusable proxy or timeout names its variable, never the value" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { name: []const u8, value: []const u8, says: []const u8 }{
        .{ .name = "HTTPS_PROXY", .value = "socks5://ana:s3cret@proxy.example:1080", .says = "only http:// proxies" },
        .{ .name = "HTTP_PROXY", .value = "http://ana:s3cret@[bad", .says = "is not a proxy URL" },
        .{ .name = "TENSORFOLD_PULL_STALL_SECONDS", .value = "0", .says = "whole seconds" },
    };
    for (cases) |c| {
        var m = try env(a, &.{.{ c.name, c.value }});
        const err = if (route.fromEnv(a, &m)) |_| return error.TestUnexpectedResult else |e| e;
        var out: std.Io.Writer.Allocating = .init(a);
        try route.refusal(&out.writer, err);
        if (@import("builtin").os.tag == .windows) {
            try std.testing.expect(std.ascii.startsWithIgnoreCase(out.written(), c.name));
        } else try std.testing.expect(std.mem.startsWith(u8, out.written(), c.name));
        try std.testing.expect(std.mem.indexOf(u8, out.written(), c.says) != null);
        try std.testing.expect(std.mem.indexOf(u8, out.written(), "s3cret") == null);
    }
}

test "pull proxy: NO_PROXY matches exact hosts, suffixes, * and host:port" {
    const list = "localhost, .internal.example,*.corp.example  hf.example 10.0.0.7:8080,[::1]:9000";
    for ([_]struct { []const u8, u16 }{
        .{ "localhost", 443 },  .{ "a.internal.example", 443 }, .{ "internal.example", 80 }, .{ "b.corp.example", 443 },
        .{ "hf.example", 443 }, .{ "cdn.hf.example", 443 },     .{ "10.0.0.7", 8080 },       .{ "::1", 9000 },
        .{ "LOCALHOST", 80 },
    }) |c| try std.testing.expect(route.bypass(list, c[0], c[1]));
    for ([_]struct { []const u8, u16 }{
        .{ "huggingface.co", 443 }, .{ "notlocalhost", 443 }, .{ "xhf.example", 443 }, .{ "10.0.0.7", 443 }, .{ "::1", 443 },
    }) |c| try std.testing.expect(!route.bypass(list, c[0], c[1]));
    try std.testing.expect(route.bypass("*", "huggingface.co", 443));
    try std.testing.expect(!route.bypass("", "huggingface.co", 443));
}
test "pull proxy: NO_PROXY ignores one DNS root dot on candidates and entries" {
    const list = "ExAmPle.OrG., .Internal.:443,*.Corp.:8443,node.:9000";
    try std.testing.expect(route.bypass(list, "EXAMPLE.ORG.", 80));
    try std.testing.expect(route.bypass(list, "cdn.Example.Org.", 80));
    try std.testing.expect(route.bypass(list, "branch.internal.", 443));
    try std.testing.expect(route.bypass(list, "edge.corp.", 8443));
    try std.testing.expect(route.bypass(list, "NODE.", 9000));
    try std.testing.expect(!route.bypass(list, "node.", 9001));
    try std.testing.expect(!route.bypass(list, "example.org..", 80));
}

test "pull proxy: the timeouts read their own variables" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try env(a, &.{ .{ route.connect_name, " 5" }, .{ route.stall_name, "7" } });
    const s = try route.fromEnv(a, &m);
    try std.testing.expectEqual(@as(u32, 5), s.connect_s);
    try std.testing.expectEqual(@as(u32, 7), s.stall_s);
}
