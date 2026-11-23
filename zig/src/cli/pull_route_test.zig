//! Pull's proxy variables, NO_PROXY matching and messages, without a network.
const std = @import("std");
const route = @import("pull_route.zig");

fn env(a: std.mem.Allocator, pairs: []const [2][]const u8) !std.process.Environ.Map {
    var m: std.process.Environ.Map = .{ .array_hash_map = .empty, .allocator = a };
    for (pairs) |p| try m.put(p[0], p[1]);
    return m;
}

test "pull proxy: repeated configure failures name the current timeout or proxy variable" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const before = route.settings;
    defer route.settings = before;
    const refused_before = route.refused;
    defer route.refused = refused_before;
    for ([_][]const u8{ "http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", route.stall_name }) |name| {
        var first = try env(a, &.{.{ route.connect_name, "0" }});
        try std.testing.expectError(error.InvalidEnvironment, route.configure(a, &first));
        try std.testing.expectEqualStrings(route.connect_name, route.refused);
        const timeout = std.mem.eql(u8, name, route.stall_name);
        var second = try env(a, &.{.{ name, if (timeout) "0" else "http://user:secret@[bad" }});
        try std.testing.expectError(if (timeout) error.InvalidEnvironment else error.ProxyUrl, route.configure(a, &second));
        try std.testing.expect(std.ascii.eqlIgnoreCase(name, route.refused));
        var out: std.Io.Writer.Allocating = .init(a);
        try route.refusal(&out.writer, if (timeout) error.InvalidEnvironment else error.ProxyUrl);
        try std.testing.expect(std.ascii.startsWithIgnoreCase(out.written(), name));
        try std.testing.expect(std.mem.indexOf(u8, out.written(), "secret") == null);
    }
}

test "pull proxy: successful configure clears a preceding refusal even without a proxy" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const before = route.settings;
    defer route.settings = before;
    const refused_before = route.refused;
    defer route.refused = refused_before;
    var invalid = try env(a, &.{.{ route.stall_name, "bad" }});
    var empty = try env(a, &.{});
    var skipped = try env(a, &.{.{ "HTTPS_PROXY", "socks5://proxy.example:1080" }});
    var valid = try env(a, &.{.{ "HTTP_PROXY", "http://proxy.example:3128" }});
    for ([_]?*const std.process.Environ.Map{ null, &empty, &skipped, &valid }) |next| {
        try std.testing.expectError(error.InvalidEnvironment, route.configure(a, &invalid));
        try std.testing.expectEqualStrings(route.stall_name, route.refused);
        try route.configure(a, next);
        try std.testing.expectEqualStrings("", route.refused);
    }
    try std.testing.expectEqual(@as(u32, 30), route.settings.connect_s);
    try std.testing.expectEqual(@as(u32, 60), route.settings.stall_s);
    try std.testing.expectEqual(@as(u16, 3128), route.settings.http.?.port);
}

test "pull proxy: every proxy parse allocation failure names the variable before or after credentials" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const before = route.settings;
    defer route.settings = before;
    const refused_before = route.refused;
    defer route.refused = refused_before;
    const cases = [_]struct { []const u8, usize }{
        .{ "proxy.example:3128", 0 },
        .{ "proxy.example:3128", 1 },
        .{ "http://proxy.example:3128", 0 },
        .{ "http://user:secret@proxy.example:3128", 0 },
        .{ "http://user:secret@proxy.example:3128", 1 },
    };
    for ([_][]const u8{ "http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY" }) |name| for (cases) |c| {
        var first = try env(a, &.{.{ route.connect_name, "0" }});
        try std.testing.expectError(error.InvalidEnvironment, route.configure(a, &first));
        var second = try env(a, &.{.{ name, c[0] }});
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = c[1] });
        try std.testing.expectError(error.OutOfMemory, route.configure(failing.allocator(), &second));
        try std.testing.expect(std.ascii.eqlIgnoreCase(name, route.refused));
        var out: std.Io.Writer.Allocating = .init(a);
        try route.refusal(&out.writer, error.OutOfMemory);
        try std.testing.expect(std.ascii.startsWithIgnoreCase(out.written(), name));
        try std.testing.expect(std.mem.indexOf(u8, out.written(), "secret") == null);
    };
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

test "pull proxy: a socks or https:// proxy is skipped by name, never refused (pull connects directly, as 1.3.3 did)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][2][]const u8{ .{ "HTTPS_PROXY", "socks5://ana:s3cret@proxy.example:1080" }, .{ "ALL_PROXY", "https://proxy.example:443" } }) |c| {
        var m = try env(a, &.{.{ c[0], c[1] }});
        const s = try route.fromEnv(a, &m);
        try std.testing.expect(s.https == null);
        try std.testing.expect(std.ascii.eqlIgnoreCase(s.skipped, c[0]));
        try std.testing.expect(std.mem.indexOf(u8, s.skipped, "s3cret") == null);
    }
    var both = try env(a, &.{ .{ "HTTPS_PROXY", "socks5://proxy.example:1080" }, .{ "HTTP_PROXY", "http://proxy.example:3128" } });
    const s = try route.fromEnv(a, &both);
    try std.testing.expect(s.https == null and s.http != null and s.http.?.port == 3128);
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

test "pull route: more distinct addresses than the table holds never fails a pull; the newest keeps its deadline" {
    const io = std.testing.io;
    const deadline: std.Io.Clock.Timestamp = .{ .raw = .zero, .clock = .awake };
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        const address: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 10, 99, @intCast(i / 256), @intCast(i % 256) }, .port = 443 } };
        try route.rememberAddress(io, address, deadline);
    }
    const newest: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 10, 99, 0, 199 }, .port = 443 } };
    try std.testing.expect(route.addressDeadline(io, &newest) != null);
}
