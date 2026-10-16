//! Which URLs cortheon_fetch may read: ones the conversation already showed, and one named package's registry JSON.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// URLs a request may read, in their compared form; at most `cap` of them.
pub const Seen = struct {
    set: std.StringHashMapUnmanaged(void) = .empty,

    const cap = 4096;

    pub fn add(s: *Seen, a: Allocator, url: []const u8) Allocator.Error!void {
        if (s.set.count() >= cap) return;
        const key = (try normal(a, url)) orelse return;
        try s.set.put(a, key, {});
    }

    /// Adds every http(s) URL written in `text`.
    pub fn addFrom(s: *Seen, a: Allocator, text: []const u8) Allocator.Error!void {
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, text, i, "http")) |at| {
            i = at + 4;
            if (!(std.ascii.startsWithIgnoreCase(text[at..], "http://") or std.ascii.startsWithIgnoreCase(text[at..], "https://"))) continue;
            var end = at;
            while (end < text.len and !stops(text[end])) end += 1;
            const url = std.mem.trimEnd(u8, text[at..end], ".,;:!?");
            try s.add(a, url);
            i = end;
        }
    }

    /// Whether `url` may be read: the conversation showed it, or it names one package's registry JSON.
    pub fn allows(s: *const Seen, a: Allocator, url: []const u8) bool {
        if (registry(url)) return true;
        const key = (normal(a, url) catch return false) orelse return false;
        return s.set.contains(key);
    }
};

fn stops(ch: u8) bool {
    return ch <= 0x20 or ch == 0x7f or std.mem.indexOfScalar(u8, "\"'<>()[]{}|\\^`", ch) != null;
}

/// `url` as compared: scheme and host lowercased, no default port, fragment or trailing slash; null if not http(s).
pub fn normal(a: Allocator, url: []const u8) Allocator.Error!?[]const u8 {
    const sep = std.mem.indexOf(u8, url, "://") orelse return null;
    const scheme = url[0..sep];
    const https = std.ascii.eqlIgnoreCase(scheme, "https");
    if (!https and !std.ascii.eqlIgnoreCase(scheme, "http")) return null;
    const rest = url[sep + 3 ..];
    const host_end = std.mem.indexOfAny(u8, rest, "/?#") orelse rest.len;
    var host = rest[0..host_end];
    if (host.len == 0) return null;
    if (std.mem.endsWith(u8, host, if (https) ":443" else ":80")) host = host[0 .. host.len - @as(usize, if (https) 4 else 3)];
    var tail = rest[host_end..];
    if (std.mem.indexOfScalar(u8, tail, '#')) |f| tail = tail[0..f];
    if (tail.len > 1 and tail[tail.len - 1] == '/' and std.mem.indexOfScalar(u8, tail, '?') == null) tail = tail[0 .. tail.len - 1];
    if (tail.len == 0) tail = "/";
    const out = try std.fmt.allocPrint(a, "{s}://{s}{s}", .{ if (https) "https" else "http", host, tail });
    for (out[0 .. out.len - tail.len]) |*ch| ch.* = std.ascii.toLower(ch.*);
    return out;
}

fn nameOk(s: []const u8, max: usize, extra: []const u8) bool {
    if (s.len == 0 or s.len > max) return false;
    for (s) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '.' or ch == '_' or ch == '-' or std.mem.indexOfScalar(u8, extra, ch) != null)) return false;
    return true;
}

fn packageName(s: []const u8) bool {
    return nameOk(s, 64, "");
}

fn versionName(s: []const u8) bool {
    return nameOk(s, 32, "+");
}

/// Whether `url` is one named package's JSON on PyPI, npm or crates.io, with no query; a name is 64 bytes at most.
pub fn registry(url: []const u8) bool {
    if (std.mem.indexOfAny(u8, url, "?#") != null) return false;
    if (std.mem.startsWith(u8, url, "https://pypi.org/pypi/")) {
        var parts = std.mem.splitScalar(u8, url["https://pypi.org/pypi/".len..], '/');
        const name = parts.next() orelse return false;
        const second = parts.next() orelse return false;
        if (std.mem.eql(u8, second, "json")) return packageName(name) and parts.next() == null;
        const third = parts.next() orelse return false;
        return packageName(name) and versionName(second) and std.mem.eql(u8, third, "json") and parts.next() == null;
    }
    if (std.mem.startsWith(u8, url, "https://registry.npmjs.org/")) {
        var rest = url["https://registry.npmjs.org/".len..];
        if (std.mem.startsWith(u8, rest, "@")) {
            const slash = std.mem.indexOfScalar(u8, rest, '/') orelse std.ascii.findIgnoreCase(rest, "%2f") orelse return false;
            if (!packageName(rest[1..slash])) return false;
            rest = rest[slash + @as(usize, if (rest[slash] == '/') 1 else 3) ..];
        }
        var parts = std.mem.splitScalar(u8, rest, '/');
        const name = parts.next() orelse return false;
        const after = parts.next();
        if (after) |v| if (!(std.mem.eql(u8, v, "latest") or versionName(v))) return false;
        return packageName(name) and parts.next() == null;
    }
    if (std.mem.startsWith(u8, url, "https://crates.io/api/v1/crates/")) {
        var parts = std.mem.splitScalar(u8, url["https://crates.io/api/v1/crates/".len..], '/');
        const name = parts.next() orelse return false;
        if (parts.next()) |v| if (!versionName(v)) return false;
        return packageName(name) and parts.next() == null;
    }
    return false;
}

test "urls compare by scheme and host case, default port, fragment and trailing slash" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("https://ziglang.org/download/0.17.0", (try normal(a, "HTTPS://ZigLang.org:443/download/0.17.0/#notes")).?);
    try std.testing.expectEqualStrings("http://a.org/", (try normal(a, "http://a.org")).?);
    try std.testing.expectEqualStrings("https://a.org/Path?Q=1", (try normal(a, "https://A.org/Path?Q=1")).?);
    try std.testing.expect((try normal(a, "ftp://a.org/x")) == null);
}

test "a request reads only what it was shown, and registry json for one package" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var s: Seen = .{};
    try s.addFrom(a, "See (https://docs.example/guide), and https://ziglang.org/news/. Not ftp://x.org or httpx.");
    try std.testing.expect(s.allows(a, "https://docs.example/guide"));
    try std.testing.expect(s.allows(a, "https://ziglang.org/news"));
    try std.testing.expect(!s.allows(a, "https://docs.example/guide?leak=secret"));
    try std.testing.expect(!s.allows(a, "https://docs.example/guide/secret"));
    try std.testing.expect(!s.allows(a, "https://evil.example/"));
    for ([_][]const u8{ "https://pypi.org/pypi/httpx/json", "https://pypi.org/pypi/httpx/0.28.1/json", "https://registry.npmjs.org/react", "https://registry.npmjs.org/react/latest", "https://registry.npmjs.org/@types/node/22.0.0", "https://registry.npmjs.org/@scope%2Fpkg", "https://crates.io/api/v1/crates/serde", "https://crates.io/api/v1/crates/serde/1.0.210" }) |u| try std.testing.expect(registry(u));
    for ([_][]const u8{ "https://pypi.org/pypi/httpx/json?x=1", "https://pypi.org/pypi/httpx", "https://pypi.org/simple/httpx/", "https://registry.npmjs.org/react/latest/extra", "https://registry.npmjs.org/a%2Fb", "https://crates.io/api/v1/crates/serde/1.0/download", "https://pypi.org/pypi/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/json", "http://pypi.org/pypi/httpx/json" }) |u| try std.testing.expect(!registry(u));
}
