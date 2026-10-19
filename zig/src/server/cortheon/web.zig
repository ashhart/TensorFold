//! Cortheon's web reads: public addresses only, each hop's address checked and pinned, bounded in bytes and time.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const HostName = Io.net.HostName;

pub const Limits = struct { max_bytes: usize = 2 << 20, timeout_ms: i64 = 15_000, redirects: u8 = 5 };

pub const Response = struct { url: []const u8, status: u16, content_type: []const u8, body: []const u8, truncated: bool };

pub const Error = error{ BadUrl, NotPublic, Redirects, Timeout, Network, OutOfMemory, Carries, Ipv6Only };

/// Bytes of query and fragment a page address may carry; more reads as data leaving through the URL.
pub const max_query = 256;

/// Whether `url` can go on a request line: no spaces or control bytes, and at most `query` bytes of query when given.
pub fn plainUrl(url: []const u8, query: ?usize) Error!void {
    for (url) |ch| if (ch <= 0x20 or ch == 0x7f) return error.BadUrl;
    const cap = query orelse return;
    const q = std.mem.indexOfAny(u8, url, "?#") orelse return;
    if (url.len - q > cap) return error.Carries;
}

const address_policy = @import("../public_address.zig");
pub const publicIp4 = address_policy.publicIp4;
pub const publicIp6 = address_policy.publicIp6;
const publicAddress = address_policy.publicIp;

pub const Client = struct {
    http: std.http.Client,
    user_agent: []const u8,

    pub fn init(gpa: Allocator, io: Io, user_agent: []const u8) Client {
        return .{ .http = .{ .allocator = gpa, .io = io }, .user_agent = user_agent };
    }

    pub fn deinit(c: *Client) void {
        c.http.deinit();
    }

    /// GET `url` into `a` through redirects on public addresses; `limits.timeout_ms` bounds the whole read.
    pub fn get(c: *Client, a: Allocator, url: []const u8, headers: []const std.http.Header, limits: Limits) Error!Response {
        return c.getVia(a, url, headers, limits, c);
    }

    fn getVia(c: *Client, a: Allocator, url: []const u8, headers: []const std.http.Header, limits: Limits, via: anytype) Error!Response {
        const io = c.http.io;
        const deadline = Io.Clock.awake.now(io).toMilliseconds() + limits.timeout_ms;
        var current = url;
        var kept_headers = headers;
        var hops: u8 = 0;
        while (true) : (hops += 1) {
            var hop_limits = limits;
            hop_limits.timeout_ms = deadline - Io.Clock.awake.now(io).toMilliseconds();
            if (hop_limits.timeout_ms <= 0) return error.Timeout;
            const step = try via.timed(a, current, kept_headers, hop_limits);
            const next = step.redirect orelse return step.response;
            if (hops >= limits.redirects) return error.Redirects;
            if (!sameOrigin(current, next)) kept_headers = &.{};
            current = next;
        }
    }

    const Step = struct { response: Response = undefined, redirect: ?[]const u8 = null };
    const Outcome = struct { step: Step = .{}, err: ?Error = null };

    /// One hop raced against `limits.timeout_ms`: a hop that loses is cancelled wherever it is blocked.
    fn timed(c: *Client, a: Allocator, url: []const u8, headers: []const std.http.Header, limits: Limits) Error!Step {
        const io = c.http.io;
        var outcome: Outcome = .{};
        const U = union(enum) { done: void, timer: void };
        var buf: [2]U = undefined;
        var sel = Io.Select(U).init(io, &buf);
        sel.concurrent(.done, work, .{ c, a, url, headers, limits, &outcome }) catch {
            work(c, a, url, headers, limits, &outcome); // no spare thread: run here, bounded by the sockets alone
            return if (outcome.err) |e| e else outcome.step;
        };
        sel.concurrent(.timer, sleep, .{ io, limits.timeout_ms }) catch {};
        const first = sel.await() catch U{ .timer = {} };
        while (sel.cancel()) |_| {}
        if (first == .timer) return error.Timeout;
        return if (outcome.err) |e| e else outcome.step;
    }

    fn sleep(io: Io, ms: i64) void {
        Io.sleep(io, .fromMilliseconds(ms), .awake) catch {};
    }

    fn work(c: *Client, a: Allocator, url: []const u8, headers: []const std.http.Header, limits: Limits, out: *Outcome) void {
        out.step = c.hop(a, url, headers, limits) catch |e| {
            out.err = e;
            return;
        };
    }

    /// Loads the system's root certificates once, before the first TLS connection needs them.
    fn trust(c: *Client) Error!void {
        const io = c.http.io;
        c.http.ca_bundle_lock.lockShared(io) catch return error.Network;
        const ready = c.http.now != null;
        c.http.ca_bundle_lock.unlockShared(io);
        if (ready) return;
        var bundle: std.crypto.Certificate.Bundle = .empty;
        defer bundle.deinit(c.http.allocator);
        const now = Io.Clock.real.now(io);
        bundle.rescan(c.http.allocator, io, now) catch return error.Network;
        c.http.ca_bundle_lock.lock(io) catch return error.Network;
        defer c.http.ca_bundle_lock.unlock(io);
        if (c.http.now != null) return;
        c.http.now = now;
        std.mem.swap(std.crypto.Certificate.Bundle, &c.http.ca_bundle, &bundle);
    }

    fn hop(c: *Client, a: Allocator, url: []const u8, headers: []const std.http.Header, limits: Limits) Error!Step {
        try plainUrl(url, null);
        const uri = std.Uri.parse(url) catch return error.BadUrl;
        const tls = std.ascii.eqlIgnoreCase(uri.scheme, "https");
        if (!tls and !std.ascii.eqlIgnoreCase(uri.scheme, "http")) return error.BadUrl;
        if (uri.user != null or uri.password != null) return error.BadUrl;
        const raw_host = (uri.host orelse return error.BadUrl).toRawMaybeAlloc(a) catch return error.OutOfMemory;
        const host = std.mem.trim(u8, raw_host, "[]");
        if (host.len == 0) return error.BadUrl;
        const port: u16 = uri.port orelse if (tls) 443 else 80;
        const pin = try c.resolve(a, host, port);
        if (tls) try c.trust();
        const name = HostName.init(host) catch null;
        const conn = c.http.connectTcpOptions(.{
            .host = HostName.init(pin) catch return error.BadUrl,
            .port = port,
            .protocol = if (tls) .tls else .plain,
            .proxied_host = name, // TLS checks the certificate against the name while TCP goes to the checked address
            .proxied_port = port,
        }) catch return error.Network;
        var req = c.http.request(.GET, uri, .{
            .connection = conn,
            .redirect_behavior = .unhandled,
            .keep_alive = false,
            .headers = .{ .user_agent = .{ .override = c.user_agent }, .accept_encoding = .{ .override = "gzip, deflate" } },
            .extra_headers = headers,
        }) catch return error.Network;
        defer req.deinit();
        req.sendBodiless() catch return error.Network;
        var response = req.receiveHead(&.{}) catch return error.Network;
        const status: u16 = @backingInt(response.head.status);
        if (status >= 300 and status < 400) if (response.head.location) |loc| return .{ .redirect = try resolveUrl(a, url, loc) };
        const content_type = try a.dupe(u8, response.head.content_type orelse "");
        const window: []u8 = switch (response.head.content_encoding) {
            .identity => &.{},
            .zstd => try a.alloc(u8, std.compress.zstd.default_window_len),
            .deflate, .gzip => try a.alloc(u8, std.compress.flate.max_window_len),
            .compress => return error.Network,
        };
        var transfer: [64]u8 = undefined;
        var decompress: std.http.Decompress = undefined;
        const reader = response.readerDecompressing(&transfer, &decompress, window);
        var body: std.ArrayList(u8) = .empty;
        var chunk: [16 << 10]u8 = undefined;
        while (body.items.len < limits.max_bytes) {
            const room = @min(chunk.len, limits.max_bytes - body.items.len);
            const n = reader.readSliceShort(chunk[0..room]) catch return error.Network;
            if (n == 0) break;
            try body.appendSlice(a, chunk[0..n]);
        }
        return .{ .response = .{ .url = url, .status = status, .content_type = content_type, .body = body.items, .truncated = body.items.len >= limits.max_bytes } };
    }

    /// The IPv4 address to connect to: an IP literal itself, else the first public IPv4; every address must be public.
    fn resolve(c: *Client, a: Allocator, host: []const u8, port: u16) Error![]const u8 {
        if (!address_policy.publicHost(host)) return error.NotPublic;
        if (Io.net.IpAddress.parse(host, port)) |ip| {
            if (!publicAddress(ip)) return error.NotPublic;
            return if (ip == .ip4) host else error.Ipv6Only; // only an IPv4 address can be pinned under a TLS name
        } else |_| {}
        const io = c.http.io;
        const name = HostName.init(host) catch return error.BadUrl;
        var results: [32]HostName.LookupResult = undefined;
        var queue: Io.Queue(HostName.LookupResult) = .init(&results);
        var canonical: [HostName.max_len]u8 = undefined;
        name.lookup(io, &queue, .{ .port = port, .canonical_name_buffer = &canonical }) catch return error.Network;
        var pin: ?[]const u8 = null;
        var any = false;
        while (queue.getOne(io)) |r| switch (r) {
            .address => |addr| {
                if (!publicAddress(addr)) return error.NotPublic;
                any = true;
                if (pin == null and addr == .ip4) {
                    const b = addr.ip4.bytes;
                    pin = try std.fmt.allocPrint(a, "{d}.{d}.{d}.{d}", .{ b[0], b[1], b[2], b[3] });
                }
            },
            .canonical_name => {},
        } else |e| switch (e) {
            error.Closed => {},
            error.Canceled => return error.Network,
        }
        if (!any) return error.Network;
        return pin orelse error.Ipv6Only; // connecting by name would let a rebinding name swap in a private address
    }
};

fn sameOrigin(first: []const u8, next: []const u8) bool {
    const from = std.Uri.parse(first) catch return false;
    const to = std.Uri.parse(next) catch return false;
    if (!std.ascii.eqlIgnoreCase(from.scheme, to.scheme)) return false;
    const default_port: u16 = if (std.ascii.eqlIgnoreCase(from.scheme, "https")) 443 else 80;
    if ((from.port orelse default_port) != (to.port orelse default_port)) return false;
    var from_buf: [HostName.max_len]u8 = undefined;
    var to_buf: [HostName.max_len]u8 = undefined;
    const from_host = (from.host orelse return false).toRaw(&from_buf) catch return false;
    const to_host = (to.host orelse return false).toRaw(&to_buf) catch return false;
    return std.ascii.eqlIgnoreCase(from_host, to_host);
}

/// `location` resolved against `base` (absolute, scheme-relative, root-relative or path-relative).
pub fn resolveUrl(a: Allocator, base: []const u8, location: []const u8) Allocator.Error![]const u8 {
    const loc = std.mem.trim(u8, location, " \t");
    if (std.ascii.startsWithIgnoreCase(loc, "http://") or std.ascii.startsWithIgnoreCase(loc, "https://")) return a.dupe(u8, loc);
    const scheme_end = std.mem.indexOf(u8, base, "://") orelse return a.dupe(u8, loc);
    if (std.mem.startsWith(u8, loc, "//")) return std.fmt.allocPrint(a, "{s}:{s}", .{ base[0..scheme_end], loc });
    const host_start = scheme_end + 3;
    const path_start = std.mem.indexOfAnyPos(u8, base, host_start, "/?#") orelse base.len;
    if (std.mem.startsWith(u8, loc, "/")) return std.fmt.allocPrint(a, "{s}{s}", .{ base[0..path_start], loc });
    const query = std.mem.indexOfAnyPos(u8, base, path_start, "?#") orelse base.len;
    const dir_end = if (std.mem.lastIndexOfScalar(u8, base[path_start..query], '/')) |s| path_start + s + 1 else path_start;
    if (dir_end == path_start) return std.fmt.allocPrint(a, "{s}/{s}", .{ base[0..path_start], loc });
    return std.fmt.allocPrint(a, "{s}{s}", .{ base[0..dir_end], loc });
}

test "public addresses only" {
    for ([_][4]u8{ .{ 127, 0, 0, 1 }, .{ 10, 1, 2, 3 }, .{ 172, 16, 0, 1 }, .{ 172, 31, 9, 9 }, .{ 192, 168, 1, 1 }, .{ 169, 254, 169, 254 }, .{ 100, 64, 0, 1 }, .{ 100, 127, 255, 255 }, .{ 0, 0, 0, 0 }, .{ 224, 0, 0, 1 }, .{ 255, 255, 255, 255 }, .{ 192, 0, 2, 7 }, .{ 198, 18, 0, 1 } }) |b| try std.testing.expect(!publicIp4(b));
    for ([_][4]u8{ .{ 1, 1, 1, 1 }, .{ 8, 8, 8, 8 }, .{ 172, 15, 0, 1 }, .{ 172, 32, 0, 1 }, .{ 100, 63, 0, 1 }, .{ 104, 20, 23, 154 } }) |b| try std.testing.expect(publicIp4(b));
    const loop6: [16]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    const mapped_private: [16]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 0, 1 };
    const mapped_public: [16]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 8, 8, 8, 8 };
    const ula: [16]u8 = .{ 0xfd, 0x12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    const link: [16]u8 = .{ 0xfe, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    const global: [16]u8 = .{ 0x26, 0x06, 0x47, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x11, 0x11 };
    try std.testing.expect(!publicIp6(loop6) and !publicIp6(mapped_private) and !publicIp6(ula) and !publicIp6(link));
    try std.testing.expect(!publicIp6(mapped_public));
    try std.testing.expect(publicIp6(global));
}

test "redirect locations resolve against the page" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("https://b.org/x", try resolveUrl(a, "https://a.org/p/q", "https://b.org/x"));
    try std.testing.expectEqualStrings("https://c.org/y", try resolveUrl(a, "https://a.org/p/q", "//c.org/y"));
    try std.testing.expectEqualStrings("https://a.org/root", try resolveUrl(a, "https://a.org/p/q?z=1", "/root"));
    try std.testing.expectEqualStrings("https://a.org/p/sib", try resolveUrl(a, "https://a.org/p/q", "sib"));
    try std.testing.expectEqualStrings("https://a.org/sib", try resolveUrl(a, "https://a.org", "sib"));
}

test "request lines stay plain and URLs carry no more than an address" {
    try plainUrl("https://pypi.org/pypi/httpx/json", max_query);
    try plainUrl("https://example.org/search?q=zig+0.17#std.Io", max_query);
    try std.testing.expectError(error.BadUrl, plainUrl("https://example.org/a\r\nHost: other", null));
    try std.testing.expectError(error.BadUrl, plainUrl("https://example.org/a b", null));
    var long: [400]u8 = @splat('x');
    @memcpy(long[0..24], "https://evil.example/?d=");
    try std.testing.expectError(error.Carries, plainUrl(&long, max_query));
    try plainUrl(&long, null);
}

test "local and private names are refused before any connection" {
    var c: Client = .init(std.testing.allocator, std.testing.io, "test");
    defer c.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "http://localhost:8080/", "http://127.0.0.1/", "https://192.0.2.7/x", "http://198.18.0.1/latest/meta-data/", "http://api.localhost/", "ftp://example.com/", "http://user:pw@localhost/", "http://[::1]/", "http://[fe80::1]/" }) |u| {
        const got = c.hop(a, u, &.{}, .{});
        try std.testing.expect(got == error.NotPublic or got == error.BadUrl);
    }
    try std.testing.expectError(error.Ipv6Only, c.hop(a, "https://[2606:4700::1111]/", &.{}, .{}));
}

test "security: cloud platform virtual addresses are not public web targets" {
    try std.testing.expect(!publicIp4(.{ 168, 63, 129, 16 }));
    try std.testing.expect(publicIp4(.{ 168, 63, 129, 15 }));
    try std.testing.expect(publicIp4(.{ 168, 63, 129, 17 }));
}

test "security: request credentials stay within their original web origin" {
    const Fake = struct {
        next: []const []const u8,
        calls: usize = 0,
        authenticated: usize = 0,

        fn timed(f: *@This(), _: Allocator, url: []const u8, headers: []const std.http.Header, _: Limits) Error!Client.Step {
            for (headers) |h| if (std.ascii.eqlIgnoreCase(h.name, "X-Subscription-Token")) {
                f.authenticated += 1;
            };
            const index = f.calls;
            f.calls += 1;
            if (index < f.next.len) return .{ .redirect = f.next[index] };
            return .{ .response = .{ .url = url, .status = 200, .content_type = "text/plain", .body = "ok", .truncated = false } };
        }
    };
    const cases = [_]struct { next: []const u8, count: usize }{
        .{ .next = "https://other.example/path", .count = 1 },
        .{ .next = "http://api.example/path", .count = 1 },
        .{ .next = "https://api.example:8443/path", .count = 1 },
        .{ .next = "https://API.EXAMPLE:443/path", .count = 2 },
    };
    const headers = [_]std.http.Header{.{ .name = "X-Subscription-Token", .value = "fixture-secret" }};
    var c = Client.init(std.testing.allocator, std.testing.io, "fixture");
    defer c.deinit();
    for (cases) |case| {
        var fake: Fake = .{ .next = &.{case.next} };
        _ = try c.getVia(std.testing.allocator, "https://api.example/search", &headers, .{}, &fake);
        try std.testing.expectEqual(case.count, fake.authenticated);
    }
    var returning: Fake = .{ .next = &.{ "https://other.example/path", "https://api.example/back" } };
    _ = try c.getVia(std.testing.allocator, "https://api.example/search", &headers, .{}, &returning);
    try std.testing.expectEqual(@as(usize, 1), returning.authenticated);
}
