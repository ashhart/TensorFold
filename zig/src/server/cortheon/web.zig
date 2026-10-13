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

/// Whether an IPv4 address is public: not private, loopback, link-local, shared, reserved or multicast.
pub fn publicIp4(b: [4]u8) bool {
    return !(b[0] == 0 or b[0] == 10 or b[0] == 127 or b[0] >= 224 or
        (b[0] == 100 and b[1] >= 64 and b[1] < 128) or
        (b[0] == 169 and b[1] == 254) or
        (b[0] == 172 and b[1] >= 16 and b[1] < 32) or
        (b[0] == 192 and b[1] == 0 and (b[2] == 0 or b[2] == 2)) or
        (b[0] == 192 and b[1] == 168) or
        (b[0] == 198 and (b[1] == 18 or b[1] == 19)) or
        (b[0] == 198 and b[1] == 51 and b[2] == 100) or
        (b[0] == 203 and b[1] == 0 and b[2] == 113));
}

/// Whether an IPv6 address is public: global unicast, and an embedded IPv4 address must be public too.
pub fn publicIp6(b: [16]u8) bool {
    const zero10 = std.mem.allEqual(u8, b[0..10], 0);
    if (zero10 and b[10] == 0xff and b[11] == 0xff) return publicIp4(b[12..16].*); // IPv4-mapped
    if (zero10 and b[10] == 0 and b[11] == 0) return false; // unspecified, loopback and IPv4-compatible
    if ((b[0] & 0xfe) == 0xfc or b[0] == 0xff) return false; // unique local, multicast
    if (b[0] == 0xfe and (b[1] & 0xc0) == 0x80) return false; // link-local
    if (b[0] == 0x20 and b[1] == 0x01 and b[2] == 0x0d and b[3] == 0xb8) return false; // documentation
    if (b[0] == 0x00 and b[1] == 0x64 and b[2] == 0xff and b[3] == 0x9b) return publicIp4(b[12..16].*); // NAT64
    return (b[0] & 0xe0) == 0x20; // 2000::/3
}

fn publicAddress(addr: Io.net.IpAddress) bool {
    return switch (addr) {
        .ip4 => |x| publicIp4(x.bytes),
        .ip6 => |x| publicIp6(x.bytes),
    };
}

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
        const io = c.http.io;
        const deadline = Io.Clock.awake.now(io).toMilliseconds() + limits.timeout_ms;
        var current = url;
        var hops: u8 = 0;
        while (true) : (hops += 1) {
            var hop_limits = limits;
            hop_limits.timeout_ms = deadline - Io.Clock.awake.now(io).toMilliseconds();
            if (hop_limits.timeout_ms <= 0) return error.Timeout;
            const step = try c.timed(a, current, headers, hop_limits);
            const next = step.redirect orelse return step.response;
            if (hops >= limits.redirects) return error.Redirects;
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
        const status: u16 = @intFromEnum(response.head.status);
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
        if (std.ascii.eqlIgnoreCase(host, "localhost") or std.ascii.endsWithIgnoreCase(host, ".localhost")) return error.NotPublic;
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
    try std.testing.expect(publicIp6(mapped_public) and publicIp6(global));
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
