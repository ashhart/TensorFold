//! Public HTTPS images for --vision-urls, as TensorFold 0.6.6's server fetched them (vision/images_http.py): an https
//! URL on port 443 without credentials or a fragment, its host resolved and every address checked public, a TLS
//! connection verified for the host on an address that was checked, GET with identity encoding, up to 3 redirects
//! (each checked the same way), a declared media type the caller accepts, the bytes within the caller's limit, all
//! inside a deadline.
const std = @import("std");
const Io = std.Io;
const net = std.Io.net;

pub const Error = error{ OutOfMemory, Media };

/// The failure's words for the client (the request is refused with them).
pub const Failure = struct { text: []const u8 = "" };

/// images.py's ImageLimits: a URL's characters, the redirects followed, one download's seconds and a request's.
pub const max_url_chars = 4096;
pub const max_redirects = 3;
pub const timeout_s = 10;
pub const total_s = 30;

/// images_http.MEDIA_TYPES: the declared types an image URL may answer with (WebP is refused after, by its bytes).
pub const image_media = [_][]const u8{ "image/jpeg", "image/png", "image/webp" };

fn fail(f: *Failure, text: []const u8) Error {
    f.text = text;
    return error.Media;
}

/// Python ipaddress: is_global and not multicast or reserved, nor 192.0.0.0/24 or 168.63.129.16; no IPv6 address
/// that embeds an IPv4 one (mapped, 6to4, Teredo).
pub fn publicIp(a: net.IpAddress) bool {
    switch (a) {
        .ip4 => |v| {
            const x = std.mem.readInt(u32, &v.bytes, .big);
            const blocked = [_]struct { [4]u8, u5 }{
                .{ .{ 0, 0, 0, 0 }, 8 },      .{ .{ 10, 0, 0, 0 }, 8 },    .{ .{ 100, 64, 0, 0 }, 10 },   .{ .{ 127, 0, 0, 0 }, 8 },
                .{ .{ 169, 254, 0, 0 }, 16 }, .{ .{ 172, 16, 0, 0 }, 12 }, .{ .{ 192, 0, 0, 0 }, 24 },    .{ .{ 192, 0, 2, 0 }, 24 },
                .{ .{ 192, 168, 0, 0 }, 16 }, .{ .{ 198, 18, 0, 0 }, 15 }, .{ .{ 198, 51, 100, 0 }, 24 }, .{ .{ 203, 0, 113, 0 }, 24 },
                .{ .{ 224, 0, 0, 0 }, 4 },    .{ .{ 240, 0, 0, 0 }, 4 },
            };
            for (blocked) |blk| {
                const mask = ~@as(u32, 0) << @intCast(32 - @as(u6, blk[1]));
                if (x & mask == std.mem.readInt(u32, &blk[0], .big) & mask) return false;
            }
            return !std.mem.eql(u8, &v.bytes, &.{ 168, 63, 129, 16 });
        },
        .ip6 => |v| {
            const b = v.bytes;
            // global unicast 2000::/3 only, less documentation (2001:db8::/32), the 2001::/23 IETF block (Teredo
            // 2001::/32 among it) and 6to4 (2002::/16)
            if (b[0] & 0xe0 != 0x20) return false;
            if (b[0] == 0x20 and b[1] == 0x01 and b[2] == 0x0d and b[3] == 0xb8) return false;
            if (b[0] == 0x20 and b[1] == 0x01 and b[2] < 0x02) return false;
            return !(b[0] == 0x20 and b[1] == 0x02);
        },
    }
}

/// A URL as a request takes it: the lowercased host and the percent-quoted path and query.
pub const Url = struct { host: []const u8, target: []const u8 };

/// images_http._url: https, a host, port 443, no credentials, fragment, backslash, whitespace or controls. A host
/// that isn't ASCII (an IDNA name) or is an IPv6 literal is refused: std has no IDNA encoder, and TLS here verifies
/// names.
pub fn parseUrl(a: std.mem.Allocator, value: []const u8, f: *Failure) Error!Url {
    if (value.len > max_url_chars) return fail(f, "image URL is too long or contains whitespace/control characters");
    for (value) |c| if (c <= 32 or c == 127) return fail(f, "image URL is too long or contains whitespace/control characters");
    const bad = "image URL must be HTTPS on port 443, without credentials or a fragment";
    if (!std.ascii.startsWithIgnoreCase(value, "https://")) return fail(f, bad);
    if (std.mem.indexOfAny(u8, value, "\\#") != null) return fail(f, bad);
    const rest = value["https://".len..];
    const end = std.mem.indexOfAny(u8, rest, "/?") orelse rest.len;
    const authority = rest[0..end];
    if (std.mem.indexOfScalar(u8, authority, '@') != null or std.mem.startsWith(u8, authority, "[")) return fail(f, bad);
    var host = authority;
    if (std.mem.lastIndexOfScalar(u8, authority, ':')) |c| {
        const port = authority[c + 1 ..];
        if (port.len > 0 and !std.mem.eql(u8, port, "443")) return fail(f, bad);
        host = authority[0..c];
    }
    if (host.len == 0 or std.mem.indexOfScalar(u8, host, '%') != null) return fail(f, bad);
    for (host) |c| if (c >= 128) return fail(f, bad);
    const lower = try std.ascii.allocLowerString(a, host);
    const name = std.mem.trimEnd(u8, lower, ".");
    for ([_][]const u8{ "localhost", "metadata.google.internal", "instance-data" }) |n| if (std.mem.eql(u8, name, n)) return fail(f, "image URLs must use public internet hosts");
    // the path and query, percent-quoted as urllib.parse.quote(safe=...) leaves them
    var t: std.ArrayList(u8) = .empty;
    const tail = rest[end..];
    if (tail.len == 0 or tail[0] == '?') try t.append(a, '/');
    var in_query = false;
    var query_at: usize = 0;
    for (tail) |c| {
        if (c == '?' and !in_query) {
            in_query = true;
            try t.append(a, '?');
            query_at = t.items.len;
            continue;
        }
        const safe = std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, "/%:@!$&'()*+,;=-._~", c) != null or (in_query and c == '?');
        if (safe) try t.append(a, c) else try t.print(a, "%{X:0>2}", .{c});
    }
    if (in_query and t.items.len == query_at) t.items.len -= 1; // an empty query, as urlsplit drops it
    return .{ .host = lower, .target = t.items };
}

/// The location a redirect names, resolved against `base` (urljoin's absolute, scheme-relative, absolute-path and
/// relative forms).
pub fn join(a: std.mem.Allocator, base: []const u8, location: []const u8) ![]const u8 {
    if (std.ascii.startsWithIgnoreCase(location, "https://") or std.ascii.startsWithIgnoreCase(location, "http://")) return a.dupe(u8, location);
    if (std.mem.startsWith(u8, location, "//")) return std.fmt.allocPrint(a, "https:{s}", .{location});
    const rest = base["https://".len..];
    const end = std.mem.indexOfAny(u8, rest, "/?") orelse rest.len;
    const origin = base[0 .. "https://".len + end];
    if (std.mem.startsWith(u8, location, "/")) return std.fmt.allocPrint(a, "{s}{s}", .{ origin, location });
    const path = rest[end..][0 .. std.mem.indexOfScalar(u8, rest[end..], '?') orelse rest.len - end];
    const dir = if (std.mem.lastIndexOfScalar(u8, path, '/')) |s| path[0 .. s + 1] else "/";
    return std.fmt.allocPrint(a, "{s}{s}{s}", .{ origin, dir, location });
}

fn remaining(io: Io, deadline: Io.Timestamp, f: *Failure) Error!i64 {
    const left = Io.Clock.awake.now(io).durationTo(deadline).toMilliseconds();
    if (left <= 0) return fail(f, "image download timed out");
    return left;
}

/// A fetched body and its declared media type.
pub const Body = struct { data: []u8, media: []const u8 };

/// What one request gave: the body, or a redirect's location.
pub const Got = union(enum) { body: Body, redirect: []const u8 };

/// The bytes of `url` and its media type (one of `media_types`), at most `max_bytes`, before `deadline` and within
/// one download's seconds, redirects followed and each checked again.
pub fn fetch(a: std.mem.Allocator, io: Io, url: []const u8, max_bytes: usize, deadline: Io.Timestamp, media_types: []const []const u8, f: *Failure) Error!Body {
    return fetchVia(a, io, url, max_bytes, deadline, media_types, f, Network{});
}

/// fetch over `via`, whose request(a, io, url, max_bytes, deadline, media_types, f) answers one checked URL.
pub fn fetchVia(a: std.mem.Allocator, io: Io, url_in: []const u8, max_bytes: usize, deadline_in: Io.Timestamp, media_types: []const []const u8, f: *Failure, via: anytype) Error!Body {
    const one = Io.Clock.awake.now(io).addDuration(.fromSeconds(timeout_s));
    const deadline = if (one.nanoseconds < deadline_in.nanoseconds) one else deadline_in;
    var url = url_in;
    var hop: usize = 0;
    while (true) : (hop += 1) {
        const u = try parseUrl(a, url, f);
        switch (try via.request(a, io, u, max_bytes, deadline, media_types, f)) {
            .body => |b| return b,
            .redirect => |location| {
                if (hop == max_redirects) return fail(f, "image download has too many redirects");
                url = try join(a, url, location);
            },
        }
    }
}

/// The GET a checked URL is asked with.
pub fn ask(w: *Io.Writer, u: Url, accept: []const u8) Io.Writer.Error!void {
    try w.print("GET {s} HTTP/1.1\r\nHost: {s}\r\nAccept: {s}\r\nAccept-Encoding: identity\r\nUser-Agent: TensorFold-native\r\nConnection: close\r\n\r\n", .{ u.target, u.host, accept });
    try w.flush();
}

/// The response read from `r`: a redirect's location, or a 200's body when its encoding, declared type and length
/// pass (Content-Length or chunked, else to the end of the stream).
pub fn answer(a: std.mem.Allocator, io: Io, r: *Io.Reader, max_bytes: usize, deadline: Io.Timestamp, media_types: []const []const u8, f: *Failure) Error!Got {
    const broke = "image download failed or timed out";
    const too_big = "image response exceeds the encoded byte limit";
    const status_line = r.takeDelimiterInclusive('\n') catch return fail(f, broke);
    if (status_line.len < 12 or !std.mem.startsWith(u8, status_line, "HTTP/1.")) return fail(f, broke);
    const status = std.fmt.parseInt(u16, status_line[9..12], 10) catch return fail(f, broke);
    var location: ?[]const u8 = null;
    var media: []const u8 = "";
    var length: ?usize = null;
    var chunked = false;
    var encoded = false;
    var header_bytes: usize = 0;
    while (true) {
        const line = r.takeDelimiterInclusive('\n') catch return fail(f, broke);
        header_bytes += line.len;
        if (header_bytes > 64 * 1024) return fail(f, broke);
        const l = std.mem.trimEnd(u8, line, "\r\n");
        if (l.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, l, ':') orelse continue;
        const key = std.mem.trim(u8, l[0..colon], " \t");
        const val = std.mem.trim(u8, l[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(key, "location")) {
            location = try a.dupe(u8, val);
        } else if (std.ascii.eqlIgnoreCase(key, "content-type")) {
            media = try std.ascii.allocLowerString(a, std.mem.trim(u8, val[0 .. std.mem.indexOfScalar(u8, val, ';') orelse val.len], " \t"));
        } else if (std.ascii.eqlIgnoreCase(key, "content-length")) {
            if (val.len == 0) return fail(f, too_big);
            for (val) |c| if (!std.ascii.isDigit(c)) return fail(f, too_big);
            length = std.fmt.parseInt(usize, val, 10) catch return fail(f, too_big);
        } else if (std.ascii.eqlIgnoreCase(key, "transfer-encoding")) {
            chunked = std.ascii.findIgnoreCase(val, "chunked") != null;
        } else if (std.ascii.eqlIgnoreCase(key, "content-encoding") and !std.ascii.eqlIgnoreCase(val, "identity")) encoded = true;
    }
    _ = try remaining(io, deadline, f);
    switch (status) {
        301, 302, 303, 307, 308 => return .{ .redirect = location orelse return fail(f, "image redirect has no destination") },
        200 => {},
        else => return fail(f, try std.fmt.allocPrint(a, "image download returned HTTP {d}", .{status})),
    }
    if (encoded) return fail(f, "compressed HTTP image responses are unsupported");
    for (media_types) |m| {
        if (std.mem.eql(u8, m, media)) break;
    } else return fail(f, "image URL content type must be JPEG, PNG or WebP");
    if (length) |n| if (n > max_bytes) return fail(f, too_big);
    var data: std.ArrayList(u8) = .empty;
    if (chunked) {
        while (true) {
            _ = try remaining(io, deadline, f);
            const size_line = r.takeDelimiterInclusive('\n') catch return fail(f, broke);
            const hex = std.mem.trim(u8, size_line[0 .. std.mem.indexOfScalar(u8, size_line, ';') orelse size_line.len], " \t\r\n");
            const n = std.fmt.parseInt(usize, hex, 16) catch return fail(f, broke);
            if (n == 0) break;
            if (n > max_bytes - data.items.len) return fail(f, too_big);
            try readInto(a, io, r, &data, n, deadline, f);
            const crlf = r.takeDelimiterInclusive('\n') catch return fail(f, broke);
            if (std.mem.trimEnd(u8, crlf, "\r\n").len != 0) return fail(f, broke);
        }
    } else if (length) |n| {
        try readInto(a, io, r, &data, n, deadline, f);
    } else while (true) { // no length: to the end of the stream (Connection: close)
        _ = try remaining(io, deadline, f);
        const piece = r.peekGreedy(1) catch |e| switch (e) {
            error.EndOfStream => break,
            else => return fail(f, broke),
        };
        if (piece.len > max_bytes - data.items.len) return fail(f, too_big);
        try data.appendSlice(a, piece);
        r.toss(piece.len);
    }
    _ = try remaining(io, deadline, f);
    return .{ .body = .{ .data = data.items, .media = media } };
}

/// Exactly `n` more bytes from `r` onto `data`, in whatever pieces the reader's buffer holds.
fn readInto(a: std.mem.Allocator, io: Io, r: *Io.Reader, data: *std.ArrayList(u8), n: usize, deadline: Io.Timestamp, f: *Failure) Error!void {
    var left = n;
    while (left > 0) {
        _ = try remaining(io, deadline, f);
        const piece = r.peekGreedy(1) catch return fail(f, "image download failed or timed out");
        const take = @min(piece.len, left);
        try data.appendSlice(a, piece[0..take]);
        r.toss(take);
        left -= take;
    }
}

/// A TCP connection to the checked address on port 443, its connect bounded by `ms` (std's connect has no timeout
/// yet: a nonblocking connect and poll) and its sends and reads by the socket's timeouts.
fn connect(addr: net.IpAddress, ms: i64, f: *Failure) Error!net.Stream {
    const c = std.c;
    const failed = "image download failed or timed out";
    var storage: c.sockaddr.storage = undefined;
    var len: c.socklen_t = undefined;
    switch (addr) {
        .ip4 => |v| {
            const sa: *c.sockaddr.in = @ptrCast(@alignCast(&storage));
            sa.* = std.mem.zeroes(c.sockaddr.in);
            sa.family = c.AF.INET;
            sa.port = std.mem.nativeToBig(u16, 443);
            sa.addr = @bitCast(v.bytes);
            len = @sizeOf(c.sockaddr.in);
        },
        .ip6 => |v| {
            const sa: *c.sockaddr.in6 = @ptrCast(@alignCast(&storage));
            sa.* = std.mem.zeroes(c.sockaddr.in6);
            sa.family = c.AF.INET6;
            sa.port = std.mem.nativeToBig(u16, 443);
            sa.addr = v.bytes;
            len = @sizeOf(c.sockaddr.in6);
        },
    }
    const fd = c.socket(@as(c_uint, storage.family), c.SOCK.STREAM, 0);
    if (fd < 0) return fail(f, failed);
    errdefer _ = c.close(fd);
    _ = c.fcntl(fd, c.F.SETFD, @as(c_int, c.FD_CLOEXEC));
    const flags = c.fcntl(fd, c.F.GETFL);
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(c.O{ .NONBLOCK = true })));
    if (flags < 0 or c.fcntl(fd, c.F.SETFL, flags | nonblock) < 0) return fail(f, failed);
    if (c.connect(fd, @ptrCast(&storage), len) != 0) {
        if (c.errno(@as(c_int, -1)) != .INPROGRESS) return fail(f, failed);
        var pfd = [_]c.pollfd{.{ .fd = fd, .events = c.POLL.OUT, .revents = 0 }};
        if (c.poll(&pfd, 1, @intCast(@min(ms, std.math.maxInt(c_int)))) != 1) return fail(f, failed);
        var err: c_int = 0;
        var err_len: c.socklen_t = @sizeOf(c_int);
        if (c.getsockopt(fd, c.SOL.SOCKET, c.SO.ERROR, &err, &err_len) != 0 or err != 0) return fail(f, failed);
    }
    if (c.fcntl(fd, c.F.SETFL, flags) < 0) return fail(f, failed);
    const tv: c.timeval = .{ .sec = @intCast(@divTrunc(ms, 1000)), .usec = @intCast(@mod(ms, 1000) * 1000) };
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.RCVTIMEO, &tv, @sizeOf(c.timeval));
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.SNDTIMEO, &tv, @sizeOf(c.timeval));
    return .{ .socket = .{ .handle = fd, .address = addr } };
}

var bundle_lock: Io.RwLock = .init;
var bundle: std.crypto.Certificate.Bundle = .empty;
var bundle_loaded = false;

/// The network: every address the host resolves to checked public, then TLS verified for the host on the first.
const Network = struct {
    fn request(_: Network, a: std.mem.Allocator, io: Io, u: Url, max_bytes: usize, deadline: Io.Timestamp, media_types: []const []const u8, f: *Failure) Error!Got {
        const unresolved = "image host could not be resolved";
        var first: ?net.IpAddress = null;
        if (net.IpAddress.parseIp4(u.host, 443)) |literal| {
            if (!publicIp(literal)) return fail(f, "image URLs must resolve only to public internet addresses");
            first = literal;
        } else |_| {
            const name = net.HostName.init(std.mem.trimEnd(u8, u.host, ".")) catch return fail(f, unresolved);
            var results: [32]net.HostName.LookupResult = undefined;
            var q: Io.Queue(net.HostName.LookupResult) = .init(&results);
            name.lookup(io, &q, .{ .port = 443 }) catch return fail(f, unresolved);
            while (q.getOneUncancelable(io)) |r| switch (r) {
                .address => |addr| {
                    if (!publicIp(addr)) return fail(f, "image URLs must resolve only to public internet addresses");
                    if (first == null) first = addr;
                },
                .canonical_name => {},
            } else |_| {}
        }
        const addr = first orelse return fail(f, unresolved);
        {
            bundle_lock.lockUncancelable(io);
            defer bundle_lock.unlock(io);
            if (!bundle_loaded) {
                bundle.rescan(std.heap.page_allocator, io, Io.Clock.real.now(io)) catch return fail(f, "image download failed: no CA certificates on the server");
                bundle_loaded = true;
            }
        }
        const stream = try connect(addr, try remaining(io, deadline, f), f);
        defer stream.close(io);
        const tls_len = std.crypto.tls.Client.min_buffer_len;
        const bufs = try a.alloc(u8, 4 * tls_len);
        var sr = stream.reader(io, bufs[0..tls_len]);
        var sw = stream.writer(io, bufs[tls_len .. 2 * tls_len]);
        var entropy: [std.crypto.tls.Client.Options.entropy_len]u8 = undefined;
        io.random(&entropy);
        var tls = std.crypto.tls.Client.init(&sr.interface, &sw.interface, .{
            .host = .{ .explicit = std.mem.trimEnd(u8, u.host, ".") },
            .ca = .{ .bundle = .{ .gpa = std.heap.page_allocator, .io = io, .lock = &bundle_lock, .bundle = &bundle } },
            .read_buffer = bufs[2 * tls_len .. 3 * tls_len],
            .write_buffer = bufs[3 * tls_len ..],
            .entropy = &entropy,
            .realtime_now = Io.Clock.real.now(io),
            .allow_truncation_attacks = true, // HTTP's own length and chunk framing end the body
        }) catch return fail(f, "image download failed or timed out"); // images_http.py: an ssl.SSLError is an OSError
        const accept = try std.mem.join(a, ", ", media_types);
        ask(&tls.writer, u, accept) catch return fail(f, "image download failed or timed out");
        sw.interface.flush() catch return fail(f, "image download failed or timed out");
        return answer(a, io, &tls.reader, max_bytes, deadline, media_types, f);
    }
};

test "public addresses as Python's ipaddress checks them" {
    const ip4 = struct {
        fn of(b: [4]u8) net.IpAddress {
            return .{ .ip4 = .{ .bytes = b, .port = 443 } };
        }
    }.of;
    const ip6 = struct {
        fn of(text: []const u8) net.IpAddress {
            return net.IpAddress.parseIp6(text, 443) catch unreachable;
        }
    }.of;
    for ([_][4]u8{ .{ 8, 8, 8, 8 }, .{ 151, 101, 1, 69 }, .{ 1, 1, 1, 1 } }) |b| try std.testing.expect(publicIp(ip4(b)));
    for ([_][4]u8{ .{ 10, 1, 2, 3 }, .{ 127, 0, 0, 1 }, .{ 169, 254, 169, 254 }, .{ 172, 20, 0, 1 }, .{ 192, 168, 1, 1 }, .{ 100, 100, 0, 1 }, .{ 0, 0, 0, 0 }, .{ 224, 0, 0, 1 }, .{ 255, 255, 255, 255 }, .{ 168, 63, 129, 16 }, .{ 192, 0, 0, 9 }, .{ 198, 18, 0, 1 }, .{ 203, 0, 113, 5 } }) |b|
        try std.testing.expect(!publicIp(ip4(b)));
    try std.testing.expect(publicIp(ip6("2606:4700::1111")));
    for ([_][]const u8{ "::1", "::", "fe80::1", "fc00::1", "ff02::1", "::ffff:8.8.8.8", "2002:808:808::1", "2001::1", "2001:db8::1", "64:ff9b::808:808" }) |t|
        try std.testing.expect(!publicIp(ip6(t)));
}

test "URLs as images_http._url takes them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Failure = .{};
    const ok = try parseUrl(a, "https://Example.COM:443/img/c.png?x=%20&y=\xc3\xa4?z", &f);
    try std.testing.expectEqualStrings("example.com", ok.host);
    try std.testing.expectEqualStrings("/img/c.png?x=%20&y=%C3%A4?z", ok.target);
    try std.testing.expectEqualStrings("/", (try parseUrl(a, "https://example.com", &f)).target);
    try std.testing.expectEqualStrings("/", (try parseUrl(a, "https://example.com?", &f)).target);
    try std.testing.expectEqualStrings("/a??", (try parseUrl(a, "https://example.com/a??", &f)).target);
    try std.testing.expectEqualStrings("/%5Ba%5D%22%3C", (try parseUrl(a, "https://example.com/[a]\"<", &f)).target);
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "https://example.com/a b.png", "image URL is too long or contains whitespace/control characters" },
        .{ "http://example.com/x.png", "image URL must be HTTPS on port 443, without credentials or a fragment" },
        .{ "https://user:pw@example.com/x", "image URL must be HTTPS on port 443, without credentials or a fragment" },
        .{ "https://example.com:8443/x", "image URL must be HTTPS on port 443, without credentials or a fragment" },
        .{ "https://example.com/x#frag", "image URL must be HTTPS on port 443, without credentials or a fragment" },
        .{ "https://example.com\\@evil/x", "image URL must be HTTPS on port 443, without credentials or a fragment" },
        .{ "https:///x", "image URL must be HTTPS on port 443, without credentials or a fragment" },
        .{ "https://[::1]/x", "image URL must be HTTPS on port 443, without credentials or a fragment" },
        .{ "https://localhost/x", "image URLs must use public internet hosts" },
        .{ "https://LOCALHOST./x", "image URLs must use public internet hosts" },
        .{ "https://metadata.google.internal/x", "image URLs must use public internet hosts" },
    };
    for (cases) |c| {
        try std.testing.expectError(error.Media, parseUrl(a, c[0], &f));
        try std.testing.expectEqualStrings(c[1], f.text);
    }
    const long = try a.alloc(u8, max_url_chars + 1);
    @memcpy(long[0.."https://e.com/".len], "https://e.com/");
    @memset(long["https://e.com/".len..], 'a');
    try std.testing.expectError(error.Media, parseUrl(a, long, &f));
    try std.testing.expectEqualStrings("https://e.com/b/c.png", try join(a, "https://e.com/b/a.png?q", "c.png"));
    try std.testing.expectEqualStrings("https://e.com/z", try join(a, "https://e.com/b/a.png", "/z"));
    try std.testing.expectEqualStrings("https://cdn.e.com/z", try join(a, "https://e.com/b/a.png", "//cdn.e.com/z"));
    try std.testing.expectEqualStrings("http://e.com/z", try join(a, "https://e.com/b", "http://e.com/z"));
}

/// Canned responses by host and path, for the HTTP rules without a network.
const Fixtures = struct {
    pages: []const struct { []const u8, []const u8 },
    asked: *std.ArrayList(u8),

    fn request(self: Fixtures, a: std.mem.Allocator, io: Io, u: Url, max_bytes: usize, deadline: Io.Timestamp, media_types: []const []const u8, f: *Failure) Error!Got {
        var w: Io.Writer.Allocating = .init(a);
        ask(&w.writer, u, try std.mem.join(a, ", ", media_types)) catch return error.OutOfMemory;
        try self.asked.appendSlice(a, w.written());
        const key = try std.fmt.allocPrint(a, "{s}{s}", .{ u.host, u.target });
        for (self.pages) |p| if (std.mem.eql(u8, p[0], key)) {
            var r: Io.Reader = .fixed(p[1]);
            return answer(a, io, &r, max_bytes, deadline, media_types, f);
        };
        return fail(f, "image host could not be resolved");
    }
};

test "responses as images_http._request reads them: redirects, types, lengths, encodings" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var asked: std.ArrayList(u8) = .empty;
    const png = "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 5\r\n\r\n\x89PNG!";
    const fx: Fixtures = .{ .asked = &asked, .pages = &.{
        .{ "e.com/a.png", png },
        .{ "e.com/chunked", "HTTP/1.1 200 OK\r\ncontent-type: Image/JPEG; q=1\r\nTransfer-Encoding: chunked\r\n\r\n3;x\r\n\xff\xd8\xff\r\n2\r\nab\r\n0\r\n\r\n" },
        .{ "e.com/eof", "HTTP/1.0 200 OK\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8\xffxyz" },
        .{ "e.com/r1", "HTTP/1.1 302 Found\r\nLocation: /r2\r\n\r\n" },
        .{ "e.com/r2", "HTTP/1.1 301 Moved\r\nLocation: https://cdn.e.com/b/x\r\n\r\n" },
        .{ "cdn.e.com/b/x", "HTTP/1.1 307 Temporary\r\nLocation: y.png\r\n\r\n" },
        .{ "cdn.e.com/b/y.png", png },
        .{ "e.com/loop", "HTTP/1.1 302 Found\r\nLocation: /loop\r\n\r\n" },
        .{ "e.com/plain", "HTTP/1.1 302 Found\r\nLocation: http://e.com/a.png\r\n\r\n" },
        .{ "e.com/inner", "HTTP/1.1 302 Found\r\nLocation: https://localhost/a.png\r\n\r\n" },
        .{ "e.com/nowhere", "HTTP/1.1 302 Found\r\n\r\n" },
        .{ "e.com/404", "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n" },
        .{ "e.com/gz", "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Encoding: gzip\r\nContent-Length: 1\r\n\r\nx" },
        .{ "e.com/html", "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 1\r\n\r\nx" },
        .{ "e.com/big", "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 11\r\n\r\n01234567890" },
        .{ "e.com/bigchunk", "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nTransfer-Encoding: chunked\r\n\r\n6\r\n012345\r\n6\r\n678901\r\n0\r\n\r\n" },
        .{ "e.com/bigeof", "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n\r\n01234567890" },
        .{ "e.com/badlen", "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: +5\r\n\r\n12345" },
        .{ "e.com/short", "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 9\r\n\r\n12345" },
        .{ "e.com/junk", "SSH-2.0-OpenSSH\r\n\r\n" },
    } };
    const far = Io.Clock.awake.now(io).addDuration(.fromSeconds(60));
    var f: Failure = .{};
    const one = try fetchVia(a, io, "https://e.com/a.png", 10, far, &image_media, &f, fx);
    try std.testing.expectEqualStrings("\x89PNG!", one.data);
    try std.testing.expectEqualStrings("image/png", one.media);
    try std.testing.expectEqualStrings("GET /a.png HTTP/1.1\r\nHost: e.com\r\nAccept: image/jpeg, image/png, image/webp\r\nAccept-Encoding: identity\r\nUser-Agent: TensorFold-native\r\nConnection: close\r\n\r\n", asked.items);
    const ch = try fetchVia(a, io, "https://e.com/chunked", 10, far, &image_media, &f, fx);
    try std.testing.expectEqualStrings("\xff\xd8\xffab", ch.data);
    try std.testing.expectEqualStrings("image/jpeg", ch.media);
    try std.testing.expectEqualStrings("\xff\xd8\xffxyz", (try fetchVia(a, io, "https://e.com/eof", 10, far, &image_media, &f, fx)).data);
    // three redirects, each joined and checked again, reach the image; a fourth is refused
    asked.clearRetainingCapacity();
    try std.testing.expectEqualStrings("\x89PNG!", (try fetchVia(a, io, "https://e.com/r1", 10, far, &image_media, &f, fx)).data);
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, asked.items, "GET "));
    try std.testing.expect(std.mem.indexOf(u8, asked.items, "GET /b/y.png HTTP/1.1\r\nHost: cdn.e.com\r\n") != null);
    const refusals = [_]struct { []const u8, []const u8 }{
        .{ "https://e.com/loop", "image download has too many redirects" },
        .{ "https://e.com/plain", "image URL must be HTTPS on port 443, without credentials or a fragment" },
        .{ "https://e.com/inner", "image URLs must use public internet hosts" },
        .{ "https://e.com/nowhere", "image redirect has no destination" },
        .{ "https://e.com/404", "image download returned HTTP 404" },
        .{ "https://e.com/gz", "compressed HTTP image responses are unsupported" },
        .{ "https://e.com/html", "image URL content type must be JPEG, PNG or WebP" },
        .{ "https://e.com/big", "image response exceeds the encoded byte limit" },
        .{ "https://e.com/bigchunk", "image response exceeds the encoded byte limit" },
        .{ "https://e.com/bigeof", "image response exceeds the encoded byte limit" },
        .{ "https://e.com/badlen", "image response exceeds the encoded byte limit" },
        .{ "https://e.com/short", "image download failed or timed out" },
        .{ "https://e.com/junk", "image download failed or timed out" },
    };
    for (refusals) |c| {
        try std.testing.expectError(error.Media, fetchVia(a, io, c[0], 10, far, &image_media, &f, fx));
        try std.testing.expectEqualStrings(c[1], f.text);
    }
    // a deadline already passed refuses before the body
    try std.testing.expectError(error.Media, fetchVia(a, io, "https://e.com/a.png", 10, Io.Clock.awake.now(io), &image_media, &f, fx));
    try std.testing.expectEqualStrings("image download timed out", f.text);
}
