//! Public HTTPS media for --vision-urls, as the Python server fetches it (vision/images_http.py): an https URL on
//! port 443 without credentials or a fragment, its host resolved and every address checked public, a TLS connection
//! verified for the host on an address that was checked, GET with identity encoding, up to 3 redirects (each checked
//! the same way), a declared media type the caller accepts, the bytes within the caller's limit, all inside a deadline.

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;

pub const Error = error{ OutOfMemory, Media };

/// The failure's words for the client (the request is refused with them).
pub const Failure = struct { text: []const u8 = "" };

pub const max_url_chars = 4096;
pub const max_redirects = 3;
pub const timeout_s = 10; // one download
pub const total_s = 30; // a request's downloads together

fn fail(f: *Failure, text: []const u8) Error {
    f.text = text;
    return error.Media;
}

/// Python ipaddress: is_global and not multicast or reserved, nor 192.0.0.0/24 or 168.63.129.16; no IPv6 address
/// that embeds an IPv4 one (mapped, 6to4, Teredo).
pub fn publicIp(a: net.IpAddress) bool {
    switch (a) {
        .ip4 => |v| {
            const b = v.bytes;
            const in = struct {
                fn net4(x: [4]u8, base: [4]u8, bits: u5) bool {
                    const xi = std.mem.readInt(u32, &x, .big);
                    const bi = std.mem.readInt(u32, &base, .big);
                    const mask: u32 = if (bits == 0) 0 else ~@as(u32, 0) << @intCast(32 - @as(u6, bits));
                    return xi & mask == bi & mask;
                }
            }.net4;
            const blocked = [_]struct { [4]u8, u5 }{
                .{ .{ 0, 0, 0, 0 }, 8 },      .{ .{ 10, 0, 0, 0 }, 8 },     .{ .{ 100, 64, 0, 0 }, 10 }, .{ .{ 127, 0, 0, 0 }, 8 },
                .{ .{ 169, 254, 0, 0 }, 16 }, .{ .{ 172, 16, 0, 0 }, 12 },  .{ .{ 192, 0, 0, 0 }, 24 },  .{ .{ 192, 0, 2, 0 }, 24 },
                .{ .{ 192, 168, 0, 0 }, 16 }, .{ .{ 198, 18, 0, 0 }, 15 },  .{ .{ 198, 51, 100, 0 }, 24 }, .{ .{ 203, 0, 113, 0 }, 24 },
                .{ .{ 224, 0, 0, 0 }, 4 },    .{ .{ 240, 0, 0, 0 }, 4 },
            };
            for (blocked) |blk| if (in(b, blk[0], blk[1])) return false;
            if (std.mem.eql(u8, &b, &.{ 168, 63, 129, 16 })) return false;
            return true;
        },
        .ip6 => |v| {
            const b = v.bytes;
            // global unicast 2000::/3 only, less documentation (2001:db8::/32), the 2001::/23 IETF block (Teredo
            // 2001::/32 among it), 6to4 (2002::/16)
            if (b[0] & 0xe0 != 0x20) return false;
            if (b[0] == 0x20 and b[1] == 0x01 and b[2] == 0x0d and b[3] == 0xb8) return false;
            if (b[0] == 0x20 and b[1] == 0x01 and b[2] < 0x02) return false;
            if (b[0] == 0x20 and b[1] == 0x02) return false;
            return true;
        },
    }
}

const Url = struct { host: []const u8, target: []const u8 };

/// grammar of images_http._url: https, a host, port 443, no credentials, fragment, backslash, whitespace or controls.
fn parseUrl(a: std.mem.Allocator, value: []const u8, f: *Failure) Error!Url {
    if (value.len > max_url_chars) return fail(f, "image URL is too long or contains whitespace/control characters");
    for (value) |c| if (c <= 32 or c == 127) return fail(f, "image URL is too long or contains whitespace/control characters");
    const bad = "image URL must be HTTPS on port 443, without credentials or a fragment";
    if (!std.ascii.startsWithIgnoreCase(value, "https://")) return fail(f, bad);
    if (std.mem.indexOfScalar(u8, value, '\\') != null or std.mem.indexOfScalar(u8, value, '#') != null) return fail(f, bad);
    const rest = value["https://".len..];
    const end = std.mem.indexOfAny(u8, rest, "/?") orelse rest.len;
    const authority = rest[0..end];
    if (std.mem.indexOfScalar(u8, authority, '@') != null) return fail(f, bad);
    var host = authority;
    if (std.mem.lastIndexOfScalar(u8, authority, ':')) |c| {
        if (authority.len > 0 and authority[0] == '[') return fail(f, bad); // a literal IPv6 host: not a name
        if (!std.mem.eql(u8, authority[c + 1 ..], "443") and authority[c + 1 ..].len > 0) return fail(f, bad);
        host = authority[0..c];
    }
    if (host.len == 0 or std.mem.indexOfScalar(u8, host, '%') != null) return fail(f, bad);
    for (host) |c| if (c >= 128) return fail(f, bad); // IDNA names: their ASCII (xn--) form
    const lower = try std.ascii.allocLowerString(a, std.mem.trimEnd(u8, host, "."));
    for ([_][]const u8{ "localhost", "metadata.google.internal", "instance-data" }) |n| if (std.mem.eql(u8, lower, n)) return fail(f, "image URLs must use public internet hosts");
    // the path and query, percent-quoted as urllib.parse.quote(safe=...) leaves them
    var t: std.ArrayList(u8) = .empty;
    var tail = rest[end..];
    if (tail.len == 0 or tail[0] == '?') try t.append(a, '/');
    const safe_path = "/%:@!$&'()*+,;=-._~";
    var in_query = false;
    for (tail) |c| {
        if (c == '?' and !in_query) {
            in_query = true;
            try t.append(a, '?');
            continue;
        }
        const safe = std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, safe_path, c) != null or (in_query and c == '?');
        if (safe) try t.append(a, c) else try t.print(a, "%{X:0>2}", .{c});
    }
    tail = t.items;
    return .{ .host = lower, .target = tail };
}

/// The location a redirect names, resolved against ``base`` (urljoin for absolute and absolute-path forms).
fn join(a: std.mem.Allocator, base: []const u8, location: []const u8) ![]const u8 {
    if (std.ascii.startsWithIgnoreCase(location, "https://") or std.ascii.startsWithIgnoreCase(location, "http://")) return a.dupe(u8, location);
    if (std.mem.startsWith(u8, location, "//")) return std.fmt.allocPrint(a, "https:{s}", .{location});
    const rest = base["https://".len..];
    const end = std.mem.indexOfAny(u8, rest, "/?") orelse rest.len;
    const origin = base[0 .. "https://".len + end];
    if (std.mem.startsWith(u8, location, "/")) return std.fmt.allocPrint(a, "{s}{s}", .{ origin, location });
    const path_end = std.mem.indexOfScalar(u8, rest[end..], '?') orelse rest.len - end;
    const path = rest[end..][0..path_end];
    const dir = if (std.mem.lastIndexOfScalar(u8, path, '/')) |s| path[0 .. s + 1] else "/";
    return std.fmt.allocPrint(a, "{s}{s}{s}", .{ origin, dir, location });
}

fn remaining(io: Io, deadline: Io.Timestamp, f: *Failure) Error!i64 {
    const left = Io.Clock.awake.now(io).durationTo(deadline).toMilliseconds();
    if (left <= 0) return fail(f, "image download timed out");
    return left;
}

fn setTimeouts(handle: net.Socket.Handle, ms: i64) void {
    const tv: std.posix.timeval = .{ .sec = @intCast(@divTrunc(ms, 1000)), .usec = @intCast(@mod(ms, 1000) * 1000) };
    std.posix.setsockopt(handle, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&tv)) catch {};
    std.posix.setsockopt(handle, std.posix.SOL.SOCKET, std.posix.SO.SNDTIMEO, std.mem.asBytes(&tv)) catch {};
}

/// A TCP connection to the checked address, its connect, sends and reads bounded by ``ms`` (SO_SNDTIMEO bounds a
/// Linux connect too; std's connect has no timeout yet).
fn connect(addr: net.IpAddress, ms: i64, f: *Failure) Error!net.Stream {
    const posix = std.posix;
    const linux = std.os.linux;
    const family: u32 = switch (addr) {
        .ip4 => posix.AF.INET,
        .ip6 => posix.AF.INET6,
    };
    const rc = linux.socket(family, posix.SOCK.STREAM | posix.SOCK.CLOEXEC, 0);
    if (linux.errno(rc) != .SUCCESS) return fail(f, "image download failed or timed out");
    const fd: posix.fd_t = @intCast(rc);
    errdefer _ = linux.close(fd);
    setTimeouts(fd, ms);
    var sa4: posix.sockaddr.in = undefined;
    var sa6: posix.sockaddr.in6 = undefined;
    const r = switch (addr) {
        .ip4 => |v| blk: {
            sa4 = .{ .port = std.mem.nativeToBig(u16, 443), .addr = @bitCast(v.bytes) };
            break :blk linux.connect(fd, @ptrCast(&sa4), @sizeOf(posix.sockaddr.in));
        },
        .ip6 => |v| blk: {
            sa6 = .{ .port = std.mem.nativeToBig(u16, 443), .flowinfo = 0, .addr = v.bytes, .scope_id = 0 };
            break :blk linux.connect(fd, @ptrCast(&sa6), @sizeOf(posix.sockaddr.in6));
        },
    };
    if (linux.errno(r) != .SUCCESS) return fail(f, "image download failed or timed out");
    return .{ .socket = .{ .handle = fd, .address = addr } };
}

/// The bytes of ``url`` (and its media type, one of ``media_types``), at most ``max_bytes``, before ``deadline``.
pub fn fetch(a: std.mem.Allocator, io: Io, url_in: []const u8, max_bytes: usize, deadline_in: Io.Timestamp, media_types: []const []const u8, f: *Failure) Error!struct { data: []u8, media: []const u8 } {
    var url = url_in;
    const deadline = blk: {
        const one = Io.Clock.awake.now(io).addDuration(.fromSeconds(timeout_s));
        break :blk if (one.nanoseconds < deadline_in.nanoseconds) one else deadline_in;
    };
    var hop: usize = 0;
    while (true) : (hop += 1) {
        const u = try parseUrl(a, url, f);
        const got = try request(a, io, u, max_bytes, deadline, media_types, f);
        switch (got) {
            .body => |b| return .{ .data = b.data, .media = b.media },
            .redirect => |loc| {
                if (hop == max_redirects) return fail(f, "image download has too many redirects");
                url = join(a, url, loc) catch return error.OutOfMemory;
            },
        }
    }
}

const Got = union(enum) { body: struct { data: []u8, media: []const u8 }, redirect: []const u8 };

var bundle_lock: Io.RwLock = .init;
var bundle: std.crypto.Certificate.Bundle = .empty;
var bundle_loaded = false;

fn request(a: std.mem.Allocator, io: Io, u: Url, max_bytes: usize, deadline: Io.Timestamp, media_types: []const []const u8, f: *Failure) Error!Got {
    // every address the name resolves to must be public; the connection goes to the first of them
    const name = net.HostName.init(u.host) catch return fail(f, "image host could not be resolved");
    var results: [32]net.HostName.LookupResult = undefined;
    var q: Io.Queue(net.HostName.LookupResult) = .init(&results);
    name.lookup(io, &q, .{ .port = 443 }) catch return fail(f, "image host could not be resolved");
    var first: ?net.IpAddress = null;
    var count: usize = 0;
    while (q.getOneUncancelable(io)) |r| switch (r) {
        .address => |addr| {
            if (!publicIp(addr)) return fail(f, "image URLs must resolve only to public internet addresses");
            if (first == null) first = addr;
            count += 1;
        },
        .canonical_name => {},
    } else |_| {}
    const addr = first orelse return fail(f, "image host could not be resolved");
    _ = try remaining(io, deadline, f);
    // the CA bundle, read once
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
        .host = .{ .explicit = u.host },
        .ca = .{ .bundle = .{ .gpa = std.heap.page_allocator, .io = io, .lock = &bundle_lock, .bundle = &bundle } },
        .read_buffer = bufs[2 * tls_len .. 3 * tls_len],
        .write_buffer = bufs[3 * tls_len ..],
        .entropy = &entropy,
        .realtime_now = Io.Clock.real.now(io),
        .allow_truncation_attacks = true, // HTTP's own lengths and framing end the body
    }) catch return fail(f, "image download failed: TLS could not be verified for the host");
    // the request
    const accept = std.mem.join(a, ", ", media_types) catch return error.OutOfMemory;
    tls.writer.print("GET {s} HTTP/1.1\r\nHost: {s}\r\nAccept: {s}\r\nAccept-Encoding: identity\r\nUser-Agent: TensorFold-native\r\nConnection: close\r\n\r\n", .{ u.target, u.host, accept }) catch return fail(f, "image download failed or timed out");
    tls.writer.flush() catch return fail(f, "image download failed or timed out");
    sw.interface.flush() catch return fail(f, "image download failed or timed out");
    const r = &tls.reader;
    // status and headers
    const status_line = r.takeDelimiterInclusive('\n') catch return fail(f, "image download failed or timed out");
    if (status_line.len < 12 or !std.mem.startsWith(u8, status_line, "HTTP/1.")) return fail(f, "image download failed");
    const status = std.fmt.parseInt(u16, status_line[9..12], 10) catch return fail(f, "image download failed");
    var location: ?[]const u8 = null;
    var media: []const u8 = "";
    var length: ?usize = null;
    var chunked = false;
    var encoded = false;
    var header_bytes: usize = 0;
    while (true) {
        const line = r.takeDelimiterInclusive('\n') catch return fail(f, "image download failed or timed out");
        header_bytes += line.len;
        if (header_bytes > 64 * 1024) return fail(f, "image download failed");
        const l = std.mem.trimEnd(u8, line, "\r\n");
        if (l.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, l, ':') orelse continue;
        const key = std.mem.trim(u8, l[0..colon], " \t");
        const val = std.mem.trim(u8, l[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(key, "location")) location = try a.dupe(u8, val);
        if (std.ascii.eqlIgnoreCase(key, "content-type")) {
            const semi = std.mem.indexOfScalar(u8, val, ';') orelse val.len;
            media = try std.ascii.allocLowerString(a, std.mem.trim(u8, val[0..semi], " \t"));
        }
        if (std.ascii.eqlIgnoreCase(key, "content-length")) {
            for (val) |c| if (!std.ascii.isDigit(c)) return fail(f, "image response exceeds the encoded byte limit");
            length = std.fmt.parseInt(usize, val, 10) catch return fail(f, "image response exceeds the encoded byte limit");
        }
        if (std.ascii.eqlIgnoreCase(key, "transfer-encoding")) chunked = std.mem.indexOf(u8, try std.ascii.allocLowerString(a, val), "chunked") != null;
        if (std.ascii.eqlIgnoreCase(key, "content-encoding") and !std.ascii.eqlIgnoreCase(val, "identity")) encoded = true;
    }
    switch (status) {
        301, 302, 303, 307, 308 => return .{ .redirect = location orelse return fail(f, "image redirect has no destination") },
        200 => {},
        else => return fail(f, try std.fmt.allocPrint(a, "image download returned HTTP {d}", .{status})),
    }
    if (encoded) return fail(f, "compressed HTTP image responses are unsupported");
    for (media_types) |m| {
        if (std.mem.eql(u8, m, media)) break;
    } else return fail(f, if (std.mem.startsWith(u8, media_types[0], "image/")) "image URL content type must be JPEG, PNG or WebP" else try std.fmt.allocPrint(a, "media URL content type must be one of {s}", .{accept}));
    if (length) |n| if (n > max_bytes) return fail(f, "image response exceeds the encoded byte limit");
    var data: std.ArrayList(u8) = .empty;
    if (chunked) {
        while (true) {
            _ = try remaining(io, deadline, f);
            const size_line = r.takeDelimiterInclusive('\n') catch return fail(f, "image download failed or timed out");
            const semi = std.mem.indexOfScalar(u8, size_line, ';') orelse size_line.len;
            const hex = std.mem.trim(u8, size_line[0..semi], " \t\r\n");
            const n = std.fmt.parseInt(usize, hex, 16) catch return fail(f, "image download failed");
            if (n == 0) break;
            if (data.items.len + n > max_bytes) return fail(f, "image response exceeds the encoded byte limit");
            const piece = r.take(n) catch return fail(f, "image download failed or timed out");
            try data.appendSlice(a, piece);
            _ = r.takeDelimiterInclusive('\n') catch return fail(f, "image download failed");
        }
    } else {
        while (length == null or data.items.len < length.?) {
            _ = try remaining(io, deadline, f);
            const want = if (length) |n| @min(n - data.items.len, 64 * 1024) else 64 * 1024;
            const piece = r.peekGreedy(1) catch |e| switch (e) {
                error.EndOfStream => break,
                else => return fail(f, "image download failed or timed out"),
            };
            const take = @min(piece.len, want);
            if (data.items.len + take > max_bytes) return fail(f, "image response exceeds the encoded byte limit");
            try data.appendSlice(a, piece[0..take]);
            r.toss(take);
        }
        if (length) |n| if (data.items.len != n) return fail(f, "image download failed or timed out");
    }
    _ = try remaining(io, deadline, f);
    return .{ .body = .{ .data = data.items, .media = media } };
}

test "public addresses as Python's ipaddress checks them" {
    const ip4 = struct {
        fn of(b: [4]u8) net.IpAddress {
            return .{ .ip4 = .{ .bytes = b, .port = 443 } };
        }
    }.of;
    try std.testing.expect(publicIp(ip4(.{ 8, 8, 8, 8 })));
    try std.testing.expect(publicIp(ip4(.{ 151, 101, 1, 69 })));
    for ([_][4]u8{ .{ 10, 1, 2, 3 }, .{ 127, 0, 0, 1 }, .{ 169, 254, 169, 254 }, .{ 172, 20, 0, 1 }, .{ 192, 168, 1, 1 }, .{ 100, 100, 0, 1 }, .{ 0, 0, 0, 0 }, .{ 224, 0, 0, 1 }, .{ 255, 255, 255, 255 }, .{ 168, 63, 129, 16 }, .{ 192, 0, 0, 9 } }) |b|
        try std.testing.expect(!publicIp(ip4(b)));
}

test "URLs as images_http._url takes them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Failure = .{};
    try std.testing.expectError(error.Media, parseUrl(a, "https://example.com/a b/c.png", &f));
    const ok = try parseUrl(a, "https://example.com:443/img/c.png?x=%20&y=ä", &f);
    try std.testing.expectEqualStrings("example.com", ok.host);
    try std.testing.expectEqualStrings("/img/c.png?x=%20&y=%C3%A4", ok.target);
    for ([_][]const u8{ "http://example.com/x.png", "https://user@example.com/x", "https://example.com:8443/x", "https://localhost/x", "https://example.com/x#frag", "https://[::1]/x" }) |bad|
        try std.testing.expectError(error.Media, parseUrl(a, bad, &f));
    try std.testing.expectEqualStrings("https://e.com/b/c.png", try join(a, "https://e.com/b/a.png", "c.png"));
    try std.testing.expectEqualStrings("https://e.com/z", try join(a, "https://e.com/b/a.png", "/z"));
}
