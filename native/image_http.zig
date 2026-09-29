const std = @import("std");
const Ip = std.Io.net.IpAddress;
const Cancellation = @import("cancellation.zig").Cancellation;
const Curl = opaque {};
const List = opaque {};
extern "c" fn curl_easy_init() ?*Curl;
extern "c" fn curl_easy_cleanup(*Curl) void;
extern "c" fn curl_easy_setopt(*Curl, c_int, ...) c_int;
extern "c" fn curl_easy_perform(*Curl) c_int;
extern "c" fn curl_slist_append(?*List, [*:0]const u8) ?*List;
extern "c" fn curl_slist_free_all(?*List) void;
extern "c" fn uidna_IDNToASCII([*]const u16, i32, [*]u16, i32, i32, ?*anyopaque, *i32) i32;

pub fn now(io: std.Io) i64 {
    return std.Io.Clock.awake.now(io).toMilliseconds();
}

fn inSubnet(ip: Ip, comptime cidr: []const u8) bool {
    const slash = comptime std.mem.indexOfScalar(u8, cidr, '/').?;
    const network = comptime Ip.parse(cidr[0..slash], 0) catch unreachable;
    const bits = comptime std.fmt.parseInt(u8, cidr[slash + 1 ..], 10) catch unreachable;
    return switch (network) {
        .ip4 => |v| ip == .ip4 and std.mem.readInt(u32, &ip.ip4.bytes, .big) >> (32 - bits) == std.mem.readInt(u32, &v.bytes, .big) >> (32 - bits),
        .ip6 => |v| ip == .ip6 and std.mem.readInt(u128, &ip.ip6.bytes, .big) >> (128 - bits) == std.mem.readInt(u128, &v.bytes, .big) >> (128 - bits),
    };
}

pub fn publicIp(ip: Ip) bool {
    if (ip == .ip4) {
        inline for (.{ "0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16", "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24", "192.168.0.0/16", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24", "224.0.0.0/3", "168.63.129.16/32" }) |block| if (inSubnet(ip, block)) return false;
        return true;
    }
    if (!inSubnet(ip, "2000::/3") and !inSubnet(ip, "fec0::/10")) return false;
    inline for (.{ "2001:1::1/128", "2001:1::2/128", "2001:3::/32", "2001:4:112::/48", "2001:20::/28", "2001:30::/28" }) |block| if (inSubnet(ip, block)) return true;
    inline for (.{ "2001::/23", "2001:db8::/32", "2002::/16", "3fff::/20" }) |block| if (inSubnet(ip, block)) return false;
    return true;
}

pub const Url = struct { host: [:0]const u8, text: [:0]const u8, base: []const u8 };

pub fn parseUrl(a: std.mem.Allocator, value: []const u8) !Url {
    if (value.len > 4 * 4096 or (std.unicode.utf8CountCodepoints(value) catch return error.InvalidImageUrl) > 4096) return error.InvalidImageUrl;
    for (value) |byte| if (byte <= 32 or byte == 127 or byte == '\\') return error.InvalidImageUrl;
    if (value.len < 8 or !std.ascii.eqlIgnoreCase(value[0..8], "https://")) return error.InvalidImageUrl;
    const authority_end = 8 + (std.mem.indexOfAny(u8, value[8..], "/?#") orelse value.len - 8);
    const authority = value[8..authority_end];
    if (authority.len == 0 or std.mem.indexOfScalar(u8, authority, '@') != null) return error.InvalidImageUrl;
    const host_end = if (authority[0] == '[') (std.mem.indexOfScalar(u8, authority, ']') orelse return error.InvalidImageUrl) + 1 else std.mem.lastIndexOfScalar(u8, authority, ':') orelse authority.len;
    if (host_end < authority.len) {
        if (authority[host_end] != ':') return error.InvalidImageUrl;
        const port = authority[host_end + 1 ..];
        for (port) |byte| if (!std.ascii.isDigit(byte)) return error.InvalidImageUrl;
        if (port.len > 0 and (std.fmt.parseInt(u16, port, 10) catch return error.InvalidImageUrl) != 443) return error.InvalidImageUrl;
    }
    const raw_host = authority[0..host_end];
    if (std.mem.indexOfScalar(u8, raw_host, '%') != null or raw_host.len == 0) return error.InvalidImageUrl;
    const host = if (raw_host[0] == '[') blk: {
        if (raw_host.len < 3 or raw_host[raw_host.len - 1] != ']') return error.InvalidImageUrl;
        _ = Ip.parseIp6(raw_host[1 .. raw_host.len - 1], 443) catch return error.InvalidImageUrl;
        const result = try a.dupeSentinel(u8, raw_host[1 .. raw_host.len - 1], 0);
        for (result) |*byte| byte.* = std.ascii.toLower(byte.*);
        break :blk result;
    } else blk: {
        const utf16 = try std.unicode.utf8ToUtf16LeAlloc(a, raw_host);
        defer a.free(utf16);
        var output: [1024]u16 = undefined;
        var status: i32 = 0;
        const count = uidna_IDNToASCII(utf16.ptr, @intCast(utf16.len), &output, output.len, 1, null, &status);
        if (status > 0 or count <= 0 or count > 254) return error.InvalidImageUrl;
        const result = try a.allocSentinel(u8, @intCast(count), 0);
        errdefer a.free(result);
        for (result, output[0..@intCast(count)]) |*byte, cp| {
            if (cp > 127 or cp <= 32 or cp == 127 or std.mem.indexOfScalar(u8, "/?#@\\:%", @intCast(cp)) != null) return error.InvalidImageUrl;
            byte.* = std.ascii.toLower(@intCast(cp));
        }
        break :blk result;
    };
    errdefer a.free(host);
    const plain = std.mem.trimEnd(u8, host, ".");
    for ([_][]const u8{ "localhost", "metadata.google.internal", "instance-data" }) |blocked| if (std.ascii.eqlIgnoreCase(plain, blocked)) return error.NonPublicImageHost;
    const suffix = value[authority_end..];
    const fragment = std.mem.indexOfScalar(u8, suffix, '#') orelse suffix.len;
    if (fragment < suffix.len and fragment != suffix.len - 1) return error.InvalidImageUrl;
    const target = suffix[0..fragment];
    const query = std.mem.indexOfScalar(u8, target, '?') orelse target.len;
    // URL paths may be Unicode; percent signs in existing escapes stay intact.
    var encoded: std.Io.Writer.Allocating = .init(a);
    defer encoded.deinit();
    if (std.mem.indexOfScalar(u8, host, ':') != null) try encoded.writer.print("https://[{s}]", .{host}) else try encoded.writer.print("https://{s}", .{host});
    const base = try std.mem.concat(a, u8, &.{ encoded.written(), target });
    const path = target[0..query];
    try quote(&encoded.writer, if (path.len == 0) "/" else path, false);
    if (query < target.len and query + 1 < target.len) {
        try encoded.writer.writeByte('?');
        try quote(&encoded.writer, target[query + 1 ..], true);
    }
    return .{ .host = host, .text = try a.dupeSentinel(u8, encoded.written(), 0), .base = base };
}

fn quote(w: *std.Io.Writer, text: []const u8, query: bool) !void {
    const hex = "0123456789ABCDEF";
    for (text) |byte| {
        if (std.ascii.isAlphanumeric(byte) or std.mem.indexOfScalar(u8, "/%:@!$&'()*+,;=-._~", byte) != null or (query and byte == '?')) try w.writeByte(byte) else try w.writeAll(&.{ '%', hex[byte >> 4], hex[byte & 15] });
    }
}

var dns_count: std.atomic.Value(usize) = .init(0);
const Dns = struct {
    refs: std.atomic.Value(usize) = .init(2),
    done: std.atomic.Value(bool) = .init(false),
    host: [:0]u8,
    result: anyerror!Ip = error.ImageDnsFailed,

    fn release(d: *Dns) void {
        if (d.refs.fetchSub(1, .acq_rel) == 1) {
            std.heap.page_allocator.free(d.host);
            std.heap.page_allocator.destroy(d);
        }
    }
    fn run(d: *Dns) void {
        defer d.release();
        defer _ = dns_count.fetchSub(1, .acq_rel);
        d.result = d.lookup();
        d.done.store(true, .release);
    }
    fn lookup(d: *Dns) !Ip {
        const hints = std.mem.zeroInit(std.c.addrinfo, .{ .socktype = std.c.SOCK.STREAM });
        var result: ?*std.c.addrinfo = null;
        if (@backingInt(std.c.getaddrinfo(d.host, "443", &hints, &result)) != 0) return error.ImageDnsFailed;
        defer if (result) |p| std.c.freeaddrinfo(p);
        var first: ?Ip = null;
        var cursor = result;
        while (cursor) |entry| : (cursor = entry.next) {
            const addr = entry.addr orelse return error.ImageDnsFailed;
            const ip: Ip = switch (addr.family) {
                std.c.AF.INET => blk: {
                    const v: *const std.c.sockaddr.in = @ptrCast(@alignCast(addr));
                    break :blk .{ .ip4 = .{ .bytes = @bitCast(v.addr), .port = 443 } };
                },
                std.c.AF.INET6 => blk: {
                    const v: *const std.c.sockaddr.in6 = @ptrCast(@alignCast(addr));
                    break :blk .{ .ip6 = .{ .bytes = v.addr, .port = 443, .interface = .none } };
                },
                else => return error.ImageDnsFailed,
            };
            try addAddress(&first, ip);
        }
        return first orelse error.ImageDnsFailed;
    }
};

fn addAddress(first: *?Ip, ip: Ip) !void {
    if (!publicIp(ip)) return error.NonPublicImageHost;
    if (first.* == null) first.* = ip;
}

fn resolve(io: std.Io, host: [:0]const u8, deadline: i64, cancellation: Cancellation) !Ip {
    try cancellation.check();
    if (now(io) >= deadline) return error.ImageDownloadTimedOut;
    const d = try startDns(host);
    return waitDns(io, d, deadline, cancellation);
}

fn startDns(host: [:0]const u8) !*Dns {
    if (dns_count.fetchAdd(1, .acq_rel) >= 4) {
        _ = dns_count.fetchSub(1, .acq_rel);
        return error.ImageDnsBusy;
    }
    errdefer _ = dns_count.fetchSub(1, .acq_rel);
    const d = try std.heap.page_allocator.create(Dns);
    errdefer std.heap.page_allocator.destroy(d);
    d.* = .{ .host = try std.heap.page_allocator.dupeSentinel(u8, host, 0) };
    errdefer std.heap.page_allocator.free(d.host);
    const thread = try std.Thread.spawn(.{}, Dns.run, .{d});
    thread.detach();
    return d;
}

fn waitDns(io: std.Io, d: *Dns, deadline: i64, cancellation: Cancellation) !Ip {
    defer d.release();
    while (!d.done.load(.acquire)) {
        try cancellation.check();
        if (now(io) >= deadline) return error.ImageDownloadTimedOut;
        try std.Io.sleep(io, .fromMilliseconds(5), .awake);
    }
    try cancellation.check();
    if (now(io) >= deadline) return error.ImageDownloadTimedOut;
    return d.result;
}

const Response = struct {
    a: std.mem.Allocator,
    limit: usize,
    status: u16 = 0,
    location: ?[]const u8 = null,
    media: bool = false,
    header_bytes: usize = 0,
    body: std.ArrayList(u8) = .empty,
    failure: ?anyerror = null,
    cancellation: Cancellation = .{},

    fn progress(context: ?*anyopaque, _: i64, _: i64, _: i64, _: i64) callconv(.c) c_int {
        const r: *Response = @ptrCast(@alignCast(context.?));
        r.cancellation.check() catch |err| {
            r.failure = err;
            return 1;
        };
        return 0;
    }

    fn redirect(r: *const Response) bool {
        return switch (r.status) {
            301, 302, 303, 307, 308 => true,
            else => false,
        };
    }
    fn header(r: *Response, line: []const u8) !void {
        r.header_bytes += line.len;
        if (r.header_bytes > 64 * 1024) return error.ImageHeadersTooLarge;
        if (std.mem.startsWith(u8, line, "HTTP/")) {
            var words = std.mem.tokenizeScalar(u8, line, ' ');
            _ = words.next();
            r.status = try std.fmt.parseInt(u16, words.next() orelse return error.InvalidImageResponse, 10);
            r.media = false;
            r.location = null;
            return;
        }
        if (r.redirect()) {
            if (headerValue(line, "location")) |v| r.location = try r.a.dupe(u8, v);
            return;
        }
        if (r.status != 200) return;
        if (headerValue(line, "content-type")) |value| {
            const media = std.mem.trim(u8, value[0 .. std.mem.indexOfScalar(u8, value, ';') orelse value.len], " \t");
            r.media = std.ascii.eqlIgnoreCase(media, "image/jpeg") or std.ascii.eqlIgnoreCase(media, "image/png") or std.ascii.eqlIgnoreCase(media, "image/webp");
        }
        if (headerValue(line, "content-encoding")) |v| if (!std.ascii.eqlIgnoreCase(v, "identity")) return error.CompressedImageResponse;
        if (headerValue(line, "content-length")) |v| {
            if (v.len == 0) return error.ImageByteLimitExceeded;
            for (v) |byte| if (!std.ascii.isDigit(byte)) return error.ImageByteLimitExceeded;
            const length = std.fmt.parseInt(usize, v, 10) catch return error.ImageByteLimitExceeded;
            if (length > r.limit) return error.ImageByteLimitExceeded;
        }
    }
    fn headerCallback(data: [*]const u8, size: usize, count: usize, context: ?*anyopaque) callconv(.c) usize {
        const r: *Response = @ptrCast(@alignCast(context.?));
        const length = std.math.mul(usize, size, count) catch return 0;
        r.header(data[0..length]) catch |err| {
            r.failure = err;
            return 0;
        };
        return length;
    }
    fn writeCallback(data: [*]const u8, size: usize, count: usize, context: ?*anyopaque) callconv(.c) usize {
        const r: *Response = @ptrCast(@alignCast(context.?));
        const length = std.math.mul(usize, size, count) catch return 0;
        if (r.redirect() or r.status != 200) return 0;
        if (!r.media) {
            r.failure = error.InvalidImageMediaType;
            return 0;
        }
        if (length > r.limit - r.body.items.len) {
            r.failure = error.ImageByteLimitExceeded;
            return 0;
        }
        r.body.appendSlice(r.a, data[0..length]) catch |err| {
            r.failure = err;
            return 0;
        };
        return length;
    }
};

fn headerValue(line: []const u8, name: []const u8) ?[]const u8 {
    const colon = std.mem.indexOfScalar(u8, line, ':') orelse return null;
    if (!std.ascii.eqlIgnoreCase(line[0..colon], name)) return null;
    return std.mem.trim(u8, line[colon + 1 ..], " \t\r\n");
}

fn set(curl: *Curl, option: c_int, value: anytype) !void {
    if (curl_easy_setopt(curl, option, value) != 0) return error.ImageTransportConfiguration;
}

fn request(a: std.mem.Allocator, io: std.Io, url: Url, ip: Ip, limit: usize, deadline: i64, cancellation: Cancellation) !Response {
    try cancellation.check();
    const remaining = deadline - now(io);
    if (remaining <= 0) return error.ImageDownloadTimedOut;
    const curl = curl_easy_init() orelse return error.OutOfMemory;
    defer curl_easy_cleanup(curl);
    var response = Response{ .a = a, .limit = limit, .cancellation = cancellation };
    errdefer response.body.deinit(a);
    const address = switch (ip) {
        .ip4 => |v| try std.fmt.allocPrint(a, "{d}.{d}.{d}.{d}", .{ v.bytes[0], v.bytes[1], v.bytes[2], v.bytes[3] }),
        .ip6 => |v| try std.fmt.allocPrint(a, "[{x}:{x}:{x}:{x}:{x}:{x}:{x}:{x}]", .{ std.mem.readInt(u16, v.bytes[0..2], .big), std.mem.readInt(u16, v.bytes[2..4], .big), std.mem.readInt(u16, v.bytes[4..6], .big), std.mem.readInt(u16, v.bytes[6..8], .big), std.mem.readInt(u16, v.bytes[8..10], .big), std.mem.readInt(u16, v.bytes[10..12], .big), std.mem.readInt(u16, v.bytes[12..14], .big), std.mem.readInt(u16, v.bytes[14..16], .big) }),
    };
    defer a.free(address);
    const pinned = try std.fmt.allocPrintSentinel(a, "::{s}:443", .{address}, 0);
    defer a.free(pinned);
    const hosts = curl_slist_append(null, pinned) orelse return error.OutOfMemory;
    defer curl_slist_free_all(hosts);
    var headers: ?*List = null;
    defer curl_slist_free_all(headers);
    for ([_][*:0]const u8{ "Accept: image/jpeg, image/png, image/webp", "Accept-Encoding: identity", "User-Agent: TensorFold/native" }) |value| headers = curl_slist_append(headers, value) orelse return error.OutOfMemory;
    try set(curl, 10002, url.text.ptr); // CURLOPT_URL
    try set(curl, 10004, @as([*:0]const u8, "")); // CURLOPT_PROXY: never bypass pinned DNS through an environment proxy
    try set(curl, 10243, hosts); // CURLOPT_CONNECT_TO: numeric destination, original TLS hostname
    try set(curl, 10023, headers); // CURLOPT_HTTPHEADER
    try set(curl, 10318, @as([*:0]const u8, "https")); // CURLOPT_PROTOCOLS_STR
    try set(curl, 52, @as(c_long, 0)); // CURLOPT_FOLLOWLOCATION: each hop is revalidated
    try set(curl, 64, @as(c_long, 1)); // CURLOPT_SSL_VERIFYPEER
    try set(curl, 81, @as(c_long, 2)); // CURLOPT_SSL_VERIFYHOST
    try set(curl, 99, @as(c_long, 1)); // CURLOPT_NOSIGNAL
    try set(curl, 234, @as(c_long, 1)); // CURLOPT_PATH_AS_IS
    try set(curl, 155, @as(c_long, @intCast(remaining))); // CURLOPT_TIMEOUT_MS
    try set(curl, 20011, &Response.writeCallback);
    try set(curl, 10001, &response);
    try set(curl, 20079, &Response.headerCallback);
    try set(curl, 10029, &response);
    try set(curl, 43, @as(c_long, 0)); // CURLOPT_NOPROGRESS
    try set(curl, 20219, &Response.progress); // CURLOPT_XFERINFOFUNCTION
    try set(curl, 10057, &response); // CURLOPT_XFERINFODATA
    const result = curl_easy_perform(curl);
    try cancellation.check();
    if (response.failure) |err| return err;
    if (result == 60) return error.ImageTlsVerificationFailed;
    if (now(io) >= deadline or result == 28) return error.ImageDownloadTimedOut;
    if (response.redirect() and (result == 0 or result == 23) and response.location != null) return response;
    if (response.status != 200) return error.InvalidImageHttpStatus;
    if (result != 0) return error.ImageDownloadFailed;
    if (!response.media) return error.InvalidImageMediaType;
    if (response.body.items.len == 0) return error.EmptyImage;
    return response;
}

pub fn fetch(a: std.mem.Allocator, io: std.Io, initial: []const u8, limit: usize, deadline: i64) ![]u8 {
    return fetchWithCancellation(a, io, initial, limit, deadline, .{});
}

pub fn fetchWithCancellation(a: std.mem.Allocator, io: std.Io, initial: []const u8, limit: usize, deadline: i64, cancellation: Cancellation) ![]u8 {
    try cancellation.check();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    var current = initial;
    for (0..4) |hop| {
        const url = try parseUrl(scratch, current);
        const ip = try resolve(io, url.host, deadline, cancellation);
        const response = try request(scratch, io, url, ip, limit, deadline, cancellation);
        if (!response.redirect()) return a.dupe(u8, response.body.items);
        if (hop == 3) return error.TooManyImageRedirects;
        current = try redirectUrl(scratch, url.base, response.location orelse return error.InvalidImageRedirect);
    }
    unreachable;
}

fn redirectUrl(a: std.mem.Allocator, base_text: []const u8, location: []const u8) ![]const u8 {
    if (location.len == 0 or location.len > 4 * 4096 or (std.unicode.utf8CountCodepoints(location) catch return error.InvalidImageUrl) > 4096) return error.InvalidImageUrl;
    for (location) |byte| if (byte <= 32 or byte == 127 or byte == '\\') return error.InvalidImageUrl;
    const base = try std.Uri.parse(base_text);
    var buffer = try a.alloc(u8, 128 * 1024);
    @memcpy(buffer[0..location.len], location);
    const redirected = try std.Uri.resolveInPlace(base, location.len, &buffer);
    const resolved = try std.fmt.allocPrint(a, "{f}", .{redirected});
    return (try parseUrl(a, resolved)).base;
}

pub fn check(io: std.Io, fixture: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, fixture, a, .limited(4 * 1024 * 1024));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    const ips = parsed.value.object.get("ips").?.array.items;
    const urls = parsed.value.object.get("urls").?.array.items;
    for (ips) |entry| {
        const input = entry.object.get("value").?.string;
        const got = if (Ip.parse(input, 443)) |ip| publicIp(ip) else |_| false;
        if (got != entry.object.get("public").?.bool) {
            std.debug.print("Image address mismatch: {s}\n", .{input});
            return error.ImageAddressPolicyMismatch;
        }
    }
    for (urls) |entry| {
        const input = entry.object.get("value").?.string;
        const want = entry.object.get("canonical").?;
        if (parseUrl(a, input)) |url| {
            if (want != .string or !std.mem.eql(u8, want.string, url.text)) {
                std.debug.print("Image URL mismatch: {s}; native {s}, upstream {f}\n", .{ input, url.text, std.json.fmt(want, .{}) });
                return error.ImageUrlPolicyMismatch;
            }
        } else |err| {
            if (want != .null) {
                std.debug.print("Image URL rejected: {s}; {s}\n", .{ input, @errorName(err) });
                return error.ImageUrlPolicyMismatch;
            }
        }
    }
    std.debug.print("PASS: {d} image address and {d} URL cases match upstream\n", .{ ips.len, urls.len });
}

pub fn fetchCheck(io: std.Io, url: []const u8, output: []const u8) !void {
    const a = std.heap.page_allocator;
    const bytes = try fetch(a, io, url, 10 * 1024 * 1024, now(io) + 10000);
    defer a.free(bytes);
    const file = try std.Io.Dir.cwd().createFile(io, output, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, bytes);
    std.debug.print("PASS: fetched {d} image bytes over validated HTTPS\n", .{bytes.len});
}

test "image URLs reject private destinations and ambiguous authorities" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "http://example.com/a", "https://user@example.com/a", "https://example.com:8443/a", "https://example.com/#x", "https://example.com\\@localhost/a", "https://example.com/a\n", "https://%31%32%37.0.0.1/a", "https://[fe80::1%25en0]/a" }) |value| try std.testing.expectError(error.InvalidImageUrl, parseUrl(a, value));
    for ([_][]const u8{ "https://localhost./a", "https://METADATA.GOOGLE.INTERNAL/a", "https://instance-data/a" }) |value| try std.testing.expectError(error.NonPublicImageHost, parseUrl(a, value));
    try std.testing.expectEqualStrings("https://xn--bcher-kva.example/%C3%A6?q=%C3%B8", (try parseUrl(a, "https://bücher.example/æ?q=ø")).text);
    for ([_][]const u8{ "127.0.0.1", "10.1.2.3", "100.100.1.1", "192.0.0.9", "168.63.129.16", "224.0.0.1", "::1", "::ffff:8.8.8.8", "2002:0808:0808::1", "2001::1", "2001:db8::1", "fc00::1", "3fff::1" }) |value| try std.testing.expect(!publicIp(try Ip.parse(value, 443)));
    for ([_][]const u8{ "8.8.8.8", "1.1.1.1", "2606:4700:4700::1111", "2001:20::1", "2001:3::1" }) |value| try std.testing.expect(publicIp(try Ip.parse(value, 443)));
}

test "image response limits apply before buffering and after chunk boundaries" {
    var r = Response{ .a = std.testing.allocator, .limit = 3 };
    defer r.body.deinit(r.a);
    try r.header("HTTP/1.1 200 OK\r\n");
    try r.header("Content-Type: image/png; charset=binary\r\n");
    try std.testing.expectEqual(@as(usize, 2), Response.writeCallback("ab", 1, 2, &r));
    try std.testing.expectEqual(@as(usize, 0), Response.writeCallback("cd", 1, 2, &r));
    try std.testing.expectEqual(error.ImageByteLimitExceeded, r.failure.?);
    try std.testing.expectError(error.ImageByteLimitExceeded, r.header("Content-Length: 4\r\n"));
    try std.testing.expectError(error.CompressedImageResponse, r.header("Content-Encoding: gzip\r\n"));
}

test "redirect destinations and all DNS candidates are validated before connection" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const base = "https://example.com/images/a.png";
    try std.testing.expectEqualStrings("https://example.com/b.png", try redirectUrl(a, base, "../b.png"));
    try std.testing.expectError(error.InvalidImageUrl, redirectUrl(a, base, "http://example.com/b.png"));
    try std.testing.expectError(error.NonPublicImageHost, redirectUrl(a, base, "//localhost/b.png"));
    var first: ?Ip = null;
    try addAddress(&first, try Ip.parse("8.8.8.8", 443));
    try std.testing.expectError(error.NonPublicImageHost, addAddress(&first, try Ip.parse("127.0.0.1", 443)));
}

test "DNS capacity and expired deadlines reject without connecting" {
    const old = dns_count.swap(4, .acq_rel);
    defer dns_count.store(old, .release);
    try std.testing.expectError(error.ImageDnsBusy, startDns("example.com"));
    try std.testing.expectEqual(@as(usize, 4), dns_count.load(.acquire));
    try std.testing.expectError(error.ImageDownloadTimedOut, resolve(std.testing.io, "example.com", now(std.testing.io) - 1, .{}));
}

test "request cancellation reaches DNS admission and the TLS transfer callback" {
    const cancelled = Cancellation{ .callback = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.ServerStopping;
        }
    }.check };
    try std.testing.expectError(error.ServerStopping, fetchWithCancellation(std.testing.allocator, std.testing.io, "https://example.com/a.png", 100, now(std.testing.io) + 10000, cancelled));
    var response = Response{ .a = std.testing.allocator, .limit = 100, .cancellation = cancelled };
    try std.testing.expectEqual(@as(c_int, 1), Response.progress(&response, 100, 0, 0, 0));
    try std.testing.expectEqual(error.ServerStopping, response.failure.?);
}
