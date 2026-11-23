//! Pull's routes: the proxy each hub or storage host goes through (proxy variables, NO_PROXY), and its timeouts.
const std = @import("std");
const environment = @import("sys").environment;
const Writer = std.Io.Writer;

/// An http:// proxy from one variable; `auth` (Basic credentials) is sent to the proxy and never shown.
pub const Proxy = struct {
    host: []const u8,
    port: u16,
    auth: ?[]const u8 = null,
    from: []const u8,

    /// scheme://host:port and the variable it came from, without credentials.
    pub fn format(p: Proxy, w: *Writer) Writer.Error!void {
        const v6 = std.mem.indexOfScalar(u8, p.host, ':') != null;
        try w.print("http://{s}{s}{s}:{d} (from {s})", .{ if (v6) "[" else "", p.host, if (v6) "]" else "", p.port, p.from });
    }
};

pub const Settings = struct {
    http: ?Proxy = null,
    https: ?Proxy = null,
    no_proxy: []const u8 = "",
    connect_s: u32 = 30,
    stall_s: u32 = 60,
    skipped: []const u8 = "", // a socks or https:// proxy pull cannot use: it connects directly, as before proxies
};

pub const connect_name = "TENSORFOLD_PULL_CONNECT_SECONDS";
pub const stall_name = "TENSORFOLD_PULL_STALL_SECONDS";

/// Set once by `configure` before any request; the socket layer reads it on every connect.
pub var settings: Settings = .{};

/// Why `configure` refused, naming the variable only: its value may carry credentials.
pub var refused: []const u8 = "";

/// Reads the proxy variables (lower case first, as curl does), NO_PROXY and the two pull timeouts.
pub fn configure(a: std.mem.Allocator, env: ?*const std.process.Environ.Map) !void {
    settings = try fromEnv(a, env);
}

pub fn fromEnv(a: std.mem.Allocator, env: ?*const std.process.Environ.Map) !Settings {
    refused = "";
    const get = struct {
        fn get(m: ?*const std.process.Environ.Map, names: []const []const u8) ?struct { []const u8, []const u8 } {
            for (names) |n| if (m) |map| if (map.get(n)) |v| if (std.mem.trim(u8, v, " \t").len > 0) return .{ n, v };
            return null;
        }
    }.get;
    var s: Settings = .{};
    if (get(env, &.{ "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY" })) |hit| s.https = try usable(a, &s, hit[0], hit[1]);
    if (get(env, &.{ "http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY" })) |hit| s.http = try usable(a, &s, hit[0], hit[1]);
    if (get(env, &.{ "no_proxy", "NO_PROXY" })) |hit| s.no_proxy = hit[1];
    const raw = struct {
        fn raw(m: ?*const std.process.Environ.Map, n: []const u8) ?[]const u8 {
            return if (m) |map| map.get(n) else null;
        }
    }.raw;
    s.connect_s = environment.value(u32, connect_name, raw(env, connect_name), 30) catch {
        refused = connect_name;
        return error.InvalidEnvironment;
    };
    s.stall_s = environment.value(u32, stall_name, raw(env, stall_name), 60) catch {
        refused = stall_name;
        return error.InvalidEnvironment;
    };
    return s;
}

/// A socks or https:// proxy is skipped, not refused: 1.3.3 and earlier ignored every proxy.
fn usable(a: std.mem.Allocator, s: *Settings, from: []const u8, value: []const u8) !?Proxy {
    return parse(a, from, value) catch |e| switch (e) {
        error.ProxyScheme => {
            refused = "";
            s.skipped = from;
            return null;
        },
        else => {
            refused = from;
            return e;
        },
    };
}

/// http://[user[:password]@]host[:port] or host[:port]; port 80 when none is given, as Python's requests does.
fn parse(a: std.mem.Allocator, from: []const u8, value: []const u8) !Proxy {
    const text = std.mem.trim(u8, value, " \t/");
    const full = if (std.mem.indexOf(u8, text, "://") == null) try std.fmt.allocPrint(a, "http://{s}", .{text}) else text;
    const uri = std.Uri.parse(full) catch return error.ProxyUrl;
    if (!std.ascii.eqlIgnoreCase(uri.scheme, "http")) return error.ProxyScheme;
    var host_buf: [std.Io.net.HostName.max_len]u8 = undefined;
    var host = (uri.host orelse return error.ProxyUrl).toRaw(&host_buf) catch return error.ProxyUrl;
    if (host.len > 1 and host[0] == '[' and host[host.len - 1] == ']') host = host[1 .. host.len - 1];
    if (host.len == 0) return error.ProxyUrl;
    const auth: ?[]const u8 = if (uri.user != null or uri.password != null) blk: {
        const basic = std.http.Client.basic_authorization;
        const buf = try a.alloc(u8, basic.valueLengthFromUri(uri));
        break :blk basic.value(uri, buf);
    } else null;
    return .{ .host = try a.dupe(u8, host), .port = uri.port orelse 80, .auth = auth, .from = from };
}

/// The one-line explanation of a `configure` refusal; it names the variable, never its value.
pub fn refusal(w: *Writer, err: anyerror) Writer.Error!void {
    switch (err) {
        error.ProxyScheme => try w.print("{s} names a proxy pull cannot use: only http:// proxies (an https:// or socks proxy is not supported)", .{refused}),
        error.InvalidEnvironment => try w.print("{s} must be whole seconds from 1 to 4294967295", .{refused}),
        else => try w.print("{s} is not a proxy URL pull can read (http://[user:password@]host[:port])", .{refused}),
    }
}

/// NO_PROXY sends `host:port` direct on "*", an exact host, a suffix (.x.org, *.x.org, x.org) or any with :port.
pub fn bypass(list: []const u8, host: []const u8, port: u16) bool {
    const candidate = if (std.mem.endsWith(u8, host, ".")) host[0 .. host.len - 1] else host;
    var it = std.mem.tokenizeAny(u8, list, ", \t");
    while (it.next()) |raw| {
        if (std.mem.eql(u8, raw, "*")) return true;
        var e = raw;
        var want: ?u16 = null;
        if (std.mem.lastIndexOfScalar(u8, e, ':')) |i| {
            const bracketed = e[0] == '[' and i > 0 and e[i - 1] == ']';
            if (bracketed or std.mem.count(u8, e, ":") == 1) {
                want = std.fmt.parseInt(u16, e[i + 1 ..], 10) catch continue;
                e = e[0..i];
            }
        }
        if (e.len > 1 and e[0] == '[' and e[e.len - 1] == ']') e = e[1 .. e.len - 1];
        if (want) |p| if (p != port) continue;
        if (std.mem.startsWith(u8, e, "*.")) e = e[1..];
        if (std.mem.endsWith(u8, e, ".")) e = e[0 .. e.len - 1];
        if (e.len == 0) continue;
        if (e[0] == '.') {
            if (std.ascii.endsWithIgnoreCase(candidate, e) or std.ascii.eqlIgnoreCase(candidate, e[1..])) return true;
            continue;
        }
        if (std.ascii.eqlIgnoreCase(candidate, e)) return true;
        if (candidate.len > e.len and std.ascii.endsWithIgnoreCase(candidate, e) and candidate[candidate.len - e.len - 1] == '.') return true;
    }
    return false;
}

/// The proxy a request to `scheme://host:port` takes, or null to go direct.
pub fn pick(s: Settings, tls: bool, host: []const u8, port: u16) ?Proxy {
    const p = (if (tls) s.https else s.http) orelse return null;
    return if (bypass(s.no_proxy, host, port)) null else p;
}

/// One host's route and the absolute deadline shared by its lookup and every TCP attempt.
const Route = struct {
    host: [std.Io.net.HostName.max_len]u8 = undefined,
    len: usize = 0,
    port: u16 = 0,
    proxy: ?Proxy = null,
    deadline: std.Io.Clock.Timestamp = .{ .raw = .zero, .clock = .awake },
};

var routes: [64]Route = @splat(.{});
var count: usize = 0;
var lock: std.Io.Mutex = .init;

/// Records the route of `uri`'s host and port before a request to it, so each redirect hop takes its own.
pub fn note(io: std.Io, uri: std.Uri) !void {
    const tls = std.ascii.eqlIgnoreCase(uri.scheme, "https");
    var host_buf: [std.Io.net.HostName.max_len]u8 = undefined;
    const host = (std.Io.net.HostName.fromUri(uri, &host_buf) catch return).bytes;
    const port = uri.port orelse @as(u16, if (tls) 443 else 80);
    const proxy = pick(settings, tls, host, port);
    const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromSeconds(settings.connect_s), .clock = .awake });
    lock.lockUncancelable(io);
    defer lock.unlock(io);
    const i = find(host, port) orelse blk: {
        if (count == routes.len) return error.TooManyHosts;
        @memcpy(routes[count].host[0..host.len], host);
        routes[count].len = host.len;
        routes[count].port = port;
        count += 1;
        break :blk count - 1;
    };
    routes[i].proxy = proxy;
    routes[i].deadline = deadline;
}

fn find(host: []const u8, port: u16) ?usize {
    for (routes[0..count], 0..) |*r, i| if (r.port == port and std.ascii.eqlIgnoreCase(r.host[0..r.len], host)) return i;
    return null;
}

pub fn hostDeadline(io: std.Io, host: []const u8, port: u16) ?std.Io.Clock.Timestamp {
    lock.lockUncancelable(io);
    defer lock.unlock(io);
    const i = find(host, port) orelse return null;
    return routes[i].deadline;
}

/// 100::/64 is a discard-only prefix (RFC 6666): no real host has these addresses.
const marker = [_]u8{ 0x01, 0x00, 0, 0, 0, 0, 0, 0, 't', 'f', 'p', 'r' };

/// The stand-in address a lookup of a proxied `host:port` returns, or null to resolve it as usual.
pub fn standIn(io: std.Io, host: []const u8, port: u16) ?std.Io.net.IpAddress {
    lock.lockUncancelable(io);
    defer lock.unlock(io);
    const i = find(host, port) orelse return null;
    if (routes[i].proxy == null) return null;
    var bytes: [16]u8 = undefined;
    @memcpy(bytes[0..12], &marker);
    std.mem.writeInt(u32, bytes[12..16], @intCast(i), .big);
    return .{ .ip6 = .{ .port = port, .bytes = bytes } };
}

pub const Tunnel = struct { host: [std.Io.net.HostName.max_len]u8, len: usize, port: u16, proxy: Proxy, deadline: std.Io.Clock.Timestamp };

/// The tunnel a stand-in address stands for.
pub fn tunnelOf(io: std.Io, address: *const std.Io.net.IpAddress) ?Tunnel {
    if (address.* != .ip6 or !std.mem.eql(u8, address.ip6.bytes[0..12], &marker)) return null;
    const i = std.mem.readInt(u32, address.ip6.bytes[12..16], .big);
    lock.lockUncancelable(io);
    defer lock.unlock(io);
    if (i >= count) return null;
    const r = routes[i];
    return .{ .host = r.host, .len = r.len, .port = r.port, .proxy = r.proxy orelse return null, .deadline = r.deadline };
}

const AddressDeadline = struct { address: std.Io.net.IpAddress = undefined, deadline: std.Io.Clock.Timestamp = .{ .raw = .zero, .clock = .awake } };
var addresses: [64]AddressDeadline = @splat(.{});
var address_count: usize = 0;
var address_next: usize = 0; // once full, the oldest entry is replaced (a pull over rotating CDN addresses never runs out)

/// Carries a name's deadline to each resolved address; when full, the oldest goes (its connect gets a fresh deadline).
pub fn rememberAddress(io: std.Io, address: std.Io.net.IpAddress, deadline: std.Io.Clock.Timestamp) error{TooManyHosts}!void {
    lock.lockUncancelable(io);
    defer lock.unlock(io);
    for (addresses[0..address_count]) |*a| if (a.address.eql(&address)) {
        a.deadline = deadline;
        return;
    };
    if (address_count < addresses.len) {
        addresses[address_count] = .{ .address = address, .deadline = deadline };
        address_count += 1;
        return;
    }
    addresses[address_next] = .{ .address = address, .deadline = deadline };
    address_next = (address_next + 1) % addresses.len;
}

pub fn addressDeadline(io: std.Io, address: *const std.Io.net.IpAddress) ?std.Io.Clock.Timestamp {
    lock.lockUncancelable(io);
    defer lock.unlock(io);
    for (addresses[0..address_count]) |a| if (a.address.eql(address)) return a.deadline;
    return null;
}

var stall_count: std.atomic.Value(u64) = .init(0);

/// Counts a socket read that saw no bytes within the stall timeout.
pub fn stalled() void {
    _ = stall_count.fetchAdd(1, .monotonic);
}

pub fn stalls() u64 {
    return stall_count.load(.monotonic);
}

/// What the last failed connect met, for the message pull prints instead of waiting silently.
pub const Failure = union(enum) {
    none,
    connect_timeout: ?Proxy,
    proxy_unreachable: struct { proxy: Proxy, err: anyerror },
    proxy_status: struct { proxy: Proxy, status: u16 },
    proxy_silent: Proxy,
};

var failure: Failure = .none;

pub fn fail(io: std.Io, f: Failure) void {
    lock.lockUncancelable(io);
    defer lock.unlock(io);
    failure = f;
}

pub fn lastFailure(io: std.Io) Failure {
    lock.lockUncancelable(io);
    defer lock.unlock(io);
    return failure;
}

pub fn reset(io: std.Io) void {
    fail(io, .none);
}

/// Why the hub is out of reach, from the last failed connect, else `err`; never a credential.
pub fn explain(io: std.Io, w: *Writer, err: anyerror) Writer.Error!void {
    const s = settings;
    switch (lastFailure(io)) {
        .connect_timeout => |p| if (p) |proxy| {
            try w.print("the proxy {f} did not accept a connection within {d} s ({s})", .{ proxy, s.connect_s, connect_name });
        } else {
            try w.print("no direct connection within {d} s ({s}); ", .{ s.connect_s, connect_name });
            try proxyInUse(w);
        },
        .proxy_unreachable => |f| try w.print("the proxy {f} is unreachable ({t})", .{ f.proxy, f.err }),
        .proxy_status => |f| if (f.status == 407) {
            try w.print("the proxy {f} wants credentials it did not accept (HTTP 407)", .{f.proxy});
        } else {
            try w.print("the proxy {f} refused the tunnel (HTTP {d})", .{ f.proxy, f.status });
        },
        .proxy_silent => |p| try w.print("the proxy {f} did not answer the tunnel request within {d} s ({s})", .{ p, s.stall_s, stall_name }),
        .none => {
            try w.print("{t}; ", .{err});
            try proxyInUse(w);
        },
    }
}

/// The proxy settings in use, without credentials.
pub fn proxyInUse(w: *Writer) Writer.Error!void {
    const s = settings;
    if (s.https == null and s.http == null) return w.writeAll("no proxy is set (HTTPS_PROXY, HTTP_PROXY, ALL_PROXY)");
    if (s.https) |p| try w.print("https through {f}", .{p});
    if (s.https != null and s.http != null) try w.writeAll(", ");
    if (s.http) |p| try w.print("http through {f}", .{p});
}

test {
    _ = @import("pull_route_test.zig");
}
