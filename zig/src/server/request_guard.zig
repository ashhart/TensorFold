//! Loopback host admission and browser origins are checked before authentication, bodies or routing.
const std = @import("std");
const auth = @import("auth.zig");
const http = @import("http_conn.zig");
pub const Reason = enum { host, origin };
const Authority = struct { host: []const u8, port: ?u16 = null };
const Origin = struct { https: bool, authority: Authority, port: u16 };

pub const Policy = struct {
    loopback_bind: bool = true,
    allowed_hosts: []const []const u8 = &.{},
    allowed_origins: []const []const u8 = &.{},

    pub fn blocked(p: Policy, headers: []const http.Header) ?Reason {
        var host: ?[]const u8 = null;
        var origin: ?[]const u8 = null;
        var hosts: usize = 0;
        var origins: usize = 0;
        for (headers) |h| {
            if (std.ascii.eqlIgnoreCase(h.name, "Host")) {
                host = h.value;
                hosts += 1;
            }
            if (std.ascii.eqlIgnoreCase(h.name, "Origin")) {
                origin = h.value;
                origins += 1;
            }
        }
        if (hosts != 1) return .host;
        const h = authority(host.?, true) orelse return .host;
        if (p.loopback_bind) {
            if (!auth.loopback(h.host)) {
                for (p.allowed_hosts) |name| {
                    const allowed = authority(name, false) orelse continue;
                    if (sameHost(h.host, allowed.host)) break;
                } else return .host;
            }
        }
        if (origins == 0) return null;
        if (origins != 1 or hosts != 1) return .origin;
        const o = parseOrigin(origin.?) orelse return .origin;

        if (!o.https and o.port == (h.port orelse 80) and sameHost(o.authority.host, h.host)) return null;
        for (p.allowed_origins) |name| {
            const allowed = parseOrigin(name) orelse continue;
            if (o.https == allowed.https and o.port == allowed.port and sameHost(o.authority.host, allowed.authority.host)) return null;
        }
        return .origin;
    }
};

pub fn validHost(name: []const u8) bool {
    return authority(name, false) != null;
}
pub fn validOrigin(name: []const u8) bool {
    return parseOrigin(name) != null;
}

fn authority(raw: []const u8, ports: bool) ?Authority {
    if (raw.len == 0 or raw.len > 300 or std.mem.indexOfScalar(u8, raw, '%') != null) return null;
    var host = raw;
    var port: ?[]const u8 = null;
    if (raw[0] == '[') {
        const end = std.mem.indexOfScalar(u8, raw, ']') orelse return null;
        host = raw[1..end];
        const ip = std.Io.net.IpAddress.parse(host, 0) catch return null;
        if (ip != .ip6) return null;
        if (end + 1 < raw.len) {
            if (raw[end + 1] != ':') return null;
            port = raw[end + 2 ..];
        }
    } else {
        if (std.mem.indexOfScalar(u8, raw, ':')) |colon| {
            host = raw[0..colon];
            port = raw[colon + 1 ..];
        }
        if (host.len == 0 or host.len > 253) return null;
        var labels = std.mem.splitScalar(u8, host, '.');
        while (labels.next()) |label| {
            if (label.len == 0 or label.len > 63 or label[0] == '-' or label[label.len - 1] == '-') return null;
            for (label) |c| if (!std.ascii.isAlphanumeric(c) and c != '-') return null;
        }
    }
    var out: Authority = .{ .host = host };
    if (port) |text| {
        if (!ports or text.len == 0) return null;
        for (text) |c| if (!std.ascii.isDigit(c)) return null;
        out.port = std.fmt.parseInt(u16, text, 10) catch return null;
    }
    return out;
}

fn parseOrigin(raw: []const u8) ?Origin {
    const split = std.mem.indexOf(u8, raw, "://") orelse return null;
    const scheme = raw[0..split];
    const https = std.ascii.eqlIgnoreCase(scheme, "https");
    if (!https and !std.ascii.eqlIgnoreCase(scheme, "http")) return null;
    const site = authority(raw[split + 3 ..], true) orelse return null;
    return .{ .https = https, .authority = site, .port = site.port orelse if (https) 443 else 80 };
}

fn sameHost(a: []const u8, b: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(a, b)) return true;
    const x = std.Io.net.IpAddress.parse(a, 0) catch return false;
    const y = std.Io.net.IpAddress.parse(b, 0) catch return false;
    if (x == .ip4 and y == .ip4) return std.mem.eql(u8, &x.ip4.bytes, &y.ip4.bytes);
    if (x == .ip6 and y == .ip6) return std.mem.eql(u8, &x.ip6.bytes, &y.ip6.bytes);
    return false;
}

pub const host_message = "Host header is not allowed on this loopback bind; use --allowed-host NAME for a trusted name";
pub const origin_message = "Origin header does not match this server; use --allowed-origin ORIGIN for a trusted origin";
pub fn allow(conn: *http.Conn, policy: Policy, a: std.mem.Allocator) bool {
    if (conn.path.len == 0 or conn.path[0] != '/' or std.mem.indexOfScalar(u8, conn.path, '#') != null or invalidTarget(conn.path)) {
        conn.close = true;
        conn.sendError(a, 400, "Unsupported HTTP request target", null) catch {};
        return false;
    }
    const reason = policy.blocked(conn.headers) orelse return true;
    const body: []const u8 = switch (reason) {
        .host => "{\"error\": {\"message\": \"" ++ host_message ++ "\", \"type\": \"invalid_request_error\"}}",
        .origin => "{\"error\": {\"message\": \"" ++ origin_message ++ "\", \"type\": \"invalid_request_error\"}}",
    };
    conn.close = true;
    conn.startResponse(403, null) catch return false;
    conn.addHeader("Content-Type", "application/json") catch return false;
    var length: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&length, "{d}", .{body.len}) catch unreachable) catch return false;
    conn.addHeader("Connection", "close") catch return false;
    conn.finish(if (std.mem.eql(u8, conn.method, "HEAD")) "" else body) catch {};
    return false;
}

test {
    _ = @import("request_guard_test.zig");
}

fn invalidTarget(path: []const u8) bool {
    for (path) |c| if (c < 0x21 or c == 0x7f) return true;
    return false;
}
