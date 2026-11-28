//! A CONNECT tunnel through an http:// proxy, opened below std's HTTP and TLS so both run through it unchanged.
const std = @import("std");
const route = @import("pull_route.zig");
const http = @import("pull_http.zig");
const Io = std.Io;
const net = Io.net;
const Error = net.IpAddress.ConnectError;

/// A socket to the proxy that now carries bytes to `t`'s host and port; refusals are recorded for pull's message.
pub fn open(io: Io, t: route.Tunnel) Error!net.Socket {
    const p = t.proxy;
    const socket = try reach(io, p, t.deadline);
    const stream: net.Stream = .{ .socket = socket };
    errdefer stream.close(io);
    const status = handshake(io, stream, t) catch |err| {
        route.fail(io, if (err == error.Timeout) .{ .proxy_silent = p } else .{ .proxy_unreachable = .{ .proxy = p, .err = err } });
        return if (err == error.Timeout) error.Timeout else error.ConnectionResetByPeer;
    };
    if (status / 100 == 2) return socket;
    route.fail(io, .{ .proxy_status = .{ .proxy = p, .status = status } });
    return if (status == 407) error.AccessDenied else error.ConnectionRefused;
}

/// The first of the proxy's addresses that accepts within the original connect deadline.
fn reach(io: Io, p: route.Proxy, deadline: Io.Clock.Timestamp) Error!net.Socket {
    return reachWith(http.nativeLookup, io.userdata, io, p, deadline);
}

pub fn reachWith(resolve: http.Resolver, userdata: ?*anyopaque, io: Io, p: route.Proxy, deadline: Io.Clock.Timestamp) Error!net.Socket {
    const name = net.HostName.init(p.host) catch |err| return unreachableProxy(io, p, err);
    var buf: [16]net.HostName.LookupResult = undefined;
    var results: Io.Queue(net.HostName.LookupResult) = .init(&buf);
    http.lookupTimed(resolve, userdata, io, name, &results, .{ .port = p.port }, deadline) catch |err| switch (err) {
        error.Timeout => {
            route.fail(io, .{ .connect_timeout = p });
            return error.Timeout;
        },
        else => return unreachableProxy(io, p, err),
    };
    var last: Error = error.HostUnreachable;
    while (results.getOne(io)) |r| switch (r) {
        .address => |address| if (http.timed(io, &address, p, deadline)) |socket| return socket else |err| {
            last = err;
        },
        .canonical_name => {},
    } else |err| switch (err) {
        error.Closed => {},
        error.Canceled => return error.Canceled,
    }
    if (last != error.Timeout) route.fail(io, .{ .proxy_unreachable = .{ .proxy = p, .err = last } });
    return last;
}

fn unreachableProxy(io: Io, p: route.Proxy, err: anyerror) Error {
    route.fail(io, .{ .proxy_unreachable = .{ .proxy = p, .err = err } });
    return error.HostUnreachable;
}

/// Sends CONNECT (with Proxy-Authorization when the proxy URL has credentials) and returns the reply's status.
fn handshake(io: Io, stream: net.Stream, t: route.Tunnel) !u16 {
    const host = t.host[0..t.len];
    const v6 = std.mem.indexOfScalar(u8, host, ':') != null;
    const open_b, const close_b = if (v6) .{ "[", "]" } else .{ "", "" };
    var out_buf: [1024]u8 = undefined;
    var w = stream.writer(io, &out_buf);
    const o = &w.interface;
    o.print("CONNECT {s}{s}{s}:{d} HTTP/1.1\r\nHost: {s}{s}{s}:{d}\r\n", .{ open_b, host, close_b, t.port, open_b, host, close_b, t.port }) catch return error.WriteFailed;
    if (t.proxy.auth) |v| o.print("Proxy-Authorization: {s}\r\n", .{v}) catch return error.WriteFailed;
    o.writeAll("\r\n") catch return error.WriteFailed;
    o.flush() catch return error.WriteFailed;
    var in_buf: [4096]u8 = undefined;
    var r = stream.reader(io, &in_buf);
    const first = r.interface.takeDelimiterInclusive('\n') catch return readError(&r);
    if (first.len < 12 or !std.mem.startsWith(u8, first, "HTTP/1.")) return error.BadProxyReply;
    const status = std.fmt.parseInt(u16, first[9..12], 10) catch return error.BadProxyReply;
    while (true) {
        const line = r.interface.takeDelimiterInclusive('\n') catch return readError(&r);
        if (std.mem.trimEnd(u8, line, "\r\n").len == 0) break;
    }
    if (r.interface.bufferedLen() != 0) return error.BadProxyReply; // nothing comes before the client speaks
    return status;
}

fn readError(r: *net.Stream.Reader) anyerror {
    if (r.err) |err| if (err == error.ConnectionTimedOut) return error.Timeout;
    return error.ProxyClosed;
}
