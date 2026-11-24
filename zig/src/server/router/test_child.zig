//! The router tests' children: a /bin/sh stand-in that prints serve's line for a fake server in the test process.
const std = @import("std");
const fake = @import("../benchmark_fake.zig");
const catalog = @import("catalog.zig");
const pool_mod = @import("pool.zig");
const posix = std.posix;
const Allocator = std.mem.Allocator;

pub const Fake = fake.Fake;
pub const Route = fake.Route;
pub const Write = fake.Write;

/// One fake model: the server its requests reach, and the entry whose child announces that server's port.
pub const Child = struct { server: *fake.Fake, entry: catalog.Entry };

/// A fake child for ``id`` and ``aliases``: ready at once, then answering ``routes`` as scripted.
pub fn start(a: Allocator, id: []const u8, aliases: []const []const u8, routes: []const fake.Route) !Child {
    var listed: std.ArrayList(u8) = .empty;
    try listed.appendSlice(a, "{\"object\": \"list\", \"data\": [");
    for ([_][]const []const u8{ &.{id}, aliases }) |names| for (names) |name| {
        if (listed.items[listed.items.len - 1] == '}') try listed.appendSlice(a, ", ");
        try listed.print(a, "{{\"id\": \"{s}\", \"object\": \"model\", \"created\": 1, \"owned_by\": \"tensorfold\", \"context_length\": 4096}}", .{name});
    };
    try listed.appendSlice(a, "]}");
    var all: std.ArrayList(fake.Route) = .empty;
    try all.append(a, .{ .method = "GET", .path = "/health", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", "{\"status\": \"ok\"}")}) });
    try all.append(a, .{ .method = "GET", .path = "/v1/models", .replies = try a.dupe([]const fake.Write, &.{try fake.jsonReply(a, "200 OK", listed.items)}) });
    try all.appendSlice(a, routes);
    const server = try fake.Fake.start(std.testing.allocator, all.items);
    const script = try std.fmt.allocPrint(a, "printf '[tensorfold] serving %s at http://127.0.0.1:%s/v1 (x)\\n' '{s}' {d}; exec sleep 60", .{ id, server.port });
    return .{ .server = server, .entry = .{ .id = id, .aliases = aliases, .argv = try a.dupe([]const u8, &.{ "/bin/sh", "-c", script }), .startup_s = 10 } };
}

/// A child that exits during its load after printing ``line``.
pub fn failing(a: Allocator, id: []const u8, line: []const u8) !catalog.Entry {
    const script = try std.fmt.allocPrint(a, "echo '{s}'; exit 3", .{line});
    return .{ .id = id, .argv = try a.dupe([]const u8, &.{ "/bin/sh", "-c", script }), .startup_s = 10 };
}

/// A child that never says it is serving.
pub fn silent(a: Allocator, id: []const u8, startup_s: f64) !catalog.Entry {
    return .{ .id = id, .argv = try a.dupe([]const u8, &.{ "/bin/sh", "-c", "exec sleep 60" }), .startup_s = startup_s };
}

/// A pool over ``entries`` whose children log under a test folder and stop within two seconds.
pub fn pool(a: Allocator, tmp: *std.testing.TmpDir, entries: []const catalog.Entry, options: pool_mod.Options) !*pool_mod.Pool {
    const env = try a.create(std.process.Environ.Map);
    env.* = .init(a);
    try env.put("PATH", "/usr/bin:/bin");
    var o = options;
    o.log_dir = try std.fs.path.join(a, &.{ try fake.tmpPath(a, tmp), "router" });
    o.grace_ms = 2_000;
    return pool_mod.Pool.init(std.testing.allocator, std.testing.io, entries, env, o);
}

/// Whether ``process`` is no longer a process.
pub fn gone(process: posix.pid_t) bool {
    posix.kill(process, @fromBackingInt(0)) catch |e| return e == error.ProcessNotFound;
    return false;
}

/// The running child's pid in slot ``i``.
pub fn pid(p: *pool_mod.Pool, i: usize) ?posix.pid_t {
    p.mutex.lockUncancelable(p.io);
    defer p.mutex.unlock(p.io);
    return if (p.slots[i].child) |m| m.pid else null;
}

/// Slot ``i``'s state.
pub fn state(p: *pool_mod.Pool, i: usize) @import("room.zig").State {
    p.mutex.lockUncancelable(p.io);
    defer p.mutex.unlock(p.io);
    return p.slots[i].state;
}

/// Seconds since ``t0`` on the awake clock.
pub fn since(t0: i96) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - t0)) / 1e9;
}

pub fn now() i96 {
    return std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
}

/// Waits up to ``ms`` for ``f(ctx)``; false when it never held.
pub fn within(ms: u32, ctx: anytype, comptime f: fn (@TypeOf(ctx)) bool) bool {
    var waited: u32 = 0;
    while (waited < ms) : (waited += 10) {
        if (f(ctx)) return true;
        fake.nap(10);
    }
    return f(ctx);
}

/// One whole exchange on a new connection to ``port``, read until the other side closes it.
pub fn exchangeWith(a: std.mem.Allocator, port: u16, request: []const u8) ![]const u8 {
    const fd = try dial(port);
    defer _ = posix.system.close(fd);
    try sendAll(fd, request);
    var out: std.ArrayList(u8) = .empty;
    try readUntil(a, fd, &out, null, 10_000);
    return out.items;
}

pub fn dial(port: u16) !posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.ConnectFailed;
    return fd;
}

pub fn sendAll(fd: posix.socket_t, bytes: []const u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = posix.system.write(fd, bytes[at..].ptr, bytes.len - at);
        if (posix.errno(n) != .SUCCESS or n <= 0) return error.WriteFailed;
        at += @intCast(n);
    }
}

/// Reads into ``out`` until ``want`` bytes are there (null: until EOF), failing after ``ms`` without them.
pub fn readUntil(a: std.mem.Allocator, fd: posix.socket_t, out: *std.ArrayList(u8), want: ?usize, ms: u32) !void {
    var waited: u32 = 0;
    while (want == null or out.items.len < want.?) {
        var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.IN, .revents = 0 }};
        if (try posix.poll(&fds, 20) == 0) {
            waited += 20;
            if (waited >= ms) return error.Timeout;
            continue;
        }
        var buf: [4096]u8 = undefined;
        const n = posix.read(fd, &buf) catch 0;
        if (n == 0) return if (want == null) {} else error.EndOfStream;
        try out.appendSlice(a, buf[0..n]);
    }
}

pub fn post(a: std.mem.Allocator, path: []const u8, body: []const u8, extra: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "POST {s} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n{s}\r\n{s}", .{ path, body.len, extra, body });
}

pub fn get(a: std.mem.Allocator, path: []const u8, extra: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "GET {s} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n{s}\r\n", .{ path, extra });
}

pub fn status(reply: []const u8) u16 {
    if (reply.len < 12) return 0;
    return std.fmt.parseInt(u16, reply[9..12], 10) catch 0;
}

pub fn bodyOf(reply: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return "";
    return reply[at + 4 ..];
}
