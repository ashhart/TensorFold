//! ``--listen-fd``: the only way in, refused beside the others; it serves one process and stays the parent's.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const cli = @import("cli.zig");
const listen_fd = @import("listen_fd.zig");
const listener = @import("listener.zig");
const server = @import("server.zig");
const witness = @import("witness.zig");
const ByteText = @import("embedding_protocol_test.zig").ByteText;
const t = std.testing;

test "--listen-fd parses a descriptor and is refused beside another way in, a key, a negative number and on Windows" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    if (builtin.os.tag == .windows) {
        try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--listen-fd", "3" }, &u));
        try t.expectEqualStrings("argument --listen-fd: not served on Windows", u.message);
        return;
    }
    try t.expectEqual(@as(?i64, 3), (try cli.parse(a, &.{ "m", "--listen-fd", "3" }, &u)).listen_fd);
    try t.expectEqual(@as(?i64, null), (try cli.parse(a, &.{"m"}, &u)).listen_fd);
    const refused = [_]struct { argv: []const []const u8, other: []const u8 }{
        .{ .argv = &.{ "--unix-socket", "/tmp/s.sock" }, .other = "--unix-socket" },
        .{ .argv = &.{ "--host", "127.0.0.1" }, .other = "--host" },
        .{ .argv = &.{"--port=8081"}, .other = "--port" },
        .{ .argv = &.{ "--api-key", "sk-one" }, .other = "--api-key" },
        .{ .argv = &.{ "--api-key-file", "/tmp/keys" }, .other = "--api-key-file" },
    };
    for (refused) |case| {
        const argv = try std.mem.concat(a, []const u8, &.{ &.{ "m", "--listen-fd=3" }, case.argv });
        try t.expectError(error.Usage, cli.parse(a, argv, &u));
        try t.expectEqualStrings(try std.fmt.allocPrint(a, "argument --listen-fd: not allowed with {s}", .{case.other}), u.message);
    }
    try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--listen-fd", "-1" }, &u));
    try t.expectEqualStrings("argument --listen-fd: expected a descriptor number from 0: '-1'", u.message);
    try t.expectError(error.Usage, cli.parse(a, &.{ "m", "--listen-fd", "fd" }, &u));
}

test "--listen-fd refuses TENSORFOLD_API_KEY and lets --execution-witness start without a key" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var u: cli.Usage = .{};
    const fd = try cli.parse(a, &.{ "m", "--listen-fd", "3", "--execution-witness" }, &u);
    try t.expectEqual(@as(?[]const u8, null), listen_fd.keyRefusal(fd, ""));
    try t.expectEqual(@as(?[]const u8, null), listen_fd.keyRefusal(fd, " , \n"));
    try t.expect(std.mem.indexOf(u8, listen_fd.keyRefusal(fd, "sk-env").?, "TENSORFOLD_API_KEY") != null);
    try t.expectEqual(@as(?[]const u8, null), listen_fd.keyRefusal(try cli.parse(a, &.{"m"}, &u), "sk-env"));
    try t.expectEqual(@as(?[]const u8, null), witness.refusal(fd, false));
    const plain = try cli.parse(a, &.{ "m", "--execution-witness" }, &u);
    try t.expect(std.mem.indexOf(u8, witness.refusal(plain, false).?, "--listen-fd") != null);
}

test "an inherited descriptor must be an open, bound Unix stream socket, and is closed on exec" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const c = std.c;
    try t.expectError(error.NotOpen, listener.Listener.inherit(std.math.maxInt(i32)));
    const file = c.open("/dev/null", .{ .ACCMODE = .RDONLY });
    try t.expect(file >= 0);
    defer _ = c.close(file);
    try t.expectError(error.NotSocket, listener.Listener.inherit(file));
    var pair: [2]c.fd_t = undefined;
    try t.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &pair));
    defer for (pair) |p| {
        _ = c.close(p);
    };
    try t.expectError(error.CannotListen, listener.Listener.inherit(pair[0]));
    const datagram = c.socket(c.AF.UNIX, c.SOCK.DGRAM, 0);
    try t.expect(datagram >= 0);
    defer _ = c.close(datagram);
    try t.expectError(error.NotStream, listener.Listener.inherit(datagram));
    const tcp = try listener.Listener.open(.{ .ip4 = .loopback(0) });
    defer tcp.close();
    try t.expectError(error.NotUnix, listener.Listener.inherit(tcp.fd));
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/tf-listen-fd-{d}.sock", .{c.getpid()}, 0);
    const fd = try bound(path);
    defer _ = c.unlink(path);
    defer _ = c.close(fd);
    const lis = try listener.Listener.inherit(fd);
    try t.expect(lis.inherited and lis.boundAddress() == null and lis.port() == 0);
    try t.expect(c.fcntl(fd, c.F.GETFD) & c.FD_CLOEXEC != 0);
    // bound but not yet listening: serve listens on it
    const quiet = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
    defer _ = c.close(quiet);
    var addr: c.sockaddr.un = .{ .path = undefined };
    @memset(&addr.path, 0);
    const other = try std.fmt.bufPrint(&addr.path, "/tmp/tf-listen-fd-quiet-{d}.sock", .{c.getpid()});
    try t.expectEqual(@as(c_int, 0), c.bind(quiet, @ptrCast(&addr), @sizeOf(c.sockaddr.un)));
    defer _ = c.unlink(@ptrCast(other.ptr));
    _ = try listener.Listener.inherit(quiet);
    const client = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
    defer _ = c.close(client);
    try t.expectEqual(@as(c_int, 0), c.connect(client, @ptrCast(&addr), @sizeOf(c.sockaddr.un)));
}

test "an inherited socket answers its one process, refuses another's connection and stays open and named for the parent" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const c = std.c;
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/tf-listen-fd-http-{d}.sock", .{c.getpid()}, 0);
    defer _ = c.unlink(path);
    // this process connects: as the one served it gets the reply, as another (the parent's pid stands in) none
    for ([_]c.pid_t{ c.getpid(), c.getppid() }, [_]bool{ true, false }) |only, served| {
        const fd = try bound(path);
        var parent_copy = c.dup(fd); // the parent keeps its own descriptor of the same socket
        defer _ = c.close(parent_copy);
        var engine: Idle = .{};
        const srv = try server.Server.init(t.allocator, t.io, engine.engine(), ByteText.model(), .{ .served_name = "m", .model_ids = &.{"m"} }, null);
        defer srv.deinit();
        var lis = try listener.Listener.inherit(fd);
        lis.only_pid = only;
        var stop: std.atomic.Value(bool) = .init(false);
        const accept = try std.Thread.spawn(.{}, server.Server.serve, .{ srv, lis, &stop });
        const reply = try exchange(t.allocator, path, "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        defer t.allocator.free(reply);
        stop.store(true, .release);
        accept.join();
        while (srv.open_connections.load(.acquire) > 0) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        lis.close();
        if (served) try t.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 ")) else try t.expectEqual(@as(usize, 0), reply.len);
        // closing the engine's copy neither shut the socket down nor removed its path: the parent can serve it again
        try t.expectEqual(@as(c_int, 0), c.access(path, c.F_OK));
        const again = try listener.Listener.inherit(parent_copy);
        try t.expect(again.inherited);
        _ = c.close(parent_copy);
        parent_copy = -1;
        _ = c.unlink(path);
    }
}

/// A listening Unix stream socket at `path`, as a parent would pass it.
fn bound(path: [:0]const u8) !std.c.fd_t {
    const c = std.c;
    var addr: c.sockaddr.un = .{ .path = undefined };
    @memset(&addr.path, 0);
    @memcpy(addr.path[0..path.len], path);
    const fd = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
    if (fd < 0) return error.Socket;
    errdefer _ = c.close(fd);
    if (c.bind(fd, @ptrCast(&addr), @sizeOf(c.sockaddr.un)) != 0 or c.listen(fd, 8) != 0) return error.Listen;
    return fd;
}

/// One request over the socket at `path`; the bytes read until the server closes the connection (none when refused).
fn exchange(a: std.mem.Allocator, path: [:0]const u8, request: []const u8) ![]u8 {
    const c = std.c;
    var addr: c.sockaddr.un = .{ .path = undefined };
    @memset(&addr.path, 0);
    @memcpy(addr.path[0..path.len], path);
    const fd = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
    if (fd < 0) return error.Socket;
    defer _ = c.close(fd);
    if (builtin.os.tag.isDarwin()) {
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.NOSIGPIPE, &one, @sizeOf(c_int)); // a refused peer's close is no signal
    }
    if (c.connect(fd, @ptrCast(&addr), @sizeOf(c.sockaddr.un)) != 0) return error.Connect;
    _ = c.send(fd, request.ptr, request.len, if (builtin.os.tag == .linux) c.MSG.NOSIGNAL else 0);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = c.read(fd, &buf, buf.len);
        if (n <= 0) break;
        try out.appendSlice(a, buf[0..@intCast(n)]);
    }
    return out.toOwnedSlice(a);
}

/// Healthy, with nothing to run.
const Idle = struct {
    fn engine(e: *Idle) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .lanes = 1 };
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Closed;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{ .healthy = true };
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};
