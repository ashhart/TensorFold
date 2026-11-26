//! ``--listen-fd``: the only way in, refused beside the others; it serves the process that passed it, while that lives.
const std = @import("std");
const builtin = @import("builtin");
const cli = @import("cli.zig");
const listen_fd = @import("listen_fd.zig");
const listener = @import("listener.zig");
const witness = @import("witness.zig");
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
    try t.expectError(error.NotBound, listener.Listener.inherit(pair[0]));
    // never bound: refused before listen, which on Linux would bind it to an abstract name of its own
    const bare = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
    try t.expect(bare >= 0);
    defer _ = c.close(bare);
    try t.expectError(error.NotBound, listener.Listener.inherit(bare));
    try t.expectError(error.NotBound, listener.Listener.inherit(bare));
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

test "an inherited socket admits the process that passed it and refuses that process's other child" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const c = std.c;
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/tf-listen-fd-admit-{d}.sock", .{c.getpid()}, 0);
    defer _ = c.unlink(path);
    const fd = try bound(path);
    defer _ = c.close(fd);
    var report: [2]c.fd_t = undefined;
    try t.expectEqual(@as(c_int, 0), c.pipe(&report));
    defer _ = c.close(report[0]);
    const server_pid = c.fork();
    if (server_pid == 0) serveChild(fd, report[1], null, 1500);
    _ = c.close(report[1]);
    _ = try readPid(report[0]);
    var buf: [16]u8 = undefined;
    try t.expectEqualStrings("ok", buf[0..ask(path, &buf)]);
    const sibling = c.fork(); // another child of the same parent: not the process that passed the socket
    if (sibling == 0) c._exit(if (ask(path, &buf) == 0) 0 else 1);
    try t.expectEqual(@as(u32, 0), exitCode(sibling));
    try t.expectEqual(@as(u32, 0), exitCode(server_pid));
    try t.expectEqual(@as(isize, 1), c.read(report[0], &buf, 1));
    try t.expectEqual(@as(u8, 'a'), buf[0]); // its parent still lived when it stopped
}

test "the parent's exit closes admission: the process that adopts the server is refused" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const c = std.c;
    const linux = builtin.os.tag == .linux;
    // on Linux this process adopts the orphaned server, so the adopter itself connects; elsewhere launchd adopts it
    if (linux) _ = std.os.linux.prctl(@backingInt(std.os.linux.PR.SET_CHILD_SUBREAPER), 1, 0, 0, 0);
    defer if (linux) {
        _ = std.os.linux.prctl(@backingInt(std.os.linux.PR.SET_CHILD_SUBREAPER), 0, 0, 0, 0);
    };
    var path_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/tf-listen-fd-orphan-{d}.sock", .{c.getpid()}, 0);
    defer _ = c.unlink(path);
    const fd = try bound(path);
    defer _ = c.close(fd);
    var report: [2]c.fd_t = undefined;
    try t.expectEqual(@as(c_int, 0), c.pipe(&report));
    defer _ = c.close(report[0]);
    const middle = c.fork(); // the parent that passes the socket, then exits once the server holds it
    if (middle == 0) {
        var taken: [2]c.fd_t = undefined;
        if (c.pipe(&taken) != 0) c._exit(2);
        if (c.fork() == 0) serveChild(fd, report[1], taken[1], 2500);
        _ = c.close(taken[1]);
        var byte: [1]u8 = undefined;
        c._exit(if (c.read(taken[0], &byte, 1) == 1) 0 else 3);
    }
    _ = c.close(report[1]);
    try t.expectEqual(@as(u32, 0), exitCode(middle));
    const server_pid = try readPid(report[0]);
    var buf: [16]u8 = undefined;
    try t.expectEqual(@as(usize, 0), ask(path, &buf));
    try t.expectEqual(@as(isize, 1), c.read(report[0], &buf, 1));
    try t.expectEqual(@as(u8, 'g'), buf[0]); // it saw its parent gone
    if (linux) _ = exitCode(server_pid);
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

/// A child serving `fd` as --listen-fd does for `ms` ms: "ok" to each connection it admits, then whether its parent lived.
fn serveChild(fd: std.c.fd_t, report: std.c.fd_t, taken: ?std.c.fd_t, ms: u32) noreturn {
    const c = std.c;
    const lis = listener.Listener.inherit(fd) catch c._exit(4);
    const pid: [4]u8 = @bitCast(@as(i32, c.getpid()));
    _ = c.write(report, &pid, pid.len);
    if (taken) |w| _ = c.write(w, "t", 1);
    var stop: std.atomic.Value(bool) = .init(false);
    const timer = std.Thread.spawn(.{}, stopAfter, .{ &stop, ms }) catch c._exit(5);
    while (lis.accept(&stop)) |conn| {
        _ = c.write(conn.fd, "ok", 2);
        _ = c.close(conn.fd);
    }
    timer.join();
    _ = c.write(report, if (lis.parentAlive()) "a" else "g", 1);
    c._exit(0);
}

fn stopAfter(stop: *std.atomic.Value(bool), ms: u32) void {
    std.Io.sleep(t.io, .fromMilliseconds(ms), .awake) catch {};
    stop.store(true, .release);
}

/// The server child's pid, written once it holds the socket.
fn readPid(fd: std.c.fd_t) !i32 {
    var pid: [4]u8 = undefined;
    if (std.c.read(fd, &pid, pid.len) != pid.len) return error.ServerDidNotStart;
    return @bitCast(pid);
}

/// A child's exit code, once it ends.
fn exitCode(pid: i32) u32 {
    var status: c_int = 0;
    _ = std.c.waitpid(pid, &status, 0);
    return std.posix.W.EXITSTATUS(@bitCast(status));
}

/// Connects to `path` and reads what the server writes until it closes (nothing when refused), at most 5 s.
fn ask(path: [:0]const u8, buf: []u8) usize {
    const c = std.c;
    var addr: c.sockaddr.un = .{ .path = undefined };
    @memset(&addr.path, 0);
    @memcpy(addr.path[0..path.len], path);
    const fd = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
    if (fd < 0) return 0;
    defer _ = c.close(fd);
    const limit: c.timeval = .{ .sec = 5, .usec = 0 };
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.RCVTIMEO, &limit, @sizeOf(c.timeval));
    if (c.connect(fd, @ptrCast(&addr), @sizeOf(c.sockaddr.un)) != 0) return 0;
    var n: usize = 0;
    while (n < buf.len) {
        const got = c.read(fd, buf[n..].ptr, buf.len - n);
        if (got <= 0) break;
        n += @intCast(got);
    }
    return n;
}
