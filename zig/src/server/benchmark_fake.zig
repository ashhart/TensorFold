//! A loopback HTTP fake for the benchmark tests: scripted replies, pauses between writes, recorded requests.
const std = @import("std");
const platform = @import("sys");
const posix = std.posix;
const Allocator = std.mem.Allocator;

/// Write bytes, then wait on ``after`` (caller releases before cleanup) and pause.
pub const Write = struct { bytes: []const u8, pause_ms: u32 = 0, after: ?*std.Io.Event = null };

/// ``METHOD path`` and its replies: the n-th match plays ``replies[min(n, len - 1)]``; ``tunnel`` records bytes after.
pub const Route = struct { method: []const u8, path: []const u8, replies: []const []const Write, tunnel: bool = false };

pub const Request = struct { method: []const u8, path: []const u8, head: []const u8, body: []const u8 };

pub const Fake = struct {
    gpa: Allocator,
    fd: posix.socket_t,
    port: u16,
    routes: []const Route,
    stop_flag: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    mutex: std.Io.Mutex = .init,
    seen: std.ArrayList(Request) = .empty,
    arena: std.heap.ArenaAllocator,

    /// Listens on 127.0.0.1 at a free port and serves ``routes`` until ``stop``.
    pub fn start(gpa: Allocator, routes: []const Route) !*Fake {
        const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
        if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const fd: posix.socket_t = @intCast(rc);
        errdefer _ = posix.system.close(fd);
        const one: c_int = 1;
        try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
        var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f000001) };
        if (posix.errno(posix.system.bind(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(fd, 64)) != .SUCCESS) return error.ListenFailed;
        var bound: posix.sockaddr.in = undefined;
        var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
        if (posix.errno(posix.system.getsockname(fd, @ptrCast(&bound), &len)) != .SUCCESS) return error.BindFailed;
        const f = try gpa.create(Fake);
        f.* = .{ .gpa = gpa, .fd = fd, .port = std.mem.bigToNative(u16, bound.port), .routes = routes, .arena = .init(gpa) };
        f.thread = try std.Thread.spawn(.{}, accept, .{f});
        return f;
    }

    pub fn stop(f: *Fake) void {
        f.stop_flag.store(true, .release);
        if (f.thread) |t| t.join();
        _ = posix.system.close(f.fd);
        f.seen.deinit(f.gpa);
        f.arena.deinit();
        f.gpa.destroy(f);
    }

    /// ``http://127.0.0.1:PORT`` plus ``suffix``.
    pub fn url(f: *const Fake, a: Allocator, suffix: []const u8) ![]const u8 {
        return std.fmt.allocPrint(a, "http://127.0.0.1:{d}{s}", .{ f.port, suffix });
    }

    pub fn count(f: *Fake) usize {
        f.mutex.lockUncancelable(std.testing.io);
        defer f.mutex.unlock(std.testing.io);
        return f.seen.items.len;
    }

    pub fn request(f: *Fake, i: usize) Request {
        f.mutex.lockUncancelable(std.testing.io);
        defer f.mutex.unlock(std.testing.io);
        return f.seen.items[i];
    }

    /// The value of header ``name`` in the i-th request, or null.
    pub fn header(f: *Fake, i: usize, name: []const u8) ?[]const u8 {
        var it = std.mem.splitSequence(u8, f.request(i).head, "\r\n");
        _ = it.next();
        while (it.next()) |h| {
            const colon = std.mem.indexOfScalar(u8, h, ':') orelse continue;
            if (std.ascii.eqlIgnoreCase(h[0..colon], name)) return std.mem.trim(u8, h[colon + 1 ..], " ");
        }
        return null;
    }

    fn accept(f: *Fake) void {
        var conns: [128]std.Thread = undefined;
        var n: usize = 0;
        defer for (conns[0..n]) |t| t.join();
        while (!f.stop_flag.load(.acquire)) {
            var fds = [_]posix.pollfd{.{ .fd = f.fd, .events = posix.POLL.IN, .revents = 0 }};
            const ready = posix.poll(&fds, 20) catch return;
            if (ready == 0) continue;
            const rc = posix.system.accept(f.fd, null, null);
            if (posix.errno(rc) != .SUCCESS) continue;
            const c: posix.socket_t = @intCast(rc);
            if (n == conns.len) {
                _ = posix.system.close(c);
                continue;
            }
            conns[n] = std.Thread.spawn(.{}, handle, .{ f, c }) catch {
                _ = posix.system.close(c);
                continue;
            };
            n += 1;
        }
    }

    fn handle(f: *Fake, c: posix.socket_t) void {
        defer _ = posix.system.close(c);
        var buf: [64 << 10]u8 = undefined;
        var end: usize = 0;
        const head_end = while (true) {
            if (std.mem.indexOf(u8, buf[0..end], "\r\n\r\n")) |i| break i;
            if (end == buf.len or !f.readMore(c, &buf, &end)) return;
        };
        const head = buf[0..head_end];
        const length = contentLength(head);
        while (end < head_end + 4 + length) if (end == buf.len or !f.readMore(c, &buf, &end)) return;
        var parts = std.mem.tokenizeScalar(u8, head[0 .. std.mem.indexOf(u8, head, "\r\n") orelse head.len], ' ');
        const method = parts.next() orelse return;
        const path = parts.next() orelse return;
        const route = f.record(method, path, head, buf[head_end + 4 .. head_end + 4 + length]) orelse {
            _ = writeAll(c, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return;
        };
        for (route.replies) |w| {
            if (f.stop_flag.load(.acquire)) return;
            if (!writeAll(c, w.bytes)) return;
            if (w.after) |gate| gate.waitUncancelable(std.testing.io);
            if (w.pause_ms > 0) nap(w.pause_ms);
        }
        while (route.tunnel) {
            var rest: [4096]u8 = undefined;
            var n: usize = 0;
            if (!f.readMore(c, &rest, &n)) return;
            _ = f.record("TUNNEL", path, "", rest[0..n]);
        }
    }

    fn readMore(f: *Fake, c: posix.socket_t, buf: []u8, end: *usize) bool {
        while (!f.stop_flag.load(.acquire)) {
            var fds = [_]posix.pollfd{.{ .fd = c, .events = posix.POLL.IN, .revents = 0 }};
            const ready = posix.poll(&fds, 20) catch return false;
            if (ready == 0) continue;
            const n = posix.read(c, buf[end.*..]) catch return false;
            if (n == 0) return false;
            end.* += n;
            return true;
        }
        return false;
    }

    const Played = struct { replies: []const Write, tunnel: bool };

    fn record(f: *Fake, method: []const u8, path: []const u8, head: []const u8, body: []const u8) ?Played {
        f.mutex.lockUncancelable(std.testing.io);
        defer f.mutex.unlock(std.testing.io);
        const a = f.arena.allocator();
        const r: Request = .{ .method = a.dupe(u8, method) catch return null, .path = a.dupe(u8, path) catch return null, .head = a.dupe(u8, head) catch return null, .body = a.dupe(u8, body) catch return null };
        var before: usize = 0;
        for (f.seen.items) |s| {
            if (std.mem.eql(u8, s.method, method) and std.mem.eql(u8, s.path, path)) before += 1;
        }
        f.seen.append(f.gpa, r) catch return null;
        for (f.routes) |route| if (std.mem.eql(u8, route.method, method) and std.mem.eql(u8, route.path, path))
            return .{ .replies = route.replies[@min(before, route.replies.len - 1)], .tunnel = route.tunnel };
        return null;
    }
};

fn contentLength(head: []const u8) usize {
    var it = std.mem.splitSequence(u8, head, "\r\n");
    while (it.next()) |h| {
        const colon = std.mem.indexOfScalar(u8, h, ':') orelse continue;
        if (std.ascii.eqlIgnoreCase(h[0..colon], "content-length")) return std.fmt.parseInt(usize, std.mem.trim(u8, h[colon + 1 ..], " "), 10) catch 0;
    }
    return 0;
}

fn writeAll(c: posix.socket_t, bytes: []const u8) bool {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = posix.system.write(c, bytes[at..].ptr, bytes.len - at);
        if (posix.errno(n) != .SUCCESS or n <= 0) return false;
        at += @intCast(n);
    }
    return true;
}

/// A pseudo-terminal pair for the hidden-prompt tests; ``open`` gives null where the system has none.
pub const Pty = struct {
    master: posix.fd_t,
    term: posix.fd_t,

    extern "c" fn posix_openpt(flags: c_int) c_int;
    extern "c" fn grantpt(fd: c_int) c_int;
    extern "c" fn unlockpt(fd: c_int) c_int;
    extern "c" fn ptsname(fd: c_int) ?[*:0]const u8;

    pub fn open() ?Pty {
        const master = posix_openpt(@bitCast(posix.O{ .ACCMODE = .RDWR, .NOCTTY = true }));
        if (master < 0) return null;
        const name = if (grantpt(master) == 0 and unlockpt(master) == 0) ptsname(master) else null;
        const term = if (name) |n| platform.c.open(n, .{ .ACCMODE = .RDWR, .NOCTTY = true }) else -1;
        if (term < 0) {
            _ = platform.c.close(master);
            return null;
        }
        return .{ .master = master, .term = term };
    }

    pub fn close(p: Pty) void {
        _ = platform.c.close(p.term);
        _ = platform.c.close(p.master);
    }
};

pub fn nap(ms: u32) void {
    var ts: posix.timespec = .{ .sec = @intCast(ms / 1000), .nsec = @intCast(@as(u64, ms % 1000) * std.time.ns_per_ms) };
    _ = platform.c.nanosleep(&ts, null);
}

/// A response head with ``Connection: close`` and no length, as the native server sends SSE.
pub fn headOf(a: Allocator, status: []const u8, content_type: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nConnection: close\r\n\r\n", .{ status, content_type });
}

/// One whole JSON reply with its length.
pub fn jsonReply(a: Allocator, status: []const u8, body: []const u8) ![]const Write {
    const w = try a.alloc(Write, 1);
    w[0] = .{ .bytes = try std.fmt.allocPrint(a, "HTTP/1.1 {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ status, body.len, body }) };
    return w;
}

/// ``data: TEXT`` and the blank line that dispatches it.
pub fn event(a: Allocator, data: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "data: {s}\n\n", .{data});
}

/// A test tmp dir's absolute path.
pub fn tmpPath(a: Allocator, tmp: *std.testing.TmpDir) ![]const u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(std.testing.io, &buf);
    return a.dupe(u8, buf[0..n]);
}

/// A completion stream of two text deliveries ``gap_ms`` apart, a finish, usage and DONE.
pub fn reply(a: Allocator, chat: bool, completion: u64, gap_ms: u32) ![]const Write {
    const first = if (chat) "{\"choices\": [{\"index\": 0, \"delta\": {\"reasoning_content\": \"synthetic\"}}]}" else "{\"choices\": [{\"index\": 0, \"text\": \"synthetic\"}]}";
    const second = if (chat) "{\"choices\": [{\"index\": 0, \"delta\": {\"content\": \" output\"}}]}" else "{\"choices\": [{\"index\": 0, \"text\": \" output\"}]}";
    const final = "{\"choices\": [{\"index\": 0, \"delta\": {}, \"finish_reason\": \"length\"}], \"tensorfold\": {\"tokens_per_second\": 42.0, \"prefill_seconds\": 0.03, \"seconds\": 0.2, \"token_sha\": \"0123456789ab\"}}";
    const usage = try std.fmt.allocPrint(a, "{{\"choices\": [], \"usage\": {{\"prompt_tokens\": 24, \"completion_tokens\": {d}, \"prompt_tokens_details\": {{\"cached_tokens\": 0}}}}}}", .{completion});
    const w = try a.alloc(Write, 3);
    w[0] = .{ .bytes = try std.mem.concat(a, u8, &.{ try headOf(a, "200 OK", "text/event-stream"), ": keepalive\n\n", try event(a, first) }), .pause_ms = gap_ms };
    w[1] = .{ .bytes = try event(a, second), .pause_ms = 0 };
    w[2] = .{ .bytes = try std.mem.concat(a, u8, &.{ try event(a, final), try event(a, usage), try event(a, "[DONE]") }) };
    return w;
}
