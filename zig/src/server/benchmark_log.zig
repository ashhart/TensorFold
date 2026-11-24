//! A managed child's output reaches its log file only through the parent: read from a pipe, keys replaced first.
const std = @import("std");
const sse = @import("benchmark_sse.zig");
const posix = std.posix;

/// Held bytes past this length are written as one replacement instead of waiting for more.
const hold_limit = 1 << 20;

pub const Sink = struct {
    io: std.Io,
    log: std.Io.File,
    read_fd: posix.fd_t,
    /// The child's stdout and stderr; the parent closes its own copy once the child holds it.
    child: std.Io.File,
    secrets: []const []const u8,
    offset: u64 = 0,
    stop: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    /// A pipe in front of `log`; `secrets` (raw keys) are replaced wherever they or their encodings occur.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, log: std.Io.File, secrets: []const []const u8) !*Sink {
        var fds: [2]posix.fd_t = undefined;
        if (std.c.pipe(&fds) != 0) return error.PipeFailed;
        for (fds) |fd| _ = std.c.fcntl(fd, posix.F.SETFD, @as(c_int, posix.FD_CLOEXEC));
        const s = gpa.create(Sink) catch |e| {
            for (fds) |fd| _ = std.c.close(fd);
            return e;
        };
        s.* = .{ .io = io, .log = log, .read_fd = fds[0], .child = .{ .handle = fds[1], .flags = .{ .nonblocking = false } }, .secrets = secrets };
        return s;
    }

    /// The child holds its end now: the parent's copy closes, so the child's exit ends the stream.
    pub fn started(s: *Sink) !void {
        _ = std.c.close(s.child.handle);
        s.thread = try std.Thread.spawn(.{}, run, .{s});
    }

    /// Waits up to `ms` for the stream to end (the child exited and its output is written).
    pub fn settle(s: *Sink, ms: u32) void {
        var waited: u32 = 0;
        while (!s.done.load(.acquire) and waited < ms) : (waited += 10) s.io.sleep(.fromMilliseconds(10), .awake) catch {};
    }

    /// Stops reading once nothing more is ready, then closes the pipe; the log stays open for its owner.
    pub fn close(s: *Sink, gpa: std.mem.Allocator) void {
        s.stop.store(true, .release);
        if (s.thread) |t| t.join() else _ = std.c.close(s.child.handle);
        _ = std.c.close(s.read_fd);
        gpa.destroy(s);
    }

    fn append(s: *Sink, bytes: []const u8) void {
        if (bytes.len == 0) return;
        s.log.writePositionalAll(s.io, bytes, s.offset) catch return;
        s.offset += bytes.len;
    }

    fn run(s: *Sink) void {
        defer s.done.store(true, .release);
        var held: std.ArrayList(u8) = .empty;
        defer held.deinit(std.heap.page_allocator);
        var buf: [64 << 10]u8 = undefined;
        var fds = [_]posix.pollfd{.{ .fd = s.read_fd, .events = posix.POLL.IN, .revents = 0 }};
        var after_stop: u32 = 0;
        while (true) {
            const stopping = s.stop.load(.acquire);
            if (stopping) after_stop += 1;
            const ready = posix.poll(&fds, if (stopping) 0 else 100) catch break;
            if (ready == 0) {
                if (stopping) break;
                continue;
            }
            if (after_stop > 256) break; // a writer that never pauses cannot hold the stop
            const n = posix.read(s.read_fd, &buf) catch break;
            if (n == 0) break;
            if (s.secrets.len == 0) {
                s.append(buf[0..n]);
                continue;
            }
            held.appendSlice(std.heap.page_allocator, buf[0..n]) catch break;
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const piece = sse.redactPiece(arena.allocator(), held.items, s.secrets) catch {
                s.append("[redacted]");
                held.clearRetainingCapacity();
                continue;
            };
            s.append(piece.text);
            const rest = piece.rest.len;
            std.mem.copyForwards(u8, held.items[0..rest], piece.rest);
            held.shrinkRetainingCapacity(rest);
            if (held.items.len > hold_limit) {
                s.append("[redacted]");
                held.clearRetainingCapacity();
            }
        }
        // a suffix still held could become a key: its place is marked, its bytes are not written
        if (held.items.len > 0) s.append("[redacted]");
    }
};

test "benchmark log: a key written to the child's stream reaches the file only as a replacement, at any split" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const secret = "fixture-log-key";
    const text = "start\nkey=fixture-log-key encoded=fixture%2Dlog%2Dkey\nready\n";
    for (0..text.len) |split| {
        const log = try tmp.dir.createFile(io, "child.log", .{ .read = true, .truncate = true });
        defer log.close(io);
        const s = try Sink.open(std.testing.allocator, io, log, &.{secret});
        const writer = std.c.dup(s.child.handle);
        try s.started();
        _ = std.c.write(writer, text.ptr, split);
        _ = std.c.write(writer, text[split..].ptr, text.len - split);
        _ = std.c.close(writer);
        s.settle(5000);
        s.close(std.testing.allocator);
        var out: [512]u8 = undefined;
        const got = out[0..try log.readPositionalAll(io, &out, 0)];
        try std.testing.expect(std.mem.indexOf(u8, got, secret) == null);
        try std.testing.expect(std.mem.indexOf(u8, got, "fixture%2Dlog") == null);
        try std.testing.expect(std.mem.startsWith(u8, got, "start\nkey=[redacted]"));
        try std.testing.expect(std.mem.endsWith(u8, got, "ready\n"));
    }
}

test "benchmark log: without keys the child's bytes reach the file unchanged" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const log = try tmp.dir.createFile(io, "child.log", .{ .read = true });
    defer log.close(io);
    const s = try Sink.open(std.testing.allocator, io, log, &.{});
    const writer = std.c.dup(s.child.handle);
    try s.started();
    _ = std.c.write(writer, "plain output\n", 13);
    _ = std.c.close(writer);
    s.settle(5000);
    s.close(std.testing.allocator);
    var out: [64]u8 = undefined;
    try std.testing.expectEqualStrings("plain output\n", out[0..try log.readPositionalAll(io, &out, 0)]);
}
