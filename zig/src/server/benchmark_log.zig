//! A managed child's output reaches its log file only through the parent: read from a pipe, a line at a time.
const std = @import("std");
const sse = @import("benchmark_sse.zig");
const credential = @import("sys").credential;
const posix = std.posix;

/// A line past this length is withheld through its newline instead of waiting for more.
const hold_limit = 1 << 20;

pub const Sink = struct {
    io: std.Io,
    log: std.Io.File,
    read_fd: posix.fd_t,
    /// The child's stdout and stderr; the parent closes its own copy once the child holds it.
    child: std.Io.File,
    secrets: []const []const u8,
    served: []const u8,
    offset: u64 = 0,
    /// Tests only: the `at`-th write (from 1) puts its first `keep` bytes in the file, then fails, once.
    fault: struct { at: u32 = 0, keep: usize = 0 } = .{},
    writes: u32 = 0,
    /// With keys, a write failed: nothing more is written, so the file stays a prefix of the checked lines.
    broken: bool = false,
    /// 1: the parent asks for the held last line now; 2: it is written (an exited child's tail is then whole).
    cut: std.atomic.Value(u8) = .init(0),
    /// Serve's port from its serving line, read before redaction and never from the file (0: not yet).
    port: std.atomic.Value(u16) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    /// A pipe in front of `log`; `secrets` (raw keys) are replaced wherever they or their encodings occur.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, log: std.Io.File, secrets: []const []const u8, served: []const u8) !*Sink {
        var fds: [2]posix.fd_t = undefined;
        if (std.c.pipe(&fds) != 0) return error.PipeFailed;
        for (fds) |fd| _ = std.c.fcntl(fd, posix.F.SETFD, @as(c_int, posix.FD_CLOEXEC));
        const s = gpa.create(Sink) catch |e| {
            for (fds) |fd| _ = std.c.close(fd);
            return e;
        };
        s.* = .{ .io = io, .log = log, .read_fd = fds[0], .child = .{ .handle = fds[1], .flags = .{ .nonblocking = false } }, .secrets = secrets, .served = served };
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
        if (bytes.len == 0 or s.broken) return;
        s.writes += 1;
        if (s.writes == s.fault.at) {
            s.log.writePositionalAll(s.io, bytes[0..@min(s.fault.keep, bytes.len)], s.offset) catch {};
            s.broken = s.secrets.len > 0;
            return;
        }
        s.log.writePositionalAll(s.io, bytes, s.offset) catch {
            s.broken = s.secrets.len > 0; // a later line written here could join this one's text into a key
            return;
        };
        s.offset += bytes.len;
    }

    /// The held last line written now, checked and with a newline after it, for an exited child whose pipe stays open.
    pub fn cutHeld(s: *Sink, ms: u32) void {
        if (s.thread == null or s.done.load(.acquire)) return;
        s.cut.store(1, .release);
        var waited: u32 = 0;
        while (s.cut.load(.acquire) != 2 and !s.done.load(.acquire) and waited < ms) : (waited += 10) s.io.sleep(.fromMilliseconds(10), .awake) catch {};
    }

    /// Serve's port once its serving line has gone through the pipe.
    pub fn servingPort(s: *const Sink) ?u16 {
        const p = s.port.load(.acquire);
        return if (p == 0) null else p;
    }

    /// With keys, the exact text written for one line: its keys replaced, and the result itself free of every key.
    fn written(s: *const Sink, a: std.mem.Allocator, raw: []const u8) []const u8 {
        const candidate = sse.redact(a, raw, s.secrets) catch return s.marker(a);
        return if (credential.contains(a, candidate, s.secrets) catch true) s.marker(a) else candidate;
    }

    /// The fixed text in place of a withheld line, or nothing when that text is itself a key.
    fn marker(s: *const Sink, a: std.mem.Allocator) []const u8 {
        return if (credential.contains(a, credential.withheld, s.secrets) catch true) "" else credential.withheld;
    }

    /// One whole line (`newline`: it ended in one, else the stream ended): readiness from its raw bytes, then the file.
    fn line(s: *Sink, a: std.mem.Allocator, raw: []const u8, newline: bool) void {
        if (s.served.len > 0 and s.port.load(.acquire) == 0) {
            if (@import("benchmark_server.zig").servingPort(raw, s.served)) |p| s.port.store(p, .release);
        }
        if (s.secrets.len == 0) return; // written as it came
        s.append(s.written(a, raw));
        if (newline) s.append("\n");
    }

    fn run(s: *Sink) void {
        defer s.done.store(true, .release);
        var held: std.ArrayList(u8) = .empty; // the current line, up to hold_limit
        defer held.deinit(std.heap.page_allocator);
        var over = false; // the current line passed hold_limit: nothing more of it is kept or written
        var buf: [64 << 10]u8 = undefined;
        var fds = [_]posix.pollfd{.{ .fd = s.read_fd, .events = posix.POLL.IN, .revents = 0 }};
        var after_stop: u32 = 0;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        while (true) {
            if (s.cut.load(.acquire) == 1) {
                if (over) {
                    if (s.secrets.len > 0) s.append("\n");
                } else if (held.items.len > 0) s.line(arena.allocator(), held.items, true);
                over = false;
                held.clearRetainingCapacity();
                s.cut.store(2, .release);
            }
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
            if (s.secrets.len == 0) s.append(buf[0..n]);
            var rest = buf[0..n];
            while (rest.len > 0) {
                const end = std.mem.indexOfScalar(u8, rest, '\n');
                const part = rest[0 .. end orelse rest.len];
                rest = if (end) |e| rest[e + 1 ..] else rest[rest.len..];
                if (!over) held.appendSlice(std.heap.page_allocator, part) catch {
                    over = true;
                };
                if (!over and held.items.len > hold_limit) over = true;
                if (over and held.items.len > 0) {
                    if (s.secrets.len > 0) s.append(s.marker(arena.allocator()));
                    held.clearRetainingCapacity();
                }
                if (end == null) continue;
                if (over) {
                    if (s.secrets.len > 0) s.append("\n");
                    over = false;
                } else s.line(arena.allocator(), held.items, true);
                held.clearRetainingCapacity();
                _ = arena.reset(.retain_capacity);
            }
        }
        // a last line without its newline is whole now: nothing can follow it
        if (!over and held.items.len > 0) s.line(arena.allocator(), held.items, false);
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
        const s = try Sink.open(std.testing.allocator, io, log, &.{secret}, "");
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
    const s = try Sink.open(std.testing.allocator, io, log, &.{}, "");
    const writer = std.c.dup(s.child.handle);
    try s.started();
    _ = std.c.write(writer, "plain output\n", 13);
    _ = std.c.close(writer);
    s.settle(5000);
    s.close(std.testing.allocator);
    var out: [64]u8 = undefined;
    try std.testing.expectEqualStrings("plain output\n", out[0..try log.readPositionalAll(io, &out, 0)]);
}

/// What the sink wrote for `text`, sent in two writes split at `split`, under `secrets`.
fn through(io: std.Io, dir: std.Io.Dir, secrets: []const []const u8, text: []const u8, split: usize, out: []u8) ![]const u8 {
    return faulted(io, dir, secrets, text, split, .{}, out);
}

/// `through`, with the sink's `at`-th file write failing after `keep` bytes; the file is writable again after.
fn faulted(io: std.Io, dir: std.Io.Dir, secrets: []const []const u8, text: []const u8, split: usize, fault: @FieldType(Sink, "fault"), out: []u8) ![]const u8 {
    const log = try dir.createFile(io, "child.log", .{ .read = true, .truncate = true });
    defer log.close(io);
    const s = try Sink.open(std.testing.allocator, io, log, secrets, "");
    s.fault = fault;
    const writer = std.c.dup(s.child.handle);
    try s.started();
    _ = std.c.write(writer, text.ptr, split);
    _ = std.c.write(writer, text[split..].ptr, text.len - split);
    _ = std.c.close(writer);
    s.settle(5000);
    s.close(std.testing.allocator);
    return out[0..try log.readPositionalAll(io, out, 0)];
}

test "benchmark log: no replacement, marker or held suffix writes a configured key, at any split" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cases = [_]struct { secrets: []const []const u8, text: []const u8 }{
        .{ .secrets = &.{"[redacted]"}, .text = "a [redacted] b\nnext\n" }, // the key is the replacement text
        // a replacement completes a key
        .{ .secrets = &.{ "[redacted]-suffix", "fixture-log-key" }, .text = "fixture-log-key-suffix\nnext\n" },
        .{ .secrets = &.{ "[redacted]", "[withheld]" }, .text = "x [redacted]\nnext\n" }, // both fixed texts are keys
        .{ .secrets = &.{"[redacted]"}, .text = "held [" }, // a suffix held at the end of the stream
        .{ .secrets = &.{"[redacted]"}, .text = "a %5Bredacted%5D b\nnext\n" }, // encoded forms of the same
        .{ .secrets = &.{"[redacted]"}, .text = "a %255Bredacted%255D b\nnext\n" },
        .{ .secrets = &.{ "[redacted]-suffix", "fixture-log-key" }, .text = "fixture%2Dlog%2Dkey-suffix\nnext\n" },
        .{ .secrets = &.{ "[redacted]", "[withheld]" }, .text = "x %5Bredacted%5D\nnext\n" },
        .{ .secrets = &.{"[redacted]"}, .text = "held %5B" },
    };
    var out: [256]u8 = undefined;
    for (cases) |case| for (0..case.text.len + 1) |split| {
        const got = try through(io, tmp.dir, case.secrets, case.text, split, &out);
        for (case.secrets) |k| try std.testing.expect(std.mem.indexOf(u8, got, k) == null);
        try std.testing.expect(!try credential.contains(std.testing.allocator, got, case.secrets)); // at any depth
        if (std.mem.endsWith(u8, case.text, "next\n")) try std.testing.expect(std.mem.endsWith(u8, got, "\nnext\n"));
    };
}

test "benchmark log: a line past the hold limit writes no configured key" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const long = try std.testing.allocator.alloc(u8, hold_limit + 4096 + 6);
    defer std.testing.allocator.free(long);
    var out: [4096]u8 = undefined;
    for ([_][]const u8{ "", "%5Bredacted%5D", "%255Bredacted%255D" }) |planted| {
        @memset(long, 'x');
        @memcpy(long[10..][0..planted.len], planted); // before the limit
        @memcpy(long[long.len - 6 - planted.len ..][0..planted.len], planted); // past it
        @memcpy(long[long.len - 6 ..], "\nnext\n");
        const got = try through(io, tmp.dir, &.{"[redacted]"}, long, 1000, &out);
        try std.testing.expect(!try credential.contains(std.testing.allocator, got, &.{"[redacted]"}));
        try std.testing.expect(std.mem.endsWith(u8, got, "\nnext\n"));
    }
}

test "benchmark log: a write that fails, whole or in part, never lets later lines join into a key in the file" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const secrets: []const []const u8 = &.{"abcdef"};
    var out: [256]u8 = undefined;
    // writes: 1 "abc", 2 its newline, 3 "def", 4 its newline; then the encoded halves
    for ([_]struct { text: []const u8, at: u32, keep: usize }{
        .{ .text = "abc\ndef\n", .at = 2, .keep = 0 }, // the separator fails, then the disk recovers
        .{ .text = "abc\ndef\n", .at = 1, .keep = 2 }, // a line written in part
        .{ .text = "abc\ndef\n", .at = 3, .keep = 1 },
        .{ .text = "%61bc\n%64ef\n", .at = 2, .keep = 0 }, // encoded halves
        .{ .text = "ab%2563\ndef\n", .at = 2, .keep = 0 }, // nested
    }) |c| for (0..c.text.len + 1) |split| {
        const got = try faulted(io, tmp.dir, secrets, c.text, split, .{ .at = c.at, .keep = c.keep }, &out);
        try std.testing.expect(!try credential.contains(std.testing.allocator, got, secrets));
    };
}

test "benchmark log: without keys, ordinary bytes pass unchanged to the end of the stream" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const text = "one\ntwo %5B\n[redacted]\nlast without newline";
    var out: [256]u8 = undefined;
    for (0..text.len + 1) |split| try std.testing.expectEqualStrings(text, try through(io, tmp.dir, &.{}, text, split, &out));
}
