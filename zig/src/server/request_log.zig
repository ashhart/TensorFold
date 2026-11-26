//! TENSORFOLD_REQUEST_LOG: each chat and completion body appended as one JSON line, images and salts redacted.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const json = @import("json");
const meta = @import("file_meta.zig").meta;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// Past this the file rotates: it becomes `.1` and a fresh file takes the newest records.
pub const file_limit: u64 = 64 << 20;
/// The current log and its one previous file, ``path`` and ``path.1``: at most twice ``file_limit`` together.
pub const files = 2;
/// The most one record holds: a longer body or reply is refused.
pub const record_limit: usize = 8 << 20;
/// A refusal's message is kept to this: a longer one is cut.
pub const error_limit: usize = 4 << 10;

fn isFilePayload(key: []const u8) bool {
    return std.mem.eql(u8, key, "file_data") or std.mem.eql(u8, key, "data") or
        std.mem.eql(u8, key, "url") or std.mem.eql(u8, key, "file_url");
}

/// The body with every image part's URL, every inline document payload and any cache salt replaced, the rest kept.
pub fn redact(a: Allocator, v: Value) Allocator.Error!Value {
    switch (v) {
        .array => |items| {
            const out = try a.alloc(Value, items.len);
            for (items, out) |item, *slot| slot.* = try redact(a, item);
            return .{ .array = out };
        },
        .object => |o| {
            if (v.typeIs("image_url") or v.typeIs("image") or v.typeIs("input_image")) {
                const copy = try json.copyObject(a, o);
                const url = try json.newObject(a);
                try url.put(a, "url", .{ .string = "<redacted>" });
                _ = copy.orderedRemove("image");
                _ = copy.orderedRemove("input_image");
                try copy.put(a, "image_url", .{ .object = url });
                return .{ .object = copy };
            }
            if (v.typeIs("file") or v.typeIs("input_file")) {
                const copy = try json.copyObject(a, o);
                if (v.get("file")) |f| if (f == .object) {
                    const file_copy = try json.copyObject(a, f.object);
                    for (file_copy.keys()) |key| if (isFilePayload(key)) try file_copy.put(a, key, .{ .string = "<redacted>" });
                    try copy.put(a, "file", .{ .object = file_copy });
                };
                for (copy.keys()) |key| if (isFilePayload(key)) try copy.put(a, key, .{ .string = "<redacted>" });
                return .{ .object = copy };
            }
            const out = try json.newObject(a);
            for (o.keys(), o.values()) |k, item| try out.put(a, k, if (std.mem.eql(u8, k, "cache_salt")) .{ .string = "<redacted>" } else try redact(a, item));
            return .{ .object = out };
        },
        else => return v,
    }
}

/// Whether ``body`` is background work: batch jobs are not client traffic and are not logged.
pub fn isBackground(body: Value) bool {
    if (body.strField("priority")) |p| if (std.mem.eql(u8, p, "background")) return true;
    return false;
}

/// Appends ``body`` to ``path`` unless it is background work (batch jobs are not client traffic).
pub fn append(a: Allocator, path: []const u8, body: Value, limit: u64) void {
    if (isBackground(body)) return;
    write(a, path, redact(a, body) catch return, limit);
}

/// The typed request record: the same redacted body ``append`` writes, beside the id its reply will name.
pub fn appendRequest(a: Allocator, path: []const u8, id: ?[]const u8, body: Value, limit: u64) void {
    if (isBackground(body)) return;
    const o = json.newObject(a) catch return;
    o.put(a, "type", .{ .string = "request" }) catch return;
    o.put(a, "id", if (id) |s| .{ .string = s } else .null) catch return;
    o.put(a, "body", redact(a, body) catch return) catch return;
    write(a, path, .{ .object = o }, limit);
}

/// One finished reply: what the client got and what the server measured, or a refusal's message.
pub const Reply = struct {
    id: []const u8,
    status: u16,
    stream: bool,
    finish_reason: ?[]const u8 = null,
    text: []const u8 = "",
    reasoning: ?[]const u8 = null,
    prompt_tokens: ?i64 = null,
    cached_tokens: ?i64 = null,
    completion_tokens: ?i64 = null,
    queue_ms: ?f64 = null,
    ttft_ms: ?f64 = null,
    decode_tps: ?f64 = null,
    truncated: bool = false,
    error_message: ?[]const u8 = null,
};

/// Appends ``r`` as one reply record (``append``'s rules: owner-only file, no headers or keys).
pub fn appendReply(a: Allocator, path: []const u8, r: Reply, limit: u64) void {
    const o = json.newObject(a) catch return;
    o.put(a, "type", .{ .string = "reply" }) catch return;
    o.put(a, "id", .{ .string = r.id }) catch return;
    o.put(a, "status", json.intValue(a, r.status) catch return) catch return;
    if (r.status >= 400) {
        const m = r.error_message orelse "";
        o.put(a, "error", .{ .string = cut(m) }) catch return;
        return write(a, path, .{ .object = o }, limit);
    }
    o.put(a, "stream", .{ .bool = r.stream }) catch return;
    o.put(a, "finish_reason", if (r.finish_reason) |s| .{ .string = s } else .null) catch return;
    o.put(a, "text", .{ .string = r.text }) catch return;
    o.put(a, "reasoning", if (r.reasoning) |s| .{ .string = s } else .null) catch return;
    const usage = json.newObject(a) catch return;
    usage.put(a, "prompt_tokens", intOrNull(a, r.prompt_tokens)) catch return;
    usage.put(a, "cached_tokens", intOrNull(a, r.cached_tokens)) catch return;
    usage.put(a, "completion_tokens", intOrNull(a, r.completion_tokens)) catch return;
    o.put(a, "usage", .{ .object = usage }) catch return;
    const timings = json.newObject(a) catch return;
    timings.put(a, "queue_ms", floatOrNull(r.queue_ms)) catch return;
    timings.put(a, "ttft_ms", floatOrNull(r.ttft_ms)) catch return;
    timings.put(a, "decode_tps", floatOrNull(r.decode_tps)) catch return;
    o.put(a, "timings", .{ .object = timings }) catch return;
    o.put(a, "truncated", .{ .bool = r.truncated }) catch return;
    if (r.error_message) |m| o.put(a, "error", .{ .string = cut(m) }) catch return;
    write(a, path, .{ .object = o }, limit);
}

fn intOrNull(a: Allocator, n: ?i64) Value {
    return if (n) |v| json.intValue(a, v) catch .null else .null;
}

fn floatOrNull(v: ?f64) Value {
    return if (v) |f| .{ .float = @round(f * 10) / 10 } else .null;
}

/// ``m`` cut to ``max`` bytes, never inside a UTF-8 sequence: a cut record still parses.
pub fn cutTo(m: []const u8, max: usize) []const u8 {
    if (m.len <= max) return m;
    var i: usize = max;
    while (i > 0 and (m[i] & 0xc0) == 0x80) i -= 1; // back over continuation bytes
    return m[0..i];
}

/// The refusal message at its bound, for the two ``appendReply`` sites.
fn cut(m: []const u8) []const u8 {
    return cutTo(m, error_limit);
}

/// One JSON line, appended through the owner-only file both records use.
fn write(a: Allocator, path: []const u8, value: Value, limit: u64) void {
    if (builtin.os.tag == .windows) return warnOnce(path, .open, "Windows keeps file access in ACLs this writer cannot set, so the log stays off there", .{});
    const line = json.stringify(a, value, .{}) catch return;
    const framed = std.mem.concat(a, u8, &.{ line, "\n" }) catch return;
    const needed: u64 = framed.len; // the trailing newline is part of the record's bound
    if (needed > @as(u64, record_limit)) return warnOnce(path, .refused, "a record past {d} MiB was refused", .{record_limit >> 20});
    if (needed > limit) return warnOnce(path, .refused, "a record past its {d} MiB file limit was refused", .{limit >> 20});
    const pathz = a.dupeSentinel(u8, path, 0) catch return;
    const lockz = std.fmt.allocPrintSentinel(a, "{s}.lock", .{path}, 0) catch return;
    const lock_fd = openLock(lockz);
    if (lock_fd < 0) return warnOnce(path, .open, "its .lock file cannot be opened, or it is a symbolic link", .{});
    defer _ = platform.c.close(lock_fd); // declared before the data file's close: the lock is released after it
    if (!private(lock_fd)) return warnOnce(path, .open, "its .lock file is not a regular file only its owner can read", .{});
    if (!lock(lock_fd)) return warnOnce(path, .io, "its .lock file cannot be locked", .{});
    var fd = openLog(pathz);
    if (fd < 0) return warnOnce(path, .open, "it cannot be opened, or it is a symbolic link", .{});
    defer {
        if (fd >= 0) _ = platform.c.close(fd);
    }
    if (!private(fd)) return warnOnce(path, .open, "it is not a regular file only its owner can read", .{});
    var size = (meta(fd, null) orelse return warnOnce(path, .io, "its size cannot be read", .{})).size;
    if (size > limit - needed) {
        // the current file becomes .1 and a fresh one takes this line: the log holds two files, never more
        _ = platform.c.close(fd);
        fd = -1;
        const older = std.fmt.allocPrintSentinel(a, "{s}.1", .{path}, 0) catch return;
        if (platform.c.rename(pathz, older) != 0) return warnOnce(path, .io, "it cannot be rotated", .{});
        // an archive above the limit keeps whole records within ``limit``: both files stay within ``2 * limit``
        const archive = platform.c.open(older, .{ .ACCMODE = .RDWR, .CLOEXEC = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0o600));
        if (archive >= 0) {
            if (size > limit) cutToRecord(archive, limit); // ``size`` is the file just renamed: the lock is still held
            _ = platform.c.close(archive);
        } else warnOnce(path, .io, "the archive cannot be opened to cut it", .{});
        fd = openLog(pathz); // the lock is held: nothing else can have made it since the rename
        if (fd < 0) return warnOnce(path, .open, "it cannot be opened, or it is a symbolic link", .{});
        if (!private(fd)) return warnOnce(path, .open, "it is not a regular file only its owner can read", .{});
        warnOnce(path, .rotated, "it reached its {d} MiB limit and rotated", .{limit >> 20});
        size = (meta(fd, null) orelse return warnOnce(path, .io, "its size cannot be read", .{})).size;
    }
    const n = platform.c.write(fd, framed.ptr, framed.len);
    if (n != @as(isize, @intCast(framed.len))) {
        if (n > 0) _ = platform.c.ftruncate(fd, @intCast(size)); // no torn record for the next line to follow
        return warnOnce(path, .io, "it cannot be written", .{});
    }
}

/// The archive cut to its last record boundary within ``limit``: a torn record never survives the cut.
fn cutToRecord(fd: platform.c.fd_t, limit: u64) void {
    var buf: [4096]u8 = undefined;
    var end = limit;
    while (end > 0) {
        const start = if (end > buf.len) end - buf.len else 0;
        const n = platform.c.pread(fd, &buf, @intCast(end - start), @intCast(start));
        if (n < 0) return; // a failed read leaves the archive as it is
        if (n == 0) break;
        if (std.mem.lastIndexOfScalar(u8, buf[0..@intCast(n)], '\n')) |i| {
            _ = platform.c.ftruncate(fd, @intCast(start + i + 1)); // the newline is kept: the archive ends on a whole record
            return;
        }
        end = start;
    }
    _ = platform.c.ftruncate(fd, 0); // no '\n' within the limit: no partial record survives
}

/// The sibling `.lock` every writer locks: it is never rotated, so a waiter never locks a moving file.
fn openLock(path: [*:0]const u8) platform.c.fd_t {
    return platform.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0o600));
}

/// The owner-only append the log opens with, before and after a rotation: a plain create keeps the append atomic.
fn openLog(path: [*:0]const u8) platform.c.fd_t {
    return platform.c.open(path, .{ .ACCMODE = .WRONLY, .APPEND = true, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0o600));
}

/// The exclusive lock a write holds across its size read, its append and its rollback; a signal's EINTR retries.
fn lock(fd: platform.c.fd_t) bool {
    while (true) {
        const rc = platform.c.flock(fd, platform.c.LOCK.EX);
        if (rc == 0) return true;
        if (platform.posix.errno(rc) != .INTR) return false;
    }
}

/// A regular file only its owner can read or write; Windows mode bits cannot answer this, so the log refuses Windows.
pub fn private(fd: platform.c.fd_t) bool {
    if (builtin.os.tag == .windows) return true; // NTFS keeps access in ACLs, not mode bits: the log refuses Windows
    const m = meta(fd, null) orelse return false;
    return m.mode & 0o170000 == 0o100000 and m.mode & 0o077 == 0;
}

/// The reasons the log can stop taking records: each warns once, and one does not silence another.
const Why = enum { refused, open, rotated, io };

var warned = std.atomic.Value(u8).init(0);

fn warnOnce(path: []const u8, comptime why: Why, comptime fmt: []const u8, args: anytype) void {
    const bit: u8 = @as(u8, 1) << @backingInt(why);
    if (warned.fetchOr(bit, .monotonic) & bit != 0) return;
    std.log.warn("TENSORFOLD_REQUEST_LOG {s}: " ++ fmt, .{path} ++ args);
}

test "request log replies: redaction removes image aliases as well as image_url" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const secret = "https://example.test/private.png";
    for ([_][]const u8{ "image_url", "image", "input_image" }) |kind| {
        const url = try json.newObject(a);
        try url.put(a, "url", .{ .string = secret });
        const part = try json.newObject(a);
        try part.put(a, "type", .{ .string = kind });
        try part.put(a, kind, .{ .object = url });
        const text = try json.stringify(a, try redact(a, .{ .object = part }), .{ .compact = true });
        try std.testing.expect(std.mem.indexOf(u8, text, secret) == null);
        try std.testing.expect(std.mem.indexOf(u8, text, "<redacted>") != null);
    }
}
test "the request log is created owner-only and keeps image URLs and cache salts out" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the log refuses Windows files
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/requests.jsonl", .{tmp.sub_path});
    const url = try json.newObject(a);
    try url.put(a, "url", .{ .string = "https://example.test/private.png" });
    const part = try json.newObject(a);
    try part.put(a, "type", .{ .string = "image_url" });
    try part.put(a, "image_url", .{ .object = url });
    const body = try json.newObject(a);
    const file = (try json.parse(a, "{\"type\": \"file\", \"file\": {\"filename\": \"a.pdf\", \"file_data\": \"JVBERi1zZWNyZXQ=\"}}")).ok;
    try body.put(a, "messages", .{ .array = try a.dupe(Value, &.{ .{ .object = part }, file }) });
    try body.put(a, "cache_salt", .{ .string = "tenant-secret-salt" });
    append(a, path, .{ .object = body }, file_limit);
    const st = try std.Io.Dir.cwd().statFile(std.testing.io, path, .{}); // Linux's libc has no stat for std.c
    try std.testing.expectEqual(@as(u32, 0o600), @as(u32, @intCast(st.permissions.toMode())) & 0o777);
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 16));
    try std.testing.expect(std.mem.indexOf(u8, text, "<redacted>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "example.test") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "tenant-secret-salt") == null and std.mem.indexOf(u8, text, "cache_salt") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "JVBERi1zZWNyZXQ=") == null and std.mem.indexOf(u8, text, "a.pdf") != null);
}

test "appends from several threads keep every line whole" {
    if (builtin.os.tag == .windows) return error.SkipZigTest; // the log refuses Windows files
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try std.process.currentPathAlloc(std.testing.io, a);
    defer a.free(cwd);
    const path = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}/log.jsonl", .{ cwd, tmp.sub_path });
    defer a.free(path);
    const threads = 8;
    const lines = 200;
    const head = "{\"line\": \"";
    const Worker = struct {
        fn run(p: []const u8, k: u8) void {
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const text = arena.allocator().alloc(u8, 300) catch return;
            @memset(text, 'a' + k);
            for (0..lines) |_| {
                const o = json.newObject(arena.allocator()) catch return;
                o.put(arena.allocator(), "line", .{ .string = text }) catch return;
                append(arena.allocator(), p, .{ .object = o }, file_limit);
            }
        }
    };
    var workers: [threads]std.Thread = undefined;
    for (&workers, 0..) |*w, k| w.* = try std.Thread.spawn(.{}, Worker.run, .{ path, @as(u8, @intCast(k)) });
    for (workers) |w| w.join();
    const got = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 24));
    defer a.free(got);
    var counts: [threads]usize = @splat(0);
    var it = std.mem.splitScalar(u8, got, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        try std.testing.expectEqual(head.len + 300 + 2, line.len);
        const k = line[head.len] - 'a';
        try std.testing.expect(k < threads);
        for (line[head.len..][0..300]) |c| try std.testing.expectEqual('a' + k, c);
        counts[k] += 1;
    }
    for (counts) |n| try std.testing.expectEqual(@as(usize, lines), n);
}

test "a log others can read, or a symbolic link, gets no requests; the owner's own file still does" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    if (builtin.os.tag == .windows) return error.SkipZigTest; // refusal by mode bits and links; NTFS keeps access in ACLs
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const body = try json.newObject(a);
    try body.put(a, "model", .{ .string = "m" });
    // a name another user created first, readable by others: nothing is appended to it
    try tmp.dir.writeFile(io, .{ .sub_path = "taken.jsonl", .data = "" });
    try tmp.dir.setFilePermissions(io, "taken.jsonl", .fromMode(0o644), .{}); // set after creating: a umask clears bits
    append(a, try std.fmt.allocPrint(a, "{s}/taken.jsonl", .{dir}), .{ .object = body }, file_limit);
    try std.testing.expectEqual(@as(usize, 0), (try tmp.dir.readFileAlloc(io, "taken.jsonl", a, .limited(1 << 10))).len);
    // a symbolic link to another file: the target is never written
    try tmp.dir.writeFile(io, .{ .sub_path = "target", .data = "", .flags = .{ .permissions = .fromMode(0o600) } });
    try tmp.dir.symLink(io, "target", "link.jsonl", .{});
    append(a, try std.fmt.allocPrint(a, "{s}/link.jsonl", .{dir}), .{ .object = body }, file_limit);
    try std.testing.expectEqual(@as(usize, 0), (try tmp.dir.readFileAlloc(io, "target", a, .limited(1 << 10))).len);
    // the owner's own 0600 file keeps its lines
    append(a, try std.fmt.allocPrint(a, "{s}/target", .{dir}), .{ .object = body }, file_limit);
    try std.testing.expect((try tmp.dir.readFileAlloc(io, "target", a, .limited(1 << 10))).len > 0);
}
