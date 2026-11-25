//! Stage a unique, exclusively created sibling file and publish it by rename.
const std = @import("std");
const platform = @import("sys.zig");
const Allocator = std.mem.Allocator;
pub const testing = if (@import("builtin").is_test) struct {
    pub var observer: ?*const fn ([]const u8) void = null;
} else struct {};

var serial: std.atomic.Value(u64) = .init(0);

threadlocal var last_failure: ?platform.c.E = null;

/// The OS error behind the latest `ShardWrite` on this thread, for client and log messages.
pub fn causeName() []const u8 {
    return if (last_failure) |e| @tagName(e) else "unknown";
}

fn failed(rc: anytype) error{ShardWrite} {
    const cause = platform.c.errno(rc);
    last_failure = if (cause == .SUCCESS) null else cause;
    return error.ShardWrite;
}

pub fn stage(a: Allocator, path: []const u8, parts: []const []const u8) ![:0]const u8 {
    return stageAs(a, path, parts, 0o644);
}

/// `stage` for private data: only the owner can read the file.
pub fn stagePrivate(a: Allocator, path: []const u8, parts: []const []const u8) ![:0]const u8 {
    return stageAs(a, path, parts, 0o600);
}

fn stageAs(a: Allocator, path: []const u8, parts: []const []const u8, mode: platform.c.mode_t) ![:0]const u8 {
    last_failure = null;
    for (0..32) |_| {
        const name = try std.fmt.allocPrintSentinel(a, "{s}.tmp.{d}.{d}", .{
            path, platform.c.getpid(), serial.fetchAdd(1, .monotonic),
        }, 0);
        const fd = platform.c.open(name, .{
            .ACCMODE = .WRONLY,
            .CREAT = true,
            .EXCL = true,
            .NOFOLLOW = true,
            .CLOEXEC = true,
        }, mode);
        if (fd < 0) {
            const cause = platform.c.errno(fd);
            a.free(name);
            if (cause == .EXIST) continue;
            last_failure = cause;
            return error.ShardWrite;
        }
        errdefer a.free(name);
        errdefer _ = platform.c.unlink(name);
        defer _ = platform.c.close(fd);
        for (parts) |part| try writeAll(fd, part);
        const synced = platform.c.fsync(fd);
        if (synced != 0) return failed(synced);
        return name;
    }
    last_failure = .EXIST;
    return error.ShardWrite;
}

pub fn commit(a: Allocator, temporary: [:0]const u8, path: []const u8) !void {
    last_failure = null;
    defer a.free(temporary);
    errdefer _ = platform.c.unlink(temporary);
    const destination = try a.dupeSentinel(u8, path, 0);
    defer a.free(destination);
    const renamed = platform.c.rename(temporary, destination);
    if (renamed != 0) return failed(renamed);
}
/// The rename or unlink of `path` made durable with its parent directory.
pub fn syncParent(path: []const u8) !void {
    defer {
        if (comptime @import("builtin").is_test) if (testing.observer) |observe| observe(path);
    }
    if (comptime @import("builtin").os.tag == .windows) return;
    var buf: [1200]u8 = undefined;
    const dir = std.fmt.bufPrintSentinel(&buf, "{s}", .{std.fs.path.dirname(path) orelse "."}, 0) catch return error.DirectorySync;
    const fd = platform.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.DirectorySync;
    defer _ = platform.c.close(fd);
    while (platform.c.fsync(fd) != 0) {
        if (platform.c.errno(-1) == .INTR) continue;
        return error.DirectorySync;
    }
}

pub fn replace(a: Allocator, path: []const u8, parts: []const []const u8) !void {
    try commit(a, try stage(a, path, parts), path);
}

/// `replace` for private data: the published file is readable by its owner only.
pub fn replacePrivate(a: Allocator, path: []const u8, parts: []const []const u8) !void {
    try commit(a, try stagePrivate(a, path, parts), path);
}

/// Clears the `<file>.tmp.<pid>.<serial>` a crash left in `dir` (only `only`'s when given); live writers keep theirs.
pub fn discardStale(io: std.Io, dir: []const u8, only: ?[]const u8) void {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return;
    defer d.close(io);
    var it = d.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        const staged = stagedBy(entry.name) orelse continue;
        if (only) |file| if (!std.mem.eql(u8, staged.file, file)) continue;
        const stale = if (platform.windows) blk: {
            // no portable way to ask whether a pid lives: a day-old temporary is stale
            const cutoff = std.Io.Clock.real.now(io).subDuration(.{ .nanoseconds = 86_400 * std.time.ns_per_s });
            const st = d.statFile(io, entry.name, .{}) catch continue;
            break :blk st.mtime.nanoseconds < cutoff.nanoseconds;
        } else writerGone(staged.pid);
        if (stale) d.deleteFile(io, entry.name) catch {};
    }
}

/// The file and writer's pid when `name` is exactly `<file>.tmp.<pid>.<serial>`, as `stageAs` prints them.
pub fn stagedBy(name: []const u8) ?struct { file: []const u8, pid: i32 } {
    const at = std.mem.lastIndexOf(u8, name, ".tmp.") orelse return null;
    if (at == 0) return null;
    const rest = name[at + ".tmp.".len ..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    if (!decimal(rest[0..dot]) or !decimal(rest[dot + 1 ..])) return null;
    const pid = std.fmt.parseInt(i32, rest[0..dot], 10) catch return null;
    if (pid <= 0) return null;
    return .{ .file = name[0..at], .pid = pid };
}

/// Digits as `{d}` prints them: at least one, no sign, separator or leading zero.
fn decimal(text: []const u8) bool {
    if (text.len == 0 or (text.len > 1 and text[0] == '0')) return false;
    for (text) |ch| if (!std.ascii.isDigit(ch)) return false;
    return true;
}

fn writerGone(pid: i32) bool {
    if (platform.windows) return false; // discardStale uses the age fallback there
    if (platform.c.kill(pid, @fromBackingInt(@intCast(0))) == 0) return false;
    return platform.c.errno(-1) == .SRCH;
}

fn writeAll(fd: platform.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = platform.c.write(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30));
        if (n < 0 and platform.c.errno(n) == .INTR) continue;
        if (n <= 0) return failed(n);
        done += @intCast(n);
    }
}
