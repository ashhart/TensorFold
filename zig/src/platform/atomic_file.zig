//! Stage a unique, exclusively created sibling file and publish it by rename.
const std = @import("std");
const platform = @import("sys.zig");
const Allocator = std.mem.Allocator;

var serial: std.atomic.Value(u64) = .init(0);

pub fn stage(a: Allocator, path: []const u8, parts: []const []const u8) ![:0]const u8 {
    return stageAs(a, path, parts, 0o644);
}

/// `stage` for private data: only the owner can read the file.
pub fn stagePrivate(a: Allocator, path: []const u8, parts: []const []const u8) ![:0]const u8 {
    return stageAs(a, path, parts, 0o600);
}

fn stageAs(a: Allocator, path: []const u8, parts: []const []const u8, mode: platform.c.mode_t) ![:0]const u8 {
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
            return error.ShardWrite;
        }
        errdefer a.free(name);
        errdefer _ = platform.c.unlink(name);
        defer _ = platform.c.close(fd);
        for (parts) |part| try writeAll(fd, part);
        if (platform.c.fsync(fd) != 0) return error.ShardWrite;
        return name;
    }
    return error.ShardWrite;
}

pub fn commit(a: Allocator, temporary: [:0]const u8, path: []const u8) !void {
    defer a.free(temporary);
    errdefer _ = platform.c.unlink(temporary);
    const destination = try a.dupeSentinel(u8, path, 0);
    defer a.free(destination);
    if (platform.c.rename(temporary, destination) != 0) return error.ShardWrite;
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
fn stagedBy(name: []const u8) ?struct { file: []const u8, pid: i32 } {
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
        if (n <= 0) return error.ShardWrite;
        done += @intCast(n);
    }
}
