//! Stage a unique, exclusively created sibling file and publish it by rename.
const std = @import("std");
const platform = @import("sys.zig");
const Allocator = std.mem.Allocator;

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

pub fn replace(a: Allocator, path: []const u8, parts: []const []const u8) !void {
    try commit(a, try stage(a, path, parts), path);
}

/// `replace` for private data: the published file is readable by its owner only.
pub fn replacePrivate(a: Allocator, path: []const u8, parts: []const []const u8) !void {
    try commit(a, try stagePrivate(a, path, parts), path);
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
