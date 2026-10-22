//! Stage a unique, exclusively created sibling file and publish it by rename.
const std = @import("std");
const platform = @import("sys.zig");
const Allocator = std.mem.Allocator;

var serial: std.atomic.Value(u64) = .init(0);

pub fn stage(a: Allocator, path: []const u8, parts: []const []const u8) ![:0]const u8 {
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
        }, @as(platform.c.mode_t, 0o644));
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

fn writeAll(fd: platform.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = platform.c.write(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30));
        if (n < 0 and platform.c.errno(n) == .INTR) continue;
        if (n <= 0) return error.ShardWrite;
        done += @intCast(n);
    }
}
