//! The directory lease shared by serving, rollback and cache mutation; closing it releases the OS lock.
const std = @import("std");
const platform = @import("sys.zig");

pub const Lease = struct {
    fd: platform.c.fd_t,

    pub fn release(l: Lease) void {
        _ = platform.c.flock(l.fd, platform.c.LOCK.UN);
        _ = platform.c.close(l.fd);
    }
};

/// Hold a folder as a reader, or exclusively as a writer; fail rather than wait on another process.
pub fn hold(a: std.mem.Allocator, dir: []const u8, exclusive: bool) error{ Busy, Unlockable, OutOfMemory }!Lease {
    return take(a, dir, exclusive, false);
}

/// Check that a selected snapshot is a real folder before taking its lease.
pub fn noFollow(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Lease {
    var checked = try std.Io.Dir.cwd().openDir(io, dir, .{ .follow_symlinks = false });
    defer checked.close(io);
    return take(a, dir, true, true);
}

fn take(a: std.mem.Allocator, dir: []const u8, exclusive: bool, no_follow: bool) error{ Busy, Unlockable, OutOfMemory }!Lease {
    const path = try a.dupeSentinel(u8, dir, 0);
    defer a.free(path);
    const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true, .NOFOLLOW = no_follow }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.Unlockable;
    const kind: c_int = if (exclusive) platform.c.LOCK.EX else platform.c.LOCK.SH;
    const rc = platform.c.flock(fd, kind | platform.c.LOCK.NB);
    if (rc != 0) {
        const why = platform.c.errno(rc);
        _ = platform.c.close(fd);
        return if (why == .AGAIN) error.Busy else error.Unlockable;
    }
    return .{ .fd = fd };
}
