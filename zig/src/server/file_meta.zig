//! A file's type, permission bits, size, identity and mtime, by path or open descriptor, on every platform.
const std = @import("std");
const platform = @import("sys");
const posix = platform.posix;
const builtin = @import("builtin");

/// A file's type and permission bits, size, identity and mtime: statx on Linux (its libc has no stat), stat elsewhere.
pub const Meta = struct { mode: u32, size: u64, dev: u64, ino: u64, mtime_ns: i128 };

/// `path` at the working directory, or the open `fd` itself when `path` is null.
pub fn meta(fd: posix.fd_t, path: ?[*:0]const u8) ?Meta {
    if (builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var sx: linux.Statx = undefined;
        const want: linux.STATX = .{ .TYPE = true, .MODE = true, .INO = true, .SIZE = true, .MTIME = true };
        const rc = if (path) |p| platform.c.statx(posix.AT.FDCWD, p, 0, want, &sx) else platform.c.statx(fd, "", 0x1000, want, &sx); // AT_EMPTY_PATH
        if (rc != 0) return null;
        const dev = (@as(u64, sx.dev_major) << 32) | sx.dev_minor;
        return .{ .mode = sx.mode, .size = sx.size, .dev = dev, .ino = sx.ino, .mtime_ns = @as(i128, sx.mtime.sec) * std.time.ns_per_s + sx.mtime.nsec };
    }
    if (builtin.os.tag == .windows) {
        // NTFS keeps access in ACLs, not mode bits: the type and size are checked, and the profile's ACL is the guard
        var st: platform.c.Stat = undefined;
        const f = if (path) |p| platform.c.open(p, .{ .ACCMODE = .RDONLY }) else fd;
        if (f < 0) return null;
        defer if (path != null) {
            _ = platform.c.close(f);
        };
        if (platform.c.fstat(f, &st) != 0) return null;
        const t = st.mtime();
        return .{ .mode = (st.mode & platform.c.S.IFMT) | 0o600, .size = @intCast(st.size), .dev = st.dev, .ino = st.ino, .mtime_ns = @as(i128, t.nsec) + @as(i128, t.sec) * std.time.ns_per_s };
    }
    var st: posix.Stat = undefined;
    const rc = if (path) |p| posix.system.stat(p, &st) else posix.system.fstat(fd, &st);
    if (posix.errno(rc) != .SUCCESS) return null;
    const t = st.mtime();
    return .{ .mode = @intCast(st.mode), .size = @intCast(st.size), .dev = @intCast(st.dev), .ino = @intCast(st.ino), .mtime_ns = std.math.lossyCast(i128, t.nsec) + @as(i128, t.sec) * std.time.ns_per_s };
}
