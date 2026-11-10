//! Whether a shard opened O_DIRECT takes its sector-aligned extent reads; one that refuses them is read buffered.
const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const sector = @import("engram_rows.zig").sector;

/// A test's disk in place of the system's pread (null: the system's).
pub var test_pread: if (builtin.is_test) ?*const fn (linux.fd_t, [*]u8, usize, i64) usize else void = if (builtin.is_test) null else {};

/// The pread of every serial and threaded Engram read.
pub fn pread(fd: linux.fd_t, buf: [*]u8, len: usize, off: i64) usize {
    if (comptime builtin.is_test) if (test_pread) |f| return f(fd, buf, len, off);
    return linux.pread(fd, buf, len, off);
}

/// `fd`, opened O_DIRECT, kept direct if it reads a sector at a sector; refused (EINVAL), O_DIRECT is cleared on it.
pub fn direct(fd: linux.fd_t, buf: []align(4096) u8) !bool {
    while (true) switch (linux.errno(pread(fd, buf.ptr, sector, sector))) {
        .INTR => continue,
        .INVAL => break,
        // bytes, the end of the file or an I/O error: the reads that follow report their own
        else => return true,
    };
    const flags = linux.fcntl(fd, linux.F.GETFL, 0);
    if (linux.errno(flags) != .SUCCESS) return error.EngramReadFailed;
    const flag: usize = @as(u32, @bitCast(linux.O{ .DIRECT = true }));
    if (linux.errno(linux.fcntl(fd, linux.F.SETFL, flags & ~flag)) != .SUCCESS) return error.EngramReadFailed;
    return false;
}
