//! Descriptor identity for persistent digest reuse; timestamps retain nanosecond precision.
const std = @import("std");
const builtin = @import("builtin");

pub const persistent = builtin.os.tag == .linux or builtin.os.tag == .macos;

pub const Key = struct {
    device: u64,
    inode: u128,
    size: u64,
    mtime: i128,
    ctime: i128,

    pub fn of(file: std.Io.File, io: std.Io) !Key {
        if (builtin.os.tag == .linux) {
            const linux = std.os.linux;
            var s: linux.Statx = undefined;
            const need: linux.STATX = .{ .TYPE = true, .INO = true, .SIZE = true, .MTIME = true, .CTIME = true };
            if (linux.errno(linux.statx(file.handle, "", linux.AT.EMPTY_PATH, need, &s)) != .SUCCESS)
                return error.DigestStatFailed;
            if (!s.mask.TYPE or !s.mask.INO or !s.mask.SIZE or !s.mask.MTIME or !s.mask.CTIME)
                return error.DigestStatIncomplete;
            if (s.mode & linux.S.IFMT != linux.S.IFREG) return error.DigestNotRegular;
            return .{ .device = (@as(u64, s.dev_major) << 32) | s.dev_minor, .inode = s.ino, .size = s.size, .mtime = @as(i128, s.mtime.sec) * 1_000_000_000 + s.mtime.nsec, .ctime = @as(i128, s.ctime.sec) * 1_000_000_000 + s.ctime.nsec };
        }
        if (builtin.os.tag == .macos) {
            var s: std.c.Stat = undefined;
            if (std.c.fstat(file.handle, &s) != 0) return error.DigestStatFailed;
            if (s.mode & std.c.S.IFMT != std.c.S.IFREG or s.size < 0) return error.DigestNotRegular;
            return .{ .device = @as(u32, @bitCast(s.dev)), .inode = s.ino, .size = @intCast(s.size), .mtime = @as(i128, s.mtime().sec) * 1_000_000_000 + s.mtime().nsec, .ctime = @as(i128, s.ctime().sec) * 1_000_000_000 + s.ctime().nsec };
        }
        const s = try file.stat(io);
        if (s.kind != .file) return error.DigestNotRegular;
        return .{ .device = 0, .inode = s.inode, .size = s.size, .mtime = s.mtime.toNanoseconds(), .ctime = s.ctime.toNanoseconds() };
    }

    pub fn equal(a: Key, b: Key) bool {
        return a.device == b.device and a.inode == b.inode and a.size == b.size and
            a.mtime == b.mtime and a.ctime == b.ctime;
    }
};
