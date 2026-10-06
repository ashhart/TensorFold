//! Positional reads that skip the page cache (O_DIRECT on Linux): 4 KiB-aligned spans into aligned buffers.

const std = @import("std");
const builtin = @import("builtin");

/// Offset, length and buffer alignment a direct read needs (the logical block size of every NVMe we run on).
pub const alignment = 4096;

pub const Buffer = []align(alignment) u8;

/// Bytes a read of `len` bytes at `offset` occupies once widened to aligned ends.
pub fn span(offset: u64, len: usize) usize {
    const lo = std.mem.alignBackward(u64, offset, alignment);
    return @intCast(std.mem.alignForward(u64, offset + len, alignment) - lo);
}

/// The largest read whose span always fits a buffer of `bytes` (whatever its offset).
pub fn fits(bytes: usize) usize {
    return bytes - 2 * alignment;
}

pub const File = struct {
    fd: std.c.fd_t,
    direct: bool,

    /// `path` read-only, direct when its file system allows it, else through the page cache.
    pub fn open(path: [:0]const u8) !File {
        if (builtin.os.tag == .linux) {
            const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .DIRECT = true });
            if (fd >= 0) return .{ .fd = fd, .direct = true };
        }
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (fd < 0) return error.FileNotFound;
        return .{ .fd = fd, .direct = false };
    }

    pub fn close(f: *File) void {
        _ = std.c.close(f.fd);
        f.* = undefined;
    }

    /// `len` bytes at `offset`, read as their aligned span into `buf`; the result is the requested bytes within it.
    pub fn read(f: File, buf: Buffer, offset: u64, len: usize) ![]u8 {
        const lo = std.mem.alignBackward(u64, offset, alignment);
        const want: usize = @intCast(offset + len - lo);
        const whole = span(offset, len);
        if (whole > buf.len) return error.BufferTooSmall;
        var got: usize = 0;
        while (got < want) {
            const n = std.c.pread(f.fd, buf.ptr + got, whole - got, @intCast(lo + got));
            if (n < 0) {
                if (std.c.errno(n) == .INTR) continue;
                std.log.err("read of {d} bytes at {d} failed: {t}", .{ whole - got, lo + got, std.c.errno(n) });
                return error.ReadFailed;
            }
            if (n == 0) return error.EndOfFile;
            got += @intCast(n);
        }
        return buf[@intCast(offset - lo)..][0..len];
    }
};

test "span and fits" {
    try std.testing.expectEqual(@as(usize, 4096), span(0, 1));
    try std.testing.expectEqual(@as(usize, 8192), span(4095, 2));
    try std.testing.expectEqual(@as(usize, 4096), span(4096, 4096));
    try std.testing.expectEqual(@as(usize, 3 * 4096), span(100, 2 * 4096));
    try std.testing.expect(span(4095, fits(1 << 20)) <= 1 << 20);
}

test "direct reads return the same bytes as buffered reads" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const gpa = std.testing.allocator;
    var f = try File.open("/proc/self/exe");
    defer f.close();
    var plain = try File.open("/proc/self/exe");
    defer plain.close();
    plain.direct = false;
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 1 << 20);
    defer gpa.free(buf);
    const want = try gpa.alloc(u8, 1 << 19);
    defer gpa.free(want);
    for ([_][2]usize{ .{ 0, 1 }, .{ 1, 4095 }, .{ 4095, 2 }, .{ 12345, 100000 }, .{ 8192, 1 << 19 } }) |c| {
        const n = std.c.pread(plain.fd, want.ptr, c[1], @intCast(c[0]));
        try std.testing.expectEqual(@as(isize, @intCast(c[1])), n);
        try std.testing.expectEqualSlices(u8, want[0..c[1]], try f.read(buf, c[0], c[1]));
    }
    try std.testing.expectError(error.BufferTooSmall, f.read(buf, 1, 1 << 20));
}
