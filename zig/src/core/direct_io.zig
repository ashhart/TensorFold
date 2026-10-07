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
        var ignored: u8 = 0;
        return readAt(f, buf, offset, len, &ignored, sysPread);
    }
};

fn sysPread(_: *anyopaque, fd: std.c.fd_t, buf: [*]u8, len: usize, offset: u64) isize {
    return std.c.pread(fd, buf, len, @intCast(offset));
}

fn readAt(f: File, buf: Buffer, offset: u64, len: usize, ctx: *anyopaque, pread: *const fn (*anyopaque, std.c.fd_t, [*]u8, usize, u64) isize) ![]u8 {
    const lo = std.mem.alignBackward(u64, offset, alignment);
    const want: usize = @intCast(offset + len - lo);
    const whole = span(offset, len);
    if (whole > buf.len) return error.BufferTooSmall;
    var got: usize = 0;
    while (got < want) {
        // A short O_DIRECT read is retried from the last 4 KiB boundary so the next pread stays aligned.
        const at: usize = if (f.direct) std.mem.alignBackward(usize, got, alignment) else got;
        const ask = whole - at;
        const n = pread(ctx, f.fd, buf.ptr + at, ask, lo + @as(u64, @intCast(at)));
        if (n < 0) {
            if (std.c.errno(n) == .INTR) continue;
            std.log.err("read of {d} bytes at {d} failed: {t}", .{ ask, lo + at, std.c.errno(n) });
            return error.ReadFailed;
        }
        if (n == 0) return error.EndOfFile;
        if (n > @as(isize, @intCast(ask))) return error.ReadFailed;
        const next = at + @as(usize, @intCast(n));
        if (next <= got) return error.ReadFailed;
        got = next;
    }
    return buf[@intCast(offset - lo)..][0..len];
}

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

const Script = struct {
    plan: []const usize,
    offs: [8]u64 = undefined,
    i: usize = 0,
    calls: usize = 0,
    bad: bool = false,

    fn pread(ctx: *anyopaque, _: std.c.fd_t, buf: [*]u8, len: usize, offset: u64) isize {
        const s: *Script = @ptrCast(@alignCast(ctx));
        if (s.calls < s.offs.len) s.offs[s.calls] = offset;
        s.calls += 1;
        if (offset % alignment != 0 or len % alignment != 0 or @intFromPtr(buf) % alignment != 0) s.bad = true;
        const step = if (s.i < s.plan.len) s.plan[s.i] else len;
        if (s.i < s.plan.len) s.i += 1;
        const n = @min(step, len);
        var k: usize = 0;
        while (k < n) : (k += 1) buf[k] = @intCast((offset + k) % 251);
        return @intCast(n);
    }
};

fn direct(script: *Script, out: []u8, offset: u64) !void {
    var storage: [8192]u8 align(alignment) = undefined;
    const f: File = .{ .fd = -1, .direct = true };
    const got = try readAt(f, &storage, offset, out.len, script, Script.pread);
    @memcpy(out, got);
}

test "a short direct read is retried from the aligned boundary" {
    var script: Script = .{ .plan = &.{ 100, 4096 } };
    var got: [4096]u8 = undefined;
    try direct(&script, &got, 0);
    try std.testing.expect(!script.bad);
    try std.testing.expectEqual(@as(usize, 2), script.calls);
    try std.testing.expectEqual(@as(u64, 0), script.offs[0]);
    try std.testing.expectEqual(@as(u64, 0), script.offs[1]);
    for (got, 0..) |b, i| try std.testing.expectEqual(@as(u8, @intCast(i % 251)), b);
    var boundary: Script = .{ .plan = &.{ 4096, 4096 } };
    var wide: [8192]u8 = undefined;
    try direct(&boundary, &wide, 0);
    try std.testing.expect(!boundary.bad);
    try std.testing.expectEqual(@as(u64, 0), boundary.offs[0]);
    try std.testing.expectEqual(@as(u64, 4096), boundary.offs[1]);
    try std.testing.expectEqual(@as(u8, @intCast(4096 % 251)), wide[4096]);
}
