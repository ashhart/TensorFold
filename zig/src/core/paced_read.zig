//! Checkpoint reads that keep out of the page cache and wait for free memory: a load never outruns the kernel's reclaim.
const std = @import("std");

/// Offsets, lengths and buffers of uncached reads align to the VM page (F_NOCACHE only skips the cache for aligned reads).
pub const alignment = 16384;

/// A reading thread's aligned staging buffer.
pub const Stage = []align(alignment) u8;

/// A thread's staging buffer of `bytes` (a multiple of `alignment`).
pub fn stage(gpa: std.mem.Allocator, bytes: usize) !Stage {
    return gpa.alignedAlloc(u8, .fromByteUnits(alignment), bytes);
}

/// Free memory the system keeps before a read goes ahead: 3% of RAM, at least 8 GB.
pub fn floor() u64 {
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &mem, &len, null, 0) != 0) mem = 0;
    return @max(8 << 30, mem / 100 * 3);
}

pub fn freeBytes() u64 {
    var pages: u32 = 0;
    var len: usize = @sizeOf(u32);
    if (std.c.sysctlbyname("vm.page_free_count", &pages, &len, null, 0) != 0) return std.math.maxInt(u64);
    var size: u64 = 0;
    var len2: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.pagesize", &size, &len2, null, 0) != 0) size = alignment;
    return @as(u64, pages) * size;
}

/// Wait (up to `limit_ms`) while free memory is under the floor, so the kernel reclaims before the next read commits pages.
pub fn waitForRoom(limit_ms: u32) void {
    const want = floor();
    var waited: u32 = 0;
    while (waited < limit_ms and freeBytes() < want) : (waited += 2) {
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 2 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
}

/// `dest.len` bytes at `at` of `fd` (opened with F_NOCACHE) through page-aligned reads into `buf`, pausing for free memory.
pub fn read(fd: std.c.fd_t, dest: []u8, at: u64, buf: Stage) !void {
    var done: usize = 0;
    while (done < dest.len) {
        waitForRoom(10_000);
        const pos = at + done;
        const lo = std.mem.alignBackward(u64, pos, alignment);
        const skip: usize = @intCast(pos - lo);
        const want = @min(dest.len - done, buf.len - skip);
        const span = std.mem.alignForward(usize, skip + want, alignment);
        var got: usize = 0;
        while (got < skip + want) { // a read past the file's end comes back short once the bytes asked for are in
            const n = std.c.pread(fd, buf.ptr + got, span - got, @intCast(lo + got));
            if (n <= 0) return error.ShortRead;
            got += @intCast(n);
        }
        @memcpy(dest[done..][0..want], buf[skip..][0..want]);
        done += want;
    }
}

test "aligned spans copy the bytes asked for, at any offset and length" {
    const gpa = std.testing.allocator;
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&name_buf, "/tmp/tf-paced-read-{d}", .{std.c.getpid()}, 0);
    const fd = std.c.open(name, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    defer {
        _ = std.c.close(fd);
        _ = std.c.unlink(name);
    }
    const data = try gpa.alloc(u8, 3 * alignment + 777);
    defer gpa.free(data);
    for (data, 0..) |*b, i| b.* = @truncate(i *% 131 +% 7);
    try std.testing.expectEqual(@as(isize, @intCast(data.len)), std.c.write(fd, data.ptr, data.len));
    const buf = try stage(gpa, 2 * alignment);
    defer gpa.free(buf);
    for ([_][2]usize{ .{ 0, 1 }, .{ 5, 100 }, .{ alignment - 3, 10 }, .{ 1, 2 * alignment + 50 }, .{ 3 * alignment, 777 }, .{ 9, data.len - 9 } }) |c| {
        const out = try gpa.alloc(u8, c[1]);
        defer gpa.free(out);
        try read(fd, out, c[0], buf);
        try std.testing.expectEqualSlices(u8, data[c[0]..][0..c[1]], out);
    }
}
