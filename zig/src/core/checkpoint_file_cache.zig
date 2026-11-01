//! Best-effort release of a copied shard's file-cache pages when its read-only descriptor is about to close.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");

/// Only after all readers joined and any deferred copies finished; this never touches the copied Metal buffers.
pub fn release(fd: platform.c.fd_t) void {
    if (builtin.os.tag == .macos) {
        const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
        if (end <= 0) return;
        const len = std.math.cast(usize, end) orelse return;
        const map = std.posix.mmap(null, len, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0) catch return;
        defer std.posix.munmap(map);
        // std.posix.msync treats EBUSY as unreachable; a cache hint must also tolerate locked pages and I/O errors.
        _ = platform.c.msync(map.ptr, map.len, platform.c.MSF.SYNC | platform.c.MSF.INVALIDATE);
    }
}

extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;

fn cached(fd: platform.c.fd_t, len: usize) !usize {
    const map = try std.posix.mmap(null, len, .{}, .{ .TYPE = .SHARED }, fd, 0);
    defer std.posix.munmap(map);
    const vec = try std.testing.allocator.alloc(u8, (len + std.heap.pageSize() - 1) / std.heap.pageSize());
    defer std.testing.allocator.free(vec);
    try std.testing.expectEqual(@as(c_int, 0), mincore(map.ptr, len, vec.ptr));
    var n: usize = 0;
    for (vec) |v| n += v & 1;
    return n;
}

test "copied shard pages leave the file cache without changing the copy or stored bytes" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const data = try a.alloc(u8, (4 << 20) + 17);
    defer a.free(data);
    for (data, 0..) |*v, i| v.* = @truncate(i *% 1772 +% 1039);
    try tmp.dir.writeFile(io, .{ .sub_path = "shard.bin", .data = data });
    const path = try tmp.dir.realPathFileAlloc(io, "shard.bin", a);
    defer a.free(path);
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    const fd = platform.c.open(z, .{ .ACCMODE = .RDONLY });
    try std.testing.expect(fd >= 0);
    defer _ = platform.c.close(fd);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.fsync(fd));
    const copy = try a.alloc(u8, data.len);
    defer a.free(copy);
    try std.testing.expectEqual(@as(isize, @intCast(copy.len)), platform.c.pread(fd, copy.ptr, copy.len, 0));
    try std.testing.expect(try cached(fd, data.len) > 0);
    release(fd);
    try std.testing.expectEqual(@as(usize, 0), try cached(fd, data.len));
    try std.testing.expectEqualSlices(u8, data, copy);
    const stored = try tmp.dir.readFileAlloc(io, "shard.bin", a, .limited(data.len + 1));
    defer a.free(stored);
    try std.testing.expectEqualSlices(u8, data, stored);
    release(-1); // Failure of a hint cannot become a checkpoint-open failure.
}

test "empty shards need no mapping to release their cache" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(io, "empty", .{});
    defer file.close(io);
    release(file.handle);
    try std.testing.expectEqual(@as(u64, 0), try file.length(io));
}
