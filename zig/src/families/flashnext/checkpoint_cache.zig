//! Drop fully copied sources after construction, preserving descriptors borrowed by runtime SSD readers.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const cache = @import("../../core/checkpoint_file_cache.zig");
const File = @import("../../core/fsr/file.zig").File;

pub fn finish(fds: []const platform.c.fd_t, runtime: []const File) void {
    copied: for (fds) |fd| {
        for (runtime) |file| if (file.fd == fd) continue :copied;
        cache.release(fd);
    }
}

extern "c" fn mincore(addr: *const anyopaque, len: usize, vec: [*]u8) c_int;

fn cached(fd: platform.c.fd_t, len: usize) !usize {
    const map = try std.posix.mmap(null, len, .{}, .{ .TYPE = .SHARED }, fd, 0);
    defer std.posix.munmap(map);
    const a = std.testing.allocator;
    const bits = try a.alloc(u8, (len + std.heap.pageSize() - 1) / std.heap.pageSize());
    defer a.free(bits);
    try std.testing.expectEqual(@as(c_int, 0), mincore(map.ptr, len, bits.ptr));
    var pages: usize = 0;
    for (bits) |b| pages += b & 1;
    return pages;
}

test "Flash Next copied cache drops, runtime SSD cache stays, both descriptors and bytes survive" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const data = try a.alloc(u8, (4 << 20) + 17);
    defer a.free(data);
    for (data, 0..) |*v, i| v.* = @truncate(i *% 1868 +% 1039);
    for ([_][]const u8{ "copy", "runtime" }) |name| try tmp.dir.writeFile(io, .{ .sub_path = name, .data = data });
    const copy = try tmp.dir.openFile(io, "copy", .{});
    defer copy.close(io);
    const runtime = try tmp.dir.openFile(io, "runtime", .{});
    defer runtime.close(io);
    const fds = [_]platform.c.fd_t{ copy.handle, runtime.handle };
    for (fds) |fd| {
        try std.testing.expectEqual(@as(c_int, 0), platform.c.fsync(fd));
        try std.testing.expect(try cached(fd, data.len) > 0);
    }
    const live = [_]File{.{ .fd = runtime.handle, .size = data.len }};
    finish(&fds, &live);
    try std.testing.expectEqual(@as(usize, 0), try cached(copy.handle, data.len));
    try std.testing.expect(try cached(runtime.handle, data.len) > 0);
    const got = try a.alloc(u8, data.len);
    defer a.free(got);
    for (fds) |fd| {
        try std.testing.expectEqual(@as(isize, @intCast(data.len)), platform.c.pread(fd, got.ptr, got.len, 0));
        try std.testing.expectEqualSlices(u8, data, got);
    }
    // Omitting the live descriptor would evict its cache too: the exemption has a real positive control.
    finish(&fds, &.{});
    try std.testing.expectEqual(@as(usize, 0), try cached(runtime.handle, data.len));
}
