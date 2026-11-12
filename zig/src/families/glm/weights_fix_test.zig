//! Actual copied-shard owners release the cache after worker joins on success and failure.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const fix = @import("weights_fix.zig");

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

test "GLM copies and transforms join before shard eviction; errors and duplicate opens retain ownership" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const data = try a.alloc(u8, (4 << 20) + 18);
    defer a.free(data);
    for (0..data.len / 2) |i| std.mem.writeInt(u16, data[i * 2 ..][0..2], @as(u16, 0x3f80) + @as(u16, @intCast(i % 16)), .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "shard", .data = data });
    const path = try tmp.dir.realPathFileAlloc(io, "shard", a);
    defer a.free(path);
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    const file = try tmp.dir.openFile(io, "shard", .{});
    defer file.close(io);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.fsync(file.handle));
    try std.testing.expect(try cached(file.handle, data.len) > 0);
    const out = try a.alloc(u8, data.len);
    defer a.free(out);
    var widened: [16]f32 = undefined;
    const copy = [_]fix.Copy{.{ .shard = 0, .off = 0, .len = data.len, .dst = out.ptr }};
    const tx = [_]fix.Transform{.{ .src = .{ .shard = 0, .off = 0, .len = 32, .dtype = .bf16, .shape = @splat(1), .rank = 1 }, .dst = @ptrCast(&widened), .fix = .to_f32 }};
    try fix.runCopies(a, &.{z}, &copy, &tx, 2);
    try std.testing.expectEqual(@as(usize, 0), try cached(file.handle, data.len));
    try std.testing.expectEqualSlices(u8, data, out);
    for (widened, 0..) |v, i| try std.testing.expectEqual(@as(f32, @bitCast(@as(u32, 0x3f80 + @as(u32, @intCast(i))) << 16)), v);
    try std.testing.expectEqual(@as(isize, @intCast(data.len)), platform.c.pread(file.handle, out.ptr, out.len, 0));
    try std.testing.expectEqualSlices(u8, data, out);
    var short: [4]u8 = undefined;
    const bad = [_]fix.Copy{.{ .shard = 0, .off = data.len - 2, .len = 4, .dst = &short }};
    try std.testing.expectError(error.ShortRead, fix.runCopies(a, &.{z}, &bad, &.{}, 2));
    try std.testing.expectEqual(@as(usize, 0), try cached(file.handle, data.len));
    // a transform's own read past the end (weights_fix.zig:49) fails the load too
    const past = [_]fix.Transform{.{ .src = .{ .shard = 0, .off = data.len - 16, .len = 32, .dtype = .bf16, .shape = @splat(1), .rank = 1 }, .dst = @ptrCast(&widened), .fix = .to_f32 }};
    try std.testing.expectError(error.TransformFailed, fix.runCopies(a, &.{z}, &.{}, &past, 2));
    var fd: platform.c.fd_t = undefined;
    {
        var files: fix.Files = .{ .a = a };
        defer files.deinit();
        fd = try files.open(z);
        try std.testing.expectEqual(fd, try files.open(z));
        try std.testing.expectEqual(@as(usize, 1), files.fds.items.len);
        const missing = try std.fmt.allocPrintSentinel(a, "{s}.absent", .{z}, 0);
        defer a.free(missing);
        try std.testing.expectError(error.OpenFailed, files.open(missing));
    }
    try std.testing.expectEqual(@as(c_int, -1), platform.c.fcntl(fd, platform.c.F.GETFD));
}
