//! Closing copied safetensors unmaps before eviction; ordinary borrowed-map closure keeps its existing policy.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const st = @import("safetensors.zig");

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

test "copied safetensors close drops cache after unmapping and preserves the stored and copied payload" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const n = (4 << 20) + 17;
    const header = try std.fmt.allocPrint(a, "{{\"payload\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}", .{ n, n });
    defer a.free(header);
    const bytes = try a.alloc(u8, 8 + header.len + n);
    defer a.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    for (bytes[8 + header.len ..], 0..) |*v, i| v.* = @truncate(i *% 1868 +% 441);
    try tmp.dir.writeFile(io, .{ .sub_path = "copy.safetensors", .data = bytes });
    const path = try tmp.dir.realPathFileAlloc(io, "copy.safetensors", a);
    defer a.free(path);
    const observer = try tmp.dir.openFile(io, "copy.safetensors", .{});
    defer observer.close(io);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.fsync(observer.handle));
    const copy = try a.alloc(u8, n);
    defer a.free(copy);
    var ordinary = try st.File.open(a, io, path);
    @memcpy(copy, ordinary.get("payload").?.bytes);
    ordinary.close(io);
    try std.testing.expect(try cached(observer.handle, bytes.len) > 0);
    var copied = try st.File.open(a, io, path);
    @memcpy(copy, copied.get("payload").?.bytes);
    copied.closeCopied(io);
    try std.testing.expectEqual(@as(usize, 0), try cached(observer.handle, bytes.len));
    try std.testing.expectEqualSlices(u8, bytes[8 + header.len ..], copy);
    const stored = try tmp.dir.readFileAlloc(io, "copy.safetensors", a, .limited(bytes.len + 1));
    defer a.free(stored);
    try std.testing.expectEqualSlices(u8, bytes, stored);
    var layer: @import("exlr8_experts.zig").Layer = .{ .number = 0, .index = try st.File.open(a, io, path), .signs_in = &.{}, .signs_out = &.{} };
    layer.files[2] = try st.File.open(a, io, path);
    layer.files[4] = try st.File.open(a, io, path);
    const fds = [_]platform.c.fd_t{ layer.index.file.handle, layer.files[2].?.file.handle, layer.files[4].?.file.handle };
    layer.closeCopied(io);
    try std.testing.expectEqual(@as(usize, 0), try cached(observer.handle, bytes.len));
    for (fds) |fd| try std.testing.expectEqual(@as(c_int, -1), platform.c.fcntl(fd, platform.c.F.GETFD));
}
