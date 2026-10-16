//! GLM learned halves use a versioned checksum envelope; legacy, corrupt and partial files never enter a cache.
const std = @import("std");
const platform = @import("sys");
const checked = @import("../../core/snapshot_file.zig");
const magic: u32 = 0x474c4d54;
pub const Head = extern struct { magic: u32 = magic, version: u32 = 1, at: u32, reserved: u32 = 0, bytes: u64 };
pub fn write(file: [:0]const u8, at: u32, payload: []const u8) !void {
    var tmp_buf: [1200]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&tmp_buf, "{s}.part", .{file}, 0);
    const fd = platform.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return error.SnapshotWrite;
    errdefer _ = platform.c.unlink(tmp);
    {
        defer _ = platform.c.close(fd);
        const head: Head = .{ .at = at, .bytes = payload.len };
        try checked.write(fd, std.mem.asBytes(&head), payload);
        if (platform.c.fsync(fd) != 0) return error.SnapshotWrite;
    }
    if (platform.c.rename(tmp, file) != 0) return error.SnapshotWrite;
}
pub fn read(file: [:0]const u8, at: u32, payload: []u8) !void {
    const fd = platform.c.open(file, .{ .ACCMODE = .RDONLY }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.SnapshotRead;
    defer _ = platform.c.close(fd);
    const expected: Head = .{ .at = at, .bytes = payload.len };
    var head: Head = undefined;
    if (platform.c.pread(fd, std.mem.asBytes(&head).ptr, @sizeOf(Head), 0) != @sizeOf(Head) or !std.mem.eql(u8, std.mem.asBytes(&head), std.mem.asBytes(&expected))) return error.SnapshotRead;
    try checked.read(fd, std.mem.asBytes(&head), payload);
}
test "both GLM half files refuse a changed body, legacy header, truncation and another position" {
    var path: [160]u8 = undefined;
    const file = try std.fmt.bufPrintSentinel(&path, "/tmp/tf-glm-file-{d}", .{platform.c.getpid()}, 0);
    defer _ = platform.c.unlink(file);
    try write(file, 128, "body");
    var out: [4]u8 = undefined;
    try read(file, 128, &out);
    try std.testing.expectEqualStrings("body", &out);
    try std.testing.expectError(error.SnapshotRead, read(file, 129, &out));
    const fd = platform.c.open(file, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.TestFile;
    defer _ = platform.c.close(fd);
    try std.testing.expectEqual(@as(isize, 1), platform.c.pwrite(fd, "X", 1, @sizeOf(Head) + 1));
    try std.testing.expectError(error.SnapshotRead, read(file, 128, &out));
    try std.testing.expectEqual(@as(c_int, 0), platform.c.ftruncate(fd, 16));
    try std.testing.expectError(error.SnapshotRead, read(file, 128, &out));
}
