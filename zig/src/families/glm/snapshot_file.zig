//! GLM learned halves use a versioned checksum envelope; legacy, corrupt and partial files never enter a cache.
const std = @import("std");
const checked = @import("../../core/snapshot_file.zig");
const magic: u32 = 0x474c4d54;
pub const Head = extern struct { magic: u32 = magic, version: u32 = 1, at: u32, reserved: u32 = 0, bytes: u64 };
pub fn write(file: [:0]const u8, at: u32, payload: []const u8) !void {
    var tmp_buf: [1200]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&tmp_buf, "{s}.part", .{file}, 0);
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.SnapshotWrite;
    errdefer _ = std.c.unlink(tmp);
    {
        defer _ = std.c.close(fd);
        const head: Head = .{ .at = at, .bytes = payload.len };
        try checked.write(fd, std.mem.asBytes(&head), payload);
        if (std.c.fsync(fd) != 0) return error.SnapshotWrite;
    }
    if (std.c.rename(tmp, file) != 0) return error.SnapshotWrite;
}
pub fn read(file: [:0]const u8, at: u32, payload: []u8) !void {
    const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.SnapshotRead;
    defer _ = std.c.close(fd);
    const expected: Head = .{ .at = at, .bytes = payload.len };
    var head: Head = undefined;
    if (std.c.pread(fd, std.mem.asBytes(&head).ptr, @sizeOf(Head), 0) != @sizeOf(Head) or !std.mem.eql(u8, std.mem.asBytes(&head), std.mem.asBytes(&expected))) return error.SnapshotRead;
    try checked.read(fd, std.mem.asBytes(&head), payload);
}
test "both GLM half files refuse a changed body, legacy header, truncation and another position" {
    var path: [160]u8 = undefined;
    const file = try std.fmt.bufPrintSentinel(&path, "/tmp/tf-glm-file-{d}", .{std.c.getpid()}, 0);
    defer _ = std.c.unlink(file);
    try write(file, 128, "body");
    var out: [4]u8 = undefined;
    try read(file, 128, &out);
    try std.testing.expectEqualStrings("body", &out);
    try std.testing.expectError(error.SnapshotRead, read(file, 129, &out));
    const fd = std.c.open(file, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.TestFile;
    defer _ = std.c.close(fd);
    try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, "X", 1, @sizeOf(Head) + 1));
    try std.testing.expectError(error.SnapshotRead, read(file, 128, &out));
    try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(fd, 16));
    try std.testing.expectError(error.SnapshotRead, read(file, 128, &out));
}
