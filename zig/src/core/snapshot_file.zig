//! Learned snapshots carry a SHA-256 of their entire header and payload; exact length is required on admission.
const std = @import("std");
const platform = @import("sys");

const Hash = std.crypto.hash.sha2.Sha256;

/// Private, exclusively staged bytes published atomically, with the directory entry made durable.
pub fn publish(file: []const u8, parts: []const []const u8) !void {
    platform.atomic_file.replacePrivate(std.heap.page_allocator, file, parts) catch return error.SnapshotWrite;
    platform.atomic_file.syncParent(file) catch return error.SnapshotWrite;
}

/// A checksummed header and payload into `file`, with exact bytes unchanged.
pub fn replace(file: []const u8, header: []const u8, payload: []const u8) !void {
    var hash = Hash.init(.{});
    hash.update(header);
    hash.update(payload);
    try publish(file, &.{ header, payload, &hash.finalResult() });
}

pub fn read(fd: c_int, header: []const u8, payload: []u8) !void {
    const length = std.math.add(usize, header.len, payload.len) catch return error.SnapshotRead;
    const total = std.math.add(usize, length, 32) catch return error.SnapshotRead;
    if (platform.c.lseek(fd, 0, platform.c.SEEK.END) != total) return error.SnapshotRead;
    if (!get(fd, payload, header.len)) return error.SnapshotRead;
    var stored: [32]u8 = undefined;
    if (!get(fd, &stored, length)) return error.SnapshotRead;
    var hash = Hash.init(.{});
    hash.update(header);
    hash.update(payload);
    if (!std.mem.eql(u8, &stored, &hash.finalResult())) return error.SnapshotRead;
}

test "payload bit flips, header changes, truncation, legacy files and extra bytes are refused" {
    var path: [128]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&path, "/tmp/tf-snapshot-test-{d}", .{platform.c.getpid()}, 0);
    defer _ = platform.c.unlink(name);
    try replace(name, "head", "body");
    const fd = platform.c.open(name, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    defer _ = platform.c.close(fd);
    var output: [4]u8 = undefined;
    try read(fd, "head", &output);
    try std.testing.expectEqualStrings("body", &output);
    try std.testing.expectError(error.SnapshotRead, read(fd, "dead", &output));
    try std.testing.expectEqual(@as(isize, 1), platform.c.pwrite(fd, "X", 1, 5));
    try std.testing.expectError(error.SnapshotRead, read(fd, "head", &output));
    try std.testing.expectEqual(@as(c_int, 0), platform.c.ftruncate(fd, 8));
    try std.testing.expectError(error.SnapshotRead, read(fd, "head", &output));
    try replace(name, "head", "body");
    const extra = platform.c.open(name, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    if (extra < 0) return error.TestFile;
    defer _ = platform.c.close(extra);
    try std.testing.expectEqual(@as(isize, 1), platform.c.pwrite(extra, "x", 1, 40));
    try std.testing.expectError(error.SnapshotRead, read(extra, "head", &output));
}
fn get(fd: c_int, bytes: []u8, at: usize) bool {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = platform.c.pread(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}
