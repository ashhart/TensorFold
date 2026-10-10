//! Learned snapshots carry a SHA-256 of their entire header and payload; exact length is required on admission.
const std = @import("std");

const Hash = std.crypto.hash.sha2.Sha256;

pub fn write(fd: c_int, header: []const u8, payload: []const u8) !void {
    var hash = Hash.init(.{});
    hash.update(header);
    hash.update(payload);
    try put(fd, header);
    try put(fd, payload);
    try put(fd, &hash.finalResult());
}

pub fn read(fd: c_int, header: []const u8, payload: []u8) !void {
    const length = std.math.add(usize, header.len, payload.len) catch return error.SnapshotRead;
    const total = std.math.add(usize, length, 32) catch return error.SnapshotRead;
    if (std.c.lseek(fd, 0, std.c.SEEK.END) != total) return error.SnapshotRead;
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
    const name = try std.fmt.bufPrintSentinel(&path, "/tmp/tf-snapshot-test-{d}", .{std.c.getpid()}, 0);
    defer _ = std.c.unlink(name);
    const fd = std.c.open(name, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    defer _ = std.c.close(fd);
    var output: [4]u8 = undefined;
    try write(fd, "head", "body");
    try read(fd, "head", &output);
    try std.testing.expectEqualStrings("body", &output);
    try std.testing.expectError(error.SnapshotRead, read(fd, "dead", &output));
    try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, "X", 1, 5));
    try std.testing.expectError(error.SnapshotRead, read(fd, "head", &output));
    try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(fd, 8));
    try std.testing.expectError(error.SnapshotRead, read(fd, "head", &output));
    _ = std.c.lseek(fd, 0, std.c.SEEK.SET);
    try write(fd, "head", "body");
    try put(fd, "x");
    try std.testing.expectError(error.SnapshotRead, read(fd, "head", &output));
}

fn put(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30));
        if (n <= 0) return error.SnapshotWrite;
        done += @intCast(n);
    }
}
fn get(fd: c_int, bytes: []u8, at: usize) bool {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pread(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}
