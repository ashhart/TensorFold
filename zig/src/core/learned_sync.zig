//! Learned-state durability: flush the device cache on Darwin, not just the kernel's buffers.
const std = @import("std");
const c = @import("sys").c;
const builtin = @import("builtin");

pub fn full(fd: c_int) bool {
    if (fd < 0) return false;
    if (builtin.os.tag.isDarwin()) {
        const e = c.errno(c.fcntl(fd, c.F.FULLFSYNC));
        return afterFull(fd, e);
    }
    return fd >= 0 and c.fsync(fd) == 0;
}

fn afterFull(fd: c_int, e: c.E) bool {
    if (e == .SUCCESS) return true;
    if (!unsupported(e)) return false;
    return fd >= 0 and c.fsync(fd) == 0;
}

fn unsupported(e: c.E) bool {
    return e == .NOTTY or e == .OPNOTSUPP or e == .INVAL;
}

test "learned full sync falls back only for unsupported operations" {
    for ([_]c.E{ .NOTTY, .OPNOTSUPP, .INVAL }) |e| try std.testing.expect(unsupported(e));
    for ([_]c.E{ .IO, .NODEV, .BADF, .NOSPC, .ACCES }) |e| try std.testing.expect(!unsupported(e));
    try std.testing.expect(!full(-1));
}

test "learned full sync does not hide device errors behind a successful fsync" {
    var buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "/tmp/tf-learn-sync-{d}", .{c.getpid()}, 0);
    const fd = c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    defer _ = c.unlink(path);
    defer _ = c.close(fd);
    try std.testing.expect(c.fsync(fd) == 0); // a fallback would conceal the injected full-sync error
    for ([_]c.E{ .IO, .NODEV, .BADF, .NOSPC, .ACCES }) |e| try std.testing.expect(!afterFull(fd, e));
    for ([_]c.E{ .NOTTY, .OPNOTSUPP, .INVAL }) |e| try std.testing.expect(afterFull(fd, e));
    try std.testing.expect(!afterFull(-1, .NOTTY)); // fallback failure is propagated too
    try std.testing.expect(full(fd));
}
