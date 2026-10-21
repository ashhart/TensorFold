//! The file work the snapshot and the rollback share: clones, hashed copies, whole writes and synced folders.
const std = @import("std");
const platform = @import("sys");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

const chunk = 4 << 20; // bytes a read or write moves at a time
const new_flags: platform.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .CLOEXEC = true };
const file_mode: platform.c.mode_t = 0o644;

/// The folder at `path` opened to read (a descriptor to lock or sync), negative when it cannot be.
pub fn openFolder(path: [:0]const u8) platform.c.fd_t {
    return platform.c.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
}

extern "c" fn clonefile(src: [*:0]const u8, dst: [*:0]const u8, flags: u32) c_int;
const clone_nofollow: u32 = 1;
const ficlone: c_int = 0x40049409; // Linux's _IOW(0x94, 9, int)

/// A clone of `src` at the new path `dst` (true), or false where this file system makes no clones.
pub fn clone(src: [:0]const u8, dst: [:0]const u8) !bool {
    switch (builtin.os.tag) {
        .macos => {
            const rc = clonefile(src, dst, clone_nofollow);
            if (rc == 0) return true;
            return switch (platform.c.errno(rc)) {
                .OPNOTSUPP, .XDEV, .INVAL => false,
                else => error.CloneFailed,
            };
        },
        .linux => {
            const in = platform.c.open(src, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
            if (in < 0) return error.ReadFailed;
            defer _ = platform.c.close(in);
            const out = platform.c.open(dst, new_flags, file_mode);
            if (out < 0) return error.WriteFailed;
            const rc = platform.c.ioctl(out, ficlone, in);
            const e = platform.c.errno(rc);
            _ = platform.c.close(out);
            if (rc == 0) return true;
            _ = platform.c.unlink(dst);
            return switch (e) {
                .OPNOTSUPP, .XDEV, .INVAL, .NOTTY, .NOSYS, .BADF => false,
                else => error.CloneFailed,
            };
        },
        else => return false,
    }
}

/// SHA-256 of the file at `path`.
pub fn hashFile(path: [:0]const u8) ![32]u8 {
    const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return error.ReadFailed;
    defer _ = platform.c.close(fd);
    const buf = try std.heap.page_allocator.alloc(u8, chunk);
    defer std.heap.page_allocator.free(buf);
    var h = Sha256.init(.{});
    while (try readSome(fd, buf)) |got| h.update(got);
    return h.finalResult();
}

/// `src` copied to the new file `dst`, synced; the SHA-256 of the bytes read from `src`.
pub fn copyHashed(src: [:0]const u8, dst: [:0]const u8) ![32]u8 {
    const in = platform.c.open(src, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    if (in < 0) return error.ReadFailed;
    defer _ = platform.c.close(in);
    const out = platform.c.open(dst, new_flags, file_mode);
    if (out < 0) return error.WriteFailed;
    defer _ = platform.c.close(out);
    const buf = try std.heap.page_allocator.alloc(u8, chunk);
    defer std.heap.page_allocator.free(buf);
    var h = Sha256.init(.{});
    while (try readSome(in, buf)) |got| {
        h.update(got);
        try writeAll(out, got);
    }
    if (platform.c.fsync(out) != 0) return error.WriteFailed;
    return h.finalResult();
}

/// The next bytes of `fd` in `buf`, or null at its end.
fn readSome(fd: platform.c.fd_t, buf: []u8) !?[]u8 {
    while (true) {
        const n = platform.c.read(fd, buf.ptr, buf.len);
        if (n < 0 and platform.c.errno(n) == .INTR) continue;
        if (n < 0) return error.ReadFailed;
        return if (n == 0) null else buf[0..@intCast(n)];
    }
}

fn writeAll(fd: platform.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = platform.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n < 0 and platform.c.errno(n) == .INTR) continue;
        if (n <= 0) return error.WriteFailed;
        done += @intCast(n);
    }
}

/// `bytes` to `path` through an exclusively created sibling file, synced, then renamed over it.
pub fn writeWhole(a: Allocator, path: []const u8, bytes: []const u8) !void {
    try platform.atomic_file.replace(a, path, &.{bytes});
}

pub fn rename(from: [:0]const u8, to: [:0]const u8) !void {
    if (platform.c.rename(from, to) != 0) return error.RenameFailed;
}

/// The folder's entries made durable (best effort: a file system that cannot sync a folder still has them).
pub fn syncDir(a: Allocator, dir: []const u8) void {
    const fd = openFolder(a.dupeSentinel(u8, dir, 0) catch return);
    if (fd < 0) return;
    _ = platform.c.fsync(fd);
    _ = platform.c.close(fd);
}
