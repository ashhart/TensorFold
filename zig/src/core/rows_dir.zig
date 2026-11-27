//! Pinned learned-row directories: descriptor-relative names, private staging, and directory durability.
const std = @import("std");
const platform = @import("sys");
const builtin = @import("builtin");

const windows_test = builtin.is_test and platform.windows;
const RenameAt = *const fn (c_int, [*:0]const u8, c_int, [*:0]const u8) callconv(.c) c_int;
/// Windows tests enter this direct Zig boundary through the fixture's exported wrapper.
pub var test_renameat: if (windows_test) ?RenameAt else void = if (windows_test) null else {};

var serial: std.atomic.Value(u64) = .init(0);

/// `path` opened without following its final link, or a symlinked/non-directory object refused as `RowsDir`.
pub fn openPath(path: [:0]const u8) !c_int {
    const fd = platform.c.open(path, directoryFlags(), @as(platform.c.mode_t, 0));
    return if (fd < 0) error.RowsDir else fd;
}

/// The configured root may be an alias; its resolved directory is pinned before any identity child is opened.
pub fn openRoot(path: [:0]const u8) !c_int {
    const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    return if (fd < 0) error.RowsDir else fd;
}

/// `name` under `fd` opened without following it, so later work cannot escape the pinned parent.
pub fn openAt(fd: c_int, name: [*:0]const u8) !c_int {
    const out = platform.c.openat(fd, name, directoryFlags(), @as(platform.c.mode_t, 0));
    return if (out < 0) error.RowsDir else out;
}

/// `name` created if missing under the pinned root, then opened without tolerating an alias or a non-directory.
pub fn openIdentity(root: c_int, name: [*:0]const u8) !c_int {
    const made = mkdirAt(root, name, 0o700);
    if (made != 0 and platform.c.errno(made) != .EXIST) return error.RowsDir;
    return openAt(root, name);
}

/// An independently owned descriptor of the same directory; a detached worker can therefore outlive its store.
pub fn clone(fd: c_int) !c_int {
    if (comptime platform.windows) return openAt(fd, ".");
    const out = platform.c.fcntl(fd, platform.c.F.DUPFD_CLOEXEC, @as(c_int, 0));
    return if (out < 0) error.RowsDir else out;
}

pub fn close(fd: c_int) void {
    if (fd >= 0) _ = platform.c.close(fd);
}

/// A scan takes its own descriptor because `fdopendir` transfers ownership to the stream.
pub fn scan(fd: c_int) !*platform.c.DIR {
    const copy = try clone(fd);
    return platform.c.fdopendir(copy) orelse {
        close(copy);
        return error.RowsDir;
    };
}

/// The pinned directory itself, not a reopened parent pathname, is made durable.
pub fn sync(fd: c_int) !void {
    if (comptime platform.windows) return;
    while (true) {
        const rc = platform.c.fsync(fd);
        if (rc == 0) return;
        if (platform.c.errno(rc) == .INTR) continue;
        return error.RowsDir;
    }
}

/// `name` removed under the pinned directory; a successful removal is followed by its directory sync.
pub fn unlink(fd: c_int, name: [*:0]const u8) !void {
    const rc = platform.c.unlinkat(fd, name, 0);
    if (rc != 0 and platform.c.errno(rc) == .NOENT) return;
    if (rc != 0) return error.RowsDir;
    try sync(fd);
}

/// A staged leaf made visible under the same pinned directory and its rename made durable.
pub fn rename(fd: c_int, temporary: [*:0]const u8, destination: [*:0]const u8) !void {
    if (renameAt(fd, temporary, fd, destination) != 0) return error.RowsDir;
    try sync(fd);
}

/// The exact file size of `name`, with links refused.
pub fn fileBytes(fd: c_int, name: [*:0]const u8) !u64 {
    const file = platform.c.openat(fd, name, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    if (file < 0) return error.RowsDir;
    defer close(file);
    const end = platform.c.lseek(file, 0, platform.c.SEEK.END);
    if (end < 0) return error.RowsDir;
    return @intCast(end);
}

/// `dest` read exactly from `name`, or false when it is absent, linked, short, or unreadable.
pub fn read(fd: c_int, name: [*:0]const u8, dest: []u8) bool {
    const file = platform.c.openat(fd, name, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0));
    if (file < 0) return false;
    defer close(file);
    return platform.c.pread(file, dest.ptr, dest.len, 0) == dest.len;
}

/// The accounting word `name` updated in place, preserving its existing format and non-atomic daily accounting.
pub fn update(fd: c_int, name: [*:0]const u8, bytes: []const u8) void {
    const file = platform.c.openat(fd, name, .{ .ACCMODE = .WRONLY, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0o600));
    if (file < 0) return;
    defer close(file);
    _ = platform.c.pwrite(file, bytes.ptr, bytes.len, 0);
}

/// A recency word staged under a private exclusive name, published, and directory-synced on a best-effort basis.
pub fn writeWord(fd: c_int, name: [*:0]const u8, word: u64) void {
    var buf: [128]u8 = undefined;
    for (0..32) |_| {
        const nonce = serial.fetchAdd(1, .monotonic);
        const part = std.fmt.bufPrintSentinel(&buf, "{s}.tmp.{d}.{d}", .{ name, platform.c.getpid(), nonce }, 0) catch return;
        const file = platform.c.openat(fd, part.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(platform.c.mode_t, 0o600));
        if (file < 0) {
            if (platform.c.errno(file) == .EXIST) continue;
            return;
        }
        const bytes = std.mem.asBytes(&word);
        const wrote = platform.c.pwrite(file, bytes.ptr, bytes.len, 0) == bytes.len;
        close(file);
        if (!wrote) {
            unlink(fd, part.ptr) catch {};
            return;
        }
        rename(fd, part.ptr, name) catch {
            unlink(fd, part.ptr) catch {};
        };
        return;
    }
}

/// `root` and its missing parents made 0700, before the root's final component is opened without following links.
pub fn makePath(a: std.mem.Allocator, root: []const u8) !void {
    const tmp = try a.dupeSentinel(u8, root, 0);
    defer a.free(tmp);
    for (tmp[1..], 1..) |ch, i| if (ch == '/') {
        tmp[i] = 0;
        _ = platform.c.mkdir(tmp.ptr, 0o700);
        tmp[i] = '/';
    };
    const rc = platform.c.mkdir(tmp.ptr, 0o700);
    if (rc != 0 and platform.c.errno(rc) != .EXIST) return error.RowsDir;
}

fn directoryFlags() platform.c.O {
    return .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true };
}

fn mkdirAt(fd: c_int, name: [*:0]const u8, mode: platform.c.mode_t) c_int {
    return platform.c.mkdirat(fd, name, mode);
}

fn renameAt(old_fd: c_int, old: [*:0]const u8, new_fd: c_int, new: [*:0]const u8) c_int {
    if (comptime windows_test) if (test_renameat) |hook| return hook(old_fd, old, new_fd, new);
    return platform.c.renameat(old_fd, old, new_fd, new);
}
