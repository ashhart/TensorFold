//! A rank group's shared key: the SHA-256 of an owner-only file, 32 random bytes per nonce, HMAC-SHA256 over parts.
const std = @import("std");
const builtin = @import("builtin");
const windows = builtin.os.tag == .windows;
const c = if (windows) @import("win_c.zig") else std.c;
const Hmac = std.crypto.auth.hmac.sha2.HmacSha256;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Key = [32]u8;
pub const Nonce = [32]u8;
pub const Proof = [Hmac.mac_length]u8;

/// The fewest secret bytes a key file holds.
pub const least_bytes = 16;
/// The most bytes a key file holds.
pub const most_bytes = 4096;

pub const Error = error{ SecretUnreadable, SecretNotAFile, SecretNotPrivate, SecretTooShort, SecretTooLong, NoRandom };

/// The key for the secret in `path`: its SHA-256, trailing whitespace dropped. A regular owner-only file up to 4 KiB.
pub fn fromFile(path: [:0]const u8) Error!Key {
    const fd = open(path);
    if (fd < 0) return error.SecretUnreadable;
    defer _ = c.close(fd);
    if (!windows) { // NTFS keeps access in ACLs, not mode bits
        const m = try mode(fd);
        if (m & 0o170000 != 0o100000) return error.SecretNotAFile;
        if (m & 0o077 != 0) return error.SecretNotPrivate;
    }
    var buf: [most_bytes]u8 = undefined;
    defer std.crypto.secureZero(u8, &buf);
    var n: usize = 0;
    while (n < buf.len) {
        const got = c.read(fd, buf[n..].ptr, buf.len - n);
        if (got < 0) return error.SecretUnreadable;
        if (got == 0) break;
        n += @intCast(got);
    }
    if (n == buf.len) {
        var more: [1]u8 = undefined;
        if (c.read(fd, &more, 1) != 0) return error.SecretTooLong;
    }
    return fromBytes(buf[0..n]);
}

/// Opens the key file to read. A symlink as its last component is refused, and a FIFO is not waited on.
fn open(path: [:0]const u8) c.fd_t {
    if (windows) return c.open(path, .{ .ACCMODE = .RDONLY });
    return c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .NOFOLLOW = true, .NONBLOCK = true });
}

/// `fd`'s type and permission bits: statx on Linux (its libc has no fstat), fstat elsewhere.
fn mode(fd: c.fd_t) Error!u32 {
    if (builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var sx: linux.Statx = undefined;
        // 0x1000 is AT_EMPTY_PATH: the open file itself
        if (c.statx(fd, "", 0x1000, .{ .TYPE = true, .MODE = true }, &sx) != 0) return error.SecretUnreadable;
        return sx.mode;
    }
    var st: c.Stat = undefined;
    if (c.fstat(fd, &st) != 0) return error.SecretUnreadable;
    return @intCast(st.mode);
}

/// The key for secret `bytes` (trailing whitespace dropped).
pub fn fromBytes(bytes: []const u8) Error!Key {
    const s = std.mem.trimEnd(u8, bytes, " \t\r\n");
    if (s.len < least_bytes) return error.SecretTooShort;
    var key: Key = undefined;
    Sha256.hash(s, &key, .{});
    return key;
}

extern "advapi32" fn SystemFunction036(buf: [*]u8, len: u32) callconv(.winapi) u8;

/// 32 bytes from the kernel's random source: /dev/urandom, or RtlGenRandom on Windows.
pub fn nonce() error{NoRandom}!Nonce {
    var n: Nonce = undefined;
    if (windows) return if (SystemFunction036(&n, n.len) == 0) error.NoRandom else n;
    const fd = c.open("/dev/urandom", .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.NoRandom;
    defer _ = c.close(fd);
    var at: usize = 0;
    while (at < n.len) {
        const got = c.read(fd, n[at..].ptr, n.len - at);
        if (got <= 0) return error.NoRandom;
        at += @intCast(got);
    }
    return n;
}

/// HMAC-SHA256 under `key` of `parts` in order. Callers pass parts of fixed length, so no two inputs collide.
pub fn mac(key: *const Key, parts: []const []const u8) Proof {
    var h = Hmac.init(key);
    for (parts) |part| h.update(part);
    var out: Proof = undefined;
    h.final(&out);
    return out;
}

/// Whether `got` is the proof `want`, compared in constant time.
pub fn same(got: *const Proof, want: *const Proof) bool {
    return std.crypto.timing_safe.eql(Proof, got.*, want.*);
}

test "rank key: a mac changes with the key, with any part, and with the order of the parts" {
    const k = try fromBytes("correct horse battery staple\n");
    const other = try fromBytes("correct horse battery staplf");
    const a = try nonce();
    const b = try nonce();
    const p = mac(&k, &.{ "label", &a, &b });
    try std.testing.expect(same(&p, &mac(&k, &.{ "label", &a, &b })));
    try std.testing.expect(!same(&p, &mac(&other, &.{ "label", &a, &b })));
    try std.testing.expect(!same(&p, &mac(&k, &.{ "labem", &a, &b })));
    try std.testing.expect(!same(&p, &mac(&k, &.{ "label", &b, &a })));
    try std.testing.expect(!std.mem.eql(u8, &a, &b));
    try std.testing.expectEqual(try fromBytes("0123456789abcdef"), try fromBytes("0123456789abcdef \r\n"));
}

/// Writes `bytes` to a new file at `path` with `perm`, whatever the umask.
fn plant(path: [:0]const u8, perm: c.mode_t, bytes: []const u8) !void {
    _ = c.unlink(path);
    const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, perm);
    try std.testing.expect(fd >= 0);
    defer _ = c.close(fd);
    _ = c.fchmod(fd, perm);
    try std.testing.expectEqual(@as(isize, @intCast(bytes.len)), c.write(fd, bytes.ptr, bytes.len));
}

test "rank key: a key file is a regular file only its owner can read, never a link, and holds 16 bytes to 4 KiB" {
    if (windows) return error.SkipZigTest; // refusal by mode bits
    try std.testing.expectError(error.SecretTooShort, fromBytes("short secret \n"));
    try std.testing.expectError(error.SecretUnreadable, fromFile("/nonexistent/rank-secret"));
    var path_buf: [64]u8 = undefined;
    var link_buf: [64]u8 = undefined;
    var dir_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/tf-rank-key-{d}", .{c.getpid()}, 0);
    const link = try std.fmt.bufPrintSentinel(&link_buf, "/tmp/tf-rank-key-{d}.link", .{c.getpid()}, 0);
    const dir = try std.fmt.bufPrintSentinel(&dir_buf, "/tmp/tf-rank-key-{d}.d", .{c.getpid()}, 0);
    defer _ = c.unlink(path);
    const contents = "0123456789abcdef0123\n";
    try plant(path, 0o600, contents);
    try std.testing.expectEqual(try fromBytes("0123456789abcdef0123"), try fromFile(path));
    for ([_]c.mode_t{ 0o644, 0o640, 0o604, 0o620 }) |perm| {
        try plant(path, perm, contents);
        try std.testing.expectError(error.SecretNotPrivate, fromFile(path));
    }
    try plant(path, 0o600, contents);
    _ = c.unlink(link);
    try std.testing.expectEqual(@as(c_int, 0), c.symlink(path, link));
    defer _ = c.unlink(link);
    try std.testing.expectError(error.SecretUnreadable, fromFile(link));
    _ = c.rmdir(dir);
    try std.testing.expectEqual(@as(c_int, 0), c.mkdir(dir, 0o700));
    defer _ = c.rmdir(dir);
    try std.testing.expectError(error.SecretNotAFile, fromFile(dir));
    const most: [most_bytes]u8 = @splat('k');
    try plant(path, 0o600, &most);
    try std.testing.expectEqual(try fromBytes(&most), try fromFile(path));
    const big: [most_bytes + 1]u8 = @splat('k');
    try plant(path, 0o600, &big);
    try std.testing.expectError(error.SecretTooLong, fromFile(path));
}
