//! The ring's shared secret: a key from an owner-only file, and each link's mutual HMAC proof over fresh nonces.
const std = @import("std");
const c = std.c;
const builtin = @import("builtin");
const Hmac = std.crypto.auth.hmac.sha2.HmacSha256;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Key = [32]u8;
pub const Nonce = [32]u8;
pub const Proof = [Hmac.mac_length]u8;

/// The fewest secret bytes a key file holds.
pub const least_bytes = 16;

pub const Error = error{ SecretUnreadable, SecretNotPrivate, SecretTooShort, SecretTooLong, NoRandom };

/// Which end of a link proves: the rank that connected, or the one that accepted.
pub const Role = enum(u8) { connector = 'C', acceptor = 'A' };

/// The key for the secret in `path`: its SHA-256, trailing whitespace dropped; owner-only and at most 4 KiB.
pub fn fromFile(path: [:0]const u8) Error!Key {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.SecretUnreadable;
    defer _ = c.close(fd);
    if (try mode(fd) & 0o077 != 0) return error.SecretNotPrivate;
    var buf: [4096]u8 = undefined;
    var n: usize = 0;
    while (n < buf.len) {
        const got = c.read(fd, buf[n..].ptr, buf.len - n);
        if (got < 0) return error.SecretUnreadable;
        if (got == 0) break;
        n += @intCast(got);
    }
    defer std.crypto.secureZero(u8, &buf);
    if (n == buf.len) {
        var more: [1]u8 = undefined;
        if (c.read(fd, &more, 1) != 0) return error.SecretTooLong;
    }
    return fromBytes(buf[0..n]);
}

/// `fd`'s permission bits: statx on Linux (its libc has no fstat), fstat elsewhere.
fn mode(fd: c.fd_t) Error!u32 {
    if (builtin.os.tag == .linux) {
        const linux = std.os.linux;
        var sx: linux.Statx = undefined;
        if (c.statx(fd, "", 0x1000, .{ .MODE = true }, &sx) != 0) return error.SecretUnreadable; // AT_EMPTY_PATH
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

/// 32 bytes from the kernel's random source (/dev/urandom on Linux and macOS alike).
pub fn nonce() Error!Nonce {
    var n: Nonce = undefined;
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

/// The proof the `role` end of a link (rank `rank` of `world`) sends: its peer's nonce, then its own.
pub fn proof(key: *const Key, role: Role, rank: u32, world: u32, fingerprint: *const [32]u8, theirs: *const Nonce, mine: *const Nonce) Proof {
    var h = Hmac.init(key);
    h.update("TFST link v1");
    h.update(&.{@backingInt(role)});
    var ints: [8]u8 = undefined;
    std.mem.writeInt(u32, ints[0..4], rank, .little);
    std.mem.writeInt(u32, ints[4..8], world, .little);
    h.update(&ints);
    h.update(fingerprint);
    h.update(theirs);
    h.update(mine);
    var out: Proof = undefined;
    h.final(&out);
    return out;
}

/// Whether `got` is the proof `want`, compared in constant time.
pub fn same(got: *const Proof, want: *const Proof) bool {
    return std.crypto.timing_safe.eql(Proof, got.*, want.*);
}

test "stage secret: proofs bind the key, the role, the rank, the fingerprint and both nonces" {
    const k = try fromBytes("correct horse battery staple\n");
    const other = try fromBytes("correct horse battery staplf");
    const fp: [32]u8 = @splat(7);
    const a = try nonce();
    const b = try nonce();
    const p = proof(&k, .connector, 1, 3, &fp, &a, &b);
    try std.testing.expect(same(&p, &proof(&k, .connector, 1, 3, &fp, &a, &b)));
    try std.testing.expect(!same(&p, &proof(&other, .connector, 1, 3, &fp, &a, &b)));
    try std.testing.expect(!same(&p, &proof(&k, .acceptor, 1, 3, &fp, &a, &b)));
    try std.testing.expect(!same(&p, &proof(&k, .connector, 2, 3, &fp, &a, &b)));
    try std.testing.expect(!same(&p, &proof(&k, .connector, 1, 3, &fp, &b, &a)));
    var fp2 = fp;
    fp2[0] ^= 1;
    try std.testing.expect(!same(&p, &proof(&k, .connector, 1, 3, &fp2, &a, &b)));
    try std.testing.expect(!std.mem.eql(u8, &a, &b));
}

test "stage secret: a key file must be owner-only and hold 16 bytes to 4 KiB" {
    try std.testing.expectError(error.SecretTooShort, fromBytes("short secret \n"));
    try std.testing.expectError(error.SecretUnreadable, fromFile("/nonexistent/stage-secret"));
    var dir_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&dir_buf, "/tmp/tf-stage-secret-{d}", .{c.getpid()}, 0);
    defer _ = c.unlink(path);
    for ([_]struct { mode: c.mode_t, ok: bool }{ .{ .mode = 0o600, .ok = true }, .{ .mode = 0o644, .ok = false }, .{ .mode = 0o640, .ok = false } }) |case| {
        _ = c.unlink(path);
        const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, case.mode);
        try std.testing.expect(fd >= 0);
        _ = c.fchmod(fd, case.mode); // the umask may have narrowed it
        _ = c.write(fd, "0123456789abcdef0123\n", 21);
        _ = c.close(fd);
        if (case.ok) {
            try std.testing.expectEqual(try fromBytes("0123456789abcdef0123"), try fromFile(path));
        } else try std.testing.expectError(error.SecretNotPrivate, fromFile(path));
    }
    _ = c.unlink(path);
    const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, @as(c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    const big: [4097]u8 = @splat('k');
    _ = c.write(fd, &big, big.len);
    _ = c.close(fd);
    try std.testing.expectError(error.SecretTooLong, fromFile(path));
}
