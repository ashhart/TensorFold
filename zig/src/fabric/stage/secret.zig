//! The ring's shared secret: the rank key file (sys.rank_auth.key), and each link's mutual HMAC proof over nonces.
const std = @import("std");
const key = @import("sys").rank_auth.key;

pub const Key = key.Key;
pub const Nonce = key.Nonce;
pub const Proof = key.Proof;

/// The fewest secret bytes a key file holds.
pub const least_bytes = key.least_bytes;

pub const Error = key.Error;

/// Which end of a link proves: the rank that connected, or the one that accepted.
pub const Role = enum(u8) { connector = 'C', acceptor = 'A' };

/// The key for the secret in a regular owner-only file of at most 4 KiB: its SHA-256, trailing whitespace dropped.
pub const fromFile = key.fromFile;
/// The key for secret bytes (trailing whitespace dropped).
pub const fromBytes = key.fromBytes;
/// 32 bytes from the kernel's random source.
pub const nonce = key.nonce;
/// Whether one proof is another, compared in constant time.
pub const same = key.same;

/// The proof the `role` end of a link (rank `rank` of `world`) sends: its peer's nonce, then its own.
pub fn proof(k: *const Key, role: Role, rank: u32, world: u32, fingerprint: *const [32]u8, theirs: *const Nonce, mine: *const Nonce) Proof {
    var ints: [8]u8 = undefined;
    std.mem.writeInt(u32, ints[0..4], rank, .little);
    std.mem.writeInt(u32, ints[4..8], world, .little);
    const parts = [_][]const u8{ "TFST link v1", &.{@backingInt(role)}, &ints, fingerprint, theirs, mine };
    return key.mac(k, &parts);
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

test "stage secret: a proof's bytes are the ones the ring has always sent" {
    // HMAC-SHA256 of the label, the role, rank 1 of 3, the fingerprint and both nonces, written out by hand
    const k = try fromBytes("0123456789abcdef");
    const fp: [32]u8 = @splat(7);
    const theirs: Nonce = @splat(1);
    const mine: Nonce = @splat(2);
    var h = std.crypto.auth.hmac.sha2.HmacSha256.init(&k);
    h.update("TFST link v1");
    h.update("C");
    h.update(&.{ 1, 0, 0, 0, 3, 0, 0, 0 });
    h.update(&fp);
    h.update(&theirs);
    h.update(&mine);
    var want: Proof = undefined;
    h.final(&want);
    try std.testing.expectEqualSlices(u8, &want, &proof(&k, .connector, 1, 3, &fp, &theirs, &mine));
}

test "stage secret: a key file must be owner-only and hold 16 bytes to 4 KiB" {
    const c = std.c;
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
