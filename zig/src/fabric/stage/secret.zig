//! The ring's shared secret: the rank key file (sys.rank_auth.key), and each link's mutual HMAC proof over nonces.
const std = @import("std");
const rank_auth = @import("sys").rank_auth;
const key = rank_auth.key;

pub const Key = key.Key;
pub const Nonce = key.Nonce;
pub const Proof = key.Proof;

/// The fewest secret bytes a stage key file holds: the ring's own 16, so a ring that starts today keeps starting.
pub const least_bytes = 16;

pub const Error = key.Error;

/// Which end of a link proves: the rank that connected, or the one that accepted.
pub const Role = enum(u8) { connector = 'C', acceptor = 'A' };

/// The key for the secret in a regular owner-only file of 16 B to 4 KiB, links followed as for a rank key.
pub fn fromFile(path: [:0]const u8) Error!Key {
    return key.fromFile(path, least_bytes);
}

/// The key for secret bytes (trailing whitespace dropped), at least 16 of them.
pub fn fromBytes(bytes: []const u8) Error!Key {
    return key.fromBytes(bytes, least_bytes);
}

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

test "stage secret: a 16-byte stage key still loads, through a link too, where a rank key needs 32" {
    const c = std.c;
    var path_buf: [64]u8 = undefined;
    var link_buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "/tmp/tf-stage-least-{d}", .{c.getpid()}, 0);
    const link = try std.fmt.bufPrintSentinel(&link_buf, "/tmp/tf-stage-least-{d}.link", .{c.getpid()}, 0);
    defer _ = c.unlink(path);
    const text: [32]u8 = @splat('k');
    for ([_]struct { len: usize, stage: bool, rank: bool }{
        .{ .len = 15, .stage = false, .rank = false },
        .{ .len = 16, .stage = true, .rank = false },
        .{ .len = 31, .stage = true, .rank = false },
        .{ .len = 32, .stage = true, .rank = true },
    }) |case| {
        _ = c.unlink(path);
        const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, @as(c.mode_t, 0o600));
        try std.testing.expect(fd >= 0);
        _ = c.fchmod(fd, 0o600); // the umask may have narrowed it
        try std.testing.expectEqual(@as(isize, @intCast(case.len)), c.write(fd, &text, case.len));
        _ = c.close(fd);
        if (case.stage) {
            try std.testing.expectEqual(try fromBytes(text[0..case.len]), try fromFile(path));
        } else try std.testing.expectError(error.SecretTooShort, fromFile(path));
        if (case.rank) {
            try std.testing.expectEqual(try fromBytes(text[0..case.len]), (try rank_auth.Secret.fromFile(path)).key);
        } else try std.testing.expectError(error.SecretTooShort, rank_auth.Secret.fromFile(path));
    }
    // the 16-byte key through a symbolic link, as a secret mount gives it: the same key, with the same mode check
    _ = c.unlink(path);
    const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, @as(c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    _ = c.fchmod(fd, 0o600);
    _ = c.write(fd, &text, 16);
    _ = c.close(fd);
    _ = c.unlink(link);
    try std.testing.expectEqual(@as(c_int, 0), c.symlink(path, link));
    defer _ = c.unlink(link);
    try std.testing.expectEqual(try fromFile(path), try fromFile(link));
    _ = c.chmod(path, 0o644);
    try std.testing.expectError(error.SecretNotPrivate, fromFile(link));
}
