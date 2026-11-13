//! Sealed records on a proven rank link: after the key proof, every byte crosses in ChaCha20-Poly1305 records.
const std = @import("std");
const key = @import("rank_key.zig");
const Aead = std.crypto.aead.chacha_poly.ChaCha20Poly1305;

/// The most bytes one record carries; a longer write is cut into several.
pub const max_plain = 16 * 1024;
pub const head_len = 4;
pub const tag_len = Aead.tag_length;
/// The most a record takes on the wire: its length, its sealed bytes, then its tag.
pub const max_record = head_len + max_plain + tag_len;

pub const Error = error{RankSealBroken};

/// One end's state for one connection. Its directions share nothing, so one thread may write while another reads.
pub const Seal = struct {
    out_key: [Aead.key_length]u8,
    in_key: [Aead.key_length]u8,
    sent: u64 = 0,
    got: u64 = 0,
    plain: [max_plain]u8 = undefined,
    at: usize = 0,
    end: usize = 0,

    /// The two directions' keys from the proof's session key; `connector` is the end that sent the hello.
    pub fn init(session: *const key.Key, connector: bool) Seal {
        const up = key.mac(session, &.{"TFRK seal v1, connector to acceptor"});
        const down = key.mac(session, &.{"TFRK seal v1, acceptor to connector"});
        return .{ .out_key = if (connector) up else down, .in_key = if (connector) down else up };
    }

    /// A record's place in its direction: each is sealed once, so one replayed, dropped or moved does not open.
    fn nonce(count: u64) [Aead.nonce_length]u8 {
        var n: [Aead.nonce_length]u8 = @splat(0);
        std.mem.writeInt(u64, n[4..12], count, .little);
        return n;
    }

    /// One record holding `bytes` (1 to max_plain of them) into `wire`; returns the record's length.
    pub fn seal(s: *Seal, bytes: []const u8, wire: *[max_record]u8) usize {
        std.debug.assert(bytes.len > 0 and bytes.len <= max_plain);
        std.mem.writeInt(u32, wire[0..head_len], @intCast(bytes.len), .little);
        var tag: [tag_len]u8 = undefined;
        Aead.encrypt(wire[head_len..][0..bytes.len], &tag, bytes, wire[0..head_len], nonce(s.sent), s.out_key);
        wire[head_len + bytes.len ..][0..tag_len].* = tag;
        s.sent += 1;
        return head_len + bytes.len + tag_len;
    }

    /// The bytes that follow a record's head on the wire: its sealed bytes and its tag.
    pub fn bodyLen(head: *const [head_len]u8) Error!usize {
        const n = std.mem.readInt(u32, head, .little);
        if (n == 0 or n > max_plain) return error.RankSealBroken;
        return n + tag_len;
    }

    /// Opens the record `head` and `body` make; its bytes then wait for `take`. Nothing of a record that fails is kept.
    pub fn open(s: *Seal, head: *const [head_len]u8, body: []const u8) Error!void {
        const n = try bodyLen(head) - tag_len;
        if (body.len != n + tag_len or s.at != s.end) return error.RankSealBroken;
        s.at = 0;
        s.end = 0;
        Aead.decrypt(s.plain[0..n], body[0..n], body[n..][0..tag_len].*, head, nonce(s.got), s.in_key) catch return error.RankSealBroken;
        s.got += 1;
        s.end = n;
    }

    /// Moves waiting bytes into `out`; returns how many.
    pub fn take(s: *Seal, out: []u8) usize {
        const n = @min(out.len, s.end - s.at);
        @memcpy(out[0..n], s.plain[s.at..][0..n]);
        s.at += n;
        return n;
    }

    /// Opened bytes no reader has taken yet: a socket that shows nothing to read may still have these.
    pub fn waiting(s: *const Seal) usize {
        return s.end - s.at;
    }
};

/// A record arriving in pieces, for a reader with deadlines: it fills `want` from its socket, then calls `fed`.
pub const Arriving = struct {
    wire: [max_record]u8 = undefined,
    have: usize = 0,

    fn end(a: *const Arriving) Error!usize {
        return if (a.have < head_len) head_len else head_len + try Seal.bodyLen(a.wire[0..head_len]);
    }

    /// Where the record's next bytes go: the rest of its head, then of its body. A head no record has is refused here.
    pub fn want(a: *Arriving) Error![]u8 {
        return a.wire[a.have..try a.end()];
    }

    /// `n` more bytes are in `want`. A record that is whole is opened into `seal`, whose `take` then has its bytes.
    pub fn fed(a: *Arriving, seal: *Seal, n: usize) Error!void {
        a.have += n;
        const whole = try a.end();
        if (a.have < head_len or a.have < whole) return;
        a.have = 0;
        try seal.open(a.wire[0..head_len], a.wire[head_len..whole]);
    }
};

const Pair = struct {
    connector: Seal,
    acceptor: Seal,

    fn of(secret: []const u8) Pair {
        var session: key.Key = undefined;
        std.crypto.hash.sha2.Sha256.hash(secret, &session, .{});
        return .{ .connector = .init(&session, true), .acceptor = .init(&session, false) };
    }
};

/// What `to` reads when `from` writes `bytes`, cut into records as a link cuts them.
fn carry(from: *Seal, to: *Seal, bytes: []const u8, out: []u8) !void {
    var wire: [max_record]u8 = undefined;
    var at: usize = 0;
    var read: usize = 0;
    while (at < bytes.len) {
        const n = @min(bytes.len - at, max_plain);
        const len = from.seal(bytes[at..][0..n], &wire);
        try to.open(wire[0..head_len], wire[head_len..len]);
        // a reader takes what it asked for, however the records fall: here seven bytes at a time, then the rest
        read += to.take(out[read..][0..@min(7, out.len - read)]);
        read += to.take(out[read..]);
        at += n;
    }
    try std.testing.expectEqual(bytes.len, read);
}

test "rank seal: bytes sealed at one end open at the other, across records and in both directions" {
    var p = Pair.of("the group's key");
    var bytes: [3 * max_plain + 100]u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = @truncate(i * 7 + 3);
    var out: [bytes.len]u8 = undefined;
    try carry(&p.connector, &p.acceptor, &bytes, &out);
    try std.testing.expectEqualSlices(u8, &bytes, &out);
    try carry(&p.acceptor, &p.connector, bytes[0..5], out[0..5]);
    try std.testing.expectEqualSlices(u8, bytes[0..5], out[0..5]);
    try carry(&p.connector, &p.acceptor, bytes[0..1], out[0..1]);
    try std.testing.expectEqual(@as(u64, 5), p.connector.sent);
    try std.testing.expectEqual(@as(u64, 5), p.acceptor.got);
}

test "security: a record altered, replayed, moved, sent back or sealed under another key does not open" {
    const words = "prompt token ids: 17 4021 9 88213";
    var wire: [max_record]u8 = undefined;
    var later: [max_record]u8 = undefined;
    // every bit of the head, the sealed bytes and the tag is covered
    for (0..head_len + words.len + tag_len) |i| {
        var p = Pair.of("the group's key");
        const len = p.connector.seal(words, &wire);
        wire[i] ^= 1;
        try std.testing.expectError(error.RankSealBroken, p.acceptor.open(wire[0..head_len], wire[head_len..len]));
        try std.testing.expectEqual(@as(usize, 0), p.acceptor.take(later[0..8])); // nothing of it reaches a reader
    }
    var p = Pair.of("the group's key");
    const first = p.connector.seal(words, &wire);
    const second = p.connector.seal("stop", &later);
    // out of order, then in order, then again
    try std.testing.expectError(error.RankSealBroken, p.acceptor.open(later[0..head_len], later[head_len..second]));
    try p.acceptor.open(wire[0..head_len], wire[head_len..first]);
    var got: [words.len]u8 = undefined;
    try std.testing.expectEqual(words.len, p.acceptor.take(&got));
    try std.testing.expectError(error.RankSealBroken, p.acceptor.open(wire[0..head_len], wire[head_len..first]));
    // a record sent back to the end that sealed it, and one from a group with another key
    try std.testing.expectError(error.RankSealBroken, p.connector.open(later[0..head_len], later[head_len..second]));
    var other = Pair.of("another group's key");
    try std.testing.expectError(error.RankSealBroken, other.acceptor.open(wire[0..head_len], wire[head_len..first]));
    // a head that claims no bytes, or more than a record holds, is refused before anything is read for it
    try std.testing.expectError(error.RankSealBroken, Seal.bodyLen(&.{ 0, 0, 0, 0 }));
    try std.testing.expectError(error.RankSealBroken, Seal.bodyLen(&.{ 1, 64, 0, 0 }));
}

test "security: the wire shows none of the bytes a record carries" {
    var p = Pair.of("the group's key");
    const words = "prompt token ids: 17 4021 9 88213";
    var wire: [max_record]u8 = undefined;
    const len = p.connector.seal(words, &wire);
    try std.testing.expectEqual(head_len + words.len + tag_len, len);
    try std.testing.expect(std.mem.indexOf(u8, wire[0..len], "prompt") == null);
    try std.testing.expect(std.mem.indexOf(u8, wire[0..len], "4021") == null);
}

test "rank seal: a record that arrives a few bytes at a time opens once it is whole, and not before" {
    var p = Pair.of("the group's key");
    const words = "prompt token ids: 17 4021 9 88213";
    var first: [max_record]u8 = undefined;
    var second: [max_record]u8 = undefined;
    const a = p.connector.seal(words, &first);
    const b = p.connector.seal("stop", &second);
    var stream: [2 * max_record]u8 = undefined;
    @memcpy(stream[0..a], first[0..a]);
    @memcpy(stream[a..][0..b], second[0..b]);
    var arriving: Arriving = .{};
    var got: [words.len + 4]u8 = undefined;
    var read: usize = 0;
    var at: usize = 0;
    while (at < a + b) {
        // a socket gives what it has: here three bytes at most, never past the record it is in
        const want = try arriving.want();
        const n = @min(3, want.len);
        @memcpy(want[0..n], stream[at..][0..n]);
        at += n;
        try arriving.fed(&p.acceptor, n);
        // nothing of a record can be read until its tag has been checked
        if (at < a) try std.testing.expectEqual(@as(usize, 0), p.acceptor.waiting());
        read += p.acceptor.take(got[read..]);
    }
    try std.testing.expectEqualStrings(words ++ "stop", &got);
}

test "security: a record altered on its way, or a head no record has, is refused piece by piece too" {
    var p = Pair.of("the group's key");
    var wire: [max_record]u8 = undefined;
    const len = p.connector.seal("stop", &wire);
    wire[head_len] ^= 1;
    var arriving: Arriving = .{};
    @memcpy((try arriving.want())[0..head_len], wire[0..head_len]);
    try arriving.fed(&p.acceptor, head_len);
    @memcpy(try arriving.want(), wire[head_len..len]);
    try std.testing.expectError(error.RankSealBroken, arriving.fed(&p.acceptor, len - head_len));
    try std.testing.expectEqual(@as(usize, 0), p.acceptor.waiting());
    // a head that claims more than a record holds: refused before a byte is read for it, and from then on
    var long: Arriving = .{};
    @memcpy((try long.want())[0..head_len], &[_]u8{ 1, 64, 0, 0 });
    try std.testing.expectError(error.RankSealBroken, long.fed(&p.acceptor, head_len));
    try std.testing.expectError(error.RankSealBroken, long.want());
}
