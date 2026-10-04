//! MCDMA link daemon protocol 1 (docs/link-daemon.md): mailbox offsets, words, half sizes and link names.
const std = @import("std");

pub const protocol: u32 = 1;
/// Each half starts with a 4 KiB control page; payloads follow it.
pub const ctrl: usize = 4096;
/// The Mac's per-MR unit and the mailbox size unit.
pub const segment: usize = 4 << 20;
pub const max_half: usize = 64 * segment;

/// Request half: the request word (both ends).
pub const request_word: usize = 0;
/// Request half, connect end: 1 while the daemon's link to the peer is up.
pub const link_up: usize = 64;
/// Request half, connect end: bumped each time the link comes up.
pub const generation: usize = 72;
/// Request half: the two half sizes as little-endian u64 values.
pub const sizes: usize = 256;
/// Reply half, connect end: the peer's ready word (pull mode).
pub const ready_word: usize = 0;
/// Reply half, connect end: the service's reply has landed.
pub const done_word: usize = 64;
/// Reply half, listen end: a reply is staged for the daemon to send.
pub const staged_word: usize = 128;

pub fn word(seq: u32, len: u32) u64 {
    return @as(u64, seq) << 32 | len;
}

pub fn seqOf(w: u64) u32 {
    return @intCast(w >> 32);
}

pub fn lenOf(w: u64) u32 {
    return @truncate(w);
}

/// A half is a whole number of 4 MiB segments, from 4 to 256 MiB, the same on both ends.
pub fn validHalf(n: u64) bool {
    return n >= segment and n <= max_half and n % segment == 0;
}

/// 1-20 characters of [A-Za-z0-9_-], as mcdma-rpcd's valid_name.
pub fn validName(name: []const u8) bool {
    if (name.len < 1 or name.len > 20) return false;
    for (name) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') return false;
    }
    return true;
}

test "words pack a sequence over a length" {
    const w = word(9, 5);
    try std.testing.expectEqual(@as(u64, (9 << 32) | 5), w);
    try std.testing.expectEqual(@as(u32, 9), seqOf(w));
    try std.testing.expectEqual(@as(u32, 5), lenOf(w));
}

test "halves and names follow mcdma-rpcd" {
    try std.testing.expect(validHalf(4 << 20) and validHalf(256 << 20) and validHalf(64 << 20));
    try std.testing.expect(!validHalf(0) and !validHalf(2 << 20) and !validHalf(260 << 20) and !validHalf((4 << 20) + 1));
    try std.testing.expect(validName("worker-a") and validName("a_1") and validName("xxxxxxxxxxxxxxxxxxxx"));
    try std.testing.expect(!validName("") and !validName("xxxxxxxxxxxxxxxxxxxxx") and !validName("a.b") and !validName("a/b"));
}
