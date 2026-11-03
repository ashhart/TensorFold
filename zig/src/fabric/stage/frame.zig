//! A stage link's frame header: 32 little-endian bytes (magic, version, kind, round, rows, payload bytes).
const std = @import("std");

pub const magic: u32 = 0x5453_4654; // "TFST" in its little-endian bytes (54 46 53 54)
pub const version: u16 = 1;
pub const size = 32;

/// What a frame carries: the link's hello, a round's plan, its hidden rows, or a clean stop.
pub const Kind = enum(u8) { hello = 1, plan = 2, rows = 3, stop = 4 };

pub const Header = struct {
    kind: Kind,
    /// The round this frame belongs to; each link's rounds run 1, 2, 3, ... in order.
    round: u64,
    /// The rows a `rows` frame carries (0 for the others).
    rows: u32,
    /// Payload bytes after the header.
    bytes: u64,
};

pub const Error = error{ BadMagic, BadVersion, BadKind, TooLarge };

pub fn encode(h: Header) [size]u8 {
    var out: [size]u8 = @splat(0);
    std.mem.writeInt(u32, out[0..4], magic, .little);
    std.mem.writeInt(u16, out[4..6], version, .little);
    out[6] = @intFromEnum(h.kind);
    std.mem.writeInt(u64, out[8..16], h.round, .little);
    std.mem.writeInt(u32, out[16..20], h.rows, .little);
    std.mem.writeInt(u64, out[24..32], h.bytes, .little);
    return out;
}

/// A header from `raw`, refused unless it is this link's magic and version, a known kind, and at most `cap` bytes.
pub fn decode(raw: *const [size]u8, cap: u64) Error!Header {
    if (std.mem.readInt(u32, raw[0..4], .little) != magic) return error.BadMagic;
    if (std.mem.readInt(u16, raw[4..6], .little) != version) return error.BadVersion;
    const kind = std.enums.fromInt(Kind, raw[6]) orelse return error.BadKind;
    const bytes = std.mem.readInt(u64, raw[24..32], .little);
    if (bytes > cap) return error.TooLarge;
    return .{ .kind = kind, .round = std.mem.readInt(u64, raw[8..16], .little), .rows = std.mem.readInt(u32, raw[16..20], .little), .bytes = bytes };
}

test "stage frame: a header round-trips with TFST on the wire, and a wrong magic, version, kind or size is refused" {
    const h: Header = .{ .kind = .rows, .round = 0x0102_0304_0506_0708, .rows = 4096, .bytes = 80 << 20 };
    const raw = encode(h);
    try std.testing.expectEqualStrings("TFST", raw[0..4]);
    try std.testing.expectEqual(h, try decode(&raw, 80 << 20));
    try std.testing.expectError(error.TooLarge, decode(&raw, (80 << 20) - 1));
    var bad = raw;
    bad[0] ^= 1;
    try std.testing.expectError(error.BadMagic, decode(&bad, 1 << 30));
    bad = raw;
    bad[4] = 2;
    try std.testing.expectError(error.BadVersion, decode(&bad, 1 << 30));
    bad = raw;
    bad[6] = 9;
    try std.testing.expectError(error.BadKind, decode(&bad, 1 << 30));
}
