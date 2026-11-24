//! A leader step header: 64 little-endian bytes that every step protocol's message starts with (T2 of #2560).
const std = @import("std");

pub const magic: u32 = 0x4853_4654; // "TFSH" in its little-endian bytes (54 46 53 48)
pub const version: u16 = 1;
pub const size = 64;

/// What a step does: a forward of rows, a state change with no forward, a wait, a clean end, or a family's own step.
pub const Kind = enum(u8) { round = 1, state = 2, idle = 3, stop = 4, family = 5 };

/// A round's rows: a window (each row's state recorded, as decode and a verify run) or a prompt chunk's rows.
pub const Mode = enum(u8) { window = 0, prompt = 1 };

pub const Header = struct {
    kind: Kind,
    /// For `round` only; every other kind carries `window` (0).
    mode: Mode = .window,
    /// The leader's header count: a receiver refuses a header that is not the round it expects.
    round: u64,
    /// The first 4 bytes of the agreed group identity digest.
    group: u32 = 0,
    /// Bit i set: section i follows.
    sections: u32 = 0,
    /// The row map's entries (section 1's, else section 8's), and the sum of their rows.
    streams: u32 = 0,
    rows: u32 = 0,
    /// Bytes after the header: every section.
    bytes: u64 = 0,
    /// The family's own step kind (0: none) and its flag word.
    family: u32 = 0,
    family_flags: u32 = 0,
};

pub const HeaderError = error{ NotStepHeader, BadVersion, BadKind, BadMode, BadReserved };

pub fn encodeHeader(h: Header) [size]u8 {
    var out: [size]u8 = @splat(0);
    std.mem.writeInt(u32, out[0..4], magic, .little);
    std.mem.writeInt(u16, out[4..6], version, .little);
    out[6] = @backingInt(h.kind);
    out[7] = @backingInt(h.mode);
    std.mem.writeInt(u64, out[8..16], h.round, .little);
    std.mem.writeInt(u32, out[16..20], h.group, .little);
    std.mem.writeInt(u32, out[20..24], h.sections, .little);
    std.mem.writeInt(u32, out[24..28], h.streams, .little);
    std.mem.writeInt(u32, out[28..32], h.rows, .little);
    std.mem.writeInt(u64, out[32..40], h.bytes, .little);
    std.mem.writeInt(u32, out[40..44], h.family, .little);
    std.mem.writeInt(u32, out[44..48], h.family_flags, .little);
    return out;
}

/// The header in `raw`, refused unless it is this version with a known kind and mode; `message.decode` checks the rest.
pub fn decodeHeader(raw: *const [size]u8) HeaderError!Header {
    if (std.mem.readInt(u32, raw[0..4], .little) != magic) return error.NotStepHeader;
    if (std.mem.readInt(u16, raw[4..6], .little) != version) return error.BadVersion;
    const kind = std.enums.fromInt(Kind, raw[6]) orelse return error.BadKind;
    const mode = std.enums.fromInt(Mode, raw[7]) orelse return error.BadMode;
    if (kind != .round and mode != .window) return error.BadMode;
    for (raw[48..size]) |b| if (b != 0) return error.BadReserved;
    return .{
        .kind = kind,
        .mode = mode,
        .round = std.mem.readInt(u64, raw[8..16], .little),
        .group = std.mem.readInt(u32, raw[16..20], .little),
        .sections = std.mem.readInt(u32, raw[20..24], .little),
        .streams = std.mem.readInt(u32, raw[24..28], .little),
        .rows = std.mem.readInt(u32, raw[28..32], .little),
        .bytes = std.mem.readInt(u64, raw[32..40], .little),
        .family = std.mem.readInt(u32, raw[40..44], .little),
        .family_flags = std.mem.readInt(u32, raw[44..48], .little),
    };
}

/// The round and group a receiver expects next.
pub const Context = struct { round: u64, group: u32 };

/// `h` refused unless it is the round, then the group, that `ctx` expects.
pub fn expect(h: Header, ctx: Context) error{ OtherRound, OtherGroup }!void {
    if (h.round != ctx.round) return error.OtherRound;
    if (h.group != ctx.group) return error.OtherGroup;
}

const t = std.testing;

test "step header: every field round-trips with TFSH on the wire, in its design offsets" {
    const h: Header = .{
        .kind = .round,
        .mode = .prompt,
        .round = 0x0102_0304_0506_0708,
        .group = 0x0A0B_0C0D,
        .sections = 0b1010,
        .streams = 3,
        .rows = 7,
        .bytes = 0x1122_3344_5566_7788,
        .family = 9,
        .family_flags = 0x8000_0001,
    };
    const raw = encodeHeader(h);
    try t.expectEqualStrings("TFSH", raw[0..4]);
    try t.expectEqual(@as(u8, 1), raw[6]);
    try t.expectEqual(@as(u8, 1), raw[7]);
    try t.expectEqual(@as(u8, 0x08), raw[8]);
    try t.expectEqual(@as(u8, 0x0D), raw[16]);
    try t.expectEqual(@as(u8, 0x88), raw[32]);
    try t.expectEqual(@as(u8, 0x01), raw[44]);
    try t.expectEqualSlices(u8, &@as([16]u8, @splat(0)), raw[48..64]);
    try t.expectEqual(h, try decodeHeader(&raw));
}

test "step header: a wrong magic, version, kind, mode or reserved byte is refused by name" {
    const raw = encodeHeader(.{ .kind = .state, .round = 3 });
    var bad = raw;
    bad[0] ^= 1;
    try t.expectError(error.NotStepHeader, decodeHeader(&bad));
    bad = raw;
    bad[4] = 2;
    try t.expectError(error.BadVersion, decodeHeader(&bad));
    bad = raw;
    bad[6] = 6;
    try t.expectError(error.BadKind, decodeHeader(&bad));
    bad = raw;
    bad[7] = 1; // prompt on a state header
    try t.expectError(error.BadMode, decodeHeader(&bad));
    bad[6] = 1; // the same mode on a round is taken
    _ = try decodeHeader(&bad);
    bad[7] = 2;
    try t.expectError(error.BadMode, decodeHeader(&bad));
    for (48..size) |i| {
        bad = raw;
        bad[i] = 1;
        try t.expectError(error.BadReserved, decodeHeader(&bad));
    }
}

test "step header: a receiver refuses another round before another group" {
    const h: Header = .{ .kind = .round, .round = 4, .group = 7 };
    try expect(h, .{ .round = 4, .group = 7 });
    try t.expectError(error.OtherRound, expect(h, .{ .round = 5, .group = 8 }));
    try t.expectError(error.OtherGroup, expect(h, .{ .round = 4, .group = 8 }));
}
