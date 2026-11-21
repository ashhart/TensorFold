//! KV handoff headers, protocol 1 (MCDMA's docs/kv-handoff.md) and 2 (#1594 step 2b): 128 little-endian bytes.
const std = @import("std");

pub const magic = "MKVH";
pub const version: u16 = 1;
/// Protocol 2: an export that holds recurrent state; a protocol 1 reader refuses its headers as foreign.
pub const recurrent_version: u16 = 2;
pub const header_bytes = 128;
/// Header flag: `crc` holds the zlib CRC-32 of the payload.
pub const checked: u32 = 1;

/// Requests the decoder sends (open, pull, close) and the producer's answers.
pub const Kind = enum(u16) { open = 1, manifest = 2, wait = 3, @"error" = 4, pull = 5, data = 6, close = 7, ack = 8 };

pub const Header = struct {
    version: u16 = version,
    kind: Kind,
    handoff: [16]u8,
    frame: u32 = 0,
    frames: u32 = 0,
    layer: u32 = 0,
    flags: u32 = 0,
    row_start: u64 = 0,
    rows: u64 = 0,
    nbytes: u64 = 0,
    crc: u32 = 0,
};

/// Why a message was refused; the text is what MCDMA's own wire.py says.
pub const Malformed = enum {
    short,
    foreign,
    kind,
    truncated,

    pub fn text(m: Malformed, kind: u16, buf: []u8) []const u8 {
        return switch (m) {
            .short => "message is shorter than a handoff header",
            .foreign => "message is not a KV handoff protocol 1 header",
            .kind => std.fmt.bufPrint(buf, "unknown handoff message kind {d}", .{kind}) catch "unknown handoff message kind",
            .truncated => "handoff payload is shorter than its header says",
        };
    }
};

pub fn pack(h: Header) [header_bytes]u8 {
    var out = std.mem.zeroes([header_bytes]u8);
    @memcpy(out[0..4], magic);
    std.mem.writeInt(u16, out[4..6], h.version, .little);
    std.mem.writeInt(u16, out[6..8], @backingInt(h.kind), .little);
    @memcpy(out[8..24], &h.handoff);
    std.mem.writeInt(u32, out[24..28], h.frame, .little);
    std.mem.writeInt(u32, out[28..32], h.frames, .little);
    std.mem.writeInt(u32, out[32..36], h.layer, .little);
    std.mem.writeInt(u32, out[36..40], h.flags, .little);
    std.mem.writeInt(u64, out[40..48], h.row_start, .little);
    std.mem.writeInt(u64, out[48..56], h.rows, .little);
    std.mem.writeInt(u64, out[56..64], h.nbytes, .little);
    std.mem.writeInt(u32, out[64..68], h.crc, .little);
    return out;
}

pub const Unpacked = union(enum) { ok: Header, bad: struct { why: Malformed, kind: u16 = 0 } };

/// Parse the header at the start of `msg` with wire.py's checks, in its order: a protocol 1 reader.
pub fn unpack(msg: []const u8) Unpacked {
    return unpackUpTo(msg, version);
}

/// As unpack, accepting protocols 1 to `most`; a later protocol's header is foreign, as protocol 1 readers find it.
pub fn unpackUpTo(msg: []const u8, most: u16) Unpacked {
    if (msg.len < header_bytes) return .{ .bad = .{ .why = .short } };
    const v = std.mem.readInt(u16, msg[4..6], .little);
    if (!std.mem.eql(u8, msg[0..4], magic) or v < version or v > most) return .{ .bad = .{ .why = .foreign } };
    const raw = std.mem.readInt(u16, msg[6..8], .little);
    const kind = std.enums.fromInt(Kind, raw) orelse return .{ .bad = .{ .why = .kind, .kind = raw } };
    const h: Header = .{
        .version = v,
        .kind = kind,
        .handoff = msg[8..24].*,
        .frame = std.mem.readInt(u32, msg[24..28], .little),
        .frames = std.mem.readInt(u32, msg[28..32], .little),
        .layer = std.mem.readInt(u32, msg[32..36], .little),
        .flags = std.mem.readInt(u32, msg[36..40], .little),
        .row_start = std.mem.readInt(u64, msg[40..48], .little),
        .rows = std.mem.readInt(u64, msg[48..56], .little),
        .nbytes = std.mem.readInt(u64, msg[56..64], .little),
        .crc = std.mem.readInt(u32, msg[64..68], .little),
    };
    const carries = switch (kind) {
        .open, .manifest, .@"error", .data => true,
        else => false,
    };
    if (carries and msg.len - header_bytes < h.nbytes) return .{ .bad = .{ .why = .truncated } };
    return .{ .ok = h };
}

/// The payload after `h`; only kinds that carry one are length-checked by unpack.
pub fn body(msg: []const u8, h: Header) []const u8 {
    const end = header_bytes +| @min(h.nbytes, msg.len - header_bytes);
    return msg[header_bytes..end];
}

/// zlib's CRC-32, as the producer computes over each DATA payload.
pub fn crc32(bytes: []const u8) u32 {
    return @import("crc32.zig").hash(bytes);
}

/// SHA-256 of the prompt's token IDs as little-endian uint32 values, in lowercase hex.
pub fn tokenSha256(tokens: []const u32) [64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (tokens) |t| {
        var le: [4]u8 = undefined;
        std.mem.writeInt(u32, &le, t, .little);
        h.update(&le);
    }
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

test "headers round trip and lay out as the table says" {
    const id: [16]u8 = .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const h: Header = .{ .kind = .data, .handoff = id, .frame = 3, .frames = 9, .layer = 2, .flags = checked, .row_start = 16, .rows = 4, .nbytes = 8, .crc = 77 };
    const b = pack(h);
    try std.testing.expectEqualStrings("MKVH", b[0..4]);
    try std.testing.expectEqual(@as(u16, 6), std.mem.readInt(u16, b[6..8], .little));
    try std.testing.expectEqual(@as(u32, 77), std.mem.readInt(u32, b[64..68], .little));
    var msg: [header_bytes + 8]u8 = undefined;
    @memcpy(msg[0..header_bytes], &b);
    try std.testing.expectEqual(h, unpack(&msg).ok);
}

test "malformed headers are refused with wire.py's reasons" {
    var buf: [64]u8 = undefined;
    try std.testing.expectEqual(Malformed.short, unpack(&@as([10]u8, @splat(0))).bad.why);
    try std.testing.expectEqual(Malformed.foreign, unpack("XXXX" ++ @as([124]u8, @splat(0))).bad.why);
    const short_data = pack(.{ .kind = .data, .handoff = @splat(0), .nbytes = 8 });
    try std.testing.expectEqual(Malformed.truncated, unpack(&short_data).bad.why);
    var odd = pack(.{ .kind = .ack, .handoff = @splat(0) });
    std.mem.writeInt(u16, odd[6..8], 42, .little);
    const bad = unpack(&odd).bad;
    try std.testing.expectEqualStrings("unknown handoff message kind 42", bad.why.text(bad.kind, &buf));
    const pull = pack(.{ .kind = .pull, .handoff = @splat(0), .nbytes = 99 });
    try std.testing.expectEqual(Kind.pull, unpack(&pull).ok.kind);
}

test "a protocol 2 header is foreign to a protocol 1 reader and read by a protocol 2 one" {
    const h: Header = .{ .version = recurrent_version, .kind = .manifest, .handoff = @splat(7) };
    const b = pack(h);
    try std.testing.expectEqual(@as(u16, 2), std.mem.readInt(u16, b[4..6], .little));
    try std.testing.expectEqual(Malformed.foreign, unpack(&b).bad.why);
    try std.testing.expectEqual(h, unpackUpTo(&b, recurrent_version).ok);
    var v3 = b;
    std.mem.writeInt(u16, v3[4..6], 3, .little);
    try std.testing.expectEqual(Malformed.foreign, unpackUpTo(&v3, recurrent_version).bad.why);
}

test "the prompt digest is over little-endian uint32 ids" {
    var expect: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&[_]u8{ 1, 0, 0, 0, 0x70, 0x11, 1, 0 }, &expect, .{});
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(expect, .lower), &tokenSha256(&.{ 1, 70000 }));
    try std.testing.expectEqual(@as(u32, 0x3610a686), crc32("hello"));
}
