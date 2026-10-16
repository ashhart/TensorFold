//! A picture's orientation tag as Pillow's getexif read it: EXIF IFD0's Orientation (0x0112), else XMP's.

const std = @import("std");

const orientation_tag = 0x0112;

/// What an unreadable EXIF block did in 0.6.6: refused the picture, or (read first by a JPEG's open) nothing.
pub const Failure = enum { refuse, ignore };

/// The orientation 0.6.6 transposed by (only 2-8 turn the picture); error.Invalid where Pillow raised on the block.
pub fn orientation(exif: ?[]const u8, xmp: ?[]const u8, failure: Failure) error{Invalid}!u32 {
    if (exif) |block| {
        const found = fromExif(block) catch |err| return if (failure == .ignore) 1 else err;
        if (found) |o| return o;
    }
    return if (xmp) |x| fromXmp(x) orelse 1 else 1;
}

/// IFD0's Orientation, its first value as a number (0 when Pillow's value matched no transpose); null when absent.
fn fromExif(block: []const u8) error{Invalid}!?u32 {
    var tiff = block;
    while (std.mem.startsWith(u8, tiff, "Exif\x00\x00")) tiff = tiff[6..];
    if (tiff.len == 0) return null;
    if (tiff.len < 8) return error.Invalid;
    const endian: std.builtin.Endian = if (std.mem.eql(u8, tiff[0..4], "MM\x00\x2a") or std.mem.eql(u8, tiff[0..4], "MM\x2a\x00"))
        .big
    else if (std.mem.eql(u8, tiff[0..4], "II\x2a\x00") or std.mem.eql(u8, tiff[0..4], "II\x00\x2a"))
        .little
    else
        return error.Invalid; // a BigTIFF header too: 0.6.6 failed reading its 8-byte offset
    var found: ?Entry = null;
    const ifd = std.mem.readInt(u32, tiff[4..8], endian);
    if (ifd > tiff.len or tiff.len - ifd < 2) return null;
    const count = std.mem.readInt(u16, tiff[ifd..][0..2], endian);
    for (0..count) |i| {
        const at = ifd + 2 + 12 * i;
        if (at + 12 > tiff.len) break; // Pillow stopped reading the directory there and kept the tags before
        const e = tiff[at..][0..12];
        const kind = std.mem.readInt(u16, e[2..4], endian);
        const n = std.mem.readInt(u32, e[4..8], endian);
        const unit = unitSize(kind) orelse continue;
        const size = @as(u64, n) * unit;
        const value: []const u8 = if (size <= 4) e[8..][0..@intCast(size)] else blk: {
            const off = std.mem.readInt(u32, e[8..12], endian);
            if (off > tiff.len or tiff.len - off < size) break; // a short read ended Pillow's directory
            break :blk tiff[off..][0..@intCast(size)];
        };
        if (value.len == 0) continue;
        if (std.mem.readInt(u16, e[0..2], endian) == orientation_tag) found = .{ .kind = kind, .value = value };
    }
    const e = found orelse return null;
    return e.number(endian);
}

const Entry = struct {
    kind: u16,
    value: []const u8,

    /// The first value as Pillow compared it with the transposes' keys; 0 when it equals none of them.
    fn number(e: Entry, endian: std.builtin.Endian) u32 {
        const v = e.value;
        const x: f64 = switch (e.kind) {
            3 => @floatFromInt(std.mem.readInt(u16, v[0..2], endian)),
            4, 13 => @floatFromInt(std.mem.readInt(u32, v[0..4], endian)),
            6 => @floatFromInt(@as(i8, @bitCast(v[0]))),
            8 => @floatFromInt(std.mem.readInt(i16, v[0..2], endian)),
            9 => @floatFromInt(std.mem.readInt(i32, v[0..4], endian)),
            5 => ratio(@floatFromInt(std.mem.readInt(u32, v[0..4], endian)), @floatFromInt(std.mem.readInt(u32, v[4..8], endian))),
            10 => ratio(@floatFromInt(std.mem.readInt(i32, v[0..4], endian)), @floatFromInt(std.mem.readInt(i32, v[4..8], endian))),
            11 => @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, v[0..4], endian)))),
            12 => @bitCast(std.mem.readInt(u64, v[0..8], endian)),
            16, 18 => @floatFromInt(std.mem.readInt(u64, v[0..8], endian)),
            17 => @floatFromInt(std.mem.readInt(i64, v[0..8], endian)),
            else => 0, // bytes and text: a value no transpose key equals
        };
        return if (x >= 2 and x <= 8 and x == @trunc(x)) @intFromFloat(x) else 0;
    }
};

/// A rational's value; Pillow's zero denominator was NaN, which equals no key.
fn ratio(num: f64, den: f64) f64 {
    return if (den == 0) 0 else num / den;
}

/// The bytes a value of a TIFF type takes, for the types Pillow loads; null for a type it skips.
fn unitSize(kind: u16) ?u64 {
    return switch (kind) {
        1, 2, 6, 7 => 1,
        3, 8 => 2,
        4, 9, 11, 13 => 4,
        5, 10, 12, 16, 17, 18 => 8,
        else => null,
    };
}

/// XMP's first `tiff:Orientation="D` or `tiff:Orientation>D`, D one digit.
fn fromXmp(xmp: []const u8) ?u32 {
    const key = "tiff:Orientation";
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, xmp, from, key)) |at| : (from = at + 1) {
        const rest = xmp[at + key.len ..];
        const quoted = std.mem.startsWith(u8, rest, "=\"");
        if (!quoted and !std.mem.startsWith(u8, rest, ">")) continue;
        const digit = rest[if (quoted) 2 else 1..];
        if (digit.len > 0 and std.ascii.isDigit(digit[0])) return digit[0] - '0';
    }
    return null;
}

test "orientation from EXIF, either byte order, and XMP only without the EXIF tag" {
    const le = "Exif\x00\x00II\x2a\x00\x08\x00\x00\x00\x01\x00\x12\x01\x03\x00\x01\x00\x00\x00\x06\x00\x00\x00";
    const be = "MM\x00\x2a\x00\x00\x00\x08\x00\x01\x01\x12\x00\x03\x00\x00\x00\x01\x00\x08\x00\x00";
    try std.testing.expectEqual(@as(u32, 6), try orientation(le, null, .refuse));
    try std.testing.expectEqual(@as(u32, 8), try orientation(be, "<x tiff:Orientation=\"3\"/>", .refuse));
    try std.testing.expectEqual(@as(u32, 3), try orientation(null, "<tiff:Orientation>3</tiff:Orientation>", .refuse));
    try std.testing.expectEqual(@as(u32, 7), try orientation(null, "tiff:Orientation tiff:Orientation=\"7\"", .refuse));
    try std.testing.expectEqual(@as(u32, 1), try orientation(null, "tiff:Orientation=7", .refuse));
    const bad = "Exif\x00\x00XX\x2a\x00\x08\x00\x00\x00";
    try std.testing.expectError(error.Invalid, orientation(bad, null, .refuse));
    try std.testing.expectEqual(@as(u32, 1), try orientation(bad, "tiff:Orientation=\"6\"", .ignore));
    // a BYTE-typed tag is present, so XMP is not read, and Pillow's bytes value turned nothing
    const byte = "II\x2a\x00\x08\x00\x00\x00\x01\x00\x12\x01\x01\x00\x01\x00\x00\x00\x06\x00\x00\x00";
    try std.testing.expectEqual(@as(u32, 0), try orientation(byte, "tiff:Orientation=\"6\"", .refuse));
}
