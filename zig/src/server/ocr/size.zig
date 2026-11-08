//! A page image's pixel size from its header alone: PNG IHDR, a JPEG frame header, WebP VP8X, VP8 or VP8L.
const std = @import("std");
const pdf = @import("../pdf.zig");

/// The width and height the header names, or null when it names none (or a zero side).
pub fn of(bytes: []const u8) ?[2]u32 {
    const wh = png(bytes) orelse pdf.jpegSize(bytes) orelse webp(bytes) orelse return null;
    return if (wh[0] > 0 and wh[1] > 0) wh else null;
}

fn png(b: []const u8) ?[2]u32 {
    if (b.len < 24 or !std.mem.startsWith(u8, b, "\x89PNG\r\n\x1a\n") or !std.mem.eql(u8, b[12..16], "IHDR")) return null;
    return .{ std.mem.readInt(u32, b[16..20], .big), std.mem.readInt(u32, b[20..24], .big) };
}

fn webp(b: []const u8) ?[2]u32 {
    if (b.len < 30 or !std.mem.eql(u8, b[0..4], "RIFF") or !std.mem.eql(u8, b[8..12], "WEBP")) return null;
    const chunk = b[12..16];
    if (std.mem.eql(u8, chunk, "VP8X")) // canvas width and height less one, 24-bit little-endian
        return .{ 1 + @as(u32, std.mem.readInt(u24, b[24..27], .little)), 1 + @as(u32, std.mem.readInt(u24, b[27..30], .little)) };
    if (std.mem.eql(u8, chunk, "VP8 ")) { // a key frame: 3-byte tag, start code 9d 01 2a, then 14-bit sides
        if (!std.mem.eql(u8, b[23..26], "\x9d\x01\x2a")) return null;
        return .{ std.mem.readInt(u16, b[26..28], .little) & 0x3fff, std.mem.readInt(u16, b[28..30], .little) & 0x3fff };
    }
    if (std.mem.eql(u8, chunk, "VP8L")) { // signature 0x2f, then 14-bit sides less one
        if (b[20] != 0x2f) return null;
        const bits = std.mem.readInt(u32, b[21..25], .little);
        return .{ (bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1 };
    }
    return null;
}
