//! Synthetic safetensors images for host loader checks and matched I/O benchmarks.
const std = @import("std");

pub const Entry = struct {
    name: []const u8,
    dtype: []const u8 = "U8",
    shape: []const usize,
    bytes: []const u8,
};

/// Header offsets follow the supplied payloads, independently of the production header parser.
pub fn image(a: std.mem.Allocator, entries: []const Entry) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.append(a, '{');
    var offset: usize = 0;
    for (entries, 0..) |entry, i| {
        if (i > 0) try header.append(a, ',');
        try header.print(a, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ entry.name, entry.dtype });
        for (entry.shape, 0..) |dim, j| try header.print(a, "{s}{d}", .{ if (j > 0) "," else "", dim });
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + entry.bytes.len });
        offset += entry.bytes.len;
    }
    try header.append(a, '}');
    const out = try a.alloc(u8, 8 + header.items.len + offset);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    var at = 8 + header.items.len;
    for (entries) |entry| {
        @memcpy(out[at..][0..entry.bytes.len], entry.bytes);
        at += entry.bytes.len;
    }
    return out;
}

/// A safetensors image whose data starts on 16 bytes: the header padded with spaces, every tensor aligned.
pub fn aligned(a: std.mem.Allocator, entries: []const Entry) ![]u8 {
    const plain = try image(a, entries);
    const header: usize = @intCast(std.mem.readInt(u64, plain[0..8], .little));
    const pad = (16 - (8 + header) % 16) % 16;
    const out = try a.alloc(u8, plain.len + pad);
    std.mem.writeInt(u64, out[0..8], header + pad, .little);
    @memcpy(out[8..][0..header], plain[8..][0..header]);
    @memset(out[8 + header ..][0..pad], ' ');
    @memcpy(out[8 + header + pad ..], plain[8 + header ..]);
    return out;
}

pub fn write(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8, entries: []const Entry) !void {
    const bytes = try image(a, entries);
    defer a.free(bytes);
    try dir.writeFile(io, .{ .sub_path = name, .data = bytes });
}
