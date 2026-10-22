//! NumPy .npy files (version 1-3, little-endian, C order): shape, dtype and the data's bytes.
const std = @import("std");

pub const Array = struct {
    descr: []const u8, // "<u2", "<f4", "<u4", "<i4", ...
    shape: [8]usize = @splat(0),
    rank: usize = 0,
    data: []const u8,

    pub fn count(self: Array) usize {
        var n: usize = 1;
        for (self.shape[0..self.rank]) |d| n *= d;
        return n;
    }
};

/// Parse `bytes` (the whole file); `data` points into it.
pub fn parse(bytes: []const u8) !Array {
    const array = try parseHeader(bytes);
    const expected = try byteCount(array);
    if (array.data.len != expected) return error.BadPayload;
    return array;
}

/// Parse a complete header and its available payload prefix for callers that stream data separately.
pub fn parseHeader(bytes: []const u8) !Array {
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.NotNpy;
    const major = bytes[6];
    if (major < 1 or major > 3 or bytes[7] != 0) return error.UnsupportedVersion;
    if (major != 1 and bytes.len < 12) return error.BadHeader;
    const header_len: usize, const at: usize = if (major == 1)
        .{ std.mem.readInt(u16, bytes[8..10], .little), 10 }
    else
        .{ std.mem.readInt(u32, bytes[8..12], .little), 12 };
    if (header_len > bytes.len - at) return error.BadHeader;
    const header = bytes[at..][0..header_len];
    var a = Array{ .descr = try field(header, "'descr':"), .data = bytes[at + header_len ..] };
    if (std.mem.indexOf(u8, header, "'fortran_order': True") != null) return error.FortranOrder;
    if (std.mem.indexOf(u8, header, "'fortran_order': False") == null) return error.BadHeader;
    const s = try shapeText(header);
    var it = std.mem.tokenizeAny(u8, s, ", ");
    while (it.next()) |d| {
        if (a.rank >= a.shape.len) return error.RankTooHigh;
        a.shape[a.rank] = std.fmt.parseInt(usize, d, 10) catch return error.BadHeader;
        a.rank += 1;
    }
    _ = try byteCount(a);
    return a;
}

fn byteCount(array: Array) !usize {
    const dtype = array.descr;
    if (dtype.len < 3 or (dtype[0] != '<' and dtype[0] != '|' and dtype[0] != '=')) return error.UnsupportedDtype;
    if (dtype[0] == '=' and @import("builtin").cpu.arch.endian() != .little) return error.UnsupportedDtype;
    const item = std.fmt.parseInt(usize, dtype[2..], 10) catch return error.UnsupportedDtype;
    const supported = switch (dtype[1]) {
        'b' => item == 1,
        'u', 'i' => item == 1 or item == 2 or item == 4 or item == 8,
        'f' => item == 2 or item == 4 or item == 8 or item == 16,
        'c' => item == 8 or item == 16 or item == 32,
        else => false,
    };
    if (!supported or (dtype[0] == '|' and item != 1)) return error.UnsupportedDtype;
    var bytes = item;
    for (array.shape[0..array.rank]) |dimension|
        bytes = std.math.mul(usize, bytes, dimension) catch return error.BadHeader;
    return bytes;
}

fn field(header: []const u8, key: []const u8) ![]const u8 {
    const i = std.mem.indexOf(u8, header, key) orelse return error.BadHeader;
    const q0 = std.mem.indexOfScalarPos(u8, header, i + key.len, '\'') orelse return error.BadHeader;
    const q1 = std.mem.indexOfScalarPos(u8, header, q0 + 1, '\'') orelse return error.BadHeader;
    return header[q0 + 1 .. q1];
}

fn shapeText(header: []const u8) ![]const u8 {
    const i = std.mem.indexOf(u8, header, "'shape':") orelse return error.BadHeader;
    const p0 = std.mem.indexOfScalarPos(u8, header, i, '(') orelse return error.BadHeader;
    const p1 = std.mem.indexOfScalarPos(u8, header, p0, ')') orelse return error.BadHeader;
    return header[p0 + 1 .. p1];
}

test "parse a v1 header" {
    const header = "{'descr': '<u2', 'fortran_order': False, 'shape': (2, 3), }";
    var buf: [128]u8 = undefined;
    @memcpy(buf[0..6], "\x93NUMPY");
    buf[6] = 1;
    buf[7] = 0;
    std.mem.writeInt(u16, buf[8..10], header.len, .little);
    @memcpy(buf[10 .. 10 + header.len], header);
    const a = try parse(buf[0 .. 10 + header.len + 12]);
    try std.testing.expectEqualStrings("<u2", a.descr);
    try std.testing.expectEqual(@as(usize, 6), a.count());
    try std.testing.expectEqual(@as(usize, 12), a.data.len);
}

test {
    _ = @import("npy_security_test.zig");
}
