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
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.NotNpy;
    const major = bytes[6];
    const header_len: usize, const at: usize = if (major == 1)
        .{ std.mem.readInt(u16, bytes[8..10], .little), 10 }
    else
        .{ std.mem.readInt(u32, bytes[8..12], .little), 12 };
    const header = bytes[at .. at + header_len];
    var a = Array{ .descr = try field(header, "'descr':"), .data = bytes[at + header_len ..] };
    if (std.mem.indexOf(u8, header, "'fortran_order': True") != null) return error.FortranOrder;
    const s = try shapeText(header);
    var it = std.mem.tokenizeAny(u8, s, ", ");
    while (it.next()) |d| {
        a.shape[a.rank] = try std.fmt.parseInt(usize, d, 10);
        a.rank += 1;
    }
    return a;
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
