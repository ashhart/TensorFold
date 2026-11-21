//! NumPy .npy arrays (versions 1 to 3, C order, little-endian f4 / i4 / i8) read for the reference checks.
const std = @import("std");

pub const Array = struct {
    shape: [4]usize = @splat(1),
    rank: usize = 0,
    kind: enum { f32, i32, i64 },
    data: []const u8,

    pub fn numel(a: Array) usize {
        var n: usize = 1;
        for (a.shape[0..a.rank]) |d| n *= d;
        return n;
    }

    /// The elements as f32 (caller frees).
    pub fn floats(a: Array, gpa: std.mem.Allocator) ![]f32 {
        const out = try gpa.alloc(f32, a.numel());
        for (out, 0..) |*o, i| o.* = switch (a.kind) {
            .f32 => @bitCast(std.mem.readInt(u32, a.data[i * 4 ..][0..4], .little)),
            .i32 => @floatFromInt(std.mem.readInt(i32, a.data[i * 4 ..][0..4], .little)),
            .i64 => @floatFromInt(std.mem.readInt(i64, a.data[i * 8 ..][0..8], .little)),
        };
        return out;
    }

    /// The elements as i32 (caller frees); float values must be whole numbers.
    pub fn ints(a: Array, gpa: std.mem.Allocator) ![]i32 {
        const out = try gpa.alloc(i32, a.numel());
        errdefer gpa.free(out);
        for (out, 0..) |*o, i| o.* = switch (a.kind) {
            .f32 => blk: {
                const f: f32 = @bitCast(std.mem.readInt(u32, a.data[i * 4 ..][0..4], .little));
                if (f != @trunc(f) or @abs(f) > 1e9) return error.NotAnInteger;
                break :blk @intFromFloat(f);
            },
            .i32 => std.mem.readInt(i32, a.data[i * 4 ..][0..4], .little),
            .i64 => std.math.cast(i32, std.mem.readInt(i64, a.data[i * 8 ..][0..8], .little)) orelse return error.NotAnInteger,
        };
        return out;
    }
};

pub fn parse(bytes: []const u8) !Array {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.NotNpy;
    const long = bytes[6] >= 2;
    const hlen: usize = if (long) std.mem.readInt(u32, bytes[8..12], .little) else std.mem.readInt(u16, bytes[8..10], .little);
    const start: usize = if (long) 12 else 10;
    if (bytes.len < start + hlen) return error.NotNpy;
    const head = bytes[start..][0..hlen];
    if (std.mem.indexOf(u8, head, "'fortran_order': False") == null) return error.UnsupportedNpy;
    const descr_at = (std.mem.indexOf(u8, head, "'descr': '") orelse return error.NotNpy) + "'descr': '".len;
    const descr = head[descr_at..][0..3];
    var a: Array = .{ .kind = undefined, .data = bytes[start + hlen ..] };
    var width: usize = 4;
    if (std.mem.eql(u8, descr, "<f4")) a.kind = .f32 else if (std.mem.eql(u8, descr, "<i4")) a.kind = .i32 else if (std.mem.eql(u8, descr, "<i8")) {
        a.kind = .i64;
        width = 8;
    } else return error.UnsupportedNpy;
    const shape_at = (std.mem.indexOf(u8, head, "'shape': (") orelse return error.NotNpy) + "'shape': (".len;
    const shape_end = std.mem.indexOfScalarPos(u8, head, shape_at, ')') orelse return error.NotNpy;
    var it = std.mem.tokenizeAny(u8, head[shape_at..shape_end], ", ");
    while (it.next()) |tok| {
        if (a.rank == a.shape.len) return error.UnsupportedNpy;
        a.shape[a.rank] = try std.fmt.parseInt(usize, tok, 10);
        a.rank += 1;
    }
    if (a.data.len != a.numel() * width) return error.NotNpy;
    return a;
}

test "npy: a version 1 float array" {
    const head = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 1), }";
    var buf: [128]u8 = undefined;
    @memcpy(buf[0..6], "\x93NUMPY");
    buf[6] = 1;
    buf[7] = 0;
    std.mem.writeInt(u16, buf[8..10], head.len, .little);
    @memcpy(buf[10..][0..head.len], head);
    const vals = [2]f32{ 1.5, -2.0 };
    @memcpy(buf[10 + head.len ..][0..8], std.mem.asBytes(&vals));
    const a = try parse(buf[0 .. 10 + head.len + 8]);
    try std.testing.expectEqual(@as(usize, 2), a.rank);
    try std.testing.expectEqual(@as(usize, 2), a.numel());
    const f = try a.floats(std.testing.allocator);
    defer std.testing.allocator.free(f);
    try std.testing.expectEqualSlices(f32, &vals, f);
}
