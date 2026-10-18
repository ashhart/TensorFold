//! Independent byte fixtures and malformed-storage boundaries for lossless scale decoding.

const std = @import("std");
const Plane = @import("nvfp4_csf.zig").Plane;
const expect = std.testing.expectEqualSlices;

test "handwritten slab preserves nibble order, bases and exception bytes" {
    var fixed: [32]u8 = @splat(0);
    fixed[0] = 240;
    fixed[1] = 1;
    fixed[16] = 0xf0;
    fixed[17] = 0x21;
    const exceptions = [_]u8{ 2, 0, 0, 255, 31, 0, 0, 128 };
    const plane = try Plane.init(16, 2, &fixed, &exceptions);
    var out: [32]u8 = undefined;
    try plane.decode(&out, 0, 16, 0, 2);
    try expect(u8, &.{ 240, 255, 255, 3 }, out[0..4]);
    try std.testing.expectEqual(@as(u8, 128), out[31]);
    var odd: [2]u8 = undefined;
    try plane.decode(&odd, 0, 2, 1, 1);
    try expect(u8, &.{ 255, 3 }, &odd);
}

const Fixture = struct {
    fixed: []u8,
    exceptions: []u8,
    source: []u8,
    rows: usize,
    columns: usize,

    fn init(rows: usize, columns: usize) !Fixture {
        const gpa = std.testing.allocator;
        const source = try gpa.alloc(u8, rows * columns);
        errdefer gpa.free(source);
        const fixed = try gpa.alloc(u8, rows + source.len / 2);
        errdefer gpa.free(fixed);
        const exceptions = try gpa.alloc(u8, ((source.len + 96) / 97) * 4);
        errdefer gpa.free(exceptions);
        @memset(fixed, 0);
        var cursor: usize = 0;
        for (0..rows) |row| {
            const base: u8 = @intCast((row * 37 + 19) % 241);
            const slab = row / 16 * (16 + columns * 8);
            fixed[slab + row % 16] = base;
            for (0..columns) |column| {
                const i = row * columns + column;
                const delta: u8 = @intCast((column * 11 + row * 3) % 16);
                source[i] = base + delta;
                const offset = slab + 16 + row % 16 * (columns / 2) + column / 2;
                fixed[offset] |= delta << @as(u3, if (column % 2 == 0) 0 else 4);
                if (i % 97 == 0) {
                    source[i] = @intCast((i / 97) % 256);
                    const record = @as(u32, @intCast(i)) | (@as(u32, source[i]) << 24);
                    std.mem.writeInt(u32, exceptions[cursor..][0..4], record, .little);
                    cursor += 4;
                }
            }
        }
        return .{ .fixed = fixed, .exceptions = exceptions, .source = source, .rows = rows, .columns = columns };
    }

    fn deinit(self: Fixture) void {
        const gpa = std.testing.allocator;
        gpa.free(self.fixed);
        gpa.free(self.exceptions);
        gpa.free(self.source);
    }

    fn check(self: Fixture, row: usize, rows: usize, column: usize, columns: usize) !void {
        const out = try std.testing.allocator.alloc(u8, rows * columns);
        defer std.testing.allocator.free(out);
        @memset(out, 0xd6);
        const plane = try Plane.init(self.rows, self.columns, self.fixed, self.exceptions);
        try plane.decode(out, row, rows, column, columns);
        for (0..rows) |r| try expect(u8, self.source[(row + r) * self.columns + column ..][0..columns], out[r * columns ..][0..columns]);
    }
};

test "full planes and row/column TP partitions reproduce independent source bytes" {
    for ([_][2]usize{ .{ 16, 2 }, .{ 32, 20 }, .{ 640, 160 }, .{ 2560, 40 } }) |shape| {
        const fixture = try Fixture.init(shape[0], shape[1]);
        defer fixture.deinit();
        try fixture.check(0, shape[0], 0, shape[1]);
        for (0..2) |rank| {
            try fixture.check(rank * shape[0] / 2, shape[0] / 2, 0, shape[1]);
            try fixture.check(0, shape[0], rank * shape[1] / 2, shape[1] / 2);
        }
        try fixture.check(15, 1, 1, shape[1] - 1);
    }
}

test "all 256 byte values survive exception decoding unchanged" {
    const fixed: [144]u8 = @splat(0);
    var exceptions: [1024]u8 = undefined;
    for (0..256) |i| std.mem.writeInt(u32, exceptions[i * 4 ..][0..4], @as(u32, @intCast(i)) * 0x1000001, .little);
    const plane = try Plane.init(16, 16, &fixed, &exceptions);
    var out: [256]u8 = undefined;
    try plane.decode(&out, 0, 16, 0, 16);
    for (out, 0..) |value, i| try std.testing.expectEqual(@as(u8, @intCast(i)), value);
}

test "the last u24 position is valid and larger planes are refused" {
    const columns = 1 << 20;
    const fixed = try std.testing.allocator.alloc(u8, 16 + 8 * columns);
    defer std.testing.allocator.free(fixed);
    @memset(fixed, 0);
    const plane = try Plane.init(16, columns, fixed, &.{ 255, 255, 255, 173 });
    var out: [1]u8 = undefined;
    try plane.decode(&out, 15, 1, columns - 1, 1);
    try expect(u8, &.{173}, &out);
    try std.testing.expectError(error.BadGeometry, Plane.init(32, columns, fixed, &.{}));
}

test "bad geometry, truncated streams, overflow bases and unordered exceptions are refused" {
    var fixed: [32]u8 = @splat(0);
    const bad = std.testing.expectError;
    for ([_][2]usize{ .{ 0, 2 }, .{ 1, 2 }, .{ 16, 0 }, .{ 16, 3 }, .{ 16, 1 << 24 }, .{ 1 << 24, 2 } }) |shape|
        try bad(error.BadGeometry, Plane.init(shape[0], shape[1], &fixed, &.{}));
    try bad(error.BadFixedStream, Plane.init(16, 2, fixed[0..31], &.{}));
    fixed[15] = 241;
    try bad(error.BadFixedStream, Plane.init(16, 2, &fixed, &.{}));
    fixed[15] = 0;
    try bad(error.BadExceptions, Plane.init(16, 2, &fixed, &.{0}));
    try bad(error.BadExceptions, Plane.init(16, 2, &fixed, &.{ 32, 0, 0, 1 }));
    try bad(error.BadExceptions, Plane.init(16, 2, &fixed, &.{ 1, 0, 0, 2, 1, 0, 0, 3 }));
    try bad(error.BadExceptions, Plane.init(16, 2, &fixed, &.{ 2, 0, 0, 2, 1, 0, 0, 3 }));
}

test "invalid slices and aliased storage fail before output changes" {
    var fixed: [32]u8 = @splat(0);
    var exceptions = [_]u8{ 0, 0, 0, 1 };
    const plane = try Plane.init(16, 2, &fixed, &exceptions);
    var out: [32]u8 = @splat(0xd6);
    const bad = std.testing.expectError;
    try bad(error.BadSlice, plane.decode(&out, 17, 0, 0, 2));
    try bad(error.BadSlice, plane.decode(&out, 0, 17, 0, 2));
    try bad(error.BadSlice, plane.decode(&out, 0, 16, 3, 0));
    try bad(error.BadSlice, plane.decode(&out, 0, 16, 1, 2));
    try bad(error.BadSlice, plane.decode(out[0..31], 0, 16, 0, 2));
    const poison: [32]u8 = @splat(0xd6);
    try expect(u8, &poison, &out);
    try bad(error.OverlappingOutput, plane.decode(&fixed, 0, 16, 0, 2));
    try bad(error.OverlappingOutput, plane.decode(&exceptions, 0, 2, 0, 2));
    try plane.decode(out[0..0], 16, 0, 2, 0);
}
