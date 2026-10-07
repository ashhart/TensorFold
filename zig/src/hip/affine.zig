//! Four-bit BF16 row projection with FP32 accumulation.
const std = @import("std");

/// Exact BF16 comparison also rejects infinities/NaNs on either side.
pub fn mismatchCount(actual: []const u8, expected: []const u8) error{Invalid}!usize {
    if (actual.len != expected.len or actual.len % 2 != 0) return error.Invalid;
    var bad: usize = 0;
    for (0..actual.len / 2) |i| {
        const x = std.mem.readInt(u16, actual[i * 2 ..][0..2], .little);
        const y = std.mem.readInt(u16, expected[i * 2 ..][0..2], .little);
        if (x != y or (x & 0x7f80) == 0x7f80 or (y & 0x7f80) == 0x7f80) bad += 1;
    }
    return bad;
}

test "every single-bit output mutation is rejected including signed zero" {
    const golden = [_]u8{ 0xc4, 0x3f, 0, 0 };
    try std.testing.expectEqual(@as(usize, 0), try mismatchCount(&golden, &golden));
    for (0..golden.len * 8) |bit| {
        var changed = golden;
        changed[bit / 8] ^= @as(u8, 1) << @as(u3, @intCast(bit % 8));
        try std.testing.expectEqual(@as(usize, 1), try mismatchCount(&changed, &golden));
    }
    try std.testing.expectEqual(@as(usize, 1), try mismatchCount(&.{ 0x80, 0x7f }, &.{ 0x80, 0x7f }));
    try std.testing.expectError(error.Invalid, mismatchCount(&.{1}, &.{1}));
}

/// Supported row-projection shapes.
pub const Shape = struct {
    rows: usize,
    outputs: usize,
    inputs: usize,
    bits: u8,
    group: usize,

    pub fn validate(self: Shape) error{Invalid}!void {
        if (self.rows == 0 or self.outputs == 0 or self.inputs == 0 or self.bits != 4)
            return error.Invalid;
        if (self.group != 32 and self.group != 64) return error.Invalid;
        if (self.inputs % self.group != 0) return error.Invalid;
        const packed_bits = std.math.mul(usize, self.inputs, self.bits) catch return error.Invalid;
        _ = std.math.mul(usize, self.rows, self.inputs) catch return error.Invalid;
        _ = std.math.mul(usize, self.rows, self.outputs) catch return error.Invalid;
        _ = std.math.mul(usize, self.outputs, packed_bits / 32) catch return error.Invalid;
    }

    pub fn byteLengths(self: Shape) error{Invalid}![5]usize {
        try self.validate();
        const counts = [5]usize{
            self.rows * self.inputs,
            self.outputs * (self.inputs * self.bits / 32),
            self.outputs * (self.inputs / self.group),
            self.outputs * (self.inputs / self.group),
            self.rows * self.outputs,
        };
        var result: [5]usize = undefined;
        for (counts, [_]usize{ 2, 4, 2, 2, 2 }, 0..) |count, width, i|
            result[i] = std.math.mul(usize, count, width) catch return error.Invalid;
        return result;
    }
};

/// Two 32-lane output warps per block, independent of batch width.
pub fn launchRowF32(
    function: @import("module.zig").Function,
    stream: @import("stream.zig").Stream,
    shape: Shape,
    buffers: [5]@import("memory.zig").DeviceBuffer,
) @import("runtime.zig").Error!void {
    const lengths = try shape.byteLengths();
    if (shape.rows > std.math.maxInt(u32) or shape.outputs > std.math.maxInt(u32) - 1)
        return error.Invalid;
    for (buffers, lengths, [_]usize{ 2, 4, 2, 2, 2 }) |buffer, len, alignment| {
        if (buffer.r != function.r or buffer.ptr == null or buffer.len < len or
            @intFromPtr(buffer.ptr.?) % alignment != 0) return error.Invalid;
    }
    var args: @import("args.zig").Args = .{};
    for (buffers) |buffer| try args.add(buffer.ptr);
    try args.add(shape.rows);
    try args.add(shape.outputs);
    try args.add(shape.inputs);
    try args.add(@as(c_uint, shape.bits));
    try args.add(@as(c_uint, @intCast(shape.group)));
    try @import("launch.zig").launch(function, .{
        .grid = .{ .x = @intCast((shape.outputs + 1) / 2), .y = @intCast(shape.rows) },
        .block = .{ .x = 64 },
    }, stream, &args);
}

test "odd row/output counts are valid but incomplete quantization groups are not" {
    const shape: Shape = .{ .rows = 3, .outputs = 17, .inputs = 96, .bits = 4, .group = 32 };
    try shape.validate();
    try std.testing.expectEqualSlices(usize, &.{ 576, 816, 102, 102, 102 }, &(try shape.byteLengths()));
    var invalid = shape;
    invalid.inputs = 95;
    try std.testing.expectError(error.Invalid, invalid.validate());
    invalid = shape;
    invalid.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.Invalid, invalid.validate());
    invalid = shape;
    invalid.group = 16;
    try std.testing.expectError(error.Invalid, invalid.validate());
    invalid.group = 128;
    try std.testing.expectError(error.Invalid, invalid.validate());
    for ([_]u8{ 0, 2, 3, 5, 6, 8 }) |bits| {
        invalid = shape;
        invalid.bits = bits;
        try std.testing.expectError(error.Invalid, invalid.validate());
    }
}

test "launch rejects short buffers without invoking HIP" {
    const Runtime = @import("runtime.zig").Runtime;
    const r: Runtime = undefined;
    const buffer = @import("memory.zig").DeviceBuffer{ .r = &r, .ptr = @ptrFromInt(16), .len = 0 };
    try std.testing.expectError(error.Invalid, launchRowF32(
        .{ .r = &r, .handle = null },
        .{ .r = &r, .handle = null },
        .{ .rows = 1, .outputs = 1, .inputs = 32, .bits = 4, .group = 32 },
        @splat(buffer),
    ));
}
