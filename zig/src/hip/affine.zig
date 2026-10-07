//! Original MLX affine bit-stream decoding; no device arithmetic qualification implied.
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

/// Generic packed-affine domain from PR #144's Python reference.
pub const Shape = struct {
    rows: usize,
    outputs: usize,
    inputs: usize,
    bits: u8,
    group: usize,

    pub fn validate(self: Shape) error{Invalid}!void {
        if (self.rows == 0 or self.outputs == 0 or self.inputs == 0 or !validBits(self.bits))
            return error.Invalid;
        if (self.group != 32 and self.group != 64 and self.group != 128) return error.Invalid;
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

/// core/row_projection Sum.f32 candidate: two 32-lane output warps per block.
pub fn launchRowF32(
    function: @import("module.zig").Function,
    stream: @import("stream.zig").Stream,
    shape: Shape,
    buffers: [5]@import("memory.zig").DeviceBuffer,
) @import("runtime.zig").Error!void {
    if (shape.bits != 4 or (shape.group != 32 and shape.group != 64)) return error.Invalid;
    return launchWithOutputsPerBlock(function, stream, shape, buffers, 2);
}

fn launchWithOutputsPerBlock(
    function: @import("module.zig").Function,
    stream: @import("stream.zig").Stream,
    shape: Shape,
    buffers: [5]@import("memory.zig").DeviceBuffer,
    outputs_per_block: usize,
) @import("runtime.zig").Error!void {
    const lengths = try shape.byteLengths();
    if (shape.rows > std.math.maxInt(u32) or shape.outputs > std.math.maxInt(u32) - 63)
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
        .grid = .{ .x = @intCast((shape.outputs + outputs_per_block - 1) / outputs_per_block), .y = @intCast(shape.rows) },
        .block = .{ .x = 64 },
    }, stream, &args);
}

pub fn validBits(bits: u8) bool {
    return switch (bits) {
        2, 3, 4, 5, 6, 8 => true,
        else => false,
    };
}

/// Decode one code from a row, including codes crossing a 32-bit word boundary.
pub fn code(words: []const u32, index: usize, bits: u8) error{Invalid}!u8 {
    if (!validBits(bits)) return error.Invalid;
    const bit = std.math.mul(usize, index, bits) catch return error.Invalid;
    const word = bit / 32;
    const shift: u5 = @intCast(bit % 32);
    if (word >= words.len) return error.Invalid;
    var value = words[word] >> shift;
    if (@as(u8, shift) + bits > 32) {
        if (word + 1 >= words.len) return error.Invalid;
        const high_shift: u5 = @intCast(32 - @as(u8, shift));
        value |= words[word + 1] << high_shift;
    }
    return @intCast(value & ((@as(u32, 1) << @as(u5, @intCast(bits))) - 1));
}

test "all supported widths match independent bit-by-bit decoding across boundaries" {
    const words = [_]u32{ 0x89abcdef, 0x76543210, 0xfedcba98, 0x13579bdf };
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| {
        for (0..words.len * 32 / bits) |index| {
            var expected: u8 = 0;
            for (0..bits) |j| {
                const at = index * bits + j;
                const v = (words[at / 32] >> @as(u5, @intCast(at % 32))) & 1;
                expected |= @as(u8, @intCast(v)) << @as(u3, @intCast(j));
            }
            try std.testing.expectEqual(expected, try code(&words, index, bits));
        }
    }
}

test "unsupported widths, missing high word and index overflow fail closed" {
    const words = [_]u32{0xffffffff};
    for ([_]u8{ 0, 1, 7, 9, 255 }) |bits|
        try std.testing.expectError(error.Invalid, code(&words, 0, bits));
    try std.testing.expectError(error.Invalid, code(&words, 10, 3));
    try std.testing.expectError(error.Invalid, code(&words, 8, 4));
    try std.testing.expectError(error.Invalid, code(&words, std.math.maxInt(usize), 8));
    try std.testing.expectError(error.Invalid, code(&.{}, 0, 4));
}

test "odd row/output counts are valid but incomplete quantization groups are not" {
    const shape: Shape = .{ .rows = 3, .outputs = 17, .inputs = 96, .bits = 3, .group = 32 };
    try shape.validate();
    try std.testing.expectEqualSlices(usize, &.{ 576, 612, 102, 102, 102 }, &(try shape.byteLengths()));
    var invalid = shape;
    invalid.inputs = 95;
    try std.testing.expectError(error.Invalid, invalid.validate());
    invalid = shape;
    invalid.rows = std.math.maxInt(usize);
    try std.testing.expectError(error.Invalid, invalid.validate());
    invalid = shape;
    invalid.group = 16;
    try std.testing.expectError(error.Invalid, invalid.validate());
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
