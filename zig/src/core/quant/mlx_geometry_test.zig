//! Affine geometry at the actual reader/upload/view seam, with independent small packed-value expectations.
const std = @import("std");
const mlx = @import("mlx.zig");
const types = @import("types.zig");
const geometry = @import("mlx_geometry.zig");
const a = std.testing.allocator;
const Tensor = types.Tensor;

fn tensor(dtype: types.DType, rows: usize, columns: usize, bytes: []const u8) Tensor {
    return .{ .dtype = dtype, .rank = 2, .shape = .{ rows, columns, 1, 1, 1 }, .bytes = bytes };
}

fn uploaded(t: Tensor) types.Buf {
    return .{ .ptr = @intFromPtr(t.bytes.ptr), .len = t.bytes.len, .dtype = t.dtype, .rank = t.rank, .shape = t.shape };
}

fn device(h: mlx.Host) mlx.Device {
    return .{ .words = uploaded(h.words), .scales = uploaded(h.scales), .biases = uploaded(h.biases), .bits = h.bits, .group = h.group };
}

fn putFloat(dtype: types.DType, bytes: []u8, index: usize, value: f32) void {
    switch (dtype) {
        .f32 => std.mem.writeInt(u32, bytes[index * 4 ..][0..4], @bitCast(value), .little),
        .f16 => std.mem.writeInt(u16, bytes[index * 2 ..][0..2], @bitCast(@as(f16, @floatCast(value))), .little),
        .bf16 => std.mem.writeInt(u16, bytes[index * 2 ..][0..2], @truncate(@as(u32, @bitCast(value)) >> 16), .little),
        else => unreachable,
    }
}

const Table = struct {
    h: mlx.Host,
    pub fn get(t: Table, key: []const u8) error{MissingTensor}!Tensor {
        if (std.mem.endsWith(u8, key, ".weight")) return t.h.words;
        if (std.mem.endsWith(u8, key, ".scales")) return t.h.scales;
        if (std.mem.endsWith(u8, key, ".biases")) return t.h.biases;
        return error.MissingTensor;
    }
    pub fn width(t: Table, _: []const u8) error{UnsupportedQuantization}!mlx.Width {
        return .{ .bits = t.h.bits, .group = t.h.group };
    }
};

test "affine geometry keeps all admitted widths groups table dtypes and independently decoded values" {
    inline for (mlx.bit_widths) |bits| inline for (.{ 32, 64, 128 }) |group| {
        const k: usize = 2 * group;
        var words: [2 * k * bits / 8]u8 = @splat(0);
        var wanted: [2 * k]f64 = undefined;
        for (&wanted, 0..) |*value, i| {
            const code: u8 = @intCast(i % (1 << bits));
            value.* = @as(f64, @floatFromInt(code)) * 2 - 1;
            for (0..bits) |bit| {
                const offset = i * bits + bit;
                words[offset / 8] |= ((code >> @intCast(bit)) & 1) << @intCast(offset % 8);
            }
        }
        inline for ([_]types.DType{ .bf16, .f16, .f32 }) |dtype| {
            var scale: [4 * dtype.size()]u8 = undefined;
            var bias: [4 * dtype.size()]u8 = undefined;
            for (0..4) |i| {
                putFloat(dtype, &scale, i, 2);
                putFloat(dtype, &bias, i, -1);
            }
            const h: mlx.Host = .{
                .words = tensor(.u32, 2, k * bits / 32, &words),
                .scales = tensor(dtype, 2, 2, &scale),
                .biases = tensor(dtype, 2, 2, &bias),
                .bits = bits,
                .group = group,
            };
            const read = try mlx.read(a, Table{ .h = h }, "projection");
            try std.testing.expectEqualSlices(u8, &words, read.words.bytes);
            try std.testing.expectEqualSlices(u8, &scale, read.scales.bytes);
            try std.testing.expectEqualSlices(u8, &bias, read.biases.bytes);
            const v = try mlx.view(device(read), false);
            try std.testing.expectEqual(@as(u32, 2), v.n);
            try std.testing.expectEqual(@as(u32, k), v.k);
            try std.testing.expectEqual(@intFromPtr(words[0..].ptr), v.matrix.words);
            var got: [2 * k]f64 = undefined;
            try mlx.reference(read, &got);
            try std.testing.expectEqualSlices(f64, &wanted, &got);
            const decoded = try mlx.dequant(a, read.words, read.scales, read.biases, group);
            defer a.free(decoded);
            for (decoded, wanted) |x, y| try std.testing.expectEqual(y, @as(f64, x));
        }
    };
}

test "affine geometry validates stacks and rejects inconsistent buffers before any device upload" {
    const words: [32]u8 = @splat(0);
    const table: [4]u8 = @splat(0);
    var h: mlx.Host = .{
        .words = tensor(.u32, 2, 4, &words),
        .scales = tensor(.bf16, 2, 1, &table),
        .biases = tensor(.bf16, 2, 1, &table),
        .bits = 4,
        .group = 32,
    };
    inline for (.{ "words", "scales", "biases" }) |field| {
        const columns = @field(h, field).shape[1];
        @field(h, field).rank = 3;
        @field(h, field).shape = .{ 2, 1, columns, 1, 1 };
    }
    const v = try mlx.view(device(h), true);
    try std.testing.expectEqual(@as(u32, 1), v.n);
    try std.testing.expectEqual(@as(u32, 32), v.k);
    try std.testing.expectError(error.UnsupportedTables, mlx.view(device(h), false));
    var short = device(h);
    short.words.len -= 1;
    try std.testing.expectError(error.UnsupportedTables, mlx.view(short, true));
    var mismatch = device(h);
    mismatch.biases.dtype = .f16;
    try std.testing.expectError(error.UnsupportedTables, mlx.view(mismatch, true));
    var wrapped = device(h);
    wrapped.words.ptr = std.math.maxInt(u64) - 7;
    try std.testing.expectError(error.UnsupportedTables, mlx.view(wrapped, true));
    var zero = device(h);
    zero.bits = 0;
    try std.testing.expectError(error.UnsupportedTables, mlx.view(zero, true));
    var calls: usize = 0;
    const never = struct {
        fn put(ctx: *anyopaque, _: Tensor) anyerror!types.Buf {
            const count: *usize = @ptrCast(@alignCast(ctx));
            count.* += 1;
            return error.UploadMustNotRun;
        }
    }.put;
    h.scales.shape[0] = 0;
    try std.testing.expectError(error.UnexpectedTensor, mlx.upload(.{ .ctx = &calls, .put = never }, h));
    try std.testing.expectEqual(@as(usize, 0), calls);
}

test "affine geometry bounds backend dimensions and intermediate products without device access" {
    if (@bitSizeOf(usize) < 64) return;
    const wide: usize = @as(usize, std.math.maxInt(u32)) + 1;
    var word: types.Buf = .{ .ptr = 16, .len = wide * 16, .dtype = .u32, .rank = 2, .shape = .{ wide, 4, 1, 1, 1 } };
    var scale: types.Buf = .{ .ptr = 16, .len = wide * 2, .dtype = .bf16, .rank = 2, .shape = .{ wide, 1, 1, 1, 1 } };
    try std.testing.expectError(error.UnexpectedTensor, geometry.shape(word, scale, scale, 4, 32, false));
    word.shape = .{ 1, wide / 8, 1, 1, 1 };
    word.len = wide / 2;
    scale.shape = .{ 1, wide / 32, 1, 1, 1 };
    scale.len = wide / 16;
    try std.testing.expectError(error.UnexpectedTensor, geometry.shape(word, scale, scale, 4, 32, false));
    try std.testing.expectError(error.UnexpectedTensor, geometry.product(std.math.maxInt(usize), 2));
    word.shape = .{ 0, 4, 1, 1, 1 };
    word.len = 0;
    try std.testing.expectError(error.UnexpectedTensor, geometry.shape(word, scale, scale, 4, 32, false));
}

test "affine geometry preserves the reader's mixed-table conversion and refuses ordinary shape disagreement" {
    const words: [16]u8 = @splat(0);
    const bf16 = [_]u8{ 0x80, 0x3f };
    const fp16 = [_]u8{ 0x00, 0xbc };
    var h: mlx.Host = .{
        .words = tensor(.u32, 1, 4, &words),
        .scales = tensor(.bf16, 1, 1, &bf16),
        .biases = tensor(.f16, 1, 1, &fp16),
        .bits = 4,
        .group = 32,
    };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const converted = try mlx.read(arena.allocator(), Table{ .h = h }, "projection");
    try std.testing.expectEqual(types.DType.f32, converted.scales.dtype);
    try std.testing.expectEqual(types.DType.f32, converted.biases.dtype);
    var values: [32]f64 = undefined;
    try mlx.reference(converted, &values);
    for (values) |value| try std.testing.expectEqual(@as(f64, -1), value);
    h.words.shape[1] = 3;
    try std.testing.expectError(error.UnexpectedTensor, geometry.shape(h.words, h.scales, h.biases, 4, 32, false));
}

test "affine geometry refuses unsupported device widths and groups but keeps the scalar oracle's domain" {
    const words: [28]u8 = @splat(0);
    const table: [2]u8 = .{ 0x80, 0x3f };
    var h: mlx.Host = .{
        .words = tensor(.u32, 1, 7, &words),
        .scales = tensor(.bf16, 1, 1, &table),
        .biases = tensor(.bf16, 1, 1, &table),
        .bits = 7,
        .group = 32,
    };
    _ = try geometry.scalarShape(h.words, h.scales, h.biases, 7, 32, false);
    try std.testing.expectError(error.UnsupportedTables, mlx.view(device(h), false));
    try std.testing.expectError(error.UnexpectedTensor, mlx.halves(a, h));
    try std.testing.expectError(error.UnevenSplit, mlx.sliceRows(a, h, &.{.{ .from = 0, .to = 1 }}));
    try std.testing.expectError(error.UnevenSplit, mlx.sliceCols(a, h, .{ .rank = 0, .world = 1 }, "unsupported"));
    h.words = tensor(.u32, 1, 2, words[0..8]);
    h.bits = 4;
    h.group = 16;
    _ = try geometry.scalarShape(h.words, h.scales, h.biases, 4, 16, false);
    try std.testing.expectError(error.UnsupportedTables, mlx.view(device(h), false));
    var values: [16]f64 = undefined;
    try mlx.reference(h, &values);
    for (values) |value| try std.testing.expectEqual(@as(f64, 1), value);
}
