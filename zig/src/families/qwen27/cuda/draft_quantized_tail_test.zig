//! Quantized host row views preserve independent K/V bits and decline invalid layout or ranges.
const std = @import("std");
const dl = @import("draft_load.zig");
const a = std.testing.allocator;

fn bf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn fixture(nq: usize, nk: usize, nv: usize, width: usize) !void {
    const total = nq + nk + nv;
    const input = try a.alloc(u8, total * width * 2);
    defer a.free(input);
    for (0..total) |row| for (0..width) |col| {
        const group = col / 64;
        const scale: f32 = @floatFromInt(1 + (row + group) % 3);
        const lo: f32 = @floatFromInt(@as(i32, @intCast((row * 7 + group * 3) % 17)) - 8);
        const j = col % 64;
        const fraction: f32 = switch (j) {
            0 => 0,
            1 => 15,
            2 => 2.5,
            3 => 3.5,
            else => @floatFromInt((j * 7 + row * 3 + group) % 16),
        };
        const value = if (group % 4 == 3) fraction * 1e-8 else lo + fraction * scale;
        std.mem.writeInt(u16, input[(row * width + col) * 2 ..][0..2], bf16(value), .little);
    };
    const words = try a.alloc(u32, total * width / 8);
    defer a.free(words);
    const scales = try a.alloc(u16, total * width / 64);
    defer a.free(scales);
    const biases = try a.alloc(u16, total * width / 64);
    defer a.free(biases);
    var first: usize = 0;
    for ([_]usize{ nq, nk, nv }) |rows| {
        dl.quantize4(input[first * width * 2 ..][0 .. rows * width * 2], rows, width, words[first * width / 8 ..][0 .. rows * width / 8], scales[first * width / 64 ..][0 .. rows * width / 64], biases[first * width / 64 ..][0 .. rows * width / 64]);
        first += rows;
    }
    const original = try a.dupe(u32, words);
    defer a.free(original);
    const expected_words = try a.alloc(u32, (nk + nv) * width / 8);
    defer a.free(expected_words);
    const expected_scales = try a.alloc(u16, (nk + nv) * width / 64);
    defer a.free(expected_scales);
    const expected_biases = try a.alloc(u16, (nk + nv) * width / 64);
    defer a.free(expected_biases);
    first = 0;
    for ([_]usize{ nk, nv }) |rows| {
        dl.quantize4(input[(nq + first) * width * 2 ..][0 .. rows * width * 2], rows, width, expected_words[first * width / 8 ..][0 .. rows * width / 8], expected_scales[first * width / 64 ..][0 .. rows * width / 64], expected_biases[first * width / 64 ..][0 .. rows * width / 64]);
        first += rows;
    }
    const tail = try dl.quantizedTail(words, scales, biases, nq, nk + nv, width);
    try std.testing.expectEqualSlices(u32, expected_words, tail.words);
    try std.testing.expectEqualSlices(u16, expected_scales, tail.scales);
    try std.testing.expectEqualSlices(u16, expected_biases, tail.biases);
    try std.testing.expectEqualSlices(u32, original, words);
    try std.testing.expect(tail.words.ptr == words[nq * width / 8 ..].ptr);
    try std.testing.expect(tail.scales.ptr == scales[nq * width / 64 ..].ptr);
    try std.testing.expect(tail.biases.ptr == biases[nq * width / 64 ..].ptr);
}

test "QKV tail equals separately quantized K/V across nonunit scales ties and unpadded rows" {
    try fixture(7, 5, 3, 64);
    try fixture(32, 8, 8, 256);
    try fixture(1, 1, 1, 5120);
}

test "RNE halfway values and signed-zero rows remain literal bits in the tail" {
    var input: [128 * 2]u8 = @splat(0);
    for (0..64) |j| {
        std.mem.writeInt(u16, input[j * 2 ..][0..2], if (j % 2 == 0) 0x8000 else 0, .little);
        const value: f32 = switch (j) {
            0 => 0,
            1 => 15,
            2 => 2.5,
            3 => 3.5,
            else => 0,
        };
        std.mem.writeInt(u16, input[(64 + j) * 2 ..][0..2], bf16(value), .little);
    }
    var words: [16]u32 = undefined;
    var scales: [2]u16 = undefined;
    var biases: [2]u16 = undefined;
    dl.quantize4(&input, 2, 64, &words, &scales, &biases);
    const tied = try dl.quantizedTail(&words, &scales, &biases, 1, 1, 64);
    try std.testing.expectEqual(@as(u32, 2), (tied.words[0] >> 8) & 15);
    try std.testing.expectEqual(@as(u32, 4), (tied.words[0] >> 12) & 15);
    try std.testing.expectEqual(bf16(1), tied.scales[0]);
    const zeros = try dl.quantizedTail(&words, &scales, &biases, 0, 1, 64);
    try std.testing.expectEqual(@as(u32, 0), zeros.words[0]);
}

test "tail validates all host arrays and bounds before exposing a borrowed view" {
    const words: [32]u32 = @splat(0);
    const scales: [4]u16 = @splat(0);
    const biases: [4]u16 = @splat(0);
    for ([_]usize{ 0, 1, 65 }) |width|
        try std.testing.expectError(error.InvalidQuantizedTail, dl.quantizedTail(&words, &scales, &biases, 0, 1, width));
    try std.testing.expectError(error.InvalidQuantizedTail, dl.quantizedTail(words[0..31], &scales, &biases, 0, 1, 64));
    try std.testing.expectError(error.InvalidQuantizedTail, dl.quantizedTail(&words, scales[0..3], &biases, 0, 1, 64));
    try std.testing.expectError(error.InvalidQuantizedTail, dl.quantizedTail(&words, &scales, biases[0..3], 0, 1, 64));
    try std.testing.expectError(error.InvalidQuantizedTail, dl.quantizedTail(&words, &scales, &biases, 5, 0, 64));
    try std.testing.expectError(error.InvalidQuantizedTail, dl.quantizedTail(&words, &scales, &biases, 3, 2, 64));
    try std.testing.expectError(error.InvalidQuantizedTail, dl.quantizedTail(&words, &scales, &biases, 1, std.math.maxInt(usize), 64));
    const empty = try dl.quantizedTail(&words, &scales, &biases, 4, 0, 64);
    try std.testing.expectEqual(@as(usize, 0), empty.words.len + empty.scales.len + empty.biases.len);
}

test "startup KV reuse flag is strict and remains disabled by default" {
    try std.testing.expect(!try dl.reuseKvFlag(null));
    try std.testing.expect(!try dl.reuseKvFlag("0"));
    try std.testing.expect(try dl.reuseKvFlag("1"));
    for ([_][]const u8{ "", "2", "true", " 1", "1 " }) |raw|
        try std.testing.expectError(error.InvalidQuantizedKvReuse, dl.reuseKvFlag(raw));
}
