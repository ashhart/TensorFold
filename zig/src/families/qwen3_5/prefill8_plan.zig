//! Host-only admission and memory contract for the opt-in FP8 prompt path.
const std = @import("std");

pub const max_rows: usize = 4096;
pub const max_columns: usize = 17408;
pub const matmul_shared: u32 = 3 * (128 * 64 + 128 * 64 / 2 + 128 * 2);

pub fn validate(rows: usize, columns: usize) !void {
    if (rows == 0 or rows > max_rows or columns == 0 or columns > max_columns or columns % 64 != 0)
        return error.InvalidFp8PrefillShape;
}

pub fn prepareShared(columns: usize) u32 {
    return @intCast((columns + 256) * 4);
}

pub fn sizes(rows: usize, columns: usize) ![3]usize {
    try validate(rows, columns);
    return .{ rows * columns, rows * (columns / 64) * 2, rows * 4 };
}

pub fn bytes(enabled: bool, rows: usize, columns: usize) !usize {
    if (!enabled) return 0;
    var sum: usize = 0;
    for (try sizes(rows, columns)) |n| sum += std.mem.alignForward(usize, n, 256);
    return sum;
}

pub fn supported(sm: u32, hidden: usize, intermediate: usize, layers: usize, moe: bool, mixed: bool, modelopt: bool, exl3: bool, pocket: usize) bool {
    return sm == 121 and hidden == 5120 and intermediate == 17408 and layers == 64 and !moe and !mixed and !modelopt and !exl3 and pocket == 0;
}

test "FP8 prefill: bounded scratch and exact GB10 Qwen3.8 admission" {
    try std.testing.expectEqual(@as(usize, 0), try bytes(false, 0, 0));
    try std.testing.expectEqual(@as(usize, 2298368), try bytes(true, 128, 17408));
    try std.testing.expectEqual(@as(u32, 70656), prepareShared(max_columns));
    try std.testing.expectEqual(@as(u32, 37632), matmul_shared);
    for ([_]usize{ 1, 16, 17, 127, 128, 2048, 4096 }) |r| try validate(r, 5120);
    try std.testing.expectError(error.InvalidFp8PrefillShape, validate(4097, 5120));
    try std.testing.expectError(error.InvalidFp8PrefillShape, validate(1, 17409));
    try std.testing.expectError(error.InvalidFp8PrefillShape, validate(1, 65));
    try std.testing.expect(supported(121, 5120, 17408, 64, false, false, false, false, 0));
    try std.testing.expect(!supported(120, 5120, 17408, 64, false, false, false, false, 0));
    try std.testing.expect(!supported(121, 5120, 17408, 64, true, false, false, false, 0));
    try std.testing.expect(!supported(121, 5120, 17408, 64, false, true, false, false, 0));
    try std.testing.expect(!supported(121, 5120, 17408, 64, false, false, true, false, 0));
    try std.testing.expect(!supported(121, 5120, 17408, 64, false, false, false, true, 0));
    try std.testing.expect(!supported(121, 5120, 17408, 64, false, false, false, false, 64));
}
