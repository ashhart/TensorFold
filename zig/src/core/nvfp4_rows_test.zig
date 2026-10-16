const std = @import("std");
const Table = @import("nvfp4_rows.zig").Table;
const Tensor = @import("safetensors.zig").Tensor;

fn tensor(dtype: @import("safetensors.zig").DType, rows: usize, cols: usize, bytes: []const u8) Tensor {
    return .{ .dtype = dtype, .rank = 2, .shape = .{ rows, cols, 1, 1, 1 }, .bytes = bytes };
}

fn rounded(value: f64) u16 {
    const v: f32 = @floatCast(value);
    if (std.math.isNan(v)) return 0x7fc0;
    const bits: u32 = @bitCast(v);
    const high = bits >> 16;
    const low = bits & 65535;
    return @intCast(high + @intFromBool(low > 32768 or (low == 32768 and high % 2 == 1)));
}

fn scale(byte: u8) f64 {
    const mag = byte & 127;
    if (mag == 127) return std.math.nan(f64);
    const e: i32 = @as(i32, mag >> 3) - 7;
    const value = if (mag < 8) @as(f64, @floatFromInt(mag)) / 512 else
        std.math.ldexp(1 + @as(f64, @floatFromInt(mag & 7)) / 8, e);
    return if (byte & 128 != 0) -value else value;
}

test "mapped NVFP4 rows preserve every nibble and scale encoding in reordered repeated gathers" {
    var codes: [256 * 8]u8 = undefined;
    var scales: [256]u8 = undefined;
    for (0..256) |row| {
        scales[row] = @intCast(row);
        for (0..8) |col| codes[row * 8 + col] = @intCast(col * 2 + ((col * 2 + 1) << 4));
    }
    const t = try Table.init(tensor(.u8, 256, 8, &codes), tensor(.f8_e4m3, 256, 1, &scales), 1.375);
    var ids: [258]usize = undefined;
    for (0..256) |i| ids[i] = 255 - i;
    ids[256] = 17;
    ids[257] = 17;
    var out: [258 * 16]u16 = undefined;
    try t.gather(&ids, &out);
    const values = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    for (ids, 0..) |id, row| for (0..16) |col| {
        const sign: f64 = if (col >= 8) -1 else 1;
        const want = rounded(sign * values[col % 8] * scale(@intCast(id)) * 1.375);
        const actual = out[row * 16 + col];
        if (id & 127 == 127) {
            try std.testing.expect(actual & 0x7fc0 == 0x7fc0);
        } else try std.testing.expectEqual(want, actual);
    };
    try t.gather(&.{}, &.{});
}

test "a bad row or destination refuses before writing and maps are never aliased" {
    var storage: [128]u8 align(8) = @splat(0x22);
    const scales = [_]u8{ 0x38, 0x40 };
    const t = try Table.init(tensor(.u8, 2, 8, storage[0..16]), tensor(.f8_e4m3, 2, 1, &scales), 1);
    var out: [32]u16 = @splat(0xbeef);
    try std.testing.expectError(error.BadNvfp4Rows, t.gather(&.{ 0, 2 }, &out));
    for (out) |v| try std.testing.expectEqual(@as(u16, 0xbeef), v);
    try std.testing.expectError(error.BadNvfp4Rows, t.gather(&.{0}, &out));
    try std.testing.expectError(error.AliasedNvfp4Rows, t.gather(&.{0}, std.mem.bytesAsSlice(u16, storage[0..32])));
    var ids: [4]usize = .{ 0, 0, 0, 0 };
    try std.testing.expectError(error.AliasedNvfp4Rows, t.gather(ids[0..1], std.mem.bytesAsSlice(u16, std.mem.sliceAsBytes(&ids))));
}

test "invalid table dtype shape extent and scale cannot create a borrowed view" {
    const codes: [16]u8 = @splat(0);
    const scales = [_]u8{0x38};
    const w = tensor(.u8, 1, 16, &codes);
    const s = tensor(.f8_e4m3, 1, 1, &scales);
    try std.testing.expectError(error.BadNvfp4Table, Table.init(w, s, 1));
    try std.testing.expectError(error.BadNvfp4Table, Table.init(tensor(.u8, 1, 7, codes[0..7]), s, 1));
    try std.testing.expectError(error.BadNvfp4Table, Table.init(tensor(.bf16, 1, 8, &codes), s, 1));
    try std.testing.expectError(error.BadNvfp4Table, Table.init(tensor(.u8, 1, 8, &codes), s, 1));
    try std.testing.expectError(error.BadNvfp4Table, Table.init(tensor(.u8, 1, 8, codes[0..8]), s, std.math.inf(f32)));
    try std.testing.expectError(error.BadNvfp4Table, Table.init(tensor(.u8, std.math.maxInt(usize), 8, &codes), s, 1));
}
