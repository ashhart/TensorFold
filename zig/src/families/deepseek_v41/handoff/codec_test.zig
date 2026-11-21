const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const s = @import("synth.zig");

test "e4m3: every code decodes as its fields say, and encoding a decoded value returns the code" {
    for (0..256) |k| {
        const code: u8 = @intCast(k);
        const v = c.e4m3(code);
        if (code & 0x7f == 0x7f) {
            try std.testing.expect(std.math.isNan(v));
            continue;
        }
        try std.testing.expectEqual(@as(f32, @floatCast(s.e4Ref(code))), v);
        if (code != 0x80) try std.testing.expectEqual(code, s.e4Encode(v)); // -0 encodes as +0 only through the sign of x
    }
    try std.testing.expectEqual(@as(f32, 448), c.e4m3(0x7e));
    try std.testing.expectEqual(@as(f32, 0x1p-9), c.e4m3(1));
}

test "e4m3: midpoints round to the even code and values past 448 saturate" {
    for (0..126) |k| {
        const mid: f32 = @floatCast((s.e4Ref(@intCast(k)) + s.e4Ref(@intCast(k + 1))) / 2);
        try std.testing.expectEqual(@as(u8, @intCast(if (k % 2 == 0) k else k + 1)), s.e4Encode(mid));
    }
    try std.testing.expectEqual(@as(u8, 0x7e), s.e4Encode(1e9));
    try std.testing.expectEqual(@as(u8, 0xfe), s.e4Encode(-1e9));
}

test "bf16: conversions round to nearest even and keep NaN a NaN" {
    try std.testing.expectEqual(@as(u16, 0x3f80), c.f32ToBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), c.f32ToBf16(@bitCast(@as(u32, 0x3f808000)))); // tie to even
    try std.testing.expectEqual(@as(u16, 0x3f82), c.f32ToBf16(@bitCast(@as(u32, 0x3f818000)))); // tie to even (up)
    try std.testing.expect(std.math.isNan(c.bf16ToF32(c.f32ToBf16(std.math.nan(f32)))));
    for ([_]u16{ 0, 1, 0x3f80, 0x7f7f, 0xff80, 0x8000 }) |x| try std.testing.expectEqual(x, c.f32ToBf16(c.bf16ToF32(x)));
}

test "comp rows decode to the byte fields' values, and a row on the fp8 grid round-trips bit for bit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var rng = s.Rng.init(7);
    for (0..200) |_| {
        var row: [512]u16 = undefined;
        s.genRow(&rng, &row);
        var rec: [g.cuda_record_bytes]u8 = undefined;
        s.encodeRecord(&row, &rec);
        var out: [512]u16 = undefined;
        try c.decodeRecord(&rec, &out);
        try std.testing.expectEqualSlices(u16, &s.recordRef(&rec), &out);
        // RoPE dims are bf16 verbatim
        try std.testing.expectEqualSlices(u16, row[448..], out[448..]);
        // the decoded row is on the grid: re-quantising it is a fixed point of the values (bytes may differ)
        var rec2: [g.cuda_record_bytes]u8 = undefined;
        s.encodeRecord(&out, &rec2);
        var again: [512]u16 = undefined;
        try c.decodeRecord(&rec2, &again);
        try std.testing.expectEqualSlices(u16, &out, &again);
        // and the NoPE error of the first quantisation is within half an e4m3 step of the tile's scale
        for (0..448) |d| {
            const scale = std.math.pow(f32, 2, @as(f32, @floatFromInt(@as(i32, rec[576 + d / 64]) - 127)));
            const err = @abs(c.bf16ToF32(out[d]) - c.bf16ToF32(row[d]));
            try std.testing.expect(err <= @max(@abs(c.bf16ToF32(row[d])) * 0x1p-4, scale * 0x1p-10) * 1.0001);
        }
    }
}

test "comp rows: a NaN code, the NaN scale byte or a non-finite RoPE value is refused" {
    var rec: [g.cuda_record_bytes]u8 = @splat(0);
    var out: [512]u16 = undefined;
    try c.decodeRecord(&rec, &out);
    rec[3] = 0x7f;
    try std.testing.expectError(error.BadKvRow, c.decodeRecord(&rec, &out));
    rec[3] = 0;
    rec[g.cuda_values_bytes + 2] = 255;
    try std.testing.expectError(error.BadKvRow, c.decodeRecord(&rec, &out));
    rec[g.cuda_values_bytes + 2] = 0;
    std.mem.writeInt(u16, rec[460..][0..2], 0x7f80, .little);
    try std.testing.expectError(error.BadKvRow, c.decodeRecord(&rec, &out));
}

test "scale bytes decode as powers of two across the byte's range, subnormal products rounding as bf16 does" {
    var rec: [g.cuda_record_bytes]u8 = @splat(0);
    var out: [512]u16 = undefined;
    rec[0] = 0x38; // 1.0 (e = 7, m = 0)
    for ([_]u8{ 0, 1, 100, 127, 200, 254 }) |sb| {
        rec[g.cuda_values_bytes] = sb;
        try c.decodeRecord(&rec, &out);
        const want = c.f32ToBf16(@floatCast(std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, sb) - 127)))));
        try std.testing.expectEqual(want, out[0]);
    }
}

test "index rows: fp8 codes times the fp32 scale, exactly; a non-finite scale or NaN code is refused" {
    var rng = s.Rng.init(11);
    for (0..100) |_| {
        var k: [128]u16 = undefined;
        s.genIndex(&rng, &k);
        var row: [g.cuda_index_fp8_bytes]u8 = undefined;
        const ref = s.encodeIndex(&k, &row);
        var out: [128]u16 = undefined;
        try c.decodeIndexFp8(&row, &out);
        try std.testing.expectEqualSlices(u16, &ref, &out);
    }
    var row: [g.cuda_index_fp8_bytes]u8 = @splat(0);
    var out: [128]u16 = undefined;
    std.mem.writeInt(u32, row[128..][0..4], 0x7f800000, .little);
    try std.testing.expectError(error.BadIndexRow, c.decodeIndexFp8(&row, &out));
    std.mem.writeInt(u32, row[128..][0..4], 0x3f800000, .little);
    row[5] = 0xff;
    try std.testing.expectError(error.BadIndexRow, c.decodeIndexFp8(&row, &out));
}

test "bf16 index rows and row stores are little-endian copies" {
    var bytes: [256]u8 = undefined;
    for (&bytes, 0..) |*b, i| b.* = @intCast(i);
    var row: [128]u16 = undefined;
    c.decodeIndexBf16(&bytes, &row);
    try std.testing.expectEqual(@as(u16, 0x0100), row[0]);
    var back: [256]u8 = undefined;
    c.storeRow(128, &row, &back);
    try std.testing.expectEqualSlices(u8, &bytes, &back);
    try std.testing.expectEqualSlices(u16, &row, &c.loadRow(128, &bytes));
}
