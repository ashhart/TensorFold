//! Directed packed inputs and names for exact replacement panels; no device work here.
const std = @import("std");
const cuda = @import("cuda");
const c = @import("core").contract.spec;
pub const Case = enum { natural, scales, ties };
pub const direct_shapes = [_]c.Shape{ .{ .rows = 1, .n = 129, .k = 4096 }, .{ .rows = 15, .n = 129, .k = 4096 } };

pub fn production() [32]c.Shape {
    var out: [32]c.Shape = undefined;
    for (0..16) |i| {
        out[i] = .{ .rows = i + 1, .n = 17408, .k = 5120 };
        out[16 + i] = .{ .rows = i + 1, .n = 5120, .k = 17408 };
    }
    return out;
}

pub fn pinName(out: cuda.qmmf.Out, clusters: bool, sk: u32) []const u8 {
    const index: usize = switch (sk) {
        1 => 0,
        2 => 1,
        4 => 3,
        8 => 5,
        else => unreachable,
    };
    const names = [2][7][]const u8{
        .{ "cuda qmmf exact tile bf16 SK1 plain", "cuda qmmf exact tile bf16 SK2 reduce", "cuda qmmf exact tile bf16 SK2 cluster", "cuda qmmf exact tile bf16 SK4 reduce", "cuda qmmf exact tile bf16 SK4 cluster", "cuda qmmf exact tile bf16 SK8 reduce", "cuda qmmf exact tile bf16 SK8 cluster" },
        .{ "cuda qmmf exact tile fp32 SK1 plain", "cuda qmmf exact tile fp32 SK2 reduce", "cuda qmmf exact tile fp32 SK2 cluster", "cuda qmmf exact tile fp32 SK4 reduce", "cuda qmmf exact tile fp32 SK4 cluster", "cuda qmmf exact tile fp32 SK8 reduce", "cuda qmmf exact tile fp32 SK8 cluster" },
    };
    return names[@backingInt(out)][index + @intFromBool(sk != 1 and clusters)];
}

pub fn directName(out: cuda.qmmf.Out, clusters: bool, which: Case) []const u8 {
    const names = [2][2][2][]const u8{
        .{ .{ "cuda qmmf exact tile bf16 scales reduce", "cuda qmmf exact tile bf16 scales cluster" }, .{ "cuda qmmf exact tile fp32 scales reduce", "cuda qmmf exact tile fp32 scales cluster" } },
        .{ .{ "cuda qmmf exact tile bf16 RN ties reduce", "cuda qmmf exact tile bf16 RN ties cluster" }, .{ "cuda qmmf exact tile fp32 RN ties reduce", "cuda qmmf exact tile fp32 RN ties cluster" } },
    };
    return names[@intFromBool(which == .ties)][@backingInt(out)][@intFromBool(clusters)];
}

pub fn fill(which: Case, x: []u16, codes: []u8, scales: []u8, rows: usize, n: usize, k: usize) void {
    if (which == .natural) return;
    @memset(x, 0);
    @memset(codes, 0x22);
    @memset(scales, 0x38);
    for (0..rows) |r| {
        x[r * k] = 0x3f80;
        if (which == .ties) x[r * k + 16] = 0x3b80 else {
            x[r * k + 2] = 1;
            x[r * k + 3] = 0x8001;
        }
    }
    const special = [_]u8{ 0, 0x80, 1, 0x7f, 0xff };
    for (0..n) |col| {
        if (which == .scales) {
            @memset(scales[col * (k / 16) ..][0 .. k / 16], special[col % special.len]);
        } else if (col % 2 != 0) scales[col * (k / 16) + 1] = 0x44;
    }
}

test "qmmf exact panels: directed scale classes and RN-even ties are planted before packing" {
    const t = std.testing;
    var x: [64]u16 = undefined;
    var codes: [8 * 64 / 2]u8 = undefined;
    var scales: [8 * 64 / 16]u8 = undefined;
    fill(.scales, &x, &codes, &scales, 1, 8, 64);
    for ([_]u8{ 0, 0x80, 1, 0x7f, 0xff }, 0..) |v, i| try t.expectEqual(v, scales[i * 4]);
    const fp8 = @import("core").quant.fp8;
    try t.expect(std.math.isNan(fp8.e4m3(scales[12])) and std.math.isNan(fp8.e4m3(scales[16])));
    try t.expectEqual(@as(f64, 1.0 / 512.0), fp8.e4m3(scales[8]));
    fill(.ties, &x, &codes, &scales, 1, 8, 64);
    for (0..8) |col| {
        const sum: f32 = c.fromBf16(x[0]) + c.fromBf16(x[16]) * @as(f32, @floatCast(fp8.e4m3(scales[col * 4 + 1])));
        try t.expectEqual(@as(f32, if (col % 2 == 0) 1.00390625 else 1.01171875), sum);
        try t.expectEqual(@as(u16, if (col % 2 == 0) 0x3f80 else 0x3f82), c.toBf16(sum));
    }
}
