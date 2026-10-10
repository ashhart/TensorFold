//! Prompt convolution and packed-state contracts refuse unsupported shape or storage before dispatch.
const std = @import("std");

pub const Conv = extern struct { rows: u32, channels: u32, taps: u32, streams: u32 };
pub const Recur = extern struct { rows: u32, nk: u32, nv: u32, dv: u32, streams: u32, pad0: u32 = 0, pad1: u32 = 0, pad2: u32 = 0 };

pub fn checkConv(a: Conv) !void {
    if (a.rows == 0 or a.rows > 2048 or a.channels == 0 or a.channels > 65536 or a.taps < 2 or a.taps > 8 or a.streams == 0 or a.streams > 64) return error.BadPromptConv;
    if (@as(u64, a.rows) * a.channels * a.streams > std.math.maxInt(u32)) return error.BadPromptConv;
}

pub fn checkRecur(a: Recur) !void {
    if (a.rows == 0 or a.rows > 2048 or a.nk == 0 or a.nk > 128 or a.nv == 0 or a.nv > 128 or a.nv % a.nk != 0 or a.dv < 8 or a.dv > 256 or a.dv % 8 != 0 or a.streams == 0 or a.streams > 64 or a.pad0 != 0 or a.pad1 != 0 or a.pad2 != 0) return error.BadPromptRecurrence;
}

pub fn canonical(partials: [32]f32) f32 {
    @setFloatMode(.strict);
    var values = partials;
    var stride: usize = 1;
    while (stride < values.len) : (stride *= 2) {
        var at: usize = 0;
        while (at < values.len) : (at += 2 * stride) values[at] += values[at + stride];
    }
    return values[0];
}

pub fn packedFold(partials: [32]f32) f32 {
    @setFloatMode(.strict);
    var values = partials;
    var roots: [4]f32 = undefined;
    for (0..4) |lane| {
        var stride: usize = 1;
        while (stride < 8) : (stride *= 2) {
            var at: usize = 0;
            while (at < 8) : (at += 2 * stride) values[lane * 8 + at] += values[lane * 8 + at + stride];
        }
        roots[lane] = values[lane * 8];
    }
    return (roots[0] + roots[1]) + (roots[2] + roots[3]);
}

test "packed local folds preserve the canonical FP32 tree without reassociation" {
    for (0..64) |seed| {
        var values: [32]f32 = undefined;
        for (&values, 0..) |*v, index| {
            const bits: u32 = @intCast((seed * 7919 + index * 65537) & 0x7fffff);
            v.* = @bitCast((if ((seed + index) % 3 == 0) @as(u32, 1 << 31) else 0) | (120 + @as(u32, @intCast(index % 16))) << 23 | bits);
        }
        try std.testing.expectEqual(@as(u32, @bitCast(canonical(values))), @as(u32, @bitCast(packedFold(values))));
    }
}

test "prompt recurrence and convolution ABI and row bounds are explicit" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Conv));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Recur));
    try checkConv(.{ .rows = 35, .channels = 10240, .taps = 4, .streams = 1 });
    try checkRecur(.{ .rows = 128, .nk = 16, .nv = 48, .dv = 128, .streams = 1 });
    try std.testing.expectError(error.BadPromptRecurrence, checkRecur(.{ .rows = 128, .nk = 16, .nv = 47, .dv = 128, .streams = 1 }));
    try std.testing.expectError(error.BadPromptConv, checkConv(.{ .rows = 2049, .channels = 10240, .taps = 4, .streams = 1 }));
}
