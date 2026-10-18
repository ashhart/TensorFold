//! Explicit rotary math policy and independent long-position reference vectors, without Metal execution.
const std = @import("std");
const f = @import("src/families/flashnext/vision_contract.zig");
const golden = @import("flashnext_precise_golden.zig");

test "precise rotary: every live source role selects precise exp2, sine and cosine" {
    for ([_]f.Which{ .mrope, .target, .mtp, .pooled, .absolute }) |which| {
        const source = f.source(which);
        for ([_][]const u8{ "metal::precise::exp2(", "metal::precise::sin(", "metal::precise::cos(" }) |call| {
            try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, source, call));
        }
        for ([_][]const u8{ "metal::fast::sin(", "metal::fast::cos(", "metal::exp2(" }) |call| {
            try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, source, call));
        }
    }
}

test "precise rotary: FP64 reference agrees with independent long-position golden values" {
    const input: [128]u16 = @splat(0x3f80);
    const weights: [128]f32 = @splat(1);
    const rotary = f.table(23.253496170043945, 0);
    for (golden.cases) |case| {
        var output: [128]u16 = undefined;
        f.normalizedRotation(&input, @splat(case.position), &weights, 0, &rotary, .f64, .sequential, &output, null);
        try std.testing.expectEqual(case.bf16, output[case.index + @as(usize, if (case.upper) 32 else 0)]);
    }
}

test "precise rotary: conditional allowance accepts the full source capacity and refuses its end" {
    const input: [128]u16 = @splat(0x3f80);
    const weights: [128]f32 = @splat(1);
    var allow: [128]f64 = undefined;
    for ([_]f64{ 0, 4 }) |factor| {
        const rotary = f.table(23.253496170043945, factor);
        for ([_]i32{ 1048575, 1048576, 1048639 }) |position| {
            try f.normalizedBounds(&input, @splat(position), &weights, 1e-5, &rotary, &allow);
            for (allow) |v| try std.testing.expect(std.math.isFinite(v) and v >= 0);
        }
        try std.testing.expectError(error.InvalidVisionBoundInput, f.normalizedBounds(&input, @splat(1048640), &weights, 1e-5, &rotary, &allow));
    }
}
