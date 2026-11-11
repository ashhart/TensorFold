//! Host-only tests for Qwen3.5 CUDA engine graph identities.
const std = @import("std");
const Shape = @import("cuda_shapes.zig").Shape;

test "dense read cost: hidden and slot variants have distinct graph keys" {
    var seen: [32]u32 = undefined;
    var n: usize = 0;
    for ([_]u8{ 1, 2, 4, 8, 16 }) |slots| {
        for ([_]bool{ false, true }) |hidden| {
            const shape: Shape = .{ .total = 1, .nseg = 1, .record = true, .prompt = false, .taps = true, .slots = slots, .hidden = hidden };
            for (seen[0..n]) |key| try std.testing.expect(key != shape.key());
            seen[n] = shape.key();
            n += 1;
        }
    }
}
