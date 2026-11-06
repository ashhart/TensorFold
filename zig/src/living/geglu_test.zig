//! GeGLU fits use the same two input rows and stored grid as SwiGLU, with Gemma's tanh activation.
const std = @import("std");
const neuron = @import("neuron.zig");
const families = @import("families.zig");
const round = @import("pocket_round.zig");
test "GeGLU neurons use the independent PyTorch tanh formula and both shared families" {
    for ([_]f32{ -9, -2, -0.5, 0, 0.5, 1, 4, 9 }) |x| {
        const z: @TypeOf(@as(f64, 0)) = x;
        const want = 0.5 * z * (1 + std.math.tanh(@sqrt(@as(f64, 2) / std.math.pi) * (z + 0.044715 * z * z * z)));
        try std.testing.expectApproxEqAbs(want, @as(f64, neuron.gelu(x)), 2e-7 + 3e-6 * @abs(want));
        const n = families.gemma4.neuron;
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)) * x / n.sharp, neuron.act(n, x, x), 2e-7 + 3e-6 * @abs(@as(f32, @floatCast(want)) * x / n.sharp));
    }
    for ([_]@import("family.zig").Family{ families.gemma4, families.gemma4_unified }) |f| {
        try f.check();
        try std.testing.expectEqual(@as(usize, 2), f.neuron.inputs());
        try std.testing.expectEqualStrings("language_model.model.layers.", f.names.layers);
    }
}
test "GeGLU rounding validates both inputs and uses the existing stable writer grid" {
    const f = families.gemma4;
    const size = 64 * 64;
    const all = try std.testing.allocator.alloc(f32, 3 * size);
    defer std.testing.allocator.free(all);
    for (all, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) * 0.003;
    const g = all[0..size];
    const u = all[size .. 2 * size];
    const d = all[2 * size ..];
    try std.testing.expectError(error.BadPocket, round.layer(f, std.testing.allocator, 64, 64, g[0 .. size - 1], u, d));
    try round.layer(f, std.testing.allocator, 64, 64, g, u, d);
    const stored = try std.testing.allocator.dupe(f32, all);
    defer std.testing.allocator.free(stored);
    try round.layer(f, std.testing.allocator, 64, 64, g, u, d);
    try std.testing.expectEqualSlices(f32, stored, all);
}
