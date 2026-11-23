//! Fake GPU time exposes width-local slow stretches without changing the target arithmetic.
const std = @import("std");
const rule = @import("cost_rule.zig");
const Timer = struct {
    calls: usize = 0,
    total: f64 = 0,
    slow: bool = false,
    pub fn time(t: *@This(), i: usize) !f64 {
        const noisy = t.slow and t.calls >= 31 * 8 and t.calls < 32 * 8;
        const value = truth(i) * (if (noisy) @as(f64, 1.08) else 1);
        t.calls += 1;
        t.total += value;
        return value;
    }
};
fn truth(i: usize) f64 {
    const width = i + 1;
    return 10 + 0.5 * @as(f64, @floatFromInt(width)) + @as(f64, if (width >= 16) 2 else 0) +
        @as(f64, if (width >= 32) 3 else 0) + @as(f64, if (width >= 48) 4 else 0);
}

test "Metal window prices: interleaved minima survive one whole-width slow stretch" {
    var timer: Timer = .{ .slow = true };
    var ms: [64]f64 = undefined;
    try rule.measure(&timer, &timer, &ms, null, false);
    for (ms, 0..) |m, i| try std.testing.expectApproxEqRel(truth(i), m, 0.01);
    var baseline: f64 = 0;
    for (0..64) |i| baseline += 8 * truth(i);
    try std.testing.expect(timer.total <= baseline * 1.2);
}

test "Metal window prices: a high outlier below the dip threshold is retimed and real tile steps survive" {
    var timer: Timer = .{};
    var ms: [64]f64 = undefined;
    for (&ms, 0..) |*m, i| m.* = truth(i);
    ms[8] *= 1.04;
    try rule.smooth(&timer, &ms);
    for (ms, 0..) |m, i| try std.testing.expectApproxEqAbs(truth(i), m, 1e-9);
    for ([_]usize{ 15, 31, 47 }) |i| try std.testing.expect(ms[i] > ms[i - 1] + 1);
    try std.testing.expect(timer.calls <= ms.len);
}
