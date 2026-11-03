//! Explicit raw BF16 diagnostic retains every whole-graph directional cell separately from frozen-store checks.
const std = @import("std");
const living = @import("living");
const native = @import("train_test.zig");
const a = std.testing.allocator;
pub fn run() !void {
    var f = try @import("learning_native_fixture.zig").Fixture.open();
    defer f.deinit();
    const r = try native.R.init(a, f.e);
    defer r.close() catch unreachable;
    const t = try native.T.init(a, std.testing.io, f.e, &r.pockets, r);
    defer t.deinit();
    t.capture_back = true;
    const tokens = [_]u32{ 3, 5, 7, 2, 9, 11, 4, 12, 19, 6, 8, 17, 2, 3, 15, 20, 7, 12, 9, 2 };
    const turn: living.lesson_route.Turn = .{ .decide = 0, .whole = true };
    var failures: usize = 0;
    for ([_]bool{ true, false }) |cut| {
        f.e.gpu.probability_bf16 = cut;
        for ([_]usize{ 0, 8, 17 }) |row| {
            @memset(t.delta, 0);
            t.nudge = .{ .layers = .{ 1, 0, 0 }, .rows = .{ row, 0, 0 } };
            _ = try t.step(&tokens, 17, .target, turn);
            var analytic: f64 = 0;
            for (t.pushAt(0, 0, 0), 0..) |g, i| analytic += g * @as(f32, if ((i * 17 + i / 13) % 5 < 2) -1 else 1);
            var square: f32 = 0;
            for (t.residual(1, row)) |v| {
                const x: f32 = @bitCast(@as(u32, v) << 16);
                square += x * x;
            }
            const initial: f32 = 0.03 * @sqrt(square) / @sqrt(@as(f32, @floatFromInt(f.e.c.hidden)));
            const baseline = try native.selectionHash(t);
            var crossings: usize = 0;
            var slopes: [6]f64 = undefined;
            for (&slopes, 0..) |*slope, k| {
                const epsilon = initial * std.math.pow(f32, 0.5, @floatFromInt(k));
                var losses: [2]f32 = undefined;
                for ([_]f32{ 1, -1 }, &losses) |sign, *loss| {
                    for (t.deltaAt(0, 0, 0), 0..) |*v, i| v.* = sign * epsilon * @as(f32, if ((i * 17 + i / 13) % 5 < 2) -1 else 1);
                    loss.* = (try t.step(&tokens, 17, .loss, turn)).loss;
                    crossings += @intFromBool(try native.selectionHash(t) != baseline);
                }
                slope.* = (@as(f64, losses[0]) - losses[1]) / (2 * @as(f64, epsilon));
            }
            var coarse: [4]f64 = slopes[0..4].*;
            std.mem.sort(f64, &coarse, {}, std.sort.asc(f64));
            const estimate = (coarse[1] + coarse[2]) * 0.5;
            const relative = @abs(analytic - estimate) / @max(@abs(analytic), 1e-30);
            std.debug.print("QWEN4_RAW cut={} row={d} epsilon={d} analytic={d} slopes={any} estimate={d} relative={d} selection_crossings={d}\n", .{ cut, row, initial, analytic, slopes, estimate, relative, crossings });
            if (!std.math.isFinite(relative) or relative > 0.15 or crossings != 0) failures += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}
