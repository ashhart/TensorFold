//! Explicit whole-graph raw BF16 diagnostic preserves every rule-v2 failure outside the local mathematical checks.
const std = @import("std");
const n = @import("src/families/gemma4/learning_numeric.zig");
const T = @import("src/families/gemma4/learning_train.zig").Trainer(n.E);
test "Gemma4 raw directional: unchanged rule v2 over all six native BF16 seam cells" {
    var f = try @import("src/families/gemma4/learning_fixture.zig").Fixture.open();
    defer f.deinit();
    const t = try T.init(std.testing.allocator, std.testing.io, f.e, &f.e.r.pockets, f.e);
    defer t.deinit();
    const tokens = [_]u32{ 3, 5, 7, 2, 9 };
    const turn: @import("living").lesson_route.Turn = .{ .decide = 0, .whole = true };
    var failures: usize = 0;
    for ([_]usize{ 0, 5 }) |layer| for ([_]usize{ 0, 1, 2 }) |row| {
        @memset(t.delta, 0);
        t.nudge = .{ .layers = .{ @as(u64, 1) << @intCast(layer), 0, 0 }, .rows = .{ row, 0, 0 } };
        _ = try t.step(&tokens, 3, .target, turn);
        var analytic: f64 = 0;
        for (t.pushAt(0, 0, layer), 0..) |g, i| analytic += g * @as(f32, if ((i * 17 + i / 13) % 5 < 2) -1 else 1);
        var square: f32 = 0;
        for (t.residual(layer + 1, row)) |v| square += n.f(v) * n.f(v);
        const initial: f32 = 0.03 * @sqrt(square) / 8;
        var slopes: [6]f64 = undefined;
        for (&slopes, 0..) |*slope, k| {
            const epsilon = initial * std.math.pow(f32, 0.5, @floatFromInt(k));
            var losses: [2]f32 = undefined;
            for ([_]f32{ 1, -1 }, &losses) |sign, *loss| {
                for (t.deltaAt(0, 0, layer), 0..) |*v, i| v.* = sign * epsilon * @as(f32, if ((i * 17 + i / 13) % 5 < 2) -1 else 1);
                loss.* = (try t.step(&tokens, 3, .loss, turn)).loss;
            }
            slope.* = (@as(f64, losses[0]) - losses[1]) / (2 * @as(f64, epsilon));
        }
        var coarse: [4]f64 = slopes[0..4].*;
        std.mem.sort(f64, &coarse, {}, std.sort.asc(f64));
        const estimate = (coarse[1] + coarse[2]) * 0.5;
        const relative = @abs(analytic - estimate) / @max(@abs(analytic), 1e-30);
        std.debug.print("GEMMA4_RAW layer={d} row={d} epsilon={d} analytic={d} slopes={any} estimate={d} relative={d}\n", .{ layer, row, initial, analytic, slopes, estimate, relative });
        if (!std.math.isFinite(relative) or relative > 0.15) failures += 1;
    };
    try std.testing.expectEqual(@as(usize, 0), failures);
}
