//! Host-only ledger separates raw BF16 perturbations from frozen-store binary64 derivatives without changing the gate.
const std = @import("std");
const core = @import("core");
const living = @import("living");
const native = @import("train_test.zig");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const reference = @import("learning_reference.zig");
const a = std.testing.allocator;
const pocket_name = "model.layers.0.mlp.pocket_output";
fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}
fn median(slopes: [6]f64) f64 {
    var coarse: [4]f64 = slopes[0..4].*;
    std.mem.sort(f64, &coarse, {}, std.sort.asc(f64));
    return (coarse[1] + coarse[2]) * 0.5;
}
fn relative(x: f64, y: f64) f64 {
    return @abs(x - y) / @max(@abs(x), 1e-30);
}
fn loss64(t: *native.T) !f64 {
    const logits = n.view(u16, try t.tape.get("head", t.rows));
    const width = t.e.c.vocab;
    var loss: f64 = 0;
    for (t.seq.start - 1..t.rows) |row| {
        const scores = logits[row * width ..][0..width];
        var maximum: f64 = -std.math.inf(f64);
        for (scores) |v| maximum = @max(maximum, bf(v));
        var total: f64 = 0;
        for (scores) |v| total += @exp(bf(v) - maximum);
        loss += maximum + @log(total) - bf(scores[t.seq.ids[row + 1]]);
    }
    return loss / @as(f64, @floatFromInt(t.seq.ids.len - t.seq.start));
}
const Ledger = struct {
    t: *native.T,
    cut: bool,
    row: usize,
    k: usize = 0,
    sign: i32 = 0,
    first: bool = false,
    fn emit(ctx: *anyopaque, name: []const u8, values: []const f64) !void {
        const o: *@This() = @ptrCast(@alignCast(ctx));
        const raw = n.view(u16, try o.t.tape.get(name, o.t.rows));
        try std.testing.expectEqual(raw.len, values.len);
        var maximum: f64 = 0;
        var at: usize = 0;
        var changed: usize = 0;
        for (raw, values, 0..) |got, want, i| {
            const gap = @abs(bf(got) - want);
            if (gap > 1e-12) changed += 1;
            if (gap > maximum) {
                maximum = gap;
                at = i;
            }
        }
        if (!o.first and changed != 0) {
            o.first = true;
            const width = values.len / o.t.rows;
            const rounded = core.quant.convert.bf16(@floatCast(values[at]));
            std.debug.print("QWEN4_FIRST cut={} row={d} k={d} sign={d} tensor={s} differing={d}/{d} at_row={d} col={d} raw={d} frozen64={d} abs={d} bf16_round_matches={}\n", .{ o.cut, o.row, o.k, o.sign, name, changed, values.len, at / width, at % width, bf(raw[at]), values[at], maximum, raw[at] == rounded });
            try std.testing.expectEqualStrings(pocket_name, name);
            // The first difference must be precisely the native nudge's documented BF16 store.
            for (raw, values) |got, want| try std.testing.expectEqual(core.quant.convert.bf16(@floatCast(want)), got);
        }
    }
};
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
    const direction = try a.alloc(f32, f.e.c.hidden);
    defer a.free(direction);
    for (direction, 0..) |*v, i| v.* = if ((i * 17 + i / 13) % 5 < 2) -1 else 1;
    for ([_]bool{ true, false }) |cut| {
        f.e.gpu.probability_bf16 = cut;
        for ([_]usize{ 0, 8, 17 }) |row| {
            @memset(t.delta, 0);
            t.nudge = .{ .layers = .{ 1, 0, 0 }, .rows = .{ row, 0, 0 } };
            const base = try t.step(&tokens, 17, .target, turn);
            const base_loss64 = try loss64(t);
            const hash = try native.selectionHash(t);
            const raw_base = try a.dupe(u16, n.view(u16, try t.tape.get(pocket_name, t.rows)));
            defer a.free(raw_base);
            var analytic: f64 = 0;
            for (t.pushAt(0, 0, 0), direction) |g, d| analytic += g * d;
            var point: reference.Point(native.T) = .{ .a = a, .t = t };
            defer point.deinit();
            try std.testing.expectApproxEqAbs(base_loss64, try point.loss(0, row, direction, 0), 1e-12);
            for ([_]f64{ 0.0001, 0.00005 }) |epsilon| {
                const hi = try point.loss(0, row, direction, epsilon);
                const lo = try point.loss(0, row, direction, -epsilon);
                const slope = (hi - lo) / (2 * epsilon);
                std.debug.print("QWEN4_LOCAL cut={} row={d} epsilon={d} analytic={d} frozen64={d} relative={d}\n", .{ cut, row, epsilon, analytic, slope, relative(analytic, slope) });
                try @import("learning_fixture.zig").near(@floatCast(analytic), slope);
            }
            var square: f32 = 0;
            for (t.residual(1, row)) |v| {
                const x: f32 = @floatCast(bf(v));
                square += x * x;
            }
            const initial: f32 = 0.03 * @sqrt(square) / @sqrt(@as(f32, @floatFromInt(f.e.c.hidden)));
            var raw_slopes: [6]f64 = undefined;
            var raw64_slopes: [6]f64 = undefined;
            var frozen_slopes: [6]f64 = undefined;
            var crossings: usize = 0;
            for (&raw_slopes, &raw64_slopes, &frozen_slopes, 0..) |*raw_slope, *raw64_slope, *frozen_slope, k| {
                const epsilon = initial * std.math.pow(f32, 0.5, @floatFromInt(k));
                var raw_losses: [2]f32 = undefined;
                var raw64_losses: [2]f64 = undefined;
                var frozen_losses: [2]f64 = undefined;
                for ([_]f32{ 1, -1 }, &raw_losses, &raw64_losses, &frozen_losses) |sign, *raw, *raw64, *frozen| {
                    for (t.deltaAt(0, 0, 0), direction) |*v, d| v.* = sign * epsilon * d;
                    raw.* = (try t.step(&tokens, 17, .loss, turn)).loss;
                    raw64.* = try loss64(t);
                    crossings += @intFromBool(try native.selectionHash(t) != hash);
                    var ledger: Ledger = .{ .t = t, .cut = cut, .row = row, .k = k, .sign = @intFromFloat(sign) };
                    point.observer = .{ .ctx = &ledger, .emit = Ledger.emit };
                    frozen.* = try point.loss(0, row, direction, sign * @as(f64, epsilon));
                    point.observer = null;
                    try std.testing.expect(ledger.first);
                }
                raw_slope.* = (@as(f64, raw_losses[0]) - raw_losses[1]) / (2 * @as(f64, epsilon));
                raw64_slope.* = (raw64_losses[0] - raw64_losses[1]) / (2 * @as(f64, epsilon));
                frozen_slope.* = (frozen_losses[0] - frozen_losses[1]) / (2 * @as(f64, epsilon));
            }
            std.debug.print("QWEN4_ROUNDING cut={} row={d} analytic={d} raw={any} raw64_loss={any} frozen64={any} raw_estimate={d} raw_relative={d} frozen_estimate={d} frozen_relative={d} selection_crossings={d}\n", .{ cut, row, analytic, raw_slopes, raw64_slopes, frozen_slopes, median(raw_slopes), relative(analytic, median(raw_slopes)), median(frozen_slopes), relative(analytic, median(frozen_slopes)), crossings });
            try std.testing.expectEqual(@as(usize, 0), crossings);
            // A strictly positive sub-half-ULP perturbation must leave the entire raw forward pass unchanged.
            var plateau: f32 = std.math.inf(f32);
            for (raw_base[row * direction.len ..][0..direction.len]) |v| {
                const magnitude = v & 0x7fff;
                const spacing: f32 = @floatCast(bf(magnitude + 1) - bf(magnitude));
                plateau = @min(plateau, spacing * 0.125);
            }
            try std.testing.expect(plateau > 0);
            for ([_]f32{ 1, -1 }) |sign| {
                for (t.deltaAt(0, 0, 0), direction) |*v, d| v.* = sign * plateau * d;
                const out = try t.step(&tokens, 17, .loss, turn);
                try std.testing.expectEqualSlices(u16, raw_base, n.view(u16, try t.tape.get(pocket_name, t.rows)));
                try std.testing.expectEqual(base.loss, out.loss);
                try std.testing.expectEqual(base_loss64, try loss64(t));
                try std.testing.expectEqual(hash, try native.selectionHash(t));
            }
            std.debug.print("QWEN4_PLATEAU cut={} row={d} epsilon={d} loss={d} loss_delta=0 raw_slope=0 analytic={d} pocket_unchanged={}\n", .{ cut, row, plateau, base.loss, analytic, true });
        }
    }
}
