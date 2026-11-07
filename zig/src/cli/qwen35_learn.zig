//! Owned 27B trainer qualification. Credit @ZackSample #911 for the directional finite-difference method.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const train = qwen.train;
const Trainer = train.Trainer;
const D = 5120;
const diagnostic = @import("qwen35_slope.zig");
const turn: train.Turn = .{ .decide = 0, .bits = .empty, .whole = true };
fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}
fn dot(a: []const f32, b: []const f32) f64 {
    var sum: f64 = 0;
    for (a, b) |x, y| sum += @as(f64, x) * y;
    return sum;
}
fn slope(t: *Trainer, sq: train.Seq, layer: usize, dir: []const f32, step: f64) !f64 {
    var fit: diagnostic.Fit = .{};
    defer @memset(t.deltaAt(0, 0, layer), 0);
    for (1..7) |m| for ([_]f64{ -1, 1 }) |sign| {
        const x = sign * @as(f64, @floatFromInt(m)) * step;
        for (t.deltaAt(0, 0, layer), dir) |*out, v| out.* = @floatCast(x * v);
        var result: [1]train.Result = undefined;
        try t.pass(&.{sq}, .loss, turn, &result);
        if (!std.math.isFinite(result[0].loss)) return error.NonfiniteLoss;
        fit.add(x, result[0].loss);
    };
    return fit.slope();
}
fn readIds(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
    defer a.free(bytes);
    const doc = try std.json.parseFromSlice([]u32, a, bytes, .{ .allocate = .alloc_always });
    defer doc.deinit();
    return a.dupe(u32, doc.value);
}
/// A separate serving pass computes the loss with an FP64 host log-sum-exp and no trainer hooks.
fn servingLoss(a: std.mem.Allocator, e: *qwen.Engine, sq: train.Seq) !f64 {
    var cache = try qwen.engine.Cache.init(a, e, sq.ids.len + 1);
    defer cache.deinit(a);
    var total: f64 = 0;
    for (sq.ids[0 .. sq.ids.len - 1], 0..) |id, row| {
        try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, &.{id}, .logits);
        if (row + 1 < sq.start) continue;
        const logits = e.rowLogits(0);
        var top = -std.math.inf(f64);
        for (logits) |v| top = @max(top, bf(v));
        var sum: f64 = 0;
        for (logits) |v| sum += @exp(bf(v) - top);
        total += top + @log(sum) - bf(logits[sq.ids[row + 1]]);
    }
    return total / @as(f64, @floatFromInt(sq.ids.len - sq.start));
}
fn cell(t: *Trainer, sq: train.Seq, layer: usize, row: usize) !void {
    var result: [1]train.Result = undefined;
    try t.pass(&.{sq}, .target, turn, &result);
    var push: [D]f32 = undefined;
    @memcpy(&push, t.pushAt(0, 0, layer));
    const length = @sqrt(dot(&push, &push));
    if (!std.math.isFinite(length) or length <= 0) return error.BadGradient;
    var dir: [D]f32 = undefined;
    for (&dir, push) |*out, v| out.* = @floatCast(v / length);
    var scratch: [D]f64 = undefined;
    const residual = try diagnostic.residual(t.residual(layer + 1, row), &scratch);
    std.debug.print("RESULT residual layer {d} row {d}: norm {e:.8}, largest {e:.8}, trimmed_rms {e:.8}, gradient {e:.8}\n", .{ layer, row, residual.norm, residual.largest, residual.trimmed_rms, length });
    var slopes: [4]f64 = undefined;
    for ([_]f64{ 0.001, 0.002, 0.004, 0.008 }, &slopes) |fraction, *fd| {
        std.debug.print("RESULT step layer {d} row {d}: fraction {e:.8}, norm_h {e:.8}, trimmed_h {e:.8}\n", .{ layer, row, fraction, residual.normStep(fraction), residual.step(fraction) });
        fd.* = try slope(t, sq, layer, &dir, residual.step(fraction));
    }
    const fractions = [_]f64{ 0.001, 0.002, 0.004, 0.008 };
    var norm_slopes: [4]f64 = undefined;
    for (fractions, &norm_slopes) |fraction, *fd|
        fd.* = try slope(t, sq, layer, &dir, residual.normStep(fraction));
    for (fractions, norm_slopes, 0..) |fraction, fd, i| {
        const fine = if (i == 0) 0 else i - 1;
        const extrapolated = diagnostic.richardson(norm_slopes[fine], norm_slopes[fine + 1]);
        std.debug.print("RESULT norm-slope layer {d} row {d}: h {e:.8}, slope {e:.8}, Richardson {e:.8}, Richardson_fine_h {e:.8}, relative {d:.6}\n", .{ layer, row, residual.normStep(fraction), fd, extrapolated, residual.normStep(fractions[fine]), @abs(fd - length) / length });
    }
    var errors: [3]f64 = undefined;
    for (&errors, 1..) |*relative, i| {
        relative.* = @abs(slopes[i] - length) / length;
        std.debug.print("RESULT slope layer {d} row {d}: h {e:.8}, slope {e:.8}, Richardson {e:.8}, relative {d:.6}\n", .{ layer, row, residual.step(@as(f64, 0.001) * @exp2(@as(f64, @floatFromInt(i)))), slopes[i], diagnostic.richardson(slopes[i - 1], slopes[i]), relative.* });
        if (!std.math.isFinite(relative.*)) return error.NonfiniteGradient;
    }
    std.mem.sort(f64, &errors, {}, std.sort.asc(f64));
    std.debug.print("RESULT owned 27B layer {d} row {d}: median {d:.5}, worst {d:.5}\n", .{ layer, row, errors[1], errors[2] });
    if (errors[1] >= 0.15 or errors[2] >= 0.15) return error.GradientMismatch;
}
pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.debug.print("usage: tf-qwen35-learn MODEL IDS_JSON_ARRAY\n", .{});
        return 2;
    }
    const ids = try readIds(a, io, args[2]);
    defer a.free(ids);
    if (ids.len < 24 or ids.len > train.max_rows + 1) return error.BadExample;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try qwen.Engine.init(a, io, &ctx, args[1], false, null);
    defer e.deinit();
    if (e.pockets.any() or e.g.pocket != 0) return error.FreshUntaughtCheckpointRequired;
    const t = try Trainer.init(a, io, e);
    defer t.deinit();
    var sq: train.Seq = .{ .ids = ids, .start = ids.len - 6, .decide = ids.len - 7 };
    const serving = try servingLoss(a, e, sq);
    var result: [1]train.Result = undefined;
    try t.pass(&.{sq}, .loss, turn, &result);
    if (!std.math.isFinite(result[0].loss) or @abs(serving - result[0].loss) > 1e-3)
        return error.TrainerServingMismatch;
    var together: [2]train.Result = undefined;
    try t.pass(&.{ sq, sq }, .loss, turn, &together);
    for (together) |p| if (@abs(p.loss - serving) > 1e-3) return error.PackedLossMismatch;
    std.debug.print("PASS owned serving/trainer/packed loss {d:.8}\n", .{serving});
    const rows = [_]usize{ 0, sq.start - 5, sq.start - 3, sq.start - 1 };
    const layers = [_]usize{ train.keepLow(e.g.layers), 23, 31, e.g.layers - 2 };
    var checks: usize = 0;
    var failures: usize = 0;
    for (rows) |row| for (layers) |layer| {
        sq.nudge = .{ .layers = .{ @as(u64, 1) << @intCast(layer), 0, 0 }, .rows = .{ row, 0, 0 } };
        cell(t, sq, layer, row) catch |err| {
            failures += 1;
            std.debug.print("FAIL owned 27B layer {d} row {d}: {s}\n", .{ layer, row, @errorName(err) });
        };
        checks += 1;
    };
    std.debug.print("RESULT {d} owned directional checks, {d} failures\n", .{ checks, failures });
    if (failures != 0) return error.GradientMismatch;
    std.debug.print("PASS configured trimmed-scale directional checks at relative-error threshold 0.15\n", .{});
    return 0;
}
