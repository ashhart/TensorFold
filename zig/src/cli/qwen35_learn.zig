//! Owned 27B trainer qualification. Credit @ZackSample #911 for the directional finite-difference method.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const train = qwen.train;
const Trainer = train.Trainer;
const D = 5120;
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
    var num: f64 = 0;
    var den: f64 = 0;
    defer @memset(t.deltaAt(0, 0, layer), 0);
    for (1..7) |m| for ([_]f64{ -1, 1 }) |sign| {
        const x = sign * @as(f64, @floatFromInt(m)) * step;
        for (t.deltaAt(0, 0, layer), dir) |*out, v| out.* = @floatCast(x * v);
        var result: [1]train.Result = undefined;
        try t.pass(&.{sq}, .loss, turn, &result);
        if (!std.math.isFinite(result[0].loss)) return error.NonfiniteLoss;
        num += x * result[0].loss;
        den += x * x;
    };
    return num / den;
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
    const e = try qwen.Engine.init(a, io, &ctx, args[1], false);
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
    const rows = [_]usize{ 0, sq.start - 5, sq.start - 1 };
    const layers = [_]usize{ train.keepLow(e.g.layers), 23, 31, e.g.layers - 2 };
    var push: [D]f32 = undefined;
    var dir: [D]f32 = undefined;
    var checks: usize = 0;
    for (rows) |row| for (layers) |layer| {
        sq.nudge = .{ .layers = .{ @as(u64, 1) << @intCast(layer), 0, 0 }, .rows = .{ row, 0, 0 } };
        try t.pass(&.{sq}, .target, turn, &result);
        @memcpy(&push, t.pushAt(0, 0, layer));
        const length = @sqrt(dot(&push, &push));
        if (!std.math.isFinite(length) or length <= 0) return error.BadGradient;
        for (&dir, push) |*out, v| out.* = @floatCast(v / length);
        var size: f64 = 0;
        for (t.residual(layer + 1, row)) |v| size += bf(v) * bf(v);
        if (!std.math.isFinite(size) or size <= 0) return error.BadResidual;
        var errors: [3]f64 = undefined;
        for ([_]f64{ 0.002, 0.004, 0.008 }, &errors) |fraction, *relative| {
            const fd = try slope(t, sq, layer, &dir, fraction * @sqrt(size));
            relative.* = @abs(fd - length) / length;
            if (!std.math.isFinite(relative.*)) return error.NonfiniteGradient;
        }
        std.mem.sort(f64, &errors, {}, std.sort.asc(f64));
        std.debug.print("RESULT owned 27B layer {d} row {d}: median {d:.5}, worst {d:.5}\n", .{ layer, row, errors[1], errors[2] });
        if (errors[1] >= 0.15 or errors[2] >= 0.15) return error.GradientMismatch;
        checks += 1;
    };
    std.debug.print("PASS {d} owned directional checks, every configured step within bound\n", .{checks});
    return 0;
}
