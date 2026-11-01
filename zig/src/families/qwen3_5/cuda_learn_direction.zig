//! The dense and MoE model qualifiers share the same directional samples and complete layer/row traversal.
const std = @import("std");
const qwen = @import("cuda.zig");
const train = qwen.train;
const gradient = @import("core").directional;
const turn: train.Turn = .{ .decide = 0, .bits = .empty, .whole = true };
fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}
fn dot(a: []const f32, b: []const f32) f64 {
    var sum: f64 = 0;
    for (a, b) |x, y| sum += @as(f64, x) * y;
    return sum;
}
pub const Case = struct {
    t: *train.Trainer,
    sq: train.Seq,
    push: []f32,
    dir: []f32,
    label: []const u8,
    sink_fractions: [3]f64 = .{ 0.002, 0.004, 0.008 },

    fn sample(v: *Case, layer: usize, step: f64) !f64 {
        var losses: [6][2]f64 = undefined;
        defer @memset(v.t.deltaAt(0, 0, layer), 0);
        for (&losses, 1..) |*pair, m| for ([_]f64{ -1, 1 }, pair) |sign, *loss| {
            const x = sign * @as(f64, @floatFromInt(m)) * step;
            for (v.t.deltaAt(0, 0, layer), v.dir) |*out, d| out.* = @floatCast(x * d);
            var result: [1]train.Result = undefined;
            try v.t.pass(&.{v.sq}, .loss, turn, &result);
            loss.* = result[0].loss;
            if (!std.math.isFinite(loss.*)) return error.NonfiniteLoss;
            std.debug.print("SAMPLE {s} layer {d} row {d}: step {e:.12}, multiple {d}, sign {d:.0}, loss {d:.16}\n", .{ v.label, layer, v.sq.nudge.rows[0], step, m, sign, loss.* });
        };
        return gradient.slope(losses, step);
    }

    pub fn check(v: *Case, row: usize, layer: usize) !gradient.Verdict {
        v.sq.nudge = .{ .layers = .{ @as(u64, 1) << @intCast(layer), 0, 0 }, .rows = .{ row, 0, 0 } };
        var result: [1]train.Result = undefined;
        try v.t.pass(&.{v.sq}, .target, turn, &result);
        @memcpy(v.push, v.t.pushAt(0, 0, layer));
        const length = @sqrt(dot(v.push, v.push));
        if (!std.math.isFinite(length) or length <= 0) {
            std.debug.print("RESULT {s} layer {d} row {d}: BadGradient\n", .{ v.label, layer, row });
            return .unresolved;
        }
        for (v.dir, v.push) |*out, d| out.* = @floatCast(d / length);
        var size: f64 = 0;
        for (v.t.residual(layer + 1, row)) |x| size += bf(x) * bf(x);
        if (!std.math.isFinite(size) or size <= 0) {
            std.debug.print("RESULT {s} layer {d} row {d}: BadResidual\n", .{ v.label, layer, row });
            return .unresolved;
        }
        var errors: [3]f64 = undefined;
        var unresolved = false;
        for (if (row == 0) v.sink_fractions else [3]f64{ 0.002, 0.004, 0.008 }, &errors) |fraction, *relative| {
            const fd = v.sample(layer, fraction * @sqrt(size)) catch |err| switch (err) {
                error.NonfiniteLoss => {
                    std.debug.print("RESULT {s} layer {d} row {d}: NonfiniteLoss\n", .{ v.label, layer, row });
                    return .unresolved;
                },
                else => return err,
            };
            relative.* = @abs(fd - length) / length;
            std.debug.print("SLOPE {s} layer {d} row {d}: fraction {d:.6}, fd {d:.8}, grad {d:.8}, relative {d:.5}\n", .{ v.label, layer, row, fraction, fd, length, relative.* });
            if (!std.math.isFinite(relative.*)) unresolved = true;
        }
        std.mem.sort(f64, &errors, {}, std.sort.asc(f64));
        std.debug.print("RESULT {s} layer {d} row {d}: median {d:.5}, worst {d:.5}\n", .{ v.label, layer, row, errors[1], errors[2] });
        return if (unresolved) .unresolved else if (errors[1] >= 0.15 or errors[2] >= 0.15) .mismatch else .pass;
    }
};

pub fn run(v: *Case, rows: []const usize, layers: []const usize) !void {
    const out = try gradient.run(v, rows, layers);
    std.debug.print("SUMMARY {s}: {d} checks, {d} failed, {d} unresolved\n", .{ v.label, out.checks, out.failed, out.unresolved });
    try out.finish();
    std.debug.print("PASS {d} {s} directional checks\n", .{ out.checks, v.label });
}
