//! The dense and MoE model qualifiers share the same directional samples and complete layer/row traversal.
const std = @import("std");
const qwen = @import("cuda.zig");
const train = qwen.train;
const routes = @import("cuda_learn_routes.zig");
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
    route_base: []i32 = &.{},
    route_now: []i32 = &.{},

    fn sample(v: *Case, layer: usize, step: f64) !f64 {
        var losses: [2]f64 = undefined;
        defer @memset(v.t.deltaAt(0, 0, layer), 0);
        for ([_]f64{ -1, 1 }, &losses) |sign, *loss| {
            for (v.t.deltaAt(0, 0, layer), v.dir) |*out, d| out.* = @floatCast(sign * step * d);
            var result: [1]train.Result = undefined;
            try v.t.pass(&.{v.sq}, .loss, turn, &result);
            loss.* = result[0].loss;
            if (v.route_base.len != 0) {
                try routes.read(v.t, v.route_now);
                const changed = routes.changes(v.route_base, v.route_now, v.t.g.top_k);
                std.debug.print("ROUTING {s} layer {d} row {d}: step {e:.16}, sign {d:.0}, changed rows {d}, replaced slots {d}\n", .{ v.label, layer, v.sq.nudge.rows[0], step, sign, changed.rows, changed.slots });
            }
            std.debug.print("SAMPLE {s} layer {d} row {d}: step {e:.16}, sign {d:.0}, loss {e:.16}\n", .{ v.label, layer, v.sq.nudge.rows[0], step, sign, loss.* });
        }
        return gradient.central(losses, step);
    }

    fn report(v: *Case, row: usize, layer: usize, slopes: [6]f64, length: f64) gradient.Verdict {
        for (gradient.fractions, slopes) |fraction, fd| {
            std.debug.print("SLOPE {s} layer {d} row {d}: fraction {d:.7}, fd {e:.16}, grad {e:.16}\n", .{ v.label, layer, row, fraction, fd, length });
        }
        const judgment = gradient.evaluate(slopes, length);
        const status: []const u8 = if (judgment.verdict == .pass) "PASS" else "FAIL";
        std.debug.print("RESULT {s} layer {d} row {d}: {s}, estimate {e:.16}, g {e:.16}, relative {e:.16}\n", .{ v.label, layer, row, status, judgment.estimate, length, judgment.relative });
        return judgment.verdict;
    }

    pub fn check(v: *Case, row: usize, layer: usize) !gradient.Verdict {
        v.sq.nudge = .{ .layers = .{ @as(u64, 1) << @intCast(layer), 0, 0 }, .rows = .{ row, 0, 0 } };
        var result: [1]train.Result = undefined;
        try v.t.pass(&.{v.sq}, .target, turn, &result);
        const count = if (v.t.g.experts > 0) v.t.g.layers * v.t.rows * (v.t.g.top_k + 1) else 0;
        v.route_base = try v.t.gpa.alloc(i32, count);
        defer {
            v.t.gpa.free(v.route_base);
            v.route_base = &.{};
        }
        v.route_now = try v.t.gpa.alloc(i32, count);
        defer {
            v.t.gpa.free(v.route_now);
            v.route_now = &.{};
        }
        if (count > 0) try routes.read(v.t, v.route_base);
        @memcpy(v.push, v.t.pushAt(0, 0, layer));
        const length = @sqrt(dot(v.push, v.push));
        if (!std.math.isFinite(length) or length <= 0) {
            std.debug.print("DETAIL {s} layer {d} row {d}: BadGradient\n", .{ v.label, layer, row });
            return v.report(row, layer, @splat(std.math.nan(f64)), length);
        }
        for (v.dir, v.push) |*out, d| out.* = @floatCast(d / length);
        std.debug.print("DIRECTION {s} layer {d} row {d}: gradient {x:0>16}, unit {x:0>16}\n", .{ v.label, layer, row, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(v.push)), std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(v.dir)) });
        var size: f64 = 0;
        for (v.t.residual(layer + 1, row)) |x| size += bf(x) * bf(x);
        if (!std.math.isFinite(size) or size <= 0) {
            std.debug.print("DETAIL {s} layer {d} row {d}: BadResidual\n", .{ v.label, layer, row });
            return v.report(row, layer, @splat(std.math.nan(f64)), length);
        }
        var slopes: [6]f64 = undefined;
        for (gradient.fractions, &slopes) |fraction, *fd| fd.* = try v.sample(layer, fraction * @sqrt(size));
        return v.report(row, layer, slopes, length);
    }
};

pub fn run(v: *Case, rows: []const usize, layers: []const usize) !void {
    const out = try gradient.run(v, rows, layers);
    std.debug.print("SUMMARY {s}: {d} checks, {d} failed\n", .{ v.label, out.checks, out.failed });
    try out.finish();
    std.debug.print("PASS {d} {s} directional checks\n", .{ out.checks, v.label });
}
