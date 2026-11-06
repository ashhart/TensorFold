//! Exhaustive tiny adjoints, served-width directions and malformed tapes use an independent f64 loss.
const std = @import("std");
const t = std.testing;
const back = @import("qsa_backward.zig");
const ref = @import("qsa_backward_reference.zig");

fn Grad(comptime T: type) type {
    return struct {
        a: std.mem.Allocator,
        q: []T,
        k: []T,
        v: []T,
        dy: []T,
        dq: []T,
        dk: []T,
        dv: []T,
        fn init(a: std.mem.Allocator, p: ref.Point) !@This() {
            var g: @This() = .{ .a = a, .q = &.{}, .k = &.{}, .v = &.{}, .dy = &.{}, .dq = &.{}, .dk = &.{}, .dv = &.{} };
            errdefer g.deinit();
            g.q = try ref.cast(T, a, p.q);
            g.k = try ref.cast(T, a, p.k);
            g.v = try ref.cast(T, a, p.v);
            g.dy = try ref.cast(T, a, p.dy);
            g.dq = try a.alloc(T, p.q.len);
            g.dk = try a.alloc(T, p.k.len);
            g.dv = try a.alloc(T, p.v.len);
            return g;
        }
        fn deinit(g: *@This()) void {
            for ([_][]T{ g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv }) |s| g.a.free(s);
        }
        fn run(g: *@This(), p: ref.Point) !void {
            try back.backward(T, g.a, p.shape, p.rows, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv);
        }
    };
}

const small = back.Shape{ .rows = 3, .heads = 4, .kv = 2, .width = 4, .capacity = 8 };
const sparse0 = [_]u32{ 2, 3 };
const sparse1 = [_]u32{ 4, 5, 6 };
const small_rows = [_]back.Row{ .{ .end = 2 }, .{ .end = 6, .ids = &sparse0 }, .{ .end = 7, .ids = &sparse1 } };

fn coordinates(comptime T: type, gated: bool, rounded: bool) !void {
    var p = try ref.Point.init(t.allocator, small, &small_rows);
    defer p.deinit();
    var gradient = try Grad(T).init(t.allocator, p);
    defer gradient.deinit();
    const context = try t.allocator.alloc(f64, p.q.len);
    defer t.allocator.free(context);
    const sigmoid = try t.allocator.alloc(f64, p.q.len);
    defer t.allocator.free(sigmoid);
    const dg = try t.allocator.alloc(T, p.q.len);
    defer t.allocator.free(dg);
    if (gated) {
        try p.rounding(rounded, context, sigmoid);
        const x = try ref.cast(T, t.allocator, context);
        defer t.allocator.free(x);
        const gate = try ref.cast(T, t.allocator, p.gate);
        defer t.allocator.free(gate);
        const s = try ref.cast(T, t.allocator, sigmoid);
        defer t.allocator.free(s);
        const dx = try t.allocator.alloc(T, p.q.len);
        defer t.allocator.free(dx);
        try back.gateBackward(T, x, gate, s, gradient.dy, dx, dg);
        @memcpy(gradient.dy, dx);
    }
    try gradient.run(p);
    var errors: ref.Error = .{};
    const values = [_][]f64{ p.q, p.k, p.v, p.gate };
    const grads = [_][]T{ gradient.dq, gradient.dk, gradient.dv, dg };
    const count: usize = if (gated) 4 else 3;
    for (values[0..count], grads[0..count]) |input, derivative| for (input, derivative, 0..) |_, actual, i| {
        for ([_]f64{ 0x1p-12, 0x1p-15, 0x1p-18 }) |step|
            try errors.check(T, try p.coordinate(input, i, step, gated), actual);
    };
    std.debug.print("QSA_BACKWARD_COORD {s} gated={} rounded={} probes={d} absolute={e:.8} relative={e:.8}\n", .{ @typeName(T), gated, rounded, errors.coordinates, errors.max_absolute, errors.max_relative });
}

test "Qwen4 QSA backward: every tiny Q K V coordinate matches independent float64 central slopes" {
    try coordinates(f64, false, false);
    try coordinates(f32, false, false);
}

test "Qwen4 QSA backward: output gate and frozen forward cuts match local straight-through finite differences" {
    inline for (.{ f64, f32 }) |T| for ([_]bool{ false, true }) |rounded| try coordinates(T, true, rounded);
}

fn direction(comptime T: type, g: Grad(T), arrays: []const []f64) f64 {
    var dot: f64 = 0;
    for ([_][]const T{ g.dq, g.dk, g.dv }, arrays) |gradient, values|
        for (gradient, values) |v, d| {
            dot += @as(f64, v) * d;
        };
    return dot;
}

fn directional(comptime T: type, p: ref.Point) !void {
    var gradient = try Grad(T).init(t.allocator, p);
    defer gradient.deinit();
    try gradient.run(p);
    const inputs = [_][]f64{ p.q, p.k, p.v };
    var saved: [3][]f64 = @splat(&.{});
    var dirs: [3][]f64 = @splat(&.{});
    defer for (saved, dirs) |x, y| {
        t.allocator.free(x);
        t.allocator.free(y);
    };
    var norm: f64 = 0;
    for (inputs, &saved, &dirs, 0..) |input, *copy, *dir, part| {
        copy.* = try t.allocator.dupe(f64, input);
        dir.* = try t.allocator.alloc(f64, input.len);
        for (dir.*, 0..) |*x, i| {
            x.* = @as(f64, @floatFromInt(@as(i64, @intCast((i * 17 + part * 13) % 31)) - 15)) / 16;
            norm += x.* * x.*;
        }
    }
    for (dirs) |dir| for (dir) |*x| {
        x.* /= @sqrt(norm);
    };
    defer for (inputs, saved) |x, copy| @memcpy(x, copy);
    const analytic = direction(T, gradient, &dirs);
    var errors: ref.Error = .{};
    for ([_]f64{ 0x1p-8, 0x1p-10, 0x1p-12 }) |step| {
        for (inputs, saved, dirs) |input, copy, dir| for (input, copy, dir) |*x, base, delta| {
            x.* = base + step * delta;
        };
        const plus = try p.loss(false);
        for (inputs, saved, dirs) |input, copy, dir| for (input, copy, dir) |*x, base, delta| {
            x.* = base - step * delta;
        };
        const central = (plus - try p.loss(false)) / (2 * step);
        try errors.check(T, central, @as(T, @floatCast(analytic)));
    }
    std.debug.print("QSA_BACKWARD_DIRECTION {s} rows={d} heads={d} kv={d} width={d} selected={d} absolute={e:.8}\n", .{ @typeName(T), p.shape.rows, p.shape.heads, p.shape.kv, p.shape.width, if (p.rows[0].ids) |ids| ids.len else p.rows[0].end, errors.max_absolute });
}

test "Qwen4 QSA backward: served 24 by 4 heads width 256 crosses dense 128 and sparse 2048 plus tail" {
    var dense = try ref.Point.init(t.allocator, .{ .rows = 1, .heads = 24, .kv = 4, .width = 256, .capacity = 131 }, &.{.{ .end = 129 }});
    defer dense.deinit();
    try directional(f64, dense);
    try directional(f32, dense);
    var ids: [2051]u32 = undefined;
    for (0..512) |block| for (0..4) |j| {
        ids[4 * block + j] = @intCast(4 * (514 - block) + j);
    };
    for (0..3) |j| ids[2048 + j] = @intCast(2060 + j);
    var sparse = try ref.Point.init(t.allocator, .{ .rows = 1, .heads = 24, .kv = 4, .width = 256, .capacity = 2065 }, &.{.{ .end = 2063, .ids = &ids }});
    defer sparse.deinit();
    try directional(f64, sparse);
    try directional(f32, sparse);
}

test "Qwen4 QSA backward: unused and future NaNs are never read and their adjoints stay zero" {
    var p = try ref.Point.init(t.allocator, .{ .rows = 2, .heads = 4, .kv = 2, .width = 4, .capacity = 6 }, &.{ .{ .end = 3, .ids = &.{ 2, 0 } }, .{ .end = 4, .ids = &.{ 0, 2, 3 } } });
    defer p.deinit();
    for ([_]usize{ 1, 4, 5 }) |key| {
        @memset(p.k[key * 8 ..][0..8], std.math.nan(f64));
        @memset(p.v[key * 8 ..][0..8], std.math.nan(f64));
    }
    var g = try Grad(f64).init(t.allocator, p);
    defer g.deinit();
    try g.run(p);
    for ([_]usize{ 1, 4, 5 }) |key| for (g.dk[key * 8 ..][0..8], g.dv[key * 8 ..][0..8]) |dk, dv| {
        try t.expectEqual(@as(f64, 0), dk);
        try t.expectEqual(@as(f64, 0), dv);
    };
    try t.expect(std.mem.indexOfNone(f64, g.dk[0..8], &.{0}) != null);
    try t.expect(std.mem.indexOfNone(f64, g.dv[0..8], &.{0}) != null);
}

test "Qwen4 QSA backward: one selected key has zero score adjoint and every GQA use accumulates" {
    const g = back.Shape{ .rows = 2, .heads = 4, .kv = 2, .width = 4, .capacity = 3 };
    var p = try ref.Point.init(t.allocator, g, &.{ .{ .end = 2, .ids = &.{1} }, .{ .end = 3, .ids = &.{1} } });
    defer p.deinit();
    var grad = try Grad(f64).init(t.allocator, p);
    defer grad.deinit();
    try grad.run(p);
    try t.expect(std.mem.allEqual(f64, grad.dq, 0));
    try t.expect(std.mem.allEqual(f64, grad.dk, 0));
    for (0..g.kv) |head| for (0..g.width) |d| {
        var sum: f64 = 0;
        for (0..g.rows) |r| for (0..2) |h| {
            sum += p.dy[(r * g.heads + 2 * head + h) * g.width + d];
        };
        try t.expectEqual(sum, grad.dv[(g.kv + head) * g.width + d]);
    };
    try t.expect(std.mem.allEqual(f64, grad.dv[0..8], 0));
    try t.expect(std.mem.allEqual(f64, grad.dv[16..], 0));
}

test "Qwen4 QSA backward: repeated calls overwrite adjoints and never mutate the forward tape" {
    var p = try ref.Point.init(t.allocator, small, &small_rows);
    defer p.deinit();
    var grad = try Grad(f32).init(t.allocator, p);
    defer grad.deinit();
    const before = try t.allocator.dupe(f32, grad.q);
    defer t.allocator.free(before);
    try grad.run(p);
    const outputs = [_][]f32{ grad.dq, grad.dk, grad.dv };
    var kept: [3][]f32 = @splat(&.{});
    defer for (kept) |values| t.allocator.free(values);
    for (&kept, outputs) |*copy, values| copy.* = try t.allocator.dupe(f32, values);
    for (0..5) |_| {
        for (outputs) |values| @memset(values, std.math.nan(f32));
        try grad.run(p);
        for (kept, outputs) |a, b| try t.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
    }
    try t.expectEqualSlices(f32, before, grad.q);
    for (grad.k, p.k) |a, b| try t.expectEqual(@as(f32, @floatCast(b)), a);
    for (grad.v, p.v) |a, b| try t.expectEqual(@as(f32, @floatCast(b)), a);
    for (grad.dy, p.dy) |a, b| try t.expectEqual(@as(f32, @floatCast(b)), a);
}

test "Qwen4 QSA backward: malformed geometry selection and aliases refuse before output writes" {
    const a = t.allocator;
    var p = try ref.Point.init(a, small, &small_rows);
    defer p.deinit();
    var g = try Grad(f64).init(a, p);
    defer g.deinit();
    @memset(g.dq, 91);
    @memset(g.dk, 92);
    @memset(g.dv, 93);
    var bad = small;
    bad.heads = 3;
    try t.expectError(error.QsaGradientShape, back.backward(f64, a, bad, p.rows, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv));
    for ([_][]const back.Row{ &.{ .{ .end = 0 }, small_rows[1], small_rows[2] }, &.{ .{ .end = 2, .ids = &.{ 0, 2 } }, small_rows[1], small_rows[2] }, &.{ .{ .end = 2, .ids = &.{ 0, 0 } }, small_rows[1], small_rows[2] } }) |rows|
        try t.expectError(error.QsaGradientSelection, back.backward(f64, a, small, rows, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv));
    try t.expectError(error.QsaGradientAlias, back.backward(f64, a, small, p.rows, g.q, g.k, g.v, g.dy, g.q, g.dk, g.dv));
    try t.expect(std.mem.allEqual(f64, g.dq, 91));
    try t.expect(std.mem.allEqual(f64, g.dk, 92));
    try t.expect(std.mem.allEqual(f64, g.dv, 93));
    g.q[0] = std.math.nan(f64);
    try t.expectError(error.QsaGradientNonfinite, g.run(p));
    try t.expect(std.mem.allEqual(f64, g.dq, 91));
    g.q[0] = std.math.floatMax(f64);
    g.k[0] = std.math.floatMax(f64);
    try t.expectError(error.QsaGradientNonfinite, g.run(p));
    try t.expect(std.mem.allEqual(f64, g.dq, 0));
    try t.expect(std.mem.allEqual(f64, g.dk, 0));
    try t.expect(std.mem.allEqual(f64, g.dv, 0));
}
