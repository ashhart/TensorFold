//! Full tiny coordinates, served head widths, alias/refusal and allocation controls for prepared attention.
const std = @import("std");
const t = std.testing;
const back = @import("attention_backward.zig");
const ref = @import("attention_backward_reference.zig");
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
const linear_rows = [_]back.Row{ .{ .end = 2 }, .{ .begin = 2, .end = 6 }, .{ .begin = 4, .end = 7 } };
const ring_rows = [_]back.Row{ .{ .begin = 6, .end = 9 }, .{ .begin = 7, .end = 10 }, .{ .begin = 10, .end = 14 } };
fn coordinates(comptime T: type, ring: bool) !void {
    var shape = small;
    shape.ring = ring;
    shape.first = if (ring) 6 else 0;
    var p = try ref.Point.init(t.allocator, shape, if (ring) &ring_rows else &linear_rows);
    defer p.deinit();
    var g = try Grad(T).init(t.allocator, p);
    defer g.deinit();
    try g.run(p);
    var errors: ref.Error = .{};
    for ([_][]f64{ p.q, p.k, p.v }, [_][]T{ g.dq, g.dk, g.dv }) |values, derivative|
        for (derivative, 0..) |got, i| for ([_]f64{ 0x1p-12, 0x1p-15, 0x1p-18 }) |step|
            try errors.check(T, try p.coordinate(values, i, step), got);
    std.debug.print("GEMMA_BACKWARD_COORD {s} ring={} probes={d} absolute={e:.8} relative={e:.8}\n", .{ @typeName(T), ring, errors.probes, errors.max_absolute, errors.max_relative });
}
test "Gemma backward: tiny scale-one global sliding and wrapped image windows match every float64 slope" {
    inline for (.{ f64, f32 }) |T| for ([_]bool{ false, true }) |ring| try coordinates(T, ring);
}
fn directional(comptime T: type, p: ref.Point) !void {
    var g = try Grad(T).init(t.allocator, p);
    defer g.deinit();
    try g.run(p);
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
    var analytic: f64 = 0;
    for (dirs, [_][]T{ g.dq, g.dk, g.dv }) |dir, gradient| for (dir, gradient) |*d, v| {
        d.* /= @sqrt(norm);
        analytic += d.* * @as(f64, v);
    };
    defer for (inputs, saved) |input, copy| @memcpy(input, copy);
    var errors: ref.Error = .{};
    for ([_]f64{ 0x1p-8, 0x1p-10, 0x1p-12 }) |step| {
        for (inputs, saved, dirs) |input, copy, dir| for (input, copy, dir) |*v, x, d| {
            v.* = x + step * d;
        };
        const plus = try p.loss();
        for (inputs, saved, dirs) |input, copy, dir| for (input, copy, dir) |*v, x, d| {
            v.* = x - step * d;
        };
        try errors.check(T, (plus - try p.loss()) / (2 * step), @as(T, @floatCast(analytic)));
    }
    std.debug.print("GEMMA_BACKWARD_DIRECTION {s} heads={d} kv={d} width={d} keys={d} ring={} absolute={e:.8}\n", .{ @typeName(T), p.shape.heads, p.shape.kv, p.shape.width, p.rows[0].end - p.rows[0].begin, p.shape.ring, errors.max_absolute });
}
test "Gemma backward: 31B and 12B sliding and global head widths cross window and image edges" {
    const cases = [_]struct { shape: back.Shape, row: back.Row }{
        .{ .shape = .{ .rows = 1, .heads = 32, .kv = 16, .width = 256, .capacity = 1031, .ring = true, .first = 1027 }, .row = .{ .begin = 1034, .end = 2058 } },
        .{ .shape = .{ .rows = 1, .heads = 32, .kv = 4, .width = 512, .capacity = 131 }, .row = .{ .end = 129 } },
        .{ .shape = .{ .rows = 1, .heads = 16, .kv = 8, .width = 256, .capacity = 1031 }, .row = .{ .begin = 5, .end = 1030 } },
        .{ .shape = .{ .rows = 1, .heads = 16, .kv = 1, .width = 512, .capacity = 131 }, .row = .{ .end = 129 } },
    };
    for (cases) |case| {
        var p = try ref.Point.init(t.allocator, case.shape, &.{case.row});
        defer p.deinit();
        try directional(f64, p);
        try directional(f32, p);
    }
}
test "Gemma backward: masked cache NaNs are not read and their adjoints stay zero" {
    var p = try ref.Point.init(t.allocator, .{ .rows = 2, .heads = 4, .kv = 2, .width = 4, .capacity = 8, .first = 6, .ring = true }, &.{ .{ .begin = 7, .end = 10 }, .{ .begin = 8, .end = 11 } });
    defer p.deinit();
    for ([_]usize{ 3, 4, 5, 6 }) |slot| {
        @memset(p.k[slot * 8 ..][0..8], std.math.nan(f64));
        @memset(p.v[slot * 8 ..][0..8], std.math.nan(f64));
    }
    var g = try Grad(f64).init(t.allocator, p);
    defer g.deinit();
    try g.run(p);
    for ([_]usize{ 3, 4, 5, 6 }) |slot| {
        try t.expect(std.mem.allEqual(f64, g.dk[slot * 8 ..][0..8], 0));
        try t.expect(std.mem.allEqual(f64, g.dv[slot * 8 ..][0..8], 0));
    }
}
test "Gemma backward: one key has zero score gradient and all GQA rows add to the shared value" {
    var p = try ref.Point.init(t.allocator, .{ .rows = 2, .heads = 4, .kv = 2, .width = 4, .capacity = 3 }, &.{ .{ .begin = 1, .end = 2 }, .{ .begin = 1, .end = 2 } });
    defer p.deinit();
    var g = try Grad(f64).init(t.allocator, p);
    defer g.deinit();
    try g.run(p);
    try t.expect(std.mem.allEqual(f64, g.dq, 0));
    try t.expect(std.mem.allEqual(f64, g.dk, 0));
    for (0..2) |h| for (0..4) |d| {
        var want: f64 = 0;
        for (0..2) |r| for (0..2) |q| {
            want += p.dy[(r * 4 + h * 2 + q) * 4 + d];
        };
        try t.expectEqual(want, g.dv[(2 + h) * 4 + d]);
    };
}
test "Gemma backward: input aliases keep distinct K V adjoints and combine by the chain rule" {
    var p = try ref.Point.init(t.allocator, small, &linear_rows);
    defer p.deinit();
    @memcpy(p.v, p.k);
    var g = try Grad(f64).init(t.allocator, p);
    defer g.deinit();
    try back.backward(f64, t.allocator, small, p.rows, g.q, g.k, g.k, g.dy, g.dq, g.dk, g.dv);
    for (p.k, 0..) |saved, i| {
        const step = 0x1p-15;
        p.k[i] = saved + step;
        p.v[i] = saved + step;
        const plus = try p.loss();
        p.k[i] = saved - step;
        p.v[i] = saved - step;
        const slope = (plus - try p.loss()) / (2 * step);
        p.k[i] = saved;
        p.v[i] = saved;
        var error_check: ref.Error = .{};
        try error_check.check(f64, slope, g.dk[i] + g.dv[i]);
    }
}
test "Gemma backward: repeats overwrite gradients without changing inputs or mask metadata" {
    var p = try ref.Point.init(t.allocator, small, &linear_rows);
    defer p.deinit();
    var g = try Grad(f32).init(t.allocator, p);
    defer g.deinit();
    try g.run(p);
    const outputs = [_][]f32{ g.dq, g.dk, g.dv };
    var copies: [3][]f32 = @splat(&.{});
    defer for (copies) |x| t.allocator.free(x);
    for (outputs, &copies) |out, *copy| copy.* = try t.allocator.dupe(f32, out);
    for (0..5) |_| {
        for (outputs) |out| @memset(out, std.math.nan(f32));
        try g.run(p);
        for (outputs, copies) |out, copy| try t.expectEqualSlices(u8, std.mem.sliceAsBytes(copy), std.mem.sliceAsBytes(out));
    }
    for ([_][]const f32{ g.q, g.k, g.v, g.dy }, [_][]const f64{ p.q, p.k, p.v, p.dy }) |input, original|
        for (input, original) |x, y| try t.expectEqual(@as(f32, @floatCast(y)), x);
    try t.expectEqualDeep(linear_rows, p.rows[0..3].*);
}
test "Gemma backward: malformed windows shapes and output overlaps refuse before writes" {
    var p = try ref.Point.init(t.allocator, small, &linear_rows);
    defer p.deinit();
    var g = try Grad(f64).init(t.allocator, p);
    defer g.deinit();
    @memset(g.dq, 91);
    @memset(g.dk, 92);
    @memset(g.dv, 93);
    var bad = small;
    bad.heads = 3;
    try t.expectError(error.GemmaGradientShape, back.backward(f64, t.allocator, bad, p.rows, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv));
    for ([_]back.Row{ .{ .end = 0 }, .{ .begin = 5, .end = 4 }, .{ .end = 9 } }) |row| {
        const rows = [_]back.Row{ row, linear_rows[1], linear_rows[2] };
        try t.expectError(error.GemmaGradientWindow, back.backward(f64, t.allocator, small, &rows, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv));
    }
    try t.expectError(error.GemmaGradientAlias, back.backward(f64, t.allocator, small, p.rows, g.q, g.k, g.v, g.dy, g.q, g.dk, g.dv));
    try t.expectError(error.GemmaGradientAlias, back.backward(f64, t.allocator, small, p.rows, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dk));
    g.q[0] = std.math.nan(f64);
    try t.expectError(error.GemmaGradientNonfinite, g.run(p));
    try t.expect(std.mem.allEqual(f64, g.dq, 91));
    try t.expect(std.mem.allEqual(f64, g.dk, 92));
    try t.expect(std.mem.allEqual(f64, g.dv, 93));
    g.q[0] = std.math.floatMax(f64);
    g.k[0] = std.math.floatMax(f64);
    try t.expectError(error.GemmaGradientNonfinite, g.run(p));
    try t.expect(std.mem.allEqual(f64, g.dq, 0));
    try t.expect(std.mem.allEqual(f64, g.dk, 0));
    try t.expect(std.mem.allEqual(f64, g.dv, 0));
}
fn allocationCase(a: std.mem.Allocator) !void {
    const shape = back.Shape{ .rows = 1, .heads = 2, .kv = 1, .width = 2, .capacity = 2 };
    const q = [_]f64{ 1, 0.5, -0.25, 0.75 };
    const k = [_]f64{ 0.5, 0.25, -0.5, 0.75 };
    const v = [_]f64{ 0.25, 0.125, 1, -0.5 };
    const dy = [_]f64{ 0.25, 1, -0.5, 0.125 };
    var dq: [4]f64 = @splat(91);
    var dk: [4]f64 = @splat(92);
    var dv: [4]f64 = @splat(93);
    back.backward(f64, a, shape, &.{.{ .end = 2 }}, &q, &k, &v, &dy, &dq, &dk, &dv) catch |err| {
        try t.expect(std.mem.allEqual(f64, &dq, 91));
        try t.expect(std.mem.allEqual(f64, &dk, 92));
        try t.expect(std.mem.allEqual(f64, &dv, 93));
        return err;
    };
}
test "Gemma backward: every scratch allocation failure leaves outputs untouched and frees prior scratch" {
    try t.checkAllAllocationFailures(t.allocator, allocationCase, .{});
}

test "Gemma backward: mask aliases and cache or byte-count overflow refuse without mutation" {
    var p = try ref.Point.init(t.allocator, small, &linear_rows);
    defer p.deinit();
    var g = try Grad(f64).init(t.allocator, p);
    defer g.deinit();
    @memset(g.dq, 91);
    @memset(g.dk, 92);
    @memset(g.dv, 93);
    var bad = small;
    bad.first = 1;
    try t.expectError(error.GemmaGradientShape, back.backward(f64, t.allocator, bad, p.rows, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv));
    bad.ring = true;
    bad.first = 6;
    try t.expectError(error.GemmaGradientWindow, back.backward(f64, t.allocator, bad, p.rows, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv));
    bad.first = std.math.maxInt(usize);
    try t.expectError(error.Overflow, back.backward(f64, t.allocator, bad, p.rows, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv));
    bad = .{ .rows = 1, .heads = 1, .kv = 1, .width = 1, .capacity = std.math.maxInt(usize) / 2 };
    try t.expectError(error.Overflow, back.backward(f64, t.allocator, bad, &.{.{ .end = 1 }}, g.q, g.k, g.v, g.dy, g.dq, g.dk, g.dv));
    try t.expect(std.mem.allEqual(f64, g.dq, 91));
    try t.expect(std.mem.allEqual(f64, g.dk, 92));
    try t.expect(std.mem.allEqual(f64, g.dv, 93));
    var raw: [4]f64 = @splat(91);
    const rows: *[1]back.Row = @ptrCast(@alignCast(&raw));
    rows[0] = .{ .end = 1 };
    const before = raw;
    var dk: [2]f64 = @splat(92);
    var dv: [2]f64 = @splat(93);
    try t.expectError(error.GemmaGradientAlias, back.backward(f64, t.allocator, .{ .rows = 1, .heads = 2, .kv = 1, .width = 2, .capacity = 1 }, rows, g.q[0..4], g.k[0..2], g.v[0..2], g.dy[0..4], &raw, &dk, &dv));
    try t.expectEqualSlices(u8, std.mem.asBytes(&before), std.mem.asBytes(&raw));
}
