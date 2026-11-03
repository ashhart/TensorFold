//! Independent binary64 loss and fixed-rounding-offset probes for the host adjoint check.
const std = @import("std");
const back = @import("qsa_backward.zig");

pub const Point = struct {
    a: std.mem.Allocator,
    shape: back.Shape,
    rows: []const back.Row,
    q: []f64,
    k: []f64,
    v: []f64,
    dy: []f64,
    gate: []f64,
    context_offset: []f64,
    sigmoid_offset: []f64,

    pub fn init(a: std.mem.Allocator, g: back.Shape, rows: []const back.Row) !Point {
        const nq = g.rows * g.heads * g.width;
        const nk = g.capacity * g.kv * g.width;
        var p: Point = .{ .a = a, .shape = g, .rows = rows, .q = &.{}, .k = &.{}, .v = &.{}, .dy = &.{}, .gate = &.{}, .context_offset = &.{}, .sigmoid_offset = &.{} };
        errdefer p.deinit();
        p.q = try a.alloc(f64, nq);
        p.k = try a.alloc(f64, nk);
        p.v = try a.alloc(f64, nk);
        p.dy = try a.alloc(f64, nq);
        p.gate = try a.alloc(f64, nq);
        p.context_offset = try a.alloc(f64, nq);
        p.sigmoid_offset = try a.alloc(f64, nq);
        var rng = std.Random.DefaultPrng.init(1804);
        for ([_][]f64{ p.q, p.k, p.v, p.dy, p.gate }) |values| for (values) |*x| {
            x.* = @as(f64, @floatFromInt(rng.random().intRangeAtMost(i32, -31, 31))) / 32;
        };
        @memset(p.context_offset, 0);
        @memset(p.sigmoid_offset, 0);
        return p;
    }
    pub fn deinit(p: *Point) void {
        for ([_][]f64{ p.q, p.k, p.v, p.dy, p.gate, p.context_offset, p.sigmoid_offset }) |v| p.a.free(v);
    }

    fn allowed(row: back.Row, key: usize) bool {
        if (key >= row.end) return false;
        const ids = row.ids orelse return true;
        for (ids) |id| if (id == key) return true;
        return false;
    }

    pub fn context(p: Point, out: []f64) !void {
        const g = p.shape;
        const scores = try p.a.alloc(f64, g.capacity);
        defer p.a.free(scores);
        const mask = try p.a.alloc(bool, g.capacity);
        defer p.a.free(mask);
        for (0..g.rows) |r| {
            for (mask, 0..) |*yes, token| yes.* = allowed(p.rows[r], token);
            for (0..g.heads) |h| {
                const at = (r * g.heads + h) * g.width;
                var maximum = -std.math.inf(f64);
                for (scores, 0..) |*s, token| {
                    s.* = -std.math.inf(f64);
                    if (!mask[token]) continue;
                    const head = h * g.kv / g.heads;
                    const base = (token * g.kv + head) * g.width;
                    var product: f64 = 0;
                    for (0..g.width) |d| product += p.q[at + d] * p.k[base + d];
                    s.* = g.scale * product;
                    maximum = @max(maximum, s.*);
                }
                var denominator: f64 = 0;
                for (scores) |*s| {
                    s.* = @exp(s.* - maximum);
                    denominator += s.*;
                }
                @memset(out[at..][0..g.width], 0);
                for (0..g.capacity) |token| {
                    if (!mask[token]) continue;
                    const base = (token * g.kv + h * g.kv / g.heads) * g.width;
                    for (0..g.width) |d| out[at + d] += scores[token] * p.v[base + d] / denominator;
                }
            }
        }
    }

    pub fn loss(p: Point, gated: bool) !f64 {
        const ctx = try p.a.alloc(f64, p.q.len);
        defer p.a.free(ctx);
        try p.context(ctx);
        var value: f64 = 0;
        for (ctx, p.gate, p.context_offset, p.sigmoid_offset, p.dy) |x, gate, xc, sc, dy|
            value += dy * (if (gated) (x + xc) * (1 / (1 + @exp(-gate)) + sc) else x);
        return value;
    }

    pub fn coordinate(p: Point, values: []f64, index: usize, step: f64, gated: bool) !f64 {
        const saved = values[index];
        defer values[index] = saved;
        values[index] = saved + step;
        const plus = try p.loss(gated);
        values[index] = saved - step;
        return (plus - try p.loss(gated)) / (2 * step);
    }

    pub fn rounding(p: Point, enabled: bool, saved_context: []f64, saved_sigmoid: []f64) !void {
        try p.context(saved_context);
        for (saved_context, saved_sigmoid, p.gate, p.context_offset, p.sigmoid_offset) |*x, *s, gate, *xc, *sc| {
            const plain_x = x.*;
            const plain_s = 1 / (1 + @exp(-gate));
            s.* = if (enabled) bf(plain_s) else plain_s;
            x.* = if (enabled) bf(plain_x) else plain_x;
            xc.* = x.* - plain_x;
            sc.* = s.* - plain_s;
        }
    }
};

pub fn bf(x: f64) f64 {
    const bits: u32 = @bitCast(@as(f32, @floatCast(x)));
    const word = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
    return @as(f32, @bitCast(word));
}

pub fn cast(comptime T: type, a: std.mem.Allocator, values: []const f64) ![]T {
    const out = try a.alloc(T, values.len);
    for (out, values) |*to, v| to.* = @floatCast(v);
    return out;
}

pub const Error = struct {
    max_absolute: f64 = 0,
    max_relative: f64 = 0,
    coordinates: usize = 0,
    pub fn check(e: *Error, comptime T: type, exact: f64, actual: T) !void {
        const delta = @abs(exact - actual);
        e.max_absolute = @max(e.max_absolute, delta);
        e.max_relative = @max(e.max_relative, delta / @max(1e-10, @abs(exact)));
        e.coordinates += 1;
        const limit = if (T == f64) 1e-8 + 1e-6 * @abs(exact) else 2e-7 + 3e-5 * @abs(exact);
        if (!std.math.isFinite(exact) or !std.math.isFinite(actual) or delta > limit) return error.QsaFiniteDifferenceMismatch;
    }
};
