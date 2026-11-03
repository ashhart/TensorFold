//! Independent dense-mask binary64 objective; does not call the adjoint's indexing or arithmetic helpers.
const std = @import("std");
const back = @import("attention_backward.zig");
pub const Point = struct {
    a: std.mem.Allocator,
    shape: back.Shape,
    rows: []const back.Row,
    q: []f64,
    k: []f64,
    v: []f64,
    dy: []f64,
    pub fn init(a: std.mem.Allocator, shape: back.Shape, rows: []const back.Row) !Point {
        var p: Point = .{ .a = a, .shape = shape, .rows = rows, .q = &.{}, .k = &.{}, .v = &.{}, .dy = &.{} };
        errdefer p.deinit();
        p.q = try a.alloc(f64, shape.rows * shape.heads * shape.width);
        p.dy = try a.alloc(f64, p.q.len);
        p.k = try a.alloc(f64, shape.capacity * shape.kv * shape.width);
        p.v = try a.alloc(f64, p.k.len);
        var rng = std.Random.DefaultPrng.init(1803);
        for ([_][]f64{ p.q, p.k, p.v, p.dy }, 0..) |values, part| for (values) |*x| {
            x.* = @as(f64, @floatFromInt(rng.random().intRangeAtMost(i32, -31, 31))) / 32;
            if (shape.width >= 256 and part < 2) x.* *= 0.25;
        };
        return p;
    }
    pub fn deinit(p: *Point) void {
        for ([_][]f64{ p.q, p.k, p.v, p.dy }) |x| p.a.free(x);
    }
    pub fn loss(p: Point) !f64 {
        const g = p.shape;
        const scores = try p.a.alloc(f64, g.capacity);
        defer p.a.free(scores);
        const mask = try p.a.alloc(bool, g.capacity);
        defer p.a.free(mask);
        var result: f64 = 0;
        for (0..g.rows) |r| {
            // Invert each physical slot within this snapshot's single logical generation.
            for (mask, 0..) |*yes, slot| {
                const logical = if (g.ring) g.first + (slot + g.capacity - g.first % g.capacity) % g.capacity else slot;
                yes.* = logical >= p.rows[r].begin and logical < p.rows[r].end;
            }
            for (0..g.heads) |h| {
                const at = (r * g.heads + h) * g.width;
                const kh = h * g.kv / g.heads;
                var top = -std.math.inf(f64);
                for (scores, 0..) |*s, slot| {
                    s.* = -std.math.inf(f64);
                    if (!mask[slot]) continue;
                    s.* = 0;
                    for (0..g.width) |d| s.* += p.q[at + d] * p.k[(slot * g.kv + kh) * g.width + d];
                    top = @max(top, s.*);
                }
                var total: f64 = 0;
                for (scores) |*s| {
                    s.* = @exp(s.* - top);
                    total += s.*;
                }
                for (0..g.width) |d| {
                    var context: f64 = 0;
                    for (scores, mask, 0..) |weight, yes, slot| {
                        if (!yes) continue;
                        context += weight / total * p.v[(slot * g.kv + kh) * g.width + d];
                    }
                    result += context * p.dy[at + d];
                }
            }
        }
        return result;
    }
    pub fn coordinate(p: Point, values: []f64, index: usize, step: f64) !f64 {
        const saved = values[index];
        defer values[index] = saved;
        values[index] = saved + step;
        const plus = try p.loss();
        values[index] = saved - step;
        return (plus - try p.loss()) / (2 * step);
    }
};
pub fn cast(comptime T: type, a: std.mem.Allocator, xs: []const f64) ![]T {
    const out = try a.alloc(T, xs.len);
    for (out, xs) |*to, x| to.* = @floatCast(x);
    return out;
}
pub const Error = struct {
    probes: usize = 0,
    max_absolute: f64 = 0,
    max_relative: f64 = 0,
    pub fn check(e: *Error, comptime T: type, slope: f64, actual: T) !void {
        const distance = @abs(slope - actual);
        e.probes += 1;
        e.max_absolute = @max(e.max_absolute, distance);
        e.max_relative = @max(e.max_relative, distance / @max(1e-10, @abs(slope)));
        const bound = if (T == f64) 1e-8 + 1e-6 * @abs(slope) else 2e-7 + 3e-5 * @abs(slope);
        if (!std.math.isFinite(slope) or !std.math.isFinite(actual) or distance > bound) return error.GemmaFiniteDifferenceMismatch;
    }
};
