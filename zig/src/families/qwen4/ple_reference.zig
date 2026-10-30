//! Independent binary64 PLE nodes retain BF16 storage cuts and propagate F32 reduction envelopes.
const std = @import("std");
const ct = @import("core").contract;
const c = ct.spec;
const V = @import("rounding_bound.zig").Value;
pub const Part = enum { gated, normed, output, tail };
pub const Shape = struct { rows: usize, hidden: usize, streams: usize, taps: usize = 4, dilation: usize = 3, eps: f32 = 1e-6 };
pub const Fixture = struct {
    p: Shape,
    part: Part,
    inputs: [7][]align(16) u8,
    want: []f64,
    error_bound: []f64,
    pub fn make(a: std.mem.Allocator, p: Shape, part: Part, fill: ct.gates.Fill, seed: u64) !Fixture {
        const w = p.hidden * p.streams;
        const tail = (p.taps - 1) * p.dilation;
        var f: Fixture = .{ .p = p, .part = part, .inputs = undefined, .want = undefined, .error_bound = undefined };
        var rng = std.Random.DefaultPrng.init(seed);
        const sizes = [_]usize{ p.rows * (w + p.hidden) * 2, p.rows * w * 2, w * 4, w * 4, w * 4, w * p.taps * 4, tail * w * 2 };
        for (&f.inputs, sizes, 0..) |*bytes, n, i| {
            bytes.* = try a.alignedAlloc(u8, .@"16", n);
            if (i < 2 or i == 6) {
                for (std.mem.bytesAsSlice(u16, bytes.*)) |*v| v.* = c.toBf16(if (fill == .stress) 32 else (rng.random().float(f32) * 2 - 1) / 4);
            } else {
                for (std.mem.bytesAsSlice(f32, bytes.*)) |*v| v.* = if (i == 5) c.fromBf16(c.toBf16((rng.random().float(f32) * 2 - 1) / 128)) else 1;
            }
        }
        if (fill == .nonfinite) {
            const x = std.mem.bytesAsSlice(u16, f.inputs[0]);
            const at = (p.rows - 1) * (w + p.hidden) + w;
            for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |v, i| x[at + i] = v;
        }
        const count = p.rows * w * (if (part == .tail) tail else 1);
        f.want = try a.alloc(f64, count);
        f.error_bound = try a.alloc(f64, count);
        try f.compute(a, .none, .sequential, f.want, f.error_bound);
        return f;
    }
    pub fn negativeSilu(a: std.mem.Allocator) !Fixture {
        const f = try make(a, .{ .rows = 1, .hidden = 256, .streams = 2 }, .output, .random, 0x367);
        for (f.inputs) |bytes| @memset(bytes, 0);
        for (f.inputs[2..5]) |bytes| @memset(std.mem.bytesAsSlice(f32, bytes), 1);
        @memset(std.mem.bytesAsSlice(u16, f.inputs[6]), c.toBf16(-89));
        const cw = std.mem.bytesAsSlice(f32, f.inputs[5]);
        for (0..512) |d| cw[d * 4] = 1;
        try f.compute(a, .none, .sequential, f.want, f.error_bound);
        return f;
    }
    pub fn denominator(a: std.mem.Allocator) !Fixture {
        const f = try make(a, .{ .rows = 1, .hidden = 5120, .streams = 4 }, .gated, .random, 0x367);
        for (f.inputs) |bytes| @memset(bytes, 0);
        @memset(std.mem.bytesAsSlice(u16, f.inputs[0]), c.toBf16(1));
        for (std.mem.bytesAsSlice(u16, f.inputs[1]), 0..) |*v, i| v.* = c.toBf16(if (i % 5120 < 201) 1 else -1);
        for (f.inputs[2..5]) |bytes| @memset(std.mem.bytesAsSlice(f32, bytes), 1);
        try f.compute(a, .none, .sequential, f.want, f.error_bound);
        return f;
    }
    fn value(f: Fixture, input: usize, at: usize) V {
        return V.exact(c.read(if (input < 2 or input == 6) .bf16 else .f32, f.inputs[input], at));
    }
    fn norm(a: std.mem.Allocator, x: []const V, scale: []const V, eps: f32, acc: c.Accum, order: ct.rows.Order) ![]V {
        const squares = try a.alloc(V, x.len);
        for (x, squares) |v, *sq| sq.* = v.mul(v);
        const inv = (try sum(a, squares, acc, order)).div(V.exact(@as(f64, @floatFromInt(x.len)))).add(V.exact(eps), .none).rsqrt();
        const out = try a.alloc(V, x.len);
        for (x, scale, out) |v, s, *y| y.* = v.mul(inv).mul(s).bf();
        return out;
    }
    pub fn compute(f: Fixture, a: std.mem.Allocator, acc: c.Accum, order: ct.rows.Order, out: []f64, errors: []f64) !void {
        const p = f.p;
        const w = p.hidden * p.streams;
        const tail_rows = (p.taps - 1) * p.dilation;
        const gated = try a.alloc(V, p.rows * w);
        const normed = try a.alloc(V, p.rows * w);
        const hidden = try a.alloc(V, p.rows * w);
        const scale = try a.alloc(V, w * 3);
        for (0..3) |i| {
            for (scale[i * w ..][0..w], 0..) |*v, d| v.* = f.value(i + 2, d);
        }
        for (0..p.rows) |r| for (0..p.streams) |s| {
            const key = try a.alloc(V, p.hidden);
            const query = try a.alloc(V, p.hidden);
            for (key, query, 0..) |*k, *q, d| {
                k.* = f.value(0, r * (w + p.hidden) + s * p.hidden + d);
                q.* = f.value(1, r * w + s * p.hidden + d);
            }
            const kn = try norm(a, key, scale[s * p.hidden ..][0..p.hidden], p.eps, acc, order);
            const qn = try norm(a, query, scale[w + s * p.hidden ..][0..p.hidden], p.eps, acc, order);
            const products = try a.alloc(V, p.hidden);
            for (kn, qn, products) |k, q, *v| v.* = k.mul(q).bf();
            const dot = (try sum(a, products, acc, order)).bf().div(V.exact(@sqrt(@as(f64, @floatFromInt(p.hidden))))).bf();
            const gate = signedRoot(dot).bf().sig().bf();
            const at = r * w + s * p.hidden;
            for (gated[at..][0..p.hidden], 0..) |*v, d| v.* = gate.mul(f.value(0, r * (w + p.hidden) + w + d)).bf();
            @memcpy(normed[at..][0..p.hidden], try norm(a, gated[at..][0..p.hidden], scale[2 * w + s * p.hidden ..][0..p.hidden], p.eps, acc, order));
        };
        for (0..p.rows) |r| for (0..w) |d| {
            const terms = try a.alloc(V, p.taps);
            for (terms, 0..) |*v, t| {
                const at = r + t * p.dilation;
                const input = if (at < tail_rows) f.value(6, at * w + d) else normed[(at - tail_rows) * w + d];
                v.* = input.mul(f.value(5, d * p.taps + t));
            }
            const act = (try sum(a, terms, acc, order)).bf().silu().bf();
            hidden[r * w + d] = f.value(1, r * w + d).add(gated[r * w + d].add(act, .none).bf(), .none).bf();
        };
        if (f.part == .tail) {
            for (0..p.rows) |r| for (0..tail_rows) |t| for (0..w) |d| {
                const at = r + t + 1;
                const v = if (at < tail_rows) f.value(6, at * w + d) else normed[(at - tail_rows) * w + d];
                const i = (r * tail_rows + t) * w + d;
                out[i] = v.v;
                errors[i] = v.e;
            };
        } else {
            const selected = switch (f.part) {
                .gated => gated,
                .normed => normed,
                .output => hidden,
                .tail => unreachable,
            };
            for (selected, out, errors) |v, *o, *e| {
                o.* = v.v;
                e.* = v.e;
            }
        }
    }
};
fn root(x: f64) f64 {
    return if (std.math.isNan(x)) x else if (x == 0) 0 else std.math.copysign(@sqrt(@max(@abs(x), 1e-6)), x);
}
fn signedRoot(x: V) V {
    const v = root(x.v);
    return .{ .v = v, .e = @max(@abs(root(x.v - x.e) - v), @abs(root(x.v + x.e) - v)) + 4 * 0x1p-24 * @abs(v) };
}
fn sum(a: std.mem.Allocator, terms: []const V, acc: c.Accum, order: ct.rows.Order) !V {
    const raw = try a.alloc(f64, terms.len);
    var error_before: f64 = 0;
    var mag: f64 = 0;
    for (terms, raw) |v, *x| {
        x.* = v.v;
        error_before += v.e;
        mag += @abs(v.v) + v.e;
    }
    const v = ct.rows.sumIn(acc, order, raw);
    return .{ .v = v, .e = if (std.math.isFinite(v)) error_before + c.gamma(@floatFromInt(terms.len), c.unit(.f32)) * mag + 0x1p-149 else 0 };
}
