//! Independent binary64 QSA RMS and rotary nodes use the exact stored F32 frequency table.
const std = @import("std");
const ct = @import("core").contract;
const c = ct.spec;
const V = @import("rounding_bound.zig").Value;
const qsa = @import("cuda_attention.zig");
pub const Part = enum { query, key, index, pool };
pub const geometry: qsa.Geometry = .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 8, .top = 128 };
pub const Fixture = struct {
    part: Part,
    rows: usize,
    inputs: [6][]align(16) u8 = @splat(&.{}),
    count: usize,
    want: []f64,
    errors: []f64,
    pub fn make(a: std.mem.Allocator, part: Part, rows: usize, fill: ct.gates.Fill, seed: u64) !Fixture {
        const g = geometry;
        const pool = part == .pool;
        const heads = switch (part) {
            .query => g.heads,
            .key => g.kv,
            .index => g.index_heads,
            .pool => 1,
        };
        const width: usize = if (part == .index or pool) 128 else 256;
        var f: Fixture = .{ .part = part, .rows = rows, .count = if (pool) 4 else 6, .want = try a.alloc(f64, rows * heads * width), .errors = try a.alloc(f64, rows * heads * width) };
        const lengths: []const usize = if (pool) &.{ (rows + 3) * g.ratio * 128 * 2, 128 * 4, 4, (3 + 32) * 4 } else &.{ rows * g.projectionWidth() * 2, 256 * 4, 256 * 4, 128 * 4, rows * 4, (3 + 32) * 4 };
        var rng = std.Random.DefaultPrng.init(seed);
        for (f.inputs[0..f.count], lengths, 0..) |*bytes, n, i| {
            bytes.* = try a.alignedAlloc(u8, .@"16", n);
            @memset(bytes.*, 0);
            if (i == 0) {
                for (std.mem.bytesAsSlice(u16, bytes.*), 0..) |*v, d| v.* = c.toBf16(if (fill == .stress) @floatFromInt(64 + d % 32) else rng.random().float(f32) * 2 - 1);
            } else if ((pool and i == 1) or (!pool and i < 4)) @memset(std.mem.bytesAsSlice(f32, bytes.*), 1);
        }
        const rope = std.mem.bytesAsSlice(f32, f.inputs[if (pool) 3 else 5]);
        rope[1] = 1;
        rope[2] = 1;
        for (rope[3..], 0..) |*v, i| v.* = @floatCast(std.math.pow(f64, 10000, -@as(f64, @floatFromInt(i)) / 32));
        if (pool) std.mem.bytesAsSlice(i32, f.inputs[2])[0] = 3 else {
            for (std.mem.bytesAsSlice(i32, f.inputs[4]), 0..) |*v, i| v.* = @intCast(3 + i);
        }
        if (fill == .nonfinite) {
            const x = std.mem.bytesAsSlice(u16, f.inputs[0]);
            const start = if (pool) 3 * g.ratio * 128 else (rows - 1) * g.projectionWidth();
            const at = start + switch (part) {
                .query, .pool => @as(usize, 0),
                .key => 2 * g.heads * 256,
                .index => 2 * (g.heads + g.kv) * 256,
            };
            for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |v, i| x[at + i] = v;
        }
        try f.compute(.none, .sequential, f.want, f.errors);
        return f;
    }
    pub fn compute(f: Fixture, acc: c.Accum, order: ct.rows.Order, out: []f64, errors: []f64) !void {
        const g = geometry;
        const pool = f.part == .pool;
        const width: usize = if (f.part == .index or pool) 128 else 256;
        const heads = switch (f.part) {
            .query => g.heads,
            .key => g.kv,
            .index => g.index_heads,
            .pool => 1,
        };
        const scale = f.inputs[
            if (pool) 1 else switch (f.part) {
                .query => @as(usize, 1),
                .key => 2,
                .index => 3,
                .pool => unreachable,
            }
        ];
        const rope = f.inputs[if (pool) 3 else 5];
        for (0..f.rows) |r| for (0..heads) |head| {
            var x: [256]V = undefined;
            var squares: [256]V = undefined;
            var normed: [256]V = undefined;
            for (x[0..width], squares[0..width], 0..) |*v, *sq, d| {
                if (pool) {
                    var terms: [8]V = undefined;
                    for (&terms, 0..) |*t, j| t.* = V.exact(c.read(.bf16, f.inputs[0], ((r + 3) * g.ratio + j) * 128 + d));
                    v.* = sum(&terms, acc, order).div(V.exact(@as(f64, 8))).bf();
                } else {
                    const base = r * g.projectionWidth() + switch (f.part) {
                        .query => head * 512,
                        .key => 2 * g.heads * 256 + head * 256,
                        .index => 2 * (g.heads + g.kv) * 256 + head * 128,
                        .pool => unreachable,
                    };
                    v.* = V.exact(c.read(.bf16, f.inputs[0], base + d));
                }
                sq.* = v.mul(v.*);
            }
            const inv = sum(squares[0..width], acc, order).div(V.exact(@as(f64, @floatFromInt(width)))).add(V.exact(@as(f32, 1e-6)), .none).rsqrt();
            for (x[0..width], normed[0..width], 0..) |v, *y, d| y.* = v.mul(inv).mul(V.exact(c.read(.f32, scale, d))).bf();
            const position: f64 = if (pool) @floatFromInt((r + 3) * g.ratio) else @floatFromInt(std.mem.bytesAsSlice(i32, f.inputs[4])[r]);
            for (normed[0..width], 0..) |v, d| {
                var y = v;
                if (d < 64) {
                    const i = d % 32;
                    const angle = V.exact(position).mul(V.exact(c.read(.f32, rope, 3 + i)));
                    const co = trig(angle, false);
                    const si = trig(angle, true);
                    y = if (d < 32) normed[d].mul(co).add(normed[d + 32].mul(si).neg(), .none) else normed[d - 32].mul(si).add(normed[d].mul(co), .none);
                }
                y = y.bf();
                const at = (r * heads + head) * width + d;
                out[at] = y.v;
                errors[at] = y.e;
            }
        };
    }
};
fn trig(x: V, sine: bool) V {
    const v = if (sine) @sin(x.v) else @cos(x.v);
    return .{ .v = v, .e = x.e + 4 * 0x1p-24 + 0x1p-149 };
}
fn sum(terms: []const V, acc: c.Accum, order: ct.rows.Order) V {
    var raw: [256]f64 = undefined;
    var e: f64 = 0;
    var mag: f64 = 0;
    for (terms, raw[0..terms.len]) |v, *x| {
        x.* = v.v;
        e += v.e;
        mag += @abs(v.v) + v.e;
    }
    const value = ct.rows.sumIn(acc, order, raw[0..terms.len]);
    return .{ .v = value, .e = if (std.math.isFinite(value)) e + c.gamma(@floatFromInt(terms.len), c.unit(.f32)) * mag + 0x1p-149 else 0 };
}
