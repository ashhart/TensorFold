//! Independent f64 convolution, normalization and delta recurrence, with absolute F32 error propagation.
const std = @import("std");
const ct = @import("core").contract;
const c = ct.spec;
const gdn = @import("cuda_gdn.zig");
const V = @import("rounding_bound.zig").Value;
pub const Part = enum { readout, state, conv };
pub const Fixture = struct {
    p: gdn.Params,
    part: Part,
    inputs: [7][]align(16) u8,
    want: []f64,
    error_bound: []f64,
    pub fn outputBytes(f: Fixture) usize {
        return f.want.len * @as(usize, if (f.part == .state) 4 else 2);
    }
    pub fn negativeSilu(a: std.mem.Allocator) !Fixture {
        const f = try make(a, .{ .rows = 1, .keys = 1, .values = 2, .eps = 1e-6, .gate = .sigmoid, .weight_flags = 11 }, .readout, .random, 0x367);
        for (f.inputs) |bytes| @memset(bytes, 0);
        const x = std.mem.bytesAsSlice(u16, f.inputs[0]);
        x[0] = c.toBf16(32);
        x[128] = c.toBf16(32);
        @memset(x[256..512], c.toBf16(-89));
        @memset(x[512..768], c.toBf16(1));
        @memset(x[768..770], c.toBf16(8));
        const cw = std.mem.bytesAsSlice(f32, f.inputs[2]);
        for (0..512) |i| cw[i * 4 + 3] = 1;
        @memset(std.mem.bytesAsSlice(f32, f.inputs[6]), 1);
        try f.compute(a, .none, .sequential, f.want, f.error_bound);
        return f;
    }
    pub fn make(a: std.mem.Allocator, p: gdn.Params, part: Part, fill: ct.gates.Fill, seed: u64) !Fixture {
        const sizes = try gdn.sizes(p);
        var f: Fixture = .{ .p = p, .part = part, .inputs = undefined, .want = undefined, .error_bound = undefined };
        const lengths = [_]usize{ sizes.projection, sizes.conv_state, sizes.conv_weight, sizes.state, sizes.scalar, sizes.scalar, sizes.norm };
        var rng = std.Random.DefaultPrng.init(seed);
        for (&f.inputs, lengths, 0..) |*bytes, len, i| {
            bytes.* = try a.alignedAlloc(u8, .@"16", len);
            if (i < 2) {
                for (std.mem.bytesAsSlice(u16, bytes.*)) |*v| v.* = c.toBf16((rng.random().float(f32) * 2 - 1) / 8);
            } else {
                for (std.mem.bytesAsSlice(f32, bytes.*)) |*v| {
                    v.* = if (i == 6) 1 else if (i == 4) -2 else (rng.random().float(f32) * 2 - 1) / 128;
                    if (i == 2 and p.weight_flags & 8 != 0) v.* = c.fromBf16(c.toBf16(v.*));
                }
            }
        }
        if (fill == .nonfinite) {
            const last = (p.rows - 1) * (p.keys * 256 + p.values * 258);
            for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }, 0..) |v, i| std.mem.bytesAsSlice(u16, f.inputs[0])[last + i] = v;
            std.mem.bytesAsSlice(u16, f.inputs[0])[last + p.keys * 128] = 0x7fc0;
        }
        if (fill == .stress) {
            const ch = p.keys * 256 + p.values * 128;
            const pw = ch + p.values * 130;
            const x = std.mem.bytesAsSlice(u16, f.inputs[0]);
            @memset(std.mem.bytesAsSlice(u16, f.inputs[1]), 0);
            @memset(std.mem.bytesAsSlice(f32, f.inputs[3]), 0);
            @memset(std.mem.bytesAsSlice(f32, f.inputs[5]), 0);
            const cw = std.mem.bytesAsSlice(f32, f.inputs[2]);
            @memset(cw, 0);
            for (0..ch) |i| cw[i * 4 + 3] = 1;
            for (0..p.rows) |r| {
                @memset(x[r * pw ..][0..ch], c.toBf16(32));
                @memset(x[r * pw + ch ..][0 .. p.values * 128], c.toBf16(1));
                @memset(x[r * pw + ch + p.values * 128 ..][0 .. 2 * p.values], 0);
            }
        }
        const count = if (part == .conv) p.rows * sizes.conv_state / 2 else p.rows * p.values * @as(usize, if (part == .readout) 128 else 128 * 128);
        f.want = try a.alloc(f64, count);
        f.error_bound = try a.alloc(f64, count);
        try f.compute(a, .none, .sequential, f.want, f.error_bound);
        return f;
    }
    fn value(f: Fixture, i: usize, at: usize) V {
        return V.exact(c.read(if (i < 2) .bf16 else .f32, f.inputs[i], at));
    }
    pub fn compute(f: Fixture, a: std.mem.Allocator, acc: c.Accum, order: ct.rows.Order, out: []f64, errors: []f64) !void {
        const p = f.p;
        const ch = p.keys * 256 + p.values * 128;
        const pw = ch + p.values * 130;
        if (f.part == .conv) {
            for (0..p.rows) |r| for (0..3) |tap| for (0..ch) |i| {
                const at = r + tap + 1;
                out[(r * 3 + tap) * ch + i] = if (at < 3) f.value(1, at * ch + i).v else f.value(0, (at - 3) * pw + i).v;
                errors[(r * 3 + tap) * ch + i] = 0;
            };
            return;
        }
        const recurrent = try a.alloc(V, p.values * 128 * 128);
        const front = try a.alloc(V, ch);
        const ys = try a.alloc(V, 128);
        for (recurrent, 0..) |*v, i| v.* = f.value(3, i);
        for (0..p.rows) |r| {
            for (front, 0..) |*v, i| {
                var terms: [4]V = undefined;
                for (&terms, 0..) |*t, tap| {
                    const at = r + tap;
                    const x = if (at < 3) f.value(1, at * ch + i) else f.value(0, (at - 3) * pw + i);
                    t.* = f.value(2, i * 4 + tap).mul(x);
                }
                const conv = sum(&terms, acc, order);
                v.* = (if (p.weight_flags & 8 != 0) conv.bf() else conv).silu().bf();
            }
            const q = front[0 .. p.keys * 128];
            const k = front[p.keys * 128 ..][0 .. p.keys * 128];
            for (0..p.keys) |head| for ([_][]V{ q[head * 128 ..][0..128], k[head * 128 ..][0..128] }, 0..) |v, which| {
                var inv = dot(v, v, acc, order).add(V.exact(@as(f32, 1e-6)), .none).rsqrt();
                if (which == 0) inv = inv.mul(V.exact(@as(f64, 1 / @sqrt(@as(f64, 128)))));
                for (v) |*vv| vv.* = vv.mul(inv);
            };
            for (0..p.values) |head| {
                const kh = head / (p.values / p.keys);
                const keys = k[kh * 128 ..][0..128];
                const queries = q[kh * 128 ..][0..128];
                const ap = f.value(0, r * pw + ch + p.values * 129 + head).add(f.value(5, head), .none);
                const softplus = if (ap.v > 20) ap else ap.exp().log1p();
                const decay = f.value(4, head).exp().mul(softplus).neg().exp();
                const beta = f.value(0, r * pw + ch + p.values * 128 + head).sig().bf();
                for (ys, 0..) |*y, d| {
                    const state = recurrent[(head * 128 + d) * 128 ..][0..128];
                    for (state) |*s| s.* = s.mul(decay);
                    const predicted = dot(state, keys, acc, order);
                    const correction = front[2 * p.keys * 128 + head * 128 + d].add(predicted.neg(), .none).mul(beta);
                    for (state, keys) |*s, key| s.* = s.add(key.mul(correction), .none);
                    y.* = dot(state, queries, acc, order).bf();
                    if (f.part == .state) for (state, 0..) |s, i| {
                        const at = ((r * p.values + head) * 128 + d) * 128 + i;
                        out[at] = s.v;
                        errors[at] = s.e;
                    };
                }
                const inv = dot(ys, ys, acc, order).div(V.exact(@as(f64, 128))).add(V.exact(p.eps), .none).rsqrt();
                for (ys, 0..) |y, d| {
                    var weighted = y.mul(inv).bf().mul(f.value(6, d));
                    if (p.weight_flags & 4 == 0) weighted = weighted.bf();
                    const z = f.value(0, r * pw + ch + head * 128 + d);
                    const got = weighted.mul(if (p.gate == .sigmoid) z.sig() else z.silu()).bf();
                    if (f.part == .readout) {
                        const at = (r * p.values + head) * 128 + d;
                        out[at] = got.v;
                        errors[at] = got.e;
                    }
                }
            }
        }
    }
};
fn sum(terms: []const V, acc: c.Accum, order: ct.rows.Order) V {
    var raw: [128]f64 = undefined;
    var e: f64 = 0;
    var mag: f64 = 0;
    for (terms, raw[0..terms.len]) |v, *x| {
        x.* = v.v;
        e += v.e;
        mag += @abs(v.v) + v.e;
    }
    const v = ct.rows.sumIn(acc, order, raw[0..terms.len]);
    if (!std.math.isFinite(v)) return .{ .v = v };
    return .{ .v = v, .e = e + c.gamma(@floatFromInt(terms.len), c.unit(.f32)) * mag + 0x1p-149 };
}
fn dot(left: []const V, right: []const V, acc: c.Accum, order: ct.rows.Order) V {
    var terms: [128]V = undefined;
    for (left, right, terms[0..left.len]) |x, y, *v| v.* = x.mul(y);
    return sum(terms[0..left.len], acc, order);
}
