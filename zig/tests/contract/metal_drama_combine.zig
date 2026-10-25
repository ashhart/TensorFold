//! CUDA-node-compatible NVFP4 combine: independent FP64 nodes and separate pair/shared-dot sum stress.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const bfr = contract.bf16_round;
const rig_ = @import("rig.zig");
const source = @import("metal_drama3.zig").source;
const c = contract.spec;
const g = contract.gates;
pub const Variant = enum { pairs, shared_gate };
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 13;
}

pub fn spec(v: Variant) c.Spec {
    return .{
        .name = switch (v) {
            .pairs => "metal drama3 d3_nvfp4_combine pair sums",
            .shared_gate => "metal drama3 d3_nvfp4_combine shared gate sums",
        },
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .in_place = true,
        .tiles = .{ .{}, .{}, .{ .size = 256 } },
        .served = &.{ .{ .rows = 8, .n = 2048, .k = 2048 }, .{ .rows = 1, .n = 2048, .k = 2048 }, .{ .rows = 168, .n = 2048, .k = 2048 } },
    };
}

const Range = struct {
    lo: f64,
    hi: f64,
    fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    fn around(v: f64, e: f64) Range {
        return .{ .lo = v - e, .hi = v + e };
    }
    fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    fn scale(a: Range, b: f64) Range {
        return if (b >= 0) .{ .lo = a.lo * b, .hi = a.hi * b } else .{ .lo = a.hi * b, .hi = a.lo * b };
    }
    fn fp32(a: Range) Range {
        const err = c.unit(.f32) * @max(@abs(a.lo), @abs(a.hi)) + 0x1p-149;
        return .{ .lo = a.lo - err, .hi = a.hi + err };
    }
    fn bf16(a: Range) Range {
        return .{ .lo = rounded(a.lo), .hi = rounded(a.hi) };
    }
};
fn rounded(v: f64) f64 {
    return bfr.bf16(@as(f32, @floatCast(v)));
}
fn sigmoid(v: f64) f64 {
    return 1 / (1 + @exp(-v));
}

pub const Combine = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    variant: Variant,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    old: []u16 = &.{},
    shared: []u16 = &.{},
    x: []u16 = &.{},
    wg: []u16 = &.{},
    down: []f32 = &.{},
    weights: []f32 = &.{},
    refs: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, variant: Variant) !Combine {
        const src = try source(a, "drama3 nvfp4 d3_nvfp4_combine");
        defer a.free(src);
        return .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .variant = variant, .pipe = try rig_.pipeline(device, src, "d3_nvfp4_combine") };
    }
    pub fn deinit(t: *Combine) void {
        t.arena.deinit();
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Combine) g.Kernel {
        return .{ .spec = spec(t.variant), .ctx = t, .vt = &vt };
    }
    const vt = blk: {
        var hooks = g.hooks(Combine);
        hooks.overlap = overlap;
        break :blk hooks;
    };
    fn overlap(_: *anyopaque) bool {
        return true;
    }

    pub fn prepare(t: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k == 0 or shape.k > 2048 or shape.n != shape.k) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        t.shape = shape;
        t.fill = fill;
        const a = t.arena.allocator();
        const n = t.count();
        t.old = try a.alloc(u16, n);
        t.shared = try a.alloc(u16, n);
        t.x = try a.alloc(u16, n);
        t.wg = try a.alloc(u16, shape.k);
        t.down = try a.alloc(f32, n * 8);
        t.weights = try a.alloc(f32, shape.rows * 8);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.old, t.shared, t.x) |*old, *shared, *x| {
            old.* = c.toBf16(rng.float(f32) * 2 - 1);
            shared.* = c.toBf16(rng.float(f32) * 2 - 1);
            x.* = c.toBf16((rng.float(f32) * 2 - 1) / 8);
        }
        for (t.wg) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) / 8);
        for (t.down) |*v| v.* = rng.float(f32) * 8 - 4;
        for (t.weights) |*v| v.* = (1 + rng.float(f32)) / 16;
        if (fill == .stress) {
            @memset(t.old, 0);
            @memset(t.shared, c.toBf16(if (t.variant == .shared_gate) 2 else 0));
            @memset(t.x, 0);
            @memset(t.wg, c.toBf16(1));
            @memset(t.down, 0);
            @memset(t.weights, 0.125);
            if (t.variant == .pairs) {
                const terms = [_]f32{ 65536, 1, 1, 1, 1, 1, 1, -65536 };
                for (0..shape.rows) |row| for (terms, 0..) |v, slot|
                    @memset(t.down[(row * 8 + slot) * shape.k ..][0..shape.k], v * 8);
            } else {
                // Exact sum 32, magnitude 2^20+32: FP16 overflows, BF16 loses the central ones in all orders.
                for (0..shape.rows) |row| for (0..64) |i| {
                    t.x[row * shape.k + i] = c.toBf16(if (i < 16) 32768 else if (i < 48) 1 else -32768);
                };
            }
        }
        if (fill == .nonfinite) {
            const values = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (values, 0..) |v, row| {
                t.down[(row * 8) * shape.k] = v;
                t.shared[row * shape.k + 1] = c.toBf16(v);
                t.old[row * shape.k + 2] = c.toBf16(v);
                t.x[row * shape.k + 3] = c.toBf16(v);
            }
            t.x[3 * shape.k] = c.toBf16(std.math.inf(f32));
            t.x[3 * shape.k + 1] = c.toBf16(-std.math.inf(f32));
            t.wg[0] = c.toBf16(1);
            t.wg[1] = c.toBf16(1);
            t.wg[3] = c.toBf16(1);
        }
        t.out = try t.rig.output(n * 2);
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.down));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.weights));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(t.shared));
        t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(t.wg));
        return true;
    }

    pub fn run(t: *Combine) !void {
        // Residual alone is allowed in place. Restore its original payload before each independent repeat.
        @memcpy(t.rig.bytes(t.out, t.count() * 2), std.mem.sliceAsBytes(t.old));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.out.buf, t.out.off, 0);
        for (t.refs, 1..) |ref, slot| p.e.setBuffer(ref.buf, ref.off, slot);
        p.e.setValue(@as(u32, @intCast(t.shape.k)), 6);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }
    pub fn count(t: *Combine) usize {
        return t.shape.rows * t.shape.k;
    }

    fn dot(t: *Combine, row: usize) struct { value: f64, magnitude: f64 } {
        var value: f64 = 0;
        var magnitude: f64 = 0;
        for (t.x[row * t.shape.k ..][0..t.shape.k], t.wg) |x, w| {
            const term = @as(f64, c.fromBf16(x)) * c.fromBf16(w);
            value += term;
            magnitude += @abs(term);
        }
        return .{ .value = value, .magnitude = magnitude };
    }

    fn pairs(t: *Combine, row: usize, col: usize, terms: *[8]f64) void {
        for (terms, 0..) |*term, slot| {
            const product = @as(f64, t.weights[row * 8 + slot]) * t.down[(row * 8 + slot) * t.shape.k + col];
            term.* = rounded(product);
        }
    }
    fn finish(t: *Combine, row: usize, col: usize, gate: f64, sum: f64) f64 {
        const i = row * t.shape.k + col;
        const moe = rounded(gate * c.fromBf16(t.shared[i]) + rounded(sum));
        return rounded(@as(f64, c.fromBf16(t.old[i])) + moe);
    }

    pub fn reference(t: *Combine, want: []f64, mag: []f64) !void {
        for (0..t.shape.rows) |row| {
            const d = t.dot(row);
            for (0..t.shape.k) |col| {
                var terms: [8]f64 = undefined;
                t.pairs(row, col, &terms);
                var sum: f64 = 0;
                var magnitude: f64 = 0;
                for (terms) |term| {
                    sum += term;
                    magnitude += @abs(term);
                }
                want[row * t.shape.k + col] = t.finish(row, col, sigmoid(d.value), sum);
                mag[row * t.shape.k + col] = magnitude + d.magnitude;
            }
        }
    }

    fn range(t: *Combine, row: usize, col: usize, gate: Range) Range {
        var sum = Range.point(0);
        var magnitude: f64 = 0;
        for (0..8) |slot| {
            const product = @as(f64, t.weights[row * 8 + slot]) * t.down[(row * 8 + slot) * t.shape.k + col];
            // A single FP32 product has an exact independently rounded value; CUDA rounds it again to BF16.
            const pair = rounded(product);
            sum = sum.add(Range.point(pair));
            magnitude += @abs(pair);
        }
        const error_sum = c.gamma(8, c.unit(.f32)) * magnitude;
        sum = Range.around(sum.lo, error_sum).bf16();
        const i = row * t.shape.k + col;
        const shared = gate.scale(c.fromBf16(t.shared[i])).fp32();
        const moe = shared.add(sum).fp32().bf16();
        return moe.add(Range.point(c.fromBf16(t.old[i]))).fp32().bf16();
    }

    pub fn intervals(t: *Combine, low: []f64, high: []f64) !void {
        if (t.fill == .stress) {
            // Integer sums/products are exact in FP32. sigmoid(32)*2 rounds to BF16 2, far from a boundary.
            const exact: f64 = if (t.variant == .pairs) 6 else 2;
            @memset(low, exact);
            @memset(high, exact);
            return;
        }
        for (0..t.shape.rows) |row| {
            const d = t.dot(row);
            const error_dot = c.gamma(@floatFromInt(depth(t.shape)), c.unit(.f32)) * d.magnitude;
            const lo = sigmoid(d.value - error_dot);
            const hi = sigmoid(d.value + error_dot);
            // precise exp's two ulps, the denominator addition and reciprocal; propagate through every BF16 node.
            const error_gate = c.gamma(6, c.unit(.f32)) * hi;
            const gate = Range{ .lo = @max(0, lo - error_gate), .hi = @min(1, hi + error_gate) };
            for (0..t.shape.k) |col| {
                const r = t.range(row, col, gate);
                low[row * t.shape.k + col] = r.lo;
                high[row * t.shape.k + col] = r.hi;
            }
        }
    }

    pub fn bounds(t: *Combine, want: []const f64, _: []const f64, allow: []f64) !void {
        const ranges = try t.a.alloc(f64, t.count() * 2);
        defer t.a.free(ranges);
        const low = ranges[0..t.count()];
        const high = ranges[t.count()..];
        try t.intervals(low, high);
        for (allow, want, low, high) |*v, w, lo, hi| v.* = @max(@abs(w - lo), @abs(hi - w));
    }

    pub fn simulate(t: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(terms);
        for (0..t.shape.rows) |row| {
            for (terms, t.wg, 0..) |*v, w, i|
                v.* = @as(f64, c.fromBf16(t.x[row * t.shape.k + i])) * c.fromBf16(w);
            const gate = sigmoid(contract.rows.sumIn(acc, order, terms));
            for (0..t.shape.k) |col| {
                var pairs_: [8]f64 = undefined;
                t.pairs(row, col, &pairs_);
                out[row * t.shape.k + col] = t.finish(row, col, gate, contract.rows.sumIn(acc, order, &pairs_));
            }
        }
    }
};
