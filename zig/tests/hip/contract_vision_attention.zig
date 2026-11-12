//! Full attention: an independent FP64 softmax over repeated key classes checks every output at 57,600 rows.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const kit = @import("contract_kit.zig");
const d = 128;
const period = 16;

fn depth(s: c.Shape) usize {
    return 128 + 6 * ((s.k + 63) / 64);
}

pub fn spec() c.Spec {
    return .{
        .name = "hip BF16 full image attention",
        .accum = .f32,
        .depth = depth,
        .ops = 16,
        .out = .f32,
        // rows counts independent images (one per call); k is both query and key count, n is head count.
        .served = &.{ .{ .rows = 1, .n = 12, .k = 64 }, .{ .rows = 1, .n = 12, .k = 57600 }, .{ .rows = 1, .n = 12, .k = 19520 }, .{ .rows = 1, .n = 12, .k = 257 }, .{ .rows = 1, .n = 12, .k = 4 } },
        .tiles = .{ .{}, .{}, .{ .size = 64 } },
    };
}

pub const Attention = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    q: []f64 = &.{},
    k: []f64 = &.{},
    v: []f64 = &.{},
    dq: u64 = 0,
    dk: u64 = 0,
    dv: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Attention {
        return .{ .rig = .{ .a = a, .r = r, .s = s }, .l = l, .arena = .init(a) };
    }
    pub fn deinit(v: *Attention) void {
        v.arena.deinit();
        v.rig.deinit();
    }
    const vt = g.hooks(Attention);
    pub fn kernel(v: *Attention) g.Kernel {
        return .{ .spec = spec(), .ctx = v, .vt = &vt };
    }

    pub fn prepare(v: *Attention, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > 8 or (s.rows > 1 and s.k > 64) or s.k == 0 or s.k > 57600 or s.n == 0 or s.n > 256) return false;
        v.rig.reset();
        _ = v.arena.reset(.retain_capacity);
        const a = v.arena.allocator();
        v.shape = s;
        v.fill = fill;
        const small = s.n * period * d;
        v.q = try a.alloc(f64, small);
        v.k = try a.alloc(f64, small);
        v.v = try a.alloc(f64, small);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (v.q, v.k, v.v) |*q, *k, *val| {
            q.* = if (fill == .stress) 0 else kit.round(.bf16, rng.float(f64) - 0.5);
            k.* = if (fill == .stress) 0 else kit.round(.bf16, rng.float(f64) - 0.5);
            val.* = kit.round(.bf16, rng.float(f64) * 2 - 1);
        }
        if (fill == .nonfinite) {
            v.q[0] = std.math.nan(f64);
            if (s.n > 1) v.k[period * d + 3] = std.math.inf(f64);
            if (s.n > 2) v.v[2 * period * d + 7] = std.math.inf(f64);
        }
        const n = s.rows * s.k * s.n * d;
        const q = try a.alloc(f32, n);
        const k = try a.alloc(u16, n);
        const values = try a.alloc(u16, n);
        for (0..s.rows) |image| for (0..s.n) |head| for (0..s.k) |row| for (0..d) |j| {
            const src = (head * period + row % period) * d + j;
            const dst = ((image * s.n + head) * s.k + row) * d + j;
            q[dst] = @floatCast(v.q[src]);
            k[dst] = c.toBf16(@floatCast(v.k[src]));
            values[dst] = c.toBf16(@floatCast(if (fill == .stress) stressValue(row, s.k) else v.v[src]));
        };
        v.dq = try v.rig.input(std.mem.sliceAsBytes(q));
        v.dk = try v.rig.input(std.mem.sliceAsBytes(k));
        v.dv = try v.rig.input(std.mem.sliceAsBytes(values));
        v.out = try v.rig.output(n * 4);
        return true;
    }

    fn stressValue(row: usize, span: usize) f64 {
        return if (row < span / 2) 1024 else if (row == span - 1) -1016 else -1024;
    }

    pub fn run(v: *Attention) !void {
        var arena = hip.Arena.counting();
        const o: hip.ops.Ops = .{ .l = v.l, .bf16 = true, .stream = v.rig.s.handle, .arena = &arena };
        const stride = v.shape.k * v.shape.n * d;
        for (0..v.shape.rows) |image| try hip.vision.fullAttention(o, v.dq + image * stride * 4, v.dk + image * stride * 2, v.dv + image * stride * 2, v.out + image * stride * 4, v.shape.k, v.shape.n, d);
        try v.rig.s.synchronize();
    }
    pub fn count(v: *Attention) usize {
        return v.shape.rows * v.shape.k * v.shape.n * d;
    }

    /// Repetition compresses the FP64 computation, not its coverage. Each query and key class has its exact count.
    pub fn reference(v: *Attention, want: []f64, mag: []f64) !void {
        const s = v.shape;
        if (v.fill == .stress) {
            var sum: f64 = 0;
            for (0..s.k) |row| sum += stressValue(row, s.k);
            @memset(want, sum / @as(f64, @floatFromInt(s.k)));
            @memset(mag, 1024);
            return;
        }
        for (0..s.n) |head| for (0..@min(s.k, period)) |query| {
            var scores: [period]f64 = @splat(-std.math.inf(f64));
            var max: f64 = -std.math.inf(f64);
            for (0..@min(s.k, period)) |key| {
                var dot: f64 = 0;
                for (0..d) |j| dot += v.q[(head * period + query) * d + j] * v.k[(head * period + key) * d + j];
                scores[key] = dot / @sqrt(@as(f64, d));
                if (std.math.isNan(scores[key]) or scores[key] > max) max = scores[key];
            }
            var denominator: f64 = 0;
            for (&scores, 0..) |*score, key| {
                const multiplicity = s.k / period + @intFromBool(key < s.k % period);
                score.* = @exp(score.* - max) * @as(f64, @floatFromInt(multiplicity));
                denominator += score.*;
            }
            for (0..d) |j| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (scores, 0..) |score, key| {
                    const term = score * v.v[(head * period + key) * d + j];
                    sum += term;
                    abs += @abs(term);
                }
                var row = query;
                while (row < s.k) : (row += period) {
                    for (0..s.rows) |image| {
                        const dst = ((image * s.n + head) * s.k + row) * d + j;
                        want[dst] = sum / denominator;
                        mag[dst] = abs / denominator;
                    }
                }
            }
        };
    }

    /// Uniform softmax is exactly one before division. These integer values make FP32 sums exact in every order.
    pub fn simulate(v: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (v.fill != .stress) return error.Invalid;
        const terms = try v.rig.a.alloc(f64, v.shape.k);
        defer v.rig.a.free(terms);
        for (terms, 0..) |*t, row| t.* = stressValue(row, terms.len);
        const sum = core.contract.rows.sumIn(acc, order, terms);
        @memset(out, sum / @as(f64, @floatFromInt(terms.len)));
    }

    pub fn bounds(v: *Attention, want: []const f64, mag: []const f64, allow: []f64) !void {
        for (want, mag, allow) |w, m, *out| {
            out.* = if (v.fill == .stress) c.boundExact(spec(), w) else c.bound(spec(), v.shape, w, m) + 0x1p-8 * m * (1 + 0x1p-23);
        }
    }
};
