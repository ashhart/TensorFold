//! tf_hy4_index_groups under the contract: a group's four keys pooled by a softmax gate, RMS-normed, bf16 out.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const ix = @import("metal_hy4_index.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;
const u = fl.u32_ulp;
const W = ix.width;
const S = ix.slots;

/// metal_index.zig's GroupArgs (index_build.metal's Hy4IndexGroups).
const GroupArgs = extern struct { first: u32, count: u32, capacity_groups: u32, epsilon: f32 };

/// A dot's 128 fma roundings, the bias add after the gate's; the 32 squares of a simdgroup in any order, 3 adds.
const dot_depth = W;
const square_depth = 34;
fn depth(_: c.Shape) usize {
    return dot_depth + 1;
}

/// Metal's precise exp to 4 ulps, its division to 2.5, rsqrt to 2: relative errors in units of u (half an ulp).
const exp_err = 8 * u;
const div_err = 5 * u;
const rsqrt_err = 4 * u;
/// Apple GPUs flush fp32 subnormals: an absolute floor on every step that can land there.
const tiny = 0x1p-123;

/// One column's pooled value and what its fp32 steps can move it by.
const Column = struct { y: f64, allow: f64 };

pub const Groups = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    eps: f32,
    fault: ?ix.Control,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    first: usize = 2,
    cap: usize = 0,
    keys: []u16 = &.{},
    wv: []f32 = &.{},
    wg: []f32 = &.{},
    bias: []f32 = &.{},
    nw: []f32 = &.{},
    /// The group keys before the launch, restored before every run: a group not launched keeps them.
    old: []u16 = &.{},
    /// Each output's allowance, worked out with the reference.
    allow: []f64 = &.{},
    ins: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub const base = "metal hy4 index_build tf_hy4_index_groups";
    /// Rows: the groups a pass touches; n: the 128 key columns; k: the 128 inputs each value and gate dot sums.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .ops = 12,
            .out = .bf16,
            // DFly2's window over three groups, a decode row's, an aligned window's, a prompt chunk's (both ways)
            .served = &.{ .{ .rows = 3, .n = W, .k = W }, .{ .rows = 1, .n = W, .k = W }, .{ .rows = 2, .n = W, .k = W }, .{ .rows = h.prompt_rows / S + 1, .n = W, .k = W }, .{ .rows = h.prompt_rows / S, .n = W, .k = W } },
            .sums = .{ .exact = false, .why = "the value and gate dots exact; the squares' fp32 sum is in the derived bound" },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, eps: f32, fault: ?ix.Control) !Groups {
        const pipe = try ix.pipeline(a, device, .build, "tf_hy4_index_groups", if (fault) |f| f.edit else null);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .eps = eps, .fault = fault };
    }
    pub fn deinit(t: *Groups) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Groups) void {
        inline for (.{ "keys", "wv", "wg", "bias", "nw", "old", "allow" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Groups) g.Kernel {
        t.name = fl.named(base, if (t.fault) |f| f.drops else null, &t.label);
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = if (t.fault) |f| f.gate else null };
    }
    const vt = g.hooks(Groups);

    /// The launch covers groups 2.. of a cache one group longer; the groups before and after keep the old keys.
    pub fn prepare(t: *Groups, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != W or shape.k != W) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        t.cap = t.first + shape.rows + 1;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.keys = try t.a.alloc(u16, t.cap * S * W);
        t.wv = try t.a.alloc(f32, W * W);
        t.wg = try t.a.alloc(f32, W * W);
        t.bias = try t.a.alloc(f32, S * W);
        t.nw = try t.a.alloc(f32, W);
        t.old = try t.a.alloc(u16, t.cap * W);
        t.allow = try t.a.alloc(f64, t.cap * W);
        for (t.old) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.keys) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        for (t.bias) |*v| v.* = (rng.float(f32) * 2 - 1) * (if (fill == .stress) @as(f32, 2) else 1);
        for (t.nw) |*v| v.* = 0.5 + rng.float(f32);
        if (fill == .stress) t.stress(rng) else {
            for (t.wv) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.15;
            for (t.wg) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.15;
        }
        if (fill == .nonfinite and shape.rows >= 4) {
            // NaN in group 0, +Inf in group 1, -Inf in group 2, both in group 3; then +Inf and -Inf norm weights
            const at = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 0 }, .{ 3, 1 }, .{ 3, 3 } };
            const vals = [_]f32{ fl.planted[0], fl.planted[1], fl.planted[2], fl.planted[1], fl.planted[2] };
            for (at, vals, 0..) |p, v, i| t.keys[((t.first + p[0]) * S + p[1]) * W + 7 * i + 3] = c.toBf16(v);
            t.nw[5] = fl.inf32;
            t.nw[9] = -fl.inf32;
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.keys));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.wv));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.wg));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.nw));
        t.out = try t.rig.output(t.old.len * 2);
        return true;
    }

    /// Gate 4's groups: products 512 x integers past fp16's range, columns k and 127 - k cancelling but for two nudges.
    fn stress(t: *Groups, rng: std.Random) void {
        for (0..t.shape.rows) |gi| for (0..S) |s| {
            const x = t.keys[((t.first + gi) * S + s) * W ..][0..W];
            for (0..W / 2) |k| {
                const v: u32 = 128 + rng.uintLessThan(u32, 125);
                const nudge: u32 = if (k == s or k == s + 4) 1 + @as(u32, @intCast((s + k) % 3)) else 0;
                x[k] = c.toBf16(@floatFromInt(v));
                x[W - 1 - k] = c.toBf16(@floatFromInt(v + nudge));
            }
        };
        for (0..W) |col| for (0..W / 2) |k| {
            const v: f32 = if (rng.boolean()) 512 else -512;
            t.wv[col * W + k] = v;
            t.wv[col * W + W - 1 - k] = -v;
            const z: f32 = if (k < 2 * S) 0 else if (rng.boolean()) 512 else -512;
            t.wg[col * W + k] = z;
            t.wg[col * W + W - 1 - k] = -z;
        };
    }

    pub fn run(t: *Groups) !void {
        @memcpy(t.rig.bytes(t.out, t.old.len * 2), std.mem.sliceAsBytes(t.old));
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.ins, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(t.out.buf, t.out.off, 5);
        e.setValue(GroupArgs{ .first = @intCast(t.first), .count = @intCast(t.shape.rows), .capacity_groups = @intCast(t.cap), .epsilon = t.eps }, 6);
        // as metal_index.zig: a threadgroup of 128 a group
        e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(W, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Groups) usize {
        return t.cap * W;
    }

    fn xv(t: *const Groups, gi: usize, s: usize, k: usize) f64 {
        return c.fromBf16(t.keys[((t.first + gi) * S + s) * W + k]);
    }

    const Sim = struct { acc: c.Accum, order: g.Order };

    /// Group gi's outputs before their bf16 rounding (sums in `sim` if given), each with its fp32 steps' allowance.
    fn group(t: *const Groups, gi: usize, sim: ?Sim, out: *[W]f64, allow: *[W]f64) void {
        const exact = t.fill == .stress;
        var pooled: [W]f64 = undefined;
        var err: [W]f64 = undefined;
        var tv: [W]f64 = undefined;
        var tz: [W]f64 = undefined;
        for (0..W) |col| {
            var v: [S]f64 = undefined;
            var z: [S]f64 = undefined;
            var ev: [S]f64 = undefined;
            var ez: [S]f64 = undefined;
            for (0..S) |s| {
                var mv: f64 = 0;
                var mz: f64 = 0;
                var sv: f64 = 0;
                var sg: f64 = 0;
                for (0..W) |k| {
                    const x = t.xv(gi, s, k);
                    tv[k] = @as(f64, t.wv[col * W + k]) * x;
                    tz[k] = @as(f64, t.wg[col * W + k]) * x;
                    sv += tv[k];
                    sg += tz[k];
                    mv += @abs(tv[k]);
                    mz += @abs(tz[k]);
                }
                if (sim) |m| {
                    sv = contract.rows.sumIn(m.acc, m.order, &tv);
                    sg = contract.rows.sumIn(m.acc, m.order, &tz);
                }
                const b: f64 = t.bias[s * W + col];
                v[s] = sv;
                z[s] = sg + b;
                ev[s] = if (exact) 0 else c.gamma(dot_depth, u) * mv;
                ez[s] = if (exact) u * @abs(z[s]) else c.gamma(dot_depth + 1, u) * (mz + @abs(b));
            }
            const top = @max(@max(z[0], z[1]), @max(z[2], z[3]));
            var w: [S]f64 = undefined;
            for (&w, z) |*o, zs| o.* = @exp(zs - top);
            const den = ((w[0] + w[1]) + w[2]) + w[3];
            var p: f64 = 0;
            for (v, w) |vs, ws| p += vs * (ws / den);
            pooled[col] = p;
            // softmax: a weight moves by its z's error and the subtraction's rounding, twice (its share, the sum's)
            const ez_most = @max(@max(ez[0], ez[1]), @max(ez[2], ez[3]));
            var d: f64 = 0;
            for (z, ez) |zs, e| d = @max(d, e + u * (@abs(zs - top) + 2 * ez_most));
            const rp = @exp(2 * d) * (1 + exp_err) / (1 - exp_err) * (1 + c.gamma(3, u)) * (1 + div_err) - 1;
            var ep: f64 = 0;
            for (v, ev, w) |vs, e, ws| {
                const ps = ws / den;
                ep += ps * (e * (1 + rp) + @abs(vs) * rp + c.gamma(4, u) * (@abs(vs) + e) * (1 + rp)) + tiny * (@abs(vs) + e + 1);
            }
            err[col] = ep;
        }
        var sq: [W]f64 = undefined;
        var total: f64 = 0;
        var et: f64 = 0;
        var big: f64 = 0;
        for (pooled, err, &sq) |p, e, *q| {
            q.* = p * p;
            total += q.*;
            const m = @abs(p) + e;
            et += (2 * @abs(p) + e) * e + u * m * m;
            big += m * m;
        }
        if (sim) |m| total = contract.rows.sumIn(m.acc, m.order, &sq);
        et += c.gamma(square_depth, u) * big * (1 + u);
        const arg = total / W + @as(f64, t.eps);
        const inv = 1 / @sqrt(arg);
        // rsqrt's argument: the squares' error, the divide by 128 and the epsilon's add
        const ra = ((et / W) * (1 + div_err + u) + (div_err + u) * arg) / arg;
        const rinv = if (ra < 0.5) (1 + rsqrt_err) / @sqrt(1 - ra) - 1 else std.math.inf(f64);
        const f = (1 + rinv) * (1 + u) * (1 + u) - 1;
        for (0..W) |col| {
            const nw: f64 = t.nw[col];
            out[col] = nw * (pooled[col] * inv);
            const e = @abs(nw) * inv * (err[col] * (1 + f) + @abs(pooled[col]) * f) + tiny;
            allow[col] = e + 0x1p-8 * (@abs(out[col]) + e) + 0x1p-133;
        }
    }

    /// Float64 of the kernel's formula before its one bf16 rounding; a group not launched keeps the old key.
    pub fn reference(t: *Groups, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        for (0..t.cap) |gi| {
            if (gi < t.first or gi >= t.first + t.shape.rows) {
                for (0..W) |col| {
                    want[gi * W + col] = c.fromBf16(t.old[gi * W + col]);
                    t.allow[gi * W + col] = 0;
                }
                continue;
            }
            t.group(gi - t.first, null, want[gi * W ..][0..W], t.allow[gi * W ..][0..W]);
        }
    }

    pub fn bounds(t: *Groups, _: []const f64, _: []const f64, allow: []f64) !void {
        @memcpy(allow, t.allow);
    }

    pub fn simulate(t: *Groups, acc: c.Accum, order: g.Order, out: []f64) !void {
        var spare: [W]f64 = undefined;
        for (0..t.cap) |gi| {
            if (gi < t.first or gi >= t.first + t.shape.rows) {
                for (0..W) |col| out[gi * W + col] = c.fromBf16(t.old[gi * W + col]);
                continue;
            }
            t.group(gi - t.first, .{ .acc = acc, .order = order }, out[gi * W ..][0..W], &spare);
        }
    }
};
