//! The decoder's head projections under the contract: ud_qk (rotary, caches, router norm) and ud_block_keys.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_uocr_decoder_common.zig");

const c = contract.spec;
const g = contract.gates;
const H = k.hidden;
const D = k.router_dim;
const Mutation = k.Mutation;
const Range = k.Range;
const bf = k.bf;
const proj = 3 * H + 2 * D;
const clamp_at: f64 = 1e4;

/// The release's clamp: NaN stays NaN.
fn clampNan(x: f64) f64 {
    return if (std.math.isNan(x)) x else std.math.clamp(x, -clamp_at, clamp_at);
}

fn clampRange(r: Range) Range {
    return .{ .lo = std.math.clamp(r.lo, -clamp_at, clamp_at), .hi = std.math.clamp(r.hi, -clamp_at, clamp_at) };
}

/// Rotary in bf16 steps for dim d of a 128-wide head: bf16(bf16(x c) + bf16(rot s)), rot = -partner below 64.
fn rope(x: f64, partner: f64, d: usize, cs: f64, sn: f64) f64 {
    const rot = if (d < 64) -partner else partner;
    return bf(bf(x * cs) + bf(rot * sn));
}

/// The same over ranges of x and its partner: each product's bf16 point, their fp32 sum, its bf16 point.
fn ropeRange(x: Range, partner: Range, d: usize, cs: f64, sn: f64) Range {
    const rot = if (d < 64) partner.mul(Range.point(-1)) else partner;
    return x.mul(Range.point(cs)).bf16().add(rot.mul(Range.point(sn)).bf16()).fp32(1).bf16();
}

/// A thread's square, the simd_sum tree, then 4 simdgroups' partials.
fn routerDepth(_: c.Shape) usize {
    return 1 + 5 + 4;
}

pub const Qk = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    p: []u16 = &.{},
    cs: []u16 = &.{},
    sn: []u16 = &.{},
    w: [2][D]u16 = undefined,
    ins: [5]rig_.Ref = undefined,
    outs: [5]rig_.Ref = undefined,

    /// Rows: tokens; n: the joint projection (4,096). Outputs: q, the k and v cache rows, the router's q and k rows.
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr decoder ud_qk rotary page2d, router norm",
            .accum = .f32,
            .depth = routerDepth,
            .ops = 6,
            .mids = 2,
            .out = .bf16,
            .served = &.{ .{ .rows = k.prompt_rows, .n = proj, .k = 1 }, .{ .rows = 1, .n = proj, .k = 1 } },
            .sums = .{ .bf16 = 0, .why = "the router's bf16 rows follow the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps" },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !Qk {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_qk"), .m = m };
    }
    pub fn deinit(t: *Qk) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Qk) void {
        inline for (.{ "p", "cs", "sn" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Qk) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Qk);

    pub fn prepare(t: *Qk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != proj) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const rows = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.p = try t.a.alloc(u16, rows * proj);
        k.fillBf16(rng, t.p, 3);
        t.cs = try t.a.alloc(u16, rows * 64);
        t.sn = try t.a.alloc(u16, rows * 64);
        for (t.cs, t.sn) |*cv, *sv| {
            const angle = rng.float(f64) * 2 * std.math.pi;
            cv.* = c.toBf16(@floatCast(@cos(angle)));
            sv.* = c.toBf16(@floatCast(@sin(angle)));
        }
        for (&t.w) |*w| k.fillBf16(rng, w, 0.5);
        if (fill == .stress) {
            // the router's rows: integers whose squares sum exactly in fp32 (128 x 1,600) and overflow fp16
            for (0..rows) |r| for (t.p[r * proj + 3 * H ..][0 .. 2 * D]) |*v| {
                v.* = c.toBf16(@floatFromInt(@as(i32, @intCast(20 + rng.uintLessThan(u32, 21))) * (if (rng.boolean()) @as(i32, 1) else -1)));
            };
            for (&t.w) |*w| @memset(w, 0);
        }
        if (fill == .nonfinite) {
            // a q value (and so its rotary partner), a value row's, the router's q row and key row
            const at = [_][2]usize{ .{ 0, 5 }, .{ 1, 2 * H + 9 }, .{ 2, 3 * H + 3 }, .{ 3, 3 * H + D + 7 } };
            const vals = [_]f32{ k.nan32, k.inf32, k.nan32, k.inf32 };
            for (at, vals) |q, v| if (q[0] < rows) {
                t.p[q[0] * proj + q[1]] = c.toBf16(v);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.p));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.cs));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.sn));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(&t.w[0]));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(&t.w[1]));
        for (&t.outs, [_]usize{ H, H, H, D, D }) |*o, width| o.* = try t.rig.output(rows * width * 2);
        return true;
    }

    pub fn run(t: *Qk) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        for (t.outs, 5..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue(extern struct { pos0: u32, eps: f32 }{ .pos0 = 0, .eps = k.eps }, 10);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(D, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Qk) usize {
        return t.shape.rows * (3 * H + 2 * D);
    }

    fn val(t: *Qk, r: usize, i: usize) f64 {
        return c.fromBf16(t.p[r * proj + i]);
    }

    /// The router's row `which` (0 q, 1 k) of token r normed: its bf16 values, or (with `range`) their spread.
    fn routerRow(t: *Qk, r: usize, which: usize, acc: c.Accum, order: g.Order, out: *[D]f64, range: ?*[D]Range) void {
        var x: [D]f64 = undefined;
        var sq: [D]f64 = undefined;
        var ss: f64 = 0;
        for (&x, &sq, 0..) |*v, *s, d| {
            v.* = t.val(r, 3 * H + which * D + d);
            s.* = v.* * v.*;
            ss += s.*;
        }
        if (acc != .none) ss = contract.rows.sumIn(acc, order, &sq);
        const inv = 1 / @sqrt(ss / D + @as(f64, k.eps));
        for (out, x, 0..) |*o, v, d| o.* = bf(clampNan(clampNan(v * inv) * (1 + @as(f64, c.fromBf16(t.w[which][d])))));
        const rg = range orelse return;
        const e = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(routerDepth(t.shape)), c.unit(.f32)) * ss;
        const ir = (Range{ .lo = 1 / @sqrt((ss + e) / D + @as(f64, k.eps)), .hi = 1 / @sqrt((ss - e) / D + @as(f64, k.eps)) }).fp32(4);
        for (rg, x, 0..) |*o, v, d| {
            const one = Range.point(1 + @as(f64, c.fromBf16(t.w[which][d]))).fp32(1);
            o.* = clampRange(clampRange(Range.point(v).mul(ir).fp32(1)).mul(one).fp32(1)).bf16();
        }
    }

    /// Every output in order, or (with `low`) their intervals.
    fn outputs(t: *Qk, acc: c.Accum, order: g.Order, out: []f64, high: ?[]f64) void {
        var i: usize = 0;
        const rows = t.shape.rows;
        for (0..3) |part| for (0..rows) |r| for (0..H) |j| {
            const d = j % 128;
            const at = part * H + j;
            const cs: f64 = c.fromBf16(t.cs[r * 64 + d % 64]);
            const sn: f64 = c.fromBf16(t.sn[r * 64 + d % 64]);
            if (part == 2) {
                out[i] = t.val(r, at);
                if (high) |h| h[i] = out[i];
            } else if (high) |h| {
                const rr = ropeRange(Range.point(t.val(r, at)), Range.point(t.val(r, at ^ 64)), d, cs, sn);
                out[i] = rr.lo;
                h[i] = rr.hi;
            } else out[i] = rope(t.val(r, at), t.val(r, at ^ 64), d, cs, sn);
            i += 1;
        };
        var nv: [D]f64 = undefined;
        var nr: [D]Range = undefined;
        for (0..2) |which| for (0..rows) |r| {
            t.routerRow(r, which, acc, order, &nv, if (high != null) &nr else null);
            for (0..D) |d| {
                const cs: f64 = c.fromBf16(t.cs[r * 64 + d % 64]);
                const sn: f64 = c.fromBf16(t.sn[r * 64 + d % 64]);
                if (high) |h| {
                    const rr = ropeRange(nr[d], nr[d ^ 64], d, cs, sn);
                    out[i] = rr.lo;
                    h[i] = rr.hi;
                } else out[i] = rope(nv[d], nv[d ^ 64], d, cs, sn);
                i += 1;
            }
        };
    }

    pub fn reference(t: *Qk, want: []f64, mag: []f64) !void {
        t.outputs(.none, .sequential, want, null);
        @memset(mag, 0);
    }

    pub fn simulate(t: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.outputs(acc, order, out, null);
    }

    /// q, k, v exact to bf16; router rows through the norm's fp32 steps, clamps, bf16, then rotary over both spreads.
    pub fn intervals(t: *Qk, low: []f64, high: []f64) !void {
        t.outputs(.none, .sequential, low, high);
    }

    pub fn bounds(t: *Qk, want: []const f64, _: []const f64, allow: []f64) !void {
        try k.boundsFromIntervals(t.a, t, want, allow);
    }
};

// --- ud_block_keys: a block's 32 router keys weighted by a softmax of their dot with head_cls / sqrt(128) -----------

/// A key's 128-term dot, one after another.
fn keysDepth(_: c.Shape) usize {
    return D;
}

pub const BlockKeys = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    rk: []u16 = &.{},
    hc: [D]u16 = undefined,
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: blocks; n: the router's width (128); k: a block's keys (32).
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr decoder ud_block_keys blocks of 32",
            .accum = .f32,
            .depth = keysDepth,
            .out = .f32,
            // a prompt chunk's blocks, and the one block a step completes
            .served = &.{ .{ .rows = k.prompt_rows / k.block, .n = D, .k = k.block }, .{ .rows = 1, .n = D, .k = k.block } },
            // the dots sum exactly in fp32 at gate 4; the softmax's mass and the weighted keys do not
            .sums = .{ .exact = false },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !BlockKeys {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_block_keys"), .m = m };
    }
    pub fn deinit(t: *BlockKeys) void {
        t.a.free(t.rk);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *BlockKeys) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(BlockKeys);

    pub fn prepare(t: *BlockKeys, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != D or shape.k != k.block) return false;
        t.rig.reset();
        t.a.free(t.rk);
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.rk = try t.a.alloc(u16, shape.rows * k.block * D);
        if (fill == .stress) {
            // keys' halves mirror (8 nudged) against head_cls mirrored with the sign turned: huge partials
            const h = D / 2;
            for (0..h) |d| {
                const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                t.hc[d] = c.toBf16(q);
                t.hc[h + d] = c.toBf16(-q);
            }
            for (0..shape.rows * k.block) |j| for (0..h) |d| {
                const v: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
                t.rk[j * D + d] = c.toBf16(v);
                t.rk[j * D + h + d] = c.toBf16(v + @as(f32, if (d < 8) 1 else 0));
            };
        } else {
            k.fillBf16(rng, t.rk, 3);
            k.fillBf16(rng, &t.hc, 1);
        }
        if (fill == .nonfinite) {
            // block 0 reads a NaN key; block 1's key 2 is +Inf at dim 7, which head_cls weighs positively
            t.rk[3 * D + 5] = c.toBf16(k.nan32);
            t.hc[7] = c.toBf16(0.5);
            if (shape.rows > 1) t.rk[(k.block + 2) * D + 7] = c.toBf16(k.inf32);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.rk));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(&t.hc));
        t.out = try t.rig.output(shape.rows * D * 4);
        return true;
    }

    pub fn run(t: *BlockKeys) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue(@as(u32, 0), 3);
        p.e.dispatchGroups(mtl.Size.of(t.shape.rows, 1, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *BlockKeys) usize {
        return t.shape.rows * D;
    }

    fn key(t: *BlockKeys, b: usize, j: usize, d: usize) f64 {
        return c.fromBf16(t.rk[(b * k.block + j) * D + d]);
    }

    /// Block b's weights (sums in `acc`, `order`); `err` gets each score's fp32 error (its dot's chain, the scale).
    fn weights(t: *BlockKeys, b: usize, acc: c.Accum, order: g.Order, w: *[k.block]f64, err: *[k.block]f64) void {
        var terms: [D]f64 = undefined;
        var aw: [k.block]f64 = undefined;
        var top = -std.math.inf(f64);
        const u = c.unit(.f32);
        for (&aw, err, 0..) |*s, *e, j| {
            var sum: f64 = 0;
            var m: f64 = 0;
            for (&terms, 0..) |*q, d| {
                q.* = t.key(b, j, d) * c.fromBf16(t.hc[d]);
                sum += q.*;
                m += @abs(q.*);
            }
            if (acc != .none) sum = contract.rows.sumIn(acc, order, &terms);
            s.* = clampNan(sum * k.inv_sqrt_hd);
            if (!std.math.isNan(s.*)) top = @max(top, s.*);
            e.* = (if (t.fill == .stress) 0 else c.gamma(D, u) * m) * k.inv_sqrt_hd + 2 * u * @abs(s.*);
        }
        var e: [k.block]f64 = undefined;
        var mass: f64 = 0;
        for (&e, aw) |*v, s| {
            v.* = @exp(s - top);
            mass += v.*;
        }
        if (acc != .none) mass = contract.rows.sumIn(acc, order, &e);
        for (w, e) |*o, v| o.* = v / @max(mass, 1e-6);
    }

    fn outputs(t: *BlockKeys, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
        var w: [k.block]f64 = undefined;
        var dm: [k.block]f64 = undefined;
        var terms: [k.block]f64 = undefined;
        for (0..t.shape.rows) |b| {
            t.weights(b, acc, order, &w, &dm);
            for (0..D) |d| {
                var s: f64 = 0;
                var m: f64 = 0;
                for (&terms, w, 0..) |*q, wj, j| {
                    q.* = wj * t.key(b, j, d);
                    s += q.*;
                    m += @abs(q.*);
                }
                if (acc != .none) s = contract.rows.sumIn(acc, order, &terms);
                out[b * D + d] = s;
                if (mag) |ms| ms[b * D + d] = if (std.math.isFinite(m)) m else 0;
            }
        }
    }

    pub fn reference(t: *BlockKeys, want: []f64, mag: []f64) !void {
        t.outputs(.none, .sequential, want, mag);
    }

    pub fn simulate(t: *BlockKeys, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.outputs(acc, order, out, null);
    }

    /// Weights within twice the worst score error, exp, the mass's chain and the division; then the sum of 32.
    pub fn bounds(t: *BlockKeys, want: []const f64, mag: []const f64, allow: []f64) !void {
        const u = c.unit(.f32);
        var w: [k.block]f64 = undefined;
        var errs: [k.block]f64 = undefined;
        for (0..t.shape.rows) |b| {
            t.weights(b, .none, .sequential, &w, &errs);
            var worst: f64 = 0;
            for (errs) |e| worst = @max(worst, e);
            const rel = 2 * (2 * worst + 4 * u) + c.gamma(k.block, u) + 2 * u;
            for (0..D) |d| {
                const i = b * D + d;
                allow[i] = if (std.math.isFinite(want[i])) (rel + c.gamma(k.block + 1, u)) * mag[i] + u * @abs(want[i]) + 0x1p-149 else 0;
            }
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    qks: std.ArrayList(*Qk) = .empty,
    keys: std.ArrayList(*BlockKeys) = .empty,

    fn add(s: *Suite, comptime T: type, list: *std.ArrayList(*T), only: ?[]const u8, m: Mutation) !void {
        const x = try s.a.create(T);
        x.* = try T.init(s.a, s.device, s.queue, m);
        const kern = x.kernel();
        if (!k.wanted(only, kern.spec.name)) {
            x.deinit();
            s.a.destroy(x);
            return;
        }
        try list.append(s.a, x);
        try s.kernels.append(s.a, kern);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Mutation{ .none, .qk_rotation_sign, .clamp_drops_nan }) |m| try s.add(Qk, &s.qks, only, m);
        for ([_]Mutation{ .none, .block_keys_unnormalized }) |m| try s.add(BlockKeys, &s.keys, only, m);
    }

    pub fn deinit(s: *Suite) void {
        inline for (.{ "qks", "keys" }) |f| {
            for (@field(s, f).items) |x| {
                x.deinit();
                s.a.destroy(x);
            }
            @field(s, f).deinit(s.a);
        }
    }
};

/// Gate 5's texts for this group's launches.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?k.Texts {
    return k.limitsText(name, &.{ "ud_qk", "ud_block_keys" });
}
