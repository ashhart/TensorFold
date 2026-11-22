//! qwen3_asr.cl's tower kernels under the contract on XPU: the Linear, conv, flatten and windowed attention.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const kit = @import("qwen3_asr_contract_kit.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const Mutation = kit.Mutation;
const bf = kit.bf;
const val = kit.val;
const u = kit.u;

/// The Linear's epilogue as the binding asks it: none (LM, conv_out), bias and residual (o, fc2), bias and GELU (fc1).
pub const Epi = enum { plain, bias_res, bias_gelu };

/// TILE_BODY: a 16-deep fma chain a k step, then the step's sum into the accumulator: at most k / 16 + 16 adds.
fn gemmDepth(s: c.Shape) usize {
    return (s.k + 15) / 16 + 17;
}

/// |gelu'| at most: its slope peaks near x = 1.4.
const gelu_slope = 1.13;

pub const Gemm = struct {
    a: std.mem.Allocator,
    rig: kit.Rig,
    k: xpu.Kernel,
    m: Mutation,
    epi: Epi,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    w: []u16 = &.{},
    b: []u16 = &.{},
    res: []u16 = &.{},
    pre: []f64 = &.{}, // each output's float64 sum plus bias, before the activation and residual
    ins: [4]?*anyopaque = undefined,
    out: ?*anyopaque = null,

    /// rows: x rows; n: outputs a row; k: the input width.
    pub fn spec(epi: Epi) c.Spec {
        const names = [_][]const u8{ "xpu qwen3-asr asr_gemm", "xpu qwen3-asr asr_gemm, bias and residual", "xpu qwen3-asr asr_gemm, bias and GELU" };
        const served: []const c.Shape = switch (epi) {
            // an odd call for the edges; the 0.6B LM's qkv, down, prompt gate | up and head; conv_out (480 x 16)
            .plain => &.{ .{ .rows = 37, .n = 130, .k = 40 }, .{ .rows = 1, .n = 4096, .k = 1024 }, .{ .rows = 1, .n = 1024, .k = 3072 }, .{ .rows = 37, .n = 6144, .k = 1024 }, .{ .rows = 1, .n = 151936, .k = 1024 }, .{ .rows = 13, .n = 896, .k = 7680 } },
            // a window's o and fc2 on the 0.6B and the 1.7B tower
            .bias_res => &.{ .{ .rows = 37, .n = 130, .k = 40 }, .{ .rows = 104, .n = 896, .k = 896 }, .{ .rows = 104, .n = 896, .k = 3584 }, .{ .rows = 104, .n = 1024, .k = 4096 } },
            // fc1
            .bias_gelu => &.{ .{ .rows = 37, .n = 130, .k = 40 }, .{ .rows = 104, .n = 3584, .k = 896 }, .{ .rows = 104, .n = 4096, .k = 1024 } },
        };
        return .{
            .name = names[@backingInt(epi)],
            .accum = .f32,
            .depth = gemmDepth,
            .ops = 2,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{ .size = 64, .min = 1 }, .{ .size = 64, .min = 1 }, .{ .size = 16, .min = 1 } },
            .served = served,
        };
    }

    pub fn init(a: std.mem.Allocator, dev: *kit.Dev, m: Mutation, epi: Epi) !Gemm {
        return .{ .a = a, .rig = .{ .a = a, .dev = dev }, .k = try dev.kernel(.qwen3_asr, m, "asr_gemm"), .m = m, .epi = epi };
    }

    pub fn deinit(t: *Gemm) void {
        t.free();
        t.k.deinit();
        t.rig.deinit();
    }

    fn free(t: *Gemm) void {
        inline for (.{ "x", "w", "b", "res" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
        t.a.free(t.pre);
        t.pre = &.{};
    }

    pub fn kernel(t: *Gemm) g.Kernel {
        return kit.mark(.{ .spec = spec(t.epi), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Gemm);

    pub fn prepare(t: *Gemm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const r = shape.rows;
        const n = shape.n;
        const kk = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, r * kk);
        t.w = try t.a.alloc(u16, n * kk);
        t.b = try t.a.alloc(u16, n);
        t.res = try t.a.alloc(u16, r * n);
        t.pre = try t.a.alloc(f64, r * n);
        @memset(t.b, 0);
        @memset(t.res, 0);
        if (fill == .stress) {
            // mirrored integer halves, 8 nudged down: a small exact dot (+sum w), partials past fp16's range
            const h = kk / 2;
            @memset(t.x, 0);
            @memset(t.w, 0);
            for (0..r) |row| for (0..h) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 255));
                t.x[row * kk + i] = c.toBf16(v);
                t.x[row * kk + h + i] = c.toBf16(v - @as(f32, if (i < 8) 1 else 0));
            };
            for (0..n) |col| for (0..h) |i| {
                const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 15));
                t.w[col * kk + i] = c.toBf16(q);
                t.w[col * kk + h + i] = c.toBf16(-q);
            };
        } else {
            kit.fill(rng, t.x, 2);
            kit.fill(rng, t.w, 0.1);
            if (t.epi != .plain) kit.fill(rng, t.b, 0.5);
            if (t.epi == .bias_res) kit.fill(rng, t.res, 2);
        }
        if (fill == .nonfinite) {
            // NaN in row 0, +Inf in row 1, -Inf in row 2, both infinities in row 3
            const at = [_][2]usize{ .{ 0, kk / 3 }, .{ 1, kk / 2 }, .{ 2, kk - 1 }, .{ 3, 0 }, .{ 3, kk - 1 } };
            const vals = [_]f32{ kit.planted[0], kit.planted[1], kit.planted[2], kit.planted[1], kit.planted[2] };
            for (at, vals) |p, v| if (p[0] < r) {
                t.x[p[0] * kk + p[1]] = c.toBf16(v);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.res));
        t.out = try t.rig.output(r * n * 2);
        return true;
    }

    pub fn run(t: *Gemm) !void {
        const s = t.shape;
        const has_b: u32 = @intFromBool(t.epi != .plain);
        const has_r: u32 = @intFromBool(t.epi == .bias_res);
        const act: u32 = @intFromBool(t.epi == .bias_gelu);
        try t.rig.launch(t.k, .{ u((s.n + 63) / 64), u((s.rows + 63) / 64), 1 }, .{ t.ins[0], t.ins[1], t.ins[2], has_b, t.ins[3], has_r, t.out, u(s.rows), u(s.n), u(s.k), u(s.n), act });
    }

    pub fn count(t: *Gemm) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Output i from its sum: the bias, GELU and residual in float64, as the epilogue orders them.
    fn finish(t: *Gemm, i: usize, sum: f64) f64 {
        const pre = sum + val(t.b[i % t.shape.n]);
        var v = if (t.epi == .bias_gelu) rows.gelu(pre) else pre;
        if (t.epi == .bias_res) v += val(t.res[i]);
        return v;
    }

    fn terms(t: *Gemm, r: usize, col: usize, buf: []f64) void {
        const kk = t.shape.k;
        for (buf, 0..) |*v, i| v.* = val(t.x[r * kk + i]) * val(t.w[col * kk + i]);
    }

    pub fn reference(t: *Gemm, want: []f64, mag: []f64) !void {
        const buf = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| for (0..t.shape.n) |col| {
            t.terms(r, col, buf);
            var s: f64 = 0;
            var m: f64 = 0;
            for (buf) |v| {
                s += v;
                m += @abs(v);
            }
            const i = r * t.shape.n + col;
            t.pre[i] = s + val(t.b[col]);
            want[i] = t.finish(i, s);
            mag[i] = m;
        };
    }

    /// The sums' chain (none under stress), then the epilogue's roundings: bias, GELU (erf 16 ulps), residual, store.
    pub fn bounds(t: *Gemm, want: []const f64, mag: []const f64, allow: []f64) !void {
        const chain = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(gemmDepth(t.shape)), kit.u32r);
        for (allow, want, mag, t.pre) |*l, w, m, pre| {
            const sum_err = chain * m + kit.u32r * @abs(pre);
            var work = switch (t.epi) {
                .plain => chain * m,
                .bias_res => sum_err + kit.ubf * (@abs(pre) + sum_err) + kit.u32r * @abs(w),
                .bias_gelu => gelu_slope * (sum_err + kit.ubf * (@abs(pre) + sum_err)) + 0.5 * @abs(pre) * 0x1p-20 + 8 * kit.u32r * (@abs(w) + @abs(pre)),
            };
            if (!std.math.isFinite(work)) work = 0;
            l.* = work + kit.ubf * (@abs(w) + work) + kit.floor_bf;
        }
    }

    pub fn simulate(t: *Gemm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| for (0..t.shape.n) |col| {
            t.terms(r, col, buf);
            const i = r * t.shape.n + col;
            out[i] = t.finish(i, rows.sumIn(acc, order, buf));
        };
    }
};

/// A conv call's geometry from its shape: rows chunks, n output channels, k = c << 16 | fi << 8 | ti.
const ConvGeo = struct { n: usize, co: usize, ci: usize, fi: usize, ti: usize, fo: usize, to: usize };

fn convGeo(s: c.Shape) ConvGeo {
    const ci = s.k >> 16;
    const fi = (s.k >> 8) & 0xff;
    const ti = s.k & 0xff;
    return .{ .n = s.rows, .co = s.n, .ci = ci, .fi = fi, .ti = ti, .fo = (fi -| 1) / 2 + 1, .to = (ti -| 1) / 2 + 1 };
}

fn convDepth(s: c.Shape) usize {
    return gemmDepth(.{ .rows = 1, .n = 1, .k = 9 * (s.k >> 16) });
}

/// The 3 x 3 stride-2 conv with zero padding 1 on channels-last rows: bf16(taps . w + b), asr_gemm's tile body.
pub const Conv = struct {
    a: std.mem.Allocator,
    rig: kit.Rig,
    k: xpu.Kernel,
    m: Mutation,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    b: []u16 = &.{},
    ins: [3]?*anyopaque = undefined,
    out: ?*anyopaque = null,

    const why = "asr_conv sums in asr_gemm's tile body, which gate 4 proves fp32 on asr_gemm; zero padding leaves " ++
        "no mirrored stress for the taps";

    pub fn spec() c.Spec {
        // an odd small grid; the tower's conv1 on two 100-frame chunks, conv2 and conv3 on a 3-frame-wide chunk's grid
        const served: []const c.Shape = &.{ .{ .rows = 2, .n = 70, .k = 5 << 16 | 13 << 8 | 11 }, .{ .rows = 2, .n = 480, .k = 1 << 16 | 128 << 8 | 100 }, .{ .rows = 1, .n = 480, .k = 480 << 16 | 64 << 8 | 3 }, .{ .rows = 1, .n = 480, .k = 480 << 16 | 32 << 8 | 3 } };
        return .{ .name = "xpu qwen3-asr asr_conv (3x3 stride 2, zero padding)", .accum = .f32, .depth = convDepth, .ops = 1, .out = .bf16, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = why } };
    }

    pub fn init(a: std.mem.Allocator, dev: *kit.Dev, m: Mutation) !Conv {
        return .{ .a = a, .rig = .{ .a = a, .dev = dev }, .k = try dev.kernel(.qwen3_asr, m, "asr_conv"), .m = m };
    }

    pub fn deinit(t: *Conv) void {
        t.free();
        t.k.deinit();
        t.rig.deinit();
    }

    fn free(t: *Conv) void {
        inline for (.{ "x", "w", "b" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }

    pub fn kernel(t: *Conv) g.Kernel {
        return kit.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Conv);

    pub fn prepare(t: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const q = convGeo(shape);
        if (q.n == 0 or q.co == 0 or q.ci == 0 or q.fi == 0 or q.ti == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, q.n * q.fi * q.ti * q.ci);
        t.w = try t.a.alloc(u16, q.co * 9 * q.ci);
        t.b = try t.a.alloc(u16, q.co);
        kit.fill(rng, t.x, 2);
        kit.fill(rng, t.w, 0.2);
        kit.fill(rng, t.b, 0.5);
        if (fill == .nonfinite) kit.plant(t.x);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Conv) !void {
        const q = convGeo(t.shape);
        const m = q.n * q.fo * q.to;
        try t.rig.launch(t.k, .{ u((q.co + 63) / 64), u((m + 63) / 64), 1 }, .{ t.ins[0], t.ins[1], t.ins[2], t.out, u(m), u(q.co), u(9 * q.ci), u(q.fi), u(q.ti), u(q.ci) });
    }

    pub fn count(t: *Conv) usize {
        const q = convGeo(t.shape);
        return q.n * q.fo * q.to * q.co;
    }

    /// Output (p, oc)'s terms: tap kk = ci 9 + kh 3 + kw, zero where the padding is.
    fn terms(t: *Conv, p: usize, oc: usize, buf: []f64) void {
        const q = convGeo(t.shape);
        const nn = p / (q.fo * q.to);
        const f = p / q.to % q.fo;
        const tt = p % q.to;
        for (buf, 0..) |*v, kk| {
            const ci = kk / 9;
            const fx = @as(isize, @intCast(2 * f + kk % 9 / 3)) - 1;
            const tx = @as(isize, @intCast(2 * tt + kk % 3)) - 1;
            v.* = 0;
            if (fx < 0 or tx < 0 or fx >= q.fi or tx >= q.ti) continue;
            const xi = ((nn * q.fi + @as(usize, @intCast(fx))) * q.ti + @as(usize, @intCast(tx))) * q.ci + ci;
            v.* = val(t.x[xi]) * val(t.w[oc * 9 * q.ci + kk]);
        }
    }

    pub fn reference(t: *Conv, want: []f64, mag: []f64) !void {
        const q = convGeo(t.shape);
        const buf = try t.a.alloc(f64, 9 * q.ci);
        defer t.a.free(buf);
        for (0..q.n * q.fo * q.to) |p| for (0..q.co) |oc| {
            t.terms(p, oc, buf);
            var s: f64 = val(t.b[oc]);
            var m: f64 = 0;
            for (buf) |v| {
                s += v;
                m += @abs(v);
            }
            want[p * q.co + oc] = s;
            mag[p * q.co + oc] = m;
        };
    }

    pub fn simulate(t: *Conv, acc: c.Accum, order: g.Order, out: []f64) !void {
        const q = convGeo(t.shape);
        const buf = try t.a.alloc(f64, 9 * q.ci);
        defer t.a.free(buf);
        for (0..q.n * q.fo * q.to) |p| for (0..q.co) |oc| {
            t.terms(p, oc, buf);
            out[p * q.co + oc] = rows.sumIn(acc, order, buf) + val(t.b[oc]);
        };
    }
};

/// [n][f][t][c] to rows [n t][c f], bit for bit; rows: chunks, n: channels, k = f << 8 | t.
pub const Flatten = struct {
    a: std.mem.Allocator,
    rig: kit.Rig,
    k: xpu.Kernel,
    m: Mutation,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    want: []u16 = &.{},
    in: ?*anyopaque = null,
    out: ?*anyopaque = null,
    msg: [160]u8 = undefined,

    pub fn spec() c.Spec {
        const served: []const c.Shape = &.{ .{ .rows = 3, .n = 8, .k = 5 << 8 | 3 }, .{ .rows = 2, .n = 480, .k = 16 << 8 | 13 }, .{ .rows = 1, .n = 1, .k = 1 << 8 | 1 } };
        return .{ .name = "xpu qwen3-asr asr_flatten (conv rows to torch's [c, f] order)", .accum = .none, .depth = kit.noSums, .out = .bf16, .served = served };
    }

    pub fn init(a: std.mem.Allocator, dev: *kit.Dev, m: Mutation) !Flatten {
        return .{ .a = a, .rig = .{ .a = a, .dev = dev }, .k = try dev.kernel(.qwen3_asr, m, "asr_flatten"), .m = m };
    }

    pub fn deinit(t: *Flatten) void {
        t.a.free(t.x);
        t.a.free(t.want);
        t.k.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *Flatten) g.Kernel {
        return kit.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Flatten);

    pub fn prepare(t: *Flatten, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const f = shape.k >> 8;
        const tt = shape.k & 0xff;
        if (shape.rows == 0 or shape.n == 0 or f == 0 or tt == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.a.free(t.want);
        t.shape = shape;
        const n = shape.rows;
        const ch = shape.n;
        t.x = try t.a.alloc(u16, n * f * tt * ch);
        var prng = std.Random.DefaultPrng.init(seed);
        kit.fill(prng.random(), t.x, 4);
        if (fill == .nonfinite) kit.plant(t.x);
        t.want = try t.a.alloc(u16, t.x.len);
        for (0..n) |nn| for (0..tt) |ti| for (0..ch) |ci| for (0..f) |fi| {
            t.want[(nn * tt + ti) * ch * f + ci * f + fi] = t.x[((nn * f + fi) * tt + ti) * ch + ci];
        };
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.out = try t.rig.output(t.x.len * 2);
        return true;
    }

    pub fn run(t: *Flatten) !void {
        const total = t.x.len;
        try t.rig.launch(t.k, .{ u((total + 255) / 256), 1, 1 }, .{ t.in, t.out, u(t.shape.k >> 8), u(t.shape.k & 0xff), u(t.shape.n), u(total) });
    }

    pub fn count(t: *Flatten) usize {
        return t.want.len;
    }

    pub fn reference(t: *Flatten, want: []f64, mag: []f64) !void {
        for (want, t.want) |*w, b| w.* = val(b);
        @memset(mag, 0);
    }

    /// Every output's bits equal to the host's copy (NaN payloads too).
    pub fn check(t: *Flatten, _: g.Gate) !?[]const u8 {
        const got = try t.rig.outputBytes();
        for (t.want, 0..) |w, i| {
            const v = std.mem.readInt(u16, got[2 * i ..][0..2], .little);
            if (v != w) return try std.fmt.bufPrint(&t.msg, "output {d} has bits {x:0>4} where the copy has {x:0>4}", .{ i, v, w });
        }
        return null;
    }
};

/// The tower's attention over one window of [q | k | v] rows (64-dim heads); rows: the window, n = heads x 64.
pub const AttnFull = struct {
    a: std.mem.Allocator,
    rig: kit.Rig,
    k: xpu.Kernel,
    m: Mutation,
    shape: c.Shape = undefined,
    qkv: []u16 = &.{},
    in: ?*anyopaque = null,
    out: ?*anyopaque = null,

    const why = "softmax weights in [0, 1] and a bf16 output: no accumulator's error survives its rounding 100x";

    fn depth(s: c.Shape) usize {
        return 66 + s.rows;
    }

    pub fn spec() c.Spec {
        // a small odd window; a full window of the 0.6B's 14 heads and of the 1.7B's 16; a short last window
        const served: []const c.Shape = &.{ .{ .rows = 45, .n = 192, .k = 64 }, .{ .rows = 104, .n = 896, .k = 64 }, .{ .rows = 104, .n = 1024, .k = 64 }, .{ .rows = 7, .n = 896, .k = 64 } };
        return .{ .name = "xpu qwen3-asr asr_attn_full (a window, 64-dim heads)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .tiles = .{ .{ .size = 16, .min = 1 }, .{}, .{} }, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = why } };
    }

    pub fn init(a: std.mem.Allocator, dev: *kit.Dev, m: Mutation) !AttnFull {
        return .{ .a = a, .rig = .{ .a = a, .dev = dev }, .k = try dev.kernel(.qwen3_asr, m, "asr_attn_full"), .m = m };
    }

    pub fn deinit(t: *AttnFull) void {
        t.a.free(t.qkv);
        t.k.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *AttnFull) g.Kernel {
        return kit.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(AttnFull);

    pub fn prepare(t: *AttnFull, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 64 != 0 or shape.k != 64) return false;
        t.rig.reset();
        t.a.free(t.qkv);
        t.shape = shape;
        t.qkv = try t.a.alloc(u16, shape.rows * 3 * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        kit.fill(prng.random(), t.qkv, 1.5);
        if (fill == .nonfinite) kit.plant(t.qkv);
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.out = try t.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(t: *AttnFull) !void {
        const s = t.shape;
        try t.rig.launch(t.k, .{ u(s.rows), u(s.n / 64), 1 }, .{ t.in, t.out, u(s.rows), u(s.n / 64), @as(f32, 0.125) });
    }

    pub fn count(t: *AttnFull) usize {
        return t.shape.rows * t.shape.n;
    }

    const Kv = struct {
        t: *const AttnFull,
        h: usize,
        pub fn k(kv: Kv, j: usize, e: usize) f64 {
            const d = kv.t.shape.n;
            return val(kv.t.qkv[j * 3 * d + d + kv.h * 64 + e]);
        }
        pub fn v(kv: Kv, j: usize, e: usize) f64 {
            const d = kv.t.shape.n;
            return val(kv.t.qkv[j * 3 * d + 2 * d + kv.h * 64 + e]);
        }
    };

    fn each(t: *AttnFull, want: []f64, allow: ?[]f64) !void {
        const d = t.shape.n;
        const len = t.shape.rows;
        var q: [64]f64 = undefined;
        for (0..len) |r| for (0..d / 64) |h| {
            for (&q, 0..) |*x, e| x.* = val(t.qkv[r * 3 * d + h * 64 + e]);
            const at = r * d + h * 64;
            try kit.attention(t.a, &q, len, Kv{ .t = t, .h = h }, 0.125, (len + 15) / 16, want[at..][0..64], if (allow) |al| al[at..][0..64] else null);
        };
    }

    pub fn reference(t: *AttnFull, want: []f64, mag: []f64) !void {
        try t.each(want, null);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(t: *AttnFull, _: []const f64, _: []const f64, allow: []f64) !void {
        const want = try t.a.alloc(f64, allow.len);
        defer t.a.free(want);
        try t.each(want, allow);
    }

    /// The scores and the two weighted sums in `acc` and `order`, exp and the division exact.
    pub fn simulate(t: *AttnFull, acc: c.Accum, order: g.Order, out: []f64) !void {
        const d = t.shape.n;
        const len = t.shape.rows;
        const buf = try t.a.alloc(f64, @max(len, 64));
        defer t.a.free(buf);
        const p = try t.a.alloc(f64, len);
        defer t.a.free(p);
        for (0..len) |r| for (0..d / 64) |h| {
            const kv: Kv = .{ .t = t, .h = h };
            var mx: f64 = -std.math.inf(f64);
            for (0..len) |j| {
                for (buf[0..64], 0..) |*x, e| x.* = val(t.qkv[r * 3 * d + h * 64 + e]) * kv.k(j, e);
                p[j] = rows.sumIn(acc, order, buf[0..64]) * 0.125;
                mx = @max(mx, p[j]);
            }
            for (p) |*x| x.* = @exp(x.* - mx);
            const l = rows.sumIn(acc, order, p);
            for (0..64) |e| {
                for (buf[0..len], 0..) |*x, j| x.* = p[j] * kv.v(j, e);
                out[r * d + h * 64 + e] = rows.sumIn(acc, order, buf[0..len]) / l;
            }
        };
    }
};
