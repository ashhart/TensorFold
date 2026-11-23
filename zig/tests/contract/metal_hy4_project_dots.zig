//! tf_hy4_absorb_q, tf_hy4_expand_v and tf_hy4_expand_v_gated: fp32 fma chains on M1-M4 (M5: mla_nax.metal's).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const pj = @import("metal_hy4_project.zig");

const fl = h.fl;
const c = contract.spec;
const g = contract.gates;
const Range = fl.Range;
const bf = fl.bf;
const u = fl.u32_ulp;
const threads = contract.reference_threads;
const mw = h.hy.mla_weights;
const huge = std.math.floatMax(f64);

/// mla_project.metal's Hy4MlaProjectArgs.
const Args = extern struct { rows: u32, heads: u32 };

const Kind = enum { absorb, expand, gated };

/// Each kind's terms a sum (k), outputs a row-head and words a row-head in and out.
const Geo = struct { k: usize, cols: usize, in_stride: usize, out_stride: usize, fun: []const u8 };
fn geo(kind: Kind) Geo {
    return switch (kind) {
        .absorb => .{ .k = mw.nope, .cols = mw.latent, .in_stride = mw.query_stride, .out_stride = mw.absorbed_stride, .fun = "tf_hy4_absorb_q" },
        .expand => .{ .k = mw.latent, .cols = mw.value_width, .in_stride = mw.latent, .out_stride = mw.value_width, .fun = "tf_hy4_expand_v" },
        .gated => .{ .k = mw.latent, .cols = mw.value_width, .in_stride = mw.latent, .out_stride = mw.value_width, .fun = "tf_hy4_expand_v_gated" },
    };
}

fn absorbDepth(_: c.Shape) usize {
    return mw.nope;
}
fn expandDepth(_: c.Shape) usize {
    return mw.latent;
}

/// Rows: the call's rows; n: its heads (sparse layers 32, sliding 64); k: the terms each output's fma chain sums.
fn served(comptime k: usize) [6]c.Shape {
    return .{ .{ .rows = h.window_rows, .n = 32, .k = k }, .{ .rows = h.decode_rows, .n = 32, .k = k }, .{ .rows = h.prompt_rows, .n = 32, .k = k }, .{ .rows = h.window_rows, .n = 64, .k = k }, .{ .rows = h.decode_rows, .n = 64, .k = k }, .{ .rows = h.prompt_rows, .n = 64, .k = k } };
}
const absorb_served = served(mw.nope);
const expand_served = served(mw.latent);

fn spec(label: []const u8, kind: Kind) c.Spec {
    return .{
        .name = label,
        .accum = .f32,
        .depth = if (kind == .absorb) absorbDepth else expandDepth,
        .out = .bf16,
        .served = if (kind == .absorb) &absorb_served else &expand_served,
        .replaces = if (kind == .gated) "tf_hy4_expand_v then tf_hy4_gate_values" else null,
    };
}

pub const Dot = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    /// The gated kernel's baseline: tf_hy4_expand_v, then tf_hy4_gate_values on its output.
    chain: ?[2]mtl.Pipeline = null,
    kind: Kind,
    ctl: ?*const pj.Control,
    label: [224]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    heads: usize = 0,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    w: []u16 = &.{},
    z: []u16 = &.{},
    x32: []f32 = &.{},
    w32: []f32 = &.{},
    sum: []f64 = &.{},
    mag: []f64 = &.{},
    have: bool = false,
    shown: std.ArrayList(u8) = .empty,
    base: std.ArrayList(u8) = .empty,
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, ctl: ?*const pj.Control) !Dot {
        const pipe = try pj.pipe(a, device, geo(kind).fun, ctl);
        errdefer pipe.deinit();
        var chain: ?[2]mtl.Pipeline = null;
        if (kind == .gated) {
            const first = try pj.pipe(a, device, "tf_hy4_expand_v", null);
            errdefer first.deinit();
            chain = .{ first, try pj.pipe(a, device, "tf_hy4_gate_values", null) };
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = pipe, .chain = chain, .kind = kind, .ctl = ctl };
    }
    pub fn deinit(t: *Dot) void {
        t.free();
        t.shown.deinit(t.a);
        t.base.deinit(t.a);
        t.rig.deinit();
        t.pipe.deinit();
        if (t.chain) |ch| for (ch) |p| p.deinit();
    }
    fn free(t: *Dot) void {
        inline for (.{ "x", "w", "z", "x32", "w32", "sum", "mag" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
        t.have = false;
    }
    pub fn kernel(t: *Dot) g.Kernel {
        const variant = if (t.kind == .gated) "(replaces tf_hy4_expand_v then tf_hy4_gate_values)" else "";
        t.name = pj.name(&t.label, geo(t.kind).fun, variant, t.ctl);
        return .{ .spec = spec(t.name, t.kind), .ctx = t, .vt = &dot_vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const dot_vt = blk: {
        var v = g.hooks(Dot);
        v.output = output;
        v.intact = intact;
        break :blk v;
    };
    fn of(ctx: *anyopaque) *Dot {
        return @ptrCast(@alignCast(ctx));
    }
    fn ends(t: *const Dot) usize {
        return t.rows * t.heads;
    }
    fn outBytes(t: *Dot) []u8 {
        return t.rig.bytes(t.out, t.ends() * geo(t.kind).out_stride * 2);
    }

    pub fn prepare(t: *Dot, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const q = geo(t.kind);
        if (shape.rows == 0 or shape.n == 0 or shape.n > 64 or shape.k != q.k) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.heads = shape.n;
        t.fill = fill;
        t.x = try t.a.alloc(u16, t.ends() * q.in_stride);
        t.w = try t.a.alloc(u16, t.heads * q.k * q.cols);
        t.z = try t.a.alloc(u16, if (t.kind == .gated) t.ends() * q.cols else 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        pj.uniform(rng, t.x, 3);
        pj.uniform(rng, t.w, 1);
        pj.uniform(rng, t.z, 8);
        if (fill == .stress) t.stress(rng);
        if (fill == .nonfinite) {
            // in one head of each row: NaN, +Inf, -Inf, then both infinities; the gated kind's logits NaN and -Inf
            const at = [_][2]usize{ .{ 0, 5 }, .{ 1, 50 }, .{ 2, 100 }, .{ 3, 7 }, .{ 3, 8 } };
            const vals = [_]f32{ fl.nan32, fl.inf32, -fl.inf32, fl.inf32, -fl.inf32 };
            for (at, vals) |p, v| if (p[0] < t.rows) {
                t.x[(p[0] * t.heads + p[0] % t.heads) * q.in_stride + p[1]] = c.toBf16(v);
            };
            if (t.z.len > 0 and t.rows > 4) {
                t.z[4 * t.heads * q.cols + 30] = c.toBf16(fl.nan32);
                t.z[4 * t.heads * q.cols + 31] = c.toBf16(-fl.inf32);
            }
        }
        t.x32 = try t.a.alloc(f32, t.x.len);
        t.w32 = try t.a.alloc(f32, t.w.len);
        for (t.x32, t.x) |*f, v| f.* = c.fromBf16(v);
        for (t.w32, t.w) |*f, v| f.* = c.fromBf16(v);
        t.sum = try t.a.alloc(f64, t.count());
        t.mag = try t.a.alloc(f64, t.count());
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        if (t.z.len > 0) t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.z));
        t.out = try t.rig.output(t.ends() * q.out_stride * 2);
        return true;
    }

    /// Gate 4's terms: integers whose partials stay below 2^23 in any order (exact in fp32), mirrored to a small sum.
    fn stress(t: *Dot, rng: std.Random) void {
        // inputs 1 to 255, the second half 8 of them one up; weights 1 to 128 (absorb) or 64, the second half negated
        const q = geo(t.kind);
        const half = q.k / 2;
        for (0..t.ends()) |rh| {
            const xs = t.x[rh * q.in_stride ..][0..q.k];
            for (0..half) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 255));
                xs[i] = c.toBf16(v);
                xs[half + i] = c.toBf16(v + @as(f32, if (i < 8) 1 else 0));
            }
        }
        for (0..t.heads) |hd| for (0..q.cols) |col| for (0..half) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, if (t.kind == .absorb) 128 else 64));
            t.w[t.wAt(hd, i, col)] = c.toBf16(v);
            t.w[t.wAt(hd, half + i, col)] = c.toBf16(-v);
        };
        @memset(t.z, 0); // a gate of exactly 0.5
    }

    /// Weight (head, term i, output col): the key projection is [head][192][512], the value's [head][256][512].
    fn wAt(t: *const Dot, hd: usize, i: usize, col: usize) usize {
        const q = geo(t.kind);
        return if (t.kind == .absorb) (hd * q.k + i) * q.cols + col else (hd * q.cols + col) * q.k + i;
    }

    pub fn run(t: *Dot) !void {
        const p = t.rig.begin();
        t.encode(p.e, t.pipe, t.out, t.kind == .gated);
        try t.rig.finish(p);
    }

    fn encode(t: *Dot, e: mtl.ComputeEncoder, pipe: mtl.Pipeline, out: rig_.Ref, logits: bool) void {
        const q = geo(t.kind);
        e.setPipeline(pipe);
        e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        e.setBuffer(out.buf, out.off, 2);
        e.setValue(Args{ .rows = @intCast(t.rows), .heads = @intCast(t.heads) }, 3);
        if (logits) e.setBuffer(t.ins[2].buf, t.ins[2].off, 4);
        e.dispatchThreads(mtl.Size.of(q.cols, t.ends(), 1), mtl.Size.of(32, 1, 1));
    }

    /// tf_hy4_expand_v into working memory, then tf_hy4_gate_values over it, as two passes of one encoder.
    pub fn replaced(t: *Dot) ![]const u8 {
        const ch = t.chain orelse return error.NoBaseline;
        const bytes = t.count() * 2;
        const work = try t.rig.scratch(bytes);
        const p = t.rig.begin();
        t.encode(p.e, ch[0], work, false);
        p.e.barrier();
        const cols = t.heads * mw.value_width;
        p.e.setPipeline(ch[1]);
        p.e.setBuffer(work.buf, work.off, 0);
        p.e.setBuffer(t.ins[2].buf, t.ins[2].off, 1);
        p.e.setValue(@as(u32, @intCast(cols)), 2);
        p.e.dispatchThreads(mtl.Size.of(cols, t.rows, 1), mtl.Size.of(128, 1, 1));
        try t.rig.finish(p);
        t.base.clearRetainingCapacity();
        try t.base.appendSlice(t.a, t.rig.bytes(work, bytes));
        return t.base.items;
    }

    /// The written words only (absorb leaves each row-head's last 64 to tf_hy4_rotate).
    fn output(ctx: *anyopaque) anyerror![]const u8 {
        const t = of(ctx);
        const q = geo(t.kind);
        t.shown.clearRetainingCapacity();
        const all = t.outBytes();
        for (0..t.ends()) |rh| try t.shown.appendSlice(t.a, all[rh * q.out_stride * 2 ..][0 .. q.cols * 2]);
        return t.shown.items;
    }

    fn intact(ctx: *anyopaque) anyerror!bool {
        const t = of(ctx);
        if (!try t.rig.intact()) return false;
        const q = geo(t.kind);
        const all = t.outBytes();
        for (0..t.ends()) |rh| for (all[(rh * q.out_stride + q.cols) * 2 ..][0 .. (q.out_stride - q.cols) * 2]) |b| {
            if (b != 0xff) return false;
        };
        return true;
    }

    pub fn count(t: *Dot) usize {
        return t.ends() * geo(t.kind).cols;
    }

    const Job = struct {
        t: *Dot,
        lo: usize,
        hi: usize,
        pub fn run(j: *Job) void {
            for (j.lo..j.hi) |rh| j.t.row(rh);
        }
    };

    /// Row-head rh's float64 sums and their terms' magnitudes; every product of two bf16 values is exact in fp32.
    fn row(t: *Dot, rh: usize) void {
        const q = geo(t.kind);
        const V = @Vector(8, f32);
        const D = @Vector(8, f64);
        const hd = rh % t.heads;
        const xs = t.x32[rh * q.in_stride ..][0..q.k];
        const out = t.sum[rh * q.cols ..][0..q.cols];
        const mags = t.mag[rh * q.cols ..][0..q.cols];
        if (t.kind == .absorb) {
            @memset(out, 0);
            @memset(mags, 0);
            for (xs, 0..) |xi, i| {
                const ws = t.w32[t.wAt(hd, i, 0)..][0..q.cols];
                var col: usize = 0;
                while (col < q.cols) : (col += 8) {
                    const p: D = @floatCast(@as(V, @splat(xi)) * @as(V, ws[col..][0..8].*));
                    out[col..][0..8].* = @as(D, out[col..][0..8].*) + p;
                    mags[col..][0..8].* = @as(D, mags[col..][0..8].*) + @abs(p);
                }
            }
            return;
        }
        for (out, mags, 0..) |*s, *m, col| {
            const ws = t.w32[t.wAt(hd, 0, col)..][0..q.k];
            var acc: D = @splat(0);
            var abs: D = @splat(0);
            var i: usize = 0;
            while (i < q.k) : (i += 8) {
                const p: D = @floatCast(@as(V, xs[i..][0..8].*) * @as(V, ws[i..][0..8].*));
                acc += p;
                abs += @abs(p);
            }
            s.* = @reduce(.Add, acc);
            m.* = @reduce(.Add, abs);
        }
    }

    fn sums(t: *Dot) void {
        if (t.have) return;
        var jobs: [threads.maximum]Job = undefined;
        const n = threads.count(threads.maximum, t.ends());
        for (jobs[0..n], 0..) |*j, i| j.* = .{ .t = t, .lo = t.ends() * i / n, .hi = t.ends() * (i + 1) / n };
        threads.run(Job, jobs[0..n], .normal);
        t.have = true;
    }

    fn logit(t: *const Dot, i: usize) f64 {
        return c.fromBf16(t.z[i]);
    }

    pub fn reference(t: *Dot, want: []f64, mag: []f64) !void {
        t.sums();
        @memcpy(mag, t.mag);
        if (t.kind != .gated) return @memcpy(want, t.sum);
        for (want, t.sum, 0..) |*w, s, i| w.* = bf(bf(s) * bf(pj.sigmoid64(t.logit(i))));
    }

    /// Gate 4: each output's terms in an accumulator and order, then (gated) the bf16 value times the bf16 gate.
    pub fn simulate(t: *Dot, acc: c.Accum, order: g.Order, out: []f64) !void {
        const q = geo(t.kind);
        const terms = try t.a.alloc(f64, q.k);
        defer t.a.free(terms);
        for (out, 0..) |*o, i| {
            const rh = i / q.cols;
            const col = i % q.cols;
            for (terms, 0..) |*x, k| x.* = @as(f64, t.x32[rh * q.in_stride + k]) * t.w32[t.wAt(rh % t.heads, k, col)];
            const s = contract.rows.sumIn(acc, order, terms);
            o.* = if (t.kind == .gated) bf(bf(s) * bf(pj.sigmoid64(t.logit(i)))) else s;
        }
    }

    /// The gated kind's allowances from its intervals; the other kinds keep the registered bound.
    pub fn bounds(t: *Dot, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (t.kind != .gated) {
            const s = spec("", t.kind);
            const shape: c.Shape = .{ .rows = t.rows, .n = t.heads, .k = geo(t.kind).k };
            for (allow, want, mag) |*l, w, m| l.* = if (t.fill == .stress) c.boundExact(s, w) else c.bound(s, shape, w, m);
            return;
        }
        try fl.allowFrom(t.a, t, want, allow);
    }

    /// The fp32 chain within gamma(depth) of its terms (gate 4: exact), bf16; gated, times the bf16 gate's range, bf16.
    pub fn intervals(t: *Dot, low: []f64, high: []f64) !void {
        t.sums();
        const gam = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(geo(t.kind).k), u);
        for (low, high, t.sum, t.mag, 0..) |*l, *hi, s, m, i| {
            const z = if (t.kind == .gated) t.logit(i) else std.math.inf(f64);
            var r = Range{ .lo = -huge, .hi = huge };
            if (std.math.isFinite(s) and !std.math.isNan(z)) {
                const v = Range{ .lo = s - gam * m, .hi = s + gam * m };
                r = if (t.kind == .gated) v.bf16().mul(sigmoidOf(z).bf16()).bf16() else v.bf16();
            }
            l.* = r.lo;
            hi.* = r.hi;
        }
    }

    fn sigmoidOf(z: f64) Range {
        return if (std.math.isFinite(z)) pj.sigmoid(z) else Range.point(pj.sigmoid64(z));
    }
};

pub fn collect(s: *pj.Suite, only: ?[]const u8) !void {
    var buf: [224]u8 = undefined;
    if (fl.wanted(only, pj.name(&buf, "tf_hy4_absorb_q", "", null))) {
        try s.dots.add(s.a, s.kernels, try Dot.init(s.a, s.device, s.queue, .absorb, null));
        try s.dots.add(s.a, s.kernels, try Dot.init(s.a, s.device, s.queue, .absorb, pj.control("tf_hy4_absorb_q", "short-sum")));
        try s.dots.add(s.a, s.kernels, try Dot.init(s.a, s.device, s.queue, .absorb, pj.control("tf_hy4_absorb_q", "half-sum")));
    }
    if (fl.wanted(only, pj.name(&buf, "tf_hy4_expand_v", "", null))) {
        try s.dots.add(s.a, s.kernels, try Dot.init(s.a, s.device, s.queue, .expand, null));
        try s.dots.add(s.a, s.kernels, try Dot.init(s.a, s.device, s.queue, .expand, pj.control("tf_hy4_expand_v", "short-sum")));
    }
    if (fl.wanted(only, pj.name(&buf, "tf_hy4_expand_v_gated", "", null))) {
        try s.dots.add(s.a, s.kernels, try Dot.init(s.a, s.device, s.queue, .gated, null));
        try s.dots.add(s.a, s.kernels, try Dot.init(s.a, s.device, s.queue, .gated, pj.control("tf_hy4_expand_v_gated", "gate-unrounded")));
    }
}
