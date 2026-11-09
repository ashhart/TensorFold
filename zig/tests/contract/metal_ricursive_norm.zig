//! Ripple's norms under the contract: rc_add_norm in the forward's three forms, and rc_prep on both layer kinds.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_ricursive_controls.zig");
const rc = @import("metal_ricursive.zig");
const c = contract.spec;
const g = contract.gates;
const rows_in = contract.rows;
const Range = rc.Range;
const R = rc.R;
const DH = rc.DH;
const u = rc.u;
const bf = rc.bf;
const xv = rc.xv;

const add_flag: u32 = 1;
const lerp_flag: u32 = 2;

/// rc_add_norm's forms: a layer's input norm with v's lerp, the attention's add, the FFN's add into the next layer.
pub const Form = enum { lerp, add, add_lerp };

/// The blend each form runs at: below one half and above it, torch.lerp's two branches.
fn blendOf(f: Form) f32 {
    return if (f == .add_lerp) 0.69921875 else 0.30078125;
}

fn flagsOf(f: Form) u32 {
    return switch (f) {
        .lerp => lerp_flag,
        .add => add_flag,
        .add_lerp => add_flag | lerp_flag,
    };
}

/// One thread's fma chain over the row (256 threads), the simd tree and the eight partials.
fn normDepth(s: c.Shape) usize {
    return s.k / 256 + 16;
}

const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const rms_sums: c.Sums = .{ .f16 = 50, .bf16 = 0, .why = rms_why };

pub fn addNormSpec(f: Form) c.Spec {
    const name = switch (f) {
        .lerp => "ricursive glue rc_add_norm lerp",
        .add => "ricursive glue rc_add_norm add",
        .add_lerp => "ricursive glue rc_add_norm add lerp",
    };
    const anyk: [3]c.Tile = .{ .{}, .{}, .{ .size = 256, .step = 1, .min = 1 } };
    return .{ .name = name, .accum = .f32, .depth = normDepth, .ops = 3, .mids = 3, .out = .bf16, .in_place = f != .lerp, .tiles = anyk, .served = &.{ .{ .rows = 16, .n = R.hidden, .k = R.hidden }, .{ .rows = 2048, .n = R.hidden, .k = R.hidden } }, .sums = rms_sums };
}

const Acc = ?struct { a: c.Accum, o: g.Order };

/// rc_add_norm on rows of the hidden width: R (and Y, w_post) with the add, V0 and the blend with the lerp.
pub const AddNorm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    form: Form,
    mutation: controls.Mutation,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    r: ?rows_in.Rows = null, // the residual (x) and w_next (w)
    y: []u16 = &.{},
    post: []u16 = &.{},
    v0: []u16 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    refs: [5]rig_.Ref = undefined, // R, Y, w_post, w_next, V0
    outs: [2]rig_.Ref = undefined, // X, VX

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, form: Form, mutation: controls.Mutation) !AddNorm {
        const lib = try controls.glueLibrary(device, a, mutation);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .form = form, .mutation = mutation, .lib = lib, .pipe = try rc.pipelineOf(lib, device, "rc_add_norm") };
    }

    fn drop(p: *AddNorm) void {
        if (p.r) |r| r.deinit();
        p.r = null;
        for ([_]*[]u16{ &p.y, &p.post, &p.v0 }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        for ([_]*[]f64{ &p.lo, &p.hi }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.rig.reset();
    }

    pub fn deinit(p: *AddNorm) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(AddNorm);

    pub fn kernel(p: *AddNorm) g.Kernel {
        return controls.mark(.{ .spec = addNormSpec(p.form), .ctx = p, .vt = &vt }, p.mutation);
    }

    fn adds(p: *const AddNorm) bool {
        return p.form != .lerp;
    }

    fn lerps(p: *const AddNorm) bool {
        return p.form != .add;
    }

    pub fn prepare(p: *AddNorm, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.k > 32 * 256) return false;
        p.drop();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6e6f);
        const rng = prng.random();
        const w = s.k;
        const n = s.rows * w;
        const r = try rows_in.Rows.init(p.a, .rms, .{ .rows = s.rows, .n = w, .k = w });
        p.r = r;
        // with the add, the planted values go into Y (the residual stays finite); without it, into R
        r.fill(if (p.adds() and fill == .nonfinite) .random else fill, seed);
        p.y = try p.a.alloc(u16, n);
        p.post = try p.a.alloc(u16, w);
        p.v0 = try p.a.alloc(u16, n);
        // Y is zero for the sums gate (the residual's stress rows stay whole), else small
        if (fill == .stress) @memset(p.y, 0) else rc.random16(rng, p.y, -0.5, 0.5);
        rc.random16(rng, p.post, 0.5, 1.5);
        rc.random16(rng, p.v0, -2, 2);
        if (fill == .nonfinite) {
            if (p.adds()) rc.plant(p.y, w, s.rows);
            if (p.lerps() and s.rows > 4) for ([_]f32{ std.math.nan(f32), std.math.inf(f32) }, 4..) |f, at| {
                p.v0[at * w + w / 4] = c.toBf16(f);
            };
        }
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.y));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.post));
        p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(r.w));
        p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.v0));
        p.refs[0] = if (p.adds()) try p.rig.output(n * 2) else try p.rig.input(std.mem.sliceAsBytes(r.x));
        p.outs[0] = try p.rig.output(n * 2);
        p.outs[1] = if (p.lerps()) try p.rig.output(n * 2) else try p.rig.scratch(n * 2);
        p.lo = try p.a.alloc(f64, p.count());
        p.hi = try p.a.alloc(f64, p.count());
        return true;
    }

    /// R (with the add), X, then VX (with the lerp).
    pub fn count(p: *AddNorm) usize {
        const n = p.shape.rows * p.shape.k;
        return n * (1 + @as(usize, @intFromBool(p.adds())) + @intFromBool(p.lerps()));
    }

    pub fn run(p: *AddNorm) !void {
        const s = p.shape;
        const r = p.r.?;
        if (p.adds()) @memcpy(p.rig.bytes(p.refs[0], r.x.len * 2), std.mem.sliceAsBytes(r.x)); // the residual before
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.refs, 0..) |x, slot| e.setBuffer(x.buf, x.off, slot);
        e.setValue(rc.Args{ .rows = @intCast(s.rows), .width = @intCast(s.k), .flags = flagsOf(p.form), .eps = rc.eps, .blend = blendOf(p.form) }, 5);
        e.setBuffer(p.outs[0].buf, p.outs[0].off, 6);
        e.setBuffer(p.outs[1].buf, p.outs[1].off, 7);
        e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    /// Row `row` at the release's rounding points into want, its range into lo and hi; `acc` replays the two sums.
    fn row(p: *AddNorm, at: usize, acc: Acc, want: []f64, t: []f64, rr: []Range) void {
        const s = p.shape;
        const w = s.k;
        const n = s.rows * w;
        const base = at * w;
        const r = p.r.?;
        const depth = normDepth(s);
        var o: usize = 0; // where X's outputs start
        for (rr, 0..) |*x, i| x.* = rc.point(xv(r.x, base + i));
        if (p.adds()) {
            o = n;
            for (t, 0..) |*x, i| x.* = xv(p.y, base + i) * xv(p.y, base + i);
            const ss = if (acc) |x| rows_in.sumIn(x.a, x.o, t) else exactSum(t);
            const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(w)) + rc.eps);
            const ir = rc.invRms(rc.point(ss), w, depth);
            for (0..w) |i| {
                const y = xv(p.y, base + i);
                const pw = xv(p.post, i);
                const res = xv(r.x, base + i);
                want[base + i] = bf(res + bf(pw * bf(y * inv)));
                const normed = rc.round(rc.mul(rc.point(pw), rc.round(rc.widen(rc.mul(rc.point(y), ir), u))));
                rr[i] = rc.round(rc.widen(rc.add(rc.point(res), normed), u));
            }
        }
        var ss: f64 = 0;
        var sr: Range = .{ 0, 0 };
        for (0..w) |i| {
            const v = if (p.adds()) want[base + i] else xv(r.x, base + i);
            t[i] = v * v;
            sr = rc.add(sr, rc.square(rr[i]));
        }
        ss = if (acc) |x| rows_in.sumIn(x.a, x.o, t) else exactSum(t);
        const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(w)) + rc.eps);
        const ir = rc.invRms(sr, w, depth);
        const blend: f64 = blendOf(p.form);
        for (0..w) |i| {
            const v = if (p.adds()) want[base + i] else xv(r.x, base + i);
            const nw = xv(r.w, i);
            const x = bf(nw * bf(v * inv));
            want[o + base + i] = x;
            const xr = rc.round(rc.mul(rc.point(nw), rc.round(rc.widen(rc.mul(rr[i], ir), u))));
            if (p.adds()) {
                p.lo[base + i] = rr[i][0];
                p.hi[base + i] = rr[i][1];
            }
            p.lo[o + base + i] = xr[0];
            p.hi[o + base + i] = xr[1];
            if (!p.lerps()) continue;
            const s0 = xv(p.v0, base + i);
            const vo = o + n + base + i;
            want[vo] = bf(lerp(s0, x, blend));
            // lerp rises with its end: the range's ends, each with the difference's rounding and the fma's
            const ends = Range{ lerp(s0, xr[0], blend), lerp(s0, xr[1], blend) };
            const err = u * (blend * @max(@abs(xr[0] - s0), @abs(xr[1] - s0)) + @max(@abs(ends[0]), @abs(ends[1])));
            const vr = rc.round(.{ ends[0] - err, ends[1] + err });
            p.lo[vo] = vr[0];
            p.hi[vo] = vr[1];
        }
    }

    fn rows(p: *AddNorm, acc: Acc, want: []f64) !void {
        const w = p.shape.k;
        const t = try p.a.alloc(f64, w);
        defer p.a.free(t);
        const rr = try p.a.alloc(Range, w);
        defer p.a.free(rr);
        for (0..p.shape.rows) |at| p.row(at, acc, want, t, rr);
    }

    pub fn reference(p: *AddNorm, want: []f64, mag: []f64) !void {
        try p.rows(null, want);
        for (mag, want) |*m, v| m.* = @abs(v);
    }

    pub fn simulate(p: *AddNorm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const lo = try p.a.dupe(f64, p.lo);
        defer p.a.free(lo);
        const hi = try p.a.dupe(f64, p.hi);
        defer p.a.free(hi);
        try p.rows(.{ .a = acc, .o = order }, out);
        @memcpy(p.lo, lo);
        @memcpy(p.hi, hi);
    }

    pub fn bounds(p: *AddNorm, want: []const f64, _: []const f64, allow: []f64) !void {
        rc.allowRange(p.lo, p.hi, want, allow);
    }

    pub fn intervals(p: *AddNorm, low: []f64, high: []f64) !void {
        @memcpy(low, p.lo);
        @memcpy(high, p.hi);
    }
};

/// torch.lerp's vectorized form (the kernel's rc_lerp) in float64.
fn lerp(s: f64, e: f64, w: f64) f64 {
    return if (@abs(w) < 0.5) s + w * (e - s) else e - (1 - w) * (e - s);
}

fn exactSum(t: []const f64) f64 {
    var s: f64 = 0;
    for (t) |x| s += x;
    return s;
}

/// rc_prep's layers: sliding (RoPE, a ring) and global (no RoPE, a slot a position).
pub const Layer = enum { sliding, global };

/// A lane's four squares, then the simd tree.
fn prepDepth(_: c.Shape) usize {
    return 12;
}

pub fn prepSpec(l: Layer) c.Spec {
    const units = R.heads + R.kv_heads;
    const stride = (R.heads + 2 * R.kv_heads) * DH;
    const name = if (l == .sliding) "ricursive glue rc_prep sliding" else "ricursive glue rc_prep global";
    return .{ .name = name, .accum = .f32, .depth = prepDepth, .ops = 6, .mids = 3, .out = .bf16, .served = &.{ .{ .rows = 16, .n = units, .k = stride }, .{ .rows = 2048, .n = units, .k = stride } }, .sums = rms_sums };
}

/// glue.metal's rc_prep_args, as forward.zig fills it.
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos: u32, stride: u32, ring: u32, rope: u32, eps: f32 };

/// rc_prep on a window (deep in a stream, its slots wrapping the ring) or a prompt chunk.
pub const Prep = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    layer: Layer,
    mutation: controls.Mutation,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    pos: usize = 0,
    ring: usize = 0,
    x: []u16 = &.{},
    qw: []u16 = &.{},
    kw: []u16 = &.{},
    inv: []f32 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    in: [4]rig_.Ref = undefined,
    out: [3]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, layer: Layer, mutation: controls.Mutation) !Prep {
        const lib = try controls.glueLibrary(device, a, mutation);
        errdefer lib.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .layer = layer, .mutation = mutation, .lib = lib, .pipe = try rc.pipelineOf(lib, device, "rc_prep") };
    }

    fn drop(p: *Prep) void {
        for ([_]*[]u16{ &p.x, &p.qw, &p.kw }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        for ([_]*[]f64{ &p.lo, &p.hi }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.a.free(p.inv);
        p.inv = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Prep) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(Prep);

    pub fn kernel(p: *Prep) g.Kernel {
        return controls.mark(.{ .spec = prepSpec(p.layer), .ctx = p, .vt = &vt }, p.mutation);
    }

    const kvw = R.kv_heads * DH;

    pub fn prepare(p: *Prep, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const stride = (R.heads + 2 * R.kv_heads) * DH;
        if (s.rows == 0 or s.k != stride or s.n != R.heads + R.kv_heads) return false;
        p.drop();
        p.shape = s;
        const window = s.rows <= 16;
        // sliding: a window across the ring's wrap, a chunk deep in a stream; global: a window at 700, a first chunk
        if (p.layer == .sliding) {
            p.ring = rc.family.sliding_ring;
            p.pos = if (window) 3 * p.ring - 7 else 100003;
        } else {
            p.pos = if (window) 700 else 0;
            p.ring = std.mem.alignForward(usize, p.pos + s.rows, 64);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.x = try p.a.alloc(u16, s.rows * stride);
        p.qw = try p.a.alloc(u16, DH);
        p.kw = try p.a.alloc(u16, DH);
        if (fill == .stress) { // integers whose 128 squares overflow fp16 and lose bits in bf16
            for (p.x) |*v| v.* = c.toBf16(@floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 129))) - 64));
        } else rc.random16(rng, p.x, -3, 3);
        rc.random16(rng, p.qw, 0.5, 1.5);
        rc.random16(rng, p.kw, 0.5, 1.5);
        if (fill == .nonfinite) rc.plant(p.x, stride, s.rows);
        p.inv = try p.a.alloc(f32, DH / 2);
        for (p.inv, 0..) |*f, i| {
            const pw: f32 = @floatCast(std.math.pow(f64, 10000, @as(f64, @floatFromInt(2 * i)) / DH));
            f.* = 1.0 / pw; // config.zig's inverseFrequencies
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.inv));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(p.qw));
        p.in[3] = try p.rig.input(std.mem.sliceAsBytes(p.kw));
        p.out[0] = try p.rig.output(s.rows * R.heads * DH * 2);
        p.out[1] = try p.rig.output(p.ring * kvw * 2);
        p.out[2] = try p.rig.output(p.ring * kvw * 2);
        p.lo = try p.a.alloc(f64, p.count());
        p.hi = try p.a.alloc(f64, p.count());
        return true;
    }

    /// QN, then the whole key and value rings.
    pub fn count(p: *Prep) usize {
        return p.shape.rows * R.heads * DH + 2 * p.ring * kvw;
    }

    pub fn run(p: *Prep) !void {
        const s = p.shape;
        // slots no row of this call reaches: unwritten by the kernel, zero in the reference
        for (p.out[1..]) |o| @memset(p.rig.bytes(o, p.ring * kvw * 2), 0);
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in[0..4], 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        const a = PArgs{ .rows = @intCast(s.rows), .heads = R.heads, .kv_heads = R.kv_heads, .pos = @intCast(p.pos), .stride = @intCast(s.k), .ring = @intCast(p.ring), .rope = @intFromBool(p.layer == .sliding), .eps = rc.eps };
        e.setValue(a, 4);
        for (p.out, 5..) |o, slot| e.setBuffer(o.buf, o.off, slot);
        e.dispatchGroups(mtl.Size.of(R.heads + R.kv_heads, s.rows, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }

    /// One head's norm and rotation at row `r`: want at the rounding points, its range; `acc` replays the sum.
    fn head(p: *Prep, r: usize, unit: usize, acc: Acc, want: []f64) void {
        const q = unit < R.heads;
        const h = if (q) unit else unit - R.heads;
        const src = r * p.shape.k + (if (q) 0 else R.heads * DH) + h * DH;
        const qn = p.shape.rows * R.heads * DH;
        const slot = (p.pos + r) % p.ring;
        const dst = if (q) (r * R.heads + h) * DH else qn + slot * kvw + h * DH;
        const w = if (q) p.qw else p.kw;
        var t: [DH]f64 = undefined;
        for (&t, 0..) |*x, d| x.* = xv(p.x, src + d) * xv(p.x, src + d);
        const ss = if (acc) |x| rows_in.sumIn(x.a, x.o, &t) else exactSum(&t);
        const inv = 1.0 / @sqrt(ss / DH + rc.eps);
        const ir = rc.invRms(rc.point(exactSum(&t)), DH, 12);
        var n: [DH]f64 = undefined;
        var nr: [DH]Range = undefined;
        for (0..DH) |d| {
            const x = xv(p.x, src + d);
            const wd = xv(w, d);
            n[d] = bf(wd * bf(x * inv));
            nr[d] = rc.round(rc.mul(rc.point(wd), rc.round(rc.widen(rc.mul(rc.point(x), ir), u))));
        }
        const position: f32 = @floatFromInt(p.pos + r);
        for (0..DH) |d| {
            var v = n[d];
            var e = nr[d];
            if (p.layer == .sliding) {
                const angle: f64 = position * p.inv[d % (DH / 2)]; // the kernel's fp32 angle
                const cs = @cos(angle);
                const sn = @sin(angle);
                const cr = rc.round(rc.widen(rc.point(cs), 4 * u)); // precise cos and sin: 4 ulps
                const sr = rc.round(rc.widen(rc.point(sn), 4 * u));
                const other = if (d < DH / 2) -n[d + DH / 2] else n[d - DH / 2];
                const or_ = if (d < DH / 2) Range{ -nr[d + DH / 2][1], -nr[d + DH / 2][0] } else nr[d - DH / 2];
                v = bf(n[d] * bf(cs) + other * bf(sn));
                const t1 = rc.mul(nr[d], cr);
                const t2 = rc.mul(or_, sr);
                const err = 2 * u * (@max(@abs(t1[0]), @abs(t1[1])) + @max(@abs(t2[0]), @abs(t2[1])));
                e = rc.round(.{ t1[0] + t2[0] - err, t1[1] + t2[1] + err });
            }
            want[dst + d] = v;
            p.lo[dst + d] = e[0];
            p.hi[dst + d] = e[1];
        }
        if (q) return;
        for (0..DH) |d| {
            const at = qn + p.ring * kvw + slot * kvw + h * DH + d;
            want[at] = xv(p.x, r * p.shape.k + (R.heads + R.kv_heads) * DH + h * DH + d);
            p.lo[at] = want[at];
            p.hi[at] = want[at];
        }
    }

    fn all(p: *Prep, acc: Acc, want: []f64) void {
        const qn = p.shape.rows * R.heads * DH;
        @memset(want[qn..], 0);
        @memset(p.lo[qn..], 0);
        @memset(p.hi[qn..], 0);
        for (0..p.shape.rows) |r| for (0..R.heads + R.kv_heads) |unit| p.head(r, unit, acc, want);
    }

    pub fn reference(p: *Prep, want: []f64, mag: []f64) !void {
        p.all(null, want);
        for (mag, want) |*m, v| m.* = @abs(v);
    }

    pub fn simulate(p: *Prep, acc: c.Accum, order: g.Order, out: []f64) !void {
        const lo = try p.a.dupe(f64, p.lo);
        defer p.a.free(lo);
        const hi = try p.a.dupe(f64, p.hi);
        defer p.a.free(hi);
        p.all(.{ .a = acc, .o = order }, out);
        @memcpy(p.lo, lo);
        @memcpy(p.hi, hi);
    }

    pub fn bounds(p: *Prep, want: []const f64, _: []const f64, allow: []f64) !void {
        rc.allowRange(p.lo, p.hi, want, allow);
    }

    pub fn intervals(p: *Prep, low: []f64, high: []f64) !void {
        @memcpy(low, p.lo);
        @memcpy(high, p.hi);
    }
};
