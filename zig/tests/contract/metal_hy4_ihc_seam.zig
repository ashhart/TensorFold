//! The seams as metal_dense.zig launches them: tail then head, partials then head, a lone tail, and the fused seam.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const grp = @import("metal_hy4_ihc.zig");
const m = @import("metal_hy4_ihc_ref.zig");

const c = contract.spec;
const g = contract.gates;
const K = grp.K;
const S = m.streams;
const W3 = h.hidden;

/// tailPass then headPass; seamPartials then headPass; seamPass; tailPass with no next collapse.
pub const Path = enum { split, bare, fused, alone };

pub const Case = struct {
    name: []const u8,
    path: Path,
    gates: u32 = 8,
    /// Two branches added in bf16 first (a routed block's experts and shared expert).
    two: bool = false,
    /// The checked output: the next post gates (fp32), else the bf16 rows (mixed streams, collapsed, normed).
    post: bool = false,
    ctl: ?[]const u8 = null,
};

const base = "metal hy4 ihc ";
pub const cases = [_]Case{
    .{ .name = base ++ "tf_hy4_head_g8 after tf_hy4_tail_g8 rows", .path = .split },
    .{ .name = base ++ "tf_hy4_head_g8 after tf_hy4_tail_g8 post", .path = .split, .post = true },
    .{ .name = base ++ "tf_hy4_head_g8 after tf_hy4_tail_g8 two branches rows", .path = .split, .two = true },
    .{ .name = base ++ "tf_hy4_head_g4 after tf_hy4_tail_g4 rows", .path = .split, .gates = 4 },
    .{ .name = base ++ "tf_hy4_head_g8 after tf_hy4_tail_g8 partials only rows", .path = .bare },
    .{ .name = base ++ "tf_hy4_head_g8 after tf_hy4_tail_g8 partials only post", .path = .bare, .post = true },
    .{ .name = base ++ "tf_hy4_seam_g8 rows", .path = .fused },
    .{ .name = base ++ "tf_hy4_seam_g8 post", .path = .fused, .post = true },
    .{ .name = base ++ "tf_hy4_seam_g8 two branches rows", .path = .fused, .two = true },
    .{ .name = base ++ "tf_hy4_seam_g4 rows", .path = .fused, .gates = 4 },
    .{ .name = base ++ "tf_hy4_tail_g8 no next", .path = .alone },
    .{ .name = base ++ "tf_hy4_tail_g8 no next two branches", .path = .alone, .two = true },
    .{ .name = base ++ "tf_hy4_head_g8 after tf_hy4_tail_g8 rows", .path = .split, .ctl = "unweighted" },
    .{ .name = base ++ "tf_hy4_seam_g8 rows", .path = .fused, .ctl = "once" },
};

/// The tail's norm: a thread's strided chain over the row, the simdgroup butterfly, the 8 simdgroups in turn.
fn normDepth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 5 + 8;
}

/// The next collapse's sums: a thread's columns' four streams, the butterfly, 8 simdgroups, then 12 threadgroups.
fn dotDepth(s: c.Shape) usize {
    return S * ((s.n + 12 * 256 - 1) / (12 * 256)) + 5 + 8 + 12;
}

fn gramDepth(s: c.Shape) usize {
    return (s.n + 12 * 256 - 1) / (12 * 256) + 5 + 8 + 12;
}

/// DFly2's window first (the edges run at it), then one decode row, and a prompt chunk or the fused seam's 16 rows.
const served = [_]c.Shape{ .{ .rows = h.window_rows, .n = W3, .k = S }, .{ .rows = h.decode_rows, .n = W3, .k = S }, .{ .rows = h.prompt_rows, .n = W3, .k = S } };
const served_fused = [_]c.Shape{ .{ .rows = h.window_rows, .n = W3, .k = S }, .{ .rows = h.decode_rows, .n = W3, .k = S }, .{ .rows = 16, .n = W3, .k = S } };

const norm_why = "a bf16 output follows the branch's sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

fn spec(v: Case, name: []const u8) c.Spec {
    var s: c.Spec = .{
        .name = name,
        .accum = .f32,
        .depth = if (v.path == .alone) normDepth else dotDepth,
        .out = if (v.post) .f32 else .bf16,
        .in_place = v.post,
        .tiles = .{ .{}, .{ .size = 256, .step = 32, .min = 32 }, .{} },
        .served = &served,
    };
    if (v.path == .fused) {
        s.tiles = .{ .{}, .{}, .{} };
        s.served = &served_fused;
        s.replaces = if (v.gates == 4) "tf_hy4_tail_g4 then tf_hy4_head_g4" else "tf_hy4_tail_g8 then tf_hy4_head_g8";
    }
    if (v.path == .alone) s.sums = .{ .bf16 = 0, .why = norm_why };
    return s;
}

fn ref(r: rig_.Ref) h.hy.ihc_metal.Ref {
    return .{ .buffer = r.buf, .offset = r.off };
}

/// One launch's working buffers: the outputs (or scratch standing in), the post gates, the partials and the barriers.
const Outs = struct { out: rig_.Ref, col: rig_.Ref, norm: rig_.Ref, post: rig_.Ref, part: rig_.Ref, bars: rig_.Ref };

pub const Seam = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    srv: *const K,
    ctl: ?grp.Ctl,
    pipe: ?mtl.Pipeline,
    v: Case,
    p: m.Params,
    label: [224]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    b0: []u16 = &.{},
    b1: []u16 = &.{},
    gain: []u16 = &.{},
    res: []u16 = &.{},
    post_in: []f32 = &.{},
    proj: []f32 = &.{},
    bias: []f32 = &.{},
    scale: []f32 = &.{},
    ngain: []u16 = &.{},
    want: []f64 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    base_bytes: []u8 = &.{},
    have: bool = false,
    ins: [8]rig_.Ref = undefined,
    o: Outs = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, srv: *const K, v: Case, p: m.Params) !Seam {
        const k = if (v.ctl) |slug| grp.control(slug) else null;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .srv = srv, .ctl = k, .pipe = if (k) |x| try grp.pipeline(a, device, x) else null, .v = v, .p = p };
    }
    pub fn deinit(t: *Seam) void {
        t.free();
        t.rig.deinit();
        if (t.pipe) |p| p.deinit();
    }
    fn free(t: *Seam) void {
        inline for (.{ "b0", "b1", "gain", "res", "post_in", "proj", "bias", "scale", "ngain", "want", "lo", "hi", "base_bytes" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
        t.have = false;
    }
    pub fn kernel(t: *Seam) g.Kernel {
        t.name = h.fl.named(t.v.name, if (t.ctl) |k| k.drops else null, &t.label);
        return .{ .spec = spec(t.v, t.name), .ctx = t, .vt = &vt, .control = if (t.ctl) |k| k.gate else null };
    }
    const vt = g.hooks(Seam);

    fn hasOut(t: *const Seam) bool {
        return !t.v.post and t.v.path != .bare;
    }
    fn hasHead(t: *const Seam) bool {
        return !t.v.post and t.v.path != .alone;
    }

    fn ihcShape(t: *const Seam) h.hy.ihc.Shape {
        return .{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.n), .gates = t.v.gates, .eps = t.p.eps, .gate_eps = t.p.gate_eps, .post_magnitude = t.p.magnitude };
    }

    pub fn prepare(t: *Seam, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const v = t.v;
        if (shape.rows == 0 or shape.n == 0 or shape.n % 2 != 0 or shape.n > W3 or shape.k != S) return false;
        if (v.path == .fused and (shape.n != W3 or shape.rows > K.seam_rows)) return false;
        if (v.post and (v.gates != 8 or v.path == .alone)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const rows = shape.rows;
        const W = shape.n;
        const wide = S * W;
        const G = v.gates;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.b0 = try t.a.alloc(u16, rows * W);
        t.b1 = try t.a.alloc(u16, if (v.two) rows * W else 0);
        t.gain = try t.a.alloc(u16, W);
        t.res = try t.a.alloc(u16, rows * wide);
        t.post_in = try t.a.alloc(f32, rows * S);
        t.proj = try t.a.alloc(f32, G * wide);
        t.bias = try t.a.alloc(f32, G);
        t.scale = try t.a.alloc(f32, G / S);
        t.ngain = try t.a.alloc(u16, W);
        if (fill == .stress) {
            @memset(t.gain, c.toBf16(1));
            @memset(t.ngain, c.toBf16(1));
            @memset(t.post_in, 1);
            @memset(t.b1, 0);
            @memset(t.bias, 0);
            @memset(t.scale, 1);
            for (0..G) |j| for (0..S) |s| for (0..W) |i| {
                t.proj[j * wide + s * W + i] = m.stressProject(j, s, i, W);
            };
            if (v.path == .alone) {
                // integers whose squares sum exactly in fp32 (at most 3,072 x 1,600) and overflow fp16 in any order
                for (t.b0) |*x| x.* = c.toBf16(@floatFromInt(@as(i32, @intCast(20 + rng.uintLessThan(u32, 21))) * (if (rng.boolean()) @as(i32, 1) else -1)));
                @memset(t.res, 0);
            } else {
                // no branch: the streams are the residual, whose sums gate 4 proves through the head
                @memset(t.b0, 0);
                for (0..rows) |r| for (0..S) |s| for (0..W) |i| {
                    t.res[r * wide + s * W + i] = c.toBf16(m.stressHidden(r, s, i, W));
                };
            }
        } else {
            for (t.b0) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.b1) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.gain) |*x| x.* = c.toBf16(0.5 + rng.float(f32));
            for (t.ngain) |*x| x.* = c.toBf16(0.5 + rng.float(f32));
            for (t.res) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
            // a sub-block's post gates lie in (hc_eps, hc_magnitude + hc_eps]
            for (t.post_in) |*x| x.* = 0.25 + rng.float(f32) * 1.5;
            for (t.proj) |*x| x.* = (rng.float(f32) * 2 - 1) * 0.05;
            for (t.bias) |*x| x.* = rng.float(f32) * 2 - 1;
            for (t.scale) |*x| x.* = 0.5 + rng.float(f32);
        }
        if (fill == .nonfinite) t.plant();
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.b0));
        t.ins[1] = if (v.two) try t.rig.input(std.mem.sliceAsBytes(t.b1)) else t.ins[0];
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.gain));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.res));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.proj));
        t.ins[5] = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        t.ins[6] = try t.rig.input(std.mem.sliceAsBytes(t.scale));
        t.ins[7] = try t.rig.input(std.mem.sliceAsBytes(t.ngain));
        // outputs in the order the rig gathers them: the mixed streams, the collapsed rows, the normed rows
        t.o.out = if (t.hasOut()) try t.rig.output(rows * wide * 2) else try t.rig.scratch(rows * wide * 2);
        t.o.col = if (t.hasHead()) try t.rig.output(rows * W * 2) else try t.rig.scratch(rows * W * 2);
        t.o.norm = if (t.hasHead()) try t.rig.output(rows * W * 2) else try t.rig.scratch(rows * W * 2);
        t.o.post = if (v.post) try t.rig.output(rows * S * 4) else try t.rig.scratch(rows * S * 4);
        try t.working(&t.o);
        return true;
    }

    /// The partials and the fused seam's barriers (zeroed: each row's arrival count starts at 0).
    fn working(t: *Seam, o: *Outs) !void {
        o.part = try t.rig.scratch(t.shape.rows * h.hy.ihc_metal.partial_floats * 4);
        o.bars = try t.rig.scratch(K.seam_rows * 8);
        @memset(t.rig.bytes(o.bars, K.seam_rows * 8), 0);
    }

    /// The residual a row each (one with both infinities), the branch's NaN and Inf, a NaN bias and gain; row 5 clean.
    fn plant(t: *Seam) void {
        const W = t.shape.n;
        const wide = S * W;
        const rows = t.shape.rows;
        const at = [_][3]usize{ .{ 0, 0, 3 }, .{ 1, 1, 5 }, .{ 2, 2, 1 }, .{ 2, 3, 4 } };
        const vals = [_]f32{ h.fl.planted[0], h.fl.planted[1], h.fl.planted[2], h.fl.planted[1] };
        for (at, vals) |q, x| if (q[0] < rows and q[2] < W) {
            t.res[q[0] * wide + q[1] * W + q[2]] = c.toBf16(x);
        };
        if (rows > 5 and W > 11) {
            t.b0[3 * W + 7] = c.toBf16(h.fl.planted[0]);
            t.b0[4 * W + 2] = c.toBf16(h.fl.planted[1]);
            if (t.v.gates == 8) t.bias[6] = h.fl.planted[0];
            t.ngain[11] = c.toBf16(h.fl.planted[0]);
        }
    }

    fn kern(t: *const Seam) K {
        return grp.with(t.srv.*, t.pipe, if (t.ctl) |k| k.slot else .tail8);
    }

    fn encode(t: *Seam, k: K, path: Path, o: Outs) !void {
        const rows: u32 = @intCast(t.shape.rows);
        const W: u32 = @intCast(t.shape.n);
        const shape = t.ihcShape();
        @memcpy(t.rig.bytes(o.post, t.post_in.len * 4), std.mem.sliceAsBytes(t.post_in));
        const p = t.rig.begin();
        const e = p.e;
        const tail: K.Tail = .{
            .branch = ref(t.ins[0]),
            .branch2 = if (t.v.two) ref(t.ins[1]) else null,
            .gain = ref(t.ins[2]),
            .residual = ref(t.ins[3]),
            .post = ref(o.post),
            .output = ref(o.out),
            .next = if (path == .alone) null else ref(t.ins[4]),
            .partials = ref(o.part),
        };
        var head: K.Head = .{ .hidden = ref(o.out), .partials = ref(o.part), .bias = ref(t.ins[5]), .scale = ref(t.ins[6]), .gain = ref(t.ins[7]), .collapsed = ref(o.col), .normalized = ref(o.norm), .post = ref(o.post) };
        switch (path) {
            .split => {
                k.tailPass(e, tail, rows, W, t.p.eps, t.v.gates);
                e.barrier();
                try k.headPass(e, head, shape, t.p.eps);
            },
            .bare => {
                head.hidden = ref(t.ins[3]);
                try k.seamPartials(e, ref(t.ins[3]), ref(t.ins[4]), ref(o.part), shape);
                e.barrier();
                try k.headPass(e, head, shape, t.p.eps);
            },
            .fused => try k.seamPass(e, tail, .{ .bias = head.bias, .scale = head.scale, .gain = head.gain, .collapsed = head.collapsed, .normalized = head.normalized, .shape = shape, .barriers = ref(o.bars) }, rows, W, t.p.eps),
            .alone => k.tailPass(e, tail, rows, W, t.p.eps, 8),
        }
        try t.rig.finish(p);
    }

    pub fn run(t: *Seam) !void {
        try t.encode(t.kern(), t.v.path, t.o);
    }

    /// The tail then the head (the loader's own pipelines) on the same inputs, into buffers of their own.
    pub fn replaced(t: *Seam) ![]const u8 {
        const rows = t.shape.rows;
        const W = t.shape.n;
        var o: Outs = undefined;
        o.out = try t.rig.scratch(rows * S * W * 2);
        o.col = try t.rig.scratch(rows * W * 2);
        o.norm = try t.rig.scratch(rows * W * 2);
        o.post = try t.rig.scratch(rows * S * 4);
        try t.working(&o);
        try t.encode(t.srv.*, .split, o);
        t.a.free(t.base_bytes);
        t.base_bytes = &.{};
        if (t.v.post) {
            t.base_bytes = try t.a.dupe(u8, t.rig.bytes(o.post, rows * S * 4));
        } else {
            t.base_bytes = try std.mem.concat(t.a, u8, &.{ t.rig.bytes(o.out, rows * S * W * 2), t.rig.bytes(o.col, rows * W * 2), t.rig.bytes(o.norm, rows * W * 2) });
        }
        return t.base_bytes;
    }

    pub fn count(t: *Seam) usize {
        const rows = t.shape.rows;
        if (t.v.post) return rows * S;
        return rows * ((if (t.hasOut()) S * t.shape.n else 0) + (if (t.hasHead()) 2 * t.shape.n else 0));
    }

    fn tailOf(t: *const Seam, r: usize) m.Tail {
        const W = t.shape.n;
        return .{ .p = t.p, .width = W, .bare = t.v.path == .bare, .b0 = t.b0[r * W ..][0..W], .b1 = if (t.v.two) t.b1[r * W ..][0..W] else null, .gain = t.gain, .res = t.res[r * S * W ..][0 .. S * W], .post = t.post_in[r * S ..][0..S], .depth = normDepth(t.shape) };
    }

    fn headOf(t: *const Seam) m.Head {
        return .{ .p = t.p, .width = t.shape.n, .n_gates = t.v.gates, .proj = t.proj, .bias = t.bias, .scale = t.scale, .gain = t.ngain, .depth = dotDepth(t.shape), .gram_depth = gramDepth(t.shape) };
    }

    const Job = struct {
        t: *Seam,
        sum: m.Sum,
        ranged: bool,
        rows: [2]usize,
        out: m.Dest,
        failed: bool = false,
        pub fn run(j: *Job) void {
            j.go() catch {
                j.failed = true;
            };
        }
        fn go(j: *Job) !void {
            const t = j.t;
            const W = t.shape.n;
            const wide = S * W;
            const all = t.shape.rows;
            const buf = try t.a.alloc(f64, 4 * wide);
            defer t.a.free(buf);
            const x = buf[0..wide];
            const lo = buf[wide .. 2 * wide];
            const hi = buf[2 * wide .. 3 * wide];
            const exact = t.fill == .stress;
            for (j.rows[0]..j.rows[1]) |r| {
                m.tailRow(t.tailOf(r), j.sum, j.ranged, exact, buf[3 * wide ..], x, lo, hi);
                if (t.v.post) {
                    m.headRow(t.headOf(), j.sum, j.ranged, exact, x, lo, hi, buf[3 * wide ..], null, null, j.out.at(r * S, S));
                    continue;
                }
                var at: usize = 0;
                if (t.hasOut()) {
                    const d = j.out.at(r * wide, wide);
                    @memcpy(d.want, x);
                    if (j.ranged) @memcpy(d.lo, lo);
                    if (j.ranged) @memcpy(d.hi, hi);
                    at = all * wide;
                }
                if (!t.hasHead()) continue;
                m.headRow(t.headOf(), j.sum, j.ranged, exact, x, lo, hi, buf[3 * wide ..], j.out.at(at + r * W, W), j.out.at(at + all * W + r * W, W), null);
            }
        }
    };

    /// Every row's outputs, in reference_threads' jobs of whole rows.
    fn rowsInto(t: *Seam, sum: m.Sum, ranged: bool, out: m.Dest) !void {
        const rows = t.shape.rows;
        const n = contract.reference_threads.count(contract.reference_threads.maximum, rows);
        var jobs: [contract.reference_threads.maximum]Job = undefined;
        for (0..n) |i| jobs[i] = .{ .t = t, .sum = sum, .ranged = ranged, .rows = .{ rows * i / n, rows * (i + 1) / n }, .out = out };
        contract.reference_threads.run(Job, jobs[0..n], .normal);
        for (jobs[0..n]) |j| if (j.failed) return error.OutOfMemory;
    }

    fn model(t: *Seam) !void {
        if (t.have) return;
        const n = t.count();
        t.want = try t.a.alloc(f64, n);
        t.lo = try t.a.alloc(f64, n);
        t.hi = try t.a.alloc(f64, n);
        try t.rowsInto(.{}, t.fill != .nonfinite, .{ .want = t.want, .lo = t.lo, .hi = t.hi });
        t.have = true;
    }

    pub fn reference(t: *Seam, want: []f64, mag: []f64) !void {
        try t.model();
        @memcpy(want, t.want);
        @memset(mag, 0);
    }

    /// Each output's range: the tail rows' ranges, then the head's sums within gamma(depth) over them, each step after.
    pub fn intervals(t: *Seam, low: []f64, high: []f64) !void {
        try t.model();
        @memcpy(low, t.lo);
        @memcpy(high, t.hi);
    }

    pub fn bounds(t: *Seam, want: []const f64, _: []const f64, allow: []f64) !void {
        try h.fl.allowFrom(t.a, t, want, allow);
    }

    pub fn simulate(t: *Seam, acc: c.Accum, order: g.Order, out: []f64) !void {
        const scratch = try t.a.alloc(f64, 2 * out.len);
        defer t.a.free(scratch);
        try t.rowsInto(.{ .acc = acc, .order = order }, false, .{ .want = out, .lo = scratch[0..out.len], .hi = scratch[out.len..] });
    }
};
