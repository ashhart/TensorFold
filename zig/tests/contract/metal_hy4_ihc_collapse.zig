//! The collapse against float64, and its two-launch form bit-equal to it; each of its two outputs registers once.
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

pub const Case = struct {
    name: []const u8,
    gates: u32,
    /// partials then finish, which must give the one launch's bits.
    two: bool = false,
    /// The checked output: the post gates (fp32), else the collapsed rows (bf16).
    post: bool = false,
    ctl: ?[]const u8 = null,
};

const base = "metal hy4 ihc ";
pub const cases = [_]Case{
    .{ .name = base ++ "tf_hy4_ihc_collapse_g4", .gates = 4 },
    .{ .name = base ++ "tf_hy4_ihc_collapse_g8 rows", .gates = 8 },
    .{ .name = base ++ "tf_hy4_ihc_collapse_g8 post", .gates = 8, .post = true },
    .{ .name = base ++ "tf_hy4_ihc_finish_g4 after tf_hy4_ihc_partials_g4", .gates = 4, .two = true },
    .{ .name = base ++ "tf_hy4_ihc_finish_g8 after tf_hy4_ihc_partials_g8 rows", .gates = 8, .two = true },
    .{ .name = base ++ "tf_hy4_ihc_finish_g8 after tf_hy4_ihc_partials_g8 post", .gates = 8, .two = true, .post = true },
    .{ .name = base ++ "tf_hy4_ihc_collapse_g8 rows", .gates = 8, .ctl = "flush" },
    .{ .name = base ++ "tf_hy4_ihc_collapse_g8 post", .gates = 8, .post = true, .ctl = "seven" },
    .{ .name = base ++ "tf_hy4_ihc_collapse_g8 post", .gates = 8, .post = true, .ctl = "half" },
    .{ .name = base ++ "tf_hy4_ihc_finish_g8 after tf_hy4_ihc_partials_g8 post", .gates = 8, .two = true, .post = true, .ctl = "slot0" },
};

/// A thread's strided chain over the row's 4 W values, the simdgroup butterfly, then the 8 simdgroups in turn.
fn depth(s: c.Shape) usize {
    return (S * s.n + 255) / 256 + 5 + 8;
}

/// DFly2's window first (the edges run at it), then one decode row and a prompt chunk.
const served = [_]c.Shape{ .{ .rows = h.window_rows, .n = h.hidden, .k = S }, .{ .rows = h.decode_rows, .n = h.hidden, .k = S }, .{ .rows = h.prompt_rows, .n = h.hidden, .k = S } };

fn spec(v: Case, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .out = if (v.post) .f32 else .bf16,
        .replaces = if (v.two) (if (v.gates == 4) "tf_hy4_ihc_collapse_g4" else "tf_hy4_ihc_collapse_g8") else null,
        .tiles = .{ .{}, .{ .size = 256, .step = 32, .min = 32 }, .{} },
        .served = &served,
    };
}

fn ref(r: rig_.Ref) h.hy.ihc_metal.Ref {
    return .{ .buffer = r.buf, .offset = r.off };
}

pub const Collapse = struct {
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
    x: []u16 = &.{},
    proj: []f32 = &.{},
    bias: []f32 = &.{},
    scale: []f32 = &.{},
    want: []f64 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    base_bytes: []u8 = &.{},
    have: bool = false,
    ins: [4]rig_.Ref = undefined,
    col: rig_.Ref = undefined,
    post: rig_.Ref = undefined,
    part: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, srv: *const K, v: Case, p: m.Params) !Collapse {
        const k = if (v.ctl) |slug| grp.control(slug) else null;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .srv = srv, .ctl = k, .pipe = if (k) |x| try grp.pipeline(a, device, x) else null, .v = v, .p = p };
    }
    pub fn deinit(t: *Collapse) void {
        t.free();
        t.rig.deinit();
        if (t.pipe) |p| p.deinit();
    }
    fn free(t: *Collapse) void {
        inline for (.{ "x", "proj", "bias", "scale", "want", "lo", "hi", "base_bytes" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
        t.have = false;
    }
    pub fn kernel(t: *Collapse) g.Kernel {
        t.name = h.fl.named(t.v.name, if (t.ctl) |k| k.drops else null, &t.label);
        return .{ .spec = spec(t.v, t.name), .ctx = t, .vt = &vt, .control = if (t.ctl) |k| k.gate else null };
    }
    const vt = g.hooks(Collapse);

    fn ihcShape(t: *const Collapse) h.hy.ihc.Shape {
        return .{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.n), .gates = t.v.gates, .eps = t.p.eps, .gate_eps = t.p.gate_eps, .post_magnitude = t.p.magnitude };
    }

    pub fn prepare(t: *Collapse, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 2 != 0 or shape.k != S or (t.v.post and t.v.gates != 8)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const rows = shape.rows;
        const W = shape.n;
        const wide = S * W;
        const G = t.v.gates;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, rows * wide);
        t.proj = try t.a.alloc(f32, G * wide);
        t.bias = try t.a.alloc(f32, G);
        t.scale = try t.a.alloc(f32, G / S);
        if (fill == .stress) {
            for (0..rows) |r| for (0..S) |s| for (0..W) |i| {
                t.x[r * wide + s * W + i] = c.toBf16(m.stressHidden(r, s, i, W));
            };
            for (0..G) |j| for (0..S) |s| for (0..W) |i| {
                t.proj[j * wide + s * W + i] = m.stressProject(j, s, i, W);
            };
            @memset(t.bias, 0);
            @memset(t.scale, 1);
        } else {
            for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (t.proj) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.05;
            for (t.bias) |*v| v.* = rng.float(f32) * 2 - 1;
            for (t.scale) |*v| v.* = 0.5 + rng.float(f32);
        }
        if (fill == .nonfinite) {
            // a row each, one row holding both infinities; a NaN post bias and an infinite scale, which saturates
            const at = [_][3]usize{ .{ 0, 0, 3 }, .{ 1, 1, 5 }, .{ 2, 2, 1 }, .{ 3, 3, 4 }, .{ 3, 0, 9 } };
            const vals = [_]f32{ h.fl.planted[0], h.fl.planted[1], h.fl.planted[2], h.fl.planted[1], h.fl.planted[2] };
            for (at, vals) |q, v| if (q[0] < rows and q[2] < W) {
                t.x[q[0] * wide + q[1] * W + q[2]] = c.toBf16(v);
            };
            if (G == 8) {
                t.bias[5] = h.fl.planted[0];
                t.scale[1] = h.fl.planted[1];
            } else t.scale[0] = h.fl.planted[1];
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.proj));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.scale));
        t.col = if (t.v.post) try t.rig.scratch(rows * W * 2) else try t.rig.output(rows * W * 2);
        t.post = if (t.v.post) try t.rig.output(rows * S * 4) else try t.rig.scratch(rows * S * 4);
        t.part = try t.rig.scratch(rows * 72 * 4);
        return true;
    }

    /// The loader's set, a control's pipeline in its slot.
    fn kern(t: *const Collapse) K {
        return grp.with(t.srv.*, t.pipe, if (t.ctl) |k| k.slot else .pre8);
    }

    fn encode(t: *Collapse, k: K, col: rig_.Ref, post: rig_.Ref, two: bool) !void {
        const p = t.rig.begin();
        try k.collapse(p.e, .{ .hidden = ref(t.ins[0]), .projection = ref(t.ins[1]), .bias = ref(t.ins[2]), .scale = ref(t.ins[3]), .output = ref(col), .post = ref(post), .partials = if (two) ref(t.part) else null }, t.ihcShape());
        try t.rig.finish(p);
    }

    pub fn run(t: *Collapse) !void {
        try t.encode(t.kern(), t.col, t.post, t.v.two);
    }

    /// The one launch's bytes of the checked output on the same inputs, run into buffers of its own.
    pub fn replaced(t: *Collapse) ![]const u8 {
        const rows = t.shape.rows;
        const col = try t.rig.scratch(rows * t.shape.n * 2);
        const post = try t.rig.scratch(rows * S * 4);
        try t.encode(t.srv.*, col, post, false);
        t.a.free(t.base_bytes);
        t.base_bytes = &.{};
        t.base_bytes = try t.a.dupe(u8, if (t.v.post) t.rig.bytes(post, rows * S * 4) else t.rig.bytes(col, rows * t.shape.n * 2));
        return t.base_bytes;
    }

    pub fn count(t: *Collapse) usize {
        return t.shape.rows * (if (t.v.post) S else t.shape.n);
    }

    fn head(t: *const Collapse) m.Head {
        return .{ .p = t.p, .width = t.shape.n, .n_gates = t.v.gates, .proj = t.proj, .bias = t.bias, .scale = t.scale, .depth = depth(t.shape) };
    }

    const Job = struct {
        t: *Collapse,
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
            const wide = S * t.shape.n;
            const buf = try t.a.alloc(f64, 2 * wide);
            defer t.a.free(buf);
            const x = buf[0..wide];
            const per = t.count() / t.shape.rows;
            for (j.rows[0]..j.rows[1]) |r| {
                for (x, t.x[r * wide ..][0..wide]) |*o, b| o.* = m.bv(b);
                const d = j.out.at(r * per, per);
                m.headRow(t.head(), j.sum, j.ranged, t.fill == .stress, x, x, x, buf[wide..], if (t.v.post) null else d, null, if (t.v.post) d else null);
            }
        }
    };

    /// Every row's outputs, in reference_threads' jobs of whole rows.
    fn rowsInto(t: *Collapse, sum: m.Sum, ranged: bool, out: m.Dest) !void {
        const rows = t.shape.rows;
        const n = contract.reference_threads.count(contract.reference_threads.maximum, rows);
        var jobs: [contract.reference_threads.maximum]Job = undefined;
        for (0..n) |i| jobs[i] = .{ .t = t, .sum = sum, .ranged = ranged, .rows = .{ rows * i / n, rows * (i + 1) / n }, .out = out };
        contract.reference_threads.run(Job, jobs[0..n], .normal);
        for (jobs[0..n]) |j| if (j.failed) return error.OutOfMemory;
    }

    fn model(t: *Collapse) !void {
        if (t.have) return;
        const n = t.count();
        t.want = try t.a.alloc(f64, n);
        t.lo = try t.a.alloc(f64, n);
        t.hi = try t.a.alloc(f64, n);
        try t.rowsInto(.{}, t.fill != .nonfinite, .{ .want = t.want, .lo = t.lo, .hi = t.hi });
        t.have = true;
    }

    pub fn reference(t: *Collapse, want: []f64, mag: []f64) !void {
        try t.model();
        @memcpy(want, t.want);
        @memset(mag, 0);
    }

    /// Each output's range: the sums within gamma(depth) (exact on gate 4's inputs), each fp32 step, each bf16 point.
    pub fn intervals(t: *Collapse, low: []f64, high: []f64) !void {
        try t.model();
        @memcpy(low, t.lo);
        @memcpy(high, t.hi);
    }

    pub fn bounds(t: *Collapse, want: []const f64, _: []const f64, allow: []f64) !void {
        try h.fl.allowFrom(t.a, t, want, allow);
    }

    pub fn simulate(t: *Collapse, acc: c.Accum, order: g.Order, out: []f64) !void {
        const scratch = try t.a.alloc(f64, 2 * out.len);
        defer t.a.free(scratch);
        try t.rowsInto(.{ .acc = acc, .order = order }, false, .{ .want = out, .lo = scratch[0..out.len], .hi = scratch[out.len..] });
    }
};
