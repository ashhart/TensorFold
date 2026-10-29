//! MOSS-Image's Metal kernels under the contract, each at its torch bf16 points; controls drop or add one such point.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const mi = @import("metal_moss_image_common.zig");
const norms = @import("metal_moss_image_norms.zig");
const steps = @import("metal_moss_image_steps.zig");

const c = mi.c;
const g = mi.g;
const ks = mi.ks;
const D = mi.D;
const kref = mi.kref;
const Ends = mi.Ends;
const Norm = norms.Norm;
const Qk = norms.Qk;
const Situ = steps.Situ;
const GateAdd = steps.GateAdd;
const Bias = steps.Bias;

// --- mi_attention_*: softmax(q k^T / sqrt(128)) v over every row, against attention_ref's bf16-P bound ------------

/// Each form's name, served lengths and edges: L % 32 > 0, L a multiple of 32 alone, and of 64.
const forms = [3]struct { name: []const u8, served: []const c.Shape, tile: c.Tile }{
    .{
        .name = "mi_attention_00",
        .served = &.{ .{ .rows = 2, .n = D, .k = 63 }, .{ .rows = 2, .n = D, .k = 4188 }, .{ .rows = 2, .n = D, .k = 4130 }, .{ .rows = 2, .n = D, .k = 34 }, .{ .rows = 2, .n = D, .k = 10 } },
        .tile = .{ .size = 32 },
    },
    .{
        .name = "mi_attention_01",
        .served = &.{ .{ .rows = 2, .n = D, .k = 32 }, .{ .rows = 2, .n = D, .k = 4128 } },
        .tile = .{ .size = 64, .step = 32, .min = 32 },
    },
    .{
        .name = "mi_attention_11",
        .served = &.{ .{ .rows = 2, .n = D, .k = 64 }, .{ .rows = 2, .n = D, .k = 4160 } },
        .tile = .{ .size = 128, .step = 64, .min = 64 },
    },
};

/// The scores' 128-deep sums, then the keys' sums.
fn attentionDepth(s: c.Shape) usize {
    return D + s.k;
}

pub const Attention = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    form: usize,
    name: []const u8,
    shape: c.Shape = undefined,
    stress: bool = false,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: heads (each the launch's own y, so two stand for the 24); n: the head width; k: the rows L.
    pub fn spec(name: []const u8, form: usize) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = attentionDepth,
            .ops = 8,
            .out = .bf16,
            .tiles = .{ .{}, .{}, forms[form].tile },
            .served = forms[form].served,
        };
    }

    pub fn deinit(t: *Attention) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Attention) void {
        t.a.free(t.q);
        t.a.free(t.k);
        t.a.free(t.v);
        t.q = &.{};
        t.k = &.{};
        t.v = &.{};
    }
    pub fn kernel(t: *Attention) g.Kernel {
        return .{ .spec = spec(t.name, t.form), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Attention);

    pub fn prepare(t: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != D or shape.k == 0 or ks.attentionForm(shape.k) != t.form) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.stress = fill == .stress;
        const L = shape.k;
        const n = shape.rows * L * D;
        t.q = try t.a.alloc(u16, n);
        t.k = try t.a.alloc(u16, n);
        t.v = try t.a.alloc(u16, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (t.stress) {
            // equal scores (q = k = 0), each column's values integers whose partial sums reach ~2^22 and cancel to 4
            @memset(t.q, 0);
            @memset(t.k, 0);
            const half = L / 2;
            for (0..shape.rows) |h| for (0..L) |j| {
                const at = if (j < half) j else j - half;
                const positive: f32 = if (at % 2 == 0) 127 else 4064;
                const residual: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
                const v: f32 = if (j < half) positive else if (j < 2 * half) residual - positive else 0;
                @memset(t.v[(h * L + j) * D ..][0..D], c.toBf16(v));
            };
        } else {
            mi.randomBf16(rng, t.q, -1, 1);
            mi.randomBf16(rng, t.k, -1, 1);
            mi.randomBf16(rng, t.v, -1, 1);
        }
        if (fill == .nonfinite) {
            // head 0: values NaN, +Inf, -Inf, and both infinities in one column; head 1: a NaN query, an infinite key
            for ([_][2]usize{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 2 }, .{ 3, 3 }, .{ 4, 3 } }, [_]u16{ mi.nan16, mi.inf16, mi.ninf16, mi.inf16, mi.ninf16 }) |p, v| if (p[0] < L) {
                t.v[p[0] * D + p[1]] = v;
            };
            if (shape.rows > 1) {
                t.q[(L + 0) * D + 5] = mi.nan16;
                if (L > 7) t.k[(L + 7) * D + 3] = mi.inf16;
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.k));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.v));
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Attention) !void {
        const p = t.rig.begin();
        ks.attend(p.e, t.pipe, kref(t.ins[0]), kref(t.ins[1]), kref(t.ins[2]), kref(t.out), t.shape.k, t.shape.rows);
        try t.rig.finish(p);
    }

    pub fn count(t: *Attention) usize {
        return t.shape.rows * t.shape.k * D;
    }

    /// attention_ref.head's formula and bound, one pass over the keys a query (its scores kept), O in [L, heads, 128].
    pub fn reference(t: *Attention, want: []f64, mag: []f64) !void {
        const L = t.shape.k;
        const heads = t.shape.rows;
        const scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, D)));
        const kf = try t.a.alloc(f64, L * D);
        defer t.a.free(kf);
        const vf = try t.a.alloc(f64, L * D);
        defer t.a.free(vf);
        const s = try t.a.alloc(f64, L);
        defer t.a.free(s);
        var qf: [D]f64 = undefined;
        var out: [D]f64 = undefined;
        var tol: [D]f64 = undefined;
        for (0..heads) |h| {
            for (kf, vf, t.k[h * L * D ..][0 .. L * D], t.v[h * L * D ..][0 .. L * D]) |*kd, *vd, kb, vb| {
                kd.* = c.fromBf16(kb);
                vd.* = c.fromBf16(vb);
            }
            for (0..L) |i| {
                var zero = true;
                for (&qf, t.q[(h * L + i) * D ..][0..D]) |*x, b| {
                    x.* = c.fromBf16(b);
                    zero = zero and x.* == 0;
                }
                var m: f64 = -std.math.inf(f64);
                var nan = false;
                for (s, 0..) |*sj, j| {
                    var dot: f64 = 0;
                    if (!zero) {
                        const Vec = @Vector(8, f64);
                        var acc: Vec = @splat(0);
                        var e: usize = 0;
                        while (e < D) : (e += 8) acc += @as(Vec, qf[e..][0..8].*) * @as(Vec, kf[j * D + e ..][0..8].*);
                        dot = @reduce(.Add, acc);
                    }
                    sj.* = dot * scale;
                    nan = nan or std.math.isNan(sj.*);
                    m = @max(m, sj.*);
                }
                if (nan) m = std.math.nan(f64);
                var l: f64 = 0;
                @memset(&out, 0);
                @memset(&tol, 0);
                for (s, 0..) |sj, j| {
                    const p = @exp(sj - m);
                    l += p;
                    for (&out, &tol, vf[j * D ..][0..D]) |*o, *tl, vv| {
                        o.* += p * vv;
                        tl.* += p * @abs(vv);
                    }
                }
                const at = (i * heads + h) * D;
                for (out, tol, 0..) |o, tl, e| {
                    want[at + e] = o / l;
                    mag[at + e] = tl / l * 0x1p-7 + @abs(o / l) * 0x1p-8 + 1e-6;
                }
            }
        }
    }

    pub fn bounds(t: *Attention, want: []const f64, tolerance: []const f64, allow: []f64) !void {
        for (want, tolerance, allow) |w, tl, *l| l.* = if (t.stress) c.boundExact(spec(t.name, t.form), w) else if (std.math.isFinite(w)) tl else 0;
    }

    pub fn simulate(t: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!t.stress) return error.AttentionStressNeedsEqualScores;
        const L = t.shape.k;
        const terms = try t.a.alloc(f64, L);
        defer t.a.free(terms);
        const ones = try t.a.alloc(f64, L);
        defer t.a.free(ones);
        @memset(ones, 1);
        const l = contract.rows.sumIn(acc, order, ones);
        for (0..t.shape.rows) |h| for (0..D) |e| {
            for (terms, 0..) |*v, j| v.* = c.fromBf16(t.v[(h * L + j) * D + e]);
            const o = contract.rows.sumIn(acc, order, terms) / l;
            for (0..L) |i| out[(i * t.shape.rows + h) * D + e] = o;
        };
    }
};

// --- the suite: the served source's pipelines, and each control's own copy of the source ------------------------

/// A control: the kernel it breaks and the one text change that drops (or adds) a rounding point.
const Control = struct { kernel: []const u8, what: []const u8, from: []const u8, to: []const u8 };

const controls = [_]?Control{
    .{ .kernel = "mi_norm", .what = "drops bf16(x inv)", .from = "mi_bf(mi_bf(float(xr[i]) * inv) * float(w[i]))", .to = "mi_bf((float(xr[i]) * inv) * float(w[i]))" },
    .{ .kernel = "mi_norm modulated", .what = "drops bf16(y s)", .from = "v = mi_bf(v * float(s[i])) + float(b[i]);", .to = "v = (v * float(s[i])) + float(b[i]);" },
    .{ .kernel = "mi_qk", .what = "drops bf16(q inv)", .from = "q[i] = mi_bf(mi_bf(q[i] * qn)", .to = "q[i] = mi_bf((q[i] * qn)" },
    .{ .kernel = "mi_qk", .what = "drops bf16(q cos)", .from = "Q[to + i] = bfloat(mi_bf(q[i] * c) + mi_bf(qp * s));", .to = "Q[to + i] = bfloat((q[i] * c) + mi_bf(qp * s));" },
    .{ .kernel = "mi_situ", .what = "drops bf16(sigmoid(g))", .from = "const float sig = mi_bf(1.0f / (1.0f + metal::precise::exp(-gv)));", .to = "const float sig = (1.0f / (1.0f + metal::precise::exp(-gv)));" },
    .{ .kernel = "mi_situ", .what = "drops bf16(tanh(h / 25))", .from = "mi_bf(25.0f * mi_bf(metal::precise::tanh(", .to = "mi_bf(25.0f * (metal::precise::tanh(" },
    .{ .kernel = "mi_gate_add", .what = "drops bf16(g y)", .from = "x[i] = bfloat(float(x[i]) + mi_bf(float(g[at.x]) * float(y[i])));", .to = "x[i] = bfloat(float(x[i]) + (float(g[at.x]) * float(y[i])));" },
    .{ .kernel = "mi_bias", .what = "adds bf16(c) before the bias", .from = "y[i] = bfloat(c[i] + float(b[at.x]));", .to = "y[i] = bfloat(mi_bf(c[i]) + float(b[at.x]));" },
};

/// The family's source as the engine compiles it, with `ctl`'s change made where it occurs, exactly once.
fn compile(a: std.mem.Allocator, device: mtl.Device, ctl: ?Control) !mtl.Library {
    const served = try tf.frags.source(device, a, sources.moss_image);
    defer a.free(served);
    const text = if (ctl) |x| blk: {
        if (std.mem.count(u8, served, x.from) != 1) return error.ControlTextNotFoundOnce;
        break :blk try std.mem.replaceOwned(u8, a, served, x.from, x.to);
    } else try a.dupe(u8, served);
    defer a.free(text);
    return mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
}

/// A source compiled the first time a kernel registered on it asks for a pipeline.
const Lazy = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    ctl: ?Control,
    lib: ?mtl.Library = null,

    fn get(l: *Lazy) !mtl.Library {
        if (l.lib == null) l.lib = try compile(l.a, l.device, l.ctl);
        return l.lib.?;
    }
    fn deinit(l: *Lazy) void {
        if (l.lib) |x| x.deinit();
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    pipes: std.ArrayList(mtl.Pipeline) = .empty,
    names: std.ArrayList([]u8) = .empty,
    norms: std.ArrayList(*Norm) = .empty,
    qks: std.ArrayList(*Qk) = .empty,
    situs: std.ArrayList(*Situ) = .empty,
    adds: std.ArrayList(*GateAdd) = .empty,
    biases: std.ArrayList(*Bias) = .empty,
    attns: std.ArrayList(*Attention) = .empty,

    fn pipeline(s: *Suite, device: mtl.Device, lib: *Lazy, name: []const u8) !mtl.Pipeline {
        const p = try mtl.Pipeline.init(device, try lib.get(), name, false);
        errdefer p.deinit();
        try s.pipes.append(s.a, p);
        return p;
    }

    fn label(s: *Suite, comptime fmt: []const u8, args: anytype) ![]const u8 {
        const name = try std.fmt.allocPrint(s.a, fmt, args);
        errdefer s.a.free(name);
        try s.names.append(s.a, name);
        return name;
    }

    fn make(s: *Suite, comptime T: type, list: *std.ArrayList(*T), value: T) !*T {
        const t = try s.a.create(T);
        errdefer s.a.destroy(t);
        t.* = value;
        try list.append(s.a, t);
        return t;
    }

    /// Each step (or, with `ctl`, the one it breaks) on `lib`'s pipelines.
    fn add(s: *Suite, device: mtl.Device, queue: mtl.Queue, lib: *Lazy, ctl: ?Control, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        const suffix = if (ctl) |x| try s.label(" control: {s}", .{x.what}) else "";
        const gate: ?g.Gate = if (ctl != null) .shapes else null;
        const rig: rig_.Rig = .{ .a = s.a, .device = device, .queue = queue };
        const ends: Ends = .{ .a = s.a };
        const want = struct {
            fn is(x: ?Control, kernel: []const u8) bool {
                return if (x) |y| std.mem.eql(u8, y.kernel, kernel) else true;
            }
            fn keep(o: ?[]const u8, name: []const u8) bool {
                return if (o) |needle| std.mem.indexOf(u8, name, needle) != null else true;
            }
        };
        for ([_]bool{ false, true }) |modulated| {
            const kind = if (modulated) "mi_norm modulated" else "mi_norm";
            const name = try s.label("metal moss image {s}{s}", .{ kind, suffix });
            if (!want.is(ctl, kind) or !want.keep(only, name)) continue;
            const t = try s.make(Norm, &s.norms, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "mi_norm"), .modulated = modulated, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]bool{ false, true }) |packed_rows| {
            const name = try s.label("metal moss image mi_qk {s}{s}", .{ if (packed_rows) "packed qkv rows" else "q k v rows", suffix });
            if (!want.is(ctl, "mi_qk") or !want.keep(only, name) or (ctl != null and packed_rows)) continue;
            const t = try s.make(Qk, &s.qks, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "mi_qk"), .packed_rows = packed_rows, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        for ([_]bool{ true, false }) |halves| {
            const name = try s.label("metal moss image mi_situ {s}{s}", .{ if (halves) "hidden | gate rows" else "two inputs", suffix });
            if (!want.is(ctl, "mi_situ") or !want.keep(only, name) or (ctl != null and !halves)) continue;
            const t = try s.make(Situ, &s.situs, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "mi_situ"), .halves = halves, .name = name, .ends = ends });
            try out.append(s.a, t.kernel(gate));
        }
        {
            const name = try s.label("metal moss image mi_gate_add{s}", .{suffix});
            if (want.is(ctl, "mi_gate_add") and want.keep(only, name)) {
                const t = try s.make(GateAdd, &s.adds, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "mi_gate_add"), .name = name, .ends = ends });
                try out.append(s.a, t.kernel(gate));
            }
        }
        {
            const name = try s.label("metal moss image mi_bias{s}", .{suffix});
            if (want.is(ctl, "mi_bias") and want.keep(only, name)) {
                const t = try s.make(Bias, &s.biases, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, "mi_bias"), .name = name, .ends = ends });
                try out.append(s.a, t.kernel(gate));
            }
        }
        if (ctl != null) return;
        for (forms, 0..) |f, form| {
            const name = try s.label("metal moss image {s}", .{f.name});
            if (!want.keep(only, name)) continue;
            const t = try s.make(Attention, &s.attns, .{ .a = s.a, .rig = rig, .pipe = try s.pipeline(device, lib, f.name), .form = form, .name = name });
            try out.append(s.a, t.kernel());
        }
    }

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        if (!device.tensorUnits()) {
            std.debug.print("UNSUPPORTED metal moss image: the family runs on tensor units (M5) alone; its kernels unchecked here\n", .{});
            return s;
        }
        for ([_]?Control{null} ++ controls) |x| {
            var lib: Lazy = .{ .a = a, .device = device, .ctl = x };
            defer lib.deinit();
            try s.add(device, queue, &lib, x, out, only);
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        const a = s.a;
        inline for (.{ &s.norms, &s.qks, &s.situs, &s.adds, &s.biases, &s.attns }) |list| {
            for (list.items) |t| {
                t.deinit();
                a.destroy(t);
            }
            list.deinit(a);
        }
        for (s.pipes.items) |p| p.deinit();
        s.pipes.deinit(a);
        for (s.names.items) |n| a.free(n);
        s.names.deinit(a);
        a.destroy(s);
    }
};
