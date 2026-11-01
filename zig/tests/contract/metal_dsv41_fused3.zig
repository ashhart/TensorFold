//! DeepSeek V4.1's fused RMS-norm launches (normq, prenorm) on Metal under the contract; controls 1-4.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_glue_ref.zig");
const rows_ = contract.rows;

const c = contract.spec;
const g = contract.gates;
const fu = tf.dsv41_metal.fused;
const base = "dsv41 fused ";
const eps: f32 = 1e-20;

/// The widths tf_dsv41_f_norm is instantiated at; each serves dim = 256 * NV exactly, so every slot is written once.
const nvs = [_]usize{ 1, 2, 5, 8, 20, 32 };

/// normq at every width; prenorm up to 2048; wide above it, where the host splits into pre_a and pre_b.
pub const Fam = enum { normq, prenorm, wide };

/// every fam but normq collapses four streams by pre into H first
fn collapses(f: Fam) bool {
    return f != .normq;
}

fn dimOf(nv: usize) usize {
    return 256 * nv;
}

fn perRow(fam: Fam) usize {
    return if (collapses(fam)) 3 else 2; // H, HN, HQ when the four streams collapse; else HN, HQ
}

/// The collapse, then a lane's chain over every 256th square, the butterfly and the eight partials.
fn normDepth(s: c.Shape) usize {
    return s.k / 256 + 12;
}

const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

/// The spec's fixed part; the served shapes live in the probe (their k is the width, chosen at runtime).
pub fn spec(name: []const u8) c.Spec {
    const t = [_]c.Tile{ .{ .size = 16 }, .{}, .{} };
    return .{ .name = name, .accum = .f32, .depth = normDepth, .ops = 8, .out = .bf16, .tiles = t, .served = &.{}, .sums = .{ .bf16 = 0, .why = rms_why } };
}

fn controlGate(ctl: u8) ?g.Gate {
    return switch (ctl) {
        1 => .nonfinite,
        2 => .shapes,
        3 => .sums,
        else => .repeat,
    };
}

fn at(r: rig_.Ref, extra: usize) fu.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

const vt = g.hooks(Proj);
const plain_vt = blk: {
    var v = g.hooks(Proj);
    v.replaced = null; // a control is proven at its own gate, not against the baseline
    break :blk v;
};

/// The fused pipelines hold every kernel of fused.metal; every probe of a control shares one compiled copy.
var shared: [5]?*fu.Pipelines = .{ null, null, null, null, null };

fn sharedPipes(a: std.mem.Allocator, device: mtl.Device, ctl: u8) !*fu.Pipelines {
    if (shared[ctl]) |p| return p;
    const p = try a.create(fu.Pipelines);
    if (ctl == 0) {
        p.* = try fu.Pipelines.load(device);
    } else {
        const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.fused });
        defer a.free(text);
        p.* = try fu.Pipelines.loadSource(device, text);
    }
    shared[ctl] = p;
    return p;
}

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    fam: Fam,
    nv: usize,
    ctl: u8,
    name: []const u8,
    pipes: *fu.Pipelines,
    served: [3]c.Shape = undefined, // {8,1,5} rows at this probe's width k = 256 * nv
    shape: c.Shape = undefined,
    dim: usize = 0,
    x: []u16 = &.{}, // normq [rows][dim]; prenorm [rows][4][dim]
    pre: []f32 = &.{}, // prenorm's [rows][4]
    w: []f32 = &.{}, // the norm weight [dim]
    in_x: rig_.Ref = undefined,
    in_pre: rig_.Ref = undefined,
    in_w: rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined, // (H,) HN, HQ in slot order
    hn: usize = 0, // HN's output slot: 1 for prenorm (H sits before it), 0 for normq
    part: rig_.Ref = undefined, // bound only on the wide split; unused at these dims

    /// `dim` is the row width: 256 * nv for the one-launch fams, given outright for the wide split.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, f: Fam, nv: usize, ctl: u8, name: []const u8, dim: usize) !Proj {
        const d = if (dim != 0) dim else dimOf(nv);
        return .{
            .a = a,
            .arena = std.heap.ArenaAllocator.init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .fam = f,
            .nv = nv,
            .ctl = ctl,
            .name = name,
            .pipes = try sharedPipes(a, device, ctl),
            .dim = d,
            .served = .{ .{ .rows = 8, .n = 1, .k = d }, .{ .rows = 1, .n = 1, .k = d }, .{ .rows = 5, .n = 1, .k = d } },
            .hn = @intFromBool(collapses(f)),
        };
    }

    pub fn deinit(p: *Proj) void {
        p.arena.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        var s = spec(p.name);
        s.served = &p.served; // the widths are per probe, so the served list lives in the probe itself
        if (p.ctl == 0) return .{ .spec = s, .ctx = p, .vt = &vt, .control = null };
        return .{ .spec = s, .ctx = p, .vt = &plain_vt, .control = controlGate(p.ctl) };
    }

    pub fn count(p: *Proj) usize {
        return p.shape.rows * perRow(p.fam) * p.dim;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != p.dim or shape.rows == 0) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        const n = shape.rows;
        const streams: usize = if (collapses(p.fam)) 4 else 1;
        p.x = try a.alloc(u16, n * streams * p.dim);
        p.w = try a.alloc(f32, p.dim);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.x) |*o| o.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (rng.uintLessThan(u32, 64) == 7) @as(f32, 20) else 1));
        for (p.w) |*o| o.* = 0.5 + rng.float(f32);
        p.pre = if (collapses(p.fam)) try a.alloc(f32, n * 4) else &.{};
        if (collapses(p.fam)) {
            for (p.pre) |*o| o.* = 0.2 + rng.float(f32);
        }
        if (fill == .nonfinite) p.plant(n);
        if (fill == .stress) try p.stress(a, n, seed);
        p.in_x = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.in_w = try p.rig.input(std.mem.sliceAsBytes(p.w));
        p.in_pre = if (collapses(p.fam)) try p.rig.input(std.mem.sliceAsBytes(p.pre)) else p.in_x;
        if (collapses(p.fam)) p.outs[0] = try p.rig.output(n * p.dim * 2);
        p.outs[p.hn] = try p.rig.output(n * p.dim * 2);
        p.outs[p.hn + 1] = try p.rig.output(n * p.dim * 2);
        p.part = try p.rig.scratch(fu.normScratch(n));
        return true;
    }

    /// Five non-finites across four rows, one a row: the reference's total is then non-finite on those rows only.
    fn plant(p: *Proj, n: usize) void {
        const streams: usize = if (collapses(p.fam)) 4 else 1;
        const spots = [_][2]usize{ .{ 0, p.dim / 3 }, .{ 1, p.dim / 2 }, .{ 2, p.dim - 1 }, .{ 3, 0 }, .{ 3, p.dim - 1 } };
        const put = [_]u16{ c.toBf16(std.math.nan(f32)), c.toBf16(std.math.inf(f32)), c.toBf16(-std.math.inf(f32)), c.toBf16(std.math.inf(f32)), c.toBf16(-std.math.inf(f32)) };
        for (spots, put) |q, v| if (q[0] < n) {
            p.x[q[0] * streams * p.dim + q[1]] = v;
        };
    }

    /// Gate 4's rows: the rms stress in stream 0 (the others zero), pre one-hot, so fp32 sums are exact integers.
    fn stress(p: *Proj, a: std.mem.Allocator, n: usize, seed: u64) !void {
        var r = try rows_.Rows.init(a, .rms, .{ .rows = n, .n = 1, .k = p.dim });
        defer r.deinit();
        r.fill(.stress, seed);
        const streams: usize = if (collapses(p.fam)) 4 else 1;
        @memset(p.x, 0);
        for (0..n) |row| {
            for (0..p.dim) |i| p.x[row * streams * p.dim + i] = r.x[row * p.dim + i];
        }
        for (p.w, r.w) |*o, v| o.* = @floatCast(c.fromBf16(v));
        if (collapses(p.fam)) {
            for (0..n) |row| {
                for (0..4) |k| p.pre[row * 4 + k] = if (k == 0) 1 else 0;
            }
        }
    }

    /// A slot's h before rounding: the stream read (normq) or the pre-weighted collapse of four (prenorm).
    fn hRaw(p: *Proj, r: usize, i: usize) f64 {
        if (p.fam == .normq) return c.fromBf16(p.x[r * p.dim + i]);
        const s = r * 4 * p.dim;
        const pr = p.pre[r * 4 ..];
        return ((c.fromBf16(p.x[s + i]) * pr[0] + c.fromBf16(p.x[s + p.dim + i]) * pr[1]) + pr[2] * c.fromBf16(p.x[s + 2 * p.dim + i])) + pr[3] * c.fromBf16(p.x[s + 3 * p.dim + i]);
    }

    fn hMag(p: *Proj, r: usize, i: usize) f64 {
        if (p.fam == .normq) return @abs(c.fromBf16(p.x[r * p.dim + i]));
        const s = r * 4 * p.dim;
        const pr = p.pre[r * 4 ..];
        var ab: f64 = 0;
        for (0..4) |k| ab += @abs(pr[k] * c.fromBf16(p.x[s + k * p.dim + i]));
        return ab;
    }

    /// (H,) HN and HQ of every row: HN is w*(h*inv); HQ is HN's bf16 as fp8 (NaN where its block is not finite).
    fn compute(p: *Proj, want: []f64, mag: []f64, acc: c.Accum, order: rows_.Order) !void {
        const n = p.shape.rows;
        const dim = p.dim;
        const a = p.arena.allocator();
        const hh = try a.alloc(f64, dim); // h as the kernel holds it: rounded to bf16 once
        const hnc = try a.alloc(f64, dim);
        const sq = try a.alloc(f64, dim);
        const h0: usize = if (collapses(p.fam)) 0 else n * dim;
        const hn0 = (if (collapses(p.fam)) n else 0) * dim;
        const hq0 = (if (collapses(p.fam)) 2 * n else n) * dim;
        var xs: [32]f32 = undefined;
        var bits: [32]u16 = undefined;
        for (0..n) |r| {
            for (0..dim) |i| {
                const raw = p.hRaw(r, i);
                if (collapses(p.fam)) {
                    want[h0 + r * dim + i] = raw;
                    mag[h0 + r * dim + i] = p.hMag(r, i);
                }
                hh[i] = c.fromBf16(c.toBf16(@floatCast(raw)));
                sq[i] = hh[i] * hh[i];
            }
            var ss: f64 = 0;
            if (acc == .none) {
                for (sq) |v| ss += v;
            } else ss = rows_.sumIn(acc, order, sq);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(dim)) + eps);
            for (0..dim) |i| {
                const hn = p.w[i] * (hh[i] * inv);
                hnc[i] = hn;
                want[hn0 + r * dim + i] = hn;
                mag[hn0 + r * dim + i] = @abs(hn) / 2;
            }
            var blk: usize = 0;
            while (blk < dim) : (blk += 32) {
                for (0..32) |j| xs[j] = @floatCast(c.fromBf16(c.toBf16(@floatCast(hnc[blk + j]))));
                ref.roundTrip(.fp8, xs[0..32], bits[0..32]);
                for (0..32) |j| {
                    want[hq0 + r * dim + blk + j] = c.fromBf16(bits[j]);
                    mag[hq0 + r * dim + blk + j] = @abs(hnc[blk + j]) / 2;
                }
            }
        }
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        try p.compute(want, mag, .none, .sequential);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const scratch = try p.arena.allocator().alloc(f64, p.count());
        try p.compute(out, scratch, acc, order);
    }

    /// HQ is E4M3 of HN's bf16: a whole quantum |v|/8, plus HN's own allowance and a floor for the scale's rounding.
    pub fn bounds(p: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        const n = p.shape.rows * p.dim;
        const hq: usize = @as(usize, if (collapses(p.fam)) 2 else 1) * n;
        const hn: usize = @as(usize, if (collapses(p.fam)) 1 else 0) * n;
        var blk: usize = 0;
        while (blk < n) : (blk += 32) {
            var amax: f64 = 0;
            for (want[hq + blk ..][0..32]) |v| amax = @max(amax, @abs(v));
            for (allow[hq + blk ..][0..32], want[hq + blk ..][0..32], allow[hn + blk ..][0..32]) |*l, v, hnl| l.* = @abs(v) * 0x1p-3 + 2 * hnl + amax * 0x1p-8;
        }
    }

    /// Rows r0 to r0 + nr of the prepared buffers: the one fused norm launch the host API offers.
    pub fn encode(p: *Proj, e: mtl.ComputeEncoder, r0: usize, nr: usize) !void {
        const rows: u32 = @intCast(nr);
        const step = p.dim * 2;
        if (p.fam == .normq) {
            try fu.normq(p.pipes, e, at(p.in_x, r0 * step), at(p.in_w, 0), at(p.outs[0], r0 * step), at(p.outs[1], r0 * step), rows, @intCast(p.dim), eps);
        } else {
            const n: fu.Norm = .{
                .x = at(p.in_x, r0 * 4 * p.dim * 2),
                .pre = at(p.in_pre, r0 * 16),
                .w = at(p.in_w, 0),
                .h = at(p.outs[0], r0 * step),
                .hn = at(p.outs[1], r0 * step),
                .hq = at(p.outs[2], r0 * step),
                .part = at(p.part, 0),
                .dim = @intCast(p.dim),
                .eps = eps,
            };
            try fu.prenorm(p.pipes, e, n, rows);
        }
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }
};

/// Every registration and its controls, plus the row-invariance probe over the baseline's launches.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { f: Fam, nv: usize, ctl: u8, name: []const u8, dim: usize = 0 };
    const list = blk: {
        var out: [29]E = undefined;
        var i: usize = 0;
        for (nvs) |nv| {
            out[i] = .{ .f = .normq, .nv = nv, .ctl = 0, .name = base ++ "normq v" ++ nvStr(nv) ++ " tf_dsv41_f_normq_v" ++ nvStr(nv) };
            i += 1;
        }
        for (nvs) |nv| {
            if (dimOf(nv) > 2048) continue; // above 2048 the host splits the norm into pre_a and pre_b
            out[i] = .{ .f = .prenorm, .nv = nv, .ctl = 0, .name = base ++ "prenorm v" ++ nvStr(nv) ++ " tf_dsv41_f_prenorm_v" ++ nvStr(nv) };
            i += 1;
        }
        out[i] = .{ .f = .normq, .nv = 8, .ctl = 1, .name = base ++ "control: normq reads every non-finite input as zero" };
        out[i + 1] = .{ .f = .normq, .nv = 8, .ctl = 2, .name = base ++ "control: normq adds one to every value of its row" };
        out[i + 2] = .{ .f = .normq, .nv = 8, .ctl = 3, .name = base ++ "control: normq lane totals flushed to fp16" };
        out[i + 3] = .{ .f = .normq, .nv = 8, .ctl = 4, .name = base ++ "control: normq writes past its row" };
        out[i + 4] = .{ .f = .prenorm, .nv = 8, .ctl = 1, .name = base ++ "control: prenorm reads every non-finite input as zero" };
        out[i + 5] = .{ .f = .prenorm, .nv = 8, .ctl = 2, .name = base ++ "control: prenorm loses the fourth residual stream" };
        out[i + 6] = .{ .f = .prenorm, .nv = 8, .ctl = 3, .name = base ++ "control: prenorm lane totals flushed to fp16" };
        out[i + 7] = .{ .f = .prenorm, .nv = 8, .ctl = 4, .name = base ++ "control: prenorm writes past its row" };
        i += 8;
        // the wide prenorm: pre_a sums a row by simdgroup into part, pre_b finishes it; the host splits above 2048
        out[i] = .{ .f = .wide, .nv = 0, .ctl = 0, .dim = 2560, .name = base ++ "pre a v20 tf_dsv41_f_pre_a_v20" };
        out[i + 1] = .{ .f = .wide, .nv = 0, .ctl = 0, .dim = 8192, .name = base ++ "pre a v32 tf_dsv41_f_pre_a_v32" };
        out[i + 2] = .{ .f = .wide, .nv = 0, .ctl = 0, .dim = 2560, .name = base ++ "pre b tf_dsv41_f_pre_b" };
        out[i + 3] = .{ .f = .wide, .nv = 0, .ctl = 1, .dim = 2560, .name = base ++ "control: pre a reads every non-finite input as zero" };
        out[i + 4] = .{ .f = .wide, .nv = 0, .ctl = 2, .dim = 2560, .name = base ++ "control: pre a loses the fourth residual stream" };
        out[i + 5] = .{ .f = .wide, .nv = 0, .ctl = 3, .dim = 2560, .name = base ++ "control: pre a lane totals flushed to fp16" };
        out[i + 6] = .{ .f = .wide, .nv = 0, .ctl = 4, .dim = 2560, .name = base ++ "control: pre a writes past its partials" };
        out[i + 7] = .{ .f = .wide, .nv = 0, .ctl = 2, .dim = 8192, .name = base ++ "control: pre a v32 loses the fourth residual stream" };
        out[i + 8] = .{ .f = .wide, .nv = 0, .ctl = 3, .dim = 8192, .name = base ++ "control: pre a v32 lane totals flushed to fp16" };
        out[i + 9] = .{ .f = .wide, .nv = 0, .ctl = 2, .dim = 2560, .name = base ++ "control: pre b drops a simdgroup partial" };
        out[i + 10] = .{ .f = .wide, .nv = 0, .ctl = 4, .dim = 2560, .name = base ++ "control: pre b writes past its act_quant" };
        break :blk out;
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.f, entry.nv, entry.ctl, entry.name, entry.dim);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        for (&shared) |*p| if (p.*) |pp| {
            pp.deinit();
            s.a.destroy(pp);
            p.* = null;
        };
        s.a.destroy(s);
    }

    /// Launches of 1, 7 and 11 rows a launch equal one launch's bytes: no value crosses a row.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var kernels: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            kernels += 1;
            const shape = c.Shape{ .rows = 33, .n = 1, .k = p.dim };
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            try p.run();
            const whole = try s.a.dupe(u8, try p.rig.outputBytes());
            defer s.a.free(whole);
            for ([_]usize{ 1, 7, 11 }) |step| {
                for (p.outs[0..perRow(p.fam)]) |o| @memset(p.rig.bytes(o, 33 * p.dim * 2), 0x55);
                const pass = p.rig.begin();
                var r0: usize = 0;
                while (r0 < shape.rows) : (r0 += step) {
                    try p.encode(pass.e, r0, @min(step, shape.rows - r0));
                    launches += 1;
                }
                try p.rig.finish(pass);
                if (!std.mem.eql(u8, whole, try p.rig.outputBytes())) {
                    bad += 1;
                    std.debug.print("FAIL row invariance {s}: {d} rows a launch differs from one launch\n", .{ p.name, step });
                }
            }
        }
        if (kernels == 0) return 0;
        std.debug.print("{s} row invariance {s}: {d} kernels at 1, 7 and 11 rows a launch ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }
};

fn nvStr(nv: usize) []const u8 {
    return switch (nv) {
        1 => "1",
        2 => "2",
        5 => "5",
        8 => "8",
        20 => "20",
        else => "32",
    };
}
