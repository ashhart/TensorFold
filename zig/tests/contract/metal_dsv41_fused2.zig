//! DeepSeek V4.1's fused act_quant and SwiGLU on Metal, against dsv41_glue_ref's bit-exact fp8 e4m3 round trip.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_glue_ref.zig");

const c = contract.spec;
const g = contract.gates;
const fu = tf.dsv41_metal.fused;
const base = "dsv41 fused ";

/// One block of the quant: 32 values, one simdgroup's lanes.
const blk: usize = 32;

const quant_rule = "a block of 32 holding a NaN or an infinity is all NaN; every other output is the round trip, exact";

pub const Fam = enum { actq, swiglu };

fn noDepth(_: c.Shape) usize {
    return 0;
}

/// SwiGLU's clamp, the same limit the served expert uses.
const limit: f32 = 7.0;

pub fn spec(f: Fam, name: []const u8) c.Spec {
    const t = [_]c.Tile{ .{ .size = 16 }, .{}, .{ .size = 256, .step = blk, .min = blk } };
    return switch (f) {
        .actq => .{ .name = name, .accum = .none, .depth = noDepth, .out = .bf16, .nan = .{ .stated = quant_rule }, .tiles = t, .served = &.{
            .{ .rows = 1, .n = 1, .k = 2048 }, .{ .rows = 16, .n = 1, .k = 2048 }, .{ .rows = 5, .n = 1, .k = 512 }, .{ .rows = 8, .n = 1, .k = 128 },
        } },
        // served[0] one row a call so gate 4's short run puts control 4's past-the-end write outside its output
        .swiglu => .{ .name = name, .accum = .none, .depth = noDepth, .out = .bf16, .tiles = t, .served = &.{
            .{ .rows = 1, .n = 1, .k = 2048 }, .{ .rows = 16, .n = 1, .k = 2048 }, .{ .rows = 5, .n = 1, .k = 512 },
        } },
    };
}

fn controlGate(f: Fam, ctl: u8) ?g.Gate {
    _ = f;
    return switch (ctl) {
        1 => .nonfinite,
        2 => .shapes,
        else => .repeat,
    };
}

fn at(r: rig_.Ref, extra: usize) fu.Ref {
    return .{ .buf = r.buf, .off = r.off + extra };
}

const vt = g.hooks(Proj);
const vt_bound = blk_: {
    var v = g.hooks(Proj);
    v.check = null; // SwiGLU's y comes from fp32 fast math: bounded, not bit-exact
    break :blk_ v;
};

fn tableFor(f: Fam, ctl: u8) *const g.VTable {
    if (ctl != 0) return &plain_vt;
    return if (f == .swiglu) &vt_bound else &vt;
}

const plain_vt = blk__: {
    var v = g.hooks(Proj);
    v.replaced = null; // a control is proven at its own gate, not against the baseline
    v.check = null;
    break :blk__ v;
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
    ctl: u8,
    name: []const u8,
    pipes: *fu.Pipelines,
    shape: c.Shape = undefined,
    gbf: []u16 = &.{}, // the gate stream (swiglu) and the only input (actq)
    ubf: []u16 = &.{}, // swiglu's up stream
    out: [2]rig_.Ref = undefined, // y, then its quant
    in_g: rig_.Ref = undefined,
    in_u: rig_.Ref = undefined,
    plants: usize = 0,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, f: Fam, ctl: u8, name: []const u8) !Proj {
        return .{
            .a = a,
            .arena = std.heap.ArenaAllocator.init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .fam = f,
            .ctl = ctl,
            .name = name,
            .pipes = try sharedPipes(a, device, ctl),
        };
    }

    pub fn deinit(p: *Proj) void {
        p.arena.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const s = spec(p.fam, p.name);
        if (p.ctl == 0) return .{ .spec = s, .ctx = p, .vt = tableFor(p.fam, 0), .control = null };
        return .{ .spec = s, .ctx = p, .vt = tableFor(p.fam, p.ctl), .control = controlGate(p.fam, p.ctl) };
    }

    pub fn count(p: *Proj) usize {
        const n = p.shape.rows * p.shape.k;
        return if (p.fam == .swiglu) 2 * n else n;
    }

    fn spread(rng: std.Random) f32 {
        return (rng.float(f32) * 2 - 1) * std.math.pow(f32, 2, @floatFromInt(rng.intRangeAtMost(i32, -6, 5)));
    }

    /// A block whose amax sits on a scale boundary, so the round trip and its ties are exercised per block.
    fn fillBlock(rng: std.Random, x: []u16, start: usize) void {
        const spike = std.math.pow(f32, 2, @floatFromInt(rng.intRangeAtMost(i32, 2, 6)));
        for (0..blk) |j| {
            const scale: f32 = if (j == 0) 1.0 else 0x1p-4;
            const v: f32 = spread(rng) * spike * scale;
            x[start + j] = c.toBf16(if (rng.boolean()) -v else v);
        }
        x[start + 7] = 0; // a zero block member, and a tie candidate
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k % blk != 0 or shape.rows == 0) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.k;
        p.gbf = try a.alloc(u16, n);
        for (0..n / blk) |b| fillBlock(rng, p.gbf, b * blk);
        if (p.fam == .swiglu) {
            p.ubf = try a.alloc(u16, n);
            for (p.ubf) |*o| o.* = c.toBf16(spread(rng) * limit);
        }
        p.plants = 0;
        if (fill == .nonfinite) {
            const nan = c.toBf16(std.math.nan(f32));
            const inf = c.toBf16(std.math.inf(f32));
            const ninf = c.toBf16(-std.math.inf(f32));
            p.gbf[blk / 3] = nan;
            p.gbf[shape.k + 1] = inf; // row 1
            p.gbf[2 * shape.k + shape.k - 1] = ninf;
            p.gbf[3 * shape.k + blk] = inf;
            p.gbf[4 * shape.k + blk + 5] = nan;
            if (p.fam == .swiglu) p.ubf[blk * 2] = ninf; // the up stream's own non-finite, its own block
        }
        p.plants = if (fill == .nonfinite) 5 else 0;
        const gb = std.mem.sliceAsBytes(p.gbf);
        p.out[0] = try p.rig.output(n * 2);
        p.out[1] = try p.rig.output(n * 2);
        if (p.fam == .swiglu) {
            if (fill == .nonfinite) p.plants += 1;
            const ub = std.mem.sliceAsBytes(p.ubf);
            p.in_g = try p.rig.input(gb);
            p.in_u = try p.rig.input(ub);
        } else {
            p.in_g = try p.rig.input(gb);
            p.in_u = p.in_g;
        }
        return true;
    }

    /// Rows r0 to r0 + n of the prepared call as one launch; the quant strides the rows contiguously.
    pub fn encode(p: *Proj, e: mtl.ComputeEncoder, r0: usize, n: usize) !void {
        const k: u32 = @intCast(p.shape.k);
        const rows: u32 = @intCast(n);
        switch (p.fam) {
            .actq => try fu.actq(p.pipes, e, at(p.in_g, r0 * p.shape.k * 2), at(p.out[0], r0 * p.shape.k * 2), .{
                .k = k,
                .nb = 1,
                .na = rows,
                .sa = k,
                .sb = 0,
                .ta = k,
                .tb = 0,
            }),
            .swiglu => fu.swiglu(p.pipes, e, at(p.in_g, r0 * p.shape.k * 2), at(p.in_u, r0 * p.shape.k * 2), at(p.out[0], r0 * p.shape.k * 2), at(p.out[1], r0 * p.shape.k * 2), k * rows, limit),
        }
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }

    /// y in the oracle's float64: the clamped gate through the sigmoid times the clamped up stream; actq reads x.
    fn yOf(p: *const Proj, i: usize) f64 {
        if (p.fam == .actq) return c.fromBf16(p.gbf[i]);
        const gg = c.fromBf16(p.gbf[i]);
        const uu = c.fromBf16(p.ubf[i]);
        // the kernel clips g only at the top (clip(g, -INF, limit)) and u both ways
        const cl: f64 = if (gg != gg) gg else @min(gg, @as(f64, limit));
        const cu: f64 = if (uu != uu) uu else @min(@max(uu, -@as(f64, limit)), @as(f64, limit));
        return cl / (1 + @exp(-cl)) * cu;
    }

    /// y where the kernel emits it (swiglu only), then every quant block from the oracle's round trip.
    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const n = p.shape.rows * p.shape.k;
        const q0 = if (p.fam == .swiglu) n else 0;
        if (p.fam == .swiglu) for (0..n) |i| {
            want[i] = p.yOf(i);
            mag[i] = @abs(want[i]);
        };
        var xs: [blk]f32 = undefined;
        var bits: [blk]u16 = undefined;
        for (0..n / blk) |b| {
            for (0..blk) |j| xs[j] = @floatCast(p.yOf(b * blk + j));
            ref.roundTrip(.fp8, xs[0..blk], bits[0..blk]);
            for (0..blk) |j| {
                want[q0 + b * blk + j] = c.fromBf16(bits[j]);
                mag[q0 + b * blk + j] = @abs(want[q0 + b * blk + j]);
            }
        }
    }

    /// y (swiglu) keeps the float64 bound; every quant slot allows one fp8 quantum, the grid it lands on.
    pub fn bounds(p: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        const n = p.shape.rows * p.shape.k;
        const q0 = if (p.fam == .swiglu) n else 0;
        // y is bf16: half a bf16 quantum; an fp32-subnormal input may flush to 0, so allow its full value
        for (allow[0..q0], mag[0..q0], 0..) |*l, m, i| {
            const sub = (c.fromBf16(p.gbf[i]) < 0x1p-126) or (p.fam == .swiglu and c.fromBf16(p.ubf[i]) < 0x1p-126);
            l.* = if (sub) m + 0x1p-133 else std.math.pow(f64, 2, @floatFromInt(std.math.frexp(m).exponent - 9)) + 0x1p-133;
        }
        for (allow[q0..], mag[q0..]) |*l, m| {
            // the fp8 quantum at m, measured from the high side so a value rounding UP to a power of 2 still fits
            const e: i32 = if (m > 0) std.math.frexp(m).exponent else -6;
            const q: i32 = if (e > -6) e - 4 else -9; // normal: half of 2^(e-3); subnormal: half of 2^-9 rounded up
            l.* = std.math.pow(f64, 2, @floatFromInt(q)) + 0x1p-133;
        }
    }

    /// Every quant block against the oracle's bits (any NaN equals any NaN); gate 1 also needs the plants landed.
    pub fn check(p: *Proj, gate: g.Gate) !?[]const u8 {
        const n = p.shape.rows * p.shape.k;
        const q = p.rig.bytes(p.out[if (p.fam == .swiglu) 1 else 0], n * 2);
        const got = std.mem.bytesAsSlice(u16, q);
        var xs: [blk]f32 = undefined;
        var want: [blk]u16 = undefined;
        var bad: usize = 0;
        for (0..n / blk) |b| {
            for (0..blk) |j| xs[j] = @floatCast(p.yOf(b * blk + j));
            ref.roundTrip(.fp8, xs[0..blk], want[0..blk]);
            for (want[0..blk], got[b * blk ..][0..blk], 0..) |w, h, i| {
                const wn = (w & 0x7fff) > 0x7f80;
                if (w != h and !(wn and (h & 0x7fff) > 0x7f80)) return try std.fmt.bufPrint(&p.msg, "block {d} element {d}: got 0x{x:0>4}, the round trip gives 0x{x:0>4}", .{ b, i, h, w });
            }
            if ((want[0] & 0x7fff) > 0x7f80) bad += 1;
        }
        if (gate == .nonfinite and (p.plants < 5 or bad < 4)) return try std.fmt.bufPrint(&p.msg, "{d} plants, {d} NaN blocks: no proof", .{ p.plants, bad });
        return null;
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        _ = p;
        _ = acc;
        _ = order;
        _ = out;
        return error.NoSums;
    }
};

/// Every registration and its controls, plus the row-invariance probe over the baseline's launches.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { f: Fam, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .f = .actq, .ctl = 0, .name = base ++ "actq tf_dsv41_f_actq" },
        .{ .f = .actq, .ctl = 1, .name = base ++ "control: actq a non-finite input read as zero" },
        .{ .f = .actq, .ctl = 4, .name = base ++ "control: actq writes past its row" },
        .{ .f = .swiglu, .ctl = 0, .name = base ++ "swiglu tf_dsv41_f_swiglu" },
        .{ .f = .swiglu, .ctl = 1, .name = base ++ "control: swiglu a non-finite input read as zero" },
        .{ .f = .swiglu, .ctl = 2, .name = base ++ "control: swiglu loses the up half of the gate" },
        .{ .f = .swiglu, .ctl = 4, .name = base ++ "control: swiglu writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.f, entry.ctl, entry.name);
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
            const shape = c.Shape{ .rows = 33, .n = 1, .k = 2048 };
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            try p.run();
            const whole = try s.a.dupe(u8, try p.rig.outputBytes());
            defer s.a.free(whole);
            for ([_]usize{ 1, 7, 11 }) |step| {
                const size = p.count() * 2;
                @memset(p.rig.bytes(p.out[0], @min(size, p.shape.rows * p.shape.k * 2)), 0x55);
                if (p.fam == .swiglu) @memset(p.rig.bytes(p.out[1], p.shape.rows * p.shape.k * 2), 0x55);
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
        std.debug.print("{s} row invariance {s} elementwise: {d} kernels at 1, 7 and 11 rows a launch ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }
};
