//! Gemma 4 E4B's per-layer kernels under the contract on Metal: gemma4/dense.metal and gemma4/ple.metal, at its shapes.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const math = @import("gemma4_math.zig");
const I = math.Interval;

const geo = tf.gemma4.config.e4b;
const kernels = tf.gemma4.kernels;
const P = geo.layer_input;
const L = geo.layers;

pub const dense_name = "gemma4 E4B dense bf16 gemma_dense4";
pub const gate_name = "gemma4 E4B per-layer gate gemma_dense4_gate";
pub const control_name = "control: gemma4 E4B dense bf16 gemma_dense4 drops the last 256-deep step of K";
pub const embed_name = "gemma4 E4B per-layer table gemma_ple_embed";
pub const combine_name = "gemma4 E4B per-layer inputs gemma_ple_combine";

/// A lane's chain: K / 32 fmas (a lane takes 8 values of every 256), then the simdgroup's five-level sum.
fn denseDepth(s: c.Shape) usize {
    return s.k / 32 + 8;
}

fn uniform(rng: std.Random) f32 {
    return rng.float(f32) * 2 - 1;
}

/// The bf16 linear y = x W^T, and the per-layer gate with its gelu(gate) * input epilogue (DeepMind _modules.py Block).
pub const Dense = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    gate: bool,
    control: bool = false,
    pipe: mtl.Pipeline,
    served: [3]c.Shape,
    shape: c.Shape = undefined,
    d: ?contract.dense.Dense = null,
    wb: []u16 = &.{},
    pli: []u16 = &.{},
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    pl: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, gate: bool) !Dense {
        const pipe = try rig_.pipeline(device, kernels.Ple.dense_text, if (gate) "gemma_dense4_gate" else "gemma_dense4");
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .gate = gate, .pipe = pipe, .served = if (gate)
            .{ .{ .rows = 1, .n = P, .k = geo.hidden }, .{ .rows = 16, .n = P, .k = geo.hidden }, .{ .rows = 128, .n = P, .k = geo.hidden } }
        else
            .{ .{ .rows = 1, .n = P, .k = geo.hidden }, .{ .rows = 16, .n = geo.hidden, .k = P }, .{ .rows = 4, .n = L * P, .k = geo.hidden } } };
    }

    /// The planted control: gemma_dense4 with its K loop stopping one 256-deep step short, red at the shapes gate.
    pub fn initControl(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Dense {
        const text = try std.mem.replaceOwned(u8, a, kernels.Ple.dense_text, "k0 < K; k0 += 256", "k0 < K - 256; k0 += 256");
        defer a.free(text);
        if (std.mem.eql(u8, text, kernels.Ple.dense_text)) return error.ControlNotPlanted;
        var p = try init(a, device, queue, false);
        p.pipe.deinit();
        p.pipe = try rig_.pipeline(device, text, "gemma_dense4");
        p.control = true;
        return p;
    }

    fn drop(p: *Dense) void {
        if (p.d) |d| d.deinit();
        p.d = null;
        p.a.free(p.wb);
        p.a.free(p.pli);
        p.wb = &.{};
        p.pli = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Dense) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Dense);

    pub fn kernel(p: *Dense) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .spec = p.spec(), .control = if (p.control) .shapes else null };
    }

    fn spec(p: *const Dense) c.Spec {
        return .{ .name = if (p.control) control_name else if (p.gate) gate_name else dense_name, .accum = .f32, .depth = denseDepth, .out = .bf16, .mids = if (p.gate) 10 else 0, .tiles = .{ .{ .size = 8 }, .{ .size = 8, .step = 4, .min = 4 }, .{ .size = 256, .step = 256, .min = 256 } }, .served = &p.served, .sums = if (p.gate) .{ .f16 = 0, .bf16 = 0, .why = "staged GeGLU can mask accumulator errors; margins reported, no precision qualification claimed" } else .{ .bf16 = 30, .why = "at the served K of 2,560 the mirrored stress sums make bf16 sums miss by 34x, not 100x" } };
    }

    pub fn prepare(p: *Dense, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.rows == 0 or shape.rows > 129 or shape.n == 0 or shape.n % 4 != 0 or shape.k == 0 or shape.k % 256 != 0) return false;
        p.shape = shape;
        const d = try contract.dense.Dense.init(p.a, shape, 1);
        p.d = d;
        d.fillX(fill, seed, 64); // gate 4: terms to 12,288, every partial sum under 2^24, so fp32 is exact and fp16 overflows
        p.wb = try p.a.alloc(u16, shape.n * shape.k);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xde75e);
        const rng = prng.random();
        for (0..shape.n) |j| for (0..shape.k) |i| {
            const v: f32 = if (fill == .stress) blk: { // gate 4: integer products, the second half cancelling the first
                const m: f32 = @floatFromInt(1 + (if (i < shape.k / 2) i * 7 + j else (i - shape.k / 2) * 7 + j) % 6);
                break :blk if (i < shape.k / 2) m else -m;
            } else uniform(rng);
            p.wb[j * shape.k + i] = c.toBf16(v);
            d.w[j * shape.k + i] = c.fromBf16(p.wb[j * shape.k + i]);
        };
        p.x = try p.rig.input(std.mem.sliceAsBytes(d.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(p.wb));
        if (p.gate) {
            p.pli = try p.a.alloc(u16, shape.rows * 4 * shape.n);
            for (p.pli) |*v| v.* = c.toBf16(uniform(rng));
            p.pl = try p.rig.input(std.mem.sliceAsBytes(p.pli));
        }
        p.y = try p.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn count(p: *Dense) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn run(p: *Dense) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.x.buf, p.x.off, 0);
        e.setBuffer(p.w.buf, p.w.off, 1);
        e.setBuffer(p.y.buf, p.y.off, 2);
        e.setValue([3]i32{ @intCast(p.shape.rows), @intCast(p.shape.n), @intCast(p.shape.k) }, 3);
        if (p.gate) {
            e.setBuffer(p.pl.buf, p.pl.off, 4);
            e.setValue([2]i32{ @intCast(2 * p.shape.n), @intCast(4 * p.shape.n) }, 5);
        }
        if (p.gate) e.dispatchGroups(.{ .width = p.shape.n / 4, .height = (p.shape.rows + kernels.gate_rows - 1) / kernels.gate_rows }, .{ .width = 32 * kernels.gate_ksplit }) else e.dispatchGroups(.{ .width = (p.shape.n / 4 + 1) / 2 }, .{ .width = 64 });
        try p.rig.finish(pass);
    }

    fn pliAt(p: *const Dense, i: usize) f64 {
        const r = i / p.shape.n;
        const j = i % p.shape.n;
        return c.fromBf16(p.pli[r * 4 * p.shape.n + 2 * p.shape.n + j]);
    }

    pub fn reference(p: *Dense, want: []f64, mag: []f64) !void {
        try p.d.?.reference(want, mag);
        if (p.gate) for (want, 0..) |*w, i| {
            w.* = math.geglu(w.*, pliAt(p, i));
        };
    }

    pub fn simulate(p: *Dense, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.d.?.simulate(acc, order, out);
        if (p.gate) for (out, 0..) |*w, i| {
            w.* = math.geglu(w.*, pliAt(p, i));
        };
    }

    /// The dot product's fp32 error (gamma times its magnitude) through the host GeGLU bound, plus an absolute floor.
    pub fn bounds(p: *Dense, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (!p.gate) {
            for (allow, want, mag) |*l, w, m| l.* = c.bound(p.spec(), p.shape, w, m);
            return;
        }
        const n = want.len;
        const dots = try p.a.alloc(f64, 2 * n);
        defer p.a.free(dots);
        try p.d.?.reference(dots[0..n], dots[n..]);
        const gamma = c.gamma(@floatFromInt(denseDepth(p.shape)), 0x1p-24);
        for (allow, want, 0..) |*l, w, i| {
            const e = gamma * dots[n + i];
            const range = math.gegluBound(.{ .lo = dots[i] - e, .hi = dots[i] + e }, I.point(pliAt(p, i)));
            l.* = @max(range.hi - w, w - range.lo, 0) + 0x1p-133;
        }
    }
};

/// gemma_ple_embed: the per-layer table's rows (4-bit codes in groups of 64, dequantized once to bf16) times sqrt(P).
pub const Embed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    served: [2]c.Shape,
    rows: usize = 0,
    ids: []u32 = &.{},
    codes: []u8 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    args: [4]rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    const vocab = 5;
    const wide = L * P;
    const scale: f32 = 16;

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Embed {
        const text = try kernels.Ple.source(a, geo);
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, text, "gemma_ple_embed"), .served = .{ .{ .rows = 1, .n = wide, .k = 1 }, .{ .rows = 17, .n = wide, .k = 1 } } };
    }

    fn drop(p: *Embed) void {
        p.a.free(p.ids);
        p.a.free(p.codes);
        p.a.free(p.scales);
        p.a.free(p.biases);
        p.ids = &.{};
        p.codes = &.{};
        p.scales = &.{};
        p.biases = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Embed) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Embed);

    fn depth(_: c.Shape) usize {
        return 2;
    }

    pub fn kernel(p: *Embed) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .spec = .{ .name = embed_name, .accum = .none, .depth = depth, .ops = 3, .mids = 1, .out = .bf16, .served = &p.served, .sums = .{ .f16 = 0, .bf16 = 0, .why = "dequantization has no accumulation" } } };
    }

    pub fn prepare(p: *Embed, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.n != wide or s.rows == 0 or s.rows > 129) return false;
        p.drop();
        p.rows = s.rows;
        p.ids = try p.a.alloc(u32, p.rows);
        for (p.ids, 0..) |*id, i| id.* = @intCast(i % vocab);
        p.codes = try p.a.alloc(u8, vocab * wide);
        const storage = try p.a.alloc(u8, vocab * wide / 2);
        defer p.a.free(storage);
        @memset(storage, 0);
        for (p.codes, 0..) |*code, i| {
            code.* = @intCast((i *% 37 +% seed) % 16);
            storage[i / 2] |= code.* << @as(u3, @intCast(4 * (i % 2)));
        }
        p.scales = try p.a.alloc(u16, vocab * wide / 64);
        p.biases = try p.a.alloc(u16, p.scales.len);
        for (p.scales, p.biases, 0..) |*sc, *bias, i| {
            sc.* = c.toBf16(@as(f32, @floatFromInt(i % 7 + 1)) / 32);
            bias.* = c.toBf16(@as(f32, @floatFromInt(i % 5)) / 8 - 0.25);
        }
        if (fill == .nonfinite) p.scales[0] = 0x7fc0;
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
        p.args[1] = try p.rig.input(storage);
        p.args[2] = try p.rig.input(std.mem.sliceAsBytes(p.scales));
        p.args[3] = try p.rig.input(std.mem.sliceAsBytes(p.biases));
        p.y = try p.rig.output(p.rows * wide * 2);
        return true;
    }

    pub fn count(p: *Embed) usize {
        return p.rows * wide;
    }

    pub fn run(p: *Embed) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.args, 0..) |arg, i| e.setBuffer(arg.buf, arg.off, i);
        e.setBuffer(p.y.buf, p.y.off, 4);
        e.setValue(scale, 5);
        e.dispatchThreads(mtl.Size.of(wide / 2, p.rows, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Embed, want: []f64, mag: []f64) !void {
        for (0..p.rows) |r| for (0..wide) |i| {
            const id = p.ids[r];
            const at = id * (wide / 64) + i / 64;
            const weight = math.roundedWeight(c.fromBf16(p.scales[at]), p.codes[id * wide + i], c.fromBf16(p.biases[at]));
            const value = weight * scale;
            want[r * wide + i] = c.fromBf16(c.toBf16(@floatCast(value)));
            mag[r * wide + i] = @abs(value);
        };
    }
};

/// gemma_ple_combine: DeepMind _modules.py encode_per_layer_input, (RMSNorm(projection/sqrt(D)) + table)/sqrt(2).
pub const Combine = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    served: [2]c.Shape,
    rows: usize = 0,
    prj: []u16 = &.{},
    tok: []u16 = &.{},
    nw: []u16 = &.{},
    args: [3]rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    const wide = L * P;
    const eps: f32 = 1e-6;
    const psc: f32 = @as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(geo.hidden)));
    const isc: f32 = @as(f32, 1.0) / @sqrt(@as(f32, 2.0));
    // The kernel multiplies by these scales as bf16 values.
    const psc_b: f64 = c.fromBf16(c.toBf16(psc));
    const isc_b: f64 = c.fromBf16(c.toBf16(isc));

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Combine {
        const text = try kernels.Ple.source(a, geo);
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, text, "gemma_ple_combine"), .served = .{ .{ .rows = 1, .n = wide, .k = P }, .{ .rows = 9, .n = wide, .k = P } } };
    }

    fn drop(p: *Combine) void {
        p.a.free(p.prj);
        p.a.free(p.tok);
        p.a.free(p.nw);
        p.prj = &.{};
        p.tok = &.{};
        p.nw = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Combine) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Combine);

    fn depth(s: c.Shape) usize {
        return s.k + 16;
    }

    pub fn kernel(p: *Combine) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .spec = .{ .name = combine_name, .accum = .f32, .depth = depth, .ops = 6, .mids = 5, .out = .bf16, .served = &p.served, .sums = .{ .f16 = 0, .bf16 = 0, .why = "the sum of squares feeds a normalization; margins reported, no precision qualification claimed" } } };
    }

    pub fn prepare(p: *Combine, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.n != wide or s.k != P or s.rows == 0 or s.rows > 129) return false;
        p.drop();
        p.rows = s.rows;
        p.prj = try p.a.alloc(u16, p.rows * wide);
        p.tok = try p.a.alloc(u16, p.rows * wide);
        p.nw = try p.a.alloc(u16, P);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xc0b1);
        const rng = prng.random();
        for (p.prj) |*v| v.* = c.toBf16(uniform(rng) * 40);
        for (p.tok) |*v| v.* = c.toBf16(uniform(rng) * 2);
        for (p.nw) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        if (fill == .stress) { // gate 4's sums are the sum of squares: equal values, exact in fp32 in any order
            for (p.prj) |*v| v.* = c.toBf16(64);
        }
        if (fill == .nonfinite) {
            p.prj[3] = 0x7fc0; // row 0, layer 0: NaN
            if (p.rows > 1) p.prj[wide + 5] = 0x7f80; // row 1: +inf
        }
        p.args[0] = try p.rig.input(std.mem.sliceAsBytes(p.prj));
        p.args[1] = try p.rig.input(std.mem.sliceAsBytes(p.tok));
        p.args[2] = try p.rig.input(std.mem.sliceAsBytes(p.nw));
        p.y = try p.rig.output(p.rows * wide * 2);
        return true;
    }

    pub fn count(p: *Combine) usize {
        return p.rows * wide;
    }

    pub fn run(p: *Combine) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        e.setBuffer(p.args[0].buf, p.args[0].off, 0);
        e.setBuffer(p.args[1].buf, p.args[1].off, 1);
        e.setBuffer(p.args[2].buf, p.args[2].off, 2);
        e.setValue(eps, 3);
        e.setValue(psc, 4);
        e.setValue(isc, 5);
        e.setBuffer(p.y.buf, p.y.off, 6);
        e.dispatchGroups(.{ .width = L, .height = p.rows }, .{ .width = P });
        try p.rig.finish(pass);
    }

    /// The values the group's norm reads: the group's projection times the bf16 scale 1/sqrt(D), rounded to bf16.
    fn group(p: *const Combine, r: usize, l: usize, xs: []u16) void {
        for (xs, 0..) |*x, i| x.* = c.toBf16(@floatCast(@as(f64, c.fromBf16(p.prj[r * wide + l * P + i])) * psc_b));
    }

    pub fn reference(p: *Combine, want: []f64, mag: []f64) !void {
        var xs: [P]u16 = undefined;
        for (0..p.rows) |r| for (0..L) |l| {
            p.group(r, l, &xs);
            const inv = math.inverse(&xs, eps).value;
            for (0..P) |i| {
                const at = r * wide + l * P + i;
                // DeepMind encode_per_layer_input: norm, weight, table add and 1/sqrt(2) scale, each rounded to bf16
                const xn = math.bf(c.fromBf16(xs[i]) * inv);
                const nrm = math.bf(xn * c.fromBf16(p.nw[i]));
                const sum = math.bf(nrm + c.fromBf16(p.tok[at]));
                want[at] = math.bf(sum * isc_b);
                mag[at] = @abs(sum * isc_b);
            }
        };
    }

    pub fn simulate(p: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        _ = acc;
        _ = order;
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        try p.reference(out, mag);
    }

    pub fn bounds(p: *Combine, want: []const f64, _: []const f64, allow: []f64) !void {
        var xs: [P]u16 = undefined;
        for (0..p.rows) |r| for (0..L) |l| {
            p.group(r, l, &xs);
            const inv = math.inverse(&xs, eps).bound;
            for (0..P) |i| {
                const at = r * wide + l * P + i;
                // each fp32 op and bf16 store as an outward interval, from the bound on 1/sqrt(mean(x^2) + eps)
                const xn = I.point(c.fromBf16(xs[i])).mul(inv).rounded();
                const nrm = xn.mul(I.point(c.fromBf16(p.nw[i]))).rounded();
                const sum = nrm.add(I.point(c.fromBf16(p.tok[at]))).rounded();
                const out = sum.mul(I.point(isc_b)).rounded();
                const w = want[at];
                allow[at] = if (std.math.isFinite(w)) @max(out.hi - w, w - out.lo, 0) + 0x1p-133 else 0;
            }
        };
    }
};

pub fn limits(a: std.mem.Allocator, device: mtl.Device, only: ?[]const u8) !usize {
    var problems: usize = 0;
    const text = try kernels.Ple.source(a, geo);
    defer a.free(text);
    const list = [_]struct { name: []const u8, function: []const u8, source: []const u8, threads: u32 }{
        .{ .name = dense_name, .function = "gemma_dense4", .source = kernels.Ple.dense_text, .threads = 64 },
        .{ .name = gate_name, .function = "gemma_dense4_gate", .source = kernels.Ple.dense_text, .threads = 32 * kernels.gate_ksplit },
        .{ .name = embed_name, .function = "gemma_ple_embed", .source = text, .threads = 256 },
        .{ .name = combine_name, .function = "gemma_ple_combine", .source = text, .threads = P },
    };
    for (list) |v| {
        if (only) |s| if (std.mem.indexOf(u8, v.name, s) == null) continue;
        problems += try @import("metal_limits.zig").onThisGpu(a, device, .{ .decl = .{ .name = v.name, .threads = v.threads, .tg_memory = if (std.mem.eql(u8, v.function, "gemma_ple_combine")) @as(u32, P / 32 * 4) else if (std.mem.eql(u8, v.function, "gemma_dense4_gate")) @as(u32, kernels.gate_ksplit * 8 * 4 * 4) else 0, .offline = false }, .function = v.function, .here = v.source, .pre_m5 = "" }, false);
    }
    return problems;
}
