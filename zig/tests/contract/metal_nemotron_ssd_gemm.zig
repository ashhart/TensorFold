//! Nemotron's SSD steps on gemm_nax's five f32 kernels on the M5: both fp32 operands cut, each product's cut one-sided.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;

/// Nemotron 3.5 Lightning's Mamba heads, head dim and state (config.checkShapes), and prefill's 256-row SSD step.
const heads = 64;
const head_dim = 64;
const state = 128;
const step = 256;

/// prefill_mixers.mamba's f32 GEMMs a step of r rows: y = sur x dtx (r x r, r x 64), dd^T x bh^T (64 x r, r x 128).
const Product = enum { y, state };

/// An entry point by its tags (A and B transposed; whole M, N and K blocks) and the shapes the step calls it at.
const Entry = struct { product: Product, tags: []const u8, served: []const c.Shape };

fn at(n: usize, k: usize) c.Shape {
    return .{ .rows = heads, .n = n, .k = k };
}

/// A shape's rows are the heads (the launch's batch), n its columns, k the step's rows r: y's M is r, the state's 64.
const entries = [_]Entry{
    .{ .product = .y, .tags = "n_n_n_n_n", .served = &.{ at(head_dim, 100), at(head_dim, 255), at(head_dim, 63), at(head_dim, 33), at(head_dim, 1) } },
    .{ .product = .y, .tags = "n_n_t_n_n", .served = &.{ at(head_dim, 64), at(head_dim, 128), at(head_dim, 192) } },
    .{ .product = .y, .tags = "n_n_t_n_t", .served = &.{at(head_dim, step)} },
    .{ .product = .state, .tags = "t_t_t_t_n", .served = &.{ at(state, 100), at(state, 255), at(state, 192), at(state, 64), at(state, 1) } },
    .{ .product = .state, .tags = "t_t_t_t_t", .served = &.{at(state, step)} },
};

/// Random operands; operands in (0, 1], where the cut's bias adds up; those stated as fp32; a source edit.
const Case = enum { plain, positive, fp32, edited };

const Reg = struct { entry: usize, case: Case, name: []const u8 };

const regs = [_]Reg{
    .{ .entry = 0, .case = .plain, .name = "metal nemotron ssd y gemm_nax_f32_n_n_n_n_n" },
    .{ .entry = 1, .case = .plain, .name = "metal nemotron ssd y gemm_nax_f32_n_n_t_n_n" },
    .{ .entry = 2, .case = .plain, .name = "metal nemotron ssd y gemm_nax_f32_n_n_t_n_t" },
    .{ .entry = 3, .case = .plain, .name = "metal nemotron ssd state gemm_nax_f32_t_t_t_t_n" },
    .{ .entry = 4, .case = .plain, .name = "metal nemotron ssd state gemm_nax_f32_t_t_t_t_t" },
    .{ .entry = 2, .case = .positive, .name = "metal nemotron ssd y gemm_nax_f32_n_n_t_n_t, operands in (0, 1]" },
    .{ .entry = 4, .case = .positive, .name = "metal nemotron ssd state gemm_nax_f32_t_t_t_t_t, operands in (0, 1]" },
    .{ .entry = 2, .case = .fp32, .name = "control: metal nemotron ssd y gemm_nax_f32_n_n_t_n_t, operands in (0, 1] stated as fp32" },
    .{ .entry = 4, .case = .fp32, .name = "control: metal nemotron ssd state gemm_nax_f32_t_t_t_t_t, operands in (0, 1] stated as fp32" },
    .{ .entry = 0, .case = .edited, .name = "control: metal nemotron ssd y gemm_nax_f32_n_n_n_n_n, its ragged K tail one short" },
};

/// The edited control's one line: the last 32-deep step past the whole K blocks reads one column fewer.
const edit_from = "max(0, rest - kk)";
const edit_to = "max(0, rest - kk - 1)";

/// gemm_step's two mma_16x32 k16 steps a 32-deep step, zero-padded past K, each mma_step roundings of |acc| + sum |ab|.
fn depth(s: c.Shape) usize {
    return ar.mma_step * 2 * ((s.k + 31) / 32);
}

fn spec(r: Reg) c.Spec {
    return .{ .name = r.name, .accum = .f32, .depth = depth, .out = .f32, .served = entries[r.entry].served };
}

fn tag(flag: bool) []const u8 {
    return if (flag) "t" else "n";
}

/// The entry point prefill_launch.gemm names for these sizes is this entry's.
fn fits(tags: []const u8, m: usize, n: usize, k: usize) bool {
    var buf: [5]u8 = undefined;
    const want = std.fmt.bufPrint(&buf, "{s}_{s}_{s}", .{ tag(m % 64 == 0), tag(n % 128 == 0), tag(k % 256 == 0) }) catch unreachable;
    return std.mem.eql(u8, tags[4..], want);
}

/// Each product's float64 cut: the operands' magnitudes shrink by up to 2^-10 each, so a product's by c (2 - c).
fn productCut(r: Reg) f64 {
    const cut: f64 = if (r.case == .fp32) 0 else ar.operand_cut;
    return cut * (2 - cut);
}

/// float64's own roundings in the reference, and fp32's least step.
const slack = 1 + 0x1p-20;
const floor = 0x1p-149;

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    reg: Reg,
    batch: usize = 0,
    m: usize = 0,
    n: usize = 0,
    k: usize = 0,
    stress: bool = false,
    /// A and B as bound: y's sur [heads, r, r], dtx [r, heads * 64]; the state's dd [r, heads * 64], bh [heads, 128, r]
    x: []f32 = &.{},
    w: []f32 = &.{},
    /// Per output: float64's value, the fp32 sums' allowance, and the cut's reach down (positive terms) and up.
    point: []f64 = &.{},
    err: []f64 = &.{},
    down: []f64 = &.{},
    up: []f64 = &.{},
    params: [16]i32 = @splat(0),
    refs: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, r: Reg) !Probe {
        const text = for (sources.prefill) |f| {
            if (std.mem.eql(u8, f.name, "gemm_nax")) break f.text;
        } else return error.UnknownSource;
        const built = try tf.frags.source(device, a, text);
        defer a.free(built);
        var src: []u8 = built;
        if (r.case == .edited) {
            if (std.mem.count(u8, built, edit_from) != 1) return error.ControlEditMissing;
            src = try std.mem.replaceOwned(u8, a, built, edit_from, edit_to);
        }
        defer if (r.case == .edited) a.free(src);
        const fun = try std.fmt.allocPrint(a, "custom_kernel_tf_gemm_nax_f32_{s}_64_128_256_2_4_float_float_int32_t_float", .{entries[r.entry].tags});
        defer a.free(fun);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, fun), .reg = r };
    }

    fn drop(p: *Probe) void {
        p.a.free(p.x);
        p.a.free(p.w);
        for ([_][]f64{ p.point, p.err, p.down, p.up }) |v| p.a.free(v);
        p.x = &.{};
        p.w = &.{};
        p.point = &.{};
        p.err = &.{};
        p.down = &.{};
        p.up = &.{};
        p.rig.reset();
    }

    fn deinit(p: *Probe) void {
        p.drop();
        p.pipe.deinit();
        p.rig.deinit();
    }

    const vt = g.hooks(Probe);

    fn kernel(p: *Probe) g.Kernel {
        const control: ?c.Gate = if (p.reg.case == .fp32 or p.reg.case == .edited) .shapes else null;
        return .{ .spec = spec(p.reg), .ctx = p, .vt = &vt, .control = control };
    }

    fn isY(p: *const Probe) bool {
        return entries[p.reg.entry].product == .y;
    }

    fn lda(p: *const Probe) usize {
        return if (p.isY()) p.k else p.batch * p.m;
    }

    fn ldb(p: *const Probe) usize {
        return if (p.isY()) p.batch * p.n else p.k;
    }

    /// Head b's A[i][kk] and B[kk][j] in the stored arrays.
    fn aAt(p: *const Probe, b: usize, i: usize, kk: usize) usize {
        return if (p.isY()) (b * p.k + i) * p.k + kk else kk * p.lda() + b * p.m + i;
    }

    fn bAt(p: *const Probe, b: usize, kk: usize, j: usize) usize {
        return if (p.isY()) kk * p.ldb() + b * p.n + j else (b * p.n + j) * p.k + kk;
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const m: usize = if (p.isY()) s.k else head_dim;
        if (s.rows == 0 or s.n == 0 or s.k == 0 or !fits(entries[p.reg.entry].tags, m, s.n, s.k)) return false;
        p.drop();
        p.batch = s.rows;
        p.m = m;
        p.n = s.n;
        p.k = s.k;
        p.stress = fill == .stress;
        const outs = p.count();
        p.x = try p.a.alloc(f32, p.batch * p.m * p.k);
        p.w = try p.a.alloc(f32, p.batch * p.k * p.n);
        for ([_]*[]f64{ &p.point, &p.err, &p.down, &p.up }) |v| v.* = try p.a.alloc(f64, outs);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (p.stress) p.stressFill(rng) else {
            const positive = p.reg.case == .positive or p.reg.case == .fp32;
            for (p.x) |*v| v.* = if (positive) 1 - rng.float(f32) else rng.float(f32) * 2 - 1;
            for (p.w) |*v| v.* = if (positive) 1 - rng.float(f32) else rng.float(f32) * 2 - 1;
        }
        if (fill == .nonfinite) {
            // NaN, +Inf and -Inf in head 0's A rows 0 to 2, both infinities in row 3, -Inf in head 1's B column 1
            const k = p.k;
            const put = [_][3]usize{ .{ 0, k / 3, 0 }, .{ 1, k / 2, 1 }, .{ 2, k - 1, 2 }, .{ 3, 0, 1 }, .{ 3, k - 1, 2 } };
            const vals = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put) |q| if (q[0] < p.m) {
                p.x[p.aAt(0, q[0], q[1])] = vals[q[2]];
            };
            if (p.batch > 1) p.w[p.bAt(1, 0, 1)] = -std.math.inf(f32);
        }
        const tn = (p.n + 127) / 128;
        const tm = (p.m + 63) / 64;
        const strides = if (p.isY()) [2]usize{ p.k * p.k, p.n } else [2]usize{ p.m, p.n * p.k };
        const v = [_]usize{ p.m, p.n, p.k, p.lda(), p.ldb(), p.n, tn, tm, 2, p.k / 256, strides[0], strides[1], p.m * p.n, 0, 0 };
        for (v, 0..) |x, i| p.params[i] = @intCast(x);
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.w));
        p.out = try p.rig.output(outs * 4);
        return true;
    }

    /// Gate 4: 11-bit integers times 5-bit ones, which the cut keeps: halves past fp16 and bf16, partials below 2^24.
    fn stressFill(p: *Probe, rng: std.Random) void {
        const h = p.k / 2;
        @memset(p.x, 0);
        @memset(p.w, 0);
        for (0..p.batch) |b| {
            for (0..p.m) |i| for (0..h) |kk| {
                const v: f32 = @floatFromInt(1024 + rng.uintLessThan(u32, 1024));
                p.x[p.aAt(b, i, kk)] = v;
                p.x[p.aAt(b, i, h + kk)] = v + @as(f32, if (kk < 8) 1 else 0);
            };
            for (0..h) |kk| for (0..p.n) |j| {
                const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 31));
                p.w[p.bAt(b, kk, j)] = w;
                p.w[p.bAt(b, h + kk, j)] = -w;
            };
        }
    }

    /// prefill_launch.gemm's dispatch: 32 x 4 x 2 threads a group, swizzle 2, one z pair a head.
    pub fn run(p: *Probe) !void {
        const tn = (p.n + 127) / 128;
        const tm = (p.m + 63) / 64;
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        pass.e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
        pass.e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
        pass.e.setBytes(std.mem.asBytes(&p.params), 2);
        pass.e.setBuffer(p.out.buf, p.out.off, 3);
        pass.e.dispatchThreads(mtl.Size.of((tn << 2) * 32, (tm + 3) / 4 * 4, p.batch * 2), mtl.Size.of(32, 4, 2));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.batch * p.m * p.n;
    }

    /// Head b's A as [m][k] and B as [n][k] in float64, each output's terms contiguous.
    fn unpack(p: *const Probe, b: usize, am: []f64, bt: []f64) void {
        for (0..p.m) |i| for (0..p.k) |kk| {
            am[i * p.k + kk] = p.x[p.aAt(b, i, kk)];
        };
        for (0..p.n) |j| for (0..p.k) |kk| {
            bt[j * p.k + kk] = p.w[p.bAt(b, kk, j)];
        };
    }

    /// D = A B in float64 on the operands as given, the fp32 sums' allowance and the cut's one-sided reach.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const K = p.k;
        const buf = try p.a.alloc(f64, (p.m + p.n) * K);
        defer p.a.free(buf);
        const am = buf[0 .. p.m * K];
        const bt = buf[p.m * K ..];
        const cut = productCut(p.reg);
        const sums = c.gamma(@floatFromInt(depth(.{ .rows = p.batch, .n = p.n, .k = K })), 0x1p-24);
        for (0..p.batch) |b| {
            p.unpack(b, am, bt);
            for (0..p.m) |i| for (0..p.n) |j| {
                var s: f64 = 0;
                var t: f64 = 0;
                var pos: f64 = 0;
                var neg: f64 = 0;
                for (am[i * K ..][0..K], bt[j * K ..][0..K]) |x, y| {
                    const q = x * y;
                    s += q;
                    t += @abs(q);
                    if (q > 0) pos += q else neg -= q;
                }
                const o = (b * p.m + i) * p.n + j;
                want[o] = s;
                mag[o] = t;
                p.err[o] = sums * t;
                p.down[o] = cut * pos;
                p.up[o] = cut * neg;
            };
        }
        @memcpy(p.point, want);
    }

    /// The sums' allowance plus the cut's wider side; gate 4's exact sums keep the exact bound.
    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const s = spec(p.reg);
        for (want, allow, p.err, p.down, p.up) |w, *l, e, d, u| {
            l.* = if (p.stress) c.boundExact(s, w) else (e + @max(d, u)) * slack + floor;
        }
    }

    /// Each output within float64's value less the sums' allowance and the positive terms' cut, plus the negative's.
    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        const s = spec(p.reg);
        for (low, high, p.point, p.err, p.down, p.up) |*l, *h, pt, e, d, u| {
            if (p.stress) {
                l.* = pt - c.boundExact(s, pt);
                h.* = pt + c.boundExact(s, pt);
            } else {
                l.* = pt - (e + d) * slack - floor;
                h.* = pt + (e + u) * slack + floor;
            }
        }
    }

    /// Each output's products summed in the accumulator and order asked (gate 4's operands survive the cut whole).
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const K = p.k;
        const buf = try p.a.alloc(f64, (p.m + p.n + 1) * K);
        defer p.a.free(buf);
        const am = buf[0 .. p.m * K];
        const bt = buf[p.m * K ..][0 .. p.n * K];
        const terms = buf[(p.m + p.n) * K ..];
        for (0..p.batch) |b| {
            p.unpack(b, am, bt);
            for (0..p.m) |i| for (0..p.n) |j| {
                for (terms, am[i * K ..][0..K], bt[j * K ..][0..K]) |*q, x, y| q.* = x * y;
                out[(b * p.m + i) * p.n + j] = contract.rows.sumIn(acc, order, terms);
            };
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    /// Only on the tensor units: before the M5, nax.h's SIMD fragments keep both fp32 operands whole.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (regs) |r| {
            if (only) |o| if (std.mem.indexOf(u8, r.name, o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, r);
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
