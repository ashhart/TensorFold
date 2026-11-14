//! EXL3's decode-row kernels under the contract on HIP: rot_in and the linear at each width, against float64.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const fmt = core.exl3_format;
const exl3 = hip.launches.exl3;

/// What a registration launches; a control runs its kind's kernel with one planted fault and must go red at gate 3.
pub const Reg = struct { rot: bool = false, halves: u32 = 8, bf16_out: bool = false, control: bool = false };

/// rot_in, the linear at every width with fp32 out (the head's) and at 4 bits with bf16 out, then a control of each.
pub const registrations = blk: {
    var list: [exl3.widths.len + 4]Reg = undefined;
    list[0] = .{ .rot = true };
    for (exl3.widths, 1..) |h, i| list[i] = .{ .halves = h };
    list[exl3.widths.len + 1] = .{ .bf16_out = true };
    list[exl3.widths.len + 2] = .{ .rot = true, .control = true };
    list[exl3.widths.len + 3] = .{ .control = true };
    break :blk list;
};

/// Whole 128-wide blocks from one block up: K and N of the linear.
const blocks: c.Tile = .{ .size = 128, .step = 128, .min = 128 };

fn sh(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

// deepseek_v41's indexer wk (16, 64, 1 rows), gate 4's (Engram's K, one block), Engram's wkv, the 8-wave edge, below it
const linear_shapes = [_]c.Shape{
    sh(16, 128, 512),   sh(64, 128, 512),  sh(1, 128, 512),    sh(1, 128, 6144),
    sh(4, 25600, 6144), sh(4, 8192, 1024), sh(16, 8192, 1024), sh(4, 8064, 1024),
};
// the head, 5120 to 129,280 at one row, registered at 8 bits: N sets only the grid
const head_shapes = linear_shapes ++ [_]c.Shape{sh(1, 129280, 5120)};
// rot_in: n is a row's width (each output sums 128 terms); the same calls' inputs
const rot_shapes = [_]c.Shape{ sh(64, 512, 128), sh(16, 6144, 128), sh(2, 5120, 128) };

/// A K sum in any order, a wave's chain and the waves', then the output rotation's seven butterflies.
fn depth(s: c.Shape) usize {
    return s.k + 64;
}

/// The product x * suh, then seven butterflies.
fn rotDepth(_: c.Shape) usize {
    return 8;
}

const rot_why = "128-term rotations of bf16 inputs: fp32 is not exact on them, so gate 4 proves fp16's miss alone";
const bf16_why = "bf16 outputs hide a bf16 sum's miss on these inputs; the fp32-output registrations prove fp32 sums";

fn name(r: Reg) []const u8 {
    if (r.control and r.rot) return "control: tf_exl3_rot_in reads the next block's suh";
    if (r.control) return "control: tf_exl3_linear reads its input rows one row late";
    if (r.rot) return "hip exl3 tf_exl3_rot_in (bf16 rows to fp16)";
    inline for (exl3.widths) |h| if (h == r.halves) {
        const fmt_ = "hip exl3 tf_exl3_linear_k{d}_w4 / _w8 ({d}-bit, 1-64 rows, {s} out)";
        if (r.bf16_out) return std.fmt.comptimePrint(fmt_, .{ h, h / 2, "bf16" });
        return std.fmt.comptimePrint(fmt_, .{ h, h / 2, "fp32" });
    };
    unreachable;
}

pub fn spec(r: Reg) c.Spec {
    if (r.rot) return .{
        .name = name(r),
        .accum = .f32,
        .depth = rotDepth,
        .ops = 2,
        .out = .f16,
        .tiles = .{ .{}, .{ .size = 512, .step = 128, .min = 128 }, .{} },
        .served = &rot_shapes,
        .sums = .{ .f16 = 4, .bf16 = 0, .exact = false, .why = rot_why },
    };
    return .{
        .name = name(r),
        .accum = .f32,
        .depth = depth,
        .ops = 3,
        .out = if (r.bf16_out) .bf16 else .f32,
        .tiles = .{ .{ .size = 16 }, blocks, blocks },
        .served = if (r.halves == 16 and !r.bf16_out and !r.control) &head_shapes else &linear_shapes,
        .sums = if (r.bf16_out) .{ .bf16 = 0, .why = bf16_why } else .{},
    };
}

pub const Case = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    reg: Reg,
    shape: c.Shape = undefined,
    /// rot_in: x (bf16) and suh; the linear: xh (fp16), svh and the trellis in the checkpoint's layout
    x: []u16 = &.{},
    scale: []u16 = &.{},
    trellis: []i16 = &.{},
    dx: u64 = 0,
    dscale: u64 = 0,
    dt: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, reg: Reg) Case {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .reg = reg };
    }

    pub fn deinit(self: *Case) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Case) void {
        self.rig.a.free(self.x);
        self.rig.a.free(self.scale);
        self.rig.a.free(self.trellis);
        self.x = &.{};
        self.scale = &.{};
        self.trellis = &.{};
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.reg), .ctx = self, .vt = &vt, .control = if (self.reg.control) .shapes else null };
    }

    const vt = g.hooks(Case);

    fn random16(rng: std.Random, out: []u16) void {
        for (out) |*v| v.* = @bitCast(@as(f16, @floatCast(rng.float(f32) * 2 - 1)));
    }

    /// Random inputs; non-finite: row 1's NaN, row 2's +Inf and -Inf; stress: see each kind's branch.
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = if (self.reg.rot) shape.n else shape.k;
        if (!exl3.fits(shape.rows, k, if (self.reg.rot) 128 else shape.n)) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.a;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // a control reads one block (suh) or one row (xh) past the call's own, so those inputs carry one more
        const extra_rows: usize = @intFromBool(self.reg.control and !self.reg.rot);
        self.x = try a.alloc(u16, (shape.rows + extra_rows) * k);
        const scales = if (self.reg.rot) k else shape.n;
        self.scale = try a.alloc(u16, scales + if (self.reg.control) @as(usize, 128) else 0);
        random16(rng, self.scale);
        if (self.reg.rot) {
            for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
            // stress: inputs to 64, as CUDA's rot_group (fp32 is not exact on 128-term rotations of them)
            if (fill == .stress) for (self.x) |*v| {
                v.* = c.toBf16(c.fromBf16(v.*) * 64);
            };
        } else {
            const w = fmt.Width{ .halves = @intCast(self.reg.halves) };
            self.trellis = try a.alloc(i16, (k / 16) * (shape.n / 16) * w.tileWords());
            random16(rng, self.x);
            rng.bytes(std.mem.sliceAsBytes(self.trellis));
            // stress: all-ones states (one weight, -0.651), svh 1, xh 4096: K equal terms past fp16's range
            if (fill == .stress) {
                @memset(self.trellis, @bitCast(@as(u16, 0xffff)));
                @memset(self.scale, 0x3c00);
                @memset(self.x, @bitCast(@as(f16, 4096)));
            }
            self.dt = try self.rig.input(std.mem.sliceAsBytes(self.trellis));
        }
        const nan: u16 = if (self.reg.rot) 0x7fc0 else 0x7e00;
        const inf: u16 = if (self.reg.rot) 0x7f80 else 0x7c00;
        if (fill == .nonfinite and shape.rows > 2) {
            self.x[k + 3] = nan;
            self.x[2 * k + 5] = inf;
            self.x[2 * k + 9] = inf | 0x8000;
        }
        self.dx = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.dscale = try self.rig.input(std.mem.sliceAsBytes(self.scale));
        self.out = try self.rig.output(self.count() * (if (self.reg.rot or self.reg.bf16_out) @as(usize, 2) else 4));
        return true;
    }

    pub fn count(self: *Case) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn run(self: *Case) !void {
        const s = self.rig.s.handle;
        const rows = self.shape.rows;
        const fault: u64 = if (self.reg.control) 1 else 0;
        if (self.reg.rot) {
            try exl3.rotIn(self.l, self.dx, 2, self.dscale + fault * 128 * 2, self.out, rows, self.shape.n, s);
        } else {
            const k = self.shape.k;
            const w = fmt.Width{ .halves = @intCast(self.reg.halves) };
            const tw: u64 = w.tileWords() / 2;
            const t: exl3.Trellis = .{
                .words = self.dt,
                .halves = self.reg.halves,
                .stride_k = (self.shape.n / 16) * tw,
                .stride_nb = 8 * tw,
            };
            const kind: c_int = if (self.reg.bf16_out) 2 else 0;
            try exl3.linear(self.l, self.dx + fault * k * 2, t, self.dscale, self.out, kind, rows, k, self.shape.n, s);
        }
        try self.rig.s.synchronize();
    }

    /// Float64 outputs and their terms' magnitude; with `acc`, each output's sums in that accumulator and order.
    fn outputs(self: *Case, want: []f64, mag: ?[]f64, acc: ?c.Accum, order: g.Order) !void {
        const a = self.rig.a;
        const rows = self.shape.rows;
        const n = self.shape.n;
        if (self.reg.rot) {
            // xh[j] = sum over q of H[q][j] / sqrt(128) * x[q] suh[q], block by block
            const h = try a.alloc(f64, 128 * 128);
            defer a.free(h);
            @memset(h, 0);
            for (0..128) |q| h[q * 128 + q] = 1;
            fmt.rotate(h, 128, 128, 1);
            var t: [128]f64 = undefined;
            for (0..rows) |r| for (0..n / 128) |b| for (0..128) |j| {
                var abs: f64 = 0;
                for (&t, 0..) |*e, q| {
                    const i = b * 128 + q;
                    e.* = h[q * 128 + j] * c.fromBf16(self.x[r * n + i]) * half(self.scale[i]);
                    abs += @abs(e.*);
                }
                want[r * n + b * 128 + j] = if (acc) |ac| contract.rows.sumIn(ac, order, &t) else sum(&t);
                if (mag) |m| m[r * n + b * 128 + j] = abs;
            };
            return;
        }
        const k = self.shape.k;
        const wq = try a.alloc(f16, k * n);
        defer a.free(wq);
        Unpack.all(self.reg.halves, self.trellis, k, n, wq);
        const s1 = try a.alloc(f64, rows * n);
        defer a.free(s1);
        if (acc) |ac| {
            const t = try a.alloc(f64, threads.maximum * k);
            defer a.free(t);
            const sims: Sims.Args = .{ .x = self.x, .wq = wq, .rows = rows, .k = k, .n = n, .acc = ac, .order = order };
            Sims.all(sims, t, want[0 .. rows * n]);
        } else Sums.all(self.x, wq, rows, k, n, want[0 .. rows * n], s1);
        rotate(want[0 .. rows * n], rows, n);
        for (0..rows) |r| for (0..n) |j| {
            const sc = half(self.scale[j]);
            want[r * n + j] *= sc;
            if (mag) |m| {
                var s: f64 = 0;
                for (j / 128 * 128..j / 128 * 128 + 128) |q| s += s1[r * n + q];
                m[r * n + j] = @abs(sc) * s / @sqrt(128.0);
            }
        };
    }

    /// Each row's 128-wide blocks through the unscaled butterflies, then the scale: exact where K sums are integers.
    fn rotate(x: []f64, rows: usize, n: usize) void {
        for (0..rows * n / 128) |b| {
            const v = x[b * 128 ..][0..128];
            var d: usize = 1;
            while (d < 128) : (d *= 2) {
                var base: usize = 0;
                while (base < 128) : (base += 2 * d) for (0..d) |j| {
                    const lo = v[base + j];
                    const hi = v[base + j + d];
                    v[base + j] = lo + hi;
                    v[base + j + d] = lo - hi;
                };
            }
            for (v) |*e| e.* /= @sqrt(128.0);
        }
    }

    fn sum(t: []const f64) f64 {
        var s: f64 = 0;
        for (t) |v| s += v;
        return s;
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null, .sequential);
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.outputs(out, null, acc, order);
    }
};

const threads = contract.reference_threads;

/// W decoded as exl3_format's unpack does, by slices of K tiles on threads.
const Unpack = struct {
    halves: u32,
    trellis: []const i16,
    n: usize,
    tk0: usize,
    tk1: usize,
    wq: []f16,

    pub fn run(j: *Unpack) void {
        const w = fmt.Width{ .halves = @intCast(j.halves) };
        const row = w.tileWords() * (j.n / 16);
        const words = j.trellis[j.tk0 * row .. j.tk1 * row];
        fmt.unpack(w, .mul1, words, j.tk1 - j.tk0, j.n / 16, j.wq[j.tk0 * 16 * j.n .. j.tk1 * 16 * j.n]);
    }

    fn all(halves: u32, trellis: []const i16, k: usize, n: usize, wq: []f16) void {
        const count = threads.count(threads.maximum, k / 16);
        var jobs: [threads.maximum]Unpack = undefined;
        for (jobs[0..count], 0..) |*j, i| j.* = .{
            .halves = halves,
            .trellis = trellis,
            .n = n,
            .tk0 = i * (k / 16) / count,
            .tk1 = (i + 1) * (k / 16) / count,
            .wq = wq,
        };
        threads.run(Unpack, jobs[0..count], .normal);
    }
};

/// Each output's K sum in float64, in K order, and its terms' magnitude: a range of columns on each thread.
const Sums = struct {
    x: []const u16,
    wq: []const f16,
    rows: usize,
    k: usize,
    n: usize,
    j0: usize,
    j1: usize,
    want: []f64,
    s1: []f64,

    pub fn run(job: *Sums) void {
        for (0..job.rows) |r| {
            const y = job.want[r * job.n ..][job.j0..job.j1];
            const m = job.s1[r * job.n ..][job.j0..job.j1];
            @memset(y, 0);
            @memset(m, 0);
            for (0..job.k) |q| {
                const xv = half(job.x[r * job.k + q]);
                for (y, m, job.wq[q * job.n ..][job.j0..job.j1]) |*e, *s, wv| {
                    const p = xv * @as(f64, wv);
                    e.* += p;
                    s.* += @abs(p);
                }
            }
        }
    }

    fn all(x: []const u16, wq: []const f16, rows: usize, k: usize, n: usize, want: []f64, s1: []f64) void {
        const count = threads.count(threads.maximum, n / 128);
        var jobs: [threads.maximum]Sums = undefined;
        for (jobs[0..count], 0..) |*j, i| j.* = .{
            .x = x,
            .wq = wq,
            .rows = rows,
            .k = k,
            .n = n,
            .j0 = i * (n / 128) / count * 128,
            .j1 = (i + 1) * (n / 128) / count * 128,
            .want = want,
            .s1 = s1,
        };
        threads.run(Sums, jobs[0..count], .normal);
    }
};

/// Gate 4's simulations: each output's K terms summed in an accumulator and order, a range of columns on each thread.
const Sims = struct {
    x: []const u16,
    wq: []const f16,
    rows: usize,
    k: usize,
    n: usize,
    j0: usize,
    j1: usize,
    acc: c.Accum,
    order: g.Order,
    t: []f64,
    out: []f64,

    pub fn run(job: *Sims) void {
        for (0..job.rows) |r| for (job.j0..job.j1) |j| {
            for (job.t, 0..) |*e, q| e.* = half(job.x[r * job.k + q]) * @as(f64, job.wq[q * job.n + j]);
            job.out[r * job.n + j] = contract.rows.sumIn(job.acc, job.order, job.t);
        };
    }

    const Args = struct {
        x: []const u16,
        wq: []const f16,
        rows: usize,
        k: usize,
        n: usize,
        acc: c.Accum,
        order: g.Order,
    };

    fn all(a: Args, t: []f64, out: []f64) void {
        const count = threads.count(threads.maximum, a.n / 128);
        var jobs: [threads.maximum]Sims = undefined;
        for (jobs[0..count], 0..) |*j, i| j.* = .{
            .x = a.x,
            .wq = a.wq,
            .rows = a.rows,
            .k = a.k,
            .n = a.n,
            .j0 = i * (a.n / 128) / count * 128,
            .j1 = (i + 1) * (a.n / 128) / count * 128,
            .acc = a.acc,
            .order = a.order,
            .t = t[i * a.k ..][0..a.k],
            .out = out,
        };
        threads.run(Sims, jobs[0..count], .normal);
    }
};

fn half(b: u16) f64 {
    return @as(f16, @bitCast(b));
}

// every served wave count is registered, and the stress sums (K terms of 4096 x -0.6513671875) are integers under 2^24
comptime {
    var four = false;
    var eight = false;
    var most: usize = 0;
    for (linear_shapes) |s| {
        if (exl3.waves(s.k, s.n) == 8) eight = true else four = true;
        most = @max(most, s.k);
    }
    if (!four or !eight) @compileError("the linear registrations miss a wave count");
    if (most * 2668 >= 1 << 24 or fmt.codebookValue(.mul1, 0xffff) != -0.6513671875)
        @compileError("the linear's stress sums are not exact in fp32");
}
