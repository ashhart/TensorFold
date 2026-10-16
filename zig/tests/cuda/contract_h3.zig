//! MiniMax H3's int8 products (h3.cu) under the contract on CUDA: plain, 256 wide, SwiGLU, wide K; and h3_spans.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

pub const Kind = enum { plain, n256, swiglu, wide };

const wide_group = 1024;

/// A kernel's tile in rows and outputs, its K slice, threads and the band of row tiles its blocks walk.
const Geo = struct { symbol: [:0]const u8, bm: usize, bn: usize, kc: usize, threads: u32, group: c_int };

pub fn geo(kind: Kind) Geo {
    return switch (kind) {
        .plain => .{ .symbol = "h3_gemm", .bm = 128, .bn = 128, .kc = 64, .threads = 128, .group = 12 },
        .n256 => .{ .symbol = "h3_gemm_n256", .bm = 128, .bn = 256, .kc = 128, .threads = 256, .group = 16 },
        .swiglu => .{ .symbol = "h3_gemm_swiglu", .bm = 128, .bn = 128, .kc = 64, .threads = 128, .group = 16 },
        .wide => .{ .symbol = "h3_gemm_wide", .bm = 256, .bn = 128, .kc = 128, .threads = 256, .group = 4 },
    };
}

pub fn shared(kind: Kind) u32 {
    const k = geo(kind);
    return @intCast(2 * (k.bm + k.bn) * k.kc);
}

/// int32 MMA sums are exact; one rounding converts the sum to fp32.
fn exactDepth(_: c.Shape) usize {
    return 1;
}

/// SwiGLU: two converted sums, then silu's exp and divide, whose relative error grows with |gate| (see `mag`).
fn swigluDepth(_: c.Shape) usize {
    return 8;
}

/// Each K group's exact sum converted and added into an fp32 chain, one fma a group.
fn wideDepth(s: c.Shape) usize {
    return 2 * (s.k / wide_group) + 2;
}

const why = "int32 MMA sums are exact, then fp32 scales; fp16 sums overflow and bf16 sums lose the result";

/// Shapes: rows, outputs a row (SwiGLU: N / 2), K; the engine's widths at 16 rows and the row tile's edges.
pub fn spec(kind: Kind) c.Spec {
    const k = geo(kind);
    const tiles = [3]c.Tile{ .{ .size = k.bm }, .{ .size = k.bn, .step = k.bn, .min = k.bn }, .{ .size = k.kc, .step = k.kc, .min = k.kc } };
    const sums: c.Sums = .{ .why = why };
    return switch (kind) {
        .plain => .{ .name = "cuda h3_gemm (int8, attention out)", .accum = .f32, .depth = exactDepth, .ops = 2, .out = .bf16, .tiles = tiles, .served = &.{.{ .rows = 16, .n = 5376, .k = 7168 }}, .sums = sums },
        .n256 => .{ .name = "cuda h3_gemm_n256 (int8, q k v gate)", .accum = .f32, .depth = exactDepth, .ops = 2, .out = .bf16, .tiles = tiles, .served = &.{.{ .rows = 16, .n = 28672, .k = 5376 }}, .sums = sums },
        .swiglu => .{ .name = "cuda h3_gemm_swiglu (int8, MLP in)", .accum = .f32, .depth = swigluDepth, .ops = 8, .out = .bf16, .tiles = .{ tiles[0], .{ .size = 64, .step = 64, .min = 64 }, tiles[2] }, .served = &.{.{ .rows = 16, .n = 14336, .k = 5376 }}, .sums = sums },
        .wide => .{ .name = "cuda h3_gemm_wide (int8, MLP out, a scale a row and 1024 inputs)", .accum = .f32, .depth = wideDepth, .ops = 2, .out = .bf16, .tiles = .{ tiles[0], tiles[1], .{ .size = wide_group, .step = wide_group, .min = wide_group } }, .served = &.{.{ .rows = 16, .n = 5376, .k = 14336 }}, .sums = sums },
    };
}

pub const Gemm = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []i8 = &.{},
    xs: []f32 = &.{},
    w: []i8 = &.{},
    ws: []f32 = &.{},
    dx: u64 = 0,
    dxs: u64 = 0,
    dw: u64 = 0,
    dws: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Gemm) void {
        self.drop();
        self.rig.deinit();
    }

    fn drop(self: *Gemm) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.xs);
        a.free(self.w);
        a.free(self.ws);
        self.x = &.{};
        self.xs = &.{};
        self.w = &.{};
        self.ws = &.{};
    }

    pub fn kernel(self: *Gemm) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Gemm);

    /// Weight rows the kernel reads: the SwiGLU's are (value, gate) pairs, two a output.
    fn wrows(self: *const Gemm) usize {
        return if (self.kind == .swiglu) 2 * self.shape.n else self.shape.n;
    }

    /// Scales a row: one, or one a group of 1024 inputs.
    fn groups(self: *const Gemm) usize {
        return if (self.kind == .wide) self.shape.k / wide_group else 1;
    }

    pub fn prepare(self: *Gemm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const k = geo(self.kind);
        const n_kernel = if (self.kind == .swiglu) 2 * shape.n else shape.n;
        if (shape.rows == 0 or n_kernel % k.bn != 0 or shape.k % k.kc != 0) return false;
        if (self.kind == .wide and shape.k % wide_group != 0) return false;
        self.rig.reset();
        self.drop();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        // the kernel reads X and (with groups) its scales up to the next whole row tile: the engine pads both
        const padded = (shape.rows + k.bm - 1) / k.bm * k.bm;
        const kk = shape.k;
        const gr = self.groups();
        self.x = try a.alloc(i8, padded * kk);
        @memset(self.x, 0);
        self.xs = try a.alloc(f32, padded * gr);
        @memset(self.xs, 0);
        self.w = try a.alloc(i8, self.wrows() * kk);
        self.ws = try a.alloc(f32, self.wrows());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integer terms whose partial sums reach about 2^19 and cancel to a few hundred: exact in fp32 in any order
            for (0..shape.rows) |r| for (0..kk / 2) |i| {
                const v: i8 = @intCast(1 + rng.uintLessThan(u8, 16));
                self.x[r * kk + i] = v;
                self.x[r * kk + kk / 2 + i] = -v + @as(i8, if (i < 8) 1 else 0);
            };
            for (0..self.wrows()) |j| for (0..kk / 2) |i| {
                const u: i8 = @intCast(1 + rng.uintLessThan(u8, 16));
                self.w[j * kk + i] = u;
                self.w[j * kk + kk / 2 + i] = u;
            };
            for (self.xs[0 .. shape.rows * gr]) |*s| s.* = 1;
            @memset(self.ws, 1);
        } else {
            for (self.x[0 .. shape.rows * kk]) |*v| v.* = @intCast(@as(i32, rng.uintLessThan(u8, 255)) - 127);
            for (self.w) |*v| v.* = @intCast(@as(i32, rng.uintLessThan(u8, 255)) - 127);
            // per-row and per-channel scales over a few binades, as an activation's and a weight's absmax / 127
            const unit: f32 = 1.0 / @as(f32, @floatFromInt(kk));
            for (self.xs[0 .. shape.rows * gr]) |*s| s.* = (0.5 + rng.float(f32)) * std.math.ldexp(unit, @as(i32, @intCast(rng.uintLessThan(u32, 4))) - 1);
            for (self.ws) |*s| s.* = (0.5 + rng.float(f32)) * 0.02;
        }
        if (fill == .nonfinite) {
            const nan = std.math.nan(f32);
            const inf = std.math.inf(f32);
            const put = [_][2]usize{ .{ 0, 0 }, .{ 1, gr - 1 }, .{ 2, 0 } };
            const vals = [_]f32{ nan, inf, -inf };
            for (put, vals) |p, v| if (p[0] < shape.rows) {
                self.xs[p[0] * gr + p[1]] = v;
            };
            self.ws[self.wrows() / 2] = inf;
            self.ws[self.wrows() - 1] = nan;
        }
        self.dx = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.dxs = try self.rig.input(std.mem.sliceAsBytes(self.xs));
        self.dw = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.dws = try self.rig.input(std.mem.sliceAsBytes(self.ws));
        self.out = try self.rig.output(self.count() * 2);
        return true;
    }

    pub fn run(self: *Gemm) !void {
        const k = geo(self.kind);
        const n_kernel = self.wrows();
        var args: cuda.Args = .{};
        args.add(self.dx);
        args.add(self.dxs);
        args.add(self.dw);
        args.add(self.dws);
        args.add(self.out);
        args.add(@as(c_int, @intCast(self.shape.rows)));
        args.add(@as(c_int, @intCast(n_kernel)));
        args.add(@as(c_int, @intCast(self.shape.k)));
        args.add(k.group);
        const blocks = (self.shape.rows + k.bm - 1) / k.bm * (n_kernel / k.bn);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = k.threads }, .shared = shared(self.kind) }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Gemm) usize {
        return self.shape.rows * self.shape.n;
    }

    /// The exact integer sum of row r against weight row j over inputs [from, to).
    fn dot(self: *const Gemm, r: usize, j: usize, from: usize, to: usize) i64 {
        const kk = self.shape.k;
        var s: i64 = 0;
        for (self.x[r * kk + from .. r * kk + to], self.w[j * kk + from .. j * kk + to]) |a, b| s += @as(i64, a) * b;
        return s;
    }

    /// One output channel's value and the magnitude its roundings act on, given its sum(s) in `sum_of`.
    fn channel(self: *const Gemm, r: usize, j: usize, comptime sum_of: anytype, ctx: anytype) [2]f64 {
        const gr = self.groups();
        var v: f64 = 0;
        var m: f64 = 0;
        for (0..gr) |q| {
            const s = sum_of(ctx, r, j, q);
            const t = s * @as(f64, self.xs[r * gr + q]);
            v += t;
            m += @abs(t);
        }
        const ws: f64 = self.ws[j];
        return .{ v * ws, m * @abs(ws) };
    }

    fn exactSum(self: *const Gemm, r: usize, j: usize, q: usize) f64 {
        const span = self.shape.k / self.groups();
        return @floatFromInt(self.dot(r, j, q * span, (q + 1) * span));
    }

    fn outputs(self: *const Gemm, want: []f64, mag: []f64, comptime sum_of: anytype, ctx: anytype) void {
        const n = self.shape.n;
        for (0..self.shape.rows) |r| for (0..n) |j| {
            const o = r * n + j;
            if (self.kind != .swiglu) {
                const vm = self.channel(r, j, sum_of, ctx);
                want[o] = vm[0];
                mag[o] = vm[1];
                continue;
            }
            const value = self.channel(r, 2 * j, sum_of, ctx);
            const gate = self.channel(r, 2 * j + 1, sum_of, ctx);
            const silu = gate[0] / (1 + @exp(-gate[0]));
            want[o] = silu * value[0];
            // silu's relative sensitivity to its input is at most 1 + |gate|; its exp's argument error scales likewise
            mag[o] = @abs(silu) * value[1] + @abs(value[0]) * (1.5 + @abs(gate[0])) * gate[1];
        };
    }

    pub fn reference(self: *Gemm, want: []f64, mag: []f64) !void {
        self.outputs(want, mag, exactSum, self);
    }

    const Sim = struct { self: *const Gemm, acc: c.Accum, order: g.Order, terms: []f64 };

    fn simSum(s: *const Sim, r: usize, j: usize, q: usize) f64 {
        const kk = s.self.shape.k;
        const span = kk / s.self.groups();
        for (s.terms[0..span], q * span..) |*t, i| t.* = @as(f64, @floatFromInt(s.self.x[r * kk + i])) * @as(f64, @floatFromInt(s.self.w[j * kk + i]));
        return rows.sumIn(s.acc, s.order, s.terms[0..span]);
    }

    pub fn simulate(self: *Gemm, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const terms = try a.alloc(f64, self.shape.k);
        defer a.free(terms);
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        const sim = Sim{ .self = self, .acc = acc, .order = order, .terms = terms };
        self.outputs(out, mag, simSum, &sim);
    }
};

/// h3_spans moves bits: rows are spans (a head each), n their 4-byte words; the source's spans sit apart.
pub fn spansSpec() c.Spec {
    return .{ .name = "cuda h3_spans (ranks' exchange packing)", .accum = .none, .depth = noDepth, .out = .f32, .tiles = .{ .{}, .{ .size = 16384 }, .{} }, .served = &.{ .{ .rows = 48, .n = 147456, .k = 1 }, .{ .rows = 48, .n = 72, .k = 1 } } };
}

fn noDepth(_: c.Shape) usize {
    return 0;
}

pub const Spans = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    src: []f32 = &.{},
    din: u64 = 0,
    out: u64 = 0,

    /// Words between the source's spans beyond a span's own.
    const gap = 5;

    pub fn deinit(self: *Spans) void {
        self.rig.gpu.gpa.free(self.src);
        self.rig.deinit();
    }

    pub fn kernel(self: *Spans) g.Kernel {
        return .{ .spec = spansSpec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Spans);

    pub fn prepare(self: *Spans, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.src);
        self.src = &.{};
        self.shape = shape;
        const stride = shape.n + gap;
        self.src = try a.alloc(f32, shape.rows * stride);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.src) |*v| v.* = (rng.float(f32) * 2 - 1) * 64;
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, i| if (i < shape.rows) {
                self.src[i * stride + (i * 7) % shape.n] = v;
            };
        }
        self.din = try self.rig.input(std.mem.sliceAsBytes(self.src));
        self.out = try self.rig.output(shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(self: *Spans) !void {
        const n = self.shape.n;
        var args: cuda.Args = .{};
        args.add(self.out);
        args.add(self.din);
        args.add(@as(c_longlong, @intCast(n * 4)));
        args.add(@as(c_longlong, @intCast((n + gap) * 4)));
        args.add(@as(c_int, @intCast(n)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((n + 16383) / 16384), .y = @intCast(self.shape.rows) }, .block = .{ .x = 256 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Spans) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Spans, want: []f64, mag: []f64) !void {
        const n = self.shape.n;
        for (0..self.shape.rows) |h| for (0..n) |i| {
            want[h * n + i] = self.src[h * (n + gap) + i];
            mag[h * n + i] = 0;
        };
    }
};
