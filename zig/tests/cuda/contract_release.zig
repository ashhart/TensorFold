//! DeepSeek V4.1's release projections under the contract on CUDA (sm_121a): FP8 and BF16 products and split sums.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const kr = @import("dsv41_release");
const ref = @import("contract_release_ref.zig");
const Rig = @import("contract_rig.zig").Rig;
const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

pub const Form = enum { fp8, bf16 };

/// Distinct activation rows a call holds; row r repeats row r % uniq_rows, so the float64 work stays bounded.
pub const uniq_rows = 19;

fn log2(s: u32) usize {
    return std.math.log2_int(u32, s);
}

/// FP8: K / 32 mma steps of 32 products each, the slices' tree; an mma rounds by up to an ulp (counted twice).
fn depthFp8(shape: c.Shape) usize {
    return 2 * (shape.k / 32 + 32) + log2(kr.split(shape.k, shape.n));
}

/// BF16: K / 16 mma steps of 16 products each, then the slices' tree, the mma's roundings counted twice as for FP8.
fn depthBf16(shape: c.Shape) usize {
    return 2 * (shape.k / 16 + 16) + log2(kr.split(shape.k, shape.n));
}

/// Every (rows, n, k) of `ps` (k, n pairs) at each row count of `ms`.
pub fn servedOf(comptime ps: []const [2]usize, comptime ms: []const usize) [ps.len * ms.len]c.Shape {
    var out: [ps.len * ms.len]c.Shape = undefined;
    for (ps, 0..) |p, i| for (ms, 0..) |m, j| {
        out[i * ms.len + j] = .{ .rows = m, .n = p[1], .k = p[0] };
    };
    return out;
}

/// The bytes of `buf` laid as [rows][ld] elements of `es` bytes starting at element `col`, compacted to [rows][n].
fn compact(gpa: std.mem.Allocator, out: *std.ArrayList(u8), buf: []const u8, m: usize, n: usize, ld: usize, col: usize, es: usize) ![]const u8 {
    try out.resize(gpa, m * n * es);
    for (0..m) |r| @memcpy(out.items[r * n * es ..][0 .. n * es], buf[(r * ld + col) * es ..][0 .. n * es]);
    return out.items;
}

pub const Proj = struct {
    rig: Rig,
    ops: kr.Ops,
    form: Form,
    name: []const u8,
    f32_out: bool,
    /// Activations read through a wider row with an offset, outputs written at a column offset of a wider row.
    view: bool,
    served: []const c.Shape,
    x: []f64 = &.{},
    codes: []u8 = &.{},
    scales: []u8 = &.{},
    wb: []u16 = &.{},
    p: kr.Proj = undefined,
    shape: c.Shape = undefined,
    packed_out: std.ArrayList(u8) = .empty,

    pub fn kernel(self: *Proj) g.Kernel {
        return .{ .spec = .{
            .name = self.name,
            .accum = .f32,
            .depth = if (self.form == .fp8) depthFp8 else depthBf16,
            .out = if (self.f32_out) .f32 else .bf16,
            .tiles = .{ .{ .size = 16 }, .{ .size = 128, .step = 32, .min = 32 }, .{ .size = 512, .step = 32, .min = 32 } },
            .served = self.served,
        }, .ctx = self, .vt = &vt };
    }

    const vt = blk: {
        var v = g.hooks(Proj);
        v.output = output;
        v.release = release;
        break :blk v;
    };

    fn of(ctx: *anyopaque) *Proj {
        return @ptrCast(@alignCast(ctx));
    }

    fn output(ctx: *anyopaque) anyerror![]const u8 {
        const self = of(ctx);
        const bytes = try self.rig.outputBytes();
        if (!self.view) return bytes;
        const es: usize = if (self.f32_out) 4 else 2;
        return compact(self.rig.gpu.gpa, &self.packed_out, bytes, self.shape.rows, self.shape.n, @intCast(self.p.y_ld), @intCast(self.p.y_col), es);
    }

    fn release(ctx: *anyopaque) void {
        const self = of(ctx);
        self.free();
        self.packed_out.deinit(self.rig.gpu.gpa);
        self.packed_out = .empty;
        self.rig.finishGates();
    }

    fn free(self: *Proj) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.codes);
        a.free(self.scales);
        a.free(self.wb);
        self.x = &.{};
        self.codes = &.{};
        self.scales = &.{};
        self.wb = &.{};
    }

    pub fn deinit(self: *Proj) void {
        release(self);
    }

    pub fn prepare(self: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.free();
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.n % 32 != 0 or shape.k % 32 != 0) return false;
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const m = shape.rows;
        const n = shape.n;
        const k = shape.k;
        const uniq = @min(m, uniq_rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(f64, uniq * k);
        const ld = if (self.view) k + 64 else k;
        const off: usize = if (self.view) 64 else 0;
        self.p = .{ .a = 0, .a_off = off, .a_ld = @intCast(ld), .w = 0, .y = 0, .y_ld = @intCast(if (self.view) n + 64 else n), .y_col = if (self.view) 32 else 0, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .f32 = @intFromBool(self.f32_out) };
        switch (self.form) {
            .fp8 => try self.fp8Inputs(rng, fill, uniq, ld, off),
            .bf16 => try self.bf16Inputs(rng, fill, uniq, ld, off),
        }
        const bytes = kr.partBytes(m, n, k);
        if (bytes > 0) self.p.part = try self.rig.scratch(bytes);
        const es: usize = if (self.f32_out) 4 else 2;
        self.p.y = try self.rig.output(m * @as(usize, @intCast(self.p.y_ld)) * es);
        return true;
    }

    /// Activation codes and scales for `uniq` rows, then W; a planted NaN code or scale makes its row or columns NaN.
    fn fp8Inputs(self: *Proj, rng: std.Random, fill: g.Fill, uniq: usize, ld: usize, off: usize) !void {
        const a = self.rig.gpu.gpa;
        const m = self.shape.rows;
        const n = self.shape.n;
        const k = self.shape.k;
        const gk = k / 32;
        const ac = try a.alloc(u8, uniq * k);
        defer a.free(ac);
        const as = try a.alloc(u8, uniq * gk);
        defer a.free(as);
        self.codes = try a.alloc(u8, n * k);
        self.scales = try a.alloc(u8, (n / 32) * gk);
        switch (fill) {
            .random, .nonfinite => {
                for (ac) |*v| v.* = finite(rng.int(u8));
                for (self.codes) |*v| v.* = finite(rng.int(u8));
                for (as) |*v| v.* = 118 + rng.uintLessThan(u8, 16);
                for (self.scales) |*v| v.* = 116 + rng.uintLessThan(u8, 12);
            },
            .stress => {
                // scales of 2^4 a side keep every partial a multiple of 2^8 below 2^32 (exact) yet past fp16's range
                @memset(as, 131);
                @memset(self.scales, 131);
                const h = k / 2;
                for (0..uniq) |u| for (0..h) |i| {
                    const v: f64 = @floatFromInt(1 + rng.uintLessThan(u8, 8));
                    ac[u * k + i] = ref.nearest(v);
                    ac[u * k + h + i] = ref.nearest(v + if (i < 8) @as(f64, 1) else 0);
                };
                for (0..n) |j| for (0..h) |i| {
                    const w = ref.nearest(@floatFromInt(1 + rng.uintLessThan(u8, 7)));
                    self.codes[j * k + i] = w;
                    self.codes[j * k + h + i] = w | 0x80;
                };
            },
        }
        if (fill == .nonfinite) {
            ac[k / 3] = 0x7f;
            if (uniq > 1) as[gk + gk - 1] = 255;
            if (uniq > 2) ac[2 * k + k - 1] = 0xff;
            self.codes[5 * k + 7] = 0x7f;
            self.scales[(n / 32 - 1) * gk] = 255;
        }
        for (0..uniq) |u| for (0..k) |i| {
            self.x[u * k + i] = ref.e4m3[ac[u * k + i]] * ref.e8m0(as[u * gk + i / 32]);
        };
        const xb = try a.alloc(u8, off + m * ld);
        defer a.free(xb);
        @memset(xb, 0x7f);
        const sld = if (self.view) gk + 4 else gk;
        const soff: usize = if (self.view) 4 else 0;
        const sb = try a.alloc(u8, soff + m * sld);
        defer a.free(sb);
        @memset(sb, 255);
        for (0..m) |r| {
            @memcpy(xb[off + r * ld ..][0..k], ac[(r % uniq) * k ..][0..k]);
            @memcpy(sb[soff + r * sld ..][0..gk], as[(r % uniq) * gk ..][0..gk]);
        }
        self.p.a = try self.rig.input(xb);
        self.p.a_off = off;
        self.p.sa = try self.rig.input(sb);
        self.p.sa_off = soff;
        self.p.sa_ld = @intCast(sld);
        self.p.w = try self.rig.input(self.codes);
        self.p.ws = try self.rig.input(self.scales);
    }

    fn bf16Inputs(self: *Proj, rng: std.Random, fill: g.Fill, uniq: usize, ld: usize, off: usize) !void {
        const a = self.rig.gpu.gpa;
        const m = self.shape.rows;
        const n = self.shape.n;
        const k = self.shape.k;
        const xs = try a.alloc(u16, uniq * k);
        defer a.free(xs);
        self.wb = try a.alloc(u16, n * k);
        switch (fill) {
            .random, .nonfinite => {
                for (xs) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
                for (self.wb) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
            },
            .stress => {
                const h = k / 2;
                for (0..uniq) |u| for (0..h) |i| {
                    const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 32));
                    xs[u * k + i] = c.toBf16(v);
                    xs[u * k + h + i] = c.toBf16(v + if (i < 8) @as(f32, 1) else 0);
                };
                for (0..n) |j| for (0..h) |i| {
                    const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                    self.wb[j * k + i] = c.toBf16(w);
                    self.wb[j * k + h + i] = c.toBf16(-w);
                };
            },
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| if (p[0] < uniq) {
                xs[p[0] * k + p[1]] = c.toBf16(v);
            };
        }
        for (self.x, xs) |*o, v| o.* = c.fromBf16(v);
        const xb = try a.alloc(u16, off / 2 + m * ld);
        defer a.free(xb);
        @memset(xb, 0x7fc0);
        for (0..m) |r| @memcpy(xb[off / 2 + r * ld ..][0..k], xs[(r % uniq) * k ..][0..k]);
        self.p.a = try self.rig.input(std.mem.sliceAsBytes(xb));
        self.p.a_off = off;
        self.p.w = try self.rig.input(std.mem.sliceAsBytes(self.wb));
    }

    /// A random code that is not NaN.
    fn finite(b: u8) u8 {
        return if (b & 0x7f == 0x7f) b ^ 1 else b;
    }

    pub fn run(self: *Proj) !void {
        try if (self.form == .fp8) self.ops.fp8(self.p) else self.ops.bf16(self.p);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Proj) usize {
        return self.shape.rows * self.shape.n;
    }

    fn projection(self: *Proj) ref.Projection {
        const w: ref.Weight = if (self.form == .fp8) .{ .fp8 = .{ .codes = self.codes, .scales = self.scales } } else .{ .bf16 = self.wb };
        return .{ .gpa = self.rig.gpu.gpa, .m = self.shape.rows, .n = self.shape.n, .k = self.shape.k, .x = self.x, .w = w };
    }

    pub fn reference(self: *Proj, want: []f64, mag: []f64) !void {
        const p = self.projection();
        return p.reference(want, mag);
    }

    pub fn simulate(self: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const p = self.projection();
        return p.simulate(acc, order, out);
    }
};

/// tf_dsv41_rel_sum alone: shape (rows, n, k = slices), FP32 partials [k][rows][n] to bf16 [rows][n].
pub const Sum = struct {
    rig: Rig,
    ops: kr.Ops,
    part: []f32 = &.{},
    shape: c.Shape = undefined,
    y: u64 = 0,
    dev: u64 = 0,

    pub const served = [_]c.Shape{
        .{ .rows = 1, .n = 1280, .k = 8 },    .{ .rows = 5, .n = 512, .k = 8 },     .{ .rows = 17, .n = 576, .k = 8 },
        .{ .rows = 64, .n = 1024, .k = 8 },   .{ .rows = 2048, .n = 1280, .k = 8 }, .{ .rows = 16, .n = 4096, .k = 2 },
        .{ .rows = 2048, .n = 5120, .k = 2 }, .{ .rows = 17, .n = 96, .k = 4 },
    };

    fn depth(shape: c.Shape) usize {
        return std.math.log2_int(usize, shape.k);
    }

    pub fn kernel(self: *Sum) g.Kernel {
        return .{ .spec = .{ .name = "dsv41_rel sum of slices, bf16 out", .accum = .f32, .depth = depth, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{ .size = 256, .step = 32, .min = 32 }, .{} }, .served = &served }, .ctx = self, .vt = &vt };
    }

    const vt = blk: {
        var v = g.hooks(Sum);
        v.release = release;
        break :blk v;
    };

    fn release(ctx: *anyopaque) void {
        const self: *Sum = @ptrCast(@alignCast(ctx));
        self.rig.gpu.gpa.free(self.part);
        self.part = &.{};
        self.rig.finishGates();
    }

    pub fn deinit(self: *Sum) void {
        release(self);
    }

    pub fn prepare(self: *Sum, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const a = self.rig.gpu.gpa;
        a.free(self.part);
        self.part = &.{};
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or !(shape.k == 2 or shape.k == 4 or shape.k == 8)) return false;
        self.shape = shape;
        const mn = shape.rows * shape.n;
        self.part = try a.alloc(f32, shape.k * mn);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.part) |*v| v.* = (rng.float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), rng.intRangeAtMost(i32, -12, 12));
        if (fill == .stress) {
            const h = shape.k / 2;
            for (0..h) |s| for (0..mn) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 1 << 20));
                self.part[s * mn + i] = v;
                self.part[(h + s) * mn + i] = -v + if (s == 0 and i % shape.n < 8) @as(f32, 1) else 0;
            };
        }
        if (fill == .nonfinite) {
            const n = shape.n;
            const at = [_][3]usize{ .{ 0, 0, n / 3 }, .{ 1, 1, n / 2 }, .{ shape.k - 1, 2, n - 1 }, .{ 0, 3, 0 }, .{ 1, 3, 0 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| if (p[1] < shape.rows) {
                self.part[p[0] * mn + p[1] * n + p[2]] = v;
            };
        }
        self.dev = try self.rig.input(std.mem.sliceAsBytes(self.part));
        self.y = try self.rig.output(mn * 2);
        return true;
    }

    pub fn run(self: *Sum) !void {
        const s = self.shape;
        try self.ops.sum(.{ .part = self.dev, .y = self.y, .y_ld = @intCast(s.n), .m = @intCast(s.rows), .n = @intCast(s.n), .s = @intCast(s.k) });
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Sum) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Sum, want: []f64, mag: []f64) !void {
        const mn = self.count();
        for (want, mag, 0..) |*w, *m, i| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..self.shape.k) |s| {
                sum += self.part[s * mn + i];
                abs += @abs(self.part[s * mn + i]);
            }
            w.* = sum;
            m.* = abs;
        }
    }

    pub fn simulate(self: *Sum, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mn = self.count();
        var t: [8]f64 = undefined;
        for (out, 0..) |*o, i| {
            for (t[0..self.shape.k], 0..) |*v, s| v.* = self.part[s * mn + i];
            o.* = contract.rows.sumIn(acc, order, t[0..self.shape.k]);
        }
    }
};

const glue = @import("contract_release_glue.zig");
const row_counts = [_]usize{ 1, 5, 16, 17, 64, 2048 };
const fp8_served = servedOf(&.{ .{ 5120, 1280 }, .{ 5120, 512 }, .{ 1280, 8192 }, .{ 1280, 4096 }, .{ 4096, 1024 }, .{ 2048, 5120 }, .{ 5120, 576 }, .{ 576, 5120 }, .{ 6144, 25600 }, .{ 15360, 5120 } }, &row_counts);
const fp8_view_served = servedOf(&.{.{ 4096, 1024 }}, &.{ 1, 5, 17, 64 }) ++ servedOf(&.{.{ 576, 5120 }}, &.{ 1, 17, 2048 }) ++ servedOf(&.{.{ 1280, 8192 }}, &.{ 1, 16 });
const bf16_served = servedOf(&.{ .{ 5120, 512 }, .{ 5120, 32384 }, .{ 5120, 32256 } }, &row_counts);
const bf16_view_served = servedOf(&.{.{ 512, 128 }}, &row_counts) ++ servedOf(&.{.{ 5120, 512 }}, &.{ 1, 17 });
const actq_served = servedOf(&.{ .{ 8192, 1 }, .{ 2048, 1 }, .{ 6144, 1 }, .{ 15360, 1 }, .{ 512, 1 } }, &row_counts);
const actq_view_served = servedOf(&.{ .{ 4096, 1 }, .{ 512, 1 } }, &.{ 1, 17, 2048 });
const rms_served = servedOf(&.{ .{ 5120, 1 }, .{ 1280, 1 } }, &row_counts);

/// The release images from the contract's own embedded set, resolved; null when they were not built.
pub fn load(d: *const cuda.Driver, mods: *[2]cuda.Module) !?kr.Functions {
    const images = @import("contract_deepseek_images.zig");
    const proj = images.image("dsv41_rel").?;
    const glue_image = images.image("dsv41_rel_glue").?;
    if (proj.len == 0 or glue_image.len == 0) return null;
    mods[0] = try cuda.Module.load(d, proj);
    errdefer mods[0].unload();
    mods[1] = try cuda.Module.load(d, glue_image);
    errdefer mods[1].unload();
    return try kr.Functions.resolve(mods[0], mods[1]);
}

/// The release registrations, sm_121 only: four projections, the slice sum, the round trip and the RMSNorm's two parts.
pub const Set = struct {
    gpa: std.mem.Allocator,
    on: bool = false,
    mods: [2]cuda.Module = undefined,
    f: kr.Functions = undefined,
    projs: [4]Proj = undefined,
    sum: Sum = undefined,
    actq: [2]glue.Actq = undefined,
    rms: [2]glue.Rms = undefined,

    pub fn init(gpu: @import("check.zig").Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa };
        if (try gpu.ctx.capability() != 121) {
            std.debug.print("N/A  dsv41_rel: the release images hold sm_121a only\n", .{});
            return set;
        }
        set.f = try load(gpu.d, &set.mods) orelse return error.ReleaseImagesNotBuilt;
        const ops: kr.Ops = .{ .f = &set.f, .s = s };
        const rig: Rig = .{ .gpu = gpu, .s = s };
        set.projs = .{
            .{ .rig = rig, .ops = ops, .form = .fp8, .name = "dsv41_rel fp8 projections, bf16 out", .f32_out = false, .view = false, .served = &fp8_served },
            .{ .rig = rig, .ops = ops, .form = .fp8, .name = "dsv41_rel fp8 views, fp32 out", .f32_out = true, .view = true, .served = &fp8_view_served },
            .{ .rig = rig, .ops = ops, .form = .bf16, .name = "dsv41_rel bf16 projections, fp32 out", .f32_out = true, .view = false, .served = &bf16_served },
            .{ .rig = rig, .ops = ops, .form = .bf16, .name = "dsv41_rel bf16 views, bf16 out", .f32_out = false, .view = true, .served = &bf16_view_served },
        };
        set.sum = .{ .rig = rig, .ops = ops };
        set.actq = .{
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel actq with bf16 values", .view = false, .rt = true, .served = &actq_served },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel actq views, codes only", .view = true, .rt = false, .served = &actq_view_served },
        };
        set.rms = .{
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel rms_actq normed rows", .codes = false, .served = &rms_served },
            .{ .rig = rig, .ops = ops, .name = "dsv41_rel rms_actq round trip", .codes = true, .served = &rms_served },
        };
        set.on = true;
        for (&set.projs) |*x| try list.append(a, x.kernel());
        try list.append(a, set.sum.kernel());
        for (&set.actq) |*x| try list.append(a, x.kernel());
        for (&set.rms) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        if (set.on) {
            for (&set.projs) |*x| x.deinit();
            set.sum.deinit();
            for (&set.actq) |*x| x.deinit();
            for (&set.rms) |*x| x.deinit();
            for (&set.mods) |*m| m.unload();
        }
        set.gpa.destroy(set);
    }
};
