//! Contract suite for `hy4_hc` and `hy4_ihc_fused`: fp64 accuracy, fp32 sums, row invariance, and 13 controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ihc_mod = @import("hy4_ihc_fused_cuda");
const hc_mod = ihc_mod.hc;
const options = @import("contract_options");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const ihc_case = @import("contract_hy4_ihc_fused.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;
const Hc = hc_mod.Hc;
const Ihc = ihc_mod.Ihc;

const ctl_img = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const mix_served = [_]c.Shape{
    .{ .rows = 8, .n = 1, .k = 4 * 3072 },
    .{ .rows = 1, .n = 1, .k = 4 * 3072 },
    .{ .rows = 3, .n = 1, .k = 4 * 256 },
};
const merge_served = [_]c.Shape{
    .{ .rows = 8, .n = 3072, .k = 1 },
    .{ .rows = 1, .n = 3072, .k = 1 },
    .{ .rows = 3, .n = 257, .k = 1 },
};

fn mixDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

fn twoDepth(_: c.Shape) usize {
    return 2;
}

pub const MixKind = enum { pre_out, pre_post, head_out };

pub const MixCase = struct {
    rig: rig_.Rig,
    fn_handle: cuda.Function,
    kind: MixKind,
    ctl: ?g.Gate,
    name: []const u8,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    fn_w: []f32 = &.{},
    base: [8]f32 = @splat(0),
    scale: [2]f32 = .{ 1, 1 },
    eps: f32 = 1e-6,
    hc_eps: f32 = 1e-4,
    magnitude: f32 = 2.0,
    dev_m: hc_mod.Mix = undefined,
    dev_x: u64 = 0,
    dev_out: u64 = 0,
    dev_post: u64 = 0,
    runs: i32 = 0,

    pub fn kernel(self: *MixCase) g.Kernel {
        const is_post = self.kind == .pre_post;
        return .{
            .spec = .{
                .name = self.name,
                .accum = .f32,
                .depth = mixDepth,
                .ops = 16,
                .out = if (is_post) .f32 else .bf16,
                .tiles = .{ .{}, .{}, .{ .size = 256, .step = 4, .min = 4 } },
                .served = &mix_served,
            },
            .ctx = self,
            .vt = &vt,
            .control = self.ctl,
        };
    }
    const vt = g.hooks(MixCase);

    pub fn deinit(self: *MixCase) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    fn mixesCount(self: *const MixCase) usize {
        return if (self.kind == .head_out) 4 else 8;
    }

    pub fn count(self: *const MixCase) usize {
        return self.shape.rows * (if (self.kind == .pre_post) 4 else self.shape.k / 4);
    }

    pub fn prepare(self: *MixCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k < 4 or shape.k % 4 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.runs = 0;
        const a = self.arena.allocator();
        const width = shape.k;
        const hidden = width / 4;
        const mcount = self.mixesCount();
        self.x = try a.alloc(u16, shape.rows * width);
        self.fn_w = try a.alloc(f32, mcount * width);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            self.scale = .{ 100.0, 100.0 };
            self.base = @splat(0);
            self.eps = 1e-6;
            self.hc_eps = 0;
            self.magnitude = 2.0;
            for (0..shape.rows) |r| for (0..hidden) |h| {
                self.x[r * width + h] = c.toBf16(8.0);
                self.x[r * width + hidden + h] = c.toBf16(-8.0);
                self.x[r * width + 2 * hidden + h] = 0;
                self.x[r * width + 3 * hidden + h] = 0;
            };
            for (0..mcount) |j| {
                const sgn: f32 = if (j % 4 == 0) 1.0 else -1.0;
                for (0..width) |k| self.fn_w[j * width + k] = if (k < 2 * hidden) sgn * 8.0 else 0.0;
                self.fn_w[j * width + hidden] = sgn * 7.875;
            }
        } else {
            self.scale = .{ 0.8, 1.1 };
            for (&self.base, 0..) |*b, i| b.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 3)) * 0.15;
            self.eps = 1e-6;
            self.hc_eps = 1e-4;
            self.magnitude = 2.0;
            for (self.x) |*v| v.* = c.toBf16(rng.float(f32) * 4.0 - 2.0);
            for (self.fn_w) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * 0.25;
            if (fill == .nonfinite) {
                if (shape.rows > 0) self.x[0] = c.toBf16(std.math.nan(f32));
                if (shape.rows > 1) self.x[width] = c.toBf16(std.math.inf(f32));
                if (shape.rows > 2) self.x[2 * width] = c.toBf16(-std.math.inf(f32));
            }
        }
        self.dev_x = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.dev_m = .{
            .fn_ = try self.rig.input(std.mem.sliceAsBytes(self.fn_w)),
            .base = try self.rig.input(std.mem.sliceAsBytes(self.base[0..mcount])),
            .scale = try self.rig.input(std.mem.sliceAsBytes(self.scale[0..if (self.kind == .head_out) 1 else 2])),
            .eps = self.eps,
            .hc_eps = self.hc_eps,
            .magnitude = self.magnitude,
        };
        if (self.kind == .pre_post) {
            self.dev_out = try self.rig.scratch(shape.rows * hidden * 2);
            self.dev_post = try self.rig.output(shape.rows * 4 * 4);
        } else {
            self.dev_out = try self.rig.output(shape.rows * hidden * 2);
            self.dev_post = if (self.kind == .pre_out) try self.rig.scratch(shape.rows * 4 * 4) else 0;
        }
        return true;
    }

    pub fn run(self: *MixCase) !void {
        const hidden = self.shape.k / 4;
        const post = if (self.kind == .head_out) 0 else self.dev_post;
        try hc_mod.launchMix(self.fn_handle, self.rig.s, self.dev_m, self.dev_x, self.dev_out, post, self.shape.rows, hidden, self.runs);
        self.runs += 1;
        try self.rig.s.synchronize();
    }

    fn evalRow(self: *const MixCase, r: usize, sq: f64, msq: f64, dot: []const f64, mdot: []const f64, want: []f64, mag: []f64) void {
        const width = self.shape.k;
        const hidden = width / 4;
        const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(width)) + @as(f64, self.eps));
        const sq_rel = if (sq > 0) 0.5 * msq / sq else 0;
        if (self.kind == .pre_post) {
            for (0..4) |i| {
                const z = (dot[4 + i] * inv) * @as(f64, self.scale[1]) + @as(f64, self.base[4 + i]);
                const e = @exp(-@abs(z));
                const sig = if (z < 0) e / (1.0 + e) else 1.0 / (1.0 + e);
                want[r * 4 + i] = @as(f64, self.magnitude) * sig + @as(f64, self.hc_eps);
                mag[r * 4 + i] = 0.25 * @abs(@as(f64, self.magnitude)) * (mdot[4 + i] * inv + @abs(dot[4 + i] * inv) * sq_rel) * @abs(@as(f64, self.scale[1]));
            }
            return;
        }
        var pre: [4]f64 = undefined;
        var wmag: f64 = 0;
        for (0..4) |i| {
            const z = (dot[i] * inv) * @as(f64, self.scale[0]) + @as(f64, self.base[i]);
            const e = @exp(-@abs(z));
            const sig = if (z < 0) e / (1.0 + e) else 1.0 / (1.0 + e);
            pre[i] = sig + @as(f64, self.hc_eps);
            wmag += 0.25 * (mdot[i] * inv + @abs(dot[i] * inv) * sq_rel) * @abs(@as(f64, self.scale[0]));
        }
        for (0..hidden) |h| {
            var s: f64 = 0;
            var xmax: f64 = 0;
            var smag: f64 = 0;
            for (0..4) |i| {
                const xv: f64 = c.fromBf16(self.x[r * width + i * hidden + h]);
                s += pre[i] * xv;
                xmax = @max(xmax, @abs(xv));
                smag += @abs(pre[i] * xv);
            }
            want[r * hidden + h] = s;
            mag[r * hidden + h] = wmag * xmax + smag;
        }
    }

    pub fn reference(self: *MixCase, want: []f64, mag: []f64) !void {
        const width = self.shape.k;
        const mcount = self.mixesCount();
        for (0..self.shape.rows) |r| {
            var sq: f64 = 0;
            var dot: [8]f64 = @splat(0);
            var mdot: [8]f64 = @splat(0);
            for (0..width) |k| {
                const v: f64 = c.fromBf16(self.x[r * width + k]);
                sq += v * v;
                for (0..mcount) |j| {
                    const p = v * @as(f64, self.fn_w[j * width + k]);
                    dot[j] += p;
                    mdot[j] += @abs(p);
                }
            }
            self.evalRow(r, sq, sq, dot[0..mcount], mdot[0..mcount], want, mag);
        }
    }

    pub fn simulate(self: *MixCase, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const width = self.shape.k;
        const mcount = self.mixesCount();
        const terms = try a.alloc(f64, width);
        defer a.free(terms);
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        for (0..self.shape.rows) |r| {
            for (terms, 0..) |*t, k| {
                const v: f64 = c.fromBf16(self.x[r * width + k]);
                t.* = v * v;
            }
            const sq = rows_.sumIn(acc, order, terms);
            var dot: [8]f64 = @splat(0);
            for (0..mcount) |j| {
                for (terms, 0..) |*t, k| t.* = @as(f64, c.fromBf16(self.x[r * width + k])) * @as(f64, self.fn_w[j * width + k]);
                dot[j] = rows_.sumIn(acc, order, terms);
            }
            self.evalRow(r, sq, sq, dot[0..mcount], dot[0..mcount], out, mag);
        }
    }
};

pub const MergeCase = struct {
    rig: rig_.Rig,
    fn_handle: cuda.Function,
    ctl: ?g.Gate,
    name: []const u8,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    res: []u16 = &.{},
    f: []u16 = &.{},
    post: []f32 = &.{},
    dev_res: u64 = 0,
    dev_f: u64 = 0,
    dev_post: u64 = 0,
    dev_out: u64 = 0,
    runs: i32 = 0,

    pub fn kernel(self: *MergeCase) g.Kernel {
        return .{
            .spec = .{
                .name = self.name,
                .accum = .none,
                .depth = zeroDepth,
                .ops = 2,
                .out = .bf16,
                .tiles = .{ .{}, .{ .size = 256 }, .{} },
                .served = &merge_served,
            },
            .ctx = self,
            .vt = &vt,
            .control = self.ctl,
        };
    }
    const vt = g.hooks(MergeCase);

    pub fn deinit(self: *MergeCase) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn count(self: *const MergeCase) usize {
        return self.shape.rows * 4 * self.shape.n;
    }

    pub fn prepare(self: *MergeCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.runs = 0;
        const a = self.arena.allocator();
        const total = shape.rows * 4 * shape.n;
        self.res = try a.alloc(u16, total);
        self.f = try a.alloc(u16, shape.rows * shape.n);
        self.post = try a.alloc(f32, shape.rows * 4);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.res) |*v| v.* = c.toBf16(rng.float(f32) * 4.0 - 2.0);
        for (self.f) |*v| v.* = c.toBf16(rng.float(f32) * 4.0 - 2.0);
        for (self.post) |*v| v.* = rng.float(f32) * 2.0;
        if (fill == .nonfinite) {
            self.res[0] = c.toBf16(std.math.nan(f32));
            if (self.f.len > 1) self.f[1] = c.toBf16(std.math.inf(f32));
            if (self.post.len > 2) self.post[2] = -std.math.inf(f32);
        }
        self.dev_res = try self.rig.input(std.mem.sliceAsBytes(self.res));
        self.dev_f = try self.rig.input(std.mem.sliceAsBytes(self.f));
        self.dev_post = try self.rig.input(std.mem.sliceAsBytes(self.post));
        self.dev_out = try self.rig.output(total * 2);
        return true;
    }

    pub fn run(self: *MergeCase) !void {
        try hc_mod.launchMerge(self.fn_handle, self.rig.s, self.dev_res, self.dev_f, self.dev_post, self.dev_out, self.shape.rows, self.shape.n, self.runs);
        self.runs += 1;
        try self.rig.s.synchronize();
    }

    pub fn reference(self: *MergeCase, want: []f64, mag: []f64) !void {
        const hidden = self.shape.n;
        for (0..self.count()) |i| {
            const row = i / (4 * hidden);
            const s = (i / hidden) % 4;
            const h = i % hidden;
            const p: f64 = self.post[row * 4 + s];
            const fv: f64 = c.fromBf16(self.f[row * hidden + h]);
            const rv: f64 = c.fromBf16(self.res[i]);
            want[i] = p * fv + rv;
            mag[i] = @abs(p * fv) + @abs(rv);
        }
    }

    pub fn bounds(self: *const MergeCase, want: []const f64, mag: []const f64, allow: []f64) !void {
        const spec: c.Spec = .{ .name = self.name, .accum = .f32, .depth = twoDepth, .ops = 1, .out = .bf16, .served = &merge_served };
        for (allow, want, mag) |*l, w, m| l.* = c.bound(spec, self.shape, w, m);
    }
};

pub const Set = struct {
    gpa: std.mem.Allocator,
    hc: Hc,
    ihc: Ihc,
    norms: cuda.norms.Norms,
    ctl_mod: ?cuda.Module = null,
    mixes: [11]MixCase = undefined,
    mixes_n: usize = 0,
    merges: [4]MergeCase = undefined,
    merges_n: usize = 0,
    ihc_cases: [7]ihc_case.Case = undefined,
    ihc_n: usize = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var hc = try Hc.load(gpu.d);
        errdefer hc.unload();
        var ihc = try Ihc.load(gpu.d);
        errdefer ihc.unload();
        var norms = try cuda.norms.Norms.load(gpu.d);
        errdefer norms.unload();
        const ctl_mod = if (options.controls) try cuda.Module.load(gpu.d, &ctl_img.bytes) else null;
        set.* = .{ .gpa = gpu.gpa, .hc = hc, .ihc = ihc, .norms = norms, .ctl_mod = ctl_mod };
        errdefer if (set.ctl_mod) |*m| m.unload();

        set.mixes[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = set.hc.pre_fn, .kind = .pre_out, .ctl = null, .name = "cuda hy4_hc hy4_hc_pre out", .arena = .init(gpu.gpa) };
        set.mixes[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = set.hc.pre_fn, .kind = .pre_post, .ctl = null, .name = "cuda hy4_hc hy4_hc_pre post", .arena = .init(gpu.gpa) };
        set.mixes[2] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = set.hc.head_fn, .kind = .head_out, .ctl = null, .name = "cuda hy4_hc hy4_hc_head", .arena = .init(gpu.gpa) };
        set.mixes_n = 3;
        set.merges[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = set.hc.merge_fn, .ctl = null, .name = "cuda hy4_hc hy4_hc_merge", .arena = .init(gpu.gpa) };
        set.merges_n = 1;

        const modes = [5]struct { mode: ihc_mod.Mode, name: []const u8 }{
            .{ .mode = .boundary, .name = "cuda hy4_ihc_fused boundary" },
            .{ .mode = .entry, .name = "cuda hy4_ihc_fused entry" },
            .{ .mode = .head, .name = "cuda hy4_ihc_fused epilogue_head" },
            .{ .mode = .post_only, .name = "cuda hy4_ihc_fused post_only" },
            .{ .mode = .head_only, .name = "cuda hy4_ihc_fused head_only" },
        };
        for (modes, 0..) |m, i| {
            set.ihc_cases[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .ihc = &set.ihc, .norms = &set.norms, .mode = m.mode, .name = m.name, .arena = .init(gpu.gpa) };
        }
        set.ihc_n = 5;

        if (set.ctl_mod) |m| {
            const pre_ctls = [_]struct { sym: [:0]const u8, gate: g.Gate, name: []const u8 }{
                .{ .sym = "hy4_hc_pre_bad_nan", .gate = .nonfinite, .name = "control: hy4_hc hy4_hc_pre drops NaN" },
                .{ .sym = "hy4_hc_pre_bad_repeat", .gate = .repeat, .name = "control: hy4_hc hy4_hc_pre drifts on repeat" },
                .{ .sym = "hy4_hc_pre_bad_tail", .gate = .shapes, .name = "control: hy4_hc hy4_hc_pre skips tail" },
                .{ .sym = "hy4_hc_pre_bad_sums", .gate = .sums, .name = "control: hy4_hc hy4_hc_pre sums in bf16" },
            };
            for (pre_ctls) |ctl| {
                set.mixes[set.mixes_n] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = try m.function(ctl.sym), .kind = .pre_out, .ctl = ctl.gate, .name = ctl.name, .arena = .init(gpu.gpa) };
                set.mixes_n += 1;
            }
            const head_ctls = [_]struct { sym: [:0]const u8, gate: g.Gate, name: []const u8 }{
                .{ .sym = "hy4_hc_head_bad_nan", .gate = .nonfinite, .name = "control: hy4_hc hy4_hc_head drops NaN" },
                .{ .sym = "hy4_hc_head_bad_repeat", .gate = .repeat, .name = "control: hy4_hc hy4_hc_head drifts on repeat" },
                .{ .sym = "hy4_hc_head_bad_tail", .gate = .shapes, .name = "control: hy4_hc hy4_hc_head skips tail" },
                .{ .sym = "hy4_hc_head_bad_sums", .gate = .sums, .name = "control: hy4_hc hy4_hc_head sums in bf16" },
            };
            for (head_ctls) |ctl| {
                set.mixes[set.mixes_n] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = try m.function(ctl.sym), .kind = .head_out, .ctl = ctl.gate, .name = ctl.name, .arena = .init(gpu.gpa) };
                set.mixes_n += 1;
            }
            const merge_ctls = [_]struct { sym: [:0]const u8, gate: g.Gate, name: []const u8 }{
                .{ .sym = "hy4_hc_merge_bad_nan", .gate = .nonfinite, .name = "control: hy4_hc hy4_hc_merge drops NaN" },
                .{ .sym = "hy4_hc_merge_bad_repeat", .gate = .repeat, .name = "control: hy4_hc hy4_hc_merge drifts on repeat" },
                .{ .sym = "hy4_hc_merge_bad_tail", .gate = .shapes, .name = "control: hy4_hc hy4_hc_merge skips tail" },
            };
            for (merge_ctls) |ctl| {
                set.merges[set.merges_n] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = try m.function(ctl.sym), .ctl = ctl.gate, .name = ctl.name, .arena = .init(gpu.gpa) };
                set.merges_n += 1;
            }
            const pre_bad_nan = try m.function("hy4_hc_pre_bad_nan");
            const pre_bad_sums = try m.function("hy4_hc_pre_bad_sums");
            set.ihc_cases[5] = .{ .rig = .{ .gpu = gpu, .s = s }, .ihc = &set.ihc, .norms = &set.norms, .ctl_pre_fn = pre_bad_nan, .mode = .entry, .fault = .bad_nan, .name = "control: hy4_ihc_fused entry drops NaN", .arena = .init(gpu.gpa) };
            set.ihc_cases[6] = .{ .rig = .{ .gpu = gpu, .s = s }, .ihc = &set.ihc, .norms = &set.norms, .ctl_pre_fn = pre_bad_sums, .mode = .entry, .fault = .bad_sums, .name = "control: hy4_ihc_fused entry sums in bf16", .arena = .init(gpu.gpa) };
            set.ihc_n = 7;
        }
        for (set.mixes[0..set.mixes_n]) |*x| try list.append(a, x.kernel());
        for (set.merges[0..set.merges_n]) |*x| try list.append(a, x.kernel());
        for (set.ihc_cases[0..set.ihc_n]) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (set.ihc_cases[0..set.ihc_n]) |*x| x.deinit();
        for (set.merges[0..set.merges_n]) |*x| x.deinit();
        for (set.mixes[0..set.mixes_n]) |*x| x.deinit();
        if (set.ctl_mod) |*m| m.unload();
        set.norms.unload();
        set.ihc.unload();
        set.hc.unload();
        set.gpa.destroy(set);
    }
};
