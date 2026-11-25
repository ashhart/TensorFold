//! Paged indexer RoPE table and 4:1 group pooling kernels (pool_1cta, pool_small, pool_persist) under the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_index_paged");
const rig_ = @import("contract_rig.zig");
const set = @import("contract_hy4_index_paged.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sumIn = contract.rows.sumIn;

const rope_served = [_]c.Shape{ .{ .rows = 8, .n = 64, .k = 1 }, .{ .rows = 1, .n = 64, .k = 1 }, .{ .rows = 64, .n = 64, .k = 1 }, .{ .rows = 257, .n = 64, .k = 1 } };

pub const RopeTable = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    fault: set.Fault = .none,
    shape: c.Shape = undefined,
    inv: [hy4.rope_pairs]f32 = undefined,
    in: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *RopeTable) void {
        self.rig.deinit();
    }

    pub fn kernel(self: *RopeTable) g.Kernel {
        var s: c.Spec = .{
            .name = "cuda hy4_index_paged tf_hy4_idx_rope_table (theta = 1e7)",
            .accum = .none,
            .depth = zeroDepth,
            .ops = 4,
            .out = .f32,
            .tiles = .{ .{ .size = 8 }, .{}, .{} },
            .served = &rope_served,
        };
        if (self.fault != .none) s.name = set.controlName("rope_table", self.fault);
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = self.fault.gate() };
    }

    const vt = g.hooks(RopeTable);

    fn zeroDepth(_: c.Shape) usize {
        return 0;
    }

    pub fn count(self: *const RopeTable) usize {
        return self.shape.rows * hy4.rope_pairs * 2;
    }

    pub fn prepare(self: *RopeTable, shape: c.Shape, fill: g.Fill, _: u64) !bool {
        if (shape.rows == 0 or shape.n != hy4.rope_pairs * 2) return false;
        self.rig.reset();
        self.shape = shape;
        self.inv = cuda.mla.ropeFreqs(1e7);
        if (fill == .nonfinite) self.inv[0] = std.math.nan(f32);
        self.in = try self.rig.input(std.mem.sliceAsBytes(&self.inv));
        self.out = try self.rig.output(self.count() * 4);
        return true;
    }

    pub fn run(self: *RopeTable) !void {
        try hy4.ropeTable(self.f, self.rig.s, self.out, 0, self.shape.rows, self.in, 0, 0, 10, 10, false);
        try self.rig.s.synchronize();
    }

    pub fn reference(self: *RopeTable, want: []f64, mag: []f64) !void {
        for (0..self.shape.rows) |r| for (0..hy4.rope_pairs) |j| {
            const p: f32 = @floatFromInt(r);
            const ang: f64 = p * self.inv[j];
            const at = (r * hy4.rope_pairs + j) * 2;
            want[at] = @cos(ang);
            want[at + 1] = @sin(ang);
            mag[at] = 1.0;
            mag[at + 1] = 1.0;
        };
    }

    pub fn bounds(_: *RopeTable, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow) |*l| l.* += 4 * 0x1p-24;
    }
};

pub const Variant = enum { cta1, small, persist };

fn poolDepth(_: c.Shape) usize {
    return hy4.dim + 16;
}

const pool_served = [_]c.Shape{ .{ .rows = 8, .n = hy4.dim, .k = hy4.dim }, .{ .rows = 1, .n = hy4.dim, .k = hy4.dim }, .{ .rows = 17, .n = hy4.dim, .k = hy4.dim }, .{ .rows = 33, .n = hy4.dim, .k = hy4.dim }, .{ .rows = 64, .n = hy4.dim, .k = hy4.dim } };

pub const Pool = struct {
    rig: rig_.Rig,
    variant: Variant,
    f: cuda.Function,
    base_1cta: cuda.Function,
    fault: set.Fault = .none,
    fill: g.Fill = .random,
    persist_clusters: usize = 1,
    shape: c.Shape = undefined,
    tk: []u16 = &.{},
    wv: []f32 = &.{},
    wg: []f32 = &.{},
    ape: []f32 = &.{},
    knw: []f32 = &.{},
    rt: []f32 = &.{},
    extra: []f64 = &.{},
    in: [6]u64 = @splat(0),
    out: u64 = 0,
    base: u64 = 0,

    pub fn deinit(self: *Pool) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Pool) void {
        const a = self.rig.gpu.gpa;
        a.free(self.tk);
        a.free(self.wv);
        a.free(self.wg);
        a.free(self.ape);
        a.free(self.knw);
        a.free(self.rt);
        a.free(self.extra);
        self.tk = &.{};
        self.wv = &.{};
        self.wg = &.{};
        self.ape = &.{};
        self.knw = &.{};
        self.rt = &.{};
        self.extra = &.{};
    }

    pub fn kernel(self: *Pool) g.Kernel {
        var s: c.Spec = .{
            .name = switch (self.variant) {
                .cta1 => "cuda hy4_index_paged tf_hy4_idx_pool_1cta (4:1 gated group pooling + RoPE)",
                .small => "cuda hy4_index_paged tf_hy4_idx_pool_small (4-CTA DSMEM 4:1 pooling + RoPE)",
                .persist => "cuda hy4_index_paged tf_hy4_idx_pool_persist (4-CTA DSMEM persistent pooling)",
            },
            .accum = .f32,
            .depth = poolDepth,
            .ops = 16,
            .out = .bf16,
            .replaces = if (self.variant != .cta1 and self.fault == .none) "tf_hy4_idx_pool_1cta" else null,
            .tiles = .{ .{ .size = 16, .step = 1, .min = 1 }, .{}, .{} },
            .served = &pool_served,
        };
        if (self.fault != .none) s.name = switch (self.variant) {
            .cta1 => set.controlName("pool_1cta", self.fault),
            .small => set.controlName("pool_small", self.fault),
            .persist => set.controlName("pool_persist", self.fault),
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = self.fault.gate() };
    }

    const vt = g.hooks(Pool);

    pub fn count(self: *const Pool) usize {
        return self.shape.rows * hy4.dim;
    }

    pub fn prepare(self: *Pool, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != hy4.dim or shape.k != hy4.dim) return false;
        self.free();
        self.rig.reset();
        self.fill = fill;
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const jobs = shape.rows;
        self.tk = try a.alloc(u16, jobs * hy4.ratio * hy4.dim);
        self.wv = try a.alloc(f32, hy4.dim * hy4.dim);
        self.wg = try a.alloc(f32, hy4.dim * hy4.dim);
        self.ape = try a.alloc(f32, hy4.ratio * hy4.dim);
        self.knw = try a.alloc(f32, hy4.dim);
        self.rt = try a.alloc(f32, jobs * hy4.rope_pairs * 2);
        self.extra = try a.alloc(f64, jobs * hy4.dim);
        @memset(self.extra, 0);
        const freqs = cuda.mla.ropeFreqs(1e7);
        for (0..jobs) |jb| for (0..hy4.rope_pairs) |j| {
            const ang = @as(f64, @floatFromInt(4 * jb)) * @as(f64, freqs[j]);
            self.rt[(jb * hy4.rope_pairs + j) * 2] = @floatCast(@cos(ang));
            self.rt[(jb * hy4.rope_pairs + j) * 2 + 1] = @floatCast(@sin(ang));
        };
        if (fill == .stress) {
            @memset(self.wg, 0);
            @memset(self.ape, 0);
            @memset(self.knw, 1);
            @memset(self.wv, 1);
            for (0..jobs * hy4.ratio) |r| {
                const big: f32 = @floatFromInt(2048 * (2 + rng.uintLessThan(u32, 3)));
                for (0..8) |i| {
                    self.tk[r * hy4.dim + i] = c.toBf16(1.0);
                    self.tk[r * hy4.dim + 64 + i] = 0;
                }
                for (8..64) |i| {
                    self.tk[r * hy4.dim + i] = c.toBf16(big);
                    self.tk[r * hy4.dim + 64 + i] = c.toBf16(-big);
                }
            }
            for (0..jobs) |jb| for (0..hy4.rope_pairs) |j| {
                self.rt[(jb * hy4.rope_pairs + j) * 2] = 1.0;
                self.rt[(jb * hy4.rope_pairs + j) * 2 + 1] = 0.0;
            };
        } else {
            for (self.tk) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
            for (self.wv) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.25;
            for (self.wg) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.25;
            for (self.ape) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.5;
            for (self.knw) |*v| v.* = 0.5 + rng.float(f32);
            if (fill == .nonfinite) {
                self.tk[0] = c.toBf16(std.math.nan(f32));
                if (jobs > 1) self.tk[1 * hy4.ratio * hy4.dim + 5] = c.toBf16(std.math.inf(f32));
                if (jobs > 2) self.tk[2 * hy4.ratio * hy4.dim + 9] = c.toBf16(-std.math.inf(f32));
            }
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.tk));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.wv));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.wg));
        self.in[3] = try self.rig.input(std.mem.sliceAsBytes(self.ape));
        self.in[4] = try self.rig.input(std.mem.sliceAsBytes(self.knw));
        self.in[5] = try self.rig.input(std.mem.sliceAsBytes(self.rt));
        self.out = try self.rig.output(self.count() * 2);
        self.base = try self.rig.scratch(self.count() * 2);
        return true;
    }

    fn launch(self: *const Pool, f: cuda.Function, v: Variant, out: u64) !void {
        const jobs = self.shape.rows;
        const cluster: ?u32 = if (v == .cta1) null else 4;
        const smem: u32 = switch (v) {
            .cta1 => hy4.pool_1cta_smem,
            .small => 0,
            .persist => hy4.pool_persist_smem,
        };
        const gx: u32 = switch (v) {
            .cta1 => @intCast((jobs + 15) / 16),
            .small => @intCast(4 * jobs),
            .persist => @intCast(4 * @min((jobs + 31) / 32, self.persist_clusters)),
        };
        const bx: u32 = if (v == .persist) 256 else 128;
        var a: cuda.Args = .{};
        inline for (.{ @as(u64, 0), @as(u64, 0), @as(c_int, 0), @as(c_int, @intCast(jobs)), self.in[0], @as(c_int, @intCast(jobs * hy4.ratio)) }) |x| a.add(x);
        inline for (.{ self.in[1], self.in[2], self.in[3], self.in[4], self.in[5], @as(c_int, 0), @as(c_int, 1), set.eps, out }) |x| a.add(x);
        try cuda.launch.launch(f, .{
            .grid = .{ .x = gx },
            .block = .{ .x = bx },
            .shared = smem,
            .cluster = if (cluster) |cl| .{ .x = cl, .y = 1, .z = 1 } else null,
        }, self.rig.s, &a);
    }

    pub fn run(self: *Pool) !void {
        try self.launch(self.f, self.variant, self.out);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Pool) ![]const u8 {
        try self.launch(self.base_1cta, .cta1, self.base);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    fn eval(self: *Pool, acc: ?c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
        var terms: [hy4.dim]f64 = undefined;
        var pooled: [hy4.dim]f64 = undefined;
        var p_mag: [hy4.dim]f64 = undefined;
        var p_err: [hy4.dim]f64 = undefined;
        var sq: [hy4.dim]f64 = undefined;
        for (0..self.shape.rows) |jb| {
            for (0..hy4.dim) |ch| {
                var vs: [hy4.ratio]f64 = undefined;
                var gs: [hy4.ratio]f64 = undefined;
                var abs_v: [hy4.ratio]f64 = undefined;
                var abs_g: [hy4.ratio]f64 = undefined;
                var mx = -std.math.inf(f64);
                for (0..hy4.ratio) |s| {
                    const row_base = (jb * hy4.ratio + s) * hy4.dim;
                    var av: f64 = 0;
                    for (0..hy4.dim) |k| {
                        const x = c.fromBf16(self.tk[row_base + k]);
                        terms[k] = x * self.wv[ch * hy4.dim + k];
                        av += @abs(terms[k]);
                    }
                    vs[s] = if (acc) |ka| sumIn(ka, order, &terms) else sumIn(.none, .sequential, &terms);
                    abs_v[s] = av;
                    var ag: f64 = 0;
                    for (0..hy4.dim) |k| {
                        terms[k] = c.fromBf16(self.tk[row_base + k]) * self.wg[ch * hy4.dim + k];
                        ag += @abs(terms[k]);
                    }
                    abs_g[s] = ag;
                    gs[s] = (if (acc) |ka| sumIn(ka, order, &terms) else sumIn(.none, .sequential, &terms)) + self.ape[s * hy4.dim + ch];
                    mx = if (std.math.isNan(mx) or std.math.isNan(gs[s])) std.math.nan(f64) else @max(mx, gs[s]);
                }
                var ex: [hy4.ratio]f64 = undefined;
                var sum_e: f64 = 0;
                for (0..hy4.ratio) |s| {
                    ex[s] = @exp(gs[s] - mx);
                    sum_e += ex[s];
                }
                var p: f64 = 0;
                var pm: f64 = 0;
                var pe: f64 = 0;
                for (0..hy4.ratio) |s| {
                    const w = ex[s] / sum_e;
                    p += vs[s] * w;
                    pm += abs_v[s] * @abs(w);
                    pe += @abs(vs[s]) * (c.gamma(hy4.dim, 0x1p-24) * abs_g[s] + 8 * 0x1p-24);
                }
                pooled[ch] = p;
                p_mag[ch] = pm;
                p_err[ch] = pe;
                sq[ch] = p * p;
            }
            const ss = if (acc) |ka| sumIn(ka, order, &sq) else sumIn(.none, .sequential, &sq);
            const rinv = 1.0 / @sqrt(ss / @as(f64, hy4.dim) + @as(f64, set.eps));
            var y: [hy4.dim]f64 = undefined;
            var ym: [hy4.dim]f64 = undefined;
            var ye: [hy4.dim]f64 = undefined;
            for (0..hy4.dim) |ch| {
                y[ch] = self.knw[ch] * (pooled[ch] * rinv);
                ym[ch] = @abs(self.knw[ch]) * (p_mag[ch] * rinv);
                ye[ch] = @abs(self.knw[ch]) * (p_err[ch] * rinv);
            }
            const dst = out[jb * hy4.dim ..][0..hy4.dim];
            for (0..64) |ch| {
                dst[ch] = y[ch];
                if (mag) |m| m[jb * hy4.dim + ch] = ym[ch];
                if (mag != null) self.extra[jb * hy4.dim + ch] = ye[ch];
            }
            for (0..hy4.rope_pairs) |pr| {
                const cv: f64 = self.rt[(jb * hy4.rope_pairs + pr) * 2];
                const sv: f64 = self.rt[(jb * hy4.rope_pairs + pr) * 2 + 1];
                const x0 = y[64 + 2 * pr];
                const x1 = y[64 + 2 * pr + 1];
                dst[64 + 2 * pr] = x0 * cv - x1 * sv;
                dst[64 + 2 * pr + 1] = x1 * cv + x0 * sv;
                if (mag) |m| {
                    const e0 = ye[64 + 2 * pr] + @abs(x0) * 0x1p-8;
                    const e1 = ye[64 + 2 * pr + 1] + @abs(x1) * 0x1p-8;
                    m[jb * hy4.dim + 64 + 2 * pr] = ym[64 + 2 * pr] * @abs(cv) + ym[64 + 2 * pr + 1] * @abs(sv);
                    m[jb * hy4.dim + 64 + 2 * pr + 1] = ym[64 + 2 * pr + 1] * @abs(cv) + ym[64 + 2 * pr] * @abs(sv);
                    self.extra[jb * hy4.dim + 64 + 2 * pr] = e0 * @abs(cv) + e1 * @abs(sv);
                    self.extra[jb * hy4.dim + 64 + 2 * pr + 1] = e1 * @abs(cv) + e0 * @abs(sv);
                }
            }
            if (acc != null) for (dst) |*v| {
                v.* = c.fromBf16(c.toBf16(@floatCast(v.*)));
            };
        }
    }

    pub fn reference(self: *Pool, want: []f64, mag: []f64) !void {
        self.eval(null, .sequential, want, mag);
    }

    pub fn bounds(self: *Pool, _: []const f64, _: []const f64, allow: []f64) !void {
        if (self.fill == .stress) return;
        for (allow, self.extra) |*l, e| if (std.math.isFinite(e)) {
            l.* += e * (1 + 0x1p-7);
        };
    }

    pub fn simulate(self: *Pool, acc: c.Accum, order: g.Order, out: []f64) !void {
        self.eval(acc, order, out, null);
    }
};
