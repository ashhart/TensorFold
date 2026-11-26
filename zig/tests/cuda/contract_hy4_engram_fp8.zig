//! Hy4-flash FP8 Engram CUDA contract set: embed, bias, gate, and planted controls in fatbin_contract_controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ce = @import("hy4_engram_fp8");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const exact = @import("contract_hy4_engram_fp8_exact.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;
const fp8 = core.quant.fp8;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

const embed_served = [_]c.Shape{ .{ .rows = 8, .n = 16, .k = 96 }, .{ .rows = 1, .n = 4, .k = 96 }, .{ .rows = 5, .n = 8, .k = 96 }, .{ .rows = 3, .n = 4, .k = 37 } };
const bias_served = [_]c.Shape{ .{ .rows = 4, .n = 5 * 3072, .k = 1 }, .{ .rows = 1, .n = 5 * 3072, .k = 1 }, .{ .rows = 3, .n = 259, .k = 1 } };
const gate_served = [_]c.Shape{ .{ .rows = 6, .n = 4, .k = 3072 }, .{ .rows = 1, .n = 4, .k = 3072 }, .{ .rows = 3, .n = 4, .k = 256 }, .{ .rows = 5, .n = 4, .k = 262 }, .{ .rows = 2, .n = 4, .k = 3070 }, .{ .rows = 2, .n = 4, .k = 3074 } };
const embed_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

fn embedDepth(s: c.Shape) usize { return (s.k + 31) / 32 + 5 + 4; }
fn zeroDepth(_: c.Shape) usize { return 0; }
fn gateDepth(s: c.Shape) usize { return (s.k + 255) / 256 + 5 + 8; }
fn sigmoid64(x: f64) f64 {
    const e = @exp(-@abs(x));
    return if (x < 0) e / (1.0 + e) else 1.0 / (1.0 + e);
}

pub const EmbedCase = struct {
    rig: rig_.Rig,
    func: cuda.Function,
    ref_fn: cuda.Function,
    ctl: ?g.Gate,
    name: []const u8,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    raw: []u8 = &.{},
    norms: []u16 = &.{},
    eps: f32 = ce.norm_eps,
    dev_raw: u64 = 0,
    dev_norms: u64 = 0,
    dev_out: u64 = 0,
    runs: i32 = 0,
    msg: [200]u8 = undefined,

    pub fn kernel(self: *EmbedCase) g.Kernel {
        return .{
            .spec = .{
                .name = self.name,
                .accum = .f32,
                .depth = embedDepth,
                .ops = 8,
                .out = .bf16,
                .tiles = .{ .{ .size = 8 }, .{ .size = 8 }, .{ .size = 32, .min = 4 } },
                .served = &embed_served,
                .sums = .{ .f16 = 100, .bf16 = 0, .why = embed_why },
            },
            .ctx = self,
            .vt = &vt,
            .control = self.ctl,
        };
    }
    const vt = g.hooks(EmbedCase);

    pub fn deinit(self: *EmbedCase) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn count(self: *const EmbedCase) usize {
        return self.shape.rows * self.shape.n * self.shape.k;
    }

    pub fn prepare(self: *EmbedCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.runs = 0;
        self.eps = ce.norm_eps;
        const a = self.arena.allocator();
        const pairs = shape.rows * shape.n;
        const rec_bytes = shape.k + 4;
        self.raw = try a.alloc(u8, pairs * rec_bytes);
        self.norms = try a.alloc(u16, shape.n * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            for (self.norms) |*w| w.* = c.toBf16(1.0);
            for (0..pairs) |p| {
                const rec = self.raw[p * rec_bytes ..][0..rec_bytes];
                @memset(rec[0..shape.k], 0);
                rec[0] = 0x38; // e4m3(1.0) * 256.0 -> sq = 65536.0 (exact in fp32, +Inf in fp16)
                std.mem.writeInt(u32, rec[shape.k..][0..4], @bitCast(@as(f32, 256.0)), .little);
            }
        } else {
            for (self.norms) |*w| w.* = c.toBf16(0.5 + rng.float(f32));
            for (0..pairs) |p| {
                const rec = self.raw[p * rec_bytes ..][0..rec_bytes];
                for (0..shape.k) |d| {
                    rec[d] = @intCast(rng.uintLessThan(u32, 0x7e) | (rng.uintLessThan(u32, 2) << 7));
                }
                const sc: f32 = 0.01 + rng.float(f32) * 0.2;
                std.mem.writeInt(u32, rec[shape.k..][0..4], @bitCast(sc), .little);
            }
            if (fill == .nonfinite) {
                self.raw[0] = 0x7f;
                if (pairs > 1) std.mem.writeInt(u32, self.raw[rec_bytes + shape.k ..][0..4], @bitCast(std.math.inf(f32)), .little);
            }
        }
        self.dev_raw = try self.rig.input(self.raw);
        self.dev_norms = try self.rig.input(std.mem.sliceAsBytes(self.norms));
        self.dev_out = try self.rig.output(self.count() * 2);
        return true;
    }

    pub fn run(self: *EmbedCase) !void {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.dev_out, .len = self.count() * 2 }).fill8(0xa5, self.rig.s.handle);
        try ce.launchEmbed(self.func, self.rig.s, self.dev_raw, self.dev_norms, self.eps, self.dev_out, self.shape.rows, self.shape.n, self.shape.k, self.runs);
        self.runs += 1;
        try self.rig.s.synchronize();
    }

    pub fn check(self: *EmbedCase, gate: g.Gate) !?[]const u8 {
        return exact.checkEmbed(self, gate);
    }

    fn evalPair(self: *const EmbedCase, p: usize, sq: f64, want: []f64, mag: []f64) void {
        const dim = self.shape.k;
        const head = p % self.shape.n;
        const rec = self.raw[p * (dim + 4) ..][0 .. dim + 4];
        const scale: f64 = @as(f32, @bitCast(std.mem.readInt(u32, rec[dim..][0..4], .little)));
        const inv = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(dim)) + @as(f64, self.eps));
        for (0..dim) |d| {
            const v = fp8.e4m3(rec[d]) * scale;
            const nw: f64 = c.fromBf16(self.norms[head * dim + d]);
            const y = (v * inv) * nw;
            want[p * dim + d] = y;
            mag[p * dim + d] = @abs(y);
        }
    }

    pub fn reference(self: *EmbedCase, want: []f64, mag: []f64) !void {
        const pairs = self.shape.rows * self.shape.n;
        const dim = self.shape.k;
        for (0..pairs) |p| {
            const rec = self.raw[p * (dim + 4) ..][0 .. dim + 4];
            const scale: f64 = @as(f32, @bitCast(std.mem.readInt(u32, rec[dim..][0..4], .little)));
            var sq: f64 = 0;
            for (0..dim) |d| {
                const v = fp8.e4m3(rec[d]) * scale;
                sq += v * v;
            }
            self.evalPair(p, sq, want, mag);
        }
    }

    pub fn simulate(self: *EmbedCase, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const pairs = self.shape.rows * self.shape.n;
        const dim = self.shape.k;
        const terms = try a.alloc(f64, dim);
        defer a.free(terms);
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        for (0..pairs) |p| {
            const rec = self.raw[p * (dim + 4) ..][0 .. dim + 4];
            const scale: f64 = @as(f32, @bitCast(std.mem.readInt(u32, rec[dim..][0..4], .little)));
            for (0..dim) |d| {
                const v = fp8.e4m3(rec[d]) * scale;
                terms[d] = v * v;
            }
            self.evalPair(p, rows_.sumIn(acc, order, terms), out, mag);
        }
    }
};

pub const BiasCase = struct {
    rig: rig_.Rig,
    func: cuda.Function,
    ref_fn: cuda.Function,
    ctl: ?g.Gate,
    name: []const u8,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    b: []u16 = &.{},
    dev_x: u64 = 0,
    dev_b: u64 = 0,
    dev_out: u64 = 0,
    runs: i32 = 0,
    msg: [200]u8 = undefined,

    pub fn kernel(self: *BiasCase) g.Kernel {
        return .{
            .spec = .{
                .name = self.name,
                .accum = .none,
                .depth = zeroDepth,
                .ops = 1,
                .out = .f32,
                .tiles = .{ .{}, .{ .size = 256 }, .{} },
                .served = &bias_served,
            },
            .ctx = self,
            .vt = &vt,
            .control = self.ctl,
        };
    }
    const vt = g.hooks(BiasCase);

    pub fn deinit(self: *BiasCase) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn count(self: *const BiasCase) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn prepare(self: *BiasCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.runs = 0;
        const a = self.arena.allocator();
        self.x = try a.alloc(f32, shape.rows * shape.n);
        self.b = try a.alloc(u16, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.x) |*v| v.* = rng.float(f32) * 4.0 - 2.0;
        for (self.b) |*v| v.* = c.toBf16(rng.float(f32) * 2.0 - 1.0);
        if (fill == .nonfinite) {
            self.x[0] = std.math.nan(f32);
            if (self.x.len > 1) self.x[1] = std.math.inf(f32);
            if (self.b.len > 2) self.b[2] = c.toBf16(-std.math.inf(f32));
        }
        self.dev_x = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.dev_b = try self.rig.input(std.mem.sliceAsBytes(self.b));
        self.dev_out = try self.rig.output(self.count() * 4);
        return true;
    }

    pub fn run(self: *BiasCase) !void {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.dev_out, .len = self.count() * 4 }).fill8(0xa5, self.rig.s.handle);
        try ce.launchBias(self.func, self.rig.s, self.dev_x, self.dev_b, self.dev_out, self.shape.rows, self.shape.n, self.runs);
        self.runs += 1;
        try self.rig.s.synchronize();
    }

    pub fn check(self: *BiasCase, gate: g.Gate) !?[]const u8 {
        return exact.checkBias(self, gate);
    }

    pub fn reference(self: *BiasCase, want: []f64, mag: []f64) !void {
        const cols = self.shape.n;
        for (0..self.count()) |i| {
            const xv: f64 = self.x[i];
            const bv: f64 = c.fromBf16(self.b[i % cols]);
            want[i] = xv + bv;
            mag[i] = @abs(xv) + @abs(bv);
        }
    }
};

pub const GateKind = enum { engram_out, streams_out };

pub const GateCase = struct {
    rig: rig_.Rig,
    func: cuda.Function,
    ref_fn: cuda.Function,
    old_reduce_fn: cuda.Function,
    kind: GateKind,
    ctl: ?g.Gate,
    check_exact: bool = false,
    name: []const u8,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    streams_in: []u16 = &.{},
    proj: []f32 = &.{},
    norm_keys: []u16 = &.{},
    norm_queries: []u16 = &.{},
    pre_gamma: []u16 = &.{},
    stream_scales: []u16 = &.{},
    mask: []u8 = &.{},
    eo_abs: []f64 = &.{},
    eps: f32 = ce.norm_eps,
    gw: ce.GateWeights = undefined,
    dev_sin: u64 = 0,
    dev_proj: u64 = 0,
    dev_mask: u64 = 0,
    dev_eo: u64 = 0,
    dev_sout: u64 = 0,
    runs: i32 = 0,
    msg: [200]u8 = undefined,

    pub fn kernel(self: *GateCase) g.Kernel {
        return .{
            .spec = .{
                .name = self.name,
                .accum = .f32,
                .depth = gateDepth,
                .ops = 24,
                .mids = if (self.kind == .streams_out) 1 else 0,
                .out = .bf16,
                .tiles = .{ .{}, .{}, .{ .size = 256, .step = 2, .min = 8 } },
                .served = &gate_served,
            },
            .ctx = self,
            .vt = &vt,
            .control = self.ctl,
        };
    }
    const vt = g.hooks(GateCase);

    pub fn deinit(self: *GateCase) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn count(self: *const GateCase) usize {
        return self.shape.rows * 4 * self.shape.k;
    }

    pub fn prepare(self: *GateCase, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 4 or shape.k < 8 or shape.k % 2 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.runs = 0;
        const a = self.arena.allocator();
        const d = shape.k;
        self.streams_in = try a.alloc(u16, shape.rows * 4 * d);
        self.proj = try a.alloc(f32, shape.rows * 5 * d);
        self.norm_keys = try a.alloc(u16, 4 * d);
        self.norm_queries = try a.alloc(u16, 4 * d);
        self.pre_gamma = try a.alloc(u16, d);
        self.stream_scales = try a.alloc(u16, 4 * d);
        self.mask = try a.alloc(u8, shape.rows);
        self.eo_abs = try a.alloc(f64, self.count());
        @memset(self.mask, 1);

        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            self.eps = 0.0;
            const half = d / 2;
            for (self.pre_gamma) |*v| v.* = c.toBf16(1024.0);
            for (self.stream_scales) |*v| v.* = c.toBf16(0.0);
            for (0..shape.rows) |r| {
                const pv = self.proj[r * 5 * d ..][0 .. 5 * d];
                @memset(pv, 0.0);
                for (0..@min(@as(usize, 192), d)) |h| pv[h] = 4.0;
                for (0..4) |s| for (0..d) |h| {
                    const pos = h < 32 or (h >= 256 and h < 288) or (h >= 512 and h < 544);
                    const neg = h >= half and ((h - half) < 32 or ((h - half) >= 256 and (h - half) < 288) or ((h - half) >= 512 and (h - half) < 544));
                    pv[(1 + s) * d + h] = if (pos or neg) 4.0 else 0.0;
                    self.streams_in[(r * 4 + s) * d + h] = c.toBf16(if (pos) 4.0 else if (neg) -4.0 else 0.0);
                };
            }
            for (0..4) |s| for (0..d) |h| {
                const big = h < 32 or (h >= 256 and h < 288) or (h >= half + 256 and h < half + 288) or (h >= half + 512 and h < half + 544);
                const small_neg = h >= half and h < half + 32;
                self.norm_keys[s * d + h] = c.toBf16(if (big) 64.0 else if (small_neg) 4.0 else 0.0);
                self.norm_queries[s * d + h] = c.toBf16(if (big) 128.0 else if (small_neg) 4.0 else 0.0);
            };
        } else {
            self.eps = ce.norm_eps;
            for (self.norm_keys) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
            for (self.norm_queries) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
            for (self.pre_gamma) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
            for (self.stream_scales) |*v| v.* = c.toBf16((rng.float(f32) * 2.0 - 1.0) * 0.2);
            for (0..shape.rows) |r| {
                const pv = self.proj[r * 5 * d ..][0 .. 5 * d];
                for (0..d) |h| pv[h] = rng.float(f32) * 2.0 - 1.0;
                for (0..4) |s| {
                    const bias_s: f32 = if (s % 2 == 0) 0.4 else -0.4;
                    for (0..d) |h| {
                        pv[(1 + s) * d + h] = 0.4 + rng.float(f32) * 0.8;
                        self.streams_in[(r * 4 + s) * d + h] = c.toBf16(bias_s + (rng.float(f32) - 0.5) * 0.6);
                    }
                }
            }
            if (fill == .random and shape.rows >= 4) self.mask[shape.rows - 2] = 0;
            if (fill == .nonfinite) {
                self.streams_in[0] = c.toBf16(std.math.nan(f32));
                if (shape.rows > 1) self.proj[5 * d] = std.math.inf(f32);
                if (shape.rows > 2) self.proj[2 * 5 * d + d] = -std.math.inf(f32);
            }
        }
        self.dev_sin = try self.rig.input(std.mem.sliceAsBytes(self.streams_in));
        self.dev_proj = try self.rig.input(std.mem.sliceAsBytes(self.proj));
        self.dev_mask = try self.rig.input(self.mask);
        self.gw = .{
            .norm_keys = try self.rig.input(std.mem.sliceAsBytes(self.norm_keys)),
            .norm_queries = try self.rig.input(std.mem.sliceAsBytes(self.norm_queries)),
            .pre_gamma = try self.rig.input(std.mem.sliceAsBytes(self.pre_gamma)),
            .stream_scales = try self.rig.input(std.mem.sliceAsBytes(self.stream_scales)),
            .eps = self.eps,
        };
        if (self.kind == .engram_out) {
            self.dev_eo = try self.rig.output(self.count() * 2);
            self.dev_sout = 0;
        } else {
            self.dev_eo = 0;
            self.dev_sout = try self.rig.output(self.count() * 2);
        }
        return true;
    }

    pub fn run(self: *GateCase) !void {
        const dst = if (self.kind == .engram_out) self.dev_eo else self.dev_sout;
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = dst, .len = self.count() * 2 }).fill8(0xa5, self.rig.s.handle);
        try ce.launchGate(self.func, self.rig.s, self.gw, self.dev_sin, self.dev_proj, self.dev_mask, self.dev_eo, self.dev_sout, self.shape.rows, self.shape.k, self.runs);
        self.runs += 1;
        try self.rig.s.synchronize();
    }

    pub fn check(self: *GateCase, gate: g.Gate) !?[]const u8 {
        return exact.checkGate(self, gate);
    }

    fn evalRow(self: *GateCase, r: usize, sq_v: f64, dot: [4]f64, mdot: [4]f64, want: []f64, mag: []f64) void {
        const d = self.shape.k;
        const df: f64 = @floatFromInt(d);
        const inv_v = 1.0 / @sqrt(sq_v / df + @as(f64, self.eps));
        const inv_sqrt_d = 1.0 / @sqrt(df);
        var gate_v: [4]f64 = undefined;
        var gate_mag: [4]f64 = undefined;
        for (0..4) |s| {
            const gv = dot[s] * inv_sqrt_d;
            const m = @sqrt(@max(@abs(gv), 1e-6));
            const ssqrt = if (std.math.isNan(gv)) gv else if (gv > 0) m else if (gv < 0) -m else 0.0;
            gate_v[s] = sigmoid64(ssqrt);
            gate_mag[s] = (0.25 * 0.5 / @max(m, 1e-3)) * mdot[s] * inv_sqrt_d;
        }
        const active = self.mask[r] != 0;
        const pv = self.proj[r * 5 * d ..][0..d];
        for (0..d) |h| {
            const main_h = @as(f64, c.fromBf16(self.pre_gamma[h])) * (pv[h] * inv_v);
            for (0..4) |s| {
                const idx = (r * 4 + s) * d + h;
                const sin_v: f64 = c.fromBf16(self.streams_in[idx]);
                const sc_v: f64 = c.fromBf16(self.stream_scales[s * d + h]);
                const eo = if (active) gate_v[s] * main_h + sc_v * main_h else 0.0;
                self.eo_abs[idx] = @abs(eo);
                const m_eo = if (active) (gate_mag[s] + @abs(gate_v[s]) + @abs(sc_v)) * @abs(main_h) else 0.0;
                want[idx] = if (self.kind == .engram_out) eo else sin_v + eo;
                mag[idx] = if (self.kind == .engram_out) m_eo else m_eo + @abs(sin_v);
            }
        }
    }

    pub fn bounds(self: *GateCase, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = want;
        _ = mag;
        if (self.kind != .streams_out) return;
        for (allow, self.eo_abs) |*l, ea| {
            if (std.math.isFinite(ea)) l.* += 0x1p-8 * ea;
        }
    }

    pub fn reference(self: *GateCase, want: []f64, mag: []f64) !void {
        const d = self.shape.k;
        const df: f64 = @floatFromInt(d);
        for (0..self.shape.rows) |r| {
            const pv = self.proj[r * 5 * d ..][0 .. 5 * d];
            var sq_v: f64 = 0;
            for (0..d) |h| sq_v += @as(f64, pv[h]) * @as(f64, pv[h]);
            var dot: [4]f64 = @splat(0);
            var mdot: [4]f64 = @splat(0);
            for (0..4) |s| {
                var sq_k: f64 = 0;
                var sq_q: f64 = 0;
                for (0..d) |h| {
                    const kv: f64 = pv[(1 + s) * d + h];
                    const qv: f64 = c.fromBf16(self.streams_in[(r * 4 + s) * d + h]);
                    sq_k += kv * kv;
                    sq_q += qv * qv;
                }
                const ik = 1.0 / @sqrt(sq_k / df + @as(f64, self.eps));
                const iq = 1.0 / @sqrt(sq_q / df + @as(f64, self.eps));
                for (0..d) |h| {
                    const kh = (@as(f64, pv[(1 + s) * d + h]) * ik) * @as(f64, c.fromBf16(self.norm_keys[s * d + h]));
                    const qh = (@as(f64, c.fromBf16(self.streams_in[(r * 4 + s) * d + h])) * iq) * @as(f64, c.fromBf16(self.norm_queries[s * d + h]));
                    const p = kh * qh;
                    dot[s] += p;
                    mdot[s] += @abs(p);
                }
            }
            self.evalRow(r, sq_v, dot, mdot, want, mag);
        }
    }

    pub fn simulate(self: *GateCase, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const d = self.shape.k;
        const df: f64 = @floatFromInt(d);
        const terms = try a.alloc(f64, d);
        defer a.free(terms);
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        for (0..self.shape.rows) |r| {
            const pv = self.proj[r * 5 * d ..][0 .. 5 * d];
            for (0..d) |h| terms[h] = @as(f64, pv[h]) * @as(f64, pv[h]);
            const sq_v = rows_.sumIn(acc, order, terms);
            var dot: [4]f64 = @splat(0);
            for (0..4) |s| {
                for (0..d) |h| terms[h] = @as(f64, pv[(1 + s) * d + h]) * @as(f64, pv[(1 + s) * d + h]);
                const ik = 1.0 / @sqrt(rows_.sumIn(acc, order, terms) / df + @as(f64, self.eps));
                for (0..d) |h| {
                    const qv: f64 = c.fromBf16(self.streams_in[(r * 4 + s) * d + h]);
                    terms[h] = qv * qv;
                }
                const iq = 1.0 / @sqrt(rows_.sumIn(acc, order, terms) / df + @as(f64, self.eps));
                for (0..d) |h| {
                    const kh = (@as(f64, pv[(1 + s) * d + h]) * ik) * @as(f64, c.fromBf16(self.norm_keys[s * d + h]));
                    const qh = (@as(f64, c.fromBf16(self.streams_in[(r * 4 + s) * d + h])) * iq) * @as(f64, c.fromBf16(self.norm_queries[s * d + h]));
                    terms[h] = kh * qh;
                }
                dot[s] = rows_.sumIn(acc, order, terms);
            }
            self.evalRow(r, sq_v, dot, dot, out, mag);
        }
    }
};

pub const Set = struct {
    gpa: std.mem.Allocator,
    ops: ce.Ops,
    ctl_mod: cuda.Module,
    embeds: [5]EmbedCase = undefined,
    biases: [4]BiasCase = undefined,
    gates: [7]GateCase = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var ops = try ce.Ops.load(gpu.d);
        errdefer ops.unload();
        var ctl_mod = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl_mod.unload();
        set.* = .{ .gpa = gpu.gpa, .ops = ops, .ctl_mod = ctl_mod };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        const ref_embed = try ctl_mod.function("tf_ref_hy4_engram_fp8_embed");
        const ref_bias = try ctl_mod.function("tf_ref_hy4_engram_fp8_bias");
        const ref_gate = try ctl_mod.function("tf_ref_hy4_engram_fp8_gate");
        const old_reduce = try ctl_mod.function("tf_ref_hy4_engram_fp8_gate_old_reduce");

        const embed_defs = [5]struct { func: cuda.Function, ctl: ?g.Gate, name: []const u8 }{
            .{ .func = ops.embed_fn, .ctl = null, .name = "cuda hy4_engram_fp8 hy4_engram_fp8_embed" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_embed_bad_nan"), .ctl = .nonfinite, .name = "control: hy4_engram_fp8 embed drops NaN" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_embed_bad_repeat"), .ctl = .repeat, .name = "control: hy4_engram_fp8 embed drifts on repeat" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_embed_bad_tail"), .ctl = .shapes, .name = "control: hy4_engram_fp8 embed skips tail" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_embed_bad_sums"), .ctl = .sums, .name = "control: hy4_engram_fp8 embed sums in fp16" },
        };
        for (embed_defs, 0..) |d, i| set.embeds[i] = .{ .rig = rig, .func = d.func, .ref_fn = ref_embed, .ctl = d.ctl, .name = d.name, .arena = .init(gpu.gpa) };

        const bias_defs = [4]struct { func: cuda.Function, ctl: ?g.Gate, name: []const u8 }{
            .{ .func = ops.bias_fn, .ctl = null, .name = "cuda hy4_engram_fp8 hy4_engram_fp8_bias" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_bias_bad_nan"), .ctl = .nonfinite, .name = "control: hy4_engram_fp8 bias drops NaN" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_bias_bad_repeat"), .ctl = .repeat, .name = "control: hy4_engram_fp8 bias drifts on repeat" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_bias_bad_tail"), .ctl = .shapes, .name = "control: hy4_engram_fp8 bias skips tail" },
        };
        for (bias_defs, 0..) |d, i| set.biases[i] = .{ .rig = rig, .func = d.func, .ref_fn = ref_bias, .ctl = d.ctl, .name = d.name, .arena = .init(gpu.gpa) };

        const gate_defs = [6]struct { func: cuda.Function, ctl: ?g.Gate, exact: bool, name: []const u8 }{
            .{ .func = ops.gate_fn, .ctl = null, .exact = true, .name = "cuda hy4_engram_fp8 hy4_engram_fp8_gate engram_out" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_gate_bad_nan"), .ctl = .nonfinite, .exact = false, .name = "control: hy4_engram_fp8 gate drops NaN" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_gate_bad_repeat"), .ctl = .repeat, .exact = false, .name = "control: hy4_engram_fp8 gate drifts on repeat" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_gate_bad_tail"), .ctl = .shapes, .exact = false, .name = "control: hy4_engram_fp8 gate skips tail" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_gate_bad_sums"), .ctl = .sums, .exact = false, .name = "control: hy4_engram_fp8 gate sums in bf16" },
            .{ .func = try ctl_mod.function("tf_ctl_hy4_engram_fp8_gate_bad_order"), .ctl = .shapes, .exact = true, .name = "control: hy4_engram_fp8 gate wrong reduction order" },
        };
        for (gate_defs, 0..) |d, i| set.gates[i] = .{ .rig = rig, .func = d.func, .ref_fn = ref_gate, .old_reduce_fn = old_reduce, .kind = .engram_out, .ctl = d.ctl, .check_exact = d.exact, .name = d.name, .arena = .init(gpu.gpa) };
        set.gates[6] = .{ .rig = rig, .func = ops.gate_fn, .ref_fn = ref_gate, .old_reduce_fn = old_reduce, .kind = .streams_out, .ctl = null, .check_exact = true, .name = "cuda hy4_engram_fp8 hy4_engram_fp8_gate streams_out", .arena = .init(gpu.gpa) };

        for (&set.embeds) |*x| try list.append(a, x.kernel());
        for (&set.biases) |*x| try list.append(a, x.kernel());
        for (&set.gates) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.embeds) |*x| x.deinit();
        for (&set.biases) |*x| x.deinit();
        for (&set.gates) |*x| x.deinit();
        set.ctl_mod.unload();
        set.ops.unload();
        set.gpa.destroy(set);
    }
};
