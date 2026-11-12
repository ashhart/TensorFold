//! dense_wmma12_gemm_silu under the contract on gfx12: the packed gate/up product's bytes, then tf_silu_mul's.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const PairShape = hip.launches.dense.PairShape;

/// k products, each exact in fp32's normal range, summed through at most k - 1 fp32 roundings in any order.
fn depth(s: c.Shape) usize {
    return @FieldType(hip.Launcher, "dense").depthWmma12(s.k);
}

/// Qwen3-8B's packed gate/up product (2 x 12,288 columns) at the rows each block serves.
const pair_served = [_]c.Shape{
    .{ .rows = 1, .n = 24576, .k = 4096 }, .{ .rows = 7, .n = 24576, .k = 4096 }, .{ .rows = 64, .n = 24576, .k = 4096 },
};
const big_served = [_]c.Shape{
    .{ .rows = 65, .n = 24576, .k = 4096 }, .{ .rows = 300, .n = 24576, .k = 4096 }, .{ .rows = 2048, .n = 24576, .k = 4096 },
};

const replaces = "dense_wmma12_gemm (bf16 out) on the packed weight, then tf_silu_mul";

/// The served block, or the control: silu on the fp32 gate sum, which must go red at gate 3 against the plain bytes.
pub fn spec(geo: PairShape, planted: bool) c.Spec {
    const pair = geo == .pair;
    return .{
        .name = if (planted) "control: fused silu product on the unrounded gate sum" else if (pair)
            "hip dense_wmma12_gemm_silu<W12Pair> (bf16 weights, bf16 out)"
        else
            "hip dense_wmma12_gemm_silu<W12Big> (bf16 weights, bf16 out)",
        .accum = .f32,
        .depth = depth,
        // expf and the division
        .ops = 2,
        // the gate, the up and the silu roundings to bf16
        .mids = 3,
        .out = .bf16,
        .replaces = replaces,
        .nan = .{ .stated = "the unfused bytes and every float64 output class" },
        .served = if (pair) &pair_served else &big_served,
        .tiles = .{ .{ .size = if (pair) 16 else 128 }, .{ .size = if (pair) 64 else 128, .step = 32, .min = 32 }, .{ .size = 4096, .step = 128, .min = 128 } },
    };
}

/// silu(g) as tf_silu_mul writes it, g / (1 + e^-g): -inf gives NaN, as torch's -inf x sigmoid(-inf) does.
fn silu(v: f64) f64 {
    return v / (1 + @exp(-v));
}

/// x [rows, k] bf16 times the packed weight [n, k] (16 gate rows, then the same features' 16 up rows): [rows, n / 2].
pub const SiluFuse = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    geo: PairShape,
    arena: std.heap.ArenaAllocator,
    /// The control's kernel, launched on the pair block's geometry, in place of the served one.
    planted: ?hip.Function = null,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, geo: PairShape) SiluFuse {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .geo = geo, .arena = .init(a) };
    }

    pub fn deinit(self: *SiluFuse) void {
        if (self.d) |d| d.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *SiluFuse) g.Kernel {
        const planted = self.planted != null;
        return .{ .spec = spec(self.geo, planted), .ctx = self, .vt = &vt, .control = if (planted) .shapes else null };
    }

    const vt = g.hooks(SiluFuse);

    /// x from the dense family (negated under stress); W as the WMMA dense test fills it, NaN in an up row too.
    pub fn prepare(self: *SiluFuse, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k % 128 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.rig.a;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        // every stress sum is negative, where silu hides the sum's error: x negated makes them all positive
        if (fill == .stress) for (d.x) |*b| {
            b.* ^= 0x8000;
        };
        const bits = try a.alloc(u16, d.w.len);
        defer a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xD3);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        // gate rows 1 and 5 hold NaN and inf; up row 19 (feature 3) only NaN: fp32's silu(g < -88) is 0, and 0 x inf
        if (fill == .nonfinite and shape.n > 19) {
            bits[1 * k + k / 4] = c.toBf16(std.math.nan(f32));
            bits[5 * k] = c.toBf16(std.math.inf(f32));
            bits[19 * k + k / 2] = c.toBf16(std.math.nan(f32));
        }
        for (d.w, bits) |*w, b| w.* = c.fromBf16(b);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(self.count() * 2);
        return true;
    }

    fn dims(self: *SiluFuse) [3]c_int {
        const s = self.d.?.shape;
        return .{ @intCast(s.rows), @intCast(s.n), @intCast(s.k) };
    }

    pub fn run(self: *SiluFuse) !void {
        const m, const n, const k = self.dims();
        const h = self.rig.s.handle;
        const dense = &self.l.dense;
        if (self.planted) |f| {
            try dense.gemmSiluWith(self.l.r, f, .pair, self.x, self.w, self.out, m, n, k, h);
        } else try dense.gemmSiluOn(self.l.r, self.geo, self.x, self.w, self.out, m, n, k, h);
        try self.rig.s.synchronize();
    }

    /// The plain product on the matching block (BF16 out), split into gate and up columns, then tf_silu_mul.
    fn unfused(self: *SiluFuse) ![]const u8 {
        const m, const n, const k = self.dims();
        const h = self.rig.s.handle;
        const half = self.count();
        const plain = try self.rig.scratch(half * 4);
        const gate = try self.rig.scratch(half * 2);
        const up = try self.rig.scratch(half * 2);
        const base = try self.rig.scratch(half * 2);
        const geo: hip.launches.dense.Shape = if (self.geo == .pair) .w12_small else .w12_big;
        try self.l.dense.gemmOn(self.l.r, .wmma12, geo, self.x, self.w, 0, plain, m, n, k, h);
        const groups: c_int = @intCast(half / 16);
        const bf16 = 2; // the elementwise kernels' kind
        try self.l.tf_copy_cols(@ptrFromInt(plain), 32, 0, @ptrFromInt(gate), bf16, groups, 16, h);
        try self.l.tf_copy_cols(@ptrFromInt(plain), 32, 16, @ptrFromInt(up), bf16, groups, 16, h);
        try self.l.tf_silu_mul(@ptrFromInt(gate), @ptrFromInt(up), @ptrFromInt(base), bf16, @intCast(half), h);
        return self.rig.bytes(base, half * 2);
    }

    /// The unfused bytes; the control's only once the served fused kernel shows them on the same inputs.
    pub fn replaced(self: *SiluFuse) ![]const u8 {
        const want = try self.arena.allocator().dupe(u8, try self.unfused());
        if (self.planted == null) return want;
        const m, const n, const k = self.dims();
        const served = try self.rig.scratch(want.len);
        try self.l.dense.gemmSiluOn(self.l.r, .pair, self.x, self.w, served, m, n, k, self.rig.s.handle);
        if (!std.mem.eql(u8, want, try self.rig.bytes(served, want.len))) return error.ServedDiffersFromBaseline;
        return want;
    }

    /// The non-finite plants must match every float64 class and the unfused kernels' bytes.
    pub fn check(self: *SiluFuse, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        if (!std.mem.eql(u8, out, try self.replaced())) return "non-finite output differs from the unfused kernels' bytes";
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        for (want, 0..) |w, i| {
            if (c.classOf(c.read(.bf16, out, i)) != c.classOf(w)) return "non-finite output class differs from float64";
        }
        return null;
    }

    pub fn count(self: *SiluFuse) usize {
        return self.d.?.outputs() / 2;
    }

    /// Feature f's gate and up columns of the packed product: 16 gate columns, then the same features' 16 up columns.
    fn columns(f: usize) [2]usize {
        const gate = f / 16 * 32 + f % 16;
        return .{ gate, gate + 16 };
    }

    /// Each feature's sums from the packed product's, in place: [rows][n] to [rows][n / 2], g and u read before writes.
    fn pairUp(self: *SiluFuse, sums: []const f64, mags: ?[]const f64, want: []f64, mag: []f64) void {
        const n = self.d.?.shape.n;
        for (0..self.d.?.shape.rows) |r| for (0..n / 2) |f| {
            const col = columns(f);
            const gs = sums[r * n + col[0]];
            const us = sums[r * n + col[1]];
            const o = r * (n / 2) + f;
            want[o] = silu(gs) * us;
            const m = mags orelse continue;
            const sig = 1 / (1 + @exp(-gs));
            const slope = sig * (1 + gs * (1 - sig));
            mag[o] = @abs(slope * us) * m[r * n + col[0]] + @abs(silu(gs)) * m[r * n + col[1]];
        };
    }

    /// silu(g) x u of the float64 sums (NaN kept), with the product's slope on each sum's magnitude.
    pub fn reference(self: *SiluFuse, want: []f64, mag: []f64) !void {
        const a = self.arena.allocator();
        const sums = try a.alloc(f64, self.d.?.outputs());
        const mags = try a.alloc(f64, self.d.?.outputs());
        try self.d.?.reference(sums, mags);
        self.pairUp(sums, mags, want, mag);
    }

    pub fn simulate(self: *SiluFuse, acc: c.Accum, order: g.Order, out: []f64) !void {
        const sums = try self.arena.allocator().alloc(f64, self.d.?.outputs());
        try self.d.?.simulate(acc, order, sums);
        self.pairUp(sums, null, out, out);
    }
};
