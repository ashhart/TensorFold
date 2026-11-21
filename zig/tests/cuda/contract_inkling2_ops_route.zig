//! Inkling 2's router and expert epilogues: tf_ink2_topk, core's tf_exl3_gateup_capped, tf_ink2_prompt_tail.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const ops = ref.ops;
const step = ref.step;

const experts = 384;
const top = 8;
const route_scale: f32 = 8;

const topk_rule = "the 8 largest score + bias, the lower id on ties, a NaN never picked, a pick's value then -inf; " ++
    "weights score / (sum of the picks' scores + 1e-20) x 8 in fp32";

/// Rows x experts x picks: every row of a call, a block each.
pub fn topkSpec(unbiased: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 16, .n = experts, .k = top }, .{ .rows = 1, .n = experts, .k = top }, .{ .rows = 128, .n = experts, .k = top }, .{ .rows = 2048, .n = experts, .k = top }, .{ .rows = 16, .n = 128, .k = top }, .{ .rows = 1, .n = 128, .k = top }, .{ .rows = 128, .n = 128, .k = top }, .{ .rows = 2048, .n = 128, .k = top } };
    const name = if (unbiased) "control: inkling2 ops tf_ink2_topk without the selection bias" else "inkling2 ops tf_ink2_topk (sigmoid scores, bias, route scale 8)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = topk_rule }, .served = if (unbiased) served[0..2] else served };
}

/// Each row's picks and weights against the host's replay of the kernel's rule in fp32 (Zig's f32 is IEEE as nvcc's).
pub const Topk = struct {
    kern: *const ops.Kernels,
    unbiased: bool = false,
    shape: c.Shape = undefined,
    scores: []f32 = &.{},
    bias: []f32 = &.{},
    dev: [3]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub fn spec(self: *const Topk) c.Spec {
        return topkSpec(self.unbiased);
    }

    pub fn control(self: *const Topk) ?g.Gate {
        return if (self.unbiased) .shapes else null;
    }

    pub fn free(self: *Topk, a: std.mem.Allocator) void {
        a.free(self.scores);
        a.free(self.bias);
        self.scores = &.{};
        self.bias = &.{};
    }

    /// Scores on 1/64 steps of (0, 1) and biases on 1/64 steps of [-1/4, 1/4]: many equal sums, so ties decide.
    pub fn prepare(self: *Topk, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > 512 * 8 or shape.k == 0 or shape.k > 32 or shape.k > shape.n) return false;
        self.free(a);
        self.shape = shape;
        const E = shape.n;
        self.scores = try a.alloc(f32, shape.rows * E);
        for (self.scores) |*v| v.* = ref.int(rng, 1, 63) / 64;
        self.bias = try a.alloc(f32, E);
        for (self.bias) |*v| v.* = ref.int(rng, -16, 16) / 64;
        if (fill == .nonfinite and shape.rows >= 4) {
            for ([_]usize{ 0, 5, 9, 100, E - 1 }) |e| self.scores[e] = ref.nan;
            self.scores[E + 7] = ref.inf;
            for (0..E / 2) |e| self.scores[2 * E + 2 * e] = -ref.inf;
            for ([_]usize{ 3, 40 }) |e| self.scores[3 * E + e] = ref.inf;
            self.scores[3 * E + 41] = -ref.inf;
        }
        const zero = try a.alloc(f32, E);
        defer a.free(zero);
        @memset(zero, 0);
        self.dev = .{ try ref.input(rig, self.scores), try ref.input(rig, self.bias), try ref.input(rig, zero) };
        self.out = .{ try rig.output(shape.rows * shape.k * 4), try rig.output(shape.rows * shape.k * 4) };
        return true;
    }

    pub fn run(self: *Topk, rig: *rig_.Rig) !void {
        const bias = if (self.unbiased) self.dev[2] else self.dev[1];
        try ops.topk(self.kern, rig.s, self.dev[0], bias, self.out[0], self.out[1], self.shape.rows, self.shape.n, self.shape.k, route_scale);
    }

    pub fn count(self: *const Topk) usize {
        return self.shape.rows * self.shape.k * 8;
    }

    pub fn check(self: *const Topk, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const E = self.shape.n;
        const K = self.shape.k;
        const rows = self.shape.rows;
        const sel = try a.alloc(f32, E);
        defer a.free(sel);
        for (0..rows) |r| {
            const sc = self.scores[r * E ..][0..E];
            for (sel, sc, self.bias) |*v, x, b| v.* = x + b;
            var ids: [32]usize = undefined;
            var t: f32 = 0;
            for (0..K) |k| {
                var best = -std.math.inf(f32);
                var id: usize = std.math.maxInt(usize);
                for (sel, 0..) |v, e| if (v > best or (v == best and e < id)) {
                    best = v;
                    id = e;
                };
                if (id == std.math.maxInt(usize)) return try std.fmt.bufPrint(msg, "row {d}: the fixture leaves no pick {d}", .{ r, k });
                ids[k] = id;
                sel[id] = -std.math.inf(f32);
                t += sc[id];
            }
            for (0..K) |k| {
                const want_w = sc[ids[k]] / (t + 1e-20) * route_scale;
                const id = ref.readI32(got, r * K + k);
                const w = ref.readF32(got, rows * K + r * K + k);
                if (id != @as(i32, @intCast(ids[k])) or !ref.sameF32(w, want_w)) return try std.fmt.bufPrint(msg, "row {d} pick {d}: ({d}, {e}) where ({d}, {e}) is due", .{ r, k, id, w, ids[k], want_w });
            }
        }
        return null;
    }
};

const had: f64 = @as(f32, 0.08838834764831845); // exl3_decode.cuh's HAD_SCALE, as fp32 holds it
const gate_cap: f32 = 4;
const up_cap: f32 = 25;
const width = 1152; // Small's width at TP=2; Mini runs one rank with the same full width.
const fixture_experts = 8;

/// fwht128 and tf_exl3_fwht alike: butterflies (lo + hi, lo - hi) at strides 1, 2, ..., 64 of the 128 values.
pub fn fwht(s: Sums, v: *[128]Iv) void {
    var h: usize = 1;
    while (h < 128) : (h *= 2) for (0..128) |i| if (i & h == 0) {
        const lo = v[i];
        v[i] = step(s, lo, v[i + h], false);
        v[i + h] = step(s, lo, v[i + h], true);
    };
}

/// The splits in order (sk adds), then 7 butterfly levels; the activations' transform 7 more.
fn gateDepth(s: c.Shape) usize {
    return s.k + 7;
}

/// (row, slot) pairs x a rank's width x the gate/up's K splits: 1 row, 16, a 128-row window, a 2,048-row window.
pub fn gateupSpec(swapped: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 8, .n = width, .k = 4 }, .{ .rows = 128, .n = width, .k = 4 }, .{ .rows = 1024, .n = width, .k = 4 }, .{ .rows = 16384, .n = width, .k = 4 }, .{ .rows = 8, .n = width, .k = 2 }, .{ .rows = 128, .n = width, .k = 2 }, .{ .rows = 1024, .n = width, .k = 2 }, .{ .rows = 16384, .n = width, .k = 2 } };
    const name = if (swapped) "control: inkling2 exl3 tf_exl3_gateup_capped with the caps swapped" else "inkling2 exl3 tf_exl3_gateup_capped (softcapped SwiGLU, gate cap 4, up cap 25)";
    return .{ .name = name, .accum = .f32, .depth = gateDepth, .out = .f16, .tiles = .{ .{}, .{ .size = 128, .step = 128, .min = 128 }, .{} }, .served = if (swapped) served[0..2] else served };
}

/// exl3_experts.cu's CappedArgs, field for field: act_mode 3, no clamp.
pub const CappedArgs = extern struct {
    z: u64,
    pick: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    xd: u64,
    p: i32,
    n: i32,
    sk: i32,
    experts: i32,
    limit: f32 = 0,
    act_mode: i32 = 3,
    gate_cap: f32,
    up_cap: f32,
};

/// xd = H(gcap tanh(g / gcap) sigmoid(g) ucap tanh(u / ucap) suh_d) 2^-3.5, g and u the split sums' H times svh.
pub const Gateup = struct {
    kern: *const ops.Kernels,
    capped: cuda.Function, // exl3_experts.Experts.capped
    swapped: bool = false,
    shape: c.Shape = undefined,
    z: []f32 = &.{}, // [(mat * sk + split) * P + pair] * n + column
    sv: [3][]u16 = .{ &.{}, &.{}, &.{} }, // fp16 [experts][n]: svh_g, svh_u, suh_d
    pick: []i32 = &.{},
    args: CappedArgs = undefined,

    pub const sums = {};

    pub fn spec(self: *const Gateup) c.Spec {
        return gateupSpec(self.swapped);
    }

    pub fn control(self: *const Gateup) ?g.Gate {
        return if (self.swapped) .shapes else null;
    }

    pub fn free(self: *Gateup, a: std.mem.Allocator) void {
        a.free(self.z);
        for (&self.sv) |*v| a.free(v.*);
        a.free(self.pick);
        self.z = &.{};
        self.sv = .{ &.{}, &.{}, &.{} };
        self.pick = &.{};
    }

    /// Splits in [-1, 1), scales near 2 (g past the gate cap's knee); stress: integers in [-2048, 2048], scales 2^-10.
    pub fn prepare(self: *Gateup, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 128 != 0 or shape.k == 0) return false;
        self.free(a);
        self.shape = shape;
        const P = shape.rows;
        const n = shape.n;
        const stress = fill == .stress;
        self.z = try a.alloc(f32, 2 * shape.k * P * n);
        for (self.z) |*v| v.* = if (stress) ref.int(rng, -2048, 2048) else rng.float(f32) * 2 - 1;
        if (fill == .nonfinite and P >= 4) {
            self.z[3] = ref.nan; // pair 0's gate: NaN through its block
            self.z[n + 5] = ref.inf; // pair 1's gate +-Inf: tanh and sigmoid keep it finite
            self.z[(shape.k * P + 2) * n + 7] = -ref.inf; // pair 2's up: tanh keeps it finite
            self.z[3 * n + 1] = ref.inf; // pair 3's gate: Inf - Inf in the transform
            self.z[3 * n + 2] = -ref.inf;
        }
        const level: f32 = if (stress) 0x1p-10 else 2;
        for (&self.sv, 0..) |*sv, mi| {
            sv.* = try a.alloc(u16, fixture_experts * n);
            for (sv.*) |*h| h.* = @bitCast(@as(f16, @floatCast(if (mi == 2) 0.5 + rng.float(f32) else level * (0.5 + rng.float(f32)))));
        }
        self.pick = try a.alloc(i32, P);
        for (self.pick) |*e| e.* = @intCast(rng.uintLessThan(u32, fixture_experts));
        self.args = .{ .z = try ref.input(rig, self.z), .pick = try ref.input(rig, self.pick), .svh_g = try ref.input(rig, self.sv[0]), .svh_u = try ref.input(rig, self.sv[1]), .suh_d = try ref.input(rig, self.sv[2]), .xd = try rig.output(P * n * 2), .p = @intCast(P), .n = @intCast(n), .sk = @intCast(shape.k), .experts = fixture_experts, .gate_cap = gate_cap, .up_cap = up_cap };
        if (self.swapped) self.args.gate_cap = up_cap;
        if (self.swapped) self.args.up_cap = gate_cap;
        return true;
    }

    pub fn run(self: *Gateup, rig: *rig_.Rig) !void {
        var a: cuda.Args = .{};
        a.add(self.args);
        const grid: cuda.Dim3 = .{ .x = @intCast(self.args.p), .y = @intCast(@divExact(self.args.n, 128)) };
        try cuda.launch.launch(self.capped, .{ .grid = grid, .block = .{ .x = 32 } }, rig.s, &a);
    }

    pub fn count(self: *const Gateup) usize {
        return self.shape.rows * self.shape.n;
    }

    fn scale(self: *const Gateup, mat: usize, p: usize, col: usize) Iv {
        const h: f16 = @bitCast(self.sv[mat][@as(usize, @intCast(self.pick[p])) * self.shape.n + col]);
        return Iv.of(h);
    }

    /// A matrix's 128-block of one pair: each column's splits added in order, then the transform.
    fn rotated(self: *const Gateup, s: Sums, mat: usize, p: usize, b: usize, v: *[128]Iv) void {
        const sk = self.shape.k;
        const P = self.shape.rows;
        const n = self.shape.n;
        for (v, 0..) |*x, j| {
            x.* = Iv.of(0);
            for (0..sk) |sp| x.* = step(s, x.*, Iv.of(self.z[((mat * sk + sp) * P + p) * n + b * 128 + j]), false);
        }
        fwht(s, v);
    }

    pub fn solve(self: *const Gateup, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        var gv: [128]Iv = undefined;
        var uv: [128]Iv = undefined;
        for (0..self.shape.rows) |p| for (0..n / 128) |b| {
            self.rotated(s, 0, p, b, &gv);
            self.rotated(s, 1, p, b, &uv);
            for (&gv, uv, 0..) |*x, uw, j| {
                const col = b * 128 + j;
                const gg = x.mul(Iv.of(had)).mul(self.scale(0, p, col));
                const u = uw.mul(Iv.of(had)).mul(self.scale(1, p, col));
                const act = Iv.of(gate_cap).mul(ref.tanh(gg.div(Iv.of(gate_cap)))).mul(gg.sigmoid()).mul(Iv.of(up_cap)).mul(ref.tanh(u.div(Iv.of(up_cap))));
                x.* = act.mul(self.scale(2, p, col));
            }
            fwht(ref.later(s), &gv);
            for (gv, 0..) |x, j| out[p * n + b * 128 + j] = ref.half(x.mul(Iv.of(had)));
        };
    }
};

/// Prompt rows x a projection's width per rank (q and gate 3,072, r 384, k and v 512, o 4,608) x a block's 128.
pub fn tailSpec(f32_out: bool, shifted: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 256, .n = 3072, .k = 128 }, .{ .rows = 2048, .n = 4608, .k = 128 }, .{ .rows = 256, .n = 512, .k = 128 }, .{ .rows = 2048, .n = 384, .k = 128 } };
    const name = if (shifted) "control: inkling2 ops tf_ink2_prompt_tail reading the next column's svh" else if (f32_out) "inkling2 ops tf_ink2_prompt_tail (fp32 out)" else "inkling2 ops tf_ink2_prompt_tail (bf16 out)";
    return .{ .name = name, .accum = .f32, .depth = struct {
        fn d(_: c.Shape) usize {
            return 7;
        }
    }.d, .out = if (f32_out) .f32 else .bf16, .tiles = .{ .{}, .{ .size = 128, .step = 128, .min = 128 }, .{} }, .served = if (shifted) served[2..3] else served };
}

/// y = H(z) x (svh x 2^-3) per 128-column block of a prompt GEMM's fp32 sums.
pub const Tail = struct {
    kern: *const ops.Kernels,
    f32_out: bool,
    shifted: bool = false,
    shape: c.Shape = undefined,
    z: []f32 = &.{},
    svh: []u16 = &.{}, // fp16 [n + 1]: the control reads one on
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Tail) c.Spec {
        return tailSpec(self.f32_out, self.shifted);
    }

    pub fn control(self: *const Tail) ?g.Gate {
        return if (self.shifted) .shapes else null;
    }

    pub fn free(self: *Tail, a: std.mem.Allocator) void {
        a.free(self.z);
        a.free(self.svh);
        self.z = &.{};
        self.svh = &.{};
    }

    /// Sums in [-64, 64); stress: integers in [4096, 8191] (a block's transform to 2^19.6: fp16 overflows).
    pub fn prepare(self: *Tail, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 128 != 0 or shape.k != 128) return false;
        self.free(a);
        self.shape = shape;
        self.z = try a.alloc(f32, shape.rows * shape.n);
        for (self.z) |*v| v.* = if (fill == .stress) ref.int(rng, 4096, 8191) else rng.float(f32) * 128 - 64;
        if (fill == .nonfinite) ref.plant(f32, self.z, shape.rows, shape.n, ref.asF32);
        self.svh = try a.alloc(u16, shape.n + 1);
        for (self.svh) |*h| h.* = @bitCast(@as(f16, @floatCast(0.01 + rng.float(f32) * 0.02)));
        self.dev = .{ try ref.input(rig, self.z), try ref.input(rig, self.svh) };
        self.out = try rig.output(shape.rows * shape.n * @as(usize, if (self.f32_out) 4 else 2));
        return true;
    }

    pub fn run(self: *Tail, rig: *rig_.Rig) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ self.dev[0], self.dev[1] + @as(u64, if (self.shifted) 2 else 0) }) |v| a.add(v);
        a.add(@as(u32, @intCast(self.shape.n)));
        a.add(self.out);
        a.add(@as(u32, @intFromBool(self.f32_out)));
        try cuda.launch.launch(self.kern.prompt_tail, .{ .grid = .{ .x = @intCast(self.shape.n / 128), .y = @intCast(self.shape.rows) }, .block = .{ .x = 32 } }, rig.s, &a);
    }

    pub fn count(self: *const Tail) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Tail, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        var v: [128]Iv = undefined;
        for (0..self.shape.rows) |r| for (0..n / 128) |b| {
            for (&v, self.z[r * n + b * 128 ..][0..128]) |*x, z| x.* = Iv.of(z);
            fwht(s, &v);
            for (v, 0..) |x, j| {
                const sv: f64 = @as(f16, @bitCast(self.svh[b * 128 + j]));
                const y = x.mul(Iv.of(sv * 0.125));
                out[r * n + b * 128 + j] = if (self.f32_out) y else y.bf();
            }
        };
    }
};
