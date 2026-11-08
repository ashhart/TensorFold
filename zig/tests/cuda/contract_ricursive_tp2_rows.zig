//! Ripple's two-rank gate, clamped SwiGLU and post-norm add (ricursive_tp2.cu) against float64's ranges.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");
const glue = @import("contract_ricursive_tp2.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Function = cuda.Function;
const launch = glue.launch;
const grid = glue.grid;
const halves = glue.halves;
const elementwise = glue.elementwise;

const eps: f32 = 1e-6;
const limit: f32 = 20; // the FFN's clamp
const gate_rows = glue.rowsAt(40 * 128, 1); // a rank's query heads x head dim
const ffn_rows = glue.rowsAt(20480, 1); // a rank's FFN width
const norm_rows = glue.rowsAt(5120, 5120);

fn halfOf(v: f64) f64 {
    return @floatCast(@as(f16, @floatCast(v)));
}

/// To fp16, nearest even: monotone, so the range's ends round to the range's ends.
fn hf(a: Iv) Iv {
    return .{ .m = halfOf(a.m), .lo = halfOf(a.lo), .hi = halfOf(a.hi) };
}

/// out = fp16(o * sigmoid(g)) in fp32: o bf16 in [-4, 4), g fp16 in [-12, 12).
pub const Gate = struct {
    f: Function,
    ctl: bool = false, // control: the sigmoid rounded to fp16 first
    shape: c.Shape = undefined,
    o: []u16 = &.{},
    gv: []f16 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(self: *const Gate) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2 tf_rtp2_gate with fp16(sigmoid(g))" else "cuda ricursive_tp2 tf_rtp2_gate (fp32 sigmoid, one fp16 rounding)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .f16, .tiles = .{ .{}, elementwise, .{} }, .served = &gate_rows };
    }

    pub fn control(self: *const Gate) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Gate, a: std.mem.Allocator) void {
        a.free(self.o);
        a.free(self.gv);
        self.o = &.{};
        self.gv = &.{};
    }

    pub fn prepare(self: *Gate, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * shape.n;
        self.o = try ref.bfs(a, rng, n, -4, 4);
        self.gv = try halves(a, rng, n, -12, 12);
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |x, i| if (2 * i + 1 < n) {
            self.gv[2 * i] = @floatCast(x);
            self.o[2 * i + 1] = c.toBf16(x);
        };
        self.dev = .{ try ref.input(rig, self.o), try ref.input(rig, self.gv) };
        self.out = try rig.output(n * 2);
        return true;
    }

    pub fn run(self: *Gate, rig: *rig_.Rig) !void {
        const n = self.shape.rows * self.shape.n;
        try launch(self.f, rig.s, grid(n), .{ self.dev[0], self.dev[1], self.out, @as(i64, @intCast(n)) });
    }

    pub fn count(self: *const Gate) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Gate, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.o, self.gv) |*v, oh, gh| v.* = hf(Iv.of(ref.bf(oh)).mul(Iv.of(gh).sigmoid()));
    }
};

/// out = fp16(silu(min(g, 20)) * clamp(u, -20, 20)) in fp32, g and u fp16 in [-30, 30): past the clamp both ways.
pub const SwiGlu = struct {
    f: Function,
    ctl: bool = false, // control: u clamped above only
    shape: c.Shape = undefined,
    gv: []f16 = &.{},
    u: []f16 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(self: *const SwiGlu) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2 tf_rtp2_swiglu, up clamped above only" else "cuda ricursive_tp2 tf_rtp2_swiglu (clamp 20, fp32, one fp16 rounding)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .f16, .nan = .{ .stated = "fminf and fmaxf: +Inf gate or up reads as the limit, -Inf up as -limit; a -Inf gate gives NaN" }, .tiles = .{ .{}, elementwise, .{} }, .served = &ffn_rows };
    }

    pub fn control(self: *const SwiGlu) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *SwiGlu, a: std.mem.Allocator) void {
        a.free(self.gv);
        a.free(self.u);
        self.gv = &.{};
        self.u = &.{};
    }

    pub fn prepare(self: *SwiGlu, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * shape.n;
        self.gv = try halves(a, rng, n, -30, 30);
        self.u = try halves(a, rng, n, -30, 30);
        if (fill == .nonfinite and n > 4) {
            self.gv[0] = std.math.inf(f16);
            self.u[1] = std.math.inf(f16);
            self.u[2] = -std.math.inf(f16);
            self.gv[3] = -std.math.inf(f16); // -Inf x sigmoid(-Inf) = -Inf x 0: the one NaN the clamps let through
        }
        self.dev = .{ try ref.input(rig, self.gv), try ref.input(rig, self.u) };
        self.out = try rig.output(n * 2);
        return true;
    }

    pub fn run(self: *SwiGlu, rig: *rig_.Rig) !void {
        const n = self.shape.rows * self.shape.n;
        try launch(self.f, rig.s, grid(n), .{ self.dev[0], self.dev[1], self.out, @as(i64, @intCast(n)), limit });
    }

    pub fn count(self: *const SwiGlu) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const SwiGlu, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.gv, self.u) |*v, gh, uh| {
            const gc = Iv.of(gh).atMost(limit);
            const uc = Iv.of(uh).atMost(limit).atLeast(-limit);
            v.* = hf(gc.mul(gc.sigmoid()).mul(uc));
        }
    }
};

/// A thread's fp32 chain over the row (one value per 256), the warp's 5 shuffles and the block's 8 partials.
fn normDepth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 5 + 8;
}

/// x += (a / rms(a)) * w in fp32, in place: a in [-2, 2), one in 64 twenty times larger; w [0.5, 1.5); x [-8, 8).
pub const NormAdd = struct {
    f: Function,
    ctl: bool = false, // control: a / rms rounded to bf16
    shape: c.Shape = undefined,
    av: []f32 = &.{},
    w: []f32 = &.{},
    x: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const NormAdd) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2 tf_rtp2_norm_add with bf16(a / rms)" else "cuda ricursive_tp2 tf_rtp2_norm_add (x += rms(a) w, fp32, in place)";
        return .{ .name = name, .accum = .f32, .depth = normDepth, .ops = 4, .out = .f32, .in_place = true, .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} }, .served = &norm_rows };
    }

    pub fn control(self: *const NormAdd) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *NormAdd, a: std.mem.Allocator) void {
        for ([_]*[]f32{ &self.av, &self.w, &self.x }) |s| {
            a.free(s.*);
            s.* = &.{};
        }
    }

    pub fn prepare(self: *NormAdd, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * shape.n;
        self.av = try ref.f32s(a, rng, n, -2, 2);
        for (self.av, 0..) |*v, i| if (i % 64 == 7) {
            v.* *= 20;
        };
        if (fill == .stress) for (self.av) |*v| {
            v.* = ref.int(rng, -8, 8);
        };
        self.w = try ref.f32s(a, rng, shape.n, 0.5, 1.5);
        self.x = try ref.f32s(a, rng, n, -8, 8);
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |v, r| if (r < shape.rows) {
            self.av[r * shape.n + shape.n / 2] = v;
        };
        self.dev = .{ try ref.input(rig, self.av), try ref.input(rig, self.w) };
        self.out = try rig.output(n * 4); // in place: restored before each launch
        return true;
    }

    pub fn run(self: *NormAdd, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.out, std.mem.sliceAsBytes(self.x));
        try launch(self.f, rig.s, self.shape.rows, .{ self.dev[0], self.dev[1], self.out, @as(i32, @intCast(self.shape.n)), eps });
    }

    pub fn count(self: *const NormAdd) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const NormAdd, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const sq = try a.alloc(Iv, n);
        defer a.free(sq);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        const depth = normDepth(self.shape);
        for (0..self.shape.rows) |row| {
            const o = row * n;
            for (sq, self.av[o..][0..n]) |*q, v| q.* = Iv.of(v).mul(Iv.of(v));
            const total = s.sum(sq, depth, scratch);
            const inv = total.div(Iv.of(@floatFromInt(n))).add(Iv.of(eps)).rsqrt(); // 1 / sqrt, both correctly rounded
            for (out[o..][0..n], self.av[o..][0..n], self.w, self.x[o..][0..n]) |*v, av, w, x| v.* = Iv.of(x).add(Iv.of(av).mul(inv).mul(Iv.of(w)));
        }
    }
};
