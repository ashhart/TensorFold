//! tf_fn_hc_mix8 under the contract on HIP: Flash Next's HC mix, 8 bf16 outputs a thread, bit-equal to tf_fn_hc_mix.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const mix = @import("contract_hc_mix.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const C = ?*const anyopaque;
const P = ?*anyopaque;
const nan = std.math.nan(f64);
const inf = std.math.inf(f64);

/// The kind of g: the up product's fp32 sums, or its bf16 rows.
pub const Gkind = enum(c_int) { f32 = 0, bf16 = 2 };

/// The streams' terms added in order, then the division by their count.
fn depth(s: c.Shape) usize {
    return s.k + 1;
}

const both = "the per-element kernel's bytes and every float64 output class";

pub fn spec(gk: Gkind) c.Spec {
    return .{
        .name = if (gk == .f32) "hip tf_fn_hc_mix8 (g fp32, bf16 x)" else "hip tf_fn_hc_mix8 (g bf16, bf16 x)",
        .accum = .f32,
        .depth = depth,
        .ops = 1,
        .out = .bf16,
        .replaces = "tf_fn_hc_mix",
        .nan = .{ .stated = both },
        // a block's 2,048 outputs in steps of a thread's 8, over the rows' streams of 2,560
        .tiles = .{ .{}, .{ .size = 2048, .step = 8, .min = 8 }, .{} },
        // a decode row, a verify window's 16 and a prompt's rows; four streams
        .served = &.{
            .{ .rows = 1, .n = 2560, .k = 4 },
            .{ .rows = 16, .n = 2560, .k = 4 },
            .{ .rows = 300, .n = 2560, .k = 4 },
        },
    };
}

/// g and x [rows][streams][dims] (shape rows, dims as n, streams as k): mixed [rows][dims] bf16.
pub const Mix8 = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    gk: Gkind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    gv: []f64 = &.{},
    x: []f64 = &.{},
    ranges: []mix.Range = &.{},
    gs: u64 = 0,
    xs: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, gk: Gkind) Mix8 {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .gk = gk, .arena = .init(a) };
    }

    pub fn deinit(self: *Mix8) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Mix8) g.Kernel {
        return .{ .spec = spec(self.gk), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Mix8);

    fn at(self: *const Mix8, r: usize, s: usize, d: usize) usize {
        return (r * self.shape.k + s) * self.shape.n + d;
    }

    /// g in [-6, 6), x in [-2, 2); gate 4: g 16 (gate 1), x 2^20, v, -2^20 and 0 a column, so only fp32 keeps v.
    pub fn prepare(self: *Mix8, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        // the launcher takes tf_fn_hc_mix8 for bf16 activations at widths a multiple of 8
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0 or shape.k == 0 or shape.k > 8) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * shape.k * shape.n;
        self.gv = try kit.floats(a, rng, n, -6, 6);
        if (self.gk == .bf16) for (self.gv) |*v| {
            v.* = mix.bf(v.*);
        };
        self.x = (try kit.values(a, rng, .bf16, n, -2, 2))[0];
        if (fill == .stress) self.stress(rng);
        if (fill == .nonfinite) self.plant();
        const gb = if (self.gk == .f32) try kit.f32Bytes(a, self.gv) else try mix.bf16Bytes(a, self.gv);
        self.gs = try self.rig.input(gb);
        self.xs = try self.rig.input(try mix.bf16Bytes(a, self.x));
        self.out = try self.rig.output(self.count() * 2);
        self.base = try self.rig.scratch(self.count() * 2);
        return true;
    }

    fn stress(self: *Mix8, rng: std.Random) void {
        @memset(self.gv, 16);
        @memset(self.x, 0);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |d| {
            const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
            const put = [_]f64{ 0x1p20, v, -0x1p20 };
            for (put, 0..) |p, s| if (s < self.shape.k) {
                self.x[self.at(r, s, d)] = p;
            };
        };
    }

    /// g: NaN, +Inf, -Inf (a gate of 0) and -Inf times an infinite x in rows 0-3; x's plants in row 4.
    fn plant(self: *Mix8) void {
        const s = self.shape;
        const k = s.k;
        const gat = [_][3]usize{ .{ 0, 0, 7 }, .{ 1, 2, 3 }, .{ 1, 1, 4 }, .{ 2, 3, 6 }, .{ 3, 0, 8 } };
        const gput = [_]f64{ nan, inf, -inf, -inf, inf };
        for (gat, gput) |p, v| if (p[0] < s.rows) {
            self.gv[self.at(p[0], p[1] % k, p[2] % s.n)] = v;
        };
        if (s.rows > 1) self.x[self.at(1, 1 % k, 4 % s.n)] = inf;
        if (s.rows > 3) self.x[self.at(3, 0, 8 % s.n)] = -inf;
        if (s.rows > 4) mix.plantX(self.x, k, s.n, 4);
    }

    pub fn run(self: *Mix8) !void {
        const s = self.shape;
        const gk = @backingInt(self.gk);
        const st = self.rig.s.handle;
        try self.l.fn_hc_mix(kit.ptr(C, self.gs), gk, kit.ptr(C, self.xs), kit.ptr(P, self.out), 2, @intCast(s.rows), @intCast(s.k), @intCast(s.n), st);
        try self.rig.s.synchronize();
    }

    /// tf_fn_hc_mix, one output a thread, on the same g and x.
    pub fn replaced(self: *Mix8) ![]const u8 {
        const s = self.shape;
        const gk = @backingInt(self.gk);
        const st = self.rig.s.handle;
        try self.l.fn_hc_mix_scalar(kit.ptr(C, self.gs), gk, kit.ptr(C, self.xs), kit.ptr(P, self.base), 2, @intCast(s.rows), @intCast(s.k), @intCast(s.n), st);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    pub fn check(self: *Mix8, at_: c.Gate) !?[]const u8 {
        if (at_ != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        return mix.classes(.bf16, out, try self.replaced(), want);
    }

    pub fn count(self: *Mix8) usize {
        return self.shape.rows * self.shape.n;
    }

    fn column(self: *const Mix8, r: usize, d: usize, gs: []f64, xs: []f64) void {
        for (gs, xs, 0..) |*gv, *x, s| {
            gv.* = self.gv[self.at(r, s, d)];
            x.* = self.x[self.at(r, s, d)];
        }
    }

    /// The mix at its bf16 points in float64; g is an input, so it carries no error.
    pub fn reference(self: *Mix8, want: []f64, mag: []f64) !void {
        const s = self.shape;
        self.ranges = try self.arena.allocator().alloc(mix.Range, self.count());
        const zero: [8]f64 = @splat(0);
        var gs: [8]f64 = undefined;
        var xs: [8]f64 = undefined;
        for (0..s.rows) |r| for (0..s.n) |d| {
            self.column(r, d, gs[0..s.k], xs[0..s.k]);
            const i = r * s.n + d;
            want[i] = mix.mixed(gs[0..s.k], xs[0..s.k], .none, .sequential);
            mag[i] = @abs(want[i]);
            self.ranges[i] = mix.range(gs[0..s.k], zero[0..s.k], xs[0..s.k]);
        };
    }

    pub fn bounds(self: *Mix8, _: []const f64, _: []const f64, allow: []f64) !void {
        mix.rangeBounds(self.ranges, allow);
    }

    pub fn intervals(self: *Mix8, lo: []f64, hi: []f64) !void {
        mix.rangeIntervals(self.ranges, lo, hi);
    }

    pub fn simulate(self: *Mix8, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        var gs: [8]f64 = undefined;
        var xs: [8]f64 = undefined;
        for (0..s.rows) |r| for (0..s.n) |d| {
            self.column(r, d, gs[0..s.k], xs[0..s.k]);
            out[r * s.n + d] = mix.mixed(gs[0..s.k], xs[0..s.k], acc, order);
        };
    }
};
