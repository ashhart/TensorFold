//! Inkling 2's DSpark SiLU and the ranks' pick under the contract: tf_ink2_dsp_silu and tf_ink2_dsp_pick.
const std = @import("std");
const core = @import("core");
const dk = @import("inkling2").dspark_ops;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;

const inter = 9216; // a TP=2 rank's FFN columns (18,432 / 2)

/// A block's 8 rows x this rank's FFN columns: no sums.
pub fn siluSpec(swapped: bool) c.Spec {
    const served = &[_]c.Shape{.{ .rows = 8, .n = inter, .k = 1 }};
    const name = if (swapped) "inkling2 dspark control: SiLU reads up as gate and gate as up" else "inkling2 dspark tf_ink2_dsp_silu (silu(gate) * up in fp32, rounded once)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .served = served };
}

/// out = bf16(g / (1 + expf(-g)) * u), gu bf16 [rows][2I] (gates, then ups).
pub const Silu = struct {
    kern: *const dk.Kernels,
    /// control: the device's rows hold ups then gates (the reference keeps gates first)
    swapped: bool = false,
    shape: c.Shape = undefined,
    gu: []u16 = &.{},
    in: u64 = 0,
    out: u64 = 0,

    pub fn spec(self: *const Silu) c.Spec {
        return siluSpec(self.swapped);
    }

    pub fn control(self: *const Silu) ?g.Gate {
        return if (self.swapped) .shapes else null;
    }

    pub fn free(self: *Silu, a: std.mem.Allocator) void {
        a.free(self.gu);
        self.gu = &.{};
    }

    /// Gates in [-8, 8), ups in [-4, 4); non-finite: a NaN gate, -Inf (silu NaN) and +Inf gates, an infinite up.
    pub fn prepare(self: *Silu, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.gu = try a.alloc(u16, shape.rows * 2 * n);
        for (0..shape.rows) |r| for (0..n) |i| {
            self.gu[r * 2 * n + i] = c.toBf16(rng.float(f32) * 16 - 8);
            self.gu[r * 2 * n + n + i] = c.toBf16(rng.float(f32) * 8 - 4);
        };
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, 3 }, .{ 1, n / 2 }, .{ 2, n - 1 }, .{ 3, 2 * n - 1 } };
            const put = [_]f32{ ref.nan, -ref.inf, ref.inf, ref.inf };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.gu[p[0] * 2 * n + p[1]] = c.toBf16(v);
            };
        }
        const dev = try a.dupe(u16, self.gu);
        defer a.free(dev);
        if (self.swapped) for (0..shape.rows) |r| for (0..n) |i| {
            std.mem.swap(u16, &dev[r * 2 * n + i], &dev[r * 2 * n + n + i]);
        };
        self.in = try ref.input(rig, dev);
        self.out = try rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(self: *Silu, rig: *rig_.Rig) !void {
        try dk.silu(self.kern, rig.s, self.in, self.out, self.shape.rows, self.shape.n);
    }

    pub fn count(self: *const Silu) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Silu, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        const n = self.shape.n;
        for (0..self.shape.rows) |r| for (0..n) |i| {
            const x = Iv.of(ref.bf(self.gu[r * 2 * n + i]));
            const u = Iv.of(ref.bf(self.gu[r * 2 * n + n + i]));
            out[r * n + i] = x.div(Iv.of(1).add(x.neg().exp())).mul(u).bf();
        };
    }
};

/// The kernel's rule, as the check below applies it independently of the kernel's loop.
const pick_rule = "walk[j + 1]: the largest value over the ranks, the lowest id among equal values (-Inf included); " ++
    "a NaN never wins; with every value NaN the id is 0x7fffffff";

/// Drafts (a launch each) x ranks (TP=2) x 1: every rank's (value, id) 16 bytes apart.
pub fn pickSpec(alone: bool) c.Spec {
    const name = if (alone) "inkling2 dspark control: the pick reads rank 0's entry alone" else "inkling2 dspark tf_ink2_dsp_pick (the ranks' best into the walk)";
    return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = pick_rule }, .served = &.{ .{ .rows = 7, .n = 2, .k = 1 }, .{ .rows = 1, .n = 2, .k = 1 } } };
}

const stride = 16;
const share = 100_608; // rank 1's ids start here

pub const Pick = struct {
    kern: *const dk.Kernels,
    /// control: launched with one rank
    alone: bool = false,
    shape: c.Shape = undefined,
    vals: []f32 = &.{},
    ids: []i32 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    pub fn spec(self: *const Pick) c.Spec {
        return pickSpec(self.alone);
    }

    pub fn control(self: *const Pick) ?g.Gate {
        return if (self.alone) .shapes else null;
    }

    pub fn free(self: *Pick, a: std.mem.Allocator) void {
        a.free(self.vals);
        a.free(self.ids);
        self.vals = &.{};
        self.ids = &.{};
    }

    /// Values on quarter steps of [-2, 2) (ties between ranks), rank 1 ahead at draft 0; non-finite: each rule's case.
    pub fn prepare(self: *Pick, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != 2) return false;
        self.free(a);
        self.shape = shape;
        const m = shape.rows * 2;
        self.vals = try a.alloc(f32, m);
        self.ids = try a.alloc(i32, m);
        for (self.vals, self.ids, 0..) |*v, *id, i| {
            v.* = ref.int(rng, -8, 7) / 4;
            id.* = @intCast(rng.uintLessThan(u32, 99_450) + (i % 2) * share);
        }
        self.vals[1] = self.vals[0] + 0.25;
        if (fill == .nonfinite) {
            const put = [_][2]f32{ .{ ref.nan, 1 }, .{ ref.inf, ref.inf }, .{ -ref.inf, -ref.inf }, .{ ref.nan, ref.nan }, .{ 0.5, ref.nan } };
            for (put, 0..) |p, r| if (r < shape.rows) {
                self.vals[2 * r] = p[0];
                self.vals[2 * r + 1] = p[1];
            };
            if (shape.rows > 1) self.ids[2] = share + 9; // draft 1: both +Inf, rank 0's id the higher, so the lower is rank 1's
            if (shape.rows > 1) self.ids[3] = 7;
        }
        const bytes = try a.alloc(u8, m * stride);
        defer a.free(bytes);
        @memset(bytes, 0x5a);
        for (self.vals, self.ids, 0..) |v, id, i| {
            std.mem.writeInt(u32, bytes[i * stride ..][0..4], @bitCast(v), .little);
            std.mem.writeInt(i32, bytes[i * stride + 4 ..][0..4], id, .little);
        }
        self.dev = try rig.input(bytes);
        self.out = try rig.output((shape.rows + 1) * 4);
        return true;
    }

    pub fn run(self: *Pick, rig: *rig_.Rig) !void {
        const world: usize = if (self.alone) 1 else 2;
        for (0..self.shape.rows) |j| try dk.pick(self.kern, rig.s, self.dev + j * 2 * stride, world, self.out, j);
    }

    pub fn count(self: *const Pick) usize {
        return (self.shape.rows + 1) * 4;
    }

    pub fn check(self: *const Pick, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        if (ref.readI32(got, 0) != -1) return "walk[0] (the anchor) was written";
        for (0..self.shape.rows) |j| {
            const vs = self.vals[2 * j ..][0..2];
            const is = self.ids[2 * j ..][0..2];
            var best: ?f32 = null;
            for (vs) |v| if (!std.math.isNan(v)) {
                best = if (best) |b| @max(b, v) else v;
            };
            var id: i32 = 0x7fffffff;
            if (best) |b| for (vs, is) |v, k| if (v == b) {
                id = @min(id, k);
            };
            const v = ref.readI32(got, j + 1);
            if (v != id) return try std.fmt.bufPrint(msg, "draft {d}: id {d} where {d} is due", .{ j, v, id });
        }
        return null;
    }
};
