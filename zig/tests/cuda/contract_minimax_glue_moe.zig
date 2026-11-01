//! MiniMax-M3.1-Flash's CUDA MLP glue under the contract: SwiGLU-OAI, the router's logits, its top-k and the partial.
const std = @import("std");
const core = @import("core");
const mm3 = @import("minimax_m3_cuda");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const kern = mm3.kernels;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Kernels = kern.Kernels;

const hidden = 6144;
const experts = 128;
const topk = 4;
const alpha: f32 = 1.702;
const limit: f32 = 7.0;

const swiglu_nan = "torch.clamp's bounds: a NaN gate or up gives NaN, +Inf reads as the limit, a -Inf up as -limit; a -Inf gate gives NaN (-Inf x 0)";

/// Rows x width x stride: the dense MLP's 6,144 and the shared expert's 1,536 a rank, DSpark's stacked gate | up.
pub fn swigluSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 16, .n = 6144, .k = 6144 }, .{ .rows = 64, .n = 1536, .k = 1536 }, .{ .rows = 7, .n = 6144, .k = 12288 }, .{ .rows = 1, .n = 1536, .k = 1536 }, .{ .rows = 2048, .n = 1536, .k = 1536 } };
    return .{ .name = "cuda minimax tf_mm3_swiglu_oai", .accum = .none, .depth = ref.noSums, .out = .bf16, .nan = .{ .stated = swiglu_nan }, .tiles = .{ .{}, .{ .size = 256, .step = 128, .min = 128 }, .{} }, .served = served };
}

/// (clamp(up) + 1) g sigmoid(g alpha), g = min(gate, limit), in the kernel's bf16 steps.
pub const Swiglu = struct {
    kern: *const Kernels,
    /// control: the device reads up - 1, so its output lacks the +1
    minus: bool = false,
    shape: c.Shape = undefined,
    gate: []u16 = &.{},
    up: []u16 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(self: *const Swiglu) c.Spec {
        var s = swigluSpec();
        if (self.minus) s.name = "control: MiniMax SwiGLU-OAI without the up's +1";
        return s;
    }

    pub fn control(self: *const Swiglu) ?g.Gate {
        return if (self.minus) .shapes else null;
    }

    pub fn free(self: *Swiglu, a: std.mem.Allocator) void {
        a.free(self.gate);
        a.free(self.up);
        self.gate = &.{};
        self.up = &.{};
    }

    fn stacked(self: *const Swiglu) bool {
        return self.shape.k == 2 * self.shape.n;
    }

    /// Gate and up in [-9, 9) (past the limit both ways); non-finite: NaN, +Inf and -Inf in each of gate and up.
    pub fn prepare(self: *Swiglu, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k < shape.n) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * shape.k;
        self.gate = try ref.bfs(a, rng, n, -9, 9);
        self.up = try ref.bfs(a, rng, n, -9, 9);
        const k = shape.k;
        if (fill == .nonfinite) {
            const put = [_]struct { usize, bool, f32 }{ .{ 0, true, ref.nan }, .{ 1, true, ref.inf }, .{ 2, true, -ref.inf }, .{ 3, false, ref.nan }, .{ 4, false, ref.inf }, .{ 5, false, -ref.inf } };
            for (put) |p| if (p[0] < shape.rows) {
                (if (p[1]) self.gate else self.up)[p[0] * k + p[0] + 1] = c.toBf16(p[2]);
            };
        }
        const dev_up = try a.dupe(u16, self.up);
        defer a.free(dev_up);
        if (self.minus) for (dev_up) |*v| {
            v.* = c.toBf16(c.fromBf16(v.*) - 1);
        };
        if (self.stacked()) { // one row of gate | up, as DSpark's stacked projection writes it
            const both = try a.alloc(u16, n);
            defer a.free(both);
            for (0..shape.rows) |r| {
                @memcpy(both[r * k ..][0..shape.n], self.gate[r * k ..][0..shape.n]);
                @memcpy(both[r * k + shape.n ..][0..shape.n], dev_up[r * k ..][0..shape.n]);
            }
            self.dev[0] = try ref.input(rig, both);
            self.dev[1] = self.dev[0] + shape.n * 2;
        } else self.dev = .{ try ref.input(rig, self.gate), try ref.input(rig, dev_up) };
        self.out = try rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(self: *Swiglu, rig: *rig_.Rig) !void {
        try kern.swigluOai(self.kern, rig.s, self.dev[0], self.dev[1], self.out, self.shape.rows, self.shape.n, self.shape.k, alpha, limit);
    }

    pub fn count(self: *const Swiglu) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Swiglu, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        const n = self.shape.n;
        for (0..self.shape.rows) |r| for (0..n) |i| {
            const at = r * self.shape.k + i;
            const gv = Iv.of(ref.bf(self.gate[at])).atMost(limit);
            const u = Iv.of(ref.bf(self.up[at])).atMost(limit).atLeast(-limit);
            const s = gv.mul(Iv.of(alpha)).bf().sigmoid().bf();
            const glu = gv.mul(s).bf();
            out[r * n + i] = u.add(Iv.of(1)).bf().mul(glu).bf();
        };
    }
};

/// A lane's fma chain (width / 32 terms), then the warp's 5 shuffles.
fn routerDepth(s: c.Shape) usize {
    return s.k / 32 + 5;
}

/// Rows x experts x hidden: a warp an (expert, row), the fp32 router as stored.
pub fn routerSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 16, .n = experts, .k = hidden }, .{ .rows = 1, .n = experts, .k = hidden }, .{ .rows = 32, .n = experts, .k = hidden }, .{ .rows = 2048, .n = experts, .k = hidden } };
    return .{ .name = "cuda minimax tf_mm3_router (bf16 rows, fp32 router)", .accum = .f32, .depth = routerDepth, .out = .f32, .served = served };
}

pub const Router = struct {
    kern: *const Kernels,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(_: *const Router) c.Spec {
        return routerSpec();
    }

    pub fn control(_: *const Router) ?g.Gate {
        return null;
    }

    pub fn free(self: *Router, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    /// Rows in [-4, 4), the router in [-0.05, 0.05); stress: integers, the second half cancelling the first.
    pub fn prepare(self: *Router, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        const k = shape.k;
        if (shape.rows == 0 or shape.n == 0 or k % 128 != 0) return false;
        self.free(a);
        self.shape = shape;
        self.x = try ref.bfs(a, rng, shape.rows * k, -4, 4);
        self.w = try ref.f32s(a, rng, shape.n * k, -0.05, 0.05);
        if (fill == .stress) {
            const h = k / 2;
            for (0..shape.rows) |r| for (0..h) |i| {
                const v = ref.int(rng, 1, 64);
                self.x[r * k + i] = c.toBf16(v);
                self.x[r * k + h + i] = c.toBf16(v + @as(f32, if (i < 8) 1 else 0));
            };
            for (0..shape.n) |e| for (0..h) |i| {
                const v = ref.int(rng, 1, 7);
                self.w[e * k + i] = v;
                self.w[e * k + h + i] = -v;
            };
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
            const put = [_]f32{ ref.nan, ref.inf, -ref.inf, ref.inf, -ref.inf };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.x[p[0] * k + p[1]] = c.toBf16(v);
            };
        }
        self.dev = .{ try ref.input(rig, self.x), try ref.input(rig, self.w) };
        self.out = try rig.output(shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(self: *Router, rig: *rig_.Rig) !void {
        try kern.router(self.kern, rig.s, self.dev[0], self.dev[1], self.out, self.shape.rows, self.shape.k, self.shape.n);
    }

    pub fn count(self: *const Router) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Router, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const k = self.shape.k;
        const terms = try a.alloc(Iv, k);
        defer a.free(terms);
        const scratch = try a.alloc(f64, k);
        defer a.free(scratch);
        for (0..self.shape.rows) |r| for (0..self.shape.n) |e| {
            for (terms, self.x[r * k ..][0..k], self.w[e * k ..][0..k]) |*t, x, w| t.* = Iv.of(ref.bf(x) * w);
            out[r * self.shape.n + e] = s.sum(terms, routerDepth(self.shape), scratch);
        };
    }
};

const route_rule = "the top-k of sigmoid(logit) + bias, a NaN never picked, ties to the lower expert; weights score / their fp32 sum, " ++
    "ascending expert id";

/// Rows x experts x top-k: a warp a row, 8 rows a block.
pub fn routeSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 16, .n = experts, .k = topk }, .{ .rows = 1, .n = experts, .k = topk }, .{ .rows = 32, .n = experts, .k = topk }, .{ .rows = 2048, .n = experts, .k = topk } };
    return .{ .name = "cuda minimax tf_mm3_route (sigmoid + bias top-4)", .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = route_rule }, .tiles = .{ .{ .size = 8 }, .{}, .{} }, .served = served };
}

/// Each row's picks against float64 scores, ties exact (logits and biases on coarse grids), weights within 2^-20.
pub const Route = struct {
    kern: *const Kernels,
    /// control: the device sees the experts in reverse order, so its ties go to the higher id
    reversed: bool = false,
    shape: c.Shape = undefined,
    l: []f32 = &.{},
    bias: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub fn spec(self: *const Route) c.Spec {
        var s = routeSpec();
        if (self.reversed) s.name = "control: MiniMax route breaks its ties to the higher expert";
        return s;
    }

    pub fn control(self: *const Route) ?g.Gate {
        return if (self.reversed) .shapes else null;
    }

    pub fn free(self: *Route, a: std.mem.Allocator) void {
        a.free(self.l);
        a.free(self.bias);
        self.l = &.{};
        self.bias = &.{};
    }

    /// Logits in quarters of [-2, 2], biases 0, 1/64 or 1/32: ties everywhere; non-finite: NaN, +Inf, -Inf, a NaN bias.
    pub fn prepare(self: *Route, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n < 16 or shape.n > 256 or shape.k != topk) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.l = try a.alloc(f32, shape.rows * n);
        for (self.l) |*v| v.* = ref.int(rng, -8, 8) / 4;
        self.bias = try a.alloc(f32, n);
        for (self.bias) |*v| v.* = ref.int(rng, 0, 2) / 64;
        if (fill == .nonfinite) {
            const put = [4]f32{ ref.nan, ref.inf, -ref.inf, ref.nan };
            for (0..@min(shape.rows, 4)) |r| for (0..3) |j| {
                self.l[r * n + 5 * j + r] = put[r];
            };
            self.l[n + 40] = ref.inf; // row 1: two +Inf logits, tied
            self.bias[17] = ref.nan;
        }
        const dl = try a.dupe(f32, self.l);
        defer a.free(dl);
        const db = try a.dupe(f32, self.bias);
        defer a.free(db);
        if (self.reversed) {
            for (0..shape.rows) |r| std.mem.reverse(f32, dl[r * n ..][0..n]);
            std.mem.reverse(f32, db);
        }
        self.dev = .{ try ref.input(rig, dl), try ref.input(rig, db) };
        self.out = .{ try rig.output(shape.rows * topk * 4), try rig.output(shape.rows * topk * 4) };
        return true;
    }

    pub fn run(self: *Route, rig: *rig_.Rig) !void {
        try kern.route(self.kern, rig.s, self.dev[0], self.dev[1], self.out[0], self.out[1], self.shape.rows, self.shape.n, topk);
    }

    pub fn count(self: *const Route) usize {
        return self.shape.rows * topk * 8;
    }

    pub fn check(self: *const Route, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        const wts = got[self.shape.rows * topk * 4 ..];
        for (0..self.shape.rows) |r| {
            var ids: [topk]usize = undefined;
            var ws: [topk]f64 = undefined;
            for (&ids, &ws, 0..) |*id, *w, k| {
                const raw = ref.readI32(got, r * topk + k);
                if (raw < 0 or raw >= n) return try std.fmt.bufPrint(msg, "row {d} slot {d}: expert {d} out of range", .{ r, k, raw });
                id.* = if (self.reversed) n - 1 - @as(usize, @intCast(raw)) else @intCast(raw);
                w.* = ref.readF32(wts, r * topk + k);
            }
            if (self.reversed) { // back to ascending ids, weights with them
                std.mem.reverse(usize, &ids);
                std.mem.reverse(f64, &ws);
            }
            const l = self.l[r * n ..][0..n];
            var total: f64 = 0;
            for (ids, 0..) |p, k| {
                if (k > 0 and p <= ids[k - 1]) return try std.fmt.bufPrint(msg, "row {d}: ids not strictly ascending ({d} after {d})", .{ r, p, ids[k - 1] });
                const ch = choice(l[p], self.bias[p]);
                if (std.math.isNan(ch)) return try std.fmt.bufPrint(msg, "row {d}: expert {d} picked with a NaN score", .{ r, p });
                total += score(l[p]);
                for (0..n) |e| {
                    if (std.mem.indexOfScalar(usize, &ids, e) != null) continue;
                    const o = choice(l[e], self.bias[e]);
                    if (std.math.isNan(o)) continue;
                    if (o > ch + 0x1p-20 * (@abs(ch) + 1)) return try std.fmt.bufPrint(msg, "row {d}: expert {d} ({d:.6}) picked over {d} ({d:.6})", .{ r, p, ch, e, o });
                    if (e < p and l[e] == l[p] and self.bias[e] == self.bias[p]) return try std.fmt.bufPrint(msg, "row {d}: expert {d} picked ahead of tied expert {d}", .{ r, p, e });
                }
            }
            for (ids, ws, 0..) |p, w, k| {
                const want = score(l[p]) / total;
                if (!(@abs(w - want) <= 0x1p-20 * want + 0x1p-140)) return try std.fmt.bufPrint(msg, "row {d} slot {d}: weight {e:.8} where float64 gives {e:.8}", .{ r, k, w, want });
            }
        }
        return null;
    }

    fn score(l: f32) f64 {
        return 1 / (1 + @exp(-@as(f64, l)));
    }

    fn choice(l: f32, b: f32) f64 {
        return score(l) + b;
    }
};

/// Rows x the slot's outputs (routed, shared) x top-k: the forward's fp32 down rows.
pub fn partialSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 16, .n = 2 * hidden, .k = topk }, .{ .rows = 1, .n = 2 * hidden, .k = topk }, .{ .rows = 32, .n = 2 * hidden, .k = topk }, .{ .rows = 256, .n = 2 * hidden, .k = topk } };
    return .{ .name = "cuda minimax tf_mm3_moe_partial (fp32 down rows)", .accum = .f32, .depth = partialDepth, .out = .f32, .served = served };
}

fn partialDepth(s: c.Shape) usize {
    return s.k; // one fma a pick, in pick order
}

/// OUT[r][0] = sum_k w[r, k] EY[r, k] (an fma chain), OUT[r][1] = the shared expert's row as it is.
pub const Partial = struct {
    kern: *const Kernels,
    /// control: the device's weights one pick late within each row
    shifted: bool = false,
    shape: c.Shape = undefined,
    ey: []f32 = &.{},
    wts: []f32 = &.{},
    sy: []f32 = &.{},
    dev: [3]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Partial) c.Spec {
        var s = partialSpec();
        if (self.shifted) s.name = "control: MiniMax MoE partial weighs each pick with the next pick's weight";
        return s;
    }

    pub fn control(self: *const Partial) ?g.Gate {
        return if (self.shifted) .shapes else null;
    }

    pub fn free(self: *Partial, a: std.mem.Allocator) void {
        a.free(self.ey);
        a.free(self.wts);
        a.free(self.sy);
        self.ey = &.{};
        self.wts = &.{};
        self.sy = &.{};
    }

    /// Down rows in [-2, 2), weights a row's normalized sigmoids; stress: integers past 2^10 times weights past 2^10.
    pub fn prepare(self: *Partial, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n % 2 != 0 or shape.k == 0 or shape.k > 8) return false;
        self.free(a);
        self.shape = shape;
        const h = shape.n / 2;
        const k = shape.k;
        const rows = shape.rows;
        self.ey = try ref.f32s(a, rng, rows * k * h, -2, 2);
        self.wts = try ref.f32s(a, rng, rows * k, 0.05, 1);
        self.sy = try ref.f32s(a, rng, rows * h, -2, 2);
        for (0..rows) |r| {
            var t: f32 = 0;
            for (self.wts[r * k ..][0..k]) |w| t += w;
            for (self.wts[r * k ..][0..k]) |*w| w.* /= t;
        }
        if (fill == .stress) for (0..rows) |r| for (0..k / 2) |j| {
            const w = ref.int(rng, 1024, 2000);
            self.wts[r * k + j] = w;
            self.wts[r * k + k / 2 + j] = -w;
            for (0..h) |i| {
                const y = ref.int(rng, 1024, 2000);
                self.ey[(r * k + j) * h + i] = y;
                self.ey[(r * k + k / 2 + j) * h + i] = y + @as(f32, if (j == 0) 1 else 0);
            }
        };
        if (fill == .nonfinite) {
            self.ey[(0 * k + 1) * h + 5] = ref.nan;
            if (rows > 1) self.ey[(1 * k + 2) * h + 9] = ref.inf;
            if (rows > 2) self.wts[2 * k] = -ref.inf;
            if (rows > 3) self.sy[3 * h + 7] = ref.nan;
        }
        const dw = try a.dupe(f32, self.wts);
        defer a.free(dw);
        if (self.shifted) for (0..rows) |r| std.mem.rotate(f32, dw[r * k ..][0..k], 1);
        self.dev = .{ try ref.input(rig, self.ey), try ref.input(rig, dw), try ref.input(rig, self.sy) };
        self.out = try rig.output(rows * shape.n * 4);
        return true;
    }

    pub fn run(self: *Partial, rig: *rig_.Rig) !void {
        try kern.moePartial(self.kern, rig.s, self.dev[0], self.dev[1], self.dev[2], self.out, self.shape.rows, self.shape.k, self.shape.n / 2);
    }

    pub fn count(self: *const Partial) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Partial, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const h = self.shape.n / 2;
        const k = self.shape.k;
        var terms: [8]Iv = undefined;
        var scratch: [8]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..h) |i| {
            for (terms[0..k], 0..) |*t, j| t.* = Iv.of(@as(f64, self.ey[(r * k + j) * h + i]) * self.wts[r * k + j]);
            out[(r * 2) * h + i] = s.sum(terms[0..k], k, &scratch);
            out[(r * 2 + 1) * h + i] = Iv.of(self.sy[r * h + i]);
        };
    }
};
