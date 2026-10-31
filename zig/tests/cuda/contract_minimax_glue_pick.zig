//! MiniMax-M3.1-Flash's CUDA greedy picks under the contract: tf_mm3_top1, DSpark's Markov scores, top1f and the merge.
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

/// TP=2's vocabulary shares: rank 0 holds 100,096 rows, rank 1 the last 99,968 from 100,096.
const share0 = 100_096;
const share1 = 99_968;

fn baseOf(n: usize) usize {
    return if (n == share1) share0 else 0;
}

const top_rule = "the largest value and its lowest id plus the share's base; a NaN never picked, +Inf the largest";

/// Rows x the rank's vocabulary: one block a row (a verify window, a batch, or one row over the whole vocabulary).
pub fn top1Spec(f32_in: bool) c.Spec {
    if (f32_in) return .{ .name = "cuda minimax tf_mm3_top1f (fp32 scores, a launch a draft)", .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = top_rule }, .served = &.{ .{ .rows = 7, .n = share0, .k = 1 }, .{ .rows = 1, .n = share1, .k = 1 }, .{ .rows = 8, .n = 5000, .k = 1 } } };
    const served = &[_]c.Shape{ .{ .rows = 16, .n = share0, .k = 1 }, .{ .rows = 1, .n = 200_064, .k = 1 }, .{ .rows = 32, .n = share1, .k = 1 }, .{ .rows = 7, .n = share0, .k = 1 } };
    return .{ .name = "cuda minimax tf_mm3_top1 (bf16 logits)", .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = top_rule }, .served = served };
}

/// Each row's (value, id) against a scan of its values: equal values everywhere, so the lowest id must win.
pub const Top1 = struct {
    kern: *const Kernels,
    f32_in: bool,
    shape: c.Shape = undefined,
    vals: []f32 = &.{},
    dev: u64 = 0,
    out: [2]u64 = @splat(0),

    pub fn spec(self: *const Top1) c.Spec {
        return top1Spec(self.f32_in);
    }

    pub fn control(_: *const Top1) ?g.Gate {
        return null;
    }

    pub fn free(self: *Top1, a: std.mem.Allocator) void {
        a.free(self.vals);
        self.vals = &.{};
    }

    /// Values on 1/8 steps of [-64, 64) (each maximum held by several ids); non-finite: NaNs, two +Infs, -Infs.
    pub fn prepare(self: *Top1, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.vals = try a.alloc(f32, shape.rows * n);
        for (self.vals) |*v| v.* = ref.int(rng, -512, 511) / 8;
        if (!self.f32_in) for (self.vals) |*v| {
            v.* = c.fromBf16(c.toBf16(v.*));
        };
        if (fill == .nonfinite) {
            const best = std.mem.indexOfMax(f32, self.vals[0..n]);
            for ([_]usize{ 0, best, n - 1 }) |i| self.vals[i] = ref.nan; // row 0: its first maximum NaN, then the next wins
            if (shape.rows > 1) for ([_]usize{ n / 2, n / 3 }) |i| {
                self.vals[n + i] = ref.inf;
            };
            if (shape.rows > 2) for (0..n / 2) |i| {
                self.vals[2 * n + i] = -ref.inf;
            };
            if (shape.rows > 3) self.vals[3 * n + 7] = -ref.inf;
        }
        if (self.f32_in) self.dev = try ref.input(rig, self.vals) else {
            const bits = try a.alloc(u16, self.vals.len);
            defer a.free(bits);
            for (bits, self.vals) |*b, v| b.* = c.toBf16(v);
            self.dev = try ref.input(rig, bits);
        }
        self.out = .{ try rig.output(shape.rows * 4), try rig.output(shape.rows * 4) };
        return true;
    }

    pub fn run(self: *Top1, rig: *rig_.Rig) !void {
        const n = self.shape.n;
        const base = baseOf(n);
        if (!self.f32_in) return kern.top1(self.kern, rig.s, self.dev, self.shape.rows, n, base, self.out[0], self.out[1]);
        for (0..self.shape.rows) |j| try kern.top1f(self.kern, rig.s, self.dev + j * n * 4, n, base, self.out[0] + j * 4, self.out[1] + j * 4);
    }

    pub fn count(self: *const Top1) usize {
        return self.shape.rows * 8;
    }

    pub fn check(self: *const Top1, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        const rows = self.shape.rows;
        for (0..rows) |r| {
            var best = -std.math.inf(f32);
            var id: usize = n;
            for (self.vals[r * n ..][0..n], 0..) |v, i| if (v > best or (v == best and id == n)) {
                best = v;
                id = i;
            };
            const v = ref.readF32(got, r);
            const i = ref.readI32(got, rows + r);
            if (v != best or i != @as(i32, @intCast(id + baseOf(n)))) return try std.fmt.bufPrint(msg, "row {d}: ({e}, {d}) where ({e}, {d}) is due", .{ r, v, i, best, id + baseOf(n) });
        }
        return null;
    }
};

/// A lane's 8 fmas of the rank-256 product, then the warp's 5 shuffles.
fn markovDepth(_: c.Shape) usize {
    return 8 + 5;
}

/// Drafts (a launch each) x the rank's vocabulary x DSpark's Markov rank (256).
pub fn markovSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 7, .n = share0, .k = 256 }, .{ .rows = 1, .n = share1, .k = 256 }, .{ .rows = 8, .n = 3000, .k = 256 } };
    return .{ .name = "cuda minimax tf_mm3_markov (logits + bf16(w2 . w1[prev]))", .accum = .f32, .depth = markovDepth, .out = .f32, .served = served };
}

const prev_rows = 4096; // w1's rows the walk's ids index (the model's: the whole vocabulary)

/// S[v] = logits[v] + bf16(w2[v] . w1[prev]), prev the walk's pick before (the anchor first), left on the device.
pub const Markov = struct {
    kern: *const Kernels,
    /// control: the device reads the walk one pick late
    late: bool = false,
    shape: c.Shape = undefined,
    logits: []u16 = &.{},
    w2: []u16 = &.{},
    w1: []u16 = &.{},
    walk: []i32 = &.{},
    dev: [4]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Markov) c.Spec {
        var s = markovSpec();
        if (self.late) s.name = "control: MiniMax Markov scores read the walk one pick late";
        return s;
    }

    pub fn control(self: *const Markov) ?g.Gate {
        return if (self.late) .shapes else null;
    }

    pub fn free(self: *Markov, a: std.mem.Allocator) void {
        a.free(self.logits);
        a.free(self.w2);
        a.free(self.w1);
        a.free(self.walk);
        self.logits = &.{};
        self.w2 = &.{};
        self.w1 = &.{};
        self.walk = &.{};
    }

    /// Logits in [-16, 16), factors in [-0.5, 0.5); stress: integers to 127 (products past 2^13), halves cancelling.
    pub fn prepare(self: *Markov, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != 256) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.logits = try ref.bfs(a, rng, shape.rows * n, -16, 16);
        self.w2 = try ref.bfs(a, rng, n * 256, -0.5, 0.5);
        self.w1 = try ref.bfs(a, rng, prev_rows * 256, -0.5, 0.5);
        self.walk = try a.alloc(i32, shape.rows + 1);
        for (self.walk) |*p| p.* = @intCast(rng.uintLessThan(u32, prev_rows));
        if (fill == .stress) {
            @memset(self.logits, 0);
            for (0..n) |v| for (0..128) |i| {
                const x = ref.int(rng, 1, 127);
                self.w2[v * 256 + i] = c.toBf16(x);
                self.w2[v * 256 + 128 + i] = c.toBf16(x + @as(f32, if (i < 2) 1 else 0));
            };
            for (0..prev_rows) |p| for (0..128) |i| {
                const v = ref.int(rng, 1, 127);
                self.w1[p * 256 + i] = c.toBf16(v);
                self.w1[p * 256 + 128 + i] = c.toBf16(-v);
            };
        }
        if (fill == .nonfinite) {
            self.logits[3] = c.toBf16(ref.nan);
            if (shape.rows > 1) self.logits[n + 5] = c.toBf16(ref.inf);
            self.w2[9 * 256 + 17] = c.toBf16(ref.nan); // vocabulary row 9: NaN in every draft
            if (shape.rows > 2) self.walk[2] = self.walk[shape.rows]; // draft 2 reads the walk's last pick's row
            self.w1[@as(usize, @intCast(self.walk[shape.rows])) * 256 + 40] = c.toBf16(-ref.inf);
        }
        self.dev = .{ try ref.input(rig, self.logits), try ref.input(rig, self.w2), try ref.input(rig, self.w1), try ref.input(rig, self.walk) };
        self.out = try rig.output(shape.rows * n * 4);
        return true;
    }

    pub fn run(self: *Markov, rig: *rig_.Rig) !void {
        const n = self.shape.n;
        const late: usize = @intFromBool(self.late);
        for (0..self.shape.rows) |j| try kern.markov(self.kern, rig.s, self.dev[0] + j * n * 2, self.dev[1], self.dev[2], self.dev[3] + (j + late) * 4, self.out + j * n * 4, n);
    }

    pub fn count(self: *const Markov) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Markov, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        var terms: [256]Iv = undefined;
        var scratch: [256]f64 = undefined;
        for (0..self.shape.rows) |j| {
            const prev: usize = @intCast(self.walk[j]);
            for (0..n) |v| {
                for (&terms, self.w2[v * 256 ..][0..256], self.w1[prev * 256 ..][0..256]) |*t, x, y| t.* = Iv.of(ref.bf(x) * ref.bf(y));
                const dot = s.sum(&terms, markovDepth(self.shape), &scratch).bf();
                out[j * n + v] = Iv.of(ref.bf(self.logits[j * n + v])).add(dot);
            }
        }
    }
};

const merge_rule = "walk[j + 1]: the ranks' best value, the lower id on a tie; a NaN never wins; with no value above -Inf the id is 0";

/// Drafts (a launch each) x ranks x 1: every rank's (value, id) at 16 bytes a rank.
pub fn mergeSpec() c.Spec {
    return .{ .name = "cuda minimax tf_mm3_pick_merge", .accum = .none, .depth = ref.noSums, .out = .raw, .nan = .{ .stated = merge_rule }, .served = &.{ .{ .rows = 7, .n = 2, .k = 1 }, .{ .rows = 1, .n = 2, .k = 1 }, .{ .rows = 8, .n = 1, .k = 1 } } };
}

const stride = 16;

pub const Merge = struct {
    kern: *const Kernels,
    shape: c.Shape = undefined,
    vals: []f32 = &.{},
    ids: []i32 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    pub fn spec(_: *const Merge) c.Spec {
        return mergeSpec();
    }

    pub fn control(_: *const Merge) ?g.Gate {
        return null;
    }

    pub fn free(self: *Merge, a: std.mem.Allocator) void {
        a.free(self.vals);
        a.free(self.ids);
        self.vals = &.{};
        self.ids = &.{};
    }

    /// Values on quarter steps of [-2, 2) (ties between ranks), each rank's id in its own share.
    pub fn prepare(self: *Merge, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > 8) return false;
        self.free(a);
        self.shape = shape;
        const m = shape.rows * shape.n;
        self.vals = try a.alloc(f32, m);
        self.ids = try a.alloc(i32, m);
        for (self.vals, self.ids, 0..) |*v, *id, i| {
            v.* = ref.int(rng, -8, 7) / 4;
            id.* = @intCast(rng.uintLessThan(u32, share1) + (i % shape.n) * share0);
        }
        if (fill == .nonfinite) {
            const put = [_]f32{ ref.nan, ref.inf, -ref.inf, ref.nan };
            for (put, 0..) |v, r| if (r < shape.rows) {
                self.vals[r * shape.n] = v;
            };
            if (shape.rows > 2) @memset(self.vals[2 * shape.n ..][0..shape.n], -ref.inf); // row 2: no value above -Inf
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

    pub fn run(self: *Merge, rig: *rig_.Rig) !void {
        const w = self.shape.n;
        for (0..self.shape.rows) |j| try kern.pickMerge(self.kern, rig.s, self.dev + j * w * stride, w, stride, self.out, j);
    }

    pub fn count(self: *const Merge) usize {
        return (self.shape.rows + 1) * 4;
    }

    pub fn check(self: *const Merge, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const w = self.shape.n;
        if (ref.readI32(got, 0) != -1) return "walk[0] (the anchor) was written";
        for (0..self.shape.rows) |j| {
            var best = -std.math.inf(f32);
            var id: i32 = 0;
            var lowest: ?i32 = null;
            for (self.vals[j * w ..][0..w], self.ids[j * w ..][0..w]) |v, k| {
                if (v > best) {
                    best = v;
                    lowest = k;
                } else if (v == best and lowest != null and k < lowest.?) lowest = k;
            }
            if (lowest) |k| id = k;
            const v = ref.readI32(got, j + 1);
            if (v != id) return try std.fmt.bufPrint(msg, "draft {d}: id {d} where {d} is due", .{ j, v, id });
        }
        return null;
    }
};
