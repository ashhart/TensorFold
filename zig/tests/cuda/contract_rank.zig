//! tf_rank_sum (#642) under the contract on CUDA: each element its ranks' fp32 terms added rank 0 first.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// The rank whose term comes from `own`; its slot in `peers` holds NaN, which a correct sum never reads.
const me = 1;

/// world - 1 adds, each rounded.
fn rankDepth(s: c.Shape) usize {
    return s.k;
}

/// Shape: rows x n elements (a rank's rows of a hidden state), k ranks.
pub fn spec() c.Spec {
    return .{
        .name = "cuda tf_rank_sum (a reduce-scatter's rank-order sum)",
        .accum = .f32,
        .depth = rankDepth,
        .out = .f32,
        .tiles = .{ .{}, .{ .size = 256 }, .{} },
        // GLM-5.3-Flash's hidden rows on 2, 4 and 5 ranks (#804); 300 rows of 4,096 wrap the grid's 1,048,576 elements
        .served = &.{ .{ .rows = 16, .n = 4096, .k = 2 }, .{ .rows = 300, .n = 4096, .k = 4 }, .{ .rows = 5, .n = 4096, .k = 5 } },
    };
}

/// Terms [rank][element]: over nine binades, NaN and infinities at known elements, or gate 4's cancelling integers.
fn terms(t: []f32, shape: c.Shape, fill: g.Fill, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const e = shape.rows * shape.n;
    const k = shape.k;
    for (t) |*v| v.* = (rng.float(f32) * 2 - 1) * std.math.ldexp(@as(f32, 1), @as(i32, @intCast(rng.uintLessThan(u32, 9))) - 4);
    switch (fill) {
        .random => {},
        .nonfinite => {
            const n = shape.n;
            const at = [_][3]usize{ .{ 0, 0, n / 3 }, .{ k - 1, 1, n / 2 }, .{ me, 2, n - 1 }, .{ 0, 3, 0 }, .{ k - 1, 3, 0 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |p, v| if (p[1] < shape.rows) {
                t[p[0] * e + p[1] * n + p[2]] = v;
            };
        },
        .stress => for (0..e) |i| {
            const h = k / 2;
            for (0..h) |r| {
                const big: f32 = @floatFromInt((@as(u32, 1) << 19) + (rng.uintLessThan(u32, 1 << 18) * 2 + 1));
                t[r * e + i] = big;
                t[(r + h) * e + i] = -big + @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 8)));
            }
            if (k % 2 == 1) t[(k - 1) * e + i] = @floatFromInt(1 + rng.uintLessThan(u32, 8));
        },
    }
}

pub const Rank = struct {
    rig: rig_.Rig,
    sum: *const cuda.rank_scatter.RankScatter,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    t: []f32 = &.{},
    peers: u64 = 0,
    own: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Rank) void {
        self.rig.gpu.gpa.free(self.t);
        self.rig.deinit();
    }

    pub fn kernel(self: *Rank) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Rank);

    pub fn prepare(self: *Rank, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.k <= me or shape.k > 8 or shape.rows * shape.n == 0) return false;
        const a = self.rig.gpu.gpa;
        const e = shape.rows * shape.n;
        a.free(self.t);
        self.t = &.{};
        self.t = try a.alloc(f32, shape.k * e);
        self.shape = shape;
        terms(self.t, shape, fill, seed);
        const slots = try a.dupe(f32, self.t);
        defer a.free(slots);
        @memset(slots[me * e ..][0..e], std.math.nan(f32));
        self.peers = try self.rig.input(std.mem.sliceAsBytes(slots));
        self.own = try self.rig.input(std.mem.sliceAsBytes(self.t[me * e ..][0..e]));
        self.out = try self.rig.output(e * 4);
        return true;
    }

    pub fn run(self: *Rank) !void {
        try self.sum.rankSum(self.rig.s, self.peers, self.own, self.shape.k, me, self.count(), self.out);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Rank) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Rank, want: []f64, mag: []f64) !void {
        const e = self.count();
        for (want, mag, 0..) |*w, *m, i| {
            var s: f64 = 0;
            var abs: f64 = 0;
            for (0..self.shape.k) |r| {
                const v: f64 = self.t[r * e + i];
                s += v;
                abs += @abs(v);
            }
            w.* = s;
            m.* = abs;
        }
    }

    pub fn simulate(self: *Rank, acc: c.Accum, order: g.Order, out: []f64) !void {
        const e = self.count();
        var buf: [8]f64 = undefined;
        for (out, 0..) |*o, i| {
            for (buf[0..self.shape.k], 0..) |*b, r| b.* = self.t[r * e + i];
            o.* = rows.sumIn(acc, order, buf[0..self.shape.k]);
        }
    }
};
