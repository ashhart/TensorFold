//! Muse Spark's router and head glue under the contract: router, top 8, argmax, soft cap and the RoPE tables.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mu = @import("contract_muse_util.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const dense = core.contract.dense;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A lane's k / 32 products in order, then the warp's 32 lanes by halves.
fn laneChain(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

pub const Kind = enum { router, topk, argmax, softcap, rope };

const topk_rule = "the top 8 of sigmoid(logit * mult) + bias, ties to the lower expert, in expert order; weights " ++
    "s / (sum s + 1e-15) * scale within 2^-20 of float64; a NaN score at expert e never leaves its tree node: e = 0 is " ++
    "picked first (every weight NaN), any other e is never picked and hides the experts j = e mod 2^ceil(log2(e + 1))";
const argmax_rule = "the first NaN, else the first maximum, over both ranks' slices up to the vocabulary";

const top = 8;
const mult: f32 = 0.17677669529663687; // output_multiplier
const cap: f32 = 20; // final_logit_softcapping
const theta: f64 = 500000;

pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .router => .{ .name = "cuda muse muse_router (fp32 rows times bf16 weights)", .accum = .f32, .depth = laneChain, .out = .f32, .tiles = .{ .{}, .{ .size = 2 }, .{ .size = 256 } }, .served = &.{ .{ .rows = 8, .n = 256, .k = 8192 }, .{ .rows = 1, .n = 256, .k = 8192 } } },
        .topk => .{ .name = "cuda muse muse_topk (sigmoid top 8 of 256)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = topk_rule }, .tiles = .{ .{}, .{ .size = 256, .step = 32, .min = 32 }, .{} }, .served = &.{ .{ .rows = 8, .n = 256, .k = top }, .{ .rows = 128, .n = 256, .k = top } } },
        .argmax => .{ .name = "cuda muse muse_argmax (two ranks' head slices)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = argmax_rule }, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{ .{ .rows = 8, .n = 202048, .k = 101120 }, .{ .rows = 1, .n = 202048, .k = 101120 } } },
        .softcap => .{ .name = "cuda muse muse_softcap (in place)", .accum = .none, .depth = noSums, .ops = 8, .out = .f32, .in_place = true, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{ .{ .rows = 8, .n = 101120, .k = 1 }, .{ .rows = 1, .n = 101120, .k = 1 } } },
        .rope => .{ .name = "cuda muse muse_rope_tables (fp64 then fp32)", .accum = .none, .depth = noSums, .out = .f32, .tiles = .{ .{ .size = 65536 }, .{}, .{} }, .served = &.{ .{ .rows = 471040, .n = 64, .k = 1 }, .{ .rows = 8192, .n = 128, .k = 1 } } },
    };
}

pub const Route = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    d: ?dense.Dense = null,
    x: []f64 = &.{}, // logits, head slices or soft-cap inputs
    bias: []f64 = &.{},
    th: f64 = theta,
    dev: [4]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Route) void {
        if (self.d) |d| d.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    const vt_ref = blk: {
        var v = g.hooks(Route);
        v.check = null;
        break :blk v;
    };
    const vt_rule = g.hooks(Route);

    pub fn kernel(self: *Route) g.Kernel {
        const ruled = self.kind == .topk or self.kind == .argmax;
        return .{ .spec = spec(self.kind), .ctx = self, .vt = if (ruled) &vt_rule else &vt_ref };
    }

    pub fn prepare(self: *Route, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        if (self.d) |d| d.deinit();
        self.d = null;
        self.shape = shape;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        switch (self.kind) {
            .router => {
                const d = try dense.Dense.init(ga, shape, 1);
                self.d = d;
                d.fillX(fill, seed, 1);
                const k = shape.k;
                for (0..shape.n) |j| for (0..k) |i| {
                    d.w[j * k + i] = if (fill != .stress) mu.uni(rng, u16, -1, 1) else if (i < k / 2) mu.int(rng, 1, 6) else -d.w[j * k + i - k / 2];
                };
                if (fill == .stress and k % 2 == 1) for (0..shape.n) |j| {
                    d.w[j * k + k - 1] = 0;
                };
                const xs = try a.alloc(f64, d.x.len);
                for (xs, d.x) |*o, b| o.* = c.fromBf16(b);
                self.dev[0] = try mu.put(&self.rig, ga, f32, xs);
                self.dev[1] = try mu.put(&self.rig, ga, u16, d.w);
                self.dev[2] = try self.rig.output(shape.rows * shape.n * 4);
            },
            .topk => {
                if (shape.n > 1024 or shape.k != top) return false;
                self.x = try a.alloc(f64, shape.rows * shape.n);
                self.bias = try a.alloc(f64, shape.n);
                for (self.x) |*v| v.* = mu.int(rng, -64, 64) / 8; // eighths: some ties
                for (self.bias) |*v| v.* = mu.int(rng, 0, 7) / 64;
                if (nf) {
                    self.x[3] = std.math.nan(f64); // never picked
                    if (shape.rows > 1) self.x[shape.n] = std.math.nan(f64); // expert 0: the root keeps it
                    if (shape.rows > 2) self.x[2 * shape.n + 5] = std.math.inf(f64);
                    if (shape.rows > 3) self.x[3 * shape.n + 1] = -std.math.inf(f64);
                }
                self.dev[0] = try mu.put(&self.rig, ga, f32, self.x);
                self.dev[1] = try mu.put(&self.rig, ga, f32, self.bias);
                self.dev[2] = try self.rig.output(shape.rows * top * 4);
                self.dev[3] = try self.rig.output(shape.rows * top * 4);
            },
            .argmax => {
                const w0 = shape.k;
                if (shape.n > 2 * w0) return false;
                const w1: usize = if (shape.n > w0) w0 else 0; // rank 1's slice, padded as rank 0's
                self.x = try a.alloc(f64, shape.rows * shape.n);
                for (self.x) |*v| v.* = mu.int(rng, -1024, 1024) / 16; // sixteenths: ties
                if (nf) {
                    const n = shape.n;
                    self.x[n / 2] = std.math.nan(f64);
                    self.x[n - 1] = std.math.nan(f64); // the first NaN wins
                    if (shape.rows > 1) self.x[n + n / 3] = std.math.inf(f64);
                    if (shape.rows > 2) self.x[2 * n + 7] = -std.math.inf(f64);
                }
                // device layout: rank 0's [rows, w0], then rank 1's [rows, w1]; NaN past the vocabulary (never read)
                const lay = try a.alloc(f64, shape.rows * (w0 + w1));
                @memset(lay, std.math.nan(f64));
                for (0..shape.rows) |r| for (0..shape.n) |col| {
                    const at = if (col < w0) r * w0 + col else shape.rows * w0 + r * w1 + col - w0;
                    lay[at] = self.x[r * shape.n + col];
                };
                self.dev[0] = try mu.put(&self.rig, ga, f32, lay);
                self.dev[1] = try self.rig.output(shape.rows * 4);
            },
            .softcap => {
                self.x = try a.alloc(f64, shape.rows * shape.n);
                for (self.x) |*v| v.* = mu.uni(rng, f32, -400, 400);
                if (nf) {
                    self.x[1] = std.math.nan(f64);
                    self.x[2] = std.math.inf(f64);
                    self.x[3] = -std.math.inf(f64);
                }
                self.dev[0] = try mu.put(&self.rig, ga, f32, self.x);
                self.dev[1] = try self.rig.output(self.x.len * 4);
            },
            .rope => {
                if (shape.n % 2 != 0) return false;
                self.th = if (nf) std.math.nan(f64) else theta; // column 0 stays finite: theta^0 = 1
                const half = shape.rows * shape.n / 2;
                self.dev[0] = try self.rig.output(2 * half * 4);
            },
        }
        return true;
    }

    /// The family's launches (ops.moe, forward's head and tables).
    pub fn run(self: *Route) !void {
        const s = self.rig.s;
        const sh = self.shape;
        var a: cuda.Args = .{};
        var grid: [2]u32 = .{ @intCast(sh.rows), 1 };
        var block: u32 = 256;
        switch (self.kind) {
            .router => {
                inline for (.{ self.dev[0], self.dev[1] }) |p| a.add(p);
                a.add(@as(c_int, 1));
                a.add(self.dev[2]);
                inline for (.{ sh.k, sh.n }) |v| a.add(@as(c_int, @intCast(v)));
                grid = .{ @intCast((sh.n + 1) / 2), @intCast(sh.rows) };
                block = 64;
            },
            .topk => {
                inline for (.{ self.dev[0], self.dev[1] }) |p| a.add(p);
                inline for (.{ sh.n, top }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(mult);
                a.add(@as(f32, 1));
                inline for (.{ self.dev[2], self.dev[3] }) |p| a.add(p);
                block = @intCast(std.math.ceilPowerOfTwo(usize, @max(sh.n, 32)) catch unreachable);
            },
            .argmax => {
                inline for (.{ self.dev[0], self.dev[1] }) |p| a.add(p);
                const w1: usize = if (sh.n > sh.k) sh.k else 0;
                inline for (.{ sh.k, w1 }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(@as(i64, @intCast(sh.rows * sh.k)));
                a.add(@as(c_int, @intCast(sh.n)));
            },
            .softcap => {
                const n = sh.rows * sh.n;
                try mu.copy(&self.rig, self.dev[1], self.dev[0], n * 4);
                a.add(self.dev[1]);
                a.add(@as(i64, @intCast(n)));
                a.add(mult);
                a.add(cap);
                grid = .{ @intCast(@min((n + 255) / 256, 4096)), 1 };
            },
            .rope => {
                const half = sh.rows * sh.n / 2;
                a.add(self.dev[0]);
                a.add(self.dev[0] + half * 4);
                a.add(@as(c_int, @intCast(sh.rows)));
                a.add(self.th);
                a.add(@as(c_int, @intCast(sh.n)));
                grid = .{ 256, 1 };
            },
        }
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = 1 }, .block = .{ .x = block } }, s, &a);
        try s.synchronize();
    }

    pub fn count(self: *Route) usize {
        const sh = self.shape;
        return switch (self.kind) {
            .router, .softcap, .rope => sh.rows * sh.n,
            .topk => sh.rows * top * 8,
            .argmax => sh.rows * 4,
        };
    }

    pub fn reference(self: *Route, want: []f64, mag: []f64) !void {
        switch (self.kind) {
            .router => return self.d.?.reference(want, mag),
            .softcap => for (want, mag, self.x) |*w, *m, x| {
                const v = x * mult;
                w.* = std.math.tanh(v / cap) * cap;
                m.* = 0;
            },
            .rope => {
                const half = self.shape.n / 2;
                const nh = self.shape.rows * half;
                for (0..nh) |i| {
                    const ang = self.angle(i);
                    want[i] = @cos(ang);
                    want[nh + i] = @sin(ang);
                    mag[i] = 0;
                    mag[nh + i] = 0;
                }
            },
            .topk, .argmax => return error.StatedRule, // gates 1 and 3 run check; no sums for gate 4
        }
    }

    fn angle(self: *const Route, i: usize) f64 {
        const half = self.shape.n / 2;
        const j: f64 = @floatFromInt(i % half);
        const p: f64 = @floatFromInt(i / half);
        return p * (1 / std.math.pow(f64, self.th, 2 * j / @as(f64, @floatFromInt(self.shape.n))));
    }

    /// The tables: fp64 pow / cos / sin within a few double ulps of the angle; the soft cap's x / cap and tanhf.
    pub fn bounds(self: *Route, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        if (self.kind != .rope) return;
        const nh = want.len / 2;
        for (allow, want, 0..) |*l, w, i| l.* = mu.allow(.f32, w, 0x1p-50 * (@abs(self.angle(i % nh)) + 1));
    }

    pub fn simulate(self: *Route, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.kind != .router) return error.NoSums;
        return self.d.?.simulate(acc, order, out);
    }

    /// Expert e's sigmoid and rank score under the stated NaN rule (expert 0 first; a NaN node hides its leaves).
    fn score(self: *const Route, r: usize, e: usize) [2]f64 {
        const row = self.x[r * self.shape.n ..][0..self.shape.n];
        const s = 1 / (1 + @exp(-(row[e] * mult)));
        const v = s + self.bias[e];
        if (std.math.isNan(v)) return .{ s, if (e == 0) std.math.inf(f64) else -std.math.inf(f64) };
        for (row[1..], 1..) |x, j| if (std.math.isNan(x) and j != e) {
            const span = std.math.ceilPowerOfTwo(usize, j + 1) catch unreachable;
            if (e % span == j % span) return .{ s, -std.math.inf(f64) };
        };
        return .{ s, v };
    }

    pub fn check(self: *Route, _: g.Gate) !?[]const u8 {
        return if (self.kind == .topk) self.checkTopk() else self.checkArgmax();
    }

    fn checkTopk(self: *Route) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const n = self.shape.n;
        const ids = try a.dupe(u8, try self.rig.bytes(self.dev[2], self.shape.rows * top * 4));
        defer a.free(ids);
        const wts = try self.rig.bytes(self.dev[3], self.shape.rows * top * 4);
        for (0..self.shape.rows) |r| {
            var taken = std.StaticBitSet(1024).empty;
            var pick: [top]usize = undefined;
            var prev: i64 = -1;
            for (&pick, 0..) |*p, k| {
                const id = std.mem.readInt(i32, ids[(r * top + k) * 4 ..][0..4], .little);
                if (id < 0 or id >= n or id <= prev) return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: expert {d} out of range or out of order", .{ r, k, id });
                prev = id;
                p.* = @intCast(id);
                taken.set(p.*);
            }
            var lowest: f64 = std.math.inf(f64);
            var sum: f64 = 0;
            for (pick) |p| {
                lowest = @min(lowest, self.score(r, p)[1]);
                sum += self.score(r, p)[0];
            }
            for (0..n) |e| if (!taken.isSet(e)) {
                const v = self.score(r, e)[1];
                const tol = 0x1p-20 * (@abs(lowest) + 1);
                if (v > lowest + tol) return try std.fmt.bufPrint(&self.msg, "row {d}: expert {d} scores {d:.6} over a pick's {d:.6}", .{ r, e, v, lowest });
                if (v == lowest) for (pick) |p| if (p > e and self.score(r, p)[1] == v)
                    return try std.fmt.bufPrint(&self.msg, "row {d}: expert {d} picked over tied expert {d}", .{ r, p, e });
            };
            for (pick, 0..) |p, k| {
                const w: f64 = @as(f32, @bitCast(std.mem.readInt(u32, wts[(r * top + k) * 4 ..][0..4], .little)));
                const want = self.score(r, p)[0] / (sum + 1e-15);
                const ok = if (std.math.isFinite(want)) @abs(w - want) <= 0x1p-20 * @abs(want) + 1e-30 else c.classOf(w) == c.classOf(want);
                if (!ok) return try std.fmt.bufPrint(&self.msg, "row {d} slot {d}: weight {d:.7} where float64 gives {d:.7}", .{ r, k, w, want });
            }
        }
        return null;
    }

    fn checkArgmax(self: *Route) !?[]const u8 {
        const got = try self.rig.bytes(self.dev[1], self.shape.rows * 4);
        const n = self.shape.n;
        for (0..self.shape.rows) |r| {
            const row = self.x[r * n ..][0..n];
            var best: usize = 0;
            for (row, 0..) |v, i| {
                const b = row[best];
                if (std.math.isNan(b)) break;
                if (std.math.isNan(v) or v > b) best = i;
            }
            const id = std.mem.readInt(i32, got[r * 4 ..][0..4], .little);
            if (id != @as(i32, @intCast(best))) return try std.fmt.bufPrint(&self.msg, "row {d}: column {d} where the rule gives {d}", .{ r, id, best });
        }
        return null;
    }
};
