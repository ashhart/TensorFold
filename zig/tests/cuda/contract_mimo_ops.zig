//! MiMo-V2's row ops under the contract (ops.cu): norms, SwiGLU, the router and top 8, the experts' sum, vs float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mu = @import("contract_mimo_util.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const dense = core.contract.dense;

const NORM = 256; // ops.zig NORM_THREADS: a block a row, 16 values a thread at most
const eps: f32 = 1e-6; // layernorm_epsilon
const top = 8; // num_experts_per_tok
const div: f32 = 128; // config.interm_div: layer 47's routing weight

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A thread's ceil(D / 256) squares in order, then 32 lanes and 8 warps by halves.
fn rmsDepth(s: c.Shape) usize {
    return (s.k + NORM - 1) / NORM + 10;
}

/// A lane's k / 32 fused products in order, then the warp's 32 lanes by halves.
fn laneChain(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

/// A row's K fused products in slot order.
fn slotChain(s: c.Shape) usize {
    return s.k;
}

pub const Kind = enum { rms, rms_res, rms_sum, silu, wsum, router, topk, topk_div };

const rms_why = "an fp16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const topk_rule = "the top 8 of sigmoid(logit) + bias in pick order (each the highest left, ties to the lower " ++
    "expert); weights s / (sum s + 1e-20) * scale within 2^-20 of float64; a NaN score at expert e never leaves its " ++
    "tree node: e = 0 is picked first (every weight NaN), any other e is never picked and hides the experts " ++
    "j = e mod 2^ceil(log2(e + 1))";

pub fn spec(kind: Kind) c.Spec {
    const hidden = &[_]c.Shape{ .{ .rows = 8, .n = 4096, .k = 4096 }, .{ .rows = 1, .n = 4096, .k = 4096 }, .{ .rows = 128, .n = 4096, .k = 4096 } };
    const norm_tiles: [3]c.Tile = .{ .{}, .{}, .{ .size = NORM } };
    const rms_sums: c.Sums = .{ .bf16 = 0, .why = rms_why };
    const dense_mlp = &[_]c.Shape{ .{ .rows = 8, .n = 8192, .k = 1 }, .{ .rows = 1, .n = 8192, .k = 1 }, .{ .rows = 128, .n = 8192, .k = 1 }, .{ .rows = 8, .n = 16384, .k = 1 } };
    const slots = &[_]c.Shape{ .{ .rows = 64, .n = 4096, .k = top }, .{ .rows = 1, .n = 4096, .k = top }, .{ .rows = 2048, .n = 4096, .k = top } };
    const short: c.Sums = .{ .f16 = 0, .bf16 = 0, .why = "eight slot products a value: too short a sum for fp16 or bf16 to miss" };
    const silu_tiles: [3]c.Tile = .{ .{}, .{ .size = 256 }, .{} };
    const route = &[_]c.Shape{ .{ .rows = 8, .n = 256, .k = top }, .{ .rows = 128, .n = 256, .k = top } };
    const pick_tiles: [3]c.Tile = .{ .{}, .{ .size = 256, .step = 32, .min = 32 }, .{} };
    return switch (kind) {
        .rms => .{ .name = "cuda mimo mimo_rms (fp32 rows, fp16 out)", .accum = .f32, .depth = rmsDepth, .ops = 8, .out = .f16, .tiles = norm_tiles, .served = hidden, .sums = rms_sums },
        .rms_res => .{ .name = "cuda mimo mimo_rms (res += x in place, then its norm: the fp16 out)", .accum = .f32, .depth = rmsDepth, .ops = 8, .out = .f16, .in_place = true, .tiles = norm_tiles, .served = hidden, .sums = rms_sums },
        .rms_sum => .{ .name = "cuda mimo mimo_rms (res += x in place: the fp32 residual)", .accum = .none, .depth = noSums, .out = .f32, .in_place = true, .tiles = norm_tiles, .served = hidden },
        .silu => .{ .name = "cuda mimo mimo_silu_mul (fp16 silu(g) * u, the dense MLP)", .accum = .none, .depth = noSums, .ops = 10, .out = .f16, .tiles = silu_tiles, .served = dense_mlp },
        .wsum => .{ .name = "cuda mimo mimo_wsum (the prompt experts' fp32 slots times their weights, slot order)", .accum = .f32, .depth = slotChain, .out = .f32, .tiles = silu_tiles, .served = slots, .sums = short },
        .router => .{ .name = "cuda mimo mimo_router (fp16 rows times fp16 weights, fp32 logits)", .accum = .f32, .depth = laneChain, .out = .f32, .tiles = .{ .{}, .{ .size = 8 }, .{ .size = 64, .step = 2, .min = 2 } }, .served = &.{ .{ .rows = 8, .n = 256, .k = 4096 }, .{ .rows = 1, .n = 256, .k = 4096 }, .{ .rows = 128, .n = 256, .k = 4096 } } },
        .topk => .{ .name = "cuda mimo mimo_topk (sigmoid top 8 of 256, scale 1)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = topk_rule }, .tiles = pick_tiles, .served = route },
        .topk_div => .{ .name = "cuda mimo mimo_topk (sigmoid top 8 of 256, layer 47's scale 128)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = topk_rule }, .tiles = pick_tiles, .served = route },
    };
}

pub const Op = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    d: ?dense.Dense = null,
    v: []f64 = &.{}, // rms: the normed rows as the kernel holds them; silu: g; wsum: the slots; topk: the logits
    x: []f64 = &.{}, // rms: the rows added; silu: u
    res: []f64 = &.{}, // rms: the residual before the add
    w: []f64 = &.{}, // rms: the norm weight; wsum: the routing weights; topk: the bias
    dev: [6]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Op) void {
        if (self.d) |d| d.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    const vt_ref = blk: {
        var v = g.hooks(Op);
        v.check = null;
        break :blk v;
    };
    const vt_rule = g.hooks(Op);

    pub fn kernel(self: *Op) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = if (self.ruled()) &vt_rule else &vt_ref };
    }

    fn ruled(self: *const Op) bool {
        return self.kind == .topk or self.kind == .topk_div;
    }

    pub fn prepare(self: *Op, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        if (self.d) |d| d.deinit();
        self.d = null;
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        return switch (self.kind) {
            .rms, .rms_res, .rms_sum => self.prepareRms(fill, rng),
            .silu => self.prepareSilu(fill, rng),
            .wsum => self.prepareWsum(fill, rng),
            .router => self.prepareRouter(fill, seed, rng),
            .topk, .topk_div => self.prepareTopk(fill, rng),
        };
    }

    /// Rows of D values: plain (x), or res + x rounded to fp32 as the kernel's in-place add leaves them.
    fn prepareRms(self: *Op, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        const D = sh.k;
        if (D > 16 * NORM) return false; // D is k (served with n = k)
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        const n = sh.rows * D;
        for ([_]*[]f64{ &self.v, &self.x, &self.res }) |p| p.* = try a.alloc(f64, n);
        self.w = try a.alloc(f64, D);
        for (self.w) |*o| o.* = mu.uni(rng, f32, 0.5, 1.5);
        const st = fill == .stress; // integer rows: their squares sum exactly in fp32 (Rows.stress)
        for (self.x, self.res) |*x, *r| {
            const v0 = if (st) mu.int(rng, -32, 32) else mu.uni(rng, f32, -2, 2);
            if (self.kind == .rms) {
                x.* = v0;
                r.* = 0;
            } else {
                x.* = if (st) mu.int(rng, -4, 4) else mu.uni(rng, f32, -2, 2);
                r.* = if (st) v0 - x.* else mu.uni(rng, f32, -2, 2);
            }
        }
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, D / 3 }, .{ 1, D / 2 }, .{ 2, D - 1 }, .{ 3, 0 }, .{ 3, D - 1 } };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (at, put) |p, x| if (p[0] < sh.rows) {
                self.x[p[0] * D + p[1]] = x;
            };
        }
        for (self.v, self.x, self.res) |*o, x, r| o.* = if (self.kind == .rms) x else mu.f32r(r + x);
        self.dev[0] = try mu.put(&self.rig, ga, f32, self.x);
        self.dev[1] = try mu.put(&self.rig, ga, f32, self.res); // copied into the working residual each run
        self.dev[2] = try mu.put(&self.rig, ga, f32, self.w);
        switch (self.kind) {
            .rms => self.dev[3] = try self.rig.output(n * 2),
            .rms_res => {
                self.dev[3] = try self.rig.output(n * 2);
                self.dev[4] = try self.rig.scratch(n * 4);
            },
            else => {
                self.dev[4] = try self.rig.output(n * 4);
                self.dev[3] = try self.rig.scratch(n * 2);
            },
        }
        return true;
    }

    fn prepareSilu(self: *Op, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        const N = sh.n;
        const n = sh.rows * N;
        self.v = try a.alloc(f64, n);
        self.x = try a.alloc(f64, n);
        for (self.v, self.x) |*gv, *uv| {
            gv.* = mu.uni(rng, f16, -6, 6);
            uv.* = mu.uni(rng, f16, -6, 6);
        }
        if (fill == .nonfinite) {
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }; // -inf: -inf / inf, NaN
            for (put, 0..) |x, i| if (i * N + i + 1 < n) {
                self.v[i * N + i + 1] = x;
            };
            if (3 * N + 4 < n) self.x[3 * N + 4] = std.math.inf(f64);
        }
        self.dev[0] = try mu.put(&self.rig, ga, f16, self.v);
        self.dev[1] = try mu.put(&self.rig, ga, f16, self.x);
        self.dev[3] = try self.rig.output(n * 2);
        return true;
    }

    /// Slots y [rows, K, D] and weights [rows, K] in fp32 (stress: small integers, exact in any sum).
    fn prepareWsum(self: *Op, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        self.v = try a.alloc(f64, sh.rows * sh.k * sh.n);
        self.w = try a.alloc(f64, sh.rows * sh.k);
        const st = fill == .stress;
        for (self.v) |*y| y.* = if (st) mu.int(rng, -64, 64) else mu.uni(rng, f32, -4, 4);
        for (self.w) |*w| w.* = if (st) mu.int(rng, 0, 4) else mu.uni(rng, f32, 0, 1);
        if (fill == .nonfinite) {
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (put, 0..) |x, i| if ((i * sh.k + i) * sh.n + 3 * i < self.v.len) {
                self.v[(i * sh.k + i) * sh.n + 3 * i] = x;
            };
            if (sh.rows > 3) self.w[3 * sh.k + 2] = std.math.nan(f64);
        }
        self.dev[0] = try mu.put(&self.rig, ga, f32, self.v);
        self.dev[1] = try mu.put(&self.rig, ga, f32, self.w);
        self.dev[2] = try self.rig.output(sh.rows * sh.n * 4);
        return true;
    }

    /// Dense's rows snapped to fp16 values (bf16 ones under 2^-14 to 0), fp16 weights; gate 4's mirrored integers.
    fn prepareRouter(self: *Op, fill: g.Fill, seed: u64, rng: std.Random) !bool {
        const sh = self.shape;
        if (sh.k % 2 != 0) return false;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        const d = try dense.Dense.init(ga, sh, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        for (d.x) |*b| if (@abs(c.fromBf16(b.*)) < 0x1p-14) {
            b.* = 0;
        };
        const k = sh.k;
        for (0..sh.n) |j| for (0..k) |i| {
            d.w[j * k + i] = if (fill != .stress) mu.uni(rng, f16, -1, 1) else if (i < k / 2) mu.int(rng, 1, 6) else -d.w[j * k + i - k / 2];
        };
        const xs = try a.alloc(f64, d.x.len);
        for (xs, d.x) |*o, b| o.* = c.fromBf16(b);
        self.dev[0] = try mu.put(&self.rig, ga, f16, xs);
        self.dev[1] = try mu.put(&self.rig, ga, f16, d.w);
        self.dev[2] = try self.rig.output(sh.rows * sh.n * 4);
        return true;
    }

    fn prepareTopk(self: *Op, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        if (sh.n > 1024 or sh.k != top) return false;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        self.v = try a.alloc(f64, sh.rows * sh.n);
        self.w = try a.alloc(f64, sh.n);
        for (self.v) |*v| v.* = mu.int(rng, -64, 64) / 8; // eighths: some ties
        for (self.w) |*v| v.* = mu.int(rng, 0, 7) / 64;
        if (fill == .nonfinite) {
            self.v[3] = std.math.nan(f64); // never picked; hides 7, 11, ...
            if (sh.rows > 1) self.v[sh.n] = std.math.nan(f64); // expert 0: the root keeps it
            if (sh.rows > 2) self.v[2 * sh.n + 5] = std.math.inf(f64);
            if (sh.rows > 3) self.v[3 * sh.n + 1] = -std.math.inf(f64);
        }
        self.dev[0] = try mu.put(&self.rig, ga, f32, self.v);
        self.dev[1] = try mu.put(&self.rig, ga, f32, self.w);
        self.dev[2] = try self.rig.output(sh.rows * top * 4);
        self.dev[3] = try self.rig.output(sh.rows * top * 4);
        return true;
    }

    /// ops.zig's launches: rms, silu and route (the router, then the top K).
    pub fn run(self: *Op) !void {
        const s = self.rig.s;
        const sh = self.shape;
        const d = self.dev;
        var a: cuda.Args = .{};
        var grid: [2]u32 = .{ @intCast(sh.rows), 1 };
        var block: u32 = NORM;
        var shared: u32 = 0;
        switch (self.kind) {
            .rms, .rms_res, .rms_sum => {
                if (self.kind != .rms) try mu.copy(&self.rig, d[4], d[1], sh.rows * sh.k * 4);
                a.add(d[0]);
                a.add(if (self.kind == .rms) @as(u64, 0) else d[4]);
                a.add(d[2]);
                a.add(d[3]);
                a.add(@as(c_int, @intCast(sh.k)));
                a.add(eps);
            },
            .silu => {
                inline for (.{ d[0], d[1], d[3] }) |p| a.add(p);
                a.add(@as(c_int, @intCast(sh.n)));
                grid[1] = @intCast((sh.n + 255) / 256);
            },
            .wsum => {
                inline for (.{ d[0], d[1], d[2] }) |p| a.add(p);
                inline for (.{ sh.k, sh.n }) |v| a.add(@as(c_int, @intCast(v)));
                grid[1] = @intCast((sh.n + 255) / 256);
            },
            .router => {
                inline for (.{ d[0], d[1], d[2] }) |p| a.add(p);
                inline for (.{ sh.k, sh.n }) |v| a.add(@as(c_int, @intCast(v)));
                grid = .{ @intCast((sh.n + 7) / 8), @intCast(sh.rows) };
                shared = @intCast(sh.k * 4);
            },
            .topk, .topk_div => {
                inline for (.{ d[0], d[1] }) |p| a.add(p);
                inline for (.{ sh.n, top }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(self.scale());
                inline for (.{ d[2], d[3] }) |p| a.add(p);
                block = @intCast(std.math.ceilPowerOfTwo(usize, @max(sh.n, 32)) catch unreachable);
            },
        }
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = 1 }, .block = .{ .x = block }, .shared = shared }, s, &a);
        try s.synchronize();
    }

    fn scale(self: *const Op) f32 {
        return if (self.kind == .topk_div) div else 1;
    }

    pub fn count(self: *Op) usize {
        const sh = self.shape;
        return switch (self.kind) {
            .rms, .rms_res, .rms_sum => sh.rows * sh.k,
            .silu, .wsum, .router => sh.rows * sh.n,
            .topk, .topk_div => sh.rows * top * 8,
        };
    }

    pub fn reference(self: *Op, want: []f64, mag: []f64) !void {
        switch (self.kind) {
            .rms, .rms_res => return self.rmsRows(.none, .sequential, want, mag),
            .rms_sum => for (want, mag, self.res, self.x) |*w, *m, r, x| {
                w.* = r + x;
                m.* = 0;
            },
            .silu => for (want, mag, self.v, self.x) |*w, *m, gv, uv| {
                w.* = gv / (1 + @exp(-gv)) * uv;
                m.* = 0;
            },
            .wsum => return self.slotSums(.none, .sequential, want, mag),
            .router => return self.d.?.reference(want, mag),
            .topk, .topk_div => return error.StatedRule, // gates 1 and 3 run check; no sums for gate 4
        }
    }

    pub fn simulate(self: *Op, acc: c.Accum, order: g.Order, out: []f64) !void {
        return switch (self.kind) {
            .rms, .rms_res => self.rmsRows(acc, order, out, null),
            .router => self.d.?.simulate(acc, order, out),
            .wsum => self.slotSums(acc, order, out, null),
            else => error.NoSums,
        };
    }

    /// v * rsqrt(mean(v^2) + eps) * w with the squares summed in `acc` (none: float64); y ~ ss^-1/2 for the mag.
    fn rmsRows(self: *Op, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) !void {
        const D = self.shape.k;
        const t = try self.arena.allocator().alloc(f64, D);
        for (0..self.shape.rows) |r| {
            const v = self.v[r * D ..][0..D];
            for (t, v) |*o, x| o.* = x * x;
            var ss: f64 = 0;
            if (acc == .none) {
                for (t) |x| ss += x;
            } else ss = rows.sumIn(acc, order, t);
            const rs = 1 / @sqrt(ss / @as(f64, @floatFromInt(D)) + eps);
            for (0..D) |i| {
                out[r * D + i] = v[i] * rs * self.w[i];
                if (mag) |m| m[r * D + i] = @abs(out[r * D + i]) / 2;
            }
        }
    }

    /// Each row's sum over slots of w * y (in `acc`; none: float64), and the products' magnitudes for the bound.
    fn slotSums(self: *Op, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) !void {
        const sh = self.shape;
        const t = try self.arena.allocator().alloc(f64, sh.k);
        for (0..sh.rows) |r| for (0..sh.n) |d| {
            var m: f64 = 0;
            for (t, 0..) |*o, k| {
                o.* = self.w[r * sh.k + k] * self.v[(r * sh.k + k) * sh.n + d];
                m += @abs(o.*);
            }
            var sum: f64 = 0;
            if (acc == .none) {
                for (t) |x| sum += x;
            } else sum = rows.sumIn(acc, order, t);
            out[r * sh.n + d] = sum;
            if (mag) |mg| mg[r * sh.n + d] = m;
        };
    }

    /// Expert e's sigmoid and rank score under the stated NaN rule (expert 0 first; a NaN node hides its leaves).
    fn score(self: *const Op, r: usize, e: usize) [2]f64 {
        const row = self.v[r * self.shape.n ..][0..self.shape.n];
        const s = 1 / (1 + @exp(-row[e]));
        const v = s + self.w[e];
        if (std.math.isNan(v)) return .{ s, if (e == 0) std.math.inf(f64) else -std.math.inf(f64) };
        for (row[1..], 1..) |x, j| if (std.math.isNan(x) and j != e) {
            const span = std.math.ceilPowerOfTwo(usize, j + 1) catch unreachable;
            if (e % span == j % span) return .{ s, -std.math.inf(f64) };
        };
        return .{ s, v };
    }

    pub fn check(self: *Op, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const n = self.shape.n;
        const ids = try a.dupe(u8, try self.rig.bytes(self.dev[2], self.shape.rows * top * 4));
        defer a.free(ids);
        const wts = try self.rig.bytes(self.dev[3], self.shape.rows * top * 4);
        for (0..self.shape.rows) |r| {
            var taken = std.StaticBitSet(1024).empty;
            var pick: [top]usize = undefined;
            for (&pick, 0..) |*p, k| {
                const id = std.mem.readInt(i32, ids[(r * top + k) * 4 ..][0..4], .little);
                if (id < 0 or id >= n or taken.isSet(@intCast(id))) return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: expert {d} out of range or picked twice", .{ r, k, id });
                p.* = @intCast(id);
                taken.set(p.*);
            }
            var lowest: f64 = std.math.inf(f64);
            var sum: f64 = 0;
            for (pick, 0..) |p, k| {
                const v = self.score(r, p)[1];
                lowest = @min(lowest, v);
                sum += self.score(r, p)[0];
                if (k == 0) continue;
                const prev = self.score(r, pick[k - 1])[1];
                if (v > prev + 0x1p-20 * (@abs(prev) + 1) or (v == prev and p < pick[k - 1]))
                    return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: expert {d} ({d:.6}) after {d} ({d:.6})", .{ r, k, p, v, pick[k - 1], prev });
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
                const want = self.score(r, p)[0] / (sum + 1e-20) * self.scale();
                const ok = if (std.math.isFinite(want)) @abs(w - want) <= 0x1p-20 * @abs(want) + 1e-30 else c.classOf(w) == c.classOf(want);
                if (!ok) return try std.fmt.bufPrint(&self.msg, "row {d} slot {d}: weight {d:.7} where float64 gives {d:.7}", .{ r, k, w, want });
            }
        }
        return null;
    }
};
