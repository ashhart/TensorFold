//! Muse Spark's DFlash2 ops under the contract (dflash2.cu): dense rows, causal convs, SiLU-mul, top 16, the walk.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mu = @import("contract_muse_util.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const dense = core.contract.dense;

const B = 8; // block_size
const KT = 2; // conv_kernel_size
const GS = 16; // conv_group_size
const TOPK = 16;
const P = B - 1; // drafted rows a stream
const V = 512; // the fixture's vocabulary (codebook rows)
const none_id: i32 = 0x7fffffff;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A lane's K / 32 products in order, then the warp by halves.
fn laneChain(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

/// KT products chained, after w = base + dyn (and y = y0 + y1, then the residual's add).
fn prepDepth(_: c.Shape) usize {
    return KT + 1;
}

fn finishDepth(_: c.Shape) usize {
    return KT + 3;
}

pub const Kind = enum { dense_f32, dense_f16, conv_prep, conv_finish, silu, topk, walk };

const topk_rule = "a row's top 16 valid columns by value, ties to the lower global id, a NaN never a candidate; missing " ++
    "places -inf with id 0x7fffffff; values and ids exact";
const walk_rule = "per stream, each row's 16 best of the ranks' merged lists (valid ids only, ties to the lower id), " ++
    "scored v + sum_k pcb[prev, k] g[k] scb[c, k]; the first maximum is within fp32's sum bound of float64's first maximum " ++
    "(a NaN score never displaces candidate 0); no valid candidate repeats the previous token";

pub fn spec(kind: Kind) c.Spec {
    const dn = &[_]c.Shape{ .{ .rows = 56, .n = 2048, .k = 8192 }, .{ .rows = 7, .n = 256, .k = 8192 } };
    const conv = &[_]c.Shape{ .{ .rows = 4, .n = 8192, .k = KT }, .{ .rows = 1, .n = 8192, .k = KT } };
    return switch (kind) {
        .dense_f32 => .{ .name = "cuda muse dfl_dense (bf16 weights, fp32 out)", .accum = .f32, .depth = laneChain, .out = .f32, .tiles = .{ .{ .size = 8 }, .{ .size = 8 }, .{ .size = 32 } }, .served = dn },
        .dense_f16 => .{ .name = "cuda muse dfl_dense (bf16 weights, fp16 out)", .accum = .f32, .depth = laneChain, .out = .f16, .tiles = .{ .{ .size = 8 }, .{ .size = 8 }, .{ .size = 32 } }, .served = dn },
        .conv_prep => .{ .name = "cuda muse dfl_conv_prep (grouped dynamic causal conv in, fp16)", .accum = .f32, .depth = prepDepth, .out = .f16, .tiles = .{ .{}, .{ .size = 256, .step = GS, .min = GS }, .{} }, .served = conv },
        .conv_finish => .{ .name = "cuda muse dfl_conv_finish (into the residual, in place)", .accum = .f32, .depth = finishDepth, .out = .f32, .in_place = true, .tiles = .{ .{}, .{ .size = 256, .step = GS, .min = GS }, .{} }, .served = conv },
        .silu => .{ .name = "cuda muse dfl_silu_mul (fp16 interm)", .accum = .none, .depth = noSums, .ops = 8, .out = .f16, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{ .{ .rows = 56, .n = 14336, .k = 1 }, .{ .rows = 7, .n = 14336, .k = 1 } } },
        .topk => .{ .name = "cuda muse dfl_topk (a rank's top 16 of its head slice)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = topk_rule }, .tiles = .{ .{}, .{ .size = 256 }, .{} }, .served = &.{ .{ .rows = 28, .n = 101120, .k = 101120 }, .{ .rows = 7, .n = 101120, .k = 100928 } } },
        .walk => .{ .name = "cuda muse dfl_walk (the candidate selector, greedy)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = walk_rule }, .tiles = .{ .{}, .{}, .{ .size = 32, .step = 32, .min = 32 } }, .served = &.{ .{ .rows = 4, .n = 2, .k = 256 }, .{ .rows = 1, .n = 2, .k = 256 } } },
    };
}

pub const Sel = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    d: ?dense.Dense = null,
    in: [6][]f64 = @splat(&.{}),
    ids: []i32 = &.{},
    anchors: []i64 = &.{},
    dev: [9]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Sel) void {
        if (self.d) |d| d.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    const vt_ref = blk: {
        var v = g.hooks(Sel);
        v.check = null;
        break :blk v;
    };
    const vt_rule = g.hooks(Sel);

    pub fn kernel(self: *Sel) g.Kernel {
        const ruled = self.kind == .topk or self.kind == .walk;
        return .{ .spec = spec(self.kind), .ctx = self, .vt = if (ruled) &vt_rule else &vt_ref };
    }

    fn rnd(self: *Sel, rng: std.Random, comptime T: type, n: usize, lo: f64, hi: f64) ![]f64 {
        const x = try self.arena.allocator().alloc(f64, n);
        for (x) |*v| v.* = mu.uni(rng, T, lo, hi);
        return x;
    }

    fn convRows(self: *const Sel) usize {
        return self.shape.rows * B;
    }

    pub fn prepare(self: *Sel, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
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
        const st = fill == .stress;
        const nan = std.math.nan(f64);
        const inf = std.math.inf(f64);
        switch (self.kind) {
            .dense_f32, .dense_f16 => {
                const d = try dense.Dense.init(ga, shape, 1);
                self.d = d;
                d.fillX(fill, seed, 1);
                for (d.x) |*b| if (@abs(c.fromBf16(b.*)) < 0x1p-14) { // fp16 holds the rest exactly
                    b.* = 0;
                };
                const k = shape.k;
                for (0..shape.n) |j| for (0..k) |i| {
                    d.w[j * k + i] = if (!st) mu.uni(rng, u16, -1, 1) else if (i < k / 2) mu.int(rng, 1, 6) else -d.w[j * k + i - k / 2];
                };
                if (st and k % 2 == 1) for (0..shape.n) |j| {
                    d.w[j * k + k - 1] = 0;
                };
                const xs = try a.alloc(f64, d.x.len);
                for (xs, d.x) |*o, b| o.* = c.fromBf16(b);
                self.dev[0] = try mu.put(&self.rig, ga, f16, xs);
                self.dev[1] = try mu.put(&self.rig, ga, u16, d.w);
                self.dev[2] = try self.rig.output(shape.rows * shape.n * (if (self.kind == .dense_f32) @as(usize, 4) else 2));
            },
            .conv_prep, .conv_finish => {
                if (shape.k != KT or shape.n % GS != 0) return false;
                const R = self.convRows();
                const D = shape.n;
                const G = D / GS;
                self.in[0] = try a.alloc(f64, 2 * KT * D); // base (prep), then base1 (finish)
                self.in[1] = try a.alloc(f64, R * 2 * KT * G); // dyn
                for (self.in[0], 0..) |*v, i| v.* = if (st) 512 + @as(f64, @floatFromInt(i / D % KT)) else mu.uni(rng, f32, -1, 1);
                for (self.in[1], 0..) |*v, i| v.* = if (!st) mu.uni(rng, f16, -0.5, 0.5) else if (i / G % KT == 0) mu.int(rng, 0, 3) else self.in[1][i - G];
                const h0 = try a.alloc(f64, D);
                for (h0) |*v| v.* = mu.int(rng, 33, 63);
                // h (prep, fp16) or y0, y1 (finish, fp32) and res: gate 4's rows alternate sign within a block
                self.in[2] = if (self.kind == .conv_prep) try self.rnd(rng, f16, R * D, -4, 4) else try self.rnd(rng, f32, R * D, -4, 4);
                self.in[3] = try self.rnd(rng, f32, R * D, -4, 4);
                self.in[4] = try self.rnd(rng, f32, R * D, -64, 64);
                if (st) for (0..R) |r| for (0..D) |i| {
                    const y = if (r % B % 2 == 0) h0[i] else -h0[i];
                    self.in[3][r * D + i] = mu.int(rng, -8, 8);
                    self.in[2][r * D + i] = if (self.kind == .conv_prep) y else y - self.in[3][r * D + i];
                    self.in[4][r * D + i] = mu.int(rng, -64, 64);
                };
                if (nf) {
                    self.in[2][3] = nan; // rows 0 and 1, column 3
                    self.in[1][2 * 2 * KT * G + 1] = inf; // row 2's first tap, group 1
                    self.in[0][(if (self.kind == .conv_prep) 0 else KT * D) + D + 7] = -inf; // the second tap, column 7
                }
                self.dev[0] = try mu.put(&self.rig, ga, f32, self.in[0]);
                self.dev[1] = try mu.put(&self.rig, ga, f16, self.in[1]);
                if (self.kind == .conv_prep) {
                    self.dev[2] = try mu.put(&self.rig, ga, f16, self.in[2]);
                    self.dev[3] = try self.rig.output(R * D * 2);
                } else {
                    for (2..5) |i| self.dev[i] = try mu.put(&self.rig, ga, f32, self.in[i]);
                    self.dev[5] = try self.rig.output(R * D * 4);
                }
            },
            .silu => {
                const n = shape.rows * shape.n;
                self.in[0] = try self.rnd(rng, f16, n, -8, 8);
                self.in[1] = try self.rnd(rng, f16, n, -4, 4);
                if (nf) {
                    self.in[0][1] = nan;
                    self.in[0][2] = inf;
                    self.in[0][3] = -inf; // -inf / (1 + inf): NaN, as float64 gives it
                    self.in[1][4] = inf;
                }
                self.dev[0] = try mu.put(&self.rig, ga, f16, self.in[0]);
                self.dev[1] = try mu.put(&self.rig, ga, f16, self.in[1]);
                self.dev[2] = try self.rig.output(n * 2);
            },
            .topk => {
                self.in[0] = try a.alloc(f64, shape.rows * shape.n);
                for (self.in[0]) |*v| v.* = mu.int(rng, -2048, 2048) / 32; // ties
                if (nf) {
                    self.in[0][0] = nan;
                    if (shape.rows > 1) self.in[0][shape.n + 3] = inf;
                    if (shape.rows > 2) for (self.in[0][2 * shape.n ..][0..shape.n]) |*v| {
                        v.* = if (rng.boolean()) nan else -inf; // fewer than 16 numbers: -inf ids, then the empty places
                    };
                }
                self.dev[0] = try mu.put(&self.rig, ga, f32, self.in[0]);
                self.dev[1] = try self.rig.output(shape.rows * 2 * TOPK * 4);
            },
            .walk => return self.prepareWalk(fill, rng),
        }
        return true;
    }

    /// recv [ranks][rows][32] (each rank's top 16 values, then ids as bits), gate [rows, R], codebooks [V, R].
    fn prepareWalk(self: *Sel, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        if (sh.k % 32 != 0 or sh.k > 1024 or sh.n == 0 or sh.n > 4) return false;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        const n1 = sh.rows * P;
        const R = sh.k;
        self.in[0] = try a.alloc(f64, sh.n * n1 * TOPK);
        self.ids = try a.alloc(i32, sh.n * n1 * TOPK);
        const span = V / sh.n;
        for (0..sh.n) |q| for (0..n1) |r| {
            var top: f64 = mu.int(rng, 0, 64) / 4;
            for (0..TOPK) |j| {
                const at = (q * n1 + r) * TOPK + j;
                self.in[0][at] = top;
                top -= mu.int(rng, 0, 4) / 8; // descending, some ties
                self.ids[at] = @intCast(q * span + (r * 37 + j * 11) % span); // each rank's own slice of ids
            }
        };
        self.in[1] = try self.rnd(rng, f16, n1 * R, -1, 1);
        self.in[2] = try self.rnd(rng, u16, V * R, -1, 1);
        self.in[3] = try self.rnd(rng, u16, V * R, -1, 1);
        if (fill == .nonfinite) {
            self.in[1][5] = std.math.nan(f64); // row 0: every score NaN, candidate 0 kept
            if (n1 > 1) self.in[0][(sh.n - 1) * n1 * TOPK + TOPK + 3] = std.math.inf(f64); // row 1: one candidate +inf
            if (n1 > 2) for (0..sh.n) |q| for (0..TOPK) |j| {
                self.ids[(q * n1 + 2) * TOPK + j] = none_id; // row 2: no candidate
            };
        }
        self.anchors = try a.alloc(i64, sh.rows);
        for (self.anchors) |*x| x.* = rng.uintLessThan(u32, V);
        const recv = try a.alloc(f32, sh.n * n1 * 2 * TOPK);
        for (0..sh.n * n1) |r| for (0..TOPK) |j| {
            recv[r * 2 * TOPK + j] = @floatCast(self.in[0][r * TOPK + j]);
            recv[r * 2 * TOPK + TOPK + j] = @bitCast(self.ids[r * TOPK + j]);
        };
        self.dev[0] = try mu.ints(&self.rig, f32, recv);
        self.dev[1] = try mu.put(&self.rig, ga, f16, self.in[1]);
        self.dev[2] = try mu.put(&self.rig, ga, u16, self.in[2]);
        self.dev[3] = try mu.put(&self.rig, ga, u16, self.in[3]);
        self.dev[4] = try mu.ints(&self.rig, i64, self.anchors);
        self.dev[5] = try self.rig.output(n1 * 4);
        return true;
    }

    /// dflash2.zig's launches.
    pub fn run(self: *Sel) !void {
        const s = self.rig.s;
        const sh = self.shape;
        const d = self.dev;
        var a: cuda.Args = .{};
        var grid: [2]u32 = .{ @intCast(sh.rows), 1 };
        var block: u32 = 256;
        switch (self.kind) {
            .dense_f32, .dense_f16 => {
                inline for (.{ d[0], d[1] }) |p| a.add(p);
                a.add(@as(c_int, 1));
                a.add(d[2]);
                a.add(@as(c_int, @intFromBool(self.kind == .dense_f32)));
                inline for (.{ sh.rows, sh.k, sh.n }) |v| a.add(@as(c_int, @intCast(v)));
                grid = .{ @intCast((sh.n + 7) / 8), @intCast((sh.rows + 7) / 8) };
            },
            .conv_prep => {
                inline for (.{ d[2], d[1], d[0], d[3] }) |p| a.add(p);
                inline for (.{ B, sh.n, KT, GS }) |v| a.add(@as(c_int, @intCast(v)));
                grid[0] = @intCast(self.convRows());
            },
            .conv_finish => {
                try mu.copy(&self.rig, d[5], d[4], self.convRows() * sh.n * 4);
                inline for (.{ d[5], d[2], d[3], d[1], d[0] + KT * sh.n * 4 }) |p| a.add(p);
                inline for (.{ B, sh.n, KT, GS }) |v| a.add(@as(c_int, @intCast(v)));
                grid[0] = @intCast(self.convRows());
            },
            .silu => {
                const n = sh.rows * sh.n;
                inline for (.{ d[0], d[1], d[2] }) |p| a.add(p);
                a.add(@as(i64, @intCast(n)));
                grid[0] = @intCast(@min((n + 255) / 256, 4096));
            },
            .topk => {
                a.add(d[0]);
                const lo: usize = if (sh.k < sh.n) sh.n else 0; // rank 1's slice: its global ids from lo
                inline for (.{ sh.n, @min(sh.k, sh.n), lo }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(d[1]);
            },
            .walk => {
                a.add(d[0]);
                inline for (.{ sh.n, sh.rows * P }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(d[1]);
                a.add(@as(c_int, @intCast(sh.k)));
                inline for (.{ d[2], d[3] }) |p| a.add(p);
                a.add(@as(c_int, 1));
                a.add(d[4]);
                a.add(@as(c_int, P));
                a.add(d[5]);
                block = 512;
            },
        }
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = 1 }, .block = .{ .x = block } }, s, &a);
        try s.synchronize();
    }

    pub fn count(self: *Sel) usize {
        const sh = self.shape;
        return switch (self.kind) {
            .dense_f32, .dense_f16, .silu => sh.rows * sh.n,
            .conv_prep, .conv_finish => self.convRows() * sh.n,
            .topk => sh.rows * 2 * TOPK * 4,
            .walk => sh.rows * P * 4,
        };
    }

    pub fn reference(self: *Sel, want: []f64, mag: []f64) !void {
        switch (self.kind) {
            .dense_f32, .dense_f16 => return self.d.?.reference(want, mag),
            .conv_prep, .conv_finish => return self.conv(.none, .sequential, want, mag),
            .silu => for (want, mag, self.in[0], self.in[1]) |*w, *m, x, uu| {
                w.* = x / (1 + @exp(-x)) * uu;
                m.* = 0;
            },
            .topk, .walk => return error.StatedRule, // gates 1 and 3 run check; no sums for gate 4
        }
    }

    pub fn simulate(self: *Sel, acc: c.Accum, order: g.Order, out: []f64) !void {
        return switch (self.kind) {
            .dense_f32, .dense_f16 => self.d.?.simulate(acc, order, out),
            .conv_prep, .conv_finish => self.conv(acc, order, out, null),
            else => error.NoSums,
        };
    }

    /// Each output's KT terms (and the residual, finish) summed in `acc`; their magnitude into mag.
    fn conv(self: *Sel, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) !void {
        const D = self.shape.n;
        const G = D / GS;
        const fin = self.kind == .conv_finish;
        const base = self.in[0][if (fin) KT * D else 0..];
        var t: [KT + 1]f64 = undefined;
        for (0..self.convRows()) |r| for (0..D) |col| {
            const b = r % B;
            var n: usize = 0;
            for (0..@min(KT, b + 1)) |tt| {
                const w = base[tt * D + col] + self.in[1][r * 2 * KT * G + (if (fin) KT * G else 0) + tt * G + col / GS];
                const at = (r - tt) * D + col;
                const y = if (fin) self.in[2][at] + self.in[3][at] else self.in[2][at];
                t[n] = w * y;
                n += 1;
            }
            const at = r * D + col;
            var s = if (acc == .none) blk: {
                var x: f64 = 0;
                for (t[0..n]) |v| x += v;
                break :blk x;
            } else rows.sumIn(acc, order, t[0..n]);
            var m: f64 = 0;
            for (t[0..n]) |v| m += @abs(v);
            if (fin) {
                s += self.in[4][at];
                m += @abs(self.in[4][at]);
            }
            out[at] = s;
            if (mag) |x| x[at] = m;
        };
    }

    pub fn check(self: *Sel, _: g.Gate) !?[]const u8 {
        return if (self.kind == .topk) self.checkTopk() else self.checkWalk();
    }

    fn beats(va: f64, ia: i64, vb: f64, ib: i64) bool {
        return va > vb or (va == vb and ia < ib);
    }

    fn checkTopk(self: *Sel) !?[]const u8 {
        const sh = self.shape;
        const got = try self.rig.bytes(self.dev[1], sh.rows * 2 * TOPK * 4);
        const lo: i64 = if (sh.k < sh.n) @intCast(sh.n) else 0;
        for (0..sh.rows) |r| {
            var v: [TOPK]f64 = @splat(-std.math.inf(f64));
            var id: [TOPK]i64 = @splat(none_id);
            for (self.in[0][r * sh.n ..][0..@min(sh.k, sh.n)], 0..) |x, col| {
                const gid = lo + @as(i64, @intCast(col));
                if (std.math.isNan(x) or !beats(x, gid, v[TOPK - 1], id[TOPK - 1])) continue;
                var k: usize = TOPK - 1;
                while (k > 0 and beats(x, gid, v[k - 1], id[k - 1])) : (k -= 1) {
                    v[k] = v[k - 1];
                    id[k] = id[k - 1];
                }
                v[k] = x;
                id[k] = gid;
            }
            for (0..TOPK) |k| {
                const gv: f64 = @as(f32, @bitCast(std.mem.readInt(u32, got[(r * 2 * TOPK + k) * 4 ..][0..4], .little)));
                const gi = std.mem.readInt(i32, got[(r * 2 * TOPK + TOPK + k) * 4 ..][0..4], .little);
                if (gv != v[k] or gi != id[k]) return try std.fmt.bufPrint(&self.msg, "row {d} place {d}: ({d}, {d}) where the rule gives ({d}, {d})", .{ r, k, gv, gi, v[k], id[k] });
            }
        }
        return null;
    }

    fn checkWalk(self: *Sel) !?[]const u8 {
        const sh = self.shape;
        const n1 = sh.rows * P;
        const R = sh.k;
        const got = try self.rig.bytes(self.dev[5], n1 * 4);
        for (0..sh.rows) |s| {
            var pred: i64 = self.anchors[s];
            for (0..P) |i| {
                const row = s * P + i;
                var cv: [TOPK]f64 = undefined;
                var cid: [TOPK]i64 = @splat(-1);
                var used = std.StaticBitSet(64).empty;
                for (0..TOPK) |k| {
                    var best: ?usize = null;
                    for (0..sh.n * TOPK) |j| {
                        const at = ((j / TOPK) * n1 + row) * TOPK + j % TOPK;
                        const id = self.ids[at];
                        if (used.isSet(j) or id < 0 or id == none_id) continue;
                        if (best == null) best = j else {
                            const bt = ((best.? / TOPK) * n1 + row) * TOPK + best.? % TOPK;
                            if (beats(self.in[0][at], id, self.in[0][bt], self.ids[bt])) best = j;
                        }
                    }
                    cv[k] = -std.math.inf(f64);
                    if (best) |j| {
                        const at = ((j / TOPK) * n1 + row) * TOPK + j % TOPK;
                        used.set(j);
                        cv[k] = self.in[0][at];
                        cid[k] = self.ids[at];
                    }
                }
                var sc: [TOPK]f64 = undefined;
                var tol: [TOPK]f64 = undefined;
                for (0..TOPK) |k| {
                    sc[k] = -std.math.inf(f64);
                    tol[k] = 0;
                    if (cid[k] < 0) continue;
                    var acc: f64 = 0;
                    var mag: f64 = 0;
                    for (0..R) |j| {
                        const t = self.in[2][@as(usize, @intCast(pred)) * R + j] * self.in[1][row * R + j] * self.in[3][@as(usize, @intCast(cid[k])) * R + j];
                        acc += t;
                        mag += @abs(t);
                    }
                    sc[k] = cv[k] + acc;
                    tol[k] = mu.gamma(@floatFromInt(R / 32 + 8)) * mag + mu.u * @abs(cv[k]);
                }
                var best: usize = 0;
                for (1..TOPK) |k| if (sc[k] > sc[best]) {
                    best = k;
                };
                const out: i64 = std.mem.readInt(i32, got[row * 4 ..][0..4], .little);
                const want = if (cid[best] >= 0) cid[best] else pred;
                if (out != want) {
                    const k = for (0..TOPK) |x| {
                        if (cid[x] == out) break x;
                    } else return try std.fmt.bufPrint(&self.msg, "stream {d} row {d}: {d} is no candidate (rule: {d})", .{ s, i, out, want });
                    if (!(sc[k] >= sc[best] - tol[k] - tol[best])) return try std.fmt.bufPrint(&self.msg, "stream {d} row {d}: {d} scores {d:.6} under the best {d:.6}", .{ s, i, out, sc[k], sc[best] });
                }
                pred = out;
            }
        }
        return null;
    }
};
