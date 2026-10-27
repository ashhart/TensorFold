//! MiMo-V2.6's DFlash drafter glue under the contract (dflash.cu): moves, head norms + RoPE, window attention, argmax.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mu = @import("contract_mimo_util.zig");
const attn_k = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;

const HD = 128;
const ROT = 64; // head_dim * partial_rotary_factor 0.5
const B = 8; // block_size
const W = 1024; // sliding_window
const eps: f32 = 1e-6; // rms_norm_eps
const scale: f32 = 0.08838834764831845; // 128^-0.5
const theta: f64 = 10000; // rope_theta

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A head's 128 squares: 32 lanes by halves, then 4 warps in order.
fn normDepth(_: c.Shape) usize {
    return 1 + 5 + 4;
}

/// A score's 128 products times the scale, or a query's visible keys chained for l and o (and the sink).
fn attnDepth(_: c.Shape) usize {
    return W + B + 1;
}

pub const Kind = enum { embed, tap, store, take, norm_q, norm_k, attn, argmax };

const half_why = "fp16 outputs after ss^-1/2: a bf16 sum's error is a few output ulps";
const attn_rule = "NaN and infinities in K and V reach the outputs as float64's two-pass softmax gives them " ++
    "(planted where no NaN enters a tile's maximum)";
const argmax_rule = "a NaN never wins: the first maximum of the other columns up to the vocabulary (all NaN: 2^31 - 1)";

pub fn spec(kind: Kind) c.Spec {
    const moves = &[_]c.Shape{ .{ .rows = 4, .n = 4096, .k = B }, .{ .rows = 1, .n = 4096, .k = B } };
    const t256: [3]c.Tile = .{ .{}, .{ .size = 256 }, .{} };
    return switch (kind) {
        .embed => .{ .name = "cuda mimo dr_block_embed (anchor rows bf16, mask rows fp32)", .accum = .none, .depth = noSums, .out = .f32, .tiles = t256, .served = moves },
        .tap => .{ .name = "cuda mimo dr_tap (fp32 residuals to fp16 tap blocks)", .accum = .none, .depth = noSums, .out = .f16, .tiles = t256, .served = &.{ .{ .rows = 128, .n = 4096, .k = 5 }, .{ .rows = 1, .n = 4096, .k = 5 } } },
        .store => .{ .name = "cuda mimo dr_store (v rows into the ring)", .accum = .none, .depth = noSums, .out = .f16, .tiles = .{ .{}, .{ .size = 2, .min = 1 }, .{} }, .served = &.{ .{ .rows = 32, .n = 8, .k = HD }, .{ .rows = 128, .n = 8, .k = HD } } },
        .take => .{ .name = "cuda mimo dr_take (a block's mask rows)", .accum = .none, .depth = noSums, .out = .f16, .tiles = t256, .served = moves },
        .norm_q => .{ .name = "cuda mimo dr_norm_rope (q heads, RoPE on 64 of 128, fp32 out)", .accum = .f32, .depth = normDepth, .out = .f32, .served = &.{ .{ .rows = 32, .n = 64, .k = HD }, .{ .rows = 8, .n = 64, .k = HD } } },
        .norm_k => .{ .name = "cuda mimo dr_norm_rope (k heads into the ring, fp16)", .accum = .f32, .depth = normDepth, .out = .f16, .served = &.{ .{ .rows = 32, .n = 8, .k = HD }, .{ .rows = 128, .n = 8, .k = HD } }, .sums = .{ .bf16 = 0, .why = half_why } },
        .attn => .{ .name = "cuda mimo dr_attn (a block's queries over the window, then the sink)", .accum = .f32, .depth = attnDepth, .ops = 1, .out = .f16, .nan = .{ .stated = attn_rule }, .tiles = .{ .{}, .{ .size = 8, .step = 8, .min = 8 }, .{} }, .served = &.{ .{ .rows = 2, .n = 64, .k = 6000 }, .{ .rows = 1, .n = 64, .k = 700 }, .{ .rows = 4, .n = 64, .k = 20 } } },
        .argmax => .{ .name = "cuda mimo dr_argmax (the ranks' head slices up to the vocabulary)", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = argmax_rule }, .tiles = t256, .served = &.{ .{ .rows = 28, .n = 152576, .k = 2 }, .{ .rows = 7, .n = 152576, .k = 1 } } },
    };
}

pub const Draft = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    stress: bool = false,
    in: [3][]f64 = .{ &.{}, &.{}, &.{} },
    tab: [2][]f64 = .{ &.{}, &.{} },
    w: []f64 = &.{},
    pos: []i32 = &.{},
    ring: usize = 0,
    kv: usize = 0,
    dev: [8]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Draft) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    const vt_ref = blk: {
        var v = g.hooks(Draft);
        v.check = null;
        break :blk v;
    };
    const vt_rule = g.hooks(Draft);

    pub fn kernel(self: *Draft) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = if (self.kind == .argmax) &vt_rule else &vt_ref };
    }

    fn rnd(self: *Draft, rng: std.Random, comptime T: type, n: usize, lo: f64, hi: f64) ![]f64 {
        const x = try self.arena.allocator().alloc(f64, n);
        for (x) |*v| v.* = mu.uni(rng, T, lo, hi);
        return x;
    }

    pub fn prepare(self: *Draft, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.stress = fill == .stress;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows;
        const D = shape.n;
        const nan = std.math.nan(f64);
        const inf = std.math.inf(f64);
        const nf = fill == .nonfinite;
        switch (self.kind) {
            .embed => {
                self.in[0] = try self.rnd(rng, u16, n * D, -2, 2); // the anchors' embedding rows
                self.in[1] = try self.rnd(rng, f32, D, -1, 1); // mask_emb
                if (nf) {
                    self.in[0][1] = nan;
                    self.in[0][D + 2] = inf;
                    self.in[1][3] = -inf;
                }
                const toks = try a.alloc(i64, n);
                for (toks, 0..) |*t, s| t.* = @intCast(n - 1 - s);
                self.dev[0] = try mu.put(&self.rig, ga, u16, self.in[0]);
                self.dev[1] = try mu.put(&self.rig, ga, f32, self.in[1]);
                self.dev[2] = try mu.ints(&self.rig, i64, toks);
                self.dev[3] = try self.rig.output(n * shape.k * D * 4);
            },
            .tap => {
                self.in[0] = try self.rnd(rng, f32, shape.k * n * D, -1000, 1000); // each tap's residual rows
                if (nf) {
                    self.in[0][0] = nan;
                    self.in[0][D + 1] = inf;
                    self.in[0][n * D + 2] = -inf;
                }
                self.dev[0] = try mu.put(&self.rig, ga, f32, self.in[0]);
                self.dev[1] = try self.rig.output(shape.k * n * D * 2);
            },
            .store, .take => {
                if (self.kind == .store and shape.k != HD) return false;
                const len = if (self.kind == .store) n * shape.n * shape.k else n * B * D;
                self.in[0] = try self.rnd(rng, f16, len, -8, 8);
                if (nf) {
                    self.in[0][0] = nan;
                    self.in[0][len / 2 + 1] = inf;
                    self.in[0][len - 1] = -inf;
                }
                self.dev[0] = try mu.put(&self.rig, ga, f16, self.in[0]);
                if (self.kind == .store) {
                    self.pos = try a.alloc(i32, n);
                    const base = rng.uintLessThan(u32, 1 << 20);
                    for (self.pos, 0..) |*p, r| p.* = @intCast(base + r);
                    const slot = try a.alloc(i32, n);
                    @memset(slot, 0);
                    self.dev[1] = try mu.ints(&self.rig, i32, self.pos);
                    self.dev[2] = try mu.ints(&self.rig, i32, slot);
                    self.dev[3] = try self.rig.output(len * 2);
                } else self.dev[1] = try self.rig.output(n * (B - 1) * D * 2);
            },
            .norm_q, .norm_k => {
                if (shape.k != HD) return false;
                const nh = shape.n;
                self.in[0] = try a.alloc(f64, n * nh * HD);
                for (self.in[0]) |*v| v.* = if (self.stress) mu.int(rng, -200, 200) else mu.uni(rng, f16, -3, 3);
                self.w = try self.rnd(rng, f32, HD, 0.5, 1.5);
                if (nf) {
                    self.in[0][3] = nan;
                    self.in[0][HD + 7] = inf;
                    self.in[0][2 * HD + 70] = -inf;
                }
                self.pos = try a.alloc(i32, n);
                const base = rng.uintLessThan(u32, 8192);
                for (self.pos, 0..) |*p, r| p.* = @intCast(base + r);
                const tl = (base + n) * ROT / 2;
                for (&self.tab, 0..) |*t, i| {
                    t.* = try a.alloc(f64, tl);
                    for (t.*, 0..) |*o, j| {
                        const ang = @as(f64, @floatFromInt(j / 32)) / std.math.pow(f64, theta, @as(f64, @floatFromInt(j % 32)) / 32);
                        o.* = mu.f32r(if (i == 0) @cos(ang) else @sin(ang));
                    }
                }
                const slot = try a.alloc(i32, n);
                @memset(slot, 0);
                self.ring = n;
                self.dev[0] = try mu.put(&self.rig, ga, f16, self.in[0]);
                self.dev[1] = try mu.put(&self.rig, ga, f32, self.w);
                for (0..2) |i| self.dev[2 + i] = try mu.put(&self.rig, ga, f32, self.tab[i]);
                self.dev[4] = try mu.ints(&self.rig, i32, self.pos);
                self.dev[5] = try mu.ints(&self.rig, i32, slot);
                self.dev[6] = try self.rig.output(n * nh * HD * (if (self.kind == .norm_q) @as(usize, 4) else 2));
            },
            .attn => return self.prepareAttn(fill, rng),
            .argmax => return self.prepareArgmax(fill, rng),
        }
        return true;
    }

    /// Streams (rows) of B queries at positions p0 .. p0 + B - 1, p0 = k - 3 s; n query heads over n / 8 KV heads.
    fn prepareAttn(self: *Draft, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        if (sh.n % 8 != 0) return false;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        const S = sh.rows;
        self.kv = sh.n / 8;
        self.ring = W + 2 * B;
        if (sh.k < 3 * (S - 1)) return false;
        self.pos = try a.alloc(i32, S);
        for (self.pos, 0..) |*p, s| p.* = @intCast(sh.k - s * 3); // p0 per stream
        self.w = try a.alloc(f64, sh.n); // the heads' sink logits
        for (self.w) |*v| v.* = if (self.stress) -std.math.inf(f64) else mu.uni(rng, f32, -4, 4); // -inf adds 0
        const cache = S * self.ring * self.kv * HD;
        self.in[0] = try a.alloc(f64, S * B * sh.n * HD);
        for (self.in[0]) |*v| v.* = if (self.stress) 0 else mu.uni(rng, f32, -1, 1);
        self.in[1] = try self.rnd(rng, f16, cache, -1, 1);
        self.in[2] = try self.rnd(rng, f16, cache, -2, 2);
        if (self.stress) try self.mirrorV(rng);
        if (fill == .nonfinite) {
            const p0: usize = @intCast(self.pos[0]);
            self.in[2][self.cacheAt(0, p0, 0) + 3] = std.math.nan(f64);
            self.in[2][self.cacheAt(0, p0 + 2, 0) + 9] = std.math.inf(f64);
            self.in[2][self.cacheAt(0, p0 + 4, 1) + 1] = -std.math.inf(f64);
            if (S > 1) self.in[1][self.cacheAt(1, @as(usize, @intCast(self.pos[1])) + 1, 0) + 4] = std.math.inf(f64);
        }
        const slot = try a.alloc(i32, S);
        for (slot, 0..) |*s, i| s.* = @intCast(i);
        self.dev[0] = try mu.put(&self.rig, ga, f32, self.in[0]);
        self.dev[1] = try mu.put(&self.rig, ga, f16, self.in[1]);
        self.dev[2] = try mu.put(&self.rig, ga, f16, self.in[2]);
        self.dev[3] = try mu.ints(&self.rig, i32, self.pos);
        self.dev[4] = try mu.ints(&self.rig, i32, slot);
        self.dev[5] = try self.rig.output(S * B * sh.n * HD * 2);
        self.dev[6] = try mu.put(&self.rig, ga, f32, self.w);
        return true;
    }

    /// n = the vocabulary, k = the ranks: recv [k, rows, ceil(n / k)], +inf past the vocabulary (never read).
    fn prepareArgmax(self: *Draft, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        const n = sh.rows;
        const V = sh.n;
        const vr = (V + sh.k - 1) / sh.k;
        self.in[0] = try a.alloc(f64, n * V);
        for (self.in[0]) |*v| v.* = mu.int(rng, -1024, 1024) / 16; // sixteenths: ties
        if (fill == .nonfinite) {
            const x = self.in[0];
            x[0] = std.math.nan(f64);
            x[V / 2] = std.math.nan(f64);
            if (n > 1) x[V + V / 3] = std.math.inf(f64);
            if (n > 2) x[2 * V + 7] = -std.math.inf(f64);
            if (n > 3) for (x[3 * V ..][0..V], 0..) |*v, i| {
                v.* = if (i == 5) std.math.nan(f64) else -std.math.inf(f64); // the first -inf wins
            };
        }
        const lay = try a.alloc(f64, sh.k * n * vr);
        @memset(lay, std.math.inf(f64));
        for (0..n) |r| for (0..V) |col| {
            lay[((col / vr) * n + r) * vr + col % vr] = self.in[0][r * V + col];
        };
        self.dev[0] = try mu.put(&self.rig, ga, f32, lay);
        self.dev[1] = try self.rig.output(n * 4);
        return true;
    }

    fn cacheAt(self: *const Draft, s: usize, kp: usize, kh: usize) usize {
        return ((s * self.ring + kp % self.ring) * self.kv + kh) * HD;
    }

    /// Keys [lo, hi] query b of stream s sees.
    fn keys(self: *const Draft, s: usize, b: usize) [2]usize {
        const p0: usize = @intCast(self.pos[s]);
        return .{ (p0 + b + 1) -| W, p0 + B - 1 };
    }

    /// Gate 4's V: zero where only some of a block's queries look, mirrored over the keys all of them see.
    fn mirrorV(self: *Draft, rng: std.Random) !void {
        const terms = try self.arena.allocator().alloc(f32, W + B);
        for (0..self.shape.rows) |s| {
            const all = self.keys(s, B - 1);
            const first = self.keys(s, 0)[0];
            for (0..self.kv) |kh| for (0..HD) |e| {
                const n = all[1] + 1 - all[0];
                attn_k.mirrored(rng, terms[0..n]);
                for (first..all[1] + 1) |kp| self.in[2][self.cacheAt(s, kp, kh) + e] = if (kp < all[0]) 0 else mu.h16(terms[kp - all[0]]);
            };
        }
    }

    /// dflash.zig's launches.
    pub fn run(self: *Draft) !void {
        const s = self.rig.s;
        const sh = self.shape;
        const n = sh.rows;
        const d = self.dev;
        var a: cuda.Args = .{};
        var grid: [2]u32 = .{ @intCast(n), 1 };
        var block: u32 = 256;
        switch (self.kind) {
            .embed => {
                inline for (.{ d[0], d[1], d[2], d[3] }) |p| a.add(p);
                inline for (.{ sh.k, sh.n }) |v| a.add(@as(c_int, @intCast(v)));
                grid[0] = @intCast(n * sh.k);
            },
            .tap => {
                for (0..sh.k) |j| {
                    var t: cuda.Args = .{};
                    t.add(d[0] + j * n * sh.n * 4);
                    t.add(d[1]);
                    inline for (.{ sh.n, sh.k, j }) |v| t.add(@as(c_int, @intCast(v)));
                    try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(n), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &t);
                }
                return s.synchronize();
            },
            .store => {
                inline for (.{ d[0], d[1], d[2], d[3] }) |p| a.add(p);
                inline for (.{ sh.n, n }) |v| a.add(@as(c_int, @intCast(v)));
            },
            .take => {
                inline for (.{ d[0], d[1] }) |p| a.add(p);
                inline for (.{ B, sh.n }) |v| a.add(@as(c_int, @intCast(v)));
                grid[0] = @intCast(n * (B - 1));
            },
            .norm_q, .norm_k => {
                inline for (.{ d[0], d[1], d[2], d[3], d[4], d[5] }) |p| a.add(p);
                a.add(if (self.kind == .norm_q) d[6] else @as(u64, 0));
                a.add(if (self.kind == .norm_k) d[6] else @as(u64, 0));
                inline for (.{ sh.n, ROT, self.ring }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(eps);
                grid[1] = @intCast(sh.n);
                block = HD;
            },
            .attn => {
                inline for (.{ d[0], d[1], d[2], d[6], d[3], d[4], d[5] }) |p| a.add(p);
                inline for (.{ B, W, self.ring, sh.n, self.kv }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(scale);
                grid = .{ @intCast(n * B), @intCast(self.kv) };
            },
            .argmax => {
                inline for (.{ d[0], d[1] }) |p| a.add(p);
                inline for (.{ sh.k, n, (sh.n + sh.k - 1) / sh.k, sh.n }) |v| a.add(@as(c_int, @intCast(v)));
            },
        }
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = 1 }, .block = .{ .x = block } }, s, &a);
        try s.synchronize();
    }

    pub fn count(self: *Draft) usize {
        const sh = self.shape;
        return switch (self.kind) {
            .embed => sh.rows * sh.k * sh.n,
            .tap => sh.k * sh.rows * sh.n,
            .store => sh.rows * sh.n * sh.k,
            .take => sh.rows * (B - 1) * sh.n,
            .norm_q, .norm_k => sh.rows * sh.n * HD,
            .attn => sh.rows * B * sh.n * HD,
            .argmax => sh.rows * 4,
        };
    }

    pub fn reference(self: *Draft, want: []f64, mag: []f64) !void {
        if (self.kind == .argmax) return error.StatedRule; // gates 1 and 3 run check
        @memset(mag, 0);
        return self.eval(.none, .sequential, want, null);
    }

    pub fn simulate(self: *Draft, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.eval(acc, order, out, null);
    }

    /// Moves are exact; norms and attention by first-order propagation, doubled; gate 4's rows (scores 0) stay exact.
    pub fn bounds(self: *Draft, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        if (self.kind != .norm_q and self.kind != .norm_k and self.kind != .attn) return;
        if (self.stress and self.kind == .attn) return;
        const tmp = try self.arena.allocator().alloc(f64, want.len);
        try self.eval(.none, .sequential, tmp, allow);
    }

    fn sum(acc: c.Accum, order: g.Order, t: []const f64) f64 {
        if (acc != .none) return rows.sumIn(acc, order, t);
        var s: f64 = 0;
        for (t) |x| s += x;
        return s;
    }

    fn eval(self: *Draft, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        const sh = self.shape;
        const n = sh.rows;
        const D = sh.n;
        switch (self.kind) {
            .embed => for (0..n) |s| for (0..sh.k) |b| for (0..D) |i| {
                out[(s * sh.k + b) * D + i] = if (b == 0) self.in[0][(n - 1 - s) * D + i] else self.in[1][i];
            },
            .tap => for (0..n) |r| for (0..sh.k) |j| for (0..D) |i| {
                out[(r * sh.k + j) * D + i] = self.in[0][(j * n + r) * D + i];
            },
            .store => for (0..n) |r| {
                const row = sh.n * sh.k;
                const p: usize = @intCast(self.pos[r]);
                @memcpy(out[(p % n) * row ..][0..row], self.in[0][r * row ..][0..row]);
            },
            .take => for (0..n) |s| for (1..B) |b| {
                @memcpy(out[(s * (B - 1) + b - 1) * D ..][0..D], self.in[0][(s * B + b) * D ..][0..D]);
            },
            .norm_q, .norm_k => self.evalNorm(acc, order, out, err),
            .attn => try self.evalAttn(acc, order, out, err),
            .argmax => return error.StatedRule,
        }
    }

    fn evalNorm(self: *Draft, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) void {
        const n = self.shape.rows;
        const nh = self.shape.n;
        const e_rs = 0.5 * mu.gamma(@floatFromInt(normDepth(self.shape))) + 3 * mu.u + mu.fast;
        var sq: [HD]f64 = undefined;
        var y: [HD]f64 = undefined;
        const e: c.Elem = if (self.kind == .norm_q) .f32 else .f16;
        for (0..n) |r| for (0..nh) |h| {
            const x = self.in[0][(r * nh + h) * HD ..][0..HD];
            for (&sq, x) |*o, v| o.* = v * v;
            const rs = 1 / @sqrt(sum(acc, order, &sq) / HD + eps);
            for (&y, x, self.w) |*o, v, w| o.* = v * rs * w;
            const p: usize = @intCast(self.pos[r]);
            const row = if (self.kind == .norm_q) r else p % self.ring;
            for (0..HD) |d| {
                const at = (row * nh + h) * HD + d;
                if (d >= ROT) { // past the rotated dims: the normed value
                    out[at] = y[d];
                    if (err) |el| el[at] = mu.allow(e, y[d], 2 * @abs(y[d]) * (e_rs + 3 * mu.u));
                    continue;
                }
                const j = d % (ROT / 2);
                const cs = self.tab[0][p * ROT / 2 + j];
                const sn = self.tab[1][p * ROT / 2 + j];
                const other = if (d < ROT / 2) -y[d + ROT / 2] else y[d - ROT / 2];
                const v = y[d] * cs + other * sn;
                out[at] = v;
                if (err) |el| el[at] = mu.allow(e, v, 2 * (@abs(y[d] * cs) + @abs(other * sn)) * (e_rs + 4 * mu.u));
            }
        };
    }

    fn evalAttn(self: *Draft, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const H = self.shape.n;
        const most = W + B;
        const buf = try a.alloc(f64, 3 * most);
        defer a.free(buf);
        const sc = buf[0..most];
        const p = buf[most .. 2 * most];
        const t = buf[2 * most ..];
        for (0..self.shape.rows) |s| for (0..B) |b| for (0..H) |h| {
            const ks = self.keys(s, b);
            const nk = ks[1] + 1 - ks[0];
            const q = self.in[0][((s * B + b) * H + h) * HD ..][0..HD];
            var m: f64 = -std.math.inf(f64);
            var e_s: f64 = 0;
            for (0..nk) |j| {
                const kr = self.in[1][self.cacheAt(s, ks[0] + j, h / (H / self.kv))..][0..HD];
                var dot: f64 = 0;
                var mag: f64 = 0;
                for (q, kr) |x, y| {
                    dot += x * y;
                    mag += @abs(x * y);
                }
                sc[j] = dot * scale;
                m = mu.fmax(m, sc[j]);
                e_s = @max(e_s, mu.gamma(HD + 1) * mag * scale);
            }
            const sk = self.w[h];
            const M = mu.fmax(m, sk); // the sink folds in last
            for (p[0..nk], sc[0..nk]) |*x, v| x.* = @exp(v - M);
            for (t[0..nk]) |*x| x.* = 1;
            const l = (if (acc == .none) sum(.none, order, p[0..nk]) else sum(acc, order, t[0..nk])) + @exp(sk - M);
            const tiles: f64 = @floatFromInt((nk + 31) / 32 + 2);
            const rel = 2 * e_s + tiles * (2 * mu.fast + 3 * mu.u) + mu.gamma(tiles + 5);
            for (0..HD) |e| {
                var abs: f64 = 0;
                for (t[0..nk], p[0..nk], 0..) |*x, pj, j| {
                    const vv = self.in[2][self.cacheAt(s, ks[0] + j, h / (H / self.kv)) + e];
                    x.* = if (acc == .none) pj * vv else vv;
                    abs += pj * @abs(vv);
                }
                const y = sum(acc, order, t[0..nk]) / l;
                const o = ((s * B + b) * H + h) * HD + e;
                out[o] = y;
                if (err) |el| el[o] = mu.allow(.f16, y, 2 * (abs / l) * (2 * rel + mu.gamma(@floatFromInt(nk + 2))));
            }
        };
    }

    pub fn check(self: *Draft, _: g.Gate) !?[]const u8 {
        const got = try self.rig.bytes(self.dev[1], self.shape.rows * 4);
        const V = self.shape.n;
        for (0..self.shape.rows) |r| {
            var best: ?usize = null;
            for (self.in[0][r * V ..][0..V], 0..) |v, i| {
                if (std.math.isNan(v)) continue;
                if (best == null or v > self.in[0][r * V + best.?]) best = i;
            }
            const want: i32 = if (best) |b| @intCast(b) else 0x7fffffff;
            const id = std.mem.readInt(i32, got[r * 4 ..][0..4], .little);
            if (id != want) return try std.fmt.bufPrint(&self.msg, "row {d}: column {d} where the rule gives {d}", .{ r, id, want });
        }
        return null;
    }
};
