//! sparse_index.cu under the contract on CUDA: indexer scores and block summaries against core/sparse_index.zig.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tc = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const si = core.sparse_index;
const gpu = cuda.sparse_index;

const D = gpu.dim;

/// The larger of a and b, NaN kept (sparse_index's and the kernels' max).
fn larger(a: f64, b: f64) f64 {
    return if (b > a or std.math.isNan(b)) b else a;
}

/// A scores test: Kolibri-style relu weighted sums over one key head, a max over 48 heads on 4, or the control.
pub const Variant = enum { flat, gqa, short };

/// Shape: query rows (rows), keys (n) and query heads (k); a key's dot chain of 128, then the heads' chain.
fn scoreDepth(s: c.Shape) usize {
    return D + s.k;
}

pub fn scoresSpec(v: Variant) c.Spec {
    const flat = &[_]c.Shape{ .{ .rows = 1, .n = 16384, .k = 64 }, .{ .rows = 4, .n = 4096, .k = 32 }, .{ .rows = 17, .n = 2048, .k = 64 } };
    const gqa = &[_]c.Shape{ .{ .rows = 1, .n = 16384, .k = 48 }, .{ .rows = 4, .n = 2048, .k = 48 } };
    return .{
        .name = switch (v) {
            .flat => "cuda sparse index tf_index_scores (relu, weighted sum, one key head)",
            .gqa => "cuda sparse index tf_index_scores (identity, max, 48 heads over 4 key heads)",
            .short => "control: sparse index scores told each row one key fewer",
        },
        .accum = .f32,
        .depth = scoreDepth,
        .ops = 2, // the weight's product and the scale's
        .out = .f32,
        .tiles = if (v == .gqa) .{ .{}, .{ .size = 64 }, .{} } else .{ .{}, .{ .size = 64 }, .{ .size = 32 } },
        .served = if (v == .gqa) gqa else flat,
        .sums = .{ .why = "integer query halves mirrored over keys of ones: one head's dot is 8 after partials near 500K" },
    };
}

pub const Scores = struct {
    rig: rig_.Rig,
    x: *const gpu.SparseIndex,
    v: Variant = .flat,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    k: []u16 = &.{},
    w: []f32 = &.{},
    counts: []i32 = &.{},
    dev: [4]u64 = @splat(0),
    out: u64 = 0,

    pub fn deinit(self: *Scores) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Scores) void {
        const a = self.rig.gpu.gpa;
        a.free(self.q);
        a.free(self.k);
        a.free(self.w);
        a.free(self.counts);
        self.q = &.{};
        self.k = &.{};
        self.w = &.{};
        self.counts = &.{};
    }

    pub fn kernel(self: *Scores) g.Kernel {
        return .{ .spec = scoresSpec(self.v), .ctx = self, .vt = &vt, .control = if (self.v == .short) .shapes else null };
    }

    const vt = g.hooks(Scores);

    fn hk(self: *const Scores) usize {
        return self.policy().key_heads;
    }

    fn ks(self: *const Scores) usize {
        return self.hk() * D + 8; // key rows a little wider than their heads
    }

    fn weighted(self: *const Scores) bool {
        return self.v != .gqa;
    }

    /// The one semantic policy: core's; the launch below is built from it.
    fn policy(self: *const Scores) si.Policy {
        return if (self.v == .gqa) .{ .topk = 1, .phi = .identity, .reduce = .max, .key_heads = 4, .scale = 0.0625 } else .{ .topk = 1, .scale = 0.0625 };
    }

    fn sh(self: *const Scores) !gpu.Scores {
        return gpu.scoresFor(self.policy(), .{}, D, self.shape.k, self.shape.rows, self.shape.n, self.ks(), self.ld());
    }

    /// Score rows 37 wider than the keys: [n, ld) must come back -inf, not stale.
    fn ld(self: *const Scores) usize {
        return self.shape.n + 37;
    }

    pub fn prepare(self: *Scores, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k > gpu.max_heads or shape.k % self.hk() != 0) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const H = shape.k;
        self.q = try a.alloc(u16, shape.rows * H * D);
        self.k = try a.alloc(u16, shape.n * self.ks());
        self.w = try a.alloc(f32, if (self.weighted()) shape.rows * H else 0);
        self.counts = try a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.counts, 0..) |*n, r| n.* = @intCast(shape.n - r * shape.n / shape.rows);
        for (self.q) |*e| e.* = attn.normal(rng);
        for (self.k) |*e| e.* = attn.normal(rng);
        for (self.w) |*e| e.* = rng.float(f32) * 2 - 1;
        switch (fill) {
            .random => {},
            .nonfinite => {
                self.q[3] = c.toBf16(std.math.nan(f32)); // row 0, head 0: that row all NaN
                if (shape.n > 2) {
                    self.k[1 * self.ks() + 5] = c.toBf16(std.math.inf(f32));
                    self.k[2 * self.ks() + 7] = c.toBf16(-std.math.inf(f32));
                }
                if (self.weighted() and shape.rows > 1 and H > 1) self.w[(shape.rows - 1) * H + 1] = std.math.nan(f32);
            },
            .stress => {
                // head 0 over ones: odd v below 32, odd 64 v to 63, then -v (8 nudged): 14-bit partials in any order
                @memset(self.q, 0);
                @memset(self.k, c.toBf16(1));
                @memset(self.w, 1);
                for (0..shape.rows) |r| {
                    const row = self.q[r * H * D ..][0..D];
                    for (0..D / 2) |d| {
                        const v: f32 = @floatFromInt((1 + 2 * rng.uintLessThan(u32, 128)) * @as(u32, if (d >= 32) 64 else 1));
                        row[d] = c.toBf16(v);
                        row[D / 2 + d] = c.toBf16((if (d < 8) @as(f32, 1) else 0) - v);
                    }
                }
            },
        }
        if (self.v != .short and shape.rows >= 3) {
            self.counts[1] = 0; // planted: a row with no keys, all -inf
            self.counts[2] = @intCast(shape.n + 5); // planted: past n, read as n (the stored count stays as told)
        }
        const told = try a.dupe(i32, self.counts);
        defer a.free(told);
        if (self.v == .short) for (told) |*n| {
            n.* -= 1;
        };
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.k));
        self.dev[2] = if (self.weighted()) try self.rig.input(std.mem.sliceAsBytes(self.w)) else 0;
        self.dev[3] = if (shape.rows == 1 and self.v != .short) 0 else try self.rig.input(std.mem.sliceAsBytes(told));
        self.out = try self.rig.output(shape.rows * self.ld() * 4);
        return true;
    }

    pub fn run(self: *Scores) !void {
        try self.x.runScores(self.rig.s, self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.out, try self.sh());
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Scores) usize {
        return self.shape.rows * self.ld();
    }

    /// Output (r, key) as sparse_index.score does it, a key head's group at a time; or the simulated sums of `acc`.
    fn each(self: *Scores, acc: ?struct { a: c.Accum, o: g.Order }, want: []f64, mag: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const H = self.shape.k;
        const G = H / self.hk();
        const qf = try a.alloc(f64, self.q.len);
        defer a.free(qf);
        for (qf, self.q) |*y, e| y.* = c.fromBf16(e);
        const kf = try a.alloc(f64, self.hk() * D);
        defer a.free(kf);
        const wf = try a.alloc(f64, H);
        defer a.free(wf);
        const t = try a.alloc(f64, D);
        defer a.free(t);
        const p = self.policy();
        for (0..self.shape.rows) |r| {
            for (wf, 0..) |*y, h| y.* = if (self.weighted()) self.w[r * H + h] else 1;
            const limit: usize = @intCast(std.math.clamp(self.counts[r], 0, @as(i32, @intCast(self.shape.n))));
            for (0..self.ld()) |key| {
                const i = r * self.ld() + key;
                if (key >= limit) {
                    want[i] = -std.math.inf(f64);
                    if (mag) |m| m[i] = 0;
                    continue;
                }
                for (kf, self.k[key * self.ks() ..][0 .. self.hk() * D]) |*y, e| y.* = c.fromBf16(e);
                if (acc == null) {
                    // core's score over the whole key row (its key heads) and the summed terms' magnitude
                    want[i] = si.score(p, .{ .q = qf[r * H * D ..][0 .. H * D], .heads = H, .weights = wf }, kf);
                    var m: f64 = 0;
                    for (0..H) |h| for (qf[(r * H + h) * D ..][0..D], kf[h / G * D ..][0..D]) |u, z| {
                        m += @abs(wf[h] * u * z);
                    };
                    if (mag) |mm| mm[i] = m * p.scale;
                    continue;
                }
                var s: f64 = if (p.reduce == .max) -std.math.inf(f64) else 0;
                for (0..self.hk()) |kh| {
                    const key_h = kf[kh * D ..][0..D];
                    const qg = qf[(r * H + kh * G) * D ..][0 .. G * D];
                    const wg = wf[kh * G ..][0..G];
                    for (0..G) |h| {
                        for (t, qg[h * D ..][0..D], key_h) |*y, u, z| y.* = u * z;
                        var dot = contract.rows.sumIn(acc.?.a, acc.?.o, t);
                        if (p.phi == .relu and dot < 0) dot = 0;
                        s = if (p.reduce == .max) larger(s, wg[h] * dot) else s + wg[h] * dot;
                    }
                }
                want[i] = s * p.scale;
            }
        }
    }

    pub fn reference(self: *Scores, want: []f64, mag: []f64) !void {
        return self.each(null, want, mag);
    }

    pub fn simulate(self: *Scores, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.each(.{ .a = acc, .o = order }, out, null);
    }
};

/// Shape: blocks (rows), keys a block (n) and the key width (k, 128); the last block holds half its keys (open).
fn blockDepth(s: c.Shape) usize {
    return s.n + 8;
}

pub fn blocksSpec(s: si.Summary, unit: bool, short: bool, open: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 256, .n = 64, .k = D }, .{ .rows = 40, .n = 128, .k = D }, .{ .rows = 9, .n = 16, .k = D } };
    const sums = s == .mean;
    return .{
        .name = if (short) "control: sparse index block means told one key fewer" else switch (s) {
            .mean => if (unit) "cuda sparse index tf_index_blocks (mean of unit-RMS keys)" else if (open) "cuda sparse index tf_index_blocks (mean)" else "cuda sparse index tf_index_blocks (mean, open block left out)",
            .max => "cuda sparse index tf_index_blocks (max)",
            .first => "cuda sparse index tf_index_blocks (first)",
            .last => "cuda sparse index tf_index_blocks (last)",
        },
        .accum = if (sums) .f32 else .none,
        .depth = blockDepth,
        .ops = if (unit) 5 else if (sums) 1 else 0, // / count; unit keys' / 128, sqrt, 1 / x and product too
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = 64 }, .{} },
        .served = served,
        .sums = if (unit) .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "unit keys' 1 / sqrt factors are irrational: no exact fp32 stress" } else .{ .why = "mirrored integer columns, 8 nudged by one" },
    };
}

pub const Blocks = struct {
    rig: rig_.Rig,
    x: *const gpu.SparseIndex,
    summary: si.Summary = .mean,
    unit: bool = false,
    short: bool = false,
    open: bool = true,
    shape: c.Shape = undefined,
    keys: usize = 0,
    k: []u16 = &.{},
    dev: u64 = 0,
    out: u64 = 0,

    const ks = D + 8;

    pub fn deinit(self: *Blocks) void {
        self.rig.gpu.gpa.free(self.k);
        self.rig.deinit();
    }

    pub fn kernel(self: *Blocks) g.Kernel {
        return .{ .spec = blocksSpec(self.summary, self.unit, self.short, self.open), .ctx = self, .vt = &vt, .control = if (self.short) .shapes else null };
    }

    const vt = g.hooks(Blocks);

    fn told(self: *const Blocks) usize {
        return if (self.short) self.keys - 1 else self.keys;
    }

    /// The one semantic policy: core's; the launch is built from it.
    fn policy(self: *const Blocks) si.Policy {
        return .{ .topk = 1, .block = self.shape.n, .summary = self.summary, .norm_first = self.unit, .open_block = self.open };
    }

    /// Summaries due for the keys: core's count, open block or not.
    fn summaries(self: *const Blocks) usize {
        return si.blocks(self.policy(), self.keys - 1);
    }

    pub fn prepare(self: *Blocks, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != D) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        self.keys = shape.rows * shape.n - (shape.n / 2);
        if (self.keys < 2) return false;
        a.free(self.k);
        self.k = &.{};
        self.k = try a.alloc(u16, self.keys * ks);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.k) |*e| e.* = attn.normal(rng);
        switch (fill) {
            .random => {},
            .nonfinite => if (self.keys > 2) {
                self.k[3] = c.toBf16(std.math.nan(f32));
                self.k[ks + 5] = c.toBf16(std.math.inf(f32));
                self.k[2 * ks + 7] = c.toBf16(-std.math.inf(f32));
                self.k[(self.keys - 1) * ks + 9] = c.toBf16(std.math.inf(f32)); // the last summary's last key
            },
            .stress => if (!self.unit) {
                // each block's column: mirrored integers, its mean 8 / its keys
                const col = try a.alloc(f32, shape.n);
                defer a.free(col);
                for (0..shape.rows) |b| {
                    const lo = b * shape.n;
                    const m = @min(shape.n, self.keys - lo);
                    for (0..D) |d| {
                        attn.mirrored(rng, col[0..m]);
                        std.mem.reverse(f32, col[m / 2 .. m]); // a full block: i and i + 32 must not cancel (strided)
                        for (col[0..m], lo..) |y, i| self.k[i * ks + d] = c.toBf16(y);
                    }
                }
            },
        }
        self.dev = try self.rig.input(std.mem.sliceAsBytes(self.k));
        if (self.summaries() == 0) return false;
        self.out = try self.rig.output(self.summaries() * D * 2);
        return true;
    }

    pub fn run(self: *Blocks) !void {
        try self.x.runBlocks(self.rig.s, self.dev, self.out, try gpu.blocksFor(self.policy(), .{}, D, self.told(), ks, D));
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Blocks) usize {
        return self.summaries() * D;
    }

    fn keysF64(self: *Blocks) ![]f64 {
        const kf = try self.rig.gpu.gpa.alloc(f64, self.keys * D);
        for (0..self.keys) |i| for (0..D) |d| {
            kf[i * D + d] = c.fromBf16(self.k[i * ks + d]);
        };
        return kf;
    }

    /// sparse_index.summarise of each block; mag: the summed terms' magnitudes over the block's keys.
    pub fn reference(self: *Blocks, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const kf = try self.keysF64();
        defer a.free(kf);
        const p = self.policy();
        const kabs = try a.dupe(f64, kf);
        defer a.free(kabs);
        for (kabs) |*v| v.* = @abs(v.*);
        for (0..self.summaries()) |b| {
            si.summarise(p, .{ .k = kf, .dim = D }, b, self.keys - 1, want[b * D ..][0..D]);
            si.summarise(p, .{ .k = kabs, .dim = D }, b, self.keys - 1, mag[b * D ..][0..D]);
        }
    }

    /// Each block's mean with its terms summed in `acc`, in `order`.
    pub fn simulate(self: *Blocks, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const kf = try self.keysF64();
        defer a.free(kf);
        const t = try a.alloc(f64, self.shape.n);
        defer a.free(t);
        for (0..self.summaries()) |b| {
            const lo = b * self.shape.n;
            const m = @min(self.shape.n, self.keys - lo);
            for (0..D) |d| {
                for (t[0..m], lo..) |*y, i| {
                    var r: f64 = 1;
                    if (self.unit) {
                        var ss: f64 = 0;
                        for (kf[i * D ..][0..D]) |v| ss += v * v;
                        r = if (ss > 0) 1 / @sqrt(ss / D) else 0;
                    }
                    y.* = kf[i * D + d] * r;
                }
                out[b * D + d] = contract.rows.sumIn(acc, order, t[0..m]) / @as(f64, @floatFromInt(m));
            }
        }
    }
};

/// The open-block test's plant: none (negative and oversized ends planted, clamped), or a control.
pub const OpenPlant = enum { none, ahead, fewer };

/// Shape: query rows (rows), keys a block (n), key width (k, 128). Each row's open block from its own end (t + 1).
pub fn openSpec(plant: OpenPlant) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 64, .n = 64, .k = D }, .{ .rows = 17, .n = 16, .k = D }, .{ .rows = 300, .n = 128, .k = D } };
    return .{
        .name = switch (plant) {
            .ahead => "control: sparse index open blocks told each row's end one key later",
            .fewer => "control: sparse index open blocks told one key fewer than the cache holds",
            .none => "cuda sparse index tf_index_open_blocks (a prompt chunk's rows, mean; negative and oversized ends)",
        },
        .accum = .f32,
        .depth = blockDepth,
        .ops = 1,
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = 64 }, .{} },
        .served = served,
        .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "the mean's sums are tf_index_blocks' (gate 4 there); here: each row's own range" },
    };
}

/// Rows of a prompt chunk or a verify chain: each summarises only its own keys, never a later row's.
pub const Open = struct {
    rig: rig_.Rig,
    x: *const gpu.SparseIndex,
    plant: OpenPlant = .none,
    shape: c.Shape = undefined,
    keys: usize = 0,
    k: []u16 = &.{},
    ends: []i32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    const ks = D + 8;

    pub fn deinit(self: *Open) void {
        self.rig.gpu.gpa.free(self.k);
        self.rig.gpu.gpa.free(self.ends);
        self.rig.deinit();
    }

    pub fn kernel(self: *Open) g.Kernel {
        return .{ .spec = openSpec(self.plant), .ctx = self, .vt = &vt, .control = if (self.plant != .none) .shapes else null };
    }

    const vt = g.hooks(Open);

    fn policy(self: *const Open) si.Policy {
        return .{ .topk = 1, .block = self.shape.n };
    }

    /// Ends spread over the keys: the full length, a block's end (no open block), one key, and many in between.
    pub fn prepare(self: *Open, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != D) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        self.keys = 4 * shape.n + 3;
        a.free(self.k);
        a.free(self.ends);
        self.k = &.{};
        self.ends = &.{};
        self.k = try a.alloc(u16, self.keys * ks);
        self.ends = try a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.k) |*e| e.* = attn.normal(rng);
        if (fill == .nonfinite) {
            self.k[(self.keys - 1) * ks + 3] = c.toBf16(std.math.nan(f32)); // only the full-length row sees it
            self.k[(self.keys - 2) * ks + 5] = c.toBf16(std.math.inf(f32));
        }
        for (self.ends, 0..) |*e, r| e.* = @intCast(switch (r) {
            0 => self.keys,
            1 => shape.n,
            2 => 1,
            else => 1 + (r * 7919 + 13) % self.keys,
        });
        const told = try a.dupe(i32, self.ends);
        defer a.free(told);
        if (self.plant == .ahead) for (told) |*e| {
            e.* = @min(e.* + 1, @as(i32, @intCast(self.keys)));
        };
        if (self.plant == .none and shape.rows > 4) { // the kernel clamps to [0, keys]: an empty row and a full one
            told[3] = -5;
            told[4] = @intCast(self.keys + 9);
            self.ends[3] = 0;
            self.ends[4] = @intCast(self.keys);
        }
        self.dev = .{ try self.rig.input(std.mem.sliceAsBytes(self.k)), try self.rig.input(std.mem.sliceAsBytes(told)) };
        self.out = try self.rig.output(shape.rows * D * 2);
        return true;
    }

    pub fn run(self: *Open) !void {
        const b = try gpu.blocksFor(self.policy(), .{}, D, if (self.plant == .fewer) self.keys - 1 else self.keys, ks, D);
        try self.x.runOpenBlocks(self.rig.s, self.dev[0], self.dev[1], self.out, self.shape.rows, b);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Open) usize {
        return self.shape.rows * D;
    }

    /// sparse_index.summarise of block end / B at t = end - 1 (0 when end is a block's end).
    fn each(self: *Open, acc: ?struct { a: c.Accum, o: g.Order }, want: []f64, mag: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const kf = try a.alloc(f64, self.keys * D);
        defer a.free(kf);
        for (0..self.keys) |i| for (0..D) |d| {
            kf[i * D + d] = c.fromBf16(self.k[i * ks + d]);
        };
        const kabs = try a.dupe(f64, kf);
        defer a.free(kabs);
        for (kabs) |*v| v.* = @abs(v.*);
        const B = self.shape.n;
        const t = try a.alloc(f64, B);
        defer a.free(t);
        for (self.ends, 0..) |e, r| {
            const end: usize = @intCast(e);
            const row = want[r * D ..][0..D];
            if (end % B == 0) {
                @memset(row, 0);
                if (mag) |m| @memset(m[r * D ..][0..D], 0);
                continue;
            }
            if (acc) |how| {
                const lo = end - end % B;
                for (0..D) |d| {
                    for (t[0 .. end - lo], lo..) |*y, i| y.* = kf[i * D + d];
                    row[d] = contract.rows.sumIn(how.a, how.o, t[0 .. end - lo]) / @as(f64, @floatFromInt(end - lo));
                }
                continue;
            }
            si.summarise(self.policy(), .{ .k = kf, .dim = D }, end / B, end - 1, row);
            if (mag) |m| si.summarise(self.policy(), .{ .k = kabs, .dim = D }, end / B, end - 1, m[r * D ..][0..D]);
        }
    }

    pub fn reference(self: *Open, want: []f64, mag: []f64) !void {
        return self.each(null, want, mag);
    }

    pub fn simulate(self: *Open, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.each(.{ .a = acc, .o = order }, out, null);
    }
};

/// The kernels and their planted controls, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    x: gpu.SparseIndex,
    scores: [3]Scores,
    blocks: [7]Blocks,
    open: [3]Open,

    pub fn init(dev: tc.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try dev.gpa.create(Set);
        errdefer dev.gpa.destroy(set);
        set.gpa = dev.gpa;
        set.x = try gpu.SparseIndex.load(dev.d);
        errdefer set.x.unload();
        for (&set.scores, [_]Variant{ .flat, .gqa, .short }) |*t, v| t.* = .{ .rig = .{ .gpu = dev, .s = s }, .x = &set.x, .v = v };
        const kinds = [_]struct { si.Summary, bool, bool, bool }{ .{ .mean, false, false, true }, .{ .mean, false, false, false }, .{ .mean, true, false, true }, .{ .max, false, false, true }, .{ .first, false, false, true }, .{ .last, false, false, true }, .{ .mean, false, true, true } };
        for (&set.blocks, kinds) |*t, kd| t.* = .{ .rig = .{ .gpu = dev, .s = s }, .x = &set.x, .summary = kd[0], .unit = kd[1], .short = kd[2], .open = kd[3] };
        for (&set.scores) |*t| try list.append(a, t.kernel());
        for (&set.blocks) |*t| try list.append(a, t.kernel());
        for (&set.open, [_]OpenPlant{ .none, .ahead, .fewer }) |*t, p| t.* = .{ .rig = .{ .gpu = dev, .s = s }, .x = &set.x, .plant = p };
        for (&set.open) |*t| try list.append(a, t.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.scores) |*t| t.deinit();
        for (&set.blocks) |*t| t.deinit();
        for (&set.open) |*t| t.deinit();
        set.x.unload();
        set.gpa.destroy(set);
    }
};
