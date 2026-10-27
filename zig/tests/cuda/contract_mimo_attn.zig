//! MiMo-V2's attention under the contract (attention.cu): q/k/v into the caches, chunked rows, merge, prompt rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mu = @import("contract_mimo_util.zig");
const attn_k = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;

const QK = 192; // q / k head dim
const VD = 128; // v head dim: the pack's v heads are 192 wide, their first 128 used
const RD = 64; // rotated dims
const CK = 256; // ops.CK: keys a chunk
const W = 128; // sliding_window
const RB = 64; // ops.PROMPT_ROWS: a prompt block's rows
const BN = 64; // attention.cu tc::BN: a key tile (rows and prompt)
const ATTN_SHARED: u32 = 2 * BN * ((QK + 8) + (VD + 8)) * 2; // ops.ATTN_SHARED: rows and prompt
const scale: f32 = 0.07216878364870323; // qk_dim^-0.5
const theta: f64 = 10000; // swa_rope_theta: the tables are the kernel's inputs

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A score's 192 fused products, or a chunk's 256 keys chained for l and o.
fn rowsDepth(_: c.Shape) usize {
    return CK + 1;
}

/// The chunk partials folded in order (and the sink), then o / l.
fn mergeDepth(s: c.Shape) usize {
    return s.k + 3;
}

/// A row's keys chained through its 64-key tiles, then the tiles' folds.
fn promptDepth(s: c.Shape) usize {
    return s.k + s.k / BN + 3;
}

pub const Kind = enum { qkv, rows_slide, rows_full, merge_sink, merge_full, prompt };

const nan_rule = "NaN and infinities in K and V reach the outputs as float64's two-pass softmax gives them, a NaN " ++
    "score dropped from the maximum as fmaxf drops it";
const two_why = "a sliding row folds two chunk partials and its sink: no long sum to test";

pub fn spec(kind: Kind) c.Spec {
    const qkv = &[_]c.Shape{ .{ .rows = 8, .n = 32, .k = 4 }, .{ .rows = 1, .n = 32, .k = 2 }, .{ .rows = 128, .n = 32, .k = 4 }, .{ .rows = 64, .n = 64, .k = 8 } };
    const slide = &[_]c.Shape{ .{ .rows = 4, .n = 32, .k = 100000 }, .{ .rows = 1, .n = 32, .k = 1500 }, .{ .rows = 8, .n = 64, .k = 300 } };
    const full = &[_]c.Shape{ .{ .rows = 4, .n = 32, .k = 12000 }, .{ .rows = 2, .n = 32, .k = 3000 }, .{ .rows = 1, .n = 64, .k = 700 } };
    const keys: [3]c.Tile = .{ .{}, .{}, .{ .size = CK } };
    return switch (kind) {
        .qkv => .{ .name = "cuda mimo mimo_qkv (NEOX RoPE on 64 of 192 dims, k and v into the cache)", .accum = .none, .depth = noSums, .out = .f16, .served = qkv },
        .rows_slide => .{ .name = "cuda mimo mimo_attn_rows (tensor cores: a 128-key window over a ring, 2 chunks, p in fp16)", .accum = .f32, .depth = rowsDepth, .out = .f32, .nan = .{ .stated = nan_rule }, .tiles = keys, .served = slide },
        .rows_full => .{ .name = "cuda mimo mimo_attn_rows (tensor cores: full layer, 256-key chunks from 0, one empty past, p in fp16)", .accum = .f32, .depth = rowsDepth, .out = .f32, .nan = .{ .stated = nan_rule }, .tiles = keys, .served = full },
        .merge_sink => .{ .name = "cuda mimo mimo_attn_merge (sliding: two chunks, then the sink)", .accum = .f32, .depth = mergeDepth, .ops = 2, .out = .f16, .served = &.{ .{ .rows = 8, .n = 32, .k = 2 }, .{ .rows = 128, .n = 32, .k = 2 } }, .sums = .{ .f16 = 0, .bf16 = 0, .why = two_why } },
        .merge_full => .{ .name = "cuda mimo mimo_attn_merge (full layer: a bucket's chunks, no sink)", .accum = .f32, .depth = mergeDepth, .ops = 2, .out = .f16, .served = &.{ .{ .rows = 8, .n = 32, .k = 32 }, .{ .rows = 1, .n = 32, .k = 1024 } } },
        .prompt => .{ .name = "cuda mimo mimo_attn_prompt (tensor cores: 64 rows a block, 64-key tiles, p in fp16)", .accum = .f32, .depth = promptDepth, .ops = 2, .out = .f16, .nan = .{ .stated = nan_rule }, .tiles = .{ .{ .size = RB }, .{}, .{} }, .served = &.{ .{ .rows = 64, .n = 32, .k = 4096 }, .{ .rows = 16, .n = 32, .k = 8192 }, .{ .rows = 8, .n = 64, .k = 1000 }, .{ .rows = 100, .n = 32, .k = 2000 } } },
    };
}

pub const Attn = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    stress: bool = false,
    hk: usize = 0,
    len: usize = 0, // cache rows a slot
    nch: usize = 0,
    pos: []i32 = &.{},
    q: []f64 = &.{},
    k: []f64 = &.{}, // qkv: k rows [n, hk, 192]; attention: the key cache [slots, len, hk, 192]
    v: []f64 = &.{}, // qkv: v rows [n, hk, 192]; attention: the value cache [slots, len, hk, 128]
    tab: [2][]f64 = .{ &.{}, &.{} },
    part: [3][]f64 = .{ &.{}, &.{}, &.{} }, // merge: po, pm, pl
    sinks: []f64 = &.{},
    dev: [10]u64 = @splat(0),

    pub fn deinit(self: *Attn) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    const vt = blk: {
        var v = g.hooks(Attn);
        v.check = null;
        break :blk v;
    };

    pub fn kernel(self: *Attn) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    fn isRows(self: *const Attn) bool {
        return self.kind == .rows_slide or self.kind == .rows_full;
    }

    /// Query heads a key head: the sliding layers' 8, the full layers' 16 (both TP=1 and TP=2).
    fn group(self: *const Attn) usize {
        return if (self.kind == .rows_slide) 8 else 16;
    }

    fn slotOf(self: *const Attn, r: usize) usize {
        return if (self.kind == .prompt) 1 else r;
    }

    /// Keys [start, end) of row r's chunk ch (attention.cu: chunks of CK from the window's first key's chunk).
    fn span(self: *const Attn, r: usize, ch: usize) [2]usize {
        const p: usize = @intCast(self.pos[r]);
        const lo = if (self.kind == .rows_slide) (p + 1) -| W else 0;
        const start = (lo / CK + ch) * CK;
        const first = @max(start, lo);
        return .{ first, @max(first, @min(start + CK, p + 1)) };
    }

    fn kAt(self: *const Attn, r: usize, key: usize, kh: usize) usize {
        return ((self.slotOf(r) * self.len + key % self.len) * self.hk + kh) * QK;
    }

    fn vAt(self: *const Attn, r: usize, key: usize, kh: usize) usize {
        return ((self.slotOf(r) * self.len + key % self.len) * self.hk + kh) * VD;
    }

    fn rnd(self: *Attn, rng: std.Random, n: usize, lo: f64, hi: f64) ![]f64 {
        const x = try self.arena.allocator().alloc(f64, n);
        for (x) |*v| v.* = mu.uni(rng, f16, lo, hi);
        return x;
    }

    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.stress = fill == .stress;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        return switch (self.kind) {
            .qkv => self.prepareQkv(fill, rng),
            .rows_slide, .rows_full, .prompt => self.prepareCache(fill, rng),
            .merge_sink, .merge_full => self.prepareMerge(fill, rng),
        };
    }

    /// Rows at consecutive positions into one slot of `rows` cache rows: every cache row written once.
    fn prepareQkv(self: *Attn, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        const n = sh.rows;
        const H = sh.n;
        self.hk = sh.k;
        self.len = n;
        self.pos = try a.alloc(i32, n);
        const base = rng.uintLessThan(u32, 4096);
        for (self.pos, 0..) |*p, r| p.* = @intCast(base + r);
        self.q = try self.rnd(rng, n * H * QK, -3, 3);
        self.k = try self.rnd(rng, n * self.hk * QK, -3, 3);
        self.v = try self.rnd(rng, n * self.hk * QK, -3, 3);
        if (fill == .nonfinite) {
            self.q[3] = std.math.nan(f64);
            self.k[QK + 7] = std.math.inf(f64);
            self.k[2 * QK + 70] = -std.math.inf(f64); // an unrotated dim: copied
            self.v[5] = std.math.nan(f64);
        }
        const tl = (base + n) * RD / 2;
        for (&self.tab, 0..) |*t, i| {
            t.* = try a.alloc(f64, tl);
            for (t.*, 0..) |*o, j| {
                const ang = @as(f64, @floatFromInt(j / 32)) / std.math.pow(f64, theta, @as(f64, @floatFromInt(j % 32)) / 32);
                o.* = mu.f32r(if (i == 0) @cos(ang) else @sin(ang));
            }
        }
        inline for (.{ self.q, self.k, self.v }, 0..) |x, i| self.dev[i] = try mu.put(&self.rig, ga, f16, x);
        for (0..2) |i| self.dev[3 + i] = try mu.put(&self.rig, ga, f32, self.tab[i]);
        const slot = try a.alloc(i32, n);
        @memset(slot, 0);
        self.dev[5] = try mu.ints(&self.rig, i32, self.pos);
        self.dev[6] = try mu.ints(&self.rig, i32, slot);
        self.dev[7] = try self.rig.output(n * H * QK * 2);
        self.dev[8] = try self.rig.output(n * self.hk * QK * 2);
        self.dev[9] = try self.rig.output(n * self.hk * VD * 2);
        return true;
    }

    /// Decode rows each in its own slot (row r at k - 1 - r), or prompt rows k - n .. k - 1 in slot 1 of 2.
    fn prepareCache(self: *Attn, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        const n = sh.rows;
        const H = sh.n;
        const G = self.group();
        if (H % G != 0 or sh.k < n) return false;
        self.hk = H / G;
        const prompt = self.kind == .prompt;
        self.pos = try a.alloc(i32, n);
        for (self.pos, 0..) |*p, r| p.* = @intCast(if (prompt) sh.k - n + r else sh.k - 1 - r);
        self.len = if (self.kind == .rows_slide) W + 64 else sh.k; // a sliding layer's ring wraps
        self.nch = if (self.kind == .rows_slide) (W + CK - 1) / CK + 1 else (sh.k - 1) / CK + 2; // one chunk past: empty
        const slots = if (prompt) 2 else n;
        self.q = try a.alloc(f64, n * H * QK);
        for (self.q) |*o| o.* = if (self.stress) 0 else mu.uni(rng, f16, -1, 1);
        self.k = try self.rnd(rng, slots * self.len * self.hk * QK, -1, 1);
        self.v = try self.rnd(rng, slots * self.len * self.hk * VD, -2, 2);
        if (self.stress) try self.mirrorV(rng);
        if (fill == .nonfinite) {
            const at = [_][4]usize{ .{ 0, 5, 0, 3 }, .{ 1, 7, 0, 10 }, .{ 2, 2, 0, 0 } };
            const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
            for (at, put) |p, x| if (p[0] < n) {
                self.v[self.vAt(p[0], self.span(p[0], 0)[0] + p[1], p[2]) + p[3]] = x;
            };
            if (n > 3) self.k[self.kAt(3, self.span(3, 0)[0] + 1, self.hk - 1) + 4] = std.math.inf(f64);
        }
        const slot = try a.alloc(i32, n);
        for (slot, 0..) |*s, r| s.* = @intCast(self.slotOf(r));
        self.dev[0] = try mu.put(&self.rig, ga, f16, self.q);
        self.dev[1] = try mu.put(&self.rig, ga, f16, self.k);
        self.dev[2] = try mu.put(&self.rig, ga, f16, self.v);
        self.dev[3] = try mu.ints(&self.rig, i32, self.pos);
        self.dev[4] = try mu.ints(&self.rig, i32, slot);
        if (prompt) {
            self.dev[5] = try self.rig.output(n * H * VD * 2);
            return true;
        }
        const pr = n * self.nch * H;
        self.dev[5] = try self.rig.output(pr * VD * 4);
        self.dev[6] = try self.rig.output(pr * 4);
        self.dev[7] = try self.rig.output(pr * 4);
        return true;
    }

    /// Gate 4's V: each chunk's values mirrored, scores 0 (q = 0); prompt rows mirror the first row's keys, then 0.
    fn mirrorV(self: *Attn, rng: std.Random) !void {
        const terms = try self.arena.allocator().alloc(f32, CK);
        const prompt = self.kind == .prompt;
        for (0..if (prompt) 1 else self.shape.rows) |r| for (0..self.nch) |ch| {
            const se = self.span(r, ch);
            const nk = se[1] - se[0];
            if (nk == 0) continue;
            for (0..self.hk) |kh| for (0..VD) |e| {
                attn_k.mirrored(rng, terms[0..nk]);
                for (0..nk) |j| self.v[self.vAt(r, se[0] + j, kh) + e] = mu.h16(terms[j]); // as the fp16 cache holds it
            };
        };
        if (!prompt) return;
        const p0: usize = @intCast(self.pos[0]);
        for (p0 + 1..self.shape.k) |key| for (0..self.hk) |kh| @memset(self.v[self.vAt(0, key, kh)..][0..VD], 0);
    }

    fn prepareMerge(self: *Attn, fill: g.Fill, rng: std.Random) !bool {
        const sh = self.shape;
        if (sh.n % 4 != 0) return false;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        const n = sh.rows;
        const H = sh.n;
        const nch = sh.k;
        self.nch = nch;
        const pr = n * nch * H;
        for (&self.part, [_]usize{ pr * VD, pr, pr }) |*x, l| x.* = try a.alloc(f64, l);
        for (0..n) |r| for (0..nch) |ch| for (0..H) |h| {
            const i = (r * nch + ch) * H + h;
            const empty = !self.stress and ch == nch - 1 and nch > 1 and r % 2 == 0; // a chunk past the row's keys
            self.part[1][i] = if (empty) -std.math.inf(f64) else if (self.stress) 0 else mu.uni(rng, f32, -4, 4);
            self.part[2][i] = if (empty) 0 else if (self.stress) mu.int(rng, 1, 256) else mu.uni(rng, f32, 1, 64);
            for (self.part[0][i * VD ..][0..VD]) |*x| x.* = if (empty) 0 else mu.uni(rng, f32, -16, 16);
        };
        if (self.stress) {
            const terms = try a.alloc(f32, nch);
            for (0..n) |r| for (0..H) |h| for (0..VD) |e| {
                attn_k.mirrored(rng, terms); // then halves' quarters interleaved: no order meets a term's mirror early
                const hh = nch / 2;
                const q = hh / 2;
                for (0..nch) |ch| {
                    const src = if (ch < q) ch else if (ch < 2 * q) hh + ch - q else if (ch < 3 * q) ch - q else ch;
                    self.part[0][((r * nch + ch) * H + h) * VD + e] = terms[src];
                }
            };
        }
        if (self.kind == .merge_sink) {
            self.sinks = try a.alloc(f64, H);
            for (self.sinks) |*x| x.* = if (self.stress) -std.math.inf(f64) else mu.uni(rng, f32, -4, 4); // -inf: adds 0
        }
        if (fill == .nonfinite) {
            self.part[0][3] = std.math.nan(f64); // head 0's chunk 0
            self.part[0][VD + 9] = std.math.inf(f64); // head 1: inf / L
            self.part[1][2] = std.math.inf(f64); // head 2's maximum: exp(inf - inf)
        }
        for (0..3) |i| self.dev[i] = try mu.put(&self.rig, ga, f32, self.part[i]);
        if (self.kind == .merge_sink) self.dev[3] = try mu.put(&self.rig, ga, f32, self.sinks);
        self.dev[4] = try self.rig.output(n * H * VD * 2);
        return true;
    }

    /// ops.zig's qkv, rows (attn_rows then attn_merge) and prompt launches, one at a time.
    pub fn run(self: *Attn) !void {
        const s = self.rig.s;
        const n = self.shape.rows;
        const H = self.shape.n;
        const d = self.dev;
        var a: cuda.Args = .{};
        var grid: [3]u32 = .{ @intCast(n), 1, 1 };
        var block: u32 = 128;
        var shared: u32 = 0;
        switch (self.kind) {
            .qkv => {
                for (d) |p| a.add(p);
                inline for (.{ self.len, H, self.hk }) |v| a.add(@as(c_int, @intCast(v)));
                grid[1] = @intCast(H + 2 * self.hk);
                block = 64;
            },
            .rows_slide, .rows_full => {
                const G = self.group();
                for (d[0..8]) |p| a.add(p);
                inline for (.{ self.len, self.nch, H, self.hk }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(@as(c_int, if (self.kind == .rows_slide) W else 0));
                a.add(scale);
                grid = .{ @intCast(n), @intCast(self.hk), @intCast(self.nch) };
                std.debug.assert(G <= 16);
                block = 128;
                shared = ATTN_SHARED;
                try self.f.allowDynamicShared(shared);
            },
            .merge_sink, .merge_full => {
                for (d[0..5]) |p| a.add(p);
                inline for (.{ self.nch, H }) |v| a.add(@as(c_int, @intCast(v)));
                grid[1] = @intCast(H / 4);
            },
            .prompt => {
                for (d[0..6]) |p| a.add(p);
                inline for (.{ n, self.len, H, self.hk }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(scale);
                grid = .{ @intCast((n + RB - 1) / RB), @intCast(H), 1 };
                block = 128;
                shared = ATTN_SHARED;
                try self.f.allowDynamicShared(shared);
            },
        }
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block }, .shared = shared }, s, &a);
        try s.synchronize();
    }

    pub fn count(self: *Attn) usize {
        const n = self.shape.rows;
        const H = self.shape.n;
        return switch (self.kind) {
            .qkv => n * (H + self.hk) * QK + n * self.hk * VD,
            .rows_slide, .rows_full => n * H * self.nch * (VD + 2),
            .merge_sink, .merge_full, .prompt => n * H * VD,
        };
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        return self.eval(.none, .sequential, want, null);
    }

    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.kind == .qkv) return error.NoSums;
        return self.eval(acc, order, out, null);
    }

    /// First-order propagation of fp32 roundings and 2-ulp expf, doubled; gate 4's runs (scores 0) stay exact.
    pub fn bounds(self: *Attn, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        if (self.stress) { // exact: the bound as registered, an empty chunk's -inf maximum aside
            if (self.isRows()) for (allow, want) |*l, w| if (!std.math.isFinite(w)) {
                l.* = mu.allow(.f32, 0, 0);
            };
            return;
        }
        const tmp = try self.arena.allocator().alloc(f64, want.len);
        try self.eval(.none, .sequential, tmp, allow);
    }

    fn sum(acc: c.Accum, order: g.Order, t: []const f64) f64 {
        if (acc != .none) return rows.sumIn(acc, order, t);
        var s: f64 = 0;
        for (t) |x| s += x;
        return s;
    }

    fn eval(self: *Attn, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        return switch (self.kind) {
            .qkv => self.evalQkv(out, err),
            .rows_slide, .rows_full => self.evalRows(acc, order, out, err),
            .merge_sink, .merge_full => self.evalMerge(acc, order, out, err),
            .prompt => self.evalPrompt(acc, order, out, err),
        };
    }

    /// y = x c -+ x' s on dims 0-63 (two products and a sum in fp32), dims 64-191 and v's 128 copied.
    fn evalQkv(self: *Attn, out: []f64, err: ?[]f64) void {
        const n = self.shape.rows;
        const H = self.shape.n;
        const hk = self.hk;
        for (0..n) |r| for (0..H + hk) |hh| {
            const is_q = hh < H;
            const src = if (is_q) self.q[(r * H + hh) * QK ..][0..QK] else self.k[(r * hk + hh - H) * QK ..][0..QK];
            const p: usize = @intCast(self.pos[r]);
            const at = if (is_q) (r * H + hh) * QK else n * H * QK + ((p % n) * hk + hh - H) * QK;
            for (0..QK) |t| {
                var y = src[t];
                var e: f64 = 0;
                if (t < RD) {
                    const cs = self.tab[0][p * RD / 2 + t % 32];
                    const sn = self.tab[1][p * RD / 2 + t % 32];
                    const xp = src[(t + 32) % RD];
                    y = src[t] * cs + (if (t < 32) -xp else xp) * sn;
                    e = 2 * (@abs(src[t] * cs) + @abs(xp * sn)) * 3 * mu.u;
                }
                out[at + t] = y;
                if (err) |el| el[at + t] = mu.allow(.f16, y, e);
            }
        };
        const vat = n * (H + hk) * QK;
        for (0..n) |r| for (0..hk) |kh| {
            const at = vat + ((@as(usize, @intCast(self.pos[r])) % n) * hk + kh) * VD;
            for (self.v[(r * hk + kh) * QK ..][0..VD], 0..) |x, i| {
                out[at + i] = x;
                if (err) |el| el[at + i] = mu.allow(.f16, x, 0);
            }
        };
    }

    /// Row r head h's scores over keys [lo, lo + nk) into sc; their maximum (NaN dropped) and worst score error.
    fn scores(self: *const Attn, r: usize, h: usize, lo: usize, nk: usize, sc: []f64) [2]f64 {
        const q = self.q[(r * self.shape.n + h) * QK ..][0..QK];
        const kh = h / self.group();
        var m: f64 = -std.math.inf(f64);
        var e_s: f64 = 0;
        for (sc[0..nk], 0..) |*o, j| {
            const kr = self.k[self.kAt(r, lo + j, kh)..][0..QK];
            var dot: f64 = 0;
            var mag: f64 = 0;
            for (q, kr) |x, y| {
                dot += x * y;
                mag += @abs(x * y);
            }
            o.* = dot * scale;
            m = mu.fmax(m, o.*);
            e_s = @max(e_s, (mu.gamma(QK + 2) * mag + mu.u * @abs(dot)) * scale);
        }
        return .{ m, e_s };
    }

    fn evalRows(self: *Attn, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        const n = self.shape.rows;
        const H = self.shape.n;
        const buf = try self.arena.allocator().alloc(f64, 3 * CK);
        const sc = buf[0..CK];
        const p = buf[CK .. 2 * CK];
        const t = buf[2 * CK ..];
        const pr = n * H * self.nch;
        for (0..n) |r| for (0..self.nch) |ch| for (0..H) |h| {
            const se = self.span(r, ch);
            const nk = se[1] - se[0];
            const o = (r * self.nch + ch) * H + h;
            const ms = self.scores(r, h, se[0], nk, sc);
            const m = ms[0];
            for (p[0..nk], sc[0..nk]) |*x, s| x.* = @exp(s - m);
            for (t[0..nk]) |*x| x.* = 1;
            const l = if (acc == .none) sum(.none, order, p[0..nk]) else sum(acc, order, t[0..nk]);
            const tiles: f64 = @floatFromInt((nk + BN - 1) / BN + 1 + 4); // 64-key tiles, then the four warps' folds
            const rel = 2 * ms[1] + tiles * (2 * mu.fast + 3 * mu.u);
            const rel_o = rel + 0x1p-11; // p rounded to fp16 for o, not l
            out[pr * VD + o] = if (nk == 0) -std.math.inf(f64) else m;
            out[pr * VD + pr + o] = l;
            if (err) |el| {
                el[pr * VD + o] = if (nk == 0) mu.allow(.f32, 0, 0) else mu.allow(.f32, m, 2 * ms[1]);
                el[pr * VD + pr + o] = mu.allow(.f32, l, 2 * l * (rel + mu.gamma(tiles + 6)));
            }
            for (0..VD) |e| {
                var abs: f64 = 0;
                for (t[0..nk], p[0..nk], 0..) |*x, pj, j| {
                    const vv = self.v[self.vAt(r, se[0] + j, h / self.group()) + e];
                    x.* = if (acc == .none) pj * vv else vv;
                    abs += pj * @abs(vv);
                }
                const y = sum(acc, order, t[0..nk]);
                out[o * VD + e] = y;
                if (err) |el| el[o * VD + e] = mu.allow(.f32, y, 2 * abs * (rel_o + mu.gamma(@as(f64, @floatFromInt(nk)) + tiles + 1)));
            }
        };
    }

    /// The chunks folded online (each fold an expf and three fp32 roundings), the sink last, then o / l to fp16.
    fn evalMerge(self: *Attn, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        const n = self.shape.rows;
        const H = self.shape.n;
        const nch = self.nch;
        const buf = try self.arena.allocator().alloc(f64, nch + 1);
        for (0..n) |r| for (0..H) |h| {
            var M: f64 = -std.math.inf(f64);
            var folds: f64 = 0;
            for (0..nch) |ch| {
                const m = self.part[1][(r * nch + ch) * H + h];
                if (m == -std.math.inf(f64)) continue;
                M = mu.fmax(M, m);
                folds += 1;
            }
            const sink: ?f64 = if (self.kind == .merge_sink) self.sinks[h] else null;
            if (sink) |sk| {
                M = mu.fmax(M, sk);
                folds += 1;
            }
            for (buf[0..nch], 0..) |*x, ch| {
                const i = (r * nch + ch) * H + h;
                x.* = if (self.part[1][i] == -std.math.inf(f64)) 0 else self.part[2][i] * @exp(self.part[1][i] - M);
            }
            buf[nch] = if (sink) |sk| @exp(sk - M) else 0;
            const L = sum(acc, order, buf[0 .. nch + 1]);
            const rel = folds * (mu.fast + 3 * mu.u) + mu.gamma(folds + 2);
            for (0..VD) |e| {
                var abs: f64 = 0;
                for (buf[0..nch], 0..) |*x, ch| {
                    const i = (r * nch + ch) * H + h;
                    x.* = if (self.part[1][i] == -std.math.inf(f64)) 0 else self.part[0][i * VD + e] * @exp(self.part[1][i] - M);
                    abs += @abs(x.*);
                }
                const y = sum(acc, order, buf[0..nch]) / L;
                const o = (r * H + h) * VD + e;
                out[o] = y;
                if (err) |el| el[o] = mu.allow(.f16, y, 2 * ((abs / L + @abs(y)) * rel + @abs(y) * mu.u));
            }
        };
    }

    /// A row's keys 0..p against float64's softmax, with the tensor-core kernel's error terms (64-key tiles, fp16 p).
    fn evalPrompt(self: *Attn, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        const n = self.shape.rows;
        const H = self.shape.n;
        const most = self.shape.k;
        const buf = try self.arena.allocator().alloc(f64, 3 * most);
        const sc = buf[0..most];
        const p = buf[most .. 2 * most];
        const t = buf[2 * most ..];
        for (0..n) |r| for (0..H) |h| {
            const nk: usize = @as(usize, @intCast(self.pos[r])) + 1;
            const ms = self.scores(r, h, 0, nk, sc);
            for (p[0..nk], sc[0..nk]) |*x, s| x.* = @exp(s - ms[0]);
            for (t[0..nk]) |*x| x.* = 1;
            const l = if (acc == .none) sum(.none, order, p[0..nk]) else sum(acc, order, t[0..nk]);
            const tiles: f64 = @floatFromInt((nk + BN - 1) / BN); // each folds with an expf and three roundings
            const rel = 2 * ms[1] + tiles * (2 * mu.fast + 3 * mu.u) + 0x1p-11 + // p rounded to fp16 for o, not l
                mu.gamma(@as(f64, @floatFromInt(nk)) + tiles + 4);
            for (0..VD) |e| {
                var abs: f64 = 0;
                for (t[0..nk], p[0..nk], 0..) |*x, pj, j| {
                    const vv = self.v[self.vAt(r, j, h / self.group()) + e];
                    x.* = if (acc == .none) pj * vv else vv;
                    abs += pj * @abs(vv);
                }
                const y = sum(acc, order, t[0..nk]) / l;
                const o = (r * H + h) * VD + e;
                out[o] = y;
                if (err) |el| el[o] = mu.allow(.f16, y, 2 * ((abs / l + @abs(y)) * rel + @abs(y) * mu.u));
            }
        };
    }
};
