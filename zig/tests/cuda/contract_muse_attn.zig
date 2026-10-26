//! Muse Spark's decode attention under the contract: QK prep and cache write, chunked rows, merge, against float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mu = @import("contract_muse_util.zig");
const attn_k = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;

const HD = 64;
const G = 8;
const CK = 256; // forward.CK
const eps: f32 = 1e-5;
const scale: f32 = 0.6841258108; // qk_scale_factor * 64^-0.5
const window = 2047; // sliding_window - 1

/// The 64 squares a lane pairs, then the warp by halves.
fn prepDepth(_: c.Shape) usize {
    return 2 + 5 + 1;
}

/// A score's 64 products (and q times the scale), or a chunk's keys chained for l and o.
fn rowsDepth(s: c.Shape) usize {
    return @max(HD + 2, @min(s.k, 2048) + 1);
}

/// Up to 64 chunks chained, then o_norm's 64 squares by halves.
fn mergeDepth(s: c.Shape) usize {
    return s.k + 7;
}

pub const Kind = enum { prep_rope, prep_nope, rows_slide, rows_full, merge_slide, merge_full };

const half_why = "fp16 outputs after ss^-1/2: a bf16 sum's error is a few output ulps";
const rows_rule = "NaN and infinities in K and V reach the partials as float64's two-pass softmax gives them " ++
    "(planted where no NaN enters the chunk maximum)";

pub fn spec(kind: Kind) c.Spec {
    const prep = &[_]c.Shape{ .{ .rows = 8, .n = 64, .k = HD }, .{ .rows = 1, .n = 64, .k = HD }, .{ .rows = 32, .n = 64, .k = HD } };
    const slide = &[_]c.Shape{ .{ .rows = 4, .n = 64, .k = 100000 }, .{ .rows = 1, .n = 64, .k = 1500 } };
    const full = &[_]c.Shape{ .{ .rows = 4, .n = 64, .k = 20000 }, .{ .rows = 2, .n = 64, .k = 3000 } };
    const merge = &[_]c.Shape{ .{ .rows = 8, .n = 64, .k = 16 }, .{ .rows = 1, .n = 64, .k = 64 } };
    const msum: c.Sums = .{ .bf16 = 0, .exact = false, .why = half_why };
    return switch (kind) {
        .prep_rope => .{ .name = "cuda muse muse_qk_prep (q / k norms, RoPE, the cache write)", .accum = .f32, .depth = prepDepth, .out = .f16, .tiles = .{ .{}, .{ .size = 8, .step = 8, .min = 8 }, .{} }, .served = prep, .sums = .{ .bf16 = 0, .why = half_why } },
        .prep_nope => .{ .name = "cuda muse muse_qk_prep (q / k norms, no RoPE, the cache write)", .accum = .f32, .depth = prepDepth, .out = .f16, .tiles = .{ .{}, .{ .size = 8, .step = 8, .min = 8 }, .{} }, .served = prep, .sums = .{ .bf16 = 0, .why = half_why } },
        .rows_slide => .{ .name = "cuda muse muse_attn_rows (sliding window over a ring, 256-key chunks)", .accum = .f32, .depth = rowsDepth, .out = .f32, .nan = .{ .stated = rows_rule }, .tiles = .{ .{}, .{}, .{ .size = 2048 } }, .served = slide },
        .rows_full => .{ .name = "cuda muse muse_attn_rows (full layer, each row's own chunk width)", .accum = .f32, .depth = rowsDepth, .out = .f32, .nan = .{ .stated = rows_rule }, .tiles = .{ .{}, .{}, .{ .size = 4096 } }, .served = full },
        .merge_slide => .{ .name = "cuda muse muse_attn_merge (sliding: fp16 sigmoid gate)", .accum = .f32, .depth = mergeDepth, .out = .f16, .served = merge, .sums = msum },
        .merge_full => .{ .name = "cuda muse muse_attn_merge (full: fp32 sigmoid gate)", .accum = .f32, .depth = mergeDepth, .out = .f16, .served = merge, .sums = msum },
    };
}

/// forward.zig's muse_chunk_keys: a full layer's chunk for a row at p.
fn chunkKeys(p: usize, full: usize) usize {
    var b: usize = 4096;
    while (b < p + 1) b <<= 1;
    b = @min(full, b);
    return @max(256, ((b + 63) / 64 + 255) / 256 * 256);
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
    full: usize = 0, // the full layer's context (CK-rounded)
    pos: []i32 = &.{},
    q: []f64 = &.{},
    k: []f64 = &.{}, // prep: k rows; attention: the caches [slots, len, hk, 64]
    v: []f64 = &.{},
    tab: [2][]f64 = .{ &.{}, &.{} },
    part: [3][]f64 = .{ &.{}, &.{}, &.{} }, // merge: po, pm, pl
    gate: []f64 = &.{},
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

    /// Keys [start, end) of row r's chunk ch.
    fn chunk(self: *const Attn, r: usize, ch: usize) [2]usize {
        const p: usize = @intCast(self.pos[r]);
        const lo = if (self.kind == .rows_slide) p -| window else 0;
        const ck = if (self.kind == .rows_slide) CK else chunkKeys(p, self.full);
        const start = lo + ch * ck;
        return .{ start, @max(start, @min(start + ck, p + 1)) };
    }

    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % G != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.stress = fill == .stress;
        self.hk = shape.n / G;
        const a = self.arena.allocator();
        const ga = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows;
        const H = shape.n;
        const nf = fill == .nonfinite;
        switch (self.kind) {
            .prep_rope, .prep_nope => {
                if (shape.k != HD) return false;
                self.len = n;
                self.pos = try a.alloc(i32, n);
                const base = rng.uintLessThan(u32, 4096);
                for (self.pos, 0..) |*p, r| p.* = @intCast(base + r);
                self.q = try a.alloc(f64, n * H * HD);
                self.k = try a.alloc(f64, n * self.hk * HD);
                self.v = try a.alloc(f64, n * self.hk * HD);
                for ([_][]f64{ self.q, self.k, self.v }) |x| for (x) |*o| {
                    o.* = if (self.stress) mu.int(rng, -200, 200) else mu.uni(rng, f16, -3, 3);
                };
                if (nf) {
                    self.q[3] = std.math.nan(f64);
                    self.k[HD + 7] = std.math.inf(f64);
                    self.k[2 * HD + 1] = -std.math.inf(f64);
                    self.v[5] = std.math.nan(f64);
                }
                const tl = (base + n) * HD / 2;
                for (&self.tab, 0..) |*t, i| {
                    t.* = try a.alloc(f64, tl);
                    for (t.*, 0..) |*o, j| {
                        const ang = @as(f64, @floatFromInt(j / 32)) / std.math.pow(f64, 500000, @as(f64, @floatFromInt(j % 32)) / 32);
                        o.* = mu.f32r(if (i == 0) @cos(ang) else @sin(ang));
                    }
                }
                inline for (.{ self.q, self.k, self.v }, 0..) |x, i| self.dev[i] = try mu.put(&self.rig, ga, f16, x);
                for (0..2) |i| self.dev[3 + i] = try mu.put(&self.rig, ga, f32, self.tab[i]);
                const slot = try a.alloc(i32, n);
                @memset(slot, 0);
                self.dev[5] = try mu.ints(&self.rig, i32, self.pos);
                self.dev[6] = try mu.ints(&self.rig, i32, slot);
                self.dev[7] = try self.rig.output(n * H * HD * 2);
                self.dev[8] = try self.rig.output(n * self.hk * HD * 2);
                self.dev[9] = try self.rig.output(n * self.hk * HD * 2);
            },
            .rows_slide, .rows_full => {
                if (shape.k < n) return false;
                self.pos = try a.alloc(i32, n);
                for (self.pos, 0..) |*p, r| p.* = @intCast(shape.k - 1 - r); // row r sees keys up to k - 1 - r
                self.full = (shape.k + 4096 + CK - 1) / CK * CK;
                self.len = if (self.kind == .rows_slide) window + 1 + 64 else shape.k; // a ring past the window
                self.nch = if (self.kind == .rows_slide) (window + 1 + CK - 1) / CK else blk: {
                    var most: usize = 0;
                    for (0..n) |r| {
                        const p: usize = @intCast(self.pos[r]);
                        most = @max(most, (p + 1 + chunkKeys(p, self.full) - 1) / chunkKeys(p, self.full));
                    }
                    break :blk most + 1; // one chunk past the row's keys: empty
                };
                const kv = n * self.len * self.hk * HD;
                self.q = try a.alloc(f64, n * H * HD);
                self.k = try a.alloc(f64, kv);
                self.v = try a.alloc(f64, kv);
                for (self.q) |*o| o.* = if (self.stress) 0 else mu.uni(rng, f16, -1, 1);
                for (self.k) |*o| o.* = mu.uni(rng, f16, -1, 1);
                for (self.v) |*o| o.* = mu.uni(rng, f16, -2, 2);
                if (self.stress) try self.mirrorV(rng);
                if (nf) {
                    const at = [_][4]usize{ .{ 0, 5, 0, 3 }, .{ 1, 7, 0, 10 }, .{ 2, 2, 0, 0 } };
                    const put = [_]f64{ std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) };
                    for (at, put) |p, x| if (p[0] < n) {
                        self.v[self.cacheAt(p[0], self.keyOf(p[0], p[1]), p[2]) + p[3]] = x;
                    };
                    if (n > 3) self.k[self.cacheAt(3, self.keyOf(3, 1), 0) + 4] = std.math.inf(f64);
                }
                const slot = try a.alloc(i32, n);
                for (slot, 0..) |*s, r| s.* = @intCast(r);
                self.dev[0] = try mu.put(&self.rig, ga, f16, self.q);
                self.dev[1] = try mu.put(&self.rig, ga, f16, self.k);
                self.dev[2] = try mu.put(&self.rig, ga, f16, self.v);
                self.dev[3] = try mu.ints(&self.rig, i32, self.pos);
                self.dev[4] = try mu.ints(&self.rig, i32, slot);
                const pr = n * H * self.nch;
                self.dev[5] = try self.rig.output(pr * HD * 4);
                self.dev[6] = try self.rig.output(pr * 4);
                self.dev[7] = try self.rig.output(pr * 4);
            },
            .merge_slide, .merge_full => {
                if (shape.k == 0 or shape.k > 64) return false;
                const nch = shape.k;
                self.nch = nch;
                const pr = n * H * nch;
                for (&self.part, [_]usize{ pr * HD, pr, pr }) |*x, l| x.* = try a.alloc(f64, l);
                for (0..pr) |i| {
                    const empty = !self.stress and i % nch == nch - 1 and nch > 1; // a chunk past the row's keys
                    self.part[1][i] = if (empty) -std.math.inf(f64) else if (self.stress) 0 else mu.uni(rng, f32, -4, 4);
                    self.part[2][i] = if (empty) 0 else if (self.stress) mu.int(rng, 1, 256) else mu.uni(rng, f32, 1, 64);
                    const o = self.part[0][i * HD ..][0..HD];
                    for (o) |*x| x.* = if (empty) 0 else mu.uni(rng, f32, -16, 16);
                }
                if (self.stress) for (0..n * H) |hh| {
                    const terms = try a.alloc(f32, nch);
                    for (0..HD) |e| {
                        attn_k.mirrored(rng, terms); // then halves' quarters interleaved: no order meets a term's mirror early
                        const h = nch / 2;
                        const q = h / 2;
                        for (0..nch) |ch| {
                            const src = if (ch < q) ch else if (ch < 2 * q) h + ch - q else if (ch < 3 * q) ch - q else ch;
                            self.part[0][(hh * nch + ch) * HD + e] = terms[src];
                        }
                    }
                };
                self.gate = try a.alloc(f64, n * H * HD);
                for (self.gate) |*x| x.* = mu.uni(rng, f16, -4, 4);
                if (nf) {
                    self.part[0][3] = std.math.nan(f64); // head 0's output
                    if (H > 1) self.part[0][nch * HD + 9] = std.math.inf(f64); // head 1: inf / L, then inf * 0
                    if (H > 2) self.gate[2 * HD + 5] = -std.math.inf(f64); // head 2: sigmoid 0 at one element
                }
                for (0..3) |i| self.dev[i] = try mu.put(&self.rig, ga, f32, self.part[i]);
                self.dev[3] = try mu.put(&self.rig, ga, f16, self.gate);
                self.dev[4] = try self.rig.output(n * H * HD * 2);
            },
        }
        return true;
    }

    /// The cache key row of row r's j-th visible key, and its element offset in the caches.
    fn keyOf(self: *const Attn, r: usize, j: usize) usize {
        return self.chunk(r, 0)[0] + j;
    }

    fn cacheAt(self: *const Attn, r: usize, key: usize, kh: usize) usize {
        return ((r * self.len + key % self.len) * self.hk + kh) * HD;
    }

    /// Gate 4's V: each chunk's values mirrored (integer sums near 2^20, a small result), every score 0 (q = 0).
    fn mirrorV(self: *Attn, rng: std.Random) !void {
        const a = self.arena.allocator();
        const terms = try a.alloc(f32, 4096);
        for (0..self.shape.rows) |r| for (0..self.nch) |ch| {
            const se = self.chunk(r, ch);
            const nk = se[1] - se[0];
            if (nk == 0) continue;
            for (0..self.hk) |kh| for (0..HD) |e| {
                attn_k.mirrored(rng, terms[0..nk]);
                for (0..nk) |j| self.v[self.cacheAt(r, se[0] + j, kh) + e] = mu.h16(terms[j]); // as the fp16 cache holds it
            };
        };
    }

    /// ops.attentionIo's three launches, one at a time.
    pub fn run(self: *Attn) !void {
        const s = self.rig.s;
        const n = self.shape.rows;
        const H = self.shape.n;
        var a: cuda.Args = .{};
        const d = self.dev;
        switch (self.kind) {
            .prep_rope, .prep_nope => {
                inline for (.{ d[0], d[1], d[2], d[3], d[4] }) |p| a.add(p);
                a.add(@as(c_int, @intFromBool(self.kind == .prep_rope)));
                inline for (.{ d[5], d[6], d[7], d[8], d[9] }) |p| a.add(p);
                inline for (.{ self.len, H, self.hk }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(eps);
                try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(n), .y = @intCast(H + self.hk), .z = 1 }, .block = .{ .x = 32 } }, s, &a);
            },
            .rows_slide, .rows_full => {
                inline for (.{ d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7] }) |p| a.add(p);
                a.add(@as(c_int, @intCast(self.len)));
                a.add(@as(c_int, @intCast(self.nch)));
                a.add(if (self.kind == .rows_slide) @as(c_int, CK) else -@as(c_int, @intCast(self.full)));
                a.add(@as(c_int, if (self.kind == .rows_slide) window else 0));
                inline for (.{ H, self.hk }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(scale);
                try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(n), .y = @intCast(self.hk), .z = @intCast(self.nch) }, .block = .{ .x = 256 } }, s, &a);
            },
            .merge_slide, .merge_full => {
                inline for (.{ d[0], d[1], d[2] }) |p| a.add(p);
                a.add(@as(c_int, @intCast(self.nch)));
                inline for (.{ d[3], d[4] }) |p| a.add(p);
                a.add(@as(c_int, @intCast(H)));
                a.add(eps);
                a.add(@as(c_int, @intFromBool(self.kind == .merge_full)));
                try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(n), .y = @intCast(H / 8), .z = 1 }, .block = .{ .x = 256 } }, s, &a);
            },
        }
        try s.synchronize();
    }

    pub fn count(self: *Attn) usize {
        const n = self.shape.rows;
        const H = self.shape.n;
        return switch (self.kind) {
            .prep_rope, .prep_nope => n * (H + 2 * self.hk) * HD,
            .rows_slide, .rows_full => n * H * self.nch * (HD + 2),
            .merge_slide, .merge_full => n * H * HD,
        };
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        return self.eval(.none, .sequential, want, null);
    }

    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.eval(acc, order, out, null);
    }

    /// First-order propagation of fp32 roundings and 2-ulp exp / rsqrt, doubled; gate 4's rows (scores 0) stay exact.
    pub fn bounds(self: *Attn, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        if (self.stress and self.isRows()) { // exact: the bound as registered, an empty chunk's -inf maximum aside
            for (allow, want) |*l, w| if (!std.math.isFinite(w)) {
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
            .prep_rope, .prep_nope => self.evalPrep(acc, order, out, err),
            .rows_slide, .rows_full => self.evalRows(acc, order, out, err),
            .merge_slide, .merge_full => self.evalMerge(acc, order, out, err),
        };
    }

    fn evalPrep(self: *Attn, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        const n = self.shape.rows;
        const H = self.shape.n;
        const e_rs = 0.5 * mu.gamma(@floatFromInt(prepDepth(self.shape))) + 3 * mu.u + mu.fast;
        var sq: [HD]f64 = undefined;
        for (0..n) |r| for (0..H + self.hk) |hh| {
            const is_q = hh < H;
            const src = if (is_q) self.q[(r * H + hh) * HD ..][0..HD] else self.k[(r * self.hk + hh - H) * HD ..][0..HD];
            for (&sq, src) |*o, x| o.* = x * x;
            const rs = 1 / @sqrt(sum(acc, order, &sq) / HD + eps);
            const p: usize = @intCast(self.pos[r]);
            const at = if (is_q) (r * H + hh) * HD else n * H * HD + ((p % n) * self.hk + hh - H) * HD; // kc: slot 0, row p % n
            for (0..32) |l| {
                const x0 = src[l] * rs;
                const x1 = src[l + 32] * rs;
                var y = [2]f64{ x0, x1 };
                var e = [2]f64{ @abs(x0) * (e_rs + mu.u), @abs(x1) * (e_rs + mu.u) };
                if (self.kind == .prep_rope) {
                    const cs = self.tab[0][p * 32 + l];
                    const sn = self.tab[1][p * 32 + l];
                    y = .{ x0 * cs - x1 * sn, x1 * cs + x0 * sn };
                    const m = [2]f64{ @abs(x0 * cs) + @abs(x1 * sn), @abs(x1 * cs) + @abs(x0 * sn) };
                    e = .{ m[0] * (e_rs + 3 * mu.u), m[1] * (e_rs + 3 * mu.u) };
                }
                for (0..2) |j| {
                    out[at + l + 32 * j] = y[j];
                    if (err) |el| el[at + l + 32 * j] = mu.allow(.f16, y[j], 2 * e[j]);
                }
            }
        };
        const vat = n * (H + self.hk) * HD;
        for (0..n) |r| {
            const at = vat + @as(usize, @intCast(self.pos[r])) % n * self.hk * HD;
            for (self.v[r * self.hk * HD ..][0 .. self.hk * HD], 0..) |x, i| {
                out[at + i] = x;
                if (err) |el| el[at + i] = mu.allow(.f16, x, 0);
            }
        }
    }

    fn evalRows(self: *Attn, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const n = self.shape.rows;
        const H = self.shape.n;
        const most = 4096 * 4;
        const buf = try a.alloc(f64, 3 * most);
        defer a.free(buf);
        const sc = buf[0..most];
        const p = buf[most .. 2 * most];
        const t = buf[2 * most ..];
        const pr = n * H * self.nch;
        for (0..n) |r| for (0..H) |h| for (0..self.nch) |ch| {
            const se = self.chunk(r, ch);
            const nk = se[1] - se[0];
            const o = (r * H + h) * self.nch + ch;
            const q = self.q[(r * H + h) * HD ..][0..HD];
            var m: f64 = -std.math.inf(f64);
            var e_s: f64 = 0;
            for (0..nk) |j| {
                const kr = self.k[self.cacheAt(r, se[0] + j, h / G)..][0..HD];
                var s: f64 = 0;
                var mag: f64 = 0;
                for (q, kr) |x, y| {
                    s += mu.f32r(x * scale) * y;
                    mag += @abs(x * scale * y);
                }
                sc[j] = s;
                m = @max(m, s);
                e_s = @max(e_s, mu.gamma(HD + 2) * mag);
            }
            for (p[0..nk], sc[0..nk]) |*x, s| x.* = @exp(s - m);
            for (t[0..nk]) |*x| x.* = 1;
            const l = if (acc == .none) sum(.none, order, p[0..nk]) else sum(acc, order, t[0..nk]) * @as(f64, if (nk > 0) 1 else 0);
            const tiles: f64 = @floatFromInt((nk + 31) / 32);
            const rel = 2 * e_s + tiles * (2 * mu.fast + 3 * mu.u);
            out[pr * HD + o] = if (nk == 0) -std.math.inf(f64) else m;
            out[pr * HD + pr + o] = l;
            if (err) |el| {
                el[pr * HD + o] = if (nk == 0) mu.allow(.f32, 0, 0) else mu.allow(.f32, m, 2 * e_s);
                el[pr * HD + pr + o] = mu.allow(.f32, l, 2 * l * (rel + mu.gamma(tiles + 6)));
            }
            for (0..HD) |e| {
                var abs: f64 = 0;
                for (t[0..nk], p[0..nk], 0..) |*x, pj, j| {
                    const vv = self.v[self.cacheAt(r, se[0] + j, h / G) + e];
                    x.* = if (acc == .none) pj * vv else vv;
                    abs += pj * @abs(vv);
                }
                const y = sum(acc, order, t[0..nk]);
                out[o * HD + e] = y;
                if (err) |el| el[o * HD + e] = mu.allow(.f32, y, 2 * abs * (rel + mu.gamma(@floatFromInt(nk + 1))));
            }
        };
    }

    fn evalMerge(self: *Attn, acc: c.Accum, order: g.Order, out: []f64, err: ?[]f64) !void {
        const n = self.shape.rows;
        const H = self.shape.n;
        const nch = self.nch;
        const g1 = mu.gamma(@floatFromInt(nch + 1));
        const lo: f64 = 0x1p-11;
        var tt: [64]f64 = undefined;
        var av: [HD]f64 = undefined;
        var ae: [HD]f64 = undefined;
        for (0..n * H) |hh| {
            const pm = self.part[1][hh * nch ..][0..nch];
            const pl = self.part[2][hh * nch ..][0..nch];
            var M: f64 = -std.math.inf(f64);
            for (pm) |x| M = @max(M, x);
            for (tt[0..nch], pm, pl) |*x, m, l| x.* = if (l == 0) 0 else l * @exp(m - M);
            const L = sum(acc, order, tt[0..nch]);
            for (0..HD) |e| {
                var abs: f64 = 0;
                for (tt[0..nch], 0..) |*x, ch| {
                    const f = if (pl[ch] == 0) 0 else @exp(pm[ch] - M);
                    x.* = self.part[0][(hh * nch + ch) * HD + e] * f;
                    abs += @abs(x.*);
                }
                av[e] = sum(acc, order, tt[0..nch]) / L;
                ae[e] = (abs / L + @abs(av[e])) * (g1 + mu.fast + 2 * mu.u) + @abs(av[e]) * lo;
            }
            var ss: f64 = 0;
            var se: f64 = 0;
            for (av, ae) |x, e| {
                ss += x * x;
                se += 2 * @abs(x) * e;
            }
            const rs = 1 / @sqrt(ss / HD + eps);
            const e_rs = 0.5 * (mu.gamma(7) + if (ss > 0) se / ss else 0) + 3 * mu.u + mu.fast;
            for (0..HD) |e| {
                const nv = av[e] * rs;
                const en = rs * ae[e] + @abs(nv) * (e_rs + mu.u + lo);
                const gv = self.gate[hh * HD + e];
                const s = 1 / (1 + @exp(-gv));
                const es = s * (mu.fast + 3 * mu.u + @as(f64, if (self.kind == .merge_slide) lo else 0));
                const y = nv * s;
                out[hh * HD + e] = y;
                if (err) |el| el[hh * HD + e] = mu.allow(.f16, y, 2 * (s * en + @abs(nv) * es + @abs(y) * mu.u));
            }
        }
    }
};
