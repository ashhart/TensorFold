//! Ling-3.1-flash's MLA chunks, the EXL3 grouped finish in Ling's clamp order and the Ling-order FP8 gate-up.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const lf = @import("contract_ling_fp8.zig");
const fx8 = cuda.experts.Fp8Experts;

const lw = 512; // the latent: the values
const kw = 576; // q and key rows: the latent, then the 64 roped columns
const chunk = 512;
const att_scale: f32 = 0.07216878364870322; // 1 / sqrt(192), as fp32
const had: f64 = @as(f32, 0.08838834764831845); // tf_exl3::HAD_SCALE
const soft_why = "online softmax over 512-key chunks, then the chunk merge: no input makes its sums exact in fp32";
const act_why = "silu and the clamps follow the sums; the down entry proves fp32 sums on the same stage";

pub fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

fn half(h: u16) f64 {
    return @as(f16, @bitCast(h));
}

fn toHalf(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// torch's order: min(silu(g), L) * clip(u, -L, L) for L > 0 (comparisons, a NaN kept), else silu(g) * u.
pub fn act(gv: f64, uv: f64, lim: f64) f64 {
    var s = silu(gv);
    var u = uv;
    if (lim > 0) {
        s = if (s > lim) lim else s;
        u = if (u > lim) lim else if (u < -lim) -lim else u;
    }
    return s * u;
}

/// act's error when g and u are off by eg and eu, with `mid` bf16 roundings of g, u and silu(g) on top (0 or 2^-8).
pub fn actSpread(gv: f64, uv: f64, eg0: f64, eu0: f64, mid: f64) f64 {
    const sg = 1 / (1 + @exp(-gv));
    const dg = mid * @abs(gv) + eg0;
    const ds = (@abs(sg * (1 + gv * (1 - sg))) + 0.55 * dg) * dg + (mid + 0x1p-20) * @abs(silu(gv));
    const du = mid * @abs(uv) + eu0;
    const x = 1.25 * (@abs(uv) * ds + @abs(silu(gv)) * du + ds * du);
    return if (std.math.isFinite(x)) x else 0;
}

pub fn widen(allow: []f64, extra: []const f64) void {
    for (allow, extra) |*l, x| l.* = if (std.math.isFinite(l.* + x)) l.* + x else std.math.floatMax(f64);
}

/// A key's 576 products, then a chunk's 512 keys and the merge's chunks.
fn attnDepth(s: c.Shape) usize {
    return kw + chunk + s.k / chunk + 1;
}

const chunk_shapes = [_]c.Shape{ .{ .rows = 8, .n = 32 * lw, .k = 2 * chunk }, .{ .rows = 4, .n = 64 * lw, .k = 2 * chunk }, .{ .rows = 64, .n = 32 * lw, .k = chunk } };

fn chunkSpec() c.Spec {
    return .{ .name = "cuda ling mla tf_mla_chunks_r16 (576-wide q and keys, dense over a linear cache), with tf_glm_merge", .accum = .f32, .depth = attnDepth, .ops = 8, .out = .bf16, .served = &chunk_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = soft_why } };
}

pub fn chunksSmem() u32 {
    return (16 * kw + 2 * 32 * kw + 16 * 32) * 2 + (16 * 4 + 2 * 16) * 4;
}

/// tf_mla_chunks_r16 as ops.attention calls it: rows x heads x 576 queries, keys 0..pos + r of a linear cache.
pub const Chunks = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    merge: cuda.Function,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    cache: []u16 = &.{},
    tol: []f64 = &.{},
    at: usize = 0,
    nch: usize = 0,
    heads: usize = 0,
    in: [3]u64 = @splat(0), // q, cache, the merge's counts (unread when dense)
    pos: u64 = 0,
    parts: [3]u64 = @splat(0),
    out: u64 = 0,
    split: ?[2]cuda.Function = null, // tf_mla_chunks_split_r16 and tf_mla_merge_part: the split form (splitRun)
    halves: [2]u64 = @splat(0), // the split form's caches: each rank's chunks 2 j + rank as page j
    got: [3]u64 = @splat(0), // the split form's [2] merged partials: o, m, l

    pub fn deinit(self: *Chunks) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Chunks) void {
        const a = self.rig.gpu.gpa;
        a.free(self.q);
        a.free(self.cache);
        a.free(self.tol);
        self.q = &.{};
        self.cache = &.{};
        self.tol = &.{};
    }

    pub fn kernel(self: *Chunks) g.Kernel {
        var s = chunkSpec();
        if (self.split != null) s.name = "cuda ling mla tf_mla_chunks_split_r16 and tf_mla_merge_part (a split cache's two halves), with tf_glm_merge";
        return .{ .spec = s, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Chunks);

    /// Queries and keys in [-1, 1); row r at position k - rows / 2 + r, so the rows straddle a chunk edge.
    pub fn prepare(self: *Chunks, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n % lw != 0 or shape.rows == 0 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.heads = shape.n / lw;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.at = shape.k - @min(shape.k, shape.rows / 2);
        const span = self.at + shape.rows;
        self.nch = (span + chunk - 1) / chunk;
        self.q = try a.alloc(u16, shape.rows * self.heads * kw);
        self.cache = try a.alloc(u16, span * kw);
        for (self.q) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.cache) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .nonfinite) {
            self.q[3] = c.toBf16(std.math.nan(f32));
            self.q[kw + 540] = c.toBf16(std.math.nan(f32)); // head 1, a roped column
            self.cache[9] = c.toBf16(std.math.inf(f32));
            self.cache[2 * kw + 530] = c.toBf16(-std.math.inf(f32)); // key 2, a roped column: scores only
        }
        const p: i32 = @intCast(self.at);
        const cnt: i32 = 0;
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.cache));
        self.in[2] = try self.rig.input(std.mem.asBytes(&cnt));
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        const n = self.nch * shape.rows * self.heads;
        self.parts = .{ try self.rig.scratch(n * lw * 4), try self.rig.scratch(n * 4), try self.rig.scratch(n * 4) };
        self.out = try self.rig.output(shape.rows * self.heads * lw * 2);
        if (self.split != null) try splitPrepare(self);
        return true;
    }

    pub fn run(self: *Chunks) !void {
        if (self.split) |f| return splitRun(self, f[0], f[1]);
        const rows: i64 = @intCast(self.shape.rows);
        const heads: i64 = @intCast(self.heads);
        var a: cuda.Args = .{};
        inline for (.{ self.in[0], self.in[1], self.pos, @as(u64, 0), @as(u64, 0), self.parts[0], self.parts[1], self.parts[2], rows, heads, @as(i64, 0), att_scale, @as(c_int, 0), @as(u64, 0), @as(i64, 0) }) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast((self.heads + 15) / 16), .z = @intCast(self.nch) }, .block = .{ .x = 256 }, .shared = chunksSmem() }, self.rig.s, &a);
        var m: cuda.Args = .{};
        inline for (.{ self.parts[0], self.parts[1], self.parts[2], self.out, self.in[2], rows, heads, @as(c_int, @intCast(self.nch)), @as(c_int, 0) }) |v| m.add(v);
        try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.heads) }, .block = .{ .x = 128 } }, self.rig.s, &m);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Chunks) usize {
        return self.shape.rows * self.heads * lw;
    }

    /// Row r, head h: softmax(q . k scale) . v over keys 0..at + r, sums in `acc` (null: float64), into `o`.
    fn attend(self: *Chunks, r: usize, h: usize, acc: ?c.Accum, order: g.Order, o: []f64, tol: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const n = self.at + r + 1;
        const s = try a.alloc(f64, 3 * n);
        defer a.free(s);
        const ds = s[n .. 2 * n];
        const p = s[2 * n ..];
        var terms: [kw]f64 = undefined;
        var most = -std.math.inf(f64);
        const sc: f64 = att_scale;
        for (0..n) |i| {
            var abs: f64 = 0;
            for (&terms, 0..) |*x, d| {
                x.* = bf(self.q[(r * self.heads + h) * kw + d]) * bf(self.cache[i * kw + d]);
                abs += @abs(x.*);
            }
            s[i] = (if (acc) |k| pages.sumIn(k, order, &terms) else pages.sumIn(.none, .sequential, &terms)) * sc;
            ds[i] = abs * sc * (kw + 8) * 0x1p-24;
            most = pages.maxNan(most, s[i]);
        }
        for (p, s[0..n]) |*x, v| x.* = @exp(v - most);
        const l = if (acc) |k| pages.sumIn(k, order, p) else pages.sumIn(.none, .sequential, p);
        const col = try a.alloc(f64, n);
        defer a.free(col);
        for (o, 0..) |*od, d| {
            var spread: f64 = 0;
            for (col, p, 0..) |*x, w, i| {
                const v = bf(self.cache[i * kw + d]);
                x.* = w * v;
                spread += w * @abs(v) * (0x1p-8 + ds[i]);
            }
            od.* = (if (acc) |k| pages.sumIn(k, order, col) else pages.sumIn(.none, .sequential, col)) / l;
            if (tol) |t| t[d] = @abs(od.*) * 0x1p-8 + spread / l + 1e-30;
        }
    }

    /// float64 o = sum p v / sum p, p = exp(s - max); bound: bf16 out and p (2^-8 each), 584 2^-24 scale sum |q k|.
    pub fn reference(self: *Chunks, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const at = (r * self.heads + h) * lw;
            try self.attend(r, h, null, .sequential, want[at..][0..lw], self.tol[at..][0..lw]);
        };
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Chunks, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    pub fn simulate(self: *Chunks, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const at = (r * self.heads + h) * lw;
            try self.attend(r, h, acc, order, out[at..][0..lw], null);
            for (out[at..][0..lw]) |*v| v.* = bf(c.toBf16(@floatCast(v.*)));
        };
    }
};

/// A host plan as grouped.zig's router writes it: pairs by expert, items of at most `per` pairs.
pub const Plan = struct {
    items: []i32,
    members: []i32,
    n: usize,

    pub fn make(a: std.mem.Allocator, picks: []const u8, ne: usize, per: usize) !Plan {
        var p: Plan = .{ .items = try a.alloc(i32, 3 * (picks.len + ne)), .members = try a.alloc(i32, picks.len), .n = 0 };
        var at: usize = 0;
        for (0..ne) |e| {
            const first = at;
            for (picks, 0..) |x, i| if (x == e) {
                p.members[at] = @intCast(i);
                at += 1;
            };
            var o = first;
            while (o < at) : (o += per) {
                p.items[3 * p.n ..][0..3].* = .{ @intCast(e), @intCast(o), @intCast(@min(per, at - o)) };
                p.n += 1;
            }
        }
        return p;
    }

    pub fn deinit(p: Plan, a: std.mem.Allocator) void {
        a.free(p.items);
        a.free(p.members);
    }
};

const Stage = extern struct { items: u64, counts: u64, members: u64, x: u64, su0: u64, su1: u64, sv0: u64, sv1: u64, x0: u64, x1: u64, z: u64, out: u64, k: i32, n: i32, p: i32, sk: i32, slots: i32, stride: i32, skip: i32, epi: i32, act_mode: i32, limit: f32 };

pub const FinishEntry = struct { name: []const u8, up: bool, limit: f32 = 0 };

pub const finish_entries = [_]FinishEntry{
    .{ .name = "cuda ling exl3_experts tf_exl3_plan_finish_after (gate-up: the limit 10 on silu's output, fp32 activation)", .up = true, .limit = 10 },
    .{ .name = "cuda ling exl3_experts tf_exl3_plan_finish_after (gate-up, no limit, fp32 activation)", .up = true },
    .{ .name = "cuda ling exl3_experts tf_exl3_plan_finish_after (down, fp32 out)", .up = false },
};

/// The split-K slices in turn, then the 128-point Hadamard's seven butterfly levels.
fn finishDepth(s: c.Shape) usize {
    return s.k / 128 + 7;
}

const finish_up = [_]c.Shape{ .{ .rows = 8, .n = 1280, .k = 512 }, .{ .rows = 70, .n = 640, .k = 512 } };
const finish_down = [_]c.Shape{ .{ .rows = 8, .n = 4096, .k = 128 }, .{ .rows = 70, .n = 4096, .k = 128 } };
pub const experts = 8;

fn finishSpec(e: FinishEntry) c.Spec {
    if (e.up) return .{ .name = e.name, .accum = .f32, .depth = finishDepth, .out = .bf16, .served = &finish_up, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = act_why } };
    return .{ .name = e.name, .accum = .f32, .depth = finishDepth, .ops = 2, .out = .f32, .served = &finish_down };
}

/// The grouped finish: z's slices summed, the Hadamard, times HAD_SCALE and svh; gate-up then silu, clamps, product.
pub const Finish = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    entry: FinishEntry,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    z: []f32 = &.{},
    sv: []u16 = &.{}, // [2][E][n]: gate then up (down: the first)
    picks: []u8 = &.{},
    extra: []f64 = &.{},
    items: usize = 0,
    stage: Stage = undefined,

    pub fn deinit(self: *Finish) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Finish) void {
        const a = self.rig.gpu.gpa;
        a.free(self.z);
        a.free(self.sv);
        a.free(self.picks);
        a.free(self.extra);
        self.z = &.{};
        self.sv = &.{};
        self.picks = &.{};
        self.extra = &.{};
    }

    pub fn kernel(self: *Finish) g.Kernel {
        return .{ .spec = finishSpec(self.entry), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Finish);

    fn sk(self: *const Finish) usize {
        return self.shape.k / 128;
    }

    /// z [mats * sk][pairs][n] in [-4, 4); stress (down): integers within 2^14, svh 1, every partial exact in fp32.
    pub fn prepare(self: *Finish, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n % 128 != 0 or shape.k % 128 != 0 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const mats: usize = if (self.entry.up) 2 else 1;
        self.z = try a.alloc(f32, mats * self.sk() * shape.rows * shape.n);
        self.sv = try a.alloc(u16, 2 * experts * shape.n);
        self.picks = try a.alloc(u8, shape.rows);
        const stress = fill == .stress and !self.entry.up;
        for (self.z) |*v| v.* = if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 1 << 15))) - (1 << 14)) else (rng.float(f32) * 2 - 1) * 4;
        for (self.sv) |*v| v.* = toHalf(if (stress) 1 else 0.5 + rng.float(f32));
        for (self.picks) |*p| p.* = rng.uintLessThan(u8, experts);
        if (fill == .nonfinite) {
            const n = shape.n;
            self.z[3] = std.math.nan(f32);
            self.z[n + 130] = std.math.inf(f32);
            self.z[2 * n + 5] = std.math.inf(f32);
            self.z[2 * n + 9] = -std.math.inf(f32);
            if (self.entry.up) self.z[(self.sk() * shape.rows + 3) * n + 7] = std.math.nan(f32); // an up NaN
        }
        const plan = try Plan.make(a, self.picks, experts, 64);
        defer plan.deinit(a);
        self.items = plan.n;
        const cnt: i32 = @intCast(plan.n);
        const items = try self.rig.input(std.mem.sliceAsBytes(plan.items[0 .. 3 * plan.n]));
        const counts = try self.rig.input(std.mem.asBytes(&cnt));
        const members = try self.rig.input(std.mem.sliceAsBytes(plan.members));
        const zd = try self.rig.input(std.mem.sliceAsBytes(self.z));
        const svd = try self.rig.input(std.mem.sliceAsBytes(self.sv));
        const out = try self.rig.output(shape.rows * shape.n * @as(usize, if (self.entry.up) 2 else 4));
        self.stage = .{ .items = items, .counts = counts, .members = members, .x = 0, .su0 = 0, .su1 = 0, .sv0 = svd, .sv1 = svd + experts * shape.n * 2, .x0 = 0, .x1 = 0, .z = zd, .out = out, .k = 0, .n = @intCast(shape.n), .p = @intCast(shape.rows), .sk = @intCast(self.sk()), .slots = 0, .stride = 0, .skip = -1, .epi = if (self.entry.up) 2 else 0, .act_mode = 1, .limit = self.entry.limit };
        return true;
    }

    pub fn run(self: *Finish) !void {
        var a: cuda.Args = .{};
        a.add(self.stage);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.items), .y = @intCast(self.shape.n / 128) }, .block = .{ .x = 32 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Finish) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Output o's terms: +-z over the slices of matrix m and its block's 128 columns, the Hadamard's signs.
    fn terms(self: *const Finish, m: usize, o: usize, t: []f64) void {
        const n = self.shape.n;
        const p = o / n;
        const col = o % n;
        const i = col % 128;
        const base = col - i;
        for (0..self.sk()) |s| for (0..128) |j| {
            const z: f64 = self.z[((m * self.sk() + s) * self.shape.rows + p) * n + base + j];
            t[s * 128 + j] = if (@popCount(i & j) % 2 == 0) z else -z;
        };
    }

    fn svOf(self: *const Finish, m: usize, o: usize) f64 {
        const n = self.shape.n;
        return half(self.sv[(m * experts + self.picks[o / n]) * n + o % n]);
    }

    fn one(self: *Finish, acc: c.Accum, order: g.Order, o: usize, t: []f64) [2]f64 {
        var v: [2]f64 = .{ 0, 0 };
        for (0..@as(usize, if (self.entry.up) 2 else 1)) |m| {
            self.terms(m, o, t);
            v[m] = pages.sumIn(acc, order, t) * had * self.svOf(m, o);
        }
        return v;
    }

    pub fn reference(self: *Finish, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.extra);
        self.extra = try a.alloc(f64, want.len);
        @memset(self.extra, 0);
        const t = try a.alloc(f64, self.shape.k);
        defer a.free(t);
        const lim: f64 = self.entry.limit;
        for (want, mag, self.extra, 0..) |*w, *m, *x, o| {
            var e: [2]f64 = .{ 0, 0 };
            var v: [2]f64 = .{ 0, 0 };
            for (0..@as(usize, if (self.entry.up) 2 else 1)) |mi| {
                self.terms(mi, o, t);
                var abs: f64 = 0;
                for (t) |y| abs += @abs(y);
                const sc = had * @abs(self.svOf(mi, o));
                v[mi] = pages.sumIn(.none, .sequential, t) * had * self.svOf(mi, o);
                e[mi] = c.gamma(@floatFromInt(finishDepth(self.shape)), 0x1p-24) * abs * sc + 2 * 0x1p-24 * @abs(v[mi]);
                if (!self.entry.up) m.* = abs * sc;
            }
            w.* = if (self.entry.up) act(v[0], v[1], lim) else v[0];
            if (self.entry.up) {
                m.* = 0;
                x.* = actSpread(v[0], v[1], e[0], e[1], 0);
            }
        }
    }

    pub fn bounds(self: *Finish, _: []const f64, _: []const f64, allow: []f64) !void {
        widen(allow, self.extra);
    }

    pub fn simulate(self: *Finish, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, self.shape.k);
        defer self.rig.gpu.gpa.free(t);
        for (out, 0..) |*v, o| {
            const x = self.one(acc, order, o, t);
            v.* = if (self.entry.up) act(x[0], x[1], self.entry.limit) else x[0];
        }
    }
};

/// Chunks' split form: each half of a split cache by tf_mla_chunks_split_r16 and tf_mla_merge_part, as served.
fn splitPrepare(self: *Chunks) !void {
    const a = self.rig.gpu.gpa;
    const span = self.at + self.shape.rows;
    for (0..2) |w| {
        const n_pages = (self.nch + 1 - w) / 2 + 1;
        const host = try a.alloc(u16, n_pages * chunk * kw);
        defer a.free(host);
        @memset(host, 0);
        var j: usize = 0;
        while (2 * j + w < self.nch) : (j += 1) {
            const first = (2 * j + w) * chunk;
            const n: usize = @min(chunk, span - first); // @min with the comptime 512 is a u10: n * kw would wrap
            @memcpy(host[j * chunk * kw ..][0 .. n * kw], self.cache[first * kw ..][0 .. n * kw]);
        }
        self.halves[w] = try self.rig.input(std.mem.sliceAsBytes(host));
    }
    const n = self.shape.rows * self.heads;
    self.got = .{ try self.rig.scratch(2 * n * lw * 4), try self.rig.scratch(2 * n * 4), try self.rig.scratch(2 * n * 4) };
}

fn splitRun(self: *Chunks, split: cuda.Function, part: cuda.Function) !void {
    const rows: i64 = @intCast(self.shape.rows);
    const heads: i64 = @intCast(self.heads);
    const n: u64 = @intCast(self.shape.rows * self.heads);
    for (0..2) |w| {
        const nloc = (self.nch + 1 - w) / 2;
        if (nloc > 0) {
            var a: cuda.Args = .{};
            inline for (.{ self.in[0], self.halves[w], self.pos, self.parts[0], self.parts[1], self.parts[2], rows, heads, att_scale, @as(c_int, 2), @as(c_int, @intCast(w)) }) |v| a.add(v);
            try cuda.launch.launch(split, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast((self.heads + 15) / 16), .z = @intCast(nloc) }, .block = .{ .x = 256 }, .shared = chunksSmem() }, self.rig.s, &a);
        }
        var m: cuda.Args = .{};
        inline for (.{ self.parts[0], self.parts[1], self.parts[2], self.got[0] + w * n * lw * 4, self.got[1] + w * n * 4, self.got[2] + w * n * 4, rows, heads, @as(c_int, @intCast(nloc)) }) |v| m.add(v);
        try cuda.launch.launch(part, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.heads) }, .block = .{ .x = 128 } }, self.rig.s, &m);
    }
    var m: cuda.Args = .{};
    inline for (.{ self.got[0], self.got[1], self.got[2], self.out, self.in[2], rows, heads, @as(c_int, 2), @as(c_int, 0) }) |v| m.add(v);
    try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.heads) }, .block = .{ .x = 128 } }, self.rig.s, &m);
    try self.rig.s.synchronize();
}

/// The three kinds on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    chunks: Chunks = undefined,
    split: Chunks = undefined,
    finish: [finish_entries.len]Finish = undefined,
    fp8s: [lf.fp8_entries.len]lf.Fp8 = undefined,
    exl3: cuda.Module = undefined,
    ex: fx8 = undefined,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, glue: cuda.Module, mla: cuda.Module, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        set.* = .{ .gpa = gpu.gpa };
        const f = try mla.function("tf_mla_chunks_r16");
        try f.allowDynamicShared(chunksSmem());
        set.chunks = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .merge = try glue.function("tf_glm_merge") };
        const fs = try mla.function("tf_mla_chunks_split_r16");
        try fs.allowDynamicShared(chunksSmem());
        set.split = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .merge = set.chunks.merge, .split = .{ fs, try mla.function("tf_mla_merge_part") } };
        set.exl3 = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
        errdefer set.exl3.unload();
        const fin = try set.exl3.function("tf_exl3_plan_finish_after");
        for (&set.finish, finish_entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = fin, .entry = e };
        set.ex = try fx8.load(gpu.d, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
        for (&set.fp8s, lf.fp8_entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .ex = &set.ex, .entry = e };
        try list.append(a, set.chunks.kernel());
        try list.append(a, set.split.kernel());
        for (&set.finish) |*x| try list.append(a, x.kernel());
        for (&set.fp8s) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        set.chunks.deinit();
        set.split.deinit();
        for (&set.finish) |*x| x.deinit();
        for (&set.fp8s) |*x| x.deinit();
        set.ex.unload();
        set.exl3.unload();
        set.gpa.destroy(set);
    }
};
