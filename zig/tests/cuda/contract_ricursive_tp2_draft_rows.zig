//! Ripple's drafter norms, SwiGLU, convolutions and head norms (ricursive_tp2_draft.cu) against float64's ranges.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");
const glue = @import("contract_ricursive_tp2.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Function = cuda.Function;
const elementwise = glue.elementwise;

pub const eps: f32 = 1e-6;
pub const theta: f32 = 10000;
pub const block = 16; // a draft block's rows
const pow_ulps = 8; // powf: CUDA's table gives fewer; taken wider

/// A hold's stream as the attention and head kernels read it (tp2_draft_load.zig's Seg).
pub const Seg = extern struct { ring: u64, begin: i64, position: i64 };

/// Served row counts at width n (k: the reduction).
pub fn at(comptime rs: []const usize, comptime n: usize, comptime k: usize) [rs.len]c.Shape {
    var out: [rs.len]c.Shape = undefined;
    for (rs, &out) |r, *s| s.* = .{ .rows = r, .n = n, .k = k };
    return out;
}

pub const hold = [_]usize{ 16, 32, 48, 64 }; // one to four streams' blocks
const absorb = [_]usize{ 16, 1, 3, 77, 128 }; // context rows into the rings, 128 at a time
const drafts = [_]usize{ 15, 30, 45, 60 }; // each block's rows past its first

/// One launch of `threads` a block over `grid`, with `shared` dynamic bytes.
pub fn go(f: Function, s: cuda.Stream, grid: [3]usize, threads: u32, shared: u32, args: anytype) !void {
    var a: cuda.Args = .{};
    inline for (args) |v| a.add(v);
    const cfg: cuda.launch.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads }, .shared = shared };
    try cuda.launch.launch(f, cfg, s, &a);
}

fn halfOf(v: f64) f64 {
    return @floatCast(@as(f16, @floatCast(v)));
}

/// To fp16, nearest even (monotone, as bf16).
fn hf(a: Iv) Iv {
    return .{ .m = halfOf(a.m), .lo = halfOf(a.lo), .hi = halfOf(a.hi) };
}

/// __expf: within 2 + floor(1.2 |x|) ulps (CUDA's table: 1.173).
pub fn fastExp(a: Iv) Iv {
    const rel = (2 + @floor(1.2 * @max(@abs(a.lo), @abs(a.hi)))) * 0x1p-23;
    return .{ .m = @exp(a.m), .lo = @exp(a.lo) * (1 - rel) - 0x1p-149, .hi = @exp(a.hi) * (1 + rel) + 0x1p-149 };
}

fn bfv(h: u16) Iv {
    return Iv.of(ref.bf(h));
}

/// A thread's fp32 chain over the row (one value per 256), the warp's 5 shuffles and the block's 8 partials.
fn normDepth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 5 + 8;
}

/// OUT row j = bf16(x / rms(x)) * w rounded to bf16 or fp16; with blocks, row j reads each block's rows past its first.
pub const Rms = struct {
    f: Function,
    half: bool = false, // fp16 out: the head's rows
    blocks: bool = false, // each block's first row dropped
    ctl: bool = false, // control: x / rms left unrounded
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []u16 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};
    const all_rows = at(&absorb ++ [_]usize{64}, 5120, 5120);
    const draft_rows = at(&drafts, 5120, 5120);

    pub fn spec(self: *const Rms) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2_draft tf_rtp2d_rms with x / rms unrounded" else if (self.half) "cuda ricursive_tp2_draft tf_rtp2d_rms (fp16 out, each block's drafts: the head's rows)" else if (self.blocks) "cuda ricursive_tp2_draft tf_rtp2d_rms (bf16 out, each block's drafts: the selector's rows)" else "cuda ricursive_tp2_draft tf_rtp2d_rms (bf16 out, every row: the layers' norms)";
        return .{ .name = name, .accum = .f32, .depth = normDepth, .ops = 4, .out = if (self.half) .f16 else .bf16, .tiles = .{ .{}, elementwise, .{} }, .served = if (self.blocks) &draft_rows else &all_rows };
    }

    pub fn control(self: *const Rms) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    fn src(self: *const Rms, j: usize) usize {
        return if (self.blocks) j / (block - 1) * block + 1 + j % (block - 1) else j;
    }

    pub fn free(self: *Rms, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    pub fn prepare(self: *Rms, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        const in = self.src(shape.rows - 1) + 1;
        self.x = try ref.f32s(a, rng, in * n, -2, 2);
        for (self.x, 0..) |*v, i| if (i % 64 == 7) {
            v.* *= 20;
        };
        if (fill == .stress) for (self.x) |*v| { // 97 values, squares to 2,304: 5,120 of them stay exact in fp32
            v.* = ref.int(rng, -48, 48);
        };
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |v, r| if (r < shape.rows) {
            self.x[self.src(r) * n + n / 2] = v;
        };
        self.w = try ref.bfs(a, rng, n, 0.5, 1.5);
        self.dev = .{ try ref.input(rig, self.x), try ref.input(rig, self.w) };
        self.out = try rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(self: *Rms, rig: *rig_.Rig) !void {
        const n: i32 = @intCast(self.shape.n);
        const kind: i32 = @intFromBool(self.half);
        const blk: i32 = if (self.blocks) block else 0;
        try go(self.f, rig.s, .{ self.shape.rows, 1, 1 }, 256, 0, .{ self.dev[0], self.dev[1], self.out, n, eps, kind, blk });
    }

    pub fn count(self: *const Rms) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Rms, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const sq = try a.alloc(Iv, n);
        defer a.free(sq);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        for (0..self.shape.rows) |j| {
            const x = self.x[self.src(j) * n ..][0..n];
            for (sq, x) |*q, v| q.* = Iv.of(v).mul(Iv.of(v));
            const total = s.sum(sq, normDepth(self.shape), scratch);
            const inv = total.div(Iv.of(@floatFromInt(n))).add(Iv.of(eps)).rsqrt();
            for (out[j * n ..][0..n], x, self.w) |*o, v, w| {
                const y = Iv.of(v).mul(inv).bf().mul(bfv(w));
                o.* = if (self.half) hf(y) else y.bf();
            }
        }
    }
};

/// OUT = bf16(g / (1 + __expf(-g)) * u), fp32 in: g in [-20, 20), u in [-8, 8).
pub const Silu = struct {
    f: Function,
    ctl: bool = false, // control: silu(g) rounded to bf16 before the up
    shape: c.Shape = undefined,
    gv: []f32 = &.{},
    u: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    const served = at(&hold, 20480, 1); // a rank's FFN width

    pub fn spec(self: *const Silu) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2_draft tf_rtp2d_silu with bf16(silu(g)) before the up" else "cuda ricursive_tp2_draft tf_rtp2d_silu (fp32 in, __expf, one bf16 rounding)";
        return .{ .name = name, .accum = .none, .depth = ref.noSums, .out = .bf16, .tiles = .{ .{}, elementwise, .{} }, .served = &served };
    }

    pub fn control(self: *const Silu) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Silu, a: std.mem.Allocator) void {
        a.free(self.gv);
        a.free(self.u);
        self.gv = &.{};
        self.u = &.{};
    }

    pub fn prepare(self: *Silu, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * shape.n;
        self.gv = try ref.f32s(a, rng, n, -20, 20);
        self.u = try ref.f32s(a, rng, n, -8, 8);
        if (fill == .nonfinite and n > 4) {
            self.gv[0] = ref.nan;
            self.gv[1] = ref.inf;
            self.gv[2] = -ref.inf; // -Inf / (1 + Inf): NaN
            self.u[3] = ref.inf;
        }
        self.dev = .{ try ref.input(rig, self.gv), try ref.input(rig, self.u) };
        self.out = try rig.output(n * 2);
        return true;
    }

    pub fn run(self: *Silu, rig: *rig_.Rig) !void {
        const n = self.shape.rows * self.shape.n;
        try go(self.f, rig.s, .{ (n + 255) / 256, 1, 1 }, 256, 0, .{ self.dev[0], self.dev[1], self.out, @as(i64, @intCast(n)) });
    }

    pub fn count(self: *const Silu) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Silu, _: std.mem.Allocator, _: Sums, out: []Iv) !void {
        for (out, self.gv, self.u) |*o, gv, u| {
            const x = Iv.of(gv);
            o.* = x.div(Iv.of(1).add(fastExp(x.neg()))).mul(Iv.of(u)).bf();
        }
    }
};

fn convDepth(_: c.Shape) usize {
    return 4;
}

/// y[t, c] = sum over lags l <= t's place in its block of (base[l, c] + dyn[t, l, c / 16]) x[t - l, c], fp32.
pub const Conv = struct {
    f: Function,
    second: bool, // conv_out: the second branch over fp32 rows plus the residual; else conv_in over bf16 rows
    ctl: bool = false, // control: the other branch's weights
    shape: c.Shape = undefined,
    xs: []f32 = &.{}, // conv_in reads these as bf16
    dyn: []f32 = &.{},
    base: []u16 = &.{},
    res: []f32 = &.{},
    dev: [4]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};
    const lags = 4;
    const group = 16;
    const served = at(&hold, 5120, lags);

    pub fn spec(self: *const Conv) c.Spec {
        const name = if (self.ctl and self.second) "control: cuda ricursive_tp2_draft tf_rtp2d_conv_out on the first branch's weights" else if (self.ctl) "control: cuda ricursive_tp2_draft tf_rtp2d_conv_in on the second branch's weights" else if (self.second) "cuda ricursive_tp2_draft tf_rtp2d_conv_out (second branch, fp32 rows plus the residual)" else "cuda ricursive_tp2_draft tf_rtp2d_conv_in (first branch, bf16 rows in and out)";
        return .{ .name = name, .accum = .f32, .depth = convDepth, .out = if (self.second) .f32 else .bf16, .tiles = .{ .{ .size = block, .step = 1, .min = 1 }, .{}, .{} }, .served = &served };
    }

    pub fn control(self: *const Conv) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Conv, a: std.mem.Allocator) void {
        for ([_]*[]f32{ &self.xs, &self.dyn, &self.res }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        a.free(self.base);
        self.base = &.{};
    }

    fn slot(self: *const Conv, t: usize, l: usize, grp: usize) usize {
        const groups = self.shape.n / group;
        return t * (2 * lags * groups) + (@as(usize, @intFromBool(self.second)) * lags + l) * groups + grp;
    }

    pub fn prepare(self: *Conv, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % group != 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        const groups = n / group;
        self.xs = try ref.f32s(a, rng, shape.rows * n, -4, 4);
        if (!self.second) for (self.xs) |*v| {
            v.* = @floatCast(ref.bf(c.toBf16(v.*)));
        };
        self.dyn = try ref.f32s(a, rng, shape.rows * 2 * lags * groups, -0.5, 0.5);
        self.base = try ref.bfs(a, rng, 2 * lags * n, -1, 1);
        self.res = try ref.f32s(a, rng, shape.rows * n, -8, 8);
        if (fill == .stress) { // lags A, -A, B, e - B: big odd partials past fp16's 2048, a small even sum
            @memset(self.xs, 1);
            @memset(self.base, 0);
            @memset(self.res, 0);
            for (0..shape.rows) |t| for (0..groups) |grp| {
                const A: f32 = @floatFromInt(2049 + 2 * rng.uintLessThan(u32, 1024));
                const B: f32 = @floatFromInt(2049 + 2 * rng.uintLessThan(u32, 1024));
                const e: f32 = @floatFromInt(2 * @as(i32, @intCast(rng.uintLessThan(u32, 9))) - 8);
                for ([_]f32{ A, -A, B, e - B }, 0..) |v, l| self.dyn[self.slot(t, l, grp)] = v;
            };
        }
        if (fill == .nonfinite and shape.rows > 3 and n > 70) {
            self.xs[n + 5] = ref.nan;
            self.dyn[self.slot(3, 0, 2)] = ref.inf;
            self.base[(@as(usize, @intFromBool(self.second)) * lags + 1) * n + 70] = c.toBf16(-ref.inf);
        }
        const xb = try a.alloc(u16, if (self.second) 0 else self.xs.len);
        defer a.free(xb);
        for (xb, self.xs[0..xb.len]) |*h, v| h.* = c.toBf16(v);
        self.dev = .{ if (self.second) try ref.input(rig, self.xs) else try ref.input(rig, xb), try ref.input(rig, self.dyn), try ref.input(rig, self.base), if (self.second) try ref.input(rig, self.res) else 0 };
        self.out = try rig.output(shape.rows * n * @as(usize, if (self.second) 4 else 2));
        return true;
    }

    pub fn run(self: *Conv, rig: *rig_.Rig) !void {
        const grid = [3]usize{ self.shape.rows, (self.shape.n + 255) / 256, 1 };
        const d: i32 = @intCast(self.shape.n);
        const k: [3]i32 = .{ group, lags, block };
        if (self.second) return go(self.f, rig.s, grid, 256, 0, .{ self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.out, d, k[0], k[1], k[2] });
        try go(self.f, rig.s, grid, 256, 0, .{ self.dev[0], self.dev[1], self.dev[2], self.out, d, k[0], k[1], k[2] });
    }

    pub fn count(self: *const Conv) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Conv, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const br = @as(usize, @intFromBool(self.second)) * lags;
        var terms: [lags]Iv = undefined;
        var scratch: [lags]f64 = undefined;
        for (0..self.shape.rows) |t| for (0..n) |col| {
            const k = @min(lags, t % block + 1);
            for (terms[0..k], 0..) |*term, l| {
                const wgt = bfv(self.base[(br + l) * n + col]).add(Iv.of(self.dyn[self.slot(t, l, col / group)]));
                term.* = wgt.mul(Iv.of(self.xs[(t - l) * n + col]));
            }
            const total = s.sum(terms[0..k], lags, &scratch);
            out[t * n + col] = if (self.second) total.add(Iv.of(self.res[t * n + col])) else total.bf();
        };
    }
};

/// RMSNorm over 128 (lane chains of 4, 5 shuffles), then bf16(bf16(x inv) w).
fn headDepth(_: c.Shape) usize {
    return 4 + 5;
}

/// One head normed and, unless `rope` is false, rotated at `pos` (rotate-half, theta 10,000), into `out` [128].
fn headRow(x: []const f32, w: []const u16, pos: i64, rope: bool, s: Sums, out: []Iv) void {
    var sq: [128]Iv = undefined;
    var scratch: [128]f64 = undefined;
    var nrm: [128]Iv = undefined;
    for (&sq, x[0..128]) |*q, v| q.* = Iv.of(v).mul(Iv.of(v));
    const inv = s.sum(&sq, 9, &scratch).div(Iv.of(128)).add(Iv.of(eps)).rsqrt();
    for (&nrm, x[0..128], w[0..128]) |*o, v, wv| o.* = Iv.of(v).mul(inv).bf().mul(bfv(wv)).bf();
    for (out[0..128], 0..) |*o, d| {
        if (!rope) {
            o.* = nrm[d];
            continue;
        }
        const e: f32 = -@as(f32, @floatFromInt(d % 64)) / 64.0;
        const p = std.math.pow(f64, theta, e);
        const r = pow_ulps * 0x1p-23;
        const angle = Iv.of(@floatFromInt(pos)).mul(.{ .m = p, .lo = p * (1 - r), .hi = p * (1 + r) });
        const cs = nrm[d].mul(angle.trig(false));
        const sn = nrm[d ^ 64].mul(angle.trig(true));
        o.* = (if (d < 64) cs.sub(sn) else cs.add(sn)).bf();
    }
}

/// Block rows' heads normed (roped at the stream's position + row) into bf16 [rows, heads, 128]; x in [-3, 3).
pub const Heads = struct {
    f: Function,
    rope: bool = true,
    ctl: bool = false, // control: rotated the other way
    served: []const c.Shape,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []u16 = &.{},
    segs: []Seg = &.{},
    dev: [3]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};
    pub const q_rows = at(&hold, 40 * 128, 128); // a rank's 40 query heads
    pub const k_rows = at(&hold, 4 * 128, 128); // and 4 key heads

    pub fn spec(self: *const Heads) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2_draft tf_rtp2d_heads rotated the other way" else if (!self.rope) "cuda ricursive_tp2_draft tf_rtp2d_heads (query heads normed, no rotation: the full layer)" else if (self.served[0].n > 512) "cuda ricursive_tp2_draft tf_rtp2d_heads (40 query heads normed and roped)" else "cuda ricursive_tp2_draft tf_rtp2d_heads (4 key heads normed and roped)";
        return .{ .name = name, .accum = .f32, .depth = headDepth, .ops = 6, .out = .bf16, .served = self.served };
    }

    pub fn control(self: *const Heads) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *Heads, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        a.free(self.segs);
        self.x = &.{};
        self.w = &.{};
        self.segs = &.{};
    }

    pub fn prepare(self: *Heads, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 128 != 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.n;
        self.x = try ref.f32s(a, rng, shape.rows * n, -3, 3);
        if (fill == .stress) for (self.x) |*v| {
            v.* = ref.int(rng, -16, 16);
        };
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |v, r| if (r < shape.rows) {
            self.x[r * n + (r * 128 + 5 + 20 * r) % n] = v;
        };
        self.w = try ref.bfs(a, rng, 128, 0.5, 1.5);
        self.segs = try a.alloc(Seg, (shape.rows + block - 1) / block);
        for (self.segs) |*sg| sg.* = .{ .ring = 0, .begin = 0, .position = rng.uintLessThan(u32, 20000) };
        self.dev = .{ try ref.input(rig, self.x), try ref.input(rig, self.w), try ref.input(rig, self.segs) };
        self.out = try rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(self: *Heads, rig: *rig_.Rig) !void {
        const nh = self.shape.n / 128;
        const args = .{ self.dev[0], @as(i32, @intCast(self.shape.n)), self.dev[1], self.out, @as(i32, @intCast(nh)), self.dev[2], @as(i32, block), eps, theta, @as(i32, @intFromBool(self.rope)) };
        try go(self.f, rig.s, .{ self.shape.rows * nh, 1, 1 }, 32, 0, args);
    }

    pub fn count(self: *const Heads) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Heads, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        for (0..self.shape.rows) |m| {
            const pos = self.segs[m / block].position + @as(i64, @intCast(m % block));
            for (0..self.shape.n / 128) |h| {
                const o = m * self.shape.n + h * 128;
                headRow(self.x[o..][0..128], self.w, pos, self.rope, s, out[o..][0..128]);
            }
        }
    }
};

/// Context rows at pos0 + r: keys normed (and roped) into ring K, values bf16 into ring V, at pos % rows.
pub const KvPut = struct {
    f: Function,
    rope: bool = true,
    ctl: bool = false, // control: values rounded toward zero
    shape: c.Shape = undefined,
    k: []f32 = &.{},
    v: []f32 = &.{},
    w: []u16 = &.{},
    pos0: i64 = 0,
    dev: [3]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub const sums = {};
    const served = at(&absorb, 4 * 128, 128); // a rank's 4 key heads

    pub fn spec(self: *const KvPut) c.Spec {
        const name = if (self.ctl) "control: cuda ricursive_tp2_draft tf_rtp2d_kv_put, values toward zero" else if (self.rope) "cuda ricursive_tp2_draft tf_rtp2d_kv_put (keys normed and roped, values bf16, into a ring)" else "cuda ricursive_tp2_draft tf_rtp2d_kv_put (keys normed, no rotation: the full layer)";
        return .{ .name = name, .accum = .f32, .depth = headDepth, .ops = 6, .out = .bf16, .served = &served };
    }

    pub fn control(self: *const KvPut) ?g.Gate {
        return if (self.ctl) .shapes else null;
    }

    pub fn free(self: *KvPut, a: std.mem.Allocator) void {
        a.free(self.k);
        a.free(self.v);
        a.free(self.w);
        self.k = &.{};
        self.v = &.{};
        self.w = &.{};
    }

    /// The ring holds as many rows as the call (every slot written once), from pos0 on, wrapping.
    pub fn prepare(self: *KvPut, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 128 != 0) return false;
        self.free(a);
        self.shape = shape;
        const n = shape.rows * shape.n;
        self.k = try ref.f32s(a, rng, n, -3, 3);
        self.v = try ref.f32s(a, rng, n, -4, 4);
        if (fill == .stress) for (self.k) |*x| {
            x.* = ref.int(rng, -16, 16);
        };
        if (fill == .nonfinite) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, 0..) |x, r| if (r < shape.rows) {
            self.k[r * shape.n + (r * 128 + 9) % shape.n] = x;
            self.v[r * shape.n + (r * 128 + 70) % shape.n] = x;
        };
        self.w = try ref.bfs(a, rng, 128, 0.5, 1.5);
        self.pos0 = rng.uintLessThan(u32, 30000);
        self.dev = .{ try ref.input(rig, self.k), try ref.input(rig, self.v), try ref.input(rig, self.w) };
        self.out = .{ try rig.output(n * 2), try rig.output(n * 2) };
        return true;
    }

    pub fn run(self: *KvPut, rig: *rig_.Rig) !void {
        const kv = self.shape.n / 128;
        const rr: i32 = @intCast(self.shape.rows);
        const args = .{ self.dev[0], self.dev[1], self.dev[2], self.out[0], self.out[1], @as(i32, @intCast(kv)), self.pos0, rr, eps, theta, @as(i32, @intFromBool(self.rope)) };
        try go(self.f, rig.s, .{ self.shape.rows * kv, 1, 1 }, 32, 0, args);
    }

    pub fn count(self: *const KvPut) usize {
        return 2 * self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const KvPut, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const vals = out[self.shape.rows * n ..];
        for (0..self.shape.rows) |r| {
            const pos = self.pos0 + @as(i64, @intCast(r));
            const slot: usize = @intCast(@mod(pos, @as(i64, @intCast(self.shape.rows))));
            for (0..n / 128) |h| {
                const o = r * n + h * 128;
                headRow(self.k[o..][0..128], self.w, pos, self.rope, s, out[slot * n + h * 128 ..][0..128]);
                for (vals[slot * n + h * 128 ..][0..128], self.v[o..][0..128]) |*x, v| x.* = Iv.of(v).bf();
            }
        }
    }
};
