//! DeepSeek V4.1's folded decode launches on Metal, bit-equal to the launches they fold and checked against float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const fold = tf.dsv41_metal.fold;
const fused = tf.dsv41_metal.fused;
const hcm = tf.dsv41_metal.hc;
const xp = tf.dsv41_metal.experts;
const base = "dsv41 fold ";
const dim: usize = 5120;
const topk: usize = 6;
const experts: u32 = 384;
const eps: f32 = 1e-20;
const nowhere: u32 = 0xFFFFFFFF;

pub const Kind = enum { comb, pre, post };

/// Up to six fmas a group from zero, the three-level group tree, the shared add, then hc_post's four products and adds.
fn combDepth(_: c.Shape) usize {
    return 16;
}

/// The collapse, then a lane's chain over every 256th square, the butterfly and the eight partials.
fn normDepth(s: c.Shape) usize {
    return s.k / 256 + 12;
}

const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

pub fn spec(k: Kind, name: []const u8) c.Spec {
    const t = [_]c.Tile{ .{ .size = 16 }, .{}, .{} };
    const served = [_]c.Shape{ .{ .rows = 16, .n = 1, .k = dim }, .{ .rows = 1, .n = 1, .k = dim }, .{ .rows = 5, .n = 1, .k = dim } };
    return switch (k) {
        .comb => .{ .name = name, .accum = .f32, .depth = combDepth, .ops = 2, .out = .bf16, .tiles = t, .served = &served, .sums = .{ .why = "each group's fmas in pick order from zero, the group tree, then the shared expert" }, .replaces = "tf_dsv41_xp_combine_f32 then tf_dsv41_f_joinpost" },
        .pre => .{ .name = name, .accum = .f32, .depth = normDepth, .ops = 8, .out = .bf16, .tiles = t, .served = &served, .sums = .{ .bf16 = 0, .why = rms_why }, .replaces = "tf_dsv41_f_pre_a then tf_dsv41_f_pre_b" },
        .post => .{ .name = name, .accum = .f32, .depth = normDepth, .ops = 8, .out = .bf16, .tiles = t, .served = &served, .sums = .{ .bf16 = 0, .why = rms_why }, .replaces = "tf_dsv41_hc_post then tf_dsv41_f_pre_a and tf_dsv41_f_pre_b" },
    };
}

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

/// A float64 value as the kernel's bf16 rounding of its fp32 value.
fn bfRound(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

fn R(comptime T: type, r: rig_.Ref, extra: usize) T {
    return .{ .buf = r.buf, .off = r.off + extra };
}

const vt = g.hooks(Proj);
const plain_vt = blk: {
    var v = g.hooks(Proj);
    v.replaced = null; // a control is proven at its own gate, not against the baseline
    break :blk v;
};

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: Kind,
    ctl: u8,
    name: []const u8,
    pipes: fold.Pipelines,
    old_f: ?fused.Pipelines = null,
    old_h: ?hcm.Pipelines = null,
    old_x: ?xp.Pipelines = null,
    rows: usize = 0,
    // host inputs: streams or residual x [rows][4][dim], pre, the norm weight; hc_post's branch, post and comb
    x: []u16 = &.{},
    pre: []f32 = &.{},
    w: []f32 = &.{},
    br: []u16 = &.{},
    po: []f32 = &.{},
    cb: []f32 = &.{},
    // the MoE's: expert outputs [rows * topk][dim], where, weights, picks and the shared expert's output
    out: []u16 = &.{},
    where: []u32 = &.{},
    wts: []f32 = &.{},
    ids: []u32 = &.{},
    sh: []u16 = &.{},
    in: [11]rig_.Ref = undefined,
    outs: [4]rig_.Ref = undefined,
    tmp: [2]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: Kind, ctl: u8, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .kind = k, .ctl = ctl, .name = name, .pipes = undefined };
        if (ctl == 0) {
            p.pipes = try fold.Pipelines.load(device);
            p.old_f = try fused.Pipelines.load(device);
            if (k == .post) p.old_h = try hcm.Pipelines.load(device);
            if (k == .comb) p.old_x = try xp.Pipelines.load(a, device);
        } else {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, sources.dsv41.fold });
            defer a.free(text);
            p.pipes = try fold.Pipelines.loadSource(device, text);
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        if (p.old_f) |*b| b.deinit();
        if (p.old_h) |*b| b.deinit();
        if (p.old_x) |*b| b.deinit();
        p.pipes.deinit();
        p.rig.deinit();
        p.arena.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            else => .repeat,
        };
        var s = spec(p.kind, p.name);
        if (p.ctl == 0) return .{ .spec = s, .ctx = p, .vt = &vt, .control = null };
        s.replaces = null;
        return .{ .spec = s, .ctx = p, .vt = &plain_vt, .control = control };
    }

    /// Outputs a row: comb writes B [dim] and Y [4][dim]; the norms H, HN, HQ [dim], and post first X1 [4][dim].
    fn perRow(k: Kind) usize {
        return switch (k) {
            .comb => 5 * dim,
            .pre => 3 * dim,
            .post => 7 * dim,
        };
    }

    pub fn count(p: *Proj) usize {
        return p.rows * perRow(p.kind);
    }

    fn bf16s(a: std.mem.Allocator, n: usize, rng: std.Random, lo: f32, hi: f32) ![]u16 {
        const v = try a.alloc(u16, n);
        for (v) |*o| o.* = c.toBf16(lo + (hi - lo) * rng.float(f32));
        return v;
    }

    fn f32s(a: std.mem.Allocator, n: usize, rng: std.Random, lo: f32, hi: f32) ![]f32 {
        const v = try a.alloc(f32, n);
        for (v) |*o| o.* = lo + (hi - lo) * rng.float(f32);
        return v;
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != dim or shape.rows == 0) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        const n = shape.rows;
        p.rows = n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        p.x = try bf16s(a, n * 4 * dim, rng, -1, 1);
        for (p.x, 0..) |*v, i| if (i % 64 == 7) {
            v.* = c.toBf16(20 * (rng.float(f32) * 2 - 1));
        };
        p.po = try f32s(a, n * 4, rng, 0, 2);
        p.cb = try f32s(a, n * 16, rng, 0, 1);
        if (p.kind == .comb) {
            p.out = try bf16s(a, n * topk * dim, rng, -4, 4);
            p.sh = try bf16s(a, n * dim, rng, -2, 2);
            p.where = try a.alloc(u32, n * topk);
            p.ids = try a.alloc(u32, n * topk);
            p.wts = try f32s(a, n * topk, rng, 0.1, 2);
            for (0..n) |r| for (0..topk) |k| {
                const at = r * topk + k;
                p.ids[at] = @intCast((k * 61 + rng.uintLessThan(u32, 61)) % experts); // distinct picks, groups spread
                p.where[at] = if (rng.uintLessThan(u32, 7) == 0 and r >= 4) nowhere else @intCast((at * 5 + 3) % (n * topk));
            };
            if (fill == .nonfinite) {
                p.out[p.where[0] * dim + dim / 3] = nan;
                p.x[1 * 4 * dim + 2 * dim + dim / 2] = inf;
                p.out[p.where[2 * topk + 1] * dim + dim - 1] = ninf;
                p.sh[3 * dim] = inf;
                p.x[3 * 4 * dim + dim - 1] = ninf;
            }
            if (fill == .stress) p.stressComb(rng);
        } else {
            p.pre = try f32s(a, n * 4, rng, 0.2, 1.2);
            p.w = try f32s(a, dim, rng, 0.5, 1.5);
            p.br = try bf16s(a, n * dim, rng, -2, 2);
            if (fill == .nonfinite) {
                const at = [_][3]usize{ .{ 0, 1, dim / 3 }, .{ 1, 0, dim / 2 }, .{ 2, 3, dim - 1 }, .{ 3, 0, 0 }, .{ 3, 2, dim - 1 } };
                const put = [_]u16{ nan, inf, ninf, inf, ninf };
                for (at, put) |q, v| if (q[0] < n) {
                    p.x[(q[0] * 4 + q[1]) * dim + q[2]] = v;
                };
            }
            if (fill == .stress) try p.stressNorm(a, seed);
        }
        try p.bind();
        return true;
    }

    /// Gate 4's MoE rows: weights one, picks at three scales mirrored so the sum is one small step; post 0, comb one.
    fn stressComb(p: *Proj, rng: std.Random) void {
        @memset(p.sh, 0);
        @memset(p.po, 0);
        for (0..p.rows) |r| {
            for (0..4) |j| for (0..4) |k| {
                p.cb[r * 16 + j * 4 + k] = if (j == k) 1 else 0;
            };
            for (0..topk) |k| {
                const at = r * topk + k;
                p.wts[at] = 1;
                p.where[at] = @intCast(at);
            }
            for (0..dim) |i| for (0..3) |k| {
                const m: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
                const v = m * std.math.pow(f32, 2, @floatFromInt(14 - 4 * @as(i32, @intCast(k))));
                const step: f32 = if (k == 2) 64 else 0;
                p.out[(r * topk + k) * dim + i] = c.toBf16(v);
                p.out[(r * topk + k + 3) * dim + i] = c.toBf16(step - v);
            };
        }
    }

    /// Gate 4's norm rows: the rms stress rows in stream 0 (the others zero), pre one-hot, post 0, comb one.
    fn stressNorm(p: *Proj, a: std.mem.Allocator, seed: u64) !void {
        var r = try rows_.Rows.init(a, .rms, .{ .rows = p.rows, .n = dim, .k = dim });
        r.fill(.stress, seed);
        @memset(p.x, 0);
        @memset(p.br, 0);
        @memset(p.po, 0);
        for (0..p.rows) |row| {
            for (0..dim) |i| p.x[row * 4 * dim + i] = r.x[row * dim + i];
            for (0..4) |k| p.pre[row * 4 + k] = if (k == 0) 1 else 0;
            for (0..4) |j| for (0..4) |k| {
                p.cb[row * 16 + j * 4 + k] = if (j == k) 1 else 0;
            };
        }
        for (p.w, r.w) |*o, v| o.* = @floatCast(bf(v));
    }

    fn bind(p: *Proj) !void {
        const n = p.rows;
        const b = std.mem.sliceAsBytes;
        if (p.kind == .comb) {
            const list = [_][]const u8{ b(p.out), b(p.where), b(p.wts), b(p.ids), b(p.sh), b(p.x), b(p.po), b(p.cb) };
            for (list, 0..) |d, i| p.in[i] = try p.rig.input(d);
            p.outs[0] = try p.rig.output(n * dim * 2);
            p.outs[1] = try p.rig.output(n * 4 * dim * 2);
            p.tmp[0] = try p.rig.scratch(n * dim * 4);
            p.tmp[1] = try p.rig.scratch(n * dim * 4);
            return;
        }
        const list = [_][]const u8{ b(p.x), b(p.pre), b(p.w), b(p.br), b(p.po), b(p.cb) };
        for (list, 0..) |d, i| p.in[i] = try p.rig.input(d);
        const first: usize = if (p.kind == .post) 1 else 0;
        if (p.kind == .post) p.outs[0] = try p.rig.output(n * 4 * dim * 2);
        for (0..3) |i| p.outs[first + i] = try p.rig.output(n * dim * 2);
        p.tmp[0] = try p.rig.scratch(n * 8 * 4);
    }

    /// Rows r0 .. r0 + n of the prepared call as one launch.
    pub fn encode(p: *Proj, e: anytype, r0: usize, n: usize) !void {
        const rows: u32 = @intCast(n);
        const F = fold.Ref;
        const d2 = dim * 2;
        switch (p.kind) {
            .comb => {
                const r: fold.Routed = .{ .out = R(F, p.in[0], 0), .where = R(F, p.in[1], r0 * topk * 4), .weights = R(F, p.in[2], r0 * topk * 4), .ids = R(F, p.in[3], r0 * topk * 4), .sum = R(F, p.tmp[1], r0 * dim * 4), .topk = topk, .per_group = experts / 8 };
                fold.combpost(&p.pipes, e, r, R(F, p.in[4], r0 * d2), R(F, p.outs[0], r0 * d2), R(F, p.in[5], r0 * 4 * d2), R(F, p.in[6], r0 * 16), R(F, p.in[7], r0 * 64), R(F, p.outs[1], r0 * 4 * d2), rows, dim);
            },
            .pre => try fold.prenorm(&p.pipes, e, R(F, p.in[0], r0 * 4 * d2), R(F, p.in[1], r0 * 16), R(F, p.in[2], 0), R(F, p.outs[0], r0 * d2), R(F, p.outs[1], r0 * d2), R(F, p.outs[2], r0 * d2), rows, dim, eps),
            .post => {
                const q: fold.Post = .{ .branch = R(F, p.in[3], r0 * d2), .post = R(F, p.in[4], r0 * 16), .comb = R(F, p.in[5], r0 * 64), .x1 = R(F, p.outs[0], r0 * 4 * d2) };
                try fold.prenormPost(&p.pipes, e, R(F, p.in[0], r0 * 4 * d2), R(F, p.in[1], r0 * 16), R(F, p.in[2], 0), R(F, p.outs[1], r0 * d2), R(F, p.outs[2], r0 * d2), R(F, p.outs[3], r0 * d2), q, rows, dim, eps);
            },
        }
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        try p.encode(pass.e, 0, p.rows);
        try p.rig.finish(pass);
    }

    /// The launches it folds, on the same inputs into the same outputs: combine and joinpost, or post, pre_a, pre_b.
    pub fn replaced(p: *Proj) ![]const u8 {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows: u32 = @intCast(p.rows);
        const FF = fused.Ref;
        switch (p.kind) {
            .comb => {
                const X = xp.Ref;
                const route: xp.Route = .{ .ids = R(X, p.in[3], 0), .weights = R(X, p.in[2], 0), .slots = R(X, p.in[1], 0), .nslots = R(X, p.in[1], 0), .pairs = R(X, p.in[1], 0), .where = R(X, p.in[1], 0) };
                const call: xp.Call = .{ .rows = rows, .k = dim, .experts = experts, .topk = topk };
                xp.combine(&p.old_x.?, e, R(X, p.in[0], 0), route, R(X, p.in[2], 0), R(X, p.tmp[0], 0), true, call);
                e.barrier();
                fused.joinpost(&p.old_f.?, e, R(FF, p.tmp[0], 0), R(FF, p.in[4], 0), R(FF, p.outs[0], 0), R(FF, p.in[5], 0), R(FF, p.in[6], 0), R(FF, p.in[7], 0), R(FF, p.outs[1], 0), rows, dim);
            },
            .pre, .post => {
                const first: usize = if (p.kind == .post) 1 else 0;
                const x = if (p.kind == .post) p.outs[0] else p.in[0];
                if (p.kind == .post) {
                    const H = hcm.Ref;
                    hcm.post(&p.old_h.?, e, R(H, p.in[3], 0), R(H, p.in[0], 0), R(H, p.in[4], 0), R(H, p.in[5], 0), R(H, p.outs[0], 0), rows, dim);
                    e.barrier();
                }
                const n: fused.Norm = .{ .x = R(FF, x, 0), .pre = R(FF, p.in[1], 0), .w = R(FF, p.in[2], 0), .h = R(FF, p.outs[first], 0), .hn = R(FF, p.outs[first + 1], 0), .hq = R(FF, p.outs[first + 2], 0), .part = R(FF, p.tmp[0], 0), .dim = dim, .eps = eps };
                try fused.prenorm(&p.old_f.?, e, n, rows);
            },
        }
        try p.rig.finish(pass);
        return p.rig.outputBytes();
    }

    /// One row of the norms in float64: (X1,) H, HN and HQ (HN's value, NaN where its act_quant block is not finite).
    fn normRow(p: *Proj, r: usize, want: []f64, mag: []f64, acc: c.Accum, order: rows_.Order) !void {
        const a = p.arena.allocator();
        const post = p.kind == .post;
        const at0: usize = if (post) 4 * dim else 0;
        const hb = try a.alloc(f64, dim);
        const sq = try a.alloc(f64, dim);
        for (0..dim) |i| {
            var v: [4]f64 = undefined;
            for (0..4) |k| v[k] = bf(p.x[(r * 4 + k) * dim + i]);
            if (post) {
                const b = bf(p.br[r * dim + i]);
                var o: [4]f64 = undefined;
                for (0..4) |k| {
                    var mm: f64 = 0;
                    var ab: f64 = @abs(p.po[r * 4 + k] * b);
                    for (0..4) |j| {
                        mm += p.cb[r * 16 + j * 4 + k] * v[j];
                        ab += @abs(p.cb[r * 16 + j * 4 + k] * v[j]);
                    }
                    const y = p.po[r * 4 + k] * b + mm;
                    want[k * dim + i] = y;
                    mag[k * dim + i] = ab;
                    o[k] = bfRound(y);
                }
                v = o;
            }
            var h: f64 = 0;
            var ab: f64 = 0;
            for (0..4) |k| {
                h += p.pre[r * 4 + k] * v[k];
                ab += @abs(p.pre[r * 4 + k] * v[k]);
            }
            want[at0 + i] = h;
            mag[at0 + i] = ab;
            hb[i] = bfRound(h);
            sq[i] = hb[i] * hb[i];
        }
        var ss: f64 = 0;
        if (acc == .none) {
            for (sq) |v| ss += v;
        } else ss = rows_.sumIn(acc, order, sq);
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(dim)) + eps);
        for (0..dim) |i| {
            const hn = p.w[i] * (hb[i] * inv);
            want[at0 + dim + i] = hn;
            mag[at0 + dim + i] = @abs(hn) / 2;
        }
        var blk: usize = 0;
        while (blk < dim) : (blk += 32) {
            var bad = false;
            for (0..32) |j| bad = bad or !std.math.isFinite(want[at0 + dim + blk + j]);
            for (0..32) |j| {
                want[at0 + 2 * dim + blk + j] = if (bad) std.math.nan(f64) else want[at0 + dim + blk + j];
                mag[at0 + 2 * dim + blk + j] = mag[at0 + dim + blk + j];
            }
        }
    }

    /// One row of the MoE's in float64: B (the routed sum plus the shared expert), then hc_post of B's bf16.
    fn combRow(p: *Proj, r: usize, want: []f64, mag: []f64, acc: c.Accum, order: rows_.Order) void {
        const n = p.rows;
        for (0..dim) |i| {
            var t: [topk + 1]f64 = undefined;
            var m: usize = 0;
            var ab: f64 = 0;
            for (0..topk) |k| {
                const at = r * topk + k;
                if (p.where[at] == nowhere) continue;
                t[m] = p.wts[at] * bf(p.out[p.where[at] * dim + i]);
                ab += @abs(t[m]);
                m += 1;
            }
            t[m] = bf(p.sh[r * dim + i]);
            ab += @abs(t[m]);
            m += 1;
            var s: f64 = 0;
            if (acc == .none) {
                for (t[0..m]) |v| s += v;
            } else s = rows_.sumIn(acc, order, t[0..m]);
            want[r * dim + i] = s;
            mag[r * dim + i] = ab;
            const b = bfRound(s);
            for (0..4) |k| {
                var y: f64 = p.po[r * 4 + k] * b;
                var yab: f64 = @abs(y);
                for (0..4) |j| {
                    const v = p.cb[r * 16 + j * 4 + k] * bf(p.x[(r * 4 + j) * dim + i]);
                    y += v;
                    yab += @abs(v);
                }
                want[n * dim + (r * 4 + k) * dim + i] = y;
                mag[n * dim + (r * 4 + k) * dim + i] = yab;
            }
        }
    }

    fn compute(p: *Proj, want: []f64, mag: []f64, acc: c.Accum, order: rows_.Order) !void {
        const per = perRow(p.kind);
        for (0..p.rows) |r| switch (p.kind) {
            .comb => p.combRow(r, want, mag, acc, order),
            else => {
                // a row's outputs gathered by buffer: (X1 [rows][4][dim],) H, HN and HQ [rows][dim] each
                const tmp_w = try p.arena.allocator().alloc(f64, per);
                const tmp_m = try p.arena.allocator().alloc(f64, per);
                try p.normRow(r, tmp_w, tmp_m, acc, order);
                const parts: usize = if (p.kind == .post) 4 else 3;
                var src: usize = 0;
                var dst: usize = 0;
                for (0..parts) |q| {
                    const len: usize = if (p.kind == .post and q == 0) 4 * dim else dim;
                    @memcpy(want[dst + r * len ..][0..len], tmp_w[src..][0..len]);
                    @memcpy(mag[dst + r * len ..][0..len], tmp_m[src..][0..len]);
                    src += len;
                    dst += p.rows * len;
                }
            },
        };
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        try p.compute(want, mag, .none, .sequential);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const scratch = try p.arena.allocator().alloc(f64, p.count());
        try p.compute(out, scratch, acc, order);
    }

    /// HQ is E4M3 of HN's bf16: its relative half step 2^-4, plus HN's own allowance and a subnormal floor.
    pub fn bounds(p: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        _ = mag;
        if (p.kind == .comb) return;
        const n = p.rows * dim;
        const hq = (if (p.kind == .post) 4 * n else 0) + 2 * n;
        var blk: usize = 0;
        while (blk < n) : (blk += 32) {
            var amax: f64 = 0;
            for (want[hq + blk ..][0..32]) |v| amax = @max(amax, @abs(v));
            for (allow[hq + blk ..][0..32], want[hq + blk ..][0..32], allow[hq - n + blk ..][0..32]) |*l, v, hn| l.* = @abs(v) * 0x1p-4 + 2 * hn + amax * 0x1p-12;
        }
    }
};

/// combpost's routed sum (a scratch beside its outputs, which the stage checks read) equals Kimi's combine's bytes.
fn sumEqual(a: std.mem.Allocator, p: *Proj) !usize {
    const len = p.rows * dim * 4;
    const mine = try a.dupe(u8, p.rig.bytes(p.tmp[1], len));
    defer a.free(mine);
    _ = try p.replaced();
    if (std.mem.eql(u8, mine, p.rig.bytes(p.tmp[0], len))) return 0;
    std.debug.print("FAIL {s}: its routed sum differs from tf_dsv41_xp_combine_f32's\n", .{p.name});
    return 1;
}

/// Every registration and its four controls, plus the row-invariance probe that splits a launch into pieces.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { k: Kind, ctl: u8, name: []const u8 };
    const list = [_]E{
        .{ .k = .comb, .ctl = 0, .name = base ++ "combpost tf_dsv41_o_combpost" },
        .{ .k = .pre, .ctl = 0, .name = base ++ "prenorm tf_dsv41_o_prenorm" },
        .{ .k = .post, .ctl = 0, .name = base ++ "postnorm tf_dsv41_o_postnorm" },
        .{ .k = .comb, .ctl = 1, .name = base ++ "control: combpost non-finite inputs read as zero" },
        .{ .k = .comb, .ctl = 2, .name = base ++ "control: combpost drops the last residual stream" },
        .{ .k = .comb, .ctl = 3, .name = base ++ "control: combpost group totals in fp16" },
        .{ .k = .comb, .ctl = 4, .name = base ++ "control: combpost writes past its streams" },
        .{ .k = .pre, .ctl = 1, .name = base ++ "control: prenorm non-finite inputs read as zero" },
        .{ .k = .pre, .ctl = 2, .name = base ++ "control: prenorm swaps two pre weights" },
        .{ .k = .pre, .ctl = 3, .name = base ++ "control: prenorm sum of squares in fp16" },
        .{ .k = .pre, .ctl = 4, .name = base ++ "control: prenorm writes past its act_quant" },
        .{ .k = .post, .ctl = 1, .name = base ++ "control: postnorm non-finite inputs read as zero" },
        .{ .k = .post, .ctl = 2, .name = base ++ "control: postnorm swaps two pre weights" },
        .{ .k = .post, .ctl = 3, .name = base ++ "control: postnorm sum of squares in fp16" },
        .{ .k = .post, .ctl = 4, .name = base ++ "control: postnorm writes past its act_quant" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.k, entry.ctl, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    /// Launches of 1, 5 and 7 rows a launch equal one launch's bytes, every buffer: no sum crosses a row.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var kernels: usize = 0;
        for (s.probes.items) |p| {
            if (p.ctl != 0) continue;
            kernels += 1;
            const shape = c.Shape{ .rows = 17, .n = 1, .k = dim };
            if (!try p.prepare(shape, .random, seed)) return error.Refused;
            try p.run();
            const whole = try s.a.dupe(u8, try p.rig.outputBytes());
            defer s.a.free(whole);
            if (p.kind == .comb) bad += try sumEqual(s.a, p);
            for ([_]usize{ 1, 5, 7 }) |step| {
                const outs: usize = if (p.kind == .post) 4 else if (p.kind == .pre) 3 else 2;
                for (p.outs[0..outs]) |o| @memset(o.buf.contents()[o.off..][0 .. o.buf.length() - o.off - rig_.guard], 0x55);
                const pass = p.rig.begin();
                var r0: usize = 0;
                while (r0 < shape.rows) : (r0 += step) {
                    try p.encode(pass.e, r0, @min(step, shape.rows - r0));
                    launches += 1;
                }
                try p.rig.finish(pass);
                if (!std.mem.eql(u8, whole, try p.rig.outputBytes())) {
                    bad += 1;
                    std.debug.print("FAIL row invariance {s}: {d} rows a launch differs from one launch\n", .{ p.name, step });
                }
            }
        }
        if (kernels == 0) return 0;
        std.debug.print("{s} row invariance {s}: {d} kernels at 1, 5 and 7 rows a launch ({d} launches) byte-equal to one launch, {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }
};
