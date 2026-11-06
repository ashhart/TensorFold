//! tf_uocr_block_keys, tf_uocr_silu_mul and tf_uocr_combine under the contract: the router's block keys and MLP glue.
const std = @import("std");
const ud = @import("contract_uocr_decoder.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = ud.c;
const g = ud.g;
const mi = ud.mi;
const Range = mi.Range;
const Ends = mi.Ends;
const bf = mi.bf;
const H = ud.H;
const D = ud.D;
const B = 32;

// --- tf_uocr_block_keys: a block's 32 router keys weighted by softmax(clamp(k . hc / sqrt(128))) --------------------

/// The weighted sum's chain over a block's 32 keys.
fn keysDepth(_: c.Shape) usize {
    return B;
}

/// The first block the test writes (its inputs and outputs start there).
const keys_c0 = 5;

pub const BlockKeys = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    /// [blocks x 32, 128] router keys, then head_cls [128]
    rk: []u16 = &.{},
    hc: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
    allow: []f64 = &.{},

    /// Rows: blocks; n: the key width (128); k: a block's keys (32).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = keysDepth,
            .out = .f32,
            .served = &.{ .{ .rows = 64, .n = D, .k = B }, .{ .rows = 1, .n = D, .k = B }, .{ .rows = 4096, .n = D, .k = B } },
        };
    }

    pub fn deinit(t: *BlockKeys) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *BlockKeys) void {
        const a = t.rig.gpu.gpa;
        a.free(t.rk);
        a.free(t.hc);
        a.free(t.allow);
        t.rk = &.{};
        t.hc = &.{};
        t.allow = &.{};
    }
    pub fn kernel(t: *BlockKeys) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(BlockKeys);

    pub fn prepare(t: *BlockKeys, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != D or shape.k != B) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        t.rk = try a.alloc(u16, shape.rows * B * D);
        t.hc = try a.alloc(u16, D);
        t.allow = try a.alloc(f64, shape.rows * D);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.rk) |*v| v.* = attn.normal(rng);
        for (t.hc) |*v| v.* = c.toBf16(rng.floatNorm(f32) * 3);
        if (fill == .stress) {
            // head_cls 0: every weight is 1/32; each column mirrored over a block's keys
            @memset(t.hc, 0);
            var col: [B]f32 = undefined;
            for (0..shape.rows) |b| for (0..D) |d| {
                attn.mirrored(rng, &col);
                for (col, 0..) |v, j| t.rk[(b * B + j) * D + d] = c.toBf16(v);
            };
        }
        if (fill == .nonfinite) {
            const at = [_][3]usize{ .{ 0, 3, 7 }, .{ 1, 9, 100 }, .{ 2, 31, 0 } };
            for (at, [_]u16{ mi.nan16, mi.inf16, mi.ninf16 }) |p, v| if (p[0] < shape.rows) {
                t.rk[(p[0] * B + p[1]) * D + p[2]] = v;
            };
        }
        t.ins[0] = try mi.input(&t.rig, t.rk);
        t.ins[1] = try mi.input(&t.rig, t.hc);
        t.out = try t.rig.output(shape.rows * D * 4);
        return true;
    }

    pub fn run(t: *BlockKeys) !void {
        const rk = t.ins[0] - keys_c0 * B * D * 2;
        const kb = t.out - keys_c0 * D * 4;
        try t.k.run("block_keys", t.rig.s, .{ t.shape.rows, 1, 1 }, 32, .{ rk, t.ins[1], kb, @as(c_int, keys_c0) });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *BlockKeys) usize {
        return t.shape.rows * D;
    }

    /// Block b's 32 weights in float64, and the worst relative error the kernel's fp32 steps may put on them.
    fn weights(t: *BlockKeys, b: usize, w: *[B]f64) f64 {
        const isq = 1 / @sqrt(@as(f64, D));
        var aw: [B]f64 = undefined;
        var err: f64 = 0;
        var top = -std.math.inf(f64);
        for (&aw, 0..) |*s, j| {
            var dot: f64 = 0;
            var abs: f64 = 0;
            for (0..D) |d| {
                const p = @as(f64, c.fromBf16(t.rk[(b * B + j) * D + d])) * c.fromBf16(t.hc[d]);
                dot += p;
                abs += @abs(p);
            }
            s.* = ud.clamp(dot * isq);
            err = @max(err, c.gamma(D, c.unit(.f32)) * abs * isq + 3 * 0x1p-24 * @abs(s.*));
            top = @max(top, s.*);
        }
        var mass: f64 = 0;
        var spread: f64 = 0;
        for (&aw, w) |s, *e| {
            e.* = @exp(s - top);
            mass += e.*;
            spread = @max(spread, @abs(s - top));
        }
        for (w) |*e| e.* /= @max(mass, 1e-6);
        // a weight: its score's and the maximum's errors, the subtraction, exp's 2 ulps; the 5-level sum, a division
        const each = 2 * err + spread * 0x1p-24 + 2 * 0x1p-23;
        return 2 * each + c.gamma(5, c.unit(.f32)) + 0x1p-24;
    }

    pub fn reference(t: *BlockKeys, want: []f64, mag: []f64) !void {
        var w: [B]f64 = undefined;
        for (0..t.shape.rows) |b| {
            const rel = t.weights(b, &w);
            for (0..D) |d| {
                var s: f64 = 0;
                var abs: f64 = 0;
                for (w, 0..) |wj, j| {
                    const v: f64 = c.fromBf16(t.rk[(b * B + j) * D + d]);
                    s += wj * v;
                    abs += @abs(wj * v);
                }
                const i = b * D + d;
                want[i] = s;
                mag[i] = abs;
                const exact = t.fill == .stress;
                t.allow[i] = if (exact) 0x1p-24 * @abs(s) + 0x1p-149 else abs * (rel + c.gamma(B + 1, c.unit(.f32))) + 0x1p-149;
            }
        }
    }
    pub fn bounds(t: *BlockKeys, _: []const f64, _: []const f64, allow: []f64) !void {
        @memcpy(allow, t.allow);
    }

    pub fn simulate(t: *BlockKeys, acc: c.Accum, order: g.Order, out: []f64) !void {
        var w: [B]f64 = undefined;
        var terms: [B]f64 = undefined;
        for (0..t.shape.rows) |b| {
            _ = t.weights(b, &w);
            for (0..D) |d| {
                for (&terms, w, 0..) |*s, wj, j| s.* = wj * c.fromBf16(t.rk[(b * B + j) * D + d]);
                out[b * D + d] = mi.rows.sumIn(acc, order, &terms);
            }
        }
    }
};

// --- tf_uocr_silu_mul: bf16(bf16(silu(g)) u) ---------------------------------------------------------------------

pub const Silu = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    /// The served form: gate and up halves of one row of 2 x width; else two buffers of width-wide rows.
    halves: bool,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    gv: []u16 = &.{},
    u: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: positions or picked experts; n: the MLP's width.
    pub fn spec(name: []const u8, halves: bool) c.Spec {
        return .{
            .name = name,
            .accum = .none,
            .depth = mi.noSums,
            .ops = 2 * (mi.exp_ulps + 2 * mi.div_ulps) + 2,
            .mids = 1,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256 }, .{} },
            .served = if (halves)
                &.{ .{ .rows = 2048, .n = ud.dense_ffn, .k = 1 }, .{ .rows = 37, .n = ud.shared_ffn, .k = 1 }, .{ .rows = ud.topk, .n = ud.expert_ffn, .k = 1 } }
            else
                &.{ .{ .rows = 2048 * ud.topk, .n = ud.expert_ffn, .k = 1 }, .{ .rows = 6, .n = ud.expert_ffn, .k = 1 } },
        };
    }

    pub fn deinit(t: *Silu) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Silu) void {
        const a = t.rig.gpu.gpa;
        a.free(t.gv);
        a.free(t.u);
        t.gv = &.{};
        t.u = &.{};
    }
    pub fn kernel(t: *Silu) g.Kernel {
        return .{ .spec = spec(t.name, t.halves), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Silu);

    pub fn prepare(t: *Silu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * shape.n;
        t.gv = try a.alloc(u16, n);
        t.u = try a.alloc(u16, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gates past +-20 reach silu's flat ends
        mi.randomBf16(rng, t.gv, -24, 24);
        mi.randomBf16(rng, t.u, -4, 4);
        if (fill == .nonfinite) for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, p| {
            t.gv[(p * 7) % n] = v;
            t.u[(p * 11 + 3) % n] = v;
        };
        if (t.halves) {
            const both = try a.alloc(u16, 2 * n);
            defer a.free(both);
            for (0..shape.rows) |r| {
                @memcpy(both[2 * r * shape.n ..][0..shape.n], t.gv[r * shape.n ..][0..shape.n]);
                @memcpy(both[(2 * r + 1) * shape.n ..][0..shape.n], t.u[r * shape.n ..][0..shape.n]);
            }
            t.ins[0] = try mi.input(&t.rig, both);
        } else {
            t.ins[0] = try mi.input(&t.rig, t.gv);
            t.ins[1] = try mi.input(&t.rig, t.u);
        }
        t.out = try t.rig.output(n * 2);
        return true;
    }

    pub fn run(t: *Silu) !void {
        const w = t.shape.n;
        const up = if (t.halves) t.ins[0] + w * 2 else t.ins[1];
        const stride = if (t.halves) 2 * w else w;
        try t.k.run("silu_mul", t.rig.s, .{ t.shape.rows, ud.kernels.blocks(w), 1 }, 256, .{ t.ins[0], up, t.out, w, stride, stride });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Silu) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Silu, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (t.gv, t.u, 0..) |gb, ub, i| {
            const gv: f64 = c.fromBf16(gb);
            const u: f64 = c.fromBf16(ub);
            const e = @exp(-gv);
            const s = bf(gv / (1 + e));
            const den = Range.of(e).fp32(mi.exp_ulps).add(Range.of(1)).fp32(0.5);
            const sr = Range.of(gv).mul(den.recip()).fp32(mi.div_ulps).bf16();
            t.ends.put(want, mag, i, bf(s * u), sr.mul(Range.of(u)).fp32(0.5).bf16());
        }
    }
    pub fn bounds(t: *Silu, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Silu, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};

// --- tf_uocr_combine: bf16(bf16(sum_j D[slot] w) + S) ------------------------------------------------------------

/// Six products and their sums in turn.
fn combineDepth(_: c.Shape) usize {
    return ud.topk + 1;
}

pub const Combine = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    d: []u16 = &.{},
    s: []u16 = &.{},
    slot: []u32 = &.{},
    w: []f32 = &.{},
    ins: [4]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    /// Rows: positions; n: the hidden width; k: the picked experts (6).
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = combineDepth,
            .ops = 1,
            .mids = 1,
            .out = .bf16,
            .served = &.{ .{ .rows = 2048, .n = H, .k = ud.topk }, .{ .rows = 1, .n = H, .k = ud.topk }, .{ .rows = 37, .n = H, .k = ud.topk } },
        };
    }

    pub fn deinit(t: *Combine) void {
        t.free();
        t.ends.deinit();
        t.rig.deinit();
    }
    fn free(t: *Combine) void {
        const a = t.rig.gpu.gpa;
        a.free(t.d);
        a.free(t.s);
        a.free(t.slot);
        a.free(t.w);
        t.d = &.{};
        t.s = &.{};
        t.slot = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Combine) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Combine);

    pub fn prepare(t: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k != ud.topk) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const a = t.rig.gpu.gpa;
        const n = shape.rows * ud.topk;
        t.d = try a.alloc(u16, n * H);
        t.s = try a.alloc(u16, shape.rows * H);
        t.slot = try a.alloc(u32, n);
        t.w = try a.alloc(f32, n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.slot, 0..) |*v, i| v.* = @intCast(i);
        rng.shuffle(u32, t.slot);
        mi.randomBf16(rng, t.d, -4, 4);
        mi.randomBf16(rng, t.s, -2, 2);
        for (t.w) |*v| v.* = rng.float(f32) * 0.5;
        if (fill == .stress) {
            // each output: +-b pairs and two small odd numbers, weights 1, so fp32 is exact and fp16 or bf16 drop bits
            @memset(t.w, 1);
            @memset(t.s, 0);
            for (0..shape.rows) |r| for (0..H) |x| {
                const b: f32 = @floatFromInt(@as(u32, 1) << @intCast(12 + rng.uintLessThan(u32, 3)));
                var v = [_]f32{ b, -b, b, -b, @floatFromInt(1 + 2 * rng.uintLessThan(u32, 8)), @floatFromInt(1 + 2 * rng.uintLessThan(u32, 8)) };
                rng.shuffle(f32, &v);
                for (v, 0..) |e, j| t.d[@as(usize, t.slot[r * ud.topk + j]) * H + x] = c.toBf16(e);
            };
        }
        if (fill == .nonfinite) {
            for ([_]u16{ mi.nan16, mi.inf16, mi.ninf16 }, 0..) |v, p| t.d[(p * 13) % (n * H)] = v;
            t.s[shape.rows * H - 1] = mi.nan16;
        }
        t.ins[0] = try mi.input(&t.rig, t.d);
        t.ins[1] = try mi.input(&t.rig, t.slot);
        t.ins[2] = try mi.input(&t.rig, t.w);
        t.ins[3] = try mi.input(&t.rig, t.s);
        t.out = try t.rig.output(shape.rows * H * 2);
        return true;
    }

    pub fn run(t: *Combine) !void {
        const i = t.ins;
        try t.k.run("combine", t.rig.s, .{ t.shape.rows, ud.kernels.blocks(H), 1 }, 256, .{ i[0], i[1], i[2], i[3], t.out });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Combine) usize {
        return t.shape.rows * H;
    }

    fn term(t: *Combine, r: usize, j: usize, x: usize) f64 {
        const i = r * ud.topk + j;
        return @as(f64, c.fromBf16(t.d[@as(usize, t.slot[i]) * H + x])) * t.w[i];
    }

    fn put(t: *Combine, ends: *Ends, r: usize, x: usize, sum: f64, sr: Range, want: []f64, mag: []f64) void {
        const s: f64 = c.fromBf16(t.s[r * H + x]);
        ends.put(want, mag, r * H + x, bf(bf(sum) + s), sr.bf16().add(Range.of(s)).fp32(0.5).bf16());
    }

    pub fn reference(t: *Combine, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        for (0..t.shape.rows) |r| for (0..H) |x| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..ud.topk) |j| {
                sum += t.term(r, j, x);
                abs += @abs(t.term(r, j, x));
            }
            const e = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(combineDepth(t.shape)), c.unit(.f32)) * abs;
            t.put(&t.ends, r, x, sum, .{ .lo = sum - e, .hi = sum + e }, want, mag);
        };
    }
    pub fn bounds(t: *Combine, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Combine, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }

    pub fn simulate(t: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = t.rig.gpu.gpa;
        const mag = try a.alloc(f64, out.len);
        defer a.free(mag);
        var scratch: Ends = .{ .a = a };
        defer scratch.deinit();
        try scratch.size(t.count());
        var terms: [ud.topk]f64 = undefined;
        for (0..t.shape.rows) |r| for (0..H) |x| {
            for (&terms, 0..) |*v, j| v.* = t.term(r, j, x);
            const sum = mi.rows.sumIn(acc, order, &terms);
            t.put(&scratch, r, x, sum, Range.of(sum), out, mag);
        };
    }
};
