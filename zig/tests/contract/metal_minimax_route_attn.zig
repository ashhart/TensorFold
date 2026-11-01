//! MiniMax-M3.1-Flash's q/k preparation (mm_prep) and the indexer's decode scores (mm_index_scores) under the contract.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const r = @import("metal_minimax_route.zig");
const sel = @import("metal_minimax_route_sort.zig");
const decode = @import("metal_minimax_route_decode.zig");

const c = contract.spec;
const g = contract.gates;
const Range = r.Range;
const bf = r.bf;
const D = r.head_dim;
const u = 0x1p-24;

// --- mm_prep: each head's Gemma norm and NeoX rotary of its first 64 dims; keys, values and index keys cached ------

/// mm_prep_args (glue.metal).
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, index_heads: u32, dim: u32, pos: u32, rotary: u32, stride: u32, eps: f32, theta: f32 };

/// A row's projection: q heads, k heads, v heads, index q heads, the index key (config.qkvRows()).
const stride = (r.heads + 2 * r.kv_heads + r.index_heads + 1) * D;
/// Heads a row prepares: q, k, index q and the index key (the values are copied with the keys).
const units = r.heads + r.kv_heads + r.index_heads + 1;

/// Four squares a lane in one fma chain, then simd_sum's five levels.
fn prepDepth(_: c.Shape) usize {
    return 4 + 5;
}

pub const Prep = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    pos: usize = 0,
    x: []u16 = &.{},
    w: [4][D]u16 = undefined,
    ins: [5]rig_.Ref = undefined,
    /// QN and IQ (the outputs gate 4 reads), then the caches' slots as scratch: KC, VC, IK.
    outs: [2]rig_.Ref = undefined,
    cache: [3]rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    /// Rows: query rows; n: query heads; k: the first row's position (its cache slot).
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 mm_prep q/k norms and rotary, 64 + 4 + 4 + 1 heads of 128",
            .accum = .f32,
            .depth = prepDepth,
            .out = .bf16,
            .nan = .{ .stated = "every output in float64's range through the kernel's fp32 steps and bf16 points (NaN where float64 gives NaN), the values copied bit for bit, the caches' slots before the first row untouched" },
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
            // a decode window near the longest window, one row, a prompt chunk, a row at position 0
            .served = &.{ .{ .rows = 16, .n = r.heads, .k = 262000 }, .{ .rows = 1, .n = r.heads, .k = 131071 }, .{ .rows = r.prompt_rows, .n = r.heads, .k = 8192 }, .{ .rows = 1, .n = r.heads, .k = 0 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Prep {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try r.gluePipe(device, "mm_prep") };
    }
    pub fn deinit(t: *Prep) void {
        t.a.free(t.x);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Prep) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Prep);

    pub fn prepare(t: *Prep, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != r.heads) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.rows = shape.rows;
        t.pos = shape.k;
        t.x = try t.a.alloc(u16, shape.rows * stride);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.x, 0..) |*v, i| v.* = c.toBf16(switch (fill) {
            // integers whose squares sum exactly in fp32 and pass fp16's range in any order
            .stress => @as(f32, @floatFromInt(rng.intRangeAtMost(i32, 32, 96))) * @as(f32, if (rng.boolean()) 1 else -1),
            else => (rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 12) else 1),
        });
        for (&t.w) |*w| for (w) |*v| {
            v.* = c.toBf16(rng.float(f32) - 0.5);
        };
        if (fill == .nonfinite) {
            // NaN in row 0's q head 0, +Inf in row 1's key head 1 (rotated), -Inf in row 2's index key (unrotated)
            const put = [_][2]usize{ .{ 0, 5 }, .{ 1, (r.heads + 1) * D + 3 }, .{ 2, (r.heads + 2 * r.kv_heads + r.index_heads) * D + 100 }, .{ 3, (r.heads + r.kv_heads + 2) * D + 9 } };
            const val = [_]f32{ r.nan32, r.inf32, -r.inf32, r.nan32 };
            for (put, val) |p, v| if (p[0] < shape.rows) {
                t.x[p[0] * stride + p[1]] = c.toBf16(v);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        for (1..5) |i| t.ins[i] = try t.rig.input(std.mem.sliceAsBytes(&t.w[i - 1]));
        t.outs[0] = try t.rig.output(shape.rows * r.heads * D * 2);
        t.outs[1] = try t.rig.output(shape.rows * r.index_heads * D * 2);
        const slots = t.pos + shape.rows;
        t.cache[0] = try t.rig.scratch(slots * r.kv_heads * D * 2);
        t.cache[1] = try t.rig.scratch(slots * r.kv_heads * D * 2);
        t.cache[2] = try t.rig.scratch(slots * D * 2);
        return true;
    }

    pub fn run(t: *Prep) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        for (t.ins, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setValue(PArgs{ .rows = @intCast(t.rows), .heads = r.heads, .kv_heads = r.kv_heads, .index_heads = r.index_heads, .dim = D, .pos = @intCast(t.pos), .rotary = r.rotary, .stride = stride, .eps = r.eps, .theta = r.theta }, 5);
        e.setBuffer(t.outs[0].buf, t.outs[0].off, 6);
        e.setBuffer(t.outs[1].buf, t.outs[1].off, 7);
        e.setBuffer(t.cache[0].buf, t.cache[0].off, 8);
        e.setBuffer(t.cache[1].buf, t.cache[1].off, 9);
        e.setBuffer(t.cache[2].buf, t.cache[2].off, 10);
        e.dispatchGroups(mtl.Size.of(units, t.rows, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    /// QN and IQ; the caches' slots are checked by `check`.
    pub fn count(t: *Prep) usize {
        return t.rows * (r.heads + r.index_heads) * D;
    }

    /// Unit `unit` of row `row`: its source offset in the row and its norm weight.
    fn unitOf(unit: usize) struct { src: usize, w: usize } {
        if (unit < r.heads) return .{ .src = unit * D, .w = 0 };
        if (unit < r.heads + r.kv_heads) return .{ .src = unit * D, .w = 1 };
        if (unit < r.heads + r.kv_heads + r.index_heads) return .{ .src = (unit + r.kv_heads) * D, .w = 2 };
        return .{ .src = (r.heads + 2 * r.kv_heads + r.index_heads) * D, .w = 3 };
    }

    /// One unit's outputs: float64 at the kernel's bf16 points (`sumsq`: gate 4's sum of squares), and each's range.
    fn unitRef(t: *Prep, row: usize, unit: usize, sumsq: ?f64, point: *[D]f64, range: ?*[D]Range) void {
        const o = unitOf(unit);
        const v = t.x[row * stride + o.src ..][0..D];
        const w = &t.w[o.w];
        var s: f64 = 0;
        for (v) |x| s += @as(f64, c.fromBf16(x)) * c.fromBf16(x);
        if (sumsq) |x| s = x;
        const inv = 1 / @sqrt(s / D + r.eps);
        // the sum's chain (all terms positive, so relative), the eps add, precise rsqrt (2 ulps), the two products
        const rel = 0.5 * (c.gamma(@floatFromInt(prepDepth(.{ .rows = 1, .n = 1, .k = 1 })), u) + u) + 2 * u + 3 * u + u;
        var nrm: [D]f64 = undefined;
        var nr: [D]Range = undefined;
        for (0..D) |d| {
            const x: f64 = c.fromBf16(v[d]);
            const gain = 1 + @as(f64, c.fromBf16(w[d]));
            nrm[d] = bf(x * inv * gain);
            const val = x * inv * gain;
            nr[d] = if (std.math.isFinite(val)) (Range{ .lo = val - rel * @abs(val), .hi = val + rel * @abs(val) }).bf16() else Range.point(val);
        }
        const half_r = r.rotary / 2;
        const position: f64 = @floatFromInt(t.pos + row);
        for (0..D) |d| {
            if (d >= r.rotary) {
                point[d] = nrm[d];
                if (range) |rg| rg[d] = nr[d];
                continue;
            }
            const i = d % half_r;
            const freq = 1 / std.math.pow(f64, r.theta, @as(f64, @floatFromInt(2 * i)) / r.rotary);
            const angle = position * freq;
            const cs = bf(@cos(angle));
            const sn = bf(@sin(angle));
            const rot = if (d < half_r) -nrm[d + half_r] else nrm[d - half_r];
            point[d] = bf(bf(nrm[d] * cs) + bf(rot * sn));
            if (range) |rg| {
                // precise pow (16 ulps) and the division, the fp32 angle, precise cos and sin (4 ulps), bf16 points
                const f = Range.point(freq).fp32(17);
                const ang = (Range{ .lo = position * f.lo, .hi = position * f.hi }).fp32(1);
                const rr = if (d < half_r) Range{ .lo = -nr[d + half_r].hi, .hi = -nr[d + half_r].lo } else nr[d - half_r];
                const a1 = Range.mul(nr[d], trig(ang, false).fp32(4).bf16()).fp32(1).bf16();
                const a2 = Range.mul(rr, trig(ang, true).fp32(4).bf16()).fp32(1).bf16();
                rg[d] = a1.add(a2).fp32(1).bf16();
            }
        }
    }

    pub fn reference(t: *Prep, want: []f64, mag: []f64) !void {
        var pt: [D]f64 = undefined;
        for (0..t.rows) |row| for (0..r.heads + r.index_heads) |h| {
            const unit = if (h < r.heads) h else h + r.kv_heads;
            t.unitRef(row, unit, null, &pt, null);
            @memcpy(want[outAt(t.rows, row, h)..][0..D], &pt);
        };
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }

    /// Where row `row`'s head `h` (q heads, then index q heads) lies among the outputs.
    fn outAt(rows: usize, row: usize, h: usize) usize {
        return if (h < r.heads) (row * r.heads + h) * D else rows * r.heads * D + (row * r.index_heads + h - r.heads) * D;
    }

    pub fn bounds(t: *Prep, want: []const f64, _: []const f64, allow: []f64) !void {
        var pt: [D]f64 = undefined;
        var rg: [D]Range = undefined;
        for (0..t.rows) |row| for (0..r.heads + r.index_heads) |h| {
            t.unitRef(row, if (h < r.heads) h else h + r.kv_heads, null, &pt, &rg);
            const at = outAt(t.rows, row, h);
            for (0..D) |d| allow[at + d] = if (std.math.isFinite(want[at + d])) r.spread(want[at + d], rg[d]) else 0;
        };
    }

    pub fn simulate(t: *Prep, acc: c.Accum, order: g.Order, out: []f64) !void {
        var sq: [D]f64 = undefined;
        var pt: [D]f64 = undefined;
        for (0..t.rows) |row| for (0..r.heads + r.index_heads) |h| {
            const unit = if (h < r.heads) h else h + r.kv_heads;
            const v = t.x[row * stride + unitOf(unit).src ..][0..D];
            for (&sq, v) |*s, x| s.* = @as(f64, c.fromBf16(x)) * c.fromBf16(x);
            t.unitRef(row, unit, contract.rows.sumIn(acc, order, &sq), &pt, null);
            @memcpy(out[outAt(t.rows, row, h)..][0..D], &pt);
        };
    }

    /// Whether bf16 `got` lies in `rg` (NaN where float64 gives NaN, infinities by value).
    fn holds(point: f64, rg: Range, got: f64) bool {
        if (std.math.isNan(point)) return std.math.isNan(got);
        if (std.math.isInf(point)) return got == point;
        return got >= rg.lo and got <= rg.hi;
    }

    pub fn check(t: *Prep, _: g.Gate) !?[]const u8 {
        const nk = r.kv_heads;
        const qn = t.rig.bytes(t.outs[0], t.rows * r.heads * D * 2);
        const iq = t.rig.bytes(t.outs[1], t.rows * r.index_heads * D * 2);
        const kc = t.rig.bytes(t.cache[0], (t.pos + t.rows) * nk * D * 2);
        const vc = t.rig.bytes(t.cache[1], (t.pos + t.rows) * nk * D * 2);
        const ik = t.rig.bytes(t.cache[2], (t.pos + t.rows) * D * 2);
        var pt: [D]f64 = undefined;
        var rg: [D]Range = undefined;
        for (0..t.rows) |row| {
            const slot = t.pos + row;
            for (0..units) |unit| {
                t.unitRef(row, unit, null, &pt, &rg);
                const got = if (unit < r.heads)
                    qn[(row * r.heads + unit) * D * 2 ..]
                else if (unit < r.heads + nk)
                    kc[(slot * nk + unit - r.heads) * D * 2 ..]
                else if (unit < units - 1)
                    iq[(row * r.index_heads + unit - r.heads - nk) * D * 2 ..]
                else
                    ik[slot * D * 2 ..];
                for (0..D) |d| {
                    const v = c.read(.bf16, got, d);
                    if (!holds(pt[d], rg[d], v)) return try std.fmt.bufPrint(&t.msg, "row {d} (position {d}) unit {d} dim {d} is {e:.6} outside [{e:.6}, {e:.6}] (float64 {e:.6})", .{ row, slot, unit, d, v, rg[d].lo, rg[d].hi, pt[d] });
                }
            }
            for (0..nk) |h| {
                const want = std.mem.sliceAsBytes(t.x[row * stride + (r.heads + nk + h) * D ..][0..D]);
                if (!std.mem.eql(u8, vc[(slot * nk + h) * D * 2 ..][0 .. D * 2], want)) return try std.fmt.bufPrint(&t.msg, "row {d}'s value head {d} is not its projection's bits", .{ row, h });
            }
        }
        // the slot before the first row keeps the rig's pattern: nothing writes past the chunk's own slots
        if (t.pos > 0) for ([_][]const u8{ kc[(t.pos - 1) * nk * D * 2 ..][0 .. nk * D * 2], vc[(t.pos - 1) * nk * D * 2 ..][0 .. nk * D * 2], ik[(t.pos - 1) * D * 2 ..][0 .. D * 2] }) |b| {
            for (b) |x| if (x != 0xa5) return try std.fmt.bufPrint(&t.msg, "a cache slot before position {d} was written", .{t.pos});
        };
        return null;
    }
};

/// cos (or sin) over an angle's range: the ends, and +-1 where the range holds a peak.
fn trig(a: Range, sine: bool) Range {
    const f = struct {
        fn of(x: f64, s: bool) f64 {
            return if (s) @sin(x) else @cos(x);
        }
    };
    var lo = @min(f.of(a.lo, sine), f.of(a.hi, sine));
    var hi = @max(f.of(a.lo, sine), f.of(a.hi, sine));
    // peaks of cos at k pi (sin: k pi + pi / 2): +1 for even k, -1 for odd
    const shift: f64 = if (sine) std.math.pi / 2.0 else 0;
    var k = @ceil((a.lo - shift) / std.math.pi);
    while (k * std.math.pi + shift <= a.hi) : (k += 1) {
        if (@mod(k, 2) == 0) hi = 1 else lo = -1;
    }
    return .{ .lo = lo - 0x1p-149, .hi = hi + 0x1p-149 };
}

// --- mm_index_scores: each (row, index head, block)'s max of q . k over the block's keys up to the row -----------

/// Each key's 16 steps of two float4 dots (four roundings each), their add and the running add.
fn scoreDepth(_: c.Shape) usize {
    return 16 * 10;
}

pub const Scores = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    rows: usize = 0,
    blocks: usize = 0,
    pos: usize = 0,
    q: []u16 = &.{},
    k: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: decode rows; n: the index blocks their keys span; k: the index head dim.
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 mm_index_scores index heads 4 of 128, blocks of 128",
            .accum = .f32,
            .depth = scoreDepth,
            .out = .f32,
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
            // decode windows (the prompt path's scores are the tensor units'): 64 blocks, the longest window, 17, 300
            .served = &.{ .{ .rows = 16, .n = 64, .k = D }, .{ .rows = 16, .n = r.context_blocks, .k = D }, .{ .rows = 1, .n = r.topk_blocks + 1, .k = D }, .{ .rows = r.batch_rows, .n = 300, .k = D } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Scores {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try r.gluePipe(device, "mm_index_scores") };
    }
    pub fn deinit(t: *Scores) void {
        t.a.free(t.q);
        t.a.free(t.k);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Scores) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Scores);

    pub fn prepare(t: *Scores, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > r.batch_rows or shape.k != D) return false;
        const pos = sel.startOf(shape.rows, shape.n) orelse return false;
        t.rig.reset();
        t.a.free(t.q);
        t.a.free(t.k);
        t.rows = shape.rows;
        t.blocks = shape.n;
        t.pos = pos;
        const keys = shape.n * r.block;
        t.q = try t.a.alloc(u16, shape.rows * r.index_heads * D);
        t.k = try t.a.alloc(u16, keys * D);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            // integers, each vector's second half mirroring its first (8 nudged): partials near 2^18, small dots
            const h = D / 2;
            for (0..shape.rows * r.index_heads) |v| for (0..h) |i| {
                const x: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                t.q[v * D + i] = c.toBf16(x);
                t.q[v * D + h + i] = c.toBf16(x + @as(f32, if (i < 8) 1 else 0));
            };
            for (0..keys) |j| for (0..h) |i| {
                const x: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                t.k[j * D + i] = c.toBf16(x);
                t.k[j * D + h + i] = c.toBf16(-x);
            };
        } else {
            // never zero: an infinity times a key's value is +-Inf, not NaN
            for (t.q) |*v| v.* = c.toBf16((0.05 + rng.float(f32)) * @as(f32, if (rng.boolean()) 1 else -1));
            for (t.k) |*v| v.* = c.toBf16((0.05 + rng.float(f32)) * @as(f32, if (rng.boolean()) 1 else -1));
        }
        if (fill == .nonfinite) {
            // NaN in row 0's head 0, +Inf in row 1's head 1, -Inf in row 2's head 2; NaN in one key of block 0
            const put = [_]f32{ r.nan32, r.inf32, -r.inf32 };
            for (put, 0..) |v, i| if (i < shape.rows) {
                t.q[(i * r.index_heads + i) * D + 3] = c.toBf16(v);
            };
            t.k[17 * D + 40] = c.toBf16(r.nan32);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.k));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Scores) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setValue(sel.IArgs{ .rows = @intCast(t.rows), .pos = @intCast(t.pos), .heads = r.index_heads, .blocks = @intCast(t.blocks) }, 2);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.dispatchGroups(mtl.Size.of(t.blocks, (t.rows + 15) / 16, 1), mtl.Size.of(128, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Scores) usize {
        return t.rows * r.index_heads * t.blocks;
    }

    fn term(t: *Scores, row: usize, h: usize, key: usize, d: usize) f64 {
        return @as(f64, c.fromBf16(t.q[(row * r.index_heads + h) * D + d])) * c.fromBf16(t.k[key * D + d]);
    }

    /// Each output from each live key's dot (`sum` over its terms): the max (NaN if any is), and the largest magnitude.
    fn each(t: *Scores, sum: anytype, want: []f64, mag: ?[]f64) void {
        var terms: [D]f64 = undefined;
        for (0..t.rows) |row| for (0..r.index_heads) |h| for (0..t.blocks) |j| {
            var best = -std.math.inf(f64);
            var m: f64 = 0;
            const last = @min((j + 1) * r.block, t.pos + row + 1);
            var key = j * r.block;
            while (key < last) : (key += 1) {
                var a: f64 = 0;
                for (&terms, 0..) |*x, d| {
                    x.* = t.term(row, h, key, d);
                    a += @abs(x.*);
                }
                const s = sum.of(&terms);
                best = if (std.math.isNan(s) or std.math.isNan(best)) std.math.nan(f64) else @max(best, s);
                if (std.math.isFinite(a)) m = @max(m, a);
            }
            const at = (row * r.index_heads + h) * t.blocks + j;
            want[at] = best;
            if (mag) |x| x[at] = m;
        };
    }

    const Exact = struct {
        fn of(_: Exact, v: []const f64) f64 {
            var s: f64 = 0;
            for (v) |x| s += x;
            return s;
        }
    };

    pub fn reference(t: *Scores, want: []f64, mag: []f64) !void {
        t.each(Exact{}, want, mag);
    }

    pub fn simulate(t: *Scores, acc: c.Accum, order: g.Order, out: []f64) !void {
        const In = struct {
            acc: c.Accum,
            order: g.Order,
            fn of(x: @This(), v: []const f64) f64 {
                return contract.rows.sumIn(x.acc, x.order, v);
            }
        };
        t.each(In{ .acc = acc, .order = order }, out, null);
    }
};

/// This file's and the decode file's kernels in the suite.
pub const Group = struct {
    prep: ?*Prep = null,
    scores: ?*Scores = null,
    decode: decode.Group = .{},

    pub fn collect(s: *Group, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !void {
        inline for (.{ .{ "prep", Prep }, .{ "scores", Scores } }) |e| {
            if (only == null or std.mem.indexOf(u8, e[1].spec().name, only.?) != null) {
                const x = try a.create(e[1]);
                x.* = try e[1].init(a, device, queue);
                @field(s, e[0]) = x;
                try kernels.append(a, x.kernel());
            }
        }
        try s.decode.collect(a, device, queue, only, kernels);
    }
    pub fn deinit(s: *Group, a: std.mem.Allocator) void {
        inline for (.{ "prep", "scores" }) |f| if (@field(s, f)) |x| {
            x.deinit();
            a.destroy(x);
        };
        s.decode.deinit(a);
    }
};
