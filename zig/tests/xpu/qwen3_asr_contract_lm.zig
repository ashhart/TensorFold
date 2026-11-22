//! The LM's kernels under the contract on XPU: asr_qk_rope (with its cache writes) and lfm3_gq2's attention pair.
const std = @import("std");
const xpu = @import("xpu");
const core = @import("core");
const kit = @import("qwen3_asr_contract_kit.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const Mutation = kit.Mutation;
const val = kit.val;
const u = kit.u;

const HD = 128;
const pos0 = 5; // the pass's first position: rows land at 5, 6, ...

/// Each lane's chain over its 8 values, then the 16-lane sub-group tree.
fn ropeDepth(_: c.Shape) usize {
    return 8 + 4 + 1;
}

/// asr_qk_rope's outputs (q, k and v rows; numbers against float64), or its cache rows (bits against the k and v rows).
pub const Part = enum { rows, cache };

const rope_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps, and " ++
    "an fp16 sum past its range zeroes rs, so an output misses by at most itself over its bound (about 85x)";

/// rows: a pass's rows; n: q heads; k: KV heads (128-dim each).
pub const QkRope = struct {
    a: std.mem.Allocator,
    rig: kit.Rig,
    k: xpu.Kernel,
    m: Mutation,
    part: Part,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    qkv: []u16 = &.{},
    qn: [HD]u16 = undefined,
    kn: [HD]u16 = undefined,
    cs: []f32 = &.{},
    sn: []f32 = &.{},
    bufs: [10]?*anyopaque = undefined,

    pub fn spec(part: Part) c.Spec {
        // a small odd pass; a decode row of the 0.6B and 1.7B LM (16 q heads on 8); a prompt pass
        const served: []const c.Shape = &.{ .{ .rows = 3, .n = 4, .k = 2 }, .{ .rows = 1, .n = 16, .k = 8 }, .{ .rows = 150, .n = 16, .k = 8 } };
        return switch (part) {
            .rows => .{ .name = "xpu qwen3-asr asr_qk_rope (q, k and v rows)", .accum = .f32, .depth = ropeDepth, .ops = 8, .mids = 3, .out = .bf16, .served = served, .sums = .{ .f16 = 50, .bf16 = 0, .why = rope_why } },
            .cache => .{ .name = "xpu qwen3-asr asr_qk_rope (k and v into the caches at pos0 + r)", .accum = .none, .depth = kit.noSums, .out = .bf16, .served = served },
        };
    }

    pub fn init(a: std.mem.Allocator, dev: *kit.Dev, m: Mutation, part: Part) !QkRope {
        return .{ .a = a, .rig = .{ .a = a, .dev = dev }, .k = try dev.kernel(.qwen3_asr, m, "asr_qk_rope"), .m = m, .part = part };
    }

    pub fn deinit(t: *QkRope) void {
        t.free();
        t.k.deinit();
        t.rig.deinit();
    }

    fn free(t: *QkRope) void {
        t.a.free(t.qkv);
        t.a.free(t.cs);
        t.a.free(t.sn);
        t.qkv = &.{};
        t.cs = &.{};
        t.sn = &.{};
    }

    pub fn kernel(t: *QkRope) g.Kernel {
        return kit.mark(.{ .spec = spec(t.part), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(QkRope);

    fn width(t: *const QkRope) usize {
        return (t.shape.n + 2 * t.shape.k) * HD;
    }

    fn cap(t: *const QkRope) usize {
        return pos0 + t.shape.rows + 2;
    }

    pub fn prepare(t: *QkRope, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.qkv = try t.a.alloc(u16, shape.rows * t.width());
        if (fill == .stress) {
            // integers -63..64: a head's sum of squares is exact in fp32 (<= 2^19) and passes fp16's range
            for (t.qkv) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 128))) - 63);
        } else kit.fill(rng, t.qkv, 2);
        kit.fill(rng, &t.qn, 1.5);
        kit.fill(rng, &t.kn, 1.5);
        if (fill == .nonfinite) {
            // NaN, +Inf, -Inf in three heads of row 0, and in a v column
            const wd = t.width();
            for (kit.planted, 0..) |v, i| t.qkv[(i * HD + 7 * i + 3) % wd] = c.toBf16(v);
            t.qkv[wd - 5] = c.toBf16(kit.planted[1]);
        }
        const n = t.cap() * HD;
        t.cs = try t.a.alloc(f32, n);
        t.sn = try t.a.alloc(f32, n);
        for (0..t.cap()) |p| for (0..HD / 2) |i| {
            const inv = 1.0 / std.math.pow(f64, 1e6, @as(f64, @floatFromInt(2 * i)) / HD);
            const ang = @as(f64, @floatFromInt(p)) * inv;
            for ([2]usize{ i, i + HD / 2 }) |col| {
                t.cs[p * HD + col] = @floatCast(kit.bf(@cos(ang)));
                t.sn[p * HD + col] = @floatCast(kit.bf(@sin(ang)));
            }
        };
        const r = shape.rows;
        const qd = shape.n * HD;
        const kvd = shape.k * HD;
        t.bufs[0] = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.bufs[1] = try t.rig.input(std.mem.sliceAsBytes(&t.qn));
        t.bufs[2] = try t.rig.input(std.mem.sliceAsBytes(&t.kn));
        t.bufs[3] = try t.rig.input(std.mem.sliceAsBytes(t.cs));
        t.bufs[4] = try t.rig.input(std.mem.sliceAsBytes(t.sn));
        const sizes = [_]usize{ r * qd, r * kvd, r * kvd, t.cap() * kvd, t.cap() * kvd };
        for (sizes, 5..) |sz, i| {
            const out = (t.part == .rows) == (i < 8);
            t.bufs[i] = if (out) try t.rig.output(sz * 2) else try t.rig.scratch(sz * 2);
        }
        return true;
    }

    pub fn run(t: *QkRope) !void {
        const s = t.shape;
        const b = t.bufs;
        try t.rig.launch(t.k, .{ u(s.n + 2 * s.k), u(s.rows), 1 }, .{ b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], u(s.n), u(s.k), @as(u32, pos0), @as(f32, 1e-6) });
    }

    pub fn count(t: *QkRope) usize {
        const s = t.shape;
        return if (t.part == .rows) s.rows * (s.n + 2 * s.k) * HD else 2 * t.cap() * s.k * HD;
    }

    /// Output i's head: its normed values (x rs w, float64) and the sum's magnitude, for want and the bound.
    fn head(t: *QkRope, r: usize, h: usize, out: *[HD]f64) f64 {
        const src = t.qkv[r * t.width() + h * HD ..][0..HD];
        const w = if (h < t.shape.n) &t.qn else &t.kn;
        var ss: f64 = 0;
        for (src) |v| ss += val(v) * val(v);
        const rs = 1 / @sqrt(ss / HD + 1e-6);
        for (out, src, w) |*o, x, wv| o.* = val(x) * rs * val(wv);
        return ss;
    }

    /// q, k (normed, rotated), v (copied) in the outputs' order, and each value's bound when `allow` is given.
    fn each(t: *QkRope, want: []f64, allow: ?[]f64) void {
        const s = t.shape;
        const qd = s.n * HD;
        const kvd = s.k * HD;
        const chain = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(ropeDepth(s)), kit.u32r);
        for (0..s.rows) |r| {
            const p = pos0 + r;
            for (0..s.n + s.k) |h| {
                var nv: [HD]f64 = undefined;
                _ = t.head(r, h, &nv);
                const at = if (h < s.n) r * qd + h * HD else s.rows * qd + r * kvd + (h - s.n) * HD;
                for (0..HD) |d| {
                    const rot = if (d < HD / 2) -nv[d + HD / 2] else nv[d - HD / 2];
                    const cs: f64 = t.cs[p * HD + d];
                    const sn: f64 = t.sn[p * HD + d];
                    want[at + d] = nv[d] * cs + rot * sn;
                    if (allow) |al| {
                        // the norm's two bf16 roundings and sum error, then each product's and the sum's roundings
                        const rel = chain / 2 + 2 * kit.ubf + 6 * kit.u32r;
                        const work = rel * (@abs(nv[d] * cs) + @abs(rot * sn)) + (kit.ubf + kit.u32r) * (@abs(nv[d] * cs) + @abs(rot * sn)) * (1 + rel) + kit.u32r * @abs(want[at + d]);
                        al[at + d] = work + kit.ubf * (@abs(want[at + d]) + work) + kit.floor_bf;
                    }
                }
            }
            const vat = s.rows * (qd + kvd) + r * kvd;
            for (0..kvd) |i| {
                want[vat + i] = val(t.qkv[r * t.width() + qd + kvd + i]);
                if (allow) |al| al[vat + i] = kit.floor_bf;
            }
        }
    }

    pub fn reference(t: *QkRope, want: []f64, mag: []f64) !void {
        t.each(want, null);
        @memset(mag, 0);
    }

    pub fn bounds(t: *QkRope, _: []const f64, _: []const f64, allow: []f64) !void {
        const want = try t.a.alloc(f64, allow.len);
        defer t.a.free(want);
        t.each(want, allow);
    }

    /// The sums of squares in `acc` and `order`, the rest exact.
    pub fn simulate(t: *QkRope, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.each(out, null);
        const s = t.shape;
        var sq: [HD]f64 = undefined;
        for (0..s.rows) |r| for (0..s.n + s.k) |h| {
            var nv: [HD]f64 = undefined;
            const ss = t.head(r, h, &nv);
            const src = t.qkv[r * t.width() + h * HD ..][0..HD];
            for (&sq, src) |*q, x| q.* = val(x) * val(x);
            // the exact rs to the simulated one: every normed value of the head scales by their ratio
            const ratio = @sqrt((ss / HD + 1e-6) / (rows.sumIn(acc, order, &sq) / HD + 1e-6));
            const at = if (h < s.n) r * s.n * HD + h * HD else s.rows * s.n * HD + r * s.k * HD + (h - s.n) * HD;
            for (out[at..][0..HD]) |*o| o.* *= ratio;
        };
    }
};

/// asr_qk_rope's cache writes: rows pos0 + r of kc and vc bit-equal to the k and v rows, every other slot untouched.
pub const QkCache = struct {
    base: QkRope,
    rig: *kit.Rig,
    msg: [160]u8 = undefined,

    /// In place: `rig` points into `base`.
    pub fn init(t: *QkCache, a: std.mem.Allocator, dev: *kit.Dev, m: Mutation) !void {
        t.* = .{ .base = try QkRope.init(a, dev, m, .cache), .rig = undefined };
        t.rig = &t.base.rig;
    }

    pub fn deinit(t: *QkCache) void {
        t.base.deinit();
    }

    pub fn kernel(t: *QkCache) g.Kernel {
        return kit.mark(.{ .spec = QkRope.spec(.cache), .ctx = t, .vt = &vt }, t.base.m);
    }
    const vt = g.hooks(QkCache);

    pub fn prepare(t: *QkCache, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        return t.base.prepare(shape, fill, seed);
    }

    pub fn run(t: *QkCache) !void {
        return t.base.run();
    }

    pub fn count(t: *QkCache) usize {
        return t.base.count();
    }

    pub fn reference(_: *QkCache, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(t: *QkCache, _: g.Gate) !?[]const u8 {
        const b = &t.base;
        const s = b.shape;
        const kvd = s.k * HD;
        const k_rows = try b.a.dupe(u8, try b.rig.bytes(b.bufs[6], s.rows * kvd * 2));
        defer b.a.free(k_rows);
        const got = try b.rig.outputBytes();
        for (0..2) |which| for (0..b.cap()) |p| for (0..kvd) |i| {
            const v = std.mem.readInt(u16, got[2 * ((which * b.cap() + p) * kvd + i) ..][0..2], .little);
            const live = p >= pos0 and p < pos0 + s.rows;
            const r = p -% pos0;
            const want: u16 = if (!live) 0xffff else if (which == 0) std.mem.readInt(u16, k_rows[2 * (r * kvd + i) ..][0..2], .little) else b.qkv[r * b.width() + (s.n + s.k) * HD + i];
            if (v != want) return try std.fmt.bufPrint(&t.msg, "{s} slot {d} value {d} has bits {x:0>4} where {x:0>4} belongs", .{ if (which == 0) "kc" else "vc", p, i, v, want });
        };
        return null;
    }
};

/// lfm3's split-K pass and merge at GQ 2, gate +inf, 128-key chunks; rows, n q heads, k positions before the pass.
pub const Attn = struct {
    a: std.mem.Allocator,
    rig: kit.Rig,
    part: xpu.Kernel,
    merge: xpu.Kernel,
    m: Mutation,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    bufs: [8]?*anyopaque = undefined,

    const chunk = 128;
    const why = "softmax weights in [0, 1] and a bf16 output: no accumulator's error survives its rounding 100x";

    fn depth(s: c.Shape) usize {
        return 14 + s.k + s.rows;
    }

    pub fn spec() c.Spec {
        // a small pass; decode rows after 300 and 1,000 positions (3 and 8 chunks); a prompt pass from 0
        const served: []const c.Shape = &.{ .{ .rows = 3, .n = 4, .k = 5 }, .{ .rows = 1, .n = 16, .k = 300 }, .{ .rows = 1, .n = 16, .k = 1000 }, .{ .rows = 150, .n = 16, .k = 0 } };
        return .{ .name = "xpu qwen3-asr lfm3_gq2 lfm3_attn_partial + lfm3_attn_merge_gate (causal GQA 2)", .accum = .f32, .depth = depth, .ops = 8, .out = .bf16, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = why } };
    }

    pub fn init(a: std.mem.Allocator, dev: *kit.Dev, m: Mutation) !Attn {
        var p = try dev.kernel(.lfm3_gq2, m, "lfm3_attn_partial");
        errdefer p.deinit();
        return .{ .a = a, .rig = .{ .a = a, .dev = dev }, .part = p, .merge = try dev.kernel(.lfm3_gq2, m, "lfm3_attn_merge_gate"), .m = m };
    }

    pub fn deinit(t: *Attn) void {
        t.free();
        t.part.deinit();
        t.merge.deinit();
        t.rig.deinit();
    }

    fn free(t: *Attn) void {
        inline for (.{ "q", "kc", "vc" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }

    pub fn kernel(t: *Attn) g.Kernel {
        return kit.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Attn);

    fn cap(t: *const Attn) usize {
        return t.shape.k + t.shape.rows + 3;
    }

    fn maxc(t: *const Attn) usize {
        return (t.cap() + chunk - 1) / chunk;
    }

    pub fn prepare(t: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 2 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const qd = shape.n * HD;
        const kvd = shape.n / 2 * HD;
        t.q = try t.a.alloc(u16, shape.rows * qd);
        t.kc = try t.a.alloc(u16, t.cap() * kvd);
        t.vc = try t.a.alloc(u16, t.cap() * kvd);
        kit.fill(rng, t.q, 1);
        kit.fill(rng, t.kc, 1);
        kit.fill(rng, t.vc, 2);
        if (fill == .nonfinite) {
            kit.plant(t.q);
            t.vc[kvd + 9] = c.toBf16(std.math.nan(f32));
            t.kc[2 * kvd + 4] = c.toBf16(-std.math.inf(f32));
        }
        const gate = try t.a.alloc(u16, shape.rows * shape.n);
        defer t.a.free(gate);
        @memset(gate, 0x7f80);
        const parts = shape.rows * shape.n * t.maxc();
        t.bufs[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.bufs[1] = try t.rig.input(std.mem.sliceAsBytes(t.kc));
        t.bufs[2] = try t.rig.input(std.mem.sliceAsBytes(t.vc));
        t.bufs[3] = try t.rig.input(std.mem.sliceAsBytes(gate));
        t.bufs[4] = try t.rig.scratch(parts * HD * 4);
        t.bufs[5] = try t.rig.scratch(parts * 4);
        t.bufs[6] = try t.rig.scratch(parts * 4);
        t.bufs[7] = try t.rig.output(shape.rows * qd * 2);
        return true;
    }

    pub fn run(t: *Attn) !void {
        const s = t.shape;
        const b = t.bufs;
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, HD)));
        const nch = (s.k + s.rows + chunk - 1) / chunk;
        try t.rig.launch(t.part, .{ u(s.n / 2), u(nch), u(s.rows) }, .{ b[0], b[1], b[2], b[4], b[5], b[6], u(s.k), @as(u32, chunk), u(t.cap()), @as(u32, 0), u(s.n / 2), u(t.maxc()), scale });
        try t.rig.launch(t.merge, .{ u(s.n), u(s.rows), 1 }, .{ b[4], b[5], b[6], b[3], b[7], u(s.k), @as(u32, chunk), @as(u32, 0), u(s.n), u(t.maxc()) });
    }

    pub fn count(t: *Attn) usize {
        return t.shape.rows * t.shape.n * HD;
    }

    const Kv = struct {
        t: *const Attn,
        kvh: usize,
        pub fn k(kv: Kv, j: usize, e: usize) f64 {
            return val(kv.t.kc[(j * kv.t.shape.n / 2 + kv.kvh) * HD + e]);
        }
        pub fn v(kv: Kv, j: usize, e: usize) f64 {
            return val(kv.t.vc[(j * kv.t.shape.n / 2 + kv.kvh) * HD + e]);
        }
    };

    fn each(t: *Attn, want: []f64, allow: ?[]f64) !void {
        const s = t.shape;
        var q: [HD]f64 = undefined;
        const scale = 1.0 / @sqrt(@as(f64, HD));
        for (0..s.rows) |r| for (0..s.n) |h| {
            for (&q, 0..) |*x, e| x.* = val(t.q[(r * s.n + h) * HD + e]);
            const keys = s.k + r + 1;
            const at = (r * s.n + h) * HD;
            try kit.attention(t.a, &q, keys, Kv{ .t = t, .kvh = h / 2 }, scale, (keys + chunk - 1) / chunk, want[at..][0..HD], if (allow) |al| al[at..][0..HD] else null);
        };
    }

    pub fn reference(t: *Attn, want: []f64, mag: []f64) !void {
        try t.each(want, null);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(t: *Attn, _: []const f64, _: []const f64, allow: []f64) !void {
        const want = try t.a.alloc(f64, allow.len);
        defer t.a.free(want);
        try t.each(want, allow);
    }

    /// The scores and the two weighted sums in `acc` and `order`, exp and the division exact.
    pub fn simulate(t: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = t.shape;
        const scale = 1.0 / @sqrt(@as(f64, HD));
        const most = s.k + s.rows;
        const buf = try t.a.alloc(f64, @max(most, HD));
        defer t.a.free(buf);
        const p = try t.a.alloc(f64, most);
        defer t.a.free(p);
        for (0..s.rows) |r| for (0..s.n) |h| {
            const kv: Kv = .{ .t = t, .kvh = h / 2 };
            const keys = s.k + r + 1;
            var mx: f64 = -std.math.inf(f64);
            for (0..keys) |j| {
                for (buf[0..HD], 0..) |*x, e| x.* = val(t.q[(r * s.n + h) * HD + e]) * kv.k(j, e);
                p[j] = rows.sumIn(acc, order, buf[0..HD]) * scale;
                mx = @max(mx, p[j]);
            }
            for (p[0..keys]) |*x| x.* = @exp(x.* - mx);
            const l = rows.sumIn(acc, order, p[0..keys]);
            for (0..HD) |e| {
                for (buf[0..keys], 0..) |*x, j| x.* = p[j] * kv.v(j, e);
                out[(r * s.n + h) * HD + e] = rows.sumIn(acc, order, buf[0..keys]) / l;
            }
        };
    }
};
