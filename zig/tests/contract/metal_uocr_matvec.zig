//! Unlimited-OCR-2's one-token projections under the contract (decoder.metal): `ud_matvec` at every projection the
//! decoder serves, and the routed experts' `ud_expert_gu` and `ud_expert_down`, which reach each slot's expert through
//! a table of GPU pointers (the weight buffers made resident with useResource, as the engine does).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");

const c = contract.spec;
const g = contract.gates;
const bf = uocr.bf;
const nan = uocr.planted[0];
const inf = uocr.planted[1];
const H = uocr.hidden;
const F = uocr.expert_ffn;

/// The engine's params structs (decoder.zig): UdMv, UdExpert and UdExpertArgs.
const UdMv = extern struct { n: u32, k: u32 };
const UdExpert = extern struct { gate: u64, up: u64, down: u64 };
const UdExpertArgs = extern struct { ffn: u32 };

/// A term's longest chain: a dot of 4 (3 adds), the two dots' add, one add a 256-wide pass, then a 5-level simd_sum.
fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 9;
}

fn fillBf(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

/// Gate 4's row: integers in [1, 32] whose second half mirrors the first less 1 at its first 8 places.
fn stressRow(rng: std.Random, x: []u16) void {
    const h = x.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 32));
        x[i] = c.toBf16(v);
        x[h + i] = c.toBf16(v - @as(f32, if (i < 8) 1 else 0));
    }
}

/// Gate 4's weight row: integers, the second half their negatives, so a row's sum is that of its first 8 weights,
/// which are in [1, 6]; the rest are in [1, 128], so even a lane's 1/32 of a 896-wide row climbs past fp16's exact
/// integers, and a whole 6848-wide row stays under 2^24 (3424 x 32 x 128) in any order.
fn stressWeights(w: []u16, salt: usize) void {
    const h = w.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt(if (i < 8) 1 + (i * 7 + salt) % 6 else 1 + (i * 37 + salt * 11) % 128);
        w[i] = c.toBf16(v);
        w[h + i] = c.toBf16(-v);
    }
}

fn dot(x: []const u16, w: []const u16, mag: *f64) f64 {
    var s: f64 = 0;
    var m: f64 = 0;
    for (x, w) |a, b| {
        const t = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
        s += t;
        m += @abs(t);
    }
    mag.* = m;
    return s;
}

fn dotIn(acc: c.Accum, order: g.Order, x: []const u16, w: []const u16, t: []f64) f64 {
    for (t, x, w) |*o, a, b| o.* = @as(f64, c.fromBf16(a)) * c.fromBf16(b);
    return contract.rows.sumIn(acc, order, t);
}

/// A control's source: the decoder's text with one edit inside `func`, which must be found.
fn edited(a: std.mem.Allocator, func: []const u8, old: []const u8, new: []const u8) ![]u8 {
    const text = uocr.source(.decoder);
    const head = try std.mem.concat(a, u8, &.{ "void ", func, "(" });
    defer a.free(head);
    const start = std.mem.indexOf(u8, text, head) orelse return error.ControlEditMissing;
    const at = std.mem.indexOfPos(u8, text, start, old) orelse return error.ControlEditMissing;
    return std.mem.concat(a, u8, &.{ text[0..at], new, text[at + old.len ..] });
}

fn build(a: std.mem.Allocator, device: mtl.Device, func: []const u8, edit: ?[2][]const u8) !mtl.Pipeline {
    const e = edit orelse return uocr.pipeline(device, .decoder, func);
    const text = try edited(a, func, e[0], e[1]);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, func, false);
}

// --- ud_matvec: y[n] = bf16(W[n, :] . x), one token a launch ----------------------------------------------------------

pub const Matvec = struct {
    pub const Variant = enum { plain, short_k, half_sums };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    xs: rig_.Ref = undefined,
    ws: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: tokens, each its own launch as decode runs them; n: output rows; k: the input width.
    pub fn spec(v: Variant) c.Spec {
        return .{
            .name = switch (v) {
                .plain => "metal uocr matvec ud_matvec",
                .short_k => "metal uocr matvec ud_matvec control: the last 8 inputs dropped",
                .half_sums => "metal uocr matvec ud_matvec control: half sums",
            },
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 256, .step = 8, .min = 8 } },
            // q/k/v/o, the router's q/k, the dense FFN's gate/up and down, the shared expert's, the lm_head
            .served = &.{
                .{ .rows = 1, .n = H, .k = H },
                .{ .rows = 1, .n = uocr.router_dim, .k = H },
                .{ .rows = 1, .n = uocr.dense_ffn, .k = H },
                .{ .rows = 1, .n = H, .k = uocr.dense_ffn },
                .{ .rows = 1, .n = uocr.shared_ffn, .k = H },
                .{ .rows = 1, .n = H, .k = uocr.shared_ffn },
                .{ .rows = 1, .n = uocr.vocab, .k = H },
            },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !Matvec {
        const edit: ?[2][]const u8 = switch (v) {
            .plain => null,
            .short_k => .{ "k < p.k; k += 256", "k + 8 < p.k; k += 256" },
            .half_sums => .{ "float acc[4]", "half acc[4]" },
        };
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, "ud_matvec", edit), .v = v };
    }
    pub fn deinit(t: *Matvec) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Matvec) void {
        t.a.free(t.x);
        t.a.free(t.w);
        t.x = &.{};
        t.w = &.{};
    }
    pub fn kernel(t: *Matvec) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = &vt, .control = switch (t.v) {
            .plain => null,
            .short_k => .shapes,
            .half_sums => .sums,
        } };
    }
    const vt = g.hooks(Matvec);

    pub fn prepare(t: *Matvec, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 8 != 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.n;
        const k = shape.k;
        t.x = try t.a.alloc(u16, shape.rows * k);
        t.w = try t.a.alloc(u16, n * k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) {
            for (0..shape.rows) |r| stressRow(rng, t.x[r * k ..][0..k]);
            for (0..n) |j| stressWeights(t.w[j * k ..][0..k], j);
        } else {
            fillBf(rng, t.x, -2, 2);
            fillBf(rng, t.w, -0.25, 0.25);
        }
        if (fill == .nonfinite) {
            // tokens 1-4 hold NaN, +Inf, -Inf and both infinities; token 0 meets them in weight rows 3, 6 and n - 1
            const xs = [_][3]usize{ .{ 1, k / 3, 0 }, .{ 2, k / 2, 1 }, .{ 3, k - 1, 2 }, .{ 4, 0, 1 }, .{ 4, k - 1, 2 } };
            for (xs) |p| if (p[0] < shape.rows) {
                t.x[p[0] * k + p[1]] = c.toBf16(uocr.planted[p[2]]);
            };
            const ws = [_][3]usize{ .{ 3, 5, 0 }, .{ 6, k - 1, 2 }, .{ n - 1, k - 2, 1 } };
            for (ws) |p| if (p[0] < n) {
                t.w[p[0] * k + p[1]] = c.toBf16(uocr.planted[p[2]]);
            };
        }
        t.xs = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ws = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    /// lin()'s launch a token: (N + 31) / 32 groups of 256 threads.
    pub fn run(t: *Matvec) !void {
        const n = t.shape.n;
        const k = t.shape.k;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ws.buf, t.ws.off, 1);
        p.e.setValue(UdMv{ .n = @intCast(n), .k = @intCast(k) }, 3);
        for (0..t.shape.rows) |r| {
            p.e.setBuffer(t.xs.buf, t.xs.off + r * k * 2, 0);
            p.e.setBuffer(t.out.buf, t.out.off + r * n * 2, 2);
            p.e.dispatchGroups(mtl.Size.of((n + 31) / 32, 1, 1), mtl.Size.of(256, 1, 1));
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Matvec) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *Matvec, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        const k = t.shape.k;
        for (want, mag, 0..) |*w, *m, i| w.* = dot(t.x[(i / n) * k ..][0..k], t.w[(i % n) * k ..][0..k], m);
    }

    pub fn simulate(t: *Matvec, acc: c.Accum, order: g.Order, out: []f64) !void {
        const n = t.shape.n;
        const k = t.shape.k;
        const terms = try t.a.alloc(f64, k);
        defer t.a.free(terms);
        for (out, 0..) |*o, i| o.* = dotIn(acc, order, t.x[(i / n) * k ..][0..k], t.w[(i % n) * k ..][0..k], terms);
    }
};

// --- ud_expert_gu and ud_expert_down: one token's routed experts through the expert table ------------------------------

/// A value's spread through fp32 steps and bf16 rounding points.
const Range = struct {
    lo: f64,
    hi: f64,
    fn of(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// An fp32 step good to `ulps` units in the last place (a correctly rounded one: 0.5).
    fn fp32(a: Range, ulps: f64) Range {
        const u = ulps * 0x1p-23;
        return .{ .lo = a.lo - u * @abs(a.lo) - 0x1p-149, .hi = a.hi + u * @abs(a.hi) + 0x1p-149 };
    }
    fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    fn join(a: Range, b: Range) Range {
        return .{ .lo = @min(a.lo, b.lo), .hi = @max(a.hi, b.hi) };
    }
};

// Metal's bounds under the engine's options (fast math on; the kernel's own precise::exp): exp 4 ulps, fast division
// 2.5, the add correctly rounded.
const exp_ulps = 4;
const div_ulps = 2.5;

/// gs / (1 + exp(-gs)) at a bf16 point through precise exp, the add and the division; past fp32's range exp is +Inf.
fn siluAt(x: f64) Range {
    var e = Range.of(@exp(-x)).fp32(exp_ulps);
    if (e.hi > std.math.floatMax(f32)) e.hi = std.math.inf(f64);
    const d = Range{ .lo = e.lo + 1, .hi = e.hi + 1 };
    const den = (Range{ .lo = d.lo - 0x1p-24 * d.lo, .hi = d.hi + 0x1p-24 * d.hi });
    return Range.of(x).mul(.{ .lo = 1 / den.hi, .hi = 1 / den.lo }).fp32(div_ulps);
}

/// SiLU's least point: below it SiLU falls, above it rises.
const silu_min_at: f64 = -1.2784645427610738;

fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

/// SiLU over every bf16 gate between `lo` and `hi`: both ends' ranges, and the least point when it lies between.
fn siluOver(lo: f64, hi: f64) Range {
    var r = siluAt(lo).join(siluAt(hi));
    if (lo < silu_min_at and silu_min_at < hi) r = r.join(siluAt(silu_min_at));
    return r;
}

/// How many experts the table holds (a layer's), how many of them the slots use, and each slot's pick among those:
/// four distinct experts at six slots, two of them picked twice (slots 1 and 3, 0 and 5); three, one twice, at four.
const table_len = uocr.experts;
const distinct = 4;
const pick = [8]usize{ 0, 1, 2, 1, 3, 0, 2, 3 };

pub const Experts = struct {
    pub const Variant = enum { gu, down, gu_slot0, gu_no_point, down_slot0 };
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Variant,
    fill: g.Fill = .random,
    shape: c.Shape = undefined,
    ids: [8]u32 = undefined,
    x: []u16 = &.{},
    /// Each distinct expert's matrices: gu, gate then up ([F, 1280] each); down, [1280, F].
    w: [distinct][]u16 = @splat(&.{}),
    xs: rig_.Ref = undefined,
    wbuf: [distinct]rig_.Ref = undefined,
    poison: rig_.Ref = undefined,
    tab: rig_.Ref = undefined,
    idbuf: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn up(v: Variant) bool {
        return v != .down and v != .down_slot0;
    }

    /// gu: rows are slots, n the expert's width F, k the residual (1280 in the kernel); down: rows are slots, n the
    /// residual (1280 in the kernel), k the width F.
    pub fn spec(v: Variant) c.Spec {
        const gu = up(v);
        return .{
            .name = switch (v) {
                .gu => "metal uocr matvec ud_expert_gu",
                .down => "metal uocr matvec ud_expert_down",
                .gu_slot0 => "metal uocr matvec ud_expert_gu control: every slot on slot 0's expert",
                .gu_no_point => "metal uocr matvec ud_expert_gu control: silu without its bf16 point",
                .down_slot0 => "metal uocr matvec ud_expert_down control: every slot on slot 0's expert",
            },
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            // gu: moe() launches F / 32 groups, so a width is whole groups; down: lanes take 8 inputs a step
            .tiles = if (gu) .{ .{}, .{ .size = 32, .step = 32, .min = 32 }, .{} } else .{ .{}, .{}, .{ .size = 256, .step = 8, .min = 8 } },
            .served = if (gu) &.{.{ .rows = uocr.topk, .n = F, .k = H }} else &.{.{ .rows = uocr.topk, .n = H, .k = F }},
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant) !Experts {
        const slot0 = [2][]const u8{ "tab[ids[s]]", "tab[ids[0]]" };
        const edit: ?[2][]const u8 = switch (v) {
            .gu, .down => null,
            .gu_slot0, .down_slot0 => slot0,
            .gu_no_point => .{ "ud_bf(gs / (1.0f + metal::precise::exp(-gs)))", "(gs / (1.0f + metal::precise::exp(-gs)))" },
        };
        const func = if (up(v)) "ud_expert_gu" else "ud_expert_down";
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, func, edit), .v = v };
    }
    pub fn deinit(t: *Experts) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Experts) void {
        t.a.free(t.x);
        t.x = &.{};
        for (&t.w) |*w| {
            t.a.free(w.*);
            w.* = &.{};
        }
    }
    pub fn kernel(t: *Experts) g.Kernel {
        return .{ .spec = spec(t.v), .ctx = t, .vt = if (up(t.v)) &vt_gu else &vt_down, .control = switch (t.v) {
            .gu, .down => null,
            else => .shapes,
        } };
    }
    const vt_gu = g.hooks(Experts);
    const vt_down = blk: {
        var v = vt_gu;
        v.bounds = null;
        v.intervals = null;
        break :blk v;
    };

    pub fn prepare(t: *Experts, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > pick.len) return false;
        if (up(t.v) and (shape.k != H or shape.n == 0 or shape.n % 32 != 0)) return false;
        if (!up(t.v) and (shape.n != H or shape.k == 0 or shape.k % 8 != 0)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // distinct ids anywhere in the layer's 64
        var chosen: [distinct]u32 = undefined;
        var have: usize = 0;
        while (have < distinct) {
            const id = rng.uintLessThan(u32, table_len);
            if (std.mem.indexOfScalar(u32, chosen[0..have], id) == null) {
                chosen[have] = id;
                have += 1;
            }
        }
        for (t.ids[0..shape.rows], pick[0..shape.rows]) |*id, p| id.* = chosen[p];
        // x: gu's token row, down's act rows (one a slot)
        const xrows: usize = if (up(t.v)) 1 else shape.rows;
        t.x = try t.a.alloc(u16, xrows * k);
        const mats: usize = if (up(t.v)) 2 else 1;
        for (&t.w) |*w| w.* = try t.a.alloc(u16, mats * n * k);
        if (fill == .stress) {
            for (0..xrows) |r| stressRow(rng, t.x[r * k ..][0..k]);
            for (t.w, 0..) |w, e| for (0..mats * n) |j| stressWeights(w[j * k ..][0..k], j + 3 * e);
        } else {
            fillBf(rng, t.x, -2, 2);
            for (t.w) |w| fillBf(rng, w, -0.25, 0.25);
        }
        if (fill == .nonfinite) t.plant();
        for (t.w, &t.wbuf) |w, *b| b.* = try t.rig.input(std.mem.sliceAsBytes(w));
        // every table entry the slots do not pick points at NaN rows, so a wrong pick shows
        const nans = try t.a.alloc(u16, mats * n * k);
        defer t.a.free(nans);
        @memset(nans, c.toBf16(nan));
        t.poison = try t.rig.input(std.mem.sliceAsBytes(nans));
        const bad = t.poison.buf.gpuAddress() + t.poison.off;
        var tab: [table_len]UdExpert = @splat(.{ .gate = bad, .up = bad, .down = bad });
        for (chosen, t.wbuf) |id, b| {
            const at = b.buf.gpuAddress() + b.off;
            tab[id] = if (up(t.v)) .{ .gate = at, .up = at + n * k * 2, .down = bad } else .{ .gate = bad, .up = bad, .down = at };
        }
        t.xs = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.tab = try t.rig.input(std.mem.sliceAsBytes(&tab));
        t.idbuf = try t.rig.input(std.mem.sliceAsBytes(t.ids[0..shape.rows]));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    /// gu (slots on experts 0 1 2 1 3 0): NaN in expert 0's gate row 0, +Inf in expert 1's up row 1, -Inf in expert
    /// 2's gate row 2 (SiLU's NaN), both infinities in expert 3's up row 3 and +Inf in its gate row 4. down: act slots
    /// 1-4 hold NaN, +Inf, -Inf and both; expert 0's down row 7 a NaN, which slots 0 and 5 meet.
    fn plant(t: *Experts) void {
        const n = t.shape.n;
        const k = t.shape.k;
        if (up(t.v)) {
            const at = [_][5]usize{ .{ 0, 0, 0, 100, 0 }, .{ 1, 1, 1, 640, 1 }, .{ 2, 0, 2, k - 1, 2 }, .{ 3, 1, 3, 0, 1 }, .{ 3, 1, 3, k - 1, 2 }, .{ 3, 0, 4, 7, 1 } };
            for (at) |p| t.w[p[0]][(p[1] * n + p[2]) * k + p[3]] = c.toBf16(uocr.planted[p[4]]);
            return;
        }
        const at = [_][3]usize{ .{ 1, k / 3, 0 }, .{ 2, k / 2, 1 }, .{ 3, k - 1, 2 }, .{ 4, 0, 1 }, .{ 4, k - 1, 2 } };
        for (at) |p| if (p[0] < t.shape.rows) {
            t.x[p[0] * k + p[1]] = c.toBf16(uocr.planted[p[2]]);
        };
        t.w[0][7 * k + 5] = c.toBf16(nan);
    }

    /// moe()'s decode launch: (F / 32, slots) groups for gu, (1280 / 32, slots) for down, 256 threads each.
    pub fn run(t: *Experts) !void {
        const p = t.rig.begin();
        const e = p.e;
        for (t.wbuf) |b| e.useResource(b.buf, mtl.ResourceUsage.read);
        e.useResource(t.poison.buf, mtl.ResourceUsage.read);
        e.setPipeline(t.pipe);
        e.setBuffer(t.xs.buf, t.xs.off, 0);
        e.setBuffer(t.tab.buf, t.tab.off, 1);
        e.setBuffer(t.idbuf.buf, t.idbuf.off, 2);
        e.setBuffer(t.out.buf, t.out.off, 3);
        const ffn = if (up(t.v)) t.shape.n else t.shape.k;
        e.setValue(UdExpertArgs{ .ffn = @intCast(ffn) }, 4);
        e.dispatchGroups(mtl.Size.of(t.shape.n / 32, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Experts) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Output i's row of the slot's expert (gu: `m` 0 gate, 1 up) and its input row.
    fn row(t: *const Experts, i: usize, m: usize) [2][]const u16 {
        const n = t.shape.n;
        const k = t.shape.k;
        const s = i / n;
        const w = t.w[pick[s]];
        const xr = if (up(t.v)) t.x[0..k] else t.x[s * k ..][0..k];
        return .{ xr, w[(m * n + i % n) * k ..][0..k] };
    }

    /// gu's float64 at its rounding points: bf16(bf16(silu(bf16(G))) * bf16(U)).
    fn act(gate: f64, upv: f64) f64 {
        return bf(bf(silu(bf(gate))) * bf(upv));
    }

    pub fn reference(t: *Experts, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const a = t.row(i, 0);
            if (!up(t.v)) {
                w.* = dot(a[0], a[1], m);
                continue;
            }
            var mg: f64 = 0;
            var mu: f64 = 0;
            const gate = dot(a[0], a[1], &mg);
            const b = t.row(i, 1);
            w.* = act(gate, dot(b[0], b[1], &mu));
            m.* = mg + mu;
        }
    }

    /// gu's range: each sum within gamma(depth) of its terms (exact on gate 4's integers), rounded to bf16, SiLU's fp32
    /// steps over every gate between, bf16, times the up's bf16 (exact in fp32), bf16.
    fn range(t: *const Experts, i: usize) Range {
        var ends: [2]Range = undefined;
        for (&ends, 0..) |*r, m| {
            const a = t.row(i, m);
            var mag: f64 = 0;
            const v = dot(a[0], a[1], &mag);
            const e = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(depth(t.shape)), c.unit(.f32)) * mag;
            r.* = (Range{ .lo = v - e, .hi = v + e }).bf16();
        }
        return siluOver(ends[0].lo, ends[0].hi).bf16().mul(ends[1]).bf16();
    }

    pub fn intervals(t: *Experts, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *h, i| {
            const r = t.range(i);
            l.* = r.lo;
            h.* = r.hi;
        }
    }

    /// The farther end of the range from float64's value at the rounding points, and the bf16 floor.
    pub fn bounds(t: *Experts, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, 0..) |*l, w, i| {
            const r = t.range(i);
            l.* = @max(@abs(w - r.lo), @abs(r.hi - w)) + 0x1p-133;
        }
    }

    pub fn simulate(t: *Experts, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(terms);
        for (out, 0..) |*o, i| {
            const a = t.row(i, 0);
            const s = dotIn(acc, order, a[0], a[1], terms);
            if (!up(t.v)) {
                o.* = s;
                continue;
            }
            const b = t.row(i, 1);
            o.* = act(s, dotIn(acc, order, b[0], b[1], terms));
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    owned: std.ArrayList(Owned) = .empty,

    const Owned = struct { ctx: *anyopaque, free: *const fn (*anyopaque, std.mem.Allocator) void };

    fn add(s: *Suite, comptime T: type, only: ?[]const u8, v: T.Variant) !void {
        if (only) |o| if (std.mem.indexOf(u8, T.spec(v).name, o) == null) return;
        const t = try s.a.create(T);
        errdefer s.a.destroy(t);
        t.* = try T.init(s.a, s.device, s.queue, v);
        const free = struct {
            fn f(p: *anyopaque, a: std.mem.Allocator) void {
                const x: *T = @ptrCast(@alignCast(p));
                x.deinit();
                a.destroy(x);
            }
        }.f;
        try s.owned.append(s.a, .{ .ctx = t, .free = free });
        try s.kernels.append(s.a, t.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Matvec.Variant)) |v| try s.add(Matvec, only, v);
        for (std.enums.values(Experts.Variant)) |v| try s.add(Experts, only, v);
    }
    pub fn deinit(s: *Suite) void {
        for (s.owned.items) |o| o.free(o.ctx, s.a);
        s.owned.deinit(s.a);
    }
};

/// Gate 5's texts for a `uocr matvec ` registration: the decoder's source (no nax.h, so one text for every chip).
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?uocr.Texts {
    if (!std.mem.startsWith(u8, name, "uocr matvec ")) return null;
    const text = uocr.source(.decoder);
    return .{ .here = text, .pre_m5 = text };
}
