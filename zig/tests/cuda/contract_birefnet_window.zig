//! tf_br_window under the contract: Swin's window attention (bias, shift mask) in fp32 against float64.
const std = @import("std");
const cm = @import("contract_birefnet_common.zig");

const c = cm.c;
const g = cm.g;
const ks = cm.ks;
const uniform = cm.uniform;
const plant = cm.plant;
const U = cm.U;
const N = 144;
const D = 32;

pub const Window = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    name: []const u8,
    shift: usize,
    shape: c.Shape = undefined,
    qkv: []f32 = &.{},
    table: []f32 = &.{},
    allow: []f64 = &.{},
    ins: [2]u64 = @splat(0),
    out: u64 = 0,

    /// The longest chain: P V's 144 products (Q K^T's are 32, the softmax's total 38).
    fn depth(_: c.Shape) usize {
        return N;
    }

    /// Rows: windows a side (the grid 12 rows on a side); n: the width C (C / 32 heads); k: the 144 keys.
    pub fn spec(name: []const u8) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .tiles = .{ .{ .size = 2 }, .{}, .{} },
            .served = &.{ .{ .rows = 3, .n = 192, .k = N }, .{ .rows = 2, .n = 384, .k = N }, .{ .rows = 1, .n = 768, .k = N }, .{ .rows = 1, .n = 1536, .k = N } },
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "a softmax's terms: no input keeps fp32's sums exact through exp and the division" },
        };
    }

    pub fn deinit(t: *Window) void {
        t.a.free(t.qkv);
        t.a.free(t.table);
        t.a.free(t.allow);
        t.rig.deinit();
    }
    pub fn kernel(t: *Window, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Window);

    fn heads(t: *const Window) usize {
        return t.shape.n / D;
    }
    fn windows(t: *const Window) usize {
        return t.shape.rows * t.shape.rows;
    }

    pub fn fill(t: *Window, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % D != 0 or shape.k != N) return false;
        t.a.free(t.qkv);
        t.a.free(t.table);
        t.shape = shape;
        t.qkv = try t.a.alloc(f32, t.windows() * N * 3 * shape.n);
        t.table = try t.a.alloc(f32, 529 * t.heads());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        uniform(rng, t.qkv, -2, 2);
        uniform(rng, t.table, -3, 3);
        if (mode == .nonfinite) plant(t.qkv, 3 * shape.n);
        return true;
    }

    pub fn prepare(t: *Window, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.table));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Window) !void {
        const side = 12 * t.shape.rows;
        try ks.window(t.k, t.rig.s, t.ins[0], t.ins[1], t.out, t.shape.n, t.heads(), side, side, t.shift);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Window) usize {
        return t.windows() * N * t.shape.n;
    }

    /// Token i's region of window (wy, wx) in the shifted grid: rows and columns split at side - 12 and side - shift.
    fn region(t: *const Window, i: usize, wy: usize, wx: usize) usize {
        const side = 12 * t.shape.rows;
        const y = wy * 12 + i / 12;
        const x = wx * 12 + i % 12;
        const ry: usize = if (y < side - 12) 0 else if (y < side - t.shift) 1 else 2;
        const rx: usize = if (x < side - 12) 0 else if (x < side - t.shift) 1 else 2;
        return ry * 3 + rx;
    }

    /// Logits (fp32-scaled q) . k + bias (+ mask), softmax, P V; the allowance tracks each logit's error through both.
    fn compute(t: *Window, acc: ?struct { c.Accum, g.Order }, want: []f64, allow: ?[]f64) !void {
        const C = t.shape.n;
        const ld = 3 * C;
        const scale: f32 = @floatCast(std.math.pow(f64, D, -0.5));
        var s: [N]f64 = undefined;
        var err: [N]f64 = undefined;
        var p: [N]f64 = undefined;
        var terms: [N]f64 = undefined;
        for (0..t.windows()) |win| for (0..t.heads()) |h| for (0..N) |qi| {
            const wy = win / t.shape.rows;
            const wx = win % t.shape.rows;
            const base = win * N;
            const q = t.qkv[(base + qi) * ld + h * D ..][0..D];
            const mine = t.region(qi, wy, wx);
            var most = -std.math.inf(f64);
            for (0..N) |j| {
                const k = t.qkv[(base + j) * ld + C + h * D ..][0..D];
                var dot: f64 = 0;
                var size: f64 = 0;
                for (0..D) |d| {
                    const qs: f32 = q[d] * scale;
                    terms[d] = @as(f64, qs) * k[d];
                    dot += terms[d];
                    size += @abs(terms[d]);
                }
                if (acc) |ao| dot = cm.rows.sumIn(ao[0], ao[1], terms[0..D]);
                const at = ((qi / 12 + 11 - j / 12) * 23 + (qi % 12 + 11 - j % 12)) * t.heads() + h;
                const bias: f64 = t.table[at];
                const masked: f64 = if (t.shift > 0 and t.region(j, wy, wx) != mine) -100 else 0;
                s[j] = dot + bias + masked;
                err[j] = c.gamma(D, U) * size + U * (@abs(dot + bias) + @abs(s[j]));
                most = @max(most, s[j]);
            }
            var total: f64 = 0;
            var worst: f64 = 0;
            for (0..N) |j| {
                p[j] = @exp(s[j] - most);
                total += p[j];
                err[j] += U * @abs(s[j] - most) + 2 * cm.exp_ulps * U;
                worst = @max(worst, err[j]);
            }
            if (acc) |ao| total = cm.rows.sumIn(ao[0], ao[1], &p);
            for (&p) |*v| v.* /= total;
            for (0..D) |d| {
                var o: f64 = 0;
                var spread: f64 = 0;
                var size: f64 = 0;
                for (0..N) |j| {
                    const v: f64 = t.qkv[(base + j) * ld + 2 * C + h * D + d];
                    terms[j] = p[j] * v;
                    o += terms[j];
                    size += @abs(terms[j]);
                    spread += @abs(terms[j]) * (err[j] + worst);
                }
                if (acc) |ao| o = cm.rows.sumIn(ao[0], ao[1], &terms);
                const i = (base + qi) * C + h * D + d;
                want[i] = o;
                const al = allow orelse continue;
                const e = 1.01 * (spread + (c.gamma(38, U) + 2 * cm.div_ulps * U + 2 * U) * size) + c.gamma(N, U) * size + U * @abs(o);
                al[i] = if (std.math.isFinite(o)) e + 0x1p-149 else 0;
            }
        };
    }

    pub fn reference(t: *Window, want: []f64, mag: []f64) !void {
        t.a.free(t.allow);
        t.allow = try t.a.alloc(f64, t.count());
        try t.compute(null, want, t.allow);
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }
    pub fn bounds(t: *Window, _: []const f64, _: []const f64, allow: []f64) !void {
        @memcpy(allow, t.allow);
    }
    pub fn simulate(t: *Window, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.compute(.{ acc, order }, out, null);
    }
};
