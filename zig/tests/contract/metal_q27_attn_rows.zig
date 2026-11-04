//! qwen27 attention's decode rows and prompt chunks on Metal: Attention.encode's whole calls against float64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const at_ = @import("metal_q27_attn.zig");

const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const q = tf.qwen27;
const attn = q.attention;
const planning = q.attention_plan;

const D = 256;
const u = 0x1p-24;
const scale = 0.0625;

/// One Mac's head counts (a rank's 12 over 2 merges as a pair, so its rows are not one call's).
pub const heads = [_]at_.Heads{ .q27b, .gqa16, .q9b };

/// decode: a round's trees and chains as decode_round.zig plans them; prompt: one chain through the prompt kernel.
pub const Which = enum { decode, prompt };

pub const Reg = struct { heads: at_.Heads, which: Which, units: bool = true };

/// A row that must not see a key gives it probability 0, and 0 times an infinite value in P V is NaN.
const masked_inf = c.Known{ .gate = .nonfinite, .finding = "a row that must not see a key gives it probability 0, and 0 times an infinite value in P V gives NaN where float64 gives the row's own value" };

/// The error is no single sum's, so `mag` carries the call's whole first-order budget in units of one fp32 rounding.
fn oneRounding(_: c.Shape) usize {
    return 1;
}

/// The fewest rows of a stream that fill `tiles` 16-row tiles at this group (prefix_2, _3, _5 and _6 on an M5).
fn rowsFor(comptime h: at_.Heads, comptime tiles: usize) usize {
    const group: usize = comptime at_.geometry(h, false).group();
    return 16 * (tiles - 1) / group + 1;
}

/// Shapes as rows of the round, query heads and the first stream's cached keys; past 16 rows, streams of 16.
pub fn served(comptime h: at_.Heads, comptime w: Which) []const c.Shape {
    const n: usize = comptime at_.geometry(h, false).heads;
    const two = comptime rowsFor(h, 2);
    const three = comptime rowsFor(h, 3);
    const five = comptime rowsFor(h, 5);
    const six = comptime rowsFor(h, 6);
    // even starts fork at 3 rows or more; 2046 and 510 split a tree over two tail chunks; 8190 and 8191 take 64 chunks
    return comptime switch (w) {
        .decode => &.{ .{ .rows = 1, .n = n, .k = 2 }, .{ .rows = two, .n = n, .k = 64 }, .{ .rows = three, .n = n, .k = 64 }, .{ .rows = two, .n = n, .k = 65 }, .{ .rows = three, .n = n, .k = 65 }, .{ .rows = 16, .n = n, .k = 2046 }, .{ .rows = 16, .n = n, .k = 2047 }, .{ .rows = 128, .n = n, .k = 510 }, .{ .rows = 128, .n = n, .k = 511 }, .{ .rows = 1, .n = n, .k = 8191 }, .{ .rows = five, .n = n, .k = 8190 }, .{ .rows = six, .n = n, .k = 8191 } },
        .prompt => &.{ .{ .rows = 16, .n = n, .k = 0 }, .{ .rows = 128, .n = n, .k = 0 }, .{ .rows = 128, .n = n, .k = 1920 }, .{ .rows = 16, .n = n, .k = 8176 } },
    };
}

fn nameOf(comptime r: Reg) []const u8 {
    return std.fmt.comptimePrint("metal qwen27 attention {s} {s}", .{ comptime at_.label(r.heads), if (r.which == .decode) "decode rows" else "prompt rows" });
}

/// fp32 scores and sums, half probabilities into P V, the gate's bf16 sigmoid; `mids`: bf16(o) and bf16(sigmoid).
pub fn spec(r: Reg) c.Spec {
    const known = r.which == .prompt or r.units;
    return switch (r.heads) {
        inline else => |h| switch (r.which) {
            inline else => |w| .{ .name = nameOf(.{ .heads = h, .which = w }), .accum = .f32, .depth = oneRounding, .ops = 8, .mids = 2, .out = .bf16, .known = if (known) masked_inf else null, .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 512 } }, .served = served(h, w) },
        },
    };
}

fn bf(x: u16) f64 {
    return c.fromBf16(x);
}

pub const Attn = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    reg: Reg,
    op: attn.Attention,
    tensor: bool,
    st: ?at_.State = null,
    rd: at_.Round = .{},
    q_in: []u16 = &.{},
    gate_in: []u16 = &.{},
    k_in: []u16 = &.{},
    v_in: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    refs: [5]rig_.Ref = undefined,
    caches: [planning.max_streams]attn.Cache = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, r: Reg) !Attn {
        const tensor = device.tensorUnits();
        if (!tensor) try q.simd_attention.check(device, queue);
        const gm = at_.geometry(r.heads, tensor);
        const text = try q.simd_attention.grouped(a, tensor, true, gm.group(), gm.kv_heads);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const op = try attn.Attention.init(device, lib);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .reg = r, .op = op, .tensor = tensor };
    }

    pub fn deinit(t: *Attn) void {
        t.release();
        t.op.deinit();
        t.rig.deinit();
    }

    fn release(t: *Attn) void {
        if (t.st) |*s| s.deinit();
        t.st = null;
        for ([_]*[]u16{ &t.q_in, &t.gate_in, &t.k_in, &t.v_in, &t.kc, &t.vc }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
    }

    pub fn kernel(t: *Attn) g.Kernel {
        return .{ .spec = spec(t.reg), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(Attn);

    fn geo(t: *const Attn) planning.Geometry {
        return at_.geometry(t.reg.heads, t.tensor);
    }

    fn at(t: *const Attn, s: usize, hk: usize, row: usize) usize {
        return t.rd.at(t.geo().kv_heads, s, hk, row);
    }

    pub fn prepare(t: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const gm = t.geo();
        const nh: usize = gm.heads;
        const hk: usize = gm.kv_heads;
        if (shape.n != nh or shape.rows == 0 or shape.rows > planning.max_rows) return false;
        // before M5 the prompt kernel is attention_prompt_sg, built where a group holds its threads
        if (t.reg.which == .prompt and t.op.prompt == null) return false;
        t.release();
        t.rig.reset();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.rd = at_.Round.init(shape, t.reg.which == .prompt, t.reg.which == .decode, rng);
        const r = &t.rd;
        t.q_in = try t.a.alloc(u16, r.rows * nh * D);
        t.gate_in = try t.a.alloc(u16, r.rows * nh * 2 * D);
        t.k_in = try t.a.alloc(u16, r.rows * hk * D);
        t.v_in = try t.a.alloc(u16, r.rows * hk * D);
        t.kc = try t.a.alloc(u16, r.streams * hk * r.cap * D);
        t.vc = try t.a.alloc(u16, t.kc.len);
        for ([_][]u16{ t.q_in, t.gate_in, t.kc, t.vc }) |xs| for (xs) |*x| {
            x.* = c.toBf16(rng.float(f32) * 4 - 2);
        };
        if (fill == .stress) t.stress(rng);
        if (fill == .nonfinite) t.plant();
        // the round's keys as they sit in the caches: append writes the same bits (io's append check proves where)
        for (0..r.streams) |s| for (0..r.counts[s]) |i| for (0..hk) |h| {
            const row = ((r.firsts[s] + i) * hk + h) * D;
            @memcpy(t.k_in[row..][0..D], t.kc[t.at(s, h, r.starts[s] + i)..][0..D]);
            @memcpy(t.v_in[row..][0..D], t.vc[t.at(s, h, r.starts[s] + i)..][0..D]);
        };
        for ([_][]u16{ t.q_in, t.k_in, t.v_in, t.gate_in }, 0..) |xs, i| t.refs[i] = try t.rig.input(std.mem.sliceAsBytes(xs));
        const per = hk * r.cap * D;
        for (0..r.streams) |s| {
            const keys = try t.rig.input(std.mem.sliceAsBytes(t.kc[s * per ..][0..per]));
            const values = try t.rig.input(std.mem.sliceAsBytes(t.vc[s * per ..][0..per]));
            t.caches[s] = .{ .keys = at_.ref(keys), .values = at_.ref(values), .capacity = @intCast(r.cap) };
        }
        t.refs[4] = try t.rig.output(r.rows * nh * D * 2);
        t.st = try at_.State.init(t.a, t.rig.device, r, gm, t.caches[0..r.streams]);
        return true;
    }

    /// Gate 1: key 0 zero, +Inf in its value; -Inf in row 1's value (row 0 must not see it); NaN in a key, query, gate.
    fn plant(t: *Attn) void {
        const gm = t.geo();
        const nh: usize = gm.heads;
        const r = &t.rd;
        const s0 = r.starts[0];
        @memset(t.kc[t.at(0, 0, 0)..][0..D], 0);
        t.vc[t.at(0, 0, 0) + 3] = c.toBf16(std.math.inf(f32));
        if (r.counts[0] > 1) t.vc[t.at(0, 0, s0 + 1) + 3] = c.toBf16(-std.math.inf(f32));
        if (r.counts[0] > 2) t.kc[t.at(0, gm.kv_heads - 1, s0 + 2) + 10] = c.toBf16(std.math.nan(f32));
        if (r.rows > 3) t.q_in[(3 * nh + 1) * D] = c.toBf16(std.math.nan(f32));
        if (r.rows > 4) t.gate_in[4 * nh * 2 * D + D + 7] = c.toBf16(std.math.nan(f32));
    }

    /// Gate 4: exact fp32 sums past fp16's range; key 0 scores 32 (value 1), the rest 0 (1,024); scores alone separate.
    fn stress(t: *Attn, rng: std.Random) void {
        // queries 32 and 33, keys 64 m (m in 33-60) mirrored in reverse: no bf16 order cancels, every sum stays < 2^24
        const gm = t.geo();
        for (0..t.rd.streams) |s| for (0..gm.kv_heads) |h| for (0..t.rd.cap) |row| {
            const k = t.kc[t.at(s, h, row)..][0..D];
            for (0..D / 2) |d| {
                const m: f32 = @floatFromInt(33 + rng.uintLessThan(u32, 28));
                k[d] = c.toBf16(64 * m);
                k[D - 1 - d] = c.toBf16(-64 * m + (if (row == 0 and d == 0) @as(f32, 16) else 0));
            }
            @memset(t.vc[t.at(s, h, row)..][0..D], c.toBf16(if (row == 0) 1 else 1024));
        };
        for (0..t.q_in.len / D) |i| for (t.q_in[i * D ..][0..D], 0..) |*x, d| {
            x.* = c.toBf16(@floatFromInt(32 + @min(d, D - 1 - d) % 2));
        };
    }

    pub fn run(t: *Attn) !void {
        const s = &t.st.?;
        const p = t.rig.begin();
        t.op.encode(p.e, &s.scratch, &s.binding, s.plan, at_.ref(t.refs[0]), at_.ref(t.refs[1]), at_.ref(t.refs[2]), at_.ref(t.refs[4]), t.caches[0..t.rd.streams], at_.ref(t.refs[3]), t.reg.which == .prompt, false) catch |err| {
            p.e.end();
            p.pool.pop();
            return err;
        };
        try t.rig.finish(p);
    }

    pub fn count(t: *Attn) usize {
        return t.rd.rows * t.geo().heads * D;
    }

    /// Float64 gated attention over the cached keys and each row's ancestors; with `acc`, every sum summed in it.
    fn outputs(t: *Attn, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const gm = t.geo();
        const nh: usize = gm.heads;
        const group: usize = gm.group();
        const r = &t.rd;
        const keys = try t.a.alloc(usize, r.cap);
        defer t.a.free(keys);
        const sc = try t.a.alloc(f64, r.cap);
        defer t.a.free(sc);
        const p = try t.a.alloc(f64, r.cap);
        defer t.a.free(p);
        const pv = try t.a.alloc(f64, r.cap);
        defer t.a.free(pv);
        var terms: [D]f64 = undefined;
        var o: [D]f64 = undefined;
        var b: [D]f64 = undefined;
        var f: [D]f64 = undefined;
        var dev: [D]f64 = undefined;
        var xs: [D]f64 = undefined;
        for (0..r.streams) |s| for (0..r.counts[s]) |i| {
            const node = r.firsts[s] + i;
            var chain: [planning.max_rows]usize = undefined;
            var depth: usize = 0;
            var x: i32 = @intCast(i);
            while (x >= 0) : (x = r.parents[r.firsts[s] + @as(usize, @intCast(x))]) {
                chain[depth] = @intCast(x);
                depth += 1;
            }
            const start = r.starts[s];
            for (0..start) |j| keys[j] = j;
            for (0..depth) |j| keys[start + j] = start + chain[depth - 1 - j];
            const n = start + depth;
            for (0..nh) |h| {
                const hk = h / group;
                const qv = t.q_in[(node * nh + h) * D ..][0..D];
                var top = -std.math.inf(f64);
                var eps: f64 = 0;
                for (0..n) |j| {
                    const kv = t.kc[t.at(s, hk, keys[j])..][0..D];
                    var dot: f64 = 0;
                    var abs: f64 = 0;
                    for (&terms, qv, kv) |*e, a, k| {
                        e.* = bf(a) * bf(k);
                        dot += e.*;
                        abs += @abs(e.*);
                    }
                    if (acc) |ac| dot = rows_.sumIn(ac, order, &terms);
                    sc[j] = scale * dot;
                    if (!std.math.isNan(sc[j])) top = @max(top, sc[j]);
                    eps = @max(eps, c.gamma(D + 1, u) * scale * abs + u * @abs(sc[j]));
                }
                var l: f64 = 0;
                for (0..n) |j| {
                    p[j] = @exp(sc[j] - top);
                    l += p[j];
                }
                @memset(&o, 0);
                @memset(&b, 0);
                @memset(&f, 0);
                for (0..n) |j| for (t.vc[t.at(s, hk, keys[j])..][0..D], 0..) |v16, d| {
                    const v = bf(v16);
                    o[d] += p[j] * v;
                    b[d] += p[j] * @abs(v);
                    f[d] += @abs(v);
                };
                if (acc) |ac| {
                    l = rows_.sumIn(ac, order, p[0..n]);
                    for (&o, 0..) |*e, d| {
                        for (pv[0..n], keys[0..n], p[0..n]) |*y, key, w| y.* = w * bf(t.vc[t.at(s, hk, key) + d]);
                        e.* = rows_.sumIn(ac, order, pv[0..n]);
                    }
                }
                for (&o) |*e| e.* /= l;
                @memset(&dev, 0);
                @memset(&xs, 0);
                for (0..n) |j| for (t.vc[t.at(s, hk, keys[j])..][0..D], 0..) |v16, d| {
                    const w = p[j] * @abs(bf(v16) - o[d]);
                    dev[d] += w;
                    xs[d] += w * (4 + 5 * @abs(sc[j] - top));
                };
                // scores' error 2 eps |v - o|, fast exp, fp32 sums of P V and of P, half P (and its floor), the divide
                const gn = c.gamma(@floatFromInt(2 * n + 64), u);
                for (0..D) |d| {
                    const e = (2 * eps * dev[d] + u * xs[d]) / l + gn * (b[d] / l + @abs(o[d])) + 0x1p-11 * b[d] / l + 0x1p-25 * f[d] / l + 4 * u * @abs(o[d]);
                    const sig = 1 / (1 + @exp(-bf(t.gate_in[(node * nh + h) * 2 * D + D + d])));
                    want[(node * nh + h) * D + d] = o[d] * sig;
                    mag[(node * nh + h) * D + d] = e * sig / c.gamma(1, u);
                }
            }
        };
    }

    pub fn reference(t: *Attn, want: []f64, mag: []f64) !void {
        return t.outputs(null, .sequential, want, mag);
    }

    pub fn simulate(t: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try t.a.alloc(f64, out.len);
        defer t.a.free(mag);
        return t.outputs(acc, order, out, mag);
    }
};
