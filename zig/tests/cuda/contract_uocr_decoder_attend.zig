//! tf_uocr_attend under the contract: a row's local window and routed blocks with their biases, against float64.
const std = @import("std");
const ud = @import("contract_uocr_decoder.zig");
const rig_ = @import("contract_rig.zig");

const c = ud.c;
const g = ud.g;
const mi = ud.mi;
const H = ud.H;
const HD = 128;
const B = 32;
const none = 0xffffffff;
const max_keys = 128 + 12 * B;
const u = 0x1p-24;

/// A score's 128-term dot (4 a lane, then the warp's 5 shuffles), the scale and the bias.
fn scoreDepth(_: c.Shape) usize {
    return 4 + 5 + 2;
}

fn fill(rng: std.Random, out: []u16, s: f32) void {
    for (out) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * s);
}

fn el(x: []const u16, i: usize) f64 {
    return c.fromBf16(x[i]);
}

pub const Attend = struct {
    rig: rig_.Rig,
    k: ud.Kernels,
    /// The prompt's form: the blocks' biases round to bf16 first.
    prompt: bool,
    name: []const u8,
    control: ?g.Gate = null,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    kv: [2][]u16 = .{ &.{}, &.{} },
    sel: []u32 = &.{},
    bias: []f32 = &.{},
    ins: [5]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: query rows, the last of `k` positions; n: the width (10 heads of 128).
    pub fn spec(name: []const u8, prompt: bool) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = scoreDepth,
            .out = .bf16,
            .served = if (prompt)
                &.{ .{ .rows = 2048, .n = H, .k = 2048 }, .{ .rows = 37, .n = H, .k = 1061 } }
            else
                &.{ .{ .rows = 1, .n = H, .k = 4096 }, .{ .rows = 1, .n = H, .k = 100 }, .{ .rows = 1, .n = H, .k = 16384 } },
            // the scores' dots sum exactly in fp32 at gate 4; the softmax's mass and the weighted values do not
            .sums = .{ .exact = false },
        };
    }

    pub fn deinit(t: *Attend) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Attend) void {
        const a = t.rig.gpu.gpa;
        a.free(t.q);
        a.free(t.sel);
        a.free(t.bias);
        for (&t.kv) |*x| {
            a.free(x.*);
            x.* = &.{};
        }
        t.q = &.{};
        t.sel = &.{};
        t.bias = &.{};
    }
    pub fn kernel(t: *Attend) g.Kernel {
        return .{ .spec = spec(t.name, t.prompt), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Attend);

    fn pos0(t: *const Attend) usize {
        return t.shape.k - t.shape.rows;
    }

    pub fn prepare(t: *Attend, shape: c.Shape, fill_: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k < shape.rows) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill_;
        const a = t.rig.gpu.gpa;
        const rows = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.q = try a.alloc(u16, rows * H);
        for (&t.kv) |*x| x.* = try a.alloc(u16, shape.k * H);
        if (fill_ == .stress) {
            // per head: q's halves mirror with the sign turned, keys' mirror (8 nudged): huge partials, small scores
            for (0..rows * ud.heads) |hh| for (0..HD / 2) |d| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                t.q[hh * HD + d] = c.toBf16(v);
                t.q[hh * HD + HD / 2 + d] = c.toBf16(-v);
            };
            for (0..shape.k * ud.heads) |hh| for (0..HD / 2) |d| {
                const v: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
                t.kv[0][hh * HD + d] = c.toBf16(v);
                t.kv[0][hh * HD + HD / 2 + d] = c.toBf16(v + @as(f32, if (d < 8) 1 else 0));
            };
            fill(rng, t.kv[1], 2);
        } else {
            fill(rng, t.q, 0.6);
            fill(rng, t.kv[0], 0.6);
            fill(rng, t.kv[1], 2);
        }
        // each row's read blocks: distinct remote blocks in a random order, gaps marked unread
        t.sel = try a.alloc(u32, rows * 12);
        t.bias = try a.alloc(f32, rows * 12);
        for (0..rows) |r| {
            const pos = t.pos0() + r;
            const nrem = if (pos / B >= 4) pos / B - 3 else 0;
            const s = t.sel[r * 12 ..][0..12];
            for (s, 0..) |*v, i| {
                v.* = none;
                if (nrem == 0 or rng.uintLessThan(u32, 8) == 0) continue;
                const pick: u32 = @intCast(rng.uintLessThan(usize, nrem));
                if (std.mem.indexOfScalar(u32, s[0..i], pick) == null) v.* = pick;
            }
            for (t.bias[r * 12 ..][0..12]) |*b| b.* = if (fill_ == .stress) 0 else rng.float(f32) * 5 - 4;
        }
        if (fill_ == .nonfinite) {
            // a key every row's window holds (head 0), a value (head 1, dim 9), a query (row 2, head 2)
            t.kv[0][(shape.k - 3) * H + 5] = mi.nan16;
            t.kv[1][(shape.k - 5) * H + HD + 9] = mi.inf16;
            if (rows > 2) t.q[2 * H + 2 * HD] = mi.ninf16;
        }
        t.ins[0] = try mi.input(&t.rig, t.q);
        t.ins[1] = try mi.input(&t.rig, t.kv[0]);
        t.ins[2] = try mi.input(&t.rig, t.kv[1]);
        t.ins[3] = try mi.input(&t.rig, t.sel);
        t.ins[4] = try mi.input(&t.rig, t.bias);
        t.out = try t.rig.output(rows * H * 2);
        return true;
    }

    pub fn run(t: *Attend) !void {
        const i = t.ins;
        try t.k.run("attend", t.rig.s, .{ ud.heads, t.shape.rows, 1 }, ud.kernels.attend_threads, .{ i[0], i[1], i[2], i[3], i[4], t.out, t.pos0(), @as(c_int, @intFromBool(t.prompt)) });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Attend) usize {
        return t.shape.rows * H;
    }

    /// Row r's keys: the window's, then each read block's before it, with their biases.
    fn keys(t: *Attend, r: usize, at: *[max_keys]usize, kb: *[max_keys]f64) usize {
        const pos = t.pos0() + r;
        const lo = if (pos >= 127) pos - 127 else 0;
        var n: usize = 0;
        for (lo..pos + 1) |p| {
            at[n] = p;
            kb[n] = 0;
            n += 1;
        }
        for (t.sel[r * 12 ..][0..12], t.bias[r * 12 ..][0..12]) |s, b| {
            if (s == none) continue;
            var p: usize = s * B;
            while (p < @min(lo, (s + 1) * B)) : (p += 1) {
                at[n] = p;
                kb[n] = if (t.prompt) mi.bf(b) else b;
                n += 1;
            }
        }
        return n;
    }

    /// Outputs and their sum of |p v| (sums in `acc`, `order`); `err` gets each row and head's worst score error.
    fn outputs(t: *Attend, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64, err: ?[]f64) !void {
        var at: [max_keys]usize = undefined;
        var kb: [max_keys]f64 = undefined;
        var sc: [max_keys]f64 = undefined;
        var terms: [max_keys]f64 = undefined;
        var dt: [HD]f64 = undefined;
        const isq: f64 = ud.inv_sqrt_hd;
        for (0..t.shape.rows) |r| {
            const n = t.keys(r, &at, &kb);
            for (0..ud.heads) |h| {
                const qo = r * H + h * HD;
                var top = -std.math.inf(f64);
                var worst: f64 = 0;
                for (sc[0..n], at[0..n], kb[0..n]) |*s, p, b| {
                    var dot: f64 = 0;
                    var m: f64 = 0;
                    for (&dt, 0..) |*q, d| {
                        q.* = el(t.q, qo + d) * el(t.kv[0], p * H + h * HD + d);
                        dot += q.*;
                        m += @abs(q.*);
                    }
                    if (acc != .none) dot = mi.rows.sumIn(acc, order, &dt);
                    s.* = dot * isq + b;
                    if (!std.math.isNan(s.*)) top = @max(top, s.*);
                    worst = @max(worst, (if (t.fill == .stress) 0 else c.gamma(HD, u) * m) * isq + 3 * u * @abs(s.*));
                }
                if (err) |e| e[r * ud.heads + h] = worst;
                var mass: f64 = 0;
                for (sc[0..n], terms[0..n]) |*s, *q| {
                    s.* = @exp(s.* - top);
                    q.* = s.*;
                    mass += s.*;
                }
                if (acc != .none) mass = mi.rows.sumIn(acc, order, terms[0..n]);
                for (0..HD) |d| {
                    var o: f64 = 0;
                    var m: f64 = 0;
                    for (terms[0..n], sc[0..n], at[0..n]) |*q, s, p| {
                        q.* = s / mass * el(t.kv[1], p * H + h * HD + d);
                        o += q.*;
                        m += @abs(q.*);
                    }
                    if (acc != .none) o = mi.bf(mi.rows.sumIn(acc, order, terms[0..n]));
                    out[qo + d] = o;
                    if (mag) |ms| ms[qo + d] = if (std.math.isFinite(m)) m else 0;
                }
            }
        }
    }

    pub fn reference(t: *Attend, want: []f64, mag: []f64) !void {
        try t.outputs(.none, .sequential, want, mag, null);
    }

    pub fn simulate(t: *Attend, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.outputs(acc, order, out, null, null);
    }

    /// Weights within twice the worst score error, exp, mass and division; the P x V chain; the bf16 point.
    pub fn bounds(t: *Attend, want: []const f64, mag: []const f64, allow: []f64) !void {
        const a = t.rig.gpu.gpa;
        const scratch = try a.alloc(f64, 2 * want.len);
        defer a.free(scratch);
        const err = try a.alloc(f64, t.shape.rows * ud.heads);
        defer a.free(err);
        try t.outputs(.none, .sequential, scratch[0..want.len], scratch[want.len..], err);
        // a warp's every-8th-key chain, the 8 warps' partials, the 4-term step
        const sums = c.gamma(max_keys / 8 + 8 + 2, u);
        for (allow, want, mag, 0..) |*l, w, m, i| {
            const rel = 2 * (2 * err[i / HD] + 4 * u) + c.gamma(max_keys, u) + 2 * u;
            l.* = if (std.math.isFinite(w)) (rel + sums) * m + 0x1p-8 * (@abs(w) + (rel + sums) * m) + 0x1p-133 else 0;
        }
    }
};
