//! ud_attend under the contract: the window's keys and the read blocks' (biased), for prompt rows and a step.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_uocr_decoder_common.zig");

const c = contract.spec;
const g = contract.gates;
const H = k.hidden;
const HD = k.head_dim;
const Mutation = k.Mutation;
const bf = k.bf;
const none: u32 = 0xffffffff;
const max_keys = k.local + k.read_blocks * k.block;

/// A score's 128-term dot (4 a lane, then the simd_sum tree), the scale and the bias.
fn scoreDepth(_: c.Shape) usize {
    return 4 + 5 + 2;
}

pub const Attend = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    /// A prompt's rows (biases rounded to bf16 first, as the prompt's bf16 mask holds them), else a step's.
    prompt: bool,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    kv: [2][]u16 = .{ &.{}, &.{} },
    sel: []u32 = &.{},
    bias: []f32 = &.{},
    ins: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: query rows, the last of `k` positions; n: the width (10 heads of 128).
    pub fn spec(prompt: bool) c.Spec {
        return .{
            .name = if (prompt) "metal uocr decoder ud_attend prompt rows, bf16 biases" else "metal uocr decoder ud_attend step, fp32 biases",
            .accum = .f32,
            .depth = scoreDepth,
            .out = .bf16,
            .served = if (prompt)
                &.{ .{ .rows = k.prompt_rows, .n = H, .k = k.prompt_rows }, .{ .rows = 37, .n = H, .k = 1061 } }
            else
                &.{ .{ .rows = 1, .n = H, .k = 4096 }, .{ .rows = 1, .n = H, .k = 100 } },
            // the scores' dots sum exactly in fp32 at gate 4; the softmax's mass and the weighted values do not
            .sums = .{ .exact = false },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation, prompt: bool) !Attend {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "ud_attend"), .m = m, .prompt = prompt };
    }
    pub fn deinit(t: *Attend) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Attend) void {
        inline for (.{ "q", "sel", "bias" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
        for (&t.kv) |*x| {
            t.a.free(x.*);
            x.* = &.{};
        }
    }
    pub fn kernel(t: *Attend) g.Kernel {
        return k.mark(.{ .spec = spec(t.prompt), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Attend);

    fn pos0(t: *const Attend) usize {
        return t.shape.k - t.shape.rows;
    }

    pub fn prepare(t: *Attend, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H or shape.k < shape.rows) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const rows = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.q = try t.a.alloc(u16, rows * H);
        for (&t.kv) |*x| x.* = try t.a.alloc(u16, shape.k * H);
        if (fill == .stress) {
            // per head: q's halves mirror with the sign turned, keys' mirror (8 nudged): huge partials, small scores
            for (0..rows * k.heads) |hh| for (0..HD / 2) |d| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                t.q[hh * HD + d] = c.toBf16(v);
                t.q[hh * HD + HD / 2 + d] = c.toBf16(-v);
            };
            for (0..shape.k * k.heads) |hh| for (0..HD / 2) |d| {
                const v: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
                t.kv[0][hh * HD + d] = c.toBf16(v);
                t.kv[0][hh * HD + HD / 2 + d] = c.toBf16(v + @as(f32, if (d < 8) 1 else 0));
            };
            k.fillBf16(rng, t.kv[1], 2);
        } else {
            k.fillBf16(rng, t.q, 0.6);
            k.fillBf16(rng, t.kv[0], 0.6);
            k.fillBf16(rng, t.kv[1], 2);
        }
        // each row's read blocks: distinct remote blocks in a random order, gaps marked unread
        t.sel = try t.a.alloc(u32, rows * k.read_blocks);
        t.bias = try t.a.alloc(f32, rows * k.read_blocks);
        for (0..rows) |r| {
            const pos = t.pos0() + r;
            const nrem = if (pos / k.block >= 4) pos / k.block - 3 else 0;
            const s = t.sel[r * k.read_blocks ..][0..k.read_blocks];
            for (s, 0..) |*v, i| {
                v.* = none;
                if (nrem == 0 or rng.uintLessThan(u32, 8) == 0) continue;
                const pick: u32 = @intCast(rng.uintLessThan(usize, nrem));
                if (std.mem.indexOfScalar(u32, s[0..i], pick) == null) v.* = pick;
            }
            for (t.bias[r * k.read_blocks ..][0..k.read_blocks]) |*b| b.* = if (fill == .stress) 0 else rng.float(f32) * 5 - 4;
        }
        if (fill == .nonfinite) {
            // a key every row's window holds (head 0), a value (head 1, dim 9), a query (row 2, head 2)
            t.kv[0][(shape.k - 3) * H + 5] = c.toBf16(k.nan32);
            t.kv[1][(shape.k - 5) * H + HD + 9] = c.toBf16(k.inf32);
            if (rows > 2) t.q[2 * H + 2 * HD] = c.toBf16(-k.inf32);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.kv[0]));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.kv[1]));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.sel));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.bias));
        t.out = try t.rig.output(rows * H * 2);
        return true;
    }

    pub fn run(t: *Attend) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 5);
        p.e.setValue(extern struct { pos0: u32, bf16_bias: u32 }{ .pos0 = @intCast(t.pos0()), .bf16_bias = @intFromBool(t.prompt) }, 6);
        p.e.dispatchGroups(mtl.Size.of(k.heads, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Attend) usize {
        return t.shape.rows * H;
    }

    /// Row r's keys: the window's, then each read block's before it, with their biases.
    fn keys(t: *Attend, r: usize, at: *[max_keys]usize, kb: *[max_keys]f64) usize {
        const pos = t.pos0() + r;
        const lo = if (pos >= k.local - 1) pos - (k.local - 1) else 0;
        var n: usize = 0;
        for (lo..pos + 1) |p| {
            at[n] = p;
            kb[n] = 0;
            n += 1;
        }
        for (t.sel[r * k.read_blocks ..][0..k.read_blocks], t.bias[r * k.read_blocks ..][0..k.read_blocks]) |s, b| {
            if (s == none) continue;
            var p: usize = s * k.block;
            while (p < @min(lo, (s + 1) * k.block)) : (p += 1) {
                at[n] = p;
                kb[n] = if (t.prompt) bf(b) else b;
                n += 1;
            }
        }
        return n;
    }

    fn el(x: []const u16, i: usize) f64 {
        return c.fromBf16(x[i]);
    }

    /// Outputs and their sum of |p v| (sums in `acc`, `order`); `err` gets each row and head's worst score error.
    fn outputs(t: *Attend, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64, err: ?[]f64) !void {
        var at: [max_keys]usize = undefined;
        var kb: [max_keys]f64 = undefined;
        var sc: [max_keys]f64 = undefined;
        var terms: [max_keys]f64 = undefined;
        var dt: [HD]f64 = undefined;
        const u = c.unit(.f32);
        for (0..t.shape.rows) |r| {
            const n = t.keys(r, &at, &kb);
            for (0..k.heads) |h| {
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
                    if (acc != .none) dot = contract.rows.sumIn(acc, order, &dt);
                    s.* = dot * k.inv_sqrt_hd + b;
                    if (!std.math.isNan(s.*)) top = @max(top, s.*);
                    worst = @max(worst, (if (t.fill == .stress) 0 else c.gamma(HD, u) * m) * k.inv_sqrt_hd + 3 * u * @abs(s.*));
                }
                if (err) |e| e[r * k.heads + h] = worst;
                var mass: f64 = 0;
                for (sc[0..n], terms[0..n]) |*s, *q| {
                    s.* = @exp(s.* - top);
                    q.* = s.*;
                    mass += s.*;
                }
                if (acc != .none) mass = contract.rows.sumIn(acc, order, terms[0..n]);
                for (0..HD) |d| {
                    var o: f64 = 0;
                    var m: f64 = 0;
                    for (terms[0..n], sc[0..n], at[0..n]) |*q, s, p| {
                        q.* = s / mass * el(t.kv[1], p * H + h * HD + d);
                        o += q.*;
                        m += @abs(q.*);
                    }
                    if (acc != .none) o = bf(contract.rows.sumIn(acc, order, terms[0..n]));
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

    /// Weights within twice the worst score error, exp, mass and division; the PV chain; the bf16 point.
    pub fn bounds(t: *Attend, want: []const f64, mag: []const f64, allow: []f64) !void {
        const u = c.unit(.f32);
        const scratch = try t.a.alloc(f64, 2 * want.len);
        defer t.a.free(scratch);
        const err = try t.a.alloc(f64, t.shape.rows * k.heads);
        defer t.a.free(err);
        try t.outputs(.none, .sequential, scratch[0..want.len], scratch[want.len..], err);
        const sums = c.gamma(max_keys / 8 + 8 + 2, u);
        for (allow, want, mag, 0..) |*l, w, m, i| {
            const rel = 2 * (2 * err[i / HD] + 4 * u) + c.gamma(max_keys, u) + 2 * u;
            l.* = if (std.math.isFinite(w)) (rel + sums) * m + 0x1p-8 * (@abs(w) + (rel + sums) * m) + 0x1p-133 else 0;
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Attend) = .empty,

    fn add(s: *Suite, only: ?[]const u8, m: Mutation, prompt: bool) !void {
        const x = try s.a.create(Attend);
        x.* = try Attend.init(s.a, s.device, s.queue, m, prompt);
        const kern = x.kernel();
        if (!k.wanted(only, kern.spec.name)) {
            x.deinit();
            s.a.destroy(x);
            return;
        }
        try s.probes.append(s.a, x);
        try s.kernels.append(s.a, kern);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        try s.add(only, .none, true);
        try s.add(only, .none, false);
        try s.add(only, .attend_no_bias, true);
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.probes.deinit(s.a);
    }
};

/// Gate 5's texts for this group's launch.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?k.Texts {
    return k.limitsText(name, &.{"ud_attend"});
}
