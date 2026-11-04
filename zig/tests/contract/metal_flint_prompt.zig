//! Flint's prompt attention on the tensor units (attention_nax.metal) as model.zig builds it, sliding and full, sinks.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const ar = contract.attention_range;
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const fl = @import("metal_flint.zig");

const c = contract.spec;
const g = contract.gates;
const dh = fl.head_dim;
const nq = fl.heads;
const kvh = fl.kv_heads;
const grp = nq / kvh;
const kw = kvh * dh;
const rt = 32; // a tile's query rows
const scale: f64 = 1.0 / @sqrt(@as(f64, dh));
const nan16 = 0x7fc0;

/// model.zig's header for attention_nax.metal.
fn head(a: std.mem.Allocator, sliding: bool) ![]u8 {
    return std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define WINDOW {d}\n#define RING {d}\n#define SCALE {e}f\n", .{ dh, nq, grp, kvh, @as(usize, if (sliding) fl.window else 0), @as(usize, if (sliding) fl.ring else 0), scale });
}

/// The source as the engine builds it on this GPU: the defines, then frags.source's text.
fn pipeline(a: std.mem.Allocator, device: mtl.Device, sliding: bool, e: ?fl.Edit) !mtl.Pipeline {
    const text = try fl.edited(a, sources.flint_attention_nax, e);
    defer a.free(text);
    const body = try tf.frags.source(device, a, text);
    defer a.free(body);
    const h = try head(a, sliding);
    defer a.free(h);
    const src = try std.mem.concat(a, u8, &.{ h, body });
    defer a.free(src);
    return rig_.pipeline(device, src, "fl_prompt_attention");
}

/// mma scores, fp32 SCALE * log2e, fast exp2, 64-key blocks; P x V takes fp32 P on bf16 V: P's error counted as bf16.
fn form(n: usize) ar.Form {
    const blocks: f64 = @floatFromInt(n / 64 + 2);
    return .{ .score = @as(f64, @floatFromInt(dh / 16)) * ar.mma_step + 2, .fast_exp = true, .chunk_exp = false, .p_rel = 0x1p-8, .num = ar.mma_step * 4 + blocks + 2, .den = blocks + 8, .score_round = 4 };
}

fn depth(_: c.Shape) usize {
    return ar.mma_step * (dh / 16);
}

const Case = struct { sliding: bool, fault: bool = false };

pub const Prompt = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Case,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    p0: usize = 0,
    slots: usize = 0,
    fill: g.Fill = .random,
    q: []u16 = &.{},
    k: []u16 = &.{},
    val: []u16 = &.{},
    sinks: []u16 = &.{},
    half: []f64 = &.{},
    refs: [5]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn base(sliding: bool) []const u8 {
        return if (sliding) "metal flint nax fl_prompt_attention sliding (513-key window, ring, sinks)" else "metal flint nax fl_prompt_attention full (sinks)";
    }
    const no_sink: fl.Edit = .{ .kernel = "fl_prompt_attention", .from = "float t = fast::exp2(sink - top[j]);", .to = "float t = 0.0f;" };

    /// Rows: a chunk's rows (past 16; the gates' small calls take fewer); n: the head dim; k: its first position.
    pub fn spec(name: []const u8, sliding: bool) c.Spec {
        // a short chunk past the window's first wrap, a chunk from the start, one deep in the stream
        const sl = [_]c.Shape{ .{ .rows = fl.window_rows + 1, .n = dh, .k = 3000 }, .{ .rows = 512, .n = dh, .k = 0 }, .{ .rows = 64, .n = dh, .k = 262000 } };
        const full = [_]c.Shape{ .{ .rows = fl.window_rows + 1, .n = dh, .k = 9000 }, .{ .rows = 512, .n = dh, .k = 0 }, .{ .rows = 64, .n = dh, .k = 8000 } };
        return .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = rt, .step = 1, .min = fl.window_rows + 1 }, .{}, .{} },
            .served = if (sliding) &sl else &full,
            .sums = .{ .bf16 = 0, .why = "the output is a bf16 mean of the values: a bf16 sum's error is a few output ulps" },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Case) !Prompt {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(a, device, v.sliding, if (v.fault) no_sink else null), .v = v };
    }
    fn free(t: *Prompt) void {
        inline for (.{ "q", "k", "val", "sinks" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
        t.a.free(t.half);
        t.half = &.{};
    }
    pub fn deinit(t: *Prompt) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Prompt) g.Kernel {
        t.name = fl.named(base(t.v.sliding), if (t.v.fault) "the sink dropped from the denominator" else null, &t.label);
        return .{ .spec = spec(t.name, t.v.sliding), .ctx = t, .vt = &vt, .control = if (t.v.fault) .shapes else null };
    }
    const vt = g.hooks(Prompt);

    fn slotOf(t: *const Prompt, i: usize) usize {
        return if (t.v.sliding) i % fl.ring else i;
    }
    /// Row r's first key: the window's (sliding), else 0.
    fn lo(t: *const Prompt, r: usize) usize {
        const p = t.p0 + r;
        return if (t.v.sliding and p + 1 > fl.window) p + 1 - fl.window else 0;
    }

    pub fn prepare(t: *Prompt, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > fl.prompt_rows or shape.n != dh) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.p0 = shape.k;
        t.fill = fill;
        // a full layer's whole 64-key blocks (state.zig's fullSlots); a sliding layer's ring
        t.slots = if (t.v.sliding) fl.ring else std.mem.alignForward(usize, t.p0 + t.rows, 64);
        const pad = std.mem.alignForward(usize, t.rows, rt); // a tile reads all its 32 rows
        t.q = try t.a.alloc(u16, pad * nq * dh);
        t.k = try t.a.alloc(u16, t.slots * kw);
        t.val = try t.a.alloc(u16, t.slots * kw);
        t.sinks = try t.a.alloc(u16, nq);
        t.half = try t.a.alloc(f64, t.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.q) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 3);
        for (t.k) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 3);
        for (t.val) |*x| x.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        for (t.sinks) |*x| x.* = c.toBf16(rng.float(f32) * 6 - 3);
        switch (fill) {
            .random => {},
            .nonfinite => {
                // in rows' own keys, which every row sees from its own position on
                for ([_]usize{ 3, 1 * dh, 2 * dh + 5 }, fl.planted, 0..) |at, x, r| {
                    t.q[r * nq * dh + (r * grp + 1) * dh + at % dh] = c.toBf16(x);
                }
                t.val[t.slotOf(t.p0) * kw + 0 * dh + 5] = c.toBf16(fl.inf32);
                t.val[t.slotOf(t.p0) * kw + 1 * dh + 6] = c.toBf16(fl.nan32);
                t.k[t.slotOf(t.p0 + 2) * kw + 2 * dh] = c.toBf16(fl.nan32);
            },
            .stress => {
                // keys alike (probabilities exactly 1), the sink -Inf, integer values: sums exact, past fp16's range
                const s: u32 = if (t.v.sliding) 4 else 1;
                for (0..t.slots) |i| @memcpy(t.k[i * kw ..][0..kw], t.k[0..kw]);
                for (0..t.slots) |i| for (0..kw) |e| {
                    t.val[i * kw + e] = c.toBf16(@floatFromInt(s * (1 + (i * 7 + e * 13) % 63)));
                };
                @memset(t.sinks, c.toBf16(-fl.inf32));
            },
        }
        // a full layer's cache past the chunk's last position holds anything: NaN, which no output may see
        if (!t.v.sliding) {
            @memset(t.k[(t.p0 + t.rows) * kw ..], nan16);
            @memset(t.val[(t.p0 + t.rows) * kw ..], nan16);
        }
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(t.k));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(t.val));
        t.refs[3] = try t.rig.input(std.mem.asBytes(&[2]i32{ @intCast(t.rows), @intCast(t.p0) }));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(t.sinks));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Prompt) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.refs[0..3], 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.setBuffer(t.refs[3].buf, t.refs[3].off, 4);
        p.e.setBuffer(t.refs[4].buf, t.refs[4].off, 5);
        // as forward.zig: (rows / 32, heads) groups, 32 threads a 64-column simdgroup
        p.e.dispatchGroups(mtl.Size.of((t.rows + rt - 1) / rt, nq, 1), mtl.Size.of(dh / 64 * 32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Prompt) usize {
        return t.rows * nq * dh;
    }

    /// Each row's heads over their keys and their sink (one more key through one more dim, its value 0).
    pub fn reference(t: *Prompt, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const E = dh + 1;
        const most = t.p0 + t.rows + 1;
        const buf = try t.a.alloc(f64, 2 * most * E + 2 * most + 3 * E);
        defer t.a.free(buf);
        const ks = buf[0 .. most * E];
        const vs = buf[most * E .. 2 * most * E];
        const w = buf[2 * most * E ..][0..most];
        const tau = buf[2 * most * E + most ..][0..most];
        const qv = buf[buf.len - 3 * E ..][0..E];
        const pt = buf[buf.len - 2 * E ..][0..E];
        const hf = buf[buf.len - E ..];
        for (0..t.rows) |r| for (0..kvh) |kv| {
            const keys = t.p0 + r + 1 - t.lo(r);
            for (0..keys) |j| {
                const s = t.slotOf(t.lo(r) + j);
                for (0..dh) |e| {
                    ks[j * E + e] = c.fromBf16(t.k[s * kw + kv * dh + e]);
                    vs[j * E + e] = c.fromBf16(t.val[s * kw + kv * dh + e]);
                }
                ks[j * E + dh] = 0;
                vs[j * E + dh] = 0;
            }
            for (0..grp) |gh| {
                const hq = kv * grp + gh;
                const sink: f64 = c.fromBf16(t.sinks[hq]);
                const n = if (std.math.isInf(sink) and sink < 0) keys else keys + 1;
                @memset(ks[keys * E ..][0..E], 0);
                @memset(vs[keys * E ..][0..E], 0);
                ks[keys * E + dh] = sink / scale;
                for (qv[0..dh], 0..) |*x, e| x.* = c.fromBf16(t.q[(r * nq + hq) * dh + e]);
                qv[dh] = 1;
                ar.head(qv, ks[0 .. n * E], vs[0 .. n * E], n, E, scale, form(n), w[0..n], tau[0..n], pt, hf);
                const at = (r * nq + hq) * dh;
                @memcpy(want[at..][0..dh], pt[0..dh]);
                @memcpy(t.half[at..][0..dh], hf[0..dh]);
            }
        };
    }

    /// The range's half width plus the output's bf16 rounding; gate 4's exact sums allow the rounding alone.
    pub fn bounds(t: *Prompt, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, t.half) |*l, w, h| {
            if (t.fill == .stress) {
                l.* = c.boundExact(spec("", t.v.sliding), w);
            } else if (!std.math.isFinite(w)) {
                l.* = 0;
            } else if (!std.math.isFinite(h)) {
                l.* = std.math.floatMax(f64); // scores too far apart for the range's first order: any finite value
            } else l.* = h + 0x1p-8 * (@abs(w) + h) + 0x1p-133;
        }
    }

    /// Gate 4: the outputs if P x V's sum ran in `acc`: every weight 1, so each row's mean over its keys.
    pub fn simulate(t: *Prompt, acc: c.Accum, order: g.Order, out: []f64) !void {
        const col = try t.a.alloc(f64, t.p0 + t.rows + 1);
        defer t.a.free(col);
        for (0..t.rows) |r| for (0..nq) |hq| for (0..dh) |e| {
            const keys = t.p0 + r + 1 - t.lo(r);
            for (col[0..keys], 0..) |*x, j| x.* = c.fromBf16(t.val[t.slotOf(t.lo(r) + j) * kw + (hq / grp) * dh + e]);
            out[(r * nq + hq) * dh + e] = fl.bf(contract.rows.sumIn(acc, order, col[0..keys]) / @as(f64, @floatFromInt(keys)));
        };
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    tests: fl.List(Prompt) = .{},

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Case{ .{ .sliding = true }, .{ .sliding = false }, .{ .sliding = true, .fault = true } }) |v| {
            if (fl.wanted(only, Prompt.base(v.sliding))) try s.tests.add(s.a, s.kernels, try Prompt.init(s.a, s.device, s.queue, v));
        }
    }
    pub fn deinit(s: *Suite) void {
        s.tests.deinit(s.a);
    }
};

/// Gate 5's texts for `flint nax ...`: the defines, then attention_nax.metal with nax.h (pre-M5: simd fragments).
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !?fl.Texts {
    if (!std.mem.startsWith(u8, name, "flint nax ")) return null;
    const h = try head(a, std.mem.indexOf(u8, name, " sliding ") != null);
    const body = try std.mem.replaceOwned(u8, a, sources.flint_attention_nax, "#include \"../nax.h\"", sources.nax);
    const pre = try std.mem.concat(a, u8, &.{ h, "#define TF_SIMD_FRAGS 1\n", body });
    return .{ .here = if (device.tensorUnits()) try std.mem.concat(a, u8, &.{ h, body }) else pre, .pre_m5 = pre };
}
