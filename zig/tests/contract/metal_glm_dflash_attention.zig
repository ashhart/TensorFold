//! GLM's DFlash2 drafter attention (glm_draft_attention.metal) under the contract: block rows over a wrapped ring.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const ar = contract.attention_range;
const dattn = tf.glm.metal_dflash_attention;
const fill_mask = tf.glm.metal_dflash_context.fillMask;

/// incoai's drafter: 32 query heads over 8 KV heads of 128, a 2,048-slot ring, an 8-row trained block.
const shape = dattn.Shape{ .heads = 32, .kv_heads = 8, .dim = 128 };
const window = 2048;
const trained = 8;
const D = 128;
const KW = 8 * D;
const QW = 32 * D;

/// One stream, three of different lengths a launch; faults: slot one past, masked value in, stream 0's ring.
const Case = enum { plain, streams, slot, masked, mixed };
pub const registered = [_]Case{ .plain, .streams, .slot, .masked, .mixed };

fn streamsOf(case: Case) usize {
    return if (case == .streams or case == .mixed) 3 else 1;
}

pub fn name(case: Case) []const u8 {
    return switch (case) {
        .plain => "metal glm dflash2 block attention (ring in place, a key a lane) +merge",
        .slot => "control: metal glm dflash2 block attention, ring slots read one past",
        .masked => "control: metal glm dflash2 block attention, a masked key's value enters the sums",
        .streams => "metal glm dflash2 block attention, three streams of different lengths in one launch",
        .mixed => "control: metal glm dflash2 block attention, every stream reads stream 0's ring",
    };
}

fn edit(case: Case) ?[2][]const u8 {
    return switch (case) {
        .plain, .streams => null,
        .slot => .{ "(st.at + key) % a.window", "(st.at + key + 1) % a.window" },
        .mixed => .{ "const device bfloat* K = GLM_RING(K, s);", "const device bfloat* K = K0;" },
        .masked => .{ "const glm_vec x = seen ? v : glm_vec(0.0f);", "const glm_vec x = v;" },
    };
}

/// A shape is (block rows, head dim, stream 0's context keys); stream s holds its context from slot at(s) on.
fn keysDepth(s: c.Shape) usize {
    return s.n + s.k + s.rows;
}

/// glm_draft_attend's arithmetic: a lane's 128-fma dot, 32-key blocks and their rescales, then the chunk merge.
fn form(keys: usize) ar.Form {
    const chunks: f64 = @floatFromInt(dattn.chunks(@intCast(keys)) + 1);
    const blocks: f64 = dattn.chunk / 32;
    return .{ .score = D + 1, .fast_exp = false, .chunk_exp = true, .p_rel = 0, .num = dattn.chunk + blocks + chunks + 2, .den = 5 + blocks + chunks + 2 };
}

/// Gate 4's value for key `j` of `n`: big halves that cancel, four ones left, so only exact fp32 sums give 4.
fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    if (j >= 2 * half) return 0;
    const at = if (j < half) j else j - half;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    const one: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
    return if (j < half) big + one else -big;
}

const nan16: u16 = 0x7fc0;

/// A stream's inputs: its keys and values in key order (context, then its own rows) and its block's mask.
const One = struct {
    context: usize = 0,
    kp: []u16 = &.{},
    vp: []u16 = &.{},
    mask: []u8 = &.{},
    fn keys(o: One, block: usize) usize {
        return o.context + block;
    }
};

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    case: Case,
    pipes: dattn.Pipes,
    block: usize = 0,
    streams: usize = 1,
    one: [3]One = @splat(.{}),
    stress: bool = false,
    q: []u16 = &.{},
    half: []f64 = &.{},
    point: []f64 = &.{},
    table: [3]dattn.Stream = undefined,
    refs: [4]rig_.Ref = undefined,
    rings: [3][2]rig_.Ref = undefined,
    part: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, case: Case) !Probe {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .case = case, .streams = streamsOf(case), .pipes = try dattn.Pipes.init(a, device, shape, edit(case)) };
    }

    pub fn deinit(p: *Probe) void {
        p.free();
        p.pipes.deinit();
        p.rig.deinit();
    }

    fn free(p: *Probe) void {
        for (&p.one) |*o| {
            p.a.free(o.kp);
            p.a.free(o.vp);
            p.a.free(o.mask);
            o.* = .{};
        }
        p.a.free(p.q);
        p.a.free(p.half);
        p.a.free(p.point);
        p.q = &.{};
        p.half = &.{};
        p.point = &.{};
    }

    fn rows(p: *const Probe) usize {
        return p.streams * p.block;
    }
    /// Stream s's first ring slot: a wrapped ring for any context past a few keys, a different one each stream.
    fn at(p: *const Probe, s: usize) usize {
        return (window - p.one[s].context / 2 + 7 * s) % window;
    }

    pub fn spec(p: *const Probe) c.Spec {
        // 16 and 8 rows over a full window (the drafter's steady state), a short context and a lone row
        const served = &[_]c.Shape{ .{ .rows = 16, .n = D, .k = 2047 }, .{ .rows = 8, .n = D, .k = 2047 }, .{ .rows = 16, .n = D, .k = 300 }, .{ .rows = 1, .n = D, .k = 64 } };
        return .{ .name = name(p.case), .accum = .f32, .depth = keysDepth, .ops = 4, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{}, .{ .size = dattn.chunk, .min = 0 } }, .served = served };
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > dattn.max_block or s.n != D or s.k >= window) return false;
        p.rig.reset();
        p.free();
        p.block = s.rows;
        p.stress = fill == .stress;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.q = try p.a.alloc(u16, p.rows() * QW);
        for (p.q) |*y| y.* = c.toBf16(rng.float(f32) * 2 - 1);
        const own_k = try p.a.alloc(u16, p.rows() * KW);
        defer p.a.free(own_k);
        const own_v = try p.a.alloc(u16, p.rows() * KW);
        defer p.a.free(own_v);
        var most: usize = 0;
        for (p.one[0..p.streams], 0..) |*o, k| {
            // stream 0 at the shape's context, the others at three fifths and a third of it
            o.context = if (k == 0) s.k else if (k == 1) s.k * 3 / 5 else s.k / 3;
            const n = o.keys(p.block);
            most = @max(most, n);
            o.kp = try p.a.alloc(u16, n * KW);
            o.vp = try p.a.alloc(u16, n * KW);
            o.mask = try p.a.alloc(u8, p.block * n);
            try fill_mask(o.mask, @intCast(o.context), @intCast(p.block), window, trained);
            for ([_][]u16{ o.kp, o.vp }) |x| for (x) |*y| {
                y.* = c.toBf16(rng.float(f32) * 2 - 1);
            };
            if (p.stress) {
                @memset(o.kp, 0);
                for (0..n) |j| @memset(o.vp[j * KW ..][0..KW], c.toBf16(stressValue(j, n)));
            }
        }
        if (p.stress) @memset(p.q, 0);
        if (fill == .nonfinite) {
            const o = &p.one[0];
            const n = o.keys(p.block);
            p.q[(0 * 32 + 1) * D + 3] = c.toBf16(std.math.nan(f32)); // row 0, head 1
            const last = n - 1; // the last own row: visible to rows past the trained block only
            o.vp[last * KW + 5] = c.toBf16(std.math.inf(f32));
            o.vp[last * KW + 6] = c.toBf16(-std.math.inf(f32));
            if (o.context > 0) o.kp[(o.context - 1) * KW + 2 * D] = c.toBf16(std.math.nan(f32)); // kv head 2
            // context key 0: row 0 sees it, a full window's later rows have slid past it (masked)
            if (o.context > 0) o.vp[1 * D + 7] = c.toBf16(std.math.inf(f32));
        }
        // each ring: its context from slot at(s) on, wrapping; every other slot NaN, so a stray read shows
        const kr = try p.a.alloc(u16, (window + 16) * KW);
        defer p.a.free(kr);
        const vr = try p.a.alloc(u16, (window + 16) * KW);
        defer p.a.free(vr);
        const mask = try p.a.alloc(u8, p.rows() * most);
        defer p.a.free(mask);
        @memset(mask, 0);
        for (p.one[0..p.streams], 0..) |o, k| {
            @memset(kr, nan16);
            @memset(vr, nan16);
            for (0..o.context) |j| {
                const slot = (p.at(k) + j) % window;
                @memcpy(kr[slot * KW ..][0..KW], o.kp[j * KW ..][0..KW]);
                @memcpy(vr[slot * KW ..][0..KW], o.vp[j * KW ..][0..KW]);
            }
            p.rings[k] = .{ try p.rig.input(std.mem.sliceAsBytes(kr)), try p.rig.input(std.mem.sliceAsBytes(vr)) };
            @memcpy(own_k[k * p.block * KW ..][0 .. p.block * KW], o.kp[o.context * KW ..]);
            @memcpy(own_v[k * p.block * KW ..][0 .. p.block * KW], o.vp[o.context * KW ..]);
            const n = o.keys(p.block);
            for (0..p.block) |r| @memcpy(mask[(k * p.block + r) * most ..][0..n], o.mask[r * n ..][0..n]);
            p.table[k] = .{ .keys = @intCast(n), .context = @intCast(o.context), .at = @intCast(p.at(k)) };
        }
        p.half = try p.a.alloc(f64, p.count());
        p.point = try p.a.alloc(f64, p.count());
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(own_k));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(own_v));
        p.refs[3] = try p.rig.input(mask);
        const args = try dattn.plan(p.table[0..p.streams], @intCast(p.block), window, 1);
        p.part = try p.rig.scratch(dattn.partBytes(shape, args));
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.rows() * QW;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const scale: f32 = 1.0 / @sqrt(@as(f32, D));
        const args = try dattn.plan(p.table[0..p.streams], @intCast(p.block), window, scale);
        const r = struct {
            fn of(x: rig_.Ref) dattn.Ref {
                return .{ .buf = x.buf, .off = x.off };
            }
        }.of;
        var rings: [3]dattn.Ring = undefined;
        for (rings[0..p.streams], p.rings[0..p.streams]) |*x, y| x.* = .{ .k = r(y[0]), .v = r(y[1]) };
        try p.pipes.encode(pass.e, args, p.table[0..p.streams], rings[0..p.streams], r(p.refs[0]), r(p.refs[1]), r(p.refs[2]), r(p.refs[3]), r(p.part), r(p.out));
        try p.rig.finish(pass);
    }

    /// Every head's float64 output over the keys its row sees in its own stream, and how far fp32 may land.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const scale: f64 = @as(f32, 1.0 / @sqrt(@as(f32, D)));
        for (p.one[0..p.streams], 0..) |o, k| {
            const n = o.keys(p.block);
            const buf = try p.a.alloc(f64, 2 * n * D + 2 * n + D);
            defer p.a.free(buf);
            const kf = buf[0 .. n * D];
            const vf = buf[n * D .. 2 * n * D];
            const w = buf[2 * n * D ..][0..n];
            const tau = buf[2 * n * D + n ..][0..n];
            const qv = buf[buf.len - D ..];
            for (0..p.block) |lr| for (0..8) |hk| {
                var seen: usize = 0;
                for (0..n) |j| if (o.mask[lr * n + j] != 0) {
                    for (0..D) |e| {
                        kf[seen * D + e] = c.fromBf16(o.kp[j * KW + hk * D + e]);
                        vf[seen * D + e] = c.fromBf16(o.vp[j * KW + hk * D + e]);
                    }
                    seen += 1;
                };
                for (hk * 4..hk * 4 + 4) |h| {
                    const at_ = ((k * p.block + lr) * 32 + h) * D;
                    for (qv, 0..) |*x, e| x.* = c.fromBf16(p.q[at_ + e]);
                    ar.head(qv, kf, vf, seen, D, scale, form(n), w, tau, want[at_..][0..D], p.half[at_..][0..D]);
                }
            };
        }
        @memcpy(mag, p.half);
        @memcpy(p.point, want);
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        const s = p.spec();
        for (allow, want, p.half) |*l, w, h| l.* = if (p.stress) c.boundExact(s, w) else ar.allowBf16(w, h, 0, 0);
    }

    pub fn intervals(p: *Probe, low: []f64, high: []f64) !void {
        for (low, high, p.point, p.half) |*l, *hi, pt, h| {
            const e = if (p.stress) [2]f64{ -std.math.floatMax(f64), std.math.floatMax(f64) } else ar.intervalBf16(pt, h, 0, 0);
            l.* = e[0];
            hi.* = e[1];
        }
    }

    /// Gate 4: every weight 1, so each output is the values its row sees summed in `acc` over their count.
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!p.stress) return error.StressRequiresEqualScores;
        var terms: std.ArrayList(f64) = .empty;
        defer terms.deinit(p.a);
        for (p.one[0..p.streams], 0..) |o, k| {
            const n = o.keys(p.block);
            for (0..p.block) |lr| for (0..KW) |e| {
                terms.clearRetainingCapacity();
                for (0..n) |j| if (o.mask[lr * n + j] != 0) try terms.append(p.a, c.fromBf16(o.vp[j * KW + e]));
                const v = c.fromBf16(c.toBf16(@floatCast(contract.rows.sumIn(acc, order, terms.items) / @as(f64, @floatFromInt(terms.items.len)))));
                for (0..4) |gq| out[(k * p.block + lr) * QW + (e / D * 4 + gq) * D + e % D] = v;
            };
        }
    }

    pub fn replaced(_: *Probe) ![]const u8 {
        return error.NotAReplacement;
    }
};

const vt = g.hooks(Probe);

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (registered) |case| {
            if (only) |o| if (std.mem.indexOf(u8, name(case), o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, case);
            errdefer p.deinit();
            try @import("contract").register(s.a, &s.probes, &s.kernels, p, .{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = switch (case) {
                .plain, .streams => null,
                .masked => .nonfinite,
                .slot, .mixed => .shapes,
            } });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
