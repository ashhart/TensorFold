//! Inkling 2 Mini Metal multi-stream prep/decode gates with independent FP64 formulas.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_attention_slots_controls.zig");
const ref = @import("metal_inkling2_attention_ref.zig");
const sources = @import("inkling2_metal");
const plan = @import("tensorfold").inkling2_metal_host.attention.planning;
const c = contract.spec;
const g = contract.gates;
const Which = controls.Which;
const Mutation = controls.Mutation;
const D = ref.D;
const H = ref.H;
const KV = ref.KV;
const R = ref.R;
const EXTENT = ref.EXTENT;

comptime {
    // q/rel/proj + partials + row metadata + 16 slot KV buffers + args = 24 <= 26 <= 31.
    const slots_buffers = 7 + 2 * plan.max_slots + 1;
    if (slots_buffers > 26 or slots_buffers > 31) @compileError("Inkling 2 slots buffer count exceeds limit");
}
const prep_shapes = [_]c.Shape{ .{ .rows = 7, .n = H * D, .k = 1030 }, .{ .rows = 3, .n = H * D, .k = 9000 } };
const sliding_shapes = [_]c.Shape{
    .{ .rows = 6, .n = H * D, .k = 1030 },  .{ .rows = 1, .n = H * D, .k = 1030 },
    .{ .rows = 2, .n = H * D, .k = 1056 },  .{ .rows = 7, .n = H * D, .k = 1060 },
    .{ .rows = 8, .n = H * D, .k = 2049 },  .{ .rows = 9, .n = H * D, .k = 1062 },
    .{ .rows = 16, .n = H * D, .k = 2049 }, .{ .rows = 18, .n = H * D, .k = 2049 },
};
const global_shapes = [_]c.Shape{
    .{ .rows = 6, .n = H * D, .k = 2050 },  .{ .rows = 1, .n = H * D, .k = 1024 },
    .{ .rows = 2, .n = H * D, .k = 1100 },  .{ .rows = 7, .n = H * D, .k = 2050 },
    .{ .rows = 8, .n = H * D, .k = 2050 },  .{ .rows = 9, .n = H * D, .k = 2055 },
    .{ .rows = 16, .n = H * D, .k = 2050 }, .{ .rows = 18, .n = H * D, .k = 2050 },
};
fn depth(s: c.Shape) usize {
    return D + 4 * s.k + 64;
}
fn spec(w: Which) c.Spec {
    var s: c.Spec = .{ .name = "", .accum = .f32, .depth = depth, .ops = 64, .mids = 8, .out = .bf16, .served = &global_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .exact = true, .why = "P and outputs have required bf16 roundings" } };
    switch (w) {
        .prep_slots => {
            s.name = "metal inkling2 attention prep slots tf_ink2_prep_slots";
            s.served = &prep_shapes;
            s.ops = 32;
            s.mids = 2;
        },
        .attn_slots_sliding => {
            s.name = "metal inkling2 attention slots sliding tf_ink2_attn_slots";
            s.served = &sliding_shapes;
        },
        .attn_slots_global => {
            s.name = "metal inkling2 attention slots global tf_ink2_attn_slots";
            s.served = &global_shapes;
        },
    }
    return s;
}

fn sum(v: []const f64, sim: ?ref.Sim) f64 {
    if (sim) |s| return contract.rows.sumIn(s.acc, s.order, v);
    var total: f64 = 0;
    for (v) |x| total += x;
    return total;
}
fn minNum(x: f64, y: f64) f64 {
    return if (std.math.isNan(x)) y else if (std.math.isNan(y)) x else @min(x, y);
}
fn headClip(x: []const f64, threshold: f64, sim: ?ref.Sim, tv: ?f64, want: []f64, mag: []f64) void {
    var sq: [D]f64 = undefined;
    @memset(mag, 0);
    for (sq[0..x.len], x, 0..) |*v, y, i| {
        v.* = y * y;
        mag[i] = @abs(y);
    }
    const f = minNum(threshold / (@sqrt(sum(sq[0..x.len], sim)) + 1e-6), 1);
    for (want, x, mag) |*w, v, *m| {
        w.* = ref.rn(v * f);
        if (tv) |t| w.* = ref.rn(w.* * t);
        m.* = @abs(v) + v * v;
    }
}
fn prepRef(p: *const Probe, sim: ?ref.Sim, want: []f64, mag: []f64) void {
    @memset(want, 0);
    @memset(mag, 0);
    const qn = p.shape.rows * H * D;
    const rn = p.shape.rows * H * R;
    var x: [D]f64 = undefined;
    for (0..p.shape.rows) |r| {
        const slot = r % p.n_slots;
        const tv: ?f64 = if (p.layer.tau) ref.tau(p.pos[r]) else null;
        for (0..H) |h| {
            for (&x, 0..) |*v, d| v.* = ref.bf(p.q[(r * H + h) * D + d]) * ref.bf(p.qs[h * D + d]);
            headClip(&x, 22.627416997969522, sim, tv, want[(r * H + h) * D ..][0..D], mag[(r * H + h) * D ..][0..D]);
            for (x[0..R], 0..) |*v, d| v.* = ref.bf(p.rel[(r * H + h) * R + d]) * ref.bf(p.rs[h * R + d]);
            const at = qn + (r * H + h) * R;
            headClip(x[0..R], 8, sim, tv, want[at..][0..R], mag[at..][0..R]);
        }
        const ring = p.slot_rings[slot];
        for (0..KV) |h| {
            const cache_at = qn + rn + 2 * p.slot_off[slot] + (h * ring + p.pos[r] % ring) * D;
            for (&x, 0..) |*v, d| v.* = ref.bf(p.kc[(r * KV + h) * D + d]);
            headClip(&x, 22.627416997969522, sim, null, want[cache_at..][0..D], mag[cache_at..][0..D]);
            var sq: [D]f64 = undefined;
            for (&x, &sq, 0..) |*v, *ss, d| {
                v.* = ref.bf(p.vc[(r * KV + h) * D + d]);
                ss.* = v.* * v.*;
            }
            const inv = 1 / @sqrt(sum(&sq, sim) / D + 1e-6);
            for (0..D) |d| {
                const at = cache_at + KV * ring * D + d;
                want[at] = ref.rn(x[d] * inv);
                mag[at] = sq[d] + @abs(x[d]);
            }
        }
    }
}
fn attentionRef(p: *const Probe, a: std.mem.Allocator, sim: ?ref.Sim, want: []f64, mag: []f64) !void {
    const score = try a.alloc(f64, p.shape.k);
    defer a.free(score);
    const weights = try a.alloc(f64, p.shape.k);
    defer a.free(weights);
    const terms = try a.alloc(f64, @max(p.shape.k, D));
    defer a.free(terms);
    for (0..p.shape.rows) |r| {
        const slot = r % p.n_slots;
        const ring = p.slot_rings[slot];
        const c_off = p.slot_off[slot];
        const pos = p.pos[r];
        const lo = if (p.layer.window > 0) (pos + 1) -| p.layer.window else 0;
        const n = pos + 1 - lo;
        for (0..H) |h| {
            const rh = r * H + h;
            const kvh = h / (H / KV);
            const z = ref.bf(p.sink[rh]);
            var peak = z;
            for (lo..pos + 1) |kp| {
                const j = kp - lo;
                const at = c_off + (kvh * ring + kp % ring) * D;
                for (0..D) |d| terms[d] = ref.bf(p.q[rh * D + d]) * ref.bf(p.kc[at + d]);
                const distance = pos - kp;
                var bias: f64 = 0;
                if (distance < EXTENT) {
                    var rterms: [R]f64 = undefined;
                    for (&rterms, 0..) |*v, rj| v.* = ref.bf(p.rel[rh * R + rj]) * ref.bf(p.proj[rj * EXTENT + distance]);
                    bias = ref.rn(sum(&rterms, sim));
                }
                score[j] = sum(terms[0..D], sim) * ref.scale + bias;
                peak = if (std.math.isNan(peak) or std.math.isNan(score[j])) std.math.nan(f64) else @max(peak, score[j]);
            }
            for (weights[0..n], score[0..n]) |*w, sc| w.* = @exp(sc - peak);
            const den = sum(weights[0..n], sim) + @exp(z - peak);
            for (0..D) |d| {
                for (lo..pos + 1) |kp| {
                    const j = kp - lo;
                    const at = c_off + (kvh * ring + kp % ring) * D;
                    terms[j] = weights[j] * ref.bf(p.vc[at + d]);
                }
                const num = sum(terms[0..n], sim);
                const y = num / den / (1 + @exp(-ref.bf(p.gate[rh * D + d])));
                want[rh * D + d] = if (sim != null) ref.rn(y) else y;
                mag[rh * D + d] = (@abs(num) + den + 1) * (1 + @abs(y));
                for (terms[0..n]) |v| mag[rh * D + d] += @abs(v);
            }
        }
    }
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: Mutation,
    lib: mtl.Library,
    main: mtl.Pipeline,
    merge: ?mtl.Pipeline = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    layer: plan.Layer = undefined,
    q: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    sink: []u16 = &.{},
    gate: []u16 = &.{},
    rel: []u16 = &.{},
    proj: []u16 = &.{},
    qs: []u16 = &.{},
    rs: []u16 = &.{},
    pos: []usize = &.{},
    n_slots: usize = 0,
    slot_rings: [plan.max_slots]usize = @splat(0),
    slot_chunks: [plan.max_slots]u32 = @splat(0),
    slot_off: [plan.max_slots + 1]usize = @splat(0),
    refs: [6]rig_.Ref = undefined,
    slot_kv: [2 * plan.max_slots]rig_.Ref = undefined,
    out: [2 + 2 * plan.max_slots]rig_.Ref = undefined,
    meta: [2]rig_.Ref = undefined,
    po: rig_.Ref = undefined,
    pml: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, m: Mutation) !Probe {
        const text = try controls.apply(a, sources.attention, m);
        defer a.free(text);
        var lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        var main = try mtl.Pipeline.init(device, lib, controls.function(w), false);
        errdefer main.deinit();
        const merge = if (controls.isAttention(w)) try mtl.Pipeline.init(device, lib, "tf_ink2_merge", false) else null;
        errdefer if (merge) |pipe| pipe.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .mutation = m, .lib = lib, .main = main, .merge = merge };
    }
    fn hostFree(p: *Probe) void {
        inline for (.{ "q", "kc", "vc", "sink", "gate", "rel", "proj", "qs", "rs", "pos" }) |f| {
            p.a.free(@field(p, f));
            @field(p, f) = &.{};
        }
    }
    pub fn deinit(p: *Probe) void {
        p.hostFree();
        p.rig.deinit();
        if (p.merge) |pipe| pipe.deinit();
        p.main.deinit();
        p.lib.deinit();
    }
    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var k: g.Kernel = .{ .spec = spec(p.which), .ctx = p, .vt = &vt };
        if (p.mutation != .none) {
            k.spec.name = p.mutation.name();
            k.control = p.mutation.gate();
        }
        return k;
    }
    fn sizes(p: *const Probe) [2]usize {
        return .{ p.shape.rows * H * D, p.shape.rows * H * R };
    }
    pub fn count(p: *const Probe) usize {
        if (controls.isAttention(p.which)) return p.shape.rows * H * D;
        const z = p.sizes();
        return z[0] + z[1] + 2 * p.slot_off[p.n_slots];
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k < s.rows or s.n != H * D) return false;
        p.hostFree();
        p.rig.reset();
        p.shape = s;
        p.fill = fill;
        const sliding = p.which == .attn_slots_sliding;
        const global = p.which == .attn_slots_global;
        const ring = std.mem.alignForward(usize, @max(2048, s.k), 64);
        p.layer = .{ .window = if (sliding) 1024 else 0, .ring = @intCast(ring), .chunk = if (sliding) 256 else plan.global_chunk, .tau = global };
        p.n_slots = @min(s.rows, plan.max_slots);
        p.slot_off[0] = 0;
        for (0..p.n_slots) |slot| {
            p.slot_rings[slot] = ring;
            p.slot_chunks[slot] = p.layer.chunk;
            p.slot_off[slot + 1] = p.slot_off[slot] + KV * ring * D;
        }
        const rows = s.rows;
        const rh = rows * H;
        p.pos = try p.a.alloc(usize, rows);
        for (p.pos, 0..) |*v, r| v.* = s.k - rows + r;
        p.q = try p.a.alloc(u16, rh * D);
        p.rel = try p.a.alloc(u16, rh * R);
        p.kc = try p.a.alloc(u16, if (controls.isAttention(p.which)) p.slot_off[p.n_slots] else rows * KV * D);
        p.vc = try p.a.alloc(u16, p.kc.len);
        @memset(p.q, 0);
        @memset(p.rel, 0);
        @memset(p.kc, 0);
        @memset(p.vc, 0);
        if (controls.isAttention(p.which)) {
            p.proj = try p.a.alloc(u16, R * EXTENT);
            p.sink = try p.a.alloc(u16, rh);
            p.gate = try p.a.alloc(u16, rh * D);
            @memset(p.proj, 0);
            @memset(p.sink, 0);
            @memset(p.gate, 0);
        } else {
            p.qs = try p.a.alloc(u16, H * D);
            p.rs = try p.a.alloc(u16, H * R);
            @memset(p.qs, 0);
            @memset(p.rs, 0);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        p.fillValues(prng.random());
        if (fill == .nonfinite) p.plant();
        try p.bind();
        return true;
    }
    fn fillValues(p: *Probe, rng: std.Random) void {
        const stress = p.fill == .stress;
        const attn = controls.isAttention(p.which);
        for (p.q) |*v| v.* = ref.bits(if (stress and attn) 0 else if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 511))) - 255) else rng.floatNorm(f64) * 0.3);
        for (p.kc) |*v| v.* = ref.bits(if (stress and attn) 0 else if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64) * 0.3);
        if (stress and attn) {
            for (0..p.n_slots) |slot| {
                const ring = p.slot_rings[slot];
                for (p.vc[p.slot_off[slot]..p.slot_off[slot + 1]], 0..) |*v, i| v.* = ref.bits(ref.periodic(i / D % ring, i % D));
            }
        } else for (p.vc) |*v| v.* = ref.bits(if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64));
        for (p.sink) |*v| v.* = ref.bits(if (stress) -std.math.inf(f64) else 2 + 4 * rng.float(f64));
        for (p.gate) |*v| v.* = ref.bits(if (stress) 32 else rng.floatNorm(f64));
        for (p.rel, 0..) |*v, i| v.* = ref.bits(if (stress and attn) 0 else if (stress) @floatFromInt(2 * (i % 31) + 1) else rng.floatNorm(f64));
        for (p.proj) |*v| v.* = ref.bits(if (stress) 0 else rng.floatNorm(f64) * 0.5);
        for (p.qs) |*v| v.* = ref.bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
        for (p.rs) |*v| v.* = ref.bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
    }
    fn plant(p: *Probe) void {
        const nan = ref.bits(std.math.nan(f64));
        const inf = ref.bits(std.math.inf(f64));
        if (controls.isAttention(p.which)) {
            p.q[D + 5] = nan;
            p.sink[2] = inf;
            p.gate[3 * D] = inf;
            p.gate[3 * D + 1] = inf | 0x8000;
            p.rel[4 * R + 2] = nan;
            p.rel[5 * R + 1] = inf;
        } else {
            p.q[D + 3] = nan;
            p.kc[5] = inf;
            p.vc[D] = inf | 0x8000;
            p.rel[2] = nan;
        }
    }
    fn maxStride(p: *const Probe) u32 {
        var stride: u32 = 1;
        for (0..p.n_slots) |slot| {
            const layer: plan.Layer = .{ .window = p.layer.window, .ring = @intCast(p.slot_rings[slot]), .chunk = p.slot_chunks[slot], .tau = p.layer.tau };
            stride = @max(stride, plan.slotsFor(layer, p.shape.k));
        }
        return stride;
    }
    fn sameChunks(p: *const Probe) bool {
        for (1..p.n_slots) |slot| if (p.slot_chunks[slot] != p.slot_chunks[0]) return false;
        return true;
    }
    fn slotRowsMax(p: *const Probe) u32 {
        var counts: [plan.max_slots]u32 = @splat(0);
        for (0..p.shape.rows) |r| counts[r % p.n_slots] += 1;
        var most: u32 = 0;
        for (0..p.n_slots) |slot| most = @max(most, counts[slot]);
        return most;
    }
    fn bind(p: *Probe) !void {
        const rows = p.shape.rows;
        if (controls.isAttention(p.which)) {
            const arrays = [_][]u16{ p.q, p.rel, p.proj, p.sink, p.gate };
            for (arrays, 0..) |x, i| p.refs[i] = try p.rig.input(std.mem.sliceAsBytes(x));
            for (0..p.n_slots) |slot| {
                const bytes = std.mem.sliceAsBytes(p.kc[p.slot_off[slot]..p.slot_off[slot + 1]]);
                const vbytes = std.mem.sliceAsBytes(p.vc[p.slot_off[slot]..p.slot_off[slot + 1]]);
                p.slot_kv[slot] = try p.rig.input(bytes);
                p.slot_kv[plan.max_slots + slot] = try p.rig.input(vbytes);
            }
        } else {
            const arrays = [_][]u16{ p.q, p.kc, p.vc, p.rel, p.qs, p.rs };
            for (arrays, 0..) |x, i| p.refs[i] = try p.rig.input(std.mem.sliceAsBytes(x));
        }
        const pos32 = try p.a.alloc(i32, rows);
        defer p.a.free(pos32);
        const seg32 = try p.a.alloc(i32, rows);
        defer p.a.free(seg32);
        for (pos32, seg32, p.pos, 0..) |*v, *sg, x, r| {
            v.* = @intCast(x);
            sg.* = @intCast(r % p.n_slots);
        }
        p.meta[0] = try p.rig.input(std.mem.sliceAsBytes(pos32));
        p.meta[1] = try p.rig.input(std.mem.sliceAsBytes(seg32));
        if (controls.isAttention(p.which)) {
            const stride = p.maxStride();
            const parts = rows * H * stride;
            p.po = try p.rig.scratch(parts * D * 4);
            p.pml = try p.rig.scratch(parts * 8);
            p.out[0] = try p.rig.output(p.count() * 2);
        } else {
            const z = p.sizes();
            p.out[0] = try p.rig.output(z[0] * 2);
            p.out[1] = try p.rig.output(z[1] * 2);
            for (0..p.n_slots) |slot| {
                const nb = KV * p.slot_rings[slot] * D * 2;
                p.out[2 + 2 * slot] = try p.rig.output(nb);
                p.out[2 + 2 * slot + 1] = try p.rig.output(nb);
            }
        }
    }
    fn bindRefs(e: mtl.ComputeEncoder, refs: []const rig_.Ref, first: usize) void {
        for (refs, first..) |r, i| e.setBuffer(r.buf, r.off, i);
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.main);
        const rows: u32 = @intCast(p.shape.rows);
        if (!controls.isAttention(p.which)) {
            for (0..p.n_slots) |slot| {
                const nb = KV * p.slot_rings[slot] * D * 2;
                @memset(p.rig.bytes(p.out[2 + 2 * slot], nb), 0);
                @memset(p.rig.bytes(p.out[2 + 2 * slot + 1], nb), 0);
            }
            bindRefs(e, &p.refs, 0);
            bindRefs(e, &p.meta, 6);
            bindRefs(e, p.out[0..2], 8);
            var rings: [plan.max_slots]u32 = undefined;
            for (0..plan.max_slots) |slot| {
                const si = if (slot < p.n_slots) slot else 0;
                e.setBuffer(p.out[2 + 2 * si].buf, p.out[2 + 2 * si].off, 10 + slot);
                e.setBuffer(p.out[2 + 2 * si + 1].buf, p.out[2 + 2 * si + 1].off, 18 + slot);
                rings[slot] = @intCast(p.slot_rings[si]);
            }
            e.setValue(plan.PrepSlotsArgs{ .heads = H, .kv_heads = KV, .global = @intFromBool(p.layer.tau), .n_slots = @intCast(p.n_slots), .rings = rings, .qk_clip = 22.627416997969522, .rel_clip = 8, .floor_pos = 8192, .alpha = 0.1, .eps = 1e-6 }, 26);
            e.dispatchGroups(mtl.Size.of(rows, H + 2 * KV, 1), mtl.Size.of(32, 1, 1));
        } else {
            const stride = p.maxStride();
            const sc: f32 = 1.0 / @sqrt(@as(f32, D));
            bindRefs(e, &.{ p.refs[0], p.refs[1], p.refs[2], p.po, p.pml, p.meta[0], p.meta[1] }, 0);
            var sm: [plan.max_slots]plan.SlotMeta = undefined;
            for (0..plan.max_slots) |slot| {
                const si = if (slot < p.n_slots) slot else 0;
                e.setBuffer(p.slot_kv[si].buf, p.slot_kv[si].off, 7 + slot);
                e.setBuffer(p.slot_kv[plan.max_slots + si].buf, p.slot_kv[plan.max_slots + si].off, 15 + slot);
                sm[slot] = .{ .ring = @intCast(p.slot_rings[si]), .chunk = p.slot_chunks[si] };
            }
            e.setValue(plan.AttnSlotsArgs{ .rows = rows, .heads = H, .kv_heads = KV, .window = p.layer.window, .c0 = 0, .chunks = stride, .n_slots = @intCast(p.n_slots), .scale = sc, .slots = sm }, 23);
            const row_blocks = (p.slotRowsMax() + plan.decode_row_block - 1) / plan.decode_row_block;
            e.dispatchGroups(mtl.Size.of(@as(u32, @intCast(p.n_slots)) * KV, stride, row_blocks), mtl.Size.of(plan.decode_threads, 1, 1));
            e.barrier();
            const same = p.sameChunks();
            const loops: usize = if (same) 1 else p.shape.rows;
            const grid_rows: u32 = if (same) rows else 1;
            for (0..loops) |r| {
                e.setPipeline(p.merge.?);
                e.setBuffer(p.po.buf, p.po.off + r * H * stride * D * 4, 0);
                e.setBuffer(p.pml.buf, p.pml.off + r * H * stride * 8, 1);
                e.setBuffer(p.refs[3].buf, p.refs[3].off + r * H * 2, 2);
                e.setBuffer(p.refs[4].buf, p.refs[4].off + r * H * D * 2, 3);
                e.setBuffer(p.meta[0].buf, p.meta[0].off + r * 4, 4);
                e.setBuffer(p.out[0].buf, p.out[0].off + r * H * D * 2, 5);
                e.setValue(plan.MergeArgs{ .heads = H, .window = p.layer.window, .chunk = p.slot_chunks[if (same) 0 else r % p.n_slots], .c0 = 0, .chunks = stride }, 6);
                e.dispatchGroups(mtl.Size.of(grid_rows * H, 1, 1), mtl.Size.of(D, 1, 1));
            }
        }
        try p.rig.finish(pass);
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (controls.isAttention(p.which)) return attentionRef(p, p.a, null, want, mag);
        prepRef(p, null, want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        const sim: ref.Sim = .{ .acc = acc, .order = order };
        if (controls.isAttention(p.which)) return attentionRef(p, p.a, sim, out, mag);
        prepRef(p, sim, out, mag);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Which)) |w| {
            if (only != null and std.mem.indexOf(u8, spec(w).name, only.?) == null) continue;
            try s.add(w, .none);
        }
        for (std.enums.values(Mutation)) |m| {
            if (m == .none) continue;
            const w = m.which();
            if (only != null and std.mem.indexOf(u8, m.name(), only.?) == null) continue;
            try s.add(w, m);
        }
    }
    fn add(s: *Suite, w: Which, m: Mutation) !void {
        const p = try s.a.create(Probe);
        errdefer s.a.destroy(p);
        p.* = Probe.init(s.a, s.device, s.queue, w, m) catch |err| {
            var k: g.Kernel = .{ .spec = spec(w), .ctx = undefined, .vt = undefined, .control = m.gate(), .init_error = err };
            if (m != .none) k.spec.name = m.name();
            try s.kernels.append(s.a, k);
            s.a.destroy(p);
            return;
        };
        try @import("contract").register(s.a, &s.probes, &s.kernels, p, p.kernel());
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
