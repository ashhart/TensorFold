//! Inkling 2 Mini Metal attention gates 1-4 with an independent float64 formula and both prompt layouts.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_attention_controls.zig");
const sources = @import("inkling2_metal");
const plan = @import("tensorfold").inkling2_metal_host.attention.planning;
const c = contract.spec;
const g = contract.gates;
const Which = controls.Which;
const Mutation = controls.Mutation;
const D = 128;
const H = 24;
const KV = 8;
const R = 16;
const EXTENT = 1024;
const scale = 0.08838834764831845;

const prep_shapes = [_]c.Shape{ .{ .rows = 7, .n = H * D, .k = 1030 }, .{ .rows = 3, .n = H * D, .k = 9000 } };
const prep_global_shapes = [_]c.Shape{ prep_shapes[0], prep_shapes[1], .{ .rows = 4, .n = H * D, .k = 1048576 } };
const rel_shapes = [_]c.Shape{ .{ .rows = 4, .n = H * EXTENT, .k = R }, .{ .rows = 1, .n = H * EXTENT, .k = R } };
const sliding_shapes = [_]c.Shape{ .{ .rows = 7, .n = H * D, .k = 1030 }, .{ .rows = 3, .n = H * D, .k = 2049 } };
const global_shapes = [_]c.Shape{ .{ .rows = 4, .n = H * D, .k = 1100 }, .{ .rows = 2, .n = H * D, .k = 1030 } };
const prompt_shapes = [_]c.Shape{ .{ .rows = 8, .n = H * D, .k = 1100 }, .{ .rows = 64, .n = H * D, .k = 80 } };
fn depth(s: c.Shape) usize {
    return D + 4 * s.k + 64;
}
fn flat(_: c.Shape) usize {
    return R;
}
fn spec(w: Which) c.Spec {
    const x: c.Spec = .{ .name = "", .accum = .f32, .depth = depth, .ops = 64, .mids = 8, .out = .bf16, .served = &global_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .exact = true, .why = "P and outputs have required bf16 roundings" } };
    var s = x;
    switch (w) {
        .prep_sliding, .prep_global => {
            s.name = if (w == .prep_global) "metal inkling2 attention prep global tf_ink2_prep" else "metal inkling2 attention prep sliding tf_ink2_prep";
            s.served = if (w == .prep_global) &prep_global_shapes else &prep_shapes;
            s.ops = 32;
            s.mids = 2;
        },
        .rel_table => {
            s.name = "metal inkling2 attention relative table tf_ink2_rel_table";
            s.served = &rel_shapes;
            s.depth = flat;
            s.ops = 4;
            s.mids = 0;
        },
        .decode_sliding => {
            s.name = "metal inkling2 attention scalar sliding tf_ink2_attn_scalar";
            s.served = &sliding_shapes;
        },
        .decode_global => {
            s.name = "metal inkling2 attention scalar global tf_ink2_attn_scalar";
            s.served = &global_shapes;
        },
        .prompt_m5 => {
            s.name = "metal inkling2 prompt m5 tf_ink2_attn_prompt";
            s.served = &prompt_shapes;
        },
        .prompt_pre_m5 => {
            s.name = "metal inkling2 prompt pre-m5 tf_ink2_attn_prompt";
            s.served = &prompt_shapes;
        },
        .prompt_sliding_m5 => {
            s.name = "metal inkling2 prompt m5 tf_ink2_attn_prompt_sliding";
            s.served = &prompt_shapes;
        },
        .prompt_sliding_pre_m5 => {
            s.name = "metal inkling2 prompt pre-m5 tf_ink2_attn_prompt_sliding";
            s.served = &prompt_shapes;
        },
    }
    return s;
}
fn bf(x: u16) f64 {
    return c.fromBf16(x);
}
fn rn(x: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(x)));
}
fn bits(x: f64) u16 {
    return c.toBf16(@floatCast(x));
}
fn minNum(x: f64, y: f64) f64 {
    return if (std.math.isNan(x)) y else if (std.math.isNan(y)) x else @min(x, y);
}
fn maxNan(x: f64, y: f64) f64 {
    return if (std.math.isNan(x) or std.math.isNan(y)) std.math.nan(f64) else @max(x, y);
}
fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}
/// Prompt P rounds to BF16 at each tile maximum; later rescaling does not round twice. The result anchors weights.
fn promptWeights(score: []const f64, weights: []f64) f64 {
    const tile_keys = 64;
    const sub_keys = 1024;
    var top = -std.math.inf(f64);
    var upper = -std.math.inf(f64);
    var first: usize = 0;
    while (first < score.len) {
        const end = @min(first + tile_keys, score.len);
        const sub_first = first - first % sub_keys;
        var tile_peak = top;
        for (score[first..end]) |sc| tile_peak = maxNan(tile_peak, sc);
        const mu = if (tile_peak == -std.math.inf(f64)) 0 else tile_peak;
        const keep = @exp(top - mu);
        for (weights[sub_first..first]) |*w| w.* *= keep;
        for (weights[first..end], score[first..end]) |*w, sc| w.* = rn(@exp(sc - mu));
        top = tile_peak;
        if (first % sub_keys == sub_keys - tile_keys or end == score.len) {
            if (top != -std.math.inf(f64)) {
                const merged = maxNan(upper, top);
                const old_scale = @exp(upper - merged);
                const new_scale = @exp(top - merged);
                for (weights[0..sub_first]) |*w| w.* *= old_scale;
                for (weights[sub_first..end]) |*w| w.* *= new_scale;
                upper = merged;
            }
            top = -std.math.inf(f64);
        }
        first = end;
    }
    return upper;
}
fn tau(pos: usize) f64 {
    return 1 + 0.1 * @log(@max(@as(f64, @floatFromInt(pos + 1)) / 8192, 1));
}
fn periodic(p: usize, col: usize) f64 {
    const i = p % 1024;
    if (i < 512) return @floatFromInt(129 + (i * 73 + col * 31) % 127);
    const j = i - 512;
    return -@as(f64, @floatFromInt(129 + ((j + 1) * 73 + col * 31) % 127)) + @as(f64, if (j < 8) 1 else 0);
}
const Sim = struct { acc: c.Accum, order: g.Order };
fn sum(v: []const f64, sim: ?Sim) f64 {
    if (sim) |s| return contract.rows.sumIn(s.acc, s.order, v);
    var total: f64 = 0;
    for (v) |x| total += x;
    return total;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: Mutation,
    lib: mtl.Library,
    main: mtl.Pipeline,
    merge_lib: ?mtl.Library = null,
    merge: mtl.Pipeline = undefined,
    fused: bool = false,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    layer: plan.Layer = undefined,
    q: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    tab: []u16 = &.{},
    sink: []u16 = &.{},
    gate: []u16 = &.{},
    rel: []u16 = &.{},
    proj: []u16 = &.{},
    qs: []u16 = &.{},
    rs: []u16 = &.{},
    pos: []usize = &.{},
    refs: [10]rig_.Ref = undefined,
    out: [4]rig_.Ref = undefined,
    meta: [3]rig_.Ref = undefined,
    po: rig_.Ref = undefined,
    pml: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, m: Mutation) !Probe {
        const plain = if (controls.isPrompt(w)) (if (w == .prompt_pre_m5 or w == .prompt_sliding_pre_m5) sources.prompt_attention_pre_m5 else sources.prompt_attention) else sources.attention;
        const text = try controls.apply(a, plain, m);
        defer a.free(text);
        var lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        var main = try mtl.Pipeline.init(device, lib, controls.function(w), false);
        errdefer main.deinit();
        var merge_lib: ?mtl.Library = null;
        errdefer if (merge_lib) |loaded| loaded.deinit();
        var merge: mtl.Pipeline = undefined;
        const fused = controls.isAttention(w) and !controls.isPromptSliding(w);
        if (fused) {
            if (controls.isPrompt(w)) merge_lib = try mtl.Library.fromSource(device, sources.attention, mtl.CompileOptions.mlx());
            const merge_library = if (merge_lib) |loaded| loaded else lib;
            merge = try mtl.Pipeline.init(device, merge_library, "tf_ink2_merge", false);
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .mutation = m, .lib = lib, .main = main, .merge_lib = merge_lib, .merge = merge, .fused = fused };
    }
    fn hostFree(p: *Probe) void {
        inline for (.{ "q", "kc", "vc", "tab", "sink", "gate", "rel", "proj", "qs", "rs", "pos" }) |f| {
            p.a.free(@field(p, f));
            @field(p, f) = &.{};
        }
    }
    pub fn deinit(p: *Probe) void {
        p.hostFree();
        p.rig.deinit();
        if (p.fused) p.merge.deinit();
        if (p.merge_lib) |loaded| loaded.deinit();
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
    fn sizes(p: *const Probe) [4]usize {
        const n = p.rigCache();
        return .{ p.shape.rows * H * D, p.shape.rows * H * R, n, n };
    }
    fn rigCache(p: *const Probe) usize {
        return KV * p.layer.ring * D;
    }
    pub fn count(p: *const Probe) usize {
        if (p.which == .rel_table) return p.shape.rows * H * EXTENT;
        if (controls.isAttention(p.which)) return p.shape.rows * H * D;
        const z = p.sizes();
        return z[0] + z[1] + z[2] + z[3];
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k < s.rows) return false;
        if (p.which == .rel_table) {
            if (s.n != H * EXTENT or s.k != R) return false;
        } else if (s.n != H * D) return false;
        p.hostFree();
        p.rig.reset();
        p.shape = s;
        p.fill = fill;
        const global = p.which == .prep_global or p.which == .decode_global or p.which == .prompt_m5 or p.which == .prompt_pre_m5;
        const sliding = p.which == .prep_sliding or p.which == .decode_sliding or controls.isPromptSliding(p.which);
        p.layer = .{ .window = if (sliding) 1024 else 0, .ring = 2048, .chunk = if (sliding) 256 else plan.global_chunk, .tau = global };
        const rows = s.rows;
        const rh = rows * H;
        const cache = KV * p.layer.ring * D;
        p.pos = try p.a.alloc(usize, rows);
        for (p.pos, 0..) |*v, r| v.* = s.k - rows + r;
        if (p.which != .rel_table) {
            p.q = try p.a.alloc(u16, rh * D);
            p.tab = try p.a.alloc(u16, rh * EXTENT);
            @memset(p.q, 0);
            @memset(p.tab, 0);
        } else {
            p.rel = try p.a.alloc(u16, rh * R);
            p.proj = try p.a.alloc(u16, R * EXTENT);
            @memset(p.rel, 0);
            @memset(p.proj, 0);
        }
        if (!controls.isAttention(p.which) and p.which != .rel_table) p.rel = try p.a.alloc(u16, rh * R);
        if (p.which != .rel_table) {
            p.kc = try p.a.alloc(u16, if (controls.isAttention(p.which)) cache else rows * KV * D);
            p.vc = try p.a.alloc(u16, p.kc.len);
            @memset(p.kc, 0);
            @memset(p.vc, 0);
        }
        if (controls.isAttention(p.which)) {
            p.sink = try p.a.alloc(u16, rh);
            p.gate = try p.a.alloc(u16, rh * D);
            @memset(p.sink, 0);
            @memset(p.gate, 0);
        }
        if (p.which == .prep_sliding or p.which == .prep_global) {
            p.qs = try p.a.alloc(u16, H * D);
            p.rs = try p.a.alloc(u16, H * R);
            @memset(p.qs, 0);
            @memset(p.rs, 0);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.fillValues(rng, global);
        if (fill == .nonfinite) p.plant();
        try p.bind();
        return true;
    }
    fn fillValues(p: *Probe, rng: std.Random, global: bool) void {
        const stress = p.fill == .stress;
        for (p.q) |*v| v.* = bits(if (stress and controls.isAttention(p.which)) 0 else if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 511))) - 255) else rng.floatNorm(f64) * 0.3);
        for (p.kc) |*v| v.* = bits(if (stress and controls.isAttention(p.which)) 0 else if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64) * 0.3);
        for (p.vc, 0..) |*v, i| v.* = if (stress and controls.isAttention(p.which)) bits(periodic(i / D % p.layer.ring, i % D)) else bits(if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64));
        for (p.tab) |*v| v.* = bits(if (stress) 0 else rng.floatNorm(f64) * 0.25);
        for (p.sink) |*v| v.* = bits(if (stress) -std.math.inf(f64) else 2 + 4 * rng.float(f64));
        for (p.gate) |*v| v.* = bits(if (stress) 32 else rng.floatNorm(f64));
        for (p.rel, 0..) |*v, i| v.* = bits(if (stress) @floatFromInt(2 * (i % 31) + 1) else rng.floatNorm(f64));
        for (p.proj, 0..) |*v, i| v.* = bits(if (stress) @floatFromInt(1 + i % 2) else rng.floatNorm(f64) * 0.5);
        for (p.qs) |*v| v.* = bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
        for (p.rs) |*v| v.* = bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
        if (global and !controls.isAttention(p.which)) {
            _ = tau(p.pos[0]);
        }
    }
    fn plant(p: *Probe) void {
        const nan = bits(std.math.nan(f64));
        const inf = bits(std.math.inf(f64));
        if (p.which == .rel_table) {
            p.rel[3] = nan;
            if (p.rel.len > R + 5) p.rel[R + 5] = inf;
            if (p.rel.len > 2 * R + 1) {
                p.rel[2 * R] = inf;
                p.rel[2 * R + 1] = inf | 0x8000;
            }
        } else if (controls.isAttention(p.which)) {
            p.q[D + 5] = nan;
            p.sink[2] = inf;
            p.gate[3 * D] = inf;
            p.gate[3 * D + 1] = inf | 0x8000;
        } else {
            p.q[D + 3] = nan;
            p.kc[5] = inf;
            p.vc[D] = inf | 0x8000;
            p.rel[2] = nan;
        }
    }
    fn bind(p: *Probe) !void {
        const rows = p.shape.rows;
        if (p.which == .rel_table) {
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.rel));
            p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.proj));
            p.out[0] = try p.rig.output(p.count() * 2);
            return;
        }
        if (controls.isAttention(p.which)) {
            const arrays = [_][]u16{ p.q, p.kc, p.vc, p.tab, p.sink, p.gate };
            for (arrays, 0..) |x, i| p.refs[i] = try p.rig.input(std.mem.sliceAsBytes(x));
        } else {
            const arrays = [_][]u16{ p.q, p.kc, p.vc, p.rel, p.qs, p.rs };
            for (arrays, 0..) |x, i| p.refs[i] = try p.rig.input(std.mem.sliceAsBytes(x));
        }
        const pos32 = try p.a.alloc(i32, rows);
        defer p.a.free(pos32);
        const seg32 = try p.a.alloc(i32, rows);
        defer p.a.free(seg32);
        @memset(seg32, 0);
        for (pos32, p.pos) |*v, x| v.* = @intCast(x);
        const seg = [1]plan.Seg{.{ .row0 = 0, .n = @intCast(rows), .p0 = @intCast(p.shape.k - rows), .off = 0 }};
        p.meta[0] = try p.rig.input(std.mem.sliceAsBytes(pos32));
        p.meta[1] = try p.rig.input(std.mem.sliceAsBytes(seg32));
        p.meta[2] = try p.rig.input(std.mem.sliceAsBytes(&seg));
        if (controls.isAttention(p.which)) {
            if (p.fused) {
                const stride = plan.slotsFor(p.layer, p.shape.k);
                const parts = rows * H * stride;
                p.po = try p.rig.scratch(parts * D * 4);
                p.pml = try p.rig.scratch(parts * 8);
            }
            p.out[0] = try p.rig.output(p.count() * 2);
        } else for (p.sizes(), 0..) |n, i| p.out[i] = try p.rig.output(n * 2);
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.main);
        const rows: u32 = @intCast(p.shape.rows);
        const ring = p.layer.ring;
        if (p.which == .rel_table) {
            e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
            e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
            e.setBuffer(p.out[0].buf, p.out[0].off, 2);
            e.dispatchGroups(mtl.Size.of(rows * H, 1, 1), mtl.Size.of(256, 1, 1));
        } else if (!controls.isAttention(p.which)) {
            @memset(p.rig.bytes(p.out[2], p.sizes()[2] * 2), 0);
            @memset(p.rig.bytes(p.out[3], p.sizes()[3] * 2), 0);
            for (p.refs[0..6], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            for (p.meta, 6..) |r, i| e.setBuffer(r.buf, r.off, i);
            for (p.out, 9..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setValue(plan.PrepArgs{ .heads = H, .kv_heads = KV, .ring = ring, .global = @intFromBool(p.layer.tau), .qk_clip = 22.627416997969522, .rel_clip = 8, .floor_pos = 8192, .alpha = 0.1, .eps = 1e-6 }, 13);
            e.dispatchGroups(mtl.Size.of(rows, H + 2 * KV, 1), mtl.Size.of(32, 1, 1));
        } else if (controls.isPromptSliding(p.which)) {
            for (p.refs[0..4], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setBuffer(p.meta[2].buf, p.meta[2].off, 4);
            e.setBuffer(p.refs[4].buf, p.refs[4].off, 5);
            e.setBuffer(p.refs[5].buf, p.refs[5].off, 6);
            e.setBuffer(p.out[0].buf, p.out[0].off, 7);
            const sc: f32 = 1.0 / @sqrt(@as(f32, D));
            e.setValue(plan.PromptArgs{ .heads = H, .kv_heads = KV, .ring = ring, .window = p.layer.window, .chunk = p.layer.chunk, .c0 = 0, .chunks = 1, .scale = sc }, 8);
            const blocks = (rows * (H / KV) + plan.prompt_pairs - 1) / plan.prompt_pairs;
            e.dispatchGroups(mtl.Size.of(blocks, KV, 1), mtl.Size.of(256, 1, 1));
        } else {
            const stride = plan.slotsFor(p.layer, p.shape.k);
            for (p.refs[0..4], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setBuffer(p.meta[2].buf, p.meta[2].off, 4);
            e.setBuffer(p.po.buf, p.po.off, 5);
            e.setBuffer(p.pml.buf, p.pml.off, 6);
            const sc: f32 = 1.0 / @sqrt(@as(f32, D));
            if (controls.isPrompt(p.which)) {
                e.setValue(plan.PromptArgs{ .heads = H, .kv_heads = KV, .ring = ring, .chunk = p.layer.chunk, .c0 = 0, .chunks = stride, .scale = sc }, 7);
                const blocks = (rows * (H / KV) + plan.prompt_pairs - 1) / plan.prompt_pairs;
                e.dispatchGroups(mtl.Size.of(blocks, KV, stride), mtl.Size.of(256, 1, 1));
            } else {
                e.setBuffer(p.meta[0].buf, p.meta[0].off, 7);
                e.setBuffer(p.meta[1].buf, p.meta[1].off, 8);
                e.setValue(plan.AttnArgs{ .heads = H, .kv_heads = KV, .ring = ring, .window = p.layer.window, .chunk = p.layer.chunk, .c0 = 0, .chunks = stride, .scale = sc }, 9);
                e.dispatchGroups(mtl.Size.of(rows * H, stride, 1), mtl.Size.of(32, 1, 1));
            }
            e.barrier();
            e.setPipeline(p.merge);
            e.setBuffer(p.po.buf, p.po.off, 0);
            e.setBuffer(p.pml.buf, p.pml.off, 1);
            e.setBuffer(p.refs[4].buf, p.refs[4].off, 2);
            e.setBuffer(p.refs[5].buf, p.refs[5].off, 3);
            e.setBuffer(p.meta[0].buf, p.meta[0].off, 4);
            e.setBuffer(p.out[0].buf, p.out[0].off, 5);
            e.setValue(plan.MergeArgs{ .heads = H, .window = p.layer.window, .chunk = p.layer.chunk, .c0 = 0, .chunks = stride }, 6);
            e.dispatchGroups(mtl.Size.of(rows * H, 1, 1), mtl.Size.of(D, 1, 1));
        }
        try p.rig.finish(pass);
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (p.which == .rel_table) return p.tableRef(null, want, mag);
        if (controls.isAttention(p.which)) return p.attentionRef(null, want, mag);
        return p.prepRef(null, want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        const sim: Sim = .{ .acc = acc, .order = order };
        if (p.which == .rel_table) return p.tableRef(sim, out, mag);
        if (controls.isAttention(p.which)) return p.attentionRef(sim, out, mag);
        return p.prepRef(sim, out, mag);
    }
    fn tableRef(p: *Probe, sim: ?Sim, want: []f64, mag: []f64) !void {
        var terms: [R]f64 = undefined;
        for (want, mag, 0..) |*w, *m, o| {
            const rh = o / EXTENT;
            const d = o % EXTENT;
            for (&terms, 0..) |*v, j| v.* = bf(p.rel[rh * R + j]) * bf(p.proj[j * EXTENT + d]);
            m.* = 0;
            for (terms) |x| m.* += @abs(x);
            const x = sum(&terms, sim);
            w.* = if (sim != null) rn(x) else x;
        }
    }
    fn headClip(x: []const f64, threshold: f64, sim: ?Sim, t: ?f64, want: []f64, mag: []f64) void {
        var sq: [D]f64 = undefined;
        mZero(mag);
        for (sq[0..x.len], x, 0..) |*v, y, i| {
            v.* = y * y;
            mag[i] = @abs(y);
        }
        const f = minNum(threshold / (@sqrt(sum(sq[0..x.len], sim)) + 1e-6), 1);
        for (want, x, mag) |*w, v, *m| {
            w.* = rn(v * f);
            if (t) |tv| w.* = rn(w.* * tv);
            m.* = @abs(v) + v * v;
        }
    }
    /// prep uses independent K/V and attention uses populated ring caches; q and rel never alias either one.
    fn prepRef(p: *Probe, sim: ?Sim, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
        const z = p.sizes();
        var x: [D]f64 = undefined;
        for (0..p.shape.rows) |r| {
            const tv: ?f64 = if (p.layer.tau) tau(p.pos[r]) else null;
            for (0..H) |h| {
                for (&x, 0..) |*v, d| v.* = bf(p.q[(r * H + h) * D + d]) * bf(p.qs[h * D + d]);
                headClip(&x, 22.627416997969522, sim, tv, want[(r * H + h) * D ..][0..D], mag[(r * H + h) * D ..][0..D]);
                for (x[0..R], 0..) |*v, d| v.* = bf(p.rel[(r * H + h) * R + d]) * bf(p.rs[h * R + d]);
                const at = z[0] + (r * H + h) * R;
                headClip(x[0..R], 8, sim, tv, want[at..][0..R], mag[at..][0..R]);
            }
            for (0..KV) |h| {
                const slot = (h * p.layer.ring + p.pos[r] % p.layer.ring) * D;
                for (&x, 0..) |*v, d| v.* = bf(p.kc[(r * KV + h) * D + d]);
                headClip(&x, 22.627416997969522, sim, null, want[z[0] + z[1] + slot ..][0..D], mag[z[0] + z[1] + slot ..][0..D]);
                var sq: [D]f64 = undefined;
                for (&x, &sq, 0..) |*v, *ss, d| {
                    v.* = bf(p.vc[(r * KV + h) * D + d]);
                    ss.* = v.* * v.*;
                }
                const inv = 1 / @sqrt(sum(&sq, sim) / D + 1e-6);
                for (0..D) |d| {
                    const at = z[0] + z[1] + z[2] + slot + d;
                    want[at] = rn(x[d] * inv);
                    mag[at] = sq[d] + @abs(x[d]);
                }
            }
        }
    }
    fn mZero(mag: []f64) void {
        @memset(mag, 0);
    }
    fn attentionRef(p: *Probe, sim: ?Sim, want: []f64, mag: []f64) !void {
        const most = p.shape.k;
        const score = try p.a.alloc(f64, most);
        defer p.a.free(score);
        const weights = try p.a.alloc(f64, most);
        defer p.a.free(weights);
        const terms = try p.a.alloc(f64, @max(most, D));
        defer p.a.free(terms);
        for (0..p.shape.rows) |r| {
            for (0..H) |h| {
                const rh = r * H + h;
                const pos = p.pos[r];
                const lo = if (p.layer.window > 0) (pos + 1) -| p.layer.window else 0;
                const n = pos + 1 - lo;
                const kvh = h / (H / KV);
                const z = bf(p.sink[rh]);
                // Prompt P is BF16 at the keys' maximum; the sink joins in the final merge using the shared maximum.
                const prompt = sim == null and controls.isPrompt(p.which);
                var peak = if (prompt) -std.math.inf(f64) else z;
                for (lo..pos + 1) |kp| {
                    const j = kp - lo;
                    const at = (kvh * p.layer.ring + kp % p.layer.ring) * D;
                    for (0..D) |d| terms[d] = bf(p.q[rh * D + d]) * bf(p.kc[at + d]);
                    const distance = pos - kp;
                    const bias = if (distance < EXTENT) bf(p.tab[rh * EXTENT + distance]) else 0;
                    score[j] = sum(terms[0..D], sim) * scale + bias;
                    peak = maxNan(peak, score[j]);
                }
                if (prompt) {
                    peak = promptWeights(score[0..n], weights[0..n]);
                } else for (weights[0..n], score[0..n]) |*w, sc| w.* = @exp(sc - peak);
                const merged_peak = if (prompt) maxNan(peak, z) else peak;
                const rescale = if (prompt) @exp(peak - merged_peak) else 1;
                const den = sum(weights[0..n], sim) * rescale + @exp(z - merged_peak);
                for (0..D) |d| {
                    for (lo..pos + 1) |kp| {
                        const j = kp - lo;
                        const at = (kvh * p.layer.ring + kp % p.layer.ring) * D;
                        terms[j] = weights[j] * bf(p.vc[at + d]);
                    }
                    const num = sum(terms[0..n], sim) * rescale;
                    const y = num / den * sigmoid(bf(p.gate[rh * D + d]));
                    want[rh * D + d] = if (sim != null) rn(y) else y;
                    mag[rh * D + d] = (@abs(num) + den + 1) * (1 + @abs(y));
                    for (terms[0..n]) |v| mag[rh * D + d] += @abs(v * rescale);
                }
            }
        }
    }
};

fn supported(device: mtl.Device, w: Which) bool {
    return switch (w) {
        .prompt_m5, .prompt_sliding_m5 => device.tensorUnits(),
        .prompt_pre_m5, .prompt_sliding_pre_m5 => !device.tensorUnits(),
        else => true,
    };
}
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Which)) |w| {
            if (!supported(s.device, w) or (only != null and std.mem.indexOf(u8, spec(w).name, only.?) == null)) continue;
            try s.add(w, .none);
        }
        for (std.enums.values(Mutation)) |m| {
            if (m == .none) continue;
            const w = m.which();
            if (!supported(s.device, w) or (only != null and std.mem.indexOf(u8, m.name(), only.?) == null)) continue;
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
        try s.probes.append(s.a, p);
        try s.kernels.append(s.a, p.kernel());
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
