//! Inkling 2 Mini Metal attention gates 1-4 with an independent float64 formula, baselines and both prompt layouts.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_attention_controls.zig");
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

const prep_shapes = [_]c.Shape{ .{ .rows = 7, .n = H * D, .k = 1030 }, .{ .rows = 3, .n = H * D, .k = 9000 } };
const prep_global_shapes = [_]c.Shape{ prep_shapes[0], prep_shapes[1], .{ .rows = 4, .n = H * D, .k = 1048576 } };
const sliding_shapes = [_]c.Shape{ .{ .rows = 7, .n = H * D, .k = 1030 }, .{ .rows = 3, .n = H * D, .k = 2049 }, .{ .rows = 5, .n = H * D, .k = 17 }, .{ .rows = 1, .n = H * D, .k = 1024 }, .{ .rows = 1, .n = H * D, .k = 1030 }, .{ .rows = 2, .n = H * D, .k = 1056 }, .{ .rows = 8, .n = H * D, .k = 1060 }, .{ .rows = 9, .n = H * D, .k = 1062 }, .{ .rows = 16, .n = H * D, .k = 2049 } };
const global_shapes = [_]c.Shape{ .{ .rows = 4, .n = H * D, .k = 1100 }, .{ .rows = 2, .n = H * D, .k = 1030 }, .{ .rows = 3, .n = H * D, .k = 17 }, .{ .rows = 1, .n = H * D, .k = 1025 }, .{ .rows = 1, .n = H * D, .k = 1024 }, .{ .rows = 7, .n = H * D, .k = 1100 }, .{ .rows = 8, .n = H * D, .k = 2049 }, .{ .rows = 9, .n = H * D, .k = 2055 }, .{ .rows = 16, .n = H * D, .k = 2060 } };
const prompt_shapes = [_]c.Shape{ .{ .rows = 8, .n = H * D, .k = 1100 }, .{ .rows = 64, .n = H * D, .k = 80 }, .{ .rows = 3, .n = H * D, .k = 17 }, .{ .rows = 1, .n = H * D, .k = 1025 } };
const long_shapes = [_]c.Shape{.{ .rows = 5, .n = H * D, .k = 17450 }};
fn depth(s: c.Shape) usize {
    return D + 4 * s.k + 64;
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
        .decode_sliding => {
            s.name = "metal inkling2 attention scalar sliding tf_ink2_attn_scalar";
            s.served = &sliding_shapes;
            s.replaces = "main tf_ink2_rel_table then tf_ink2_attn_scalar";
        },
        .decode_global => {
            s.name = "metal inkling2 attention scalar global tf_ink2_attn_scalar";
            s.served = &global_shapes;
            s.replaces = "main tf_ink2_rel_table then tf_ink2_attn_scalar";
        },
        .decode_global_long => {
            s.name = "metal inkling2 attention scalar global long tf_ink2_attn_scalar";
            s.accum = .none;
            s.sums = .{ .f16 = 0, .bf16 = 0, .why = "byte-exact against main on multi-chunk long context" };
            s.served = &long_shapes;
            s.replaces = "main tf_ink2_rel_table then tf_ink2_attn_scalar";
        },
        .prompt_m5 => {
            s.name = "metal inkling2 prompt m5 tf_ink2_attn_prompt";
            s.served = &prompt_shapes;
            s.replaces = "main tf_ink2_rel_table then tf_ink2_attn_prompt";
        },
        .prompt_m5_long => {
            s.name = "metal inkling2 prompt m5 long tf_ink2_attn_prompt";
            s.accum = .none;
            s.sums = .{ .f16 = 0, .bf16 = 0, .why = "byte-exact against main on multi-chunk long context" };
            s.served = &long_shapes;
            s.replaces = "main tf_ink2_rel_table then tf_ink2_attn_prompt";
        },
        .prompt_pre_m5 => {
            s.name = "metal inkling2 prompt pre-m5 tf_ink2_attn_prompt";
            s.served = &prompt_shapes;
            s.replaces = "main tf_ink2_rel_table then tf_ink2_attn_prompt";
        },
        .prompt_pre_m5_long => {
            s.name = "metal inkling2 prompt pre-m5 long tf_ink2_attn_prompt";
            s.accum = .none;
            s.sums = .{ .f16 = 0, .bf16 = 0, .why = "byte-exact against main on multi-chunk long context" };
            s.served = &long_shapes;
            s.replaces = "main tf_ink2_rel_table then tf_ink2_attn_prompt";
        },
    }
    return s;
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
    base: ?ref.Baseline = null,
    fused: bool = false,
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
    refs: [10]rig_.Ref = undefined,
    out: [4]rig_.Ref = undefined,
    meta: [3]rig_.Ref = undefined,
    po: rig_.Ref = undefined,
    pml: rig_.Ref = undefined,
    base_tab: rig_.Ref = undefined,
    base_po: rig_.Ref = undefined,
    base_pml: rig_.Ref = undefined,
    base_out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, m: Mutation) !Probe {
        const plain = if (controls.isPrompt(w)) (if (controls.isPreM5(w)) sources.prompt_attention_pre_m5 else sources.prompt_attention) else sources.attention;
        const text = try controls.apply(a, plain, m);
        defer a.free(text);
        var lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        var main = try mtl.Pipeline.init(device, lib, controls.function(w), false);
        errdefer main.deinit();
        var merge_lib: ?mtl.Library = null;
        errdefer if (merge_lib) |loaded| loaded.deinit();
        var merge: mtl.Pipeline = undefined;
        const fused = controls.isAttention(w);
        if (fused) {
            if (controls.isPrompt(w)) merge_lib = try mtl.Library.fromSource(device, sources.attention, mtl.CompileOptions.mlx());
            merge = try mtl.Pipeline.init(device, merge_lib orelse lib, "tf_ink2_merge", false);
        }
        errdefer if (fused) merge.deinit();
        const base = if (fused and m == .none) try ref.Baseline.init(a, device, w) else null;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .mutation = m, .lib = lib, .main = main, .merge_lib = merge_lib, .merge = merge, .base = base, .fused = fused };
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
        if (p.base) |*b| b.deinit();
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
            k.spec.replaces = null;
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
        if (controls.isAttention(p.which)) return p.shape.rows * H * D;
        const z = p.sizes();
        return z[0] + z[1] + z[2] + z[3];
    }
    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k < s.rows or s.n != H * D) return false;
        p.hostFree();
        p.rig.reset();
        p.shape = s;
        p.fill = fill;
        const global = p.which == .prep_global or p.which == .decode_global or p.which == .decode_global_long or controls.isPrompt(p.which);
        const sliding = p.which == .prep_sliding or p.which == .decode_sliding;
        const ring: u32 = if (controls.isAttention(p.which)) @intCast(std.mem.alignForward(usize, @max(2048, s.k), 64)) else 2048;
        p.layer = .{ .window = if (sliding) 1024 else 0, .ring = ring, .chunk = if (sliding) 256 else plan.global_chunk, .tau = global };
        const rows = s.rows;
        const rh = rows * H;
        const cache = KV * p.layer.ring * D;
        p.pos = try p.a.alloc(usize, rows);
        for (p.pos, 0..) |*v, r| v.* = s.k - rows + r;
        p.q = try p.a.alloc(u16, rh * D);
        p.rel = try p.a.alloc(u16, rh * R);
        p.kc = try p.a.alloc(u16, if (controls.isAttention(p.which)) cache else rows * KV * D);
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
        p.fillValues(prng.random(), global);
        if (fill == .nonfinite) p.plant();
        try p.bind();
        return true;
    }
    fn fillValues(p: *Probe, rng: std.Random, global: bool) void {
        const stress = p.fill == .stress;
        const attn = controls.isAttention(p.which);
        for (p.q) |*v| v.* = ref.bits(if (stress and attn) 0 else if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 511))) - 255) else rng.floatNorm(f64) * 0.3);
        for (p.kc) |*v| v.* = ref.bits(if (stress and attn) 0 else if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64) * 0.3);
        for (p.vc, 0..) |*v, i| v.* = if (stress and attn) ref.bits(ref.periodic(i / D % p.layer.ring, i % D)) else ref.bits(if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64));
        for (p.sink) |*v| v.* = ref.bits(if (stress) -std.math.inf(f64) else 2 + 4 * rng.float(f64));
        for (p.gate) |*v| v.* = ref.bits(if (stress) 32 else rng.floatNorm(f64));
        for (p.rel, 0..) |*v, i| v.* = ref.bits(if (stress and attn) 0 else if (stress) @floatFromInt(2 * (i % 31) + 1) else rng.floatNorm(f64));
        for (p.proj) |*v| v.* = ref.bits(if (stress) 0 else rng.floatNorm(f64) * 0.5);
        for (p.qs) |*v| v.* = ref.bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
        for (p.rs) |*v| v.* = ref.bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
        if (global and !attn) _ = ref.tau(p.pos[0]);
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
    fn bind(p: *Probe) !void {
        const rows = p.shape.rows;
        if (controls.isAttention(p.which)) {
            const arrays = [_][]u16{ p.q, p.kc, p.vc, p.rel, p.proj, p.sink, p.gate };
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
            const stride = plan.slotsFor(p.layer, p.shape.k);
            const parts = rows * H * stride;
            p.po = try p.rig.scratch(parts * D * 4);
            p.pml = try p.rig.scratch(parts * 8);
            p.out[0] = try p.rig.output(p.count() * 2);
            if (p.base != null) {
                p.base_tab = try p.rig.scratch(rows * H * EXTENT * 2);
                p.base_po = try p.rig.scratch(parts * D * 4);
                p.base_pml = try p.rig.scratch(parts * 8);
                p.base_out = try p.rig.scratch(p.count() * 2);
            }
        } else for (p.sizes(), 0..) |n, i| p.out[i] = try p.rig.output(n * 2);
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.main);
        const rows: u32 = @intCast(p.shape.rows);
        const ring = p.layer.ring;
        if (!controls.isAttention(p.which)) {
            @memset(p.rig.bytes(p.out[2], p.sizes()[2] * 2), 0);
            @memset(p.rig.bytes(p.out[3], p.sizes()[3] * 2), 0);
            for (p.refs[0..6], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            for (p.meta, 6..) |r, i| e.setBuffer(r.buf, r.off, i);
            for (p.out, 9..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setValue(plan.PrepArgs{ .heads = H, .kv_heads = KV, .ring = ring, .global = @intFromBool(p.layer.tau), .qk_clip = 22.627416997969522, .rel_clip = 8, .floor_pos = 8192, .alpha = 0.1, .eps = 1e-6 }, 13);
            e.dispatchGroups(mtl.Size.of(rows, H + 2 * KV, 1), mtl.Size.of(32, 1, 1));
        } else {
            const stride = plan.slotsFor(p.layer, p.shape.k);
            for (p.refs[0..5], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setBuffer(p.meta[2].buf, p.meta[2].off, 5);
            e.setBuffer(p.po.buf, p.po.off, 6);
            e.setBuffer(p.pml.buf, p.pml.off, 7);
            const sc: f32 = 1.0 / @sqrt(@as(f32, D));
            if (controls.isPrompt(p.which)) {
                e.setValue(plan.PromptArgs{ .heads = H, .kv_heads = KV, .ring = ring, .chunk = p.layer.chunk, .c0 = 0, .chunks = stride, .scale = sc }, 8);
                const blocks = (rows * (H / KV) + plan.prompt_pairs - 1) / plan.prompt_pairs;
                e.dispatchGroups(mtl.Size.of(blocks, KV, stride), mtl.Size.of(256, 1, 1));
            } else {
                e.setBuffer(p.meta[0].buf, p.meta[0].off, 8);
                e.setBuffer(p.meta[1].buf, p.meta[1].off, 9);
                e.setValue(plan.AttnArgs{ .rows = rows, .heads = H, .kv_heads = KV, .ring = ring, .window = p.layer.window, .chunk = p.layer.chunk, .c0 = 0, .chunks = stride, .scale = sc }, 10);
                const row_blocks = (rows + plan.decode_row_block - 1) / plan.decode_row_block;
                e.dispatchGroups(mtl.Size.of(KV, stride, row_blocks), mtl.Size.of(plan.decode_threads, 1, 1));
            }
            e.barrier();
            ref.encodeMerge(e, p.merge, p.po, p.pml, p.refs[5], p.refs[6], p.meta[0], p.out[0], p.layer, rows, stride);
        }
        try p.rig.finish(pass);
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        const b = p.base orelse return error.NoBaseline;
        return b.run(&p.rig, p.which, p.layer, p.shape, &p.refs, &p.meta, p.base_tab, p.base_po, p.base_pml, p.base_out);
    }
    fn inputs(p: *const Probe) ref.Inputs {
        return .{ .which = p.which, .shape = p.shape, .layer = p.layer, .q = p.q, .kc = p.kc, .vc = p.vc, .sink = p.sink, .gate = p.gate, .rel = p.rel, .proj = p.proj, .qs = p.qs, .rs = p.rs, .pos = p.pos, .sizes = p.sizes() };
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (controls.isAttention(p.which)) return ref.attentionRef(p.a, p.inputs(), null, want, mag);
        return ref.prepRef(p.inputs(), null, want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        const sim: ref.Sim = .{ .acc = acc, .order = order };
        if (controls.isAttention(p.which)) return ref.attentionRef(p.a, p.inputs(), sim, out, mag);
        return ref.prepRef(p.inputs(), sim, out, mag);
    }
};

fn supported(device: mtl.Device, w: Which) bool {
    return switch (w) {
        .prompt_m5, .prompt_m5_long => device.tensorUnits(),
        .prompt_pre_m5, .prompt_pre_m5_long => !device.tensorUnits(),
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
