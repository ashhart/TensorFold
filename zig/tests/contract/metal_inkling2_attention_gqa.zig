//! Inkling 2 Mini Metal GQA 3:1 and batched multi-segment attention gates 1-4 with 0-ULP scalar parity.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_attention_gqa_controls.zig");
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
const max_segs = 3;

const sliding_shapes = [_]c.Shape{ .{ .rows = 7, .n = H * D, .k = 1030 }, .{ .rows = 3, .n = H * D, .k = 2049 }, .{ .rows = 5, .n = H * D, .k = 17 }, .{ .rows = 1, .n = H * D, .k = 1024 } };
const global_shapes = [_]c.Shape{ .{ .rows = 4, .n = H * D, .k = 1100 }, .{ .rows = 2, .n = H * D, .k = 1030 }, .{ .rows = 3, .n = H * D, .k = 17 }, .{ .rows = 1, .n = H * D, .k = 1025 } };
const long_shapes = [_]c.Shape{.{ .rows = 5, .n = H * D, .k = 17450 }};
const batched_prep_sliding_shapes = [_]c.Shape{ .{ .rows = 6, .n = H * D, .k = 1030 }, .{ .rows = 3, .n = H * D, .k = 2049 }, .{ .rows = 2, .n = H * D, .k = 17 } };
const batched_prep_global_shapes = [_]c.Shape{ .{ .rows = 6, .n = H * D, .k = 9000 }, .{ .rows = 3, .n = H * D, .k = 1048576 }, .{ .rows = 2, .n = H * D, .k = 1030 } };
const batched_sliding_shapes = [_]c.Shape{ .{ .rows = 6, .n = H * D, .k = 2049 }, .{ .rows = 4, .n = H * D, .k = 1030 }, .{ .rows = 2, .n = H * D, .k = 257 } };
const batched_global_shapes = [_]c.Shape{ .{ .rows = 5, .n = H * D, .k = 1100 }, .{ .rows = 3, .n = H * D, .k = 1030 }, .{ .rows = 2, .n = H * D, .k = 1025 } };

fn depth(s: c.Shape) usize {
    return D + 4 * s.k + 64;
}

fn bindRefs(e: mtl.ComputeEncoder, refs: []const rig_.Ref, first: usize) void {
    for (refs, first..) |r, slot| e.setBuffer(r.buf, r.off, slot);
}

fn offRef(r: rig_.Ref, delta: usize) rig_.Ref {
    return .{ .buf = r.buf, .off = r.off + delta };
}

fn spec(w: Which) c.Spec {
    var s: c.Spec = .{ .name = "", .accum = .f32, .depth = depth, .ops = 64, .mids = 8, .out = .bf16, .served = &global_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .exact = true, .why = "P and outputs have required bf16 roundings" } };
    switch (w) {
        .decode_gqa_sliding => {
            s.name = "metal inkling2 attention scalar gqa sliding tf_ink2_attn_scalar_gqa";
            s.served = &sliding_shapes;
            s.replaces = "tf_ink2_attn_scalar then tf_ink2_merge";
        },
        .decode_gqa_global => {
            s.name = "metal inkling2 attention scalar gqa global tf_ink2_attn_scalar_gqa";
            s.served = &global_shapes;
            s.replaces = "tf_ink2_attn_scalar then tf_ink2_merge";
        },
        .decode_gqa_global_long => {
            s.name = "metal inkling2 attention scalar gqa global long tf_ink2_attn_scalar_gqa";
            s.accum = .none;
            s.sums = .{ .f16 = 0, .bf16 = 0, .why = "byte-exact against scalar baseline on multi-chunk long context" };
            s.served = &long_shapes;
            s.replaces = "tf_ink2_attn_scalar then tf_ink2_merge";
        },
        .prep_batched_sliding, .prep_batched_global => {
            s.name = if (w == .prep_batched_global) "metal inkling2 attention prep batched global tf_ink2_prep_batched" else "metal inkling2 attention prep batched sliding tf_ink2_prep_batched";
            s.served = if (w == .prep_batched_global) &batched_prep_global_shapes else &batched_prep_sliding_shapes;
            s.ops = 32;
            s.mids = 2;
            s.replaces = "per-segment tf_ink2_prep";
        },
        .decode_gqa_batched_sliding => {
            s.name = "metal inkling2 attention scalar gqa batched sliding tf_ink2_attn_scalar_gqa_batched";
            s.served = &batched_sliding_shapes;
            s.replaces = "per-segment tf_ink2_attn_scalar then tf_ink2_merge";
        },
        .decode_gqa_batched_global => {
            s.name = "metal inkling2 attention scalar gqa batched global tf_ink2_attn_scalar_gqa_batched";
            s.served = &batched_global_shapes;
            s.replaces = "per-segment tf_ink2_attn_scalar then tf_ink2_merge";
        },
        .decode_gqa_batched_global_long => {
            s.name = "metal inkling2 attention scalar gqa batched global long tf_ink2_attn_scalar_gqa_batched";
            s.accum = .none;
            s.sums = .{ .f16 = 0, .bf16 = 0, .why = "byte-exact against per-segment scalar baseline on multi-chunk long context" };
            s.served = &long_shapes;
            s.replaces = "per-segment tf_ink2_attn_scalar then tf_ink2_merge";
        },
    }
    return s;
}

const SegInfo = struct { row0: usize = 0, rows: usize = 0, k: usize = 0, ring: u32 = 0 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: Mutation,
    lib: mtl.Library,
    main: mtl.Pipeline,
    merge: mtl.Pipeline,
    base_lib: ?mtl.Library = null,
    base_pipe: mtl.Pipeline = undefined,
    base_merge: mtl.Pipeline = undefined,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    n_segs: usize = 1,
    segs: [max_segs]SegInfo = @splat(.{}),
    q: []u16 = &.{},
    rel: []u16 = &.{},
    kc_in: []u16 = &.{},
    vc_in: []u16 = &.{},
    kc_seg: [max_segs][]u16 = @splat(&.{}),
    vc_seg: [max_segs][]u16 = @splat(&.{}),
    sink: []u16 = &.{},
    gate: []u16 = &.{},
    proj: []u16 = &.{},
    qs: []u16 = &.{},
    rs: []u16 = &.{},
    pos: []usize = &.{},
    seg_of: []i32 = &.{},
    // [0]=q, [1]=rel, [2]=kc_in/proj, [3]=vc_in/sink, [4]=qs/gate, [5]=rs
    in_refs: [6]rig_.Ref = undefined,
    // [0]=pos, [1]=seg_of, [2]=single_seg
    meta: [3]rig_.Ref = undefined,
    kv_in: [max_segs][2]rig_.Ref = undefined,
    kv_out: [max_segs][2]rig_.Ref = undefined,
    base_kv: [max_segs][2]rig_.Ref = undefined,
    // [0]=po/out_q, [1]=pml/out_rel, [2]=out_y
    work: [3]rig_.Ref = undefined,
    // [0]=base_po/base_q, [1]=base_pml/base_rel, [2]=base_out
    base_work: [3]rig_.Ref = undefined,
    base_bytes: std.ArrayList(u8) = .empty,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, m: Mutation) !Probe {
        const text = try controls.apply(a, sources.attention, m);
        defer a.free(text);
        var lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        var main = try mtl.Pipeline.init(device, lib, w.function(), false);
        errdefer main.deinit();
        var merge = try mtl.Pipeline.init(device, lib, "tf_ink2_merge", false);
        errdefer merge.deinit();
        var base_lib: ?mtl.Library = null;
        var base_pipe: mtl.Pipeline = undefined;
        var base_merge: mtl.Pipeline = undefined;
        if (m == .none) {
            base_lib = try mtl.Library.fromSource(device, sources.attention, mtl.CompileOptions.mlx());
            errdefer base_lib.?.deinit();
            base_pipe = try mtl.Pipeline.init(device, base_lib.?, if (w.isAttention()) "tf_ink2_attn_scalar" else "tf_ink2_prep", false);
            errdefer base_pipe.deinit();
            base_merge = try mtl.Pipeline.init(device, base_lib.?, "tf_ink2_merge", false);
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .mutation = m, .lib = lib, .main = main, .merge = merge, .base_lib = base_lib, .base_pipe = base_pipe, .base_merge = base_merge };
    }

    fn hostFree(p: *Probe) void {
        inline for (.{ "q", "rel", "kc_in", "vc_in", "sink", "gate", "proj", "qs", "rs", "pos", "seg_of" }) |f| {
            p.a.free(@field(p, f));
            @field(p, f) = &.{};
        }
        for (&p.kc_seg, &p.vc_seg) |*ks, *vs| {
            p.a.free(ks.*);
            p.a.free(vs.*);
            ks.* = &.{};
            vs.* = &.{};
        }
    }

    pub fn deinit(p: *Probe) void {
        p.hostFree();
        p.base_bytes.deinit(p.a);
        p.rig.deinit();
        if (p.base_lib) |bl| {
            p.base_merge.deinit();
            p.base_pipe.deinit();
            bl.deinit();
        }
        p.merge.deinit();
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

    fn layerFor(p: *const Probe, ring: u32) plan.Layer {
        const global = p.which.isGlobal();
        return .{ .window = if (global) 0 else 1024, .ring = ring, .chunk = if (global) plan.global_chunk else plan.sliding_chunk, .tau = global };
    }

    pub fn count(p: *const Probe) usize {
        if (p.which.isAttention()) return p.shape.rows * H * D;
        var total = p.shape.rows * H * D + p.shape.rows * H * R;
        for (p.segs[0..p.n_segs]) |s| total += 2 * KV * s.ring * D;
        return total;
    }

    fn maxChunks(p: *const Probe) u32 {
        var mc: u32 = 1;
        for (p.segs[0..p.n_segs]) |s| mc = @max(mc, plan.activeChunks(p.layerFor(s.ring), s.k - 1));
        return mc;
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k < s.rows or s.n != H * D) return false;
        if (p.which.isBatched() and s.rows < 2) return false;
        p.hostFree();
        p.rig.reset();
        p.shape = s;
        p.fill = fill;
        p.n_segs = if (p.which.isBatched()) @min(s.rows, max_segs) else 1;
        var row0: usize = 0;
        const base_rows = s.rows / p.n_segs;
        const rem = s.rows % p.n_segs;
        for (0..p.n_segs) |si| {
            const sr = base_rows + @as(usize, if (si < rem) 1 else 0);
            const sk = if (si + 1 == p.n_segs) s.k else if (si == 0) @max(sr, @min(s.k, 17)) else @max(sr, @min(s.k, 1030));
            const ring_base = if (p.which.isAttention()) std.mem.alignForward(usize, @max(2048, sk), 64) else 2048;
            p.segs[si] = .{ .row0 = row0, .rows = sr, .k = sk, .ring = @intCast(ring_base + si * 64) };
            row0 += sr;
        }
        p.pos = try p.a.alloc(usize, s.rows);
        p.seg_of = try p.a.alloc(i32, s.rows);
        for (p.segs[0..p.n_segs], 0..) |sg, si| for (0..sg.rows) |r| {
            p.pos[sg.row0 + r] = sg.k - sg.rows + r;
            p.seg_of[sg.row0 + r] = @intCast(si);
        };
        const rh = s.rows * H;
        p.q = try p.a.alloc(u16, rh * D);
        p.rel = try p.a.alloc(u16, rh * R);
        if (p.which.isAttention()) {
            for (p.segs[0..p.n_segs], 0..) |sg, si| {
                p.kc_seg[si] = try p.a.alloc(u16, KV * sg.ring * D);
                p.vc_seg[si] = try p.a.alloc(u16, KV * sg.ring * D);
            }
            p.proj = try p.a.alloc(u16, R * EXTENT);
            p.sink = try p.a.alloc(u16, rh);
            p.gate = try p.a.alloc(u16, rh * D);
        } else {
            p.kc_in = try p.a.alloc(u16, s.rows * KV * D);
            p.vc_in = try p.a.alloc(u16, s.rows * KV * D);
            p.qs = try p.a.alloc(u16, H * D);
            p.rs = try p.a.alloc(u16, H * R);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        p.fillValues(prng.random());
        if (fill == .nonfinite) p.plant();
        try p.bind();
        return true;
    }

    fn fillValues(p: *Probe, rng: std.Random) void {
        const stress = p.fill == .stress;
        const attn = p.which.isAttention();
        for (p.q) |*v| v.* = ref.bits(if (stress and attn) 0 else if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 511))) - 255) else rng.floatNorm(f64) * 0.3);
        for (p.rel, 0..) |*v, i| v.* = ref.bits(if (stress and attn) 0 else if (stress) @floatFromInt(2 * (i % 31) + 1) else rng.floatNorm(f64));
        if (attn) {
            for (p.segs[0..p.n_segs], 0..) |sg, si| {
                for (p.kc_seg[si]) |*v| v.* = ref.bits(if (stress) 0 else rng.floatNorm(f64) * 0.3);
                for (p.vc_seg[si], 0..) |*v, i| v.* = if (stress) ref.bits(ref.periodic(i / D % sg.ring, i % D)) else ref.bits(rng.floatNorm(f64));
            }
            for (p.sink) |*v| v.* = ref.bits(if (stress) -std.math.inf(f64) else 2 + 4 * rng.float(f64));
            for (p.gate) |*v| v.* = ref.bits(if (stress) 32 else rng.floatNorm(f64));
            for (p.proj) |*v| v.* = ref.bits(if (stress) 0 else rng.floatNorm(f64) * 0.5);
        } else {
            for (p.kc_in) |*v| v.* = ref.bits(if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64) * 0.3);
            for (p.vc_in) |*v| v.* = ref.bits(if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64));
            for (p.qs) |*v| v.* = ref.bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
            for (p.rs) |*v| v.* = ref.bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
        }
    }

    fn plant(p: *Probe) void {
        const nan = ref.bits(std.math.nan(f64));
        const inf = ref.bits(std.math.inf(f64));
        if (p.which.isAttention()) {
            p.q[D + 5] = nan;
            p.sink[2] = inf;
            p.gate[3 * D] = inf;
            p.gate[3 * D + 1] = inf | 0x8000;
            p.rel[4 * R + 2] = nan;
            p.rel[5 * R + 1] = inf;
        } else {
            p.q[D + 3] = nan;
            p.kc_in[5] = inf;
            p.vc_in[D] = inf | 0x8000;
            p.rel[2] = nan;
        }
    }

    fn bind(p: *Probe) !void {
        const rows = p.shape.rows;
        p.in_refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.q));
        p.in_refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.rel));
        const pos32 = try p.a.alloc(i32, rows);
        defer p.a.free(pos32);
        for (pos32, p.pos) |*v, x| v.* = @intCast(x);
        var seg_tbl: [max_segs]plan.Seg = undefined;
        for (p.segs[0..p.n_segs], 0..) |sg, si| seg_tbl[si] = .{ .row0 = 0, .n = @intCast(sg.rows), .p0 = @intCast(sg.k - sg.rows), .off = 0 };
        p.meta = .{
            try p.rig.input(std.mem.sliceAsBytes(pos32)),
            try p.rig.input(std.mem.sliceAsBytes(p.seg_of)),
            try p.rig.input(std.mem.sliceAsBytes(seg_tbl[0..p.n_segs])),
        };
        if (p.which.isAttention()) {
            for (0..p.n_segs) |si| {
                p.kv_in[si] = .{ try p.rig.input(std.mem.sliceAsBytes(p.kc_seg[si])), try p.rig.input(std.mem.sliceAsBytes(p.vc_seg[si])) };
            }
            p.in_refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.proj));
            p.in_refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.sink));
            p.in_refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.gate));
            var max_slots: u32 = p.maxChunks();
            for (p.segs[0..p.n_segs]) |sg| max_slots = @max(max_slots, plan.slotsFor(p.layerFor(sg.ring), sg.k));
            const parts = rows * H * max_slots;
            p.work = .{ try p.rig.scratch(parts * D * 4), try p.rig.scratch(parts * 8), try p.rig.output(rows * H * D * 2) };
            if (p.base_lib != null) p.base_work = .{ try p.rig.scratch(parts * D * 4), try p.rig.scratch(parts * 8), try p.rig.scratch(rows * H * D * 2) };
        } else {
            p.in_refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.kc_in));
            p.in_refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.vc_in));
            p.in_refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.qs));
            p.in_refs[5] = try p.rig.input(std.mem.sliceAsBytes(p.rs));
            p.work[0] = try p.rig.output(rows * H * D * 2);
            p.work[1] = try p.rig.output(rows * H * R * 2);
            for (p.segs[0..p.n_segs], 0..) |sg, si| p.kv_out[si] = .{ try p.rig.output(KV * sg.ring * D * 2), try p.rig.output(KV * sg.ring * D * 2) };
            if (p.base_lib != null) {
                p.base_work[0] = try p.rig.scratch(rows * H * D * 2);
                p.base_work[1] = try p.rig.scratch(rows * H * R * 2);
                for (p.segs[0..p.n_segs], 0..) |sg, si| p.base_kv[si] = .{ try p.rig.scratch(KV * sg.ring * D * 2), try p.rig.scratch(KV * sg.ring * D * 2) };
            }
        }
    }

    fn makeBatchedSegs(p: *Probe, e: mtl.ComputeEncoder, kv: *const [max_segs][2]rig_.Ref, usage: usize, out: *[max_segs]plan.BatchedSeg) []const plan.BatchedSeg {
        for (p.segs[0..p.n_segs], 0..) |sg, si| {
            e.useResource(kv[si][0].buf, usage);
            e.useResource(kv[si][1].buf, usage);
            out[si] = .{ .k_cache = kv[si][0].buf.gpuAddress() + kv[si][0].off, .v_cache = kv[si][1].buf.gpuAddress() + kv[si][1].off, .row0 = @intCast(sg.row0), .n = @intCast(sg.rows), .p0 = @intCast(sg.k - sg.rows), .ring = sg.ring };
        }
        return out[0..p.n_segs];
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows: u32 = @intCast(p.shape.rows);
        const sc: f32 = 1.0 / @sqrt(@as(f32, D));
        e.setPipeline(p.main);
        if (!p.which.isAttention()) {
            for (p.segs[0..p.n_segs], 0..) |sg, si| {
                @memset(p.rig.bytes(p.kv_out[si][0], KV * sg.ring * D * 2), 0);
                @memset(p.rig.bytes(p.kv_out[si][1], KV * sg.ring * D * 2), 0);
            }
            var bsegs: [max_segs]plan.BatchedSeg = undefined;
            const segs = p.makeBatchedSegs(e, &p.kv_out, mtl.ResourceUsage.read | mtl.ResourceUsage.write, &bsegs);
            bindRefs(e, &.{ p.in_refs[0], p.in_refs[2], p.in_refs[3], p.in_refs[1], p.in_refs[4], p.in_refs[5], p.meta[0], p.meta[1] }, 0);
            e.setBytes(std.mem.sliceAsBytes(segs), 8);
            bindRefs(e, p.work[0..2], 9);
            e.setValue(plan.PrepArgs{ .heads = H, .kv_heads = KV, .ring = 0, .global = @intFromBool(p.which.isGlobal()), .qk_clip = 22.627416997969522, .rel_clip = 8, .floor_pos = 8192, .alpha = 0.1, .eps = 1e-6 }, 11);
            e.dispatchGroups(mtl.Size.of(rows, H + 2 * KV, 1), mtl.Size.of(32, 1, 1));
        } else if (p.which.isBatched()) {
            const mc = p.maxChunks();
            const layer0 = p.layerFor(p.segs[0].ring);
            var bsegs: [max_segs]plan.BatchedSeg = undefined;
            const segs = p.makeBatchedSegs(e, &p.kv_in, mtl.ResourceUsage.read, &bsegs);
            bindRefs(e, &.{ p.in_refs[0], p.in_refs[1], p.in_refs[2] }, 0);
            e.setBytes(std.mem.sliceAsBytes(segs), 3);
            bindRefs(e, &.{ p.work[0], p.work[1], p.meta[0], p.meta[1] }, 4);
            e.setValue(plan.AttnArgs{ .heads = H, .kv_heads = KV, .ring = 0, .window = layer0.window, .chunk = layer0.chunk, .c0 = 0, .chunks = mc, .scale = sc }, 8);
            e.dispatchGroups(mtl.Size.of(rows * KV, mc, 1), mtl.Size.of(32, 1, 1));
            e.barrier();
            ref.encodeMerge(e, p.merge, p.work[0], p.work[1], p.in_refs[3], p.in_refs[4], p.meta[0], p.work[2], layer0, rows, mc);
        } else {
            const layer = p.layerFor(p.segs[0].ring);
            const stride = plan.slotsFor(layer, p.shape.k);
            bindRefs(e, &.{ p.in_refs[0], p.kv_in[0][0], p.kv_in[0][1], p.in_refs[1], p.in_refs[2], p.meta[2], p.work[0], p.work[1], p.meta[0], p.meta[1] }, 0);
            e.setValue(plan.AttnArgs{ .heads = H, .kv_heads = KV, .ring = layer.ring, .window = layer.window, .chunk = layer.chunk, .c0 = 0, .chunks = stride, .scale = sc }, 10);
            e.dispatchGroups(mtl.Size.of(rows * KV, stride, 1), mtl.Size.of(32, 1, 1));
            e.barrier();
            ref.encodeMerge(e, p.merge, p.work[0], p.work[1], p.in_refs[3], p.in_refs[4], p.meta[0], p.work[2], layer, rows, stride);
        }
        try p.rig.finish(pass);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        if (p.base_lib == null) return error.NoBaseline;
        const pass = p.rig.begin();
        const e = pass.e;
        const sc: f32 = 1.0 / @sqrt(@as(f32, D));
        if (!p.which.isAttention()) {
            @memset(p.rig.bytes(p.base_work[0], p.shape.rows * H * D * 2), 0xff);
            @memset(p.rig.bytes(p.base_work[1], p.shape.rows * H * R * 2), 0xff);
            for (p.segs[0..p.n_segs], 0..) |sg, si| {
                const nbytes = KV * sg.ring * D * 2;
                @memset(p.rig.bytes(p.base_kv[si][0], nbytes), 0);
                @memset(p.rig.bytes(p.base_kv[si][1], nbytes), 0);
                e.setPipeline(p.base_pipe);
                bindRefs(e, &.{ offRef(p.in_refs[0], sg.row0 * H * D * 2), offRef(p.in_refs[2], sg.row0 * KV * D * 2), offRef(p.in_refs[3], sg.row0 * KV * D * 2), offRef(p.in_refs[1], sg.row0 * H * R * 2), p.in_refs[4], p.in_refs[5], offRef(p.meta[0], sg.row0 * 4), offRef(p.meta[1], sg.row0 * 4), p.meta[2], offRef(p.base_work[0], sg.row0 * H * D * 2), offRef(p.base_work[1], sg.row0 * H * R * 2), p.base_kv[si][0], p.base_kv[si][1] }, 0);
                e.setValue(plan.PrepArgs{ .heads = H, .kv_heads = KV, .ring = sg.ring, .global = @intFromBool(p.which.isGlobal()), .qk_clip = 22.627416997969522, .rel_clip = 8, .floor_pos = 8192, .alpha = 0.1, .eps = 1e-6 }, 13);
                e.dispatchGroups(mtl.Size.of(@intCast(sg.rows), H + 2 * KV, 1), mtl.Size.of(32, 1, 1));
            }
            try p.rig.finish(pass);
            p.base_bytes.clearRetainingCapacity();
            try p.base_bytes.appendSlice(p.a, p.rig.bytes(p.base_work[0], p.shape.rows * H * D * 2));
            try p.base_bytes.appendSlice(p.a, p.rig.bytes(p.base_work[1], p.shape.rows * H * R * 2));
            for (p.segs[0..p.n_segs], 0..) |sg, si| {
                const nbytes = KV * sg.ring * D * 2;
                try p.base_bytes.appendSlice(p.a, p.rig.bytes(p.base_kv[si][0], nbytes));
                try p.base_bytes.appendSlice(p.a, p.rig.bytes(p.base_kv[si][1], nbytes));
            }
            return p.base_bytes.items;
        }
        const out_bytes = p.shape.rows * H * D * 2;
        @memset(p.rig.bytes(p.base_work[2], out_bytes), 0xff);
        for (p.segs[0..p.n_segs], 0..) |sg, si| {
            const layer = p.layerFor(sg.ring);
            const stride = plan.slotsFor(layer, sg.k);
            e.setPipeline(p.base_pipe);
            bindRefs(e, &.{ offRef(p.in_refs[0], sg.row0 * H * D * 2), p.kv_in[si][0], p.kv_in[si][1], offRef(p.in_refs[1], sg.row0 * H * R * 2), p.in_refs[2], p.meta[2], p.base_work[0], p.base_work[1], offRef(p.meta[0], sg.row0 * 4), offRef(p.meta[1], sg.row0 * 4) }, 0);
            e.setValue(plan.AttnArgs{ .heads = H, .kv_heads = KV, .ring = sg.ring, .window = layer.window, .chunk = layer.chunk, .c0 = 0, .chunks = stride, .scale = sc }, 10);
            e.dispatchGroups(mtl.Size.of(@intCast(sg.rows * H), stride, 1), mtl.Size.of(32, 1, 1));
            e.barrier();
            ref.encodeMerge(e, p.base_merge, p.base_work[0], p.base_work[1], offRef(p.in_refs[3], sg.row0 * H * 2), offRef(p.in_refs[4], sg.row0 * H * D * 2), offRef(p.meta[0], sg.row0 * 4), offRef(p.base_work[2], sg.row0 * H * D * 2), layer, @intCast(sg.rows), stride);
            e.barrier();
        }
        try p.rig.finish(pass);
        return p.rig.bytes(p.base_work[2], out_bytes);
    }

    fn evalAll(p: *Probe, sim: ?ref.Sim, want: []f64, mag: []f64) !void {
        const ref_which: @import("metal_inkling2_attention_controls.zig").Which = if (p.which.isAttention()) (if (p.which.isGlobal()) .decode_global else .decode_sliding) else (if (p.which.isGlobal()) .prep_global else .prep_sliding);
        if (p.which.isAttention()) {
            for (p.segs[0..p.n_segs], 0..) |sg, si| {
                const in: ref.Inputs = .{ .which = ref_which, .shape = .{ .rows = sg.rows, .n = H * D, .k = sg.k }, .layer = p.layerFor(sg.ring), .q = p.q[sg.row0 * H * D ..][0 .. sg.rows * H * D], .kc = p.kc_seg[si], .vc = p.vc_seg[si], .sink = p.sink[sg.row0 * H ..][0 .. sg.rows * H], .gate = p.gate[sg.row0 * H * D ..][0 .. sg.rows * H * D], .rel = p.rel[sg.row0 * H * R ..][0 .. sg.rows * H * R], .proj = p.proj, .qs = &.{}, .rs = &.{}, .pos = p.pos[sg.row0..][0..sg.rows], .sizes = .{ 0, 0, 0, 0 } };
                try ref.attentionRef(p.a, in, sim, want[sg.row0 * H * D ..][0 .. sg.rows * H * D], mag[sg.row0 * H * D ..][0 .. sg.rows * H * D]);
            }
            return;
        }
        @memset(want, 0);
        @memset(mag, 0);
        const rel_base = p.shape.rows * H * D;
        var kv_off = rel_base + p.shape.rows * H * R;
        for (p.segs[0..p.n_segs]) |sg| {
            const qn = sg.rows * H * D;
            const rn = sg.rows * H * R;
            const cn = KV * sg.ring * D;
            const tmp_w = try p.a.alloc(f64, qn + rn + 2 * cn);
            defer p.a.free(tmp_w);
            const tmp_m = try p.a.alloc(f64, qn + rn + 2 * cn);
            defer p.a.free(tmp_m);
            const in: ref.Inputs = .{ .which = ref_which, .shape = .{ .rows = sg.rows, .n = H * D, .k = sg.k }, .layer = p.layerFor(sg.ring), .q = p.q[sg.row0 * H * D ..][0..qn], .kc = p.kc_in[sg.row0 * KV * D ..][0 .. sg.rows * KV * D], .vc = p.vc_in[sg.row0 * KV * D ..][0 .. sg.rows * KV * D], .sink = &.{}, .gate = &.{}, .rel = p.rel[sg.row0 * H * R ..][0..rn], .proj = &.{}, .qs = p.qs, .rs = p.rs, .pos = p.pos[sg.row0..][0..sg.rows], .sizes = .{ qn, rn, cn, cn } };
            try ref.prepRef(in, sim, tmp_w, tmp_m);
            @memcpy(want[sg.row0 * H * D ..][0..qn], tmp_w[0..qn]);
            @memcpy(mag[sg.row0 * H * D ..][0..qn], tmp_m[0..qn]);
            @memcpy(want[rel_base + sg.row0 * H * R ..][0..rn], tmp_w[qn..][0..rn]);
            @memcpy(mag[rel_base + sg.row0 * H * R ..][0..rn], tmp_m[qn..][0..rn]);
            @memcpy(want[kv_off..][0 .. 2 * cn], tmp_w[qn + rn ..][0 .. 2 * cn]);
            @memcpy(mag[kv_off..][0 .. 2 * cn], tmp_m[qn + rn ..][0 .. 2 * cn]);
            kv_off += 2 * cn;
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        return p.evalAll(null, want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        return p.evalAll(.{ .acc = acc, .order = order }, out, mag);
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
            if (only != null and std.mem.indexOf(u8, m.name(), only.?) == null) continue;
            try s.add(m.which(), m);
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
        try contract.register(s.a, &s.probes, &s.kernels, p, p.kernel());
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
