//! Inkling 2 Mini Metal tree-attention gates 1-4 (prep_tree, scalar_tree, prompt prefix/tail).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_tree_controls.zig");
const ref = @import("metal_inkling2_attention_ref.zig");
const sources = @import("inkling2_metal");
const host = @import("tensorfold").inkling2_metal_host;
const plan = host.attention.planning;
const tree_plan = host.tree.plan;
const c = contract.spec;
const g = contract.gates;
const Which = controls.Which;
const Mutation = controls.Mutation;

const D = ref.D;
const H = ref.H;
const KV = ref.KV;
const R = ref.R;
const EXTENT = ref.EXTENT;
const scale = ref.scale;

const prep_shapes = [_]c.Shape{ .{ .rows = 5, .n = H * D, .k = 1030 }, .{ .rows = 4, .n = H * D, .k = 1048576 } };
const scalar_shapes = [_]c.Shape{ .{ .rows = 5, .n = H * D, .k = 1030 }, .{ .rows = 6, .n = H * D, .k = 2049 } };
const prefix_shapes = [_]c.Shape{
    .{ .rows = 5, .n = H * D, .k = 100 },
    .{ .rows = 6, .n = H * D, .k = 1100 },
    .{ .rows = 5, .n = H * D, .k = 16500 },
};
const tail_shapes = [_]c.Shape{
    .{ .rows = 5, .n = H * D, .k = 40 },
    .{ .rows = 5, .n = H * D, .k = 1031 },
    .{ .rows = 6, .n = H * D, .k = 1100 },
    .{ .rows = 6, .n = H * D, .k = 16388 },
};

fn depth(s: c.Shape) usize {
    return D + 4 * s.k + 64;
}

pub fn spec(w: Which) c.Spec {
    return switch (w) {
        .prep_tree => .{
            .name = "metal inkling2 attention prep tree tf_ink2_prep_tree",
            .accum = .f32,
            .depth = depth,
            .ops = 32,
            .mids = 2,
            .out = .bf16,
            .served = &prep_shapes,
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = true, .why = "Q/rel round to bf16 before tau; KV round once" },
        },
        .attn_scalar_tree => .{
            .name = "metal inkling2 attention scalar tree tf_ink2_attn_scalar_tree",
            .accum = .f32,
            .depth = depth,
            .ops = 64,
            .mids = 8,
            .out = .bf16,
            .served = &scalar_shapes,
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = true, .why = "P and outputs have required bf16 roundings" },
        },
        .prompt_m5_tree_prefix, .prompt_pre_m5_tree_prefix => .{
            .name = if (w == .prompt_m5_tree_prefix)
                "metal inkling2 prompt m5 tree prefix tf_ink2_attn_prompt_tree_prefix"
            else
                "metal inkling2 prompt pre-m5 tree prefix tf_ink2_attn_prompt_tree_prefix",
            .accum = .f32,
            .depth = depth,
            .ops = 64,
            .mids = 8,
            .out = .bf16,
            .served = &prefix_shapes,
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = true, .why = "P and outputs have required bf16 roundings" },
        },
        .prompt_m5_tree_tail, .prompt_pre_m5_tree_tail => .{
            .name = if (w == .prompt_m5_tree_tail)
                "metal inkling2 prompt m5 tree tail tf_ink2_attn_prompt_tree_tail"
            else
                "metal inkling2 prompt pre-m5 tree tail tf_ink2_attn_prompt_tree_tail",
            .accum = .f32,
            .depth = depth,
            .ops = 64,
            .mids = 8,
            .out = .bf16,
            .served = &tail_shapes,
            .sums = .{ .f16 = 0, .bf16 = 0, .exact = true, .why = "P and outputs have required bf16 roundings" },
        },
        else => unreachable,
    };
}

const bf = ref.bf;
const rn = ref.rn;
const bits = ref.bits;
const tau = ref.tau;
const periodic = ref.periodic;
fn minNum(x: f64, y: f64) f64 {
    return if (std.math.isNan(x)) y else if (std.math.isNan(y)) x else @min(x, y);
}
fn maxNan(x: f64, y: f64) f64 {
    return if (std.math.isNan(x) or std.math.isNan(y)) std.math.nan(f64) else @max(x, y);
}
fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

const Sim = struct { acc: c.Accum, order: g.Order };
fn sum(v: []const f64, sim: ?Sim) f64 {
    if (sim) |s| return contract.rows.sumIn(s.acc, s.order, v);
    var total: f64 = 0;
    for (v) |x| total += x;
    return total;
}

fn promptChunkWeights(score: []const f64, weights: []f64) f64 {
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
                const old_s = @exp(upper - merged);
                const new_s = @exp(top - merged);
                for (weights[0..sub_first]) |*w| w.* *= old_s;
                for (weights[sub_first..end]) |*w| w.* *= new_s;
                upper = merged;
            }
            top = -std.math.inf(f64);
        }
        first = end;
    }
    return upper;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: Mutation,
    lib: mtl.Library,
    main: mtl.Pipeline,
    plain_lib: ?mtl.Library = null,
    helper: ?mtl.Pipeline = null,
    attn_lib: ?mtl.Library = null,
    merge: ?mtl.Pipeline = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    layer: plan.Layer = undefined,
    start: u32 = 0,
    last_pos: u32 = 0,
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
    paths: []u32 = &.{},
    refs: [10]rig_.Ref = undefined,
    out: [4]rig_.Ref = undefined,
    meta: [3]rig_.Ref = undefined,
    tree_paths_ref: rig_.Ref = undefined,
    po: rig_.Ref = undefined,
    pml: rig_.Ref = undefined,
    tail_o: rig_.Ref = undefined,
    tail_ml: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, m: Mutation) !Probe {
        const text = try controls.apply(a, w.baseSource(), m);
        defer a.free(text);
        var lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        var main = try mtl.Pipeline.init(device, lib, w.function(), false);
        errdefer main.deinit();
        var plain_lib: ?mtl.Library = null;
        var helper: ?mtl.Pipeline = null;
        var attn_lib: ?mtl.Library = null;
        var merge: ?mtl.Pipeline = null;
        errdefer {
            if (merge) |*p| p.deinit();
            if (attn_lib) |*l| l.deinit();
            if (helper) |*p| p.deinit();
            if (plain_lib) |*l| l.deinit();
        }
        if (w.isAttention()) {
            const merge_src = if (w.isPrompt()) blk: {
                plain_lib = try mtl.Library.fromSource(device, w.baseSource(), mtl.CompileOptions.mlx());
                const peer = if (w == .prompt_m5_tree_prefix or w == .prompt_pre_m5_tree_prefix)
                    "tf_ink2_attn_prompt_tree_tail"
                else
                    "tf_ink2_attn_prompt_tree_prefix";
                helper = try mtl.Pipeline.init(device, plain_lib.?, peer, false);
                attn_lib = try mtl.Library.fromSource(device, sources.attention, mtl.CompileOptions.mlx());
                break :blk attn_lib.?;
            } else lib;
            merge = try mtl.Pipeline.init(device, merge_src, "tf_ink2_merge", false);
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = w, .mutation = m, .lib = lib, .main = main, .plain_lib = plain_lib, .helper = helper, .attn_lib = attn_lib, .merge = merge };
    }

    fn hostFree(p: *Probe) void {
        inline for (.{ "q", "kc", "vc", "sink", "gate", "rel", "proj", "qs", "rs", "pos", "paths" }) |f| {
            p.a.free(@field(p, f));
            @field(p, f) = &.{};
        }
    }

    pub fn deinit(p: *Probe) void {
        p.hostFree();
        p.rig.deinit();
        if (p.merge) |*pl| pl.deinit();
        if (p.attn_lib) |*l| l.deinit();
        if (p.helper) |*pl| pl.deinit();
        if (p.plain_lib) |*l| l.deinit();
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

    fn rigCache(p: *const Probe) usize {
        return KV * p.layer.ring * D;
    }
    fn sizes(p: *const Probe) [4]usize {
        const n = p.rigCache();
        return .{ p.shape.rows * H * D, p.shape.rows * H * R, n, n };
    }
    pub fn count(p: *const Probe) usize {
        if (p.which.isAttention()) return p.shape.rows * H * D;
        const z = p.sizes();
        return z[0] + z[1] + z[2] + z[3];
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > tree_plan.window_rows or s.k < s.rows or s.n != H * D) return false;
        if (p.which != .prep_tree and s.k > 65536) return false;
        p.hostFree();
        p.rig.reset();
        p.shape = s;
        p.fill = fill;
        const global = p.which == .prep_tree or p.which.isPrompt();
        const ring: u32 = if (p.which.isPrompt()) @intCast(std.mem.alignForward(usize, @max(s.k, 2048), 64)) else 2048;
        p.layer = .{ .window = if (global) 0 else 1024, .ring = ring, .chunk = if (global) plan.global_chunk else 256, .tau = global };
        const rows = s.rows;
        const rh = rows * H;
        p.start = @intCast(s.k - rows);
        var parents: [tree_plan.window_rows]i32 = undefined;
        var depths: [tree_plan.window_rows]u32 = undefined;
        parents[0] = -1;
        for (1..rows) |r| parents[r] = switch (r) {
            1, 2 => 0,
            3 => 1,
            4 => 2,
            else => @intCast(r - 1),
        };
        try tree_plan.validateParents(parents[0..rows], depths[0..rows]);
        p.paths = try p.a.alloc(u32, rows * tree_plan.window_rows);
        tree_plan.buildTreePaths(parents[0..rows], depths[0..rows], p.paths);
        p.pos = try p.a.alloc(usize, rows);
        var max_d: u32 = 0;
        for (0..rows) |r| {
            max_d = @max(max_d, depths[r]);
            p.pos[r] = p.start + depths[r];
        }
        p.last_pos = p.start + max_d;
        p.q = try p.a.alloc(u16, rh * D);
        p.rel = try p.a.alloc(u16, rh * R);
        @memset(p.q, 0);
        @memset(p.rel, 0);
        p.kc = try p.a.alloc(u16, if (p.which.isAttention()) p.rigCache() else rows * KV * D);
        p.vc = try p.a.alloc(u16, p.kc.len);
        @memset(p.kc, 0);
        @memset(p.vc, 0);
        if (p.which.isAttention()) {
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
        const attn = p.which.isAttention();
        for (p.q) |*v| v.* = bits(if (stress and attn) 0 else if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 511))) - 255) else rng.floatNorm(f64) * 0.3);
        for (p.kc) |*v| v.* = bits(if (stress and attn) 0 else if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64) * 0.3);
        for (p.vc, 0..) |*v, i| v.* = if (stress and attn) bits(periodic(i / D % p.layer.ring, i % D)) else bits(if (stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 255))) - 127) else rng.floatNorm(f64));
        for (p.sink) |*v| v.* = bits(if (stress) -std.math.inf(f64) else 2 + 4 * rng.float(f64));
        for (p.gate) |*v| v.* = bits(if (stress) 32 else rng.floatNorm(f64));
        for (p.rel) |*v| v.* = bits(if (stress and attn) 0 else if (stress) 3 else rng.floatNorm(f64));
        for (p.proj) |*v| v.* = bits(if (stress) 0 else rng.floatNorm(f64) * 0.5);
        for (p.qs) |*v| v.* = bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
        for (p.rs) |*v| v.* = bits(if (stress) 1 else 0.5 + 2 * rng.float(f64));
    }

    fn plant(p: *Probe) void {
        const nan = bits(std.math.nan(f64));
        const inf = bits(std.math.inf(f64));
        if (p.which.isAttention()) {
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
        if (p.which.isAttention()) {
            for ([_][]u16{ p.q, p.kc, p.vc, p.rel, p.proj, p.sink, p.gate }, 0..) |x, i| p.refs[i] = try p.rig.input(std.mem.sliceAsBytes(x));
        } else {
            for ([_][]u16{ p.q, p.kc, p.vc, p.rel, p.qs, p.rs }, 0..) |x, i| p.refs[i] = try p.rig.input(std.mem.sliceAsBytes(x));
        }
        const pos32 = try p.a.alloc(i32, rows);
        defer p.a.free(pos32);
        const seg32 = try p.a.alloc(i32, rows);
        defer p.a.free(seg32);
        @memset(seg32, 0);
        for (pos32, p.pos) |*v, x| v.* = @intCast(x);
        const seg = [1]plan.Seg{.{ .row0 = 0, .n = @intCast(rows), .p0 = @intCast(p.start), .off = 0 }};
        p.meta[0] = try p.rig.input(std.mem.sliceAsBytes(pos32));
        p.meta[1] = try p.rig.input(std.mem.sliceAsBytes(seg32));
        p.meta[2] = try p.rig.input(std.mem.sliceAsBytes(&seg));
        if (p.which.isAttention()) {
            p.tree_paths_ref = try p.rig.input(std.mem.sliceAsBytes(p.paths));
            const r = plan.Rows{ .rows = @intCast(rows), .heads = H, .kv_heads = KV, .last_pos = p.last_pos };
            const stride = plan.slotsFor(p.layer, p.last_pos + 1);
            const parts = rows * H * stride;
            p.po = try p.rig.scratch(parts * D * 4);
            p.pml = try p.rig.scratch(parts * 8);
            if (p.which.isPrompt()) {
                p.tail_o = try p.rig.scratch(2 * rows * H * D * 4);
                p.tail_ml = try p.rig.scratch(rows * H * 16);
            }
            _ = r;
            p.out[0] = try p.rig.output(p.count() * 2);
        } else for (p.sizes(), 0..) |n, i| p.out[i] = try p.rig.output(n * 2);
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows: u32 = @intCast(p.shape.rows);
        if (p.which == .prep_tree) {
            e.setPipeline(p.main);
            @memset(p.rig.bytes(p.out[2], p.sizes()[2] * 2), 0);
            @memset(p.rig.bytes(p.out[3], p.sizes()[3] * 2), 0);
            for (p.refs[0..6], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            for (p.meta, 6..) |r, i| e.setBuffer(r.buf, r.off, i);
            for (p.out, 9..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setValue(plan.PrepArgs{ .heads = H, .kv_heads = KV, .ring = p.layer.ring, .global = 1, .qk_clip = 22.627416997969522, .rel_clip = 8, .floor_pos = 8192, .alpha = 0.1, .eps = 1e-6 }, 13);
            e.dispatchGroups(mtl.Size.of(rows, H + 2 * KV, 1), mtl.Size.of(32, 1, 1));
        } else {
            const r = plan.Rows{ .rows = rows, .heads = H, .kv_heads = KV, .last_pos = p.last_pos };
            const stride = plan.slotsFor(p.layer, p.last_pos + 1);
            const launch = try plan.attendTree(r, p.layer, stride, p.start);
            const sc: f32 = 1.0 / @sqrt(@as(f32, D));
            if (p.which.isPrompt()) {
                const is_prefix = p.which == .prompt_m5_tree_prefix or p.which == .prompt_pre_m5_tree_prefix;
                const args = plan.PromptTreeArgs{ .heads = H, .kv_heads = KV, .ring = p.layer.ring, .chunk = p.layer.chunk, .c0 = 0, .chunks = stride, .scale = sc, .p_tile = launch.p_tile, .c_off = launch.c_off, .max_pos = p.last_pos };
                if (launch.p_tile > 0) {
                    e.setPipeline(if (is_prefix) p.main else p.helper.?);
                    for (p.refs[0..5], 0..) |x, i| e.setBuffer(x.buf, x.off, i);
                    e.setBuffer(p.meta[2].buf, p.meta[2].off, 5);
                    e.setBuffer(p.po.buf, p.po.off, 6);
                    e.setBuffer(p.pml.buf, p.pml.off, 7);
                    e.setBuffer(p.meta[0].buf, p.meta[0].off, 8);
                    e.setBuffer(p.tail_o.buf, p.tail_o.off, 9);
                    e.setBuffer(p.tail_ml.buf, p.tail_ml.off, 10);
                    e.setValue(args, 11);
                    e.dispatchGroups(mtl.Size.of(launch.prefix_grid[0], launch.prefix_grid[1], launch.prefix_grid[2]), mtl.Size.of(launch.threads, 1, 1));
                    e.barrier();
                }
                e.setPipeline(if (!is_prefix) p.main else p.helper.?);
                for (p.refs[0..5], 0..) |x, i| e.setBuffer(x.buf, x.off, i);
                e.setBuffer(p.meta[2].buf, p.meta[2].off, 5);
                e.setBuffer(p.po.buf, p.po.off, 6);
                e.setBuffer(p.pml.buf, p.pml.off, 7);
                e.setBuffer(p.meta[0].buf, p.meta[0].off, 8);
                e.setBuffer(p.tree_paths_ref.buf, p.tree_paths_ref.off, 9);
                e.setBuffer(p.tail_o.buf, p.tail_o.off, 10);
                e.setBuffer(p.tail_ml.buf, p.tail_ml.off, 11);
                e.setValue(args, 12);
                e.dispatchGroups(mtl.Size.of(launch.tail_grid[0], launch.tail_grid[1], launch.tail_grid[2]), mtl.Size.of(launch.threads, 1, 1));
            } else {
                e.setPipeline(p.main);
                for (p.refs[0..5], 0..) |x, i| e.setBuffer(x.buf, x.off, i);
                e.setBuffer(p.meta[2].buf, p.meta[2].off, 5);
                e.setBuffer(p.po.buf, p.po.off, 6);
                e.setBuffer(p.pml.buf, p.pml.off, 7);
                e.setBuffer(p.meta[0].buf, p.meta[0].off, 8);
                e.setBuffer(p.meta[1].buf, p.meta[1].off, 9);
                e.setBuffer(p.tree_paths_ref.buf, p.tree_paths_ref.off, 10);
                e.setValue(plan.AttnArgs{ .heads = H, .kv_heads = KV, .ring = p.layer.ring, .window = p.layer.window, .chunk = p.layer.chunk, .c0 = 0, .chunks = stride, .scale = sc }, 11);
                e.dispatchGroups(mtl.Size.of(launch.tail_grid[0], launch.tail_grid[1], launch.tail_grid[2]), mtl.Size.of(launch.threads, 1, 1));
            }
            e.barrier();
            ref.encodeMerge(e, p.merge.?, p.po, p.pml, p.refs[5], p.refs[6], p.meta[0], p.out[0], p.layer, rows, stride);
        }
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (p.which == .prep_tree) return p.prepRef(null, want, mag);
        return p.attentionRef(null, want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        const sim: Sim = .{ .acc = acc, .order = order };
        if (p.which == .prep_tree) return p.prepRef(sim, out, mag);
        return p.attentionRef(sim, out, mag);
    }

    fn headClip(x: []const f64, threshold: f64, sim: ?Sim, t: ?f64, want: []f64, mag: []f64) void {
        var sq: [D]f64 = undefined;
        @memset(mag, 0);
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

    fn prepRef(p: *Probe, sim: ?Sim, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
        const z = p.sizes();
        var x: [D]f64 = undefined;
        for (0..p.shape.rows) |r| {
            const tv = tau(p.pos[r]);
            for (0..H) |h| {
                for (&x, 0..) |*v, d| v.* = bf(p.q[(r * H + h) * D + d]) * bf(p.qs[h * D + d]);
                headClip(&x, 22.627416997969522, sim, tv, want[(r * H + h) * D ..][0..D], mag[(r * H + h) * D ..][0..D]);
                for (x[0..R], 0..) |*v, d| v.* = bf(p.rel[(r * H + h) * R + d]) * bf(p.rs[h * R + d]);
                const at = z[0] + (r * H + h) * R;
                headClip(x[0..R], 8, sim, tv, want[at..][0..R], mag[at..][0..R]);
            }
            const kv_pos = @as(usize, p.start) + r;
            for (0..KV) |h| {
                const slot = (h * p.layer.ring + kv_pos % p.layer.ring) * D;
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

    fn attentionRef(p: *Probe, sim: ?Sim, want: []f64, mag: []f64) !void {
        const most = p.last_pos + 1;
        const score = try p.a.alloc(f64, most);
        defer p.a.free(score);
        const weights = try p.a.alloc(f64, most);
        defer p.a.free(weights);
        const terms = try p.a.alloc(f64, @max(most, D));
        defer p.a.free(terms);
        for (0..p.shape.rows) |r| {
            const pos = p.pos[r];
            const lo = if (p.layer.window > 0) (pos + 1) -| p.layer.window else 0;
            const n = pos + 1 - lo;
            for (0..H) |h| {
                const rh = r * H + h;
                const kvh = h / (H / KV);
                const z = bf(p.sink[rh]);
                const prompt = sim == null and p.which.isPrompt();
                var peak = if (prompt) -std.math.inf(f64) else z;
                for (lo..pos + 1) |kp| {
                    const j = kp - lo;
                    const phys = if (kp < p.start) kp else @as(usize, p.start) + p.paths[r * tree_plan.window_rows + (kp - p.start)];
                    const at = (kvh * p.layer.ring + phys % p.layer.ring) * D;
                    for (0..D) |d| terms[d] = bf(p.q[rh * D + d]) * bf(p.kc[at + d]);
                    const distance = pos - kp;
                    var bias: f64 = 0;
                    if (distance < EXTENT) {
                        var rterms: [R]f64 = undefined;
                        for (&rterms, 0..) |*v, rj| v.* = bf(p.rel[rh * R + rj]) * bf(p.proj[rj * EXTENT + distance]);
                        bias = rn(sum(&rterms, sim));
                    }
                    score[j] = sum(terms[0..D], sim) * scale + bias;
                    if (!prompt) peak = maxNan(peak, score[j]);
                }
                if (prompt) {
                    const chunk = p.layer.chunk;
                    var mm = -std.math.inf(f64);
                    var first: usize = 0;
                    while (first < n) {
                        const c_end = @min(first + chunk, n);
                        const cp = promptChunkWeights(score[first..c_end], weights[first..c_end]);
                        const mn = maxNan(mm, cp);
                        const x = @exp(mm - mn);
                        const y = @exp(cp - mn);
                        for (weights[0..first]) |*w| w.* *= x;
                        for (weights[first..c_end]) |*w| w.* *= y;
                        mm = mn;
                        first = c_end;
                    }
                    peak = mm;
                } else for (weights[0..n], score[0..n]) |*w, sc| w.* = @exp(sc - peak);
                const merged_peak = if (prompt) maxNan(peak, z) else peak;
                const rescale = if (prompt) @exp(peak - merged_peak) else 1;
                const den = sum(weights[0..n], sim) * rescale + @exp(z - merged_peak);
                for (0..D) |d| {
                    for (lo..pos + 1) |kp| {
                        const j = kp - lo;
                        const phys = if (kp < p.start) kp else @as(usize, p.start) + p.paths[r * tree_plan.window_rows + (kp - p.start)];
                        const at = (kvh * p.layer.ring + phys % p.layer.ring) * D;
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
