//! Mini's weights on one GPU: bf16 projections for the lane matmul, NVFP4/MXFP8 core experts, the small tensors.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const names = @import("names.zig");
const plan_ = @import("plan.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Lane = cuda.qmmf.Weight;
const ex = cuda.experts;
const mx = cuda.experts_mx;
const Tensor = core.checkpoint.Tensor;

/// NVFP4's outer scale denominator as Mini ships it (scales/RESULTS.md); another value is refused.
pub const nvfp4_denominator = 2688;

/// One packed buffer a layer [q|gate|rel|sink|k|v], views at one shared K slice: fused rows keep the separate bits.
pub const Fused = struct {
    /// Row offsets of the parts in the fused output.
    q: usize,
    gate: usize,
    rel: usize,
    sink: usize,
    k: usize,
    v: usize,
    /// The q-side rows (q, gate, rel, sink: one input), and every row (q, k and v from one input).
    qx: usize,
    all: usize,

    pub fn of(c: *const config.Text) Fused {
        const qw: usize = @as(usize, c.local.q) * c.local.dim;
        const kvw: usize = @as(usize, c.local.kv) * c.local.dim;
        const rel = 2 * qw;
        const sink = rel + @as(usize, c.local.q) * c.d_rel;
        const k = sink + std.mem.alignForward(usize, c.local.q, 64);
        return .{ .q = 0, .gate = qw, .rel = rel, .sink = sink, .k = k, .v = k + kvw, .qx = k, .all = k + 2 * kvw };
    }
};

pub const Attn = struct {
    qx: Lane, // the fused q-side rows (one input: q, gate, rel, sink)
    all: Lane, // every fused row (q, k and v from one input)
    wq: Lane,
    wk: Lane,
    wv: Lane,
    wr: Lane, // relative-bias queries [heads * d_rel, H]
    gate: Lane,
    sink: Lane,
    wo: Lane,
    k_taps: u64 = 0, // global: bf16 [kv * 4, H] (dense rows), the input-dependent conv taps
    v_taps: u64 = 0,
    q_scale: u64, // bf16 [heads, 128]
    rel_scale: u64, // bf16 [heads, d_rel]
    rel_proj: u64, // bf16 [d_rel, extent]
    k_conv: u64, // sliding: bf16 [kv * 128][4]; global: static_w bf16 [4][kv * 128]
    v_conv: u64,
    norms: [3]u64, // sliding: attn_norm three times; global: the q, k and v input norms
};

pub const Controller = struct { down: u64, forget: u64, gain: u64 };

pub const Layer = struct {
    attn: Attn,
    attn_conv: u64, // bf16 [H][4]
    mlp_conv: u64,
    mlp_norm: u64,
    router: u64, // bf16 [E, H]
    router_bias: u64, // fp32 [E]
    attn_query: u64 = 0, // fp32 [1 or 3, H]; 0 for layer 0
    mlp_query: u64, // fp32 [H]
    controllers: ?[4]Controller = null, // core layers: attn (q), k, v inputs, then the MLP input
    experts: ex.Layer,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: config.Config,
    plan: plan_.Plan,
    layers: []Layer = &.{},
    embed: u64 = 0, // bf16 [V, H]
    embed_norm: u64 = 0,
    norm: u64 = 0,
    unembed: Lane = undefined, // the first unpadded_vocab rows
    output_query: u64 = 0, // fp32 [H]
    cross_main: u64 = 0, // fp32 [2 core, H]
    cross_kv: u64 = 0,
    ones: u64 = 0, // fp32 [max taps] of 1.0: dense rows' scaled form rounds the dynamic taps to bf16
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    fn upload(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        var b = try DeviceBuffer.fromHost(d, bytes);
        errdefer b.free();
        try w.buffers.append(w.gpa, b);
        w.bytes += bytes.len;
        return b.ptr;
    }

    /// The six input projections packed as Fused orders them; every view pinned to the parts' K slices.
    fn fused(w: *Weights, d: *const cuda.Driver, parts: [6]Tensor, a: *Attn) !void {
        const f = Fused.of(&w.config.text);
        const k = parts[0].dim(1);
        const offs = [6]usize{ f.q, f.gate, f.rel, f.sink, f.k, f.v };
        const npad = cuda.modelopt.padded(f.all);
        const src = try w.gpa.alloc(u16, f.all * k);
        defer w.gpa.free(src);
        @memset(src, 0);
        const ends = [6]usize{ f.gate, f.rel, f.sink, f.k, f.v, f.all };
        for (parts, offs, ends) |t, o, e| {
            if (t.dim(1) != k or o + t.dim(0) > e) return error.Inkling2FusedShape;
            @memcpy(std.mem.sliceAsBytes(src[o * k ..][0 .. t.dim(0) * k]), t.bytes[0 .. t.dim(0) * k * 2]);
        }
        const out = try w.gpa.alloc(u16, npad * k);
        defer w.gpa.free(out);
        cuda.qmmf.packBf16(out, src, f.all, k);
        const base = try w.upload(d, std.mem.sliceAsBytes(out));
        const sk = cuda.qmmf.splitK(parts[0].dim(0), k);
        for (parts) |t| if (cuda.qmmf.splitK(t.dim(0), k) != sk) return error.Inkling2FusedSlices;
        const view = struct {
            fn at(ptr: u64, row: usize, n: usize, kk: usize, slices: u32) !Lane {
                return cuda.qmmf.pinned(cuda.modelopt.bf16Weight(ptr + row * kk * 2, n, kk), slices);
            }
        };
        a.qx = try view.at(base, 0, f.qx, k, sk);
        a.all = try view.at(base, 0, f.all, k, sk);
        const lanes = [_]*Lane{ &a.wq, &a.gate, &a.wr, &a.sink, &a.wk, &a.wv };
        for (lanes, parts, offs) |l, t, o| l.* = try view.at(base, o, t.dim(0), k, sk);
    }

    /// A bf16 [n, k] projection (its first `rows` rows) in the lane's order, padded to 128 rows.
    fn lane(w: *Weights, d: *const cuda.Driver, t: Tensor, rows: usize) !Lane {
        const k = t.dim(t.rank - 1);
        const n = rows;
        const npad = cuda.modelopt.padded(n);
        const out = try w.gpa.alloc(u16, npad * k);
        defer w.gpa.free(out);
        const src = try w.gpa.alloc(u16, n * k);
        defer w.gpa.free(src);
        @memcpy(std.mem.sliceAsBytes(src), t.bytes[0 .. n * k * 2]);
        cuda.qmmf.packBf16(out, src, n, k);
        return cuda.modelopt.bf16Weight(try w.upload(d, std.mem.sliceAsBytes(out)), n, k);
    }
};

fn get(ck: *core.checkpoint.Checkpoint, comptime fmt: []const u8, args: anytype) !Tensor {
    var buf: [128]u8 = undefined;
    return ck.get(try std.fmt.bufPrint(&buf, fmt, args));
}

fn raw(w: *Weights, d: *const cuda.Driver, ck: *core.checkpoint.Checkpoint, comptime fmt: []const u8, args: anytype) !u64 {
    return w.upload(d, (try get(ck, fmt, args)).bytes);
}

/// rel_logits_proj [d_rel][e] widened to the kernels' [d_rel][1024] with zero columns: no bias past e, as in the model.
fn relProj(w: *Weights, d: *const cuda.Driver, t: Tensor) !u64 {
    const ext = @import("kernels.zig").extent;
    const rows = t.dim(0);
    const e = t.dim(1);
    if (e > ext) return error.Inkling2RelExtent;
    if (e == ext) return w.upload(d, t.bytes);
    const out = try w.gpa.alloc(u16, rows * ext);
    defer w.gpa.free(out);
    @memset(out, 0);
    for (0..rows) |r| @memcpy(std.mem.sliceAsBytes(out[r * ext ..][0..e]), t.bytes[r * e * 2 ..][0 .. e * 2]);
    return w.upload(d, std.mem.sliceAsBytes(out));
}

fn proj(w: *Weights, d: *const cuda.Driver, ck: *core.checkpoint.Checkpoint, comptime fmt: []const u8, args: anytype) !Lane {
    const t = try get(ck, fmt, args);
    return w.lane(d, t, t.dim(0));
}

/// The header of every shard against the weight map: a missing, extra or misshapen tensor refuses the load by name.
fn admit(gpa: std.mem.Allocator, ck: *core.checkpoint.Checkpoint, c: *const config.Config, prec: []plan_.Precision) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var all: core.safetensors.Header = .empty;
    for (ck.files.items) |*f| try names.merge(a, &all, f.names);
    try names.precisions(all, c.text.layers, prec);
    const r = try names.check(a, c, all, .model);
    if (r.clean()) return;
    for (r.missing.items) |n| std.log.err("inkling2 mini: the checkpoint has no {s}", .{n});
    for (r.wrong.items) |n| std.log.err("inkling2 mini: {s} has another dtype or shape than the config implies", .{n});
    for (r.unknown.items) |n| std.log.err("inkling2 mini: {s} is not a tensor the config implies", .{n});
    return error.Inkling2WeightMap;
}

/// config.json read and admitted (inkling2_loop, ModelOpt via core), the map checked, then layer by layer to the GPU.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Weights {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const json = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(json);
    var why: config.Why = .{};
    const c = config.parse(gpa, json, &why) catch |e| {
        std.log.err("inkling2: {s}", .{why.text()});
        return e;
    };
    if (c.text.loop == null) return error.Inkling2NotLooped; // Small serves on its own engine
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    var prec: [config.max_layers]plan_.Precision = undefined;
    try admit(gpa, &ck, &c, prec[0..c.text.layers]);
    var w: Weights = .{ .gpa = gpa, .config = c, .plan = try plan_.build(&c.text, prec[0..c.text.layers]) };
    errdefer w.deinit();
    const t = &c.text;
    w.layers = try gpa.alloc(Layer, t.layers);
    w.embed = try raw(&w, d, &ck, "model.llm.embed.weight", .{});
    w.embed_norm = try raw(&w, d, &ck, "model.llm.embed_norm.weight", .{});
    w.norm = try raw(&w, d, &ck, "model.llm.norm.weight", .{});
    w.output_query = try raw(&w, d, &ck, "model.llm.attn_res.output_query", .{});
    w.cross_main = try raw(&w, d, &ck, "model.llm.middle_cycle_cross_pass_queries", .{});
    w.cross_kv = try raw(&w, d, &ck, "model.llm.middle_cycle_cross_pass_kv_queries", .{});
    w.unembed = try w.lane(d, try ck.get("model.llm.unembed.weight"), t.unpadded_vocab);
    w.ones = try w.upload(d, std.mem.sliceAsBytes(&@as([64]f32, @splat(1.0))));
    for (w.layers, 0..) |*l, i| {
        l.* = try layer(&w, d, &ck, @intCast(i), prec[i]);
        ck.release(); // the mapped pages already copied: drop them from the page cache's working set
    }
    return w;
}

fn layer(w: *Weights, d: *const cuda.Driver, ck: *core.checkpoint.Checkpoint, i: u32, prec: plan_.Precision) !Layer {
    const c = &w.config.text;
    const L = .{i};
    const global = config.isGlobal(i);
    var a: Attn = .{
        .qx = undefined,
        .all = undefined,
        .wq = undefined,
        .wk = undefined,
        .wv = undefined,
        .wr = undefined,
        .gate = undefined,
        .sink = undefined,
        .wo = try proj(w, d, ck, "model.llm.layers.{d}.attn.wo_ud.weight", L),
        .q_scale = try raw(w, d, ck, "model.llm.layers.{d}.attn.q_l2_clip.scale", L),
        .rel_scale = try raw(w, d, ck, "model.llm.layers.{d}.attn.rel_l2_clip.scale", L),
        .rel_proj = try relProj(w, d, try get(ck, "model.llm.layers.{d}.attn.rel_logits_proj.proj", L)),
        .k_conv = undefined,
        .v_conv = undefined,
        .norms = undefined,
    };
    const pre = "model.llm.layers.{d}.attn.";
    try w.fused(d, .{
        try get(ck, pre ++ "wq_du.weight", L), try get(ck, pre ++ "gate_attn.weight", L),
        try get(ck, pre ++ "wr_du.weight", L), try get(ck, pre ++ "attn_sink_proj.weight", L),
        try get(ck, pre ++ "wk_dv.weight", L), try get(ck, pre ++ "wv_dv.weight", L),
    }, &a);
    if (global) {
        a.k_conv = try raw(w, d, ck, "model.llm.layers.{d}.attn.k_sconv.static_w", L);
        a.v_conv = try raw(w, d, ck, "model.llm.layers.{d}.attn.v_sconv.static_w", L);
        a.k_taps = try raw(w, d, ck, "model.llm.layers.{d}.attn.k_sconv.weight_proj", L);
        a.v_taps = try raw(w, d, ck, "model.llm.layers.{d}.attn.v_sconv.weight_proj", L);
        a.norms = .{
            try raw(w, d, ck, "model.llm.layers.{d}.q_input_norm.weight", L),
            try raw(w, d, ck, "model.llm.layers.{d}.k_input_norm.weight", L),
            try raw(w, d, ck, "model.llm.layers.{d}.v_input_norm.weight", L),
        };
    } else {
        a.k_conv = try raw(w, d, ck, "model.llm.layers.{d}.attn.k_sconv.weight", L);
        a.v_conv = try raw(w, d, ck, "model.llm.layers.{d}.attn.v_sconv.weight", L);
        const n = try raw(w, d, ck, "model.llm.layers.{d}.attn_norm.weight", L);
        a.norms = .{ n, n, n };
    }
    var l: Layer = .{
        .attn = a,
        .attn_conv = try raw(w, d, ck, "model.llm.layers.{d}.attn_sconv.weight", L),
        .mlp_conv = try raw(w, d, ck, "model.llm.layers.{d}.mlp_sconv.weight", L),
        .mlp_norm = try raw(w, d, ck, "model.llm.layers.{d}.mlp_norm.weight", L),
        .router = try raw(w, d, ck, "model.llm.layers.{d}.mlp.gate.weight", L),
        .router_bias = try raw(w, d, ck, "model.llm.layers.{d}.mlp.gate.bias", L),
        .mlp_query = try raw(w, d, ck, "model.llm.attn_res.mlp_queries.{d}", L),
        .experts = try experts(w, d, ck, i, prec),
    };
    // attn_queries.j feeds layer j + 1's attention input; layer 0 norms the embedding alone
    if (i > 0) l.attn_query = try raw(w, d, ck, "model.llm.attn_res.attn_queries.{d}", .{i - 1});
    if (c.loop) |lp| if (i >= lp.start and i < lp.end) {
        var cs: [4]Controller = undefined;
        inline for (names.controllers, 0..) |g, k| cs[k] = .{
            .down = try raw(w, d, ck, "model.llm.layers.{d}.controllers." ++ g ++ ".down_weight", L),
            .forget = try raw(w, d, ck, "model.llm.layers.{d}.controllers." ++ g ++ ".forget_weight", L),
            .gain = try raw(w, d, ck, "model.llm.layers.{d}.controllers." ++ g ++ ".gain_weight", L),
        };
        l.controllers = cs;
    };
    return l;
}

/// A layer's experts as stored, packed for core experts: w13's interleaved rows split into gate and up first.
fn experts(w: *Weights, d: *const cuda.Driver, ck: *core.checkpoint.Checkpoint, i: u32, prec: plan_.Precision) !ex.Layer {
    const c = &w.config.text;
    const e: usize = c.experts;
    const h: usize = c.hidden;
    const m: usize = c.expert_inter;
    const nv = prec == .nvfp4;
    const pre = "model.llm.layers.{d}.mlp.experts.";
    const L = .{i};
    const w13 = try get(ck, pre ++ "w13_weight", L);
    const s13 = try get(ck, pre ++ "w13_weight_scale", L);
    const w2 = try get(ck, pre ++ "w2_weight", L);
    const s2 = try get(ck, pre ++ "w2_weight_scale", L);
    const up_words = (if (nv) ex.Nvfp4Experts.bytes(1, m, h, 2) else mx.Mxfp8Experts.bytes(1, m, h, 2)) / 4;
    const down_words = (if (nv) ex.Nvfp4Experts.bytes(1, h, m, 1) else mx.Mxfp8Experts.bytes(1, h, m, 1)) / 4;
    const up = try w.gpa.alloc(u32, e * up_words);
    defer w.gpa.free(up);
    const down = try w.gpa.alloc(u32, e * down_words);
    defer w.gpa.free(down);
    const cb = w13.bytes.len / e;
    const sb = s13.bytes.len / e;
    const split = try w.gpa.alloc(u8, cb + sb);
    defer w.gpa.free(split);
    const c2 = w2.bytes.len / e;
    const sc2 = s2.bytes.len / e;
    for (0..e) |x| {
        mx.splitGateUp(w13.bytes[x * cb ..][0..cb], cb / (2 * m), split[0 .. cb / 2], split[cb / 2 .. cb]);
        mx.splitGateUp(s13.bytes[x * sb ..][0..sb], sb / (2 * m), split[cb..][0 .. sb / 2], split[cb + sb / 2 ..][0 .. sb / 2]);
        const gate = [2][]const u8{ split[0 .. cb / 2], split[cb..][0 .. sb / 2] };
        const upm = [2][]const u8{ split[cb / 2 .. cb], split[cb + sb / 2 ..][0 .. sb / 2] };
        const o = up[x * up_words ..][0..up_words];
        const dn = down[x * down_words ..][0..down_words];
        if (nv) {
            ex.Nvfp4Experts.packGateUp(o, gate, upm, m, h);
            ex.Nvfp4Experts.packOne(dn, w2.bytes[x * c2 ..][0..c2], s2.bytes[x * sc2 ..][0..sc2], h, m);
        } else {
            mx.Mxfp8Experts.packGateUp(o, gate, upm, m, h);
            mx.Mxfp8Experts.packOne(dn, w2.bytes[x * c2 ..][0..c2], s2.bytes[x * sc2 ..][0..sc2], h, m);
        }
    }
    var l: ex.Layer = .{ .format = if (nv) .nvfp4 else .mxfp8, .up = try w.upload(d, std.mem.sliceAsBytes(up)), .down = try w.upload(d, std.mem.sliceAsBytes(down)), .width = m, .dims = h, .experts = e, .caps = .{ c.gate_cap, c.up_cap } };
    if (nv) {
        var f13 = try w.gpa.alloc(f32, 2 * e);
        defer w.gpa.free(f13);
        var f2 = try w.gpa.alloc(f32, e);
        defer w.gpa.free(f2);
        inline for (.{ "w13", "w2" }, 0..) |name, k| {
            const amax = try get(ck, pre ++ name ++ "_weight_amax", L);
            const den = std.mem.bytesToValue(f32, (try get(ck, pre ++ name ++ "_weight_outer_scale_denominator", L)).bytes[0..4]);
            if (den != nvfp4_denominator) return error.Inkling2Nvfp4Denominator;
            for (0..e) |x| {
                const f = mx.nvfp4Factor(std.mem.bytesToValue(f32, amax.bytes[x * 4 ..][0..4]), den);
                if (k == 0) {
                    f13[2 * x] = f; // one amax covers the gate and the up rows
                    f13[2 * x + 1] = f;
                } else f2[x] = f;
            }
        }
        l.up_scale = try w.upload(d, std.mem.sliceAsBytes(f13));
        l.down_scale = try w.upload(d, std.mem.sliceAsBytes(f2));
    }
    return l;
}

test "the fused rows: parts in order, sink padded to 64 rows, every view on a 64-row block edge" {
    var c: config.Text = undefined;
    c.local = .{ .q = 24, .kv = 8, .dim = 128 };
    c.d_rel = 16;
    const f = Fused.of(&c);
    try std.testing.expectEqual(Fused{ .q = 0, .gate = 3072, .rel = 6144, .sink = 6528, .k = 6592, .v = 7616, .qx = 6592, .all = 8640 }, f);
    for ([_]usize{ f.k, f.v, f.qx, f.all }) |r| try std.testing.expect(r % 64 == 0);
    // Mini's six parts share 4 K slices at K 2304 (the fused views pin them); the tiny checkpoint's share 1 at K 256
    for ([_]usize{ 3072, 1024, 384, 24 }) |n| try std.testing.expectEqual(@as(u32, 4), cuda.qmmf.splitK(n, 2304));
    for ([_]usize{ 512, 256, 64, 4 }) |n| try std.testing.expectEqual(@as(u32, 1), cuda.qmmf.splitK(n, 256));
}
