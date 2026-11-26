//! Inkling 2 Mini's resident text model: validated checkpoint weights and every Metal pipeline.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const cfg = @import("../config.zig");
const layer_plan = @import("../plan.zig");
const ck = @import("../../../core/checkpoint_metal.zig");
const shards = @import("core").checkpoint_host;
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const experts = @import("experts.zig");
const loop = @import("loop.zig");
const dense_file = @import("dense.zig");
const tree = @import("tree.zig");
const sample_file = @import("sample.zig");
const ink2_tfq3 = @import("tfq3.zig");
const affine4_file = @import("affine4.zig");
const model_affine4 = @import("model_affine4.zig");

pub const Tensor = ck.Tensor;
pub const Affine4Tensor = affine4_file.Affine4Tensor;
pub const DenseWeight = affine4_file.DenseWeight;
pub const DynamicConv = struct { static_w: Tensor, projection: Tensor };
pub const Conv = union(enum) { static: Tensor, dynamic: DynamicConv };
pub const Controller = struct { down: Tensor, forget: Tensor, gain: Tensor };
pub const AttentionWeights = struct {
    wq: DenseWeight,
    wk: DenseWeight,
    wv: DenseWeight,
    wr: DenseWeight,
    wo: DenseWeight,
    gate: DenseWeight,
    sink: DenseWeight,
    q_scale: Tensor,
    rel_scale: Tensor,
    rel_projection: Tensor,
    k_conv: Conv,
    v_conv: Conv,
    q_norm: Tensor,
    k_norm: Tensor,
    v_norm: Tensor,
};
pub const ExpertMatrix = struct {
    weight: Tensor,
    scale: Tensor,
    amax: ?Tensor = null,
    denominator: ?Tensor = null,
};
pub const LayerWeights = struct {
    attn: AttentionWeights,
    attn_conv: Tensor,
    mlp_norm: Tensor,
    mlp_conv: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    gu: ExpertMatrix,
    down: ExpertMatrix,
    tfq3: ?ink2_tfq3.Layer = null,
    controllers: ?[4]Controller,
};
pub const Weights = struct {
    embed: DenseWeight,
    embed_norm: Tensor,
    norm: Tensor,
    unembed: DenseWeight,
    output_query: Tensor,
    cross_queries: Tensor,
    cross_kv_queries: Tensor,
    attn_queries: [23]Tensor,
    mlp_queries: [24]Tensor,
    layers: [24]LayerWeights,
};

/// Scalar copies and residuals around dispatch stay on Metal; forward never reads tensors back to the host.
pub const runtime_source = sources.inkling2_runtime;

pub const RuntimeOp = enum { copy_bf16, copy_f32, widen, attention_partial, mlp_partial, capture_tap, stage_conv, stage_conv_segs, stage_conv_static, commit_histories };
pub const runtime_names = [_][:0]const u8{
    "tf_ink2_copy_bf16",
    "tf_ink2_copy_f32",
    "tf_ink2_widen",
    "tf_ink2_attention_partial",
    "tf_ink2_mlp_partial",
    "tf_ink2_capture_tap",
    "tf_ink2_stage_conv",
    "tf_ink2_stage_conv_segs",
    "tf_ink2_stage_conv_static",
    "tf_ink2_commit_histories",
};

pub fn admitConfig(c: *const cfg.Config) !void {
    if (c.quant != .modelopt_nvfp4 and c.quant != .mlx_affine4) return error.UnsupportedInkling2Quant;
    const t = c.text;
    const l = t.loop orelse return error.Inkling2NotMini;
    if (l.passes != 3 or l.core_start != 6 or l.core_end != 18 or l.gate_rank != 64)
        return error.UnsupportedInkling2Loop;
    if (t.hidden != 2304 or t.layers != 24 or t.vocab != 201024 or t.unpadded_vocab != 200058 or
        t.window != 1024 or t.d_rel != 16 or t.rel_extent != 1024 or t.log_floor != 8192 or
        t.block != 6 or t.experts != 128 or t.topk != 8 or t.expert_inter != 1152 or
        t.max_context != 1048576) return error.UnsupportedInkling2Geometry;
    for ([_]cfg.Heads{ t.local, t.global }) |h| if (h.q != 24 or h.kv != 8 or h.dim != 128)
        return error.UnsupportedInkling2Geometry;
    if (t.eps != 1e-6 or t.log_alpha != 0.1 or t.qk_clip != 22.627416997969522 or
        t.rel_clip != 8 or t.route_scale != 8 or t.gate_cap != 4 or t.up_cap != 25 or
        t.logit_cap != 30) return error.UnsupportedInkling2Constants;
}

fn require(t: Tensor, dtype: ck.DType, dims: []const usize) !Tensor {
    if (t.dtype != dtype or t.rank != dims.len) return error.UnexpectedInkling2Tensor;
    for (dims, 0..) |d, i| if (t.shape[i] != d) return error.UnexpectedInkling2Tensor;
    var count: usize = 1;
    for (dims) |d| count = std.math.mul(usize, count, d) catch return error.UnexpectedInkling2Tensor;
    if (t.bytes != count * dtype.size()) return error.UnexpectedInkling2Tensor;
    return t;
}

fn get(c: *const ck.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: ck.DType, dims: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    return require(try c.get(name), dtype, dims);
}

fn matrix(c: *const ck.Checkpoint, layer: u32, comptime suffix: []const u8, precision: layer_plan.Precision, rows: usize, k: usize) !ExpertMatrix {
    const base = "model.llm.layers.{d}.mlp.experts." ++ suffix;
    const e = 128;
    switch (precision) {
        .nvfp4 => return .{
            .weight = try get(c, base ++ "_weight", .{layer}, .u8, &.{ e, rows, k / 2 }),
            .scale = try get(c, base ++ "_weight_scale", .{layer}, .f8_e4m3, &.{ e, rows, k / 16 }),
            .amax = try get(c, base ++ "_weight_amax", .{layer}, .f32, &.{e}),
            .denominator = try get(c, base ++ "_weight_outer_scale_denominator", .{layer}, .f32, &.{1}),
        },
        .mxfp8 => return .{
            .weight = try get(c, base ++ "_weight", .{layer}, .f8_e4m3, &.{ e, rows, k }),
            .scale = try get(c, base ++ "_weight_scale", .{layer}, .u8, &.{ e, rows, k / 32 }),
        },
        .exl3, .tfq3 => return error.UnsupportedInkling2ExpertFormat,
    }
}

fn loadConv(c: *const ck.Checkpoint, layer: u32, comptime role: []const u8, global: bool) !Conv {
    if (!global) return .{ .static = try get(c, "model.llm.layers.{d}.attn." ++ role ++ "_sconv.weight", .{layer}, .bf16, &.{ 1024, 1, 4 }) };
    return .{ .dynamic = .{
        .static_w = try get(c, "model.llm.layers.{d}.attn." ++ role ++ "_sconv.static_w", .{layer}, .bf16, &.{ 4, 1024 }),
        .projection = try get(c, "model.llm.layers.{d}.attn." ++ role ++ "_sconv.weight_proj", .{layer}, .bf16, &.{ 32, 2304 }),
    } };
}

fn loadWeights(m: *Model, dir: []const u8) !Weights {
    const c = &m.checkpoint;
    const config = &m.config;
    const wp = &m.weight_plan;
    var w: Weights = undefined;
    w.embed = try affine4_file.loadBf16Dense(c, "model.llm.embed", .{}, 201024, 2304);
    w.embed_norm = try get(c, "model.llm.embed_norm.weight", .{}, .bf16, &.{2304});
    w.norm = try get(c, "model.llm.norm.weight", .{}, .bf16, &.{2304});
    w.unembed = try affine4_file.loadBf16Dense(c, "model.llm.unembed", .{}, 201024, 2304);
    w.output_query = try get(c, "model.llm.attn_res.output_query", .{}, .f32, &.{2304});
    w.cross_queries = try get(c, "model.llm.middle_cycle_cross_pass_queries", .{}, .f32, &.{ 24, 2304 });
    w.cross_kv_queries = try get(c, "model.llm.middle_cycle_cross_pass_kv_queries", .{}, .f32, &.{ 24, 2304 });
    for (&w.attn_queries, 0..) |*q, i| {
        const global = cfg.isGlobal(@intCast(i + 1));
        const dims: []const usize = if (global) &.{ 3, 2304 } else &.{2304};
        q.* = try get(c, "model.llm.attn_res.attn_queries.{d}", .{i}, .f32, dims);
    }
    for (&w.mlp_queries, 0..) |*q, i| q.* = try get(c, "model.llm.attn_res.mlp_queries.{d}", .{i}, .f32, &.{2304});
    for (&w.layers, 0..) |*l, i| {
        const layer: u32 = @intCast(i);
        const p = wp.at(layer);
        const global = p.kind == .global;
        const q_width: u32 = p.heads.q * p.heads.dim;
        const local_norm = if (!global) try get(c, "model.llm.layers.{d}.attn_norm.weight", .{layer}, .bf16, &.{2304}) else undefined;
        l.attn = .{
            .wq = try affine4_file.loadBf16Dense(c, "model.llm.layers.{d}.attn.wq_du", .{layer}, q_width, 2304),
            .wk = try affine4_file.loadBf16Dense(c, "model.llm.layers.{d}.attn.wk_dv", .{layer}, 1024, 2304),
            .wv = try affine4_file.loadBf16Dense(c, "model.llm.layers.{d}.attn.wv_dv", .{layer}, 1024, 2304),
            .wr = try affine4_file.loadBf16Dense(c, "model.llm.layers.{d}.attn.wr_du", .{layer}, 384, 2304),
            .wo = try affine4_file.loadBf16Dense(c, "model.llm.layers.{d}.attn.wo_ud", .{layer}, 2304, q_width),
            .gate = try affine4_file.loadBf16Dense(c, "model.llm.layers.{d}.attn.gate_attn", .{layer}, q_width, 2304),
            .sink = try affine4_file.loadBf16Dense(c, "model.llm.layers.{d}.attn.attn_sink_proj", .{layer}, 24, 2304),
            .q_scale = try get(c, "model.llm.layers.{d}.attn.q_l2_clip.scale", .{layer}, .bf16, &.{ 24, 128 }),
            .rel_scale = try get(c, "model.llm.layers.{d}.attn.rel_l2_clip.scale", .{layer}, .bf16, &.{ 24, 16 }),
            .rel_projection = try get(c, "model.llm.layers.{d}.attn.rel_logits_proj.proj", .{layer}, .bf16, &.{ 16, p.extent }),
            .k_conv = try loadConv(c, layer, "k", global),
            .v_conv = try loadConv(c, layer, "v", global),
            .q_norm = if (global) try get(c, "model.llm.layers.{d}.q_input_norm.weight", .{layer}, .bf16, &.{2304}) else local_norm,
            .k_norm = if (global) try get(c, "model.llm.layers.{d}.k_input_norm.weight", .{layer}, .bf16, &.{2304}) else local_norm,
            .v_norm = if (global) try get(c, "model.llm.layers.{d}.v_input_norm.weight", .{layer}, .bf16, &.{2304}) else local_norm,
        };
        l.attn_conv = try get(c, "model.llm.layers.{d}.attn_sconv.weight", .{layer}, .bf16, &.{ 2304, 1, 4 });
        l.mlp_norm = try get(c, "model.llm.layers.{d}.mlp_norm.weight", .{layer}, .bf16, &.{2304});
        l.mlp_conv = try get(c, "model.llm.layers.{d}.mlp_sconv.weight", .{layer}, .bf16, &.{ 2304, 1, 4 });
        l.router_weight = try get(c, "model.llm.layers.{d}.mlp.gate.weight", .{layer}, .bf16, &.{ 128, 2304 });
        l.router_bias = try get(c, "model.llm.layers.{d}.mlp.gate.bias", .{layer}, .f32, &.{128});
        if (p.precision == .tfq3) {
            l.gu = undefined;
            l.down = undefined;
            l.tfq3 = try ink2_tfq3.loadLayer(
                m.gpa,
                m.device,
                &m.tfq3_bufs,
                &m.tfq3_maps,
                &m.tfq3_bytes,
                dir,
                .{
                    .layer = layer,
                    .experts = config.text.experts,
                    .hidden = config.text.hidden,
                    .inter = config.text.expert_inter,
                },
                m.tfq3_tables,
                m.tfq4_tables,
            );
        } else {
            l.gu = try matrix(c, layer, "w13", p.precision, 2304, 2304);
            l.down = try matrix(c, layer, "w2", p.precision, 2304, 1152);
            l.tfq3 = null;
        }
        l.controllers = null;
        if (config.text.loop) |lp| if (layer >= lp.core_start and layer < lp.core_end) {
            var controllers: [4]Controller = undefined;
            inline for (.{ "attn", "k", "v", "mlp" }, 0..) |role, j| controllers[j] = .{
                .down = try get(c, "model.llm.layers.{d}.controllers." ++ role ++ ".down_weight", .{layer}, .bf16, &.{ 64, 4608 }),
                .forget = try get(c, "model.llm.layers.{d}.controllers." ++ role ++ ".forget_weight", .{layer}, .bf16, &.{ 2304, 64 }),
                .gain = try get(c, "model.llm.layers.{d}.controllers." ++ role ++ ".gain_weight", .{layer}, .bf16, &.{ 2304, 64 }),
            };
            l.controllers = controllers;
        };
    }
    return w;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ck.Checkpoint,
    weights: Weights,
    weight_plan: layer_plan.Plan,
    attention: attention.Kernels,
    glue: glue.Pipelines,
    experts: experts.Pipelines,
    loop: loop.Pipelines,
    tree: tree.Pipelines,
    dense: [6]mtl.Pipeline,
    norm: mtl.Pipeline,
    /// A prompt's projections on simdgroup matrices, where the GPU has no tensor units.
    dense_mma: ?dense_file.Mma,
    /// A prompt's register-tiled bf16 projections, where the GPU has tensor units (M5+).
    dense_tiled: ?dense_file.Tiled,
    /// MLX 4-bit affine g64 pipelines; null on official BF16 checkpoints (zero startup cost).
    affine4: ?affine4_file.Pipelines = null,
    runtime: [runtime_names.len]mtl.Pipeline,
    sample: sample_file.SamplePipelines,
    tfq3_tables: ?mtl.Buffer = null,
    tfq4_tables: ?ink2_tfq3.LoadedTables4 = null,
    tfq3_maps: std.ArrayList(ink2_tfq3.MappedShard) = .empty,
    tfq3_bufs: std.ArrayList(mtl.Buffer) = .empty,
    tfq3_pipes: ?ink2_tfq3.Pipelines = null,
    tfq3_bytes: usize = 0,

    pub fn residentBytes(m: *const Model) usize {
        const xrot: usize = if (m.tfq3_pipes != null) ink2_tfq3.xrot_bytes else 0;
        return m.checkpoint.residentBytes() + m.tfq3_bytes + xrot;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        const config_path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(config_path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, config_path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        var why: cfg.Why = .{};
        m.config = cfg.parse(gpa, text, &why) catch |err| {
            std.log.err("inkling2_loop: {s}", .{why.text()});
            return err;
        };
        try admitConfig(&m.config);
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        m.tfq3_tables = null;
        m.tfq4_tables = null;
        m.tfq3_maps = .empty;
        m.tfq3_bufs = .empty;
        m.tfq3_pipes = null;
        m.tfq3_bytes = 0;
        errdefer {
            for (m.tfq3_bufs.items) |b| b.deinit();
            m.tfq3_bufs.deinit(m.gpa);
            for (m.tfq3_maps.items) |map| map.deinit();
            m.tfq3_maps.deinit(m.gpa);
        }
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);

        // Mini uses TFQ3/TFQ4 when tables + layer_NN.tffpw exist, else NVFP4 for layers 0-19 and MXFP8 for 20-23.
        var precisions: [24]layer_plan.Precision = undefined;
        var has_tfq3 = false;
        const has_tables = ink2_tfq3.hasTables(dir);
        const has_tables4 = ink2_tfq3.hasTables4(dir);
        for (&precisions, 0..) |*p, i| {
            if (ink2_tfq3.hasLayer(dir, i)) {
                if (!has_tables and !has_tables4) return error.MissingTfq3Tables;
                p.* = .tfq3;
                has_tfq3 = true;
            } else {
                p.* = if (i < 20) .nvfp4 else .mxfp8;
            }
        }
        m.weight_plan = try layer_plan.build(&m.config.text, &precisions);
        if (m.weight_plan.n_visits != 48) return error.UnsupportedInkling2Loop;
        if (m.config.quant == .mlx_affine4) {
            for (files) |path| try m.checkpoint.addFileSplitPrefixes(m.device, path, "", &.{ "model.language_model.", "lm_head." }, null);
            m.weights = try model_affine4.loadWeights(&m.checkpoint, m.device, &m.config, &m.weight_plan);
        } else {
            for (files) |path| try m.checkpoint.addFileSelected(m.device, path, "", "model.llm.");
            if (has_tfq3) {
                if (has_tables) {
                    m.tfq3_tables = try ink2_tfq3.loadTables(gpa, m.device, &m.tfq3_bufs, &m.tfq3_maps, &m.tfq3_bytes, dir);
                }
                m.tfq4_tables = try ink2_tfq3.loadTables4Optional(gpa, m.device, &m.tfq3_bufs, &m.tfq3_maps, &m.tfq3_bytes, dir);
            }
            m.weights = try loadWeights(m, dir);
        }
        try initPipelines(m);
        return m;
    }

    pub fn initPipelines(m: *Model) !void {
        m.attention = try attention.Kernels.init(m.gpa, m.device);
        errdefer m.attention.deinit();
        m.glue = try glue.Pipelines.load(m.device);
        errdefer m.glue.deinit();
        m.experts = try experts.Pipelines.load(m.device);
        errdefer m.experts.deinit();
        m.loop = try loop.Pipelines.load(m.device);
        errdefer m.loop.deinit();
        m.tree = try tree.Pipelines.load(m.device);
        errdefer m.tree.deinit();
        {
            const lib = try mtl.Library.fromSource(m.device, sources.spark25_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.dense[0..made]) |p| p.deinit();
            for ([_]u32{ 1, 2, 4, 8, 16, 32 }, 0..) |rows, i| {
                var buf: [32]u8 = undefined;
                const name = try std.fmt.bufPrintSentinel(&buf, "sp_matvec_{d}", .{rows}, 0);
                m.dense[i] = try mtl.Pipeline.init(m.device, lib, name, false);
                made += 1;
                if (m.dense[i].maxThreads() < 256) return error.Inkling2Threads;
            }
            const sp_norm = try mtl.Pipeline.init(m.device, lib, "sp_norm", false);
            errdefer sp_norm.deinit();
            if (sp_norm.maxThreads() < 256) return error.Inkling2Threads;
            m.norm = sp_norm;
        }
        errdefer {
            m.norm.deinit();
            for (m.dense) |p| p.deinit();
        }
        m.dense_mma = try dense_file.Mma.load(m.device);
        errdefer if (m.dense_mma) |mma| mma.deinit();
        m.dense_tiled = try dense_file.Tiled.load(m.device);
        errdefer if (m.dense_tiled) |tiled| tiled.deinit();
        m.affine4 = if (m.config.quant == .mlx_affine4) try affine4_file.Pipelines.load(m.device) else null;
        errdefer if (m.affine4) |p| p.deinit();
        {
            const lib = try mtl.Library.fromSource(m.device, runtime_source, mtl.CompileOptions.mlx());
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.runtime[0..made]) |p| p.deinit();
            for (runtime_names, 0..) |name, i| {
                m.runtime[i] = try mtl.Pipeline.init(m.device, lib, name, false);
                made += 1;
                if (m.runtime[i].maxThreads() < 256) return error.Inkling2Threads;
            }
        }
        errdefer for (m.runtime) |p| p.deinit();
        m.sample = try sample_file.SamplePipelines.load(m.device);
        errdefer m.sample.deinit();
        m.tfq3_pipes = null;
        var has_tfq3 = false;
        for (m.weight_plan.layers[0..m.weight_plan.n]) |l| {
            if (l.precision == .tfq3) {
                has_tfq3 = true;
                break;
            }
        }
        if (has_tfq3) m.tfq3_pipes = try ink2_tfq3.Pipelines.load(m.device);
    }

    pub fn deinit(m: *Model) void {
        if (m.tfq3_pipes) |*p| p.deinit();
        m.sample.deinit();
        for (m.runtime) |p| p.deinit();
        if (m.affine4) |p| p.deinit();
        m.norm.deinit();
        for (m.dense) |p| p.deinit();
        if (m.dense_tiled) |tiled| tiled.deinit();
        if (m.dense_mma) |mma| mma.deinit();
        m.tree.deinit();
        m.loop.deinit();
        m.experts.deinit();
        m.glue.deinit();
        m.attention.deinit();
        m.checkpoint.deinit();
        for (m.tfq3_bufs.items) |b| b.deinit();
        m.tfq3_bufs.deinit(m.gpa);
        for (m.tfq3_maps.items) |map| map.deinit();
        m.tfq3_maps.deinit(m.gpa);
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
