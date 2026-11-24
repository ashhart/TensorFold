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

pub const Tensor = ck.Tensor;
pub const DynamicConv = struct { static_w: Tensor, projection: Tensor };
pub const Conv = union(enum) { static: Tensor, dynamic: DynamicConv };
pub const Controller = struct { down: Tensor, forget: Tensor, gain: Tensor };
pub const AttentionWeights = struct {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wr: Tensor,
    wo: Tensor,
    gate: Tensor,
    sink: Tensor,
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
    controllers: ?[4]Controller,
};
pub const Weights = struct {
    embed: Tensor,
    embed_norm: Tensor,
    norm: Tensor,
    unembed: Tensor,
    output_query: Tensor,
    cross_queries: Tensor,
    cross_kv_queries: Tensor,
    attn_queries: [23]Tensor,
    mlp_queries: [24]Tensor,
    layers: [24]LayerWeights,
};

/// Scalar copies and residuals around dispatch stay on Metal; forward never reads tensors back to the host.
pub const runtime_source = sources.inkling2_runtime;

pub const RuntimeOp = enum { copy_bf16, copy_f32, widen, attention_partial, mlp_partial, capture_tap };
pub const runtime_names = [_][:0]const u8{
    "tf_ink2_copy_bf16",
    "tf_ink2_copy_f32",
    "tf_ink2_widen",
    "tf_ink2_attention_partial",
    "tf_ink2_mlp_partial",
    "tf_ink2_capture_tap",
};

pub fn admitConfig(c: *const cfg.Config) !void {
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
    }
}

fn loadConv(c: *const ck.Checkpoint, layer: u32, comptime role: []const u8, global: bool) !Conv {
    if (!global) return .{ .static = try get(c, "model.llm.layers.{d}.attn." ++ role ++ "_sconv.weight", .{layer}, .bf16, &.{ 1024, 1, 4 }) };
    return .{ .dynamic = .{
        .static_w = try get(c, "model.llm.layers.{d}.attn." ++ role ++ "_sconv.static_w", .{layer}, .bf16, &.{ 4, 1024 }),
        .projection = try get(c, "model.llm.layers.{d}.attn." ++ role ++ "_sconv.weight_proj", .{layer}, .bf16, &.{ 32, 2304 }),
    } };
}

fn loadWeights(c: *const ck.Checkpoint, config: *const cfg.Config, wp: *const layer_plan.Plan) !Weights {
    var w: Weights = undefined;
    w.embed = try get(c, "model.llm.embed.weight", .{}, .bf16, &.{ 201024, 2304 });
    w.embed_norm = try get(c, "model.llm.embed_norm.weight", .{}, .bf16, &.{2304});
    w.norm = try get(c, "model.llm.norm.weight", .{}, .bf16, &.{2304});
    w.unembed = try get(c, "model.llm.unembed.weight", .{}, .bf16, &.{ 201024, 2304 });
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
        const q_width: usize = @as(usize, p.heads.q) * p.heads.dim;
        const local_norm = if (!global) try get(c, "model.llm.layers.{d}.attn_norm.weight", .{layer}, .bf16, &.{2304}) else undefined;
        l.attn = .{
            .wq = try get(c, "model.llm.layers.{d}.attn.wq_du.weight", .{layer}, .bf16, &.{ q_width, 2304 }),
            .wk = try get(c, "model.llm.layers.{d}.attn.wk_dv.weight", .{layer}, .bf16, &.{ 1024, 2304 }),
            .wv = try get(c, "model.llm.layers.{d}.attn.wv_dv.weight", .{layer}, .bf16, &.{ 1024, 2304 }),
            .wr = try get(c, "model.llm.layers.{d}.attn.wr_du.weight", .{layer}, .bf16, &.{ 384, 2304 }),
            .wo = try get(c, "model.llm.layers.{d}.attn.wo_ud.weight", .{layer}, .bf16, &.{ 2304, q_width }),
            .gate = try get(c, "model.llm.layers.{d}.attn.gate_attn.weight", .{layer}, .bf16, &.{ q_width, 2304 }),
            .sink = try get(c, "model.llm.layers.{d}.attn.attn_sink_proj.weight", .{layer}, .bf16, &.{ 24, 2304 }),
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
        l.gu = try matrix(c, layer, "w13", p.precision, 2304, 2304);
        l.down = try matrix(c, layer, "w2", p.precision, 2304, 1152);
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
    dense: [6]mtl.Pipeline,
    norm: mtl.Pipeline,
    runtime: [runtime_names.len]mtl.Pipeline,

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
        const files = try shards.shardFiles(gpa, io, dir);
        defer shards.freeShardFiles(gpa, files);

        // Mini uses NVFP4 for layers 0-19 and MXFP8 for 20-23; matrix() rechecks all stored tensor metadata and bytes.
        var precisions: [24]layer_plan.Precision = undefined;
        for (&precisions, 0..) |*p, i| p.* = if (i < 20) .nvfp4 else .mxfp8;
        m.weight_plan = try layer_plan.build(&m.config.text, &precisions);
        if (m.weight_plan.n_visits != 48) return error.UnsupportedInkling2Loop;
        for (files) |path| try m.checkpoint.addFileSelected(m.device, path, "", "model.llm.");
        m.weights = try loadWeights(&m.checkpoint, &m.config, &m.weight_plan);

        m.attention = try attention.Kernels.init(gpa, m.device);
        errdefer m.attention.deinit();
        m.glue = try glue.Pipelines.load(m.device);
        errdefer m.glue.deinit();
        m.experts = try experts.Pipelines.load(m.device);
        errdefer m.experts.deinit();
        m.loop = try loop.Pipelines.load(m.device);
        errdefer m.loop.deinit();
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
            }
            m.norm = try mtl.Pipeline.init(m.device, lib, "sp_norm", false);
        }
        errdefer {
            m.norm.deinit();
            for (m.dense) |p| p.deinit();
        }
        {
            const lib = try mtl.Library.fromSource(m.device, runtime_source, mtl.CompileOptions.mlx());
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (m.runtime[0..made]) |p| p.deinit();
            for (runtime_names, 0..) |name, i| {
                m.runtime[i] = try mtl.Pipeline.init(m.device, lib, name, false);
                made += 1;
            }
        }
        errdefer for (m.runtime) |p| p.deinit();
        return m;
    }

    pub fn deinit(m: *Model) void {
        for (m.runtime) |p| p.deinit();
        m.norm.deinit();
        for (m.dense) |p| p.deinit();
        m.loop.deinit();
        m.experts.deinit();
        m.glue.deinit();
        m.attention.deinit();
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
