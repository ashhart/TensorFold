//! GLM-5.3-Flash's pipelines, compiled at run time with MLX's custom-kernel options: the Python family's kernels
//! (generated, one library each as MLX builds them), our glue, and the MLX replicas for softmax and embedding.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");

pub const max_rows = 16;

pub const Kernels = struct {
    qmv_kda_in: mtl.Pipeline,
    qmv_kda_out: mtl.Pipeline, // also the MTP's eh_proj (K 8192, N 4096)
    qmv_x: mtl.Pipeline,
    qmv_qr: mtl.Pipeline,
    qmv_mla_out: mtl.Pipeline,
    qmv_dense_gu: mtl.Pipeline,
    qmv_dense_down: mtl.Pipeline,
    qmv_head: mtl.Pipeline,
    gemv_t_igate: mtl.Pipeline,
    gemv_t_values: mtl.Pipeline,
    gemv_scores_lt4: mtl.Pipeline,
    gemv_scores_le32: mtl.Pipeline,
    gemv_scores: mtl.Pipeline,
    hc_expand_11: mtl.Pipeline,
    hc_expand_01: mtl.Pipeline,
    hc_expand_10: mtl.Pipeline,
    hc_mix: mtl.Pipeline,
    hc_split_norm: mtl.Pipeline,
    kda_rows: mtl.Pipeline,
    router: [max_rows]mtl.Pipeline, // by window rows (RR = 1 .. 16)
    moe_route: mtl.Pipeline,
    moe_gateup_1: mtl.Pipeline,
    moe_down_1: mtl.Pipeline,
    moe_gateup_2: mtl.Pipeline,
    moe_down_2: mtl.Pipeline,
    moe_combine: mtl.Pipeline,
    sparse_attention: mtl.Pipeline,
    cast_f32: mtl.Pipeline,
    rms: mtl.Pipeline,
    layer_norm: mtl.Pipeline,
    absorb: mtl.Pipeline,
    unabsorb: mtl.Pipeline,
    scale: mtl.Pipeline,
    pool: mtl.Pipeline,
    stream_mean: mtl.Pipeline,
    add: mtl.Pipeline,
    swiglu: mtl.Pipeline,
    argmax: mtl.Pipeline,
    index_scores: mtl.Pipeline,
    index_select: mtl.Pipeline,
    exp_f32: mtl.Pipeline,
    streams: mtl.Pipeline,
    copy_u32: mtl.Pipeline,
    route_rows: mtl.Pipeline,
    act2: mtl.Pipeline,
    dense_indices: mtl.Pipeline,
    softmax: mtl.Pipeline,
    embed: mtl.Pipeline,
    latent_scores: mtl.Pipeline,
    latent_values: mtl.Pipeline,

    pub fn deinit(k: *Kernels) void {
        const info = @typeInfo(Kernels).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == mtl.Pipeline) @field(k, name).deinit() else for (&@field(k, name)) |*p| p.deinit();
        }
    }
};

/// The generated kernel `key` (tools/zig/gen_glm_kernels.py) and the field its first function fills.
const generated = [_]struct { key: []const u8, field: []const u8 }{
    .{ .key = "qmv_4096_24896", .field = "qmv_kda_in" },
    .{ .key = "qmv_8192_4096", .field = "qmv_kda_out" },
    .{ .key = "qmv_4096_2208", .field = "qmv_x" },
    .{ .key = "qmv_1536_20480", .field = "qmv_qr" },
    .{ .key = "qmv_16384_4096", .field = "qmv_mla_out" },
    .{ .key = "qmv_4096_24576", .field = "qmv_dense_gu" },
    .{ .key = "qmv_12288_4096", .field = "qmv_dense_down" },
    .{ .key = "qmv_4096_154880", .field = "qmv_head" },
    .{ .key = "gemv_t_igate", .field = "gemv_t_igate" },
    .{ .key = "gemv_t_values", .field = "gemv_t_values" },
    .{ .key = "gemv_scores_lt4", .field = "gemv_scores_lt4" },
    .{ .key = "gemv_scores_le32", .field = "gemv_scores_le32" },
    .{ .key = "gemv_scores", .field = "gemv_scores" },
    .{ .key = "hc_expand_11", .field = "hc_expand_11" },
    .{ .key = "hc_expand_01", .field = "hc_expand_01" },
    .{ .key = "hc_expand_10", .field = "hc_expand_10" },
    .{ .key = "hc_mix", .field = "hc_mix" },
    .{ .key = "hc_split_norm", .field = "hc_split_norm" },
    .{ .key = "kda_rows", .field = "kda_rows" },
    .{ .key = "router", .field = "router" },
    .{ .key = "moe_route", .field = "moe_route" },
    .{ .key = "moe_gateup_1", .field = "moe_gateup_1" },
    .{ .key = "moe_down_1", .field = "moe_down_1" },
    .{ .key = "moe_gateup_2", .field = "moe_gateup_2" },
    .{ .key = "moe_down_2", .field = "moe_down_2" },
    .{ .key = "moe_combine", .field = "moe_combine" },
    .{ .key = "sparse_attention", .field = "sparse_attention" },
};

const glue = [_]struct { name: [:0]const u8, field: []const u8 }{
    .{ .name = "glm_cast_f32", .field = "cast_f32" },         .{ .name = "glm_rms", .field = "rms" },
    .{ .name = "glm_layer_norm", .field = "layer_norm" },     .{ .name = "glm_absorb", .field = "absorb" },
    .{ .name = "glm_unabsorb", .field = "unabsorb" },         .{ .name = "glm_scale", .field = "scale" },
    .{ .name = "glm_pool", .field = "pool" },                 .{ .name = "glm_stream_mean", .field = "stream_mean" },
    .{ .name = "glm_add", .field = "add" },                   .{ .name = "glm_swiglu", .field = "swiglu" },
    .{ .name = "glm_argmax", .field = "argmax" },             .{ .name = "glm_index_scores", .field = "index_scores" },
    .{ .name = "glm_index_select", .field = "index_select" }, .{ .name = "glm_exp_f32", .field = "exp_f32" },
    .{ .name = "glm_streams", .field = "streams" },         .{ .name = "glm_copy_u32", .field = "copy_u32" },
    .{ .name = "glm_route_rows", .field = "route_rows" },   .{ .name = "glm_act2", .field = "act2" },
    .{ .name = "glm_dense_indices", .field = "dense_indices" },
};

const Job = struct {
    device: mtl.Device,
    source: []const u8,
    names: []const [:0]const u8,
    out: []mtl.Pipeline,
    failed: bool = false,

    fn run(job: *Job) void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = mtl.Library.fromSource(job.device, job.source, mtl.CompileOptions.mlx()) catch {
            job.failed = true;
            return;
        };
        defer lib.deinit();
        for (job.names, job.out) |name, *p| {
            p.* = mtl.Pipeline.init(job.device, lib, name, false) catch {
                job.failed = true;
                return;
            };
        }
    }
};

fn kernelOf(comptime key: []const u8) sources.glm.Kernel {
    for (sources.glm.all) |k| if (comptime std.mem.eql(u8, k.key, key)) return k;
    @compileError("no generated GLM kernel " ++ key);
}

/// Compile every library on worker threads.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !*Kernels {
    const k = try gpa.create(Kernels);
    errdefer gpa.destroy(k);
    var jobs: [generated.len + 4]Job = undefined;
    inline for (generated, 0..) |g, i| {
        const src = comptime kernelOf(g.key);
        const FT = @FieldType(Kernels, g.field);
        const single = FT == mtl.Pipeline;
        if (src.functions.len != (if (single) 1 else @typeInfo(FT).array.len)) @compileError("GLM kernel " ++ g.key ++ ": functions and pipelines differ");
        const out: []mtl.Pipeline = if (single) @as(*[1]mtl.Pipeline, &@field(k, g.field)) else &@field(k, g.field);
        jobs[i] = .{ .device = device, .source = src.source, .names = src.functions, .out = out };
    }
    var glue_out: [glue.len]mtl.Pipeline = undefined;
    const glue_names = comptime blk: {
        var n: [glue.len][:0]const u8 = undefined;
        for (glue, 0..) |g, i| n[i] = g.name;
        break :blk n;
    };
    jobs[generated.len] = .{ .device = device, .source = sources.glm_glue, .names = &glue_names, .out = &glue_out };
    jobs[generated.len + 1] = .{ .device = device, .source = sources.ops_softmax, .names = &.{"tf_softmax_bf16"}, .out = @as(*[1]mtl.Pipeline, &k.softmax) };
    jobs[generated.len + 2] = .{ .device = device, .source = sources.ops_embed_norm, .names = &.{"tf_embed_b4_g64"}, .out = @as(*[1]mtl.Pipeline, &k.embed) };
    const attn_src = try frags.source(device, gpa, sources.glm_attn);
    defer gpa.free(attn_src);
    var attn_out: [2]mtl.Pipeline = undefined;
    jobs[generated.len + 3] = .{ .device = device, .source = attn_src, .names = &.{ "glm_latent_scores", "glm_latent_values" }, .out = &attn_out };
    var next = std.atomic.Value(usize).init(0);
    const Worker = struct {
        fn run(all: []Job, counter: *std.atomic.Value(usize)) void {
            while (true) {
                const i = counter.fetchAdd(1, .monotonic);
                if (i >= all.len) return;
                all[i].run();
            }
        }
    };
    var threads: [8]?std.Thread = @splat(null);
    for (&threads) |*t| t.* = std.Thread.spawn(.{}, Worker.run, .{ &jobs, &next }) catch null;
    Worker.run(&jobs, &next);
    for (threads) |t| if (t) |th| th.join();
    for (jobs) |j| if (j.failed) return error.KernelCompile;
    inline for (glue, 0..) |g, i| @field(k, g.field) = glue_out[i];
    k.latent_scores = attn_out[0];
    k.latent_values = attn_out[1];
    return k;
}
