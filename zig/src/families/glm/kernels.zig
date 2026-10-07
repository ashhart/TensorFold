//! GLM-5.3-Flash's pipelines, compiled at run time: the Python family's kernels (generated), our glue and latent attention.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const moe_route = @import("../../core/moe_route.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const hc = @import("../../core/hc.zig");

/// The route's shape (config.zig refuses other checkpoints).
pub const route_shape: moe_route.Shape = .{ .hidden = 4096, .experts = 288, .topk = 8 };
/// The MLA indexer's key gate as the same gemv_t (x [rows, 4096] times the packed [128, 4096] gate, bf16 out).
pub const igate_shape: moe_route.Shape = .{ .hidden = 4096, .experts = 128, .topk = 8, .out_bf16 = true };

/// The hyper-connection boundary's shape (config.zig refuses other checkpoints).
pub const hc_shape: hc.Shape = .{ .width = 4096, .sinkhorn = 20, .eps_e9 = 1000 };

pub const max_rows = 16;

pub const Kernels = struct {
    source_hash: u64, // every compiled source, hashed: part of a learned prompt state's identity
    qmv_kda_in: mtl.Pipeline,
    qmv_kda_out: mtl.Pipeline, // also the MTP's eh_proj (K 8192, N 4096)
    qmv_x: mtl.Pipeline,
    qmv_qr: mtl.Pipeline,
    qmv_mla_out: mtl.Pipeline,
    qmv_dense_gu: mtl.Pipeline,
    qmv_dense_down: mtl.Pipeline,
    qmv_head: mtl.Pipeline,
    qmv_kda_in_tp: mtl.Pipeline, // TP2: the KDA input projection's rows of one Mac's heads
    qmvp_kda_out: mtl.Pipeline, // TP2: the KDA out-projection over one Mac's heads, fp32 partials
    qmv_head_tp: mtl.Pipeline, // TP2: the head's rows of one Mac's half of the vocabulary
    qmv_qr_tp: mtl.Pipeline, // TP2: q_b's rows of one Mac's MLA heads, the indexer's queries whole
    qmvp_mla_out: mtl.Pipeline, // TP2: the MLA out-projection over one Mac's heads, fp32 partials
    sparse_attention_tp: mtl.Pipeline, // TP2: the sparse kernel over one Mac's 32 heads
    qmv_dense_gu_tp: mtl.Pipeline, // TP2: the dense MLP's gate and up rows of one Mac's half
    qmvp_dense_down: mtl.Pipeline, // TP2: its down projection over that half, fp32 partials
    gemv_t_igate: mtl.Pipeline,
    gemv_t_values: mtl.Pipeline,
    gemv_scores_lt4: mtl.Pipeline,
    gemv_scores_le32: mtl.Pipeline,
    gemv_scores: mtl.Pipeline,
    hc_expand_10: mtl.Pipeline,
    hc_core: [3]mtl.Pipeline, // core/hc.zig: the expand (or the first boundary's read) with partial sums, then the split
    kda_rows: mtl.Pipeline,
    kda_rows_tp: mtl.Pipeline, // TP2: the fused step over one Mac's 32 heads
    router: [max_rows]mtl.Pipeline, // by window rows (RR = 1 .. 16)
    moe_route: mtl.Pipeline,
    moe_gateup_1: mtl.Pipeline,
    moe_down_1: mtl.Pipeline,
    moe_gateup_2: mtl.Pipeline,
    moe_down_2: mtl.Pipeline,
    moe_combine: mtl.Pipeline,
    moe_gateup_2h: mtl.Pipeline, // expert parallel by rows: half of every expert's gate/up rows
    moe_down_2h: mtl.Pipeline, // and its down inputs, fp32 partials
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
    route_logits: mtl.Pipeline,
    route_select: mtl.Pipeline,
    igate_logits: mtl.Pipeline,
    kda_pre: mtl.Pipeline, // a prompt chunk's KDA layer in three passes (glm_kda_prompt.metal): gates, conv, norms
    kda_scan: mtl.Pipeline, // the recurrence alone
    kda_post: mtl.Pipeline, // the output norm and gate
    kda_pre_tp: mtl.Pipeline, // TP2: the three passes over one Mac's 32 heads
    kda_scan_tp: mtl.Pipeline,
    kda_post_tp: mtl.Pipeline,
    sparse_nax: mtl.Pipeline, // a prompt chunk's sparse MLA attention on the tensor units (glm_sparse_nax.metal)
    absorb_nax: mtl.Pipeline, // and its absorb (glm_absorb_nax.metal)
    sparse_nax_tp: mtl.Pipeline, // TP2: both over one Mac's 32 heads
    absorb_nax_tp: mtl.Pipeline,
    mm_bf16: affine_mm.Pipes, // prompt chunks' 4-bit g64 matmuls on the tensor units (core/affine_mm.zig): dense, gathers
    mm_f32: affine_mm.Pipes, // and with fp32 out: expert parallel by rows' down partials

    pub fn deinit(k: *Kernels) void {
        const info = @typeInfo(Kernels).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == mtl.Pipeline) @field(k, name).deinit() else if (T != u64) for (&@field(k, name)) |*p| p.deinit();
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
    .{ .key = "qmv_4096_12576", .field = "qmv_kda_in_tp" },
    .{ .key = "qmvp_4096_4096", .field = "qmvp_kda_out" },
    .{ .key = "qmv_4096_77440", .field = "qmv_head_tp" },
    .{ .key = "qmv_1536_12288", .field = "qmv_qr_tp" },
    .{ .key = "qmvp_8192_4096", .field = "qmvp_mla_out" },
    .{ .key = "sparse_attention_tp", .field = "sparse_attention_tp" },
    .{ .key = "qmv_4096_12288", .field = "qmv_dense_gu_tp" },
    .{ .key = "qmvp_6144_4096", .field = "qmvp_dense_down" },
    .{ .key = "gemv_t_igate", .field = "gemv_t_igate" },
    .{ .key = "gemv_t_values", .field = "gemv_t_values" },
    .{ .key = "gemv_scores_lt4", .field = "gemv_scores_lt4" },
    .{ .key = "gemv_scores_le32", .field = "gemv_scores_le32" },
    .{ .key = "gemv_scores", .field = "gemv_scores" },
    .{ .key = "hc_expand_10", .field = "hc_expand_10" },
    .{ .key = "kda_rows", .field = "kda_rows" },
    .{ .key = "kda_rows_tp", .field = "kda_rows_tp" },
    .{ .key = "router", .field = "router" },
    .{ .key = "moe_route", .field = "moe_route" },
    .{ .key = "moe_gateup_1", .field = "moe_gateup_1" },
    .{ .key = "moe_down_1", .field = "moe_down_1" },
    .{ .key = "moe_gateup_2", .field = "moe_gateup_2" },
    .{ .key = "moe_down_2", .field = "moe_down_2" },
    .{ .key = "moe_combine", .field = "moe_combine" },
    .{ .key = "moe_gateup_2h", .field = "moe_gateup_2h" },
    .{ .key = "moe_down_2h", .field = "moe_down_2h" },
    .{ .key = "sparse_attention", .field = "sparse_attention" },
};

const glue = [_]struct { name: [:0]const u8, field: []const u8 }{
    .{ .name = "glm_cast_f32", .field = "cast_f32" },           .{ .name = "glm_rms", .field = "rms" },
    .{ .name = "glm_layer_norm", .field = "layer_norm" },       .{ .name = "glm_absorb", .field = "absorb" },
    .{ .name = "glm_unabsorb", .field = "unabsorb" },           .{ .name = "glm_scale", .field = "scale" },
    .{ .name = "glm_pool", .field = "pool" },                   .{ .name = "glm_stream_mean", .field = "stream_mean" },
    .{ .name = "glm_add", .field = "add" },                     .{ .name = "glm_swiglu", .field = "swiglu" },
    .{ .name = "glm_argmax", .field = "argmax" },               .{ .name = "glm_index_scores", .field = "index_scores" },
    .{ .name = "glm_index_select", .field = "index_select" },   .{ .name = "glm_exp_f32", .field = "exp_f32" },
    .{ .name = "glm_streams", .field = "streams" },             .{ .name = "glm_copy_u32", .field = "copy_u32" },
    .{ .name = "glm_route_rows", .field = "route_rows" },       .{ .name = "glm_act2", .field = "act2" },
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
    @setEvalBranchQuota(20000);
    for (sources.glm.all) |k| if (comptime std.mem.eql(u8, k.key, key)) return k;
    @compileError("no generated GLM kernel " ++ key);
}

/// Compile every library on worker threads.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !*Kernels {
    const k = try gpa.create(Kernels);
    errdefer gpa.destroy(k);
    var jobs: [generated.len + 14]Job = undefined;
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
    const route_src = try moe_route.source(gpa, route_shape);
    defer gpa.free(route_src);
    var route_out: [2]mtl.Pipeline = undefined;
    jobs[generated.len + 4] = .{ .device = device, .source = route_src, .names = &moe_route.names, .out = &route_out };
    const igate_src = try moe_route.source(gpa, igate_shape);
    defer gpa.free(igate_src);
    var igate_out: [1]mtl.Pipeline = undefined;
    jobs[generated.len + 5] = .{ .device = device, .source = igate_src, .names = &.{moe_route.names[0]}, .out = &igate_out };
    const m16_src = try affine_mm.source(device, gpa, .{});
    defer gpa.free(m16_src);
    const m32_src = try affine_mm.source(device, gpa, .{ .out_f32 = true });
    defer gpa.free(m32_src);
    jobs[generated.len + 6] = .{ .device = device, .source = m16_src, .names = &affine_mm.names, .out = &k.mm_bf16 };
    jobs[generated.len + 7] = .{ .device = device, .source = m32_src, .names = &affine_mm.names, .out = &k.mm_f32 };
    const kda_src = try std.mem.concat(gpa, u8, &.{ comptime kernelOf("kda_rows").source, sources.glm_kda_prompt });
    defer gpa.free(kda_src);
    var kda_out: [6]mtl.Pipeline = undefined;
    jobs[generated.len + 8] = .{ .device = device, .source = kda_src, .names = &.{ "glm_kda_pre", "glm_kda_scan", "glm_kda_post", "glm_kda_pre_tp", "glm_kda_scan_tp", "glm_kda_post_tp" }, .out = &kda_out };
    const sparse_src = try frags.source(device, gpa, sources.glm_sparse_nax);
    defer gpa.free(sparse_src);
    jobs[generated.len + 9] = .{ .device = device, .source = sparse_src, .names = &.{"glm_sparse_nax"}, .out = @as(*[1]mtl.Pipeline, &k.sparse_nax) };
    const absorb_src = try frags.source(device, gpa, sources.glm_absorb_nax);
    defer gpa.free(absorb_src);
    jobs[generated.len + 10] = .{ .device = device, .source = absorb_src, .names = &.{"glm_absorb_nax"}, .out = @as(*[1]mtl.Pipeline, &k.absorb_nax) };
    const hc_src = try hc.source(gpa, hc_shape);
    defer gpa.free(hc_src);
    jobs[generated.len + 11] = .{ .device = device, .source = hc_src, .names = &hc.names, .out = &k.hc_core };
    const tp_heads = "#define GLM_HEADS 32\n"; // TP2: one Mac's MLA heads
    const sparse_tp_raw = try std.mem.concat(gpa, u8, &.{ tp_heads, sources.glm_sparse_nax });
    defer gpa.free(sparse_tp_raw);
    const sparse_tp_src = try frags.source(device, gpa, sparse_tp_raw);
    defer gpa.free(sparse_tp_src);
    jobs[generated.len + 12] = .{ .device = device, .source = sparse_tp_src, .names = &.{"glm_sparse_nax"}, .out = @as(*[1]mtl.Pipeline, &k.sparse_nax_tp) };
    const absorb_tp_raw = try std.mem.concat(gpa, u8, &.{ tp_heads, sources.glm_absorb_nax });
    defer gpa.free(absorb_tp_raw);
    const absorb_tp_src = try frags.source(device, gpa, absorb_tp_raw);
    defer gpa.free(absorb_tp_src);
    jobs[generated.len + 13] = .{ .device = device, .source = absorb_tp_src, .names = &.{"glm_absorb_nax"}, .out = @as(*[1]mtl.Pipeline, &k.absorb_nax_tp) };
    var sources_seen = std.hash.Wyhash.init(0x6b);
    for (jobs) |j| sources_seen.update(j.source);
    k.source_hash = sources_seen.final();
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
    k.route_logits = route_out[0];
    k.route_select = route_out[1];
    k.igate_logits = igate_out[0];
    k.kda_pre = kda_out[0];
    k.kda_scan = kda_out[1];
    k.kda_post = kda_out[2];
    k.kda_pre_tp = kda_out[3];
    k.kda_scan_tp = kda_out[4];
    k.kda_post_tp = kda_out[5];
    return k;
}
