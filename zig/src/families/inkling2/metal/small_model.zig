//! Inkling 2 Small EXL3 resident text model: dense BF16 tensors, borrowed EXL3 expert descriptors and Metal pipelines.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const st = @import("core").safetensors;
const exl3 = @import("core").exl3_format;
const exl3_proj = @import("core").exl3_projection;
const ck = @import("../../../core/checkpoint_metal.zig");
const cfg = @import("../config.zig");
const exl3_cfg = @import("../exl3_config.zig");
const exl3_pack = @import("../exl3_pack.zig");
const layer_plan = @import("../plan.zig");
const attention = @import("attention.zig");
const glue = @import("glue.zig");
const dense_file = @import("dense.zig");
const small_ops = @import("small_ops.zig");
const small_state = @import("small_state.zig");

pub const Tensor = ck.Tensor;
pub const DynamicConv = struct { static_w: Tensor, projection: Tensor };
pub const Conv = union(enum) { static: Tensor, dynamic: DynamicConv };

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

pub const ExpertTables = struct {
    w13_descs: mtl.Buffer,
    w2_descs: mtl.Buffer,
    shard_ids: []mtl.objc.Id,
    count: usize,
};

pub const LayerWeights = struct {
    attn: AttentionWeights,
    attn_conv: Tensor,
    mlp_norm: Tensor,
    mlp_conv: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    experts: ExpertTables,
};

pub const Weights = struct {
    embed: Tensor,
    embed_norm: Tensor,
    norm: Tensor,
    unembed: Tensor,
    output_query: Tensor,
    attn_queries: []Tensor,
    mlp_queries: []Tensor,
    layers: []LayerWeights,
};

pub const RuntimeOp = enum { copy_bf16, copy_f32, widen, mlp_partial };
pub const runtime_names = [_][:0]const u8{
    "tf_ink2_copy_bf16",
    "tf_ink2_copy_f32",
    "tf_ink2_widen",
    "tf_ink2_mlp_partial",
};

pub const Pipelines = struct {
    attention: attention.Kernels,
    glue: glue.Pipelines,
    small: small_ops.Pipelines,
    exl3: exl3_proj.Pipelines,
    dense: [6]mtl.Pipeline,
    norm: mtl.Pipeline,
    dense_mma: ?dense_file.Mma,
    runtime: [runtime_names.len]mtl.Pipeline,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device) !Pipelines {
        var p: Pipelines = undefined;
        p.attention = try attention.Kernels.init(gpa, device);
        errdefer p.attention.deinit();
        p.glue = try glue.Pipelines.load(device);
        errdefer p.glue.deinit();
        p.small = try small_ops.Pipelines.load(device);
        errdefer p.small.deinit();
        p.exl3 = try exl3_proj.Pipelines.load(device);
        errdefer p.exl3.deinit();
        {
            const lib = try mtl.Library.fromSource(device, sources.spark25_glue, .{
                .language = mtl.CompileOptions.mlx().language,
                .math_functions = .precise,
            });
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (p.dense[0..made]) |pipe| pipe.deinit();
            for ([_]u32{ 1, 2, 4, 8, 16, 32 }, 0..) |rows, i| {
                var buf: [32]u8 = undefined;
                const name = try std.fmt.bufPrintSentinel(&buf, "sp_matvec_{d}", .{rows}, 0);
                p.dense[i] = try mtl.Pipeline.init(device, lib, name, false);
                made += 1;
                if (p.dense[i].maxThreads() < 256) return error.Inkling2Threads;
            }
            const sp_norm = try mtl.Pipeline.init(device, lib, "sp_norm", false);
            errdefer sp_norm.deinit();
            if (sp_norm.maxThreads() < 256) return error.Inkling2Threads;
            p.norm = sp_norm;
        }
        errdefer {
            p.norm.deinit();
            for (p.dense) |pipe| pipe.deinit();
        }
        p.dense_mma = try dense_file.Mma.load(device);
        errdefer if (p.dense_mma) |mma| mma.deinit();
        {
            const lib = try mtl.Library.fromSource(device, sources.inkling2_runtime, mtl.CompileOptions.mlx());
            defer lib.deinit();
            var made: usize = 0;
            errdefer for (p.runtime[0..made]) |pipe| pipe.deinit();
            for (runtime_names, 0..) |name, i| {
                p.runtime[i] = try mtl.Pipeline.init(device, lib, name, false);
                made += 1;
                if (p.runtime[i].maxThreads() < 256) return error.Inkling2Threads;
            }
        }
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.runtime) |pipe| pipe.deinit();
        if (p.dense_mma) |mma| mma.deinit();
        p.norm.deinit();
        for (p.dense) |pipe| pipe.deinit();
        p.exl3.deinit();
        p.small.deinit();
        p.glue.deinit();
        p.attention.deinit();
    }
};

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

fn loadConv(c: *const ck.Checkpoint, layer: u32, comptime role: []const u8, global: bool) !Conv {
    if (!global) return .{ .static = try get(c, "model.llm.layers.{d}.attn." ++ role ++ "_sconv.weight", .{layer}, .bf16, &.{ small_state.kv_width, 1, 4 }) };
    return .{ .dynamic = .{
        .static_w = try get(c, "model.llm.layers.{d}.attn." ++ role ++ "_sconv.static_w", .{layer}, .bf16, &.{ 4, small_state.kv_width }),
        .projection = try get(c, "model.llm.layers.{d}.attn." ++ role ++ "_sconv.weight_proj", .{layer}, .bf16, &.{ 32, small_state.hidden }),
    } };
}

fn readAll(fd: platform.c.fd_t, dest: []u8, at: usize) !void {
    const max_pread = 1 << 30;
    var done: usize = 0;
    while (done < dest.len) {
        const n = platform.c.pread(fd, dest.ptr + done, @min(dest.len - done, max_pread), @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

fn buildProjectionDesc(
    hdr: st.Header,
    buf: mtl.Buffer,
    expert: u32,
    comptime proj: []const u8,
    k: usize,
    n: usize,
    want_bits: ?u8,
) !exl3_proj.ExpertDescriptor {
    var nb: [64]u8 = undefined;
    const t_name = std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".trellis", .{expert}) catch unreachable;
    const te = hdr.get(t_name) orelse return error.MissingTensor;
    if (te.dtype != .i16 or te.rank != 3 or te.shape[0] != k / 16 or te.shape[1] != n / 16 or te.shape[2] == 0 or te.shape[2] % 16 != 0)
        return error.WrongTensor;
    const bits_usize = te.shape[2] / 16;
    if (bits_usize < exl3_pack.min_bits or bits_usize > exl3_pack.max_bits) return error.UnsupportedWidth;
    const bits: u8 = @intCast(bits_usize);
    if (want_bits) |wb| if (wb != bits) return error.WidthMismatch;

    const su_name = std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".suh", .{expert}) catch unreachable;
    const su = hdr.get(su_name) orelse return error.MissingTensor;
    if (su.dtype != .f16 or su.rank != 1 or su.shape[0] != k) return error.WrongTensor;

    const sv_name = std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".svh", .{expert}) catch unreachable;
    const sv = hdr.get(sv_name) orelse return error.MissingTensor;
    if (sv.dtype != .f16 or sv.rank != 1 or sv.shape[0] != n) return error.WrongTensor;

    const m_name = std.fmt.bufPrint(&nb, "experts.{d}." ++ proj ++ ".mul1", .{expert}) catch unreachable;
    const me = hdr.get(m_name) orelse return error.UnsupportedCodebook;
    if (me.dtype != .i32 or me.rank != 0 or me.end - me.begin != 4) return error.UnsupportedCodebook;
    const marker = std.mem.readInt(u32, buf.contents()[me.begin..][0..4], .little);
    if (marker != exl3.Mul1Mul) return error.UnsupportedCodebook;

    return exl3_proj.descriptor(.{
        .trellis = .{ .buffer = buf, .offset = te.begin },
        .suh = .{ .buffer = buf, .offset = su.begin },
        .svh = .{ .buffer = buf, .offset = sv.begin },
        .k = k,
        .n = n,
        .width = .{ .halves = @as(u16, bits) * 2 },
        .codebook = .mul1,
        .suh_format = .f16,
        .svh_format = .f16,
    });
}

fn loadExpertShard(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    path: [:0]const u8,
    batch: exl3_pack.Batch,
    k_by_expert: ?[]const u8,
    w13_out: []exl3_proj.ExpertDescriptor,
    w2_out: []exl3_proj.ExpertDescriptor,
) !mtl.Buffer {
    const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.OpenFailed;
    defer _ = platform.c.close(fd);
    var head: [8]u8 = undefined;
    try readAll(fd, &head, 0);
    const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (end < 8) return error.BadSafetensors;
    const file_size = std.math.cast(usize, end) orelse return error.BadSafetensors;
    const hlen = try st.headerSize(std.mem.readInt(u64, &head, .little), file_size);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const header_json = try a.alloc(u8, hlen);
    try readAll(fd, header_json, 8);
    const data_start = 8 + hlen;
    const data_len = file_size - data_start;
    const hdr = try st.parseHeader(a, header_json, data_len);
    if (hdr.count() != @as(usize, batch.end - batch.start) * 8) return error.UnexpectedTensor;

    const buf = try device.buffer(@max(data_len, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    try readAll(fd, buf.contents()[0..data_len], data_start);

    for (batch.start..batch.end) |e_usize| {
        const e: u32 = @intCast(e_usize);
        const want = if (k_by_expert) |ks| ks[e] else null;
        w13_out[e] = try buildProjectionDesc(hdr, buf, e, "w13", small_state.hidden, 2 * small_state.intermediate, want);
        w2_out[e] = try buildProjectionDesc(hdr, buf, e, "w2", small_state.intermediate, small_state.hidden, want);
    }
    return buf;
}

fn loadDenseWeights(gpa: std.mem.Allocator, c: *const ck.Checkpoint, config: *const cfg.Config, wp: *const layer_plan.Plan) !Weights {
    const n_layers: usize = config.text.layers;
    const h: usize = config.text.hidden;
    const vocab: usize = config.text.vocab;
    const attn_queries = try gpa.alloc(Tensor, n_layers - 1);
    errdefer gpa.free(attn_queries);
    const mlp_queries = try gpa.alloc(Tensor, n_layers);
    errdefer gpa.free(mlp_queries);
    const layers = try gpa.alloc(LayerWeights, n_layers);
    errdefer gpa.free(layers);

    for (attn_queries, 0..) |*q, i| {
        const global = cfg.isGlobal(@intCast(i + 1));
        const dims: []const usize = if (global) &.{ 3, h } else &.{h};
        q.* = try get(c, "model.llm.attn_res.attn_queries.{d}", .{i}, .f32, dims);
    }
    for (mlp_queries, 0..) |*q, i| q.* = try get(c, "model.llm.attn_res.mlp_queries.{d}", .{i}, .f32, &.{h});

    for (layers, 0..) |*l, i| {
        const layer: u32 = @intCast(i);
        const p = wp.at(layer);
        const global = p.kind == .global;
        const q_width: usize = @as(usize, p.heads.q) * p.heads.dim;
        const local_norm = if (!global) try get(c, "model.llm.layers.{d}.attn_norm.weight", .{layer}, .bf16, &.{h}) else undefined;
        l.attn = .{
            .wq = try get(c, "model.llm.layers.{d}.attn.wq_du.weight", .{layer}, .bf16, &.{ q_width, h }),
            .wk = try get(c, "model.llm.layers.{d}.attn.wk_dv.weight", .{layer}, .bf16, &.{ small_state.kv_width, h }),
            .wv = try get(c, "model.llm.layers.{d}.attn.wv_dv.weight", .{layer}, .bf16, &.{ small_state.kv_width, h }),
            .wr = try get(c, "model.llm.layers.{d}.attn.wr_du.weight", .{layer}, .bf16, &.{ small_state.rel_width, h }),
            .wo = try get(c, "model.llm.layers.{d}.attn.wo_ud.weight", .{layer}, .bf16, &.{ h, q_width }),
            .gate = try get(c, "model.llm.layers.{d}.attn.gate_attn.weight", .{layer}, .bf16, &.{ q_width, h }),
            .sink = try get(c, "model.llm.layers.{d}.attn.attn_sink_proj.weight", .{layer}, .bf16, &.{ p.heads.q, h }),
            .q_scale = try get(c, "model.llm.layers.{d}.attn.q_l2_clip.scale", .{layer}, .bf16, &.{ p.heads.q, p.heads.dim }),
            .rel_scale = try get(c, "model.llm.layers.{d}.attn.rel_l2_clip.scale", .{layer}, .bf16, &.{ p.heads.q, config.text.d_rel }),
            .rel_projection = try get(c, "model.llm.layers.{d}.attn.rel_logits_proj.proj", .{layer}, .bf16, &.{ config.text.d_rel, p.extent }),
            .k_conv = try loadConv(c, layer, "k", global),
            .v_conv = try loadConv(c, layer, "v", global),
            .q_norm = if (global) try get(c, "model.llm.layers.{d}.q_input_norm.weight", .{layer}, .bf16, &.{h}) else local_norm,
            .k_norm = if (global) try get(c, "model.llm.layers.{d}.k_input_norm.weight", .{layer}, .bf16, &.{h}) else local_norm,
            .v_norm = if (global) try get(c, "model.llm.layers.{d}.v_input_norm.weight", .{layer}, .bf16, &.{h}) else local_norm,
        };
        l.attn_conv = try get(c, "model.llm.layers.{d}.attn_sconv.weight", .{layer}, .bf16, &.{ h, 1, 4 });
        l.mlp_norm = try get(c, "model.llm.layers.{d}.mlp_norm.weight", .{layer}, .bf16, &.{h});
        l.mlp_conv = try get(c, "model.llm.layers.{d}.mlp_sconv.weight", .{layer}, .bf16, &.{ h, 1, 4 });
        l.router_weight = try get(c, "model.llm.layers.{d}.mlp.gate.weight", .{layer}, .bf16, &.{ config.text.experts, h });
        l.router_bias = try get(c, "model.llm.layers.{d}.mlp.gate.bias", .{layer}, .f32, &.{config.text.experts});
        l.experts = undefined;
    }

    return .{
        .embed = try get(c, "model.llm.embed.weight", .{}, .bf16, &.{ vocab, h }),
        .embed_norm = try get(c, "model.llm.embed_norm.weight", .{}, .bf16, &.{h}),
        .norm = try get(c, "model.llm.norm.weight", .{}, .bf16, &.{h}),
        .unembed = try get(c, "model.llm.unembed.weight", .{}, .bf16, &.{ vocab, h }),
        .output_query = try get(c, "model.llm.attn_res.output_query", .{}, .f32, &.{h}),
        .attn_queries = attn_queries,
        .mlp_queries = mlp_queries,
        .layers = layers,
    };
}

fn fileExists(io: std.Io, path: []const u8) !bool {
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    f.close(io);
    return true;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ck.Checkpoint,
    weights: Weights,
    weight_plan: layer_plan.Plan,
    expert_buffers: std.ArrayList(mtl.Buffer) = .empty,
    pipes: Pipelines,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.expert_buffers = .empty;

        const single_manifest = try std.fs.path.join(gpa, &.{ dir, "experts", "checkpoint.json" });
        defer gpa.free(single_manifest);
        const split_manifest = try std.fs.path.join(gpa, &.{ dir, "checkpoint.json" });
        defer gpa.free(split_manifest);
        const has_single = try fileExists(io, single_manifest);
        const has_split = try fileExists(io, split_manifest);
        if (has_single == has_split) return error.MissingManifest;
        const layout: exl3_pack.Layout = if (has_single) .single_dense else .split_dense;

        const cfg_path = if (layout == .single_dense)
            try std.fs.path.join(gpa, &.{ dir, "config.json" })
        else
            try std.fs.path.join(gpa, &.{ dir, "auxiliary", "config.json" });
        defer gpa.free(cfg_path);
        const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 24));
        defer gpa.free(cfg_text);
        var why: cfg.Why = .{};
        m.config = cfg.parse(gpa, cfg_text, &why) catch |err| {
            std.log.err("inkling2_small: {s}", .{why.text()});
            return err;
        };
        exl3_pack.admitSmallConfig(&m.config, &why) catch |err| {
            std.log.err("inkling2_small: {s}", .{why.text()});
            return err;
        };
        m.weight_plan = try exl3_cfg.layerPlan(&m.config);

        const manifest_path = if (layout == .single_dense) single_manifest else split_manifest;
        const manifest_json = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, gpa, .limited(64 << 20));
        defer gpa.free(manifest_json);
        var manifest = exl3_pack.parseManifest(gpa, manifest_json, &m.config.text, &why) catch |err| {
            std.log.err("inkling2_small: {s}", .{why.text()});
            return err;
        };
        defer manifest.deinit();

        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ck.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();

        if (layout == .single_dense) {
            const p = try std.fs.path.joinZ(gpa, &.{ dir, "model-dense.safetensors" });
            defer gpa.free(p);
            try m.checkpoint.addFileSelected(m.device, p, "", "model.llm.");
        } else {
            const gpath = try std.fs.path.joinZ(gpa, &.{ dir, "dense", "global.safetensors" });
            defer gpa.free(gpath);
            try m.checkpoint.addFileSelected(m.device, gpath, "", "model.llm.");
            for (0..m.config.text.layers) |li| {
                var lb: [32]u8 = undefined;
                const fname = std.fmt.bufPrint(&lb, "layer-{d:0>2}.safetensors", .{li}) catch unreachable;
                const lpath = try std.fs.path.joinZ(gpa, &.{ dir, "dense", fname });
                defer gpa.free(lpath);
                try m.checkpoint.addFileSelected(m.device, lpath, "", "model.llm.");
            }
        }

        m.weights = try loadDenseWeights(gpa, &m.checkpoint, &m.config, &m.weight_plan);
        errdefer {
            gpa.free(m.weights.attn_queries);
            gpa.free(m.weights.mlp_queries);
            gpa.free(m.weights.layers);
        }

        const expert_root = if (layout == .single_dense)
            try std.fs.path.join(gpa, &.{ dir, "experts" })
        else
            try gpa.dupe(u8, dir);
        defer gpa.free(expert_root);

        var loaded_layers: usize = 0;
        errdefer {
            for (m.weights.layers[0..loaded_layers]) |*lw| {
                lw.experts.w13_descs.deinit();
                lw.experts.w2_descs.deinit();
                gpa.free(lw.experts.shard_ids);
            }
            for (m.expert_buffers.items) |b| b.deinit();
            m.expert_buffers.deinit(gpa);
        }

        const n_experts: usize = m.config.text.experts;
        const desc_bytes = n_experts * @sizeOf(exl3_proj.ExpertDescriptor);
        for (manifest.layers, 0..) |lm, li| {
            const w13_buf = try m.device.buffer(desc_bytes, mtl.ResourceOptions.shared);
            errdefer w13_buf.deinit();
            const w2_buf = try m.device.buffer(desc_bytes, mtl.ResourceOptions.shared);
            errdefer w2_buf.deinit();
            const shard_ids = try gpa.alloc(mtl.objc.Id, lm.batches.len);
            errdefer gpa.free(shard_ids);
            const w13_slice = w13_buf.slice(exl3_proj.ExpertDescriptor, n_experts);
            const w2_slice = w2_buf.slice(exl3_proj.ExpertDescriptor, n_experts);

            for (lm.batches, 0..) |batch, bi| {
                const spath = try std.fs.path.joinZ(gpa, &.{ expert_root, batch.file });
                defer gpa.free(spath);
                const sbuf = try loadExpertShard(gpa, m.device, spath, batch, lm.k_by_expert, w13_slice, w2_slice);
                errdefer sbuf.deinit();
                try m.expert_buffers.append(gpa, sbuf);
                shard_ids[bi] = sbuf.id;
            }
            m.weights.layers[li].experts = .{
                .w13_descs = w13_buf,
                .w2_descs = w2_buf,
                .shard_ids = shard_ids,
                .count = n_experts,
            };
            loaded_layers += 1;
        }

        m.pipes = try Pipelines.load(gpa, m.device);
        return m;
    }

    pub fn deinit(m: *Model) void {
        m.pipes.deinit();
        for (m.weights.layers) |*lw| {
            lw.experts.w13_descs.deinit();
            lw.experts.w2_descs.deinit();
            m.gpa.free(lw.experts.shard_ids);
        }
        for (m.expert_buffers.items) |b| b.deinit();
        m.expert_buffers.deinit(m.gpa);
        m.gpa.free(m.weights.attn_queries);
        m.gpa.free(m.weights.mlp_queries);
        m.gpa.free(m.weights.layers);
        m.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};
