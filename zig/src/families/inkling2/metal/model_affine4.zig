//! MLX 4-bit affine group-64 Inkling 2 Mini checkpoint weight loader (`mlx-community/Inkling-2-Mini-4bit`).
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("../config.zig");
const layer_plan = @import("../plan.zig");
const ck = @import("../../../core/checkpoint_metal.zig");
const model = @import("model.zig");
const affine4 = @import("affine4.zig");
pub const pack = @import("affine4_pack.zig");
pub const pinned_pack = pack.pinned_pack;
pub const admitPackFiles = pack.admitPackFiles;
pub const verifyPinnedDigests = pack.verifyPinnedDigests;

const Tensor = ck.Tensor;
const align_bytes = ck.alignment;

const Stage = struct {
    buf: mtl.Buffer,
    used: usize = 0,

    fn alloc(s: *Stage, dtype: ck.DType, dims: []const usize) !Tensor {
        var count: usize = 1;
        for (dims) |d| count = std.math.mul(usize, count, d) catch return error.UnexpectedInkling2Tensor;
        const bytes = std.math.mul(usize, count, dtype.size()) catch return error.UnexpectedInkling2Tensor;
        const off = std.mem.alignForward(usize, s.used, align_bytes);
        if (off + bytes > s.buf.length()) return error.OutOfMemory;
        s.used = off + bytes;
        var t = Tensor{ .buffer = s.buf, .offset = off, .bytes = bytes, .dtype = dtype, .rank = dims.len };
        for (dims, 0..) |d, i| t.shape[i] = d;
        return t;
    }

    fn slice(t: Tensor, comptime T: type) []T {
        const ptr: [*]u8 = t.buffer.contents() + t.offset;
        return @as([*]T, @ptrCast(@alignCast(ptr)))[0 .. t.bytes / @sizeOf(T)];
    }
};

fn rawBytes(t: Tensor) []const u8 {
    return (t.buffer.contents() + t.offset)[0..t.bytes];
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
    return require(try c.get(try std.fmt.bufPrint(&buf, fmt, args)), dtype, dims);
}

fn getBf16OrF32(c: *const ck.Checkpoint, stage: *Stage, comptime fmt: []const u8, args: anytype, dims: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    const t = try c.get(try std.fmt.bufPrint(&buf, fmt, args));
    if (t.dtype == .bf16) return require(t, .bf16, dims);
    const f32_t = try require(t, .f32, dims);
    const out = try stage.alloc(.bf16, dims);
    affine4.convertF32BytesToBf16(rawBytes(f32_t), Stage.slice(out, u16));
    return out;
}

fn loadConv(c: *const ck.Checkpoint, stage: *Stage, layer: u32, comptime role: []const u8, global: bool) !model.Conv {
    if (!global) return .{ .static = try get(c, "model.language_model.layers.{d}.self_attn." ++ role ++ "_sconv.conv1d.weight", .{layer}, .bf16, &.{ 1024, 4, 1 }) };
    const static_w = try get(c, "model.language_model.layers.{d}.self_attn." ++ role ++ "_sconv.static_weight", .{layer}, .bf16, &.{ 4, 1024 });
    var mbuf: [256]u8 = undefined;
    const mod = try std.fmt.bufPrint(&mbuf, "model.language_model.layers.{d}.self_attn." ++ role ++ "_sconv.weight_proj", .{layer});
    var sbuf: [280]u8 = undefined;
    const has_comp = c.has(try std.fmt.bufPrint(&sbuf, "{s}.scales", .{mod})) or
        c.has(try std.fmt.bufPrint(&sbuf, "{s}.biases", .{mod}));
    const proj = if (has_comp) blk: {
        const q = try affine4.readTriple(c, mod, 32, 2304);
        const out = try stage.alloc(.bf16, &.{ 32, 2304 });
        affine4.dequantAffine4ToBf16(rawBytes(q.weight), rawBytes(q.scales), rawBytes(q.biases), 32, 2304, Stage.slice(out, u16));
        break :blk out;
    } else try affine4.readBf16NoCompanions(c, mod, 32, 2304);
    return .{ .dynamic = .{ .static_w = static_w, .projection = proj } };
}

fn loadExpert(c: *const ck.Checkpoint, denom: Tensor, layer: u32, comptime role: []const u8, precision: layer_plan.Precision, rows: usize, k: usize) !model.ExpertMatrix {
    const base = "model.language_model.layers.{d}.mlp.experts." ++ role;
    const e: usize = 128;
    switch (precision) {
        .nvfp4 => return .{
            .weight = try get(c, base ++ ".weight", .{layer}, .u32, &.{ e, rows, k / 8 }),
            .scale = try get(c, base ++ ".scales", .{layer}, .u8, &.{ e, rows, k / 16 }),
            .amax = try get(c, base ++ ".global_scale", .{layer}, .f32, &.{e}),
            .denominator = denom,
        },
        .mxfp8 => return .{
            .weight = try get(c, base ++ ".weight", .{layer}, .u32, &.{ e, rows, k / 4 }),
            .scale = try get(c, base ++ ".scales", .{layer}, .u8, &.{ e, rows, k / 32 }),
        },
        .exl3 => return error.UnsupportedInkling2ExpertFormat,
    }
}

/// Staging bytes for NVFP4 denom, split message queries, dynamic conv weight_proj, and F32->BF16 router/gates.
pub fn stageCapacity() usize {
    const a = struct {
        fn up(n: usize) usize {
            return std.mem.alignForward(usize, n, align_bytes);
        }
    }.up;
    return a(4) + // nvfp4 denominator
        2 * a(24 * 2304 * 4) + // cross_queries + cross_kv_queries
        8 * a(32 * 2304 * 2) + // 4 global layers * (k, v) weight_proj
        24 * a(128 * 2304 * 2) + // 24 router weights
        12 * 4 * 2 * a(2304 * 64 * 2); // 12 core layers * 4 roles * (forget, gain)
}

pub fn loadWeights(c: *ck.Checkpoint, device: mtl.Device, config: *const cfg.Config, wp: *const layer_plan.Plan) !model.Weights {
    try pack.admitHeaders(c, config, wp);
    const cap = stageCapacity();
    const buf = try device.buffer(cap, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    var owned = false;
    errdefer if (!owned) buf.deinit();
    try c.own(buf, cap);
    owned = true;

    var stage = Stage{ .buf = buf };
    const denom = try stage.alloc(.f32, &.{1});
    Stage.slice(denom, f32)[0] = affine4.nvfp4_global_denom;

    var w: model.Weights = undefined;
    w.embed = .{ .affine4 = try affine4.loadAffine4(c, "model.language_model.embed_tokens", .{}, 201024, 2304) };
    w.embed_norm = try get(c, "model.language_model.embed_norm.weight", .{}, .bf16, &.{2304});
    w.norm = try get(c, "model.language_model.norm.weight", .{}, .bf16, &.{2304});
    w.unembed = .{ .affine4 = try affine4.loadAffine4(c, "lm_head", .{}, 201024, 2304) };
    w.output_query = try get(c, "model.language_model.final_input.query", .{}, .f32, &.{2304});

    const msg_q = try get(c, "model.language_model.message_input.query", .{}, .f32, &.{ 12, 4, 2304 });
    w.cross_queries = try stage.alloc(.f32, &.{ 24, 2304 });
    w.cross_kv_queries = try stage.alloc(.f32, &.{ 24, 2304 });
    affine4.splitMessageQueries(rawBytes(msg_q), Stage.slice(w.cross_queries, f32), Stage.slice(w.cross_kv_queries, f32));

    for (&w.attn_queries, 0..) |*q, i| {
        const layer: u32 = @intCast(i + 1);
        const dims: []const usize = if (cfg.isGlobal(layer)) &.{ 3, 2304 } else &.{2304};
        q.* = try get(c, "model.language_model.layers.{d}.attention_input.query", .{layer}, .f32, dims);
    }
    for (&w.mlp_queries, 0..) |*q, i| {
        q.* = try get(c, "model.language_model.layers.{d}.mlp_input.query", .{i}, .f32, &.{2304});
    }

    for (&w.layers, 0..) |*l, i| {
        const layer: u32 = @intCast(i);
        const p = wp.at(layer);
        const global = p.kind == .global;
        const q_width: u32 = p.heads.q * p.heads.dim;
        const local_norm = if (!global) try get(c, "model.language_model.layers.{d}.input_layernorm.weight", .{layer}, .bf16, &.{2304}) else undefined;
        l.attn = .{
            .wq = .{ .affine4 = try affine4.loadAffine4(c, "model.language_model.layers.{d}.self_attn.q_proj", .{layer}, q_width, 2304) },
            .wk = .{ .affine4 = try affine4.loadAffine4(c, "model.language_model.layers.{d}.self_attn.k_proj", .{layer}, 1024, 2304) },
            .wv = .{ .affine4 = try affine4.loadAffine4(c, "model.language_model.layers.{d}.self_attn.v_proj", .{layer}, 1024, 2304) },
            .wr = .{ .affine4 = try affine4.loadAffine4(c, "model.language_model.layers.{d}.self_attn.r_proj", .{layer}, 384, 2304) },
            .wo = .{ .affine4 = try affine4.loadAffine4(c, "model.language_model.layers.{d}.self_attn.o_proj", .{layer}, 2304, q_width) },
            .gate = .{ .affine4 = try affine4.loadAffine4(c, "model.language_model.layers.{d}.self_attn.gate_proj", .{layer}, q_width, 2304) },
            .sink = .{ .affine4 = try affine4.loadAffine4(c, "model.language_model.layers.{d}.self_attn.sink_proj", .{layer}, 24, 2304) },
            .q_scale = try get(c, "model.language_model.layers.{d}.self_attn.q_l2_clip.scale", .{layer}, .bf16, &.{ 24, 128 }),
            .rel_scale = try get(c, "model.language_model.layers.{d}.self_attn.rel_l2_clip.scale", .{layer}, .bf16, &.{ 24, 16 }),
            .rel_projection = try get(c, "model.language_model.layers.{d}.self_attn.rel_logits_proj.proj", .{layer}, .bf16, &.{ 16, p.extent }),
            .k_conv = try loadConv(c, &stage, layer, "k", global),
            .v_conv = try loadConv(c, &stage, layer, "v", global),
            .q_norm = if (global) try get(c, "model.language_model.layers.{d}.q_input_layernorm.weight", .{layer}, .bf16, &.{2304}) else local_norm,
            .k_norm = if (global) try get(c, "model.language_model.layers.{d}.k_input_layernorm.weight", .{layer}, .bf16, &.{2304}) else local_norm,
            .v_norm = if (global) try get(c, "model.language_model.layers.{d}.v_input_layernorm.weight", .{layer}, .bf16, &.{2304}) else local_norm,
        };
        l.attn_conv = try get(c, "model.language_model.layers.{d}.post_attention_convolution.conv1d.weight", .{layer}, .bf16, &.{ 2304, 4, 1 });
        l.mlp_norm = try get(c, "model.language_model.layers.{d}.post_attention_layernorm.weight", .{layer}, .bf16, &.{2304});
        l.mlp_conv = try get(c, "model.language_model.layers.{d}.post_mlp_convolution.conv1d.weight", .{layer}, .bf16, &.{ 2304, 4, 1 });
        l.router_weight = try getBf16OrF32(c, &stage, "model.language_model.layers.{d}.mlp.gate.weight", .{layer}, &.{ 128, 2304 });
        l.router_bias = try get(c, "model.language_model.layers.{d}.mlp.gate.e_score_correction_bias", .{layer}, .f32, &.{128});
        l.gu = try loadExpert(c, denom, layer, "gate_up_proj", p.precision, 2304, 2304);
        l.down = try loadExpert(c, denom, layer, "down_proj", p.precision, 2304, 1152);
        l.controllers = null;
        if (config.text.loop) |lp| if (layer >= lp.core_start and layer < lp.core_end) {
            var controllers: [4]model.Controller = undefined;
            inline for (.{ "q", "k", "v", "mlp" }, 0..) |role, j| controllers[j] = .{
                .down = try get(c, "model.language_model.layers.{d}." ++ role ++ "_message_gate.down_proj.weight", .{layer}, .bf16, &.{ 64, 4608 }),
                .forget = try getBf16OrF32(c, &stage, "model.language_model.layers.{d}." ++ role ++ "_message_gate.forget_gate_proj.weight", .{layer}, &.{ 2304, 64 }),
                .gain = try getBf16OrF32(c, &stage, "model.language_model.layers.{d}." ++ role ++ "_message_gate.message_gate_proj.weight", .{layer}, &.{ 2304, 64 }),
            };
            l.controllers = controllers;
        };
    }
    return w;
}
