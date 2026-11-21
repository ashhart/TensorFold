//! Views into the converted checkpoint: affine projections, experts back to back, bf16 / f32 tensors.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const row = @import("../../core/row_projection.zig");
const c = @import("config.zig");

pub const Tensor = ck.Tensor;
pub const Linear = row.Weights;

pub const Hc = struct { fn_: Tensor, base: Tensor, scale: Tensor };

pub const Compressor = struct {
    /// [512, 5120] bf16; with a ratio of 2 the pooling gate beside it (both multiplied in fp32).
    wkv: Tensor,
    wgate: ?Tensor,
    norm: Tensor,
};

pub const Indexer = struct {
    wq_b: Linear, // [32 x 128, 1280]
    weights_proj: Tensor, // [32, 5120] bf16
    /// Only a KV source derives index keys from its latent: [128, 512] bf16 and the key norm.
    wk: ?Tensor,
    k_norm: ?Tensor,
};

pub const Attention = struct {
    wq_a: Linear,
    q_norm: Tensor,
    wq_b: Linear,
    wkv: Linear,
    kv_norm: Tensor,
    /// [8 x 1024, 4096] bf16: group g projects heads 8g..8g+7 (block diagonal).
    wo_a: WoA,
    wo_b: Linear,
    sink: Tensor, // f32 [64]
    compressor: ?Compressor,
    indexer: ?Indexer,
};

pub const Shared = struct { w1: Linear, w2: Linear, w3: Linear };

pub const Moe = struct {
    router: Tensor, // bf16 [E, 5120]
    bias: Tensor, // f32 [E]
    bias_vl: Tensor, // f32 [E], image-span tokens
    w1: row.Experts,
    w2: row.Experts,
    w3: row.Experts,
    shared: Shared,
};

pub const Engram = struct { wkv: Linear, q_weight: Tensor, k_weight: Tensor };

pub const Layer = struct {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    attn: Attention,
    moe: Moe,
    hc_attn: Hc,
    hc_ffn: Hc,
    engram: ?Engram,
};

pub const Stage = struct {
    block: Layer,
    main_proj: ?Linear, // stage 0: [5120, 3 x 5120]
    main_norm: ?Tensor,
    norm: ?Tensor, // the last stage's head norm and Markov head (its confidence head is not used)
    markov_embed: ?Tensor,
    markov_head: ?Tensor,
};

/// The block-diagonal wo_a: affine, stacked by group (convert.py now), or bf16 [8192, 4096] (older checkpoints).
pub const WoA = union(enum) { affine: row.Experts, bf16: Tensor };

fn woA(ckpt: *const ck.Checkpoint, p: []const u8, dense: c.Width) !WoA {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}attn.wo_a.weight", .{p});
    const t = try ckpt.get(name);
    if (t.dtype == .bf16) return .{ .bf16 = try tensor(ckpt, p, "attn.wo_a.weight", .bf16, &.{ c.o_groups * c.o_rank, c.group_in }) };
    return .{ .affine = try stack(ckpt, p, "attn.wo_a", c.o_groups, c.o_rank, c.group_in, dense, 1) };
}

pub const Weights = struct {
    embed: Tensor, // bf16 [vocab, 5120]
    norm: Tensor,
    head: Linear,
    layers: [c.layers]Layer,
    stages: ?[c.stages]Stage,
};

fn expect(ckpt: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    const t = ckpt.get(name) catch |e| {
        std.log.err("deepseek_v41: the checkpoint has no {s}", .{name});
        return e;
    };
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape) or t.offset % 4 != 0) {
        std.log.err("{s}: unexpected tensor dtype, shape or alignment", .{name});
        return error.UnexpectedDeepseekTensor;
    }
    return t;
}

fn tensor(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    return expect(ckpt, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }), dtype, shape);
}

fn optional(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: ck.DType, shape: []const usize) !?Tensor {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
    if (!ckpt.has(name)) return null;
    return try expect(ckpt, name, dtype, shape);
}

/// `count` [n, k] affine matrices (0: one of rank 2) at the tensors' own width; `expected` is the config's.
fn affine(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, count: usize, n: usize, k: usize, expected: c.Width) !Linear {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
    var w = expected;
    {
        var nb: [272]u8 = undefined;
        const qt = ckpt.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{name})) catch null;
        const st_ = ckpt.get(try std.fmt.bufPrint(&nb, "{s}.scales", .{name})) catch null;
        if (qt != null and st_ != null and qt.?.rank >= 2 and st_.?.rank >= 2) {
            const words = qt.?.shape[qt.?.rank - 1];
            const groups = st_.?.shape[st_.?.rank - 1];
            if (groups > 0 and k % groups == 0 and (words * 32) % k == 0)
                w = .{ .bits = @intCast(words * 32 / k), .group = @intCast(k / groups) };
        }
    }
    const lead: []const usize = if (count == 0) &.{} else &.{count};
    var shape: [3]usize = undefined;
    @memcpy(shape[0..lead.len], lead);
    const r = lead.len + 2;
    shape[lead.len] = n;
    shape[lead.len + 1] = k * w.bits / 32;
    const q = try tensor(ckpt, name, ".weight", .u32, shape[0..r]);
    shape[lead.len + 1] = k / w.group;
    const s = try tensor(ckpt, name, ".scales", .bf16, shape[0..r]);
    const b = try tensor(ckpt, name, ".biases", .bf16, shape[0..r]);
    const out: Linear = .{ .w = q.buffer, .w_off = q.offset, .scales = s.buffer, .s_off = s.offset, .biases = b.buffer, .b_off = b.offset, .n = n, .k = k, .group = w.group, .bits = w.bits };
    try out.validate();
    return out;
}

fn stack(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, experts: usize, n: usize, k: usize, w: c.Width, repeat: usize) !row.Experts {
    const out: row.Experts = .{ .one = try affine(ckpt, prefix, suffix, experts, n, k, w), .experts = experts, .repeat = repeat };
    try out.validate();
    return out;
}

fn hcOf(ckpt: *const ck.Checkpoint, p: []const u8, site: []const u8) !Hc {
    var a: [32]u8 = undefined;
    var b: [32]u8 = undefined;
    var d: [32]u8 = undefined;
    return .{
        .fn_ = try tensor(ckpt, p, try std.fmt.bufPrint(&a, "hc_{s}_fn", .{site}), .f32, &.{ c.hc_mix, c.hc * c.hidden }),
        .base = try tensor(ckpt, p, try std.fmt.bufPrint(&b, "hc_{s}_base", .{site}), .f32, &.{c.hc_mix}),
        .scale = try tensor(ckpt, p, try std.fmt.bufPrint(&d, "hc_{s}_scale", .{site}), .f32, &.{3}),
    };
}

/// A backbone block (`layer` < 40) or a DSpark stage's (`layer` = 40 + stage: window-only attention, 128 experts).
fn block(ckpt: *const ck.Checkpoint, p: []const u8, layer: usize, dense: c.Width, ew: c.Width) !Layer {
    const backbone = layer < c.layers;
    const ratio: u8 = if (backbone) c.ratios[layer] else 0;
    const n_exp: usize = if (backbone) c.experts else c.draft_experts;
    const k_top: usize = if (backbone) c.topk else c.draft_topk;
    var attn: Attention = .{
        .wq_a = try affine(ckpt, p, "attn.wq_a", 0, c.q_rank, c.hidden, dense),
        .q_norm = try tensor(ckpt, p, "attn.q_norm.weight", .bf16, &.{c.q_rank}),
        .wq_b = try affine(ckpt, p, "attn.wq_b", 0, c.heads * c.head_dim, c.q_rank, dense),
        .wkv = try affine(ckpt, p, "attn.wkv", 0, c.head_dim, c.hidden, dense),
        .kv_norm = try tensor(ckpt, p, "attn.kv_norm.weight", .bf16, &.{c.head_dim}),
        .wo_a = try woA(ckpt, p, dense),
        .wo_b = try affine(ckpt, p, "attn.wo_b", 0, c.hidden, c.o_groups * c.o_rank, dense),
        .sink = try tensor(ckpt, p, "attn.attn_sink", .f32, &.{c.heads}),
        .compressor = null,
        .indexer = null,
    };
    if (backbone and c.isKvSource(layer)) attn.compressor = .{
        .wkv = try tensor(ckpt, p, "attn.compressor.wkv.weight", .bf16, &.{ c.head_dim, c.hidden }),
        .wgate = if (ratio > 1) try tensor(ckpt, p, "attn.compressor.wgate.weight", .bf16, &.{ c.head_dim, c.hidden }) else null,
        .norm = try tensor(ckpt, p, "attn.compressor.norm.weight", .bf16, &.{c.head_dim}),
    };
    if (backbone and c.isIndexSource(layer)) attn.indexer = .{
        .wq_b = try affine(ckpt, p, "attn.indexer.wq_b", 0, c.index_heads * c.index_dim, c.q_rank, dense),
        .weights_proj = try tensor(ckpt, p, "attn.indexer.weights_proj.weight", .bf16, &.{ c.index_heads, c.hidden }),
        .wk = try optional(ckpt, p, "attn.indexer.wk.weight", .bf16, &.{ c.index_dim, c.head_dim }),
        .k_norm = try optional(ckpt, p, "attn.indexer.k_norm.weight", .bf16, &.{c.index_dim}),
    };
    if (backbone and (attn.indexer != null) and ((attn.indexer.?.wk != null) != c.isKvSource(layer))) return error.UnexpectedDeepseekTensor;
    return .{
        .attn_norm = try tensor(ckpt, p, "attn_norm.weight", .bf16, &.{c.hidden}),
        .ffn_norm = try tensor(ckpt, p, "ffn_norm.weight", .bf16, &.{c.hidden}),
        .attn = attn,
        .moe = .{
            .router = try tensor(ckpt, p, "ffn.gate.weight", .bf16, &.{ n_exp, c.hidden }),
            .bias = try tensor(ckpt, p, "ffn.gate.bias", .f32, &.{n_exp}),
            .bias_vl = try tensor(ckpt, p, "ffn.gate.bias_vl", .f32, &.{n_exp}),
            .w1 = try stack(ckpt, p, "ffn.experts.w1", n_exp, c.moe_inter, c.hidden, ew, k_top),
            .w3 = try stack(ckpt, p, "ffn.experts.w3", n_exp, c.moe_inter, c.hidden, ew, k_top),
            .w2 = try stack(ckpt, p, "ffn.experts.w2", n_exp, c.hidden, c.moe_inter, ew, 1),
            .shared = .{
                .w1 = try affine(ckpt, p, "ffn.shared_experts.w1", 0, c.moe_inter, c.hidden, dense),
                .w2 = try affine(ckpt, p, "ffn.shared_experts.w2", 0, c.hidden, c.moe_inter, dense),
                .w3 = try affine(ckpt, p, "ffn.shared_experts.w3", 0, c.moe_inter, c.hidden, dense),
            },
        },
        .hc_attn = try hcOf(ckpt, p, "attn"),
        .hc_ffn = try hcOf(ckpt, p, "ffn"),
        .engram = if (backbone and c.engramIndex(layer) != null) .{
            .wkv = try affine(ckpt, p, "engram.wkv", 0, c.hidden * (c.hc + 1), c.engram_cols * c.engram_dim, dense),
            .q_weight = try tensor(ckpt, p, "engram.q_weight", .bf16, &.{ c.hc, c.hidden }),
            .k_weight = try tensor(ckpt, p, "engram.k_weight", .bf16, &.{ c.hc, c.hidden }),
        } else null,
    };
}

pub fn load(ckpt: *const ck.Checkpoint, config: *const c.Config, drafts: bool) !Weights {
    var w: Weights = .{
        .embed = try expect(ckpt, "embed.weight", .bf16, &.{ c.vocab, c.hidden }),
        .norm = try expect(ckpt, "norm.weight", .bf16, &.{c.hidden}),
        .head = try affine(ckpt, "", "head", 0, c.vocab, c.hidden, config.dense),
        .layers = undefined,
        .stages = null,
    };
    for (&w.layers, 0..) |*l, i| {
        var buf: [32]u8 = undefined;
        l.* = try block(ckpt, try std.fmt.bufPrint(&buf, "layers.{d}.", .{i}), i, config.dense, config.experts[i]);
    }
    if (drafts) {
        var stages: [c.stages]Stage = undefined;
        for (&stages, 0..) |*s, i| {
            var buf: [32]u8 = undefined;
            const p = try std.fmt.bufPrint(&buf, "mtp.{d}.", .{i});
            const last = i == c.stages - 1;
            s.* = .{
                .block = try block(ckpt, p, c.layers + i, config.dense, config.draft_experts[i]),
                .main_proj = if (i == 0) try affine(ckpt, p, "main_proj", 0, c.hidden, c.hidden * c.draft_targets.len, config.dense) else null,
                .main_norm = if (i == 0) try tensor(ckpt, p, "main_norm.weight", .bf16, &.{c.hidden}) else null,
                .norm = if (last) try tensor(ckpt, p, "norm.weight", .bf16, &.{c.hidden}) else null,
                .markov_embed = if (last) try tensor(ckpt, p, "markov_head.embed.weight", .bf16, &.{ c.vocab, c.markov_rank }) else null,
                .markov_head = if (last) try tensor(ckpt, p, "markov_head.head.weight", .bf16, &.{ c.vocab, c.markov_rank }) else null,
            };
        }
        w.stages = stages;
    }
    return w;
}
