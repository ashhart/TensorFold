//! Dense BF16 backbone, FP32 queries and loop controllers borrowed from the mapped checkpoint.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const config = @import("config.zig");

pub const Tensor = ck.Tensor;

pub const Error = error{UnexpectedInklingTensor};

/// One layer's attention tensors: sliding layers one input norm and conv taps; global ones one norm per q, k, v.
pub const Attention = struct {
    wq: Tensor, // [q*dim, H]
    wk: Tensor, // [kv*dim, H]
    wv: Tensor,
    wr: Tensor, // [q*d_rel, H]
    wo: Tensor, // [H, q*dim]
    gate: Tensor, // [q*dim, H]
    sink: Tensor, // [q, H]
    q_scale: Tensor, // [q, dim]
    rel_scale: Tensor, // [q, d_rel]
    rel_proj: Tensor, // [d_rel, extent]
    /// Sliding: the weight [kv*dim, 1, 4]. Global: static_w [4, kv*dim].
    k_conv: Tensor,
    v_conv: Tensor,
    /// Global only: weight_proj [kv*4, H], whose per-token rows add to the static taps.
    k_taps: ?Tensor,
    v_taps: ?Tensor,
    /// Sliding: attn_norm in [0]. Global: q, k, v input norms.
    norms: [3]?Tensor,
};

pub const Controller = struct { down: Tensor, forget: Tensor, gain: Tensor };
pub const CrossQueries = struct { main: Tensor, kv: Tensor };

pub const Layer = struct {
    attn: Attention,
    attn_conv: Tensor, // [H, 1, 4]
    mlp_conv: Tensor, // [H, 1, 4]
    mlp_norm: Tensor, // [H]
    router: Tensor, // [E, H] bf16, read in f32
    router_bias: Tensor, // [E] f32
    /// FP32 queries of the attention input (none for layer 0; three for a global layer) and the MLP input.
    attn_query: ?Tensor,
    mlp_query: Tensor,
    controllers: ?[4]Controller,
};

pub const Weights = struct {
    embed: Tensor, // [vocab, H]
    embed_norm: Tensor, // [H]
    norm: Tensor, // [H]
    unembed: Tensor, // [vocab, H]
    output_query: Tensor, // [H] f32
    cross: ?CrossQueries,
    layers: []Layer,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        gpa.free(w.layers);
    }
};

/// Check the size and scalar/vector alignment used by the kernels.
fn get(c: *const ck.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: ck.DType, dims: []const usize) !Tensor {
    var buf: [160]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    const t = try c.get(name);
    var bytes: usize = dtype.size();
    for (dims) |n| bytes *= n;
    const alignment: usize = if (dtype == .f32) 4 else 16;
    const ok = t.dtype == dtype and t.rank == dims.len and std.mem.eql(usize, t.shape[0..t.rank], dims) and t.bytes == bytes and t.offset % alignment == 0;
    if (!ok) {
        std.log.err("inkling2: {s} is not {s}{any} (found {s}{any})", .{ name, @tagName(dtype), dims, @tagName(t.dtype), t.shape[0..t.rank] });
        return Error.UnexpectedInklingTensor;
    }
    return t;
}

fn attention(c: *const ck.Checkpoint, t: *const config.Text, i: u32) !Attention {
    const h: usize = t.hidden;
    const hd = t.heads(i);
    const q: usize = @as(usize, hd.q) * hd.dim;
    const kv: usize = @as(usize, hd.kv) * hd.dim;
    const taps: usize = config.conv_taps;
    const L = .{i};
    const p = "model.llm.layers.{d}.attn.";
    var a: Attention = undefined;
    a.wq = try get(c, p ++ "wq_du.weight", L, .bf16, &.{ q, h });
    a.wk = try get(c, p ++ "wk_dv.weight", L, .bf16, &.{ kv, h });
    a.wv = try get(c, p ++ "wv_dv.weight", L, .bf16, &.{ kv, h });
    a.wr = try get(c, p ++ "wr_du.weight", L, .bf16, &.{ @as(usize, hd.q) * t.d_rel, h });
    a.wo = try get(c, p ++ "wo_ud.weight", L, .bf16, &.{ h, q });
    a.gate = try get(c, p ++ "gate_attn.weight", L, .bf16, &.{ q, h });
    a.sink = try get(c, p ++ "attn_sink_proj.weight", L, .bf16, &.{ hd.q, h });
    a.q_scale = try get(c, p ++ "q_l2_clip.scale", L, .bf16, &.{ hd.q, hd.dim });
    a.rel_scale = try get(c, p ++ "rel_l2_clip.scale", L, .bf16, &.{ hd.q, t.d_rel });
    a.rel_proj = try get(c, p ++ "rel_logits_proj.proj", L, .bf16, &.{ t.d_rel, t.extent(i) });
    a.norms = .{ null, null, null };
    if (config.isGlobal(i)) {
        a.k_conv = try get(c, p ++ "k_sconv.static_w", L, .bf16, &.{ taps, kv });
        a.v_conv = try get(c, p ++ "v_sconv.static_w", L, .bf16, &.{ taps, kv });
        a.k_taps = try get(c, p ++ "k_sconv.weight_proj", L, .bf16, &.{ @as(usize, hd.kv) * taps, h });
        a.v_taps = try get(c, p ++ "v_sconv.weight_proj", L, .bf16, &.{ @as(usize, hd.kv) * taps, h });
        a.norms[0] = try get(c, "model.llm.layers.{d}.q_input_norm.weight", L, .bf16, &.{h});
        a.norms[1] = try get(c, "model.llm.layers.{d}.k_input_norm.weight", L, .bf16, &.{h});
        a.norms[2] = try get(c, "model.llm.layers.{d}.v_input_norm.weight", L, .bf16, &.{h});
    } else {
        a.k_conv = try get(c, p ++ "k_sconv.weight", L, .bf16, &.{ kv, 1, taps });
        a.v_conv = try get(c, p ++ "v_sconv.weight", L, .bf16, &.{ kv, 1, taps });
        a.k_taps = null;
        a.v_taps = null;
        a.norms[0] = try get(c, "model.llm.layers.{d}.attn_norm.weight", L, .bf16, &.{h});
    }
    return a;
}

/// The checkpoint's dense tensors checked against the config, every matrix in place.
pub fn load(gpa: std.mem.Allocator, c: *const ck.Checkpoint, cfg: *const config.Config) !Weights {
    const t = &cfg.text;
    const h: usize = t.hidden;
    const layers = try gpa.alloc(Layer, t.layers);
    errdefer gpa.free(layers);
    for (layers, 0..) |*l, k| {
        const i: u32 = @intCast(k);
        const L = .{i};
        l.attn = try attention(c, t, i);
        l.attn_conv = try get(c, "model.llm.layers.{d}.attn_sconv.weight", L, .bf16, &.{ h, 1, config.conv_taps });
        l.mlp_conv = try get(c, "model.llm.layers.{d}.mlp_sconv.weight", L, .bf16, &.{ h, 1, config.conv_taps });
        l.mlp_norm = try get(c, "model.llm.layers.{d}.mlp_norm.weight", L, .bf16, &.{h});
        l.router = try get(c, "model.llm.layers.{d}.mlp.gate.weight", L, .bf16, &.{ t.experts, h });
        l.router_bias = try get(c, "model.llm.layers.{d}.mlp.gate.bias", L, .f32, &.{t.experts});
        l.mlp_query = try get(c, "model.llm.attn_res.mlp_queries.{d}", L, .f32, &.{h});
        // queries.j feeds layer j + 1's attention input: layer 0 reads the embedding alone
        l.attn_query = null;
        if (i > 0) {
            const dims: []const usize = if (config.isGlobal(i)) &.{ 3, h } else &.{h};
            l.attn_query = try get(c, "model.llm.attn_res.attn_queries.{d}", .{i - 1}, .f32, dims);
        }
        l.controllers = null;
        if (t.loop) |loop| if (i >= loop.start and i < loop.end) {
            var controllers: [4]Controller = undefined;
            inline for (.{ "attn", "k", "v", "mlp" }, 0..) |name, g| {
                const p = "model.llm.layers.{d}.controllers." ++ name;
                controllers[g] = .{
                    .down = try get(c, p ++ ".down_weight", L, .bf16, &.{ loop.rank, 2 * h }),
                    .forget = try get(c, p ++ ".forget_weight", L, .bf16, &.{ h, loop.rank }),
                    .gain = try get(c, p ++ ".gain_weight", L, .bf16, &.{ h, loop.rank }),
                };
            }
            l.controllers = controllers;
        };
    }
    return .{
        .embed = try get(c, "model.llm.embed.weight", .{}, .bf16, &.{ t.vocab, h }),
        .embed_norm = try get(c, "model.llm.embed_norm.weight", .{}, .bf16, &.{h}),
        .norm = try get(c, "model.llm.norm.weight", .{}, .bf16, &.{h}),
        .unembed = try get(c, "model.llm.unembed.weight", .{}, .bf16, &.{ t.vocab, h }),
        .output_query = try get(c, "model.llm.attn_res.output_query", .{}, .f32, &.{h}),
        .cross = if (t.loop) |loop| .{
            .main = try get(c, "model.llm.middle_cycle_cross_pass_queries", .{}, .f32, &.{ 2 * (loop.end - loop.start), h }),
            .kv = try get(c, "model.llm.middle_cycle_cross_pass_kv_queries", .{}, .f32, &.{ 2 * (loop.end - loop.start), h }),
        } else null,
        .layers = layers,
    };
}
