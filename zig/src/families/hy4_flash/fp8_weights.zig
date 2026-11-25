//! Official Hy4-flash-FP8 (W8A16) checkpoint tensor map; each projection's format comes from its stored tensors.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const engram = @import("engram.zig");

const Text = config.Text;
const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;
const fp8 = core.quant.fp8;

/// A projection [n, k] as stored: bf16 alone, or e4m3 codes with bf16 scale_inv [ceil(n/128), ceil(k/128)].
pub const Proj = union(enum) {
    bf16: Tensor,
    fp8: fp8.Host,

    pub fn rows(p: Proj) usize {
        return switch (p) {
            .bf16 => |t| t.dim(0),
            .fp8 => |f| f.weight.dim(0),
        };
    }
};

/// One iHC site: fn [2*hc, hc*hidden] fp32, base [2*hc] fp32, scale [2] fp32 (pre, post).
pub const Hc = struct { fn_: Tensor, base: Tensor, scale: Tensor };

pub const Indexer = struct {
    query: Tensor,
    head_weights: Tensor,
    token_key: Tensor,
    token_key_norm: Tensor,
    values: Tensor,
    gates: Tensor,
    key_norm: Tensor,
    ape: Tensor,
};

pub const Attn = struct {
    q_a: Tensor,
    q_a_norm: Tensor,
    q_b: Proj,
    kv_a: Tensor,
    kv_a_norm: Tensor,
    kv_b: Tensor,
    gate: Proj,
    o: Proj,
    post_norm: Tensor,
    sinks: Tensor,
    indexer: ?Indexer,
};

pub const Dense = struct { gate: Tensor, up: Tensor, down: Tensor, post_norm: Tensor };

pub const Moe = struct {
    router: Tensor,
    bias: Tensor,
    latent_down: Tensor,
    latent_up: Tensor,
    gate_up: Tensor,
    gate_up_s: Tensor,
    down: Tensor,
    down_s: Tensor,
    shared: Dense,
};

pub const Engram = struct {
    table: Tensor,
    map: Tensor,
    head_norms: [engram.columns]Tensor,
    value: Tensor,
    value_bias: Tensor,
    keys: [4]Tensor,
    key_biases: [4]Tensor,
    norm_keys: [4]Tensor,
    norm_queries: [4]Tensor,
    pre_gamma: Tensor,
    stream_scales: [4]Tensor,
};

pub const Layer = struct {
    input_norm: Tensor,
    post_attn_norm: Tensor,
    hc_attn: Hc,
    hc_mlp: Hc,
    attn: Attn,
    mlp: union(enum) { dense: Dense, moe: Moe },
    engram: ?Engram,
};

/// The tensors outside the decoder layers.
pub const Top = struct { embed: Tensor, norm: Tensor, head: Tensor, hc_head: Hc };

pub const Map = struct {
    gpa: std.mem.Allocator,
    top: Top,
    layers: []Layer,

    pub fn deinit(m: *Map) void {
        m.gpa.free(m.layers);
        m.* = undefined;
    }
};

const Names = struct {
    ck: *Checkpoint,
    i: usize,
    buf: [192]u8 = undefined,

    fn name(self: *Names, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&self.buf, "model.layers.{d}." ++ fmt, .{self.i} ++ args);
    }

    fn t(self: *Names, dtype: core.checkpoint.DType, shape: []const usize, comptime fmt: []const u8, args: anytype) !Tensor {
        return self.ck.expect(try self.name(fmt, args), dtype, shape);
    }

    fn bf(self: *Names, shape: []const usize, comptime fmt: []const u8, args: anytype) !Tensor {
        return self.t(.bf16, shape, fmt, args);
    }

    fn f32s(self: *Names, shape: []const usize, comptime fmt: []const u8, args: anytype) !Tensor {
        return self.t(.f32, shape, fmt, args);
    }

    /// Read `module` [n, k] in whichever format its tensors store: bf16 alone, or e4m3 with bf16 scale_inv.
    fn proj(self: *Names, n: usize, k: usize, comptime module: []const u8) !Proj {
        const w = try self.ck.get(try self.name(module ++ ".weight", .{}));
        switch (w.dtype) {
            .bf16 => {
                if (self.ck.has(try self.name(module ++ ".weight_scale_inv", .{}))) return error.UnexpectedTensor;
                return .{ .bf16 = try self.bf(&.{ n, k }, module ++ ".weight", .{}) };
            },
            .f8_e4m3 => {
                const prefix = try self.name(module, .{});
                var own: [192]u8 = undefined;
                @memcpy(own[0..prefix.len], prefix);
                return .{ .fp8 = try fp8.read(self.ck, own[0..prefix.len], n, k, .bf16, null) };
            },
            else => return error.UnsupportedQuantization,
        }
    }

    fn hc(self: *Names, text: Text, comptime site: []const u8) !Hc {
        const m: usize = text.hc_mult;
        const d: usize = text.hidden_size;
        return .{
            .fn_ = try self.f32s(&.{ 2 * m, m * d }, site ++ ".hc_pre.hc_fn", .{}),
            .base = try self.f32s(&.{2 * m}, site ++ ".hc_pre.hc_base", .{}),
            .scale = try self.f32s(&.{2}, site ++ ".hc_pre.hc_scale", .{}),
        };
    }
};

fn attn(nm: *Names, text: Text) !Attn {
    const h = text.headShape(@intCast(nm.i));
    const d: usize = text.hidden_size;
    const p = "self_attn.";
    return .{
        .q_a = try nm.bf(&.{ text.q_lora_rank, d }, p ++ "q_a_proj.weight", .{}),
        .q_a_norm = try nm.bf(&.{text.q_lora_rank}, p ++ "q_a_layernorm.weight", .{}),
        .q_b = try nm.proj(h.queryRows(), text.q_lora_rank, p ++ "q_b_proj"),
        .kv_a = try nm.bf(&.{ @as(usize, text.kv_lora_rank) + h.rotary, d }, p ++ "kv_a_proj_with_mqa.weight", .{}),
        .kv_a_norm = try nm.bf(&.{text.kv_lora_rank}, p ++ "kv_a_layernorm.weight", .{}),
        .kv_b = try nm.bf(&.{ h.kvRows(), text.kv_lora_rank }, p ++ "kv_b_proj.weight", .{}),
        .gate = try nm.proj(@as(usize, h.heads) * h.value, d, p ++ "linear_gate"),
        .o = try nm.proj(d, @as(usize, h.heads) * h.value, p ++ "o_proj"),
        .post_norm = try nm.bf(&.{d}, p ++ "post_proj_norm.weight", .{}),
        .sinks = try nm.f32s(&.{h.heads}, p ++ "learnable_sink_param", .{}),
        .indexer = if (text.layer_types[nm.i] == .deepseek_sparse_attention) try indexer(nm, text) else null,
    };
}

fn indexer(nm: *Names, text: Text) !Indexer {
    const p = "self_attn.compressed_indexer.";
    const n: usize = text.compressed_indexer_head_dim;
    const d: usize = text.hidden_size;
    return .{
        .query = try nm.bf(&.{ @as(usize, text.compressed_indexer_num_attention_heads) * n, text.q_lora_rank }, p ++ "query_projection.weight", .{}),
        .head_weights = try nm.bf(&.{ text.compressed_indexer_num_attention_heads, d }, p ++ "head_weight_projection.weight", .{}),
        .token_key = try nm.bf(&.{ n, d }, p ++ "token_key_projection.weight", .{}),
        .token_key_norm = try nm.bf(&.{n}, p ++ "token_key_norm.weight", .{}),
        .values = try nm.f32s(&.{ n, n }, p ++ "key_value_projection.weight", .{}),
        .gates = try nm.f32s(&.{ n, n }, p ++ "key_gate_projection.weight", .{}),
        .key_norm = try nm.f32s(&.{n}, p ++ "key_norm.weight", .{}),
        .ape = try nm.f32s(&.{ text.compressed_indexer_compress_ratio, n }, p ++ "intra_group_ape", .{}),
    };
}

fn moe(nm: *Names, text: Text) !Moe {
    const e: usize = text.n_routed_experts;
    const w: usize = text.moe_intermediate_size;
    const l: usize = text.routed_expert_hidden_size;
    const d: usize = text.hidden_size;
    const shared_w: usize = @as(usize, text.moe_intermediate_size) * text.n_shared_experts;
    return .{
        .router = try nm.bf(&.{ e, d }, "mlp.gate.weight", .{}),
        .bias = try nm.f32s(&.{e}, "mlp.gate.e_score_correction_bias", .{}),
        .latent_down = try nm.bf(&.{ l, d }, "mlp.routed_expert_down_proj.weight", .{}),
        .latent_up = try nm.bf(&.{ d, l }, "mlp.routed_expert_up_proj.weight", .{}),
        .gate_up = try nm.t(.f8_e4m3, &.{ e, 2 * w, l }, "mlp.experts.gate_up_proj", .{}),
        .gate_up_s = try nm.bf(&.{ e, fp8.blocks(2 * w), fp8.blocks(l) }, "mlp.experts.gate_up_proj_scale_inv", .{}),
        .down = try nm.t(.f8_e4m3, &.{ e, l, w }, "mlp.experts.down_proj", .{}),
        .down_s = try nm.bf(&.{ e, fp8.blocks(l), fp8.blocks(w) }, "mlp.experts.down_proj_scale_inv", .{}),
        .shared = .{
            .gate = try nm.bf(&.{ shared_w, d }, "mlp.shared_experts.gate_proj.weight", .{}),
            .up = try nm.bf(&.{ shared_w, d }, "mlp.shared_experts.up_proj.weight", .{}),
            .down = try nm.bf(&.{ d, shared_w }, "mlp.shared_experts.down_proj.weight", .{}),
            .post_norm = try nm.bf(&.{d}, "mlp.combine_output_norm.weight", .{}),
        },
    };
}

fn engramTensors(nm: *Names, text: Text, number: u32) !Engram {
    if (text.hc_mult != 4) return error.BadArchitecture;
    const p = "engram_layer.";
    const d: usize = text.hidden_size;
    const head_dim: usize = text.engram_n_embed_per_ngram / text.engram_n_head_per_ngram;
    const k: usize = engram.columns * head_dim;
    const rows = try text.engramRows(number);
    var g: Engram = undefined;
    g.table = try nm.bf(&.{ rows, head_dim }, p ++ "multi_head_embedding.embedding.weight", .{});
    g.map = try nm.t(.i32, &.{text.engram_lookup_table_size}, p ++ "lookup_table", .{});
    for (&g.head_norms, 0..) |*n, h| n.* = try nm.bf(&.{head_dim}, p ++ "lookup_per_head_norms.{d}.weight", .{h});
    g.value = try nm.bf(&.{ d, k }, p ++ "value_proj.weight", .{});
    g.value_bias = try nm.bf(&.{d}, p ++ "value_proj.bias", .{});
    for (0..4) |s| {
        g.keys[s] = try nm.bf(&.{ d, k }, p ++ "key_projs.{d}.weight", .{s});
        g.key_biases[s] = try nm.bf(&.{d}, p ++ "key_projs.{d}.bias", .{s});
        g.norm_keys[s] = try nm.bf(&.{d}, p ++ "norm_key.{d}.weight", .{s});
        g.norm_queries[s] = try nm.bf(&.{d}, p ++ "norm_query.{d}.weight", .{s});
        g.stream_scales[s] = try nm.bf(&.{d}, p ++ "unified_per_hc_scales.{d}", .{s});
    }
    g.pre_gamma = try nm.bf(&.{d}, p ++ "unified_pre_gamma", .{});
    return g;
}

/// Read zero-based layer `i`'s W8A16 checkpoint tensors.
pub fn layer(ck: *Checkpoint, text: Text, i: usize) !Layer {
    var nm: Names = .{ .ck = ck, .i = i };
    const d: usize = text.hidden_size;
    const one_based: u32 = @intCast(i + 1);
    const has_engram = std.mem.indexOfScalar(u32, text.engram_layer_ids, one_based) != null;
    return .{
        .input_norm = try nm.bf(&.{d}, "input_layernorm.weight", .{}),
        .post_attn_norm = try nm.bf(&.{d}, "post_attention_layernorm.weight", .{}),
        .hc_attn = try nm.hc(text, "hc_attn_layer"),
        .hc_mlp = try nm.hc(text, "hc_mlp_layer"),
        .attn = try attn(&nm, text),
        .mlp = if (text.mlp_layer_types[i] == .dense) .{ .dense = .{
            .gate = try nm.bf(&.{ text.intermediate_size, d }, "mlp.gate_proj.weight", .{}),
            .up = try nm.bf(&.{ text.intermediate_size, d }, "mlp.up_proj.weight", .{}),
            .down = try nm.bf(&.{ d, text.intermediate_size }, "mlp.down_proj.weight", .{}),
            .post_norm = try nm.bf(&.{d}, "mlp.post_fc2_norm.weight", .{}),
        } } else .{ .moe = try moe(&nm, text) },
        .engram = if (has_engram) try engramTensors(&nm, text, one_based) else null,
    };
}

/// Read the top-level embeddings, final norm, lm_head, and hc_head tensors.
pub fn top(ck: *Checkpoint, text: Text) !Top {
    const m: usize = text.hc_mult;
    const d: usize = text.hidden_size;
    return .{
        .embed = try ck.expect("model.embed_tokens.weight", .bf16, &.{ text.vocab_size, d }),
        .norm = try ck.expect("model.norm.weight", .bf16, &.{d}),
        .head = try ck.expect("lm_head.weight", .bf16, &.{ text.vocab_size, d }),
        .hc_head = .{
            .fn_ = try ck.expect("model.hc_head.hc_head_fn", .f32, &.{ m, m * d }),
            .base = try ck.expect("model.hc_head.hc_head_base", .f32, &.{m}),
            .scale = try ck.expect("model.hc_head.hc_head_scale", .f32, &.{1}),
        },
    };
}

/// Map every backbone tensor and mark DFly2 and vision prefixes as skipped on purpose.
pub fn mapped(gpa: std.mem.Allocator, ck: *Checkpoint, text: Text) !Map {
    const m: Map = .{ .gpa = gpa, .top = try top(ck, text), .layers = try gpa.alloc(Layer, text.num_hidden_layers) };
    errdefer gpa.free(m.layers);
    for (m.layers, 0..) |*l, i| l.* = try layer(ck, text, i);
    try ck.skip("model.dfly_layers.");
    try ck.skip("vit.");
    return m;
}

/// Map the W8A16 backbone and refuse any unmapped checkpoint tensor, naming it in `why` when given.
pub fn map(gpa: std.mem.Allocator, ck: *Checkpoint, text: Text, why: ?*core.modelopt.Why) !Map {
    var m = try mapped(gpa, ck, text);
    errdefer m.deinit();
    if (why) |w| for (ck.files.items) |*file| for (file.names.keys()) |key| if (!ck.used.contains(key)) {
        w.set("unused checkpoint tensor {s}", .{key});
        return error.UnusedTensors;
    };
    if (ck.unused() != 0) return error.UnusedTensors;
    return m;
}
