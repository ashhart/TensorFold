//! Checkpoint-backed weights for the target graph and DFly2. GPU loaders consume one layer at a time.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const checkpoint = @import("loader.zig");
const ranks = @import("rank_weights.zig");
const mla = @import("mla_weights.zig");
const ihc = @import("ihc.zig");
const engram = @import("engram.zig");
const engram_rows = @import("engram_rows.zig");
const rope = @import("rope.zig");

const Tensor = core.checkpoint_host.Tensor;
const Affine = core.quant.mlx.Host;
const Allocator = std.mem.Allocator;
const Rank = core.quant.Rank;

/// Prefix-scoped, checked views. The checkpoint owns these bytes.
const Reader = struct {
    model: *checkpoint.Loaded,
    prefix: []const u8,

    fn tensor(self: Reader, field: []const u8, format: core.checkpoint_host.DType, dims: []const usize) !Tensor {
        var scratch: [160]u8 = undefined;
        const key = try std.fmt.bufPrint(&scratch, "{s}{s}", .{ self.prefix, field });
        return self.model.checkpoint.expect(key, format, dims);
    }

    fn affine(self: Reader, field: []const u8, dims: []const usize) !Affine {
        var scratch: [128]u8 = undefined;
        const key = try std.fmt.bufPrint(&scratch, "{s}{s}", .{ self.prefix, field });
        return ranks.affine(self.model, key, dims);
    }

    fn bf16(self: Reader, field: []const u8, dims: []const usize) !Tensor {
        return self.tensor(field, .bf16, dims);
    }
};

pub const Head = struct {
    embedding: Affine,
    lm_head: Affine,
    norm: Tensor,
    merge: ihc.Weights,

    pub fn read(model: *checkpoint.Loaded, rows: u32) !Head {
        const t = model.config.value.text_config;
        const r: Reader = .{ .model = model, .prefix = "" };
        return .{
            .embedding = try r.affine("model.embed_tokens", &.{ t.vocab_size, t.hidden_size }),
            .lm_head = try r.affine("lm_head", &.{ t.vocab_size, t.hidden_size }),
            .norm = try r.bf16("model.norm.weight", &.{t.hidden_size}),
            .merge = try ihc.Weights.read(model, .head, rows),
        };
    }
};

fn Mlp(comptime Matrix: type) type {
    return struct { gate: Matrix, up: Matrix, down: Matrix };
}

fn affineMlp(r: Reader, h: usize, inner: usize) !Mlp(Affine) {
    return .{
        .gate = try r.affine("gate_proj", &.{ inner, h }),
        .up = try r.affine("up_proj", &.{ inner, h }),
        .down = try r.affine("down_proj", &.{ h, inner }),
    };
}

fn draftMlp(r: Reader, h: usize, inner: usize) !Mlp(Tensor) {
    return .{
        .gate = try r.bf16("gate_proj.weight", &.{ inner, h }),
        .up = try r.bf16("up_proj.weight", &.{ inner, h }),
        .down = try r.bf16("down_proj.weight", &.{ h, inner }),
    };
}

pub const Index = struct {
    token_keys: Affine,
    token_norm: Tensor,
    query: Affine,
    head_weights: Affine,
    group_values: Tensor,
    group_gates: Tensor,
    group_bias: Tensor,
    group_norm: Tensor,

    fn read(r: Reader, t: config.Text) !Index {
        const width = t.compressed_indexer_head_dim;
        const heads = t.compressed_indexer_num_attention_heads;
        return .{
            .token_keys = try r.affine("token_key_projection", &.{ width, t.hidden_size }),
            .token_norm = try r.bf16("token_key_norm.weight", &.{width}),
            .query = try r.affine("query_projection", &.{ @as(usize, heads) * width, t.q_lora_rank }),
            .head_weights = try r.affine("head_weight_projection", &.{ heads, t.hidden_size }),
            .group_values = try r.tensor("key_value_projection.weight", .f32, &.{ width, width }),
            .group_gates = try r.tensor("key_gate_projection.weight", .f32, &.{ width, width }),
            .group_bias = try r.tensor("intra_group_ape", .f32, &.{ t.compressed_indexer_compress_ratio, width }),
            .group_norm = try r.tensor("key_norm.weight", .f32, &.{width}),
        };
    }
};

pub const Routed = struct {
    router: Tensor,
    correction: Tensor,
    down: Affine,
    up: Affine,
    experts: ranks.Experts,
    shared: Mlp(Affine),
    norm: Tensor,
};

pub const Feed = union(enum) {
    dense: struct { projections: Mlp(Affine), norm: Tensor },
    routed: Routed,

    fn read(a: Allocator, model: *checkpoint.Loaded, layer: u32, rank: Rank) !Feed {
        const t = model.config.value.text_config;
        var buf: [80]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.", .{layer});
        const r: Reader = .{ .model = model, .prefix = prefix };
        const h = t.hidden_size;
        if (t.mlp_layer_types[layer] == .dense) {
            return .{ .dense = .{
                .projections = try affineMlp(r, h, t.intermediate_size),
                .norm = try r.bf16("post_fc2_norm.weight", &.{h}),
            } };
        }
        var shared_buf: [96]u8 = undefined;
        const shared: Reader = .{
            .model = model,
            .prefix = try std.fmt.bufPrint(&shared_buf, "{s}shared_experts.", .{prefix}),
        };
        const latent = t.routed_expert_hidden_size;
        return .{ .routed = .{
            .router = try r.bf16("gate.weight", &.{ t.n_routed_experts, h }),
            .correction = try r.tensor("gate.e_score_correction_bias", .f32, &.{t.n_routed_experts}),
            .down = try r.affine("routed_expert_down_proj", &.{ latent, h }),
            .up = try r.affine("routed_expert_up_proj", &.{ h, latent }),
            .experts = try ranks.Experts.load(a, model, layer, rank),
            .shared = try affineMlp(shared, h, @as(usize, t.moe_intermediate_size) * t.n_shared_experts),
            .norm = try r.bf16("combine_output_norm.weight", &.{h}),
        } };
    }
};

pub const Engram = struct {
    const Stream = struct { key: Affine, bias: Tensor, key_norm: Tensor, query_norm: Tensor, residual_scale: Tensor };

    table: engram_rows.Table,
    key: [ihc.streams]Stream,
    lookup_norm: [engram.columns]Tensor,
    value: Affine,
    value_bias: Tensor,
    pre_gamma: Tensor,

    fn read(model: *checkpoint.Loaded, layer: u32) !Engram {
        const t = model.config.value.text_config;
        if (t.hc_mult != ihc.streams) return error.BadHy4IhcShape;
        var prefix_buf: [76]u8 = undefined;
        const r: Reader = .{
            .model = model,
            .prefix = try std.fmt.bufPrint(&prefix_buf, "model.layers.{d}.engram_layer.", .{layer}),
        };
        const table = try engram_rows.Table.open(model, layer + 1);
        const h = t.hidden_size;
        const input = table.elementsPerToken();
        var out: Engram = .{
            .table = table,
            .key = undefined,
            .lookup_norm = undefined,
            .value = try r.affine("value_proj", &.{ h, input }),
            .value_bias = try r.bf16("value_proj.bias", &.{h}),
            .pre_gamma = try r.bf16("unified_pre_gamma", &.{h}),
        };
        var name: [72]u8 = undefined;
        for (&out.lookup_norm, 0..) |*weight, i| {
            weight.* = try r.bf16(try std.fmt.bufPrint(&name, "lookup_per_head_norms.{d}.weight", .{i}), &.{table.width});
        }
        for (&out.key, 0..) |*weight, i| {
            weight.* = .{
                .key = try r.affine(try std.fmt.bufPrint(&name, "key_projs.{d}", .{i}), &.{ h, input }),
                .bias = try r.bf16(try std.fmt.bufPrint(&name, "key_projs.{d}.bias", .{i}), &.{h}),
                .key_norm = try r.bf16(try std.fmt.bufPrint(&name, "norm_key.{d}.weight", .{i}), &.{h}),
                .query_norm = try r.bf16(try std.fmt.bufPrint(&name, "norm_query.{d}.weight", .{i}), &.{h}),
                .residual_scale = try r.bf16(try std.fmt.bufPrint(&name, "unified_per_hc_scales.{d}", .{i}), &.{h}),
            };
        }
        return out;
    }
};

pub const TargetLayer = struct {
    project_query: Affine,
    project_kv: Affine,
    norm_query: Tensor,
    norm_kv: Tensor,
    attention: ranks.Attention,
    absorbed: mla.Weights,
    sinks: [64]f32,
    position: rope.Parameters,
    index: ?Index,
    memory: ?Engram,
    attention_hc: ihc.Weights,
    attention_norm: Tensor,
    attention_output_norm: Tensor,
    mlp_hc: ihc.Weights,
    mlp_norm: Tensor,
    feed: Feed,

    pub fn read(a: Allocator, model: *checkpoint.Loaded, layer: u32, rank: Rank, rows: u32) !TargetLayer {
        _ = try model.attentionTargetLayer(.target, layer);
        const t = model.config.value.text_config;
        const h = t.hidden_size;
        var path: [92]u8 = undefined;
        const r: Reader = .{ .model = model, .prefix = try std.fmt.bufPrint(&path, "model.layers.{d}.", .{layer}) };
        var ix_path: [96]u8 = undefined;
        const ix: Reader = .{
            .model = model,
            .prefix = try std.fmt.bufPrint(&ix_path, "{s}self_attn.compressed_indexer.", .{r.prefix}),
        };
        const shape = t.headShape(layer);
        var attention = try ranks.Attention.load(a, model, layer, rank);
        errdefer attention.deinit(a);
        var absorbed = try mla.read(a, model, .target, layer, attention.heads);
        errdefer absorbed.deinit(a);
        var all_sinks: [64]f32 = undefined;
        const sink = try model.attentionSink(.target, layer, &all_sinks);
        var packed_sinks: [64]f32 = @splat(0);
        const first = attention.heads.first;
        const count = attention.heads.count;
        @memcpy(packed_sinks[0..count], sink[first..][0..count]);
        const kind = t.layer_types[layer];
        return .{
            .project_query = try r.affine("self_attn.q_a_proj", &.{ t.q_lora_rank, h }),
            .project_kv = try r.affine("self_attn.kv_a_proj_with_mqa", &.{ @as(usize, t.kv_lora_rank) + shape.rotary, h }),
            .norm_query = try r.bf16("self_attn.q_a_layernorm.weight", &.{t.q_lora_rank}),
            .norm_kv = try r.bf16("self_attn.kv_a_layernorm.weight", &.{t.kv_lora_rank}),
            .attention = attention,
            .absorbed = absorbed,
            .sinks = packed_sinks,
            .position = try t.rotary(kind),
            .index = if (kind == .deepseek_sparse_attention) try Index.read(ix, t) else null,
            .memory = if (std.mem.indexOfScalar(u32, t.engram_layer_ids, layer + 1) != null) try Engram.read(model, layer) else null,
            .attention_hc = try ihc.Weights.read(model, .{ .attention = layer }, rows),
            .attention_norm = try r.bf16("input_layernorm.weight", &.{h}),
            .attention_output_norm = try r.bf16("self_attn.post_proj_norm.weight", &.{h}),
            .mlp_hc = try ihc.Weights.read(model, .{ .feed_forward = layer }, rows),
            .mlp_norm = try r.bf16("post_attention_layernorm.weight", &.{h}),
            .feed = try Feed.read(a, model, layer, rank),
        };
    }

    pub fn deinit(self: *TargetLayer, a: Allocator) void {
        self.absorbed.deinit(a);
        self.attention.deinit(a);
        self.* = undefined;
    }
};

pub const DraftHead = struct {
    anchor_norm: Tensor,
    hidden_norm: Tensor,
    fusion: Tensor,
    fusion_gate: Tensor,
    norm: Tensor,
    correction_hidden_norm: Tensor,
    correction_embedding_norm: Tensor,
    correction: Mlp(Tensor),

    pub fn read(model: *checkpoint.Loaded) !DraftHead {
        const h = model.config.value.text_config.hidden_size;
        const r: Reader = .{ .model = model, .prefix = "model.dfly_layers." };
        const correction: Reader = .{ .model = model, .prefix = "model.dfly_layers.hidden_correction." };
        return .{
            .anchor_norm = try r.bf16("mtp_fusion.enorm.weight", &.{h}),
            .hidden_norm = try r.bf16("mtp_fusion.hnorm.weight", &.{h}),
            .fusion = try r.bf16("mtp_fusion.eh_proj.weight", &.{ h, @as(usize, h) * 2 }),
            .fusion_gate = try r.bf16("mtp_fusion.gate", &.{h}),
            .norm = try r.bf16("final_norm.weight", &.{h}),
            .correction_hidden_norm = try correction.bf16("hidden_norm.weight", &.{h}),
            .correction_embedding_norm = try correction.bf16("embed_norm.weight", &.{h}),
            .correction = .{
                .gate = try correction.bf16("gate_proj.weight", &.{ h, @as(usize, h) * 2 }),
                .up = try correction.bf16("up_proj.weight", &.{ h, @as(usize, h) * 2 }),
                .down = try correction.bf16("down_proj.weight", &.{ h, h }),
            },
        };
    }
};

pub const DraftLayer = struct {
    target_kv: u32,
    target_index: u32,
    heads: mla.HeadRange,
    input_norm: Tensor,
    query_low: Tensor,
    query_norm: Tensor,
    query_high: Tensor,
    kv_low: Tensor,
    kv_norm: Tensor,
    absorbed: mla.Weights,
    gate: Tensor,
    output: Tensor,
    owns_output: bool,
    output_norm: Tensor,
    sinks: [64]f32,
    mlp_norm: Tensor,
    mlp: Mlp(Tensor),

    pub fn read(a: Allocator, model: *checkpoint.Loaded, layer: u32, rank: Rank) !DraftLayer {
        const target = try model.attentionTargetLayer(.draft, layer);
        const t = model.config.value.text_config;
        const s = t.headShape(target);
        const range = try ranks.Range.init(s.heads, rank, "Hy4-flash DFly2 attention heads");
        const heads: mla.HeadRange = .{ .first = @intCast(range.first), .count = @intCast(range.count) };
        const values = @as(usize, s.heads) * s.value;
        const query = @as(usize, s.non_rotary) + s.rotary;
        const value_start = range.first * s.value;
        const value_end = value_start + range.count * s.value;
        const h = t.hidden_size;
        var path: [80]u8 = undefined;
        const r: Reader = .{
            .model = model,
            .prefix = try std.fmt.bufPrint(&path, "model.dfly_layers.{d}.", .{layer}),
        };
        var mlp_path: [88]u8 = undefined;
        const ffn: Reader = .{ .model = model, .prefix = try std.fmt.bufPrint(&mlp_path, "{s}mlp.", .{r.prefix}) };
        const inner = model.config.value.dfly_intermediate_size orelse try std.math.mul(u32, h, 3);
        const o = try r.bf16("self_attn.o_proj.weight", &.{ h, values });
        const output = if (rank.world == 1) o else try core.quant.slice.takeCols(a, o, value_start, value_end);
        errdefer if (rank.world != 1) a.free(output.bytes);
        var absorbed = try mla.read(a, model, .draft, layer, heads);
        errdefer absorbed.deinit(a);
        var all_sinks: [64]f32 = undefined;
        const sink = try model.attentionSink(.draft, layer, &all_sinks);
        var packed_sinks: [64]f32 = @splat(0);
        @memcpy(packed_sinks[0..heads.count], sink[heads.first..][0..heads.count]);
        const q = try r.bf16("self_attn.q_b_proj.weight", &.{ s.queryRows(), t.q_lora_rank });
        const gate = try r.bf16("self_attn.linear_gate.weight", &.{ values, h });
        return .{
            .target_kv = target,
            .target_index = model.config.value.dfly_target_index_layer[layer],
            .heads = heads,
            .input_norm = try r.bf16("input_layernorm.weight", &.{h}),
            .query_low = try r.bf16("self_attn.q_a_proj.weight", &.{ t.q_lora_rank, h }),
            .query_norm = try r.bf16("self_attn.q_a_layernorm.weight", &.{t.q_lora_rank}),
            .query_high = try core.quant.slice.takeRows(a, q, &.{.{ .from = range.first * query, .to = (range.first + range.count) * query }}),
            .kv_low = try r.bf16("self_attn.kv_a_proj_with_mqa.weight", &.{ @as(usize, t.kv_lora_rank) + s.rotary, h }),
            .kv_norm = try r.bf16("self_attn.kv_a_layernorm.weight", &.{t.kv_lora_rank}),
            .absorbed = absorbed,
            .gate = try core.quant.slice.takeRows(a, gate, &.{.{ .from = value_start, .to = value_end }}),
            .output = output,
            .owns_output = rank.world != 1,
            .output_norm = try r.bf16("self_attn.post_proj_norm.weight", &.{h}),
            .sinks = packed_sinks,
            .mlp_norm = try r.bf16("post_attention_layernorm.weight", &.{h}),
            .mlp = try draftMlp(ffn, h, inner),
        };
    }

    pub fn deinit(self: *DraftLayer, a: Allocator) void {
        self.absorbed.deinit(a);
        if (self.owns_output) a.free(self.output.bytes);
        self.* = undefined;
    }
};
