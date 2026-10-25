//! Qwen4 dense admission. Kernel observations never supply missing checkpoint dimensions.
const std = @import("std");
const f = @import("fields.zig");
const q = @import("quant.zig");
pub const Problem = f.Problem;
pub const Kind = enum { linear_attention, indexed_attention };
pub const Gate = enum { sigmoid, silu };
pub const NormWeights = enum { hf_gamma, scale };
pub const max_layers = 512;
pub const Config = struct {
    doc: std.json.Parsed(std.json.Value),
    root: f.Object,
    text: f.Object,
    hidden: usize,
    vocab: usize,
    layers: usize,
    intermediate: usize,
    context: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: usize,
    theta: f64,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv: usize,
    streams: usize,
    lowrank: usize,
    eps: f32,
    gate: Gate,
    norm_weights: NormWeights = .hf_gamma,
    norm_explicit: bool = false,
    index_heads: usize,
    index_dim: usize,
    index_budget: usize,
    index_ratio: usize,
    kinds: [max_layers]Kind = undefined,
    ple: [max_layers]bool = @splat(false),
    ple_count: usize = 0,
    ple_dim: usize,
    ple_conv: usize,
    ngram: usize,
    ngram_heads: usize,
    ngram_vocab: usize,
    ngram_parts: usize,
    ngram_divisor: usize,
    seed: u64,
    eos: [16]u32 = @splat(0),
    eos_count: usize = 0,
    ple_eos: u32 = 0,
    tie_embeddings: bool,
    quant: q.Config,

    pub fn deinit(c: *Config) void {
        c.doc.deinit();
        c.* = undefined;
    }
    pub fn wide(c: *const Config) usize {
        return c.hidden * c.streams;
    }
    pub fn keyWidth(c: *const Config) usize {
        return c.key_heads * c.key_dim;
    }
    pub fn valueWidth(c: *const Config) usize {
        return c.value_heads * c.value_dim;
    }
    pub fn convWidth(c: *const Config) usize {
        return 2 * c.keyWidth() + c.valueWidth();
    }
};

pub fn parse(a: std.mem.Allocator, bytes: []const u8, p: *Problem) !Config {
    p.len = 0;
    var doc = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    errdefer doc.deinit();
    const root = try f.object(doc.value, "config.json", p);
    const text = if (root.get("text_config")) |v| try f.object(v, "text_config", p) else root;
    const tag = root.get("model_type") orelse return p.reject("model_type");
    if (!f.word(tag, "qwen4") and !f.word(tag, "qwen4_text")) return p.reject("model_type");
    if (root.get("text_config") != null) {
        const t = text.get("model_type") orelse return p.reject("text_config.model_type");
        if (!f.word(t, "qwen4_text")) return p.reject("text_config.model_type");
    }
    if (root.get("vision_config")) |v| if (v != .null) return p.reject("vision_config");
    if (text.get("vision_config")) |v| if (v != .null) return p.reject("vision_config");
    if (root.get("text_config") != null) try f.known(root, &.{
        "model_type",           "text_config",   "vision_config",       "architectures", "torch_dtype",         "dtype",
        "transformers_version", "_name_or_path", "_commit_hash",        "quantization",  "quantization_config", "eos_token_id",
        "bos_token_id",         "pad_token_id",  "tie_word_embeddings",
    }, p);
    try f.known(text, &.{
        "model_type",                      "architectures",       "torch_dtype",           "dtype",                              "transformers_version",    "_name_or_path",         "_commit_hash",
        "hidden_size",                     "vocab_size",          "num_hidden_layers",     "intermediate_size",                  "max_position_embeddings", "rms_norm_eps",          "hidden_act",
        "attention_bias",                  "attention_dropout",   "attn_output_gate",      "use_cache",                          "num_attention_heads",     "num_key_value_heads",   "head_dim",
        "rope_parameters",                 "rope_scaling",        "linear_num_key_heads",  "linear_num_value_heads",             "linear_key_head_dim",     "linear_value_head_dim", "linear_conv_kernel_dim",
        "mamba_ssm_dtype",                 "hc_count",            "hc_lowrank",            "indexer_n_heads",                    "indexer_kv_heads",        "indexer_head_dim",      "indexer_budget",
        "indexer_compress_ratio",          "output_gate_type",    "layer_types",           "full_attention_interval",            "ple_layer_ids",           "ple_embed_dim",         "ple_conv_kernel_size",
        "ngram_size",                      "heads_per_ngram",     "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "split_ngram_parts",       "seed",                  "eos_token_id",
        "bos_token_id",                    "pad_token_id",        "tie_word_embeddings",   "initializer_range",                  "num_experts",             "num_experts_per_tok",   "moe_intermediate_size",
        "shared_expert_intermediate_size", "decoder_sparse_step", "mlp_only_layers",       "mlp_layer_types",                    "mtp_num_hidden_layers",   "mtp",                   "quantization",
        "quantization_config",             "vision_config",       "norm_weight_mode",
    }, p);

    for ([_][]const u8{ "num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size", "decoder_sparse_step", "mlp_only_layers", "rope_scaling" }) |name| {
        if (text.get(name)) |v| if (v != .null) return p.reject(name);
    }
    // No dense MTP schema or head tensor map is in the supplied source. Refuse rather than silently drop a head.
    if (try f.natural(text, "mtp_num_hidden_layers", 0, p) != 0) return p.reject("mtp_num_hidden_layers");
    if (text.get("mtp")) |v| if (v != .null) return p.reject("mtp");
    try f.fixed(text, "hidden_act", .{ .string = "silu" }, p);
    try f.fixed(text, "attention_bias", .{ .bool = false }, p);
    try f.fixed(text, "attn_output_gate", .{ .bool = true }, p);
    try f.fixed(text, "mamba_ssm_dtype", .{ .string = "float32" }, p);
    if (text.get("attention_dropout")) |v| if (try f.number(v, "attention_dropout", p) != 0) return p.reject("attention_dropout");
    const h = try f.int(text, "hidden_size", p);
    const nh = try f.int(text, "num_attention_heads", p);
    const nkv = try f.int(text, "num_key_value_heads", p);
    const hd = try f.int(text, "head_dim", p);
    const rope = if (text.get("rope_parameters")) |v| try f.object(v, "rope_parameters", p) else return p.reject("rope_parameters");
    try f.known(rope, &.{ "rope_type", "type", "factor", "partial_rotary_factor", "rope_theta", "mrope_section" }, p);
    try f.fixed(rope, "rope_type", .{ .string = "default" }, p);
    try f.fixed(rope, "type", .{ .string = "default" }, p);
    if (rope.get("factor")) |v| if (try f.number(v, "rope_parameters.factor", p) != 1) return p.reject("rope_parameters.factor");
    // Text-only positions have one axis. A nontrivial multi-axis mapping cannot be ignored.
    if (rope.get("mrope_section")) |v| if (v != .null) return p.reject("rope_parameters.mrope_section");
    const part = try f.number(rope.get("partial_rotary_factor") orelse return p.reject("partial_rotary_factor"), "partial_rotary_factor", p);
    const theta = try f.number(rope.get("rope_theta") orelse return p.reject("rope_theta"), "rope_theta", p);
    if (part < 0 or part > 1 or theta <= 0) return p.reject("rope_parameters");
    const rd = @as(f64, @floatFromInt(hd)) * part;
    if (rd != @floor(rd) or rd > @as(f64, @floatFromInt(std.math.maxInt(u32)))) return p.reject("partial_rotary_factor");
    const rotary: usize = @intFromFloat(rd);
    const eps = try f.number(text.get("rms_norm_eps") orelse return p.reject("rms_norm_eps"), "rms_norm_eps", p);
    if (eps <= 0 or eps > std.math.floatMax(f32) or @as(f32, @floatCast(eps)) == 0) return p.reject("rms_norm_eps");
    const gate = text.get("output_gate_type") orelse .null;
    const norm_weights = text.get("norm_weight_mode");
    const root_tied = try f.flag(root, "tie_word_embeddings", false, p);
    const tied = try f.flag(text, "tie_word_embeddings", root_tied, p);
    if (text.get("tie_word_embeddings") != null and root.get("tie_word_embeddings") != null and tied != root_tied)
        return p.reject("tie_word_embeddings");
    var c: Config = .{
        .doc = doc,
        .root = root,
        .text = text,
        .hidden = h,
        .vocab = try f.int(text, "vocab_size", p),
        .layers = try f.int(text, "num_hidden_layers", p),
        .intermediate = try f.int(text, "intermediate_size", p),
        .context = try f.int(text, "max_position_embeddings", p),
        .heads = nh,
        .kv_heads = nkv,
        .head_dim = hd,
        .rotary = rotary,
        .theta = theta,
        .eps = @floatCast(eps),
        .key_heads = try f.int(text, "linear_num_key_heads", p),
        .value_heads = try f.int(text, "linear_num_value_heads", p),
        .key_dim = try f.int(text, "linear_key_head_dim", p),
        .value_dim = try f.int(text, "linear_value_head_dim", p),
        .conv = try f.int(text, "linear_conv_kernel_dim", p),
        .streams = try f.int(text, "hc_count", p),
        .lowrank = try f.int(text, "hc_lowrank", p),
        .index_heads = try f.int(text, "indexer_n_heads", p),
        .index_dim = try f.int(text, "indexer_head_dim", p),
        .index_budget = try f.int(text, "indexer_budget", p),
        .index_ratio = try f.int(text, "indexer_compress_ratio", p),
        .gate = if (gate == .null or f.word(gate, "silu")) .silu else if (f.word(gate, "sigmoid")) .sigmoid else return p.reject("output_gate_type"),
        .norm_weights = if (norm_weights) |v| if (f.word(v, "hf_gamma")) .hf_gamma else if (f.word(v, "scale")) .scale else return p.reject("norm_weight_mode") else .hf_gamma,
        .norm_explicit = norm_weights != null,
        .ple_dim = try f.int(text, "ple_embed_dim", p),
        .ple_conv = try f.int(text, "ple_conv_kernel_size", p),
        .ngram = try f.int(text, "ngram_size", p),
        .ngram_heads = try f.int(text, "heads_per_ngram", p),
        .ngram_vocab = try f.int(text, "ngram_vocab_size_base", p),
        .ngram_parts = try f.int(text, "split_ngram_parts", p),
        .ngram_divisor = try f.int(text, "make_ngram_vocab_size_divisible_by", p),
        .seed = try f.natural(text, "seed", 1234, p),
        .tie_embeddings = tied,
        .quant = try q.parse(root, text, p),
    };
    if (c.layers > max_layers) return p.reject("num_hidden_layers");
    if (nh % nkv != 0) return p.reject("num_key_value_heads");
    if (c.value_heads % c.key_heads != 0) return p.reject("linear_num_value_heads");
    if (c.streams <= 1) return p.reject("hc_count");
    if (c.conv < 2) return p.reject("linear_conv_kernel_dim");
    if (try f.int(text, "indexer_kv_heads", p) != 1) return p.reject("indexer_kv_heads");
    if (c.index_budget % c.index_ratio != 0) return p.reject("indexer_budget");
    if (rotary % 2 != 0 or rotary > hd or rotary > c.index_dim) return p.reject("partial_rotary_factor");
    // Check all products before any runtime-sized state or tensor allocation.
    for ([_][2]usize{ .{ h, c.streams }, .{ c.key_heads, c.key_dim }, .{ c.value_heads, c.value_dim }, .{ nh, hd }, .{ nkv, hd }, .{ c.intermediate, h }, .{ c.vocab, h }, .{ c.layers, c.context } }) |pair| {
        _ = std.math.mul(usize, pair[0], pair[1]) catch return p.reject("dimensions");
    }
    _ = std.math.add(usize, c.valueWidth(), std.math.mul(usize, 2, c.keyWidth()) catch return p.reject("linear_key_head_dim")) catch return p.reject("linear_value_head_dim");
    const signed_max: usize = std.math.maxInt(i32);
    if (c.wide() > signed_max or c.convWidth() > signed_max or (nh * hd) > signed_max / 2 or c.context > signed_max or c.layers > signed_max) return p.reject("dimensions");
    const index_width = std.math.mul(usize, std.math.add(usize, c.index_heads, 1) catch return p.reject("indexer_n_heads"), c.index_dim) catch return p.reject("indexer_head_dim");
    if (index_width > signed_max or c.lowrank > signed_max or c.intermediate > signed_max or c.vocab > signed_max) return p.reject("dimensions");
    _ = std.math.mul(usize, c.wide(), c.lowrank) catch return p.reject("hc_lowrank");
    _ = std.math.mul(usize, c.wide(), c.context) catch return p.reject("max_position_embeddings");
    const kinds = text.get("layer_types") orelse return p.reject("layer_types");
    if (kinds != .array or kinds.array.items.len != c.layers) return p.reject("layer_types");
    for (kinds.array.items, 0..) |v, i| {
        c.kinds[i] = if (f.word(v, "linear_attention")) .linear_attention else if (f.word(v, "indexed_attention") or f.word(v, "full_attention")) .indexed_attention else return p.reject("layer_types");
    }

    if (text.get("full_attention_interval")) |v| {
        const interval = try f.int(text, "full_attention_interval", p);
        _ = v;
        for (c.kinds[0..c.layers], 0..) |kind, i| {
            if ((kind == .indexed_attention) != ((i + 1) % interval == 0)) return p.reject("full_attention_interval");
        }
    }
    if (text.get("mlp_layer_types")) |v| {
        if (v != .array or v.array.items.len != c.layers) return p.reject("mlp_layer_types");
        for (v.array.items) |kind| if (!f.word(kind, "dense")) return p.reject("mlp_layer_types");
    }
    const ple = text.get("ple_layer_ids") orelse return p.reject("ple_layer_ids");
    if (ple != .array) return p.reject("ple_layer_ids");
    for (ple.array.items) |v| {
        if (v != .integer or v.integer <= 0 or v.integer > c.layers) return p.reject("ple_layer_ids");
        const i: usize = @intCast(v.integer - 1);
        if (c.kinds[i] != .linear_attention or c.ple[i]) return p.reject("ple_layer_ids");
        c.ple[i] = true;
        c.ple_count += 1;
    }
    if (c.ngram < 2 or c.ple_conv < 2) return p.reject("ngram_size");
    if (c.ngram > 16) return p.reject("ngram_size");
    if (c.ngram_heads > 16) return p.reject("heads_per_ngram");
    if (c.ngram_vocab > std.math.maxInt(u32) or c.ngram_parts > 65536 or c.ngram_divisor > 65536) return p.reject("ngram_vocab_size_base");
    const ng = std.math.mul(usize, c.ngram - 1, c.ngram_heads) catch return p.reject("heads_per_ngram");
    if (c.ple_dim % ng != 0) return p.reject("ple_embed_dim");
    try readEos(root.get("eos_token_id") orelse text.get("eos_token_id"), c.vocab, &c.eos, &c.eos_count, p, "eos_token_id");
    var lexical: [16]u32 = @splat(0);
    var lexical_count: usize = 0;
    const lexical_name: []const u8 = if (root.get("text_config") != null) "text_config.eos_token_id" else "eos_token_id";
    try readEos(text.get("eos_token_id"), c.vocab, &lexical, &lexical_count, p, lexical_name);
    if (c.ple_count > 0 and lexical_count == 0) return p.reject(lexical_name);
    if (lexical_count > 0) c.ple_eos = lexical[0];
    return c;
}

fn readEos(value: ?std.json.Value, vocab: usize, out: *[16]u32, count: *usize, p: *Problem, field: []const u8) !void {
    const v = value orelse return;
    if (v == .null) return;
    const ids = if (v == .array) v.array.items else &.{v};
    if (ids.len > out.len) return p.reject(field);
    for (ids) |id| {
        if (id != .integer or id.integer < 0 or id.integer >= vocab or id.integer > std.math.maxInt(u32)) return p.reject(field);
        out[count.*] = @intCast(id.integer);
        count.* += 1;
    }
}
