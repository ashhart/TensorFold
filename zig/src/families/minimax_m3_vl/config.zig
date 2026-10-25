//! MiniMax-M3.1-Flash's shape from config.json's text_config (minimax_m3_vl): MoE + the sparse-indexer attention (MSA).

const std = @import("std");

pub const max_layers = 60;

/// config.json's `quantization_config` codebook, in the core reader's order (`core/exl3_format.zig`: mul1 is id 2).
pub const Codebook = enum { unknown, inst3, mcg, mul1 };

pub const Config = struct {
    hidden: u32 = 6144,
    layers: u32 = 60,
    run: u32 = 60, // backbone layers loaded and run: all, or the first few for a check that must fit
    vocab: u32 = 200064,
    head_unit: u32 = 64, // the vocabulary's TP cut in whole units: 128 for EXL3, whose matrices split at 128 columns
    exl3: bool = false, // quant_method "exl3" in config.json, or the standalone quantization_config.json
    qc_avg_bits: f32 = 0, // the quantization_config's avg_bits (3.06 for Mia-AiLab's pack)
    qc_head_bits: u32 = 0, // the language head's own width (head_bits 8: lm_head is the only 8-bit tensor)
    qc_codebook: Codebook = .unknown, // Mia-AiLab's pack: mul1, odd half-bit units, mixed widths per matrix
    eps: f32 = 1e-6,
    dense_layers: u32 = 3, // the leading MoE-free layers: moe_layer_freq's first 1, else first_k_dense_replace
    dense_inter: u32 = 12288,
    experts: u32 = 128,
    topk: u32 = 4,
    moe_inter: u32 = 3072,
    shared_inter: u32 = 3072, // n_shared_experts 1's width: shared_intermediate_size
    routed_scale: f32 = 2.0,
    sigmoid_router: bool = true, // scoring_func "sigmoid": the router scores without a softmax
    routing_bias: bool = true, // use_routing_bias: the router keeps one bias per expert
    norm_topk: bool = false, // no norm_topk_prob in the config: the routed weights stay as scored
    swiglu_alpha: f32 = 1.702, // the swigluoai gate's alpha
    swiglu_limit: f32 = 7.0, // the swigluoai gate's alpha-limit clamp
    attn_heads: u32 = 64,
    kv_heads: u32 = 4, // GQA
    head_dim: u32 = 128,
    rotary_dim: u32 = 64, // partial rotary: rotary_dim 64 of head_dim 128
    partial_rotary: f32 = 0.5,
    qk_norm: bool = true, // use_qk_norm: q and k carry their own per-head norms
    i_heads: u32 = 4, // the MSA indexer's heads (sparse_num_index_heads)
    i_dim: u32 = 128, // the indexer head's dim (sparse_index_dim)
    i_topk: u32 = 16, // whole pooled blocks a row attends (sparse_topk_blocks)
    block: u32 = 128, // the pooled key block's width (sparse_block_size)
    score_max: bool = true, // sparse_score_type "max" over the indexer heads
    tp: u32 = 1, // tensor parallel: the ranks splitting the attention heads
    tp_rank: u32 = 0,

    /// Keys a sparse row attends: the selected pools' whole blocks (stage-2's gather adds the init/local ones).
    pub fn keyWidth(c: *const Config) u32 {
        return c.i_topk * c.block;
    }

    /// The head's rows [lo, hi) this rank computes, in whole units of `head_unit` (EXL3 splits at 128 columns).
    pub fn vocabPart(c: *const Config) [2]u32 {
        const n = c.vocab / c.tp / c.head_unit * c.head_unit;
        return .{ c.tp_rank * n, if (c.tp_rank + 1 == c.tp) c.vocab else (c.tp_rank + 1) * n };
    }
};

fn int(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = o.get(key) orelse return error.BadConfig;
    return switch (v) {
        .integer => |i| i,
        else => error.BadConfig,
    };
}

fn float(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.BadConfig;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn boolOr(o: std.json.ObjectMap, key: []const u8, or_else: bool) bool {
    const v = o.get(key) orelse return or_else;
    return switch (v) {
        .bool => |b| b,
        else => or_else,
    };
}

fn stringIs(o: std.json.ObjectMap, key: []const u8, want: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, want);
}

/// config.json's quantization_config (top level, else text_config's): glm/cuda_exl3.zig's block().
fn quantBlock(root: std.json.Value) ?std.json.ObjectMap {
    if (root != .object) return null;
    if (root.object.get("quantization_config")) |q| if (q == .object) return q.object;
    if (root.object.get("text_config")) |t| if (t == .object) if (t.object.get("quantization_config")) |q| if (q == .object) return q.object;
    return null;
}

fn codebook(name: []const u8) Codebook {
    if (std.mem.eql(u8, name, "mul1")) return .mul1;
    if (std.mem.eql(u8, name, "mcg")) return .mcg;
    if (std.mem.eql(u8, name, "inst3")) return .inst3;
    return .unknown;
}

/// One quantization_config's exl3 detection: the method widens the vocab TP cut to 128, the fields feed a later gate.
fn detectExl3(c: *Config, q: std.json.ObjectMap) void {
    if (!stringIs(q, "quant_method", "exl3")) return;
    c.exl3 = true;
    c.head_unit = 128;
    if (q.get("avg_bits")) |v| if (v == .float or v == .integer) {
        c.qc_avg_bits = @floatCast(float(q, "avg_bits") catch return);
    };
    if (q.get("head_bits")) |v| if (v == .integer) {
        if (v.integer > 0) c.qc_head_bits = @intCast(v.integer);
    };
    if (q.get("codebook")) |v| if (v == .string) {
        c.qc_codebook = codebook(v.string);
    };
}

/// text_config with M3.1-Flash's widths; smaller synthetic counts pass; `qc_json` reads a sibling quant config's JSON.
pub fn parse(gpa: std.mem.Allocator, json: []const u8, standalone_qc: ?[]const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const t = (root.get("text_config") orelse parsed.value).object;
    var c = Config{};
    const want = Config{};
    c.hidden = @intCast(try int(t, "hidden_size"));
    c.layers = @intCast(try int(t, "num_hidden_layers"));
    c.run = c.layers;
    c.vocab = @intCast(try int(t, "vocab_size"));
    c.attn_heads = @intCast(try int(t, "num_attention_heads"));
    c.kv_heads = @intCast(try int(t, "num_key_value_heads"));
    c.head_dim = @intCast(try int(t, "head_dim"));
    c.eps = @floatCast(try float(t, "rms_norm_eps"));
    c.experts = @intCast(try int(t, "num_local_experts"));
    c.topk = @intCast(try int(t, "num_experts_per_tok"));
    c.moe_inter = @intCast(try int(t, "intermediate_size"));
    c.dense_inter = @intCast(try int(t, "dense_intermediate_size"));
    c.shared_inter = @intCast(try int(t, "shared_intermediate_size"));
    c.routed_scale = @floatCast(try float(t, "routed_scaling_factor"));
    c.swiglu_alpha = @floatCast(try float(t, "swiglu_alpha"));
    c.swiglu_limit = @floatCast(try float(t, "swiglu_limit"));
    c.sigmoid_router = stringIs(t, "scoring_func", "sigmoid");
    c.routing_bias = boolOr(t, "use_routing_bias", false);
    c.qk_norm = boolOr(t, "use_qk_norm", false);
    if (t.get("qk_norm_type")) |v| {
        if (!(v == .string and std.mem.eql(u8, v.string, "per_head"))) return error.NotMiniMaxM3VL;
    } else if (c.qk_norm) return error.NotMiniMaxM3VL; // the per-head kind is this family's only one
    if (c.swiglu_limit <= 0) return error.NotMiniMaxM3VL; // the swigluoai gate's alpha-limit clamp
    if (t.get("rotary_dim")) |v| {
        if (v != .integer) return error.BadConfig;
        c.rotary_dim = @intCast(v.integer);
    }
    if (t.get("partial_rotary_factor")) |v| {
        if (v != .float and v != .integer) return error.BadConfig;
        c.partial_rotary = @floatCast(try float(t, "partial_rotary_factor"));
    }
    if (t.get("moe_layer_freq")) |v| {
        if (v != .array) return error.BadConfig;
        var dense: u32 = c.layers; // the sentinel: no MoE wave seen yet
        for (v.array.items, 0..) |e, i| {
            if (e != .integer) return error.BadConfig;
            if (e.integer == 1) {
                if (dense == c.layers) dense = @intCast(i);
            } else if (e.integer != 0) return error.NotMiniMaxM3VL;
            if (dense != c.layers and e.integer == 0) return error.NotMiniMaxM3VL; // one dense->MoE wave only
        }
        c.dense_layers = @min(dense, c.layers);
    } else {
        c.dense_layers = @intCast(try int(t, "first_k_dense_replace")); // the deepseek spelling's fallback
    }
    if (t.get("sparse_attention_config")) |v| {
        if (v != .object) return error.BadConfig;
        const s = v.object;
        c.i_heads = @intCast(try int(s, "sparse_num_index_heads"));
        c.i_dim = @intCast(try int(s, "sparse_index_dim"));
        c.i_topk = @intCast(try int(s, "sparse_topk_blocks"));
        c.block = @intCast(try int(s, "sparse_block_size"));
        c.score_max = stringIs(s, "sparse_score_type", "max");
    } else {
        // the GLM DSA indexer's flat spellings: the same machinery's token-topk shape, expressed in blocks
        c.i_heads = @intCast(try int(t, "index_n_heads"));
        c.i_dim = @intCast(try int(t, "index_head_dim"));
        const kpool = try int(t, "index_kpool");
        if (kpool <= 0) return error.BadConfig;
        c.block = @intCast(kpool);
        c.i_topk = @intCast(@divExact(try int(t, "index_topk"), kpool)); // whole pools only: a partial one refuses
    }
    var qc_parsed: ?std.json.Parsed(std.json.Value) = null;
    defer if (qc_parsed) |p| p.deinit();
    var qc: ?std.json.ObjectMap = quantBlock(parsed.value);
    if (qc == null) {
        if (standalone_qc) |s| {
            qc_parsed = try std.json.parseFromSlice(std.json.Value, gpa, s, .{});
            const p = qc_parsed.?;
            if (p.value == .object) {
                qc = p.value.object;
            } else {
                qc_parsed = null;
            }
        }
    }
    if (qc) |q| detectExl3(&c, q);
    if (c.layers > max_layers) return error.NotMiniMaxM3VL;
    const widths = c.hidden == want.hidden and c.dense_inter == want.dense_inter and c.topk == want.topk and
        c.moe_inter == want.moe_inter and c.shared_inter == want.shared_inter and c.attn_heads == want.attn_heads and
        c.kv_heads == want.kv_heads and c.head_dim == want.head_dim and c.i_heads == want.i_heads and
        c.i_dim == want.i_dim and c.i_topk == want.i_topk and c.block == want.block and
        c.eps == want.eps and c.swiglu_alpha == want.swiglu_alpha;
    const counts = c.layers >= 1 and c.dense_layers <= c.layers and c.experts >= c.topk and
        c.experts <= want.experts and c.vocab <= want.vocab and c.vocab % 128 == 0;
    if (!widths or !counts) return error.NotMiniMaxM3VL;
    return c;
}

/// MiniMax-M3.1-Flash's own counts: the shapes the engines' kernels are built for (smaller synthetic ones pass).
pub fn whole(c: Config) bool {
    const want = Config{};
    return c.layers == want.layers and c.vocab == want.vocab and c.dense_layers == want.dense_layers and c.experts == want.experts;
}

test "the MiniMax-M3.1-Flash EXL3 pack's config.json parses and carries the model_type the registry dispatches" {
    const gpa = std.testing.allocator;
    const json = @embedFile("test_pack_config.json");
    var root = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer root.deinit();
    try std.testing.expectEqualStrings("minimax_m3_vl", root.value.object.get("model_type").?.string);
    const c = try parse(gpa, json, null);
    try std.testing.expect(whole(c));
    try std.testing.expect(c.exl3);
    try std.testing.expectEqual(Codebook.mul1, c.qc_codebook); // mul1: the core reader's codebook id 2
    try std.testing.expectEqual(@as(f32, 3.06), c.qc_avg_bits);
    try std.testing.expectEqual(@as(u32, 8), c.qc_head_bits); // head_bits 8: only lm_head carries it
    try std.testing.expectEqual(@as(u32, 128), c.head_unit); // the vocabulary's TP cut once EXL3 is on
    try std.testing.expectEqual(@as(u32, 60), c.layers);
    try std.testing.expectEqual(@as(u32, 3), c.dense_layers); // moe_layer_freq's wave turns MoE from layer 3
    try std.testing.expectEqual(@as(u32, 128), c.experts);
    try std.testing.expectEqual(@as(u32, 4), c.topk);
    try std.testing.expectEqual(@as(u32, 3072), c.moe_inter);
    try std.testing.expectEqual(@as(u32, 12288), c.dense_inter);
    try std.testing.expectEqual(@as(f32, 1.702), c.swiglu_alpha); // the swigluoai gate
    try std.testing.expectEqual(@as(f32, 7.0), c.swiglu_limit);
    try std.testing.expect(c.sigmoid_router and c.routing_bias); // sigmoid scores plus a bias per expert
    try std.testing.expect(!c.norm_topk);
    try std.testing.expect(c.qk_norm); // use_qk_norm with qk_norm_type "per_head"
    try std.testing.expectEqual(@as(u32, 4), c.kv_heads); // GQA
    try std.testing.expectEqual(@as(u32, 64), c.rotary_dim);
    try std.testing.expectEqual(@as(u32, 4), c.i_heads); // the MSA indexer: 4 heads of 128, 16 blocks of 128, max
    try std.testing.expectEqual(@as(u32, 128), c.i_dim);
    try std.testing.expectEqual(@as(u32, 16), c.i_topk);
    try std.testing.expectEqual(@as(u32, 128), c.block);
    try std.testing.expect(c.score_max);
    try std.testing.expectEqual(@as(u32, 2048), c.keyWidth());
    try std.testing.expectEqual([2]u32{ 0, 200064 }, c.vocabPart());
}

test "a standalone quantization_config.json admits the same EXL3 gate when config.json's inline copy is absent" {
    const gpa = std.testing.allocator;
    const qc = @embedFile("test_quantization_config.json");
    const noqc = try std.fmt.allocPrint(gpa, "{{\"model_type\": \"minimax_m3_vl\", \"text_config\": {{" ++
        "\"hidden_size\": 6144, \"num_hidden_layers\": 2, \"vocab_size\": 2048, \"num_attention_heads\": 64," ++
        "\"num_key_value_heads\": 4, \"head_dim\": 128, \"rms_norm_eps\": 1e-06, \"num_local_experts\": 4," ++
        "\"num_experts_per_tok\": 4, \"intermediate_size\": 3072, \"dense_intermediate_size\": 12288," ++
        "\"shared_intermediate_size\": 3072, \"routed_scaling_factor\": 2.0, \"swiglu_alpha\": 1.702," ++
        "\"swiglu_limit\": 7.0, \"use_qk_norm\": true, \"qk_norm_type\": \"per_head\", \"moe_layer_freq\": [0, 1]," ++
        "\"sparse_attention_config\": {{\"sparse_num_index_heads\": 4, \"sparse_index_dim\": 128," ++
        "\"sparse_topk_blocks\": 16, \"sparse_block_size\": 128, \"sparse_score_type\": \"max\"}}}}}}", .{});
    defer gpa.free(noqc);
    const c = try parse(gpa, noqc, qc);
    try std.testing.expect(c.exl3);
    try std.testing.expectEqual(Codebook.mul1, c.qc_codebook);
    try std.testing.expectEqual(@as(f32, 3.06), c.qc_avg_bits);
    try std.testing.expectEqual(@as(u32, 128), c.head_unit);
    const d = try parse(gpa, noqc, null); // without the standalone summary there is no EXL3 either
    try std.testing.expect(!d.exl3);
    try std.testing.expectEqual(Codebook.unknown, d.qc_codebook);
    try std.testing.expectEqual(@as(u32, 64), d.head_unit);
}

test "the GLM DSA indexer's flat spellings also parse, and other widths do not" {
    const gpa = std.testing.allocator;
    const flat = try std.fmt.allocPrint(gpa, "{{\"text_config\": {{\"hidden_size\": 6144, \"num_hidden_layers\": 2," ++
        "\"vocab_size\": 2048, \"num_attention_heads\": 64, \"num_key_value_heads\": 4, \"head_dim\": 128," ++
        "\"rms_norm_eps\": 1e-06, \"num_local_experts\": 4, \"num_experts_per_tok\": 4, \"intermediate_size\": 3072," ++
        "\"dense_intermediate_size\": 12288, \"shared_intermediate_size\": 3072, \"routed_scaling_factor\": 2.0," ++
        "\"swiglu_alpha\": 1.702, \"swiglu_limit\": 7.0, \"use_qk_norm\": true, \"qk_norm_type\": \"per_head\"," ++
        "\"first_k_dense_replace\": 0, \"index_n_heads\": 4, \"index_head_dim\": 128, \"index_topk\": 2048," ++
        "\"index_kpool\": 128}}}}", .{});
    defer gpa.free(flat);
    const c = try parse(gpa, flat, null);
    try std.testing.expectEqual(@as(u32, 4), c.i_heads);
    try std.testing.expectEqual(@as(u32, 16), c.i_topk); // 2048 tokens of a 128-wide pool: 16 blocks
    try std.testing.expectEqual(@as(u32, 128), c.block);
    try std.testing.expectEqual(@as(u32, 0), c.dense_layers); // first_k_dense_replace's fallback
    const wide = try std.fmt.allocPrint(gpa, "{{\"text_config\": {{\"hidden_size\": 4096, \"num_hidden_layers\": 2," ++
        "\"vocab_size\": 2048, \"num_attention_heads\": 64, \"num_key_value_heads\": 4, \"head_dim\": 128," ++
        "\"rms_norm_eps\": 1e-06, \"num_local_experts\": 4, \"num_experts_per_tok\": 4, \"intermediate_size\": 3072," ++
        "\"dense_intermediate_size\": 12288, \"shared_intermediate_size\": 3072, \"routed_scaling_factor\": 2.0," ++
        "\"swiglu_alpha\": 1.702, \"swiglu_limit\": 7.0, \"use_qk_norm\": true, \"qk_norm_type\": \"per_head\"," ++
        "\"moe_layer_freq\": [0, 1], \"sparse_attention_config\": {{\"sparse_num_index_heads\": 4," ++
        "\"sparse_index_dim\": 128, \"sparse_topk_blocks\": 16, \"sparse_block_size\": 128," ++
        "\"sparse_score_type\": \"max\"}}}}}}", .{});
    defer gpa.free(wide);
    try std.testing.expectError(error.NotMiniMaxM3VL, parse(gpa, wide, null));
}