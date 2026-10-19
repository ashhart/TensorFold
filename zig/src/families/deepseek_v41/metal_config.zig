//! Metal capability view of the shared typed config; no independent checkpoint parser.
const std = @import("std");
const shared = @import("config.zig");
const unsigned = shared.unsigned;

pub const Rope = struct {
    rope_type: []const u8,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position_embeddings: u32,
};

pub const Text = struct {
    vocab_size: u32,
    hidden_size: u32,
    moe_intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    qk_rope_head_dim: u32,
    q_lora_rank: u32,
    o_lora_rank: u32,
    o_groups: u32,
    swiglu_limit: f32,
    rms_norm_eps: f32,
    max_position_embeddings: u32,
    rope_theta: f32,
    rope_scaling: Rope,
    n_routed_experts: u32,
    n_shared_experts: u32,
    num_experts_per_tok: u32,
    scoring_func: []const u8,
    topk_method: []const u8,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    sliding_window: u32,
    compress_ratios: []const u32,
    compress_rope_theta: f32,
    kv_source_layer_ids: []const u32,
    index_source_layer_ids: []const u32,
    index_n_heads: u32,
    index_head_dim: u32,
    index_topk: u32,
    candidate_source_layer_id: u32,
    candidate_topk_blocks: u32,
    candidate_block_size: u32,
    hc_mult: u32,
    hc_sinkhorn_iters: u32,
    hc_eps: f32,
    engram_layer_ids: []const u32,
    engram_num_embeddings: []const u64,
    engram_max_ngram_size: u32,
    engram_vocab_size: u32,
    engram_n_heads: u32,
    engram_head_dim: u32,
    engram_pad_token_id: u32,
    engram_compressed_vocab_size: u32,
    num_nextn_predict_layers: u32,
};

pub const Config = struct {
    model_type: []const u8,
    text_config: Text,
    quantization_config: struct { quant_method: []const u8, version: []const u8 },
    engram_table_dir: ?[]const u8 = null,

    pub fn load(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(arena, &.{ dir, "config.json" });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 24));
        return parse(arena, bytes);
    }

    pub fn parse(arena: std.mem.Allocator, bytes: []const u8) !Config {
        const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{ .parse_numbers = false });
        const text = try field(root, "text_config");
        if (text != .object) return error.UnexpectedToken;
        const quant = try field(root, "quantization_config");
        inline for (@typeInfo(Text).@"struct".fields) |f| _ = try field(text, f.name);
        const c = try arena.create(shared.Config);
        c.* = try shared.Config.parse(arena, bytes);
        try c.validateRelease();
        const result: Config = .{
            .model_type = try string(try field(root, "model_type")),
            .quantization_config = .{ .quant_method = try string(try field(quant, "quant_method")), .version = try string(try field(quant, "version")) },
            .engram_table_dir = if (root.object.get("engram_table_dir")) |v| (if (v == .null) null else try string(v)) else null,
            .text_config = .{
                .vocab_size = c.vocab,
                .hidden_size = c.hidden,
                .moe_intermediate_size = c.expert_width,
                .num_hidden_layers = c.layers,
                .num_attention_heads = c.heads,
                .num_key_value_heads = try unsigned(u32, try field(text, "num_key_value_heads")),
                .head_dim = c.head_dim,
                .qk_rope_head_dim = c.rope_dim,
                .q_lora_rank = c.q_lora,
                .o_lora_rank = c.o_lora,
                .o_groups = c.o_groups,
                .swiglu_limit = c.swiglu_limit,
                .rms_norm_eps = c.eps,
                .max_position_embeddings = @intCast(c.rope.max_positions),
                .rope_theta = @floatCast(c.rope.theta),
                .rope_scaling = .{ .rope_type = if (c.rope.yarn) "yarn" else "default", .factor = @floatCast(c.rope.factor), .beta_fast = @floatCast(c.rope.beta_fast), .beta_slow = @floatCast(c.rope.beta_slow), .original_max_position_embeddings = @intCast(c.rope.original_max_positions) },
                .n_routed_experts = c.experts,
                .n_shared_experts = c.shared_experts,
                .num_experts_per_tok = c.topk,
                .scoring_func = "sqrtsoftplus",
                .topk_method = "noaux_tc",
                .norm_topk_prob = c.norm_topk_prob,
                .routed_scaling_factor = c.routed_scale,
                .sliding_window = c.window,
                .compress_ratios = c.compress_ratios.items(),
                .compress_rope_theta = @floatCast(c.rope.compress_theta),
                .kv_source_layer_ids = c.kv_sources.items(),
                .index_source_layer_ids = c.index_sources.items(),
                .index_n_heads = c.index_heads,
                .index_head_dim = c.index_dim,
                .index_topk = c.index_topk,
                .candidate_source_layer_id = c.candidate_source,
                .candidate_topk_blocks = c.candidate_blocks,
                .candidate_block_size = c.candidate_block_size,
                .hc_mult = c.hc_mult,
                .hc_sinkhorn_iters = c.hc_sinkhorn_iters,
                .hc_eps = c.hc_eps,
                .engram_layer_ids = c.engram_layers.items(),
                .engram_num_embeddings = c.engram_rows.items(),
                .engram_max_ngram_size = c.engram_max_ngram,
                .engram_vocab_size = c.engram_vocab,
                .engram_n_heads = c.engram_heads,
                .engram_head_dim = c.engram_head_dim,
                .engram_pad_token_id = c.engram_pad,
                .engram_compressed_vocab_size = c.engram_compressed_vocab,
                .num_nextn_predict_layers = c.mtp_layers,
            },
        };
        try result.validate();
        return result;
    }

    pub fn validate(c: Config) !void {
        const t = c.text_config;
        if (!std.mem.eql(u8, c.model_type, "deepseek_v41") or
            !std.mem.eql(u8, c.quantization_config.quant_method, "exl3")) return error.UnsupportedCheckpoint;
        if (t.hidden_size != 5120 or t.moe_intermediate_size != 2304 or t.num_hidden_layers != 40 or
            t.num_attention_heads != 64 or t.num_key_value_heads != 1 or t.head_dim != 512 or
            t.qk_rope_head_dim != 64 or t.q_lora_rank != 1280 or t.o_lora_rank != 1024 or t.o_groups != 8 or
            t.vocab_size != 129280 or t.max_position_embeddings != 1048576) return error.UnsupportedArchitecture;
        if (t.n_routed_experts != 384 or t.n_shared_experts != 1 or t.num_experts_per_tok != 6 or
            !std.mem.eql(u8, t.scoring_func, "sqrtsoftplus") or !std.mem.eql(u8, t.topk_method, "noaux_tc") or
            !t.norm_topk_prob or t.routed_scaling_factor != 1.5 or t.swiglu_limit != 10 or
            t.sliding_window != 128 or t.hc_mult != 4 or t.hc_sinkhorn_iters != 20 or
            t.index_n_heads != 32 or t.index_head_dim != 128 or t.index_topk != 512 or
            t.candidate_source_layer_id != 20 or t.candidate_topk_blocks != 2048 or
            t.candidate_block_size != 8) return error.UnsupportedArchitecture;
        if (!std.mem.eql(u32, t.kv_source_layer_ids, &.{ 2, 8, 14, 20 }) or
            !std.mem.eql(u32, t.index_source_layer_ids, &.{ 2, 8, 14, 20, 24, 28, 32, 36 }) or
            !std.mem.eql(u32, t.engram_layer_ids, &.{ 1, 14 }) or t.compress_ratios.len != 43 or
            !std.mem.eql(u64, t.engram_num_embeddings, &.{ 384006168, 384016682 })) return error.UnsupportedCacheLayout;
        for (t.compress_ratios, 0..) |ratio, layer| {
            const expected: u32 = if (layer < 2 or layer >= 40) 0 else if (layer < 20) 2 else 1;
            if (ratio != expected) return error.UnsupportedCacheLayout;
        }
        if (t.engram_vocab_size != 16000000 or
            t.engram_max_ngram_size != 4 or t.engram_n_heads != 8 or t.engram_head_dim != 256 or
            t.engram_pad_token_id != 2 or t.engram_compressed_vocab_size != 99092 or
            t.num_nextn_predict_layers != 3 or !std.mem.eql(u8, t.rope_scaling.rope_type, "yarn") or
            t.rope_scaling.factor != 16 or t.rope_scaling.original_max_position_embeddings != 65536 or
            !std.math.isFinite(t.rms_norm_eps) or t.rms_norm_eps <= 0 or
            !std.math.isFinite(t.hc_eps) or t.hc_eps <= 0) return error.UnsupportedArchitecture;
        if (!std.math.isFinite(t.rope_theta) or t.rope_theta <= 1 or
            !std.math.isFinite(t.compress_rope_theta) or t.compress_rope_theta <= 1 or
            !std.math.isFinite(t.rope_scaling.beta_fast) or t.rope_scaling.beta_fast <= 0 or
            !std.math.isFinite(t.rope_scaling.beta_slow) or t.rope_scaling.beta_slow <= 0)
            return error.UnsupportedArchitecture;
    }

    pub fn context(c: Config, requested: u32) !u32 {
        const limit = c.text_config.max_position_embeddings;
        if (requested > limit) return error.ContextFull;
        return if (requested == 0) limit else requested;
    }
};

fn field(value: std.json.Value, name: []const u8) !std.json.Value {
    if (value != .object) return error.UnexpectedToken;
    return value.object.get(name) orelse error.MissingField;
}
fn string(value: std.json.Value) ![]const u8 {
    if (value != .string) return error.UnexpectedToken;
    return value.string;
}
test "config integers retain exact unsigned boundaries and integral notation" {
    try std.testing.expectEqual(@as(u32, 4294967295), try unsigned(u32, .{ .number_string = "4294967295" }));
    try std.testing.expectEqual(@as(u64, 18446744073709551615), try unsigned(u64, .{ .number_string = "18446744073709551615.0" }));
    try std.testing.expectEqual(@as(u32, 5120), try unsigned(u32, .{ .number_string = "5.120e3" }));
    try std.testing.expectEqual(@as(u32, 5120), try unsigned(u32, .{ .string = "5120" }));
    try std.testing.expectEqual(@as(u32, 0), try unsigned(u32, .{ .number_string = "-0.0" }));
    try std.testing.expectError(error.Overflow, unsigned(u32, .{ .number_string = "4294967296" }));
    try std.testing.expectError(error.Overflow, unsigned(u64, .{ .number_string = "18446744073709551616" }));
    try std.testing.expectError(error.Overflow, unsigned(u64, .{ .number_string = "18446744073709551616.0" }));
    try std.testing.expectError(error.InvalidNumber, unsigned(u64, .{ .number_string = "18446744073709551615.1" }));
    try std.testing.expectError(error.InvalidNumber, unsigned(u32, .{ .number_string = "5120.0000000000000000001" }));
    try std.testing.expectError(error.InvalidNumber, unsigned(u32, .{ .number_string = "1e-100" }));
    try std.testing.expectError(error.Overflow, unsigned(u32, .{ .number_string = "-1.0" }));
    try std.testing.expectError(error.Overflow, unsigned(u32, .{ .number_string = "1e100" }));
}

test "Metal config refuses missing metadata and wrong JSON shapes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.MissingField, Config.parse(arena.allocator(), "{}"));
    try std.testing.expectError(error.UnexpectedToken, Config.parse(arena.allocator(), "[]"));
    try std.testing.expectError(error.UnexpectedToken, Config.parse(arena.allocator(), "{\"text_config\":false}"));
}

test "released architecture rejects inconsistent Engram buckets and nonfinite rotary parameters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const checkpoint =
        \\{"model_type":"deepseek_v41","quantization_config":{"quant_method":"exl3","version":"1"},
        \\"text_config":{"vocab_size":129280,"hidden_size":5120,"moe_intermediate_size":2304,
        \\"num_hidden_layers":40,"num_attention_heads":64,"num_key_value_heads":1,"head_dim":512,
        \\"qk_rope_head_dim":64,"q_lora_rank":1280,"o_lora_rank":1024,"o_groups":8,
        \\"swiglu_limit":10,"rms_norm_eps":0.000001,"max_position_embeddings":1048576,
        \\"rope_theta":10000,"rope_scaling":{"rope_type":"yarn","factor":16,"beta_fast":32,
        \\"beta_slow":1,"original_max_position_embeddings":65536},"n_routed_experts":384,
        \\"n_shared_experts":1,"num_experts_per_tok":6,"scoring_func":"sqrtsoftplus",
        \\"topk_method":"noaux_tc","norm_topk_prob":true,"routed_scaling_factor":1.5,
        \\"sliding_window":128,"compress_ratios":[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
        \\1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],"compress_rope_theta":10000,
        \\"kv_source_layer_ids":[2,8,14,20],"index_source_layer_ids":[2,8,14,20,24,28,32,36],
        \\"index_n_heads":32,"index_head_dim":128,"index_topk":512,"candidate_source_layer_id":20,
        \\"candidate_topk_blocks":2048,"candidate_block_size":8,"hc_mult":4,"hc_sinkhorn_iters":20,
        \\"hc_eps":0.000001,"engram_layer_ids":[1,14],"engram_num_embeddings":[384006168,384016682],
        \\"engram_max_ngram_size":4,"engram_vocab_size":16000000,"engram_n_heads":8,"engram_head_dim":256,
        \\"engram_pad_token_id":2,"engram_compressed_vocab_size":99092,"num_nextn_predict_layers":3}
    ;
    const absent = try Config.parse(arena.allocator(), try std.fmt.allocPrint(arena.allocator(), "{s}}}", .{checkpoint}));
    var invalid = absent;
    invalid.text_config.engram_vocab_size = 1;
    try std.testing.expectError(error.UnsupportedArchitecture, invalid.validate());
    invalid = absent;
    invalid.text_config.engram_vocab_size = invalid.text_config.vocab_size;
    try std.testing.expectError(error.UnsupportedArchitecture, invalid.validate());
    invalid = absent;
    invalid.text_config.rope_theta = 1;
    try std.testing.expectError(error.UnsupportedArchitecture, invalid.validate());
    invalid = absent;
    invalid.text_config.rope_scaling.beta_slow = std.math.nan(f32);
    try std.testing.expectError(error.UnsupportedArchitecture, invalid.validate());
}
