//! The admitted DeepSeek V4.1 EXL3 architecture and its source-owned global caches.
const std = @import("std");

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
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 20));
        return parse(arena, bytes);
    }

    pub fn parse(arena: std.mem.Allocator, bytes: []const u8) !Config {
        const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{ .parse_numbers = false });
        const text = try field(root, "text_config");
        const rope = try field(text, "rope_scaling");
        const quant = try field(root, "quantization_config");
        const c: Config = .{
            .model_type = try string(try field(root, "model_type")),
            .text_config = .{
                .vocab_size = try unsigned(u32, try field(text, "vocab_size")),
                .hidden_size = try unsigned(u32, try field(text, "hidden_size")),
                .moe_intermediate_size = try unsigned(u32, try field(text, "moe_intermediate_size")),
                .num_hidden_layers = try unsigned(u32, try field(text, "num_hidden_layers")),
                .num_attention_heads = try unsigned(u32, try field(text, "num_attention_heads")),
                .num_key_value_heads = try unsigned(u32, try field(text, "num_key_value_heads")),
                .head_dim = try unsigned(u32, try field(text, "head_dim")),
                .qk_rope_head_dim = try unsigned(u32, try field(text, "qk_rope_head_dim")),
                .q_lora_rank = try unsigned(u32, try field(text, "q_lora_rank")),
                .o_lora_rank = try unsigned(u32, try field(text, "o_lora_rank")),
                .o_groups = try unsigned(u32, try field(text, "o_groups")),
                .swiglu_limit = try float(try field(text, "swiglu_limit")),
                .rms_norm_eps = try float(try field(text, "rms_norm_eps")),
                .max_position_embeddings = try unsigned(u32, try field(text, "max_position_embeddings")),
                .rope_theta = try float(try field(text, "rope_theta")),
                .rope_scaling = .{
                    .rope_type = try string(try field(rope, "rope_type")),
                    .factor = try float(try field(rope, "factor")),
                    .beta_fast = try float(try field(rope, "beta_fast")),
                    .beta_slow = try float(try field(rope, "beta_slow")),
                    .original_max_position_embeddings = try unsigned(u32, try field(rope, "original_max_position_embeddings")),
                },
                .n_routed_experts = try unsigned(u32, try field(text, "n_routed_experts")),
                .n_shared_experts = try unsigned(u32, try field(text, "n_shared_experts")),
                .num_experts_per_tok = try unsigned(u32, try field(text, "num_experts_per_tok")),
                .scoring_func = try string(try field(text, "scoring_func")),
                .topk_method = try string(try field(text, "topk_method")),
                .norm_topk_prob = try boolean(try field(text, "norm_topk_prob")),
                .routed_scaling_factor = try float(try field(text, "routed_scaling_factor")),
                .sliding_window = try unsigned(u32, try field(text, "sliding_window")),
                .compress_ratios = try integers(u32, arena, try field(text, "compress_ratios")),
                .compress_rope_theta = try float(try field(text, "compress_rope_theta")),
                .kv_source_layer_ids = try integers(u32, arena, try field(text, "kv_source_layer_ids")),
                .index_source_layer_ids = try integers(u32, arena, try field(text, "index_source_layer_ids")),
                .index_n_heads = try unsigned(u32, try field(text, "index_n_heads")),
                .index_head_dim = try unsigned(u32, try field(text, "index_head_dim")),
                .index_topk = try unsigned(u32, try field(text, "index_topk")),
                .candidate_source_layer_id = try unsigned(u32, try field(text, "candidate_source_layer_id")),
                .candidate_topk_blocks = try unsigned(u32, try field(text, "candidate_topk_blocks")),
                .candidate_block_size = try unsigned(u32, try field(text, "candidate_block_size")),
                .hc_mult = try unsigned(u32, try field(text, "hc_mult")),
                .hc_sinkhorn_iters = try unsigned(u32, try field(text, "hc_sinkhorn_iters")),
                .hc_eps = try float(try field(text, "hc_eps")),
                .engram_layer_ids = try integers(u32, arena, try field(text, "engram_layer_ids")),
                .engram_num_embeddings = try integers(u64, arena, try field(text, "engram_num_embeddings")),
                .engram_max_ngram_size = try unsigned(u32, try field(text, "engram_max_ngram_size")),
                .engram_vocab_size = try unsigned(u32, try field(text, "engram_vocab_size")),
                .engram_n_heads = try unsigned(u32, try field(text, "engram_n_heads")),
                .engram_head_dim = try unsigned(u32, try field(text, "engram_head_dim")),
                .engram_pad_token_id = try unsigned(u32, try field(text, "engram_pad_token_id")),
                .engram_compressed_vocab_size = try unsigned(u32, try field(text, "engram_compressed_vocab_size")),
                .num_nextn_predict_layers = try unsigned(u32, try field(text, "num_nextn_predict_layers")),
            },
            .quantization_config = .{
                .quant_method = try string(try field(quant, "quant_method")),
                .version = try string(try field(quant, "version")),
            },
            .engram_table_dir = if (root.object.get("engram_table_dir")) |v|
                (if (v == .null) null else try string(v))
            else
                null,
        };
        try c.validate();
        return c;
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

fn boolean(value: std.json.Value) !bool {
    if (value != .bool) return error.UnexpectedToken;
    return value.bool;
}

fn numberText(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .number_string, .string => |s| s,
        else => error.UnexpectedToken,
    };
}

fn float(value: std.json.Value) !f32 {
    const result = try std.fmt.parseFloat(f32, try numberText(value));
    if (!std.math.isFinite(result)) return error.Overflow;
    return result;
}

/// Decode integral decimal/exponent notation from digits exactly, without floating-point coercion.
fn unsigned(comptime T: type, value: std.json.Value) !T {
    const text = try numberText(value);
    if (text.len == 0) return error.InvalidCharacter;
    if (std.fmt.parseInt(T, text, 10)) |n| return n else |_| {}
    var start: usize = 0;
    const negative = text[0] == '-';
    if (negative or text[0] == '+') start = 1;
    const exp_pos = std.mem.indexOfAnyPos(u8, text, start, "eE") orelse text.len;
    const exponent: i64 = if (exp_pos < text.len)
        try std.fmt.parseInt(i32, text[exp_pos + 1 ..], 10)
    else
        0;
    var digits: usize = 0;
    var fractional: usize = 0;
    var dot = false;
    var nonzero = false;
    for (text[start..exp_pos]) |ch| {
        if (ch == '.' and !dot) {
            dot = true;
            continue;
        }
        if (ch < '0' or ch > '9') return error.InvalidCharacter;
        digits += 1;
        if (dot) fractional += 1;
        nonzero = nonzero or ch != '0';
    }
    if (digits == 0) return error.InvalidCharacter;
    if (!nonzero) return 0;
    if (negative) return error.Overflow;
    const scale = exponent - @as(i64, @intCast(fractional));
    const discarded: usize = if (scale < 0) @intCast(-scale) else 0;
    if (discarded >= digits) return error.InvalidNumber;
    const kept = digits - discarded;
    var n: T = 0;
    var index: usize = 0;
    for (text[start..exp_pos]) |ch| {
        if (ch == '.') continue;
        if (index < kept) {
            const digit: T = @intCast(ch - '0');
            if (n > (std.math.maxInt(T) - digit) / 10) return error.Overflow;
            n = n * 10 + digit;
        } else if (ch != '0') return error.InvalidNumber;
        index += 1;
    }
    // Any nonzero ordinary integer overflows after at most 20 decimal shifts.
    if (scale > 20) return error.Overflow;
    var shift: i64 = 0;
    while (shift < scale) : (shift += 1) {
        if (n > std.math.maxInt(T) / 10) return error.Overflow;
        n *= 10;
    }
    return n;
}

fn integers(comptime T: type, arena: std.mem.Allocator, value: std.json.Value) ![]const T {
    if (value != .array) return error.UnexpectedToken;
    const result = try arena.alloc(T, value.array.items.len);
    for (value.array.items, result) |item, *out| out.* = try unsigned(T, item);
    return result;
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

test "config decoding rejects missing fields and wrong shapes without defaults" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.MissingField, Config.parse(arena.allocator(), "{}"));
    try std.testing.expectError(error.UnexpectedToken, Config.parse(arena.allocator(), "[]"));
    try std.testing.expectError(error.UnexpectedToken, Config.parse(arena.allocator(), "{\"text_config\":false}"));
    try std.testing.expectError(error.UnexpectedToken, boolean(.{ .string = "true" }));
    try std.testing.expectError(error.UnexpectedToken, unsigned(u32, .{ .bool = true }));
    try std.testing.expectError(error.Overflow, float(.{ .number_string = "1e100" }));
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
