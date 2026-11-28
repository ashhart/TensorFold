const std = @import("std");
const number = @import("config_number");
const rope = @import("rope.zig");

pub const Attention = enum { sliding_attention, deepseek_sparse_attention };
pub const FeedForward = enum { dense, sparse };

pub const HeadShape = struct {
    heads: u32,
    non_rotary: u32,
    rotary: u32,
    value: u32,

    pub fn queryRows(self: HeadShape) u32 {
        return self.heads * (self.non_rotary + self.rotary);
    }

    pub fn kvRows(self: HeadShape) u32 {
        return self.heads * (self.non_rotary + self.value);
    }
};

pub const Text = struct {
    hidden_size: u32,
    intermediate_size: u32,
    moe_intermediate_size: u32,
    routed_expert_hidden_size: u32,
    num_hidden_layers: u32,
    vocab_size: u32,
    max_position_embeddings: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    num_attention_heads: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    swa_num_attention_heads: u32,
    swa_qk_nope_head_dim: u32,
    swa_qk_rope_head_dim: u32,
    swa_v_head_dim: u32,
    sliding_window: u32,
    compressed_indexer_compress_ratio: u32,
    compressed_indexer_num_attention_heads: u32,
    compressed_indexer_head_dim: u32,
    compressed_indexer_group_topk: u32,
    compressed_indexer_candidate_topk: u32,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    n_shared_experts: u32,
    n_group: u32,
    topk_group: u32,
    norm_topk_prob: bool,
    swiglu_limit: f32,
    routed_scaling_factor: f32,
    enable_ihc: bool,
    hc_mult: u32,
    hc_eps: f32,
    hc_magnitude: f32,
    rms_norm_eps: f32,
    rope_parameters: struct {
        sliding_attention: rope.Config,
        deepseek_sparse_attention: rope.Config,
    },
    layer_types: []const Attention,
    mlp_layer_types: []const FeedForward,
    engram_layer_ids: []const u32,
    engram_max_ngram_size: u32,
    engram_n_head_per_ngram: u32,
    engram_n_embed_per_ngram: u32,
    engram_pad_id: u32,
    engram_seed: u64,
    engram_fp8_embedding: bool,
    engram_lookup_table_size: u32 = 120818,
    engram_layer_head_vocab_sizes: std.json.ArrayHashMap([]const []const u32),

    pub fn rotary(self: Text, attention: Attention) !rope.Parameters {
        return switch (attention) {
            .sliding_attention => self.rope_parameters.sliding_attention.parameters(self.swa_qk_rope_head_dim),
            .deepseek_sparse_attention => self.rope_parameters.deepseek_sparse_attention.parameters(self.qk_rope_head_dim),
        };
    }

    pub fn headShape(self: Text, layer: u32) HeadShape {
        return switch (self.layer_types[layer]) {
            .sliding_attention => .{
                .heads = self.swa_num_attention_heads,
                .non_rotary = self.swa_qk_nope_head_dim,
                .rotary = self.swa_qk_rope_head_dim,
                .value = self.swa_v_head_dim,
            },
            .deepseek_sparse_attention => .{
                .heads = self.num_attention_heads,
                .non_rotary = self.qk_nope_head_dim,
                .rotary = self.qk_rope_head_dim,
                .value = self.v_head_dim,
            },
        };
    }

    pub fn engramRows(self: Text, layer: u32) !usize {
        var layer_name: [10]u8 = undefined;
        const key = try std.fmt.bufPrint(&layer_name, "{d}", .{layer});
        const orders = self.engram_layer_head_vocab_sizes.map.get(key) orelse return error.BadEngramGeometry;
        if (self.engram_max_ngram_size < 2 or orders.len != self.engram_max_ngram_size - 1)
            return error.BadEngramGeometry;
        var rows: usize = 0;
        for (orders) |heads| {
            if (heads.len != self.engram_n_head_per_ngram) return error.BadEngramGeometry;
            for (heads) |prime| {
                if (prime < 2) return error.BadEngramGeometry;
                rows = try std.math.add(usize, rows, prime);
            }
        }
        return rows;
    }
};

pub const Model = struct {
    model_type: []const u8,
    quantization: ?struct {
        bits: u8,
        group_size: u16,
        mode: []const u8 = "affine",
    } = null,
    text_config: Text,
    vision_config: struct {
        hidden_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        out_hidden_size: u32,
        patch_size: u32,
        spatial_merge_size: u32,
    },
    speculative_config: struct {
        method: []const u8,
        num_speculative_tokens: u32,
    },
    dfly_num_hidden_layers: u32,
    dfly_intermediate_size: ?u32 = null,
    dfly_target_index_topk: u32,
    dfly_target_index_layer: []const u32,
    dfly_target_kv_layers: []const u32,
    dfly_enable_hidden_correction: bool,
    dfly_mask_token_id: u32,
    dfly_mtp_fusion: []const u8,

    pub fn validate(self: Model) !void {
        if (!std.mem.eql(u8, self.model_type, "hy_v4_flash")) return error.WrongFamily;
        const t = self.text_config;
        _ = try t.rotary(.sliding_attention);
        _ = try t.rotary(.deepseek_sparse_attention);
        if (t.layer_types.len == 0 or t.layer_types.len != t.num_hidden_layers or
            t.mlp_layer_types.len != t.num_hidden_layers or t.hidden_size == 0 or
            t.q_lora_rank == 0 or t.kv_lora_rank == 0 or t.sliding_window == 0 or
            !t.enable_ihc or t.hc_mult == 0 or self.vision_config.out_hidden_size != t.hidden_size)
            return error.BadArchitecture;
        if (t.num_experts_per_tok == 0 or t.num_experts_per_tok > t.n_routed_experts or
            t.routed_expert_hidden_size == 0 or t.moe_intermediate_size == 0 or t.n_group != 1 or t.topk_group != 1 or
            !t.norm_topk_prob or !std.math.isFinite(t.swiglu_limit) or t.swiglu_limit <= 0 or
            !std.math.isFinite(t.routed_scaling_factor) or t.routed_scaling_factor <= 0)
            return error.BadExpertGeometry;
        if (t.compressed_indexer_compress_ratio == 0 or t.compressed_indexer_group_topk == 0 or
            @as(u64, t.compressed_indexer_compress_ratio) * t.compressed_indexer_group_topk != t.compressed_indexer_candidate_topk)
            return error.BadSparseGeometry;
        if (t.engram_max_ngram_size < 2 or t.engram_n_head_per_ngram == 0 or t.engram_n_embed_per_ngram == 0 or
            t.engram_n_embed_per_ngram % t.engram_n_head_per_ngram != 0 or t.engram_pad_id >= t.vocab_size)
            return error.BadEngramGeometry;
        // Engram layer numbers are one-based, whereas DFly2 target layers are zero-based.
        var previous: u32 = 0;
        for (t.engram_layer_ids) |n| {
            if (n <= previous or n > t.num_hidden_layers) return error.BadEngramGeometry;
            _ = try t.engramRows(n);
            previous = n;
        }
        if (!std.mem.eql(u8, self.speculative_config.method, "dfly") or
            !std.mem.eql(u8, self.dfly_mtp_fusion, "slot0") or !self.dfly_enable_hidden_correction or
            self.dfly_num_hidden_layers == 0 or self.speculative_config.num_speculative_tokens == 0 or
            self.dfly_target_kv_layers.len != self.dfly_num_hidden_layers or
            self.dfly_target_index_layer.len != self.dfly_num_hidden_layers or
            self.dfly_target_index_topk != t.compressed_indexer_candidate_topk or self.dfly_mask_token_id >= t.vocab_size)
            return error.BadDraftGeometry;
        for (self.dfly_target_index_layer, self.dfly_target_kv_layers) |index, kv| {
            if (index >= t.num_hidden_layers or kv >= t.num_hidden_layers or
                t.layer_types[index] != .deepseek_sparse_attention) return error.BadDraftGeometry;
        }
    }
};

const EngramVocab = @FieldType(Text, "engram_layer_head_vocab_sizes");

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Model) {
    const doc = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{
        .allocate = .alloc_always,
        .parse_numbers = false,
    });
    errdefer doc.deinit();
    const model = try readField(Model, doc.arena.allocator(), doc.value);
    try model.validate();
    return .{ .arena = doc.arena, .value = model };
}

// Decode the checkpoint schema from the JSON tree without going through a floating-point integer.
fn readField(comptime T: type, a: std.mem.Allocator, v: std.json.Value) std.json.ParseFromValueError!T {
    if (T == EngramVocab) {
        if (v != .object) return error.UnexpectedToken;
        var vocab: EngramVocab = .{};
        try vocab.map.ensureTotalCapacity(a, v.object.count());
        for (v.object.keys(), v.object.values()) |key, value| {
            vocab.map.putAssumeCapacityNoClobber(key, try readField([]const []const u32, a, value));
        }
        return vocab;
    }
    switch (@typeInfo(T)) {
        .int => {
            if (v != .number_string and v != .string) return error.UnexpectedToken;
            return number.unsigned(T, v) catch |err| switch (err) {
                error.BadConfig => error.UnexpectedToken,
                else => |numeric_error| numeric_error,
            };
        },
        .bool => return if (v == .bool) v.bool else error.UnexpectedToken,
        .float => return std.fmt.parseFloat(T, try numericText(v)),
        .@"enum" => |e| {
            const text = try numericText(v);
            if (std.meta.stringToEnum(T, text)) |tag| return tag;
            if (!std.json.isNumberFormattedLikeAnInteger(text)) return error.InvalidEnumTag;
            const n = std.fmt.parseInt(e.tag_type, text, 10) catch return error.InvalidEnumTag;
            return std.enums.fromInt(T, n) orelse error.InvalidEnumTag;
        },
        .optional => |o| return if (v == .null) null else try readField(o.child, a, v),
        .array => |array| {
            if (v != .array) return error.UnexpectedToken;
            if (v.array.items.len != array.len) return error.LengthMismatch;
            var items: T = undefined;
            for (&items, v.array.items) |*item, value| item.* = try readField(array.child, a, value);
            return items;
        },
        .pointer => |ptr| {
            if (T == []const u8 and v == .string) return v.string;
            if (v != .array) return error.UnexpectedToken;
            const items = try a.alloc(ptr.child, v.array.items.len);
            for (items, v.array.items) |*item, value| item.* = try readField(ptr.child, a, value);
            return items;
        },
        .@"struct" => |info| {
            if (v != .object) return error.UnexpectedToken;
            var result: T = undefined;
            inline for (info.field_names, info.field_types, info.field_attrs) |name, field_type, attrs| {
                @field(result, name) = if (v.object.get(name)) |field|
                    try readField(field_type, a, field)
                else
                    attrs.defaultValue(field_type) orelse return error.MissingField;
            }
            return result;
        },
        else => @compileError("unsupported Hy4 config field type: " ++ @typeName(T)),
    }
}

fn numericText(v: std.json.Value) error{UnexpectedToken}![]const u8 {
    return number.numberText(v) catch error.UnexpectedToken;
}

fn expectLegacyModel(bytes: []const u8) !void {
    const a = std.testing.allocator;
    const legacy = try std.json.parseFromSlice(Model, a, bytes, .{ .ignore_unknown_fields = true });
    defer legacy.deinit();
    const parsed = try parse(a, bytes);
    defer parsed.deinit();
    const expected = try std.json.Stringify.valueAlloc(a, legacy.value, .{});
    defer a.free(expected);
    const actual = try std.json.Stringify.valueAlloc(a, parsed.value, .{});
    defer a.free(actual);
    try std.testing.expectEqualStrings(expected, actual);
}

test "Hy4 config matches the previous decoder for the checkpoint and numeric variants" {
    const fixture = @embedFile("config_fixture.json");
    try expectLegacyModel(fixture);
    const a = std.testing.allocator;
    const modified = try std.mem.replaceOwned(u8, a, fixture, "\"compressed_indexer_candidate_topk\": 2048", "\"compressed_indexer_candidate_topk\": \"2.048e3\"");
    defer a.free(modified);
    try expectLegacyModel(modified);
    const seeded = try std.mem.replaceOwned(u8, a, modified, "\"engram_seed\": 0", "\"engram_seed\": 18446744073709551615.0");
    defer a.free(seeded);
    const quantized = try std.fmt.allocPrint(a, "{{\"quantization\": {{\"bits\": 4, \"group_size\": 64}}, \"dfly_intermediate_size\": 8192,{s}", .{seeded[1..]});
    defer a.free(quantized);
    try expectLegacyModel(quantized);
}

test "released Hy4-flash geometry" {
    const parsed = try parse(std.testing.allocator, @embedFile("config_fixture.json"));
    defer parsed.deinit();
    const m = parsed.value;
    const t = m.text_config;
    try std.testing.expectEqual(@as(u32, 8192), t.headShape(0).queryRows());
    try std.testing.expectEqual(@as(u32, 16384), t.headShape(2).queryRows());
    try std.testing.expectEqual(@as(u32, 14336), t.headShape(0).kvRows());
    try std.testing.expectEqual(@as(u32, 28672), t.headShape(2).kvRows());
    try std.testing.expectEqualSlices(u32, &.{ 2, 12, 22, 32 }, t.engram_layer_ids);
    try std.testing.expectEqual(@as(usize, 165008942), try t.engramRows(32));
    try std.testing.expectEqualSlices(u32, &.{ 25, 29, 33, 37, 41 }, m.dfly_target_kv_layers);
    try std.testing.expectEqual(@as(u32, 5), m.speculative_config.num_speculative_tokens);
    try std.testing.expectEqual(@as(u32, 512), t.n_routed_experts);
}

test "DFly2 fails closed on a sliding index source" {
    const parsed = try parse(std.testing.allocator, @embedFile("config_fixture.json"));
    defer parsed.deinit();
    var broken = parsed.value;
    broken.dfly_target_index_layer = &.{ 2, 29, 33, 37, 41 };
    try std.testing.expectError(error.BadDraftGeometry, broken.validate());
}
