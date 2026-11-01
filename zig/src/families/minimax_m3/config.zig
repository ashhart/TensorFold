//! MiniMax-M3.1-Flash's shape from the release's config.json (text_config): routed or dense MLP, sparse attention.
const std = @import("std");

pub const max_layers = 128;

/// A layer's attention (block-sparse with the lightning indexer, or dense) and its MLP (routed or dense).
pub const Layer = struct { sparse: bool, moe: bool };

pub const Config = struct {
    hidden: u32 = 6144,
    layers: u32 = 60,
    vocab: u32 = 200064,
    eps: f32 = 1e-6,
    heads: u32 = 64,
    kv_heads: u32 = 4,
    head_dim: u32 = 128,
    /// Rotated dims of each head (and of each index head): the first `rotary`, NeoX halves.
    rotary: u32 = 64,
    theta: f32 = 1e7,
    dense_intermediate: u32 = 12288,
    moe_intermediate: u32 = 3072,
    shared_intermediate: u32 = 3072,
    experts: u32 = 128,
    topk: u32 = 4,
    routed_scale: f32 = 2.0,
    swiglu_alpha: f32 = 1.702,
    swiglu_limit: f32 = 7.0,
    index_heads: u32 = 4,
    index_dim: u32 = 128,
    /// Keys an index block pools: the unit of selection.
    block: u32 = 128,
    /// Blocks a query keeps at most (its own included).
    topk_blocks: u32 = 16,
    context: u32 = 1048576,
    image_token: u32 = 200025,
    layer: [max_layers]Layer = undefined,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// Key and value width of one token in one layer.
    pub fn kvWidth(c: *const Config) u32 {
        return c.kv_heads * c.head_dim;
    }

    /// Rows of the stacked q | k | v | index q | index k projection.
    pub fn qkvRows(c: *const Config) u32 {
        return (c.heads + 2 * c.kv_heads) * c.head_dim + (c.index_heads + 1) * c.index_dim;
    }

    /// Index blocks a context of `len` keys spans.
    pub fn blocks(c: *const Config, len: usize) usize {
        return std.math.divCeil(usize, len, c.block) catch unreachable;
    }
};

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse {
        std.log.err("minimax_m3: config.json lacks {s}", .{key});
        return error.BadMiniMaxConfig;
    };
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadMiniMaxConfig,
    };
}

fn int(o: std.json.ObjectMap, key: []const u8) !u32 {
    const v = try field(o, key);
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.BadMiniMaxConfig;
    return @intCast(v.integer);
}

fn flag(o: std.json.ObjectMap, key: []const u8, want: bool) !void {
    const v = try field(o, key);
    if (v != .bool or v.bool != want) return error.UnsupportedMiniMaxConfig;
}

fn says(o: std.json.ObjectMap, key: []const u8, want: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, want)) return error.UnsupportedMiniMaxConfig;
}

/// A per-layer 0/1 list of `layers` entries.
fn perLayer(o: std.json.ObjectMap, key: []const u8, layers: u32, out: []bool) !void {
    const v = try field(o, key);
    if (v != .array or v.array.items.len != layers) return error.BadMiniMaxConfig;
    for (v.array.items, 0..) |x, i| {
        if (x != .integer or x.integer < 0 or x.integer > 1) return error.BadMiniMaxConfig;
        out[i] = x.integer == 1;
    }
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    var doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadMiniMaxConfig;
    const root = doc.value.object;
    try says(root, "model_type", "minimax_m3_vl");
    const text = try field(root, "text_config");
    if (text != .object) return error.BadMiniMaxConfig;
    const t = text.object;
    var c = Config{};
    c.hidden = try int(t, "hidden_size");
    c.layers = try int(t, "num_hidden_layers");
    c.vocab = try int(t, "vocab_size");
    c.eps = @floatCast(try number(try field(t, "rms_norm_eps")));
    c.heads = try int(t, "num_attention_heads");
    c.kv_heads = try int(t, "num_key_value_heads");
    c.head_dim = try int(t, "head_dim");
    c.rotary = try int(t, "rotary_dim");
    c.theta = @floatCast(try number(try field(t, "rope_theta")));
    c.dense_intermediate = try int(t, "dense_intermediate_size");
    c.moe_intermediate = try int(t, "intermediate_size");
    c.shared_intermediate = try int(t, "shared_intermediate_size");
    c.experts = try int(t, "num_local_experts");
    c.topk = try int(t, "num_experts_per_tok");
    c.routed_scale = @floatCast(try number(try field(t, "routed_scaling_factor")));
    c.swiglu_alpha = @floatCast(try number(try field(t, "swiglu_alpha")));
    c.swiglu_limit = @floatCast(try number(try field(t, "swiglu_limit")));
    c.context = try int(t, "max_position_embeddings");
    if (root.get("image_token_index")) |v| if (v == .integer and v.integer > 0) {
        c.image_token = @intCast(v.integer);
    };
    if (c.layers > max_layers) return error.BadMiniMaxConfig;
    // the release's arithmetic: Gemma norms on every head, sigmoid scores with the correction bias, one shared expert
    try flag(t, "use_gemma_norm", true);
    try flag(t, "use_qk_norm", true);
    try flag(t, "use_routing_bias", true);
    try flag(t, "attention_output_gate", false);
    try flag(t, "tie_word_embeddings", false);
    try says(t, "scoring_func", "sigmoid");
    try says(t, "hidden_act", "swigluoai");
    try says(t, "qk_norm_type", "per_head");
    if ((try int(t, "n_shared_experts")) != 1) return error.UnsupportedMiniMaxConfig;
    const sp = try field(t, "sparse_attention_config");
    if (sp != .object) return error.BadMiniMaxConfig;
    const s = sp.object;
    try flag(s, "use_sparse_attention", true);
    try says(s, "sparse_score_type", "max");
    c.index_heads = try int(s, "sparse_num_index_heads");
    c.index_dim = try int(s, "sparse_index_dim");
    c.block = try int(s, "sparse_block_size");
    c.topk_blocks = try int(s, "sparse_topk_blocks");
    // init blocks 0 and one local block (the query's own) are what the selection kernels force
    const init = try field(s, "sparse_init_block");
    if (init != .integer or init.integer != 0 or (try int(s, "sparse_local_block")) != 1) return error.UnsupportedMiniMaxAttention;
    var sparse: [max_layers]bool = undefined;
    var moe: [max_layers]bool = undefined;
    var no_value: [max_layers]bool = undefined;
    try perLayer(s, "sparse_attention_freq", c.layers, &sparse);
    try perLayer(s, "sparse_disable_index_value", c.layers, &no_value);
    try perLayer(t, "moe_layer_freq", c.layers, &moe);
    for (0..c.layers) |i| {
        // the index-value branch is off on every sparse layer of the release; dense attention layers are not built
        if (!sparse[i] or !no_value[i]) return error.UnsupportedMiniMaxAttention;
        c.layer[i] = .{ .sparse = sparse[i], .moe = moe[i] };
    }
    try validate(&c);
    return c;
}

/// The kernels' shape limits: a head in 32 lanes' registers, four index heads (one a key head), 128-key blocks.
pub fn validate(c: *const Config) error{ UnsupportedMiniMaxAttention, UnsupportedMiniMaxExperts }!void {
    if (c.head_dim != 128 or c.index_dim != 128 or c.heads % c.kv_heads != 0 or c.heads / c.kv_heads > 32 or
        c.index_heads != c.kv_heads or c.index_heads != 4 or c.rotary == 0 or c.rotary > c.head_dim or c.rotary % 2 != 0 or c.block != 128 or
        c.topk_blocks == 0 or c.topk_blocks > 32) return error.UnsupportedMiniMaxAttention;
    if (c.hidden == 0 or c.hidden % 1024 != 0 or c.experts == 0 or c.experts % 8 != 0 or c.experts > 256 or c.topk == 0 or c.topk > 8 or
        c.moe_intermediate % 64 != 0 or c.shared_intermediate % 64 != 0 or c.dense_intermediate % 64 != 0) return error.UnsupportedMiniMaxExperts;
}

test "minimax_m3 config: the release's shape, three dense layers, every layer sparse" {
    const c = try parse(std.testing.allocator, @embedFile("config_fixture.json"));
    try std.testing.expectEqual(@as(u32, 60), c.layers);
    try std.testing.expect(!c.layer[0].moe and !c.layer[2].moe and c.layer[3].moe and c.layer[59].moe);
    try std.testing.expect(c.layer[0].sparse and c.layer[59].sparse);
    try std.testing.expectEqual(@as(u32, 64), c.rotary);
    try std.testing.expectEqual(@as(u32, 9856), c.qkvRows());
    try std.testing.expectEqual(@as(usize, 17), c.blocks(2049));
    try std.testing.expectEqual(@as(f32, 2.0), c.routed_scale);
}

test "minimax_m3 config: the flat converted form, softmax scores, init blocks, an index-value branch or a dense layer are refused" {
    const fixture = @embedFile("config_fixture.json");
    const Case = struct { from: []const u8, to: []const u8, err: anyerror };
    const cases = [_]Case{
        .{ .from = "\"model_type\": \"minimax_m3_vl\"", .to = "\"model_type\": \"minimax_m3\"", .err = error.UnsupportedMiniMaxConfig },
        .{ .from = "\"scoring_func\": \"sigmoid\"", .to = "\"scoring_func\": \"softmax\"", .err = error.UnsupportedMiniMaxConfig },
        .{ .from = "\"sparse_init_block\": 0", .to = "\"sparse_init_block\": 1", .err = error.UnsupportedMiniMaxAttention },
        .{ .from = "\"sparse_disable_index_value\": [\n    1,", .to = "\"sparse_disable_index_value\": [\n    0,", .err = error.UnsupportedMiniMaxAttention },
        .{ .from = "\"sparse_attention_freq\": [\n    1,", .to = "\"sparse_attention_freq\": [\n    0,", .err = error.UnsupportedMiniMaxAttention },
        .{ .from = "\"head_dim\": 128", .to = "\"head_dim\": 96", .err = error.UnsupportedMiniMaxAttention },
        .{ .from = "\"num_experts_per_tok\": 4", .to = "\"num_experts_per_tok\": 9", .err = error.UnsupportedMiniMaxExperts },
    };
    for (cases) |x| {
        const bytes = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, x.from, x.to);
        defer std.testing.allocator.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, fixture));
        try std.testing.expectError(x.err, parse(std.testing.allocator, bytes));
    }
}
