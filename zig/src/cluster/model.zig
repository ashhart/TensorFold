//! A model's shape from its config.json (text_config when nested): what the planner and the memory budgets need.
const std = @import("std");

pub const Attention = enum(u8) { gqa, mla, linear };

pub const max_layers = 256;

pub const Shape = struct {
    family: []const u8 = "",
    hidden: u64 = 0,
    layers: u32 = 0,
    vocab: u64 = 0,
    heads: u32 = 0,
    kv_heads: u32 = 0,
    head_dim: u32 = 0,
    experts: u32 = 0,
    top_k: u32 = 0,
    moe_inter: u64 = 0,
    /// The routed experts' width: routed_expert_hidden_size for latent MoE, else the hidden size.
    latent: u64 = 0,
    shared: u32 = 0,
    first_dense: u32 = 0,
    inter: u64 = 0,
    q_lora: u64 = 0,
    kv_lora: u64 = 0,
    rope_dim: u32 = 0,
    nope_dim: u32 = 0,
    v_dim: u32 = 0,
    linear_heads: u32 = 0,
    linear_head_dim: u32 = 0,
    conv_kernel: u32 = 0,
    quant_group: u32 = 0,
    quant_bits: u32 = 0,
    /// DSA: each query attends to the indexer's top `index_topk` tokens; 0 means dense attention.
    index_topk: u32 = 0,
    index_heads: u32 = 0,
    index_dim: u32 = 0,
    mtp_layers: u32 = 0,
    max_context: u64 = 0,
    kinds: [max_layers]Attention = @splat(.gqa),

    pub fn kind(s: *const Shape, layer: u32) Attention {
        return s.kinds[layer];
    }

    pub fn moe(s: *const Shape, layer: u32) bool {
        return s.experts > 0 and layer >= s.first_dense;
    }

    pub fn count(s: *const Shape, k: Attention) u32 {
        var n: u32 = 0;
        for (s.kinds[0..s.layers]) |x| n += @intFromBool(x == k);
        return n;
    }

    /// KV bytes one token adds to one layer's cache (bf16); linear-attention layers keep a fixed state instead.
    pub fn kvPerToken(s: *const Shape, layer: u32) u64 {
        return switch (s.kinds[layer]) {
            .mla => (s.kv_lora + s.rope_dim) * 2,
            .gqa => 2 * @as(u64, s.kv_heads) * s.head_dim * 2,
            .linear => 0,
        };
    }

    /// The DSA indexer's key bytes per token in one layer (bf16), beside the latent cache.
    pub fn indexPerToken(s: *const Shape, layer: u32) u64 {
        return if (s.index_topk > 0 and s.kinds[layer] == .mla) @as(u64, s.index_dim) * 2 else 0;
    }

    /// Tokens of a stream's cache one query reads in one layer: all of them, or the indexer's top-k under DSA.
    pub fn attended(s: *const Shape, context: u64) u64 {
        return if (s.index_topk > 0) @min(context, s.index_topk) else context;
    }

    /// A stream's fixed state in one linear-attention layer: fp32 per-head state plus the short convolutions' tails.
    pub fn statePerStream(s: *const Shape, layer: u32) u64 {
        if (s.kinds[layer] != .linear) return 0;
        const width = @as(u64, s.linear_heads) * s.linear_head_dim;
        return width * s.linear_head_dim * 4 + 3 * width * (s.conv_kernel -| 1) * 2;
    }
};

pub const Error = error{BadConfig};

/// Parse the fields the planner needs; unknown families still get their dense and MoE shapes.
pub fn fromConfig(a: std.mem.Allocator, text: []const u8) !Shape {
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return error.BadConfig;
    if (root != .object) return error.BadConfig;
    const c = if (root.object.get("text_config")) |t| (if (t == .object) t else root) else root;
    const o = c.object;
    var s: Shape = .{
        .family = str(o.get("model_type")) orelse str(root.object.get("model_type")) orelse "",
        .hidden = int(o, "hidden_size"),
        .layers = @intCast(int(o, "num_hidden_layers")),
        .vocab = int(o, "vocab_size"),
        .heads = @intCast(int(o, "num_attention_heads")),
        .experts = @intCast(@max(int(o, "num_experts"), int(o, "n_routed_experts"))),
        .top_k = @intCast(@max(int(o, "num_experts_per_token"), int(o, "num_experts_per_tok"))),
        .moe_inter = int(o, "moe_intermediate_size"),
        .shared = @intCast(@max(int(o, "num_shared_experts"), int(o, "n_shared_experts"))),
        .first_dense = @intCast(int(o, "first_k_dense_replace")),
        .inter = int(o, "intermediate_size"),
        .q_lora = int(o, "q_lora_rank"),
        .kv_lora = int(o, "kv_lora_rank"),
        .rope_dim = @intCast(int(o, "qk_rope_head_dim")),
        .nope_dim = @intCast(int(o, "qk_nope_head_dim")),
        .v_dim = @intCast(int(o, "v_head_dim")),
        .mtp_layers = @intCast(int(o, "num_nextn_predict_layers")),
        .max_context = int(o, "max_position_embeddings"),
    };
    if (s.layers == 0 or s.layers > max_layers or s.hidden == 0) return error.BadConfig;
    s.kv_heads = @intCast(int(o, "num_key_value_heads"));
    if (s.kv_heads == 0) s.kv_heads = s.heads;
    s.head_dim = @intCast(int(o, "head_dim"));
    if (s.head_dim == 0 and s.heads > 0) s.head_dim = @intCast(s.hidden / s.heads);
    s.latent = int(o, "routed_expert_hidden_size");
    if (s.latent == 0) s.latent = s.hidden;
    const base: Attention = if (s.kv_lora > 0) .mla else .gqa;
    @memset(s.kinds[0..s.layers], base);
    if (o.get("linear_attn_config")) |l| if (l == .object) {
        s.linear_heads = @intCast(int(l.object, "num_heads"));
        s.linear_head_dim = @intCast(int(l.object, "head_dim"));
        s.conv_kernel = @intCast(int(l.object, "short_conv_kernel_size"));
        if (l.object.get("kda_layers")) |ks| if (ks == .array) for (ks.array.items) |k| {
            const one = uintOf(k) orelse continue;
            if (one >= 1 and one <= s.layers) s.kinds[one - 1] = .linear;
        };
    };
    s.index_topk = @intCast(int(o, "index_topk"));
    s.index_heads = @intCast(int(o, "index_n_heads"));
    s.index_dim = @intCast(int(o, "index_head_dim"));
    if (o.get("quantization_config")) |q| if (q == .object) {
        s.quant_group = @intCast(deep(q, &.{ "config_groups", "group_0", "weights", "group_size" }) orelse int(q.object, "group_size"));
        s.quant_bits = @intCast(deep(q, &.{ "config_groups", "group_0", "weights", "num_bits" }) orelse int(q.object, "bits"));
        if (std.mem.eql(u8, str(q.object.get("quant_method")) orelse "", "fp8")) s.quant_bits = 8;
        if (q.object.get("weight_block_size")) |b| if (b == .array and b.array.items.len > 0) {
            s.quant_group = @intCast(uintOf(b.array.items[0]) orelse 0);
        };
    };
    return s;
}

fn deep(v: std.json.Value, path: []const []const u8) ?u64 {
    var x = v;
    for (path) |k| {
        if (x != .object) return null;
        x = x.object.get(k) orelse return null;
    }
    return uintOf(x);
}

fn int(o: std.json.ObjectMap, key: []const u8) u64 {
    return uintOf(o.get(key) orelse return 0) orelse 0;
}

fn uintOf(v: std.json.Value) ?u64 {
    return switch (v) {
        .integer => |i| if (i < 0) null else @intCast(i),
        else => null,
    };
}

fn str(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

/// Kimi K3's shape as its config states it (moonshotai/Kimi-K3 at f831ab66), for tests that run without the file.
pub fn k3() Shape {
    var s: Shape = .{
        .family = "kimi_linear", .hidden = 7168, .layers = 93, .vocab = 163840, .heads = 96, .kv_heads = 96, .head_dim = 74,
        .experts = 896, .top_k = 16, .moe_inter = 3072, .latent = 3584, .shared = 2, .first_dense = 1, .inter = 33792,
        .q_lora = 1536, .kv_lora = 512, .rope_dim = 64, .nope_dim = 128, .v_dim = 128, .linear_heads = 96, .linear_head_dim = 128, .conv_kernel = 4,
        .quant_group = 32, .quant_bits = 4, .max_context = 1 << 20,
    };
    @memset(s.kinds[0..93], .linear);
    var layer: u32 = 3;
    while (layer < 92) : (layer += 4) s.kinds[layer] = .mla;
    s.kinds[92] = .mla;
    return s;
}

/// GLM-5.3 (glm_moe_dsa, FP8) from the lead's figures; head dims, vocab and indexer heads assumed as DeepSeek-V3.2's.
pub fn glm53() Shape {
    var s: Shape = .{
        .family = "glm_moe_dsa", .hidden = 6144, .layers = 78, .vocab = 151552, .heads = 64, .kv_heads = 64, .head_dim = 96,
        .experts = 256, .top_k = 8, .moe_inter = 2048, .latent = 6144, .shared = 1, .first_dense = 3, .inter = 12288,
        .q_lora = 2048, .kv_lora = 512, .rope_dim = 64, .nope_dim = 192, .v_dim = 256, .quant_group = 128, .quant_bits = 8,
        .index_topk = 2048, .index_heads = 32, .index_dim = 128, .mtp_layers = 1, .max_context = 202752,
    };
    @memset(s.kinds[0..78], .mla);
    return s;
}

test "Kimi K3's literal shape: 69 KDA layers, 24 MLA layers, latent experts" {
    const s = k3();
    try std.testing.expectEqual(@as(u32, 69), s.count(.linear));
    try std.testing.expectEqual(@as(u32, 24), s.count(.mla));
    try std.testing.expect(s.kind(0) == .linear and s.kind(3) == .mla and s.kind(91) == .mla and s.kind(92) == .mla);
    try std.testing.expect(!s.moe(0) and s.moe(1));
    try std.testing.expectEqual(@as(u64, 1152), s.kvPerToken(3));
    try std.testing.expectEqual(@as(u64, 96 * 128 * 128 * 4 + 3 * 12288 * 3 * 2), s.statePerStream(0));
}

test "GLM-5.3: every layer MLA with a 2,048-token indexer, 75 MoE layers, one MTP layer" {
    const s = glm53();
    try std.testing.expectEqual(@as(u32, 78), s.count(.mla));
    try std.testing.expect(!s.moe(2) and s.moe(3));
    try std.testing.expectEqual(@as(u64, 1152), s.kvPerToken(10));
    try std.testing.expectEqual(@as(u64, 256), s.indexPerToken(10));
    try std.testing.expectEqual(@as(u64, 2048), s.attended(32768));
    try std.testing.expectEqual(@as(u64, 1000), s.attended(1000));
    try std.testing.expectEqual(@as(u64, 32768), k3().attended(32768));
}

test "the real K3 config (TF_K3_DIR) parses to the literal" {
    const dir = std.testing.environ.getPosix("TF_K3_DIR") orelse return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 20));
    var got = try fromConfig(a, text);
    var want = k3();
    try std.testing.expectEqualStrings(want.family, got.family);
    got.family = "";
    want.family = "";
    try std.testing.expectEqual(want, got);
}
