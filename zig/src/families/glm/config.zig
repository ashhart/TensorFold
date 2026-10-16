//! GLM-5.3-Flash's shape from config.json's text_config: the engine's kernels are generated for exactly this shape.
const std = @import("std");

pub const max_layers = 46;

pub const Kind = enum { kda, mla };

/// List entries a decode attention block takes: the same at every window width, so every row's sums keep one order.
pub const attn_span = 128;

pub const Config = struct {
    hidden: u32 = 4096,
    layers: u32 = 45,
    run: u32 = 45, // backbone layers loaded and run: all, or the first few for a check that must fit (GLM_LAYERS)
    vocab: u32 = 154880,
    head_unit: u32 = 64, // the vocabulary's TP cut in whole units: 128 for EXL3, whose matrices split at 128 columns
    eps: f32 = 1e-5,
    dense_layers: u32 = 3,
    dense_inter: u32 = 12288, // this Mac's dense MLP rows: all, or its half in TP2
    experts: u32 = 288,
    own: [2]u32 = .{ 0, 288 }, // routed experts [lo, hi) this Mac holds: all, or its half in expert-parallel mode
    inter: [2]u32 = .{ 0, 2048 }, // each routed expert's intermediate rows [lo, hi) this Mac holds (half by rows)
    topk: u32 = 8,
    moe_inter: u32 = 2048,
    shared_inter: u32 = 2048, // the shared expert's width: moe_intermediate_size x n_shared_experts
    routed_scale: f32 = 2.5,
    norm_topk: bool = true, // norm_topk_prob: the routed weights renormalized over the top-k
    swiglu_limit: f32 = 10.0,
    kda_heads: u32 = 64, // this Mac's KDA heads: all, or its share in TP2 (heads [tp_rank kda_heads, ...))
    kda_dim: u32 = 128,
    conv: u32 = 4,
    lower_bound: f32 = -5.0,
    mla_heads: u32 = 64, // this Mac's MLA heads: all, or its share in TP2
    nope: u32 = 256,
    v_dim: u32 = 256,
    q_lora: u32 = 1536,
    kv_lora: u32 = 512,
    i_heads: u32 = 32,
    i_dim: u32 = 128,
    i_topk: u32 = 2048,
    kpool: u32 = 4,
    hc: u32 = 4,
    hc_eps: f32 = 1e-6,
    sinkhorn: u32 = 20,
    mtp: u32 = 1,
    tp: u32 = 1, // tensor parallel: the Macs splitting the attention heads
    tp_rank: u32 = 0,
    eos: [3]u32 = .{ 154820, 154827, 154829 },
    eos_n: u32 = 3,
    /// Layer i is MLA (sparse attention) when set; KDA otherwise.
    mla: std.StaticBitSet(max_layers) = defaultMla(),

    pub fn kind(c: *const Config, i: u32) Kind {
        return if (c.mla.isSet(i)) .mla else .kda;
    }

    /// Expert parallel by rows: this Mac holds part of every routed expert's intermediate rows.
    pub fn byRows(c: *const Config) bool {
        return c.inter[1] - c.inter[0] != c.moe_inter;
    }

    pub fn isMoe(c: *const Config, i: u32) bool {
        return i >= c.dense_layers;
    }

    pub fn kdaWidth(c: *const Config) u32 {
        return c.kda_heads * c.kda_dim;
    }

    /// The stacked KDA input projection's rows: q, k, v, f_a, g_a, b.
    pub fn kdaProj(c: *const Config) u32 {
        return 3 * c.kdaWidth() + 2 * c.kda_dim + c.kda_heads;
    }

    /// The stacked MLA projection of the layer input: q_a, kv_a, the indexer's keys and head weights.
    pub fn xProj(c: *const Config) u32 {
        return c.q_lora + c.kv_lora + c.i_dim + c.i_heads;
    }

    /// The stacked projection of the normed q_a: q_b and the indexer's queries.
    pub fn qrProj(c: *const Config) u32 {
        return c.mla_heads * c.nope + c.i_heads * c.i_dim;
    }

    /// Keys a sparse row attends: index_topk from whole blocks plus the tail of its partial block.
    pub fn keyWidth(c: *const Config) u32 {
        return c.i_topk + c.kpool - 1;
    }

    /// A decode row's key list in blocks of `attn_span`, each block's softmax its own, combined in block order.
    pub fn attnBlocks(c: *const Config) u32 {
        return (c.keyWidth() + attn_span - 1) / attn_span;
    }

    /// The head's rows [lo, hi) this Mac computes: the whole vocabulary, or its half in TP2.
    pub fn vocabPart(c: *const Config) [2]u32 {
        const n = c.vocab / c.tp;
        return .{ c.tp_rank * n, if (c.tp_rank + 1 == c.tp) c.vocab else (c.tp_rank + 1) * n };
    }

    pub fn countKind(c: *const Config, k: Kind) u32 {
        var n: u32 = 0;
        for (0..c.layers) |i| n += @intFromBool(c.kind(@intCast(i)) == k);
        return n;
    }

    /// Dense index of layer i among the layers of its kind (state arenas are per kind).
    pub fn kindIndex(c: *const Config, i: u32) u32 {
        var n: u32 = 0;
        for (0..i) |j| n += @intFromBool(c.kind(@intCast(j)) == c.kind(i));
        return n;
    }
};

fn defaultMla() std.StaticBitSet(max_layers) {
    var s = std.StaticBitSet(max_layers).empty;
    var l: usize = 3;
    while (l < 45) : (l += 4) s.set(l);
    s.set(45); // the MTP layer
    return s;
}

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

/// text_config with GLM-5.3-Flash's widths; smaller counts pass (synthetic), experts >= top-k, vocab a multiple of 128.
pub fn parse(gpa: std.mem.Allocator, json: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const t = (root.get("text_config") orelse parsed.value).object;
    var c = Config{};
    const want = c;
    c.hidden = @intCast(try int(t, "hidden_size"));
    c.layers = @intCast(try int(t, "num_hidden_layers"));
    c.run = c.layers;
    c.vocab = @intCast(try int(t, "vocab_size"));
    if (root.get("quantization_config")) |q| if (q == .object) if (q.object.get("quant_method")) |m| if (m == .string and std.mem.eql(u8, m.string, "exl3")) {
        c.head_unit = 128;
    };
    c.eps = @floatCast(try float(t, "rms_norm_eps"));
    c.dense_layers = @intCast(try int(t, "first_k_dense_replace"));
    c.dense_inter = @intCast(try int(t, "intermediate_size"));
    c.experts = @intCast(try int(t, "n_routed_experts"));
    c.own = .{ 0, c.experts };
    c.inter = .{ 0, c.moe_inter };
    c.topk = @intCast(try int(t, "num_experts_per_tok"));
    c.moe_inter = @intCast(try int(t, "moe_intermediate_size"));
    c.shared_inter = c.moe_inter * @as(u32, @intCast(if (t.get("n_shared_experts")) |v| v.integer else 1));
    if (t.get("norm_topk_prob")) |v| c.norm_topk = v != .bool or v.bool;
    c.routed_scale = @floatCast(try float(t, "routed_scaling_factor"));
    c.swiglu_limit = @floatCast(try float(t, "swiglu_limit"));
    c.mla_heads = @intCast(try int(t, "num_attention_heads"));
    c.nope = @intCast(try int(t, "qk_nope_head_dim"));
    c.v_dim = @intCast(try int(t, "v_head_dim"));
    c.q_lora = @intCast(try int(t, "q_lora_rank"));
    c.kv_lora = @intCast(try int(t, "kv_lora_rank"));
    c.i_heads = @intCast(try int(t, "index_n_heads"));
    c.i_dim = @intCast(try int(t, "index_head_dim"));
    c.i_topk = @intCast(try int(t, "index_topk"));
    c.kpool = @intCast(try int(t, "index_kpool"));
    c.hc = @intCast(try int(t, "hc_mult"));
    c.hc_eps = @floatCast(try float(t, "hc_eps"));
    c.sinkhorn = @intCast(try int(t, "hc_sinkhorn_iters"));
    c.mtp = @intCast(try int(t, "num_nextn_predict_layers"));
    const la = (t.get("linear_attn_config") orelse return error.BadConfig).object;
    c.kda_heads = @intCast(try int(la, "num_heads"));
    c.kda_dim = @intCast(try int(la, "head_dim"));
    c.conv = @intCast(try int(la, "short_conv_kernel_size"));
    c.lower_bound = @floatCast(try float(la, "gate_lower_bound"));
    c.mla = std.StaticBitSet(max_layers).empty;
    for ((la.get("full_attn_layers") orelse return error.BadConfig).array.items) |v| {
        if (v != .integer or v.integer < 0 or v.integer >= max_layers) return error.BadConfig;
        c.mla.set(@intCast(v.integer));
    }
    if (c.layers + c.mtp > max_layers) return error.BadConfig;
    if (c.mtp > 0) c.mla.set(c.layers);
    if (t.get("eos_token_id")) |v| if (v == .array and v.array.items.len <= 3) {
        c.eos_n = @intCast(v.array.items.len);
        for (v.array.items, 0..) |e, i| c.eos[i] = @intCast(e.integer);
    };
    const rope = t.get("qk_rope_head_dim");
    const plain = (if (t.get("n_group")) |v| v.integer else 1) == 1 and (if (t.get("topk_group")) |v| v.integer else 1) == 1;
    const widths = c.hidden == want.hidden and c.dense_inter == want.dense_inter and c.topk == want.topk and
        c.moe_inter == want.moe_inter and c.shared_inter == want.shared_inter and c.mla_heads == want.mla_heads and
        c.nope == want.nope and c.v_dim == want.v_dim and c.q_lora == want.q_lora and c.kv_lora == want.kv_lora and
        c.i_heads == want.i_heads and c.i_dim == want.i_dim and c.i_topk == want.i_topk and c.kpool == want.kpool and
        c.hc == want.hc and c.sinkhorn == want.sinkhorn and c.kda_heads == want.kda_heads and
        c.kda_dim == want.kda_dim and c.conv == want.conv and plain and
        (rope == null or (rope.? == .integer and rope.?.integer == 0));
    var eos_in = true;
    for (c.eos[0..c.eos_n]) |id| eos_in = eos_in and id < c.vocab;
    const counts = c.layers >= 1 and c.layers <= want.layers and c.dense_layers <= c.layers and
        c.experts >= c.topk and c.experts <= want.experts and c.vocab <= want.vocab and c.vocab % 128 == 0 and eos_in;
    if (!widths or !counts) return error.NotGlm53Flash;
    return c;
}

/// GLM-5.3-Flash's own counts, the shapes the Metal kernels are built for (parse also passes smaller synthetic ones).
pub fn whole(c: Config) bool {
    const want = Config{};
    return c.layers == want.layers and c.vocab == want.vocab and c.dense_layers == want.dense_layers and
        c.experts == want.experts and c.mla.eql(want.mla);
}

/// The first `n` backbone layers only (the MTP layer stays): a check whose weights must fit where the whole model does not.
pub fn subset(c: *Config, n: u32) !void {
    if (n == 0 or n > c.layers) return error.BadLayerCount;
    c.run = n;
}

/// Expert parallel by rows over `ranks` Macs: rank r holds every routed expert's intermediate rows [r n, (r + 1) n).
pub fn splitRows(c: *Config, rank: u32, ranks: u32) !void {
    if (ranks == 0 or rank >= ranks or c.moe_inter % (64 * ranks) != 0) return error.BadExpertSplit;
    const n = c.moe_inter / ranks;
    c.own = .{ 0, c.experts };
    c.inter = .{ rank * n, (rank + 1) * n };
}

/// Tensor parallel over `ranks` Macs: rank r runs its share of every layer's KDA and MLA heads and dense MLP rows.
pub fn splitHeads(c: *Config, rank: u32, ranks: u32) !void {
    if (ranks == 0 or rank >= ranks or c.tp != 1) return error.BadHeadSplit;
    if (c.kda_heads % ranks != 0 or c.mla_heads % ranks != 0 or c.dense_inter % (64 * ranks) != 0) return error.BadHeadSplit;
    c.kda_heads /= ranks;
    c.mla_heads /= ranks;
    c.dense_inter /= ranks;
    c.tp = ranks;
    c.tp_rank = rank;
}

/// Expert parallel over `ranks` Macs: rank r holds routed experts [r * experts / ranks, (r + 1) * experts / ranks).
pub fn split(c: *Config, rank: u32, ranks: u32) !void {
    if (ranks == 0 or rank >= ranks or c.experts % ranks != 0) return error.BadExpertSplit;
    const per = c.experts / ranks;
    c.own = .{ rank * per, (rank + 1) * per };
}

test "GLM-5.3-Flash's layer kinds: 34 KDA and 11 MLA in the backbone, the MTP layer MLA" {
    const c = Config{};
    try std.testing.expectEqual(@as(u32, 34), c.countKind(.kda));
    try std.testing.expectEqual(@as(u32, 11), c.countKind(.mla));
    try std.testing.expectEqual(Kind.mla, c.kind(3));
    try std.testing.expectEqual(Kind.mla, c.kind(43));
    try std.testing.expectEqual(Kind.kda, c.kind(44));
    try std.testing.expectEqual(Kind.mla, c.kind(45));
    try std.testing.expectEqual(@as(u32, 24896), c.kdaProj());
    try std.testing.expectEqual(@as(u32, 2208), c.xProj());
    try std.testing.expectEqual(@as(u32, 20480), c.qrProj());
    try std.testing.expectEqual(@as(u32, 10), c.kindIndex(43));
}

/// GLM-5.3-Flash's text_config with the counts given (the widths as the checkpoint has them), for the tests.
pub fn textConfig(gpa: std.mem.Allocator, layers: u32, dense: u32, mla: []const u32, experts: u32, vocab: u32, hidden: u32) ![]u8 {
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    for (mla, 0..) |l, i| try list.print(gpa, "{s}{d}", .{ if (i > 0) ", " else "", l });
    return std.fmt.allocPrint(gpa,
        \\{{"model_type": "glm5_next", "text_config": {{"hidden_size": {d}, "num_hidden_layers": {d}, "vocab_size": {d},
        \\"rms_norm_eps": 1e-05, "first_k_dense_replace": {d}, "intermediate_size": 12288, "n_routed_experts": {d},
        \\"num_experts_per_tok": 8, "moe_intermediate_size": 2048, "n_shared_experts": 1, "norm_topk_prob": true,
        \\"routed_scaling_factor": 2.5, "swiglu_limit": 10.0, "num_attention_heads": 64, "qk_nope_head_dim": 256,
        \\"v_head_dim": 256, "q_lora_rank": 1536, "kv_lora_rank": 512, "index_n_heads": 32, "index_head_dim": 128,
        \\"index_topk": 2048, "index_kpool": 4, "hc_mult": 4, "hc_eps": 1e-06, "hc_sinkhorn_iters": 20,
        \\"num_nextn_predict_layers": 1, "eos_token_id": [{d}], "linear_attn_config": {{"num_heads": 64, "head_dim": 128,
        \\"short_conv_kernel_size": 4, "gate_lower_bound": -5.0, "full_attn_layers": [{s}]}}}}}}
    , .{ hidden, layers, vocab, dense, experts, vocab - 2, list.items });
}

test "the real counts parse to the defaults; smaller counts parse, other widths do not" {
    const gpa = std.testing.allocator;
    const real = try textConfig(gpa, 45, 3, &.{ 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 }, 288, 154880, 4096);
    defer gpa.free(real);
    const c = try parse(gpa, real);
    try std.testing.expect(c.mla.eql((Config{}).mla));
    try std.testing.expectEqual(@as(u32, 45), c.layers);
    const small = try textConfig(gpa, 2, 1, &.{1}, 16, 8192, 4096);
    defer gpa.free(small);
    const s = try parse(gpa, small);
    try std.testing.expect(whole(c) and !whole(s)); // the Metal engine refuses the synthetic one
    try std.testing.expectEqual(Kind.kda, s.kind(0));
    try std.testing.expectEqual(Kind.mla, s.kind(1));
    try std.testing.expectEqual(Kind.mla, s.kind(2)); // the MTP layer
    try std.testing.expectEqual(@as(u32, 16), s.experts);
    const wide = try textConfig(gpa, 2, 1, &.{1}, 16, 8192, 2048);
    defer gpa.free(wide);
    try std.testing.expectError(error.NotGlm53Flash, parse(gpa, wide));
    const few = try textConfig(gpa, 2, 1, &.{1}, 4, 8192, 4096);
    defer gpa.free(few);
    try std.testing.expectError(error.NotGlm53Flash, parse(gpa, few));
}

test "a layer subset keeps the MTP layer's place and refuses an empty or oversized count" {
    var c = Config{};
    try subset(&c, 8);
    try std.testing.expectEqual(@as(u32, 8), c.run);
    try std.testing.expectEqual(@as(u32, 45), c.layers);
    try std.testing.expectError(error.BadLayerCount, subset(&c, 0));
    try std.testing.expectError(error.BadLayerCount, subset(&c, 46));
}

test "by rows, two ranks hold every expert's halves" {
    var c = Config{};
    try splitRows(&c, 1, 2);
    try std.testing.expectEqual([2]u32{ 0, 288 }, c.own);
    try std.testing.expectEqual([2]u32{ 1024, 2048 }, c.inter);
    try std.testing.expect(c.byRows());
    try std.testing.expect(!(Config{}).byRows());
}

test "TP2: each rank runs half the KDA heads; its projection keeps f_a and g_a whole" {
    var c = Config{};
    try splitHeads(&c, 1, 2);
    try std.testing.expectEqual(@as(u32, 32), c.kda_heads);
    try std.testing.expectEqual(@as(u32, 4096), c.kdaWidth());
    try std.testing.expectEqual(@as(u32, 12576), c.kdaProj());
    try std.testing.expectEqual([2]u32{ 77440, 154880 }, c.vocabPart());
    try std.testing.expectEqual(@as(u32, 12288), c.qrProj());
    try std.testing.expectError(error.BadHeadSplit, splitHeads(&c, 0, 2));
    var d = Config{};
    try std.testing.expectError(error.BadHeadSplit, splitHeads(&d, 2, 2));
}

test "two ranks hold the routed experts' halves" {
    var c = Config{};
    try split(&c, 0, 2);
    try std.testing.expectEqual([2]u32{ 0, 144 }, c.own);
    try split(&c, 1, 2);
    try std.testing.expectEqual([2]u32{ 144, 288 }, c.own);
    try std.testing.expectError(error.BadExpertSplit, split(&c, 2, 2));
    try std.testing.expectError(error.BadExpertSplit, split(&c, 0, 5));
}
