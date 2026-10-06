//! GLM-5.3-Flash's shape from config.json's text_config: the engine's kernels are generated for exactly this shape.
const std = @import("std");

pub const max_layers = 46;

pub const Kind = enum { kda, mla };

pub const Config = struct {
    hidden: u32 = 4096,
    layers: u32 = 45,
    run: u32 = 45, // backbone layers loaded and run: all, or the first few for a check that must fit (GLM_LAYERS)
    vocab: u32 = 154880,
    eps: f32 = 1e-5,
    dense_layers: u32 = 3,
    dense_inter: u32 = 12288,
    experts: u32 = 288,
    own: [2]u32 = .{ 0, 288 }, // routed experts [lo, hi) this Mac holds: all, or its half in expert-parallel mode
    inter: [2]u32 = .{ 0, 2048 }, // each routed expert's intermediate rows [lo, hi) this Mac holds (half by rows)
    topk: u32 = 8,
    moe_inter: u32 = 2048,
    routed_scale: f32 = 2.5,
    swiglu_limit: f32 = 10.0,
    kda_heads: u32 = 64,
    kda_dim: u32 = 128,
    conv: u32 = 4,
    lower_bound: f32 = -5.0,
    mla_heads: u32 = 64,
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

/// The checkpoint's text_config, refused unless it is GLM-5.3-Flash's shape (the kernels are built for it).
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
    c.eps = @floatCast(try float(t, "rms_norm_eps"));
    c.dense_layers = @intCast(try int(t, "first_k_dense_replace"));
    c.dense_inter = @intCast(try int(t, "intermediate_size"));
    c.experts = @intCast(try int(t, "n_routed_experts"));
    c.own = .{ 0, c.experts };
    c.inter = .{ 0, c.moe_inter };
    c.topk = @intCast(try int(t, "num_experts_per_tok"));
    c.moe_inter = @intCast(try int(t, "moe_intermediate_size"));
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
    const same = c.hidden == want.hidden and c.layers == want.layers and c.vocab == want.vocab and
        c.dense_layers == want.dense_layers and c.dense_inter == want.dense_inter and c.experts == want.experts and
        c.topk == want.topk and c.moe_inter == want.moe_inter and c.mla_heads == want.mla_heads and
        c.nope == want.nope and c.v_dim == want.v_dim and c.q_lora == want.q_lora and c.kv_lora == want.kv_lora and
        c.i_heads == want.i_heads and c.i_dim == want.i_dim and c.i_topk == want.i_topk and c.kpool == want.kpool and
        c.hc == want.hc and c.sinkhorn == want.sinkhorn and c.kda_heads == want.kda_heads and
        c.kda_dim == want.kda_dim and c.conv == want.conv and c.mla.eql(want.mla) and plain and
        (rope == null or (rope.? == .integer and rope.?.integer == 0));
    if (!same) return error.NotGlm53Flash;
    return c;
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

test "two ranks hold the routed experts' halves" {
    var c = Config{};
    try split(&c, 0, 2);
    try std.testing.expectEqual([2]u32{ 0, 144 }, c.own);
    try split(&c, 1, 2);
    try std.testing.expectEqual([2]u32{ 144, 288 }, c.own);
    try std.testing.expectError(error.BadExpertSplit, split(&c, 2, 2));
    try std.testing.expectError(error.BadExpertSplit, split(&c, 0, 5));
}
