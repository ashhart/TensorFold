//! GLM-5.3's shape (model_type glm_moe_dsa) from config.json, and each Mac's TP/PP slice.
const std = @import("std");

pub const max_layers = 96;

pub const Config = struct {
    hidden: u32 = 6144,
    layers: u32 = 78,
    vocab: u32 = 154880,
    eps: f32 = 1e-5,
    dense_layers: u32 = 3,
    dense_inter: u32 = 12288,
    experts: u32 = 256,
    topk: u32 = 8,
    moe_inter: u32 = 2048,
    routed_scale: f32 = 2.5,
    heads: u32 = 64,
    nope: u32 = 192,
    rope: u32 = 64,
    v_dim: u32 = 256,
    q_lora: u32 = 2048,
    kv_lora: u32 = 512,
    rope_theta: f64 = 8e6,
    i_heads: u32 = 32,
    i_dim: u32 = 128,
    i_topk: u32 = 2048,
    mtp: u32 = 1,
    eos: [3]u32 = .{ 154820, 154827, 154829 },
    eos_n: u32 = 3,
    /// Layer i runs its own indexer when set; otherwise it reuses the last full layer's top-k.
    full: std.StaticBitSet(max_layers) = .empty,
    /// World & TP/PP coordinates.
    rank: u32 = 0,
    ranks: u32 = 1,
    tp: u32 = 1,
    pp: u32 = 1,
    tp_rank: u32 = 0,
    pp_rank: u32 = 0,
    /// Owned slice ranges (explicit so heterogeneous nodes can hold proportional shares).
    layer0: u32 = 0,
    layer1: u32 = 78,
    head_lo: u32 = 0,
    head_hi: u32 = 64,
    half_lo: u32 = 0,
    half_hi: u32 = 16,
    dense_lo: u32 = 0,
    dense_hi: u32 = 12288,
    vocab_lo: u32 = 0,
    vocab_hi: u32 = 154880,

    pub fn isMoe(c: *const Config, i: u32) bool {
        return i >= c.dense_layers;
    }
    pub fn qHead(c: *const Config) u32 {
        return c.nope + c.rope;
    }
    /// The attention heads this Mac runs: [head0, head0 + localHeads).
    pub fn localHeads(c: *const Config) u32 {
        return c.head_hi - c.head_lo;
    }
    pub fn head0(c: *const Config) u32 {
        return c.head_lo;
    }
    /// Each routed or shared expert's 128-channel halves this Mac holds: [half0, half0 + localHalves).
    pub fn localHalves(c: *const Config) u32 {
        return c.half_hi - c.half_lo;
    }
    pub fn half0(c: *const Config) u32 {
        return c.half_lo;
    }
    pub fn localDense(c: *const Config) u32 {
        return c.dense_hi - c.dense_lo;
    }
    pub fn dense0(c: *const Config) u32 {
        return c.dense_lo;
    }
    /// The head's vocabulary rows [lo, hi) this Mac computes.
    pub fn vocabPart(c: *const Config) [2]u32 {
        return .{ c.vocab_lo, c.vocab_hi };
    }
    pub fn localLayers(c: *const Config) u32 {
        return c.layer1 - c.layer0;
    }
    pub fn isFirstStage(c: *const Config) bool {
        return c.pp_rank == 0;
    }
    pub fn isLastStage(c: *const Config) bool {
        return c.pp_rank + 1 == c.pp;
    }
    /// The full layer whose top-k layer i uses: itself when it has an indexer.
    pub fn indexFrom(c: *const Config, i: u32) u32 {
        var l = i;
        while (l > 0 and !c.full.isSet(l)) l -= 1;
        return l;
    }
};

fn int(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = o.get(key) orelse return error.BadConfig;
    return if (v == .integer) v.integer else error.BadConfig;
}

fn float(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.BadConfig;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn str(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

/// config.json of a glm_moe_dsa checkpoint; a shape the kernels do not take is refused.
pub fn parse(gpa: std.mem.Allocator, json: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const t = parsed.value.object;
    if (!std.mem.eql(u8, str(t, "model_type") orelse "", "glm_moe_dsa")) return error.NotGlmMoeDsa;
    var c: Config = .{};
    c.hidden = @intCast(try int(t, "hidden_size"));
    c.layers = @intCast(try int(t, "num_hidden_layers"));
    c.vocab = @intCast(try int(t, "vocab_size"));
    c.eps = @floatCast(try float(t, "rms_norm_eps"));
    c.dense_layers = @intCast(try int(t, "first_k_dense_replace"));
    c.dense_inter = @intCast(try int(t, "intermediate_size"));
    c.experts = @intCast(try int(t, "n_routed_experts"));
    c.topk = @intCast(try int(t, "num_experts_per_tok"));
    c.moe_inter = @intCast(try int(t, "moe_intermediate_size"));
    c.routed_scale = @floatCast(try float(t, "routed_scaling_factor"));
    c.heads = @intCast(try int(t, "num_attention_heads"));
    c.nope = @intCast(try int(t, "qk_nope_head_dim"));
    c.rope = @intCast(try int(t, "qk_rope_head_dim"));
    c.v_dim = @intCast(try int(t, "v_head_dim"));
    c.q_lora = @intCast(try int(t, "q_lora_rank"));
    c.kv_lora = @intCast(try int(t, "kv_lora_rank"));
    c.i_heads = @intCast(try int(t, "index_n_heads"));
    c.i_dim = @intCast(try int(t, "index_head_dim"));
    c.i_topk = @intCast(try int(t, "index_topk"));
    c.mtp = @intCast(try int(t, "num_nextn_predict_layers"));
    c.layer0 = 0;
    c.layer1 = c.layers;
    c.head_lo = 0;
    c.head_hi = c.heads;
    c.half_lo = 0;
    c.half_hi = c.moe_inter / 128;
    c.dense_lo = 0;
    c.dense_hi = c.dense_inter;
    c.vocab_lo = 0;
    c.vocab_hi = c.vocab;
    const rp = t.get("rope_parameters") orelse return error.BadConfig;
    if (rp != .object or !std.mem.eql(u8, str(rp.object, "rope_type") orelse "", "default")) return error.UnsupportedRope;
    c.rope_theta = try float(rp.object, "rope_theta");
    const types = t.get("indexer_types") orelse return error.BadConfig;
    if (types != .array or types.array.items.len != c.layers or c.layers + c.mtp > max_layers) return error.BadConfig;
    for (types.array.items, 0..) |v, i| {
        if (v != .string) return error.BadConfig;
        if (std.mem.eql(u8, v.string, "full")) c.full.set(i) else if (!std.mem.eql(u8, v.string, "shared")) return error.BadConfig;
    }
    if (!c.full.isSet(0)) return error.BadConfig;
    if (c.mtp > 0) c.full.set(c.layers);
    if (t.get("eos_token_id")) |v| if (v == .array and v.array.items.len <= 3) {
        c.eos_n = @intCast(v.array.items.len);
        for (v.array.items, 0..) |e, i| c.eos[i] = @intCast(e.integer);
    };
    const plain = (if (t.get("n_group")) |v| v.integer else 1) == 1 and (if (t.get("topk_group")) |v| v.integer else 1) == 1;
    const sigmoid = std.mem.eql(u8, str(t, "scoring_func") orelse "sigmoid", "sigmoid");
    const shaped = c.hidden == 6144 and c.kv_lora == 512 and c.rope == 64 and c.nope == 192 and c.v_dim == 256 and
        c.q_lora == 2048 and c.i_heads == 32 and c.i_dim == 128 and c.i_topk == 2048 and c.moe_inter == 2048;
    if (!plain or !sigmoid or !shaped) return error.UnsupportedGlmShape;
    return c;
}

/// Even tensor parallel over `ranks` Macs (1, 2, 4, 8): heads, dense rows, expert halves and vocab split evenly.
pub fn split(c: *Config, rank: u32, ranks: u32) !void {
    if (ranks == 0 or rank >= ranks or c.heads % ranks != 0 or (c.moe_inter / 128) % ranks != 0 or c.dense_inter % (128 * ranks) != 0)
        return error.BadSplit;
    c.rank = rank;
    c.ranks = ranks;
    c.tp = ranks;
    c.pp = 1;
    c.tp_rank = rank;
    c.pp_rank = 0;
    c.layer0 = 0;
    c.layer1 = c.layers;
    const lh = c.heads / ranks;
    c.head_lo = rank * lh;
    c.head_hi = (rank + 1) * lh;
    const lm = (c.moe_inter / 128) / ranks;
    c.half_lo = rank * lm;
    c.half_hi = (rank + 1) * lm;
    const ld = c.dense_inter / ranks;
    c.dense_lo = rank * ld;
    c.dense_hi = (rank + 1) * ld;
    const lv = c.vocab / ranks;
    c.vocab_lo = rank * lv;
    c.vocab_hi = if (rank + 1 == ranks) c.vocab else (rank + 1) * lv;
}

/// Apply N-rank TP, PP, hybrid TPxPP, or heterogeneous shard ranges from `tp_settings.Settings`.
pub fn splitSettings(c: *Config, s: anytype, layer_override: ?u32) !void {
    const total_layers = layer_override orelse c.layers;
    if (total_layers == 0 or total_layers > c.layers) return error.BadGlmLayers;
    const ranks = s.worldSize();
    const tp = s.tpSize();
    const pp = s.ppSize();
    const tp_rank = s.tpRank();
    const pp_rank = s.ppRank();
    if (ranks == 0 or tp == 0 or pp == 0 or tp * pp != ranks or s.rank >= ranks) return error.BadSplit;
    c.rank = s.rank;
    c.ranks = ranks;
    c.tp = tp;
    c.pp = pp;
    c.tp_rank = tp_rank;
    c.pp_rank = pp_rank;
    // Pipeline layer range (explicit heterogeneous range or even division snapped to full-indexer layers).
    if (s.layer_range) |lr| {
        if (lr[0] >= lr[1] or lr[1] > total_layers) return error.BadSplit;
        if (lr[0] > 0 and !c.full.isSet(lr[0])) return error.BadSplit;
        c.layer0 = lr[0];
        c.layer1 = lr[1];
    } else if (pp == 1) {
        c.layer0 = 0;
        c.layer1 = total_layers;
    } else {
        const snap = struct {
            fn toFull(cfg_: *const Config, raw: u32, max_l: u32) u32 {
                if (raw == 0 or raw >= max_l) return raw;
                var l = raw;
                while (l > 0 and !cfg_.full.isSet(l)) l -= 1;
                return l;
            }
        }.toFull;
        const base = total_layers / pp;
        const extra = total_layers % pp;
        const r0 = pp_rank * base + @min(pp_rank, extra);
        const r1 = if (pp_rank + 1 == pp) total_layers else (pp_rank + 1) * base + @min(pp_rank + 1, extra);
        c.layer0 = snap(c, r0, total_layers);
        c.layer1 = if (pp_rank + 1 == pp) total_layers else snap(c, r1, total_layers);
        if (c.layer0 >= c.layer1) return error.BadSplit;
    }
    // Attention heads (must be even-aligned so kv_b_proj 448*h0 aligns to FP8 128-row scale blocks).
    if (s.head_range) |hr| {
        if (hr[0] >= hr[1] or hr[1] > c.heads or hr[0] % 2 != 0 or hr[1] % 2 != 0) return error.BadSplit;
        c.head_lo = hr[0];
        c.head_hi = hr[1];
    } else {
        if (c.heads % tp != 0) return error.BadSplit;
        const lh = c.heads / tp;
        if (lh % 2 != 0 and tp > 1) return error.BadSplit;
        c.head_lo = tp_rank * lh;
        c.head_hi = (tp_rank + 1) * lh;
    }
    // MoE 128-channel halves.
    const total_halves = c.moe_inter / 128;
    if (s.moe_half_range) |mr| {
        if (mr[0] >= mr[1] or mr[1] > total_halves) return error.BadSplit;
        c.half_lo = mr[0];
        c.half_hi = mr[1];
    } else {
        if (total_halves % tp != 0) return error.BadSplit;
        const lm = total_halves / tp;
        c.half_lo = tp_rank * lm;
        c.half_hi = (tp_rank + 1) * lm;
    }
    // Dense MLP intermediate channels (128-aligned for FP8 scale blocks).
    if (s.dense_range) |dr| {
        if (dr[0] >= dr[1] or dr[1] > c.dense_inter or dr[0] % 128 != 0 or dr[1] % 128 != 0) return error.BadSplit;
        c.dense_lo = dr[0];
        c.dense_hi = dr[1];
    } else {
        if (c.dense_inter % (128 * tp) != 0) return error.BadSplit;
        const ld = c.dense_inter / tp;
        c.dense_lo = tp_rank * ld;
        c.dense_hi = (tp_rank + 1) * ld;
    }
    // Vocab rows for lm_head.
    if (s.vocab_range) |vr| {
        if (vr[0] >= vr[1] or vr[1] > c.vocab) return error.BadSplit;
        c.vocab_lo = vr[0];
        c.vocab_hi = vr[1];
    } else {
        const lv = c.vocab / tp;
        c.vocab_lo = tp_rank * lv;
        c.vocab_hi = if (tp_rank + 1 == tp) c.vocab else (tp_rank + 1) * lv;
    }
}

test "GLM-5.3 indexer pattern and TP=2, TP=4, and heterogeneous TP+PP splits" {
    var c: Config = .{};
    for ([_]u32{ 0, 1, 2, 6, 10, 74 }) |i| c.full.set(i);
    try std.testing.expectEqual(@as(u32, 2), c.indexFrom(5));
    try std.testing.expectEqual(@as(u32, 6), c.indexFrom(9));
    try std.testing.expectEqual(@as(u32, 74), c.indexFrom(77));
    try split(&c, 1, 2);
    try std.testing.expectEqual(@as(u32, 32), c.head0());
    try std.testing.expectEqual(@as(u32, 8), c.half0());
    try std.testing.expectEqual([2]u32{ 77440, 154880 }, c.vocabPart());
    try split(&c, 3, 4);
    try std.testing.expectEqual(@as(u32, 48), c.head0());
    try std.testing.expectEqual(@as(u32, 16), c.localHeads());
    try std.testing.expectEqual(@as(u32, 12), c.half0());
    try std.testing.expectEqual(@as(u32, 4), c.localHalves());
    try std.testing.expectEqual(@as(u32, 9216), c.dense0());
    try std.testing.expectEqual(@as(u32, 3072), c.localDense());
    try std.testing.expectError(error.BadSplit, split(&c, 0, 3));
}
