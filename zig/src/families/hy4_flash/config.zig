//! Hy4-flash's shape from config.json (hy_v4_flash), refusing every layout the engine does not run.
const std = @import("std");

/// Backbone layers the per-layer tables hold (Hy4-flash has 42).
pub const max_layers = 64;

/// The most drafter taps into the backbone (Hy4-flash's DFly has 5).
pub const max_taps = 8;

/// A layer's attention: sparse layers pick rows with the indexer, sliding layers read the last `window` rows.
pub const Kind = enum { sparse, sliding };

/// One MLA geometry: sliding layers read the `swa_*` keys, sparse layers the plain ones.
pub const Heads = struct { q: u32, nope: u32, rope: u32, v: u32 };

/// Why parse refused the config, one sentence for the caller's log line (tests read it too).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Text = struct {
    hidden: u32,
    layers: u32,
    vocab: u32,
    eps: f32,
    max_context: u32,
    sparse: Heads,
    sliding: Heads,
    q_rank: u32,
    kv_rank: u32, // the latent a KV row stores, beside its rope part
    window: u32,
    index_topk: u32,
    index_heads: u32,
    index_dim: u32,
    compress: u32, // rows one compressed indexer key stands for
    group_topk: u32,
    hc: u32, // residual streams of the identity hyper-connections
    dense_inter: u32,
    experts: u32,
    topk: u32,
    latent: u32, // the routed experts' input and output width
    expert_inter: u32,
    shared: u32,
    route_scale: f32,
    swiglu_limit: f32,
    kinds: [max_layers]Kind,
    moe: [max_layers]bool, // false: the layer's MLP is dense
    engram: [max_layers]bool,

    pub fn heads(c: *const Text, i: u32) Heads {
        return if (c.kinds[i] == .sliding) c.sliding else c.sparse;
    }

    pub fn count(c: *const Text, k: Kind) u32 {
        var n: u32 = 0;
        for (c.kinds[0..c.layers]) |x| n += @intFromBool(x == k);
        return n;
    }

    /// Bytes of one fp8 KV row: the e4m3 latent, its four fp32 tile scales and the bf16 rope part.
    pub fn kvRowBytes(c: *const Text) u32 {
        return c.kv_rank + 16 + 2 * c.sparse.rope;
    }
};

/// The n-gram tables: `heads` hash heads for each n-gram size from 2 to `max_ngram`, `dim` values a head.
pub const Engram = struct { layers: u32, max_ngram: u32, heads: u32, dim: u32, pad: u32, fp8: bool };

/// The checkpoint's own drafter (DFly): its layers, tokens a round and the backbone layers it reads.
pub const Drafter = struct { layers: u32, tokens: u32, taps: [max_taps]u32, taps_n: u32 };

pub const Config = struct {
    text: Text,
    engram: Engram,
    drafter: ?Drafter,
    has_vision: bool,
    eos: u32,
    image_token: u32,
};

fn get(o: std.json.ObjectMap, key: []const u8, why: *Why) !std.json.Value {
    return o.get(key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
}

fn int(o: std.json.ObjectMap, key: []const u8, why: *Why) !u32 {
    const v = try get(o, key, why);
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) {
        why.set("config.json's {s} is not a whole number", .{key});
        return error.BadConfig;
    }
    return @intCast(v.integer);
}

fn float(o: std.json.ObjectMap, key: []const u8, why: *Why) !f32 {
    return switch (try get(o, key, why)) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => {
            why.set("config.json's {s} is not a number", .{key});
            return error.BadConfig;
        },
    };
}

fn flag(o: std.json.ObjectMap, key: []const u8, default: bool, why: *Why) !bool {
    const v = o.get(key) orelse return default;
    if (v != .bool) {
        why.set("config.json's {s} is not true or false", .{key});
        return error.BadConfig;
    }
    return v.bool;
}

fn text(o: std.json.ObjectMap, key: []const u8) []const u8 {
    const v = o.get(key) orelse return "";
    return if (v == .string) v.string else "";
}

fn list(o: std.json.ObjectMap, key: []const u8, n: u32, why: *Why) ![]const std.json.Value {
    const v = try get(o, key, why);
    if (v != .array or v.array.items.len != n) {
        why.set("config.json's {s} must list {d} layers", .{ key, n });
        return error.LayerList;
    }
    return v.array.items;
}

/// A switch the engine runs only one way: `key` must read `want` (a missing key reads `default`).
fn only(o: std.json.ObjectMap, key: []const u8, default: bool, want: bool, err: anyerror, why: *Why) !void {
    if (try flag(o, key, default, why) == want) return;
    why.set("{s} must be {}: the engine runs no other", .{ key, want });
    return err;
}

/// A rank key the engine runs without: absent or null.
fn none(o: std.json.ObjectMap, key: []const u8, why: *Why) !void {
    const v = o.get(key) orelse return;
    if (v == .null) return;
    why.set("{s} must be null: the engine's gate and output projections are direct", .{key});
    return error.Mla;
}

fn mla(o: std.json.ObjectMap, c: *Text, why: *Why) !void {
    try only(o, "use_mla", true, true, error.Mla, why);
    try only(o, "gated_mla", false, true, error.Mla, why);
    try only(o, "learnable_sink", false, true, error.Mla, why);
    try only(o, "qk_norm", true, true, error.Mla, why);
    try only(o, "full_nope", false, false, error.Mla, why);
    try only(o, "attention_bias", false, false, error.Mla, why);
    if (!std.mem.eql(u8, text(o, "gating_type"), "elementwise")) {
        why.set("gating_type must be \"elementwise\": the engine runs no other", .{});
        return error.Mla;
    }
    try none(o, "gate_lora_rank", why);
    try none(o, "o_lora_rank", why);
    c.sparse = .{
        .q = try int(o, "num_attention_heads", why),
        .nope = try int(o, "qk_nope_head_dim", why),
        .rope = try int(o, "qk_rope_head_dim", why),
        .v = try int(o, "v_head_dim", why),
    };
    c.sliding = .{
        .q = try int(o, "swa_num_attention_heads", why),
        .nope = try int(o, "swa_qk_nope_head_dim", why),
        .rope = try int(o, "swa_qk_rope_head_dim", why),
        .v = try int(o, "swa_v_head_dim", why),
    };
    c.q_rank = try int(o, "q_lora_rank", why);
    c.kv_rank = try int(o, "kv_lora_rank", why);
    if (c.kv_rank != 512 or c.sparse.rope != 64 or c.sliding.rope != 64) {
        why.set("kv_lora_rank {d} with rope {d}/{d}: a KV row holds a 512 latent and 64 rope values", .{ c.kv_rank, c.sparse.rope, c.sliding.rope });
        return error.Mla;
    }
    if (c.sparse.q == 0 or c.sliding.q == 0 or c.q_rank == 0) {
        why.set("{d} sparse and {d} sliding heads at q rank {d}: not an attention shape", .{ c.sparse.q, c.sliding.q, c.q_rank });
        return error.Mla;
    }
}

fn indexer(o: std.json.ObjectMap, c: *Text, why: *Why) !void {
    try only(o, "use_dsa", false, true, error.Indexer, why);
    try only(o, "use_compressed_indexer", false, true, error.Indexer, why);
    c.window = try int(o, "sliding_window", why);
    c.index_topk = try int(o, "index_topk", why);
    c.index_heads = try int(o, "index_n_heads", why);
    c.index_dim = try int(o, "index_head_dim", why);
    c.compress = try int(o, "compressed_indexer_compress_ratio", why);
    c.group_topk = try int(o, "compressed_indexer_group_topk", why);
    if (c.index_topk == 0 or c.index_topk % 128 != 0 or c.index_topk < c.window or c.window == 0) {
        why.set("index_topk {d} with window {d}: the top-k is in 128-row tiles and covers the window", .{ c.index_topk, c.window });
        return error.Indexer;
    }
    if (c.compress == 0 or c.group_topk == 0 or c.group_topk * c.compress != c.index_topk) {
        why.set("{d} groups of {d} rows for a top-k of {d}: the groups must fill it", .{ c.group_topk, c.compress, c.index_topk });
        return error.Indexer;
    }
}

fn mlp(o: std.json.ObjectMap, c: *Text, why: *Why) !void {
    if (!std.mem.eql(u8, text(o, "hidden_act"), "silu")) {
        why.set("hidden_act must be \"silu\": the engine runs no other", .{});
        return error.Activation;
    }
    c.dense_inter = try int(o, "intermediate_size", why);
    c.experts = try int(o, "n_routed_experts", why);
    c.topk = try int(o, "num_experts_per_tok", why);
    c.latent = try int(o, "routed_expert_hidden_size", why);
    c.expert_inter = try int(o, "moe_intermediate_size", why);
    c.shared = try int(o, "n_shared_experts", why);
    c.route_scale = try float(o, "routed_scaling_factor", why);
    c.swiglu_limit = try float(o, "swiglu_limit", why);
    if (c.topk == 0 or c.topk > c.experts or c.latent == 0 or c.expert_inter == 0) {
        why.set("{d} of {d} experts at latent {d}, width {d}: not an expert shape", .{ c.topk, c.experts, c.latent, c.expert_inter });
        return error.Router;
    }
    if (try int(o, "n_group", why) != 1 or try int(o, "topk_group", why) != 1) {
        why.set("n_group and topk_group must be 1: the engine's router has no groups", .{});
        return error.Router;
    }
    try only(o, "moe_router_use_sigmoid", false, true, error.Router, why);
    try only(o, "norm_topk_prob", false, true, error.Router, why);
}

fn layerTables(o: std.json.ObjectMap, c: *Text, why: *Why) !void {
    const kinds = try list(o, "layer_types", c.layers, why);
    const mlps = try list(o, "mlp_layer_types", c.layers, why);
    const swa = try list(o, "swa_pattern", c.layers, why);
    for (0..c.layers) |i| {
        const k = if (kinds[i] == .string) kinds[i].string else "";
        const sliding = std.mem.eql(u8, k, "sliding_attention");
        if (!sliding and !std.mem.eql(u8, k, "deepseek_sparse_attention")) {
            why.set("layer {d} is \"{s}\": the engine runs deepseek_sparse_attention and sliding_attention", .{ i, k });
            return error.LayerList;
        }
        if (swa[i] != .integer or (swa[i].integer == 1) != sliding) {
            why.set("swa_pattern and layer_types disagree at layer {d}", .{i});
            return error.LayerList;
        }
        const m = if (mlps[i] == .string) mlps[i].string else "";
        const dense = std.mem.eql(u8, m, "dense");
        if (!dense and !std.mem.eql(u8, m, "sparse")) {
            why.set("layer {d}'s MLP is \"{s}\": the engine runs dense and sparse", .{ i, m });
            return error.LayerList;
        }
        c.kinds[i] = if (sliding) .sliding else .sparse;
        c.moe[i] = !dense;
    }
}

fn engramTables(o: std.json.ObjectMap, c: *Text, why: *Why) !Engram {
    var e = std.mem.zeroes(Engram);
    e.max_ngram = try int(o, "engram_max_ngram_size", why);
    e.heads = try int(o, "engram_n_head_per_ngram", why);
    e.pad = try int(o, "engram_pad_id", why);
    e.fp8 = try flag(o, "engram_fp8_embedding", false, why);
    const width = try int(o, "engram_n_embed_per_ngram", why);
    if (e.max_ngram < 2 or e.heads == 0 or width % e.heads != 0) {
        why.set("{d}-grams with {d} heads over {d} values: the heads must divide the width", .{ e.max_ngram, e.heads, width });
        return error.Engram;
    }
    e.dim = width / e.heads;
    const ids = try get(o, "engram_layer_ids", why);
    const sizes = try get(o, "engram_layer_head_vocab_sizes", why);
    if (ids != .array or sizes != .object) {
        why.set("engram_layer_ids and engram_layer_head_vocab_sizes: a list and a table", .{});
        return error.Engram;
    }
    try only(o, "enable_ihc", false, true, error.Engram, why);
    for (ids.array.items) |v| {
        // the ids count from 1: id 2 is layers.1 in the checkpoint
        if (v != .integer or v.integer < 1 or v.integer > c.layers) {
            why.set("engram_layer_ids holds a layer outside 1 to {d}", .{c.layers});
            return error.Engram;
        }
        var key: [12]u8 = undefined;
        const name = std.fmt.bufPrint(&key, "{d}", .{v.integer}) catch unreachable;
        const table = sizes.object.get(name);
        if (table == null or table.? != .array or table.?.array.items.len != e.max_ngram - 1) {
            why.set("engram layer {d} has no head sizes for its {d} n-gram sizes", .{ v.integer, e.max_ngram - 1 });
            return error.Engram;
        }
        for (table.?.array.items) |row| if (row != .array or row.array.items.len != e.heads) {
            why.set("engram layer {d}: each n-gram size needs {d} head sizes", .{ v.integer, e.heads });
            return error.Engram;
        };
        c.engram[@intCast(v.integer - 1)] = true;
        e.layers += 1;
    }
    return e;
}

fn drafter(top: std.json.ObjectMap, layers: u32, why: *Why) !?Drafter {
    const spec = top.get("speculative_config") orelse return null;
    if (spec != .object) return null;
    if (!std.mem.eql(u8, text(spec.object, "method"), "dfly")) {
        why.set("speculative_config's method must be \"dfly\": the engine runs no other drafter", .{});
        return error.Drafter;
    }
    var d = std.mem.zeroes(Drafter);
    d.layers = try int(top, "dfly_num_hidden_layers", why);
    d.tokens = try int(spec.object, "num_speculative_tokens", why);
    const taps = try get(top, "dfly_target_kv_layers", why);
    if (taps != .array or taps.array.items.len != d.layers or d.layers > max_taps) {
        why.set("dfly_target_kv_layers must name one backbone layer for each of the {d} drafter layers", .{d.layers});
        return error.Drafter;
    }
    for (taps.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0 or v.integer >= layers) {
            why.set("dfly_target_kv_layers holds a layer outside the {d} backbone layers", .{layers});
            return error.Drafter;
        }
        d.taps[i] = @intCast(v.integer);
    }
    d.taps_n = d.layers;
    return d;
}

/// config.json as the reference (transformers' hy_v4_flash, vLLM's HYV4FlashConfig) reads it.
pub fn parse(gpa: std.mem.Allocator, json: []const u8, why: *Why) !Config {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch {
        why.set("config.json is not JSON", .{});
        return error.BadConfig;
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const top = parsed.value.object;
    const inner = top.get("text_config");
    if (!std.mem.eql(u8, text(top, "model_type"), "hy_v4_flash") or inner == null or inner.? != .object) {
        why.set("not a Hy4-flash checkpoint: model_type hy_v4_flash with a text_config", .{});
        return error.NotHy4Flash;
    }
    const o = inner.?.object;
    var c = std.mem.zeroes(Config);
    const t = &c.text;
    t.hidden = try int(o, "hidden_size", why);
    t.layers = try int(o, "num_hidden_layers", why);
    t.vocab = try int(o, "vocab_size", why);
    t.eps = try float(o, "rms_norm_eps", why);
    t.max_context = try int(o, "max_position_embeddings", why);
    t.hc = try int(o, "hc_mult", why);
    if (t.layers == 0 or t.hidden == 0 or t.vocab == 0 or t.hc == 0) {
        why.set("{d} layers at width {d}, {d} tokens, {d} streams: not a model shape", .{ t.layers, t.hidden, t.vocab, t.hc });
        return error.BadConfig;
    }
    if (t.layers > max_layers) {
        why.set("{d} layers: the loader holds {d}", .{ t.layers, max_layers });
        return error.TooManyLayers;
    }
    try only(o, "sandwich_norm", false, true, error.Norm, why);
    try only(o, "tie_word_embeddings", false, false, error.Norm, why);
    try mla(o, t, why);
    try indexer(o, t, why);
    try mlp(o, t, why);
    try layerTables(o, t, why);
    c.engram = try engramTables(o, t, why);
    c.drafter = try drafter(top, t.layers, why);
    c.has_vision = top.get("vision_config") != null;
    c.eos = try int(o, "eos_token_id", why);
    c.image_token = if (top.get("image_token_id")) |_| try int(top, "image_token_id", why) else 0;
    return c;
}
