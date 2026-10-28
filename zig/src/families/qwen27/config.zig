//! Dense Qwen text shapes, layer schedule, rotary metadata and ordered checkpoint EOS ids.
const std = @import("std");

pub const Kind = enum { linear, attention };
pub const max_layers = 128;
pub const max_eos = 8;

pub const Config = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    k_heads: usize,
    v_heads: usize,
    dk: usize,
    dv: usize,
    conv_kernel: usize,
    eps: f32,
    rope_dims: usize,
    rope_theta: f64,
    mrope_section: [3]usize = .{ 11, 11, 10 },
    kinds: [max_layers]Kind = @splat(.linear),
    eos: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,
    max_position: usize = 262144,
    mrope_interleaved: bool = true,
    rank: u32 = 0,
    ranks: u32 = 1,
    split_attention: bool = false,

    pub fn kind(c: Config, layer: usize) Kind {
        return c.kinds[layer];
    }

    pub fn qDim(c: Config) usize {
        return 2 * c.heads * c.head_dim;
    }

    pub fn kvDim(c: Config) usize {
        return c.kv_heads * c.head_dim;
    }

    pub fn gdnValueDim(c: Config) usize {
        return c.v_heads * c.dv;
    }

    pub fn gdnQkvDim(c: Config) usize {
        return 2 * c.k_heads * c.dk + c.gdnValueDim();
    }

    pub fn isEos(c: Config, token: u32) bool {
        return std.mem.indexOfScalar(u32, c.eos[0..c.eos_count], token) != null;
    }

    // Rank/ranks: 0/1 on one Mac. The fused zba row holds this rank's z, then every rank's b, then every rank's a.

    /// This rank's first value head: where its b, a, A_log and dt_bias rows start; 0 on one Mac or a replicated shard.
    pub fn valueHead0(c: Config) usize {
        return if (c.split_attention) c.rank * c.v_heads else 0;
    }

    /// The `b` block's width: every rank's, because the fused projection keeps it replicated.
    fn zbaHeads(c: Config) usize {
        return if (c.split_attention) c.ranks * c.v_heads else c.v_heads;
    }

    /// The rank's `b` row inside the fused `zba` row.
    pub fn zbaB(c: Config) usize {
        return c.gdnValueDim() + c.valueHead0();
    }

    /// The rank's `a` row inside the fused `zba` row, after the `b` block.
    pub fn zbaA(c: Config) usize {
        return c.gdnValueDim() + c.zbaHeads() + c.valueHead0();
    }

    /// The fused `zba` row's width. A multiple of 32 keeps `projection.prepare` on its tiled kernel.
    pub fn zbaDim(c: Config) usize {
        return c.gdnValueDim() + 2 * c.zbaHeads();
    }
};

/// A rank's slice of one tensor: elements `[from, from+len)` of the checkpoint's rows, or of an input row.
pub const Span = struct {
    from: usize = 0,
    len: usize = 0,

    pub fn end(s: Span) usize {
        return s.from + s.len;
    }
};

/// A rank's share of the 27B: the checkpoint holds `global`'s shapes, `local` is what this rank runs.
pub const Shard = struct {
    rank: u32,
    ranks: u32,
    global: Config,
    local: Config,

    /// This rank's first output row in a module whose `local` rows each rank takes a block of.
    pub fn row0(s: Shard, local: usize) usize {
        return @as(usize, s.rank) * local;
    }

    /// This rank's rows of a global-row module: local rows on a split dimension, the whole tensor on a replicated one.
    fn rows(s: Shard, global: usize, local: usize) Span {
        return if (local == global) .{ .from = 0, .len = global } else .{ .from = s.row0(local), .len = local };
    }

    /// Whether attention and linear attention split across ranks; an mlp shard leaves them whole, so one seam a layer.
    pub fn splitsAttention(s: Shard) bool {
        return s.local.split_attention;
    }

    /// self_attn.q_proj rows: q and its gate interleave every head_dim, so slices land on a 2*head_dim boundary.
    pub fn qRows(s: Shard) Span {
        return s.rows(s.global.qDim(), s.local.qDim());
    }

    /// `self_attn.k_proj` and `v_proj` rows (stacked k|v in that order): whole KV heads, one GQA group each.
    pub fn kvRows(s: Shard) Span {
        return s.rows(s.global.kvDim(), s.local.kvDim());
    }

    /// linear_attn.in_proj_qkv rows and conv1d channels: one q | k | v tensor, each block split on its own head count.
    pub fn qkvSpans(s: Shard) [3]Span {
        const gqk = s.global.k_heads * s.global.dk;
        const q = s.rows(gqk, s.local.k_heads * s.local.dk);
        const k = s.rows(gqk, s.local.k_heads * s.local.dk);
        const v = s.rows(s.global.gdnValueDim(), s.local.gdnValueDim());
        return .{ q, .{ .from = gqk + k.from, .len = k.len }, .{ .from = 2 * gqk + v.from, .len = v.len } };
    }

    /// linear_attn.conv1d channels in qkvSpans order: the same three blocks, so local conv state indexes like qkv rows.
    pub fn convSpans(s: Shard) [3]Span {
        return s.qkvSpans();
    }

    /// `linear_attn.in_proj_z` rows: this rank's `v_heads * dv` value channels, one contiguous block.
    pub fn zRows(s: Shard) Span {
        return s.rows(s.global.gdnValueDim(), s.local.gdnValueDim());
    }

    /// linear_attn.in_proj_a and in_proj_b: replicated whole into zba; a 24-row split would drop the tiled projection.
    pub fn abRows(s: Shard) Span {
        return .{ .from = 0, .len = s.global.v_heads };
    }

    /// `linear_attn.A_log` and `dt_bias`: one constant a value head, so a rank takes its own heads' slice.
    pub fn valueRows(s: Shard) Span {
        return s.rows(s.global.v_heads, s.local.v_heads);
    }

    /// `mlp.gate_proj` and `up_proj` rows (stacked gate|up): this rank's share of the intermediate width.
    pub fn intermediateRows(s: Shard) Span {
        return s.rows(s.global.intermediate, s.local.intermediate);
    }

    /// `lm_head` rows: this rank's share of the vocabulary, one contiguous block of tokens.
    pub fn vocabRows(s: Shard) Span {
        return s.rows(s.global.vocab, s.local.vocab);
    }

    /// `o_proj`/`out_proj`/`down_proj` rows: this rank's block over the whole K, or the width when replicated.
    pub fn outputRows(s: Shard) Span {
        return if (s.splitsAttention()) s.rows(s.global.hidden, s.global.hidden / s.ranks) else .{ .from = 0, .len = s.global.hidden };
    }
};

/// Mode: full halves attention, linear attention and MLP; mlp splits only the MLP and head, so one seam a layer.
pub const Mode = enum { full, mlp };

/// Split the even dimensions: heads, GQA groups, GDN heads, MLP width and vocabulary; embeddings and norms stay whole.
pub fn split(c: *Config, rank: u32, ranks: u32, mode: Mode) !Shard {
    if (ranks < 2 or ranks > 8 or rank >= ranks) return error.BadShard;
    if (c.intermediate % ranks != 0 or c.vocab % ranks != 0) return error.BadShard;
    if (mode == .full and (c.heads % ranks != 0 or c.kv_heads % ranks != 0 or c.k_heads % ranks != 0 or c.v_heads % ranks != 0 or c.hidden % ranks != 0)) return error.BadShard;
    const global = c.*;
    if (mode == .full) {
        c.heads /= ranks;
        c.kv_heads /= ranks;
        c.k_heads /= ranks;
        c.v_heads /= ranks;
    }
    c.intermediate /= ranks;
    c.vocab /= ranks;
    c.rank = rank;
    c.ranks = ranks;
    c.split_attention = mode == .full;
    // the local shapes must keep the group structure the attention, GDN and projection kernels run
    if (c.heads == 0 or c.kv_heads == 0 or c.k_heads == 0 or c.v_heads == 0) return error.BadShard;
    if (c.heads % c.kv_heads != 0 or c.v_heads % c.k_heads != 0) return error.BadShard;
    if (c.heads / c.kv_heads != global.heads / global.kv_heads) return error.BadShard;
    if (c.qDim() % 4 != 0 or c.kvDim() % 4 != 0 or c.gdnQkvDim() % 4 != 0 or c.gdnValueDim() % 4 != 0) return error.BadShard;
    if (c.vocab % 4 != 0 or c.intermediate % 4 != 0) return error.BadShard;
    if (c.layers != global.layers) return error.BadShard;
    try shapeProducts(c.*);
    return .{ .rank = rank, .ranks = ranks, .global = global, .local = c.* };
}

pub fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadConfig;
}

fn integer(v: std.json.Value, zero: bool) !usize {
    if (v != .integer or v.integer < 0 or (!zero and v.integer == 0)) return error.BadConfig;
    if (v.integer > std.math.maxInt(u32)) return error.BadConfig;
    return @intCast(v.integer);
}

fn required(o: std.json.ObjectMap, key: []const u8) !usize {
    return integer(o.get(key) orelse return error.BadConfig, false);
}

fn number(v: std.json.Value) !f64 {
    const n: f64 = switch (v) {
        .integer => |x| @floatFromInt(x),
        .float => |x| x,
        else => return error.BadConfig,
    };
    return if (std.math.isFinite(n)) n else error.BadConfig;
}

fn optional(o: std.json.ObjectMap, key: []const u8, fallback: f64) !f64 {
    const v = o.get(key) orelse return fallback;
    return if (v == .null) fallback else number(v);
}

fn flag(o: std.json.ObjectMap, key: []const u8) !bool {
    const v = o.get(key) orelse return false;
    return if (v == .bool) v.bool else error.BadConfig;
}

fn eos(c: *Config, v: std.json.Value) !void {
    if (v == .null) return;
    if (v == .array) {
        for (v.array.items) |id| try eos(c, id);
        return;
    }
    const id: u32 = @intCast(try integer(v, true));
    if (id >= c.vocab) return error.BadEos;
    if (c.isEos(id)) return;
    if (c.eos_count == max_eos) return error.TooManyEos;
    c.eos[c.eos_count] = id;
    c.eos_count += 1;
}

fn shapeProducts(c: Config) !void {
    const q = try std.math.mul(usize, c.heads, c.head_dim);
    _ = try std.math.mul(usize, q, 2);
    _ = try std.math.mul(usize, c.kv_heads, c.head_dim);
    const k = try std.math.mul(usize, c.k_heads, c.dk);
    const v = try std.math.mul(usize, c.v_heads, c.dv);
    const conv = try std.math.add(usize, try std.math.mul(usize, k, 2), v);
    _ = try std.math.mul(usize, conv, c.conv_kernel);
    _ = try std.math.mul(usize, try std.math.mul(usize, v, c.dk), 4);
    _ = try std.math.mul(usize, c.hidden, c.intermediate);
    _ = try std.math.mul(usize, c.hidden, c.vocab);
}

pub fn parse(gpa: std.mem.Allocator, json: []const u8, generation: ?[]const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    const t = if (root.get("text_config")) |v| try object(v) else root;
    const model = t.get("model_type") orelse root.get("model_type") orelse return error.BadConfig;
    if (model != .string or (!std.mem.eql(u8, model.string, "qwen3_5") and !std.mem.eql(u8, model.string, "qwen3_5_text"))) return error.NotQwenDense;
    if (t.get("num_experts")) |v| if (try integer(v, true) != 0) return error.NotQwenDense;
    if (try flag(root, "tie_word_embeddings") or try flag(t, "tie_word_embeddings")) return error.TiedEmbeddingHead;
    if (t.get("attn_output_gate")) |v| if (v != .bool or !v.bool) return error.UnsupportedAttentionGate;
    inline for (.{ .{ "hidden_act", "silu" }, .{ "output_gate_type", "swish" }, .{ "linear_attn_state_dtype", "float32" } }) |entry| {
        if (t.get(entry[0])) |v| if (v != .string or !std.mem.eql(u8, v.string, entry[1])) return error.UnsupportedActivation;
    }
    const heads = try required(t, "num_attention_heads");
    const hidden = try required(t, "hidden_size");
    const head_dim = if (t.get("head_dim")) |v| blk: {
        if (v == .null or (v == .integer and v.integer == 0)) break :blk hidden / heads;
        break :blk try integer(v, false);
    } else hidden / heads;
    const rope = if (t.get("rope_parameters")) |v| (if (v == .null) null else try object(v)) else null;
    const factor = if (rope) |r| try optional(r, "partial_rotary_factor", try optional(t, "partial_rotary_factor", 0.25)) else try optional(t, "partial_rotary_factor", 0.25);
    if (factor <= 0 or factor > 1) return error.BadConfig;
    var c = Config{
        .hidden = hidden,
        .intermediate = try required(t, "intermediate_size"),
        .layers = try required(t, "num_hidden_layers"),
        .vocab = try required(t, "vocab_size"),
        .heads = heads,
        .kv_heads = try required(t, "num_key_value_heads"),
        .head_dim = head_dim,
        .k_heads = try required(t, "linear_num_key_heads"),
        .v_heads = try required(t, "linear_num_value_heads"),
        .dk = try required(t, "linear_key_head_dim"),
        .dv = try required(t, "linear_value_head_dim"),
        .conv_kernel = try required(t, "linear_conv_kernel_dim"),
        .eps = @floatCast(try optional(t, "rms_norm_eps", 1e-6)),
        .rope_dims = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * factor),
        .rope_theta = if (rope) |r| try optional(r, "rope_theta", try optional(t, "rope_theta", 10000000)) else try optional(t, "rope_theta", 10000000),
    };
    if (t.get("max_position_embeddings")) |v| c.max_position = try integer(v, false);
    if (rope) |r| if (r.get("mrope_interleaved")) |v| {
        if (v != .bool or !v.bool) return error.UnsupportedMrope;
        c.mrope_interleaved = v.bool;
    };
    if (c.layers > max_layers or c.head_dim == 0 or c.heads % c.kv_heads != 0 or c.v_heads % c.k_heads != 0) return error.BadConfig;
    if (!std.math.isFinite(c.eps) or c.eps <= 0 or c.rope_theta <= 0 or c.rope_dims == 0 or c.rope_dims % 2 != 0) return error.BadConfig;
    if (rope) |r| if (r.get("mrope_section")) |v| {
        if (v != .array or v.array.items.len != 3) return error.BadConfig;
        for (v.array.items, 0..) |x, i| c.mrope_section[i] = try integer(x, true);
    };
    if (t.get("layer_types")) |v| {
        if (v != .array or v.array.items.len != c.layers) return error.BadConfig;
        for (v.array.items, 0..) |x, i| {
            if (x != .string) return error.BadConfig;
            c.kinds[i] = if (std.mem.eql(u8, x.string, "linear_attention")) .linear else if (std.mem.eql(u8, x.string, "full_attention")) .attention else return error.UnsupportedLayer;
        }
    } else {
        const interval = if (t.get("full_attention_interval")) |v| try integer(v, false) else 4;
        for (c.kinds[0..c.layers], 0..) |*k, i| k.* = if ((i + 1) % interval == 0) .attention else .linear;
    }
    if (root.get("eos_token_id") orelse t.get("eos_token_id")) |v| try eos(&c, v);
    if (generation) |text| {
        const gen = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer gen.deinit();
        if ((try object(gen.value)).get("eos_token_id")) |v| try eos(&c, v);
    }
    if (c.eos_count == 0) return error.MissingEos;
    try shapeProducts(c);
    return c;
}

/// The embedding table stays whole on every rank, so its gather bound is the table's rows, not a shard's width.
pub fn embedBound(c: *const Config, rows: u64) !u32 {
    if (rows < c.vocab) return error.BadEmbedding;
    return std.math.cast(u32, rows) orelse error.BadEmbedding;
}

test {
    _ = @import("config_test.zig");
}
