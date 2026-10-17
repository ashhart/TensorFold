//! DeepSeek-V4.1-Flash text config (config.json -> text_config) and the per-layer CSA2 topology.

const std = @import("std");

/// Fixed-capacity id lists of the config (layers, sources, Engram tables); the release's longest has 43 entries.
pub fn List(comptime T: type, comptime cap: usize) type {
    return struct {
        const Self = @This();
        buf: [cap]T = undefined,
        len: usize = 0,

        pub fn of(values: []const T) Self {
            var s: Self = .{};
            for (values) |v| s.push(v) catch unreachable;
            return s;
        }

        pub fn push(s: *Self, v: T) !void {
            if (s.len == cap) return error.ConfigListTooLong;
            s.buf[s.len] = v;
            s.len += 1;
        }

        pub fn items(s: *const Self) []const T {
            return s.buf[0..s.len];
        }

        pub fn contains(s: *const Self, v: T) bool {
            return std.mem.indexOfScalar(T, s.items(), v) != null;
        }
    };
}

pub const Ids = List(u32, 64);

pub const Rope = struct {
    theta: f64 = 10000.0,
    compress_theta: f64 = 160000.0,
    factor: f64 = 16.0,
    original_max_positions: u64 = 65536,
    beta_fast: f64 = 32.0,
    beta_slow: f64 = 1.0,
    max_positions: u64 = 1048576,
    yarn: bool = true,
};

/// The attention a layer runs: sliding window only, KV source + indexer, own indexer on a source's keys, or reuse.
pub const Mode = enum { swa, full, reindex, reuse };

pub const Config = struct {
    vocab: u32 = 129280,
    hidden: u32 = 5120,
    layers: u32 = 40,
    heads: u32 = 64,
    head_dim: u32 = 512,
    rope_dim: u32 = 64,
    q_lora: u32 = 1280,
    o_lora: u32 = 1024,
    o_groups: u32 = 8,
    eps: f32 = 1e-20,
    window: u32 = 128,
    experts: u32 = 384,
    shared_experts: u32 = 1,
    topk: u32 = 6,
    expert_width: u32 = 2304,
    routed_scale: f32 = 1.5,
    norm_topk: bool = true,
    swiglu_limit: f32 = 10.0,
    compress_ratios: Ids = Ids.of(&.{ 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 }),
    kv_sources: Ids = Ids.of(&.{ 2, 8, 14, 20 }),
    index_sources: Ids = Ids.of(&.{ 2, 8, 14, 20, 24, 28, 32, 36 }),
    index_heads: u32 = 32,
    index_dim: u32 = 128,
    index_topk: u32 = 512,
    candidate_source: i64 = 20,
    candidate_blocks: u32 = 2048,
    candidate_block_size: u32 = 8,
    hc_mult: u32 = 4,
    hc_sinkhorn_iters: u32 = 20,
    hc_eps: f32 = 1e-6,
    hc_post_alpha: f32 = 2.0,
    engram_layers: Ids = Ids.of(&.{ 1, 14 }),
    engram_rows: List(u64, 8) = List(u64, 8).of(&.{ 384006168, 384016682 }),
    engram_max_ngram: u32 = 4,
    engram_vocab: u64 = 16000000,
    engram_heads: u32 = 8,
    engram_head_dim: u32 = 256,
    engram_pad: u32 = 2,
    engram_compressed_vocab: u32 = 99092,
    engram_gate_clamp: f32 = 1e-6,
    mtp_layers: u32 = 3,
    dspark_block: u32 = 5,
    dspark_noise_token: u32 = 128799,
    dspark_targets: Ids = Ids.of(&.{ 37, 38, 39 }),
    dspark_markov_rank: u32 = 256,
    dspark_experts: u32 = 128,
    dspark_topk: u32 = 3,
    rope: Rope = .{},
    bos: u32 = 0,
    eos: u32 = 1,

    /// config.json in `dir`.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    /// Checkpoint JSON (text_config, else top level); absent keys keep release values, unknown routing is refused.
    pub fn parse(gpa: std.mem.Allocator, json: []const u8) !Config {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadConfig;
        const top = parsed.value.object;
        const text = if (top.get("text_config")) |t| (if (t == .object) t.object else return error.BadConfig) else top;
        var c: Config = .{};
        const ints = .{
            .{ "vocab", "vocab_size" },                         .{ "hidden", "hidden_size" },
            .{ "layers", "num_hidden_layers" },                 .{ "heads", "num_attention_heads" },
            .{ "head_dim", "head_dim" },                        .{ "rope_dim", "qk_rope_head_dim" },
            .{ "q_lora", "q_lora_rank" },                       .{ "o_lora", "o_lora_rank" },
            .{ "o_groups", "o_groups" },                        .{ "window", "sliding_window" },
            .{ "experts", "n_routed_experts" },                 .{ "shared_experts", "n_shared_experts" },
            .{ "topk", "num_experts_per_tok" },                 .{ "expert_width", "moe_intermediate_size" },
            .{ "index_heads", "index_n_heads" },                .{ "index_dim", "index_head_dim" },
            .{ "index_topk", "index_topk" },                    .{ "candidate_source", "candidate_source_layer_id" },
            .{ "candidate_blocks", "candidate_topk_blocks" },   .{ "candidate_block_size", "candidate_block_size" },
            .{ "hc_mult", "hc_mult" },                          .{ "hc_sinkhorn_iters", "hc_sinkhorn_iters" },
            .{ "engram_max_ngram", "engram_max_ngram_size" },   .{ "engram_vocab", "engram_vocab_size" },
            .{ "engram_heads", "engram_n_heads" },              .{ "engram_head_dim", "engram_head_dim" },
            .{ "engram_pad", "engram_pad_token_id" },           .{ "engram_compressed_vocab", "engram_compressed_vocab_size" },
            .{ "mtp_layers", "num_nextn_predict_layers" },      .{ "dspark_block", "dspark_block_size" },
            .{ "dspark_noise_token", "dspark_noise_token_id" }, .{ "dspark_markov_rank", "dspark_markov_rank" },
            .{ "dspark_experts", "dspark_n_routed_experts" },   .{ "dspark_topk", "dspark_num_experts_per_tok" },
        };
        inline for (ints) |p| if (text.get(p[1])) |v| {
            @field(c, p[0]) = try intAs(@TypeOf(@field(c, p[0])), v);
        };
        const floats = .{
            .{ "eps", "rms_norm_eps" }, .{ "routed_scale", "routed_scaling_factor" }, .{ "swiglu_limit", "swiglu_limit" },
            .{ "hc_eps", "hc_eps" },    .{ "hc_post_alpha", "hc_post_alpha" },        .{ "engram_gate_clamp", "engram_gate_clamp" },
        };
        inline for (floats) |p| if (text.get(p[1])) |v| {
            @field(c, p[0]) = @floatCast(try float(v));
        };
        if (text.get("norm_topk_prob")) |v| c.norm_topk = if (v == .bool) v.bool else return error.BadConfig;
        const lists = .{
            .{ "compress_ratios", "compress_ratios" },        .{ "kv_sources", "kv_source_layer_ids" },
            .{ "index_sources", "index_source_layer_ids" },   .{ "engram_layers", "engram_layer_ids" },
            .{ "dspark_targets", "dspark_target_layer_ids" }, .{ "engram_rows", "engram_num_embeddings" },
        };
        inline for (lists) |p| if (text.get(p[1])) |v| {
            @field(c, p[0]) = try list(@TypeOf(@field(c, p[0])), v);
        };
        // routing the kernels implement: sqrt(softplus) scores, bias-corrected top-k (vLLM noaux_tc)
        if (text.get("scoring_func")) |v| if (v != .string or !std.mem.eql(u8, v.string, "sqrtsoftplus")) return error.UnsupportedScoring;
        if (text.get("topk_method")) |v| if (v != .string or !std.mem.eql(u8, v.string, "noaux_tc")) return error.UnsupportedScoring;
        c.rope = try rope(text);
        if (top.get("bos_token_id")) |v| c.bos = try intAs(u32, v);
        if (top.get("eos_token_id")) |v| c.eos = try intAs(u32, v);
        try c.validate();
        return c;
    }

    /// Shapes the kernels and the loader assume; a release that breaks one is refused before any allocation.
    pub fn validate(c: *const Config) !void {
        if (c.hidden % 128 != 0 or c.expert_width % 128 != 0 or c.vocab % 128 != 0) return error.BadConfig;
        if (c.heads % c.o_groups != 0 or c.rope_dim > c.head_dim or c.topk > c.experts) return error.BadConfig;
        if (c.dspark_topk > c.dspark_experts and c.dspark_experts != 0) return error.BadConfig;
        if (c.compress_ratios.len < c.layers) return error.BadConfig;
        for (c.compress_ratios.items()) |r| if (r > 2) return error.BadConfig;
        for (c.kv_sources.items()) |s| if (s >= c.layers or c.compressRatio(s) == 0) return error.BadConfig;
        for (c.index_sources.items()) |s| if (s >= c.layers or c.compressRatio(s) == 0) return error.BadConfig;
        if (c.engram_layers.len != c.engram_rows.len) return error.BadConfig;
        // every compressed layer needs a KV source and an index source at or before it
        var l: u32 = 0;
        while (l < c.layers) : (l += 1) if (c.compressRatio(l) > 0 and (c.kvSource(l) == null or c.indexSource(l) == null)) return error.BadConfig;
    }

    /// Layers the forward runs: the backbone, then the DSpark blocks (checkpoint mtp.*).
    pub fn allLayers(c: *const Config) u32 {
        return c.layers + c.mtp_layers;
    }

    pub fn compressRatio(c: *const Config, layer: u32) u32 {
        return if (layer < c.compress_ratios.len) c.compress_ratios.buf[layer] else 0;
    }

    pub fn isBackbone(c: *const Config, layer: u32) bool {
        return layer < c.layers;
    }

    /// The latest source at or before `layer` (null for sliding-window layers).
    fn latest(c: *const Config, sources: *const Ids, layer: u32) ?u32 {
        if (c.compressRatio(layer) == 0) return null;
        var best: ?u32 = null;
        for (sources.items()) |s| if (s <= layer and (best == null or s > best.?)) {
            best = s;
        };
        return best;
    }

    pub fn kvSource(c: *const Config, layer: u32) ?u32 {
        return c.latest(&c.kv_sources, layer);
    }

    pub fn indexSource(c: *const Config, layer: u32) ?u32 {
        return c.latest(&c.index_sources, layer);
    }

    pub fn isKvSource(c: *const Config, layer: u32) bool {
        return c.isBackbone(layer) and c.kv_sources.contains(layer);
    }

    pub fn isIndexSource(c: *const Config, layer: u32) bool {
        return c.isBackbone(layer) and c.index_sources.contains(layer);
    }

    pub fn isCandidateSource(c: *const Config, layer: u32) bool {
        return c.candidate_source == layer and c.candidate_blocks > 0;
    }

    pub fn usesCandidates(c: *const Config, layer: u32) bool {
        return c.isIndexSource(layer) and c.candidate_blocks > 0 and c.candidate_source >= 0 and c.candidate_source < layer;
    }

    pub fn mode(c: *const Config, layer: u32) Mode {
        if (c.compressRatio(layer) == 0) return .swa;
        if (c.isKvSource(layer)) return .full;
        if (c.isIndexSource(layer)) return .reindex;
        return .reuse;
    }

    pub fn isEngram(c: *const Config, layer: u32) bool {
        return c.engram_layers.contains(layer);
    }

    /// (routed experts, experts a token) of a backbone layer or a DSpark block.
    pub fn expertsOf(c: *const Config, layer: u32) struct { count: u32, topk: u32 } {
        if (c.isBackbone(layer)) return .{ .count = c.experts, .topk = c.topk };
        return .{ .count = if (c.dspark_experts != 0) c.dspark_experts else c.experts, .topk = if (c.dspark_topk != 0) c.dspark_topk else c.topk };
    }

    /// Engram's hash columns: (max n-gram - 1) x heads.
    pub fn hashCols(c: *const Config) u32 {
        return (c.engram_max_ngram - 1) * c.engram_heads;
    }
};

fn intAs(comptime T: type, v: std.json.Value) !T {
    return switch (v) {
        .integer => |i| std.math.cast(T, i) orelse error.BadConfig,
        .float => |f| if (@floor(f) == f) std.math.cast(T, @as(i64, @intFromFloat(f))) orelse error.BadConfig else error.BadConfig,
        else => error.BadConfig,
    };
}

fn float(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadConfig,
    };
}

fn list(comptime L: type, v: std.json.Value) !L {
    if (v != .array) return error.BadConfig;
    var out: L = .{};
    const T = @TypeOf(out.buf[0]);
    for (v.array.items) |x| try out.push(try intAs(T, x));
    return out;
}

/// rope_theta, compress_rope_theta and rope_scaling (or rope_parameters), as RopeConfig.from_dict.
fn rope(text: std.json.ObjectMap) !Rope {
    var r: Rope = .{};
    if (text.get("rope_theta")) |v| r.theta = try float(v);
    if (text.get("compress_rope_theta")) |v| r.compress_theta = try float(v);
    if (text.get("max_position_embeddings")) |v| r.max_positions = try intAs(u64, v);
    const scaling = text.get("rope_scaling") orelse text.get("rope_parameters");
    r.factor = 1.0;
    r.original_max_positions = r.max_positions;
    r.yarn = false;
    if (scaling) |s| if (s == .object) {
        const o = s.object;
        if (o.get("factor")) |v| r.factor = try float(v);
        if (o.get("original_max_position_embeddings")) |v| r.original_max_positions = try intAs(u64, v);
        if (o.get("beta_fast")) |v| r.beta_fast = try float(v);
        if (o.get("beta_slow")) |v| r.beta_slow = try float(v);
        const kind = o.get("rope_type") orelse o.get("type");
        r.yarn = if (kind) |k| (k == .string and !std.mem.eql(u8, k.string, "default")) else false;
    };
    return r;
}

/// A small config with every V4.1 mechanism; split widths are 256 so TP=2 keeps 128-value blocks.
pub fn tiny() Config {
    return .{
        .vocab = 256,
        .hidden = 256,
        .layers = 8,
        .heads = 4,
        .head_dim = 64,
        .rope_dim = 8,
        .q_lora = 32,
        .o_lora = 128,
        .o_groups = 2,
        .eps = 1e-6,
        .window = 4,
        .experts = 8,
        .topk = 2,
        .expert_width = 256,
        .compress_ratios = Ids.of(&.{ 0, 0, 2, 2, 1, 1, 1, 1, 0, 0 }),
        .kv_sources = Ids.of(&.{ 2, 4 }),
        .index_sources = Ids.of(&.{ 2, 4, 6 }),
        .index_heads = 2,
        .index_dim = 16,
        .index_topk = 3,
        .candidate_source = 4,
        .candidate_blocks = 2,
        .candidate_block_size = 2,
        .engram_layers = Ids.of(&.{1}),
        .engram_rows = List(u64, 8).of(&.{0}),
        .engram_max_ngram = 3,
        .engram_vocab = 31,
        .engram_heads = 2,
        .engram_head_dim = 8,
        .engram_compressed_vocab = 50,
        .mtp_layers = 2,
        .dspark_block = 3,
        .dspark_noise_token = 96,
        .dspark_targets = Ids.of(&.{ 5, 6, 7 }),
        .dspark_markov_rank = 8,
        .dspark_experts = 4,
        .dspark_topk = 2,
        .rope = .{ .factor = 4.0, .original_max_positions = 64, .max_positions = 256 },
    };
}
