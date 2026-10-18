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
    engram_table_dir: List(u8, 4096) = .{},
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
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{ .parse_numbers = false });
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadConfig;
        const top = parsed.value.object;
        const text = if (top.get("text_config")) |t| (if (t == .object) t.object else return error.BadConfig) else top;
        // The exporter uses only the canonical checked pad field. Legacy aliases are refused at either level.
        if (top.contains("engram_pad_id") or text.contains("engram_pad_id")) return error.BadConfig;
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
        if (top.get("engram_table_dir")) |v| {
            if (v != .null) {
                if (v != .string) return error.BadConfig;
                for (v.string) |byte| try c.engram_table_dir.push(byte);
            }
        }
        try c.validate();
        try requireCheckpoint(top, text);
        try c.validateRelease();
        return c;
    }

    /// Shapes the kernels and the loader assume; a release that breaks one is refused before any allocation.
    pub fn validate(c: *const Config) !void {
        if (c.o_groups == 0 or c.heads == 0 or c.hidden == 0 or c.vocab == 0 or c.expert_width == 0 or
            c.layers == 0 or c.layers > 64 or c.mtp_layers > 64 - c.layers or c.engram_max_ngram < 2 or
            c.engram_max_ngram > 8 or c.engram_heads == 0 or c.engram_heads > 64 / (c.engram_max_ngram - 1) or
            c.engram_layers.len > 4 or c.engram_pad >= c.vocab or c.bos >= c.vocab or c.eos >= c.vocab or
            c.topk == 0 or c.experts == 0 or c.head_dim == 0 or c.window == 0) return error.BadConfig;
        inline for (.{ c.eps, c.hc_eps, c.engram_gate_clamp, c.routed_scale, c.swiglu_limit }) |v|
            if (!std.math.isFinite(v) or v <= 0) return error.BadConfig;
        inline for (.{ c.rope.theta, c.rope.compress_theta, c.rope.factor, c.rope.beta_fast, c.rope.beta_slow }) |v|
            if (!std.math.isFinite(v) or v <= 0) return error.BadConfig;
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

    /// Shared release capability; small logical fixtures are constructed explicitly with tiny().
    pub fn validateRelease(c: *const Config) !void {
        try c.validate();
        const release: Config = .{};
        inline for (.{ "vocab", "hidden", "layers", "heads", "head_dim", "rope_dim", "q_lora", "o_lora", "o_groups", "window", "experts", "shared_experts", "topk", "expert_width", "routed_scale", "norm_topk", "swiglu_limit", "index_heads", "index_dim", "index_topk", "candidate_source", "candidate_blocks", "candidate_block_size", "hc_mult", "hc_sinkhorn_iters", "hc_post_alpha", "engram_max_ngram", "engram_vocab", "engram_heads", "engram_head_dim", "engram_pad", "engram_compressed_vocab", "mtp_layers", "dspark_block", "dspark_noise_token", "dspark_markov_rank", "dspark_experts", "dspark_topk", "bos", "eos" }) |name| {
            if (@field(c.*, name) != @field(release, name)) return error.UnsupportedArchitecture;
        }
        inline for (.{ "compress_ratios", "kv_sources", "index_sources", "engram_layers", "dspark_targets" }) |name| {
            if (!std.mem.eql(u32, @field(c.*, name).items(), @field(release, name).items()))
                return error.UnsupportedCacheLayout;
        }
        if (!std.mem.eql(u64, c.engram_rows.items(), release.engram_rows.items())) return error.UnsupportedCacheLayout;
        if (!c.rope.yarn or c.rope.factor != 16 or c.rope.original_max_positions != 65536 or
            c.rope.max_positions != 1048576 or c.rope.theta <= 1 or c.rope.compress_theta <= 1)
            return error.UnsupportedArchitecture;
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

fn numberText(v: std.json.Value) ![]const u8 {
    return switch (v) {
        .number_string, .string => |s| s,
        else => error.BadConfig,
    };
}
fn intAs(comptime T: type, v: std.json.Value) !T {
    const n = unsigned(u64, v) catch return error.BadConfig;
    return std.math.cast(T, n) orelse error.BadConfig;
}
fn float(v: std.json.Value) !f64 {
    const n = std.fmt.parseFloat(f64, try numberText(v)) catch return error.BadConfig;
    if (!std.math.isFinite(n)) return error.BadConfig;
    return n;
}
pub fn unsigned(comptime T: type, value: std.json.Value) !T {
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

fn requireCheckpoint(top: std.json.ObjectMap, text: std.json.ObjectMap) !void {
    const model = top.get("model_type") orelse return error.MissingField;
    if (model != .string or !std.mem.eql(u8, model.string, "deepseek_v41")) return error.UnsupportedCheckpoint;
    const quant = top.get("quantization_config") orelse return error.MissingField;
    if (quant != .object) return error.BadConfig;
    const method = quant.object.get("quant_method") orelse return error.MissingField;
    const version = quant.object.get("version") orelse return error.MissingField;
    if (method != .string or !std.mem.eql(u8, method.string, "exl3") or version != .string or
        (!std.mem.eql(u8, version.string, "1") and !std.mem.startsWith(u8, version.string, "1."))) return error.UnsupportedCheckpoint;
    inline for (.{ "candidate_block_size", "candidate_source_layer_id", "candidate_topk_blocks", "compress_ratios", "compress_rope_theta", "engram_compressed_vocab_size", "engram_head_dim", "engram_layer_ids", "engram_max_ngram_size", "engram_n_heads", "engram_num_embeddings", "engram_pad_token_id", "engram_vocab_size", "hc_eps", "hc_mult", "hc_sinkhorn_iters", "head_dim", "hidden_size", "index_head_dim", "index_n_heads", "index_source_layer_ids", "index_topk", "kv_source_layer_ids", "max_position_embeddings", "moe_intermediate_size", "n_routed_experts", "n_shared_experts", "norm_topk_prob", "num_attention_heads", "num_experts_per_tok", "num_hidden_layers", "num_key_value_heads", "num_nextn_predict_layers", "o_groups", "o_lora_rank", "q_lora_rank", "qk_rope_head_dim", "rms_norm_eps", "rope_theta", "routed_scaling_factor", "scoring_func", "sliding_window", "swiglu_limit", "topk_method", "vocab_size" }) |key| {
        if (!text.contains(key)) return error.MissingField;
    }
    if (try intAs(u32, text.get("num_key_value_heads").?) != 1) return error.UnsupportedArchitecture;
    const r = text.get("rope_scaling") orelse return error.MissingField;
    if (r != .object) return error.BadConfig;
    inline for (.{ "rope_type", "factor", "beta_fast", "beta_slow", "original_max_position_embeddings" }) |key|
        if (!r.object.contains(key)) return error.MissingField;
    const kind = r.object.get("rope_type").?;
    if (kind != .string or !std.mem.eql(u8, kind.string, "yarn")) return error.UnsupportedArchitecture;
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

test "Engram pad alias cannot override the admitted token id" {
    const a = std.testing.allocator;
    const good = @embedFile("fixtures/config.json");
    const bad = try std.fmt.allocPrint(a, "{{\"engram_pad_id\":129280,{s}", .{good[1..]});
    defer a.free(bad);
    try std.testing.expectError(error.BadConfig, Config.parse(a, bad));
    const nested = try std.mem.replaceOwned(u8, a, good, "\"engram_pad_token_id\": 2", "\"engram_pad_token_id\": 2, \"engram_pad_id\": 129280");
    defer a.free(nested);
    try std.testing.expectError(error.BadConfig, Config.parse(a, nested));
    const conflict = try std.fmt.allocPrint(a, "{{\"engram_pad_id\":3,{s}", .{good[1..]});
    defer a.free(conflict);
    try std.testing.expectError(error.BadConfig, Config.parse(a, conflict));
}
