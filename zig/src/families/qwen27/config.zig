//! Qwen3.5-class text shapes (dense or routed), layer schedule, rotary metadata and ordered checkpoint EOS ids.
const std = @import("std");
const layer_pattern = @import("qwen35_layers");
const rules = @import("qwen35_rules");
const rope_config = @import("rope_config");
const yarn = @import("yarn.zig");
const fields = @import("config_fields");
const eos_ids = @import("checkpoint_eos");

pub const MhcRole = if (@hasDecl(@import("core"), "sensenova_loop"))
    @import("core").sensenova_loop.config.MhcRole
else
    enum { none, enter, inner, exit, both };

pub const Kind = enum { linear, attention };
pub const max_layers = 128;
pub const max_eos = 8;
pub const max_mhc_iterations: u32 = 1000;
pub const ParseOptions = struct { rope_yarn: ?f64 = null, loops: ?u32 = null };
pub const Problem = fields.Problem;
comptime {
    std.debug.assert(eos_ids.qwen27_metal.max.? == max_eos);
}

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
    looped: bool = false,
    physical_layers: usize = 0,
    exec_phys: [max_layers]u8 = initExecPhys(),
    mhc_roles: [max_layers]MhcRole = @splat(.none),
    mhc_iterations: u32 = 0,
    mhc_pre_scale: f32 = 1.0,
    shift_norms: bool = false,
    rope: yarn.Rope = .plain,
    /// Routed experts (qwen3_5_moe): their count and the picks a row makes; 0: a dense MLP of `intermediate`.
    experts: usize = 0,
    top_k: usize = 0,
    /// A routed expert's width; `intermediate` is then the shared expert's (the same, as the CUDA family requires).
    moe_intermediate: usize = 0,
    /// Whether a row's top-k softmax weights are renormalized to sum to one (`norm_topk_prob`).
    renorm: bool = false,

    fn initExecPhys() [max_layers]u8 {
        var out: [max_layers]u8 = undefined;
        for (&out, 0..) |*x, i| x.* = @intCast(i);
        return out;
    }

    pub fn physicalLayers(c: Config) usize {
        return if (c.physical_layers != 0) c.physical_layers else c.layers;
    }

    pub fn physKind(c: Config, phys: usize) Kind {
        for (0..c.layers) |exec| if (c.exec_phys[exec] == phys) return c.kinds[exec];
        return c.kinds[phys];
    }

    pub fn moe(c: Config) bool {
        return c.experts != 0;
    }

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
    // routed experts and looped mHC run on one Mac only: no rank split is qualified
    if (c.moe() or c.looped) return error.BadShard;
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

/// A count, 0 allowed when `zero`, at most u32; the field layer names the field, the refusal stays BadConfig.
fn integer(v: std.json.Value, zero: bool, at: fields.At, p: *Problem) !usize {
    return fields.int(usize, v, .{ .min = @intFromBool(!zero), .max = std.math.maxInt(u32) }, at, p) catch error.BadConfig;
}

fn required(r: fields.Reader, key: []const u8) !usize {
    return integer(r.value(key) catch return error.BadConfig, false, r.at(key), r.problem);
}

/// The number, or `fallback` when absent or null.
fn optional(r: fields.Reader, key: []const u8, fallback: f64) !f64 {
    return (r.optionalFloat(key, .absent) catch return error.BadConfig) orelse fallback;
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
    var p: Problem = .{};
    return parseWithOptionsWhy(gpa, json, generation, .{}, &p);
}

/// `parse`, with `p` naming the field behind a BadConfig the field layer refused.
pub fn parseWhy(gpa: std.mem.Allocator, json: []const u8, generation: ?[]const u8, p: *Problem) !Config {
    return parseWithOptionsWhy(gpa, json, generation, .{}, p);
}

/// Whether `json`'s root object declares `model_type == "sensenova_loop"`.
pub fn isSenseNovaRoot(gpa: std.mem.Allocator, json: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const mt = parsed.value.object.get("model_type") orelse return false;
    return mt == .string and std.mem.eql(u8, mt.string, "sensenova_loop");
}

/// Resolved context ceiling for a `sensenova_loop` checkpoint directory under optional `--rope-yarn`.
pub fn senseNovaMaxContext(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, rope_yarn: ?f64) !usize {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(16 << 20));
    defer gpa.free(text);
    const gen_path = try std.fs.path.join(gpa, &.{ dir, "generation_config.json" });
    defer gpa.free(gen_path);
    const gen_text = std.Io.Dir.cwd().readFileAlloc(io, gen_path, gpa, .limited(1 << 20)) catch null;
    defer if (gen_text) |gt| gpa.free(gt);
    return (try parseWithOptions(gpa, text, gen_text, .{ .rope_yarn = rope_yarn })).max_position;
}

pub fn parseWithOptions(gpa: std.mem.Allocator, json: []const u8, generation: ?[]const u8, opts: ParseOptions) !Config {
    var p: Problem = .{};
    return parseWithOptionsWhy(gpa, json, generation, opts, &p);
}

/// `parseWithOptions`, with `p` naming the field behind a BadConfig the field layer refused.
pub fn parseWithOptionsWhy(gpa: std.mem.Allocator, json: []const u8, generation: ?[]const u8, opts: ParseOptions, p: *Problem) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    const t = if (root.get("text_config")) |v| try object(v) else root;
    const tower: fields.Reader = .{ .object = t, .source = if (root.get("text_config") == null) "config.json" else "config.json text_config", .problem = p };
    const root_model = root.get("model_type");
    if (root_model) |rm| if (rm == .string and std.mem.eql(u8, rm.string, "sensenova_loop_text")) return error.NotQwenDense;
    const looped = root_model != null and root_model.? == .string and std.mem.eql(u8, root_model.?.string, "sensenova_loop");
    const model = if (looped) root_model.? else (t.get("model_type") orelse root_model orelse return error.BadConfig);
    const kinds_of = [_][]const u8{ "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text" };
    if (model != .string) return error.NotQwenDense;
    const routed = if (looped) false else for (kinds_of, 0..) |k, i| {
        if (std.mem.eql(u8, model.string, k)) break i >= 2;
    } else return error.NotQwenDense;
    const experts = if (t.get("num_experts")) |v| try integer(v, true, tower.at("num_experts"), p) else 0;
    if (routed != (experts != 0)) return error.NotQwenDense;
    if (try rules.tied(root, t, .metal27)) return error.TiedEmbeddingHead;
    try rules.check(t, .metal27);
    const heads = try required(tower, "num_attention_heads");
    const hidden = try required(tower, "hidden_size");
    const head_dim = if (t.get("head_dim")) |v| blk: {
        if (v == .null or (v == .integer and v.integer == 0)) break :blk hidden / heads;
        break :blk try integer(v, false, tower.at("head_dim"), p);
    } else hidden / heads;
    const rope = rope_config.params(t) catch return error.BadConfig;
    // theta alone: non-SenseNova checkpoints refuse YaRN/rope_scaling and --rope-yarn as before
    if (!looped and (opts.rope_yarn != null or rope_config.typed(rope) != null or rope_config.legacy(if (t.contains("rope_scaling")) t else root))) return error.UnsupportedRopeScaling;
    const rp: ?fields.Reader = if (rope) |r| .{ .object = r, .source = "config.json rope_parameters", .problem = p } else null;
    const factor = if (rp) |r| try optional(r, "partial_rotary_factor", try optional(tower, "partial_rotary_factor", 0.25)) else try optional(tower, "partial_rotary_factor", 0.25);
    if (factor <= 0 or factor > 1) return error.BadConfig;
    var c = Config{
        .hidden = hidden,
        .intermediate = if (routed) try required(tower, "shared_expert_intermediate_size") else try required(tower, "intermediate_size"),
        .layers = try required(tower, "num_hidden_layers"),
        .vocab = try required(tower, "vocab_size"),
        .heads = heads,
        .kv_heads = try required(tower, "num_key_value_heads"),
        .head_dim = head_dim,
        .k_heads = try required(tower, "linear_num_key_heads"),
        .v_heads = try required(tower, "linear_num_value_heads"),
        .dk = try required(tower, "linear_key_head_dim"),
        .dv = try required(tower, "linear_value_head_dim"),
        .conv_kernel = try required(tower, "linear_conv_kernel_dim"),
        .eps = @floatCast(try optional(tower, "rms_norm_eps", 1e-6)),
        .rope_dims = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * factor),
        .rope_theta = if (rp) |r| try optional(r, "rope_theta", try optional(tower, "rope_theta", 10000000)) else try optional(tower, "rope_theta", 10000000),
    };
    if (routed) {
        c.experts = experts;
        c.top_k = try required(tower, "num_experts_per_tok");
        c.moe_intermediate = try required(tower, "moe_intermediate_size");
        // absent means true, as Qwen3.5-MoE's reference config and the CUDA family read it
        c.renorm = (tower.optionalFlag("norm_topk_prob", .refused) catch return error.BadConfig) orelse true;
        if (c.experts > 256 or c.top_k >= c.experts or c.top_k > 15 or c.moe_intermediate != c.intermediate) return error.UnsupportedRouting;
    }
    if (t.get("max_position_embeddings")) |v| {
        c.max_position = try integer(v, false, tower.at("max_position_embeddings"), p);
    } else if (looped) if (root.get("max_position_embeddings")) |v| {
        c.max_position = try integer(v, false, .{ .source = "config.json", .key = "max_position_embeddings" }, p);
    };
    if (looped) {
        const res = try yarn.resolve(t, root, rope, c.rope_theta, c.max_position, opts.rope_yarn);
        c.rope = res.rope;
        c.max_position = res.ceiling;
    }
    if (rope) |r| if (r.get("mrope_interleaved")) |v| {
        if (v != .bool or !v.bool) return error.UnsupportedMrope;
        c.mrope_interleaved = v.bool;
    };
    if (c.layers > max_layers or c.head_dim == 0 or c.heads % c.kv_heads != 0 or c.v_heads % c.k_heads != 0) return error.BadConfig;
    if (!std.math.isFinite(c.eps) or c.eps <= 0 or c.rope_theta <= 0 or c.rope_dims == 0 or c.rope_dims % 2 != 0) return error.BadConfig;
    if (rope) |r| if (r.get("mrope_section")) |v| {
        if (v != .array or v.array.items.len != 3) return error.BadConfig;
        for (v.array.items, 0..) |x, i| c.mrope_section[i] = try integer(x, true, .{ .source = "config.json rope_parameters", .key = "mrope_section" }, p);
    };
    if (looped) {
        if (comptime @hasDecl(@import("core"), "sensenova_loop")) {
            const sn_cfg = @import("core").sensenova_loop.config;
            const sn = try sn_cfg.Config.parseWithGeneration(gpa, json, generation);
            var step_buf: [max_layers]sn_cfg.ExecutionStep = undefined;
            const steps = try sn.fillPlan(&step_buf, opts.loops);
            c.looped = true;
            c.shift_norms = true;
            c.physical_layers = sn.num_hidden_layers;
            c.layers = steps.len;
            if (sn.mhc) |mhc| {
                const scale32: f32 = @floatCast(mhc.pre_scale);
                if (mhc.streams != 2 or mhc.fixed_identity or mhc.iterations > max_mhc_iterations or !std.math.isFinite(scale32) or scale32 <= 0) {
                    return error.UnsupportedSenseNovaMhc;
                }
                c.mhc_iterations = mhc.iterations;
                c.mhc_pre_scale = scale32;
            }
            for (steps, 0..) |st, exec| {
                c.kinds[exec] = if (st.is_linear) .linear else .attention;
                c.exec_phys[exec] = st.phys;
                c.mhc_roles[exec] = st.mhc_role;
            }
        } else return error.NotQwenDense;
    } else {
        var full: [max_layers]bool = undefined;
        try layer_pattern.read(full[0..c.layers], t);
        for (c.kinds[0..c.layers], full[0..c.layers]) |*k, f| k.* = if (f) .attention else .linear;
    }
    // generation_config.json is parsed before config.json's ids: only when both are bad does the refusal differ
    const gen = if (generation) |text| try std.json.parseFromSlice(std.json.Value, gpa, text, .{}) else null;
    defer if (gen) |g| g.deinit();
    const gen_doc: std.json.Value = if (gen) |g| .{ .object = try object(g.value) } else .null;
    const ids = eos_ids.collect(&c.eos, eos_ids.qwen27_metal, parsed.value, .{ .object = t }, gen_doc, c.vocab) catch |e| return switch (e) {
        error.BadEosValue => error.BadConfig,
        else => e,
    };
    c.eos_count = ids.len;
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
