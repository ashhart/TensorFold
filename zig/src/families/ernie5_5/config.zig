//! ERNIE 5.5's shape from config.json (ernie5_5_moe), refusing every layout the engine does not run.
const std = @import("std");
const fields = @import("config_fields");

pub const Problem = fields.Problem;
pub const Error = error{ BadConfig, NotErnie55, Unsupported, Layout, Quant };

/// The most layers the plan tables hold (the release has 43).
pub const max_layers = 128;
/// The most target layers the draft head may tap (the release taps 3).
pub const max_taps = 8;

/// The compressed-attention (HCA) layers: one key/value head shared by every query head, output in groups.
pub const Hca = struct { heads: u32, head_dim: u32, nope: u32, rope: u32, q_lora: u32, o_groups: u32, o_lora: u32, window: u32 };
/// The sparse-attention (DSA) layers' MLA heads.
pub const Mla = struct { heads: u32, q_lora: u32, kv_lora: u32, nope: u32, rope: u32, value: u32 };
pub const Indexer = struct { heads: u32, dim: u32, topk: u32 };
/// Routed experts run in a latent space of `latent` wide; the shared expert and the dense layers in the hidden width.
pub const Moe = struct { experts: u32, topk: u32, inter: u32, latent: u32, dense_layers: u32, dense_inter: u32, scale: f64 };
pub const HyperConnections = struct { streams: u32, sinkhorn: u32, eps: f64 };
pub const Rope = struct { theta: f64, compress_theta: f64, factor: f64, original: u32, beta_fast: f64, beta_slow: f64 };
/// The draft head bundled as `model.mtp.*`: its layers tap these target layers.
pub const Dspark = struct { block: u32, rank: u32, noise_token: u32, taps: [max_taps]u32, taps_n: u32 };

pub const Config = struct {
    hidden: u32,
    layers: u32,
    vocab: u32,
    max_context: u32,
    eps: f64,
    /// Per layer: a positive value is an HCA layer's compress ratio, zero or less marks a DSA layer.
    compress: [max_layers]i32,
    hca: Hca,
    mla: Mla,
    index: Indexer,
    moe: Moe,
    hc: HyperConnections,
    rope: Rope,
    situ_beta: f64,
    situ_linear_beta: f64,
    dspark: Dspark,

    pub fn isDsa(c: *const Config, i: u32) bool {
        return c.compress[i] <= 0;
    }

    pub fn isDense(c: *const Config, i: u32) bool {
        return i < c.moe.dense_layers;
    }
};

/// Why a key is held to the release's value: the weight map follows it, the vendor runs one value, or it is unread.
const Basis = enum { weights, vendor, unread };

const Pin = struct {
    key: []const u8,
    want: union(enum) { flag: bool, text: []const u8, int: i64 },
    basis: Basis,
};

const pins = [_]Pin{
    .{ .key = "enable_hyper_connections", .want = .{ .flag = true }, .basis = .weights },
    .{ .key = "use_vha_attention", .want = .{ .flag = true }, .basis = .weights },
    .{ .key = "use_gated_attn", .want = .{ .flag = true }, .basis = .weights },
    .{ .key = "gated_attn_use_q_lora", .want = .{ .flag = true }, .basis = .weights },
    .{ .key = "add_full_attention_sink_bias", .want = .{ .flag = true }, .basis = .weights },
    .{ .key = "latent_moe_use_norm", .want = .{ .flag = true }, .basis = .weights },
    .{ .key = "gated_linear_unit", .want = .{ .flag = true }, .basis = .weights },
    .{ .key = "moe_shared_expert_gate", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "n_shared_experts", .want = .{ .int = 1 }, .basis = .weights },
    .{ .key = "tie_word_embeddings", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "attention_bias", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "use_bias", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "use_mla", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "multimax", .want = .{ .text = "lm_head" }, .basis = .weights },
    .{ .key = "num_nextn_predict_layers", .want = .{ .int = 0 }, .basis = .weights },
    .{ .key = "use_dense_mtp", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "moe_split_feature_routing", .want = .{ .flag = true }, .basis = .vendor },
    .{ .key = "routed_scaling_factor_learnable", .want = .{ .flag = true }, .basis = .vendor },
    .{ .key = "n_group", .want = .{ .int = 1 }, .basis = .vendor },
    .{ .key = "topk_group", .want = .{ .int = 1 }, .basis = .vendor },
    .{ .key = "scoring_func", .want = .{ .text = "sigmoid" }, .basis = .vendor },
    .{ .key = "gated_attn_act", .want = .{ .text = "sigmoid" }, .basis = .vendor },
    .{ .key = "use_fast_hadamard", .want = .{ .flag = true }, .basis = .vendor },
    .{ .key = "hidden_act", .want = .{ .text = "situ" }, .basis = .vendor },
    .{ .key = "norm_topk_prob", .want = .{ .flag = true }, .basis = .vendor },
    .{ .key = "experimental_attention_variant", .want = .{ .text = "dsv4_hybrid" }, .basis = .unread },
    .{ .key = "hybrid_mla_attention", .want = .{ .text = "mqa_dsa" }, .basis = .unread },
    .{ .key = "csa_dense_mode", .want = .{ .flag = true }, .basis = .unread },
    .{ .key = "use_attn_merger", .want = .{ .flag = true }, .basis = .unread },
    .{ .key = "mqa_split_kv_b_proj", .want = .{ .flag = true }, .basis = .unread },
    .{ .key = "use_qk_norm", .want = .{ .flag = true }, .basis = .unread },
    .{ .key = "qk_norm_type", .want = .{ .text = "per_head" }, .basis = .unread },
    .{ .key = "rope_interleave", .want = .{ .flag = true }, .basis = .unread },
    .{ .key = "hca_rope_type", .want = .{ .text = "yarn" }, .basis = .unread },
    .{ .key = "topk_method", .want = .{ .text = "noaux_tc" }, .basis = .unread },
    .{ .key = "moe_group_experts", .want = .{ .flag = false }, .basis = .unread },
    .{ .key = "moe_layer_freq", .want = .{ .int = 1 }, .basis = .unread },
    .{ .key = "mtp_shared_last_layer", .want = .{ .flag = true }, .basis = .unread },
};

fn say(p: *Problem, comptime fmt: []const u8, args: anytype) void {
    var w: std.Io.Writer = .fixed(&p.bytes);
    w.print(fmt, args) catch {}; // full: the text is cut, not refused
    p.len = w.end;
}

fn refuse(p: *Problem, e: Error, comptime fmt: []const u8, args: anytype) Error {
    say(p, fmt, args);
    return e;
}

fn pinned(r: fields.Reader, pin: Pin) Error!void {
    const v = r.value(pin.key) catch return error.BadConfig;
    const same = switch (pin.want) {
        .flag => |b| v == .bool and v.bool == b,
        .text => |s| v == .string and std.mem.eql(u8, v.string, s),
        .int => |n| v == .integer and v.integer == n,
    };
    if (same) return;
    const why = switch (pin.basis) {
        .weights => "the weight map follows it",
        .vendor => "the vendor's plugin runs no other value",
        .unread => "its meaning is not read yet, so the engine holds the release's value",
    };
    return switch (pin.want) {
        .flag => |b| refuse(r.problem, error.Unsupported, "config.json: {s} must be {}: {s}", .{ pin.key, b, why }),
        .text => |s| refuse(r.problem, error.Unsupported, "config.json: {s} must be \"{s}\": {s}", .{ pin.key, s, why }),
        .int => |n| refuse(r.problem, error.Unsupported, "config.json: {s} must be {d}: {s}", .{ pin.key, n, why }),
    };
}

fn int(r: fields.Reader, key: []const u8) Error!u32 {
    return r.requiredInt(u32, key, .{ .min = 1, .max = std.math.maxInt(u32) }) catch error.BadConfig;
}

fn positive(r: fields.Reader, key: []const u8) Error!f64 {
    const x = r.requiredFloat(key) catch return error.BadConfig;
    if (x <= 0) return refuse(r.problem, error.BadConfig, "{s}: {s} is {d}, not positive", .{ r.source, key, x });
    return x;
}

fn object(r: fields.Reader, key: []const u8) Error!fields.Object {
    const v = r.value(key) catch return error.BadConfig;
    if (v != .object) return refuse(r.problem, error.BadConfig, "{s}: {s} is not an object", .{ r.source, key });
    return v.object;
}

/// The key's string, or null when it is absent or holds another kind.
fn textOf(o: fields.Object, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn text(r: fields.Reader, key: []const u8) Error![]const u8 {
    const v = r.value(key) catch return error.BadConfig;
    if (v != .string) return refuse(r.problem, error.BadConfig, "{s}: {s} is not a string", .{ r.source, key });
    return v.string;
}

/// Layer indices in a JSON array, each below `bound`, into `out`; the count is returned.
fn indices(r: fields.Reader, key: []const u8, bound: u32, out: []u32) Error!u32 {
    const v = r.value(key) catch return error.BadConfig;
    if (v != .array or v.array.items.len == 0 or v.array.items.len > out.len)
        return refuse(r.problem, error.Layout, "{s}: {s} must list 1 to {d} layers", .{ r.source, key, out.len });
    for (v.array.items, 0..) |e, i| {
        if (e != .integer or e.integer < 0 or e.integer >= bound)
            return refuse(r.problem, error.Layout, "{s}: {s} names a layer past the {d} layers", .{ r.source, key, bound });
        out[i] = @intCast(e.integer);
    }
    return @intCast(v.array.items.len);
}

fn compressRatios(r: fields.Reader, layers: u32, out: *[max_layers]i32) Error!void {
    const v = r.value("compress_ratios") catch return error.BadConfig;
    if (v != .array or v.array.items.len != layers)
        return refuse(r.problem, error.Layout, "config.json: compress_ratios must hold one entry for each of the {d} layers", .{layers});
    for (v.array.items, 0..) |e, i| {
        if (e != .integer or e.integer == 0 or (e.integer < 0 and e.integer != -2) or e.integer > std.math.maxInt(i32))
            return refuse(r.problem, error.Layout, "config.json: compress_ratios[{d}] must be a positive ratio or -2 (a DSA layer)", .{i});
        out[i] = @intCast(e.integer);
    }
}

/// The checkpoint stores FP8 projections in 128x128 blocks and MXFP4 routed experts; nothing else is read.
fn quantization(r: fields.Reader) Error!void {
    const q: fields.Reader = .{ .object = try object(r, "quantization_config"), .source = "config.json quantization_config", .problem = r.problem };
    const fp8 = std.mem.eql(u8, try text(q, "quant_method"), "fp8") and std.mem.eql(u8, try text(q, "fmt"), "e4m3");
    const experts = std.mem.eql(u8, textOf(q.object, "expert_dtype") orelse "", "mxfp4");
    const blocks = q.object.get("weight_block_size") orelse .null;
    const square = blocks == .array and blocks.array.items.len == 2 and for (blocks.array.items) |b| {
        if (b != .integer or b.integer != 128) break false;
    } else true;
    if (!fp8 or !experts or !square)
        return refuse(r.problem, error.Quant, "quantization_config: the loader reads FP8 E4M3 in 128x128 blocks with MXFP4 routed experts (the FP4 release)", .{});
}

fn hca(r: fields.Reader) Error!Hca {
    const h: Hca = .{
        .heads = try int(r, "num_attention_heads"),
        .head_dim = try int(r, "head_dim"),
        .nope = try int(r, "qk_nope_head_dim"),
        .rope = try int(r, "qk_rope_head_dim"),
        .q_lora = try int(r, "q_lora_rank"),
        .o_groups = try int(r, "o_groups"),
        .o_lora = try int(r, "o_lora_rank"),
        .window = try int(r, "sliding_window"),
    };
    if (try int(r, "num_key_value_heads") != 1 or try int(r, "v_head_dim") != h.head_dim or h.nope + h.rope != h.head_dim or h.heads % h.o_groups != 0)
        return refuse(r.problem, error.Layout, "config.json: HCA layers need one key/value head, values of head_dim, nope + rope = head_dim and heads that split over o_groups", .{});
    return h;
}

fn mla(r: fields.Reader) Error!Mla {
    const m: Mla = .{
        .heads = try int(r, "hybrid_mla_num_attention_heads"),
        .q_lora = try int(r, "hybrid_mla_q_lora_rank"),
        .kv_lora = try int(r, "hybrid_mla_kv_lora_rank"),
        .nope = try int(r, "hybrid_mla_qk_nope_head_dim"),
        .rope = try int(r, "hybrid_mla_qk_rope_head_dim"),
        .value = try int(r, "hybrid_mla_v_head_dim"),
    };
    if (try int(r, "hybrid_mla_num_key_value_heads") != m.heads)
        return refuse(r.problem, error.Layout, "config.json: DSA layers need as many key/value heads as query heads (MLA)", .{});
    return m;
}

fn moe(r: fields.Reader, layers: u32) Error!Moe {
    const m: Moe = .{
        .experts = try int(r, "n_routed_experts"),
        .topk = try int(r, "num_experts_per_tok"),
        .inter = try int(r, "moe_intermediate_size"),
        .latent = try int(r, "moe_latent_size"),
        .dense_layers = r.requiredInt(u32, "first_k_dense_replace", .{ .min = 0, .max = layers }) catch return error.BadConfig,
        .dense_inter = try int(r, "intermediate_size"),
        .scale = try positive(r, "routed_scaling_factor"),
    };
    if (m.topk > m.experts or m.latent % 32 != 0 or m.inter % 32 != 0)
        return refuse(r.problem, error.Layout, "config.json: experts per token within n_routed_experts, and latent and expert widths in MXFP4's groups of 32", .{});
    return m;
}

fn rope(r: fields.Reader) Error!Rope {
    const y: fields.Reader = .{ .object = try object(r, "rope_scaling"), .source = "config.json rope_scaling", .problem = r.problem };
    if (!std.mem.eql(u8, try text(y, "type"), "yarn"))
        return refuse(r.problem, error.Unsupported, "config.json rope_scaling: type must be \"yarn\" (the HCA layers' scaling)", .{});
    return .{
        .theta = try positive(r, "rope_theta"),
        .compress_theta = try positive(r, "compress_rope_theta"),
        .factor = try positive(y, "factor"),
        .original = try int(y, "original_max_position_embeddings"),
        .beta_fast = try positive(y, "beta_fast"),
        .beta_slow = try positive(y, "beta_slow"),
    };
}

fn dspark(r: fields.Reader, layers: u32, vocab: u32) Error!Dspark {
    var d: Dspark = .{ .block = try int(r, "dspark_block_size"), .rank = try int(r, "dspark_markov_rank"), .noise_token = try int(r, "dspark_noise_token_id"), .taps = @splat(0), .taps_n = 0 };
    d.taps_n = try indices(r, "dspark_target_layer_ids", layers, &d.taps);
    if (d.noise_token >= vocab) return refuse(r.problem, error.Layout, "config.json: dspark_noise_token_id is not a token of the {d}-token vocabulary", .{vocab});
    return d;
}

/// config.json as the release ships it; a refusal names its key in `p`.
pub fn parse(gpa: std.mem.Allocator, json: []const u8, p: *Problem) Error!Config {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch return refuse(p, error.BadConfig, "config.json is not JSON", .{});
    defer parsed.deinit();
    if (parsed.value != .object) return refuse(p, error.BadConfig, "config.json is not an object", .{});
    const r: fields.Reader = .{ .object = parsed.value.object, .source = "config.json", .problem = p };
    const arch = r.object.get("architectures") orelse .null;
    const named = arch == .array and arch.array.items.len == 1 and arch.array.items[0] == .string and
        std.mem.eql(u8, arch.array.items[0].string, "Ernie5_5_MoeForCausalLM");
    if (!named or !std.mem.eql(u8, textOf(r.object, "model_type") orelse "", "ernie5_5_moe"))
        return refuse(p, error.NotErnie55, "not an ERNIE 5.5 checkpoint: architectures Ernie5_5_MoeForCausalLM, model_type ernie5_5_moe", .{});
    for (pins) |pin| try pinned(r, pin);
    try quantization(r);
    var c: Config = undefined;
    c.hidden = try int(r, "hidden_size");
    c.layers = try int(r, "num_hidden_layers");
    if (c.layers > max_layers) return refuse(p, error.Layout, "config.json: {d} layers, the plan tables hold {d}", .{ c.layers, max_layers });
    c.vocab = try int(r, "vocab_size");
    c.max_context = try int(r, "max_position_embeddings");
    c.eps = try positive(r, "rms_norm_eps");
    @memset(&c.compress, 0);
    try compressRatios(r, c.layers, &c.compress);
    c.hca = try hca(r);
    c.mla = try mla(r);
    c.index = .{ .heads = try int(r, "index_n_heads"), .dim = try int(r, "index_head_dim"), .topk = try int(r, "index_topk") };
    c.moe = try moe(r, c.layers);
    c.hc = .{ .streams = try int(r, "hc_mult"), .sinkhorn = try int(r, "hc_sinkhorn_iters"), .eps = try positive(r, "hc_eps") };
    c.rope = try rope(r);
    c.situ_beta = try positive(r, "activation_situ_beta");
    c.situ_linear_beta = try positive(r, "activation_situ_linear_beta");
    c.dspark = try dspark(r, c.layers, c.vocab);
    return c;
}
