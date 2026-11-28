//! GigaChat 4's shape from config.json (model_type gigachat4), refusing every layout the engine does not run.
const std = @import("std");
const fields = @import("config_fields");

pub const Problem = fields.Problem;
pub const Error = error{ BadConfig, NotGigaChat4, Unsupported, Layout, Quant };

/// The most decoder layers the plan tables hold (the releases have 32 and 64).
pub const max_layers = 128;
/// The most routed experts the bounded weight map holds (the releases have 192 and 256).
pub const max_experts = 1024;
/// The widest dimension a config may give; every product of two stays far inside u64.
const max_dim = 1 << 24;

/// How the checkpoint stores the projections the FP8 release quantizes; every other tensor is BF16 in both.
pub const Storage = enum { fp8, bf16 };

/// The sparse-attention layers' multi-head latent attention (MLA).
pub const Mla = struct { heads: u32, q_lora: u32, kv_lora: u32, nope: u32, rope: u32, value: u32 };
/// The DSA indexer of each MLA layer outside dense_attention_layers; its key norm is a plain RMSNorm.
pub const Indexer = struct { heads: u32, dim: u32, topk: u32, eps: f64 };
/// The Gated DeltaNet layers: heads and dims of keys and values, the short convolution and the output gate.
pub const DeltaNet = struct { key_heads: u32, value_heads: u32, key_dim: u32, value_dim: u32, conv: u32, gate_scale: f64, eps: f64 };
/// Routed experts with one shared expert; the first `dense_layers` layers run a dense MLP of `dense_inter`.
pub const Moe = struct { experts: u32, topk: u32, inter: u32, shared: u32, dense_layers: u32, dense_inter: u32, scale: f64 };
/// GatedNorm, the model-wide norm: its epsilon and the constant its sigmoid gate is multiplied by.
pub const Norm = struct { eps: f64, gating_weight: f64 };
pub const Rope = struct { theta: f64, factor: f64, original: u32, beta_fast: f64, beta_slow: f64, mscale: f64, mscale_all_dim: f64 };

pub const Config = struct {
    hidden: u32,
    layers: u32,
    vocab: u32,
    max_context: u32,
    storage: Storage,
    /// Per layer: an MLA layer (full_attention_layers); every other layer is Gated DeltaNet.
    full: [max_layers]bool,
    /// Per layer: an MLA layer without an indexer (dense_attention_layers).
    dense_attention: [max_layers]bool,
    mla: Mla,
    index: Indexer,
    delta: DeltaNet,
    moe: Moe,
    norm: Norm,
    rope: Rope,
    /// SwiGLU clamps (gate at most L, up within -L..L); 0 turns them off.
    swiglu_limit: f64,
    /// tanh soft-capping of the final logits; 0 when the config gives none.
    softcap: f64,
    /// The multi-token-prediction layers stored after the decoder layers (num_nextn_predict_layers).
    mtp_layers: u32,

    pub fn isMla(c: *const Config, i: u32) bool {
        return c.full[i];
    }

    pub fn isDsa(c: *const Config, i: u32) bool {
        return c.full[i] and !c.dense_attention[i];
    }

    pub fn isDense(c: *const Config, i: u32) bool {
        return i < c.moe.dense_layers;
    }
};

/// Why a key is held to the release's value: the weight map follows it, the reference code runs one value, or unread.
const Basis = enum { weights, reference, unread };

const Pin = struct {
    key: []const u8,
    want: union(enum) { flag: bool, text: []const u8, int: i64, none },
    basis: Basis,
};

const pins = [_]Pin{
    .{ .key = "norm_type", .want = .{ .text = "GatedNorm" }, .basis = .weights },
    .{ .key = "layernorm_type", .want = .{ .text = "pre_post" }, .basis = .weights },
    .{ .key = "linear_attention_type", .want = .{ .text = "GigaChat4GatedDeltaNet" }, .basis = .weights },
    .{ .key = "linear_use_short_conv", .want = .{ .flag = true }, .basis = .weights },
    .{ .key = "gated_attention", .want = .{ .flag = true }, .basis = .weights },
    .{ .key = "use_shared_expert_sigmoid", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "n_shared_experts", .want = .{ .int = 1 }, .basis = .weights },
    .{ .key = "num_krope_heads", .want = .{ .int = 1 }, .basis = .weights },
    .{ .key = "num_nextn_predict_layers", .want = .{ .int = 1 }, .basis = .weights },
    .{ .key = "nextn_is_linear", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "tie_word_embeddings", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "attention_bias", .want = .{ .flag = false }, .basis = .weights },
    .{ .key = "dtype", .want = .{ .text = "bfloat16" }, .basis = .weights },
    .{ .key = "linear_gating_type", .want = .{ .text = "gated_rmsnorm_sigmoid" }, .basis = .reference },
    .{ .key = "use_mla_scaling_factor", .want = .{ .flag = true }, .basis = .reference },
    .{ .key = "rope_interleave", .want = .{ .flag = true }, .basis = .reference },
    .{ .key = "hidden_act", .want = .{ .text = "silu" }, .basis = .reference },
    .{ .key = "n_group", .want = .{ .int = 1 }, .basis = .reference },
    .{ .key = "topk_group", .want = .{ .int = 1 }, .basis = .reference },
    .{ .key = "norm_topk_prob", .want = .{ .flag = true }, .basis = .reference },
    .{ .key = "nextn_is_sparse", .want = .{ .flag = false }, .basis = .unread },
    .{ .key = "indexer_rotate_activation_enabled", .want = .none, .basis = .unread },
};

/// Keys config.json leaves out, whose reference defaults the engine holds; another value is refused.
const defaults = [_]struct { key: []const u8, want: []const u8 }{
    .{ .key = "indexer_norm_type", .want = "LlamaRMSNorm" },
    .{ .key = "attention_type", .want = "LlamaLatentSparseAttention" },
};

/// The FP8 release's ignored_layers: the modules it keeps in BF16, which the weight map follows.
const bf16_modules = [_][]const u8{
    "lm_head",
    "embed_tokens",
    "shared_head.head",
    "mlp.gate.weight",
    "eh_proj",
    "self_attn.A_log",
    "self_attn.conv1d",
    "self_attn.dt_bias",
    "self_attn.in_proj_ba",
    "self_attn.norm.weight",
    "self_attn.gate_proj",
    "gate_up_projection",
    "gate_down_projection",
    "shared_expert_sigmoid",
    "indexer.wk",
    "indexer.weights_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.shared_experts",
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

fn because(basis: Basis) []const u8 {
    return switch (basis) {
        .weights => "the weight map follows it",
        .reference => "the reference code runs no other value here",
        .unread => "its meaning is not read yet, so the engine holds the release's value",
    };
}

fn pinned(r: fields.Reader, pin: Pin) Error!void {
    const v = r.value(pin.key) catch return error.BadConfig;
    const same = switch (pin.want) {
        .flag => |b| v == .bool and v.bool == b,
        .text => |s| v == .string and std.mem.eql(u8, v.string, s),
        .int => |n| v == .integer and v.integer == n,
        .none => v == .null,
    };
    if (same) return;
    const why = because(pin.basis);
    return switch (pin.want) {
        .flag => |b| refuse(r.problem, error.Unsupported, "config.json: {s} must be {}: {s}", .{ pin.key, b, why }),
        .text => |s| refuse(r.problem, error.Unsupported, "config.json: {s} must be \"{s}\": {s}", .{ pin.key, s, why }),
        .int => |n| refuse(r.problem, error.Unsupported, "config.json: {s} must be {d}: {s}", .{ pin.key, n, why }),
        .none => refuse(r.problem, error.Unsupported, "config.json: {s} must be null: {s}", .{ pin.key, why }),
    };
}

fn int(r: fields.Reader, key: []const u8) Error!u32 {
    return r.requiredInt(u32, key, .{ .min = 1, .max = max_dim }) catch error.BadConfig;
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

fn textOf(o: fields.Object, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn text(r: fields.Reader, key: []const u8) Error![]const u8 {
    const v = r.value(key) catch return error.BadConfig;
    if (v != .string) return refuse(r.problem, error.BadConfig, "{s}: {s} is not a string", .{ r.source, key });
    return v.string;
}

/// A JSON array of distinct layer indices below `layers` into `out`; `empty_ok` admits [].
fn layerSet(r: fields.Reader, key: []const u8, layers: u32, empty_ok: bool, out: *[max_layers]bool) Error!u32 {
    @memset(out, false);
    const v = r.value(key) catch return error.BadConfig;
    if (v != .array or (v.array.items.len == 0 and !empty_ok))
        return refuse(r.problem, error.Layout, "config.json: {s} must list layers", .{key});
    for (v.array.items) |e| {
        if (e != .integer or e.integer < 0 or e.integer >= layers)
            return refuse(r.problem, error.Layout, "config.json: {s} names a layer past the {d} layers", .{ key, layers });
        const i: usize = @intCast(e.integer);
        if (out[i]) return refuse(r.problem, error.Layout, "config.json: {s} names layer {d} twice", .{ key, i });
        out[i] = true;
    }
    return @intCast(v.array.items.len);
}

/// FP8 E4M3 in 128x128 blocks with dynamic activations and the release's BF16 modules, or no quantization at all.
fn storage(r: fields.Reader) Error!Storage {
    const v = r.object.get("quantization_config") orelse return .bf16;
    if (v != .object) return refuse(r.problem, error.Quant, "config.json: quantization_config is not an object", .{});
    const q: fields.Reader = .{ .object = v.object, .source = "config.json quantization_config", .problem = r.problem };
    const fp8 = std.mem.eql(u8, textOf(q.object, "quant_method") orelse "", "fp8") and
        std.mem.eql(u8, textOf(q.object, "fmt") orelse "", "e4m3") and
        std.mem.eql(u8, textOf(q.object, "activation_scheme") orelse "", "dynamic");
    const blocks = q.object.get("weight_block_size") orelse .null;
    const square = blocks == .array and blocks.array.items.len == 2 and for (blocks.array.items) |b| {
        if (b != .integer or b.integer != 128) break false;
    } else true;
    if (!fp8 or !square)
        return refuse(r.problem, error.Quant, "quantization_config: the loader reads FP8 E4M3 in 128x128 blocks with dynamic activations", .{});
    const ignored = q.object.get("ignored_layers") orelse .null;
    const same = ignored == .array and ignored.array.items.len == bf16_modules.len and for (ignored.array.items, bf16_modules) |e, m| {
        if (e != .string or !std.mem.eql(u8, e.string, m)) break false;
    } else true;
    if (!same) return refuse(r.problem, error.Quant, "quantization_config: ignored_layers must be the release's 20 BF16 modules: the weight map follows it", .{});
    return .fp8;
}

fn mla(r: fields.Reader) Error!Mla {
    const m: Mla = .{
        .heads = try int(r, "num_attention_heads"),
        .q_lora = try int(r, "q_lora_rank"),
        .kv_lora = try int(r, "kv_lora_rank"),
        .nope = try int(r, "qk_nope_head_dim"),
        .rope = try int(r, "qk_rope_head_dim"),
        .value = try int(r, "v_head_dim"),
    };
    if (try int(r, "num_key_value_heads") != m.heads)
        return refuse(r.problem, error.Layout, "config.json: MLA layers need as many key/value heads as query heads", .{});
    if (try int(r, "qk_head_dim") != @as(u64, m.nope) + m.rope or try int(r, "head_dim") != m.rope)
        return refuse(r.problem, error.Layout, "config.json: qk_head_dim must be qk_nope + qk_rope and head_dim qk_rope", .{});
    return m;
}

fn delta(r: fields.Reader, eps: f64) Error!DeltaNet {
    const d: DeltaNet = .{
        .key_heads = try int(r, "linear_num_key_heads"),
        .value_heads = try int(r, "linear_num_value_heads"),
        .key_dim = try int(r, "linear_key_head_dim"),
        .value_dim = try int(r, "linear_value_head_dim"),
        .conv = try int(r, "linear_conv_kernel_dim"),
        .gate_scale = try positive(r, "linear_sigmoid_gate_scale"),
        .eps = (r.optionalFloat("linear_attn_o_norm_eps", .absent) catch return error.BadConfig) orelse eps,
    };
    if (d.value_heads % d.key_heads != 0 or d.conv < 2 or d.eps <= 0)
        return refuse(r.problem, error.Layout, "config.json: Gated DeltaNet value heads must split over key heads, a convolution of 2 or more, a positive eps", .{});
    return d;
}

fn moe(r: fields.Reader, layers: u32) Error!Moe {
    const m: Moe = .{
        .experts = try int(r, "n_routed_experts"),
        .topk = try int(r, "num_experts_per_tok"),
        .inter = try int(r, "moe_intermediate_size"),
        .shared = try int(r, "n_shared_experts"),
        .dense_layers = r.requiredInt(u32, "first_k_dense_replace", .{ .min = 0, .max = layers }) catch return error.BadConfig,
        .dense_inter = try int(r, "intermediate_size"),
        .scale = try positive(r, "routed_scaling_factor"),
    };
    if (m.experts > max_experts)
        return refuse(r.problem, error.Layout, "config.json: n_routed_experts must not exceed {d} for the bounded weight map", .{max_experts});
    if (m.topk > m.experts) return refuse(r.problem, error.Layout, "config.json: num_experts_per_tok exceeds n_routed_experts", .{});
    return m;
}

fn rope(r: fields.Reader) Error!Rope {
    const y: fields.Reader = .{ .object = try object(r, "rope_scaling"), .source = "config.json rope_scaling", .problem = r.problem };
    const kind = textOf(y.object, "rope_type") orelse textOf(y.object, "type") orelse "";
    if (!std.mem.eql(u8, kind, "yarn"))
        return refuse(r.problem, error.Unsupported, "config.json rope_scaling: type must be \"yarn\"", .{});
    return .{
        .theta = try positive(r, "rope_theta"),
        .factor = try positive(y, "factor"),
        .original = try int(y, "original_max_position_embeddings"),
        .beta_fast = try positive(y, "beta_fast"),
        .beta_slow = try positive(y, "beta_slow"),
        .mscale = try positive(y, "mscale"),
        .mscale_all_dim = try positive(y, "mscale_all_dim"),
    };
}

fn indexer(r: fields.Reader, m: Mla) Error!Indexer {
    const x: Indexer = .{
        .heads = try int(r, "index_n_heads"),
        .dim = try int(r, "index_head_dim"),
        .topk = try int(r, "index_topk"),
        .eps = (r.optionalFloat("indexer_norm_eps", .absent) catch return error.BadConfig) orelse 1e-6,
    };
    if (x.dim < m.rope or x.eps <= 0) return refuse(r.problem, error.Layout, "config.json: index_head_dim must hold the qk_rope dims", .{});
    return x;
}

/// config.json as the releases ship it; a refusal names its key in `p`.
pub fn parse(gpa: std.mem.Allocator, json: []const u8, p: *Problem) Error!Config {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, json, .{}) catch return refuse(p, error.BadConfig, "config.json is not JSON", .{});
    defer parsed.deinit();
    if (parsed.value != .object) return refuse(p, error.BadConfig, "config.json is not an object", .{});
    const r: fields.Reader = .{ .object = parsed.value.object, .source = "config.json", .problem = p };
    const arch = r.object.get("architectures") orelse .null;
    const named = arch == .array and arch.array.items.len == 1 and arch.array.items[0] == .string and
        std.mem.eql(u8, arch.array.items[0].string, "GigaChat4ForCausalLM");
    if (!named or !std.mem.eql(u8, textOf(r.object, "model_type") orelse "", "gigachat4"))
        return refuse(p, error.NotGigaChat4, "not a GigaChat 4 checkpoint: architectures GigaChat4ForCausalLM, model_type gigachat4", .{});
    for (pins) |pin| try pinned(r, pin);
    for (defaults) |d| {
        const v = r.object.get(d.key) orelse continue;
        if (v != .string or !std.mem.eql(u8, v.string, d.want))
            return refuse(p, error.Unsupported, "config.json: {s} must be \"{s}\" or absent: the reference default", .{ d.key, d.want });
    }
    var c: Config = undefined;
    c.storage = try storage(r);
    c.hidden = try int(r, "hidden_size");
    c.layers = try int(r, "num_hidden_layers");
    if (c.layers > max_layers) return refuse(p, error.Layout, "config.json: {d} layers, the plan tables hold {d}", .{ c.layers, max_layers });
    c.vocab = try int(r, "vocab_size");
    c.max_context = r.requiredInt(u32, "max_position_embeddings", .{ .min = 1, .max = std.math.maxInt(u32) }) catch return error.BadConfig;
    c.norm = .{ .eps = try positive(r, "rms_norm_eps"), .gating_weight = try positive(r, "layernorm_gating_weight") };
    _ = try layerSet(r, "full_attention_layers", c.layers, false, &c.full);
    _ = try layerSet(r, "dense_attention_layers", c.layers, true, &c.dense_attention);
    for (0..c.layers) |i| if (c.dense_attention[i] and !c.full[i])
        return refuse(p, error.Layout, "config.json: dense_attention_layers must be a subset of full_attention_layers", .{});
    c.mla = try mla(r);
    c.index = try indexer(r, c.mla);
    c.delta = try delta(r, c.norm.eps);
    c.moe = try moe(r, c.layers);
    c.rope = try rope(r);
    c.swiglu_limit = r.requiredFloat("swiglu_limit") catch return error.BadConfig;
    if (c.swiglu_limit < 0) return refuse(p, error.BadConfig, "config.json: swiglu_limit is negative", .{});
    c.softcap = (r.optionalFloat("final_logit_softcapping", .absent) catch return error.BadConfig) orelse 0;
    if (c.softcap < 0) return refuse(p, error.BadConfig, "config.json: final_logit_softcapping is negative", .{});
    c.mtp_layers = 1;
    return c;
}
