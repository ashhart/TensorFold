//! gpt-oss config admission for host metadata; arithmetic details absent from the checkpoint remain unqualified.
const std = @import("std");
pub const max_layers = 128;
pub const max_topk = 32;
pub const Kind = enum { sliding, full };
pub const Why = struct {
    buf: [320]u8 = @splat(0),
    len: usize = 0,

    pub fn set(w: *Why, key: []const u8) void {
        const message = std.fmt.bufPrint(&w.buf, "gpt_oss refuses field {s}", .{key}) catch w.buf[0..];
        w.len = message.len;
    }
    pub fn text(w: *const Why) []const u8 {
        return w.buf[0..w.len];
    }
};

pub const Yarn = struct { factor: f64, beta_fast: f64, beta_slow: f64, original: u32, truncate: bool };
pub const Config = struct {
    hidden: u32,
    layers: u32,
    vocab: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    experts: u32,
    topk: u32,
    context: u32,
    initial_context: u32,
    window: u32,
    eps: f64,
    rope_theta: f64,
    swiglu_limit: f64,
    eos: u32,
    pad: u32,
    yarn: Yarn,
    layer: [max_layers]Kind,

    pub fn kind(c: *const Config, i: usize) !Kind {
        if (i >= c.layers) return error.BadLayer;
        return c.layer[i];
    }
    pub fn qWidth(c: *const Config) u32 {
        return c.heads * c.head_dim;
    }
    pub fn kvWidth(c: *const Config) u32 {
        return c.kv_heads * c.head_dim;
    }
};

pub fn bad(why: *Why, key: []const u8) error{BadGptOssConfig} {
    why.set(key);
    return error.BadGptOssConfig;
}

pub fn field(o: std.json.ObjectMap, key: []const u8, why: *Why) !std.json.Value {
    return o.get(key) orelse return bad(why, key);
}

pub fn uint(o: std.json.ObjectMap, key: []const u8, why: *Why) !u32 {
    const v = try field(o, key, why);
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return bad(why, key);
    return @intCast(v.integer);
}

fn positive(o: std.json.ObjectMap, key: []const u8, limit: u32, why: *Why) !u32 {
    const n = try uint(o, key, why);
    if (n == 0 or n > limit) return bad(why, key);
    return n;
}

fn number(o: std.json.ObjectMap, key: []const u8, why: *Why) !f64 {
    const v = try field(o, key, why);
    const n: f64 = switch (v) {
        .integer => |x| @floatFromInt(x),
        .float => |x| x,
        else => return bad(why, key),
    };
    if (!std.math.isFinite(n)) return bad(why, key);
    return n;
}

fn word(o: std.json.ObjectMap, key: []const u8, expected: []const u8, why: *Why) !void {
    const v = try field(o, key, why);
    if (v != .string or !std.mem.eql(u8, v.string, expected)) return bad(why, key);
}

fn flag(o: std.json.ObjectMap, key: []const u8, expected: bool, why: *Why) !void {
    const v = try field(o, key, why);
    if (v != .bool or v.bool != expected) return bad(why, key);
}

fn quant(o: std.json.ObjectMap, why: *Why) !void {
    const v = try field(o, "quantization_config", why);
    if (v != .object) return bad(why, "quantization_config");
    word(v.object, "quant_method", "mxfp4", why) catch return bad(why, "quantization_config.quant_method");
    if (v.object.count() != 2) return bad(why, "quantization_config");
    const list = v.object.get("modules_to_not_convert") orelse return bad(why, "quantization_config.modules_to_not_convert");
    const expected = [_][]const u8{ "model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head" };
    if (list != .array or list.array.items.len != expected.len) return bad(why, "quantization_config.modules_to_not_convert");
    var seen: [4]bool = @splat(false);
    for (list.array.items) |entry| {
        if (entry != .string) return bad(why, "quantization_config.modules_to_not_convert");
        for (expected, 0..) |name, i| {
            if (std.mem.eql(u8, name, entry.string) and !seen[i]) {
                seen[i] = true;
                break;
            }
        } else return bad(why, "quantization_config.modules_to_not_convert");
    }
}

fn rope(o: std.json.ObjectMap, initial: u32, why: *Why) !Yarn {
    const v = try field(o, "rope_scaling", why);
    if (v != .object) return bad(why, "rope_scaling");
    const p = v.object;
    word(p, "rope_type", "yarn", why) catch return bad(why, "rope_scaling.rope_type");
    flag(p, "truncate", false, why) catch return bad(why, "rope_scaling.truncate");
    const y: Yarn = .{
        .factor = number(p, "factor", why) catch return bad(why, "rope_scaling.factor"),
        .beta_fast = number(p, "beta_fast", why) catch return bad(why, "rope_scaling.beta_fast"),
        .beta_slow = number(p, "beta_slow", why) catch return bad(why, "rope_scaling.beta_slow"),
        .original = uint(p, "original_max_position_embeddings", why) catch return bad(why, "rope_scaling.original_max_position_embeddings"),
        .truncate = false,
    };
    if (p.count() != 6) return bad(why, "rope_scaling");
    if (y.factor < 1) return bad(why, "rope_scaling.factor");
    if (y.beta_fast <= 0) return bad(why, "rope_scaling.beta_fast");
    if (y.beta_slow <= 0 or y.beta_slow >= y.beta_fast) return bad(why, "rope_scaling.beta_slow");
    if (y.original != initial) return bad(why, "rope_scaling.original_max_position_embeddings");
    return y;
}

pub fn parse(a: std.mem.Allocator, bytes: []const u8, why: *Why) !Config {
    why.len = 0;
    var doc = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch |err| {
        if (err == error.OutOfMemory) return err;
        return bad(why, "config.json");
    };
    defer doc.deinit();
    if (doc.value != .object) return bad(why, "config.json");
    const o = doc.value.object;
    try word(o, "model_type", "gpt_oss", why);
    const arch = try field(o, "architectures", why);
    if (arch != .array or arch.array.items.len != 1 or arch.array.items[0] != .string or
        !std.mem.eql(u8, arch.array.items[0].string, "GptOssForCausalLM")) return bad(why, "architectures");
    try flag(o, "attention_bias", true, why);
    try flag(o, "tie_word_embeddings", false, why);
    try flag(o, "use_cache", true, why);
    try flag(o, "output_router_logits", false, why);
    try word(o, "hidden_act", "silu", why);
    if (try number(o, "attention_dropout", why) != 0) return bad(why, "attention_dropout");
    try quant(o, why);
    const initial = try positive(o, "initial_context_length", 1 << 30, why);
    var c: Config = .{
        .hidden = try positive(o, "hidden_size", 65536, why),
        .layers = try positive(o, "num_hidden_layers", max_layers, why),
        .vocab = try positive(o, "vocab_size", 1 << 24, why),
        .heads = try positive(o, "num_attention_heads", 512, why),
        .kv_heads = try positive(o, "num_key_value_heads", 512, why),
        .head_dim = try positive(o, "head_dim", 256, why),
        .intermediate = try positive(o, "intermediate_size", 65536, why),
        .experts = try positive(o, "num_local_experts", 512, why),
        .topk = try positive(o, "experts_per_token", max_topk, why),
        .context = try positive(o, "max_position_embeddings", 1 << 30, why),
        .initial_context = initial,
        .window = try positive(o, "sliding_window", 1 << 30, why),
        .eps = try number(o, "rms_norm_eps", why),
        .rope_theta = try number(o, "rope_theta", why),
        .swiglu_limit = try number(o, "swiglu_limit", why),
        .eos = try uint(o, "eos_token_id", why),
        .pad = try uint(o, "pad_token_id", why),
        .yarn = try rope(o, initial, why),
        .layer = @splat(.full),
    };
    if (c.hidden % 32 != 0) return bad(why, "hidden_size");
    if (c.intermediate % 32 != 0) return bad(why, "intermediate_size");
    if (c.heads % c.kv_heads != 0) return bad(why, "num_attention_heads");
    if (c.head_dim % 2 != 0) return bad(why, "head_dim");
    if (c.topk > c.experts) return bad(why, "experts_per_token");
    if (o.contains("num_experts_per_tok") and try uint(o, "num_experts_per_tok", why) != c.topk) return bad(why, "num_experts_per_tok");
    if (c.eps <= 0) return bad(why, "rms_norm_eps");
    if (c.rope_theta <= 0) return bad(why, "rope_theta");
    if (c.swiglu_limit <= 0) return bad(why, "swiglu_limit");
    if (c.initial_context > c.context) return bad(why, "initial_context_length");
    if (c.window > c.context) return bad(why, "sliding_window");
    if (c.eos >= c.vocab) return bad(why, "eos_token_id");
    if (c.pad >= c.vocab) return bad(why, "pad_token_id");
    const kinds = try field(o, "layer_types", why);
    if (kinds != .array or kinds.array.items.len != c.layers) return bad(why, "layer_types");
    for (kinds.array.items, 0..) |v, i| {
        if (v != .string) return bad(why, "layer_types");
        c.layer[i] = if (std.mem.eql(u8, v.string, "sliding_attention")) .sliding else if (std.mem.eql(u8, v.string, "full_attention")) .full else return bad(why, "layer_types");
    }
    const known = [_][]const u8{ "architectures", "attention_bias", "attention_dropout", "eos_token_id", "experts_per_token", "head_dim", "hidden_act", "hidden_size", "initial_context_length", "initializer_range", "intermediate_size", "layer_types", "max_position_embeddings", "model_type", "num_attention_heads", "num_experts_per_tok", "num_hidden_layers", "num_key_value_heads", "num_local_experts", "output_router_logits", "pad_token_id", "quantization_config", "rms_norm_eps", "rope_scaling", "rope_theta", "router_aux_loss_coef", "sliding_window", "swiglu_limit", "tie_word_embeddings", "transformers_version", "use_cache", "vocab_size", "_name_or_path", "_commit_hash" };
    for (o.keys()) |key| {
        for (known) |name| {
            if (std.mem.eql(u8, key, name)) break;
        } else return bad(why, key);
    }
    return c;
}
