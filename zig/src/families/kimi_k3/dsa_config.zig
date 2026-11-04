//! The supported K3.1 release geometry; unknown DSA semantics or shapes must never fall back to dense K3.
const std = @import("std");
const Value = std.json.Value;
const Object = std.json.ObjectMap;

pub const prefix_limit: u32 = 2047;
pub const Indexer = struct { heads: u32 = 32, dim: u32 = 128, topk: u32 = 2048 };
pub const limit_message = "Kimi K3.1 supports at most 2047 total evaluated tokens, including the cached prefix and reply";

fn object(v: Value) !Object {
    return if (v == .object) v.object else error.BadK31Config;
}

fn integer(o: Object, key: []const u8, want: i64) !void {
    const v = o.get(key) orelse return error.BadK31Config;
    if (v != .integer or v.integer != want) return error.UnsupportedK31Shape;
}

fn number(o: Object, key: []const u8, want: f64) !void {
    const v = o.get(key) orelse return error.BadK31Config;
    const got: f64 = switch (v) {
        .integer => @floatFromInt(v.integer),
        .float => v.float,
        else => return error.BadK31Config,
    };
    if (got != want) return error.UnsupportedK31Shape;
}

fn string(o: Object, key: []const u8, want: []const u8) !void {
    const v = o.get(key) orelse return error.BadK31Config;
    if (v != .string or !std.mem.eql(u8, v.string, want)) return error.UnsupportedK31Shape;
}

fn boolean(o: Object, key: []const u8, want: bool) !void {
    const v = o.get(key) orelse return error.BadK31Config;
    if (v != .bool or v.bool != want) return error.UnsupportedK31Shape;
}

fn layerSet(o: Object, key: []const u8) !std.StaticBitSet(128) {
    const v = o.get(key) orelse return error.BadK31Config;
    if (v != .array) return error.BadK31Config;
    var set = std.StaticBitSet(128).empty;
    for (v.array.items) |l| {
        if (l != .integer or l.integer < 1 or l.integer > 93) return error.BadK31Layers;
        const i: usize = @intCast(l.integer - 1);
        if (set.isSet(i)) return error.BadK31Layers;
        set.set(i);
    }
    return set;
}

pub fn parse(root: Object, text: Object) !?Indexer {
    const kind = root.get("model_type");
    const k31 = if (kind) |v| v == .string and std.mem.eql(u8, v.string, "kimi_k31") else false;
    const dv = text.get("dsa_config") orelse return if (k31) error.MissingK31Dsa else null;
    if (!k31 and root.contains("text_config")) return error.UnsupportedK31Identity;
    try string(text, "model_type", "kimi_linear");
    const d = try object(dv);
    const keys = [_][]const u8{
        "dsa_layers",        "index_topk",      "index_head_dim",         "index_num_heads", "index_rope_dim",
        "index_k_norm_type", "index_topk_freq", "index_skip_topk_offset",
    };
    if (d.count() != keys.len) return error.UnsupportedK31Dsa;
    try integer(d, "index_topk", 2048);
    try integer(d, "index_head_dim", 128);
    try integer(d, "index_num_heads", 32);
    try integer(d, "index_rope_dim", 0);
    try string(d, "index_k_norm_type", "layernorm");
    try integer(d, "index_topk_freq", 1);
    try integer(d, "index_skip_topk_offset", 1);
    const ints = .{
        .{ "hidden_size", 7168 },               .{ "num_hidden_layers", 93 },    .{ "vocab_size", 163840 },
        .{ "first_k_dense_replace", 1 },        .{ "intermediate_size", 33792 }, .{ "num_experts", 896 },
        .{ "num_experts_per_token", 16 },       .{ "num_shared_experts", 2 },    .{ "moe_intermediate_size", 3072 },
        .{ "routed_expert_hidden_size", 3584 }, .{ "num_attention_heads", 96 },  .{ "num_key_value_heads", 96 },
        .{ "q_lora_rank", 1536 },               .{ "kv_lora_rank", 512 },        .{ "qk_nope_head_dim", 128 },
        .{ "qk_rope_head_dim", 64 },            .{ "v_head_dim", 128 },          .{ "attn_res_block_size", 12 },
        .{ "num_nextn_predict_layers", 0 },     .{ "num_expert_group", 1 },      .{ "topk_group", 1 },
        .{ "moe_layer_freq", 1 },
    };
    inline for (ints) |pair| try integer(text, pair[0], pair[1]);
    try number(text, "rms_norm_eps", 1e-5);
    try number(text, "activation_situ_beta", 4);
    try number(text, "activation_situ_linear_beta", 25);
    try number(text, "routed_scaling_factor", 1);
    try string(text, "hidden_act", "situ");
    try string(text, "moe_router_activation_func", "sigmoid");
    inline for (.{ "latent_moe_use_norm", "mla_use_nope", "mla_use_output_gate", "moe_renormalize" }) |key|
        try boolean(text, key, true);
    const la = try object(text.get("linear_attn_config") orelse return error.BadK31Config);
    try integer(la, "num_heads", 96);
    try integer(la, "head_dim", 128);
    try integer(la, "short_conv_kernel_size", 4);
    try number(la, "gate_lower_bound", -5);
    try boolean(la, "use_full_rank_gate", true);
    const ds = try layerSet(d, "dsa_layers");
    const full = try layerSet(la, "full_attn_layers");
    var expected = std.StaticBitSet(128).empty;
    var i: usize = 3;
    while (i < 92) : (i += 4) expected.set(i);
    expected.set(92);
    if (!std.meta.eql(ds, full) or !std.meta.eql(ds, expected)) return error.BadK31Layers;
    const kd = try layerSet(la, "kda_layers");
    for (0..93) |l| if (kd.isSet(l) == ds.isSet(l)) return error.BadK31Layers;
    return .{};
}

/// Stop one token below our GLM all-visible boundary, including the offset's possible one-position shift.
pub fn checkEnd(pos: u32, rows: u32) !void {
    if (@as(u64, pos) + rows > prefix_limit) return error.K31ContextLimit;
}
