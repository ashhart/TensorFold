//! Sage-A1.5 (qwen3_5_moe, 35B-A3B) on an Intel GPU: the config.json fields the XPU path needs, and the shape checks.

const std = @import("std");

pub const Rope = struct {
    rope_theta: f64,
    partial_rotary_factor: f32 = 1.0,
    mrope_interleaved: bool = false,
    mrope_section: []const u32 = &.{},
};

pub const Text = struct {
    hidden_size: u32,
    vocab_size: u32,
    num_hidden_layers: u32,
    layer_types: []const []const u8,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    attn_output_gate: bool = false,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    num_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    shared_expert_intermediate_size: u32,
    mtp_num_hidden_layers: u32 = 0,
    rms_norm_eps: f32,
    rope_parameters: Rope,
};

pub const Vision = struct {
    depth: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_heads: u32,
    out_hidden_size: u32,
    patch_size: u32,
    spatial_merge_size: u32,
    temporal_patch_size: u32,
    num_position_embeddings: u32,
    deepstack_visual_indexes: []const u32 = &.{},
};

/// EXL3 and AutoRound name a quant_method. MLX writes only group_size, bits and mode here.
pub const Quant = struct {
    quant_method: []const u8 = "mlx",
    codebook: ?[]const u8 = null,
};

pub const Config = struct {
    model_type: []const u8,
    text_config: Text,
    vision_config: ?Vision = null,
    quantization_config: ?Quant = null,
};

pub const Kind = enum { gdn, attention };

/// The sizes the kernels take as arguments, derived once from the config.
pub const Shape = struct {
    hidden: u32,
    layers: u32,
    vocab: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rope_dims: u32,
    gdn_k_heads: u32,
    gdn_v_heads: u32,
    gdn_head_dim: u32,
    experts: u32,
    top_k: u32,
    expert_ff: u32,
    shared_ff: u32,
    mtp: bool,

    /// Attention input projection out: q and its gate per head, then k, then v.
    pub fn attIn(s: Shape) u32 {
        return 2 * s.q_heads * s.head_dim + 2 * s.kv_heads * s.head_dim;
    }
    /// GDN input projection out: q|k (key heads), v and z (value heads), a and b (one per value head).
    pub fn gdnIn(s: Shape) u32 {
        return 2 * s.gdn_k_heads * s.gdn_head_dim + 2 * s.gdn_v_heads * s.gdn_head_dim + 2 * s.gdn_v_heads;
    }
};

/// The -D values zig/kernels/xpu/qwen27/shape.h takes, for a qwen27 kernel image built at shape `s`.
/// `qpg`: query heads one attn_decode subgroup holds (4 keeps a lane's q and o at 128 floats).
pub fn kernelDefines(buf: []u8, s: Shape, qpg: u32) ![]const u8 {
    const nqk = s.gdn_k_heads * 128;
    const vd = s.gdn_v_heads * 128;
    const nconv = 2 * nqk + vd;
    const gqa = s.q_heads / s.kv_heads;
    const att_k = s.q_heads * 512;
    if (s.hidden % 512 != 0 or s.gdn_v_heads % s.gdn_k_heads != 0 or gqa % qpg != 0) return error.UnsupportedConfig;
    return std.fmt.bufPrint(buf, "-DQ27_D={d} -DQ27_D512={d} -DQ27_GK={d} -DQ27_GV={d} -DQ27_GVK={d} -DQ27_NQK={d} " ++
        "-DQ27_NCONV={d} -DQ27_VD={d} -DQ27_GDN_IN={d} -DQ27_GA={d} -DQ27_GB={d} -DQ27_AQ={d} -DQ27_AKV={d} -DQ27_AH={d} " ++
        "-DQ27_AV0={d} -DQ27_GQA={d} -DQ27_QPG={d} -DQ27_QPARTS={d} -DQ27_ATT_IN={d} -DQ27_AK0={d} -DQ27_AV0C={d} " ++
        "-DQ27_OD={d} -DQ27_OB={d} -DQ27_KVROW={d}", .{
        s.hidden,                          s.hidden / 512,      s.gdn_k_heads,              s.gdn_v_heads,
        s.gdn_v_heads / s.gdn_k_heads,     nqk,                 nconv,                      vd,
        (nconv + vd + 2 * s.gdn_v_heads + 255) / 256 * 256,     nconv + vd,                 nconv + vd + s.gdn_v_heads,
        s.q_heads,                         s.kv_heads,          s.q_heads + 2 * s.kv_heads, s.q_heads + s.kv_heads,
        gqa,                               qpg,                 gqa / qpg,                  s.attIn(),
        att_k,                             att_k + s.kv_heads * 256,                        s.q_heads * 256,
        s.q_heads * 2,                     s.kv_heads * 256,
    });
}

/// What zig/build/xpu.zig compiles the Sage images with: kernelDefines of Sage-A1.5's config, qpg 4 (tested below).
pub const sage_defines = "-DQ27_D=2048 -DQ27_D512=4 -DQ27_GK=16 -DQ27_GV=32 -DQ27_GVK=2 -DQ27_NQK=2048 -DQ27_NCONV=8192 " ++
    "-DQ27_VD=4096 -DQ27_GDN_IN=12544 -DQ27_GA=12288 -DQ27_GB=12320 -DQ27_AQ=16 -DQ27_AKV=2 -DQ27_AH=20 -DQ27_AV0=18 " ++
    "-DQ27_GQA=8 -DQ27_QPG=4 -DQ27_QPARTS=2 -DQ27_ATT_IN=9216 -DQ27_AK0=8192 -DQ27_AV0C=8704 -DQ27_OD=4096 -DQ27_OB=32 " ++
    "-DQ27_KVROW=512";

pub fn kind(c: Text, layer: usize) Kind {
    return if (std.mem.eql(u8, c.layer_types[layer], "full_attention")) .attention else .gdn;
}

/// The kernels were written for these sizes; refuse a checkpoint they were not checked against.
pub fn validate(c: Config) !Shape {
    if (!std.mem.eql(u8, c.model_type, "qwen3_5_moe")) return error.UnsupportedConfig;
    const t = c.text_config;
    if (t.layer_types.len != t.num_hidden_layers) return error.UnsupportedConfig;
    for (t.layer_types, 0..) |name, i| {
        const want: []const u8 = if ((i + 1) % 4 == 0) "full_attention" else "linear_attention";
        if (!std.mem.eql(u8, name, want)) return error.UnsupportedConfig;
    }
    const ok = t.hidden_size == 2048 and t.head_dim == 256 and t.attn_output_gate and
        t.num_attention_heads % t.num_key_value_heads == 0 and t.linear_key_head_dim == 128 and
        t.linear_value_head_dim == 128 and t.linear_conv_kernel_dim == 4 and t.num_experts <= 256 and
        t.num_experts % 16 == 0 and t.num_experts_per_tok <= 8 and t.moe_intermediate_size % 128 == 0 and
        t.shared_expert_intermediate_size % 128 == 0 and t.mtp_num_hidden_layers <= 1;
    if (!ok) return error.UnsupportedConfig;
    if (c.vision_config) |v| if (v.deepstack_visual_indexes.len != 0) return error.UnsupportedConfig;
    if (c.quantization_config) |q| {
        if (std.mem.eql(u8, q.quant_method, "exl3") and !std.mem.eql(u8, q.codebook orelse "", "mul1")) return error.UnsupportedConfig;
    }
    const rope: u32 = @intFromFloat(@as(f32, @floatFromInt(t.head_dim)) * t.rope_parameters.partial_rotary_factor);
    return .{
        .hidden = t.hidden_size,
        .layers = t.num_hidden_layers,
        .vocab = t.vocab_size,
        .q_heads = t.num_attention_heads,
        .kv_heads = t.num_key_value_heads,
        .head_dim = t.head_dim,
        .rope_dims = rope,
        .gdn_k_heads = t.linear_num_key_heads,
        .gdn_v_heads = t.linear_num_value_heads,
        .gdn_head_dim = t.linear_key_head_dim,
        .experts = t.num_experts,
        .top_k = t.num_experts_per_tok,
        .expert_ff = t.moe_intermediate_size,
        .shared_ff = t.shared_expert_intermediate_size,
        .mtp = t.mtp_num_hidden_layers == 1,
    };
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Config) {
    return std.json.parseFromSlice(Config, gpa, bytes, .{ .ignore_unknown_fields = true });
}

test "Sage-A1.5 EXL3 config gives the shape the XPU kernels take" {
    const bytes = @embedFile("fixtures/sage_a15_config.json");
    const p = try parse(std.testing.allocator, bytes);
    defer p.deinit();
    const s = try validate(p.value);
    try std.testing.expectEqual(@as(u32, 40), s.layers);
    try std.testing.expectEqual(@as(u32, 64), s.rope_dims);
    try std.testing.expectEqual(@as(u32, 2 * 4096 + 2 * 512), s.attIn());
    try std.testing.expectEqual(@as(u32, 4096 + 8192 + 64), s.gdnIn());
    try std.testing.expectEqual(Kind.attention, kind(p.value.text_config, 3));
    try std.testing.expectEqual(Kind.gdn, kind(p.value.text_config, 4));
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings(sage_defines, try kernelDefines(&buf, s, 4));
}

test "the 27B's shape gives shape.h's defaults" {
    const s: Shape = .{ .hidden = 5120, .layers = 64, .vocab = 248320, .q_heads = 24, .kv_heads = 4, .head_dim = 256, .rope_dims = 64, .gdn_k_heads = 16, .gdn_v_heads = 48, .gdn_head_dim = 128, .experts = 0, .top_k = 0, .expert_ff = 0, .shared_ff = 0, .mtp = false };
    var buf: [1024]u8 = undefined;
    try std.testing.expectEqualStrings("-DQ27_D=5120 -DQ27_D512=10 -DQ27_GK=16 -DQ27_GV=48 -DQ27_GVK=3 -DQ27_NQK=2048 " ++
        "-DQ27_NCONV=10240 -DQ27_VD=6144 -DQ27_GDN_IN=16640 -DQ27_GA=16384 -DQ27_GB=16432 -DQ27_AQ=24 -DQ27_AKV=4 -DQ27_AH=32 " ++
        "-DQ27_AV0=28 -DQ27_GQA=6 -DQ27_QPG=6 -DQ27_QPARTS=1 -DQ27_ATT_IN=14336 -DQ27_AK0=12288 -DQ27_AV0C=13312 " ++
        "-DQ27_OD=6144 -DQ27_OB=48 -DQ27_KVROW=1024", try kernelDefines(&buf, s, 6));
}
