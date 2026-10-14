//! Flash Next's tensors as the HIP loader reads them: names, shapes and centred norm storage.
const std = @import("std");
const Config = @import("../config.zig").Config;

/// Every text tensor's prefix (the vision tower's sit outside it).
pub const root = "language_model.";
pub const embed = "language_model.model.embed_tokens";
pub const lm_head = "language_model.lm_head";
pub const mixer = "language_model.model.hyper_connection_mixer";

/// Decoder layer `i`'s stem.
pub fn layer(buf: []u8, i: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "language_model.model.layers.{d}", .{i});
}

/// The MTP head's tensors (one decoder layer, `layer0`).
pub const mtp = struct {
    pub const stem = "language_model.mtp";
    pub const layer0 = "language_model.mtp.layers.0";
    pub const fc_e = "fc_embedding";
    pub const fc_h = "fc_hidden";
    pub const fc_embedding = stem ++ "." ++ fc_e;
    pub const fc_hidden = stem ++ "." ++ fc_h;
    pub const norm_embedding = "language_model.mtp.pre_fc_norm_embedding.weight";
    pub const norm_hidden = "language_model.mtp.pre_fc_norm_hidden.weight";
    pub const mixer = "language_model.mtp.hyper_connection_mixer";
};

/// A hyper-connection under a layer's stem (`attn` or `mlp`) or the mixers: its norm and projections.
pub const hc = struct {
    pub const attn = "attn_hyper_connection";
    pub const mlp = "mlp_hyper_connection";
    pub const norm = "hc_norm.weight";
    pub const down = "input_mix_weight_down";
    pub const up = "input_mix_weight_up";
    pub const inject = "block_inject_weight";
};

/// The MoE block under a layer's stem: the router, the shared expert's gate and projections, the routed experts.
pub const moe = struct {
    pub const prefix = "mlp.";
    pub const gate = "mlp.gate";
    pub const shared_gate = "mlp.shared_expert_gate";
    pub const shared = [_][]const u8{
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
        "mlp.shared_expert.down_proj",
    };
    pub const routed = [_][]const u8{
        "mlp.switch_mlp.gate_proj",
        "mlp.switch_mlp.up_proj",
        "mlp.switch_mlp.down_proj",
    };
};

/// The Gated DeltaNet block under a layer's stem.
pub const linear = struct {
    pub const qkv = "linear_attn.in_proj_qkv";
    pub const z = "linear_attn.in_proj_z";
    pub const b = "linear_attn.in_proj_b";
    pub const a = "linear_attn.in_proj_a";
    pub const out = "linear_attn.out_proj";
    pub const a_log = "linear_attn.A_log";
    pub const dt_bias = "linear_attn.dt_bias";
    pub const conv = "linear_attn.conv1d.weight";
    pub const norm = "linear_attn.norm.weight";
    /// The input projections in the order they share one product.
    pub const inputs = [_][]const u8{ qkv, z, b, a };
};

/// The sparse attention block under a layer's stem, with its indexer.
pub const sparse = struct {
    pub const q = "self_attn.q_proj";
    pub const k = "self_attn.k_proj";
    pub const v = "self_attn.v_proj";
    pub const o = "self_attn.o_proj";
    pub const index = "self_attn.indexer.index_qk_proj";
    pub const q_norm = "self_attn.q_norm.weight";
    pub const k_norm = "self_attn.k_norm.weight";
    pub const iq_norm = "self_attn.indexer.q_layernorm.weight";
    pub const ik_norm = "self_attn.indexer.k_layernorm.weight";
    /// The input projections in the order they share one product.
    pub const inputs = [_][]const u8{ q, k, v, index };
};

/// The n-gram (PLE) block under its layer's stem.
pub const ple = struct {
    pub const key = "ple.key_proj";
    pub const value = "ple.value_proj";
    pub const norm_key = "ple.norm_key.weight";
    pub const norm_query = "ple.norm_query.weight";
    pub const norm_conv = "ple.norm_conv.weight";
    pub const conv = "ple.conv1d.weight";
    pub const vectors = [_][]const u8{
        conv,
        norm_conv,
        norm_key,
        norm_query,
        "ple.ple_embedding.layer_multipliers",
        "ple.ple_embedding.ngram_heads_offsets",
        "ple.ple_embedding.ngram_heads_vocab_sizes",
    };

    /// Shard `i` of the tables under `spelling` (`shard_` or `shards.`), without its `.weight`/`.scales`/`.biases`.
    pub fn table(buf: []u8, layer_stem: []const u8, spelling: []const u8, i: usize) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}.ple.ple_embedding.ngram_embedding.{s}{d}", .{ layer_stem, spelling, i });
    }
};

/// A projection's output rows and input width.
pub const Shape = struct { n: usize, k: usize };

/// The shape of the projection named `suffix` (one of the names above), or null for a name this table does not size.
pub fn shape(c: *const Config, suffix: []const u8) !?Shape {
    const wide = try c.wide();
    const kw = c.key_heads * c.key_dim;
    const vw = c.value_heads * c.value_dim;
    const table = [_]struct { []const u8, Shape }{
        .{ hc.down, .{ .n = c.hc_lowrank, .k = wide } },
        .{ hc.up, .{ .n = wide, .k = c.hc_lowrank } },
        .{ hc.inject, .{ .n = c.hc_count, .k = wide } },
        .{ linear.qkv, .{ .n = 2 * kw + vw, .k = c.hidden } },
        .{ linear.z, .{ .n = vw, .k = c.hidden } },
        .{ linear.a, .{ .n = c.value_heads, .k = c.hidden } },
        .{ linear.b, .{ .n = c.value_heads, .k = c.hidden } },
        .{ linear.out, .{ .n = c.hidden, .k = vw } },
        .{ sparse.q, .{ .n = 2 * c.heads * c.head_dim, .k = c.hidden } },
        .{ sparse.k, .{ .n = c.kv_heads * c.head_dim, .k = c.hidden } },
        .{ sparse.v, .{ .n = c.kv_heads * c.head_dim, .k = c.hidden } },
        .{ sparse.o, .{ .n = c.hidden, .k = c.heads * c.head_dim } },
        .{ sparse.index, .{ .n = (c.index_heads + c.index_kv_heads) * c.index_dim, .k = c.hidden } },
        .{ ple.key, .{ .n = wide, .k = c.ple_dim } },
        .{ ple.value, .{ .n = c.hidden, .k = c.ple_dim } },
        .{ embed, .{ .n = c.vocab, .k = c.hidden } },
        .{ lm_head, .{ .n = c.vocab, .k = c.hidden } },
        .{ mtp.fc_embedding, .{ .n = c.hidden, .k = c.hidden } },
        .{ mtp.fc_hidden, .{ .n = c.hidden, .k = c.hidden } },
    };
    for (table) |e| if (std.mem.endsWith(u8, suffix, e[0])) return e[1];
    return null;
}

/// Gamma (true) or gamma - 1 stored, from the layers' attention hc norm means: under 8 say zero, a mix is refused.
pub fn normsAroundOne(means: []f64) !bool {
    if (means.len < 8) return false;
    var above: usize = 0;
    for (means) |m| {
        if (m > 0.5) above += 1;
    }
    std.mem.sort(f64, means, {}, std.sort.asc(f64));
    const median = means[means.len / 2];
    const share = @as(f64, @floatFromInt(above)) / @as(f64, @floatFromInt(means.len));
    const around_one = share >= 0.9 and 0.75 <= median and median <= 1.5;
    const around_zero = share <= 0.1 and -0.5 <= median and median <= 0.25;
    if (!around_one and !around_zero) return error.AmbiguousNormStorage;
    return around_one;
}

/// A centred norm's scale from its stored value: 1 + w, w the stored value less one when gamma is stored.
pub fn centred(stored: f32, around_one: bool) f32 {
    const w = if (around_one) stored - 1.0 else stored;
    return 1.0 + w;
}

test "every sized projection of the 4-bit checkpoint has the shape its config gives" {
    var c = try @import("../config.zig").parse(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer c.deinit();
    // the published checkpoint's headers: rows and inputs of each projection
    const want = [_]struct { []const u8, Shape }{
        .{ hc.down, .{ .n = 320, .k = 10240 } },   .{ hc.up, .{ .n = 10240, .k = 320 } },
        .{ hc.inject, .{ .n = 4, .k = 10240 } },
        .{
            linear.qkv,
            .{ .n = 10240, .k = 2560 },
        },
        .{ linear.z, .{ .n = 6144, .k = 2560 } },  .{ linear.a, .{ .n = 48, .k = 2560 } },
        .{
            linear.out,
            .{ .n = 2560, .k = 6144 },
        },
        .{ sparse.q, .{ .n = 12288, .k = 2560 } }, .{ sparse.k, .{ .n = 512, .k = 2560 } },
        .{ sparse.o, .{ .n = 2560, .k = 6144 } },
        .{
            sparse.index,
            .{ .n = 640, .k = 2560 },
        },
        .{ ple.key, .{ .n = 10240, .k = 2560 } },  .{ ple.value, .{ .n = 2560, .k = 2560 } },
        .{ lm_head, .{ .n = 248320, .k = 2560 } },
        .{
            mtp.fc_hidden,
            .{ .n = 2560, .k = 2560 },
        },
    };
    for (want) |w| try std.testing.expectEqual(w[1], (try shape(&c, w[0])).?);
    try std.testing.expect((try shape(&c, moe.gate)) == null);
}

test "norm storage: gamma near one, gamma - 1 near zero, a mix refused, few layers around zero" {
    var one = [_]f64{ 1.0, 0.9, 1.1, 1.0, 1.2, 0.95, 1.0, 1.05 };
    try std.testing.expect(try normsAroundOne(&one));
    var zero = [_]f64{ 0.0, 0.1, -0.1, 0.0, 0.2, 0.05, 0.0, 0.02 };
    try std.testing.expect(!try normsAroundOne(&zero));
    var mixed = [_]f64{ 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 };
    try std.testing.expectError(error.AmbiguousNormStorage, normsAroundOne(&mixed));
    var few = [_]f64{ 1.0, 1.0 };
    try std.testing.expect(!try normsAroundOne(&few));
    try std.testing.expectEqual(@as(f32, 2.25), centred(1.25, false));
    try std.testing.expectEqual(@as(f32, 1.25), centred(1.25, true));
}
