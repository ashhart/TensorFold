//! Muse Spark Open's checkpoint facts (text model) and a rank's EXL3 cuts, as exllamav3's muse_spark reads them.

const std = @import("std");

pub const model_type = "muse_spark";
pub const text_type = "muse_spark_text";
pub const max_layers = 64;
/// The tensors' prefix in the pack: the text model of the multimodal checkpoint.
pub const prefix = "model.language_model.";

pub const Config = struct {
    layers: usize,
    hidden: usize, // 8192
    heads: usize, // query heads; a rank's under TP
    kv_heads: usize,
    head_dim: usize, // 64
    window: usize, // past keys a sliding row sees: sliding_window - 1 (the reference mask counts the row itself)
    full: [max_layers]bool = @splat(false), // full attention, else sliding
    theta: [max_layers]f64 = @splat(0), // RoPE theta of the layer, 0 = NoPE
    latent: usize, // moe_hidden_size: the experts' model width (4096)
    width: usize, // moe_intermediate_size: an expert's width; a rank's under TP
    experts: usize,
    top_k: usize,
    vocab: usize, // real tokens; the head's rows are padded to a multiple of 128
    eps: f64, // rms_norm_eps: input, pre-ffn, pre-expert, qk, o and final norms
    post_eps: f64, // post_norm_eps: post-attention, post-ffn and post-expert norms
    qk_scale: f64, // qk_scale_factor: folds into the softmax scale
    out_mult: f64, // output_multiplier: router logits and head pre-scale
    softcap: f64, // final_logit_softcapping, 0 = none
    gate_temp: f64, // residual_gate_temperature
    route_scale: f64, // routed_scaling_factor
    eos: [4]u32 = @splat(0),
    eos_len: usize = 0,

    pub fn isEos(c: Config, t: u32) bool {
        return std.mem.indexOfScalar(u32, c.eos[0..c.eos_len], t) != null;
    }
    pub fn rope(c: Config, i: usize) bool {
        return c.theta[i] != 0;
    }
    /// The softmax scale: qk_scale_factor * head_dim ** -0.5 (exllamav3's sm_scale), as a Python float.
    pub fn smScale(c: Config) f64 {
        return c.qk_scale * std.math.pow(f64, @floatFromInt(c.head_dim), -0.5);
    }

    /// config.json (+ generation_config.json's EOS ids); refuses every variant the forward does not wire.
    pub fn parse(gpa: std.mem.Allocator, config_json: []const u8, generation_json: ?[]const u8) !Config {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
        defer parsed.deinit();
        const top = parsed.value.object;
        if (!std.mem.eql(u8, str(top, "model_type") orelse "", model_type)) return error.NotMuseSpark;
        const q = if (top.get("quantization_config")) |v| v.object else return error.NotExl3;
        if (!std.mem.eql(u8, str(q, "quant_method") orelse "", "exl3")) return error.NotExl3;
        if (!std.mem.eql(u8, str(q, "codebook") orelse "mul1", "mul1")) return error.NotMul1;
        const o = (top.get("text_config") orelse return error.MissingField).object;
        if (!std.mem.eql(u8, str(o, "model_type") orelse text_type, text_type)) return error.NotMuseSpark;
        if (boolean(o, "attention_bias") or boolean(o, "tie_word_embeddings")) return error.Unsupported;
        if (!std.mem.eql(u8, str(o, "hidden_act") orelse "silu", "silu")) return error.UnsupportedActivation;
        if ((int(o, "num_shared_experts") orelse 0) != 0) return error.UnsupportedRouting;
        if (o.get("rope_parameters")) |rp| if (rp == .object) {
            if (!std.mem.eql(u8, str(rp.object, "rope_type") orelse "default", "default")) return error.UnsupportedRope;
        };
        var c: Config = .{
            .layers = try need(o, "num_hidden_layers"),
            .hidden = try need(o, "hidden_size"),
            .heads = try need(o, "num_attention_heads"),
            .kv_heads = try need(o, "num_key_value_heads"),
            .head_dim = try need(o, "head_dim"),
            .window = (try need(o, "sliding_window")) - 1,
            .latent = try need(o, "moe_hidden_size"),
            .width = try need(o, "moe_intermediate_size"),
            .experts = try need(o, "num_local_experts"),
            .top_k = try need(o, "num_experts_per_tok"),
            .vocab = try need(o, "vocab_size"),
            .eps = try numNeed(o, "rms_norm_eps"),
            .post_eps = if (o.get("post_norm_eps")) |v| num(v) else try numNeed(o, "rms_norm_eps"),
            .qk_scale = if (o.get("qk_scale_factor")) |v| num(v) else 1.0,
            .out_mult = if (o.get("output_multiplier")) |v| num(v) else 1.0,
            .softcap = if (o.get("final_logit_softcapping")) |v| num(v) else 0.0,
            .gate_temp = try numNeed(o, "residual_gate_temperature"),
            .route_scale = if (o.get("routed_scaling_factor")) |v| num(v) else 1.0,
        };
        if (c.layers > max_layers) return error.TooManyLayers;
        if (c.head_dim != 64) return error.HeadDim; // the attention kernels: a lane holds dims l and l + 32
        if (c.heads % c.kv_heads != 0 or c.heads / c.kv_heads != 8) return error.GroupSize; // a warp a query head of the group
        if (c.top_k > 32 or c.experts > 1024) return error.UnsupportedRouting;
        const types = (o.get("layer_types") orelse return error.MissingField).array.items;
        if (types.len != c.layers) return error.LayerPattern;
        if (o.get("mlp_layer_types")) |m| for (m.array.items) |t| {
            if (!std.mem.eql(u8, t.string, "sparse")) return error.DenseMlp;
        };
        var default_theta: f64 = 500000.0;
        if (o.get("rope_parameters")) |rp| if (rp == .object) if (rp.object.get("rope_theta")) |v| {
            default_theta = num(v);
        };
        const thetas: ?[]std.json.Value = if (o.get("layer_rope_theta")) |v| v.array.items else null;
        if (thetas != null and thetas.?.len != c.layers) return error.LayerPattern;
        for (types, 0..) |t, i| {
            if (std.mem.eql(u8, t.string, "full_attention")) {
                c.full[i] = true;
            } else if (!std.mem.eql(u8, t.string, "sliding_attention")) return error.LayerPattern;
            c.theta[i] = if (thetas) |th| num(th[i]) else default_theta;
        }
        if (o.get("eos_token_id")) |v| c.addEos(v);
        if (generation_json) |g| {
            const gp = try std.json.parseFromSlice(std.json.Value, gpa, g, .{});
            defer gp.deinit();
            if (gp.value.object.get("eos_token_id")) |v| c.addEos(v);
        }
        return c;
    }

    fn addEos(c: *Config, v: std.json.Value) void {
        switch (v) {
            .integer => |i| c.pushEos(@intCast(i)),
            .array => |a| for (a.items) |x| if (x == .integer) c.pushEos(@intCast(x.integer)),
            else => {},
        }
    }
    fn pushEos(c: *Config, t: u32) void {
        if (c.isEos(t) or c.eos_len == c.eos.len) return;
        c.eos[c.eos_len] = t;
        c.eos_len += 1;
    }

    /// A rank's view under TP: heads, KV heads and the expert width split; the latent stays whole.
    pub fn rankView(c: Config, world: usize) !Config {
        if (world != 1 and world != 2) return error.World;
        if (c.heads % world != 0 or c.kv_heads % world != 0) return error.HeadsSplit;
        if (c.width % (128 * world) != 0 or c.latent % (128 * world) != 0) return error.WidthSplit;
        if ((c.heads / world) * c.head_dim % 128 != 0 or (c.kv_heads / world) * c.head_dim % 128 != 0) return error.HeadsSplit;
        var r = c;
        r.heads = c.heads / world;
        r.kv_heads = c.kv_heads / world;
        r.width = c.width / world;
        return r;
    }
};

/// One EXL3 matrix a rank loads: its tensors' prefix and the cut: outputs (`n`) or inputs (`k`).
pub const Cut = struct {
    prefix: []const u8,
    axis: enum { whole, n, k } = .whole,
    lo: usize = 0,
    hi: usize = 0,
};

/// The layer's matrices in load order: q, k, v, gate, o, pre_expert, post_expert, then (gate, up, down) per expert.
pub const attn_mats = 5;
pub const latent_mats = 2;

/// Every EXL3 matrix of layer `i` a rank loads, by heads, latent halves and expert width; `buf` holds the prefixes.
pub fn layerCuts(full: Config, i: usize, rank: usize, world: usize, buf: *std.ArrayList(u8), gpa: std.mem.Allocator, out: *std.ArrayList(Cut)) !void {
    const hq = full.heads / world * full.head_dim;
    const hk = full.kv_heads / world * full.head_dim;
    const lat = full.latent / world;
    const wid = full.width / world;
    const r = rank;
    const one = struct {
        fn f(g: std.mem.Allocator, b: *std.ArrayList(u8), o: *std.ArrayList(Cut), comptime fmt: []const u8, args: anytype, axis: @FieldType(Cut, "axis"), lo: usize, hi: usize, w: usize) !void {
            const start = b.items.len;
            try b.print(g, prefix ++ fmt, args);
            try o.append(g, .{ .prefix = b.items[start..], .axis = if (w == 1) .whole else axis, .lo = lo, .hi = hi });
        }
    }.f;
    try one(gpa, buf, out, "layers.{d}.self_attn.q_proj", .{i}, .n, r * hq, (r + 1) * hq, world);
    try one(gpa, buf, out, "layers.{d}.self_attn.k_proj", .{i}, .n, r * hk, (r + 1) * hk, world);
    try one(gpa, buf, out, "layers.{d}.self_attn.v_proj", .{i}, .n, r * hk, (r + 1) * hk, world);
    try one(gpa, buf, out, "layers.{d}.self_attn.gate_proj", .{i}, .n, r * hq, (r + 1) * hq, world);
    try one(gpa, buf, out, "layers.{d}.self_attn.o_proj", .{i}, .k, r * hq, (r + 1) * hq, world);
    try one(gpa, buf, out, "layers.{d}.mlp.pre_expert_proj", .{i}, .n, r * lat, (r + 1) * lat, world);
    try one(gpa, buf, out, "layers.{d}.mlp.post_expert_proj", .{i}, .k, r * lat, (r + 1) * lat, world);
    for (0..full.experts) |e| {
        try one(gpa, buf, out, "layers.{d}.mlp.experts.{d}.gate_proj", .{ i, e }, .n, r * wid, (r + 1) * wid, world);
        try one(gpa, buf, out, "layers.{d}.mlp.experts.{d}.up_proj", .{ i, e }, .n, r * wid, (r + 1) * wid, world);
        try one(gpa, buf, out, "layers.{d}.mlp.experts.{d}.down_proj", .{ i, e }, .k, r * wid, (r + 1) * wid, world);
    }
}

/// lm_head's rows of a rank in 128-row blocks of the padded head (1579 blocks: 790 + 789 at TP=2).
pub fn headCut(n: usize, rank: usize, world: usize) Cut {
    if (world == 1) return .{ .prefix = "lm_head", .lo = 0, .hi = n };
    const blocks = n / 128;
    const first = (blocks + 1) / 2;
    const lo = if (rank == 0) 0 else first * 128;
    const hi = if (rank == 0) first * 128 else n;
    return .{ .prefix = "lm_head", .axis = .n, .lo = lo, .hi = hi };
}

/// Columns a rank's head slice occupies in the gather buffer: the widest rank's.
pub fn headStride(n: usize, world: usize) usize {
    return if (world == 1) n else ((n / 128 + 1) / 2) * 128;
}

fn str(o: std.json.ObjectMap, k: []const u8) ?[]const u8 {
    const v = o.get(k) orelse return null;
    return if (v == .string) v.string else null;
}
fn int(o: std.json.ObjectMap, k: []const u8) ?usize {
    const v = o.get(k) orelse return null;
    return switch (v) {
        .integer => |i| @intCast(i),
        else => null,
    };
}
fn need(o: std.json.ObjectMap, k: []const u8) !usize {
    return int(o, k) orelse error.MissingField;
}
fn num(v: std.json.Value) f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |s| std.fmt.parseFloat(f64, s) catch 0,
        else => 0,
    };
}
fn numNeed(o: std.json.ObjectMap, k: []const u8) !f64 {
    return num(o.get(k) orelse return error.MissingField);
}
fn boolean(o: std.json.ObjectMap, k: []const u8) bool {
    const v = o.get(k) orelse return false;
    return v == .bool and v.bool;
}

// The fields Config.parse reads, as the 2-layer dry-run pack's config.json has them (text_config of Muse Spark Open).
const fixture =
    \\{"model_type": "muse_spark", "quantization_config": {"quant_method": "exl3", "codebook": "mul1", "bits": 2.04},
    \\ "text_config": {"attention_bias": false, "eos_token_id": 200001, "final_logit_softcapping": 20.0, "head_dim": 64,
    \\ "hidden_act": "silu", "hidden_size": 8192, "intermediate_size": 16384, "layer_rope_theta": [500000.0, 0],
    \\ "layer_types": ["sliding_attention", "full_attention"], "mlp_layer_types": ["sparse", "sparse"],
    \\ "model_type": "muse_spark_text", "moe_hidden_size": 4096, "moe_intermediate_size": 4096,
    \\ "num_attention_heads": 128, "num_experts_per_tok": 8, "num_hidden_layers": 2, "num_key_value_heads": 16,
    \\ "num_local_experts": 256, "num_shared_experts": 0, "output_multiplier": 0.17677669529663687, "post_norm_eps": 1e-08,
    \\ "qk_scale_factor": 5.4730064863875, "residual_gate_temperature": 0.3, "rms_norm_eps": 1e-05,
    \\ "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}, "routed_scaling_factor": 1.0,
    \\ "sliding_window": 2048, "tie_word_embeddings": false, "vocab_size": 202048}}
;

test "Config.parse reads the 2-layer pack as exllamav3's MuseSparkConfig does" {
    const c = try Config.parse(std.testing.allocator, fixture, "{\"eos_token_id\": [200001, 200007]}");
    try std.testing.expectEqual(@as(usize, 2), c.layers);
    try std.testing.expectEqual(@as(usize, 2047), c.window);
    try std.testing.expect(!c.full[0] and c.full[1]);
    try std.testing.expect(c.rope(0) and !c.rope(1));
    try std.testing.expectEqual(@as(f64, 500000.0), c.theta[0]);
    try std.testing.expectEqual(@as(f64, 1e-8), c.post_eps);
    try std.testing.expectEqualSlices(u32, &.{ 200001, 200007 }, c.eos[0..c.eos_len]);
    try std.testing.expectApproxEqRel(@as(f64, 5.4730064863875 / 8.0), c.smScale(), 1e-15);
    const r = try c.rankView(2);
    try std.testing.expectEqual(@as(usize, 64), r.heads);
    try std.testing.expectEqual(@as(usize, 8), r.kv_heads);
    try std.testing.expectEqual(@as(usize, 2048), r.width);
    try std.testing.expectEqual(@as(usize, 4096), r.latent);
}

test "Config.parse refuses what the forward does not wire" {
    const a = std.testing.allocator;
    const bad = [_][2][]const u8{
        .{ "\"num_shared_experts\": 0", "\"num_shared_experts\": 1" },
        .{ "\"attention_bias\": false", "\"attention_bias\": true" },
        .{ "\"mlp_layer_types\": [\"sparse\", \"sparse\"]", "\"mlp_layer_types\": [\"dense\", \"sparse\"]" },
        .{ "\"head_dim\": 64", "\"head_dim\": 128" },
        .{ "\"quant_method\": \"exl3\"", "\"quant_method\": \"fp8\"" },
    };
    for (bad) |b| {
        const t = try std.mem.replaceOwned(u8, a, fixture, b[0], b[1]);
        defer a.free(t);
        try std.testing.expect(std.meta.isError(Config.parse(a, t, null)));
    }
}

test "rank cuts" {
    const a = std.testing.allocator;
    const c = try Config.parse(a, fixture, null);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try buf.ensureTotalCapacity(a, 1 << 20); // prefixes slice into it: no reallocation
    var cuts: std.ArrayList(Cut) = .empty;
    defer cuts.deinit(a);
    try layerCuts(c, 1, 1, 2, &buf, a, &cuts);
    try std.testing.expectEqual(@as(usize, attn_mats + latent_mats + 3 * 256), cuts.items.len);
    try std.testing.expectEqualStrings("model.language_model.layers.1.self_attn.q_proj", cuts.items[0].prefix);
    try std.testing.expectEqual(@as(usize, 64 * 64), cuts.items[0].lo); // q: heads 64..127
    try std.testing.expectEqual(@as(usize, 8 * 64), cuts.items[1].lo); // k: KV heads 8..15
    try std.testing.expectEqual(@as(usize, 16 * 64), cuts.items[1].hi);
    try std.testing.expect(cuts.items[4].axis == .k); // o: input columns of its heads
    try std.testing.expectEqual(@as(usize, 2048), cuts.items[5].lo); // pre_expert_proj: latent outputs 2048..4095
    try std.testing.expect(cuts.items[6].axis == .k); // post_expert_proj: latent inputs
    try std.testing.expectEqual(@as(usize, 2048), cuts.items[7].lo); // expert width 4096 / 2
    try std.testing.expectEqualStrings("model.language_model.layers.1.mlp.experts.255.down_proj", cuts.items[cuts.items.len - 1].prefix);
    cuts.clearRetainingCapacity();
    try layerCuts(c, 0, 0, 1, &buf, a, &cuts);
    try std.testing.expect(cuts.items[0].axis == .whole);
    const h0 = headCut(202112, 0, 2);
    const h1 = headCut(202112, 1, 2);
    try std.testing.expectEqual(@as(usize, 101120), h0.hi);
    try std.testing.expectEqual(@as(usize, 101120), h1.lo);
    try std.testing.expectEqual(@as(usize, 202112), h1.hi);
    try std.testing.expectEqual(@as(usize, 101120), headStride(202112, 2));
}
