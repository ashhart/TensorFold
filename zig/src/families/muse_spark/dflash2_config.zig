//! RadixArk's DFlash2 drafter for Muse Spark: its config and a rank's matrix cuts (z-lab/dflash semantics).

const std = @import("std");
const config = @import("config.zig");
const Cut = config.Cut;

pub const DConfig = struct {
    hidden: usize, // 8192: the target's
    inter: usize, // 28672 (a rank's under TP: rankView)
    layers: usize, // 5
    heads: usize, // 64 query heads (a rank's under TP)
    kv_heads: usize, // 8
    head_dim: usize, // 128
    eps: f64, // rms_norm_eps: every drafter norm
    theta: f64, // RoPE theta, NeoX over the whole head
    window: usize, // sliding_window: keys q - k < window behind each query, the whole block ahead (is_causal false)
    block: usize, // block_size: the anchor and block - 1 mask rows
    mask_id: usize,
    taps: [8]usize = @splat(0), // target layers whose OUTPUT (residual) feeds the context, increasing
    ntaps: usize = 0,
    conv_k: usize, // conv_kernel_size: taps of each grouped dynamic convolution
    conv_g: usize, // conv_group_size: channels sharing a dynamic weight
    sel_rank: usize, // selector_rank
    sel_k: usize, // selector_top_k (the walk kernel takes 16)
    out_mult: f64, // output_multiplier on the draft logits (the target head's own: applied once)
    softcap: f64, // final_logit_softcapping (likewise)
    vocab: usize,

    pub fn tapList(c: *const DConfig) []const usize {
        return c.taps[0..c.ntaps];
    }

    /// config.json of the drafter (the bf16 release or exllamav3's EXL3 export of it); refuses what is not wired.
    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !DConfig {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const top = parsed.value.object;
        const arch = if (top.get("architectures")) |a| (if (a == .array and a.array.items.len > 0 and a.array.items[0] == .string) a.array.items[0].string else "") else "";
        if (!std.mem.eql(u8, arch, "DFlash2DraftModel")) return error.NotDFlash2;
        const df: ?std.json.ObjectMap = if (top.get("dflash_config")) |v| (if (v == .object) v.object else null) else null;
        var c: DConfig = .{
            .hidden = try need(top, "hidden_size"),
            .inter = try need(top, "intermediate_size"),
            .layers = try need(top, "num_hidden_layers"),
            .heads = try need(top, "num_attention_heads"),
            .kv_heads = try need(top, "num_key_value_heads"),
            .head_dim = int(top, "head_dim") orelse return error.MissingField,
            .eps = num(top.get("rms_norm_eps") orelse return error.MissingField),
            .theta = 0,
            .window = int(top, "sliding_window") orelse 2048,
            .block = try pick(df, top, "block_size"),
            .mask_id = try pick(df, top, "mask_token_id"),
            .conv_k = pickOr(df, top, "conv_kernel_size", 2),
            .conv_g = pickOr(df, top, "conv_group_size", 16),
            .sel_rank = try pick(df, top, "selector_rank"),
            .sel_k = try pick(df, top, "selector_top_k"),
            .out_mult = pickNum(df, top, "output_multiplier", 1.0),
            .softcap = pickNum(df, top, "final_logit_softcapping", 0.0),
            .vocab = try need(top, "vocab_size"),
        };
        c.theta = if (top.get("rope_theta")) |v| num(v) else 0;
        if (top.get("rope_parameters")) |rp| if (rp == .object) {
            if (rp.object.get("rope_theta")) |v| c.theta = num(v);
            if (!std.mem.eql(u8, str(rp.object, "rope_type") orelse "default", "default")) return error.UnsupportedRope;
        };
        if (c.theta == 0) c.theta = 10000.0;
        if (boolean(top, "attention_bias")) return error.Unsupported;
        if (!std.mem.eql(u8, str(top, "hidden_act") orelse "silu", "silu")) return error.UnsupportedActivation;
        // the reference reads is_causal (default: causal on sliding layers); this checkpoint ships false
        const causal = if (top.get("is_causal")) |v| (v == .bool and v.bool) else true;
        if (causal) return error.CausalDrafter;
        const lt = top.get("layer_types") orelse return error.LayerPattern;
        if (lt.array.items.len != c.layers) return error.LayerPattern;
        for (lt.array.items) |t| if (!std.mem.eql(u8, t.string, "sliding_attention")) return error.LayerPattern;
        if (pickNum(df, top, "input_embedding_scale", 1.0) != 1.0) return error.Unsupported;
        if (pickOr(df, top, "tap_shift", 0) != 0) return error.Unsupported; // DFlash2: hidden_states[id + 1] = layer id's output
        const ids = (if (df) |d| d.get("target_layer_ids") else null) orelse top.get("target_layer_ids") orelse return error.MissingField;
        for (ids.array.items) |v| {
            if (c.ntaps == c.taps.len) return error.TooManyTaps;
            c.taps[c.ntaps] = @intCast(v.integer);
            if (c.ntaps > 0 and c.taps[c.ntaps] <= c.taps[c.ntaps - 1]) return error.TapsOutOfOrder;
            c.ntaps += 1;
        }
        if (c.head_dim != 128) return error.HeadDim; // dflash.cu's norm/rope and attention: 128 threads a head
        if (c.heads % c.kv_heads != 0 or c.heads / c.kv_heads > 8) return error.GroupWiderThan8;
        if (c.sel_k != 16) return error.SelectorTopK; // dfl_walk: a warp a candidate, 16 warps
        if (c.block < 2 or c.block > 16) return error.BlockSize;
        if (c.conv_k == 0 or c.conv_k > c.block or c.conv_k > 4) return error.ConvKernel;
        if (c.hidden % c.conv_g != 0) return error.ConvGroups;
        if (c.sel_rank % 32 != 0 or c.sel_rank > 1024) return error.SelectorRank;
        return c;
    }

    /// A rank's view at two ranks: query / KV heads and the MLP width split (the context projection fc by outputs).
    pub fn rankView(c: DConfig, world: usize) !DConfig {
        if (world != 2) return error.MuseSparkRequiresTwoRanks;
        if (c.heads % world != 0 or c.kv_heads % world != 0) return error.HeadsSplit;
        if ((c.kv_heads / world) * c.head_dim % 128 != 0) return error.HeadsSplit;
        if (c.inter % (128 * world) != 0 or c.hidden % (128 * world) != 0) return error.WidthSplit;
        var r = c;
        r.heads = c.heads / world;
        r.kv_heads = c.kv_heads / world;
        r.inter = c.inter / world;
        return r;
    }
};

/// The drafter's matrices: fc, then per layer q, k, v, o, gate, up, down, the two convolutions' kernel projections.
pub const Mat = enum { q, k, v, o, gate, up, down, aconv, mconv };
pub const mat_names = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "attention_conv.kernel_projection", "mlp_conv.kernel_projection" };

/// A layer matrix's cut for `rank`: q / o by heads, k / v by KV heads, the MLP by width; kernel projections whole.
pub fn layerCut(c: DConfig, mat: Mat, prefix: []const u8, rank: usize, world: usize) Cut {
    if (world == 1) return .{ .prefix = prefix };
    const hq = c.heads / world * c.head_dim;
    const hk = c.kv_heads / world * c.head_dim;
    const wi = c.inter / world;
    const r = rank;
    return switch (mat) {
        .q => .{ .prefix = prefix, .axis = .n, .lo = r * hq, .hi = (r + 1) * hq },
        .k, .v => .{ .prefix = prefix, .axis = .n, .lo = r * hk, .hi = (r + 1) * hk },
        .o => .{ .prefix = prefix, .axis = .k, .lo = r * hq, .hi = (r + 1) * hq },
        .gate, .up => .{ .prefix = prefix, .axis = .n, .lo = r * wi, .hi = (r + 1) * wi },
        .down => .{ .prefix = prefix, .axis = .k, .lo = r * wi, .hi = (r + 1) * wi },
        .aconv, .mconv => .{ .prefix = prefix },
    };
}

/// fc (taps -> hidden) by outputs: each rank projects its half of the context row; the halves are gathered (fp16).
pub fn fcCut(c: DConfig, rank: usize, world: usize) Cut {
    if (world == 1) return .{ .prefix = "fc" };
    const w = c.hidden / world;
    return .{ .prefix = "fc", .axis = .n, .lo = rank * w, .hi = (rank + 1) * w };
}

/// Weight bytes a draft pass reads on one rank (bpw for the EXL3 matrices; the head and the context update excluded).
pub fn passBytes(c: DConfig, world: usize, bpw: f64) f64 {
    const H = @as(f64, @floatFromInt(c.hidden));
    const q = H * @as(f64, @floatFromInt(c.heads * c.head_dim));
    const kv = 2 * H * @as(f64, @floatFromInt(c.kv_heads * c.head_dim));
    const mlp = 3 * H * @as(f64, @floatFromInt(c.inter));
    const conv = 2 * H * @as(f64, @floatFromInt(2 * c.conv_k * (c.hidden / c.conv_g)));
    const w: f64 = @floatFromInt(world);
    return @as(f64, @floatFromInt(c.layers)) * ((2 * q + kv + mlp) / w + conv) * bpw / 8;
}

fn str(o: std.json.ObjectMap, k: []const u8) ?[]const u8 {
    const v = o.get(k) orelse return null;
    return if (v == .string) v.string else null;
}
fn int(o: std.json.ObjectMap, k: []const u8) ?usize {
    const v = o.get(k) orelse return null;
    return if (v == .integer) @intCast(v.integer) else null;
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
fn boolean(o: std.json.ObjectMap, k: []const u8) bool {
    const v = o.get(k) orelse return false;
    return v == .bool and v.bool;
}
/// dflash_config->key first, then the top level (the reference's _draft_value).
fn pickOr(df: ?std.json.ObjectMap, top: std.json.ObjectMap, k: []const u8, default: usize) usize {
    if (df) |d| if (int(d, k)) |v| return v;
    return int(top, k) orelse default;
}
fn pick(df: ?std.json.ObjectMap, top: std.json.ObjectMap, k: []const u8) !usize {
    if (df) |d| if (int(d, k)) |v| return v;
    return int(top, k) orelse error.MissingField;
}
fn pickNum(df: ?std.json.ObjectMap, top: std.json.ObjectMap, k: []const u8, default: f64) f64 {
    if (df) |d| if (d.get(k)) |v| return num(v);
    return if (top.get(k)) |v| num(v) else default;
}

// RadixArk/Muse-Spark-Open-DFlash2 @ 46c90f41's config.json (the fields parse reads).
pub const fixture =
    \\{"architectures": ["DFlash2DraftModel"], "attention_bias": false, "dflash_config": {"attention_mode": "gqa",
    \\ "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "final_logit_softcapping": 20, "mask_token_id": 200002,
    \\ "output_multiplier": 0.1767766952966369, "selector_rank": 256, "selector_top_k": 16,
    \\ "target_layer_ids": [1, 16, 30, 44, 59]}, "head_dim": 128, "hidden_act": "silu", "hidden_size": 8192,
    \\ "intermediate_size": 28672, "is_causal": false, "layer_types": ["sliding_attention", "sliding_attention",
    \\ "sliding_attention", "sliding_attention", "sliding_attention"], "num_attention_heads": 64, "num_hidden_layers": 5,
    \\ "num_key_value_heads": 8, "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 500000, "rope_type": "default"},
    \\ "rope_theta": 500000, "sliding_window": 2048, "vocab_size": 202048}
;

test "DConfig.parse reads RadixArk's Muse Spark DFlash2" {
    const c = try DConfig.parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(@as(usize, 5), c.layers);
    try std.testing.expectEqual(@as(usize, 8), c.block);
    try std.testing.expectEqual(@as(usize, 200002), c.mask_id);
    try std.testing.expectEqualSlices(usize, &.{ 1, 16, 30, 44, 59 }, c.tapList());
    try std.testing.expectEqual(@as(f64, 500000), c.theta);
    try std.testing.expectEqual(@as(usize, 2048), c.window);
    try std.testing.expectEqual(@as(usize, 2), c.conv_k);
    try std.testing.expectEqual(@as(f64, 20), c.softcap);
    try std.testing.expectApproxEqRel(@as(f64, 0.17677669529663687), c.out_mult, 1e-12);
    try std.testing.expectError(error.MuseSparkRequiresTwoRanks, c.rankView(1));
    const r = try c.rankView(2);
    try std.testing.expectEqual(@as(usize, 32), r.heads);
    try std.testing.expectEqual(@as(usize, 4), r.kv_heads);
    try std.testing.expectEqual(@as(usize, 14336), r.inter);
}

test "DConfig.parse refuses what the drafter does not wire" {
    const a = std.testing.allocator;
    const bad = [_][2][]const u8{
        .{ "\"is_causal\": false", "\"is_causal\": true" },
        .{ "\"selector_top_k\": 16", "\"selector_top_k\": 8" },
        .{ "\"head_dim\": 128", "\"head_dim\": 64" },
        .{ "\"DFlash2DraftModel\"", "\"DFlashDraftModel\"" },
        .{ "[1, 16, 30, 44, 59]", "[16, 1, 30, 44, 59]" },
        .{ "\"sliding_attention\", \"sliding_attention\"]", "\"sliding_attention\", \"full_attention\"]" },
    };
    for (bad) |b| {
        const t = try std.mem.replaceOwned(u8, a, fixture, b[0], b[1]);
        defer a.free(t);
        try std.testing.expect(std.meta.isError(DConfig.parse(a, t)));
    }
}

test "drafter cuts at TP=2 sit on 128-column blocks" {
    const c = try DConfig.parse(std.testing.allocator, fixture);
    const q1 = layerCut(c, .q, "layers.0.self_attn.q_proj", 1, 2);
    try std.testing.expectEqual(@as(usize, 4096), q1.lo);
    try std.testing.expectEqual(@as(usize, 8192), q1.hi);
    const k1 = layerCut(c, .k, "x", 1, 2);
    try std.testing.expectEqual(@as(usize, 512), k1.lo);
    try std.testing.expect(layerCut(c, .o, "x", 0, 2).axis == .k);
    const d1 = layerCut(c, .down, "x", 1, 2);
    try std.testing.expectEqual(@as(usize, 14336), d1.lo);
    try std.testing.expect(d1.axis == .k);
    try std.testing.expect(layerCut(c, .aconv, "x", 1, 2).axis == .whole);
    try std.testing.expectEqual(@as(usize, 4096), fcCut(c, 1, 2).lo);
    for (std.enums.values(Mat)) |m| {
        const cc = layerCut(c, m, "x", 1, 2);
        try std.testing.expect(cc.lo % 128 == 0 and cc.hi % 128 == 0);
    }
    try std.testing.expect(layerCut(c, .q, "x", 0, 1).axis == .whole);
    // a draft pass reads about half the drafter a rank when split
    const split = passBytes(c, 2, 4.0) / 1e9;
    const repl = passBytes(c, 1, 4.0) / 1e9;
    try std.testing.expect(split > 1.0 and split < 1.3);
    try std.testing.expect(repl > 2.0 and repl < 2.4);
}
