//! MiMo-V2's checkpoint facts and a rank's share of its weights, as the Python family's weights.py reads them.

const std = @import("std");

pub const model_type = "mimo_v2";
pub const max_layers = 64;
/// exllamav3 folds 1/128 into layer 47's experts' up_proj (fp16 overflow); a routing weight puts it back.
pub const interm_div_layer = 47;
pub const interm_div: f32 = 128.0;

pub const Config = struct {
    layers: usize,
    hidden: usize,
    heads: usize, // query heads (both kinds of layer); a rank's under TP
    kv_full: usize,
    kv_swa: usize,
    qk_dim: usize, // 192
    v_dim: usize, // 128
    rope_dim: usize, // 64: int(head_dim * partial_rotary_factor)
    theta_full: f64,
    theta_swa: f64,
    window: usize,
    full: [max_layers]bool = @splat(false), // full attention, else sliding with sink logits
    moe: [max_layers]bool = @splat(false), // routed experts, else the dense MLP
    vocab: usize,
    experts: usize,
    top_k: usize,
    moe_width: usize,
    dense_width: usize,
    eps: f64,
    eos: [4]u32 = @splat(0),
    eos_len: usize = 0,
    sinks_full: bool,
    sinks_swa: bool,

    pub fn kvHeads(c: Config, i: usize) usize {
        return if (c.full[i]) c.kv_full else c.kv_swa;
    }
    pub fn theta(c: Config, i: usize) f64 {
        return if (c.full[i]) c.theta_full else c.theta_swa;
    }
    pub fn isEos(c: Config, t: u32) bool {
        return std.mem.indexOfScalar(u32, c.eos[0..c.eos_len], t) != null;
    }

    /// Config.read: config.json (+ generation_config.json's EOS ids), with the Python family's checks.
    pub fn parse(gpa: std.mem.Allocator, config_json: []const u8, generation_json: ?[]const u8) !Config {
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
        defer parsed.deinit();
        const o = parsed.value.object;
        if (!std.mem.eql(u8, str(o, "model_type") orelse "", model_type)) return error.NotMiMoV2;
        const q = if (o.get("quantization_config")) |v| v.object else return error.NotExl3;
        if (!std.mem.eql(u8, str(q, "quant_method") orelse "", "exl3")) return error.NotExl3;
        if (!std.mem.eql(u8, str(o, "scoring_func") orelse "sigmoid", "sigmoid") or
            !std.mem.eql(u8, str(o, "topk_method") orelse "noaux_tc", "noaux_tc") or
            (int(o, "n_group") orelse 1) != 1) return error.UnsupportedRouting;
        if ((int(o, "n_shared_experts") orelse 0) != 0) return error.UnsupportedRouting;
        if (o.get("routed_scaling_factor")) |v| switch (v) {
            .null => {},
            else => if (num(v) != 1.0) return error.UnsupportedRouting,
        };
        const hd = try need(o, "head_dim");
        const vd = try need(o, "v_head_dim");
        if ((int(o, "swa_head_dim") orelse hd) != hd or (int(o, "swa_v_head_dim") orelse vd) != vd) return error.HeadSizes;
        var prf: f64 = 1.0;
        var theta_full: ?f64 = if (o.get("rope_theta")) |v| num(v) else null;
        if (o.get("rope_parameters")) |rp| if (rp == .object) {
            if (rp.object.get("partial_rotary_factor")) |v| prf = num(v);
            if (rp.object.get("rope_theta")) |v| theta_full = num(v);
        };
        if (o.get("rope_parameters") == null or o.get("rope_parameters").? != .object or o.get("rope_parameters").?.object.get("partial_rotary_factor") == null) {
            if (o.get("partial_rotary_factor")) |v| prf = num(v);
        }
        var c: Config = .{
            .layers = try need(o, "num_hidden_layers"),
            .hidden = try need(o, "hidden_size"),
            .heads = try need(o, "num_attention_heads"),
            .kv_full = try need(o, "num_key_value_heads"),
            .kv_swa = try need(o, "swa_num_key_value_heads"),
            .qk_dim = hd,
            .v_dim = vd,
            .rope_dim = @intFromFloat(@floor(@as(f64, @floatFromInt(hd)) * prf)),
            .theta_full = theta_full orelse return error.MissingField,
            .theta_swa = num(o.get("swa_rope_theta") orelse return error.MissingField),
            .window = try need(o, "sliding_window"),
            .vocab = try need(o, "vocab_size"),
            .experts = try need(o, "n_routed_experts"),
            .top_k = try need(o, "num_experts_per_tok"),
            .moe_width = try need(o, "moe_intermediate_size"),
            .dense_width = try need(o, "intermediate_size"),
            .eps = if (o.get("layernorm_epsilon")) |v| num(v) else if (o.get("rms_norm_eps")) |v| num(v) else 1e-6,
            .sinks_full = boolean(o, "add_full_attention_sink_bias"),
            .sinks_swa = boolean(o, "add_swa_attention_sink_bias"),
        };
        if (c.layers > max_layers) return error.TooManyLayers;
        const hyb = (o.get("hybrid_layer_pattern") orelse return error.MissingField).array.items;
        const freq = (o.get("moe_layer_freq") orelse return error.MissingField).array.items;
        if (hyb.len != c.layers or freq.len != c.layers) return error.LayerPattern;
        for (hyb, freq, 0..) |h, m, i| {
            c.full[i] = h.integer == 0;
            c.moe[i] = m.integer != 0;
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

    /// A rank's view under tensor parallelism (weights.load's `replace`): heads split, everything else whole.
    pub fn rankView(c: Config, world: usize) !Config {
        if (world != 2) return error.MimoV2RequiresTwoRanks; // only two ranks were validated
        if (c.heads % world != 0 or c.kv_full % world != 0 or c.kv_swa % world != 0) return error.HeadsSplit;
        if (c.moe_width % (128 * world) != 0 or c.dense_width % (128 * world) != 0 or c.vocab % (128 * world) != 0) return error.WidthSplit;
        var r = c;
        r.heads = c.heads / world;
        r.kv_full = c.kv_full / world;
        r.kv_swa = c.kv_swa / world;
        return r;
    }
};

/// One EXL3 matrix a rank loads: its tensors' prefix and the cut (weights.py `cut`): outputs (`n`) or inputs (`k`).
pub const Cut = struct {
    prefix: []const u8,
    axis: enum { whole, n, k } = .whole,
    lo: usize = 0,
    hi: usize = 0,
};

/// Every EXL3 matrix of layer `i` a rank loads, in weights.load's order; `buf` holds the prefixes.
pub fn layerCuts(full: Config, i: usize, rank: usize, world: usize, buf: *std.ArrayList(u8), gpa: std.mem.Allocator, out: *std.ArrayList(Cut)) !void {
    const hq = full.heads / world;
    const hk = full.kvHeads(i) / world;
    const dq, const dv = .{ full.qk_dim, full.v_dim };
    const r = rank;
    const ni, const nd = .{ full.moe_width / world, full.dense_width / world };
    const one = struct {
        fn f(g: std.mem.Allocator, b: *std.ArrayList(u8), o: *std.ArrayList(Cut), comptime fmt: []const u8, args: anytype, axis: @FieldType(Cut, "axis"), lo: usize, hi: usize, w: usize) !void {
            const start = b.items.len;
            try b.print(g, fmt, args);
            try o.append(g, .{ .prefix = b.items[start..], .axis = if (w == 1) .whole else axis, .lo = lo, .hi = hi });
        }
    }.f;
    try one(gpa, buf, out, "model.layers.{d}.self_attn.q_proj", .{i}, .n, r * hq * dq, (r + 1) * hq * dq, world);
    try one(gpa, buf, out, "model.layers.{d}.self_attn.k_proj", .{i}, .n, r * hk * dq, (r + 1) * hk * dq, world);
    try one(gpa, buf, out, "model.layers.{d}.self_attn.v_proj", .{i}, .n, r * hk * dq, (r + 1) * hk * dq, world); // 192 a head: V + 64 pad
    try one(gpa, buf, out, "model.layers.{d}.self_attn.o_proj", .{i}, .k, r * hq * dv, (r + 1) * hq * dv, world);
    if (full.moe[i]) {
        for (0..full.experts) |e| {
            try one(gpa, buf, out, "model.layers.{d}.mlp.experts.{d}.gate_proj", .{ i, e }, .n, r * ni, (r + 1) * ni, world);
            try one(gpa, buf, out, "model.layers.{d}.mlp.experts.{d}.up_proj", .{ i, e }, .n, r * ni, (r + 1) * ni, world);
            try one(gpa, buf, out, "model.layers.{d}.mlp.experts.{d}.down_proj", .{ i, e }, .k, r * ni, (r + 1) * ni, world);
        }
    } else {
        try one(gpa, buf, out, "model.layers.{d}.mlp.gate_proj", .{i}, .n, r * nd, (r + 1) * nd, world);
        try one(gpa, buf, out, "model.layers.{d}.mlp.up_proj", .{i}, .n, r * nd, (r + 1) * nd, world);
        try one(gpa, buf, out, "model.layers.{d}.mlp.down_proj", .{i}, .k, r * nd, (r + 1) * nd, world);
    }
}

/// lm_head's rows of a rank: vocab / world.
pub fn headCut(full: Config, rank: usize, world: usize) Cut {
    const v = full.vocab / world;
    return .{ .prefix = "lm_head", .axis = if (world == 1) .whole else .n, .lo = rank * v, .hi = (rank + 1) * v };
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
fn boolean(o: std.json.ObjectMap, k: []const u8) bool {
    const v = o.get(k) orelse return false;
    return v == .bool and v.bool;
}

// The fields Config.read uses, as MiMo-V2.6-Flash-RL's config.json has them (the 3.0 bpw pack).
pub const fixture =
    \\{"add_full_attention_sink_bias": false, "add_swa_attention_sink_bias": true, "eos_token_id": 151645,
    \\ "head_dim": 192, "hidden_size": 4096, "intermediate_size": 16384, "layernorm_epsilon": 1e-06,
    \\ "hybrid_layer_pattern": [0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0],
    \\ "moe_layer_freq": [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    \\ "model_type": "mimo_v2", "moe_intermediate_size": 2048, "n_group": 1, "n_routed_experts": 256, "n_shared_experts": null,
    \\ "norm_topk_prob": true, "num_attention_heads": 64, "num_experts_per_tok": 8, "num_hidden_layers": 48,
    \\ "num_key_value_heads": 4, "partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "routed_scaling_factor": null,
    \\ "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "rope_type": "default", "type": "default"},
    \\ "scoring_func": "sigmoid", "sliding_window": 128, "swa_head_dim": 192, "swa_num_attention_heads": 64,
    \\ "swa_num_key_value_heads": 8, "swa_rope_theta": 10000.0, "swa_v_head_dim": 128, "topk_method": "noaux_tc",
    \\ "v_head_dim": 128, "vocab_size": 152576, "quantization_config": {"quant_method": "exl3", "codebook": "mul1"}}
;

// Python's Config.read on the pack: rope_dim 64, theta 1e7 / 1e4, window 128, eps 1e-6, sinks on sliding layers only.
test "Config.parse reads MiMo-V2.6-Flash as the Python family does" {
    const c = try Config.parse(std.testing.allocator, fixture, "{\"bos_token_id\": 151643, \"eos_token_id\": [151643, 151645, 151672]}");
    try std.testing.expectEqual(@as(f64, 10000000.0), c.theta_full);
    try std.testing.expectEqual(@as(f64, 10000.0), c.theta_swa);
    try std.testing.expectEqual(@as(usize, 128), c.window);
    try std.testing.expectEqual(@as(f64, 1e-6), c.eps);
    try std.testing.expectEqualSlices(u32, &.{ 151645, 151643, 151672 }, c.eos[0..c.eos_len]);
    try std.testing.expectEqual(@as(usize, 48), c.layers);
    try std.testing.expectEqual(@as(usize, 64), c.rope_dim); // int(192 * 0.334) = 64
    try std.testing.expect(c.full[0] and !c.full[1] and c.full[5] and c.full[47]);
    try std.testing.expect(!c.moe[0] and c.moe[1] and c.moe[47]);
    try std.testing.expect(!c.sinks_full and c.sinks_swa);
    var n_full: usize = 0;
    for (c.full[0..c.layers]) |f| n_full += @intFromBool(f);
    try std.testing.expectEqual(@as(usize, 9), n_full);
    inline for (.{ 1, 3, 4 }) |world| try std.testing.expectError(error.MimoV2RequiresTwoRanks, c.rankView(world));
    const r = try c.rankView(2);
    try std.testing.expectEqual(@as(usize, 32), r.heads);
    try std.testing.expectEqual(@as(usize, 2), r.kv_full);
    try std.testing.expectEqual(@as(usize, 4), r.kv_swa);
}

test "rank cuts are weights.load's" {
    const a = std.testing.allocator;
    const c = try Config.parse(a, fixture, null);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try buf.ensureTotalCapacity(a, 1 << 20); // prefixes slice into it: no reallocation
    var cuts: std.ArrayList(Cut) = .empty;
    defer cuts.deinit(a);
    try layerCuts(c, 1, 1, 2, &buf, a, &cuts); // a sliding MoE layer, rank 1
    try std.testing.expectEqual(@as(usize, 4 + 3 * 256), cuts.items.len);
    try std.testing.expectEqualStrings("model.layers.1.self_attn.q_proj", cuts.items[0].prefix);
    try std.testing.expectEqual(@as(usize, 32 * 192), cuts.items[0].lo); // q: heads 32..63
    try std.testing.expectEqual(@as(usize, 4 * 192), cuts.items[1].lo); // k: sliding kv heads 4..7
    try std.testing.expectEqual(@as(usize, 8 * 192), cuts.items[1].hi);
    try std.testing.expectEqual(@as(usize, 32 * 128), cuts.items[3].lo); // o: input columns of its heads
    try std.testing.expectEqualStrings("model.layers.1.mlp.experts.255.down_proj", cuts.items[cuts.items.len - 1].prefix);
    try std.testing.expectEqual(@as(usize, 1024), cuts.items[4].lo); // expert width 2048 / 2
    cuts.clearRetainingCapacity();
    try layerCuts(c, 0, 0, 2, &buf, a, &cuts); // full attention, dense MLP
    try std.testing.expectEqual(@as(usize, 2 * 192), cuts.items[1].hi); // full kv heads 0..1
    try std.testing.expectEqual(@as(usize, 8192), cuts.items[4].hi);
    const h = headCut(c, 1, 2);
    try std.testing.expectEqual(@as(usize, 76288), h.lo);
}
