//! Admission for DeepSeek-V4.1-Flash (deepseek_v41): pinned geometry, affine widths, engram tables by the shards.
const std = @import("std");

pub const hidden = 5120;
pub const vocab = 129280;
pub const layers = 40;
/// DSpark draft stages (the checkpoint's mtp.* blocks).
pub const stages = 3;
pub const heads = 64;
pub const head_dim = 512;
pub const rope_dim = 64;
pub const q_rank = 1280;
pub const o_groups = 8;
pub const o_rank = 1024;
/// Query heads an output group projects: [heads / o_groups * head_dim] inputs a group.
pub const group_in = heads / o_groups * head_dim;
pub const window = 128;
pub const experts = 384;
pub const topk = 6;
pub const moe_inter = 2304;
pub const draft_experts = 128;
pub const draft_topk = 3;
pub const index_heads = 32;
pub const index_dim = 128;
pub const index_topk = 512;
pub const candidate_blocks = 2048;
pub const candidate_block = 8;
pub const hc = 4;
pub const hc_mix = (2 + hc) * hc;
pub const sinkhorn = 20;
pub const hc_eps: f32 = 1e-6;
pub const norm_eps: f32 = 1e-20;
pub const swiglu_limit: f32 = 10.0;
pub const route_scale: f32 = 1.5;
pub const engram_heads = 8;
pub const engram_dim = 256;
pub const engram_ngram = 4;
/// Table rows a position looks up a layer: (n-gram sizes 2..4) x heads.
pub const engram_cols = (engram_ngram - 1) * engram_heads;
/// A row as stored: the 256 fp8 e4m3 values, then their 8 e8m0 scales.
pub const engram_row = engram_dim + engram_dim / 32;
pub const engram_layers = [_]usize{ 1, 14 };
pub const draft_block = 5;
pub const draft_targets = [_]usize{ 37, 38, 39 };
pub const markov_rank = 256;

/// Layers' compressed-KV ratios (0: window only), the KV and index sources; layer 20 also picks candidate blocks.
pub const ratios = blk: {
    var r: [layers]u8 = undefined;
    for (&r, 0..) |*x, i| x.* = if (i < 2) 0 else if (i < 20) 2 else 1;
    break :blk r;
};
pub const kv_sources = [_]usize{ 2, 8, 14, 20 };
pub const index_sources = [_]usize{ 2, 8, 14, 20, 24, 28, 32, 36 };
pub const candidate_source = 20;

/// The KV source a compressing layer reads (the last source at or before it), else null.
pub fn kvSource(layer: usize) ?usize {
    if (ratios[layer] == 0) return null;
    var out: ?usize = null;
    for (kv_sources) |s| if (s <= layer) {
        out = s;
    };
    return out;
}

pub fn isIndexSource(layer: usize) bool {
    return std.mem.indexOfScalar(usize, &index_sources, layer) != null;
}

pub fn isKvSource(layer: usize) bool {
    return std.mem.indexOfScalar(usize, &kv_sources, layer) != null;
}

pub fn engramIndex(layer: usize) ?usize {
    return std.mem.indexOfScalar(usize, &engram_layers, layer);
}

/// An affine width (bits, group) as the row kernels read it.
pub const Width = struct { bits: u8, group: u16 };

pub const Rope = struct {
    theta: f32,
    compress_theta: f32,
    factor: f32,
    original: u32,
    beta_fast: f32,
    beta_slow: f32,
};

pub const Config = struct {
    context: usize,
    rope: Rope,
    /// The global width: every affine projection that names none of its own (attention, shared experts, head).
    dense: Width,
    /// Routed experts' width by layer, then by draft stage.
    experts: [layers]Width,
    draft_experts: [stages]Width,
    engram_rows: [engram_layers.len]usize,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadDeepseekConfig;
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse {
        std.log.err("deepseek_v41: config.json has no {s}", .{key});
        return error.BadDeepseekConfig;
    };
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDeepseekConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = try field(o, key);
    if (v != .integer or v.integer < 0) return error.BadDeepseekConfig;
    return @intCast(v.integer);
}

fn ints(o: std.json.ObjectMap, key: []const u8, want: []const usize) !void {
    const v = try field(o, key);
    if (v != .array or v.array.items.len != want.len) return error.UnsupportedDeepseekGeometry;
    for (v.array.items, want) |x, w| if (x != .integer or x.integer != w) {
        std.log.err("deepseek_v41: {s} differs from DeepSeek-V4.1-Flash's", .{key});
        return error.UnsupportedDeepseekGeometry;
    };
}

fn width(v: std.json.Value) !Width {
    const o = try object(v);
    const bits = try integer(o, "bits");
    const group = try integer(o, "group_size");
    if (o.get("mode")) |mode| if (mode != .string or !std.mem.eql(u8, mode.string, "affine")) return error.UnsupportedDeepseekQuantization;
    if (std.mem.indexOfScalar(usize, &.{ 2, 3, 4, 5, 6, 8 }, bits) == null or std.mem.indexOfScalar(usize, &.{ 32, 64, 128 }, group) == null)
        return error.UnsupportedDeepseekQuantization;
    return .{ .bits = @intCast(bits), .group = @intCast(group) };
}

/// A stack's width: its w1, w2 and w3 entries must agree; none names the global width.
fn stackWidth(q: std.json.ObjectMap, prefix: []const u8, global: Width) !Width {
    var out: ?Width = null;
    for ([_][]const u8{ "w1", "w2", "w3" }) |w| {
        var buf: [96]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "{s}ffn.experts.{s}", .{ prefix, w });
        const this = if (q.get(key)) |v| try width(v) else global;
        if (out) |o| if (!std.meta.eql(o, this)) return error.UnsupportedDeepseekQuantization;
        out = this;
    }
    return out.?;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    const t = try field(root, "model_type");
    if (t != .string or !std.mem.eql(u8, t.string, "deepseek_v41")) return error.UnsupportedDeepseekConfig;
    const dims = .{
        .{ "hidden_size", hidden },                         .{ "vocab_size", vocab },                       .{ "num_hidden_layers", layers },
        .{ "num_attention_heads", heads },                  .{ "num_key_value_heads", 1 },                  .{ "head_dim", head_dim },
        .{ "qk_rope_head_dim", rope_dim },                  .{ "q_lora_rank", q_rank },                     .{ "o_lora_rank", o_rank },
        .{ "o_groups", o_groups },                          .{ "sliding_window", window },                  .{ "n_routed_experts", experts },
        .{ "num_experts_per_tok", topk },                   .{ "moe_intermediate_size", moe_inter },        .{ "n_shared_experts", 1 },
        .{ "index_n_heads", index_heads },                  .{ "index_head_dim", index_dim },               .{ "index_topk", index_topk },
        .{ "candidate_source_layer_id", candidate_source }, .{ "candidate_topk_blocks", candidate_blocks }, .{ "candidate_block_size", candidate_block },
        .{ "hc_mult", hc },                                 .{ "hc_sinkhorn_iters", sinkhorn },             .{ "engram_n_heads", engram_heads },
        .{ "engram_head_dim", engram_dim },                 .{ "engram_max_ngram_size", engram_ngram },     .{ "num_nextn_predict_layers", stages },
        .{ "dspark_block_size", draft_block },              .{ "dspark_n_routed_experts", draft_experts },  .{ "dspark_num_experts_per_tok", draft_topk },
        .{ "dspark_markov_rank", markov_rank },             .{ "dspark_noise_token_id", 128799 },
    };
    inline for (dims) |d| if (try integer(root, d[0]) != d[1]) {
        std.log.err("deepseek_v41: {s} is not DeepSeek-V4.1-Flash's {d}", .{ d[0], d[1] });
        return error.UnsupportedDeepseekGeometry;
    };
    var r43: [layers + stages]usize = undefined;
    for (&r43, 0..) |*x, i| x.* = if (i < layers) ratios[i] else 0;
    try ints(root, "compress_ratios", &r43);
    try ints(root, "kv_source_layer_ids", &kv_sources);
    try ints(root, "index_source_layer_ids", &index_sources);
    try ints(root, "engram_layer_ids", &engram_layers);
    try ints(root, "dspark_target_layer_ids", &draft_targets);
    for ([_][]const u8{ "scoring_func", "topk_method" }, [_][]const u8{ "sqrtsoftplus", "noaux_tc" }) |k, want| {
        const v = try field(root, k);
        if (v != .string or !std.mem.eql(u8, v.string, want)) return error.UnsupportedDeepseekConfig;
    }
    const checks = .{ .{ "rms_norm_eps", norm_eps }, .{ "swiglu_limit", swiglu_limit }, .{ "routed_scaling_factor", route_scale }, .{ "hc_eps", hc_eps } };
    inline for (checks) |c| if (@as(f32, @floatCast(try number(try field(root, c[0])))) != c[1]) return error.UnsupportedDeepseekConfig;

    const rs = try object(try field(root, "rope_scaling"));
    const rt = try field(rs, "rope_type");
    if (rt != .string or !std.mem.eql(u8, rt.string, "yarn")) return error.UnsupportedDeepseekConfig;
    const q = try object(try field(root, "quantization"));
    const dense = try width(.{ .object = q });
    var out: Config = .{
        .context = try integer(root, "max_position_embeddings"),
        .rope = .{
            .theta = @floatCast(try number(try field(root, "rope_theta"))),
            .compress_theta = @floatCast(try number(try field(root, "compress_rope_theta"))),
            .factor = @floatCast(try number(try field(rs, "factor"))),
            .original = @intCast(try integer(rs, "original_max_position_embeddings")),
            .beta_fast = @floatCast(try number(try field(rs, "beta_fast"))),
            .beta_slow = @floatCast(try number(try field(rs, "beta_slow"))),
        },
        .dense = dense,
        .experts = undefined,
        .draft_experts = undefined,
        .engram_rows = undefined,
    };
    for (&out.experts, 0..) |*w, i| {
        var buf: [32]u8 = undefined;
        w.* = try stackWidth(q, try std.fmt.bufPrint(&buf, "layers.{d}.", .{i}), dense);
    }
    for (&out.draft_experts, 0..) |*w, i| {
        var buf: [32]u8 = undefined;
        w.* = try stackWidth(q, try std.fmt.bufPrint(&buf, "mtp.{d}.", .{i}), dense);
    }
    const rows = try field(root, "engram_num_embeddings");
    if (rows != .array or rows.array.items.len != engram_layers.len) return error.UnsupportedDeepseekGeometry;
    for (&out.engram_rows, rows.array.items) |*n, v| n.* = if (v == .integer and v.integer > 0) @intCast(v.integer) else return error.BadDeepseekConfig;
    return out;
}

test "the layer plan: sources, ratios and engram layers" {
    try std.testing.expectEqual(@as(?usize, null), kvSource(1));
    try std.testing.expectEqual(@as(?usize, 2), kvSource(7));
    try std.testing.expectEqual(@as(?usize, 14), kvSource(19));
    try std.testing.expectEqual(@as(?usize, 20), kvSource(39));
    try std.testing.expectEqual(@as(u8, 2), ratios[19]);
    try std.testing.expectEqual(@as(u8, 1), ratios[20]);
    try std.testing.expectEqual(@as(?usize, 1), engramIndex(14));
    try std.testing.expectEqual(@as(usize, 264), engram_row);
}
