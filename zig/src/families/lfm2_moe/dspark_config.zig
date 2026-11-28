//! LiquidAI's DSpark drafter for LFM2.5-24B-A2B: its config admitted, any other shape refused with its reason.
const std = @import("std");
const c = @import("config.zig");

/// The geometry the drafter's kernels take: the target's attention shape, five taps at most eight layers.
pub const Config = struct {
    taps: [8]u32 = @splat(0),
    tap_count: u32,
    layers: u32,
    intermediate: u32,
    /// Rows a block takes: the pending token and block - 1 mask tokens, one draft a row.
    block: u32,
    mask: u32,
    rank: u32,
    theta: f32,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer parsed.deinit();
        const o = if (parsed.value == .object) parsed.value.object else return error.BadDraftConfig;
        const archs = o.get("architectures") orelse return error.BadDraftConfig;
        if (archs != .array or archs.array.items.len != 1 or archs.array.items[0] != .string) return error.BadDraftConfig;
        if (!std.mem.eql(u8, archs.array.items[0].string, "Lfm2DSparkDraftModel")) return error.NotLfmDSpark;
        const d = if (o.get("dflash_config")) |v| (if (v == .object) v.object else return error.BadDraftConfig) else return error.BadDraftConfig;
        var out: Config = .{
            .tap_count = 0,
            .layers = try small(o, "num_hidden_layers"),
            .intermediate = try small(o, "intermediate_size"),
            .block = try small(o, "block_size"),
            .mask = try small(d, "mask_token_id"),
            .rank = try small(o, "markov_rank"),
            .theta = @floatCast(try number(o, "rope_theta")),
        };
        // the target's widths: the drafter shares its embedding, head and attention kernels
        if (try small(o, "hidden_size") != c.hidden or try small(o, "vocab_size") != c.vocab) return error.DraftForAnotherTarget;
        if (try small(d, "num_target_layers") != c.layers) return error.DraftForAnotherTarget;
        const heads = try small(o, "num_attention_heads");
        const kv = try small(o, "num_key_value_heads");
        if (heads != c.query_heads or kv != c.kv_heads or try small(o, "head_dim") != c.head_dim) return error.UnsupportedDraftShape;
        if (@as(f32, @floatCast(try number(o, "rms_norm_eps"))) != c.eps) return error.UnsupportedDraftShape;
        const taps = d.get("target_layer_ids") orelse return error.BadDraftConfig;
        if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > out.taps.len) return error.BadDraftConfig;
        for (taps.array.items, 0..) |t, i| {
            if (t != .integer or t.integer < 0) return error.BadDraftConfig;
            if (t.integer >= c.layers) return error.DraftForAnotherTarget;
            out.taps[i] = @intCast(t.integer);
        }
        out.tap_count = @intCast(taps.array.items.len);
        try kinds(o, out.layers);
        // the block's MLP is the fixed SiLU gate
        const act = o.get("hidden_act") orelse return error.BadDraftConfig;
        if (act != .string or !std.mem.eql(u8, act.string, "silu")) return error.UnsupportedDraftActivation;
        // the vendored defaults decide the block: non-causal attention, interleaved RoPE, a plain Markov head
        if (o.get("is_causal") != null) return error.UnsupportedDraftMask;
        const neox = o.get("rope_is_neox_style") orelse return error.BadDraftConfig;
        if (neox != .bool or neox.bool) return error.UnsupportedDraftRope;
        const head = o.get("markov_head_type") orelse return error.BadDraftConfig;
        if (head != .string or !std.mem.eql(u8, head.string, "vanilla")) return error.UnsupportedMarkovHead;
        if (o.get("sample_from_anchor")) |v| if (v != .bool or !v.bool) return error.UnsupportedDraftAnchor;
        if (out.layers == 0 or out.layers > 8 or out.block < 2 or out.block > 16) return error.UnsupportedDraftShape;
        if (out.intermediate == 0 or out.intermediate % 64 != 0 or out.rank == 0 or out.rank > 256 or out.rank % 64 != 0) return error.UnsupportedDraftShape;
        if (out.mask >= c.vocab or !std.math.isFinite(out.theta) or !(out.theta > 0)) return error.BadDraftConfig;
        return out;
    }
};

/// Every layer full attention (the block's mask has no sliding window).
fn kinds(o: std.json.ObjectMap, layers: u32) !void {
    const v = o.get("layer_types") orelse return;
    if (v != .array or v.array.items.len != layers) return error.BadDraftConfig;
    for (v.array.items) |k| if (k != .string or !std.mem.eql(u8, k.string, "full_attention")) return error.UnsupportedDraftMask;
}

fn small(o: std.json.ObjectMap, key: []const u8) !u32 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadDraftConfig;
    return @intCast(v.integer);
}

fn number(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

/// LiquidAI/LFM2.5-24B-A2B-DSpark's config.json at e052e5b1dd10, as released.
pub const stock =
    \\{"architectures":["Lfm2DSparkDraftModel"],"model_type":"qwen3","dtype":"bfloat16","hidden_size":2048,
    \\"num_hidden_layers":5,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":64,"intermediate_size":6144,
    \\"hidden_act":"silu","rms_norm_eps":1e-05,"vocab_size":128000,"rope_theta":10000000.0,
    \\"max_position_embeddings":131072,"layer_types":["full_attention","full_attention","full_attention",
    \\"full_attention","full_attention"],"block_size":9,"dflash_config":{"mask_token_id":125017,
    \\"target_layer_ids":[2,10,18,26,34],"num_target_layers":40},"markov_rank":256,"rope_is_neox_style":false,
    \\"enable_confidence_head":true,"markov_head_type":"vanilla"}
;

fn edited(from: []const u8, to: []const u8) !Config {
    const a = std.testing.allocator;
    const text = try std.mem.replaceOwned(u8, a, stock, from, to);
    defer a.free(text);
    return Config.parse(a, text);
}

test "lfm2 dspark: the released config is admitted with its taps, block, mask token and Markov rank" {
    const cfg = try Config.parse(std.testing.allocator, stock);
    try std.testing.expectEqualSlices(u32, &.{ 2, 10, 18, 26, 34 }, cfg.taps[0..cfg.tap_count]);
    try std.testing.expectEqual(@as(u32, 5), cfg.layers);
    try std.testing.expectEqual(@as(u32, 9), cfg.block);
    try std.testing.expectEqual(@as(u32, 125017), cfg.mask);
    try std.testing.expectEqual(@as(u32, 256), cfg.rank);
    try std.testing.expectEqual(@as(u32, 6144), cfg.intermediate);
    try std.testing.expectEqual(@as(f32, 1e7), cfg.theta);
}

test "lfm2 dspark: other drafters and shapes are refused with their reason" {
    try std.testing.expectError(error.NotLfmDSpark, edited("Lfm2DSparkDraftModel", "DFlashDraftModel"));
    try std.testing.expectError(error.DraftForAnotherTarget, edited("\"hidden_size\":2048", "\"hidden_size\":4096"));
    try std.testing.expectError(error.DraftForAnotherTarget, edited("\"num_target_layers\":40", "\"num_target_layers\":24"));
    try std.testing.expectError(error.DraftForAnotherTarget, edited("26,34]", "26,44]"));
    try std.testing.expectError(error.UnsupportedDraftShape, edited("\"head_dim\":64", "\"head_dim\":128"));
    try std.testing.expectError(error.UnsupportedDraftActivation, edited("\"silu\"", "\"gelu\""));
    try std.testing.expectError(error.UnsupportedDraftRope, edited("\"rope_is_neox_style\":false", "\"rope_is_neox_style\":true"));
    try std.testing.expectError(error.UnsupportedMarkovHead, edited("\"vanilla\"", "\"gated\""));
    try std.testing.expectError(error.UnsupportedDraftMask, edited("\"full_attention\"]", "\"sliding_attention\"]"));
    try std.testing.expectError(error.UnsupportedDraftMask, edited("\"block_size\"", "\"is_causal\":true,\"block_size\""));
    try std.testing.expectError(error.UnsupportedDraftAnchor, edited("\"block_size\"", "\"sample_from_anchor\":false,\"block_size\""));
    try std.testing.expectError(error.UnsupportedDraftShape, edited("\"block_size\":9", "\"block_size\":17"));
    try std.testing.expectError(error.UnsupportedDraftShape, edited("\"markov_rank\":256", "\"markov_rank\":512"));
    try std.testing.expectError(error.BadDraftConfig, edited("10000000.0", "1e100"));
    try std.testing.expectError(error.BadDraftConfig, edited("\"mask_token_id\":125017", "\"mask_token_id\":128000"));
}
