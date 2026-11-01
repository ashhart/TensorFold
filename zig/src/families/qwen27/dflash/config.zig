//! DFlash2 is a separate five-layer GQA contract; the target's GDN and draft conventions are not interchangeable.
const std = @import("std");
const target_config = @import("../config.zig");
pub const repository = "z-lab/Qwen3.8-27B-DFlash2";
/// The stock DFlash2 taps, padded to the eight a DFlash v1 draft may name.
pub const tap_ids = [8]u32{ 5, 19, 33, 47, 61, 0, 0, 0 };
pub const max_taps = 8;
/// DFlash2 (conv kernels and a candidate selector) or z-lab's DFlash v1 (a plain qwen3 body, taps fused by fc).
pub const Variant = enum { dflash2, v1 };
pub const Config = struct {
    variant: Variant = .dflash2,
    hidden: u32 = 5120,
    intermediate: u32 = 17408,
    layers: u32 = 5,
    vocab: u32 = 248320,
    heads: u32 = 32,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    target_layers: u32 = 64,
    block: u32 = 8,
    mask: u32 = 248070,
    window: u32 = 2048,
    group: u32 = 16,
    conv: u32 = 2,
    rank: u32 = 256,
    topk: u32 = 16,
    eps: f32 = 1e-6,
    theta: f64 = 10000000,
    taps: [8]u32 = tap_ids,
    tap_count: u32 = 5,
    /// v1: the layers whose layer_types entry is full_attention (DFlash2 layers are all sliding).
    full: [8]bool = @splat(false),
    pub fn qWidth(c: Config) u32 {
        return c.heads * c.head_dim;
    }
    pub fn kvWidth(c: Config) u32 {
        return c.kv_heads * c.head_dim;
    }
    pub fn tapWidth(c: Config) u32 {
        return c.hidden * c.tap_count;
    }
    pub fn dynamicWidth(c: Config) u32 {
        return 4 * (c.hidden / c.group);
    }
    pub fn depth(c: Config) u32 {
        return c.block - 1;
    }
    /// DFlash2: 15 a layer and six globals; v1: 11 a layer beside fc, hidden_norm and norm.
    pub fn tensorCount(c: Config) usize {
        return if (c.variant == .v1) 11 * @as(usize, c.layers) + 3 else 15 * @as(usize, c.layers) + 6;
    }
    /// DFlash2: seven linears and two conv projections a layer, fc and the selector's projection; v1: seven and fc.
    pub fn linearCount(c: Config) usize {
        return if (c.variant == .v1) 7 * @as(usize, c.layers) + 1 else 9 * @as(usize, c.layers) + 2;
    }
    pub fn check(c: Config) !void {
        if (c.variant == .v1) return c.checkV1();
        const dimensions = [_]u32{ c.hidden, c.intermediate, c.layers, c.vocab, c.heads, c.kv_heads, c.head_dim, c.target_layers, c.group, c.rank, c.topk };
        for (dimensions) |n| if (n == 0) return error.BadDraftConfig;
        if (c.hidden % 64 != 0 or c.intermediate % 64 != 0 or c.hidden % c.group != 0 or c.heads % c.kv_heads != 0 or c.head_dim % 2 != 0 or c.block != 8 or c.conv != 2 or c.layers > 8 or c.mask >= c.vocab or c.topk > c.vocab or c.window < c.block) return error.BadDraftConfig;
        if (!std.math.isFinite(c.eps) or c.eps <= 0 or !std.math.isFinite(c.theta) or c.theta <= 0) return error.BadDraftConfig;
        if (c.target_layers == 64 and !std.mem.eql(u32, &c.taps, &tap_ids)) return error.UnsupportedDraftTaps;
        if (c.tap_count != 5) return error.UnsupportedDraftTaps;
        for (c.taps[0..5], 0..) |tap, i| for (c.taps[0..i]) |previous| if (tap == previous) return error.UnsupportedDraftTaps;
        for (c.taps[0..5]) |i| if (i >= c.target_layers) return error.BadDraftConfig;
        const q = try std.math.mul(u32, c.heads, c.head_dim);
        _ = try std.math.mul(u32, c.kv_heads, c.head_dim);
        _ = try std.math.mul(u32, c.hidden, 5);
        _ = try std.math.mul(u32, c.hidden / c.group, 4);
        if (q % 64 != 0) return error.BadDraftConfig;
    }
    /// A draft reads one target's taps and vocabulary: its width, vocabulary and tapped depth.
    pub fn bind(c: Config, hidden: usize, vocab: usize, layers: usize) !void {
        if (hidden != c.hidden or vocab != c.vocab or layers != c.target_layers) return error.DraftForAnotherTarget;
    }
    /// v1 has no conv kernels and no selector (group, conv and rank zero); its lattice keeps the head's top-k.
    fn checkV1(c: Config) !void {
        const dimensions = [_]u32{ c.hidden, c.intermediate, c.layers, c.vocab, c.heads, c.kv_heads, c.head_dim, c.target_layers, c.topk, c.tap_count };
        for (dimensions) |n| if (n == 0) return error.BadDraftConfig;
        if (c.group != 0 or c.conv != 0 or c.rank != 0) return error.BadDraftConfig;
        if (c.hidden % 64 != 0 or c.intermediate % 64 != 0 or c.heads % c.kv_heads != 0 or c.head_dim % 2 != 0 or c.block < 2 or c.block > 16 or c.layers > 8 or c.tap_count > max_taps or c.mask >= c.vocab or c.topk > 16 or c.topk > c.vocab or c.window < c.block) return error.BadDraftConfig;
        if (!std.math.isFinite(c.eps) or c.eps <= 0 or !std.math.isFinite(c.theta) or c.theta <= 0) return error.BadDraftConfig;
        for (c.taps[0..c.tap_count]) |i| if (i >= c.target_layers) return error.BadDraftConfig;
        for (c.taps[0..c.tap_count], 0..) |tap, i| for (c.taps[0..i]) |previous| if (tap == previous) return error.UnsupportedDraftTaps;
        for (c.taps[c.tap_count..]) |unused| if (unused != 0) return error.BadDraftConfig;
        for (c.full[c.layers..]) |unused| if (unused) return error.BadDraftConfig;
        const q = try std.math.mul(u32, c.heads, c.head_dim);
        _ = try std.math.mul(u32, c.kv_heads, c.head_dim);
        _ = try std.math.mul(u32, c.hidden, c.tap_count);
        if (q % 64 != 0) return error.BadDraftConfig;
    }
    pub fn requireStock(c: Config) !void {
        try c.check();
        if (c.hidden != 5120 or c.intermediate != 17408 or c.layers != 5 or c.vocab != 248320 or c.heads != 32 or c.kv_heads != 8 or c.head_dim != 128 or c.target_layers != 64 or c.mask != 248070 or c.window != 2048 or c.group != 16 or c.rank != 256 or c.topk != 16 or c.eps != 1e-6 or c.theta != 10000000) return error.NotStockDFlash2;
    }
};
fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadDraftConfig;
}
fn integer(o: std.json.ObjectMap, name: []const u8) !u32 {
    const v = o.get(name) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadDraftConfig;
    return @intCast(v.integer);
}
fn number(o: std.json.ObjectMap, name: []const u8) !f64 {
    const v = o.get(name) orelse return error.BadDraftConfig;
    const value: f64 = switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        else => return error.BadDraftConfig,
    };
    return if (std.math.isFinite(value)) value else error.BadDraftConfig;
}
fn text(o: std.json.ObjectMap, name: []const u8, wanted: []const u8) !void {
    const v = o.get(name) orelse return error.BadDraftConfig;
    if (v != .string or !std.mem.eql(u8, v.string, wanted)) return error.BadDraftConfig;
}
pub fn parse(gpa: std.mem.Allocator, raw: []const u8) !Config {
    const p = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer p.deinit();
    const o = try object(p.value);
    const d = try object(o.get("dflash_config") orelse return error.BadDraftConfig);
    const architectures = o.get("architectures") orelse return error.BadDraftConfig;
    if (architectures == .array and architectures.array.items.len == 1 and architectures.array.items[0] == .string and std.mem.eql(u8, architectures.array.items[0].string, "DFlashDraftModel")) return parseV1(o, d);
    if (architectures != .array or architectures.array.items.len != 1 or architectures.array.items[0] != .string or !std.mem.eql(u8, architectures.array.items[0].string, "DFlash2DraftModel")) return error.BadDraftConfig;
    try text(o, "model_type", "qwen3");
    try text(o, "dtype", "bfloat16");
    try text(o, "hidden_act", "silu");
    for ([_][]const u8{ "is_causal", "attention_bias", "tie_word_embeddings" }) |key| {
        const v = o.get(key) orelse return error.BadDraftConfig;
        if (v != .bool or v.bool) return error.BadDraftConfig;
    }
    const sliding = o.get("use_sliding_window") orelse return error.BadDraftConfig;
    if (sliding != .bool or !sliding.bool or try number(o, "attention_dropout") != 0) return error.BadDraftConfig;
    const rope = try object(o.get("rope_parameters") orelse return error.BadDraftConfig);
    try text(rope, "rope_type", "default");
    if (o.get("rope_scaling")) |v| if (v != .null) return error.BadDraftConfig; // a scaled rope read as plain rope
    var c = Config{ .hidden = try integer(o, "hidden_size"), .intermediate = try integer(o, "intermediate_size"), .layers = try integer(o, "num_hidden_layers"), .vocab = try integer(o, "vocab_size"), .heads = try integer(o, "num_attention_heads"), .kv_heads = try integer(o, "num_key_value_heads"), .head_dim = try integer(o, "head_dim"), .target_layers = try integer(o, "num_target_layers"), .block = try integer(d, "block_size"), .mask = try integer(d, "mask_token_id"), .window = try integer(o, "sliding_window"), .group = try integer(d, "conv_group_size"), .conv = try integer(d, "conv_kernel_size"), .rank = try integer(d, "selector_rank"), .topk = try integer(d, "selector_top_k"), .eps = @floatCast(try number(o, "rms_norm_eps")), .theta = try number(rope, "rope_theta") };
    const taps = d.get("target_layer_ids") orelse return error.BadDraftConfig;
    if (taps != .array or taps.array.items.len != 5) return error.BadDraftConfig;
    for (taps.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadDraftConfig;
        c.taps[i] = @intCast(v.integer);
    }
    const types = o.get("layer_types") orelse return error.BadDraftConfig;
    if (types != .array or types.array.items.len != c.layers) return error.BadDraftConfig;
    for (types.array.items) |v| if (v != .string or !std.mem.eql(u8, v.string, "sliding_attention")) return error.BadDraftConfig;
    try c.requireStock();
    return c;
}
/// z-lab's DFlash v1 layout (DFlashDraftModel): its taps, block, mask and layer kinds from the config.
fn parseV1(o: std.json.ObjectMap, d: std.json.ObjectMap) !Config {
    try text(o, "model_type", "qwen3");
    try text(o, "dtype", "bfloat16");
    try text(o, "hidden_act", "silu");
    for ([_][]const u8{ "attention_bias", "tie_word_embeddings" }) |key| {
        const v = o.get(key) orelse return error.BadDraftConfig;
        if (v != .bool or v.bool) return error.BadDraftConfig;
    }
    // the draft attends to its whole block both ways; a config that says otherwise is not this reading
    if (o.get("is_causal")) |v| if (v != .bool or v.bool) return error.BadDraftConfig;
    for ([_][]const u8{ "conv_group_size", "conv_kernel_size", "selector_rank", "selector_top_k" }) |key| if (d.get(key) != null) return error.BadDraftConfig;
    const sliding = o.get("use_sliding_window") orelse return error.BadDraftConfig;
    if (sliding != .bool or !sliding.bool or try number(o, "attention_dropout") != 0) return error.BadDraftConfig;
    const rope = try object(o.get("rope_parameters") orelse return error.BadDraftConfig);
    try text(rope, "rope_type", "default");
    if (o.get("rope_scaling")) |v| if (v != .null) return error.BadDraftConfig;
    var c = Config{ .variant = .v1, .hidden = try integer(o, "hidden_size"), .intermediate = try integer(o, "intermediate_size"), .layers = try integer(o, "num_hidden_layers"), .vocab = try integer(o, "vocab_size"), .heads = try integer(o, "num_attention_heads"), .kv_heads = try integer(o, "num_key_value_heads"), .head_dim = try integer(o, "head_dim"), .target_layers = try integer(o, "num_target_layers"), .block = try integer(d, "block_size"), .mask = try integer(d, "mask_token_id"), .window = try integer(o, "sliding_window"), .group = 0, .conv = 0, .rank = 0, .topk = 16, .eps = @floatCast(try number(o, "rms_norm_eps")), .theta = try number(rope, "rope_theta"), .taps = @splat(0), .tap_count = 0 };
    const taps = d.get("target_layer_ids") orelse return error.BadDraftConfig;
    if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > max_taps) return error.BadDraftConfig;
    for (taps.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadDraftConfig;
        c.taps[i] = @intCast(v.integer);
    }
    c.tap_count = @intCast(taps.array.items.len);
    if (c.layers == 0 or c.layers > 8) return error.BadDraftConfig;
    const types = o.get("layer_types") orelse return error.BadDraftConfig;
    if (types != .array or types.array.items.len != c.layers) return error.BadDraftConfig;
    for (types.array.items, 0..) |v, i| {
        if (v != .string) return error.BadDraftConfig;
        if (std.mem.eql(u8, v.string, "full_attention")) c.full[i] = true else if (!std.mem.eql(u8, v.string, "sliding_attention")) return error.BadDraftConfig;
    }
    try c.check();
    return c;
}

/// --drafter's config against the target's, from the two folders' config.json: a refusal comes before either loads.
pub fn admit(gpa: std.mem.Allocator, io: std.Io, target_dir: []const u8, draft_dir: []const u8) !Config {
    const target_text = try readConfig(gpa, io, target_dir, 16 << 20);
    defer gpa.free(target_text);
    const draft_text = try readConfig(gpa, io, draft_dir, 1 << 20);
    defer gpa.free(draft_text);
    const t = try target_config.parse(gpa, target_text, null);
    const c = try parse(gpa, draft_text);
    try c.bind(t.hidden, t.vocab, t.layers);
    return c;
}
fn readConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, limit: usize) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(limit));
}

test "stock contracts count81native tensors47eligiblelinears and seven mask predictions" {
    const c = Config{};
    try c.requireStock();
    try std.testing.expectEqual(@as(usize, 81), c.tensorCount());
    try std.testing.expectEqual(@as(usize, 47), c.linearCount());
    try std.testing.expectEqual(@as(u32, 7), c.depth());
}
test "tap convention is fixed after decoder blocks and cannot be shifted" {
    var c = Config{};
    c.taps[0] = 4;
    try std.testing.expectError(error.UnsupportedDraftTaps, c.check());
}
test "block slots and finite dimensions are admitted before operators" {
    var c = Config{};
    c.block = 16;
    try std.testing.expectError(error.BadDraftConfig, c.check());
    c = .{};
    c.hidden = 0;
    try std.testing.expectError(error.BadDraftConfig, c.check());
    c = .{};
    c.eps = std.math.nan(f32);
    try std.testing.expectError(error.BadDraftConfig, c.check());
}

const fixture =
    \\{"architectures":["DFlash2DraftModel"],"model_type":"qwen3","dtype":"bfloat16","hidden_act":"silu",
    \\"hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":5,"vocab_size":248320,
    \\"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"num_target_layers":64,
    \\"is_causal":false,"attention_bias":false,"tie_word_embeddings":false,"attention_dropout":0.0,
    \\"use_sliding_window":true,"sliding_window":2048,"rms_norm_eps":0.000001,
    \\"rope_parameters":{"rope_type":"default","rope_theta":10000000},
    \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
    \\"dflash_config":{"block_size":8,"mask_token_id":248070,"conv_group_size":16,"conv_kernel_size":2,
    \\"selector_rank":256,"selector_top_k":16,"target_layer_ids":[5,19,33,47,61]}}
;
test "actual nested cached schema resolves the stock noncausal draft and tap order" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqualDeep(Config{}, c);
}
test "a causal reading or changed native storage cannot silently become the default" {
    const a = std.testing.allocator;
    var p = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer p.deinit();
    try p.value.object.put(p.arena.allocator(), "is_causal", .{ .bool = true });
    const text_json = try std.json.Stringify.valueAlloc(a, p.value, .{});
    defer a.free(text_json);
    try std.testing.expectError(error.BadDraftConfig, parse(a, text_json));
}
test "a draft whose rope is scaled by a rope_scaling block is refused, not read as plain rope" {
    const a = std.testing.allocator;
    const scaled = try std.mem.replaceOwned(u8, a, fixture, "\"layer_types\"", "\"rope_scaling\":{\"type\":\"yarn\",\"factor\":4.0},\"layer_types\"");
    defer a.free(scaled);
    try std.testing.expectError(error.BadDraftConfig, parse(a, scaled));
}

fn targetJson(a: std.mem.Allocator, hidden: u32, intermediate: u32, layers: u32, heads: u32, v_heads: u32) ![]u8 {
    return std.fmt.allocPrint(a,
        \\{{"model_type":"qwen3_5","text_config":{{"model_type":"qwen3_5_text","hidden_size":{d},"intermediate_size":{d},
        \\"num_hidden_layers":{d},"num_attention_heads":{d},"num_key_value_heads":4,"head_dim":256,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":{d},"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\"linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"eos_token_id":248044}}}}
    , .{ hidden, intermediate, layers, heads, v_heads });
}
/// Two checkpoint folders holding only their config.json: `admit` reads no weights, since none exist.
fn admitFolders(target: []const u8, draft: []const u8) !Config {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "target");
    try tmp.dir.createDirPath(io, "draft");
    try tmp.dir.writeFile(io, .{ .sub_path = "target/config.json", .data = target });
    try tmp.dir.writeFile(io, .{ .sub_path = "draft/config.json", .data = draft });
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(root);
    const t = try std.fs.path.join(a, &.{ root, "target" });
    defer a.free(t);
    const d = try std.fs.path.join(a, &.{ root, "draft" });
    defer a.free(d);
    return admit(a, io, t, d);
}
test "a 9B target refuses the 27B's DFlash2 and a 9B-shaped DFlash2 by name from their configs alone" {
    const a = std.testing.allocator;
    const nine = try targetJson(a, 4096, 12288, 32, 16, 32);
    defer a.free(nine);
    const big = try targetJson(a, 5120, 17408, 64, 24, 48);
    defer a.free(big);
    try std.testing.expectEqualDeep(Config{}, try admitFolders(big, fixture));
    try std.testing.expectError(error.DraftForAnotherTarget, admitFolders(nine, fixture));
    var small = try a.dupe(u8, fixture);
    defer a.free(small);
    const edits = [_][2][]const u8{ .{ "\"hidden_size\":5120", "\"hidden_size\":4096" }, .{ "\"intermediate_size\":17408", "\"intermediate_size\":12288" }, .{ "\"num_target_layers\":64", "\"num_target_layers\":32" }, .{ "[5,19,33,47,61]", "[1,8,15,22,29]" } };
    for (edits) |e| {
        const next = try std.mem.replaceOwned(u8, a, small, e[0], e[1]);
        a.free(small);
        small = next;
    }
    try std.testing.expectError(error.NotStockDFlash2, admitFolders(nine, small));
}
// z-lab/Qwen3.5-9B-DFlash's config.json, as published (Ornith-1.5-9B-DFlash publishes the same file).
const v1_fixture =
    \\{"architectures":["DFlashDraftModel"],"attention_bias":false,"attention_dropout":0.0,
    \\"auto_map":{"AutoModel":"dflash.DFlashDraftModel"},"bos_token_id":null,
    \\"dflash_config":{"block_size":16,"mask_token_id":248077,"target_layer_ids":[1,5,9,13,17,21,25,29]},
    \\"dtype":"bfloat16","eos_token_id":248044,"head_dim":128,"hidden_act":"silu","hidden_size":4096,
    \\"initializer_range":0.02,"intermediate_size":12288,
    \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
    \\"max_position_embeddings":262144,"max_window_layers":6,"model_type":"qwen3","num_attention_heads":32,
    \\"num_hidden_layers":6,"num_key_value_heads":8,"num_target_layers":32,"pad_token_id":null,"rms_norm_eps":1e-06,
    \\"rope_parameters":{"rope_theta":10000000,"rope_type":"default"},"sliding_window":4096,"tie_word_embeddings":false,
    \\"transformers_version":"5.7.0","use_cache":true,"use_sliding_window":true,"vocab_size":248320}
;
test "the 9B target takes its own DFlash v1 drafter, and the 27B target refuses it by name" {
    const a = std.testing.allocator;
    const nine = try targetJson(a, 4096, 12288, 32, 16, 32);
    defer a.free(nine);
    const big = try targetJson(a, 5120, 17408, 64, 24, 48);
    defer a.free(big);
    try std.testing.expectEqual(Variant.v1, (try admitFolders(nine, v1_fixture)).variant);
    try std.testing.expectError(error.DraftForAnotherTarget, admitFolders(big, v1_fixture));
}
test "a DFlash v1 config reads its own taps, layer kinds, block and window" {
    const c = try parse(std.testing.allocator, v1_fixture);
    try std.testing.expectEqual(Variant.v1, c.variant);
    try std.testing.expectEqual(@as(u32, 8), c.tap_count);
    try std.testing.expectEqualSlices(u32, &.{ 1, 5, 9, 13, 17, 21, 25, 29 }, c.taps[0..8]);
    try std.testing.expectEqual(@as(u32, 16), c.block);
    try std.testing.expectEqual(@as(u32, 15), c.depth());
    try std.testing.expectEqual(@as(u32, 248077), c.mask);
    try std.testing.expectEqual(@as(u32, 4096), c.window);
    try std.testing.expectEqual(@as(u32, 4096), c.hidden);
    try std.testing.expectEqual(@as(u32, 12288), c.intermediate);
    try std.testing.expectEqual(@as(u32, 6), c.layers);
    try std.testing.expectEqual(@as(u32, 32), c.target_layers);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false, false, true }, c.full[0..6]);
    try std.testing.expectEqual(@as(u32, 32768), c.tapWidth());
    try std.testing.expectEqual(@as(u32, 0), c.rank);
    try std.testing.expectEqual(@as(u32, 16), c.topk);
}
test "a DFlash v1 draft counts eleven tensors a layer and seven linears a layer beside fc" {
    const c = try parse(std.testing.allocator, v1_fixture);
    try std.testing.expectEqual(@as(usize, 69), c.tensorCount());
    try std.testing.expectEqual(@as(usize, 43), c.linearCount());
}
test "a DFlash v1 config whose taps fall past the target or repeat is refused" {
    const a = std.testing.allocator;
    const past = try std.mem.replaceOwned(u8, a, v1_fixture, "25,29]", "25,32]");
    defer a.free(past);
    try std.testing.expectError(error.BadDraftConfig, parse(a, past));
    const repeated = try std.mem.replaceOwned(u8, a, v1_fixture, "25,29]", "25,25]");
    defer a.free(repeated);
    try std.testing.expectError(error.UnsupportedDraftTaps, parse(a, repeated));
}
test "a DFlash v1 config that asks for a causal draft is refused" {
    const a = std.testing.allocator;
    const causal = try std.mem.replaceOwned(u8, a, v1_fixture, "\"attention_bias\":false", "\"attention_bias\":false,\"is_causal\":true");
    defer a.free(causal);
    try std.testing.expectError(error.BadDraftConfig, parse(a, causal));
}
test "the stock DFlash2 config keeps its five taps and twenty-five-thousand-wide rows" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(Variant.dflash2, c.variant);
    try std.testing.expectEqual(@as(u32, 5), c.tap_count);
    try std.testing.expectEqual(@as(u32, 25600), c.tapWidth());
}
