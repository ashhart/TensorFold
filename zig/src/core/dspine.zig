//! DSpine (Tencent's code-bridge block drafter) for any backend: its config contract and the exact-sender table walk.
const std = @import("std");

/// The serial second pass's candidates a row (head_topk), as every backend's table fixes them.
pub const head_topk = 16;
/// The widest sliding window admitted.
pub const most_window = 8192;

/// A DSpine checkpoint's contract, read from its config.json alone.
pub const Shape = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    block: usize,
    mask: u32,
    window: usize, // sliding_window: a block row sees context rows at most window - 1 behind it
    code: usize, // the bridge's code width
    rank: usize, // its message width
    topk: usize,
    eps: f32,
    bridge_eps: f32,
    theta: f64,
    target_layers: usize,
    taps: [8]u32 = @splat(0),
    n_taps: usize = 0,

    pub fn tapIds(s: *const Shape) []const u32 {
        return s.taps[0..s.n_taps];
    }
};

/// Whether a parsed config.json declares DSpine: a dflash_config with a selected-prefix code bridge.
pub fn declared(root: std.json.Value) bool {
    if (root != .object) return false;
    const d = root.object.get("dflash_config") orelse return false;
    return d == .object and d.object.get("selected_prefix_code_bridge") != null;
}

fn object(o: std.json.ObjectMap, key: []const u8) !std.json.ObjectMap {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return if (v == .object) v.object else error.BadDraftConfig;
}

fn text(o: std.json.ObjectMap, key: []const u8, want: []const u8) !void {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .string) return error.BadDraftConfig;
    if (!std.mem.eql(u8, v.string, want)) return error.UnsupportedDraftConfig;
}

fn flag(o: std.json.ObjectMap, key: []const u8, want: bool) !void {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .bool) return error.BadDraftConfig;
    if (v.bool != want) return error.UnsupportedDraftConfig;
}

fn int(o: std.json.ObjectMap, key: []const u8) !usize {
    const v = o.get(key) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer <= 0) return error.BadDraftConfig;
    return @intCast(v.integer);
}

fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v: f64 = switch (o.get(key) orelse return error.BadDraftConfig) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.BadDraftConfig,
    };
    return if (std.math.isFinite(v) and v > 0) v else error.BadDraftConfig;
}

/// The block-causal bridge with the exact-sender serial table, all-sliding plain-rope layers; anything else refused.
pub fn parse(root: std.json.Value) !Shape {
    if (root != .object) return error.BadDraftConfig;
    const o = root.object;
    const arch = o.get("architectures") orelse return error.BadDraftConfig;
    if (arch != .array or arch.array.items.len != 1 or arch.array.items[0] != .string) return error.BadDraftConfig;
    if (!std.mem.eql(u8, arch.array.items[0].string, "DFlashDraftModel")) return error.UnsupportedDraftConfig;
    const d = try object(o, "dflash_config");
    const bridge = try object(d, "selected_prefix_code_bridge");
    const second = try object(o, "code_bridge_second_pass");
    const refine = try object(d, "selected_prefix_last_write_refine");
    try text(second, "mode", "serial_table");
    try text(refine, "kind", "exact_sender");
    try text(d, "selected_prefix_decode_mode", "serial");
    try text(d, "selected_prefix_feature_write_mode", "independent");
    try flag(d, "draft_block_causal", true);
    try flag(d, "selected_prefix_feature_writes", true);
    try flag(d, "selected_prefix_terminal", false);
    try flag(o, "is_causal", true);
    try flag(o, "use_sliding_window", true);
    const rope = try object(o, "rope_parameters");
    if (rope.get("rope_type")) |v| if (v != .string or !std.mem.eql(u8, v.string, "default")) return error.UnsupportedDraftConfig;
    if (o.get("rope_scaling")) |v| if (v != .null) return error.UnsupportedDraftConfig;
    var s: Shape = .{
        .hidden = try int(o, "hidden_size"),
        .intermediate = try int(o, "intermediate_size"),
        .layers = try int(o, "num_hidden_layers"),
        .vocab = try int(o, "vocab_size"),
        .heads = try int(o, "num_attention_heads"),
        .kv_heads = try int(o, "num_key_value_heads"),
        .head_dim = try int(o, "head_dim"),
        .block = try int(o, "block_size"),
        .mask = @intCast(try int(d, "mask_token_id")),
        .window = try int(o, "sliding_window"),
        .code = try int(bridge, "code_size"),
        .rank = try int(bridge, "rank"),
        .topk = try int(second, "head_topk"),
        .eps = @floatCast(try num(o, "rms_norm_eps")),
        .bridge_eps = @floatCast(try num(bridge, "eps")),
        .theta = try num(rope, "rope_theta"),
        .target_layers = try int(o, "num_target_layers"),
    };
    const types = o.get("layer_types") orelse return error.BadDraftConfig;
    if (types != .array or types.array.items.len != s.layers) return error.BadDraftConfig;
    for (types.array.items) |t| {
        if (t != .string) return error.BadDraftConfig;
        if (!std.mem.eql(u8, t.string, "sliding_attention")) return error.UnsupportedDraftConfig;
    }
    const writes = d.get("selected_prefix_feature_layers") orelse return error.BadDraftConfig;
    if (writes != .array or writes.array.items.len != s.layers) return error.UnsupportedDraftConfig;
    for (writes.array.items, 1..) |v, i| if (v != .integer or v.integer != i) return error.UnsupportedDraftConfig;
    const taps = d.get("target_layer_ids") orelse return error.BadDraftConfig;
    if (taps != .array or taps.array.items.len == 0 or taps.array.items.len > s.taps.len) return error.BadDraftConfig;
    for (taps.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0 or v.integer >= s.target_layers) return error.BadDraftConfig;
        s.taps[i] = @intCast(v.integer);
    }
    s.n_taps = taps.array.items.len;
    if (try int(refine, "head_topk") != s.topk or s.topk != head_topk) return error.UnsupportedDraftConfig;
    if (s.block != 16 or s.mask >= s.vocab or s.heads % s.kv_heads != 0 or s.head_dim % 2 != 0) return error.UnsupportedDraftConfig;
    if (s.code % 64 != 0 or s.rank % 64 != 0 or s.hidden % 64 != 0) return error.UnsupportedDraftConfig;
    if (s.window < s.block or s.window > most_window) return error.UnsupportedDraftConfig;
    return s;
}

/// A served window (TF_DSPINE_WINDOW): a block up to the checkpoint's sliding_window, never wider; 0 keeps it.
pub fn narrowed(s: Shape, window: usize) error{DSpineWindowOutOfRange}!Shape {
    if (window == 0) return s;
    if (window < s.block or window > s.window) return error.DSpineWindowOutOfRange;
    var out = s;
    out.window = window;
    return out;
}

/// Row r's sender for `pred`: 0 on row 0 (all hold the pending token), else its index in the row before's candidates.
pub fn sender(r: usize, previous: []align(1) const u32, pred: u32) ?usize {
    if (r == 0) return 0;
    for (previous, 0..) |id, j| if (id == pred) return j;
    return null;
}

/// The greedy chain through `table` [row][sender][candidate] over `candidates` [row][k]: ties to the earlier one.
pub fn chain(candidates: []const u32, table: []const f32, k: usize, out: []u32) void {
    var s: usize = 0;
    for (out, 0..) |*o, r| {
        const row = table[(r * k + s) * k ..][0..k];
        var best: usize = 0;
        for (row, 0..) |v, j| if (v > row[best]) {
            best = j;
        };
        o.* = candidates[r * k + best];
        s = best;
    }
}

test {
    _ = @import("dspine_test.zig");
}
