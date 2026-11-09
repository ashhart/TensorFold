//! DSpine (Tencent's code-bridge block drafter) for any backend: its config contract and the exact-sender table walk.
const std = @import("std");
const bf16 = @import("bf16_round");

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

/// A checkpoint tensor's dimension in the config's terms (`early`: the layers before the last).
pub const Dim = enum { hidden, intermediate, vocab, code, rank, layers, early, q_width, kv_width, head_dim, tap_width };
pub const Kind = enum { bf16, i64 };
/// One checkpoint tensor: its name (bridge names under bridge_prefix, layer names under "layers.{i}."), dtype and dims.
pub const Tensor = struct { name: []const u8, kind: Kind = .bf16, dims: []const Dim };
pub const bridge_prefix = "selected_prefix_bridge.";
/// The tensors outside the bridge and the layers; the i64 ones are version scalars the reference requires at 1.
pub const globals = [_]Tensor{
    .{ .name = "fc.weight", .dims = &.{ .hidden, .tap_width } },
    .{ .name = "hidden_norm.weight", .dims = &.{.hidden} },
    .{ .name = "norm.weight", .dims = &.{.hidden} },
    .{ .name = "draft_block_causal_version", .kind = .i64, .dims = &.{} },
    .{ .name = "last_write_refine_version", .kind = .i64, .dims = &.{} },
};
/// The code bridge: reads [early, code, hidden] then the last's, writes [layers, hidden, rank], gates [rank, code].
pub const bridge_tensors = [_]Tensor{
    .{ .name = "codebook", .dims = &.{ .vocab, .code } },
    .{ .name = "early_read_weights", .dims = &.{ .early, .code, .hidden } },
    .{ .name = "read_weight", .dims = &.{ .code, .hidden } },
    .{ .name = "feature_write_weights", .dims = &.{ .layers, .hidden, .rank } },
    .{ .name = "gate.weight", .dims = &.{ .rank, .code } },
    .{ .name = "gate.bias", .dims = &.{.rank} },
    .{ .name = "predecessor_gate.weight", .dims = &.{ .rank, .code } },
    .{ .name = "prev.weight", .dims = &.{ .rank, .code } },
    .{ .name = "out.weight", .dims = &.{ .code, .rank } },
    .{ .name = "independent_write_version", .kind = .i64, .dims = &.{} },
    .{ .name = "terminal_disabled_version", .kind = .i64, .dims = &.{} },
};
/// Each body layer's tensors.
pub const layer_tensors = [_]Tensor{
    .{ .name = "input_layernorm.weight", .dims = &.{.hidden} },
    .{ .name = "post_attention_layernorm.weight", .dims = &.{.hidden} },
    .{ .name = "self_attn.q_proj.weight", .dims = &.{ .q_width, .hidden } },
    .{ .name = "self_attn.k_proj.weight", .dims = &.{ .kv_width, .hidden } },
    .{ .name = "self_attn.v_proj.weight", .dims = &.{ .kv_width, .hidden } },
    .{ .name = "self_attn.o_proj.weight", .dims = &.{ .hidden, .q_width } },
    .{ .name = "self_attn.q_norm.weight", .dims = &.{.head_dim} },
    .{ .name = "self_attn.k_norm.weight", .dims = &.{.head_dim} },
    .{ .name = "mlp.gate_proj.weight", .dims = &.{ .intermediate, .hidden } },
    .{ .name = "mlp.up_proj.weight", .dims = &.{ .intermediate, .hidden } },
    .{ .name = "mlp.down_proj.weight", .dims = &.{ .hidden, .intermediate } },
};

/// The checkpoint's tensor count: 71 for five layers.
pub fn tensorCount(layers: usize) usize {
    return globals.len + bridge_tensors.len + layer_tensors.len * layers;
}

/// A dimension's size for `s`: a Shape, or any backend config with the same fields.
pub fn size(s: anytype, d: Dim) usize {
    return switch (d) {
        .hidden => s.hidden,
        .intermediate => s.intermediate,
        .vocab => s.vocab,
        .code => s.code,
        .rank => s.rank,
        .layers => s.layers,
        .early => s.layers - 1,
        .q_width => s.heads * s.head_dim,
        .kv_width => s.kv_heads * s.head_dim,
        .head_dim => s.head_dim,
        .tap_width => s.hidden * s.n_taps,
    };
}

/// A tensor's dims for `s`, in `out`.
pub fn dims(s: anytype, t: Tensor, out: *[3]usize) []const usize {
    for (t.dims, out[0..t.dims.len]) |d, *o| o.* = size(s, d);
    return out[0..t.dims.len];
}

/// The tensor a full checkpoint name stands for (a layer's under layers.{i} for i below `layers`); null: none.
pub fn find(name: []const u8, layers: usize) ?Tensor {
    for (globals) |t| if (std.mem.eql(u8, t.name, name)) return t;
    if (std.mem.startsWith(u8, name, bridge_prefix)) {
        for (bridge_tensors) |t| if (std.mem.eql(u8, t.name, name[bridge_prefix.len..])) return t;
        return null;
    }
    if (!std.mem.startsWith(u8, name, "layers.")) return null;
    const rest = name["layers.".len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const i = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
    if (i >= layers or (dot > 1 and rest[0] == '0')) return null;
    for (layer_tensors) |t| if (std.mem.eql(u8, t.name, rest[dot + 1 ..])) return t;
    return null;
}

/// Every name the checkpoint holds, each once: globals, the bridge's, then each layer's.
pub fn names(gpa: std.mem.Allocator, layers: usize) ![][]const u8 {
    const result = try gpa.alloc([]const u8, tensorCount(layers));
    var made: usize = 0;
    errdefer freeNames(gpa, result[0..made]);
    for (globals) |t| {
        result[made] = try gpa.dupe(u8, t.name);
        made += 1;
    }
    for (bridge_tensors) |t| {
        result[made] = try std.mem.concat(gpa, u8, &.{ bridge_prefix, t.name });
        made += 1;
    }
    for (0..layers) |i| for (layer_tensors) |t| {
        result[made] = try std.fmt.allocPrint(gpa, "layers.{d}.{s}", .{ i, t.name });
        made += 1;
    };
    return result;
}

pub fn freeNames(gpa: std.mem.Allocator, list: []const []const u8) void {
    for (list) |name| gpa.free(name);
    gpa.free(list);
}

/// TG's device input for column k: bf16(norm[h] * W_write[last][h][k]); TG[v][k] = target head row v . it.
pub fn tgInput(norm: []align(1) const u16, write_last: []align(1) const u16, rank: usize, k: usize, out: []u16) void {
    for (out, norm, 0..) |*o, w, h| o.* = bf16.bits(bf16.value(w) * bf16.value(write_last[h * rank + k]));
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
