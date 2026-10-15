//! The checkpoint's config.json against what the CUDA path is built for: another shape or setting is refused at load.

const std = @import("std");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");

/// One config field and the value the kernels and buffers assume.
const Want = struct { name: []const u8, value: i64 };

/// The text config's shapes the CUDA path's constants hold.
const shapes = [_]Want{
    .{ .name = "hidden_size", .value = wt.dims },
    .{ .name = "num_hidden_layers", .value = wt.layers_n },
    .{ .name = "full_attention_interval", .value = 4 },
    .{ .name = "vocab_size", .value = wt.vocab },
    .{ .name = "num_attention_heads", .value = wt.q_heads },
    .{ .name = "num_key_value_heads", .value = wt.kv_heads },
    .{ .name = "head_dim", .value = wt.head_dim },
    .{ .name = "num_experts", .value = wt.experts_n },
    .{ .name = "num_experts_per_tok", .value = wt.slots - 1 },
    .{ .name = "moe_intermediate_size", .value = wt.moe_w },
    .{ .name = "shared_expert_intermediate_size", .value = wt.moe_w },
    .{ .name = "hc_count", .value = wt.streams },
    .{ .name = "hc_lowrank", .value = wt.low },
    .{ .name = "linear_num_key_heads", .value = gdn.nk },
    .{ .name = "linear_num_value_heads", .value = gdn.nv },
    .{ .name = "linear_key_head_dim", .value = gdn.dk },
    .{ .name = "linear_value_head_dim", .value = gdn.dv },
    .{ .name = "linear_conv_kernel_dim", .value = gdn.taps },
    .{ .name = "indexer_n_heads", .value = wt.index_heads },
    .{ .name = "indexer_head_dim", .value = wt.index_dim },
    .{ .name = "indexer_compress_ratio", .value = wt.index_ratio },
    .{ .name = "mtp_num_hidden_layers", .value = 1 },
};

/// RoPE's base: Scratch.inv's 10,000,000 (the Python engine's default when the config names none).
pub const rope_theta: f64 = 10_000_000;
/// The rotated share of a head: 64 of 256 features, `wt.half` pairs.
pub const partial_rotary: f64 = @as(f64, @floatFromInt(2 * wt.half)) / @as(f64, @floatFromInt(wt.head_dim));

/// Settings the kernels fix, absent meaning the value they assume: a whole number, a float, a word or a flag.
const Fixed = struct { name: []const u8, value: union(enum) { int: i64, float: f64, word: []const u8, flag: bool } };

/// The text config's semantic settings the evaluation hard-codes (norms, gates, activations, layer roles, n-grams).
const fixed = [_]Fixed{
    .{ .name = "output_gate_type", .value = .{ .word = "sigmoid" } },
    .{ .name = "hidden_act", .value = .{ .word = "silu" } },
    .{ .name = "attention_bias", .value = .{ .flag = false } },
    .{ .name = "attn_output_gate", .value = .{ .flag = true } },
    .{ .name = "norm_topk_prob", .value = .{ .flag = true } },
    .{ .name = "mamba_ssm_dtype", .value = .{ .word = "float32" } },
    .{ .name = "tie_word_embeddings", .value = .{ .flag = false } },
    .{ .name = "mtp_use_dedicated_embeddings", .value = .{ .flag = false } },
    .{ .name = "indexer_budget", .value = .{ .int = wt.index_top * wt.index_ratio } },
    .{ .name = "indexer_kv_heads", .value = .{ .int = 1 } },
    .{ .name = "ple_embed_dim", .value = .{ .int = wt.dims } },
    .{ .name = "ple_conv_kernel_size", .value = .{ .int = 4 } },
    .{ .name = "ngram_size", .value = .{ .int = 3 } },
};

fn int(o: std.json.ObjectMap, name: []const u8) ?i64 {
    const v = o.get(name) orelse return null;
    return if (v == .integer) v.integer else null;
}

fn float(v: std.json.Value) ?f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn object(o: std.json.ObjectMap, name: []const u8) ?std.json.ObjectMap {
    const v = o.get(name) orelse return null;
    return if (v == .object) v.object else null;
}

/// A value present and not null.
fn given(o: std.json.ObjectMap, name: []const u8) ?std.json.Value {
    const v = o.get(name) orelse return null;
    return if (v == .null) null else v;
}

fn holds(v: std.json.Value, want: @FieldType(Fixed, "value")) bool {
    return switch (want) {
        .int => |i| v == .integer and v.integer == i,
        .float => |f| float(v) == f,
        .word => |w| v == .string and std.mem.eql(u8, v.string, w),
        .flag => |b| v == .bool and v.bool == b,
    };
}

/// The first config value that differs from what the CUDA path serves, or null when every one matches.
pub fn mismatch(text: std.json.ObjectMap, quant: ?std.json.ObjectMap) ?[]const u8 {
    const q = quant orelse return "quantization";
    if (int(q, "bits") != 4) return "quantization.bits";
    if (int(q, "group_size") != 32) return "quantization.group_size";
    if (q.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return "quantization.mode";
    for (shapes) |w| if (int(text, w.name) != w.value) return w.name;
    return semantics(text);
}

/// The first semantic setting the kernels would ignore: eps, RoPE, gates, activations, layer roles, the head's layer.
pub fn semantics(text: std.json.ObjectMap) ?[]const u8 {
    const eps = float(given(text, "rms_norm_eps") orelse return "rms_norm_eps") orelse return "rms_norm_eps";
    if (@as(f32, @floatCast(eps)) != wt.eps) return "rms_norm_eps"; // the kernels take eps as fp32
    for (fixed) |f| if (given(text, f.name)) |v| if (!holds(v, f.value)) return f.name;
    if (rope(text)) |field| return field;
    if (given(text, "layer_types")) |v| {
        if (v != .array or v.array.items.len != wt.layers_n) return "layer_types";
        for (v.array.items, 0..) |kind, i| {
            const want = if (wt.linearLayer(i)) "linear_attention" else "full_attention";
            if (!holds(kind, .{ .word = want })) return "layer_types";
        }
    }
    const ple = given(text, "ple_layer_ids") orelse return "ple_layer_ids";
    if (ple != .array or ple.array.items.len != 1 or !holds(ple.array.items[0], .{ .int = wt.ple_layer + 1 })) return "ple_layer_ids";
    if (object(text, "mtp")) |m| {
        if (given(m, "rope_theta")) |v| if (float(v) != rope_theta) return "mtp.rope_theta";
        if (given(m, "num_hidden_layers")) |v| if (!holds(v, .{ .int = 1 })) return "mtp.num_hidden_layers";
        if (given(m, "layer_types")) |v| {
            if (v != .array or v.array.items.len != 1 or !holds(v.array.items[0], .{ .word = "full_attention" })) return "mtp.layer_types";
        }
    }
    return null;
}

/// RoPE as the kernels apply it: base 10,000,000 on 64 of 256 features, no scaling (text positions only).
fn rope(text: std.json.ObjectMap) ?[]const u8 {
    if (given(text, "rope_scaling") != null) return "rope_scaling";
    const params = object(text, "rope_parameters");
    const theta = (if (params) |p| given(p, "rope_theta") else null) orelse given(text, "rope_theta");
    if (theta) |v| if (float(v) != rope_theta) return "rope_theta";
    const partial = (if (params) |p| given(p, "partial_rotary_factor") else null) orelse given(text, "partial_rotary_factor");
    if (partial) |v| if (float(v) != partial_rotary) return "partial_rotary_factor";
    const p = params orelse return null;
    for ([_][]const u8{ "rope_type", "type" }) |name| if (given(p, name)) |v| if (!holds(v, .{ .word = "default" })) return "rope_parameters.rope_type";
    if (given(p, "factor")) |v| if (float(v) != 1) return "rope_parameters.factor";
    return null;
}

/// Reads `dir`/config.json; error.UnsupportedCheckpoint names the field in a log line when one differs.
pub fn check(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.UnsupportedCheckpoint;
    const root = parsed.value.object;
    const text = if (root.get("text_config")) |v| (if (v == .object) v.object else root) else root;
    const quant = if (root.get("quantization")) |v| (if (v == .object) v.object else null) else null;
    var field = mismatch(text, quant);
    if (field == null) if (given(root, "tie_word_embeddings")) |v| if (!holds(v, .{ .flag = false })) {
        field = "tie_word_embeddings";
    };
    std.log.err("flashnext: config.json's {s} differs from the MLX 4-bit group-32 Flash Next the CUDA path serves", .{field orelse return});
    return error.UnsupportedCheckpoint;
}

/// The served config's text fields, as the checked-in checkpoint has them (a test's base to change one at a time).
fn served(gpa: std.mem.Allocator, text: *std.json.ObjectMap, rope_params: *std.json.ObjectMap, mtp: *std.json.ObjectMap, kinds: *std.json.Array, ple: *std.json.Array) !void {
    for (shapes) |w| try text.put(gpa, w.name, .{ .integer = w.value });
    try text.put(gpa, "rms_norm_eps", .{ .float = 1e-06 });
    try text.put(gpa, "output_gate_type", .{ .string = "sigmoid" });
    try text.put(gpa, "hidden_act", .{ .string = "silu" });
    try text.put(gpa, "attention_bias", .{ .bool = false });
    try text.put(gpa, "partial_rotary_factor", .{ .float = 0.25 });
    try text.put(gpa, "indexer_budget", .{ .integer = 2048 });
    try text.put(gpa, "ple_conv_kernel_size", .{ .integer = 4 });
    try text.put(gpa, "ngram_size", .{ .integer = 3 });
    for (0..wt.layers_n) |i| try kinds.append(.{ .string = if (wt.linearLayer(i)) "linear_attention" else "full_attention" });
    try text.put(gpa, "layer_types", .{ .array = kinds.* });
    try ple.append(.{ .integer = 2 });
    try text.put(gpa, "ple_layer_ids", .{ .array = ple.* });
    try rope_params.put(gpa, "rope_theta", .{ .integer = 10_000_000 });
    try rope_params.put(gpa, "partial_rotary_factor", .{ .float = 0.25 });
    try rope_params.put(gpa, "type", .{ .string = "default" });
    try text.put(gpa, "rope_parameters", .{ .object = rope_params.* });
    try mtp.put(gpa, "rope_theta", .{ .integer = 10_000_000 });
    try text.put(gpa, "mtp", .{ .object = mtp.* });
}

test "the served shapes pass and a 6-bit checkpoint or another width is named" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var text: std.json.ObjectMap = .empty;
    var rope_params: std.json.ObjectMap = .empty;
    var mtp: std.json.ObjectMap = .empty;
    var kinds: std.json.Array = .init(gpa);
    var ple: std.json.Array = .init(gpa);
    try served(gpa, &text, &rope_params, &mtp, &kinds, &ple);
    var quant: std.json.ObjectMap = .empty;
    try quant.put(gpa, "bits", .{ .integer = 4 });
    try quant.put(gpa, "group_size", .{ .integer = 32 });
    try std.testing.expectEqual(@as(?[]const u8, null), mismatch(text, quant));
    try quant.put(gpa, "bits", .{ .integer = 6 });
    try std.testing.expectEqualStrings("quantization.bits", mismatch(text, quant).?);
    try quant.put(gpa, "bits", .{ .integer = 4 });
    try text.put(gpa, "hidden_size", .{ .integer = 4096 });
    try std.testing.expectEqualStrings("hidden_size", mismatch(text, quant).?);
    try std.testing.expectEqualStrings("quantization", mismatch(text, null).?);
}

test "a changed epsilon, RoPE, gate, activation or layer role is refused by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const Change = struct { field: []const u8, value: std.json.Value, refused: []const u8 };
    const changes = [_]Change{
        .{ .field = "rms_norm_eps", .value = .{ .float = 1e-5 }, .refused = "rms_norm_eps" },
        .{ .field = "output_gate_type", .value = .{ .string = "silu" }, .refused = "output_gate_type" },
        .{ .field = "hidden_act", .value = .{ .string = "gelu" }, .refused = "hidden_act" },
        .{ .field = "attention_bias", .value = .{ .bool = true }, .refused = "attention_bias" },
        .{ .field = "attn_output_gate", .value = .{ .bool = false }, .refused = "attn_output_gate" },
        .{ .field = "norm_topk_prob", .value = .{ .bool = false }, .refused = "norm_topk_prob" },
        .{ .field = "rope_scaling", .value = .{ .string = "yarn" }, .refused = "rope_scaling" },
        .{ .field = "indexer_budget", .value = .{ .integer = 4096 }, .refused = "indexer_budget" },
        .{ .field = "ngram_size", .value = .{ .integer = 4 }, .refused = "ngram_size" },
        .{ .field = "ple_layer_ids", .value = .{ .array = .init(gpa) }, .refused = "ple_layer_ids" },
        .{ .field = "layer_types", .value = .{ .array = .init(gpa) }, .refused = "layer_types" },
    };
    for (changes) |c| {
        var text: std.json.ObjectMap = .empty;
        var rope_params: std.json.ObjectMap = .empty;
        var mtp: std.json.ObjectMap = .empty;
        var kinds: std.json.Array = .init(gpa);
        var ple: std.json.Array = .init(gpa);
        try served(gpa, &text, &rope_params, &mtp, &kinds, &ple);
        try std.testing.expectEqual(@as(?[]const u8, null), semantics(text));
        try text.put(gpa, c.field, c.value);
        try std.testing.expectEqualStrings(c.refused, semantics(text) orelse "accepted");
    }
}

test "each layer's label is exactly the role the kernels run: a sliding or misspelt one is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const full = 3; // the first full-attention layer
    try std.testing.expect(!wt.linearLayer(full) and wt.linearLayer(0));
    const Case = struct { layer: usize, label: []const u8 };
    const cases = [_]Case{
        .{ .layer = full, .label = "sliding_attention" },
        .{ .layer = full, .label = "attention" },
        .{ .layer = full, .label = "Full_Attention" },
        .{ .layer = wt.layers_n - 1, .label = "sliding_attention" },
        .{ .layer = 0, .label = "full_attention" },
        .{ .layer = 0, .label = "sliding_attention" },
    };
    for (cases) |c| {
        var text: std.json.ObjectMap = .empty;
        var rope_params: std.json.ObjectMap = .empty;
        var mtp: std.json.ObjectMap = .empty;
        var kinds: std.json.Array = .init(gpa);
        var ple: std.json.Array = .init(gpa);
        try served(gpa, &text, &rope_params, &mtp, &kinds, &ple);
        try std.testing.expectEqual(@as(?[]const u8, null), semantics(text));
        kinds.items[c.layer] = .{ .string = c.label };
        try text.put(gpa, "layer_types", .{ .array = kinds });
        try std.testing.expectEqualStrings("layer_types", semantics(text) orelse "accepted");
    }
}

test "RoPE's base and type inside rope_parameters, and the MTP layer's base, are checked too" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const Case = struct { in_mtp: bool, field: []const u8, value: std.json.Value, refused: []const u8 };
    const cases = [_]Case{
        .{ .in_mtp = false, .field = "rope_theta", .value = .{ .integer = 1_000_000 }, .refused = "rope_theta" },
        .{ .in_mtp = false, .field = "partial_rotary_factor", .value = .{ .float = 0.5 }, .refused = "partial_rotary_factor" },
        .{ .in_mtp = false, .field = "type", .value = .{ .string = "yarn" }, .refused = "rope_parameters.rope_type" },
        .{ .in_mtp = false, .field = "factor", .value = .{ .float = 4 }, .refused = "rope_parameters.factor" },
        .{ .in_mtp = true, .field = "rope_theta", .value = .{ .integer = 5_000_000 }, .refused = "mtp.rope_theta" },
        .{ .in_mtp = true, .field = "layer_types", .value = .{ .array = .init(gpa) }, .refused = "mtp.layer_types" },
    };
    for (cases) |c| {
        var text: std.json.ObjectMap = .empty;
        var rope_params: std.json.ObjectMap = .empty;
        var mtp: std.json.ObjectMap = .empty;
        var kinds: std.json.Array = .init(gpa);
        var ple: std.json.Array = .init(gpa);
        try served(gpa, &text, &rope_params, &mtp, &kinds, &ple);
        if (c.in_mtp) {
            try mtp.put(gpa, c.field, c.value);
            try text.put(gpa, "mtp", .{ .object = mtp });
        } else {
            try rope_params.put(gpa, c.field, c.value);
            try text.put(gpa, "rope_parameters", .{ .object = rope_params });
        }
        try std.testing.expectEqualStrings(c.refused, semantics(text) orelse "accepted");
    }
}
