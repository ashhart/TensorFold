//! Native DFlash tensors remain BF16 borrows; prepared quantized copies and the target embedding have separate owners.
const std = @import("std");
const ck = @import("../checkpoint.zig");
const Config = @import("config.zig").Config;
pub const Tensor = ck.Tensor;
pub const Attention = struct { q: Tensor, k: Tensor, v: Tensor, o: Tensor, q_norm: Tensor, k_norm: Tensor };
pub const Conv = struct { projection: Tensor, base: Tensor };
pub const Mlp = struct { gate: Tensor, up: Tensor, down: Tensor };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, attention: Attention, mlp: Mlp, attention_conv: Conv, mlp_conv: Conv };
pub const Inventory = struct { source: ck.Source, names: []const []const u8 };
pub const Graph = struct {
    gpa: std.mem.Allocator,
    config: Config,
    layers: []Layer,
    fusion: Tensor,
    hidden_norm: Tensor,
    norm: Tensor,
    hidden_projection: Tensor,
    predecessor: Tensor,
    successor: Tensor,
    embed: ck.Linear,
    pub fn deinit(g: *Graph) void {
        g.gpa.free(g.layers);
        g.* = undefined;
    }
    pub fn nativeBytes(g: Graph) !usize {
        var total: usize = 0;
        for ([_]Tensor{ g.fusion, g.hidden_norm, g.norm, g.hidden_projection, g.predecessor, g.successor }) |t| total = try std.math.add(usize, total, t.bytes.len);
        for (g.layers) |l| for ([_]Tensor{ l.input_norm, l.post_norm, l.attention.q, l.attention.k, l.attention.v, l.attention.o, l.attention.q_norm, l.attention.k_norm, l.mlp.gate, l.mlp.up, l.mlp.down, l.attention_conv.projection, l.attention_conv.base, l.mlp_conv.projection, l.mlp_conv.base }) |t| {
            total = try std.math.add(usize, total, t.bytes.len);
        };
        return total;
    }
};
const suffixes = [_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "attention_conv.kernel_projection.weight", "attention_conv.base_kernel", "mlp_conv.kernel_projection.weight", "mlp_conv.base_kernel" };
const globals = [_][]const u8{ "fc.weight", "hidden_norm.weight", "norm.weight", "candidate_selector.hidden_projection.weight", "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook" };
/// A DFlash v1 draft's own tensors: the first eleven suffixes a layer, and fc, hidden_norm and norm.
const v1_suffixes = suffixes[0..11];
const v1_globals = globals[0..3];
/// A tensor a v1 draft does not have (its conv and selector slots): no bytes, never bound to a kernel.
pub const absent = Tensor{ .dtype = .bf16, .rank = 0, .shape = @splat(1), .bytes = &.{} };
pub fn names(gpa: std.mem.Allocator, c: Config) ![][]const u8 {
    const result = try gpa.alloc([]const u8, c.tensorCount());
    var made: usize = 0;
    errdefer {
        for (result[0..made]) |name| gpa.free(name);
        gpa.free(result);
    }
    const v1 = c.variant == .v1;
    for (if (v1) v1_globals else &globals) |name| {
        result[made] = try gpa.dupe(u8, name);
        made += 1;
    }
    for (0..c.layers) |i| for (if (v1) v1_suffixes else &suffixes) |suffix| {
        result[made] = try std.fmt.allocPrint(gpa, "layers.{d}.{s}", .{ i, suffix });
        made += 1;
    };
    return result;
}
pub fn freeNames(gpa: std.mem.Allocator, list: [][]const u8) void {
    for (list) |name| gpa.free(name);
    gpa.free(list);
}
fn inventory(gpa: std.mem.Allocator, list: []const []const u8, c: Config) !void {
    const expected = try names(gpa, c);
    defer freeNames(gpa, expected);
    if (list.len != expected.len) return error.DraftTensorCoverage;
    var seen = std.StringHashMap(void).init(gpa);
    defer seen.deinit();
    for (list) |name| {
        if ((try seen.getOrPut(name)).found_existing) return error.DraftTensorCoverage;
        var found = false;
        for (expected) |wanted| if (std.mem.eql(u8, wanted, name)) {
            found = true;
            break;
        };
        if (!found) return error.DraftTensorCoverage;
    }
}
fn get(src: ck.Source, name: []const u8, dims: []const usize) !Tensor {
    const t = try src.get(name);
    if (t.dtype != .bf16) return error.DraftNativeDType;
    try ck.shape(t, dims);
    return t;
}
fn layerTensor(src: ck.Source, i: usize, suffix: []const u8, dims: []const usize) !Tensor {
    var name: [160]u8 = undefined;
    return get(src, try std.fmt.bufPrint(&name, "layers.{d}.{s}", .{ i, suffix }), dims);
}
fn target(p: ck.Linear, c: Config) !void {
    switch (p) {
        .dense => |t| {
            if (!ck.floating(t.dtype)) return error.TargetBinding;
            try ck.shape(t, &.{ c.vocab, c.hidden });
        },
        .quantized => |q| {
            _ = try ck.validate(q.weight, q.scales, q.biases, q.format, c.vocab, c.hidden);
        },
    }
}
pub fn load(gpa: std.mem.Allocator, source: Inventory, c: Config, embed: ck.Linear) !Graph {
    try c.check();
    try inventory(gpa, source.names, c);
    try target(embed, c);
    const src = source.source;
    const layers = try gpa.alloc(Layer, c.layers);
    errdefer gpa.free(layers);
    if (c.variant == .v1) return loadV1(gpa, src, c, layers, embed);
    for (layers, 0..) |*l, i| l.* = .{
        .input_norm = try layerTensor(src, i, "input_layernorm.weight", &.{c.hidden}),
        .post_norm = try layerTensor(src, i, "post_attention_layernorm.weight", &.{c.hidden}),
        .attention = .{ .q = try layerTensor(src, i, "self_attn.q_proj.weight", &.{ c.qWidth(), c.hidden }), .k = try layerTensor(src, i, "self_attn.k_proj.weight", &.{ c.kvWidth(), c.hidden }), .v = try layerTensor(src, i, "self_attn.v_proj.weight", &.{ c.kvWidth(), c.hidden }), .o = try layerTensor(src, i, "self_attn.o_proj.weight", &.{ c.hidden, c.qWidth() }), .q_norm = try layerTensor(src, i, "self_attn.q_norm.weight", &.{c.head_dim}), .k_norm = try layerTensor(src, i, "self_attn.k_norm.weight", &.{c.head_dim}) },
        .mlp = .{ .gate = try layerTensor(src, i, "mlp.gate_proj.weight", &.{ c.intermediate, c.hidden }), .up = try layerTensor(src, i, "mlp.up_proj.weight", &.{ c.intermediate, c.hidden }), .down = try layerTensor(src, i, "mlp.down_proj.weight", &.{ c.hidden, c.intermediate }) },
        .attention_conv = .{ .projection = try layerTensor(src, i, "attention_conv.kernel_projection.weight", &.{ c.dynamicWidth(), c.hidden }), .base = try layerTensor(src, i, "attention_conv.base_kernel", &.{ 2, 2, c.hidden }) },
        .mlp_conv = .{ .projection = try layerTensor(src, i, "mlp_conv.kernel_projection.weight", &.{ c.dynamicWidth(), c.hidden }), .base = try layerTensor(src, i, "mlp_conv.base_kernel", &.{ 2, 2, c.hidden }) },
    };
    return .{ .gpa = gpa, .config = c, .layers = layers, .fusion = try get(src, "fc.weight", &.{ c.hidden, c.tapWidth() }), .hidden_norm = try get(src, "hidden_norm.weight", &.{c.hidden}), .norm = try get(src, "norm.weight", &.{c.hidden}), .hidden_projection = try get(src, "candidate_selector.hidden_projection.weight", &.{ c.rank, c.hidden }), .predecessor = try get(src, "candidate_selector.predecessor_codebook", &.{ c.vocab, c.rank }), .successor = try get(src, "candidate_selector.successor_codebook", &.{ c.vocab, c.rank }), .embed = embed };
}

/// v1 layers: the qwen3 body alone; the conv and selector slots stay `absent`.
fn loadV1(gpa: std.mem.Allocator, src: ck.Source, c: Config, layers: []Layer, embed: ck.Linear) !Graph {
    const none = Conv{ .projection = absent, .base = absent };
    for (layers, 0..) |*l, i| l.* = .{
        .input_norm = try layerTensor(src, i, "input_layernorm.weight", &.{c.hidden}),
        .post_norm = try layerTensor(src, i, "post_attention_layernorm.weight", &.{c.hidden}),
        .attention = .{ .q = try layerTensor(src, i, "self_attn.q_proj.weight", &.{ c.qWidth(), c.hidden }), .k = try layerTensor(src, i, "self_attn.k_proj.weight", &.{ c.kvWidth(), c.hidden }), .v = try layerTensor(src, i, "self_attn.v_proj.weight", &.{ c.kvWidth(), c.hidden }), .o = try layerTensor(src, i, "self_attn.o_proj.weight", &.{ c.hidden, c.qWidth() }), .q_norm = try layerTensor(src, i, "self_attn.q_norm.weight", &.{c.head_dim}), .k_norm = try layerTensor(src, i, "self_attn.k_norm.weight", &.{c.head_dim}) },
        .mlp = .{ .gate = try layerTensor(src, i, "mlp.gate_proj.weight", &.{ c.intermediate, c.hidden }), .up = try layerTensor(src, i, "mlp.up_proj.weight", &.{ c.intermediate, c.hidden }), .down = try layerTensor(src, i, "mlp.down_proj.weight", &.{ c.hidden, c.intermediate }) },
        .attention_conv = none,
        .mlp_conv = none,
    };
    return .{ .gpa = gpa, .config = c, .layers = layers, .fusion = try get(src, "fc.weight", &.{ c.hidden, c.tapWidth() }), .hidden_norm = try get(src, "hidden_norm.weight", &.{c.hidden}), .norm = try get(src, "norm.weight", &.{c.hidden}), .hidden_projection = absent, .predecessor = absent, .successor = absent, .embed = embed };
}

test "a DFlash v1 inventory is its sixty-nine qwen3 names, no conv or selector" {
    const a = std.testing.allocator;
    var c = Config{ .variant = .v1, .hidden = 4096, .intermediate = 12288, .layers = 6, .target_layers = 32, .block = 16, .mask = 248077, .window = 4096, .group = 0, .conv = 0, .rank = 0, .topk = 16, .taps = .{ 1, 5, 9, 13, 17, 21, 25, 29 }, .tap_count = 8 };
    c.full[5] = true;
    try c.check();
    const list = try names(a, c);
    defer freeNames(a, list);
    try inventory(a, list, c);
    try std.testing.expectEqual(@as(usize, 69), list.len);
    for (list) |name| {
        try std.testing.expect(std.mem.indexOf(u8, name, "conv") == null);
        try std.testing.expect(std.mem.indexOf(u8, name, "selector") == null);
    }
    try std.testing.expectEqualStrings("fc.weight", list[0]);
    try std.testing.expectEqualStrings("layers.5.mlp.down_proj.weight", list[list.len - 1]);
}

test "inventory is81exact names including unsuffixed BF16codebooks" {
    const a = std.testing.allocator;
    const list = try names(a, .{});
    defer freeNames(a, list);
    try inventory(a, list, .{});
    try std.testing.expectEqual(@as(usize, 81), list.len);
    const previous = list[0];
    list[0] = list[1];
    defer list[0] = previous;
    try std.testing.expectError(error.DraftTensorCoverage, inventory(a, list, .{}));
}
