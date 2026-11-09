//! DSpine's 71 checkpoint tensors as BF16 borrows: the plain five-layer body, fc, norms and the code bridge.
const std = @import("std");
const ck = @import("../checkpoint.zig");
const config = @import("config.zig");
const Shape = config.Shape;
pub const Tensor = ck.Tensor;
pub const Attention = struct { q: Tensor, k: Tensor, v: Tensor, o: Tensor, q_norm: Tensor, k_norm: Tensor };
pub const Mlp = struct { gate: Tensor, up: Tensor, down: Tensor };
pub const Layer = struct { input_norm: Tensor, post_norm: Tensor, attention: Attention, mlp: Mlp };
/// The bridge: reads [layers - 1, code, hidden] then the last's, writes [layers, hidden, rank], gates [rank, code].
pub const Bridge = struct { codebook: Tensor, early_read: Tensor, read: Tensor, write: Tensor, gate: Tensor, gate_bias: Tensor, predecessor_gate: Tensor, prev: Tensor };
pub const Inventory = struct { source: ck.Source, names: []const []const u8 };
pub const Graph = struct {
    gpa: std.mem.Allocator,
    shape: Shape,
    layers: []Layer,
    fusion: Tensor,
    hidden_norm: Tensor,
    norm: Tensor,
    bridge: Bridge,
    embed: ck.Linear,
    pub fn deinit(g: *Graph) void {
        g.gpa.free(g.layers);
        g.* = undefined;
    }
    /// One layer's read: early_read_weights[l], or read_weight on the last layer, [code, hidden] bf16.
    pub fn readOf(g: Graph, l: usize) []const u8 {
        const n = g.shape.code * g.shape.hidden * 2;
        return if (l + 1 == g.shape.layers) g.bridge.read.bytes else g.bridge.early_read.bytes[l * n ..][0..n];
    }
    /// One layer's write: feature_write_weights[l], [hidden, rank] bf16.
    pub fn writeOf(g: Graph, l: usize) []const u8 {
        const n = g.shape.hidden * g.shape.rank * 2;
        return g.bridge.write.bytes[l * n ..][0..n];
    }
};
const suffixes = [_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight" };
const bridge_names = [_][]const u8{ "codebook", "early_read_weights", "read_weight", "feature_write_weights", "gate.weight", "gate.bias", "predecessor_gate.weight", "prev.weight", "out.weight", "independent_write_version", "terminal_disabled_version" };
const globals = [_][]const u8{ "fc.weight", "hidden_norm.weight", "norm.weight", "draft_block_causal_version", "last_write_refine_version" };
/// Versions the reference requires at 1: Tencent's loader and DSpineCodeBridge.prepare refuse any other.
const versions = [_][]const u8{ "draft_block_causal_version", "last_write_refine_version", "selected_prefix_bridge.independent_write_version", "selected_prefix_bridge.terminal_disabled_version" };

pub fn tensorCount(s: Shape) usize {
    return globals.len + bridge_names.len + suffixes.len * s.layers;
}

pub fn names(gpa: std.mem.Allocator, s: Shape) ![][]const u8 {
    const result = try gpa.alloc([]const u8, tensorCount(s));
    var made: usize = 0;
    errdefer {
        for (result[0..made]) |name| gpa.free(name);
        gpa.free(result);
    }
    for (globals) |name| {
        result[made] = try gpa.dupe(u8, name);
        made += 1;
    }
    for (bridge_names) |name| {
        result[made] = try std.fmt.allocPrint(gpa, "selected_prefix_bridge.{s}", .{name});
        made += 1;
    }
    for (0..s.layers) |i| for (suffixes) |suffix| {
        result[made] = try std.fmt.allocPrint(gpa, "layers.{d}.{s}", .{ i, suffix });
        made += 1;
    };
    return result;
}

pub fn freeNames(gpa: std.mem.Allocator, list: [][]const u8) void {
    for (list) |name| gpa.free(name);
    gpa.free(list);
}

fn inventory(gpa: std.mem.Allocator, list: []const []const u8, s: Shape) !void {
    const expected = try names(gpa, s);
    defer freeNames(gpa, expected);
    if (list.len != expected.len) return error.DraftTensorCoverage;
    var seen = std.StringHashMap(void).init(gpa);
    defer seen.deinit();
    for (list) |name| {
        if ((try seen.getOrPut(name)).found_existing) return error.DraftTensorCoverage;
        for (expected) |wanted| {
            if (std.mem.eql(u8, wanted, name)) break;
        } else return error.DraftTensorCoverage;
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

fn bridgeTensor(src: ck.Source, name: []const u8, dims: []const usize) !Tensor {
    var full: [160]u8 = undefined;
    return get(src, try std.fmt.bufPrint(&full, "selected_prefix_bridge.{s}", .{name}), dims);
}

/// Every version scalar is I64 1, and the terminal projection out.weight is all zero, as the reference demands.
fn contract(src: ck.Source, s: Shape) !void {
    for (versions) |name| {
        const t = try src.get(name);
        if (t.dtype != .i64) return error.DraftNativeDType;
        try ck.shape(t, &.{});
        if (std.mem.readInt(i64, t.bytes[0..8], .little) != 1) return error.UnsupportedDSpineVersion;
    }
    const out = try bridgeTensor(src, "out.weight", &.{ s.code, s.rank });
    for (std.mem.bytesAsSlice(u16, out.bytes)) |v| if (v & 0x7fff != 0) return error.DSpineTerminalNonzero;
}

fn target(p: ck.Linear, s: Shape) !void {
    switch (p) {
        .dense => |t| {
            if (!ck.floating(t.dtype)) return error.TargetBinding;
            try ck.shape(t, &.{ s.vocab, s.hidden });
        },
        .quantized => |q| _ = try ck.validate(q.weight, q.scales, q.biases, q.format, s.vocab, s.hidden),
    }
}

pub fn load(gpa: std.mem.Allocator, source: Inventory, s: Shape, embed: ck.Linear) !Graph {
    try config.pin(s);
    try inventory(gpa, source.names, s);
    try target(embed, s);
    const src = source.source;
    try contract(src, s);
    const h = s.hidden;
    const q = config.qWidth(s);
    const kv = config.kvWidth(s);
    const layers = try gpa.alloc(Layer, s.layers);
    errdefer gpa.free(layers);
    for (layers, 0..) |*l, i| l.* = .{
        .input_norm = try layerTensor(src, i, "input_layernorm.weight", &.{h}),
        .post_norm = try layerTensor(src, i, "post_attention_layernorm.weight", &.{h}),
        .attention = .{
            .q = try layerTensor(src, i, "self_attn.q_proj.weight", &.{ q, h }),
            .k = try layerTensor(src, i, "self_attn.k_proj.weight", &.{ kv, h }),
            .v = try layerTensor(src, i, "self_attn.v_proj.weight", &.{ kv, h }),
            .o = try layerTensor(src, i, "self_attn.o_proj.weight", &.{ h, q }),
            .q_norm = try layerTensor(src, i, "self_attn.q_norm.weight", &.{s.head_dim}),
            .k_norm = try layerTensor(src, i, "self_attn.k_norm.weight", &.{s.head_dim}),
        },
        .mlp = .{
            .gate = try layerTensor(src, i, "mlp.gate_proj.weight", &.{ s.intermediate, h }),
            .up = try layerTensor(src, i, "mlp.up_proj.weight", &.{ s.intermediate, h }),
            .down = try layerTensor(src, i, "mlp.down_proj.weight", &.{ h, s.intermediate }),
        },
    };
    const bridge: Bridge = .{
        .codebook = try bridgeTensor(src, "codebook", &.{ s.vocab, s.code }),
        .early_read = try bridgeTensor(src, "early_read_weights", &.{ s.layers - 1, s.code, h }),
        .read = try bridgeTensor(src, "read_weight", &.{ s.code, h }),
        .write = try bridgeTensor(src, "feature_write_weights", &.{ s.layers, h, s.rank }),
        .gate = try bridgeTensor(src, "gate.weight", &.{ s.rank, s.code }),
        .gate_bias = try bridgeTensor(src, "gate.bias", &.{s.rank}),
        .predecessor_gate = try bridgeTensor(src, "predecessor_gate.weight", &.{ s.rank, s.code }),
        .prev = try bridgeTensor(src, "prev.weight", &.{ s.rank, s.code }),
    };
    return .{ .gpa = gpa, .shape = s, .layers = layers, .fusion = try get(src, "fc.weight", &.{ h, config.tapWidth(s) }), .hidden_norm = try get(src, "hidden_norm.weight", &.{h}), .norm = try get(src, "norm.weight", &.{h}), .bridge = bridge, .embed = embed };
}

const testing = std.testing;

test "DSpine inventory: the checkpoint's 71 names exactly, a renamed or missing one refused" {
    const s = try config.parse(testing.allocator, @embedFile("testdata/config.json"));
    const list = try names(testing.allocator, s);
    defer freeNames(testing.allocator, list);
    try testing.expectEqual(@as(usize, 71), list.len);
    try inventory(testing.allocator, list, s);
    const previous = list[0];
    list[0] = list[1];
    defer list[0] = previous;
    try testing.expectError(error.DraftTensorCoverage, inventory(testing.allocator, list, s));
    try testing.expectError(error.DraftTensorCoverage, inventory(testing.allocator, list[1..], s));
}
