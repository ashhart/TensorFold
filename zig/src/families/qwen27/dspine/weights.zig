//! DSpine's 71 checkpoint tensors as BF16 borrows: the plain five-layer body, fc, norms and the code bridge.
const std = @import("std");
const core = @import("core");
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
const dspine = core.dspine;

pub fn tensorCount(s: Shape) usize {
    return dspine.tensorCount(s.layers);
}

pub fn names(gpa: std.mem.Allocator, s: Shape) ![][]const u8 {
    return dspine.names(gpa, s.layers);
}

pub fn freeNames(gpa: std.mem.Allocator, list: [][]const u8) void {
    dspine.freeNames(gpa, list);
}

fn inventory(gpa: std.mem.Allocator, list: []const []const u8, s: Shape) !void {
    if (list.len != tensorCount(s)) return error.DraftTensorCoverage;
    var seen = std.StringHashMap(void).init(gpa);
    defer seen.deinit();
    for (list) |name| {
        if ((try seen.getOrPut(name)).found_existing) return error.DraftTensorCoverage;
        _ = dspine.find(name, s.layers) orelse return error.DraftTensorCoverage;
    }
}

/// A tensor as core.dspine's table states it: its dtype and its dims for `s`.
fn get(src: ck.Source, s: Shape, name: []const u8) !Tensor {
    const want = dspine.find(name, s.layers) orelse return error.DraftTensorCoverage;
    const t = try src.get(name);
    if (t.dtype != @as(ck.DType, switch (want.kind) {
        .bf16 => .bf16,
        .i64 => .i64,
    })) return error.DraftNativeDType;
    var dims: [3]usize = undefined;
    try ck.shape(t, dspine.dims(s, want, &dims));
    return t;
}

fn layerTensor(src: ck.Source, s: Shape, i: usize, suffix: []const u8) !Tensor {
    var name: [160]u8 = undefined;
    return get(src, s, try std.fmt.bufPrint(&name, "layers.{d}.{s}", .{ i, suffix }));
}

fn bridgeTensor(src: ck.Source, s: Shape, name: []const u8) !Tensor {
    var full: [160]u8 = undefined;
    return get(src, s, try std.fmt.bufPrint(&full, "{s}{s}", .{ dspine.bridge_prefix, name }));
}

fn version(src: ck.Source, s: Shape, name: []const u8) !void {
    const t = try get(src, s, name);
    if (std.mem.readInt(i64, t.bytes[0..8], .little) != 1) return error.UnsupportedDSpineVersion;
}

/// Every version scalar is I64 1, and the terminal projection out.weight is all zero, as the reference demands.
fn contract(src: ck.Source, s: Shape) !void {
    for (dspine.globals) |t| if (t.kind == .i64) try version(src, s, t.name);
    for (dspine.bridge_tensors) |t| if (t.kind == .i64) {
        var full: [160]u8 = undefined;
        try version(src, s, try std.fmt.bufPrint(&full, "{s}{s}", .{ dspine.bridge_prefix, t.name }));
    };
    const out = try bridgeTensor(src, s, "out.weight");
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
    const layers = try gpa.alloc(Layer, s.layers);
    errdefer gpa.free(layers);
    for (layers, 0..) |*l, i| l.* = .{
        .input_norm = try layerTensor(src, s, i, "input_layernorm.weight"),
        .post_norm = try layerTensor(src, s, i, "post_attention_layernorm.weight"),
        .attention = .{
            .q = try layerTensor(src, s, i, "self_attn.q_proj.weight"),
            .k = try layerTensor(src, s, i, "self_attn.k_proj.weight"),
            .v = try layerTensor(src, s, i, "self_attn.v_proj.weight"),
            .o = try layerTensor(src, s, i, "self_attn.o_proj.weight"),
            .q_norm = try layerTensor(src, s, i, "self_attn.q_norm.weight"),
            .k_norm = try layerTensor(src, s, i, "self_attn.k_norm.weight"),
        },
        .mlp = .{
            .gate = try layerTensor(src, s, i, "mlp.gate_proj.weight"),
            .up = try layerTensor(src, s, i, "mlp.up_proj.weight"),
            .down = try layerTensor(src, s, i, "mlp.down_proj.weight"),
        },
    };
    const bridge: Bridge = .{
        .codebook = try bridgeTensor(src, s, "codebook"),
        .early_read = try bridgeTensor(src, s, "early_read_weights"),
        .read = try bridgeTensor(src, s, "read_weight"),
        .write = try bridgeTensor(src, s, "feature_write_weights"),
        .gate = try bridgeTensor(src, s, "gate.weight"),
        .gate_bias = try bridgeTensor(src, s, "gate.bias"),
        .predecessor_gate = try bridgeTensor(src, s, "predecessor_gate.weight"),
        .prev = try bridgeTensor(src, s, "prev.weight"),
    };
    return .{ .gpa = gpa, .shape = s, .layers = layers, .fusion = try get(src, s, "fc.weight"), .hidden_norm = try get(src, s, "hidden_norm.weight"), .norm = try get(src, s, "norm.weight"), .bridge = bridge, .embed = embed };
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
