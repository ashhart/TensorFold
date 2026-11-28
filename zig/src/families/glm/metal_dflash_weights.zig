//! GLM's DFlash2 drafter graph: incoai's tensor names and shapes, borrowed BF16, the target's embedding read elsewhere.
const std = @import("std");
const ck = @import("../qwen27/checkpoint.zig");
const shared = @import("../qwen27/dflash/weights.zig");
const Config = @import("../qwen27/dflash/config.zig").Config;

pub const Tensor = shared.Tensor;
pub const Layer = shared.Layer;
pub const Graph = shared.Graph;
pub const Inventory = shared.Inventory;
pub const names = shared.names;
pub const freeNames = shared.freeNames;

/// The graph's embedding record: the GLM backend reads the target's own affine, so this carries the shape alone.
pub fn embedBinding(c: Config) ck.Linear {
    return .{ .dense = .{ .dtype = .bf16, .rank = 2, .shape = .{ c.vocab, c.hidden, 1, 1, 1 }, .bytes = &.{} } };
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

/// The embedding record names the target's table; its bytes belong to the target, so only its shape is checked.
fn embedShape(p: ck.Linear, c: Config) !void {
    switch (p) {
        .dense => |t| {
            if (t.rank != 2 or t.shape[0] != c.vocab or t.shape[1] != c.hidden) return error.TargetBinding;
        },
        .quantized => return error.TargetBinding,
    }
}

/// incoai's drafter graph from `source`, whose tensors are borrowed BF16 views of the drafter's checkpoint.
pub fn load(gpa: std.mem.Allocator, source: Inventory, c: Config, embed: ck.Linear) !Graph {
    try c.check();
    try inventory(gpa, source.names, c);
    try embedShape(embed, c);
    const src = source.source;
    const layers = try gpa.alloc(Layer, c.layers);
    errdefer gpa.free(layers);
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

const toy = Config{ .hidden = 64, .intermediate = 128, .vocab = 64, .heads = 8, .kv_heads = 4, .head_dim = 8, .mask = 63, .window = 16, .rank = 16, .topk = 2, .target_layers = 45, .taps = @import("metal_dflash_config.zig").stock.taps, .tap_count = 5 };

const Store = struct {
    arena: std.heap.ArenaAllocator,
    keys: [][]const u8,
    tensors: []Tensor,
    fn init(gpa: std.mem.Allocator) !Store {
        var s = Store{ .arena = std.heap.ArenaAllocator.init(gpa), .keys = undefined, .tensors = undefined };
        errdefer s.deinit();
        const a = s.arena.allocator();
        s.keys = try names(a, toy);
        s.tensors = try a.alloc(Tensor, s.keys.len);
        for (s.keys, s.tensors) |name, *t| {
            const shape = dims(name);
            var count: usize = 2;
            for (shape.values[0..shape.rank]) |n| count = try std.math.mul(usize, count, n);
            const data = try a.alloc(u8, count);
            @memset(data, 0);
            t.* = .{ .dtype = .bf16, .rank = @intCast(shape.rank), .shape = shape.values, .bytes = data };
        }
        return s;
    }
    fn deinit(s: *Store) void {
        s.arena.deinit();
    }
    fn graph(s: *Store, gpa: std.mem.Allocator) !Graph {
        return load(gpa, .{ .source = .{ .ptr = s, .getFn = read }, .names = s.keys }, toy, embedBinding(toy));
    }
    fn read(ptr: *anyopaque, name: []const u8) !Tensor {
        const s: *Store = @ptrCast(@alignCast(ptr));
        for (s.keys, s.tensors) |key, t| if (std.mem.eql(u8, key, name)) return t;
        return error.MissingTensor;
    }
    fn dims(name: []const u8) struct { values: @FieldType(Tensor, "shape"), rank: usize } {
        if (std.mem.eql(u8, name, "fc.weight")) return .{ .values = .{ toy.hidden, toy.tapWidth(), 1, 1, 1 }, .rank = 2 };
        if (std.mem.eql(u8, name, "candidate_selector.hidden_projection.weight")) return .{ .values = .{ toy.rank, toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.startsWith(u8, name, "candidate_selector.")) return .{ .values = .{ toy.vocab, toy.rank, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "base_kernel")) return .{ .values = .{ 2, 2, toy.hidden, 1, 1 }, .rank = 3 };
        if (std.mem.endsWith(u8, name, "kernel_projection.weight")) return .{ .values = .{ toy.dynamicWidth(), toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "q_proj.weight")) return .{ .values = .{ toy.qWidth(), toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "k_proj.weight") or std.mem.endsWith(u8, name, "v_proj.weight")) return .{ .values = .{ toy.kvWidth(), toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "o_proj.weight")) return .{ .values = .{ toy.hidden, toy.qWidth(), 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "gate_proj.weight") or std.mem.endsWith(u8, name, "up_proj.weight")) return .{ .values = .{ toy.intermediate, toy.hidden, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "down_proj.weight")) return .{ .values = .{ toy.hidden, toy.intermediate, 1, 1, 1 }, .rank = 2 };
        if (std.mem.endsWith(u8, name, "q_norm.weight") or std.mem.endsWith(u8, name, "k_norm.weight")) return .{ .values = .{ toy.head_dim, 1, 1, 1, 1 }, .rank = 1 };
        return .{ .values = .{ toy.hidden, 1, 1, 1, 1 }, .rank = 1 };
    }
};

test "incoai's names and shapes load as one borrowed BF16 graph" {
    const a = std.testing.allocator;
    var s = try Store.init(a);
    defer s.deinit();
    var g = try s.graph(a);
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 81), s.keys.len);
    try std.testing.expectEqual(s.tensors[0].bytes.ptr, g.fusion.bytes.ptr);
    try std.testing.expect(try g.nativeBytes() > 0);
}

test "a missing, duplicated or foreign tensor name is refused before any weight is read" {
    const a = std.testing.allocator;
    var s = try Store.init(a);
    defer s.deinit();
    const previous = s.keys[0];
    s.keys[0] = s.keys[1];
    defer s.keys[0] = previous;
    try std.testing.expectError(error.DraftTensorCoverage, s.graph(a));
}

test "a tensor of another dtype or shape cannot be cast into the graph" {
    const a = std.testing.allocator;
    var s = try Store.init(a);
    defer s.deinit();
    s.tensors[0].dtype = .f16;
    try std.testing.expectError(error.DraftNativeDType, s.graph(a));
    s.tensors[0].dtype = .bf16;
    s.tensors[0].shape[1] -= 1;
    try std.testing.expectError(error.ProjectionShape, s.graph(a));
}

test "an embedding record of another vocabulary or width is refused by the target binding" {
    const a = std.testing.allocator;
    var s = try Store.init(a);
    defer s.deinit();
    var wide = embedBinding(toy);
    wide.dense.shape[0] += 1;
    try std.testing.expectError(error.TargetBinding, load(a, .{ .source = .{ .ptr = &s, .getFn = Store.read }, .names = s.keys }, toy, wide));
    try std.testing.expectError(error.TargetBinding, load(a, .{ .source = .{ .ptr = &s, .getFn = Store.read }, .names = s.keys }, toy, .{ .quantized = undefined }));
}
