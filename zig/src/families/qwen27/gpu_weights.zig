//! The Metal graph borrows native scalar tensors and owns only byte-preserving affine layouts.
const std = @import("std");
const mtl = @import("metal");
const cp = @import("core").checkpoint_metal;
const Config = @import("config.zig").Config;
const projection = @import("projection.zig");
const lane = @import("core").lane_projection;
const Ref = projection.Ref;
const Linear = projection.Linear;
pub const Tensor = cp.Tensor;

pub const Gdn = struct { qkv: Linear, zba: Linear, out: Linear, conv: Tensor, a_log: Tensor, dt: Tensor, norm: Tensor };
pub const Attention = struct { q: Linear, kv: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Layer = struct { norm: Tensor, post_norm: Tensor, gu: Linear, down: Linear, mixer: union(@import("config.zig").Kind) { linear: Gdn, attention: Attention } };

pub const Weights = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    embed: projection.Triple,
    norm: Tensor,
    head: Linear,
    owned: std.ArrayList(mtl.Buffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(w.allocator);
        w.allocator.free(w.layers);
    }
};

/// Modules the graph reads only through repacked copies: their tensors stay in the files until load stages them.
const repacked_modules = [_][]const u8{ ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj", ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.in_proj_b", ".linear_attn.in_proj_a", ".linear_attn.out_proj", ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj" };

/// Whether `name` is a weight, scales or biases tensor of a repacked module or the LM head.
pub fn repacked(name: []const u8, _: usize) bool {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return false;
    const leaf = name[dot + 1 ..];
    if (!std.mem.eql(u8, leaf, "weight") and !std.mem.eql(u8, leaf, "scales") and !std.mem.eql(u8, leaf, "biases")) return false;
    const module = name[0..dot];
    if (std.mem.eql(u8, module, "lm_head") or std.mem.endsWith(u8, module, ".lm_head")) return true;
    for (repacked_modules) |m| if (std.mem.endsWith(u8, module, m)) return true;
    return false;
}

const Builder = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    checkpoint: *const cp.Checkpoint,
    prefix: []const u8,
    owned: *std.ArrayList(mtl.Buffer),
    stage: *cp.Stage,

    fn get(b: Builder, suffix: []const u8) !Tensor {
        const name = try std.mem.concat(b.allocator, u8, &.{ b.prefix, suffix });
        defer b.allocator.free(name);
        return b.checkpoint.get(name);
    }

    fn tensor(b: Builder, suffix: []const u8, dtype: cp.DType, shape: []const usize) !Tensor {
        return checked(try b.get(suffix), dtype, shape);
    }

    fn quant(b: Builder, module: []const u8, n: usize, k: usize) !projection.Triple {
        const suffixes = [_][]const u8{ ".weight", ".scales", ".biases" };
        var tensors: [3]Tensor = undefined;
        for (suffixes, &tensors, 0..) |suffix, *t, i| {
            const name = try std.mem.concat(b.allocator, u8, &.{ module, suffix });
            defer b.allocator.free(name);
            t.* = try b.tensor(name, if (i == 0) .u32 else .bf16, &.{ n, if (i == 0) k / 8 else k / 64 });
        }
        return .{ .weight = tensors[0], .scales = tensors[1], .biases = tensors[2] };
    }

    /// `parts` (stacked in order, each `n` rows of `k`) staged from the files and repacked into one owned Linear.
    fn linear(b: Builder, parts: []const Part, k: usize) !Linear {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        const names = try arena.allocator().alloc([]const u8, parts.len * 3);
        for (parts, 0..) |p, i| for ([_][]const u8{ ".weight", ".scales", ".biases" }, 0..) |suffix, j| {
            names[i * 3 + j] = try std.mem.concat(arena.allocator(), u8, &.{ b.prefix, p.module, suffix });
        };
        try b.stage.reserve(b.checkpoint, names);
        defer b.stage.reset();
        var triples: [3]projection.Triple = undefined;
        if (parts.len > triples.len) return error.BadProjectionShape;
        for (parts, triples[0..parts.len], 0..) |p, *t, i| t.* = .{
            .weight = try checked(try b.stage.read(b.checkpoint, names[i * 3]), .u32, &.{ p.n, k / 8 }),
            .scales = try checked(try b.stage.read(b.checkpoint, names[i * 3 + 1]), .bf16, &.{ p.n, k / 64 }),
            .biases = try checked(try b.stage.read(b.checkpoint, names[i * 3 + 2]), .bf16, &.{ p.n, k / 64 }),
        };
        return projection.prepare(b.device, b.allocator, b.owned, triples[0..parts.len], 32);
    }

    fn layer(b: Builder, c: Config, i: usize) !Layer {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const prefix = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
        const Name = struct {
            fn full(allocator: std.mem.Allocator, p: []const u8, leaf: []const u8) ![]const u8 {
                return std.mem.concat(allocator, u8, &.{ p, leaf });
            }
        };
        const mixer: @FieldType(Layer, "mixer") = switch (c.kind(i)) {
            .linear => .{ .linear = .{
                .qkv = try b.linear(&.{.{ .module = try Name.full(a, prefix, "linear_attn.in_proj_qkv"), .n = c.gdnQkvDim() }}, c.hidden),
                .zba = try b.linear(&.{ .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_z"), .n = c.gdnValueDim() }, .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_b"), .n = c.v_heads }, .{ .module = try Name.full(a, prefix, "linear_attn.in_proj_a"), .n = c.v_heads } }, c.hidden),
                .out = try b.linear(&.{.{ .module = try Name.full(a, prefix, "linear_attn.out_proj"), .n = c.hidden }}, c.gdnValueDim()),
                .conv = try b.tensor(try Name.full(a, prefix, "linear_attn.conv1d.weight"), .bf16, &.{ c.gdnQkvDim(), c.conv_kernel, 1 }),
                .a_log = try b.tensor(try Name.full(a, prefix, "linear_attn.A_log"), .bf16, &.{c.v_heads}),
                .dt = try b.tensor(try Name.full(a, prefix, "linear_attn.dt_bias"), .bf16, &.{c.v_heads}),
                .norm = try b.tensor(try Name.full(a, prefix, "linear_attn.norm.weight"), .bf16, &.{c.dv}),
            } },
            .attention => .{ .attention = .{
                .q = try b.linear(&.{.{ .module = try Name.full(a, prefix, "self_attn.q_proj"), .n = c.qDim() }}, c.hidden),
                .kv = try b.linear(&.{ .{ .module = try Name.full(a, prefix, "self_attn.k_proj"), .n = c.kvDim() }, .{ .module = try Name.full(a, prefix, "self_attn.v_proj"), .n = c.kvDim() } }, c.hidden),
                .out = try b.linear(&.{.{ .module = try Name.full(a, prefix, "self_attn.o_proj"), .n = c.hidden }}, c.heads * c.head_dim),
                .q_norm = try b.tensor(try Name.full(a, prefix, "self_attn.q_norm.weight"), .bf16, &.{c.head_dim}),
                .k_norm = try b.tensor(try Name.full(a, prefix, "self_attn.k_norm.weight"), .bf16, &.{c.head_dim}),
            } },
        };
        const gu = try b.linear(&.{ .{ .module = try Name.full(a, prefix, "mlp.gate_proj"), .n = c.intermediate }, .{ .module = try Name.full(a, prefix, "mlp.up_proj"), .n = c.intermediate } }, c.hidden);
        const down = try b.linear(&.{.{ .module = try Name.full(a, prefix, "mlp.down_proj"), .n = c.hidden }}, c.intermediate);
        return .{ .norm = try b.tensor(try Name.full(a, prefix, "input_layernorm.weight"), .bf16, &.{c.hidden}), .post_norm = try b.tensor(try Name.full(a, prefix, "post_attention_layernorm.weight"), .bf16, &.{c.hidden}), .gu = gu, .down = down, .mixer = mixer };
    }
};

const Part = struct { module: []const u8, n: usize };

fn checked(t: Tensor, dtype: cp.DType, shape: []const usize) !Tensor {
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.BadNativeTensor;
    return t;
}

/// The graph from `checkpoint`, whose `repacked` tensors stay in their files: each Linear is staged, repacked, dropped.
pub fn load(allocator: std.mem.Allocator, device: mtl.Device, checkpoint: *const cp.Checkpoint, c: Config) !Weights {
    const prefix: []const u8 = if (checkpoint.has("language_model.model.embed_tokens.weight")) "language_model." else if (checkpoint.has("model.embed_tokens.weight")) "" else return error.MissingTextNamespace;
    if (checkpoint.has("language_model.model.embed_tokens.weight") and checkpoint.has("model.embed_tokens.weight")) return error.AmbiguousTextNamespace;
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    errdefer {
        for (owned.items) |b| b.deinit();
        owned.deinit(allocator);
    }
    const layers = try allocator.alloc(Layer, c.layers);
    errdefer allocator.free(layers);
    var stage = cp.Stage{ .device = device };
    defer stage.deinit();
    const b = Builder{ .allocator = allocator, .device = device, .checkpoint = checkpoint, .prefix = prefix, .owned = &owned, .stage = &stage };
    for (layers, 0..) |*l, i| l.* = try b.layer(c, i);
    const embed = try b.quant("model.embed_tokens", c.vocab, c.hidden);
    const norm = try b.tensor("model.norm.weight", .bf16, &.{c.hidden});
    const head = try b.linear(&.{.{ .module = "lm_head", .n = c.vocab }}, c.hidden);
    if (device.tensorUnits()) try laneOrder(allocator, device, layers, head);
    return .{ .allocator = allocator, .layers = layers, .embed = embed, .norm = norm, .head = head, .owned = owned };
}

/// Tensor-unit projections leave load in lane_projection.reg_layout, the order quant_gpu's reg kernel reads.
fn laneOrder(a: std.mem.Allocator, device: mtl.Device, layers: []const Layer, head: Linear) !void {
    var jobs: std.ArrayList(lane.Order) = .empty;
    defer jobs.deinit(a);
    for (layers) |l| {
        const list = switch (l.mixer) {
            .linear => |g| [_]Linear{ g.qkv, g.zba, g.out, l.gu, l.down },
            .attention => |t| [_]Linear{ t.q, t.kv, t.out, l.gu, l.down },
        };
        for (list) |p| try jobs.append(a, .{ .w = .{ .buf = p.words.buffer, .off = p.words.offset }, .n = p.n, .k = p.k });
    }
    try jobs.append(a, .{ .w = .{ .buf = head.words.buffer, .off = head.words.offset }, .n = head.n, .k = head.k });
    try lane.regOrder(device, jobs.items);
}

pub fn ref(t: Tensor) Ref {
    return .{ .buffer = t.buffer, .offset = t.offset };
}

test "repacked names the modules the graph reads only through owned copies" {
    try std.testing.expect(repacked("language_model.model.layers.3.mlp.gate_proj.weight", 2));
    try std.testing.expect(repacked("model.layers.3.self_attn.o_proj.scales", 2));
    try std.testing.expect(repacked("language_model.model.layers.0.linear_attn.out_proj.biases", 2));
    try std.testing.expect(repacked("language_model.lm_head.weight", 2));
    try std.testing.expect(repacked("lm_head.scales", 2));
    try std.testing.expect(!repacked("language_model.model.embed_tokens.weight", 2));
    try std.testing.expect(!repacked("language_model.model.layers.0.linear_attn.conv1d.weight", 2));
    try std.testing.expect(!repacked("language_model.model.layers.3.self_attn.q_norm.weight", 2));
    try std.testing.expect(!repacked("language_model.model.norm.weight", 2));
    try std.testing.expect(!repacked("language_model.model.layers.0.linear_attn.A_log", 2));
}
