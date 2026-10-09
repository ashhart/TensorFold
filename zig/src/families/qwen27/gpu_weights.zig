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

const Builder = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    checkpoint: *const cp.Checkpoint,
    prefix: []const u8,
    owned: *std.ArrayList(mtl.Buffer),

    fn get(b: Builder, suffix: []const u8) !Tensor {
        const name = try std.mem.concat(b.allocator, u8, &.{ b.prefix, suffix });
        defer b.allocator.free(name);
        return b.checkpoint.get(name);
    }

    fn tensor(b: Builder, suffix: []const u8, dtype: cp.DType, shape: []const usize) !Tensor {
        const t = try b.get(suffix);
        if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.BadNativeTensor;
        return t;
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

    fn one(b: Builder, module: []const u8, n: usize, k: usize, tile: u32) !Linear {
        return projection.prepare(b.device, b.allocator, b.owned, &.{try b.quant(module, n, k)}, tile);
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
        const gate = try b.quant(try Name.full(a, prefix, "mlp.gate_proj"), c.intermediate, c.hidden);
        const up = try b.quant(try Name.full(a, prefix, "mlp.up_proj"), c.intermediate, c.hidden);
        const mixer: @FieldType(Layer, "mixer") = switch (c.kind(i)) {
            .linear => .{ .linear = .{
                .qkv = try b.one(try Name.full(a, prefix, "linear_attn.in_proj_qkv"), c.gdnQkvDim(), c.hidden, 32),
                .zba = try projection.prepare(b.device, b.allocator, b.owned, &.{ try b.quant(try Name.full(a, prefix, "linear_attn.in_proj_z"), c.gdnValueDim(), c.hidden), try b.quant(try Name.full(a, prefix, "linear_attn.in_proj_b"), c.v_heads, c.hidden), try b.quant(try Name.full(a, prefix, "linear_attn.in_proj_a"), c.v_heads, c.hidden) }, 32),
                .out = try b.one(try Name.full(a, prefix, "linear_attn.out_proj"), c.hidden, c.gdnValueDim(), 32),
                .conv = try b.tensor(try Name.full(a, prefix, "linear_attn.conv1d.weight"), .bf16, &.{ c.gdnQkvDim(), c.conv_kernel, 1 }),
                .a_log = try b.tensor(try Name.full(a, prefix, "linear_attn.A_log"), .bf16, &.{c.v_heads}),
                .dt = try b.tensor(try Name.full(a, prefix, "linear_attn.dt_bias"), .bf16, &.{c.v_heads}),
                .norm = try b.tensor(try Name.full(a, prefix, "linear_attn.norm.weight"), .bf16, &.{c.dv}),
            } },
            .attention => .{ .attention = .{
                .q = try b.one(try Name.full(a, prefix, "self_attn.q_proj"), c.qDim(), c.hidden, 32),
                .kv = try projection.prepare(b.device, b.allocator, b.owned, &.{ try b.quant(try Name.full(a, prefix, "self_attn.k_proj"), c.kvDim(), c.hidden), try b.quant(try Name.full(a, prefix, "self_attn.v_proj"), c.kvDim(), c.hidden) }, 32),
                .out = try b.one(try Name.full(a, prefix, "self_attn.o_proj"), c.hidden, c.heads * c.head_dim, 32),
                .q_norm = try b.tensor(try Name.full(a, prefix, "self_attn.q_norm.weight"), .bf16, &.{c.head_dim}),
                .k_norm = try b.tensor(try Name.full(a, prefix, "self_attn.k_norm.weight"), .bf16, &.{c.head_dim}),
            } },
        };
        return .{ .norm = try b.tensor(try Name.full(a, prefix, "input_layernorm.weight"), .bf16, &.{c.hidden}), .post_norm = try b.tensor(try Name.full(a, prefix, "post_attention_layernorm.weight"), .bf16, &.{c.hidden}), .gu = try projection.prepare(b.device, b.allocator, b.owned, &.{ gate, up }, 32), .down = try b.one(try Name.full(a, prefix, "mlp.down_proj"), c.hidden, c.intermediate, 32), .mixer = mixer };
    }
};

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
    const b = Builder{ .allocator = allocator, .device = device, .checkpoint = checkpoint, .prefix = prefix, .owned = &owned };
    for (layers, 0..) |*l, i| l.* = try b.layer(c, i);
    const embed = try b.quant("model.embed_tokens", c.vocab, c.hidden);
    const norm = try b.tensor("model.norm.weight", .bf16, &.{c.hidden});
    const head = try b.one("lm_head", c.vocab, c.hidden, 32);
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
