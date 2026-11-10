//! LFM3-VL's 138 dense BF16 vision tensors: names, shapes and admission against a checkpoint index, without Metal.
const std = @import("std");

pub const width = 1024;
pub const intermediate = 2560;
pub const depth = 14;
/// The official exports' `vision_tower.` and `multi_modal_projector.` tensors: all admitted, none left over.
pub const tensor_count = 138;
/// The safetensors index's most dimensions (`safetensors.max_rank`).
pub const max_rank = 5;

pub fn Layout(comptime Ref: type) type {
    return struct {
        pub const Conv = struct { in: Ref, out: Ref, spatial: Ref };
        pub const Attention = struct { q: Ref, k: Ref, v: Ref, out: Ref, q_norm: Ref, k_norm: Ref };
        pub const Layer = struct {
            operator_norm: Ref,
            ffn_norm: Ref,
            gate: Ref,
            up: Ref,
            down: Ref,
            mixer: union(enum) { conv: Conv, attention: Attention },
        };
        pub const Weights = struct {
            patch: Ref,
            norm: Ref,
            blocks: [depth]Layer,
            projector_norm: Ref,
            projector_norm_bias: Ref,
            linear1: Ref,
            bias1: Ref,
            linear2: Ref,
            bias2: Ref,
        };
    };
}

/// What a checkpoint index holds for one tensor: the owner's handle to it, its header and the bytes behind it.
pub fn Found(comptime Ref: type) type {
    return struct {
        ref: Ref,
        bf16: bool,
        rank: usize,
        shape: [max_rank]usize,
        offset: usize,
        bytes: usize,
        /// The length of the buffer the tensor sits in.
        storage: usize,
    };
}

fn Loader(comptime Ref: type, comptime Source: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        source: Source,
        seen: std.StringHashMapUnmanaged(void) = .empty,

        fn get(self: *Self, prefix: []const u8, suffix: []const u8, shape: []const usize) !Ref {
            var buf: [192]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
            const t = (try self.source.find(name)) orelse return error.MissingTensor;
            if (!t.bf16 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape))
                return error.UnexpectedLfm3VisionTensor;
            var bytes: usize = 2;
            for (shape) |dim| bytes = std.math.mul(usize, bytes, dim) catch return error.TensorBytes;
            const end = std.math.add(usize, t.offset, bytes) catch return error.TensorBytes;
            if (t.bytes != bytes or end > t.storage or t.offset % 4 != 0) return error.TensorBytes;
            try self.seen.put(self.a, try self.a.dupe(u8, name), {});
            return t.ref;
        }
    };
}

/// The tower's tensors from `source`: `find(name)`, null when absent, and `keys()`, an iterator of every name it holds.
pub fn load(comptime Ref: type, gpa: std.mem.Allocator, source: anytype) !Layout(Ref).Weights {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var l: Loader(Ref, @TypeOf(source)) = .{ .a = arena.allocator(), .source = source };
    var out: Layout(Ref).Weights = .{
        .patch = try l.get("vision_tower.", "patchifier.weight", &.{ width, 1536 }),
        .norm = try l.get("vision_tower.", "embedding_norm.weight", &.{width}),
        .blocks = undefined,
        .projector_norm = try l.get("multi_modal_projector.", "layer_norm.weight", &.{width}),
        .projector_norm_bias = try l.get("multi_modal_projector.", "layer_norm.bias", &.{width}),
        .linear1 = try l.get("multi_modal_projector.", "linear_1.weight", &.{ 4096, 4096 }),
        .bias1 = try l.get("multi_modal_projector.", "linear_1.bias", &.{4096}),
        .linear2 = try l.get("multi_modal_projector.", "linear_2.weight", &.{ 2048, 4096 }),
        .bias2 = try l.get("multi_modal_projector.", "linear_2.bias", &.{2048}),
    };
    for (&out.blocks, 0..) |*block, i| {
        var buf: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "vision_tower.layers.{d}.", .{i});
        block.operator_norm = try l.get(p, "operator_norm.weight", &.{width});
        block.ffn_norm = try l.get(p, "ffn_norm.weight", &.{width});
        block.gate = try l.get(p, "feed_forward.w1.weight", &.{ intermediate, width });
        block.up = try l.get(p, "feed_forward.w3.weight", &.{ intermediate, width });
        block.down = try l.get(p, "feed_forward.w2.weight", &.{ width, intermediate });
        block.mixer = if (i >= 2 and i % 2 == 0) .{ .attention = .{
            .q = try l.get(p, "self_attn.q_proj.weight", &.{ width, width }),
            .k = try l.get(p, "self_attn.k_proj.weight", &.{ 512, width }),
            .v = try l.get(p, "self_attn.v_proj.weight", &.{ 512, width }),
            .out = try l.get(p, "self_attn.out_proj.weight", &.{ width, width }),
            .q_norm = try l.get(p, "self_attn.q_layernorm.weight", &.{64}),
            .k_norm = try l.get(p, "self_attn.k_layernorm.weight", &.{64}),
        } } else .{ .conv = .{
            .in = try l.get(p, "conv.in_proj.weight", &.{ 3 * width, width }),
            .out = try l.get(p, "conv.out_proj.weight", &.{ width, width }),
            .spatial = try l.get(p, "conv.spatial_conv.weight", &.{ width, 3, 3, 1 }),
        } };
    }
    var names = l.source.keys();
    while (names.next()) |name| {
        if (std.mem.startsWith(u8, name, "vision_tower.") or std.mem.startsWith(u8, name, "multi_modal_projector."))
            if (!l.seen.contains(name)) return error.UnexpectedLfm3VisionTensor;
    }
    if (l.seen.count() != tensor_count) return error.UnexpectedLfm3VisionTensor;
    return out;
}
