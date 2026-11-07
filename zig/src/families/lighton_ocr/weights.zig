//! LightOnOCR's BF16 tensor map: validated checkpoint views and aligned offsets for one owned device allocation.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const Allocator = std.mem.Allocator;

pub const TextLayer = struct {
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    gate: u64,
    up: u64,
    down: u64,
};

pub const VisionLayer = struct {
    input_norm: u64,
    post_norm: u64,
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    gate: u64,
    up: u64,
    down: u64,
};

/// Borrowed from the mapped checkpoint until upload completes; offsets are relative to the device arena's base.
pub const Entry = struct { offset: u64, bytes: []const u8 };

pub const Layout = struct {
    gpa: Allocator,
    c: config.Config,
    bytes: usize = 0,
    entries: std.ArrayList(Entry) = .empty,
    text: []TextLayer = &.{},
    vision: []VisionLayer = &.{},
    embedding: u64 = 0,
    final_norm: u64 = 0,
    patch: u64 = 0,
    pre_norm: u64 = 0,
    projector_norm: u64 = 0,
    merger: u64 = 0,
    project1: u64 = 0,
    project2: u64 = 0,

    /// All required tensors are shape-checked before the caller starts any device allocation or upload.
    pub fn init(gpa: Allocator, ck: anytype, c: config.Config) !Layout {
        var l: Layout = .{ .gpa = gpa, .c = c };
        errdefer l.deinit();
        l.text = try gpa.alloc(TextLayer, c.layers);
        l.vision = try gpa.alloc(VisionLayer, c.vision.layers);
        const h: usize = c.hidden;
        const mid: usize = c.intermediate;
        const qw = @as(usize, c.heads) * c.head_dim;
        const kw = @as(usize, c.kv_heads) * c.head_dim;
        const vh: usize = c.vision.hidden;
        const vmid: usize = c.vision.intermediate;
        const text = "language_model.model.";
        const tower = "model.vision_encoder.";
        const projector = "model.vision_projection.";
        l.embedding = try l.tensor(ck, text ++ "embed_tokens.weight", &.{ c.vocab, h });
        l.final_norm = try l.tensor(ck, text ++ "norm.weight", &.{h});
        for (l.text, 0..) |*b, i| {
            var name: [192]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&name, text ++ "layers.{d}.", .{i});
            b.* = .{
                .input_norm = try l.named(ck, prefix, "input_layernorm.weight", &.{h}),
                .post_norm = try l.named(ck, prefix, "post_attention_layernorm.weight", &.{h}),
                .q_norm = try l.named(ck, prefix, "self_attn.q_norm.weight", &.{c.head_dim}),
                .k_norm = try l.named(ck, prefix, "self_attn.k_norm.weight", &.{c.head_dim}),
                .q = try l.named(ck, prefix, "self_attn.q_proj.weight", &.{ qw, h }),
                .k = try l.named(ck, prefix, "self_attn.k_proj.weight", &.{ kw, h }),
                .v = try l.named(ck, prefix, "self_attn.v_proj.weight", &.{ kw, h }),
                .o = try l.named(ck, prefix, "self_attn.o_proj.weight", &.{ h, qw }),
                .gate = try l.named(ck, prefix, "mlp.gate_proj.weight", &.{ mid, h }),
                .up = try l.named(ck, prefix, "mlp.up_proj.weight", &.{ mid, h }),
                .down = try l.named(ck, prefix, "mlp.down_proj.weight", &.{ h, mid }),
            };
        }
        l.patch = try l.tensor(ck, tower ++ "patch_conv.weight", &.{ vh, 3, c.vision.patch, c.vision.patch });
        l.pre_norm = try l.tensor(ck, tower ++ "ln_pre.weight", &.{vh});
        for (l.vision, 0..) |*b, i| {
            var name: [192]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&name, tower ++ "transformer.layers.{d}.", .{i});
            b.* = .{
                .input_norm = try l.named(ck, prefix, "attention_norm.weight", &.{vh}),
                .post_norm = try l.named(ck, prefix, "ffn_norm.weight", &.{vh}),
                .q = try l.named(ck, prefix, "attention.q_proj.weight", &.{ vh, vh }),
                .k = try l.named(ck, prefix, "attention.k_proj.weight", &.{ vh, vh }),
                .v = try l.named(ck, prefix, "attention.v_proj.weight", &.{ vh, vh }),
                .o = try l.named(ck, prefix, "attention.o_proj.weight", &.{ vh, vh }),
                .gate = try l.named(ck, prefix, "feed_forward.gate_proj.weight", &.{ vmid, vh }),
                .up = try l.named(ck, prefix, "feed_forward.up_proj.weight", &.{ vmid, vh }),
                .down = try l.named(ck, prefix, "feed_forward.down_proj.weight", &.{ vh, vmid }),
            };
        }
        l.projector_norm = try l.tensor(ck, projector ++ "norm.weight", &.{vh});
        l.merger = try l.tensor(ck, projector ++ "patch_merger.merging_layer.weight", &.{ vh, vh * c.merge * c.merge });
        l.project1 = try l.tensor(ck, projector ++ "linear_1.weight", &.{ h, vh });
        l.project2 = try l.tensor(ck, projector ++ "linear_2.weight", &.{ h, h });
        return l;
    }

    pub fn deinit(l: *Layout) void {
        l.gpa.free(l.text);
        l.gpa.free(l.vision);
        l.entries.deinit(l.gpa);
        l.* = undefined;
    }

    fn named(l: *Layout, ck: anytype, prefix: []const u8, name: []const u8, shape: []const usize) !u64 {
        var buf: [256]u8 = undefined;
        return l.tensor(ck, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name }), shape);
    }

    fn tensor(l: *Layout, ck: anytype, name: []const u8, shape: []const usize) !u64 {
        const t = try ck.expect(name, .bf16, shape);
        var elements: usize = 1;
        for (shape) |d| elements = try std.math.mul(usize, elements, d);
        if (t.bytes.len != try std.math.mul(usize, elements, 2)) return error.UnexpectedTensor;
        const offset = std.mem.alignForward(usize, l.bytes, 256);
        const end = try std.math.add(usize, offset, t.bytes.len);
        try l.entries.append(l.gpa, .{ .offset = offset, .bytes = t.bytes });
        l.bytes = std.mem.alignForward(usize, end, 256);
        return offset;
    }
};

test {
    _ = @import("weights_test.zig");
}
