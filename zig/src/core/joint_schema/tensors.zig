//! The joint schema head's tensors (`joint_head.safetensors`): every name with its exact shape, in one walk.
const std = @import("std");
const safetensors = @import("../safetensors.zig");
const Config = @import("config.zig").Config;

pub const Tensor = safetensors.Tensor;

pub const Linear = struct { weight: Tensor, bias: ?Tensor };
pub const Norm = struct { weight: Tensor, bias: Tensor };
/// torch's MultiheadAttention: `in_proj` packs the q, k and v projections by rows.
pub const Attention = struct { in_weight: Tensor, in_bias: Tensor, out: Linear };
pub const Evidence = struct { query_norm: Norm, memory_norm: Norm, attention: Attention, feedforward_norm: Norm, up: Linear, down: Linear };
/// nn.TransformerDecoderLayer, norm first: self-attention, cross-attention to the memory, feed-forward.
pub const Decoder = struct { norm1: Norm, norm2: Norm, norm3: Norm, self_attn: Attention, cross_attn: Attention, up: Linear, down: Linear };

pub const Weights = struct {
    cfg: Config,
    hidden_norm: Norm,
    memory: Tensor,
    question: Tensor,
    option_question: Tensor,
    global: Tensor,
    option_context: Tensor,
    option_lexical: Tensor,
    type_embedding: Tensor,
    evidence: []Evidence,
    option_summary_norm: Norm,
    layers: []Decoder,
    field_norm: Norm,
    option_norm: Norm,
    scorer_up: Linear,
    scorer_down: Linear,
    prior_logit_scale: Tensor,
    joint_logit_scale: Tensor,
    residual_gate: Tensor,
};

/// The head's tensor count: 22 outside the layers, 14 per evidence layer, 18 per decoder layer (122 for Clef-Flash).
pub fn count(cfg: Config) usize {
    return 22 + 14 * @as(usize, cfg.routing_layers) + 18 * @as(usize, cfg.layers);
}

/// Every tensor in the safetensors names, each asked of `src.take(name, shape) !Tensor`; slices from `a`.
pub fn fill(a: std.mem.Allocator, cfg: Config, src: anytype) !Weights {
    const H: usize = cfg.hidden_size;
    const W: usize = cfg.width;
    const F: usize = cfg.feedforward;
    var w: Weights = undefined;
    w.cfg = cfg;
    var n: Names(@TypeOf(src)) = .{ .src = src };
    w.hidden_norm = try n.norm("hidden_norm", H);
    w.memory = try n.take("memory_projection.weight", &.{ W, H });
    w.question = try n.take("question_projection.weight", &.{ W, H });
    w.option_question = try n.take("option_question_projection.weight", &.{ W, H });
    w.global = try n.take("global_projection.weight", &.{ W, H });
    w.option_context = try n.take("option_context_projection.weight", &.{ W, H });
    w.option_lexical = try n.take("option_lexical_projection.weight", &.{ W, H });
    w.type_embedding = try n.take("type_embedding.weight", &.{ 3, W });
    w.evidence = try a.alloc(Evidence, cfg.routing_layers);
    for (w.evidence, 0..) |*e, i| {
        n.at("evidence_layers.{d}.", i);
        e.* = .{
            .query_norm = try n.norm("query_norm", W),
            .memory_norm = try n.norm("memory_norm", W),
            .attention = try n.attention("attention", W),
            .feedforward_norm = try n.norm("feedforward_norm", W),
            .up = try n.linear("feedforward.0", F, W),
            .down = try n.linear("feedforward.3", W, F),
        };
    }
    n.plen = 0;
    w.option_summary_norm = try n.norm("option_summary_norm", W);
    w.layers = try a.alloc(Decoder, cfg.layers);
    for (w.layers, 0..) |*d, i| {
        n.at("layers.{d}.", i);
        d.* = .{
            .norm1 = try n.norm("norm1", W),
            .norm2 = try n.norm("norm2", W),
            .norm3 = try n.norm("norm3", W),
            .self_attn = try n.attention("self_attn", W),
            .cross_attn = try n.attention("multihead_attn", W),
            .up = try n.linear("linear1", F, W),
            .down = try n.linear("linear2", W, F),
        };
    }
    n.plen = 0;
    w.field_norm = try n.norm("field_norm", W);
    w.option_norm = try n.norm("option_norm", W);
    w.scorer_up = try n.linear("residual_scorer.0", W, 4 * W);
    w.scorer_down = try n.linear("residual_scorer.3", 1, W);
    w.prior_logit_scale = try n.take("prior_logit_scale", &.{});
    w.joint_logit_scale = try n.take("joint_logit_scale", &.{});
    w.residual_gate = try n.take("residual_gate", &.{});
    return w;
}

/// Full names under the current layer prefix, each handed to the source with its shape.
fn Names(comptime S: type) type {
    return struct {
        src: S,
        prefix: [32]u8 = undefined,
        plen: usize = 0,
        const Self = @This();

        fn at(self: *Self, comptime fmt: []const u8, i: usize) void {
            self.plen = (std.fmt.bufPrint(&self.prefix, fmt, .{i}) catch unreachable).len;
        }
        fn take(self: *Self, name: []const u8, shape: []const usize) !Tensor {
            var buf: [96]u8 = undefined;
            const full = std.fmt.bufPrint(&buf, "{s}{s}", .{ self.prefix[0..self.plen], name }) catch unreachable;
            return self.src.take(full, shape);
        }
        fn suffixed(self: *Self, name: []const u8, suffix: []const u8, shape: []const usize) !Tensor {
            var buf: [64]u8 = undefined;
            return self.take(std.fmt.bufPrint(&buf, "{s}{s}", .{ name, suffix }) catch unreachable, shape);
        }
        fn norm(self: *Self, name: []const u8, dim: usize) !Norm {
            return .{ .weight = try self.suffixed(name, ".weight", &.{dim}), .bias = try self.suffixed(name, ".bias", &.{dim}) };
        }
        fn linear(self: *Self, name: []const u8, out: usize, in: usize) !Linear {
            return .{ .weight = try self.suffixed(name, ".weight", &.{ out, in }), .bias = try self.suffixed(name, ".bias", &.{out}) };
        }
        fn attention(self: *Self, name: []const u8, w: usize) !Attention {
            var buf: [64]u8 = undefined;
            const out = std.fmt.bufPrint(&buf, "{s}.out_proj", .{name}) catch unreachable;
            return .{ .in_weight = try self.suffixed(name, ".in_proj_weight", &.{ 3 * w, w }), .in_bias = try self.suffixed(name, ".in_proj_bias", &.{3 * w}), .out = try self.linear(out, w, w) };
        }
    };
}

/// A source that only lists: each name and shape appended, a zero-length tensor returned.
pub const Listing = struct {
    a: std.mem.Allocator,
    names: std.ArrayList([]const u8) = .empty,
    shapes: std.ArrayList([]const usize) = .empty,

    pub fn take(l: *Listing, name: []const u8, shape: []const usize) !Tensor {
        try l.names.append(l.a, try l.a.dupe(u8, name));
        try l.shapes.append(l.a, try l.a.dupe(usize, shape));
        var t: Tensor = .{ .dtype = .bf16, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = &.{} };
        @memcpy(t.shape[0..shape.len], shape);
        return t;
    }
};

test "joint schema tensors: Clef-Flash's 122 names, unique, with the release's shapes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var l: Listing = .{ .a = arena.allocator() };
    _ = try fill(arena.allocator(), Config.clef_flash, &l);
    try std.testing.expectEqual(@as(usize, 122), l.names.items.len);
    try std.testing.expectEqual(count(Config.clef_flash), l.names.items.len);
    var seen = std.StringHashMap(void).init(arena.allocator());
    for (l.names.items) |n| try std.testing.expect(!(try seen.getOrPut(n)).found_existing);
    const spot = [_]struct { []const u8, []const usize }{
        .{ "evidence_layers.1.attention.in_proj_weight", &.{ 3072, 1024 } }, .{ "layers.3.multihead_attn.out_proj.bias", &.{1024} },
        .{ "residual_scorer.0.weight", &.{ 1024, 4096 } },                   .{ "residual_scorer.3.bias", &.{1} },
        .{ "evidence_layers.0.feedforward.3.weight", &.{ 1024, 4096 } },     .{ "residual_gate", &.{} },
        .{ "hidden_norm.bias", &.{4096} },                                   .{ "type_embedding.weight", &.{ 3, 1024 } },
    };
    for (spot) |s| {
        const i = for (l.names.items, 0..) |n, i| {
            if (std.mem.eql(u8, n, s[0])) break i;
        } else return error.TestUnexpectedResult;
        try std.testing.expectEqualSlices(usize, s[1], l.shapes.items[i]);
    }
}
