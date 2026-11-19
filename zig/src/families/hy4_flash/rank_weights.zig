//! Exact partitions of Hy4-flash attention heads and routed experts.
const std = @import("std");
const core = @import("core");
const loader = @import("loader.zig");
const mla = @import("mla_weights.zig");

const Affine = core.quant.mlx.Host;
const Span = core.quant.Span;
const Rank = core.quant.Rank;
const group = 64;

pub const Range = struct {
    first: usize,
    count: usize,

    pub fn init(whole: usize, r: Rank, name: []const u8) !Range {
        if (r.world == 0 or r.rank >= r.world or whole == 0) return error.BadHy4Rank;
        const count = try core.quant.slice.even(whole, r.world, name);
        return .{ .first = r.rank * count, .count = count };
    }

    fn span(self: Range, width: usize) Span {
        return .{ .from = self.first * width, .to = (self.first + self.count) * width };
    }
};

/// Q-B and the gate own this rank's output heads. The O projection owns those same input columns.
/// Q-A and KV-A are shared between ranks. The sink and absorbed K/V projection follow the head range.
pub const Attention = struct {
    heads: mla.HeadRange,
    query: Affine,
    gate: Affine,
    output: Affine,
    owns_output: bool,

    pub fn load(a: std.mem.Allocator, model: *loader.Loaded, layer: u32, rank: Rank) !Attention {
        _ = try model.attentionTargetLayer(.target, layer);
        const config = model.config.value.text_config;
        const shape = config.headShape(layer);
        const heads = try Range.init(shape.heads, rank, "Hy4-flash attention heads");
        const query_stride = @as(usize, shape.non_rotary) + shape.rotary;
        const value_width = shape.value;
        var buffer: [108]u8 = undefined;
        const prefix = "model.layers.{d}.self_attn.";
        const query = try affine(model, try std.fmt.bufPrint(&buffer, prefix ++ "q_b_proj", .{layer}), &.{
            shape.heads * query_stride, config.q_lora_rank,
        });
        const gate = try affine(model, try std.fmt.bufPrint(&buffer, prefix ++ "linear_gate", .{layer}), &.{
            @as(usize, shape.heads) * value_width, config.hidden_size,
        });
        const output = try affine(model, try std.fmt.bufPrint(&buffer, prefix ++ "o_proj", .{layer}), &.{
            config.hidden_size, @as(usize, shape.heads) * value_width,
        });
        return .{
            .heads = .{ .first = @intCast(heads.first), .count = @intCast(heads.count) },
            .query = try takeRows(a, query, heads.span(query_stride)),
            .gate = try takeRows(a, gate, heads.span(value_width)),
            .output = if (rank.world == 1) output else try takeCols(a, output, heads.span(value_width)),
            .owns_output = rank.world != 1,
        };
    }

    pub fn deinit(self: *Attention, a: std.mem.Allocator) void {
        if (self.owns_output) {
            a.free(self.output.words.bytes);
            a.free(self.output.scales.bytes);
            a.free(self.output.biases.bytes);
        }
        self.* = undefined;
    }
};

/// Each rank reads whole experts from the two stacked affine projections, keeping global expert IDs.
pub const Experts = struct {
    ids: Range,
    gate_up: Affine,
    down: Affine,

    pub fn load(a: std.mem.Allocator, model: *loader.Loaded, layer: u32, rank: Rank) !Experts {
        _ = try model.attentionTargetLayer(.target, layer);
        const config = model.config.value.text_config;
        if (config.mlp_layer_types[layer] != .sparse) return error.BadHy4ExpertLayer;
        const total = config.n_routed_experts;
        const width = config.moe_intermediate_size;
        const latent = config.routed_expert_hidden_size;
        const ids = try Range.init(total, rank, "Hy4-flash routed experts");
        var name: [108]u8 = undefined;
        const prefix = "model.layers.{d}.mlp.experts.";
        const gate_up = try affine(model, try std.fmt.bufPrint(&name, prefix ++ "gate_up_proj", .{layer}), &.{ total, @as(usize, width) * 2, latent });
        const down = try affine(model, try std.fmt.bufPrint(&name, prefix ++ "down_proj", .{layer}), &.{ total, latent, width });
        return .{
            .ids = ids,
            .gate_up = try takeRows(a, gate_up, ids.span(1)),
            .down = try takeRows(a, down, ids.span(1)),
        };
    }
};

pub fn affine(model: *loader.Loaded, key: []const u8, dims: []const usize) !Affine {
    if (dims.len < 2 or dims.len > 3) return error.BadHy4ProjectionWidth;
    var name: [120]u8 = undefined;
    var shape: [3]usize = undefined;
    @memcpy(shape[0..dims.len], dims);
    const k = dims[dims.len - 1];
    if (k == 0 or k % group != 0) return error.BadHy4ProjectionWidth;
    shape[dims.len - 1] = k * model.bits / 32;
    const words = try model.checkpoint.expect(try std.fmt.bufPrint(&name, "{s}.weight", .{key}), .u32, shape[0..dims.len]);
    shape[dims.len - 1] = k / group;
    return .{
        .words = words,
        .scales = try model.checkpoint.expect(try std.fmt.bufPrint(&name, "{s}.scales", .{key}), .bf16, shape[0..dims.len]),
        .biases = try model.checkpoint.expect(try std.fmt.bufPrint(&name, "{s}.biases", .{key}), .bf16, shape[0..dims.len]),
        .bits = model.bits,
        .group = group,
    };
}

fn takeRows(a: std.mem.Allocator, p: Affine, span: Span) !Affine {
    var cut = p;
    cut.words = try core.quant.slice.takeRows(a, p.words, &.{span});
    cut.scales = try core.quant.slice.takeRows(a, p.scales, &.{span});
    cut.biases = try core.quant.slice.takeRows(a, p.biases, &.{span});
    return cut;
}

fn takeCols(a: std.mem.Allocator, p: Affine, cols: Span) !Affine {
    if (cols.from % group != 0 or cols.to % group != 0) return error.BadHy4ProjectionWidth;
    var cut = p;
    cut.words = try core.quant.slice.takeCols(a, p.words, cols.from * p.bits / 32, cols.to * p.bits / 32);
    errdefer a.free(cut.words.bytes);
    cut.scales = try core.quant.slice.takeCols(a, p.scales, cols.from / group, cols.to / group);
    errdefer a.free(cut.scales.bytes);
    cut.biases = try core.quant.slice.takeCols(a, p.biases, cols.from / group, cols.to / group);
    return cut;
}

test "both rank cuts align attention and routed expert partitions" {
    try std.testing.expectEqual(Range{ .first = 16, .count = 16 }, try Range.init(32, .{ .rank = 1, .world = 2 }, "sparse"));
    try std.testing.expectEqual(Range{ .first = 0, .count = 32 }, try Range.init(64, .{ .rank = 0, .world = 2 }, "sliding"));
    const ids = try Range.init(512, .{ .rank = 1, .world = 2 }, "experts");
    try std.testing.expectEqual(Span{ .from = 256, .to = 512 }, ids.span(1));
    try std.testing.expectError(error.BadHy4Rank, Range.init(32, .{ .rank = 0, .world = 0 }, "sparse"));
}
