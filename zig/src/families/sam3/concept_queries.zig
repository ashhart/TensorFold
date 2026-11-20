const std = @import("std");
const runtime = @import("runtime.zig");
const elementwise = @import("elementwise.zig");
const projection = @import("projection.zig");
const attention = @import("attention.zig");
const encodings = @import("encodings.zig");
const concept_context = @import("concept_context.zig");

const Runtime = runtime.Runtime;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;
const Context = concept_context.Context;
const heads = concept_context.heads;
const layerNorm = concept_context.layerNorm;

pub const query_count = 200;
pub const query_layers = 6;

/// `tokens` `[1 + count, hidden]` with the presence token first; `boxes` `[count, 4]` as (cx, cy, w, h).
pub const Queries = struct { tokens: Tensor, boxes: Tensor, count: usize };

pub const Axis = enum { x, y };

/// Six refinement layers of the object queries; must start with no batch open (host reads between stages).
pub fn refine(rt: *Runtime, context: Context, grounded: Tensor) !Queries {
    const c = rt.config;
    const width = c.hidden;
    const side = c.featureSide();
    const grid_rows = side * side;
    if (grounded.count != grid_rows * width or context.cues.count != context.cue_rows * width) return error.SamQueryShape;
    if (rt.gpu.batch != null) return error.SamBatchOpen;
    const decoder = try Scope.root(rt, c.roots.detector).sub("transformer.decoder");
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    errdefer if (rt.gpu.batch != null) rt.gpu.endBatch() catch {};

    try rt.gpu.beginBatch();
    const presence = try decoder.param("presence_token.weight");
    const learned = try decoder.param("query_embed.weight");
    const references = try decoder.param("reference_points.weight");
    if (presence.count != width or learned.count != query_count * width or references.count != query_count * 4)
        return error.SamQueryShape;
    var tokens = try elementwise.concat(rt, &.{ presence, learned });
    var boxes = try elementwise.sigmoid(rt, references);
    const zero = try elementwise.hostFill(rt, width, 0);
    try rt.gpu.endBatch();

    for (0..query_layers) |j| {
        const layer = try decoder.item("layers", j);
        const code = try encodings.boxCode(rt, boxes);
        const box_values = boxes.slice(f32, boxes.count);
        const fx = try rt.hostTensor(query_count * side * 2);
        const fy = try rt.hostTensor(query_count * side * 2);
        edgeFeatures(fx.slice(f32, fx.count), box_values, side, .x);
        edgeFeatures(fy.slice(f32, fy.count), box_values, side, .y);

        try rt.gpu.beginBatch();
        const query_positions = try projection.reluChain(decoder, "ref_point_head", code, 2);
        const positions = try elementwise.concat(rt, &.{ zero, query_positions });
        const placed = try elementwise.add(rt, tokens, positions);
        const own = try attention.packedHeads(try layer.sub("self_attn"), placed, placed, tokens, heads, .{});
        tokens = try layerNorm(layer, "norm2", try elementwise.add(rt, tokens, own));
        const cue_query = try elementwise.add(rt, tokens, positions);
        const cued = try attention.packedHeads(try layer.sub("ca_text"), cue_query, context.cues, context.cues, heads, .{});
        tokens = try layerNorm(layer, "catext_norm", try elementwise.add(rt, tokens, cued));
        const ox = try projection.reluChain(decoder, "boxRPB_embed_x", fx, 2);
        const oy = try projection.reluChain(decoder, "boxRPB_embed_y", fy, 2);
        try rt.gpu.endBatch();

        if (ox.count != query_count * side * heads or oy.count != ox.count) return error.SamQueryShape;
        const box_offsets = try rt.hostTensor(heads * (query_count + 1) * grid_rows);
        const offset_values = box_offsets.slice(f32, box_offsets.count);
        assembleBoxOffsets(offset_values, ox.slice(f32, ox.count), oy.slice(f32, oy.count), query_count, side, heads);

        try rt.gpu.beginBatch();
        const image_query = try elementwise.add(rt, tokens, positions);
        const keys = try elementwise.add(rt, grounded, context.grid_code);
        const options = attention.Options{ .bias = box_offsets };
        const looked = try attention.packedHeads(try layer.sub("cross_attn"), image_query, keys, grounded, heads, options);
        tokens = try layerNorm(layer, "norm1", try elementwise.add(rt, tokens, looked));
        const fed = try projection.expandContract(layer, tokens, .relu);
        tokens = try layerNorm(layer, "norm3", try elementwise.add(rt, tokens, fed));
        const normed = try layerNorm(decoder, "norm", runtime.part(tokens, width, query_count * width));
        const box_update = try projection.reluChain(decoder, "bbox_embed", normed, 3);
        try rt.gpu.endBatch();

        if (box_update.count != query_count * 4) return error.SamQueryShape;
        const next = try rt.hostTensor(query_count * 4);
        updateBoxes(next.slice(f32, next.count), box_values, box_update.slice(f32, box_update.count));
        boxes = next;
        rt.rewind(m, &.{ zero, tokens, boxes });
    }
    rt.rewind(m, &.{ tokens, boxes });
    return .{ .tokens = tokens, .boxes = boxes, .count = query_count };
}

/// sign(g)·log₂(|g| + 1)/3 with g = (j/side − edge)·8 and sign(0) = 0.
pub fn edgeDistance(cell: usize, edge: f32, side: usize) f32 {
    const u = @as(f32, @floatFromInt(cell)) / @as(f32, @floatFromInt(side));
    const g = (u - edge) * 8;
    return std.math.sign(g) * @log2(@abs(g) + 1) / 3;
}

/// Row (i, j) = side·i + j holds the distances of cell j to the low and high edges of box i along `axis`.
pub fn edgeFeatures(out: []f32, boxes: []const f32, side: usize, axis: Axis) void {
    const at: usize = @backingInt(axis);
    for (0..boxes.len / 4) |i| {
        const centre = boxes[4 * i + at];
        const extent = boxes[4 * i + 2 + at];
        const low = centre - 0.5 * extent;
        const high = centre + 0.5 * extent;
        for (0..side) |j| {
            const row = (i * side + j) * 2;
            out[row] = edgeDistance(j, low, side);
            out[row + 1] = edgeDistance(j, high, side);
        }
    }
}

/// `[heads, 1 + queries, side²]`: zero for the presence row, else the column term plus the row term.
pub fn assembleBoxOffsets(out: []f32, ox: []const f32, oy: []const f32, queries: usize, side: usize, head_count: usize) void {
    const rows = queries + 1;
    const cells = side * side;
    for (0..head_count) |h| {
        @memset(out[h * rows * cells ..][0..cells], 0);
        for (0..queries) |i| {
            const base = (h * rows + 1 + i) * cells;
            for (0..side) |y| for (0..side) |x| {
                out[base + y * side + x] = ox[(side * i + x) * head_count + h] + oy[(side * i + y) * head_count + h];
            };
        }
    }
}

/// ln(max(p̂, 1e-3) / max(1 − p̂, 1e-3)) with p̂ = clamp(p, 0, 1).
pub fn inverseSigmoid(p: f32) f32 {
    const floor: f32 = 1e-3;
    const clamped = std.math.clamp(p, 0, 1);
    return @log(@max(clamped, floor) / @max(1 - clamped, floor));
}

pub fn sigmoid(z: f32) f32 {
    return 1 / (1 + @exp(-z));
}

/// Box update in logit space: σ(inverse sigmoid of p + η), per coordinate.
pub fn updateBoxes(out: []f32, boxes: []const f32, box_update: []const f32) void {
    for (out, boxes, box_update) |*next, p, eta| next.* = sigmoid(inverseSigmoid(p) + eta);
}

const testing = std.testing;

test "SAM box edge distance is a signed log of the scaled offset" {
    try testing.expectEqual(@as(f32, 0), edgeDistance(18, 0.25, 72));
    try testing.expectEqual(@as(f32, 0), edgeDistance(0, 0, 72));
    try testing.expectApproxEqAbs(-@log2(@as(f32, 5)) / 3, edgeDistance(0, 0.5, 72), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 1) / 3, edgeDistance(9, 0, 72), 1e-7);
    for (0..72) |j| for ([_]f32{ 0, 0.1, 0.37, 0.5, 0.93 }) |e| {
        const u = @as(f32, @floatFromInt(j)) / 72;
        const forward = edgeDistance(j, e, 72);
        const g = (e - u) * 8;
        const mirrored = std.math.sign(g) * @log2(@abs(g) + 1) / 3;
        try testing.expectEqual(forward, -mirrored);
    };
}

test "SAM box offsets put the column term first and leave the presence row empty" {
    const queries = 2;
    const side = 72;
    const head_count = 8;
    const ox = try testing.allocator.alloc(f32, queries * side * head_count);
    defer testing.allocator.free(ox);
    const oy = try testing.allocator.alloc(f32, queries * side * head_count);
    defer testing.allocator.free(oy);
    for (0..queries) |i| for (0..side) |j| for (0..head_count) |h| {
        ox[(side * i + j) * head_count + h] = @floatFromInt(1000 * i + 10 * j + h);
        oy[(side * i + j) * head_count + h] = 0.5 * @as(f32, @floatFromInt(j));
    };
    const out = try testing.allocator.alloc(f32, head_count * (queries + 1) * side * side);
    defer testing.allocator.free(out);
    @memset(out, std.math.nan(f32));
    assembleBoxOffsets(out, ox, oy, queries, side, head_count);
    try testing.expectEqual(@as(usize, 8 * 3 * 5184), out.len);
    for (0..head_count) |h| {
        for (out[h * 3 * 5184 ..][0..5184]) |v| try testing.expectEqual(@as(f32, 0), v);
        for (0..queries) |i| for ([_]usize{ 0, 5, 71 }) |y| for ([_]usize{ 0, 33, 71 }) |x| {
            const expected = ox[(72 * i + x) * head_count + h] + oy[(72 * i + y) * head_count + h];
            try testing.expectEqual(expected, out[(h * 3 + 1 + i) * 5184 + 72 * y + x]);
        };
    }
}

test "SAM box update uses the floored inverse sigmoid" {
    try testing.expectEqual(@as(f32, 0.5), sigmoid(inverseSigmoid(0.5) + 0));
    const low = @log(@as(f32, 1e-3));
    try testing.expectApproxEqAbs(@as(f32, -6.9077554), low, 1e-6);
    try testing.expectEqual(low, inverseSigmoid(0));
    try testing.expectEqual(low, inverseSigmoid(-0.2));
    try testing.expectEqual(-low, inverseSigmoid(1));
    try testing.expectEqual(-low, inverseSigmoid(1.5));
    try testing.expectEqual(@log(@as(f32, 0.9995) / @as(f32, 1e-3)), inverseSigmoid(0.9995));
    var out: [4]f32 = undefined;
    updateBoxes(&out, &.{ 0, 1, 0.25, 0.75 }, &.{ -3, 3, 0.5, -0.5 });
    for (out) |v| try testing.expect(v > 0 and v < 1);
}
