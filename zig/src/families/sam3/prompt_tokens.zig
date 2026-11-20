const std = @import("std");
const prompts = @import("prompts.zig");
const runtime = @import("runtime.zig");
const elementwise = @import("elementwise.zig");
const convolution = @import("convolution.zig");
const spatial = @import("spatial.zig");
const norms = @import("norms.zig");
const encodings = @import("encodings.zig");

const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;
const Scope = runtime.Scope;

const mask_norm_eps: f32 = 1e-6;

pub const PromptTokens = struct { sparse: ?Tensor = null, sparse_rows: usize = 0, dense: ?Tensor = null };

const Kind = enum(u8) { negative_point = 0, positive_point = 1, top_left = 2, bottom_right = 3, terminator = 4 };
const Record = struct { kind: Kind, point: [2]f32 = .{ 0, 0 } };

fn rowCount(prompt: prompts.Prompt) usize {
    const coordinates = 2 * prompt.boxes.len + prompt.points.len;
    return if (coordinates == 0) 0 else coordinates + 1;
}

fn unit(x: f32, y: f32, side: f32) [2]f32 {
    return .{ (x + 0.5) / side, (y + 0.5) / side };
}

// Box corners first, then points, then one terminator whenever a point or box is present.
fn records(allocator: std.mem.Allocator, prompt: prompts.Prompt, size: prompts.Size, side: usize) ![]Record {
    for (prompt.boxes) |box| if (!box.positive) return error.NegativeBoxUnsupported;
    const count = rowCount(prompt);
    const out = try allocator.alloc(Record, count);
    errdefer allocator.free(out);
    if (count == 0) return out;
    const transform = prompts.Transform{ .source = size, .side = side };
    const extent: f32 = @floatFromInt(side);
    var i: usize = 0;
    for (prompt.boxes) |box| {
        const corners = try transform.box(box);
        out[i] = .{ .kind = .top_left, .point = unit(corners[0], corners[1], extent) };
        out[i + 1] = .{ .kind = .bottom_right, .point = unit(corners[2], corners[3], extent) };
        i += 2;
    }
    for (prompt.points) |point| {
        const at = try transform.point(point);
        out[i] = .{ .kind = if (point.positive) .positive_point else .negative_point, .point = unit(at[0], at[1], extent) };
        i += 1;
    }
    out[i] = .{ .kind = .terminator };
    return out;
}

fn learnedRow(encoder: Scope, leaf: []const u8, width: usize) ![]const f32 {
    const row = try encoder.param(leaf);
    if (row.count != width) return error.SamPromptShape;
    return row.slice(f32, width);
}

fn sparseRows(rt: *Runtime, encoder: Scope, rows: []const Record) !Tensor {
    const width = rt.config.hidden;
    const basis = try encodings.FourierBasis.load(rt, try encoder.key("pe_layer.positional_encoding_gaussian_matrix"));
    if (basis.width != width) return error.SamPromptShape;
    var types: [4][]const f32 = undefined;
    for (&types, 0..) |*row, t| row.* = try learnedRow(encoder, try rt.print("point_embeddings.{d}.weight", .{t}), width);
    const terminator = try learnedRow(encoder, "not_a_point_embed.weight", width);
    const tensor = try rt.hostTensor(rows.len * width);
    const values = tensor.slice(f32, rows.len * width);
    for (rows, 0..) |record, i| {
        const row = values[i * width ..][0..width];
        if (record.kind == .terminator) {
            // The zero code is added so that a stored -0 becomes +0.
            for (row, terminator) |*out, learned| out.* = @as(f32, 0) + learned;
            continue;
        }
        basis.encode(record.point, row);
        for (row, types[@backingInt(record.kind)]) |*out, learned| out.* = out.* + learned;
    }
    return tensor;
}

fn normStage(rt: *Runtime, scope: Scope, conv_leaf: []const u8, norm_leaf: []const u8, x: FeatureMap) !FeatureMap {
    const y = try convolution.conv(scope, conv_leaf, x, .{ .stride = 2, .pad = 0 });
    const normed = try norms.layerNorm(scope, norm_leaf, y.tensor, mask_norm_eps, .plain);
    return FeatureMap.of(try elementwise.gelu(rt, normed), y.height, y.width, y.channels);
}

// Probabilities on the model frame, the 4 × 4 stride-4 downsampler (252), then bilinear to the mask grid.
fn probabilityGrid(rt: *Runtime, tracker: Scope, downsample: []const u8, input: FeatureMap) !FeatureMap {
    const config = rt.config;
    const side = config.maskSide();
    const source = config.image_size;
    const enlarged = try spatial.resample(rt, input, source, source, .bilinear);
    const reduced = try convolution.conv(tracker, downsample, enlarged, .{ .stride = 4, .pad = 0 });
    if (reduced.height == side) return reduced;
    return spatial.resample(rt, reduced, side, side, .bilinear);
}

fn denseRows(rt: *Runtime, encoder: Scope, mask: prompts.Mask) !Tensor {
    const config = rt.config;
    const grid = config.featureSide();
    const side = config.maskSide();
    const tracker = Scope.root(rt, config.roots.tracker);
    const downsample = if (config.version == .sam31) "interactive_mask_downsample" else "mask_downsample";
    const scaling = try encoder.sub("mask_downscaling");
    const input = FeatureMap.of(try rt.upload(mask.values), mask.size.height, mask.size.width, 1);
    var x = switch (mask.format) {
        .probabilities => try probabilityGrid(rt, tracker, downsample, input),
        .logits => try spatial.resample(rt, input, side, side, .bilinear),
    };
    try rt.gpu.beginBatch();
    errdefer if (rt.gpu.batch != null) rt.gpu.endBatch() catch {};
    if (x.height != side or x.width != side or x.channels != 1) return error.SamPromptShape;
    x = try normStage(rt, scaling, "0", "1", x);
    x = try normStage(rt, scaling, "3", "4", x);
    x = try convolution.conv(scaling, "6", x, .{});
    try rt.gpu.endBatch();
    if (x.height != grid or x.width != grid or x.channels != config.hidden) return error.SamPromptShape;
    return x.tensor;
}

pub fn encode(rt: *Runtime, prompt: prompts.Prompt, size: prompts.Size) !PromptTokens {
    try prompt.validate(size);
    const rows = try records(rt.scratch, prompt, size, rt.config.image_size);
    const width = rt.config.hidden;
    const grid = rt.config.featureSide();
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const encoder = Scope.root(rt, rt.config.roots.prompt_encoder);
    var tokens = PromptTokens{ .sparse_rows = rows.len };
    if (rows.len > 0) tokens.sparse = try sparseRows(rt, encoder, rows);
    if (prompt.mask) |mask| tokens.dense = try denseRows(rt, encoder, mask);
    if (tokens.sparse) |sparse| try rt.observe("prompt.sparse", sparse, &.{ 1, rows.len, width });
    if (tokens.dense) |dense| try rt.observe("prompt.dense", dense, &.{ 1, grid, grid, width });
    var keep: [2]Tensor = undefined;
    var kept: usize = 0;
    for ([_]?Tensor{ tokens.sparse, tokens.dense }) |maybe| if (maybe) |tensor| {
        keep[kept] = tensor;
        kept += 1;
    };
    rt.rewind(m, keep[0..kept]);
    return tokens;
}

fn expectKinds(prompt: prompts.Prompt, expected: []const Kind) !void {
    const a = std.testing.allocator;
    const rows = try records(a, prompt, .{ .width = 960, .height = 540 }, 1008);
    defer a.free(rows);
    try std.testing.expectEqual(expected.len, rows.len);
    try std.testing.expectEqual(expected.len, rowCount(prompt));
    for (rows, expected) |row, kind| try std.testing.expectEqual(kind, row.kind);
}

test "SAM prompt rows put box corners before points and end with one terminator" {
    const point = prompts.Point{ .x = 480, .y = 270 };
    const negative = prompts.Point{ .x = 480, .y = 270, .positive = false };
    const box = prompts.Box{ .x0 = 10, .y0 = 20, .x1 = 300, .y1 = 400 };
    try expectKinds(.{ .points = &.{point} }, &.{ .positive_point, .terminator });
    try expectKinds(.{ .points = &.{negative} }, &.{ .negative_point, .terminator });
    try expectKinds(.{ .boxes = &.{box} }, &.{ .top_left, .bottom_right, .terminator });
    try expectKinds(.{ .points = &.{ point, point }, .boxes = &.{box} }, &.{ .top_left, .bottom_right, .positive_point, .positive_point, .terminator });
    const mask = prompts.Mask{ .size = .{ .width = 1, .height = 1 }, .values = &.{0.5} };
    try expectKinds(.{ .mask = mask }, &.{});
    const refused = records(std.testing.allocator, .{ .boxes = &.{.{ .x0 = 0, .y0 = 0, .x1 = 4, .y1 = 4, .positive = false }} }, .{ .width = 960, .height = 540 }, 1008);
    try std.testing.expectError(error.NegativeBoxUnsupported, refused);
}

test "SAM prompt coordinates shift by half a pixel in the model frame" {
    const a = std.testing.allocator;
    const rows = try records(a, .{ .points = &.{.{ .x = 480, .y = 270 }} }, .{ .width = 960, .height = 540 }, 1008);
    defer a.free(rows);
    const expected: f32 = @as(f32, 504.5) / @as(f32, 1008);
    try std.testing.expectEqual([2]f32{ expected, expected }, rows[0].point);
    try std.testing.expectEqual([2]f32{ 0, 0 }, rows[1].point);
}
