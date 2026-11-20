const std = @import("std");
const runtime = @import("runtime.zig");
const elementwise = @import("elementwise.zig");
const projection = @import("projection.zig");
const norms = @import("norms.zig");
const attention = @import("attention.zig");
const convolution = @import("convolution.zig");
const spatial = @import("spatial.zig");
const concept_context = @import("concept_context.zig");
const concept_queries = @import("concept_queries.zig");

const Runtime = runtime.Runtime;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;
const Context = concept_context.Context;
const Queries = concept_queries.Queries;
const heads = concept_context.heads;
const layerNorm = concept_context.layerNorm;
const sigmoid = concept_queries.sigmoid;
const query_count = concept_queries.query_count;

pub const score_clamp: f32 = 12;
pub const joint_clamp: f32 = 10;
pub const pixel_decoder_groups = 8;
pub const pixel_decoder_stages = 2;

/// One kept query: a one-channel logit map on the mask grid, its normalised corner box and its score.
pub const Detection = struct { mask: FeatureMap, box: [4]f32, score: f32 };

pub const DetectOptions = struct {
    threshold: f32 = 0.5,
    max_results: usize = 200,

    pub fn validate(self: DetectOptions) error{InvalidDetectionOptions}!void {
        if (!std.math.isFinite(self.threshold) or self.threshold < 0 or self.threshold > 1 or
            self.max_results < 1 or self.max_results > query_count) return error.InvalidDetectionOptions;
    }
};

/// Context, grounding, queries, scoring and mask read-out; must start with no batch open.
pub fn detect(rt: *Runtime, levels: [3]FeatureMap, text_features: Tensor, options: DetectOptions) ![]Detection {
    try options.validate();
    const c = rt.config;
    const width = c.hidden;
    if (text_features.count == 0 or text_features.count % width != 0) return error.SamConceptShape;
    for (levels, 0..) |level, i| {
        const side = c.featureSide() << @intCast(2 - i);
        if (level.height != side or level.width != side or level.channels != width or level.tensor.count != side * side * width)
            return error.SamConceptShape;
    }
    if (rt.gpu.batch != null) return error.SamBatchOpen;
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    errdefer if (rt.gpu.batch != null) rt.gpu.endBatch() catch {};

    const context = try concept_context.build(rt, levels[2], text_features);
    const grounded = try concept_context.ground(rt, context);
    const queries = try concept_queries.refine(rt, context, grounded);
    const scored = try scoreQueries(rt, context, queries);
    const order = try rt.scratch.alloc(usize, query_count);
    const kept = keepQueries(scored.query_scores, options.threshold, options.max_results, order);
    if (kept == 0) {
        rt.rewind(m, &.{});
        return rt.scratch.alloc(Detection, 0);
    }

    try rt.gpu.beginBatch();
    const basis = try maskBasis(rt, levels, context, grounded);
    const kept_rows = try rt.scratch.alloc(Tensor, kept);
    for (order[0..kept], kept_rows) |q, *row| row.* = runtime.part(scored.normed, q * width, width);
    const segmentation = try Scope.root(rt, c.roots.detector).sub("segmentation_head");
    const embeds = try projection.reluChain(segmentation, "mask_predictor.mask_embed", try elementwise.concat(rt, kept_rows), 3);
    const cells = basis.height * basis.width;
    const logits = try projection.matmul(rt, basis.tensor, embeds, cells, kept, width);
    const detections = try rt.scratch.alloc(Detection, kept);
    const keep = try rt.scratch.alloc(Tensor, kept);
    const boxes = queries.boxes.slice(f32, queries.boxes.count);
    for (detections, keep, order[0..kept], 0..) |*d, *held, q, column| {
        held.* = try elementwise.columns(rt, logits, cells, kept, column, 1);
        d.* = .{
            .mask = FeatureMap.of(held.*, basis.height, basis.width, 1),
            .box = corners(boxes[4 * q ..][0..4].*),
            .score = scored.query_scores[q],
        };
    }
    try rt.gpu.endBatch();
    rt.rewind(m, keep);
    return detections;
}

const Scored = struct { normed: Tensor, query_scores: []f32 };

// Score inputs are read on the host between batches; the pooled cue vector is projected on the device.
fn scoreQueries(rt: *Runtime, context: Context, queries: Queries) !Scored {
    const c = rt.config;
    const width = c.hidden;
    const detector = Scope.root(rt, c.roots.detector);
    const decoder = try detector.sub("transformer.decoder");
    const scoring = try detector.sub("dot_prod_scoring");
    if (queries.tokens.count != (queries.count + 1) * width) return error.SamQueryShape;

    try rt.gpu.beginBatch();
    const normed = try layerNorm(decoder, "norm", runtime.part(queries.tokens, width, queries.count * width));
    const lifted = try projection.reluChain(scoring, "prompt_mlp", context.cues, 2);
    const cue_rows = try layerNorm(scoring, "prompt_mlp.out_norm", try elementwise.add(rt, context.cues, lifted));
    const embedded = try projection.affine(scoring, "hs_proj", normed);
    const presence_row = try layerNorm(decoder, "presence_token_out_norm", runtime.part(queries.tokens, 0, width));
    const presence = try projection.reluChain(decoder, "presence_token_head", presence_row, 3);
    try rt.gpu.endBatch();

    if (presence.count != 1 or embedded.count != queries.count * width) return error.SamQueryShape;
    const pooled = try rt.hostTensor(width);
    cueMean(pooled.slice(f32, width), cue_rows.slice(f32, cue_rows.count), context.cue_rows, width);
    try rt.gpu.beginBatch();
    const target = try projection.affine(scoring, "prompt_proj", pooled);
    try rt.gpu.endBatch();

    if (target.count != width) return error.SamQueryShape;
    const query_scores = try rt.scratch.alloc(f32, queries.count);
    const t = target.slice(f32, width);
    const e = embedded.slice(f32, embedded.count);
    const presence_logit = presence.slice(f32, 1)[0];
    const joint = c.version == .sam31;
    for (query_scores, 0..) |*s, q| {
        const logit = scoreLogit(dot(t, e[q * width ..][0..width]), width);
        s.* = if (joint) jointScore(presence_logit, logit) else composeScore(presence_logit, logit);
    }
    return .{ .normed = normed, .query_scores = query_scores };
}

// Cue-conditioned grid, two-stage pixel decoder (144, then 288) and the per-pixel mask basis.
fn maskBasis(rt: *Runtime, levels: [3]FeatureMap, context: Context, grounded: Tensor) !FeatureMap {
    const c = rt.config;
    const side = c.featureSide();
    const segmentation = try Scope.root(rt, c.roots.detector).sub("segmentation_head");
    const decoder = try segmentation.sub("pixel_decoder");
    const normed = try layerNorm(segmentation, "cross_attn_norm", grounded);
    const cued = try attention.packedHeads(try segmentation.sub("cross_attend_prompt"), normed, context.cues, context.cues, heads, .{});
    var v = FeatureMap.of(try elementwise.add(rt, grounded, cued), side, side, c.hidden);
    for (0..pixel_decoder_stages) |stage| {
        const skip = levels[1 - stage];
        const enlarged = try spatial.resample(rt, v, skip.height, skip.width, .nearest);
        const joined = FeatureMap.of(try elementwise.add(rt, enlarged.tensor, skip.tensor), skip.height, skip.width, skip.channels);
        const convolved = try convolution.conv(decoder, try rt.print("conv_layers.{d}", .{stage}), joined, .{ .pad = 1 });
        const grouped = try norms.groupNorm(decoder, try rt.print("norms.{d}", .{stage}), convolved, pixel_decoder_groups);
        v = FeatureMap.of(try elementwise.relu(rt, grouped.tensor), grouped.height, grouped.width, grouped.channels);
    }
    return convolution.conv(segmentation, "instance_seg_head", v, .{});
}

fn dot(a: []const f32, b: []const f32) f32 {
    var sum: f32 = 0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

/// clamp(dot / √width, −12, 12).
pub fn scoreLogit(product: f32, width: usize) f32 {
    return std.math.clamp(product / @sqrt(@as(f32, @floatFromInt(width))), -score_clamp, score_clamp);
}

/// σ(presence logit)·σ(score logit).
pub fn composeScore(presence_logit: f32, score_logit: f32) f32 {
    return sigmoid(presence_logit) * sigmoid(score_logit);
}

/// SAM3.1: the composed score as a logit clamped to ±10, then σ again.
pub fn jointScore(presence_logit: f32, score_logit: f32) f32 {
    const p = composeScore(presence_logit, score_logit);
    return sigmoid(std.math.clamp(@log(p / (1 - p)), -joint_clamp, joint_clamp));
}

/// Mean of `rows` rows of `width` values: rows summed in order, then one division by the row count.
pub fn cueMean(out: []f32, values: []const f32, rows: usize, width: usize) void {
    @memset(out, 0);
    for (0..rows) |r| for (out, values[r * width ..][0..width]) |*acc, v| {
        acc.* += v;
    };
    const count: f32 = @floatFromInt(rows);
    for (out) |*acc| acc.* /= count;
}

/// Queries scoring strictly above the threshold, capped to the best scores (ties to lower), in ascending query order.
pub fn keepQueries(scores: []const f32, threshold: f32, max_results: usize, order: []usize) usize {
    var kept: usize = 0;
    for (scores, 0..) |s, q| if (s > threshold) {
        order[kept] = q;
        kept += 1;
    };
    std.mem.sort(usize, order[0..kept], scores, before);
    kept = @min(kept, max_results);
    std.mem.sort(usize, order[0..kept], {}, std.sort.asc(usize));
    return kept;
}

fn before(scores: []const f32, a: usize, b: usize) bool {
    return scores[a] > scores[b] or (scores[a] == scores[b] and a < b);
}

/// (cx, cy, w, h) → (x0, y0, x1, y1), not clamped.
pub fn corners(box: [4]f32) [4]f32 {
    return .{ box[0] - 0.5 * box[2], box[1] - 0.5 * box[3], box[0] + 0.5 * box[2], box[1] + 0.5 * box[3] };
}

/// Normalised corners scaled by (W, H, W, H) of the source image.
pub fn scaleBox(box: [4]f32, width: usize, height: usize) [4]f32 {
    const w: f32 = @floatFromInt(width);
    const h: f32 = @floatFromInt(height);
    return .{ box[0] * w, box[1] * h, box[2] * w, box[3] * h };
}

const testing = std.testing;

test "SAM detection corners are not clamped" {
    try testing.expectEqual([4]f32{ 0.4, 0.3, 0.6, 0.7 }, corners(.{ 0.5, 0.5, 0.2, 0.4 }));
    const left = corners(.{ 0.05, 0.5, 0.2, 0.2 });
    try testing.expectApproxEqAbs(@as(f32, -0.05), left[0], 1e-7);
    try testing.expect(left[0] < 0);
}

test "SAM detection scores compose presence and clamped query logits" {
    try testing.expectEqual(@as(f32, 0.25), composeScore(0, scoreLogit(0, 256)));
    try testing.expectEqual(@as(f32, 12), scoreLogit(1000, 256));
    try testing.expectEqual(@as(f32, -12), scoreLogit(-1000, 256));
    try testing.expectEqual(sigmoid(0) * sigmoid(12), composeScore(0, scoreLogit(1000, 256)));
    try testing.expectEqual(@as(f32, 0.5), scoreLogit(8, 256));
    try testing.expectEqual(sigmoid(10), jointScore(12, 12));
    try testing.expectEqual(sigmoid(-10), jointScore(-12, -12));
    try testing.expectApproxEqAbs(composeScore(1, 0.5), jointScore(1, 0.5), 1e-6);
}

test "SAM detection selection is strict, ordered and capped" {
    var order: [4]usize = undefined;
    const scores = [_]f32{ 0.9, 0.7, 0.9, 0.6 };
    try testing.expectEqual(@as(usize, 4), keepQueries(&scores, 0.5, 200, &order));
    try testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, &order);
    try testing.expectEqual(@as(usize, 2), keepQueries(&scores, 0.5, 2, &order));
    try testing.expectEqualSlices(usize, &.{ 0, 2 }, order[0..2]);
    try testing.expectEqual(@as(usize, 2), keepQueries(&.{ 0.6, 0.7, 0.9 }, 0.5, 2, &order));
    try testing.expectEqualSlices(usize, &.{ 1, 2 }, order[0..2]);
    try testing.expectEqual(@as(usize, 1), keepQueries(&.{ 0.5, 0.75, std.math.nan(f32) }, 0.5, 200, &order));
    try testing.expectEqual(@as(usize, 1), order[0]);
    try testing.expectEqual(@as(usize, 0), keepQueries(&.{ 0, 0 }, 0, 200, &order));
}

test "SAM detection options are checked before any work" {
    for ([_]f32{ std.math.nan(f32), -0.01, 1.01, std.math.inf(f32) }) |threshold|
        try testing.expectError(error.InvalidDetectionOptions, (DetectOptions{ .threshold = threshold }).validate());
    for ([_]usize{ 0, 201 }) |cap| try testing.expectError(error.InvalidDetectionOptions, (DetectOptions{ .max_results = cap }).validate());
    for ([_]f32{ 0, 1 }) |threshold| try (DetectOptions{ .threshold = threshold }).validate();
    for ([_]usize{ 1, 200 }) |cap| try (DetectOptions{ .max_results = cap }).validate();
}

test "SAM cue mean includes every cue row" {
    var out: [2]f32 = undefined;
    cueMean(&out, &.{ 1, 10, 2, 20, 6, 60 }, 3, 2);
    try testing.expectEqual([2]f32{ 3, 30 }, out);
}

test "SAM detection boxes scale by the source size" {
    try testing.expectEqual([4]f32{ -6.4, 30, 64, 90 }, scaleBox(.{ -0.05, 0.25, 0.5, 0.75 }, 128, 120));
}
