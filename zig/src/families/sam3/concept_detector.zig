const std = @import("std");
const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;
const geometry = @import("concept_geometry.zig");
const masks = @import("concept_masks.zig");
const position = @import("position.zig");

pub const Detection = struct { mask: ops.Feature, box: [4]f32, score: f32 };
pub const Options = struct { threshold: f32 = 0.5, max_results: usize = 200 };

const Scene = struct {
    image: ops.Tensor,
    position: ops.Tensor,
    prompt: ops.Tensor,

    fn encode(self: Scene, root: Layer) !ops.Tensor {
        const r = root.runtime;
        const mark = r.mark();
        var image = self.image;
        for (0..6) |index| {
            const layer = try root.indexed("layers", index);
            const normalized = try layer.norm("norm1", image, 1e-5);
            const keys = try r.add(normalized, self.position);
            image = try r.add(image, try attention(layer, "self_attn", keys, keys, normalized, null));
            const query = try layer.norm("norm2", image, 1e-5);
            image = try r.add(image, try attention(layer, "cross_attn_image", query, self.prompt, self.prompt, null));
            image = try r.add(image, try layer.feedForward(try layer.norm("norm3", image, 1e-5), false));
            r.trim(mark, &.{image});
        }
        return image;
    }
};

fn attention(layer: Layer, name: []const u8, q: ops.Tensor, k: ops.Tensor, v: ops.Tensor, bias: ?ops.Tensor) !ops.Tensor {
    return layer.runtime.mha(try layer.named(name), q, k, v, 8, layer.runtime.config.hidden, bias);
}

const ObjectQueries = struct {
    values: ops.Tensor,
    boxes: ops.Tensor,
    count: usize,

    fn init(root: Layer) !ObjectQueries {
        const r = root.runtime;
        const anchors = try root.parameter("reference_points.weight");
        const learned = try root.parameter("query_embed.weight");
        const count = anchors.count / 4;
        if (anchors.count % 4 != 0 or count != 200 or learned.count != count * r.config.hidden) return error.SamConceptQueryShape;
        return .{ .values = try r.concat(&.{ try root.parameter("presence_token.weight"), learned }), .boxes = try r.unary(anchors, 2), .count = count };
    }

    fn refine(self: *ObjectQueries, root: Layer, scene: Scene, memory: ops.Tensor) !void {
        const r = root.runtime;
        const width = r.config.hidden;
        const mark = r.mark();
        const empty = try r.alloc(width);
        @memset(empty.slice(f32, width), 0);
        for (0..6) |index| {
            const layer = try root.indexed("layers", index);
            const locations = try root.mlp("ref_point_head", try geometry.boxEmbedding(r, self.boxes), 2);
            const positional = try r.concat(&.{ empty, locations });
            const keys = try r.add(self.values, positional);
            var value = try layer.norm("norm2", try r.add(self.values, try attention(layer, "self_attn", keys, keys, self.values, null)), 1e-5);
            value = try layer.norm("catext_norm", try r.add(value, try attention(layer, "ca_text", try r.add(value, positional), scene.prompt, scene.prompt, null)), 1e-5);
            const spatial_keys = try r.add(memory, scene.position);
            const bias = try geometry.boxBias(root, self.boxes);
            value = try layer.norm("norm1", try r.add(value, try attention(layer, "cross_attn", try r.add(value, positional), spatial_keys, memory, bias)), 1e-5);
            self.values = try layer.norm("norm3", try r.add(value, try layer.feedForward(value, false)), 1e-5);
            const normalized = try root.norm("norm", ops.view(self.values, width, self.count * width), 1e-5);
            self.boxes = try geometry.refine(r, self.boxes, try root.mlp("bbox_embed", normalized, 3));
            r.trim(mark, &.{ empty, self.values, self.boxes });
        }
    }
};

const Candidate = struct {
    query: usize,
    score: f32,

    fn before(_: void, left: Candidate, right: Candidate) bool {
        return left.score > right.score or (left.score == right.score and left.query < right.query);
    }
};

fn rank(root: Layer, prompt: ops.Tensor, objects: ObjectQueries, normalized: ops.Tensor, options: Options) ![]Candidate {
    const r = root.runtime;
    const width = r.config.hidden;
    const scorer = Layer{ .runtime = r, .prefix = "detector.dot_prod_scoring" };
    const transformed = try scorer.norm("prompt_mlp.out_norm", try r.add(prompt, try scorer.mlp("prompt_mlp", prompt, 2)), 1e-5);
    const pool = try r.alloc(width);
    for (pool.slice(f32, width), 0..) |*value, channel| {
        var sum: f32 = 0;
        for (0..prompt.count / width) |token| sum += transformed.get(token * width + channel);
        value.* = sum / @as(f32, @floatFromInt(prompt.count / width));
    }
    const text = try scorer.linear("prompt_proj", pool);
    const image = try scorer.linear("hs_proj", normalized);
    const presence = try root.norm("presence_token_out_norm", ops.view(objects.values, 0, width), 1e-5);
    const present = try root.mlp("presence_token_head", presence, 3);
    const probability = 1 / (1 + @exp(-present.get(0)));
    const candidates = try r.a.alloc(Candidate, objects.count);
    var accepted: usize = 0;
    for (0..objects.count) |query| {
        var dot: f32 = 0;
        for (0..width) |channel| dot += text.get(channel) * image.get(query * width + channel);
        const score = probability / (1 + @exp(-std.math.clamp(dot / @sqrt(@as(f32, @floatFromInt(width))), -12, 12)));
        if (score < options.threshold) continue;
        candidates[accepted] = .{ .query = query, .score = score };
        accepted += 1;
    }
    std.mem.sort(Candidate, candidates[0..accepted], {}, Candidate.before);
    return candidates[0..@min(accepted, options.max_results)];
}

pub fn detect(r: *ops.Runtime, features: [3]ops.Feature, text: ops.Tensor, options: Options) ![]Detection {
    if (!std.math.isFinite(options.threshold) or options.threshold < 0 or options.threshold > 1 or options.max_results == 0 or options.max_results > 200)
        return error.InvalidDetectionOptions;
    const mark = r.mark();
    errdefer r.trim(mark, &.{});
    const spatial = try position.spatial(r, features[2].height, features[2].width, r.config.hidden);
    const scene = Scene{
        .image = features[2].tensor,
        .position = spatial,
        .prompt = try r.concat(&.{ text, try geometry.classToken(r, features[2].tensor, spatial) }),
    };
    const encoder = Layer{ .runtime = r, .prefix = "detector.transformer.encoder" };
    const decoder = Layer{ .runtime = r, .prefix = "detector.transformer.decoder" };
    const memory = try scene.encode(encoder);
    var objects = try ObjectQueries.init(decoder);
    try objects.refine(decoder, scene, memory);
    const normalized = try decoder.norm("norm", ops.view(objects.values, r.config.hidden, objects.count * r.config.hidden), 1e-5);
    const selected = try rank(decoder, scene.prompt, objects, normalized, options);
    const output = try r.a.alloc(Detection, selected.len);
    if (selected.len == 0) {
        r.trim(mark, &.{});
        return output;
    }
    const chosen = try r.alloc(selected.len * r.config.hidden);
    for (selected, 0..) |candidate, index| for (0..r.config.hidden) |channel| {
        chosen.slice(f32, chosen.count)[index * r.config.hidden + channel] = normalized.get(candidate.query * r.config.hidden + channel);
    };
    const pixels = try masks.pixels(r, features, memory, scene.prompt);
    const logits = try masks.classify(r, pixels, chosen);
    const retained = try r.a.alloc(ops.Tensor, selected.len);
    for (selected, output, retained, 0..) |candidate, *result, *keep, index| {
        const box = ops.view(objects.boxes, candidate.query * 4, 4);
        const cx = box.get(0);
        const cy = box.get(1);
        const rx = box.get(2) * 0.5;
        const ry = box.get(3) * 0.5;
        keep.* = try r.columns(logits.tensor, pixels.height * pixels.width, selected.len, index, 1);
        result.* = .{ .mask = ops.feature(keep.*, pixels.height, pixels.width, 1), .box = .{ cx - rx, cy - ry, cx + rx, cy + ry }, .score = candidate.score };
    }
    r.trim(mark, retained);
    return output;
}
