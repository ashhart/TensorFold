const std = @import("std");
const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;
const geometry = @import("concept_geometry.zig");
const masks = @import("concept_masks.zig");
const position = @import("position.zig");

pub const Detection = struct { mask: ops.Feature, box: [4]f32, score: f32 };
pub const Options = struct { threshold: f32 = 0.5, max_results: usize = 200 };

fn attention(block: Layer, name: []const u8, query: ops.Tensor, key: ops.Tensor, value: ops.Tensor, bias: ?ops.Tensor) !ops.Tensor {
    return block.runtime.mha(try block.named(name), query, key, value, 8, block.runtime.config.hidden, bias);
}

const Context = struct {
    image: ops.Tensor,
    grid: ops.Tensor,
    prompt: ops.Tensor,

    fn fuse(self: Context, root: Layer) !ops.Tensor {
        const r = root.runtime;
        const start = r.mark();
        var value = self.image;
        for (0..6) |index| {
            const block = try root.indexed("layers", index);
            const norm1 = try block.norm("norm1", value, 1e-5);
            const spatial = try r.add(norm1, self.grid);
            const local = try attention(block, "self_attn", spatial, spatial, norm1, null);
            value = try r.add(value, local);
            const norm2 = try block.norm("norm2", value, 1e-5);
            const prompted = try attention(block, "cross_attn_image", norm2, self.prompt, self.prompt, null);
            value = try r.add(value, prompted);
            const norm3 = try block.norm("norm3", value, 1e-5);
            value = try r.add(value, try block.feedForward(norm3, false));
            r.trim(start, &.{value});
        }
        return value;
    }
};

const Queries = struct {
    tokens: ops.Tensor,
    boxes: ops.Tensor,
    count: usize,

    fn initialize(root: Layer) !Queries {
        const r = root.runtime;
        const anchors = try root.parameter("reference_points.weight");
        const symbols = try root.parameter("query_embed.weight");
        if (anchors.count != 800 or symbols.count != 200 * r.config.hidden) return error.SamConceptQueryShape;
        const presence = try root.parameter("presence_token.weight");
        return .{ .tokens = try r.concat(&.{ presence, symbols }), .boxes = try r.unary(anchors, 2), .count = 200 };
    }

    fn decode(self: *Queries, root: Layer, context: Context, memory: ops.Tensor) !void {
        const r = root.runtime;
        const width = r.config.hidden;
        const start = r.mark();
        const presence_position = try r.alloc(width);
        @memset(presence_position.slice(f32, width), 0);
        for (0..6) |index| {
            const block = try root.indexed("layers", index);
            const fourier = try geometry.boxEmbedding(r, self.boxes);
            const learned_position = try root.mlp("ref_point_head", fourier, 2);
            const positions = try r.concat(&.{ presence_position, learned_position });
            const positioned = try r.add(self.tokens, positions);
            const local = try attention(block, "self_attn", positioned, positioned, self.tokens, null);
            var value = try block.norm("norm2", try r.add(self.tokens, local), 1e-5);
            const text_query = try r.add(value, positions);
            const prompted = try attention(block, "ca_text", text_query, context.prompt, context.prompt, null);
            value = try block.norm("catext_norm", try r.add(value, prompted), 1e-5);
            const image_keys = try r.add(memory, context.grid);
            const bias = try geometry.boxBias(root, self.boxes);
            const image_query = try r.add(value, positions);
            const localized = try attention(block, "cross_attn", image_query, image_keys, memory, bias);
            value = try block.norm("norm1", try r.add(value, localized), 1e-5);
            self.tokens = try block.norm("norm3", try r.add(value, try block.feedForward(value, false)), 1e-5);
            const objects = try root.norm("norm", ops.view(self.tokens, width, self.count * width), 1e-5);
            self.boxes = try geometry.refine(r, self.boxes, try root.mlp("bbox_embed", objects, 3));
            r.trim(start, &.{ presence_position, self.tokens, self.boxes });
        }
    }
};

const Candidate = struct {
    index: usize,
    score: f32,

    fn higher(_: void, a: Candidate, b: Candidate) bool {
        if (a.score == b.score) return a.index < b.index;
        return a.score > b.score;
    }
};

fn classify(root: Layer, prompt: ops.Tensor, queries: Queries, normalized: ops.Tensor, options: Options) ![]Candidate {
    const r = root.runtime;
    const width = r.config.hidden;
    const scorer = Layer{ .runtime = r, .prefix = "detector.dot_prod_scoring" };
    const prompt_update = try scorer.mlp("prompt_mlp", prompt, 2);
    const prompt_hidden = try scorer.norm("prompt_mlp.out_norm", try r.add(prompt, prompt_update), 1e-5);
    const pool = try r.alloc(width);
    const length = prompt.count / width;
    for (pool.slice(f32, width), 0..) |*value, channel| {
        var sum: f32 = 0;
        for (0..length) |row| sum += prompt_hidden.get(row * width + channel);
        value.* = sum / @as(f32, @floatFromInt(length));
    }
    const text = try scorer.linear("prompt_proj", pool);
    const objects = try scorer.linear("hs_proj", normalized);
    const presence = try root.norm("presence_token_out_norm", ops.view(queries.tokens, 0, width), 1e-5);
    const logit = try root.mlp("presence_token_head", presence, 3);
    const probability = 1 / (1 + @exp(-logit.get(0)));
    const candidates = try r.a.alloc(Candidate, queries.count);
    var count: usize = 0;
    for (0..queries.count) |index| {
        var dot: f32 = 0;
        for (0..width) |channel| dot += text.get(channel) * objects.get(index * width + channel);
        const scaled = dot / @sqrt(@as(f32, @floatFromInt(width)));
        const score = probability / (1 + @exp(-std.math.clamp(scaled, -12, 12)));
        if (score < options.threshold) continue;
        candidates[count] = .{ .index = index, .score = score };
        count += 1;
    }
    std.mem.sort(Candidate, candidates[0..count], {}, Candidate.higher);
    return candidates[0..@min(count, options.max_results)];
}

pub fn detect(r: *ops.Runtime, features: [3]ops.Feature, text: ops.Tensor, options: Options) ![]Detection {
    if (!std.math.isFinite(options.threshold) or options.threshold < 0 or options.threshold > 1 or
        options.max_results < 1 or options.max_results > 200) return error.InvalidDetectionOptions;
    if (text.count == 0 or text.count % r.config.hidden != 0) return error.SamConceptQueryShape;
    const start = r.mark();
    errdefer r.trim(start, &.{});
    const grid = try position.spatial(r, features[2].height, features[2].width, r.config.hidden);
    const class_token = try geometry.classToken(r, features[2].tensor, grid);
    const context = Context{ .image = features[2].tensor, .grid = grid, .prompt = try r.concat(&.{ text, class_token }) };
    const root = Layer{ .runtime = r, .prefix = "detector.transformer" };
    const encoder = try root.child("encoder");
    const decoder = try root.child("decoder");
    const memory = try context.fuse(encoder);
    var queries = try Queries.initialize(decoder);
    try queries.decode(decoder, context, memory);
    const width = r.config.hidden;
    const normalized = try decoder.norm("norm", ops.view(queries.tokens, width, queries.count * width), 1e-5);
    const candidates = try classify(decoder, context.prompt, queries, normalized, options);
    const results = try r.a.alloc(Detection, candidates.len);
    if (results.len == 0) {
        r.trim(start, &.{});
        return results;
    }
    const selected = try r.alloc(results.len * width);
    const data = selected.slice(f32, selected.count);
    for (candidates, 0..) |candidate, row| {
        for (data[row * width ..][0..width], 0..) |*value, channel| value.* = normalized.get(candidate.index * width + channel);
    }
    const image = try masks.pixels(r, features, memory, context.prompt);
    const logits = try masks.classify(r, image, selected);
    const retained = try r.a.alloc(ops.Tensor, results.len);
    for (candidates, results, retained, 0..) |candidate, *result, *keep, column| {
        keep.* = try r.columns(logits.tensor, image.height * image.width, results.len, column, 1);
        const offset = 4 * candidate.index;
        const cx = queries.boxes.get(offset);
        const cy = queries.boxes.get(offset + 1);
        const rx = 0.5 * queries.boxes.get(offset + 2);
        const ry = 0.5 * queries.boxes.get(offset + 3);
        result.* = .{
            .mask = ops.feature(keep.*, image.height, image.width, 1),
            .box = .{ cx - rx, cy - ry, cx + rx, cy + ry },
            .score = candidate.score,
        };
    }
    r.trim(start, retained);
    return results;
}
