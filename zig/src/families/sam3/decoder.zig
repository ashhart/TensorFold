const std = @import("std");
const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;
const position = @import("position.zig");

pub const Result = struct {
    masks: ops.Feature,
    input_masks: ?ops.Feature = null,
    pointer: ops.Tensor,
    count: usize,
    scores: [16]f32,
    ious: [16]f32,

    pub fn retained(self: Result, storage: *[3]ops.Tensor) []const ops.Tensor {
        storage[0] = self.masks.tensor;
        storage[1] = self.pointer;
        if (self.input_masks) |mask| {
            storage[2] = mask.tensor;
            return storage;
        }
        return storage[0..2];
    }
};

pub const Options = struct {
    multiplex: bool = false,
    active: u16 = 1,
    multimask: bool = true,
    dense: ?ops.Tensor = null,
    sparse: ?ops.Tensor = null,
};

const TokenTable = struct {
    slots: usize,
    masks: usize,
    width: usize,

    fn token(self: TokenTable, values: ops.Tensor, offset: usize, slot: usize) ops.Tensor {
        return ops.view(values, (offset + slot) * self.width, self.width);
    }

    fn mask(self: TokenTable, values: ops.Tensor, slot: usize, candidate: usize) ops.Tensor {
        return self.token(values, self.slots * 2, slot * self.masks + candidate);
    }
};

fn queries(layer: Layer, table: TokenTable, options: Options) !ops.Tensor {
    const r = layer.runtime;
    const cache = try r.name("decoder.queries.{s}.{d}", .{ layer.prefix, options.active });
    if (r.constant(cache)) |cached| return cached;
    var masks = try layer.parameter("mask_tokens.weight");
    if (options.multiplex) {
        const owner = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
        const present = try owner.parameter("output_valid_embed");
        const absent = try owner.parameter("output_invalid_embed");
        const visibility = try r.alloc(masks.count);
        for (0..table.slots) |slot| {
            const chosen = if (options.active & (@as(u16, 1) << @intCast(slot)) == 0) absent else present;
            for (0..table.masks) |candidate| for (0..table.width) |channel| {
                visibility.slice(f32, visibility.count)[(slot * table.masks + candidate) * table.width + channel] = chosen.get(slot * table.width + channel);
            };
        }
        masks = try r.add(masks, visibility);
    }
    const value = try r.concat(&.{ try layer.parameter("obj_score_token.weight"), try layer.parameter("iou_token.weight"), masks });
    return if (options.active == 1) r.remember(cache, value) else value;
}

const Refinement = struct {
    image: ops.Tensor,
    queries: ops.Tensor,
    query_position: ops.Tensor,
    image_position: ops.Tensor,

    fn attention(layer: Layer, name: []const u8, q: ops.Tensor, k: ops.Tensor, v: ops.Tensor) !ops.Tensor {
        return layer.runtime.splitMha(try layer.named(name), q, k, v, 8);
    }

    fn step(self: *Refinement, layer: Layer, first: bool) !void {
        const r = layer.runtime;
        const positioned = if (first) self.queries else try r.add(self.queries, self.query_position);
        const attended = try attention(layer, "self_attn", positioned, positioned, self.queries);
        self.queries = try layer.norm("norm1", if (first) attended else try r.add(self.queries, attended), 1e-5);
        const image_keys = try r.add(self.image, self.image_position);
        const from_image = try attention(layer, "cross_attn_token_to_image", try r.add(self.queries, self.query_position), image_keys, self.image);
        self.queries = try layer.norm("norm2", try r.add(self.queries, from_image), 1e-5);
        const mlp = try layer.child("mlp");
        const nonlinear = try r.unary(try mlp.linear("lin1", self.queries), 0);
        self.queries = try layer.norm("norm3", try r.add(self.queries, try mlp.linear("lin2", nonlinear)), 1e-5);
        const from_tokens = try attention(layer, "cross_attn_image_to_token", image_keys, try r.add(self.queries, self.query_position), self.queries);
        self.image = try layer.norm("norm4", try r.add(self.image, from_tokens), 1e-5);
    }

    fn run(self: *Refinement, layer: Layer) !void {
        const mark = layer.runtime.mark();
        for (0..2) |index| {
            try self.step(try layer.indexed("layers", index), index == 0);
            layer.runtime.trim(mark, &.{ self.queries, self.image });
        }
        const r = layer.runtime;
        const attended = try attention(layer, "final_attn_token_to_image", try r.add(self.queries, self.query_position), try r.add(self.image, self.image_position), self.image);
        self.queries = try layer.norm("norm_final_attn", try r.add(self.queries, attended), 1e-5);
        r.trim(mark, &.{ self.queries, self.image });
    }
};

fn upsample(layer: Layer, image: ops.Tensor, skips: [3]ops.Feature) !ops.Feature {
    const r = layer.runtime;
    const side = r.config.featureSide();
    var output = ops.feature(image, side, side, r.config.hidden);
    for (0..2) |stage| {
        output = try r.conv(try layer.named(if (stage == 0) "output_upscaling.0" else "output_upscaling.3"), output, 2, 0, true, false);
        const skip = try r.conv(try layer.named(if (stage == 0) "conv_s1" else "conv_s0"), skips[1 - stage], 1, 0, false, false);
        output.tensor = try r.add(output.tensor, skip.tensor);
        if (stage == 0) output.tensor = try layer.norm("output_upscaling.1", output.tensor, 1e-6);
        output.tensor = try r.gelu(output.tensor);
    }
    return output;
}

const Choice = struct { candidate: usize = 0, score: f32 = -10, iou: f32 = 0 };

fn choose(table: TokenTable, quality: []const f32, scores: []const f32, multimask: bool) [16]Choice {
    var selected: [16]Choice = @splat(.{});
    for (selected[0..table.slots], 0..) |*choice, slot| {
        choice.candidate = @intFromBool(multimask);
        const count = if (table.slots > 1 or multimask) table.masks else 1;
        for (choice.candidate + 1..count) |candidate| {
            if (quality[slot * table.masks + candidate] > quality[slot * table.masks + choice.candidate]) choice.candidate = candidate;
        }
        choice.iou = quality[slot * table.masks + choice.candidate];
        choice.score = scores[slot];
    }
    return selected;
}

pub fn decode(r: *ops.Runtime, features: [3]ops.Feature, image: ops.Tensor, requested: Options) !Result {
    const image_side = r.config.featureSide();
    if (image.count != image_side * image_side * r.config.hidden) return error.SamDecoderShape;
    for (features, 0..) |feature, i| {
        const expected = image_side * (@as(usize, 4) >> @intCast(i));
        if (feature.height != expected or feature.width != expected or feature.channels != r.config.hidden or
            feature.tensor.count != expected * expected * feature.channels) return error.SamDecoderShape;
    }
    const mark = r.mark();
    errdefer r.trim(mark, &.{});
    var options = requested;
    options.multiplex = requested.multiplex and r.config.version == .sam31;
    const layer = Layer{ .runtime = r, .prefix = r.config.decoderPrefix(!options.multiplex) };
    const owner = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
    const symbols = try r.store.get(try layer.named("mask_tokens.weight"));
    const table = TokenTable{ .slots = if (options.multiplex) r.config.bucket_size else 1, .masks = if (options.multiplex) r.config.mask_tokens else 4, .width = r.config.hidden };
    if (!symbols.is(symbols.dtype, &.{ table.slots * table.masks, table.width })) return error.SamDecoderShape;
    var initial = try queries(layer, table, options);
    if (options.sparse) |sparse| {
        initial = try r.concat(&.{ initial, sparse });
    } else if (!options.multiplex) {
        const prompt = Layer{ .runtime = r, .prefix = r.config.promptPrefix() };
        const padding = try prompt.parameter("not_a_point_embed.weight");
        initial = try r.concat(&.{ initial, padding, padding });
    }
    const positional_key = if (options.multiplex) try owner.named("image_pe_layer.positional_encoding_gaussian_matrix") else try r.name("{s}.pe_layer.positional_encoding_gaussian_matrix", .{r.config.promptPrefix()});
    const dense = options.dense orelse if (options.multiplex) null else try r.weight(try r.name("{s}.no_mask_embed.weight", .{r.config.promptPrefix()}));
    const image_shape = [_]usize{ 1, features[2].height, features[2].width, r.config.hidden };
    try r.observe("decoder.image", image, &image_shape);
    if (options.dense == null and !options.multiplex) if (dense) |value| try r.observe("prompt.no_mask", value, &.{ 1, r.config.hidden });
    var state = Refinement{ .image = if (dense) |value| try r.add(image, value) else image, .queries = initial, .query_position = initial, .image_position = try position.grid(r, positional_key) };
    try r.observe("decoder.position", state.image_position, &image_shape);
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    try state.run(try layer.child("transformer"));
    const quality_logits = try layer.mlp("iou_prediction_head", ops.view(state.queries, table.slots * table.width, table.slots * table.width), 3);
    const quality = if (r.config.version == .sam3) try r.unary(quality_logits, 1) else quality_logits;
    const scores = try layer.mlp("pred_obj_score_head", ops.view(state.queries, 0, table.slots * table.width), 3);
    try r.g.endBatch();
    if (quality.count != table.slots * table.masks or scores.count != table.slots) return error.SamDecoderShape;
    try r.observe("decoder.quality", quality, &.{ table.slots, table.masks });
    const selected = choose(table, quality.slice(f32, quality.count), scores.slice(f32, scores.count), options.multimask and !options.multiplex);
    const side = r.config.featureSide() * 4;
    const empty_mask = try r.alloc(side * side);
    const empty_pointer = try r.alloc(table.width);
    @memset(empty_mask.slice(f32, empty_mask.count), -1024);
    @memset(empty_pointer.slice(f32, empty_pointer.count), 0);
    var mask_rows: [16]ops.Tensor = @splat(empty_mask);
    var pointer_rows: [16]ops.Tensor = @splat(empty_pointer);
    var output = Result{ .masks = undefined, .pointer = undefined, .count = table.slots, .scores = @splat(-10), .ious = @splat(0) };
    try r.g.beginBatch();
    const pixels: ?ops.Feature = if (needsPixels(selected[0..table.slots], options)) try upsample(layer, state.image, features) else null;
    for (selected[0..table.slots], 0..) |choice, slot| {
        output.scores[slot] = choice.score;
        output.ious[slot] = choice.iou;
        if (options.multiplex and options.active & (@as(u16, 1) << @intCast(slot)) == 0) continue;
        const mask_token = table.mask(state.queries, slot, choice.candidate);
        if (!(choice.score <= 0)) {
            const visible = pixels.?;
            const head = try r.name("output_hypernetworks_mlps.{d}", .{choice.candidate});
            const classifier = try layer.mlp(head, mask_token, 3);
            mask_rows[slot] = try r.matmul(visible.tensor, classifier, side * side, 1, visible.channels);
        }
        var pointer = try owner.mlp(if (r.config.version == .sam31 and !options.multiplex) "interactive_obj_ptr_proj" else "obj_ptr_proj", mask_token, 3);
        if (choice.score <= 0) {
            pointer = if (r.config.version == .sam31) try owner.linear("no_obj_ptr_linear", pointer) else try owner.parameter("no_obj_ptr");
            mask_rows[slot] = empty_mask;
        }
        pointer_rows[slot] = pointer;
    }
    output.masks = ops.feature(if (table.slots == 1) mask_rows[0] else try r.transpose(try r.concat(mask_rows[0..table.slots]), table.slots, side * side), side, side, table.slots);
    output.pointer = try r.concat(pointer_rows[0..table.slots]);
    try r.g.endBatch();
    r.trim(mark, &.{ output.masks.tensor, output.pointer });
    return output;
}

fn needsPixels(selected: []const Choice, options: Options) bool {
    for (selected, 0..) |choice, slot| {
        if (options.multiplex and options.active & (@as(u16, 1) << @intCast(slot)) == 0) continue;
        if (!(choice.score <= 0)) return true;
    }
    return false;
}

test "SAM absent masks skip pixels without hiding active NaN scores" {
    var selected: [16]Choice = @splat(.{});
    const options = Options{ .multiplex = true, .active = 1 };
    for ([_]f32{ -std.math.inf(f32), -1, -0.0, 0 }) |score| {
        selected[0].score = score;
        try std.testing.expect(!needsPixels(&selected, options));
    }
    for ([_]f32{ 0x1p-149, std.math.inf(f32), std.math.nan(f32) }) |score| {
        selected[0].score = score;
        try std.testing.expect(needsPixels(&selected, options));
    }
    selected[0].score = -1;
    selected[1].score = std.math.nan(f32);
    try std.testing.expect(!needsPixels(&selected, options));
    try std.testing.expect(needsPixels(&selected, .{ .multiplex = true, .active = 2 }));
    selected[0].score = 1;
    try std.testing.expect(needsPixels(selected[0..1], .{ .active = 0 }));
}

test "SAM tracked masks exclude candidate zero while explicit single-mask prompts retain it" {
    const table = TokenTable{ .slots = 1, .masks = 4, .width = 256 };
    const quality = [_]f32{ 0.99, 0.4, 0.8, 0.9 };
    const scores = [_]f32{2};
    const tracked = choose(table, &quality, &scores, (Options{}).multimask);
    try std.testing.expectEqual(@as(usize, 3), tracked[0].candidate);
    try std.testing.expectEqual(quality[3], tracked[0].iou);
    const single = choose(table, &quality, &scores, false);
    try std.testing.expectEqual(@as(usize, 0), single[0].candidate);
    const tied = choose(table, &.{ 0.99, 0.8, 0.8, 0.7 }, &scores, true);
    try std.testing.expectEqual(@as(usize, 1), tied[0].candidate);
    const multiplex = choose(.{ .slots = 2, .masks = 3, .width = 256 }, &.{ 0.9, 0.2, 0.3, 0.4, 0.5, 0.8 }, &.{ 2, -3 }, false);
    try std.testing.expectEqual(@as(usize, 0), multiplex[0].candidate);
    try std.testing.expectEqual(@as(usize, 2), multiplex[1].candidate);
    try std.testing.expectEqual(@as(f32, -3), multiplex[1].score);
}
