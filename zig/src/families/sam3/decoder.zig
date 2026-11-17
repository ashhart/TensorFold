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

    fn row(self: TokenTable, tensor: ops.Tensor, index: usize) ops.Tensor {
        return ops.view(tensor, index * self.width, self.width);
    }
};

const Choice = struct { candidate: usize = 0, score: f32 = -10, iou: f32 = 0 };

fn choose(table: TokenTable, quality: []const f32, scores: []const f32, multimask: bool) [16]Choice {
    var result: [16]Choice = @splat(.{});
    const first: usize = @intFromBool(multimask);
    const end = if (multimask or table.slots > 1) table.masks else 1;
    for (0..table.slots) |slot| {
        const row = quality[slot * table.masks ..][0..table.masks];
        var winner = first;
        for (first + 1..end) |candidate| {
            if (row[candidate] > row[winner]) winner = candidate;
        }
        result[slot] = .{ .candidate = winner, .score = scores[slot], .iou = row[winner] };
    }
    return result;
}

fn enabled(options: Options, slot: usize) bool {
    return !options.multiplex or options.active & (@as(u16, 1) << @intCast(slot)) != 0;
}

fn needsPixels(choices: []const Choice, options: Options) bool {
    for (choices, 0..) |choice, slot| if (enabled(options, slot) and !(choice.score <= 0)) return true;
    return false;
}

const Decoder = struct {
    network: Layer,
    owner: Layer,
    prompts: Layer,
    layout: TokenTable,
    options: Options,

    fn init(r: *ops.Runtime, requested: Options) !Decoder {
        var options = requested;
        options.multiplex = options.multiplex and r.config.version == .sam31;
        const layout = TokenTable{
            .slots = if (options.multiplex) r.config.bucket_size else 1,
            .masks = if (options.multiplex) r.config.mask_tokens else 4,
            .width = r.config.hidden,
        };
        const network = Layer{ .runtime = r, .prefix = r.config.decoderPrefix(!options.multiplex) };
        const tensor = try r.store.get(try network.named("mask_tokens.weight"));
        if (layout.slots == 0 or layout.slots > 16 or layout.masks == 0 or
            !tensor.is(tensor.dtype, &.{ layout.slots * layout.masks, layout.width })) return error.SamDecoderShape;
        return .{
            .network = network,
            .owner = .{ .runtime = r, .prefix = r.config.trackerPrefix() },
            .prompts = .{ .runtime = r, .prefix = r.config.promptPrefix() },
            .layout = layout,
            .options = options,
        };
    }

    fn learnedTokens(self: Decoder) !ops.Tensor {
        const r = self.network.runtime;
        const key = try r.name("decoder.queries.{s}.{d}", .{ self.network.prefix, self.options.active });
        if (r.constant(key)) |value| return value;
        var masks = try self.network.parameter("mask_tokens.weight");
        if (self.options.multiplex) {
            const present = try self.owner.parameter("output_valid_embed");
            const absent = try self.owner.parameter("output_invalid_embed");
            const offsets = try r.alloc(masks.count);
            const values = offsets.slice(f32, offsets.count);
            for (0..self.layout.slots) |slot| {
                const symbol = if (enabled(self.options, slot)) present else absent;
                for (0..self.layout.masks) |candidate| {
                    const row = values[(slot * self.layout.masks + candidate) * self.layout.width ..][0..self.layout.width];
                    for (row, 0..) |*value, channel| value.* = symbol.get(slot * self.layout.width + channel);
                }
            }
            masks = try r.add(masks, offsets);
        }
        const tokens = try r.concat(&.{
            try self.network.parameter("obj_score_token.weight"),
            try self.network.parameter("iou_token.weight"),
            masks,
        });
        return if (self.options.active == 1) r.remember(key, tokens) else tokens;
    }

    fn inputTokens(self: Decoder) !ops.Tensor {
        const r = self.network.runtime;
        const learned = try self.learnedTokens();
        if (self.options.sparse) |points| return r.concat(&.{ learned, points });
        if (self.options.multiplex) return learned;
        const padding = try self.prompts.parameter("not_a_point_embed.weight");
        return r.concat(&.{ learned, padding, padding });
    }

    fn imagePosition(self: Decoder) !ops.Tensor {
        const key = if (self.options.multiplex)
            try self.owner.named("image_pe_layer.positional_encoding_gaussian_matrix")
        else
            try self.prompts.named("pe_layer.positional_encoding_gaussian_matrix");
        return position.grid(self.network.runtime, key);
    }

    fn upsample(self: Decoder, pixels: ops.Tensor, features: [3]ops.Feature) !ops.Feature {
        const r = self.network.runtime;
        const side = r.config.featureSide();
        var value = ops.feature(pixels, side, side, self.layout.width);
        const stages = [_]struct { convolution: []const u8, skip: []const u8, norm: ?[]const u8, level: usize }{
            .{ .convolution = "output_upscaling.0", .skip = "conv_s1", .norm = "output_upscaling.1", .level = 1 },
            .{ .convolution = "output_upscaling.3", .skip = "conv_s0", .norm = null, .level = 0 },
        };
        for (stages) |stage| {
            value = try r.conv(try self.network.named(stage.convolution), value, 2, 0, true, false);
            const skip = try r.conv(try self.network.named(stage.skip), features[stage.level], 1, 0, false, false);
            value.tensor = try r.add(value.tensor, skip.tensor);
            if (stage.norm) |norm| value.tensor = try self.network.norm(norm, value.tensor, 1e-6);
            value.tensor = try r.gelu(value.tensor);
        }
        return value;
    }

    fn output(self: Decoder, tokens: ops.Tensor, image: ops.Tensor, features: [3]ops.Feature, choices: []const Choice) !Result {
        const r = self.network.runtime;
        const side = r.config.featureSide() * 4;
        const empty = try r.alloc(side * side);
        const zero = try r.alloc(self.layout.width);
        @memset(empty.slice(f32, empty.count), -1024);
        @memset(zero.slice(f32, zero.count), 0);
        var masks: [16]ops.Tensor = @splat(empty);
        var pointers: [16]ops.Tensor = @splat(zero);
        var result = Result{ .masks = undefined, .pointer = undefined, .count = choices.len, .scores = @splat(-10), .ious = @splat(0) };
        const pixels = if (needsPixels(choices, self.options)) try self.upsample(image, features) else null;
        const pointer_head = if (r.config.version == .sam31 and !self.options.multiplex) "interactive_obj_ptr_proj" else "obj_ptr_proj";
        for (choices, 0..) |choice, slot| {
            result.scores[slot] = choice.score;
            result.ious[slot] = choice.iou;
            if (!enabled(self.options, slot)) continue;
            const row = self.layout.row(tokens, 2 * self.layout.slots + slot * self.layout.masks + choice.candidate);
            if (!(choice.score <= 0)) {
                const classifier = try self.network.mlp(try r.name("output_hypernetworks_mlps.{d}", .{choice.candidate}), row, 3);
                masks[slot] = try r.matmul(pixels.?.tensor, classifier, side * side, 1, pixels.?.channels);
            }
            pointers[slot] = try self.owner.mlp(pointer_head, row, 3);
            if (choice.score <= 0) {
                pointers[slot] = if (r.config.version == .sam31)
                    try self.owner.linear("no_obj_ptr_linear", pointers[slot])
                else
                    try self.owner.parameter("no_obj_ptr");
            }
        }
        const mask_tensor = if (choices.len == 1) masks[0] else try r.transpose(try r.concat(masks[0..choices.len]), choices.len, side * side);
        result.masks = ops.feature(mask_tensor, side, side, choices.len);
        result.pointer = try r.concat(pointers[0..choices.len]);
        return result;
    }
};

fn attention(layer: Layer, name: []const u8, query: ops.Tensor, key: ops.Tensor, value: ops.Tensor) !ops.Tensor {
    return layer.runtime.splitMha(try layer.named(name), query, key, value, 8);
}

fn twoWay(root: Layer, tokens: ops.Tensor, pixels: ops.Tensor, grid: ops.Tensor) ![2]ops.Tensor {
    const r = root.runtime;
    const start = r.mark();
    var q = tokens;
    var image = pixels;
    for (0..2) |index| {
        const block = try root.indexed("layers", index);
        const positioned = if (index == 0) q else try r.add(q, tokens);
        const self_attended = try attention(block, "self_attn", positioned, positioned, q);
        q = try block.norm("norm1", if (index == 0) self_attended else try r.add(q, self_attended), 1e-5);
        const image_keys = try r.add(image, grid);
        q = try block.norm("norm2", try r.add(q, try attention(block, "cross_attn_token_to_image", try r.add(q, tokens), image_keys, image)), 1e-5);
        const mlp = try block.child("mlp");
        const hidden = try r.unary(try mlp.linear("lin1", q), 0);
        q = try block.norm("norm3", try r.add(q, try mlp.linear("lin2", hidden)), 1e-5);
        const image_update = try attention(block, "cross_attn_image_to_token", image_keys, try r.add(q, tokens), q);
        image = try block.norm("norm4", try r.add(image, image_update), 1e-5);
        r.trim(start, &.{ q, image });
    }
    const final = try attention(root, "final_attn_token_to_image", try r.add(q, tokens), try r.add(image, grid), image);
    q = try root.norm("norm_final_attn", try r.add(q, final), 1e-5);
    r.trim(start, &.{ q, image });
    return .{ q, image };
}

pub fn decode(r: *ops.Runtime, features: [3]ops.Feature, image: ops.Tensor, requested: Options) !Result {
    const side = r.config.featureSide();
    const width = r.config.hidden;
    if (image.count != side * side * width) return error.SamDecoderShape;
    for (features, 0..) |feature, level| {
        const extent = side * (@as(usize, 4) >> @intCast(level));
        if (feature.height != extent or feature.width != extent or feature.channels != width or
            feature.tensor.count != extent * extent * width) return error.SamDecoderShape;
    }
    const start = r.mark();
    errdefer r.trim(start, &.{});
    const decoder = try Decoder.init(r, requested);
    const tokens = try decoder.inputTokens();
    const grid = try decoder.imagePosition();
    const dense = decoder.options.dense orelse if (decoder.options.multiplex) null else try decoder.prompts.parameter("no_mask_embed.weight");
    const shape = [_]usize{ 1, side, side, width };
    try r.observe("decoder.image", image, &shape);
    if (requested.dense == null and !decoder.options.multiplex) if (dense) |value| try r.observe("prompt.no_mask", value, &.{ 1, width });
    try r.observe("decoder.position", grid, &shape);
    const pixels = if (dense) |value| try r.add(image, value) else image;
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const refined = try twoWay(try decoder.network.child("transformer"), tokens, pixels, grid);
    const count = decoder.layout.slots;
    const quality_logits = try decoder.network.mlp("iou_prediction_head", ops.view(refined[0], count * width, count * width), 3);
    const quality = if (r.config.version == .sam3) try r.unary(quality_logits, 1) else quality_logits;
    const confidence = try decoder.network.mlp("pred_obj_score_head", ops.view(refined[0], 0, count * width), 3);
    try r.g.endBatch();
    if (quality.count != count * decoder.layout.masks or confidence.count != count) return error.SamDecoderShape;
    try r.observe("decoder.quality", quality, &.{ count, decoder.layout.masks });
    const choices = choose(decoder.layout, quality.slice(f32, quality.count), confidence.slice(f32, confidence.count), requested.multimask and !decoder.options.multiplex);
    try r.g.beginBatch();
    const result = try decoder.output(refined[0], refined[1], features, choices[0..count]);
    try r.g.endBatch();
    r.trim(start, &.{ result.masks.tensor, result.pointer });
    return result;
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
