// SAM architecture adapted from facebookresearch/sam3 under LICENSES/SAM.txt.
const std = @import("std");
const ops = @import("ops.zig");
const vision = @import("vision.zig");
const Runtime = ops.Runtime;
const Tensor = ops.Tensor;
const Feature = ops.Feature;
const feature = ops.feature;
const view = ops.view;

pub const Result = struct {
    masks: Feature,
    input_masks: ?Feature = null,
    pointer: Tensor,
    count: usize,
    scores: [16]f32,
    ious: [16]f32,

    pub fn retained(self: Result, storage: *[3]Tensor) []const Tensor {
        storage[0..2].* = .{ self.masks.tensor, self.pointer };
        if (self.input_masks) |input| {
            storage[2] = input.tensor;
            return storage;
        }
        return storage[0..2];
    }
};

pub const Options = struct {
    multiplex: bool = false,
    active: u16 = 1,
    multimask: bool = false,
    dense: ?Tensor = null,
    sparse: ?Tensor = null,
};

fn twoWay(r: *Runtime, key: []const u8, image: Tensor, pos: Tensor, tokens: Tensor) ![2]Tensor {
    const start = r.mark();
    var q = tokens;
    var k = image;
    for (0..2) |i| {
        const layer = try r.name("{s}.layers.{d}", .{ key, i });
        const qs = if (i == 0) q else try r.add(q, tokens);
        const self = try r.splitMha(try r.name("{s}.self_attn", .{layer}), qs, qs, q, 8);
        q = try r.norm(try r.name("{s}.norm1", .{layer}), if (i == 0) self else try r.add(q, self), 256, 1e-5);
        q = try r.norm(try r.name("{s}.norm2", .{layer}), try r.add(q, try r.splitMha(try r.name("{s}.cross_attn_token_to_image", .{layer}), try r.add(q, tokens), try r.add(k, pos), k, 8)), 256, 1e-5);
        const up = try r.unary(try r.linear(try r.name("{s}.mlp.lin1", .{layer}), q, 256), 0);
        q = try r.norm(try r.name("{s}.norm3", .{layer}), try r.add(q, try r.linear(try r.name("{s}.mlp.lin2", .{layer}), up, 2048)), 256, 1e-5);
        k = try r.norm(try r.name("{s}.norm4", .{layer}), try r.add(k, try r.splitMha(try r.name("{s}.cross_attn_image_to_token", .{layer}), try r.add(k, pos), try r.add(q, tokens), q, 8)), 256, 1e-5);
        r.trim(start, &.{ q, k });
    }
    q = try r.norm(try r.name("{s}.norm_final_attn", .{key}), try r.add(q, try r.splitMha(try r.name("{s}.final_attn_token_to_image", .{key}), try r.add(q, tokens), try r.add(k, pos), k, 8)), 256, 1e-5);
    r.trim(start, &.{ q, k });
    return .{ q, k };
}

pub fn decode(r: *Runtime, features: [3]Feature, image: Tensor, options: Options) !Result {
    const start = r.mark();
    const multiplex = options.multiplex and r.config.version == .sam31;
    const count: usize = if (multiplex) 16 else 1;
    const per: usize = if (multiplex) 3 else 4;
    const root = r.config.trackerPrefix();
    const key = r.config.decoderPrefix(!multiplex);
    const token_key = try r.name("{s}.tokens.{d}", .{ key, if (multiplex) options.active else @as(u16, 1) });
    const base_tokens = r.constant(token_key) orelse tokens: {
        const scores = try r.weight(try r.name("{s}.obj_score_token.weight", .{key}));
        const ious = try r.weight(try r.name("{s}.iou_token.weight", .{key}));
        var masks = try r.weight(try r.name("{s}.mask_tokens.weight", .{key}));
        if (multiplex) {
            const extra = try r.alloc(masks.count);
            const valid = try r.weight(try r.name("{s}.output_valid_embed", .{root}));
            const invalid = try r.weight(try r.name("{s}.output_invalid_embed", .{root}));
            for (0..16) |slot| for (0..3) |m| for (0..256) |c| {
                const active = options.active & (@as(u16, 1) << @intCast(slot)) != 0;
                extra.slice(f32, extra.count)[(slot * 3 + m) * 256 + c] =
                    if (active) valid.get(slot * 256 + c) else invalid.get(slot * 256 + c);
            };
            masks = try r.add(masks, extra);
        }
        const combined = try r.concat(&.{ scores, ious, masks });
        break :tokens if (!multiplex or options.active == 1) try r.remember(token_key, combined) else combined;
    };
    var tokens = base_tokens;
    if (options.sparse) |sparse| {
        tokens = try r.concat(&.{ tokens, sparse });
    } else if (!multiplex) {
        const nullpoint = try r.weight(try r.name("{s}.not_a_point_embed.weight", .{r.config.promptPrefix()}));
        tokens = try r.concat(&.{ tokens, nullpoint, nullpoint });
    }
    const pe = try vision.imagePe(r, if (multiplex)
        try r.name("{s}.image_pe_layer.positional_encoding_gaussian_matrix", .{root})
    else
        try r.name("{s}.pe_layer.positional_encoding_gaussian_matrix", .{r.config.promptPrefix()}));
    const dense = options.dense orelse if (!multiplex)
        try r.weight(try r.name("{s}.no_mask_embed.weight", .{r.config.promptPrefix()}))
    else
        null;
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const result = try twoWay(r, try r.name("{s}.transformer", .{key}), if (dense) |d| try r.add(image, d) else image, pe, tokens);
    var up = try r.conv(try r.name("{s}.output_upscaling.0", .{key}), feature(result[1], 72, 72, 256), 2, 0, true, false);
    const high1 = try r.conv(try r.name("{s}.conv_s1", .{key}), features[1], 1, 0, false, false);
    up.tensor = try r.gelu(try r.norm(try r.name("{s}.output_upscaling.1", .{key}), try r.add(up.tensor, high1.tensor), 64, 1e-6));
    up = try r.conv(try r.name("{s}.output_upscaling.3", .{key}), up, 2, 0, true, false);
    const high0 = try r.conv(try r.name("{s}.conv_s0", .{key}), features[0], 1, 0, false, false);
    up.tensor = try r.gelu(try r.add(up.tensor, high0.tensor));
    const iou = try r.mlp(try r.name("{s}.iou_prediction_head", .{key}), view(result[0], count * 256, count * 256), 256, 3);
    const object_score = try r.mlp(try r.name("{s}.pred_obj_score_head", .{key}), view(result[0], 0, count * 256), 256, 3);
    try r.g.endBatch();
    var scores: [16]f32 = @splat(-10);
    var ious: [16]f32 = @splat(0);
    var best: [16]usize = @splat(0);
    for (0..count) |slot| {
        scores[slot] = object_score.get(slot);
        const first: usize = if (!multiplex and options.multimask) 1 else 0;
        const end: usize = if (multiplex or options.multimask) per else 1;
        best[slot] = first;
        for (first + 1..end) |i| if (iou.get(slot * per + i) > iou.get(slot * per + best[slot])) {
            best[slot] = i;
        };
        ious[slot] = iou.get(slot * per + best[slot]);
    }
    const empty_mask = try r.alloc(288 * 288);
    @memset(empty_mask.slice(f32, empty_mask.count), -1024);
    const empty_pointer = try r.alloc(256);
    @memset(empty_pointer.slice(f32, 256), 0);
    var masks: [16]Tensor = @splat(empty_mask);
    var pointers: [16]Tensor = @splat(empty_pointer);
    try r.g.beginBatch();
    for (0..count) |slot| {
        if (multiplex and options.active & (@as(u16, 1) << @intCast(slot)) == 0) continue;
        const mask_token = view(result[0], (count * 2 + slot * per + best[slot]) * 256, 256);
        const hyper = try r.mlp(try r.name("{s}.output_hypernetworks_mlps.{d}", .{ key, best[slot] }), mask_token, 256, 3);
        masks[slot] = try r.matmul(up.tensor, hyper, 288 * 288, 1, 32);
        const pointer_token = if (multiplex)
            view(result[0], (count * 2 + slot * per) * 256, 256)
        else
            mask_token;
        const projection = if (r.config.version == .sam31 and !multiplex) "interactive_obj_ptr_proj" else "obj_ptr_proj";
        var pointer = try r.mlp(try r.name("{s}.{s}", .{ root, projection }), pointer_token, 256, 3);
        if (scores[slot] <= 0) {
            pointer = if (r.config.version == .sam31)
                try r.linear(try r.name("{s}.no_obj_ptr_linear", .{root}), pointer, 256)
            else
                try r.weight(try r.name("{s}.no_obj_ptr", .{root}));
            masks[slot] = empty_mask;
        }
        pointers[slot] = pointer;
    }
    const combined = if (count == 1) masks[0] else try r.transpose(try r.concat(masks[0..count]), count, 288 * 288);
    const pointer = try r.concat(pointers[0..count]);
    try r.g.endBatch();
    r.trim(start, &.{ combined, pointer });
    return .{ .masks = feature(combined, 288, 288, count), .pointer = pointer, .count = count, .scores = scores, .ious = ious };
}
