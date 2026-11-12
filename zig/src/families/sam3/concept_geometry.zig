const std = @import("std");
const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;

pub fn classToken(r: *ops.Runtime, image: ops.Tensor, position: ops.Tensor) !ops.Tensor {
    const encoder = Layer{ .runtime = r, .prefix = "detector.geometry_encoder" };
    const width = r.config.hidden;
    const mark = r.mark();
    var token = try encoder.norm("norm", try encoder.linear("final_proj", try encoder.parameter("cls_embed.weight")), 1e-5);
    const image_keys = try r.add(image, position);
    for (0..3) |index| {
        const layer = try encoder.indexed("encode", index);
        const normalized = try layer.norm("norm1", token, 1e-5);
        const self_attention = try r.mha(try layer.named("self_attn"), normalized, normalized, normalized, 8, width, null);
        token = try r.add(token, self_attention);
        const from_image = try r.mha(try layer.named("cross_attn_image"), try layer.norm("norm2", token, 1e-5), image_keys, image, 8, width, null);
        token = try r.add(token, from_image);
        token = try r.add(token, try layer.feedForward(try layer.norm("norm3", token, 1e-5), false));
        r.trim(mark, &.{ token, image_keys });
    }
    token = try encoder.norm("encode_norm", token, 1e-5);
    r.trim(mark, &.{token});
    return token;
}

pub fn boxEmbedding(r: *ops.Runtime, boxes: ops.Tensor) !ops.Tensor {
    if (boxes.count % 4 != 0) return error.SamConceptBoxShape;
    const bands = r.config.hidden / 2;
    const output = try r.alloc(boxes.count * bands);
    const axes = [_]usize{ 1, 0, 2, 3 };
    for (0..boxes.count / 4) |box| {
        for (axes, 0..) |axis, output_axis| {
            const coordinate = boxes.get(box * 4 + axis) * (2 * std.math.pi);
            for (0..bands) |frequency| {
                const exponent = @as(f32, @floatFromInt(frequency / 2 * 2)) / @as(f32, @floatFromInt(bands));
                const phase = coordinate / std.math.pow(f32, 10000, exponent);
                output.slice(f32, output.count)[(box * 4 + output_axis) * bands + frequency] = if (frequency % 2 == 0) @sin(phase) else @cos(phase);
            }
        }
    }
    return output;
}

fn gridDistance(cell: usize, side: usize, edge: f32) f32 {
    const coordinate = @as(f32, @floatFromInt(cell)) / @as(f32, @floatFromInt(side));
    const distance = (coordinate - edge) * 8;
    return std.math.sign(distance) * @log2(@abs(distance) + 1) / 3;
}

fn edgeDistances(r: *ops.Runtime, boxes: ops.Tensor, axis: usize, side: usize) !ops.Tensor {
    const output = try r.alloc(boxes.count / 4 * side * 2);
    for (0..boxes.count / 4) |box| {
        const centre = boxes.get(box * 4 + axis);
        const half = boxes.get(box * 4 + axis + 2) * 0.5;
        for (0..side) |cell| {
            for ([_]f32{ centre - half, centre + half }, 0..) |edge, index| {
                output.slice(f32, output.count)[(box * side + cell) * 2 + index] = gridDistance(cell, side, edge);
            }
        }
    }
    return output;
}

pub fn boxBias(root: Layer, boxes: ops.Tensor) !ops.Tensor {
    const r = root.runtime;
    const side = r.config.featureSide();
    const count = boxes.count / 4;
    const heads = 8;
    const x = try root.mlp("boxRPB_embed_x", try edgeDistances(r, boxes, 0, side), 2);
    const y = try root.mlp("boxRPB_embed_y", try edgeDistances(r, boxes, 1, side), 2);
    const output = try r.alloc(heads * (count + 1) * side * side);
    const values = output.slice(f32, output.count);
    @memset(values, 0);
    for (0..heads) |head| for (0..count) |box| {
        for (0..side * side) |pixel| {
            values[(head * (count + 1) + box + 1) * side * side + pixel] = x.get((box * side + pixel % side) * heads + head) + y.get((box * side + pixel / side) * heads + head);
        }
    };
    return output;
}

pub fn refine(r: *ops.Runtime, boxes: ops.Tensor, offsets: ops.Tensor) !ops.Tensor {
    if (boxes.count != offsets.count) return error.SamConceptBoxShape;
    const output = try r.alloc(boxes.count);
    for (output.slice(f32, output.count), 0..) |*value, index| {
        const p = boxes.get(index);
        const logit = @log(p / (1 - p + 1e-6) + 1e-6);
        value.* = 1 / (1 + @exp(-(logit + offsets.get(index))));
    }
    return output;
}

test "concept box coordinates use a half-open feature grid" {
    try std.testing.expectEqual(@as(f32, 0), gridDistance(36, 72, 0.5));
    const expected: f32 = @floatCast(-@log(@as(f64, 1) + 8.0 / 72.0) / @log(@as(f64, 8)));
    try std.testing.expectApproxEqAbs(expected, gridDistance(71, 72, 1), 1e-7);
    try std.testing.expect(gridDistance(71, 72, 1) < 0);
}
