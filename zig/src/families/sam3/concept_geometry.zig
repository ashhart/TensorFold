const std = @import("std");
const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;

fn attentionUpdate(block: Layer, name: []const u8, query: ops.Tensor, key: ops.Tensor, value: ops.Tensor) !ops.Tensor {
    return block.runtime.mha(try block.named(name), query, key, value, 8, block.runtime.config.hidden, null);
}

pub fn classToken(r: *ops.Runtime, image: ops.Tensor, position: ops.Tensor) !ops.Tensor {
    const root = Layer{ .runtime = r, .prefix = "detector.geometry_encoder" };
    const start = r.mark();
    const learned = try root.parameter("cls_embed.weight");
    var value = try root.norm("norm", try root.linear("final_proj", learned), 1e-5);
    const image_keys = try r.add(image, position);
    for (0..3) |index| {
        const block = try root.indexed("encode", index);
        const norm1 = try block.norm("norm1", value, 1e-5);
        value = try r.add(value, try attentionUpdate(block, "self_attn", norm1, norm1, norm1));
        const norm2 = try block.norm("norm2", value, 1e-5);
        value = try r.add(value, try attentionUpdate(block, "cross_attn_image", norm2, image_keys, image));
        const norm3 = try block.norm("norm3", value, 1e-5);
        value = try r.add(value, try block.feedForward(norm3, false));
        r.trim(start, &.{ value, image_keys });
    }
    value = try root.norm("encode_norm", value, 1e-5);
    r.trim(start, &.{value});
    return value;
}

pub fn boxEmbedding(r: *ops.Runtime, boxes: ops.Tensor) !ops.Tensor {
    if (boxes.count % 4 != 0) return error.SamConceptBoxShape;
    const bands = r.config.hidden / 2;
    const output = try r.alloc(boxes.count * bands);
    const values = output.slice(f32, output.count);
    const permutation = [4]usize{ 1, 0, 2, 3 };
    for (0..boxes.count) |coordinate| {
        const source = coordinate / 4 * 4 + permutation[coordinate % 4];
        const radians = boxes.get(source) * (2 * std.math.pi);
        const row = values[coordinate * bands ..][0..bands];
        for (0..bands / 2) |band| {
            const exponent = @as(f32, @floatFromInt(2 * band)) / @as(f32, @floatFromInt(bands));
            const phase = radians / std.math.pow(f32, 10000, exponent);
            row[2 * band] = @sin(phase);
            row[2 * band + 1] = @cos(phase);
        }
    }
    return output;
}

fn gridDistance(cell: usize, side: usize, edge: f32) f32 {
    const coordinate = @as(f32, @floatFromInt(cell)) / @as(f32, @floatFromInt(side));
    const delta = (coordinate - edge) * 8;
    return std.math.sign(delta) * @log2(@abs(delta) + 1) / 3;
}

fn axisFeatures(r: *ops.Runtime, boxes: ops.Tensor, axis: usize, side: usize) !ops.Tensor {
    const count = boxes.count / 4;
    const features = try r.alloc(count * side * 2);
    const data = features.slice(f32, features.count);
    for (0..count) |box| {
        const centre = boxes.get(4 * box + axis);
        const radius = 0.5 * boxes.get(4 * box + axis + 2);
        for (0..side) |cell| {
            const offset = 2 * (box * side + cell);
            data[offset] = gridDistance(cell, side, centre - radius);
            data[offset + 1] = gridDistance(cell, side, centre + radius);
        }
    }
    return features;
}

pub fn boxBias(root: Layer, boxes: ops.Tensor) !ops.Tensor {
    if (boxes.count % 4 != 0) return error.SamConceptBoxShape;
    const r = root.runtime;
    const side = r.config.featureSide();
    const count = boxes.count / 4;
    const x = try root.mlp("boxRPB_embed_x", try axisFeatures(r, boxes, 0, side), 2);
    const y = try root.mlp("boxRPB_embed_y", try axisFeatures(r, boxes, 1, side), 2);
    const heads = 8;
    const pixels = side * side;
    const output = try r.alloc(heads * (count + 1) * pixels);
    const data = output.slice(f32, output.count);
    for (0..heads) |head| {
        const rows = data[head * (count + 1) * pixels ..][0 .. (count + 1) * pixels];
        @memset(rows[0..pixels], 0);
        for (0..count) |box| {
            const row = rows[(box + 1) * pixels ..][0..pixels];
            for (row, 0..) |*value, pixel| {
                value.* = x.get((box * side + pixel % side) * heads + head) +
                    y.get((box * side + pixel / side) * heads + head);
            }
        }
    }
    return output;
}

pub fn refine(r: *ops.Runtime, boxes: ops.Tensor, offsets: ops.Tensor) !ops.Tensor {
    if (boxes.count != offsets.count) return error.SamConceptBoxShape;
    const output = try r.alloc(boxes.count);
    for (output.slice(f32, output.count), 0..) |*value, coordinate| {
        const p = boxes.get(coordinate);
        const prior = @log(p / (1 - p + 1e-6) + 1e-6);
        const logit = prior + offsets.get(coordinate);
        value.* = 1 / (1 + @exp(-logit));
    }
    return output;
}

test "concept box coordinates use a half-open feature grid" {
    try std.testing.expectEqual(@as(f32, 0), gridDistance(36, 72, 0.5));
    const expected: f32 = @floatCast(-@log(@as(f64, 1) + 8.0 / 72.0) / @log(@as(f64, 8)));
    try std.testing.expectApproxEqAbs(expected, gridDistance(71, 72, 1), 1e-7);
    try std.testing.expect(gridDistance(71, 72, 1) < 0);
}
