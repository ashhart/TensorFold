const std = @import("std");
const ops = @import("ops.zig");
const prompts = @import("prompts.zig");
const position = @import("position.zig");
const Layer = @import("layers.zig").Layer;

pub const Encoded = struct { sparse: ?ops.Tensor = null, dense: ?ops.Tensor = null };

const Label = enum { negative, positive, upper_left, lower_right, padding };
const Coordinate = struct { xy: [2]f32 = .{ 0, 0 }, label: Label };

fn coordinates(r: *ops.Runtime, prompt: prompts.Prompt, size: prompts.Size) ![]Coordinate {
    const pad = prompt.points.len > 0 and prompt.boxes.len == 0;
    const corners = try std.math.mul(usize, 2, prompt.boxes.len);
    const count = try std.math.add(usize, corners, try std.math.add(usize, prompt.points.len, @intFromBool(pad)));
    const records = try r.a.alloc(Coordinate, count);
    const transform = prompts.Transform{ .source = size, .side = r.config.image_size };
    for (prompt.points, records[0..prompt.points.len]) |point, *record| {
        record.* = .{ .xy = try transform.point(point), .label = if (point.positive) .positive else .negative };
    }
    for (prompt.boxes, 0..) |box, index| {
        if (!box.positive) return error.NegativeBoxNeedsConceptPrompt;
        const xyxy = try transform.box(box);
        const offset = prompt.points.len + index * 2;
        records[offset] = .{ .xy = .{ xyxy[0], xyxy[1] }, .label = .upper_left };
        records[offset + 1] = .{ .xy = .{ xyxy[2], xyxy[3] }, .label = .lower_right };
    }
    if (pad) records[count - 1] = .{ .label = .padding };
    return records;
}

fn coordinateEmbeddings(root: Layer, records: []const Coordinate) !?ops.Tensor {
    if (records.len == 0) return null;
    const r = root.runtime;
    const basis = try position.Fourier.init(r, try root.named("pe_layer.positional_encoding_gaussian_matrix"));
    const embedding = try r.alloc(try std.math.mul(usize, records.len, basis.width));
    const data = embedding.slice(f32, embedding.count);
    const extent: f32 = @floatFromInt(r.config.image_size);
    for (records, 0..) |record, row| {
        const target = data[row * basis.width ..][0..basis.width];
        const name = switch (record.label) {
            .padding => "not_a_point_embed.weight",
            else => try r.name("point_embeddings.{d}.weight", .{@backingInt(record.label)}),
        };
        const learned = try root.parameter(name);
        if (learned.count != target.len) return error.SamPromptEmbeddingShape;
        if (record.label == .padding) {
            @memset(target, 0);
        } else {
            basis.write(.{ (record.xy[0] + 0.5) / extent, (record.xy[1] + 0.5) / extent }, target);
        }
        for (target, 0..) |*value, channel| value.* += learned.get(channel);
    }
    return embedding;
}

fn maskEmbedding(root: Layer, mask: prompts.Mask) !ops.Tensor {
    const r = root.runtime;
    var image = ops.feature(try r.upload(mask.values), mask.size.height, mask.size.width, 1);
    if (mask.format == .probabilities) {
        const owner = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
        const projection = if (r.config.version == .sam31) "interactive_mask_downsample" else "mask_downsample";
        image = try r.resize(image, r.config.image_size, r.config.image_size, true);
        image = try r.conv(try owner.named(projection), image, 4, 0, false, false);
    }
    image = try r.resize(image, 4 * r.config.featureSide(), 4 * r.config.featureSide(), true);
    const network = try root.child("mask_downscaling");
    for (0..3) |level| {
        const stride: usize = if (level == 2) 1 else 2;
        image = try r.conv(try network.named(try r.name("{d}", .{3 * level})), image, stride, 0, false, false);
        if (level < 2) {
            image.tensor = try network.norm(try r.name("{d}", .{3 * level + 1}), image.tensor, 1e-6);
            image.tensor = try r.gelu(image.tensor);
        }
    }
    return image.tensor;
}

pub fn encode(r: *ops.Runtime, prompt: prompts.Prompt, size: prompts.Size) !Encoded {
    try prompt.validate(size);
    const start = r.mark();
    errdefer r.trim(start, &.{});
    const root = Layer{ .runtime = r, .prefix = r.config.promptPrefix() };
    const records = try coordinates(r, prompt, size);
    var result: Encoded = .{};
    result.sparse = try coordinateEmbeddings(root, records);
    if (prompt.mask) |mask| result.dense = try maskEmbedding(root, mask);
    var keep: [2]ops.Tensor = undefined;
    var used: usize = 0;
    const side = r.config.featureSide();
    if (result.sparse) |value| {
        try r.observe("prompt.sparse", value, &.{ 1, records.len, r.config.hidden });
        keep[used] = value;
        used += 1;
    }
    if (result.dense) |value| {
        try r.observe("prompt.dense", value, &.{ 1, side, side, r.config.hidden });
        keep[used] = value;
        used += 1;
    }
    r.trim(start, keep[0..used]);
    return result;
}
