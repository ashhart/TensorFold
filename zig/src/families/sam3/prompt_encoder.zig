const std = @import("std");
const ops = @import("ops.zig");
const prompts = @import("prompts.zig");
const Fourier = @import("position.zig").Fourier;
const Layer = @import("layers.zig").Layer;

pub const Encoded = struct { sparse: ?ops.Tensor = null, dense: ?ops.Tensor = null };

const SparseWriter = struct {
    layer: Layer,
    transform: prompts.Transform,
    mapping: Fourier,
    output: ops.Tensor,
    cursor: usize = 0,

    fn append(self: *SparseWriter, coordinate: ?[2]f32, kind: usize) !void {
        const width = self.mapping.width;
        const row = self.output.slice(f32, self.output.count)[self.cursor * width ..][0..width];
        if (coordinate) |xy| {
            const extent: f32 = @floatFromInt(self.layer.runtime.config.image_size);
            self.mapping.write(.{ (xy[0] + 0.5) / extent, (xy[1] + 0.5) / extent }, row);
        } else @memset(row, 0);
        const parameter = if (coordinate == null) "not_a_point_embed.weight" else try self.layer.runtime.name("point_embeddings.{d}.weight", .{kind});
        const learned = try self.layer.parameter(parameter);
        if (learned.count != width) return error.SamPromptEmbeddingShape;
        for (row, 0..) |*component, c| component.* += learned.get(c);
        self.cursor += 1;
    }
};

fn sparse(r: *ops.Runtime, prompt: prompts.Prompt, size: prompts.Size) !?ops.Tensor {
    const padded = prompt.points.len != 0 and prompt.boxes.len == 0;
    const corners = try std.math.mul(usize, prompt.boxes.len, 2);
    const count = try std.math.add(usize, try std.math.add(usize, corners, prompt.points.len), @intFromBool(padded));
    if (count == 0) return null;
    const layer = Layer{ .runtime = r, .prefix = r.config.promptPrefix() };
    const mapping = try Fourier.init(r, try layer.named("pe_layer.positional_encoding_gaussian_matrix"));
    var writer = SparseWriter{
        .layer = layer,
        .transform = .{ .source = size },
        .mapping = mapping,
        .output = try r.alloc(try std.math.mul(usize, count, mapping.width)),
    };
    for (prompt.points) |point| try writer.append(try writer.transform.point(point), @intFromBool(point.positive));
    for (prompt.boxes) |box| {
        if (!box.positive) return error.NegativeBoxNeedsConceptPrompt;
        const rectangle = try writer.transform.box(box);
        try writer.append(rectangle[0..2].*, 2);
        try writer.append(rectangle[2..4].*, 3);
    }
    if (padded) try writer.append(null, 0);
    return writer.output;
}

fn dense(r: *ops.Runtime, mask: prompts.Mask) !ops.Tensor {
    var image = ops.feature(try r.upload(mask.values), mask.size.height, mask.size.width, 1);
    const tracker = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
    if (mask.format == .probabilities) {
        image = try r.resize(image, r.config.image_size, r.config.image_size, true);
        image = try r.conv(try tracker.named(if (r.config.version == .sam31) "interactive_mask_downsample" else "mask_downsample"), image, 4, 0, false, false);
    }
    const side = r.config.featureSide() * 4;
    image = try r.resize(image, side, side, true);
    const network = Layer{ .runtime = r, .prefix = try r.name("{s}.mask_downscaling", .{r.config.promptPrefix()}) };
    const stages = [_]struct { convolution: []const u8, norm: ?[]const u8 }{
        .{ .convolution = "0", .norm = "1" }, .{ .convolution = "3", .norm = "4" }, .{ .convolution = "6", .norm = null },
    };
    for (stages) |stage| {
        image = try r.conv(try network.named(stage.convolution), image, if (stage.norm != null) 2 else 1, 0, false, false);
        if (stage.norm) |normalizer| image.tensor = try r.gelu(try network.norm(normalizer, image.tensor, 1e-6));
    }
    return image.tensor;
}

pub fn encode(r: *ops.Runtime, prompt: prompts.Prompt, size: prompts.Size) !Encoded {
    try prompt.validate(size);
    const mark = r.mark();
    errdefer r.trim(mark, &.{});
    const output = Encoded{ .sparse = try sparse(r, prompt, size), .dense = if (prompt.mask) |mask| try dense(r, mask) else null };
    if (output.sparse) |tensor| try r.observe("prompt.sparse", tensor, &.{ 1, tensor.count / r.config.hidden, r.config.hidden });
    if (output.dense) |tensor| try r.observe("prompt.dense", tensor, &.{ 1, r.config.featureSide(), r.config.featureSide(), r.config.hidden });
    var retained: [2]ops.Tensor = undefined;
    var count: usize = 0;
    for ([_]?ops.Tensor{ output.sparse, output.dense }) |maybe| if (maybe) |tensor| {
        retained[count] = tensor;
        count += 1;
    };
    r.trim(mark, retained[0..count]);
    return output;
}
