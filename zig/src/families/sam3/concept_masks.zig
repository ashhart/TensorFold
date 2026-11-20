const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;

pub fn pixels(r: *ops.Runtime, features: [3]ops.Feature, memory: ops.Tensor, prompt: ops.Tensor) !ops.Feature {
    const root = Layer{ .runtime = r, .prefix = "detector.segmentation_head" };
    const queried = try root.norm("cross_attn_norm", memory, 1e-5);
    const attended = try r.mha(try root.named("cross_attend_prompt"), queried, prompt, prompt, 8, r.config.hidden, null);
    var image = ops.feature(try r.add(memory, attended), features[2].height, features[2].width, r.config.hidden);
    const mark = r.mark();
    for (0..3) |index| {
        if (index != 0) {
            const skip = features[2 - index];
            image = try r.resize(image, skip.height, skip.width, false);
            image.tensor = try r.add(image.tensor, skip.tensor);
        }
        image = try r.conv(try root.named(try r.name("pixel_decoder.conv_layers.{d}", .{index})), image, 1, 1, false, false);
        image = try r.groupNorm(try root.named(try r.name("pixel_decoder.norms.{d}", .{index})), image, 8);
        image.tensor = try r.unary(image.tensor, 0);
        r.trim(mark, &.{image.tensor});
    }
    image = try r.conv(try root.named("instance_seg_head"), image, 1, 0, false, false);
    return image;
}

pub fn classify(r: *ops.Runtime, image: ops.Feature, queries: ops.Tensor) !ops.Feature {
    const root = Layer{ .runtime = r, .prefix = "detector.segmentation_head.mask_predictor" };
    const classifiers = try root.mlp("mask_embed", queries, 3);
    const masks = try r.matmul(image.tensor, try r.transpose(classifiers, queries.count / r.config.hidden, image.channels), image.height * image.width, queries.count / r.config.hidden, image.channels);
    return ops.feature(masks, image.height, image.width, queries.count / r.config.hidden);
}
