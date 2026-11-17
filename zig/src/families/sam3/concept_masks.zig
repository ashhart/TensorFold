const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;

pub fn pixels(r: *ops.Runtime, features: [3]ops.Feature, memory: ops.Tensor, prompt: ops.Tensor) !ops.Feature {
    const root = Layer{ .runtime = r, .prefix = "detector.segmentation_head" };
    const normalized = try root.norm("cross_attn_norm", memory, 1e-5);
    const update = try r.mha(try root.named("cross_attend_prompt"), normalized, prompt, prompt, 8, r.config.hidden, null);
    var value = ops.feature(try r.add(memory, update), features[2].height, features[2].width, r.config.hidden);
    const scratch = r.mark();
    const decoder = try root.child("pixel_decoder");
    for ([_]usize{ 2, 1, 0 }, 0..) |level, index| {
        if (index > 0) {
            const skip = features[level];
            value = try r.resize(value, skip.height, skip.width, false);
            value.tensor = try r.add(value.tensor, skip.tensor);
        }
        const convolution = try decoder.named(try r.name("conv_layers.{d}", .{index}));
        const normalizer = try decoder.named(try r.name("norms.{d}", .{index}));
        value = try r.conv(convolution, value, 1, 1, false, false);
        value = try r.groupNorm(normalizer, value, 8);
        value.tensor = try r.unary(value.tensor, 0);
        r.trim(scratch, &.{value.tensor});
    }
    return r.conv(try root.named("instance_seg_head"), value, 1, 0, false, false);
}

pub fn classify(r: *ops.Runtime, image: ops.Feature, queries: ops.Tensor) !ops.Feature {
    const root = Layer{ .runtime = r, .prefix = "detector.segmentation_head.mask_predictor" };
    if (queries.count % r.config.hidden != 0) return error.SamConceptQueryShape;
    const count = queries.count / r.config.hidden;
    const embeddings = try root.mlp("mask_embed", queries, 3);
    if (embeddings.count != count * image.channels) return error.SamConceptQueryShape;
    const matrix = try r.transpose(embeddings, count, image.channels);
    const output = try r.matmul(image.tensor, matrix, image.height * image.width, count, image.channels);
    return ops.feature(output, image.height, image.width, count);
}
