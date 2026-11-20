const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;

fn absolutePosition(root: Layer) !ops.Tensor {
    const r = root.runtime;
    const key = "image.absolute_position";
    if (r.constant(key)) |cached| return cached;
    const learned = try root.parameter("pos_embed");
    const width = r.config.vision_width;
    const tile = r.config.window;
    const side = r.config.featureSide();
    if (learned.count != (tile * tile + 1) * width) return error.SamImagePositionShape;
    const output = try r.alloc(side * side * width);
    for (0..side * side) |pixel| {
        const source = (1 + (pixel / side % tile) * tile + pixel % side % tile) * width;
        const row = output.slice(f32, output.count)[pixel * width ..][0..width];
        for (row, 0..) |*value, channel| value.* = learned.get(source + channel);
    }
    return r.remember(key, output);
}

fn block(layer: Layer, input: ops.Tensor, global: bool) !ops.Tensor {
    const r = layer.runtime;
    const width = r.config.vision_width;
    const normalized = try r.visionNorm(try layer.named("norm1"), input, !global);
    const projections = try r.visionLinear(try layer.named("attn.qkv"), normalized, width, false);
    const qkv = try r.visionQkv(try layer.named("attn.freqs_cis"), projections, global);
    const attended = try r.visionAttention(qkv[0], qkv[1], qkv[2], global);
    const residual = try r.visionResidual(try layer.named("attn.proj"), attended, width, input, !global);
    const hidden = try r.visionLinear(try layer.named("mlp.fc1"), try r.visionNorm(try layer.named("norm2"), residual, false), width, true);
    return r.visionResidual(try layer.named("mlp.fc2"), hidden, r.config.vision_mlp, residual, false);
}

pub fn encode(r: *ops.Runtime, input: ops.Feature) !ops.Feature {
    const root = Layer{ .runtime = r, .prefix = "detector.backbone.vision_backbone.trunk" };
    const mark = r.mark();
    errdefer r.trim(mark, &.{});
    const positional = try absolutePosition(root);
    for (0..r.config.vision_layers) |index| {
        const layer = try root.indexed("blocks", index);
        _ = try r.visionRotaryTable(try layer.named("attn.freqs_cis"), (index + 1) % 8 == 0);
    }
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const patches = try r.conv(try root.named("patch_embed.proj"), input, r.config.patch_size, 0, false, false);
    const side = r.config.featureSide();
    const width = r.config.vision_width;
    if (patches.height != side or patches.width != side or patches.channels != width) return error.SamImagePatchShape;
    try r.observe("trunk.patch_embed", patches.tensor, &.{ 1, side, side, width });
    var value = try root.norm("ln_pre", try r.add(patches.tensor, positional), 1e-5);
    try r.observe("trunk.ln_pre", value, &.{ 1, side, side, width });
    r.trim(mark, &.{value});
    for (0..r.config.vision_layers) |index| {
        value = try block(try root.indexed("blocks", index), value, (index + 1) % 8 == 0);
        try r.observe(try r.name("trunk.blocks.{d}", .{index}), value, &.{ 1, side, side, width });
        r.trim(mark, &.{value});
        if (!r.vision_ready and (index + 1) % 4 == 0) {
            try r.g.endBatch();
            try r.g.beginBatch();
        }
    }
    try r.g.endBatch();
    r.vision_ready = true;
    return ops.feature(value, side, side, width);
}

pub fn pyramid(r: *ops.Runtime, encoded: ops.Feature, kind: []const u8) ![3]ops.Feature {
    const root = Layer{ .runtime = r, .prefix = try r.name("detector.backbone.vision_backbone.{s}", .{kind}) };
    var output: [3]ops.Feature = undefined;
    const mark = r.mark();
    errdefer r.trim(mark, &.{});
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    for (&output, 0..) |*level, index| {
        const branch = try root.child(try r.name("{d}", .{index}));
        level.* = encoded;
        if (index == 0) {
            level.* = try r.conv(try branch.named("dconv_2x2_0"), level.*, 2, 0, true, false);
            level.tensor = try r.gelu(level.tensor);
            level.* = try r.conv(try branch.named("dconv_2x2_1"), level.*, 2, 0, true, false);
        } else if (index == 1) level.* = try r.conv(try branch.named("dconv_2x2"), level.*, 2, 0, true, false);
        level.* = try r.conv(try branch.named("conv_1x1"), level.*, 1, 0, false, false);
        level.* = try r.conv(try branch.named("conv_3x3"), level.*, 1, 1, false, false);
        var retained: [4]ops.Tensor = undefined;
        retained[0] = encoded.tensor;
        for (output[0 .. index + 1], 1..) |done, slot| retained[slot] = done.tensor;
        r.trim(mark, retained[0 .. index + 2]);
    }
    try r.g.endBatch();
    return output;
}
