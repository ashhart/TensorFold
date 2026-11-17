const ops = @import("ops.zig");
const Layer = @import("layers.zig").Layer;

const Trunk = struct {
    root: Layer,
    side: usize,
    width: usize,

    fn position(self: Trunk) !ops.Tensor {
        const r = self.root.runtime;
        if (r.constant("image.absolute_position")) |cached| return cached;
        const table = try self.root.parameter("pos_embed");
        const window = r.config.window;
        if (table.count != (1 + window * window) * self.width) return error.SamImagePositionShape;
        const output = try r.alloc(self.side * self.side * self.width);
        const data = output.slice(f32, output.count);
        for (0..self.side) |y| for (0..self.side) |x| {
            const offset = (1 + (y % window) * window + x % window) * self.width;
            const row = data[(y * self.side + x) * self.width ..][0..self.width];
            for (row, 0..) |*value, channel| value.* = table.get(offset + channel);
        };
        return r.remember("image.absolute_position", output);
    }

    fn global(self: Trunk, index: usize) bool {
        return (index + 1) % (self.root.runtime.config.vision_layers / 4) == 0;
    }

    fn layer(self: Trunk, input: ops.Tensor, index: usize) !ops.Tensor {
        const r = self.root.runtime;
        const weights = try self.root.indexed("blocks", index);
        const windowed = !self.global(index);
        const norm1 = try r.visionNorm(try weights.named("norm1"), input, windowed);
        const projections = try r.visionLinear(try weights.named("attn.qkv"), norm1, self.width, false);
        const rotary = try r.visionQkv(try weights.named("attn.freqs_cis"), projections, !windowed);
        const attention = try r.visionAttention(rotary[0], rotary[1], rotary[2], !windowed);
        const residual = try r.visionResidual(try weights.named("attn.proj"), attention, self.width, input, windowed);
        const norm2 = try r.visionNorm(try weights.named("norm2"), residual, false);
        const expansion = try r.visionLinear(try weights.named("mlp.fc1"), norm2, self.width, true);
        return r.visionResidual(try weights.named("mlp.fc2"), expansion, r.config.vision_mlp, residual, false);
    }

    fn observe(self: Trunk, label: []const u8, tensor: ops.Tensor) !void {
        try self.root.runtime.observe(label, tensor, &.{ 1, self.side, self.side, self.width });
    }
};

pub fn encode(r: *ops.Runtime, input: ops.Feature) !ops.Feature {
    if (r.config.vision_layers < 4 or r.config.vision_layers % 4 != 0) return error.SamImageLayerShape;
    const tower = Trunk{
        .root = .{ .runtime = r, .prefix = "detector.backbone.vision_backbone.trunk" },
        .side = r.config.featureSide(),
        .width = r.config.vision_width,
    };
    const start = r.mark();
    errdefer r.trim(start, &.{});
    const position = try tower.position();
    for (0..r.config.vision_layers) |index| {
        const weights = try tower.root.indexed("blocks", index);
        _ = try r.visionRotaryTable(try weights.named("attn.freqs_cis"), tower.global(index));
    }
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const patches = try r.conv(try tower.root.named("patch_embed.proj"), input, r.config.patch_size, 0, false, false);
    if (patches.height != tower.side or patches.width != tower.side or patches.channels != tower.width) return error.SamImagePatchShape;
    try tower.observe("trunk.patch_embed", patches.tensor);
    var value = try tower.root.norm("ln_pre", try r.add(patches.tensor, position), 1e-5);
    try tower.observe("trunk.ln_pre", value);
    r.trim(start, &.{value});
    for (0..r.config.vision_layers) |index| {
        value = try tower.layer(value, index);
        try tower.observe(try r.name("trunk.blocks.{d}", .{index}), value);
        r.trim(start, &.{value});
        if (!r.vision_ready and (index + 1) % 4 == 0) {
            try r.g.endBatch();
            try r.g.beginBatch();
        }
    }
    try r.g.endBatch();
    r.vision_ready = true;
    return ops.feature(value, tower.side, tower.side, tower.width);
}

pub fn pyramid(r: *ops.Runtime, encoded: ops.Feature, kind: []const u8) ![3]ops.Feature {
    const root = Layer{ .runtime = r, .prefix = try r.name("detector.backbone.vision_backbone.{s}", .{kind}) };
    const paths = [_][]const []const u8{
        &.{ "dconv_2x2_0", "dconv_2x2_1" }, &.{"dconv_2x2"}, &.{},
    };
    const start = r.mark();
    errdefer r.trim(start, &.{});
    var result: [3]ops.Feature = undefined;
    var keep: [4]ops.Tensor = undefined;
    keep[0] = encoded.tensor;
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    for (paths, 0..) |path, level| {
        const branch = try root.child(try r.name("{d}", .{level}));
        var value = encoded;
        for (path, 0..) |name, step| {
            value = try r.conv(try branch.named(name), value, 2, 0, true, false);
            if (step + 1 < path.len) value.tensor = try r.gelu(value.tensor);
        }
        value = try r.conv(try branch.named("conv_1x1"), value, 1, 0, false, false);
        result[level] = try r.conv(try branch.named("conv_3x3"), value, 1, 1, false, false);
        keep[level + 1] = result[level].tensor;
        r.trim(start, keep[0 .. level + 2]);
    }
    try r.g.endBatch();
    return result;
}
