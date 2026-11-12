const ops = @import("ops.zig");
const decoder = @import("decoder.zig");
const Layer = @import("layers.zig").Layer;
const Size = @import("tensor_metal").mtl.Size;

pub const Projected = struct { key: ops.Tensor, value: ops.Tensor };
pub const Memory = struct {
    index: usize,
    conditioning: bool,
    encoded: ops.Tensor,
    image: ops.Tensor,
    pointer: ops.Tensor,
    score: f32 = 0,
    projected: ?[4]Projected = null,
    spatial_live: bool = true,

    pub fn retained(self: Memory, destination: *[11]ops.Tensor) []const ops.Tensor {
        destination[0] = self.pointer;
        var count: usize = 1;
        if (self.spatial_live) {
            destination[1] = self.encoded;
            destination[2] = self.image;
            count = 3;
            if (self.projected) |cache| for (cache) |layer| {
                destination[count] = layer.key;
                destination[count + 1] = layer.value;
                count += 2;
            };
        }
        return destination[0..count];
    }

    pub fn release(self: Memory, r: *ops.Runtime) void {
        var tensors: [11]ops.Tensor = undefined;
        for (self.retained(&tensors)) |tensor| r.release(tensor);
    }

    pub fn expireSpatial(self: *Memory, r: *ops.Runtime) void {
        if (!self.spatial_live) return;
        var tensors: [11]ops.Tensor = undefined;
        for (self.retained(&tensors)[1..]) |tensor| r.release(tensor);
        self.projected = null;
        self.spatial_live = false;
    }
};

fn snapshot(r: *ops.Runtime, tensor: ops.Tensor) !ops.Tensor {
    const output = try r.own(try r.g.allocDedicated(tensor.count, tensor.dtype));
    try r.g.dispatch("im_sam_copy", &.{ tensor, output }, @as(u32, @intCast(tensor.count)), Size.of(tensor.count, 1, 1), Size.of(256, 1, 1));
    return output;
}

pub const MaskParams = extern struct {
    pixels: u32,
    channels: u32,
    active: u32,
    conditioning: u32,
    hard: u32,
    scale: f32,
    bias: f32,
};

const Encoder = struct {
    owner: Layer,
    backbone: Layer,
    multiplex: bool,
    sparse: bool,

    fn absence(self: Encoder, prediction: decoder.Result, active: u16) !ops.Tensor {
        const r = self.owner.runtime;
        const embedding = try self.owner.parameter("no_obj_embed_spatial");
        const output = try r.alloc(r.config.memory_width);
        @memset(output.slice(f32, output.count), 0);
        for (0..r.config.bucket_size) |slot| {
            if (active & (@as(u16, 1) << @intCast(slot)) != 0 and prediction.scores[slot] > 0) continue;
            for (output.slice(f32, output.count), 0..) |*value, channel| value.* += embedding.get(slot * output.count + channel);
        }
        return output;
    }

    fn masks(self: Encoder, input: ops.Feature, active: u16, conditioning: u16, hard: bool) !ops.Feature {
        const r = self.owner.runtime;
        const side = if (self.multiplex) r.config.featureSide() * 16 else r.config.image_size;
        const resized = try r.resize(input, side, side, true);
        const channels: usize = if (self.multiplex and !self.sparse) r.config.bucket_size * 2 else 1;
        const output = try r.device(side * side * channels);
        if (self.sparse) {
            try r.g.dispatch("im_sam_mask_values", &.{ resized.tensor, output }, @as(u32, @intCast(side * side)), Size.of(output.count, 1, 1), Size.of(256, 1, 1));
        } else {
            const parameters = MaskParams{
                .pixels = @intCast(side * side),
                .channels = @intCast(channels),
                .active = active,
                .conditioning = conditioning & active,
                .hard = @intFromBool(hard and !self.multiplex),
                .scale = if (self.multiplex) 2 else 20,
                .bias = if (self.multiplex) -1 else -10,
            };
            try r.g.dispatch("im_sam_memory_masks", &.{ resized.tensor, output }, parameters, Size.of(output.count, 1, 1), Size.of(256, 1, 1));
        }
        const encoded = ops.feature(output, side, side, channels);
        return if (self.multiplex) encoded else r.resize(encoded, r.config.featureSide() * 16, r.config.featureSide() * 16, true);
    }

    fn downsample(self: Encoder, input_mask: ops.Feature, conditioning: u16, keep: ops.Tensor, mark: usize) !ops.Feature {
        const r = self.owner.runtime;
        const network = try self.backbone.child("mask_downsampler.encoder");
        var image = input_mask;
        for (0..4) |stage| {
            const convolution = try r.name("{d}", .{stage * 3});
            if (stage == 0 and self.sparse) {
                const side = image.height / 2;
                const bias = try network.parameter("0.bias");
                const output = try r.device(side * side * bias.count);
                try r.g.dispatch("im_sam_mask_downsample", &.{ image.tensor, try network.parameter("0.weight"), bias, output }, @as(u32, @intFromBool(conditioning & 1 != 0)), Size.of(output.count, 1, 1), Size.of(256, 1, 1));
                image = ops.feature(output, side, side, bias.count);
            } else image = try r.conv(try network.named(convolution), image, 2, 1, false, false);
            image.tensor = try r.gelu(try network.norm(try r.name("{d}", .{stage * 3 + 1}), image.tensor, 1e-6));
            r.trim(mark, &.{ image.tensor, keep });
        }
        return r.conv(try network.named("12"), image, 1, 0, false, false);
    }

    fn fuse(self: Encoder, image: ops.Tensor, mask: ops.Feature) !ops.Tensor {
        const r = self.owner.runtime;
        const side = r.config.featureSide();
        const projected = try r.conv(try self.backbone.named("pix_feat_proj"), ops.feature(image, side, side, r.config.hidden), 1, 0, false, false);
        var output = ops.feature(try r.add(mask.tensor, projected.tensor), side, side, projected.channels);
        for (0..2) |stage| {
            const block = try self.backbone.indexed("fuser.layers", stage);
            const residual = output.tensor;
            const spatial = try r.conv(try block.named("dwconv"), output, 1, 3, false, true);
            const normalized = try block.norm("norm", spatial.tensor, 1e-6);
            const expanded = try r.gelu(try block.linear("pwconv1", normalized));
            const contracted = try block.linear("pwconv2", expanded);
            output.tensor = try r.add(residual, try r.mul(contracted, try block.parameter("gamma")));
        }
        if (r.store.has(try self.backbone.named("out_proj.weight")))
            output = try r.conv(try self.backbone.named("out_proj"), output, 1, 0, false, false);
        return output.tensor;
    }

    fn projections(self: Encoder, image: ops.Tensor, encoded: ops.Tensor) !?[4]Projected {
        if (!self.multiplex or self.owner.runtime.precision != .mixed) return null;
        const r = self.owner.runtime;
        var cache: [4]Projected = undefined;
        for (&cache, 0..) |*entry, index| {
            const mark = r.mark();
            const block = try self.owner.indexed("transformer.encoder.layers", index);
            const image_key = try block.linear("image_cross_attn_k_proj", image);
            entry.key = try snapshot(r, try r.add(image_key, try block.linear("cross_attn_k_proj", encoded)));
            entry.value = try snapshot(r, try block.linear("cross_attn_v_proj", encoded));
            r.trim(mark, &.{ entry.key, entry.value });
        }
        return cache;
    }
};

pub fn encode(r: *ops.Runtime, image: ops.Tensor, prediction: decoder.Result, active: u16, conditioning: u16, from_points: bool, index: usize) !Memory {
    const mark = r.mark();
    errdefer r.trim(mark, &.{});
    const owner = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
    const encoder = Encoder{ .owner = owner, .backbone = try owner.child("maskmem_backbone"), .multiplex = r.config.version == .sam31, .sparse = r.config.version == .sam31 and r.precision == .mixed and active == 1 };
    const absent = try encoder.absence(prediction, active);
    var mask = prediction.input_masks orelse prediction.masks;
    if (prediction.input_masks != null) mask = try r.resize(mask, r.config.image_size, r.config.image_size, true);
    if (encoder.sparse) mask = ops.feature(try r.columns(mask.tensor, mask.height * mask.width, r.config.bucket_size, 0, 1), mask.height, mask.width, 1);
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const down = try encoder.downsample(try encoder.masks(mask, active, conditioning, from_points), conditioning, absent, mark);
    const encoded = try r.add(try encoder.fuse(image, down), absent);
    const visible = if (prediction.scores[0] > 0) 2 / (1 + @exp(-prediction.scores[0])) - 1 else 0;
    const memory = Memory{
        .index = index,
        .conditioning = conditioning != 0,
        .encoded = encoded,
        .image = try snapshot(r, image),
        .pointer = try snapshot(r, prediction.pointer),
        .score = visible * prediction.ious[0],
        .projected = try encoder.projections(image, encoded),
    };
    var retained: [11]ops.Tensor = undefined;
    r.trim(mark, memory.retained(&retained));
    try r.g.endBatch();
    return memory;
}
