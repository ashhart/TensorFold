const std = @import("std");
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
        if (!self.spatial_live) return destination[0..1];
        destination[1] = self.encoded;
        destination[2] = self.image;
        var length: usize = 3;
        if (self.projected) |entries| {
            for (entries) |entry| {
                destination[length..][0..2].* = .{ entry.key, entry.value };
                length += 2;
            }
        }
        return destination[0..length];
    }

    pub fn release(self: Memory, runtime: *ops.Runtime) void {
        var storage: [11]ops.Tensor = undefined;
        for (self.retained(&storage)) |tensor| runtime.release(tensor);
    }

    pub fn expireSpatial(self: *Memory, runtime: *ops.Runtime) void {
        if (self.spatial_live) {
            var storage: [11]ops.Tensor = undefined;
            for (self.retained(&storage)[1..]) |tensor| runtime.release(tensor);
            self.spatial_live = false;
            self.projected = null;
        }
    }
};

pub const MaskParams = extern struct {
    pixels: u32,
    channels: u32,
    active: u32,
    conditioning: u32,
    hard: u32,
    scale: f32,
    bias: f32,
};

const Packing = struct {
    multiplex: bool,
    sparse: bool,

    fn of(root: Layer, active: u16) !Packing {
        const multiplex = root.runtime.config.version == .sam31;
        var sparse = false;
        if (multiplex and active == 1) {
            const weights = try root.parameter("maskmem_backbone.mask_downsampler.encoder.0.weight");
            sparse = sparseWeights(weights.slice(f32, weights.count));
        }
        return .{ .multiplex = multiplex, .sparse = sparse };
    }

    fn input(self: Packing, root: Layer, prediction: decoder.Result, active: u16, conditioning: u16, hard: bool) !ops.Feature {
        const r = root.runtime;
        var mask = prediction.input_masks orelse prediction.masks;
        if (self.sparse) {
            const column = try r.columns(mask.tensor, mask.height * mask.width, r.config.bucket_size, 0, 1);
            mask = ops.feature(column, mask.height, mask.width, 1);
        }
        const side = r.config.image_size;
        if (prediction.input_masks != null) mask = try r.resize(mask, side, side, true);
        mask = try r.resize(mask, side, side, true);
        const channels = if (self.multiplex and !self.sparse) 2 * r.config.bucket_size else 1;
        const output = try r.device(side * side * channels);
        if (self.sparse and r.precision != .fp32) {
            try r.g.dispatch("im_sam_mask_values", &.{ mask.tensor, output }, @as(u32, @intCast(side * side)), Size.of(output.count, 1, 1), Size.of(256, 1, 1));
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
            try r.g.dispatch("im_sam_memory_masks", &.{ mask.tensor, output }, parameters, Size.of(output.count, 1, 1), Size.of(256, 1, 1));
        }
        const extent = 16 * r.config.featureSide();
        return r.resize(ops.feature(output, side, side, channels), extent, extent, true);
    }

    fn downsample(self: Packing, root: Layer, mask: ops.Feature, conditioning: u16, absence: ops.Tensor, start: usize) !ops.Feature {
        const r = root.runtime;
        const network = try root.child("maskmem_backbone.mask_downsampler.encoder");
        var value = mask;
        for (0..4) |level| {
            if (level != 0 or !self.sparse) {
                value = try r.conv(try network.named(try r.name("{d}", .{3 * level})), value, 2, 1, false, false);
            } else {
                const weights = try network.parameter("0.weight");
                const bias = try network.parameter("0.bias");
                const extent = value.height / 2;
                const tensor = try r.device(extent * extent * bias.count);
                try r.g.dispatch("im_sam_mask_downsample", &.{ value.tensor, weights, bias, tensor }, @as(u32, @intFromBool(conditioning & 1 != 0)), Size.of(tensor.count, 1, 1), Size.of(256, 1, 1));
                value = ops.feature(tensor, extent, extent, bias.count);
            }
            value.tensor = try network.norm(try r.name("{d}", .{3 * level + 1}), value.tensor, 1e-6);
            value.tensor = try r.gelu(value.tensor);
            r.trim(start, &.{ value.tensor, absence });
        }
        return r.conv(try network.named("12"), value, 1, 0, false, false);
    }
};

fn absentEmbedding(root: Layer, prediction: decoder.Result, active: u16) !ops.Tensor {
    const r = root.runtime;
    const symbols = try root.parameter("no_obj_embed_spatial");
    const result = try r.alloc(r.config.memory_width);
    const row = result.slice(f32, result.count);
    @memset(row, 0);
    for (0..r.config.bucket_size) |slot| {
        const visible = active & (@as(u16, 1) << @intCast(slot)) != 0 and prediction.scores[slot] > 0;
        if (!visible) for (row, 0..) |*value, channel| {
            value.* += symbols.get(slot * row.len + channel);
        };
    }
    return result;
}

fn fuseImage(root: Layer, mask: ops.Feature, image: ops.Tensor) !ops.Tensor {
    const r = root.runtime;
    const network = try root.child("maskmem_backbone");
    const side = r.config.featureSide();
    const pixels = ops.feature(image, side, side, r.config.hidden);
    var fused = try r.conv(try network.named("pix_feat_proj"), pixels, 1, 0, false, false);
    fused.tensor = try r.add(mask.tensor, fused.tensor);
    for (0..2) |index| {
        const block = try network.indexed("fuser.layers", index);
        const spatial = try r.conv(try block.named("dwconv"), fused, 1, 3, false, true);
        const normalized = try block.norm("norm", spatial.tensor, 1e-6);
        const expansion = try r.gelu(try block.linear("pwconv1", normalized));
        const update = try block.linear("pwconv2", expansion);
        const weighted = try r.mul(update, try block.parameter("gamma"));
        fused.tensor = try r.add(fused.tensor, weighted);
    }
    if (r.store.has(try network.named("out_proj.weight"))) {
        fused = try r.conv(try network.named("out_proj"), fused, 1, 0, false, false);
    }
    return fused.tensor;
}

fn snapshot(runtime: *ops.Runtime, tensor: ops.Tensor) !ops.Tensor {
    const copy = try runtime.own(try runtime.g.allocDedicated(tensor.count, tensor.dtype));
    try runtime.g.dispatch("im_sam_copy", &.{ tensor, copy }, @as(u32, @intCast(tensor.count)), Size.of(tensor.count, 1, 1), Size.of(256, 1, 1));
    return copy;
}

fn projectionCache(root: Layer, image: ops.Tensor, memory: ops.Tensor) !?[4]Projected {
    const r = root.runtime;
    if (r.config.version != .sam31 or r.precision != .mixed) return null;
    var entries: [4]Projected = undefined;
    for (0..entries.len) |index| {
        const start = r.mark();
        const layer = try root.indexed("transformer.encoder.layers", index);
        const image_key = try layer.linear("image_cross_attn_k_proj", image);
        const memory_key = try layer.linear("cross_attn_k_proj", memory);
        const key = try snapshot(r, try r.add(image_key, memory_key));
        const value = try snapshot(r, try layer.linear("cross_attn_v_proj", memory));
        entries[index] = .{ .key = key, .value = value };
        r.trim(start, &.{ key, value });
    }
    return entries;
}

pub fn encode(r: *ops.Runtime, image: ops.Tensor, prediction: decoder.Result, active: u16, conditioning: u16, from_points: bool, index: usize) !Memory {
    const start = r.mark();
    errdefer r.trim(start, &.{});
    const root = Layer{ .runtime = r, .prefix = r.config.trackerPrefix() };
    const packing = try Packing.of(root, active);
    const absent = try absentEmbedding(root, prediction, active);
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const mask = try packing.input(root, prediction, active, conditioning, from_points);
    try r.observe("tracker.memory_mask_input", mask.tensor, &.{ 1, mask.height, mask.width, mask.channels });
    const reduced = try packing.downsample(root, mask, conditioning, absent, start);
    const encoded = try r.add(try fuseImage(root, reduced, image), absent);
    const visible = if (prediction.scores[0] > 0) 2 / (1 + @exp(-prediction.scores[0])) - 1 else 0;
    const result = Memory{
        .index = index,
        .conditioning = conditioning != 0,
        .encoded = encoded,
        .image = try snapshot(r, image),
        .pointer = try snapshot(r, prediction.pointer),
        .score = visible * prediction.ious[0],
        .projected = try projectionCache(root, image, encoded),
    };
    var storage: [11]ops.Tensor = undefined;
    r.trim(start, result.retained(&storage));
    try r.g.endBatch();
    return result;
}

fn sparseWeights(values: []const f32) bool {
    for (values) |value| {
        const magnitude = @abs(value);
        if (magnitude == 0) continue;
        if (!(magnitude >= 0x1p-36 and magnitude <= 0x1p36)) return false;
    }
    return true;
}

test "SAM sparse memory masks require bounded finite weights" {
    try std.testing.expect(sparseWeights(&.{ 0, -0.0, 0x1p-36, -0x1p-36, 0x1p36, -0x1p36 }));
    for ([_]f32{ 0x1p-37, 0x1p37, 0x1p-149, std.math.floatMax(f32), std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |value| {
        try std.testing.expect(!sparseWeights(&.{ 0, value, 1 }));
    }
}
