// SAM memory architecture adapted from facebookresearch/sam3 under LICENSES/SAM.txt.
const std = @import("std");
const ops = @import("ops.zig");
const decoder = @import("decoder.zig");
const Runtime = ops.Runtime;
const Tensor = ops.Tensor;
const feature = ops.feature;
const Size = @import("tensor_metal").mtl.Size;

pub const Projected = struct { key: Tensor, value: Tensor };
pub const Memory = struct {
    index: usize,
    conditioning: bool,
    encoded: Tensor,
    image: Tensor,
    pointer: Tensor,
    score: f32 = 0,
    projected: ?[4]Projected = null,
    spatial_live: bool = true,

    pub fn retained(self: Memory, storage: *[11]Tensor) []const Tensor {
        storage[0] = self.pointer;
        if (!self.spatial_live) return storage[0..1];
        storage[1..3].* = .{ self.encoded, self.image };
        if (self.projected) |layers| {
            for (layers, 0..) |layer, i| storage[3 + i * 2 ..][0..2].* = .{ layer.key, layer.value };
            return storage;
        }
        return storage[0..3];
    }

    pub fn release(self: Memory, r: *Runtime) void {
        var storage: [11]Tensor = undefined;
        for (self.retained(&storage)) |tensor| r.release(tensor);
    }

    pub fn expireSpatial(self: *Memory, r: *Runtime) void {
        if (!self.spatial_live) return;
        r.release(self.encoded);
        r.release(self.image);
        if (self.projected) |layers| for (layers) |layer| {
            r.release(layer.key);
            r.release(layer.value);
        };
        self.projected = null;
        self.spatial_live = false;
    }
};

pub fn copy(r: *Runtime, tensor: Tensor) !Tensor {
    const limit = r.g.buffer_cache_limit;
    r.g.buffer_cache_limit = 0;
    const cached = r.device(tensor.count);
    r.g.buffer_cache_limit = limit;
    const out = try cached;
    try r.g.dispatch("im_sam_copy", &.{ tensor, out }, @as(u32, @intCast(tensor.count)), Size.of(tensor.count, 1, 1), Size.of(256, 1, 1));
    return out;
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

pub fn encode(r: *Runtime, image: Tensor, result: decoder.Result, active: u16, conditioning: u16, from_points: bool, index: usize) !Memory {
    const start = r.mark();
    const root = r.config.trackerPrefix();
    const multiplex = r.config.version == .sam31;
    const width = r.config.memory_width;
    const absent = try r.weight(try r.name("{s}.no_obj_embed_spatial", .{root}));
    const summed = try r.alloc(width);
    @memset(summed.slice(f32, width), 0);
    for (0..r.config.bucket_size) |slot| {
        if (active & (@as(u16, 1) << @intCast(slot)) != 0 and result.scores[slot] > 0) continue;
        for (0..width) |c| summed.slice(f32, width)[c] += absent.get(slot * width + c);
    }
    const sparse = multiplex and active == 1 and r.precision == .mixed;
    var masks = result.input_masks orelse result.masks;
    if (result.input_masks != null) masks = try r.resize(masks, 1008, 1008, true);
    if (sparse) masks = feature(try r.columns(masks.tensor, masks.height * masks.width, 16, 0, 1), masks.height, masks.width, 1);
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const side: usize = if (multiplex) 1152 else 1008;
    const large = try r.resize(masks, side, side, true);
    const channels: usize = if (multiplex and !sparse) 32 else 1;
    const mux = try r.device(side * side * channels);
    if (sparse) {
        try r.g.dispatch("im_sam_mask_values", &.{ large.tensor, mux }, @as(u32, @intCast(side * side)), Size.of(mux.count, 1, 1), Size.of(256, 1, 1));
    } else {
        try r.g.dispatch("im_sam_memory_masks", &.{ large.tensor, mux }, MaskParams{
            .pixels = @intCast(side * side),
            .channels = @intCast(channels),
            .active = active,
            .conditioning = conditioning & active,
            .hard = @intFromBool(!multiplex and from_points),
            .scale = if (multiplex) 2 else 20,
            .bias = if (multiplex) -1 else -10,
        }, Size.of(mux.count, 1, 1), Size.of(256, 1, 1));
    }
    var f = feature(mux, side, side, if (multiplex) 32 else 1);
    if (!multiplex) f = try r.resize(f, 1152, 1152, true);
    const key = try r.name("{s}.maskmem_backbone", .{root});
    for (0..4) |i| {
        const widths = if (multiplex) [_]usize{ 16, 64, 256, 1024 } else [_]usize{ 4, 16, 64, 256 };
        if (sparse and i == 0) {
            const weight = try r.weight(try r.name("{s}.mask_downsampler.encoder.0.weight", .{key}));
            const bias = try r.weight(try r.name("{s}.mask_downsampler.encoder.0.bias", .{key}));
            const out = try r.device(576 * 576 * 16);
            try r.g.dispatch("im_sam_mask_downsample", &.{ mux, weight, bias, out }, @as(u32, @intFromBool(conditioning & 1 != 0)), Size.of(out.count, 1, 1), Size.of(256, 1, 1));
            f = feature(out, 576, 576, 16);
        } else f = try r.conv(try r.name("{s}.mask_downsampler.encoder.{d}", .{ key, i * 3 }), f, 2, 1, false, false);
        f.tensor = try r.gelu(try r.norm(try r.name("{s}.mask_downsampler.encoder.{d}", .{ key, i * 3 + 1 }), f.tensor, widths[i], 1e-6));
        r.trim(start, &.{ f.tensor, summed });
    }
    f = try r.conv(try r.name("{s}.mask_downsampler.encoder.12", .{key}), f, 1, 0, false, false);
    const projected = try r.conv(try r.name("{s}.pix_feat_proj", .{key}), feature(image, 72, 72, 256), 1, 0, false, false);
    f.tensor = try r.add(f.tensor, projected.tensor);
    for (0..2) |i| {
        const layer = try r.name("{s}.fuser.layers.{d}", .{ key, i });
        const residual = f.tensor;
        f = try r.conv(try r.name("{s}.dwconv", .{layer}), f, 1, 3, false, true);
        f.tensor = try r.norm(try r.name("{s}.norm", .{layer}), f.tensor, 256, 1e-6);
        const up = try r.gelu(try r.linear(try r.name("{s}.pwconv1", .{layer}), f.tensor, 256));
        const down = try r.linear(try r.name("{s}.pwconv2", .{layer}), up, 1024);
        f.tensor = try r.add(residual, try r.mul(down, try r.weight(try r.name("{s}.gamma", .{layer}))));
    }
    if (!multiplex) f = try r.conv(try r.name("{s}.out_proj", .{key}), f, 1, 0, false, false);
    const encoded = try r.add(f.tensor, summed);
    const confidence = if (result.scores[0] > 0) (2 / (1 + @exp(-result.scores[0])) - 1) * result.ious[0] else 0;
    var memory = Memory{ .index = index, .conditioning = conditioning != 0, .score = confidence, .encoded = encoded, .image = try copy(r, image), .pointer = try copy(r, result.pointer) };
    if (multiplex and r.precision == .mixed) {
        var layers: [4]Projected = undefined;
        for (&layers, 0..) |*cached, i| {
            const mark = r.mark();
            const layer = try r.name("{s}.transformer.encoder.layers.{d}", .{ root, i });
            cached.key = try copy(r, try r.add(try r.linear(try r.name("{s}.image_cross_attn_k_proj", .{layer}), image, 256), try r.linear(try r.name("{s}.cross_attn_k_proj", .{layer}), encoded, 256)));
            cached.value = try copy(r, try r.linear(try r.name("{s}.cross_attn_v_proj", .{layer}), encoded, 256));
            r.trim(mark, &.{ cached.key, cached.value });
        }
        memory.projected = layers;
    }
    var retained: [11]Tensor = undefined;
    r.trim(start, memory.retained(&retained));
    try r.g.endBatch();
    return memory;
}
