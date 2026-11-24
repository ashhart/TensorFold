const std = @import("std");
const gpu = @import("tensor_metal");
const runtime = @import("runtime.zig");
const variant = @import("variant.zig");
const kernel_abi = @import("kernel_abi.zig");
const elementwise = @import("elementwise.zig");
const projection = @import("projection.zig");
const convolution = @import("convolution.zig");
const spatial = @import("spatial.zig");
const norms = @import("norms.zig");
const memory_bank = @import("memory_bank.zig");
const outcome = @import("outcome.zig");

const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;
const Scope = runtime.Scope;
const Entry = memory_bank.Entry;
const KeyValue = memory_bank.KeyValue;
const Outcome = outcome.Outcome;

const downsampler_eps: f32 = 1e-6;
const fuser_eps: f32 = 1e-6;
const reader_layers = 4;
const fuser_layers = 2;
const first_conv_channels = 16;

pub const Write = struct { active: u16, prompted: u16, hard: bool, frame: usize };

// The single-channel first convolution needs every nonzero weight within [2^-36, 2^36]; NaN and infinities fail.
fn singleChannel(weights: []const f32) bool {
    for (weights) |w| {
        if (w == 0) continue;
        const magnitude = @abs(w);
        if (!(magnitude >= 0x1p-36 and magnitude <= 0x1p36)) return false;
    }
    return true;
}

fn squashParams(version: variant.Version, sparse: bool, precision: runtime.Precision, hard: bool, slots: usize, active: u16, pixels: usize) kernel_abi.Squash {
    var params = kernel_abi.Squash{
        .pixels = @intCast(pixels),
        .stride = 1,
        .slots = 1,
        .sources = 1,
        .active = active,
        .threshold = 0,
        .accurate = 0,
        .gain = 2,
        .offset = -1,
    };
    switch (version) {
        .sam3 => {
            params.gain = 20;
            params.offset = -10;
            params.threshold = @intFromBool(hard);
        },
        .sam31 => if (sparse) {
            params.accurate = @intFromBool(precision == .mixed);
        } else {
            params.stride = @intCast(2 * slots);
            params.slots = @intCast(slots);
            params.sources = @intCast(slots);
        },
    }
    return params;
}

// Bit s is set for slots that contribute an absent-object row: unoccupied, or occupied without positive presence.
fn absentSlots(active: u16, presence: []const f32) u16 {
    var bits: u16 = 0;
    for (presence, 0..) |logit, s| {
        const bit = @as(u16, 1) << @intCast(s);
        if (!(active & bit != 0 and logit > 0)) bits |= bit;
    }
    return bits;
}

fn confidence(presence: f32, quality: f32) f32 {
    if (!(presence > 0)) return 0;
    return (2 / (1 + @exp(-presence)) - 1) * quality;
}

fn absentVector(rt: *Runtime, tracker: Scope, active: u16, presence: []const f32) !Tensor {
    const width = rt.config.memory_width;
    const rows = try tracker.param("no_obj_embed_spatial");
    if (rows.count != presence.len * width) return error.SamMemoryShape;
    const table = rows.slice(f32, rows.count);
    const vector = try rt.hostTensor(width);
    const sum = vector.slice(f32, width);
    @memset(sum, 0);
    const bits = absentSlots(active, presence);
    for (0..presence.len) |s| {
        if (bits & (@as(u16, 1) << @intCast(s)) == 0) continue;
        for (sum, table[s * width ..][0..width]) |*out, value| out.* = out.* + value;
    }
    return vector;
}

fn firstConv(rt: *Runtime, encoder: Scope, canvas: FeatureMap, prompted: bool) !FeatureMap {
    const info = try encoder.info("0.weight");
    if (!std.mem.eql(usize, info.shape[0..info.rank], &.{ first_conv_channels, 2 * first_conv_channels, 3, 3 }) or
        canvas.height != 16 * rt.config.featureSide() or canvas.width != canvas.height or canvas.channels != 1)
        return error.SamMemoryShape;
    const weight = try encoder.param("0.weight");
    const bias = try encoder.param("0.bias");
    const side = canvas.height / 2;
    const count = side * side * first_conv_channels;
    const out = try rt.deviceTensor(count, .f32);
    const flag: u32 = @intFromBool(prompted);
    try rt.gpu.dispatch(kernel_abi.mask_first_conv, &.{ canvas.tensor, weight, bias, out }, flag, gpu.mtl.Size.of(count, 1, 1), gpu.mtl.Size.of(256, 1, 1));
    return FeatureMap.of(out, side, side, first_conv_channels);
}

fn normGelu(rt: *Runtime, scope: Scope, leaf: []const u8, x: FeatureMap) !FeatureMap {
    const normed = try norms.layerNorm(scope, leaf, x.tensor, downsampler_eps, .plain);
    return FeatureMap.of(try elementwise.gelu(rt, normed), x.height, x.width, x.channels);
}

fn canvasOf(rt: *Runtime, source: FeatureMap, params: kernel_abi.Squash, sparse: bool, request: Write) !FeatureMap {
    const frame = rt.config.image_size;
    const slots = rt.config.bucket_size;
    var logits = source;
    if (sparse) logits = FeatureMap.of(
        try elementwise.columns(rt, source.tensor, source.height * source.width, slots, 0, 1),
        source.height,
        source.width,
        1,
    );
    const enlarged = try spatial.resample(rt, logits, frame, frame, .bilinear);
    if (enlarged.channels != params.sources) return error.SamMemoryShape;
    const channels: usize = params.stride;
    const squashed = try rt.deviceTensor(frame * frame * channels, .f32);
    const threads = frame * frame * @as(usize, params.slots);
    try rt.gpu.dispatch(kernel_abi.mask_squash, &.{ enlarged.tensor, squashed }, params, gpu.mtl.Size.of(threads, 1, 1), gpu.mtl.Size.of(256, 1, 1));
    if (channels > params.slots)
        try elementwise.fillChannels(rt, squashed, frame * frame, channels, slots, slots, request.prompted & request.active);
    const side = 16 * rt.config.featureSide();
    const canvas = try spatial.resample(rt, FeatureMap.of(squashed, frame, frame, channels), side, side, .bilinear);
    try rt.observe("tracker.memory_mask_input", canvas.tensor, &.{ 1, side, side, channels });
    return canvas;
}

fn downsample(rt: *Runtime, encoder: Scope, canvas: FeatureMap, sparse: bool, request: Write) !FeatureMap {
    const mark = rt.mark();
    var x = if (sparse)
        try firstConv(rt, encoder, canvas, request.prompted & 1 != 0)
    else
        try convolution.conv(encoder, "0", canvas, .{ .stride = 2, .pad = 1 });
    x = try normGelu(rt, encoder, "1", x);
    rt.rewind(mark, &.{x.tensor});
    for ([_][2][]const u8{ .{ "3", "4" }, .{ "6", "7" }, .{ "9", "10" } }) |stage| {
        x = try convolution.conv(encoder, stage[0], x, .{ .stride = 2, .pad = 1 });
        x = try normGelu(rt, encoder, stage[1], x);
        rt.rewind(mark, &.{x.tensor});
    }
    return convolution.conv(encoder, "12", x, .{});
}

fn withImage(rt: *Runtime, backbone: Scope, mask_code: FeatureMap, coarse_image: Tensor) !FeatureMap {
    const grid = rt.config.featureSide();
    const width = rt.config.hidden;
    const image_term = try convolution.conv(backbone, "pix_feat_proj", FeatureMap.of(coarse_image, grid, grid, width), .{});
    if (mask_code.tensor.count != image_term.tensor.count) return error.SamMemoryShape;
    var code = try elementwise.add(rt, mask_code.tensor, image_term.tensor);
    for (0..fuser_layers) |j| {
        const layer = try backbone.item("fuser.layers", j);
        const spread = try convolution.depthwise(layer, "dwconv", FeatureMap.of(code, grid, grid, width), 3);
        const normed = try norms.layerNorm(layer, "norm", spread.tensor, fuser_eps, .plain);
        const expanded = try elementwise.gelu(rt, try projection.affine(layer, "pwconv1", normed));
        const contracted = try projection.affine(layer, "pwconv2", expanded);
        code = try elementwise.add(rt, code, try elementwise.mul(rt, contracted, try layer.param("gamma")));
    }
    var out = FeatureMap.of(code, grid, grid, width);
    if (backbone.has("out_proj.weight")) out = try convolution.conv(backbone, "out_proj", out, .{});
    return out;
}

fn cachedProjections(rt: *Runtime, tracker: Scope, coarse_image: Tensor, encoding: Tensor) ![reader_layers]KeyValue {
    var pairs: [reader_layers]KeyValue = undefined;
    for (&pairs, 0..) |*pair, i| {
        const mark = rt.mark();
        const layer = try tracker.item("transformer.encoder.layers", i);
        const image_key = try projection.affine(layer, "image_cross_attn_k_proj", coarse_image);
        const key = try elementwise.add(rt, image_key, try projection.affine(layer, "cross_attn_k_proj", encoding));
        const value = try projection.affine(layer, "cross_attn_v_proj", encoding);
        pair.* = .{ .key = try elementwise.dedicatedCopy(rt, key), .value = try elementwise.dedicatedCopy(rt, value) };
        rt.rewind(mark, &.{ pair.key, pair.value });
    }
    return pairs;
}

pub fn write(rt: *Runtime, coarse_image: Tensor, group: Outcome, request: Write) !Entry {
    const config = rt.config;
    const slots = config.bucket_size;
    const grid = config.featureSide();
    if (group.slots != slots or slots > outcome.max_slots) return error.SamMemoryShape;
    if (coarse_image.count != grid * grid * config.hidden or group.pointers.count != slots * config.hidden)
        return error.SamMemoryShape;
    const source = group.full_logits orelse group.mask_logits;
    if (source.channels != slots or source.tensor.count != source.height * source.width * slots) return error.SamMemoryShape;
    const tracker = Scope.root(rt, config.roots.tracker);
    const backbone = try tracker.sub("maskmem_backbone");
    const encoder = try backbone.sub("mask_downsampler.encoder");
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const grouped = config.version == .sam31;
    const first_weight = try encoder.param("0.weight");
    const sparse = grouped and request.active == 1 and singleChannel(first_weight.slice(f32, first_weight.count));
    const absent = try absentVector(rt, tracker, request.active, group.presence[0..slots]);
    const params = squashParams(config.version, sparse, rt.routes.precision, request.hard, slots, request.active, config.image_size * config.image_size);

    try rt.gpu.beginBatch();
    errdefer if (rt.gpu.batch != null) rt.gpu.endBatch() catch {};
    const canvas = try canvasOf(rt, source, params, sparse, request);
    const mask_code = try downsample(rt, encoder, canvas, sparse, request);
    const encoded = try withImage(rt, backbone, mask_code, coarse_image);
    if (encoded.channels != config.memory_width) return error.SamMemoryShape;
    var entry = Entry{
        .frame = request.frame,
        .prompted = request.prompted != 0,
        .encoding = try elementwise.add(rt, encoded.tensor, absent),
        .coarse_image = try elementwise.dedicatedCopy(rt, coarse_image),
        .pointer = try elementwise.dedicatedCopy(rt, group.pointers),
        .confidence = confidence(group.presence[0], group.quality[0]),
        .grid_live = true,
    };
    if (grouped and rt.observer == null)
        entry.projections = try cachedProjections(rt, tracker, coarse_image, entry.encoding);
    var buffer: [11]Tensor = undefined;
    rt.rewind(m, entry.tensors(&buffer));
    try rt.gpu.endBatch();
    return entry;
}

test "SAM memory single-channel predicate bounds every nonzero first-stage weight" {
    const good = [_]f32{ 0, -0.0, 0x1p-36, -0x1p36, 0.75, -3 };
    try std.testing.expect(singleChannel(&good));
    const bad = [_]f32{
        std.math.nextAfter(f32, 0x1p-36, 0),
        std.math.nextAfter(f32, 0x1p36, std.math.inf(f32)),
        -0x1p-40,
        0x1p60,
        std.math.nan(f32),
        -std.math.inf(f32),
    };
    for (bad) |value| {
        var weights: [good.len + 1]f32 = undefined;
        @memcpy(weights[0..good.len], &good);
        weights[good.len] = value;
        try std.testing.expect(!singleChannel(&weights));
    }
}

test "SAM memory squash parameters per version, path, precision and hard flag" {
    const pixels = 1008 * 1008;
    const fp32 = squashParams(.sam31, true, .fp32, false, 16, 1, pixels);
    try std.testing.expectEqual(kernel_abi.Squash{ .pixels = pixels, .stride = 1, .slots = 1, .sources = 1, .active = 1, .threshold = 0, .accurate = 0, .gain = 2, .offset = -1 }, fp32);
    const mixed = squashParams(.sam31, true, .mixed, false, 16, 1, pixels);
    try std.testing.expectEqual(@as(u32, 1), mixed.accurate);
    for ([_]runtime.Precision{ .fp32, .mixed }) |precision| {
        const general = squashParams(.sam31, false, precision, false, 16, 0b101, pixels);
        try std.testing.expectEqual(kernel_abi.Squash{ .pixels = pixels, .stride = 32, .slots = 16, .sources = 16, .active = 0b101, .threshold = 0, .accurate = 0, .gain = 2, .offset = -1 }, general);
    }
    for ([_]bool{ false, true }) |hard| {
        const sam3 = squashParams(.sam3, false, .fp32, hard, 1, 1, pixels);
        try std.testing.expectEqual(kernel_abi.Squash{ .pixels = pixels, .stride = 1, .slots = 1, .sources = 1, .active = 1, .threshold = @intFromBool(hard), .accurate = 0, .gain = 20, .offset = -10 }, sam3);
    }
}

test "SAM memory absent slots include unoccupied slots and non-positive or NaN presence" {
    var presence: [16]f32 = @splat(-1);
    try std.testing.expectEqual(@as(u16, 0xffff), absentSlots(0b1, &presence));
    presence = @splat(2);
    try std.testing.expectEqual(@as(u16, 0xfffe), absentSlots(0b1, &presence));
    presence[0] = std.math.nan(f32);
    try std.testing.expectEqual(@as(u16, 0xffff), absentSlots(0b1, &presence));
}

test "SAM memory confidence is zero unless presence is positive" {
    for ([_]f32{ 0, -1, std.math.nan(f32) }) |presence| try std.testing.expectEqual(@as(f32, 0), confidence(presence, 0.5));
    var p: f32 = 2;
    var q: f32 = 0.5;
    _ = .{ &p, &q };
    const expected = (2 / (1 + @exp(-p)) - 1) * q;
    try std.testing.expectEqual(expected, confidence(p, q));
}
