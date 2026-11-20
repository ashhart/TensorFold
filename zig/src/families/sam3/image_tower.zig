const std = @import("std");
const runtime = @import("runtime.zig");
const variant = @import("variant.zig");
const prompts = @import("prompts.zig");
const elementwise = @import("elementwise.zig");
const norms = @import("norms.zig");
const convolution = @import("convolution.zig");
const spatial = @import("spatial.zig");
const rotary = @import("rotary.zig");
const image_block = @import("image_block.zig");

const Runtime = runtime.Runtime;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;
const LevelSet = variant.LevelSet;
const Config = variant.Config;

const position_constant = "image_tower.position";

/// Validated RGB image → normalised FP32 map resampled bilinearly to the model frame.
pub fn prepare(rt: *Runtime, image: prompts.Image) !FeatureMap {
    try image.validate();
    const stride = try image.stride();
    const w = image.size.width;
    const h = image.size.height;
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const normalized = try rt.hostTensor(try std.math.mul(usize, try image.size.pixels(), 3));
    normalize(normalized.slice(f32, normalized.count), image.rgb, w, h, stride);
    try rt.observe("input.normalized_rgb", normalized, &.{ 1, h, w, 3 });
    const side = rt.config.image_size;
    const prepared = try spatial.resample(rt, FeatureMap.of(normalized, h, w, 3), side, side, .bilinear);
    rt.rewind(m, &.{prepared.tensor});
    return prepared;
}

/// `[1008, 1008, 3]` prepared map → trunk output `[72, 72, 1024]`.
pub fn encode(rt: *Runtime, prepared: FeatureMap) !FeatureMap {
    const c = rt.config;
    const side = c.featureSide();
    const width = c.vision_width;
    if (c.vision_layers < 4 or c.vision_layers % 4 != 0) return error.SamTrunkShape;
    if (prepared.height != c.image_size or prepared.width != c.image_size or prepared.channels != 3 or
        prepared.tensor.count != c.image_size * c.image_size * 3) return error.SamImageShape;
    const trunk = Scope.root(rt, c.roots.trunk);
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const position = try positionTable(rt, trunk);
    const tables = try rt.scratch.alloc(rotary.Table, c.vision_layers);
    for (tables, 0..) |*table, i| table.* = try rotary.trunkTable(rt, try trunk.item("blocks", i), image_block.fullGrid(c, i));
    try rt.gpu.beginBatch();
    errdefer if (rt.gpu.batch != null) rt.gpu.endBatch() catch {};
    const shape = [_]usize{ 1, side, side, width };
    const patches = try convolution.conv(trunk, "patch_embed.proj", prepared, .{ .stride = c.patch_size, .pad = 0 });
    if (patches.height != side or patches.width != side or patches.channels != width) return error.SamTrunkShape;
    try rt.observe("trunk.patch_embed", patches.tensor, &shape);
    const placed = try elementwise.add(rt, patches.tensor, position);
    var x = try norms.layerNorm(trunk, "ln_pre", placed, 1e-5, .trunk);
    try rt.observe("trunk.ln_pre", x, &shape);
    rt.rewind(m, &.{x});
    for (0..c.vision_layers) |i| {
        x = try image_block.run(rt, try trunk.item("blocks", i), x, image_block.fullGrid(c, i), tables[i]);
        if (rt.observer != null) try rt.observe(try rt.print("trunk.blocks.{d}", .{i}), x, &shape);
        rt.rewind(m, &.{x});
        if (!rt.trunk_warm and (i + 1) % 4 == 0) {
            try rt.gpu.endBatch();
            try rt.gpu.beginBatch();
        }
    }
    try rt.gpu.endBatch();
    rt.trunk_warm = true;
    return FeatureMap.of(x, side, side, width);
}

/// `encode(prepare(image))`; the prepared map is released once the trunk is done.
pub fn encodeImage(rt: *Runtime, image: prompts.Image) !FeatureMap {
    const prepared = try prepare(rt, image);
    const trunk = encode(rt, prepared) catch |err| {
        rt.release(prepared.tensor);
        return err;
    };
    rt.release(prepared.tensor);
    return trunk;
}

/// Three-level feature stack (288, 144, 72; `hidden` channels) of one level set.
pub fn levels(rt: *Runtime, trunk: FeatureMap, set: LevelSet) ![3]FeatureMap {
    const c = rt.config;
    const side = c.featureSide();
    if (trunk.height != side or trunk.width != side or trunk.channels != c.vision_width or
        trunk.tensor.count != side * side * c.vision_width) return error.SamTrunkShape;
    const root = try Scope.root(rt, c.roots.levels).sub(c.levelRoot(set));
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const opened = rt.gpu.batch == null;
    if (opened) try rt.gpu.beginBatch();
    errdefer if (opened and rt.gpu.batch != null) rt.gpu.endBatch() catch {};
    var out: [3]FeatureMap = undefined;
    var keep: [4]Tensor = undefined;
    keep[0] = trunk.tensor;
    for (level_plans, 0..) |plan, i| {
        const s = try root.sub(try rt.print("{d}", .{i}));
        var u = trunk;
        for (plan.upsamples, 0..) |leaf, step| {
            if (step > 0) u = FeatureMap.of(try elementwise.gelu(rt, u.tensor), u.height, u.width, u.channels);
            u = try convolution.upsample2(s, leaf, u);
        }
        const v = try convolution.conv(s, "conv_1x1", u, .{});
        const level = try convolution.conv(s, "conv_3x3", v, .{ .pad = 1 });
        const expected = levelSide(side, i);
        if (level.height != expected or level.width != expected or level.channels != c.hidden) return error.SamLevelShape;
        out[i] = level;
        keep[i + 1] = level.tensor;
        rt.rewind(m, keep[0 .. i + 2]);
    }
    if (opened) try rt.gpu.endBatch();
    return out;
}

const LevelPlan = struct { upsamples: []const []const u8 };

/// Transposed 2 × 2 convolutions per level, GELU between consecutive ones; then `conv_1x1` and `conv_3x3`.
const level_plans = [3]LevelPlan{
    .{ .upsamples = &.{ "dconv_2x2_0", "dconv_2x2_1" } },
    .{ .upsamples = &.{"dconv_2x2"} },
    .{ .upsamples = &.{} },
};

fn levelSide(grid: usize, level: usize) usize {
    return grid << @intCast(level_plans[level].upsamples.len);
}

/// One byte → [−1, 1]: an `f32` division by 127.5, then a subtraction of 1.
pub fn normalizeByte(value: u8) f32 {
    const v: f32 = @floatFromInt(value);
    return v / 127.5 - 1;
}

/// Channel-last normalisation of the visible 3·w bytes of each row; row padding is never read.
pub fn normalize(out: []f32, rgb: []const u8, w: usize, h: usize, stride: usize) void {
    for (0..h) |y| for (0..w * 3) |i| {
        out[y * w * 3 + i] = normalizeByte(rgb[y * stride + i]);
    };
}

/// Row of the stored `[1 + tile², width]` position table used by grid cell (y, x).
pub fn positionRow(y: usize, x: usize, tile: usize) usize {
    return 1 + tile * (y % tile) + (x % tile);
}

// Built on the host with no batch open: the stored tile block repeated over the grid.
fn positionTable(rt: *Runtime, trunk: Scope) !Tensor {
    if (rt.constant(position_constant)) |table| return table;
    const c = rt.config;
    const side = c.featureSide();
    const tile = c.tile_side;
    const width = c.vision_width;
    const stored = try trunk.param("pos_embed");
    if (stored.count != (1 + tile * tile) * width) return error.SamTrunkShape;
    const source = stored.slice(f32, stored.count);
    const host = try rt.hostTensor(side * side * width);
    const values = host.slice(f32, host.count);
    for (0..side) |y| for (0..side) |x| {
        const row = positionRow(y, x, tile);
        @memcpy(values[(y * side + x) * width ..][0..width], source[row * width ..][0..width]);
    };
    return rt.keepConstant(position_constant, host);
}

const testing = std.testing;

test "SAM image normalisation maps bytes by one division and one subtraction" {
    try testing.expectEqual(@as(f32, -1), normalizeByte(0));
    try testing.expectEqual(@as(f32, 1), normalizeByte(255));
    const expected: f32 = @as(f32, 128) / @as(f32, 127.5) - @as(f32, 1);
    try testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(normalizeByte(128))));
}

test "SAM image normalisation never reads row padding" {
    var rgb: [14]u8 = undefined;
    for (&rgb, 0..) |*byte, i| byte.* = @intCast(i * 17);
    var first: [12]f32 = undefined;
    normalize(&first, &rgb, 2, 2, 8);
    for (0..6) |i| try testing.expectEqual(normalizeByte(rgb[i]), first[i]);
    for (0..6) |i| try testing.expectEqual(normalizeByte(rgb[8 + i]), first[6 + i]);
    rgb[6] = 0xff;
    rgb[7] = 0x00;
    var second: [12]f32 = undefined;
    normalize(&second, &rgb, 2, 2, 8);
    try testing.expectEqualSlices(f32, &first, &second);
}

test "SAM absolute positions tile the stored 24 × 24 block over the grid" {
    try testing.expectEqual(@as(usize, 1), positionRow(0, 0, 24));
    try testing.expectEqual(@as(usize, 1), positionRow(0, 24, 24));
    try testing.expectEqual(@as(usize, 1), positionRow(24, 48, 24));
    try testing.expectEqual(@as(usize, 576), positionRow(71, 71, 24));
    try testing.expectEqual(@as(usize, 553), positionRow(23, 0, 24));
    var uses: [577]usize = @splat(0);
    for (0..72) |y| for (0..72) |x| {
        uses[positionRow(y, x, 24)] += 1;
    };
    try testing.expectEqual(@as(usize, 0), uses[0]);
    for (uses[1..]) |count| try testing.expectEqual(@as(usize, 9), count);
}

test "SAM level sets keep 288, 144 and 72 sides with checkpoint channel counts" {
    try testing.expectEqual(@as(usize, 288), levelSide(72, 0));
    try testing.expectEqual(@as(usize, 144), levelSide(72, 1));
    try testing.expectEqual(@as(usize, 72), levelSide(72, 2));
    // Weight shapes of each level in order: transposed [in, out, 2, 2], then conv_1x1 and conv_3x3 [out, in, k, k].
    const Shape = struct { in: usize, out: usize };
    const transposed = [3][]const Shape{
        &.{ .{ .in = 1024, .out = 512 }, .{ .in = 512, .out = 256 } },
        &.{.{ .in = 1024, .out = 512 }},
        &.{},
    };
    const one_by_one = [3]Shape{ .{ .in = 256, .out = 256 }, .{ .in = 512, .out = 256 }, .{ .in = 1024, .out = 256 } };
    for (level_plans, 0..) |plan, level| {
        try testing.expectEqual(plan.upsamples.len, transposed[level].len);
        var channels: usize = 1024;
        for (transposed[level]) |step| {
            try testing.expectEqual(channels, step.in);
            channels = step.out;
        }
        try testing.expectEqual(channels, one_by_one[level].in);
        try testing.expectEqual(@as(usize, 256), one_by_one[level].out);
    }
}
