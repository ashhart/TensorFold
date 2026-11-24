const std = @import("std");
const gpu = @import("tensor_metal");
const abi = @import("kernel_abi.zig");
const runtime = @import("runtime.zig");
const Runtime = runtime.Runtime;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;
const DType = runtime.DType;
const Size = gpu.mtl.Size;
const group = Size.of(256, 1, 1);

pub const Table = struct { tensor: Tensor, positions: usize, head_width: usize };

const stored_suffix = ".attn.freqs_cis";

fn word(value: usize) !u32 {
    return std.math.cast(u32, value) orelse error.SamRotaryShape;
}

fn tableName(rt: *Runtime, side: usize, head_width: usize, position_scale: f32) ![]const u8 {
    return rt.print("rotary.{d}.{d}.{x:0>8}", .{ side, head_width, @as(u32, @bitCast(position_scale)) });
}

fn tableCount(side: usize, head_width: usize) !usize {
    if (side == 0 or head_width == 0 or head_width % 4 != 0) return error.SamRotaryShape;
    return std.math.mul(usize, try std.math.mul(usize, side, side), head_width);
}

pub fn generate(rt: *Runtime, side: usize, head_width: usize, position_scale: f32) !Table {
    const count = try tableCount(side, head_width);
    const positions = side * side;
    const name = try tableName(rt, side, head_width, position_scale);
    if (rt.constant(name)) |tensor| return .{ .tensor = tensor, .positions = positions, .head_width = head_width };
    const fresh = try rt.newConstant(count, .f32);
    errdefer fresh.deinit();
    const params = abi.Rotary{
        .rows = 0,
        .heads = 0,
        .head_width = try word(head_width),
        .rotated_rows = 0,
        .positions = try word(positions),
        .side = try word(side),
        .scale = position_scale,
    };
    try rt.gpu.dispatch(abi.rotary_table, &.{fresh}, params, Size.of(count / 2, 1, 1), group);
    const tensor = try rt.adoptConstant(name, fresh);
    return .{ .tensor = tensor, .positions = positions, .head_width = head_width };
}

pub fn trunkTable(rt: *Runtime, block: Scope, full_grid: bool) !Table {
    std.debug.assert(rt.gpu.batch == null);
    const grid = rt.config.featureSide();
    const tile = rt.config.tile_side;
    const side = if (full_grid) grid else tile;
    const position_scale: f32 = if (full_grid) @as(f32, @floatFromInt(tile)) / @as(f32, @floatFromInt(grid)) else 1;
    const head_width = rt.config.vision_width / rt.config.vision_heads;
    const count = try tableCount(side, head_width);
    const positions = side * side;
    const name = try block.key("attn.freqs_cis");
    if (rt.constant(name)) |tensor| return .{ .tensor = tensor, .positions = positions, .head_width = head_width };
    const stored = try rt.store.rotaryPairs(name, positions, head_width) orelse
        return generate(rt, side, head_width, position_scale);
    const bytes = stored.bytes;
    if (bytes.len != count * @sizeOf(f32)) return error.SamRotaryShape;
    if (rt.findConstant(stored_suffix, bytes)) |existing|
        return .{ .tensor = try rt.shareConstant(name, existing), .positions = positions, .head_width = head_width };
    const fresh = try rt.newConstant(count, .f32);
    errdefer fresh.deinit();
    @memcpy(fresh.slice(u8, bytes.len), bytes);
    const tensor = try rt.adoptConstant(name, fresh);
    return .{ .tensor = tensor, .positions = positions, .head_width = head_width };
}

pub fn rotate(rt: *Runtime, x: Tensor, heads: usize, width: usize, table: Table, rotated_rows: usize) !Tensor {
    if (x.dtype != .f32) return error.SamRotaryType;
    if (heads == 0 or width == 0 or width % heads != 0 or width / heads != table.head_width or x.count % width != 0)
        return error.SamRotaryShape;
    const rows = x.count / width;
    if (rotated_rows > rows or table.positions == 0) return error.SamRotaryShape;
    const out = try rt.deviceTensor(x.count, .f32);
    const params = abi.Rotary{
        .rows = try word(rows),
        .heads = try word(heads),
        .head_width = try word(table.head_width),
        .rotated_rows = try word(rotated_rows),
        .positions = try word(table.positions),
        .side = 0,
        .scale = 1,
    };
    try rt.gpu.dispatch(abi.rotate, &.{ x, table.tensor, out }, params, Size.of(x.count / 4, 1, 1), group);
    return out;
}

pub const SplitFlagged = struct { q: Tensor, k: Tensor, v: Tensor, flags: Tensor };

pub fn splitRotateFlag(rt: *Runtime, qkv: Tensor, table: Table, flag_count: usize) !SplitFlagged {
    const width = rt.config.vision_width;
    if (qkv.dtype != .f32) return error.SamRotaryType;
    if (flag_count == 0 or table.head_width == 0 or table.head_width % 4 != 0 or width % table.head_width != 0 or qkv.count % (3 * width) != 0 or qkv.count == 0)
        return error.SamRotaryShape;
    const rows = qkv.count / (3 * width);
    var parts: [3]Tensor = undefined;
    for (&parts) |*item| item.* = try rt.deviceTensor(rows * width, .f32);
    const flags = try rt.deviceTensor(flag_count, .f32);
    const params = abi.RotarySplit{
        .rows = try word(rows),
        .width = try word(width),
        .head_width = try word(table.head_width),
        .positions = try word(table.positions),
    };
    try rt.gpu.dispatch("sam_attention_device_clear", &.{flags}, try word(flag_count), Size.of(flag_count, 1, 1), Size.of(128, 1, 1));
    try rt.gpu.dispatch(abi.rotate_split_f32_flag, &.{ qkv, table.tensor, parts[0], parts[1], parts[2], flags }, params, Size.of(rows * width / 4, 1, 1), group);
    return .{ .q = parts[0], .k = parts[1], .v = parts[2], .flags = flags };
}

pub fn splitRotate(rt: *Runtime, qkv: Tensor, table: Table, out: DType) ![3]Tensor {
    const width = rt.config.vision_width;
    if (qkv.dtype != .f32 or (out != .f32 and out != .f16)) return error.SamRotaryType;
    if (table.head_width == 0 or table.head_width % 4 != 0 or width % table.head_width != 0 or qkv.count % (3 * width) != 0 or qkv.count == 0)
        return error.SamRotaryShape;
    const rows = qkv.count / (3 * width);
    var parts: [3]Tensor = undefined;
    for (&parts) |*item| item.* = try rt.deviceTensor(rows * width, out);
    const params = abi.RotarySplit{
        .rows = try word(rows),
        .width = try word(width),
        .head_width = try word(table.head_width),
        .positions = try word(table.positions),
    };
    const name = if (out == .f16) abi.rotate_split_f16 else abi.rotate_split_f32;
    try rt.gpu.dispatch(name, &.{ qkv, table.tensor, parts[0], parts[1], parts[2] }, params, Size.of(rows * width / 4, 1, 1), group);
    return parts;
}

const Coordinate = struct { position: usize, pair: usize, axis: enum { x, y }, value: usize };

fn tableCoordinate(index: usize, side: usize, head_width: usize) Coordinate {
    const pairs = head_width / 2;
    const pair = index % pairs;
    const position = index / pairs;
    if (pair < head_width / 4) return .{ .position = position, .pair = pair, .axis = .x, .value = position % side };
    return .{ .position = position, .pair = pair, .axis = .y, .value = position / side };
}

test "SAM rotary tables use the x angle for the first quarter of pairs and the y angle above" {
    const first = tableCoordinate(75 * 32, 72, 64);
    try std.testing.expectEqual(@as(usize, 75), first.position);
    try std.testing.expect(first.axis == .x);
    try std.testing.expectEqual(@as(usize, 3), first.value);
    const upper = tableCoordinate(75 * 32 + 16, 72, 64);
    try std.testing.expectEqual(@as(usize, 16), upper.pair);
    try std.testing.expect(upper.axis == .y);
    try std.testing.expectEqual(@as(usize, 1), upper.value);
    try std.testing.expectEqual(@as(usize, 72 * 72 * 64), try tableCount(72, 64));
    try std.testing.expectError(error.SamRotaryShape, tableCount(72, 30));
    try std.testing.expectEqual(@as(u32, 0x3EAAAAAB), @as(u32, @bitCast(@as(f32, 24) / @as(f32, 72))));
}
