const std = @import("std");
const gpu = @import("tensor_metal");
const runtime = @import("runtime.zig");
const variant = @import("variant.zig");
const kernel_abi = @import("kernel_abi.zig");
const elementwise = @import("elementwise.zig");
const projection = @import("projection.zig");
const norms = @import("norms.zig");
const attention = @import("attention.zig");
const spatial = @import("spatial.zig");
const rotary = @import("rotary.zig");

const Runtime = runtime.Runtime;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;
const DType = runtime.DType;
const Config = variant.Config;
const Size = gpu.mtl.Size;

const norm_eps: f32 = 1e-5;

/// One trunk block on the grid-ordered `[G², width]` FP32 stream; `table` is its prepared `rotary.trunkTable`.
pub fn run(rt: *Runtime, block: Scope, x: Tensor, full_grid: bool, table: rotary.Table) !Tensor {
    const c = rt.config;
    const side = c.featureSide();
    if (x.dtype != .f32 or x.count != side * side * c.vision_width) return error.SamTrunkShape;
    const shape = tableShape(c, full_grid);
    if (table.positions != shape.side * shape.side or table.head_width != shape.head_width) return error.SamTrunkShape;
    return switch (rt.routes.precision) {
        .fp32 => runFp32(rt, block, x, full_grid, table),
        .mixed => runMixed(rt, block, x, full_grid, table),
    };
}

/// True for the last block of every quarter of the stack.
pub fn fullGrid(config: Config, index: usize) bool {
    const quarter = config.vision_layers / 4;
    if (quarter == 0) return false;
    return (index + 1) % quarter == 0;
}

/// Head width of the trunk attention, side and position scale of a block's rotary table.
pub fn tableShape(config: Config, full_grid: bool) struct { side: usize, head_width: usize, scale: f32 } {
    const grid = config.featureSide();
    const tile = config.tile_side;
    const scale: f32 = if (full_grid) @as(f32, @floatFromInt(tile)) / @as(f32, @floatFromInt(grid)) else 1;
    return .{ .side = if (full_grid) grid else tile, .head_width = config.vision_width / config.vision_heads, .scale = scale };
}

// Planes (route N-planes, route P-planes) go only to projections that take them; other steps keep FP32 tensors.
fn runFp32(rt: *Runtime, block: Scope, x: Tensor, full_grid: bool, table: rotary.Table) !Tensor {
    const width = rt.config.vision_width;
    const tiled = !full_grid;
    const m = rt.mark();
    const qkv_planes = try projection.takesPlanes(block, "attn.qkv", .qkv);
    const qkv_wide = !qkv_planes and try projection.takesWide(block, "attn.qkv");
    const z = if (qkv_planes) blk: {
        const n1 = try norms.trunkPlanes(block, "norm1", x, norm_eps, true, tiled) orelse
            try norms.layerNorm(block, "norm1", x, norm_eps, .trunk);
        try observeInput(block, "attn.qkv", n1, width);
        break :blk try projection.trunkPlanes(block, "attn.qkv", n1, .qkv, null, false);
    } else if (qkv_wide) blk: {
        if (try norms.trunkWide(block, "norm1", x, norm_eps, tiled)) |n1| {
            try observeInput(block, "attn.qkv", n1.tensor, width);
            break :blk (try projection.trunkWide(block, "attn.qkv", n1.tensor, .qkv, n1.flag, null, false, false)).tensor;
        }
        var n1 = try norms.layerNorm(block, "norm1", x, norm_eps, .trunk);
        if (tiled) n1 = try spatial.tiles(rt, n1, width, .into_tiles);
        try observeInput(block, "attn.qkv", n1, width);
        break :blk try projection.trunkFp32(block, "attn.qkv", n1, .qkv);
    } else blk: {
        var n1 = try norms.layerNorm(block, "norm1", x, norm_eps, .trunk);
        if (tiled) n1 = try spatial.tiles(rt, n1, width, .into_tiles);
        try observeInput(block, "attn.qkv", n1, width);
        break :blk try projection.trunkFp32(block, "attn.qkv", n1, .qkv);
    };
    const proj_planes = try projection.takesPlanes(block, "attn.proj", .attention_out);
    const proj_wide = !proj_planes and try projection.takesWide(block, "attn.proj");
    const x1 = if (proj_wide and rt.routes.exactDevice() and rt.config.vision_width / rt.config.vision_heads == 64) blk: {
        const flag_count = try attention.trunkFlagCount(rt, full_grid);
        const split = try rotary.splitRotateFlag(rt, z, table, flag_count);
        const attended = try attention.trunkPrechecked(rt, split.q, split.k, split.v, full_grid, split.flags);
        try observeInput(block, "attn.proj", attended.tensor, width);
        break :blk (try projection.trunkWide(block, "attn.proj", attended.tensor, .attention_out, attended.wide_flag, x, tiled, false)).tensor;
    } else blk: {
        const q, const k, const v = try rotary.splitRotate(rt, z, table, .f32);
        const o = try attention.trunk(rt, q, k, v, full_grid);
        try observeInput(block, "attn.proj", o, width);
        break :blk if (proj_planes)
            try projection.trunkPlanes(block, "attn.proj", o, .attention_out, x, tiled)
        else if (proj_wide)
            (try projection.trunkWide(block, "attn.proj", o, .attention_out, null, x, tiled, false)).tensor
        else
            try attentionOut(rt, block, o, x, tiled);
    };
    rt.rewind(m, &.{x1});
    const fc2_planes = try projection.takesPlanes(block, "mlp.fc2", .contract);
    const fc1_planes = fc2_planes and try projection.takesPlanes(block, "mlp.fc1", .expand);
    if (!fc2_planes and try projection.takesWide(block, "mlp.fc1") and try projection.takesWide(block, "mlp.fc2")) {
        if (try norms.trunkWide(block, "norm2", x1, norm_eps, false)) |n2| {
            try observeInput(block, "mlp.fc1", n2.tensor, width);
            const e = try projection.trunkWide(block, "mlp.fc1", n2.tensor, .expand, n2.flag, null, false, true);
            try observeInput(block, "mlp.fc2", e.tensor, rt.config.vision_mlp);
            return (try projection.trunkWide(block, "mlp.fc2", e.tensor, .contract, e.flag, x1, false, false)).tensor;
        }
    }
    const n2 = try norms.trunkPlanes(block, "norm2", x1, norm_eps, fc1_planes, false) orelse
        try norms.layerNorm(block, "norm2", x1, norm_eps, .trunk);
    try observeInput(block, "mlp.fc1", n2, width);
    const e = if (fc1_planes)
        try projection.trunkPlanes(block, "mlp.fc1", n2, .expand, null, false)
    else
        try projection.trunkFp32(block, "mlp.fc1", n2, .expand);
    if (fc2_planes) return projection.trunkPlanes(block, "mlp.fc2", e, .contract, x1, false);
    try observeInput(block, "mlp.fc2", e, rt.config.vision_mlp);
    const contracted = try projection.trunkFp32(block, "mlp.fc2", e, .contract);
    return elementwise.add(rt, x1, contracted);
}

// Under an observer, records an FP32 trunk projection's input as `<leaf key>.input`.
fn observeInput(block: Scope, leaf: []const u8, x: Tensor, width: usize) !void {
    const rt = block.rt;
    if (rt.observer == null or x.dtype != .f32) return;
    try rt.observe(try rt.print("{s}.{s}.input", .{ block.path, leaf }), x, &.{ x.count / width, width });
}

fn attentionOut(rt: *Runtime, block: Scope, o: Tensor, x: Tensor, tiled: bool) !Tensor {
    var p = try projection.trunkFp32(block, "attn.proj", o, .attention_out);
    if (tiled) p = try spatial.tiles(rt, p, rt.config.vision_width, .from_tiles);
    return elementwise.add(rt, x, p);
}

fn runMixed(rt: *Runtime, block: Scope, x: Tensor, full_grid: bool, table: rotary.Table) !Tensor {
    const c = rt.config;
    const rows = c.featureSide() * c.featureSide();
    const width = c.vision_width;
    const m = rt.mark();
    const n1 = try tileNorm(rt, try block.sub("norm1"), x, !full_grid);
    const z = try tileProject(rt, kernel_abi.tile_project, try weights(block, "attn.qkv"), n1, null, rows, 3 * width, width, false);
    const q, const k, const v = try rotary.splitRotate(rt, z, table, if (full_grid) .f16 else .f32);
    var o = try attention.trunk(rt, q, k, v, full_grid);
    if (o.dtype == .f32) o = try elementwise.cast(rt, o, .f16);
    const attn_out = try weights(block, "attn.proj");
    const x1 = try tileProject(rt, kernel_abi.tile_project_residual, attn_out, o, x, rows, width, width, !full_grid);
    rt.rewind(m, &.{x1});
    const n2 = try tileNorm(rt, try block.sub("norm2"), x1, false);
    const expand = try weights(block, "mlp.fc1");
    const e = try tileProject(rt, kernel_abi.tile_project_gelu, expand, n2, null, rows, c.vision_mlp, width, false);
    const contract = try weights(block, "mlp.fc2");
    return tileProject(rt, kernel_abi.tile_project_residual, contract, e, x1, rows, width, c.vision_mlp, false);
}

const Weights = struct { weight: Tensor, bias: Tensor };

fn weights(block: Scope, leaf: []const u8) !Weights {
    const s = try block.sub(leaf);
    return .{ .weight = try s.rt.paramAs(try s.key("weight"), .f16), .bias = try s.param("bias") };
}

/// Layer norm of FP32 rows into FP16; with `gather`, output row r is the norm of grid row `gridRow(r)` (tile order).
fn tileNorm(rt: *Runtime, s: Scope, x: Tensor, gather: bool) !Tensor {
    const c = rt.config;
    const side = c.featureSide();
    const width = c.vision_width;
    const rows = side * side;
    const weight = try s.param("weight");
    const bias = try s.param("bias");
    if (x.dtype != .f32 or x.count != rows * width or weight.count != width or bias.count != width)
        return error.SamTileShape;
    const out = try rt.deviceTensor(x.count, .f16);
    const params = kernel_abi.TileNorm{
        .width = try narrow(width),
        .gather = @intFromBool(gather),
        .grid = try narrow(side),
        .tile = try narrow(c.tile_side),
        .eps = norm_eps,
    };
    try rt.gpu.dispatch(kernel_abi.tile_norm_f16, &.{ x, weight, bias, out }, params, Size.of(rows * 256, 1, 1), Size.of(256, 1, 1));
    return out;
}

/// Checks the tile-kernel preconditions; returns the output dtype of `name`.
pub fn tileCheck(name: []const u8, x: TileOperand, w: TileOperand, bias: TileOperand, residual: ?TileOperand, rows: usize, cols: usize, depth: usize, scatter: bool, grid: usize) !DType {
    if (rows == 0 or rows % 64 != 0 or cols == 0 or cols % 128 != 0 or depth == 0) return error.SamTileShape;
    if (x.dtype != .f16 or x.count != rows * depth or w.dtype != .f16 or w.count != cols * depth) return error.SamTileShape;
    if (bias.dtype != .f32 or bias.count != cols) return error.SamTileShape;
    if (scatter and rows != grid * grid) return error.SamTileShape;
    const residual_kernel = std.mem.eql(u8, name, kernel_abi.tile_project_residual);
    if (residual_kernel != (residual != null)) return error.SamTileShape;
    if (residual) |r| if (r.dtype != .f32 or r.count != rows * cols) return error.SamTileShape;
    if (scatter and !residual_kernel) return error.SamTileShape;
    return if (std.mem.eql(u8, name, kernel_abi.tile_project_gelu)) .f16 else .f32;
}

pub const TileOperand = struct { dtype: DType, count: usize };

fn operand(t: Tensor) TileOperand {
    return .{ .dtype = t.dtype, .count = t.count };
}

fn tileProject(rt: *Runtime, name: []const u8, p: Weights, x: Tensor, residual: ?Tensor, rows: usize, cols: usize, depth: usize, scatter: bool) !Tensor {
    const c = rt.config;
    const grid = c.featureSide();
    const r: ?TileOperand = if (residual) |t| operand(t) else null;
    const dtype = try tileCheck(name, operand(x), operand(p.weight), operand(p.bias), r, rows, cols, depth, scatter, grid);
    const out = try rt.deviceTensor(rows * cols, dtype);
    const params = kernel_abi.TileProject{
        .rows = try narrow(rows),
        .cols = try narrow(cols),
        .depth = try narrow(depth),
        .scatter = @intFromBool(scatter),
        .grid = try narrow(grid),
        .tile = try narrow(c.tile_side),
    };
    const threads = Size.of(rows / 64 * 32, cols / 128 * 4, 2);
    const group = Size.of(32, 4, 2);
    if (residual) |res|
        try rt.gpu.dispatch(name, &.{ x, p.weight, p.bias, out, res }, params, threads, group)
    else
        try rt.gpu.dispatch(name, &.{ x, p.weight, p.bias, out }, params, threads, group);
    return out;
}

fn narrow(value: usize) !u32 {
    return std.math.cast(u32, value) orelse return error.SamTileShape;
}

const testing = std.testing;

test "SAM trunk full-grid blocks close every quarter of the stack" {
    var c = Config.of(.sam31);
    for (0..32) |i| try testing.expectEqual(i == 7 or i == 15 or i == 23 or i == 31, fullGrid(c, i));
    c.vision_layers = 8;
    for (0..8) |i| try testing.expectEqual(i % 2 == 1, fullGrid(c, i));
    c.vision_layers = 2;
    try testing.expect(!fullGrid(c, 1));
}

test "SAM trunk rotary tables use the grid or the tile side" {
    const c = Config.of(.sam31);
    const full = tableShape(c, true);
    try testing.expectEqual(@as(usize, 72), full.side);
    try testing.expectEqual(@as(usize, 64), full.head_width);
    try testing.expectEqual(@as(u32, 0x3EAAAAAB), @as(u32, @bitCast(full.scale)));
    const tiled = tableShape(c, false);
    try testing.expectEqual(@as(usize, 24), tiled.side);
    try testing.expectEqual(@as(f32, 1), tiled.scale);
}

test "SAM tile kernels refuse shapes outside their preconditions" {
    const half = struct {
        fn of(count: usize) TileOperand {
            return .{ .dtype = .f16, .count = count };
        }
    }.of;
    const full = struct {
        fn of(count: usize) TileOperand {
            return .{ .dtype = .f32, .count = count };
        }
    }.of;
    const rows: usize = 5184;
    try testing.expectEqual(DType.f32, try tileCheck(kernel_abi.tile_project, half(rows * 1024), half(3072 * 1024), full(3072), null, rows, 3072, 1024, false, 72));
    try testing.expectEqual(DType.f16, try tileCheck(kernel_abi.tile_project_gelu, half(rows * 1024), half(4736 * 1024), full(4736), null, rows, 4736, 1024, false, 72));
    const residual = kernel_abi.tile_project_residual;
    try testing.expectEqual(DType.f32, try tileCheck(residual, half(rows * 1024), half(1024 * 1024), full(1024), full(rows * 1024), rows, 1024, 1024, true, 72));
    try testing.expectError(error.SamTileShape, tileCheck(residual, half(rows * 1024), half(1024 * 1024), full(1024), half(rows * 1024), rows, 1024, 1024, false, 72));
    try testing.expectError(error.SamTileShape, tileCheck(residual, half(64 * 1024), half(1024 * 1024), full(1024), full(64 * 1024), 64, 1024, 1024, true, 72));
    try testing.expectError(error.SamTileShape, tileCheck(kernel_abi.tile_project, half(5183 * 1024), half(1024 * 1024), full(1024), null, 5183, 1024, 1024, false, 72));
    try testing.expectError(error.SamTileShape, tileCheck(kernel_abi.tile_project, half(rows * 1024), half(1000 * 1024), full(1000), null, rows, 1000, 1024, false, 72));
    try testing.expectError(error.SamTileShape, tileCheck(kernel_abi.tile_project, full(rows * 1024), half(1024 * 1024), full(1024), null, rows, 1024, 1024, false, 72));
    try testing.expectError(error.SamTileShape, tileCheck(kernel_abi.tile_project, half(rows * 1024), half(1024 * 1023), full(1024), null, rows, 1024, 1024, false, 72));
}
