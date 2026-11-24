const std = @import("std");
const gpu = @import("tensor_metal");
const runtime = @import("runtime.zig");
const variant = @import("variant.zig");
const kernel_abi = @import("kernel_abi.zig");
const elementwise = @import("elementwise.zig");
const projection = @import("projection.zig");
const attention_split = @import("attention_split.zig");
const attention_split32 = @import("attention_split32.zig");
const attention_device = @import("attention_device.zig");

const Runtime = runtime.Runtime;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;
const Routes = runtime.Routes;
const ops = gpu.ops;
const Size = gpu.mtl.Size;

pub const Options = struct { bias: ?Tensor = null, causal: bool = false };

// Rows and keys of one 72 × 72 grid: the device dim-32 route and the FP16 flash promotion both key on it.
const grid_tokens = 5184;

const Route = enum { split, split32, flash_f32, flash_f16, padded, core_causal, core_cross, refused, rows_kernel };

const Case = struct {
    routes: Routes,
    version: variant.Version,
    bias: bool,
    causal: bool,
    d: usize,
    rows: usize,
    keys: usize,
};

fn routeFor(c: Case) Route {
    const plain = !c.bias and !c.causal;
    const device = c.routes.exactDevice();
    if (device and plain and c.d == 256) return .split;
    if (device and c.version == .sam31 and plain and c.d == 32 and c.rows == grid_tokens and c.keys >= c.rows) return .split32;
    if (c.routes.precision == .mixed and plain and c.d == 32)
        return if (c.rows < grid_tokens or c.keys < grid_tokens) .flash_f32 else .flash_f16;
    if (plain and c.d == 16) return .padded;
    if (!c.bias and (c.d == 32 or c.d == 64 or c.d == 128 or c.d == 256)) return if (c.causal) .core_causal else .core_cross;
    if (c.causal or c.d > 256) return .refused;
    return .rows_kernel;
}

fn scaleFor(d: usize) f32 {
    return 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
}

fn narrow(value: usize) error{SamAttentionShape}!u32 {
    return std.math.cast(u32, value) orelse error.SamAttentionShape;
}

/// Scaled dot-product attention of FP32 q `[rows, width]` over k, v `[keys, width]` in `heads` heads.
pub fn attend(rt: *Runtime, q: Tensor, k: Tensor, v: Tensor, heads: usize, width: usize, options: Options) !Tensor {
    if (heads == 0 or width == 0 or width % heads != 0 or q.count % width != 0 or k.count != v.count or
        k.count % width != 0 or q.dtype != .f32 or k.dtype != .f32 or v.dtype != .f32) return error.SamAttentionShape;
    const d = width / heads;
    const rows = q.count / width;
    const keys = k.count / width;
    if (rows == 0 or keys == 0) return error.SamAttentionShape;
    if (options.bias) |bias| if (bias.dtype != .f32 or bias.count != heads * rows * keys) return error.SamAttentionShape;
    const g = rt.gpu;
    const route = routeFor(.{
        .routes = rt.routes,
        .version = rt.config.version,
        .bias = options.bias != null,
        .causal = options.causal,
        .d = d,
        .rows = rows,
        .keys = keys,
    });
    switch (route) {
        .split => return rt.hold(try attention_split.run(g, q, k, v, .{ 1, heads, rows, keys, 256 })),
        .split32 => return rt.hold(try attention_split32.run(g, q, k, v, .{ 1, heads, rows, keys, 32 })),
        .flash_f32 => return rt.hold(try flash(g, q, k, v, rows, keys, heads, 1)),
        .flash_f16 => {
            const start = rt.mark();
            const qh = try elementwise.cast(rt, q, .f16);
            const kh = try elementwise.cast(rt, k, .f16);
            const vh = try elementwise.cast(rt, v, .f16);
            const attended = try rt.hold(try flash(g, qh, kh, vh, rows, keys, heads, 1));
            const out = try elementwise.cast(rt, attended, .f32);
            rt.rewind(start, &.{out});
            return out;
        },
        .padded => {
            const qp = try rt.hold(try ops.headsCopy(g, q, rows, heads, 16, 32, width, 0));
            const kp = try rt.hold(try ops.headsCopy(g, k, keys, heads, 16, 32, width, 0));
            const vp = try rt.hold(try ops.headsCopy(g, v, keys, heads, 16, 32, width, 0));
            const attended = try rt.hold(try ops.attentionCross(g, qp, kp, vp, rows, keys, heads, 0.25));
            return rt.hold(try ops.headsCopy(g, attended, rows, heads, 32, 16, heads * 32, 0));
        },
        .core_causal => return rt.hold(try ops.attention(g, q, k, v, rows, heads, heads, true, scaleFor(d))),
        .core_cross => return rt.hold(try ops.attentionCross(g, q, k, v, rows, keys, heads, scaleFor(d))),
        .refused => return error.SamAttentionShape,
        .rows_kernel => return attendRows(rt, q, k, v, rows, keys, heads, d, options.bias),
    }
}

fn attendRows(rt: *Runtime, q: Tensor, k: Tensor, v: Tensor, rows: usize, keys: usize, heads: usize, d: usize, bias: ?Tensor) !Tensor {
    const constants = kernel_abi.RowsAttention{
        .rows = try narrow(rows),
        .keys = try narrow(keys),
        .heads = try narrow(heads),
        .head_width = try narrow(d),
        .has_bias = @intFromBool(bias != null),
    };
    const out = try rt.deviceTensor(q.count, .f32);
    try rt.gpu.dispatch(kernel_abi.attention_rows, &.{ q, k, v, bias orelse q, out }, constants, Size.of(rows * 32, heads, 1), Size.of(32, 1, 1));
    return out;
}

const TrunkRoute = enum { split32_full, device_tiles, flash_full, flash_tiles, core_full, core_tiles };

fn trunkRouteFor(routes: Routes, full_grid: bool) TrunkRoute {
    if (routes.exactDevice()) return if (full_grid) .split32_full else .device_tiles;
    if (routes.precision == .mixed) return if (full_grid) .flash_full else .flash_tiles;
    return if (full_grid) .core_full else .core_tiles;
}

pub const TrunkOut = struct { tensor: Tensor, wide_flag: Tensor };

/// Flag words needed for a prechecked trunk attention call (domain + per-tile rescale + 1 output wide flag).
pub fn trunkFlagCount(rt: *Runtime, full_grid: bool) !usize {
    const config = rt.config;
    const heads = config.vision_heads;
    if (heads == 0 or config.vision_width % heads != 0 or config.tile_side == 0) return error.SamAttentionShape;
    const side = config.featureSide();
    if (side % config.tile_side != 0) return error.SamAttentionShape;
    const tokens = side * side;
    const tile = config.tile_side * config.tile_side;
    const tiles = tokens / tile;
    const rows = if (full_grid) tokens else tile;
    const batch = if (full_grid) @as(usize, 1) else tiles;
    const nq = (rows + 63) / 64;
    return 2 + nq * heads * batch;
}

/// Prechecked FP32 device trunk attention: skips clear/domain and writes the output wide flag at flags[count - 1].
pub fn trunkPrechecked(rt: *Runtime, q: Tensor, k: Tensor, v: Tensor, full_grid: bool, flags: Tensor) !TrunkOut {
    const config = rt.config;
    const heads = config.vision_heads;
    if (heads == 0 or config.vision_width % heads != 0 or config.tile_side == 0) return error.SamAttentionShape;
    const d = config.vision_width / heads;
    const side = config.featureSide();
    if (side % config.tile_side != 0) return error.SamAttentionShape;
    const tokens = side * side;
    const tile = config.tile_side * config.tile_side;
    const tiles = tokens / tile;
    if (q.count != tokens * config.vision_width or k.count != q.count or v.count != q.count) return error.SamAttentionShape;
    const wide_idx = (try trunkFlagCount(rt, full_grid)) - 1;
    if (flags.count <= wide_idx) return error.SamAttentionShape;
    const out = try rt.hold(if (full_grid)
        try attention_split32.runPrechecked(rt.gpu, q, k, v, .{ 1, heads, tokens, tokens, d }, flags, wide_idx)
    else
        try attention_device.runPrechecked(rt.gpu, q, k, v, .{ tiles, heads, tile, tile, d }, flags, wide_idx));
    return .{ .tensor = out, .wide_flag = runtime.part(flags, wide_idx, 1) };
}

/// Trunk attention over the grid (one batch of all tokens) or over its tiles (one batch item per tile, tile order).
pub fn trunk(rt: *Runtime, q: Tensor, k: Tensor, v: Tensor, full_grid: bool) !Tensor {
    const config = rt.config;
    const heads = config.vision_heads;
    if (heads == 0 or config.vision_width % heads != 0 or config.tile_side == 0) return error.SamAttentionShape;
    const d = config.vision_width / heads;
    const side = config.featureSide();
    if (side % config.tile_side != 0) return error.SamAttentionShape;
    const tokens = side * side;
    const tile = config.tile_side * config.tile_side;
    const tiles = tokens / tile;
    if (q.count != tokens * config.vision_width or k.count != q.count or v.count != q.count) return error.SamAttentionShape;
    const g = rt.gpu;
    const scale = scaleFor(d);
    return rt.hold(try switch (trunkRouteFor(rt.routes, full_grid)) {
        .split32_full => attention_split32.run(g, q, k, v, .{ 1, heads, tokens, tokens, d }),
        .device_tiles => attention_device.run(g, q, k, v, .{ tiles, heads, tile, tile, d }),
        .flash_full => flash(g, q, k, v, tokens, tokens, heads, 1),
        .flash_tiles => flash(g, q, k, v, tile, tile, heads, tiles),
        .core_full => ops.attention(g, q, k, v, tokens, heads, heads, false, scale),
        .core_tiles => ops.attentionBatched(g, q, k, v, tile, heads, heads, false, scale, tiles),
    });
}

/// Reserves the single-head attention workspace for (rows, keys); call with no batch open.
pub fn reserve(rt: *Runtime, rows: usize, keys: usize) !void {
    std.debug.assert(rt.gpu.batch == null);
    if (rt.routes.exactDevice()) {
        _ = try rt.gpu.splitAttentionWorkspace(rows, keys, 1, 1, @import("sam3_kernels").attention_split.chunk_blocks);
    } else try rt.gpu.reserveAttention(rows, keys, 1, 1);
}

/// Flash attention tiles on q `[batch, rows, heads, d]`, k, v `[batch, keys, heads, d]`; the caller owns the result.
pub fn flash(g: *gpu.Gpu, q: Tensor, k: Tensor, v: Tensor, rows: usize, keys: usize, heads: usize, batch: usize) !Tensor {
    const planes = std.math.mul(usize, std.math.mul(usize, rows, heads) catch return error.SamAttentionShape, batch) catch
        return error.SamAttentionShape;
    if (planes == 0 or keys == 0 or q.count % planes != 0) return error.SamAttentionShape;
    const d = q.count / planes;
    const key_count = std.math.mul(usize, std.math.mul(usize, batch, keys) catch return error.SamAttentionShape, heads * d) catch
        return error.SamAttentionShape;
    if (k.dtype != q.dtype or v.dtype != q.dtype or k.count != key_count or v.count != key_count) return error.SamAttentionShape;
    const name = try kernel_abi.flash(q.dtype, d);
    const constants = kernel_abi.Flash{ .rows = try narrow(rows), .keys = try narrow(keys), .heads = try narrow(heads), .batch = try narrow(batch) };
    const out = try g.allocDevice(q.count, q.dtype);
    errdefer out.deinit();
    try g.dispatch(name, &.{ q, k, v, out }, constants, Size.of((rows + 63) / 64 * 32, heads * 4, batch), Size.of(32, 4, 1));
    return out;
}

/// Thirds of the packed projection for q, k and v, attention, then `out_proj`.
pub fn packedHeads(s: Scope, q: Tensor, k: Tensor, v: Tensor, heads: usize, options: Options) !Tensor {
    const bias = try s.info("in_proj_bias");
    if (bias.rank != 1 or bias.dim(0) % 3 != 0) return error.SamAttentionShape;
    const width = bias.dim(0) / 3;
    const qp = try projection.packedPart(s, q, .query);
    const kp = try projection.packedPart(s, k, .key);
    const vp = try projection.packedPart(s, v, .value);
    const attended = try attend(s.rt, qp, kp, vp, heads, width, options);
    return projection.affine(s, "out_proj", attended);
}

/// Separate `q_proj`, `k_proj`, `v_proj`, attention without bias, then `out_proj`.
pub fn separateHeads(s: Scope, q: Tensor, k: Tensor, v: Tensor, heads: usize) !Tensor {
    const query = try s.info("q_proj.weight");
    if (query.rank != 2) return error.SamAttentionShape;
    const qp = try projection.affine(s, "q_proj", q);
    const kp = try projection.affine(s, "k_proj", k);
    const vp = try projection.affine(s, "v_proj", v);
    const attended = try attend(s.rt, qp, kp, vp, heads, query.dim(0), .{});
    return projection.affine(s, "out_proj", attended);
}

fn case(routes: Routes, version: variant.Version, bias: bool, causal: bool, d: usize, rows: usize, keys: usize) Case {
    return .{ .routes = routes, .version = version, .bias = bias, .causal = causal, .d = d, .rows = rows, .keys = keys };
}

test "SAM attention route table, first match wins" {
    const expect = std.testing.expectEqual;
    const fp32 = Routes{ .precision = .fp32 };
    const mixed = Routes{ .precision = .mixed, .tensor_units = true };
    const device = Routes{ .precision = .fp32, .tensor_units = true };
    try expect(Route.split, routeFor(case(device, .sam31, false, false, 256, 5184, 5184)));
    try expect(Route.split, routeFor(case(device, .sam3, false, false, 256, 7, 9)));
    try expect(Route.core_cross, routeFor(case(mixed, .sam3, false, false, 256, 7, 9)));
    try expect(Route.rows_kernel, routeFor(case(device, .sam31, true, false, 256, 7, 9)));
    try expect(Route.core_causal, routeFor(case(device, .sam31, false, true, 256, 7, 7)));
    try expect(Route.split32, routeFor(case(device, .sam31, false, false, 32, 5184, 5184)));
    try expect(Route.split32, routeFor(case(device, .sam31, false, false, 32, 5184, 9000)));
    try expect(Route.core_cross, routeFor(case(fp32, .sam31, false, false, 32, 5184, 5184)));
    try expect(Route.core_cross, routeFor(case(device, .sam31, false, false, 32, 5184, 5183)));
    try expect(Route.core_cross, routeFor(case(device, .sam3, false, false, 32, 5184, 5184)));
    try expect(Route.core_cross, routeFor(case(device, .sam31, false, false, 32, 5183, 5184)));
    try expect(Route.flash_f16, routeFor(case(mixed, .sam31, false, false, 32, 5184, 5184)));
    try expect(Route.flash_f32, routeFor(case(mixed, .sam31, false, false, 32, 5183, 5184)));
    try expect(Route.flash_f32, routeFor(case(mixed, .sam31, false, false, 32, 5184, 5183)));
    try expect(Route.flash_f32, routeFor(case(mixed, .sam31, false, false, 32, 7, 9)));
    try expect(Route.padded, routeFor(case(fp32, .sam31, false, false, 16, 7, 9)));
    try expect(Route.padded, routeFor(case(mixed, .sam31, false, false, 16, 7, 9)));
    try expect(Route.refused, routeFor(case(fp32, .sam31, false, true, 16, 7, 7)));
    try expect(Route.core_causal, routeFor(case(fp32, .sam31, false, true, 64, 7, 7)));
    try expect(Route.core_causal, routeFor(case(mixed, .sam31, false, true, 32, 7, 7)));
    for ([_]usize{ 32, 64, 128, 256 }) |d| try expect(Route.core_cross, routeFor(case(fp32, .sam31, false, false, d, 7, 9)));
    try expect(Route.refused, routeFor(case(fp32, .sam31, false, false, 288, 7, 9)));
    try expect(Route.refused, routeFor(case(fp32, .sam31, true, true, 48, 7, 7)));
    try expect(Route.rows_kernel, routeFor(case(fp32, .sam31, true, false, 32, 7, 9)));
    try expect(Route.rows_kernel, routeFor(case(mixed, .sam31, true, false, 32, 5184, 5184)));
    try expect(Route.rows_kernel, routeFor(case(fp32, .sam31, false, false, 48, 7, 9)));
    try expect(Route.rows_kernel, routeFor(case(fp32, .sam31, true, false, 16, 7, 9)));
}

test "SAM trunk attention route per precision and tensor units" {
    const expect = std.testing.expectEqual;
    const device = Routes{ .precision = .fp32, .tensor_units = true };
    try expect(TrunkRoute.split32_full, trunkRouteFor(device, true));
    try expect(TrunkRoute.device_tiles, trunkRouteFor(device, false));
    try expect(TrunkRoute.flash_full, trunkRouteFor(.{ .precision = .mixed, .tensor_units = true }, true));
    try expect(TrunkRoute.flash_tiles, trunkRouteFor(.{ .precision = .mixed, .tensor_units = true }, false));
    try expect(TrunkRoute.core_full, trunkRouteFor(.{ .precision = .fp32 }, true));
    try expect(TrunkRoute.core_tiles, trunkRouteFor(.{ .precision = .fp32 }, false));
}

test "SAM attention scale is the f32 reciprocal of the f32 square root" {
    try std.testing.expectEqual(@as(f32, 0.125), scaleFor(64));
    try std.testing.expectEqual(@as(f32, 0.0625), scaleFor(256));
    try std.testing.expectEqual(@as(f32, 1.0) / @sqrt(@as(f32, 32.0)), scaleFor(32));
}
