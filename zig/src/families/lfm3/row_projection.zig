//! Independent LFM3 row pairs dispatch the unchanged shared row projection arithmetic.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const shared = @import("../../core/row_projection.zig");
pub const bit_widths = [_]u8{ 4, 6, 8 };
pub const names = [_][:0]const u8{ "lfm3_row_pairs_q4", "lfm3_row_pairs_q6", "lfm3_row_pairs_q8" };

pub fn source(a: std.mem.Allocator) ![]u8 {
    return std.mem.concat(a, u8, &.{ sources.core_row_projection, "\n", sources.lfm3_projection });
}
pub const Pipelines = struct {
    plain: [bit_widths.len]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const a = std.heap.c_allocator;
        const text = try source(a);
        defer a.free(text);
        const library = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer library.deinit();
        var out: Pipelines = undefined;
        var loaded: usize = 0;
        errdefer for (out.plain[0..loaded]) |pipeline| pipeline.deinit();
        for (names, 0..) |name, i| {
            out.plain[i] = try mtl.Pipeline.init(device, library, name, false);
            loaded += 1;
        }
        return out;
    }
    pub fn deinit(p: *Pipelines) void {
        for (p.plain) |pipeline| pipeline.deinit();
    }
};
pub const Call = struct { pipeline: mtl.Pipeline, dims: [6]i32, grid: mtl.Size };

pub fn call(p: *const Pipelines, weights: shared.Weights, rows: usize) !Call {
    try weights.validate();
    if (weights.sum != .f32 or (weights.group != 32 and weights.group != 64)) return error.UnsupportedLfmRowPairs;
    const index = std.mem.indexOfScalar(u8, &bit_widths, weights.bits) orelse return error.UnsupportedLfmRowPairBits;
    if (rows == 0 or rows > std.math.maxInt(i32) or weights.n > std.math.maxInt(i32) or weights.k > std.math.maxInt(i32)) return error.UnsupportedLfmRowPairShape;
    return .{ .pipeline = p.plain[index], .dims = .{ @intCast(weights.k), @intCast(weights.n), @intCast(weights.group), @intCast(rows), 1, 1 }, .grid = .of((rows + 1) / 2, weights.n / 8, 1) };
}

pub fn encode(e: mtl.ComputeEncoder, p: *const Pipelines, weights: shared.Weights, x: mtl.Buffer, x_off: usize, y: mtl.Buffer, y_off: usize, rows: usize) !void {
    const dispatch = try call(p, weights, rows);
    e.setPipeline(dispatch.pipeline);
    e.setBuffer(x, x_off, 0);
    e.setBuffer(weights.w, weights.w_off, 1);
    e.setBuffer(weights.scales, weights.s_off, 2);
    e.setBuffer(weights.biases, weights.b_off, 3);
    e.setBytes(std.mem.asBytes(&dispatch.dims), 4);
    e.setBuffer(y, y_off, 5);
    e.dispatchGroups(dispatch.grid, .of(64, 1, 1));
}
