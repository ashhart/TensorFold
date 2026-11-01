//! Two stored affine projections of the same geometry in one dispatch, without packing or changing their row sums.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const row = @import("../../core/row_projection.zig");
pub const names = [_][:0]const u8{ "lfm3_projection_dual_q4", "lfm3_projection_dual_q6", "lfm3_projection_dual_q8" };
pub fn source(a: std.mem.Allocator) ![]u8 {
    return std.mem.concat(a, u8, &.{ sources.core_row_projection, "\n", sources.lfm3_projection_dual });
}
pub const Pipelines = struct {
    plain: [3]mtl.Pipeline,
    pub fn load(device: mtl.Device) !Pipelines {
        const a = std.heap.c_allocator;
        const text = try source(a);
        defer a.free(text);
        const library = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer library.deinit();
        var out: Pipelines = undefined;
        var made: usize = 0;
        errdefer for (out.plain[0..made]) |p| p.deinit();
        for (names, 0..) |name, i| {
            out.plain[i] = try mtl.Pipeline.init(device, library, name, false);
            made += 1;
        }
        return out;
    }
    pub fn deinit(p: *Pipelines) void {
        for (p.plain) |pipeline| pipeline.deinit();
    }
};
pub fn compatible(a: row.Weights, b: row.Weights) bool {
    return a.n == b.n and a.k == b.k and a.group == b.group and a.bits == b.bits and a.sum == .f32 and b.sum == .f32 and
        (a.group == 32 or a.group == 64) and (a.bits == 4 or a.bits == 6 or a.bits == 8);
}
pub fn encode(e: mtl.ComputeEncoder, p: *const Pipelines, a: row.Weights, b: row.Weights, x: mtl.Buffer, x_off: usize, y: mtl.Buffer, y_off: usize, z: mtl.Buffer, z_off: usize, rows: usize) !void {
    try a.validate();
    try b.validate();
    if (!compatible(a, b) or rows == 0 or rows > std.math.maxInt(i32) or a.n > std.math.maxInt(i32) or a.k > std.math.maxInt(i32))
        return error.UnsupportedLfmDualProjection;
    e.setPipeline(p.plain[
        switch (a.bits) {
            4 => @as(usize, 0),
            6 => 1,
            8 => 2,
            else => unreachable,
        }
    ]);
    e.setBuffer(x, x_off, 0);
    e.setBuffer(a.w, a.w_off, 1);
    e.setBuffer(a.scales, a.s_off, 2);
    e.setBuffer(a.biases, a.b_off, 3);
    const dims = [_]i32{ @intCast(a.k), @intCast(a.n), @intCast(a.group), @intCast(rows), 1, 1 };
    e.setBytes(std.mem.asBytes(&dims), 4);
    e.setBuffer(y, y_off, 5);
    e.setBuffer(b.w, b.w_off, 6);
    e.setBuffer(b.scales, b.s_off, 7);
    e.setBuffer(b.biases, b.b_off, 8);
    e.setBuffer(z, z_off, 9);
    e.dispatchGroups(.of((rows + 1) / 2, a.n / 8, 2), .of(64, 1, 1));
}
