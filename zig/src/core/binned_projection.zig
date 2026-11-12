//! Affine BF16 rows through exact exponent-bin/nibble group dots and compensated FP32 affine accumulation.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("frags.zig");
const row = @import("row_projection.zig");
pub const bit_widths = [_]u8{ 4, 6, 8 };
pub const groups = [_]usize{ 32, 64 };
pub const names = [_][:0]const u8{
    "tf_binned_q4_g32", "tf_binned_q4_g64", "tf_binned_q6_g32", "tf_binned_q6_g64", "tf_binned_q8_g32", "tf_binned_q8_g64",
};
pub const twin_names = [_][:0]const u8{
    "tf_binned_twin_q4_g32", "tf_binned_twin_q4_g64", "tf_binned_twin_q6_g32",
    "tf_binned_twin_q6_g64", "tf_binned_twin_q8_g32", "tf_binned_twin_q8_g64",
};
pub const Args = extern struct { k: i32, n: i32, rows: i32, group: i32 };
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

pub fn source(device: mtl.Device, a: std.mem.Allocator) ![]u8 {
    return frags.source(device, a, sources.core_binned_projection);
}

pub const Pipelines = struct {
    sums: mtl.Pipeline,
    dense: [names.len]mtl.Pipeline,
    twin: [names.len]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const a = std.heap.c_allocator;
        const text = try source(device, a);
        defer a.free(text);
        return loadSource(device, text);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        p.sums = try mtl.Pipeline.init(device, lib, "tf_binned_sums", false);
        errdefer p.sums.deinit();
        var made: usize = 0;
        errdefer for (p.dense[0..made]) |pipeline| pipeline.deinit();
        for (names, 0..) |name, i| {
            p.dense[i] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        var twins: usize = 0;
        errdefer for (p.twin[0..twins]) |pipeline| pipeline.deinit();
        for (twin_names, 0..) |name, i| {
            p.twin[i] = try mtl.Pipeline.init(device, lib, name, false);
            twins += 1;
        }
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.sums.deinit();
        for (p.dense) |pipeline| pipeline.deinit();
        for (p.twin) |pipeline| pipeline.deinit();
    }
};

pub fn supports(w: row.Weights) bool {
    return w.sum == .f32 and w.n > 0 and w.n % 32 == 0 and w.k > 0 and w.k % 64 == 0 and
        std.mem.indexOfScalar(u8, &bit_widths, w.bits) != null and std.mem.indexOfScalar(usize, &groups, w.group) != null;
}

pub fn args(w: row.Weights, rows: usize) !Args {
    try w.validate();
    if (!supports(w) or rows == 0 or rows > std.math.maxInt(i32) or w.k > std.math.maxInt(i32) / 8 or
        w.n > std.math.maxInt(i32) or w.n > std.math.maxInt(i32) / (w.k / w.group)) return error.UnsupportedBinnedProjection;
    return .{ .k = @intCast(w.k), .n = @intCast(w.n), .rows = @intCast(rows), .group = @intCast(w.group) };
}

pub fn scratchBytes(k: usize, group: usize, rows: usize) !usize {
    if ((group != 32 and group != 64) or k == 0 or k % group != 0) return error.UnsupportedBinnedGroup;
    return std.math.mul(usize, try std.math.mul(usize, rows, k / group), 8);
}

pub fn prepare(e: mtl.ComputeEncoder, p: *const Pipelines, w: row.Weights, x: Ref, sums: Ref, meta: Ref, rows: usize) !void {
    const a = try args(w, rows);
    const bytes = try scratchBytes(w.k, w.group, rows);
    if (sums.off > sums.buf.length() or bytes > sums.buf.length() - sums.off or
        meta.off > meta.buf.length() or bytes > meta.buf.length() - meta.off) return error.BinnedScratchExceeded;
    e.setPipeline(p.sums);
    e.setBuffer(x.buf, x.off, 0);
    e.setValue(a, 1);
    e.setBuffer(sums.buf, sums.off, 2);
    e.setBuffer(meta.buf, meta.off, 3);
    e.dispatchThreads(.of(w.k / w.group, rows, 1), .of(@min(w.k / w.group, 64), 1, 1));
}

pub fn dense(e: mtl.ComputeEncoder, p: *const Pipelines, w: row.Weights, x: Ref, sums: Ref, meta: Ref, y: Ref, rows: usize) !void {
    const a = try args(w, rows);
    const bits = std.mem.indexOfScalar(u8, &bit_widths, w.bits).?;
    const group = std.mem.indexOfScalar(usize, &groups, w.group).?;
    const twin = rows <= 32;
    e.setPipeline(if (twin) p.twin[bits * groups.len + group] else p.dense[bits * groups.len + group]);
    e.setBuffer(x.buf, x.off, 0);
    e.setBuffer(w.w, w.w_off, 1);
    e.setBuffer(w.scales, w.s_off, 2);
    e.setBuffer(w.biases, w.b_off, 3);
    e.setValue(a, 4);
    e.setBuffer(y.buf, y.off, 5);
    e.setBuffer(sums.buf, sums.off, 6);
    e.setBuffer(meta.buf, meta.off, 7);
    e.dispatchGroups(if (twin) .of(w.n / 8, (rows + 1) / 2, 1) else .of(w.n / 32, (rows + 31) / 32, 1), .of(64, 1, 1));
}

pub fn encode(e: mtl.ComputeEncoder, p: *const Pipelines, w: row.Weights, x: Ref, sums: Ref, meta: Ref, y: Ref, rows: usize) !void {
    try prepare(e, p, w, x, sums, meta, rows);
    try dense(e, p, w, x, sums, meta, y, rows);
}

test "binned input preparation allocates both float-pair metadata streams without overflow" {
    try std.testing.expectEqual(@as(usize, 64 * 2048 / 32 * 8), try scratchBytes(2048, 32, 64));
    try std.testing.expectEqual(@as(usize, 64 * 2048 / 64 * 8), try scratchBytes(2048, 64, 64));
    try std.testing.expectError(error.UnsupportedBinnedGroup, scratchBytes(2048, 128, 64));
    try std.testing.expectError(error.Overflow, scratchBytes(2048, 32, std.math.maxInt(usize)));
}
