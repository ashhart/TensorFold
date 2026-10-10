//! Tap planes store native post-block residuals until accepted rows and their successor ids are known.
const std = @import("std");
const mtl = @import("metal");
const Ref = @import("projection.zig").Ref;
const Glue = @import("glue_gpu.zig").Kernels;

pub const layers = [_]usize{ 5, 19, 33, 47, 61 };

pub const Taps = struct {
    buffer: mtl.Buffer,
    capacity: u32,
    width: u32,
    seen: u8 = 0,
    ids: [5]usize = layers,
    rows: u32 = 0,

    pub fn init(device: mtl.Device, capacity: u32, width: u32) !Taps {
        if (capacity == 0 or capacity > 128 or width == 0) return error.BadTapShape;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, capacity, width), 2 * layers.len);
        return .{ .buffer = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked), .capacity = capacity, .width = width };
    }

    pub fn deinit(t: Taps) void {
        t.buffer.deinit();
    }

    pub fn begin(t: *Taps) void {
        t.seen = 0;
        t.rows = 0;
    }

    pub fn record(t: *Taps, glue: Glue, e: mtl.ComputeEncoder, layer: usize, hidden: Ref, rows: u32) !void {
        if (rows == 0 or rows > t.capacity) return error.BadTapShape;
        const index = std.mem.indexOfScalar(usize, &t.ids, layer) orelse return;
        if (t.seen != 0 and rows != t.rows) return error.BadTapShape;
        const bit = @as(u8, 1) << @intCast(index);
        if (t.seen & bit != 0) return error.DuplicateTap;
        try glue.unstack(e, hidden, try t.plane(index), .{ .rows = rows, .width = t.width, .stride = t.width, .offset = 0 });
        e.barrier();
        t.seen |= bit;
        t.rows = rows;
    }

    pub fn plane(t: Taps, index: usize) !Ref {
        if (index >= layers.len) return error.BadTapIndex;
        return .{ .buffer = t.buffer, .offset = index * @as(usize, t.capacity) * t.width * 2 };
    }

    pub fn complete(t: Taps) !void {
        if (t.seen != (1 << layers.len) - 1) return error.IncompleteTaps;
    }

    pub fn accepted(t: Taps, glue: Glue, e: mtl.ComputeEncoder, path: []const u32, kept: Ref, out: Ref) !void {
        try t.complete();
        if (path.len == 0) return;
        if (path.len > t.rows) return error.BadTapGather;
        for (path) |row| if (row >= t.rows) return error.BadTapGather;
        try glue.tapRows(e, .{ .buffer = t.buffer }, kept, out, .{ .rows = @intCast(path.len), .width = t.width, .capacity = t.capacity, .planes = layers.len });
        e.barrier();
    }
};

test "DFlash tap ids name completed target blocks" {
    try std.testing.expectEqualSlices(usize, &.{ 5, 19, 33, 47, 61 }, &layers);
}
