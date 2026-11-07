//! Tap planes (up to eight) store native post-block residuals until accepted rows and their successor ids are known.
const std = @import("std");
const mtl = @import("metal");
const Ref = @import("projection.zig").Ref;
const Glue = @import("glue_gpu.zig").Kernels;

/// The 27B drafter's tap layers, the default before a drafter sets its own.
pub const layers = [_]usize{ 5, 19, 33, 47, 61 };
pub const max_planes = 8;

pub const Taps = struct {
    buffer: mtl.Buffer,
    capacity: u32,
    width: u32,
    seen: u16 = 0,
    ids: [max_planes]usize = layers ++ [_]usize{ 0, 0, 0 },
    count: usize = layers.len,
    /// Planes the buffer holds: the 27B's five until a drafter of more taps takes them (use).
    planes: usize = layers.len,
    rows: u32 = 0,

    pub fn init(device: mtl.Device, capacity: u32, width: u32) !Taps {
        if (capacity == 0 or capacity > @import("gpu_frame.zig").max_rows or width == 0) return error.BadTapShape;
        return .{ .buffer = try planesBuffer(device, capacity, width, layers.len), .capacity = capacity, .width = width };
    }

    /// A drafter's tap layers; the buffer grows to their count when it holds fewer planes, never shrinks.
    pub fn use(t: *Taps, device: mtl.Device, ids: []const u32) !void {
        if (ids.len == 0 or ids.len > max_planes) return error.BadTapIndex;
        if (ids.len > t.planes) {
            const grown = try planesBuffer(device, t.capacity, t.width, ids.len);
            t.buffer.deinit();
            t.buffer = grown;
            t.planes = ids.len;
        }
        for (ids, t.ids[0..ids.len]) |id, *to| to.* = id;
        t.count = ids.len;
    }

    fn planesBuffer(device: mtl.Device, capacity: u32, width: u32, planes: usize) !mtl.Buffer {
        const bytes = try std.math.mul(usize, try std.math.mul(usize, capacity, width), 2 * planes);
        return device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
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
        const index = std.mem.indexOfScalar(usize, t.ids[0..t.count], layer) orelse return;
        if (t.seen != 0 and rows != t.rows) return error.BadTapShape;
        const bit = @as(u16, 1) << @intCast(index);
        if (t.seen & bit != 0) return error.DuplicateTap;
        try glue.unstack(e, hidden, try t.plane(index), .{ .rows = rows, .width = t.width, .stride = t.width, .offset = 0 });
        e.barrier();
        t.seen |= bit;
        t.rows = rows;
    }

    pub fn plane(t: Taps, index: usize) !Ref {
        if (index >= t.count or index >= t.planes) return error.BadTapIndex;
        return .{ .buffer = t.buffer, .offset = index * @as(usize, t.capacity) * t.width * 2 };
    }

    pub fn complete(t: Taps) !void {
        if (t.count == 0 or t.count > t.planes or t.seen != (@as(u16, 1) << @intCast(t.count)) - 1) return error.IncompleteTaps;
    }

    pub fn accepted(t: Taps, glue: Glue, e: mtl.ComputeEncoder, path: []const u32, kept: Ref, out: Ref) !void {
        try t.complete();
        if (path.len == 0) return;
        if (path.len > t.rows) return error.BadTapGather;
        for (path) |row| if (row >= t.rows) return error.BadTapGather;
        try glue.tapRows(e, .{ .buffer = t.buffer }, kept, out, .{ .rows = @intCast(path.len), .width = t.width, .capacity = t.capacity, .planes = @intCast(t.count) });
        e.barrier();
    }
};

test "DFlash tap ids name completed target blocks" {
    try std.testing.expectEqualSlices(usize, &.{ 5, 19, 33, 47, 61 }, &layers);
}
