//! Tap planes store native post-block residuals until accepted rows and their successor ids are known.
const std = @import("std");
const mtl = @import("metal");
const Ref = @import("projection.zig").Ref;
const Glue = @import("glue_gpu.zig").Kernels;

pub const layers = [_]usize{ 5, 19, 33, 47, 61 };
/// The most tap planes a drafter may name (a DFlash v1 draft names up to eight).
pub const max_planes = 8;

/// A drafter's fusion of a row's taps (fc then hidden_norm), encoded where the taps complete.
pub const Fuser = struct { ptr: *anyopaque, width: u32, encode: *const fn (*anyopaque, mtl.ComputeEncoder, Ref, Ref, u32) anyerror!void };
/// Where accepted rows gather from: the raw planes, or one plane of fused rows.
pub const Source = struct { buffer: mtl.Buffer, width: u32, planes: u32 };

pub const Taps = struct {
    buffer: mtl.Buffer,
    fuser: ?Fuser = null,
    fused: ?mtl.Buffer = null, // [capacity][fuser width]
    rowmajor: ?mtl.Buffer = null, // [capacity][count * width]: a row's planes side by side, fc's input
    identity: ?mtl.Buffer = null, // 0, 1, ... capacity - 1: every row, for the row-major gather
    capacity: u32,
    width: u32,
    seen: u8 = 0,
    ids: [max_planes]usize = .{ 5, 19, 33, 47, 61, 0, 0, 0 },
    count: u32 = layers.len,
    rows: u32 = 0,

    pub fn init(device: mtl.Device, capacity: u32, width: u32) !Taps {
        if (capacity == 0 or capacity > @import("gpu_frame.zig").max_rows or width == 0) return error.BadTapShape;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, capacity, width), 2 * layers.len);
        return .{ .buffer = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked), .capacity = capacity, .width = width };
    }

    pub fn deinit(t: Taps) void {
        t.dropFused();
        t.buffer.deinit();
    }

    fn dropFused(t: Taps) void {
        if (t.fused) |b| b.deinit();
        if (t.rowmajor) |b| b.deinit();
        if (t.identity) |b| b.deinit();
    }

    /// Completed taps fused for a drafter whose ring keeps fused rows; null: planes stay raw.
    pub fn setFuser(t: *Taps, device: mtl.Device, f: ?Fuser) !void {
        t.dropFused();
        t.fuser = null;
        t.fused = null;
        t.rowmajor = null;
        t.identity = null;
        const fuser = f orelse return;
        if (fuser.width == 0) return error.BadTapShape;
        const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const fused = try device.buffer(@as(usize, t.capacity) * fuser.width * 2, options);
        errdefer fused.deinit();
        const rowmajor = try device.buffer(@as(usize, t.capacity) * t.count * t.width * 2, options);
        errdefer rowmajor.deinit();
        const identity = try device.buffer(@as(usize, t.capacity) * 4, options);
        for (identity.slice(u32, t.capacity), 0..) |*x, i| x.* = @intCast(i);
        t.fuser = fuser;
        t.fused = fused;
        t.rowmajor = rowmajor;
        t.identity = identity;
    }

    /// The rows accepted taps gather from.
    pub fn source(t: Taps) Source {
        if (t.fused) |f| return .{ .buffer = f, .width = t.fuser.?.width, .planes = 1 };
        return .{ .buffer = t.buffer, .width = t.width, .planes = t.count };
    }

    /// The drafter's tap layers; planes are re-made only when their count changes.
    pub fn setLayers(t: *Taps, device: mtl.Device, ids: []const u32) !void {
        if (ids.len == 0 or ids.len > max_planes or (t.fuser != null and ids.len != t.count)) return error.BadTapShape;
        if (ids.len != t.count) {
            const bytes = try std.math.mul(usize, try std.math.mul(usize, t.capacity, t.width), 2 * ids.len);
            const buffer = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            t.buffer.deinit();
            t.buffer = buffer;
            t.count = @intCast(ids.len);
        }
        t.ids = @splat(0);
        for (ids, t.ids[0..ids.len]) |id, *to| to.* = id;
        t.begin();
    }

    pub fn begin(t: *Taps) void {
        t.seen = 0;
        t.rows = 0;
    }

    pub fn record(t: *Taps, glue: Glue, e: mtl.ComputeEncoder, layer: usize, hidden: Ref, rows: u32) !void {
        if (rows == 0 or rows > t.capacity) return error.BadTapShape;
        const index = std.mem.indexOfScalar(usize, t.ids[0..t.count], layer) orelse return;
        if (t.seen != 0 and rows != t.rows) return error.BadTapShape;
        const bit = @as(u8, 1) << @intCast(index);
        if (t.seen & bit != 0) return error.DuplicateTap;
        try glue.unstack(e, hidden, try t.plane(index), .{ .rows = rows, .width = t.width, .stride = t.width, .offset = 0 });
        e.barrier();
        t.seen |= bit;
        t.rows = rows;
        if (t.fuser) |f| if (t.seen == fullMask(t.count)) {
            const kept = Ref{ .buffer = t.identity.? };
            try glue.tapRows(e, .{ .buffer = t.buffer }, kept, .{ .buffer = t.rowmajor.? }, .{ .rows = rows, .width = t.width, .capacity = t.capacity, .planes = t.count });
            e.barrier();
            try f.encode(f.ptr, e, .{ .buffer = t.rowmajor.? }, .{ .buffer = t.fused.? }, rows);
            e.barrier();
        };
    }

    pub fn plane(t: Taps, index: usize) !Ref {
        if (index >= t.count) return error.BadTapIndex;
        return .{ .buffer = t.buffer, .offset = index * @as(usize, t.capacity) * t.width * 2 };
    }

    pub fn complete(t: Taps) !void {
        if (t.seen != fullMask(t.count)) return error.IncompleteTaps;
    }

    pub fn accepted(t: Taps, glue: Glue, e: mtl.ComputeEncoder, path: []const u32, kept: Ref, out: Ref) !void {
        try t.complete();
        if (path.len == 0) return;
        if (path.len > t.rows) return error.BadTapGather;
        for (path) |row| if (row >= t.rows) return error.BadTapGather;
        const s = t.source();
        try glue.tapRows(e, .{ .buffer = s.buffer }, kept, out, .{ .rows = @intCast(path.len), .width = s.width, .capacity = t.capacity, .planes = s.planes });
        e.barrier();
    }
};

fn fullMask(count: u32) u8 {
    return @intCast((@as(u16, 1) << @intCast(count)) - 1);
}

test "eight tap planes complete when all eight are seen" {
    try std.testing.expectEqual(@as(u8, 0xff), fullMask(8));
    try std.testing.expectEqual(@as(u8, 0x1f), fullMask(5));
}

test "DFlash tap ids name completed target blocks" {
    try std.testing.expectEqualSlices(usize, &.{ 5, 19, 33, 47, 61 }, &layers);
}
