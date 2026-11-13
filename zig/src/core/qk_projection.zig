//! qk (core/qk.metal): 4-bit g64 products on weight_layout as load leaves it, tf_lane_reg's words on the same K split.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const lane = @import("lane_projection.zig");

/// tf_qk_args.
pub const Args = extern struct { rows: i32, k: i32, n: i32, groups: i32, stride: i32 };

/// The largest K split a tile takes: its slices add up through threadgroup memory.
pub const max_split = 8;

/// Rows a threadgroup takes: all of `rows`, rounded to 8, up to 48 while slices fit threadgroup memory, else 32.
pub fn blockRows(rows: u32, sk: u32) u32 {
    const whole = (rows + 7) / 8 * 8;
    if (whole <= 48 and threadgroupBytes(whole, sk) <= 32768) return whole;
    return 32;
}

/// Threadgroup memory a tile's slices add up through: each slice past the first keeps a lane's values for 32 lanes.
pub fn threadgroupBytes(block: u32, sk: u32) usize {
    return @as(usize, sk -| 1) * block * 32 * @sizeOf(f32);
}

pub const Pipelines = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    cache: std.AutoHashMapUnmanaged([2]u32, mtl.Pipeline) = .empty,

    pub fn init(allocator: std.mem.Allocator, device: mtl.Device) Pipelines {
        return .{ .allocator = allocator, .device = device };
    }

    pub fn deinit(q: *Pipelines) void {
        var it = q.cache.valueIterator();
        while (it.next()) |p| p.deinit();
        q.cache.deinit(q.allocator);
    }

    /// The pipeline of a K split and a block of rows, compiled the first time it is asked for.
    pub fn get(q: *Pipelines, sk: u32, block: u32) !mtl.Pipeline {
        if (sk == 0 or sk > max_split or block == 0 or block % 8 != 0 or block > 48 or threadgroupBytes(block, sk) > 32768) return error.InvalidSchedule;
        const key = [2]u32{ sk, block };
        if (q.cache.get(key)) |p| return p;
        if (!q.device.tensorUnits()) return error.TensorUnitsRequired;
        const text = try std.fmt.allocPrint(q.allocator, "#define TF_SK {d}\n#define TF_ROWS {d}\n{s}", .{ sk, block, ks.core_qk });
        defer q.allocator.free(text);
        const lib = try mtl.Library.fromSource(q.device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const p = try mtl.Pipeline.init(q.device, lib, "tf_qk", false);
        errdefer p.deinit();
        if (p.simdWidth() != 32 or p.maxThreads() < 32 * sk) return error.UnsupportedPipelineShape;
        try q.cache.put(q.allocator, key, p);
        return p;
    }

    /// y = x W^T for `rows` rows: w in weight_layout, sb in metadata_layout, x's group sums; one dispatch, no barrier.
    pub fn encode(q: *Pipelines, e: mtl.ComputeEncoder, n: usize, k: usize, sk: u32, x: lane.Ref, w: lane.Ref, sb: lane.Ref, sums: lane.Sums, y: lane.Ref, rows: u32) !void {
        if (n == 0 or n % 32 != 0 or k == 0 or k % 64 != 0 or n > std.math.maxInt(i32) / 16 or k > std.math.maxInt(i32) / 16) return error.InvalidShape;
        if (rows == 0 or sums.stride < rows) return error.InvalidRows;
        const groups = k / 64;
        try checkRef(x, @as(usize, rows) * k * 2, 2);
        try checkRef(w, n * k / 2, 16);
        try checkRef(sb, n * groups * 4, 16);
        try checkRef(sums.ref, ((groups - 1) * @as(usize, sums.stride) + rows) * 4, 4);
        try checkRef(y, @as(usize, rows) * n * 2, 8);
        const block = blockRows(rows, sk);
        e.setPipeline(try q.get(sk, block));
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setBuffer(sb.buf, sb.off, 2);
        e.setBuffer(sums.ref.buf, sums.ref.off, 3);
        e.setBuffer(y.buf, y.off, 4);
        e.setValue(Args{ .rows = @intCast(rows), .k = @intCast(k), .n = @intCast(n), .groups = @intCast(groups), .stride = @intCast(sums.stride) }, 5);
        e.dispatchGroups(mtl.Size.of(n / 32, (rows + block - 1) / block, 1), mtl.Size.of(32 * sk, 1, 1));
    }
};

/// Whether this GPU compiles qk: tensor units, and a runtime compiler that takes uint4b_format.
pub fn available(allocator: std.mem.Allocator, device: mtl.Device) bool {
    if (!device.tensorUnits()) return false;
    var q = Pipelines.init(allocator, device);
    defer q.deinit();
    _ = q.get(1, 16) catch return false;
    return true;
}

fn checkRef(r: lane.Ref, need: usize, alignment: usize) !void {
    if (r.off % alignment != 0) return error.MisalignedBuffer;
    const len = r.buf.length();
    if (r.off > len or need > len - r.off) return error.BufferTooSmall;
}

test "a block takes every row up to 48 while its slices fit threadgroup memory, else 32" {
    try std.testing.expectEqual(@as(u32, 8), blockRows(1, 2));
    try std.testing.expectEqual(@as(u32, 16), blockRows(13, 4));
    try std.testing.expectEqual(@as(u32, 48), blockRows(48, 4));
    try std.testing.expectEqual(@as(u32, 32), blockRows(48, 8)); // 7 slices of 48 rows: 43,008 bytes
    try std.testing.expectEqual(@as(u32, 32), blockRows(64, 1));
    try std.testing.expectEqual(@as(u32, 32), blockRows(2048, 2));
    for ([_]u32{ 1, 2, 4, 8 }) |sk| for (1..129) |rows| {
        try std.testing.expect(threadgroupBytes(blockRows(@intCast(rows), sk), sk) <= 32768);
    };
}
