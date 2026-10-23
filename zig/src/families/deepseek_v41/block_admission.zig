//! Decode admission bounds every scratch dimension before emitting a view or launch.
const std = @import("std");
const Config = @import("config.zig").Config;

pub fn nativePrefill(c: *const Config, world: u32) !void {
    if (world == 0 or c.heads % world != 0) return error.BadWorld;
    const tile: u32 = @intCast(@import("block_prefill.zig").bmq);
    if (c.heads / world != 16 and (c.heads / world) % tile != 0) return error.UnsupportedPrefillHeads;
}

pub fn options(c: *const Config, o: anytype, layers: []const u32, n: i64) !void {
    const rank = if (@hasField(@TypeOf(o), "rank")) o.rank else 0;
    const part = @import("partition.zig").share(c.expert_width, rank, o.world) catch return error.InvalidBlock;
    if (o.expert_rows < 0 or o.dspark_rows < 0 or o.branch_rows < 0 or o.prefill_rows < 0 or o.image_rows < -1) return error.InvalidBlock;
    for (layers) |l| {
        if (l >= c.allLayers()) return error.InvalidBlock;
        if ((if (c.isBackbone(l)) o.expert_rows else o.dspark_rows) < n) return error.InvalidBlock;
    }
    inline for (.{ .{ o.expert_rows, @as(i64, c.topk) + 1 }, .{ o.dspark_rows, @as(i64, c.dspark_topk) + 1 } }) |cap| {
        const slots = std.math.mul(i64, cap[0], cap[1]) catch return error.InvalidBlock;
        const width = @max(8 * @as(i64, part.len), @as(i64, c.hidden));
        const items = std.math.mul(i64, slots, width) catch return error.InvalidBlock;
        _ = std.math.mul(i64, items, 4) catch return error.InvalidBlock;
    }
    if (o.expert_topp) |p| if (!std.math.isFinite(p) or p <= 0 or p > 1) return error.InvalidBlock;
}
pub fn tensor(shape: []const i64, bytes: u64) !void {
    var elements: i64 = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        if (shape[i] < 0) return error.InvalidBlock;
        elements = std.math.mul(i64, elements, shape[i]) catch return error.InvalidBlock;
    }
    _ = std.math.mul(u64, @intCast(elements), bytes) catch return error.InvalidBlock;
}
pub fn view(shape: []const i64, stride: []const i64, offset: i64, bytes: u64) !void {
    if (shape.len != stride.len or offset < 0) return error.InvalidBlock;
    for (shape, stride) |n, s| if (n < 0 or s < 0) return error.InvalidBlock;
    if (std.mem.indexOfScalar(i64, shape, 0) != null) return;
    var last: u64 = 0;
    for (shape, stride) |n, s| {
        if (n < 0 or s < 0) return error.InvalidBlock;
        const reach = std.math.mul(u64, @intCast(n - 1), @intCast(s)) catch return error.InvalidBlock;
        last = std.math.add(u64, last, reach) catch return error.InvalidBlock;
    }
    const count = std.math.add(u64, last, 1) catch return error.InvalidBlock;
    const size = std.math.mul(u64, count, bytes) catch return error.InvalidBlock;
    _ = std.math.add(u64, @intCast(offset), size) catch return error.InvalidBlock;
}
