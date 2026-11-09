//! Pocket products from the ordinary Metal prompt, unioned with the restored prefix's receipt.
const lanes = @import("lanes");
const usage = lanes.pocket_usage;
const Metal = @import("backend.zig").Metal;
const Cache = @import("state.zig").Cache;

pub fn chunk(b: *Metal, cache: *Cache, rows: usize) !void {
    if (!b.usage_enabled or cache.untaught) return;
    try b.drain(); // each layer's shared product buffer must be read before the next chunk overwrites it
    var mask = cache.usage orelse usage.Masks{ .complete = false };
    for (b.m.weights.layers[0..b.m.config.layers], 0..) |l, i| switch (l) {
        .moe => |m| if (m.pocket) |p| {
            const acts = p.z.slice(f32, rows * p.used);
            for (acts, 0..) |x, at| {
                if (!@import("std").math.isFinite(x)) mask.complete = false;
                if (usage.fires(x)) mask.bits[i] |= @as(u64, 1) << @intCast(at % p.used);
            }
        },
        else => {},
    };
    cache.usage = mask;
}
