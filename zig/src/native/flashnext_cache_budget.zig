//! Automatic cache room must respect both the footprint and allocated pool buffers.
const math = @import("cache_budget.zig");
const margin = @import("sys").memory_policy.margin;
pub fn automatic(cap: u64, ready: u64, allocated: u64) !u64 {
    const room = cap -| ready -| margin;
    const physical = (try math.fitUnder(cap, ready, null, false, .{ .margin = margin, .ceiling = @min(room, @max(16 * math.GiB, room / 2)) })).budget;
    return @min(physical, cap -| allocated -| margin);
}
