//! Byte accounting for one physical expert cache, its persistent map and bounded PLE row staging.
const std = @import("std");
const cache = @import("cache.zig");
const reader = @import("reader.zig");
const arena = @import("arena.zig");
pub const GiB: u64 = 1 << 30;
pub const Pool = struct {
    requested: u64,
    slots: usize,
    stride: u64,
    slabs: u64,
    metadata: u64,
    unused: u64,
    pub fn charged(p: Pool) !u64 {
        return std.math.add(u64, p.slabs, p.metadata);
    }
};

pub fn bytes(gib: f64) !u64 {
    const value = gib * GiB;
    if (!std.math.isFinite(value) or value < 1 or value >= 0x1p63) return error.BadExpertBudget;
    return @intFromFloat(value);
}

/// Every component uses its largest admitted expert stride, preserving mixed stored widths without conversion.
pub fn pool(requested: u64, parts: [9]u64, keys: usize, max_union: usize) !Pool {
    if (keys == 0 or max_union == 0 or max_union > keys) return error.InvalidPoolShape;
    var stride: u64 = 0;
    for (parts, 0..) |n, i| {
        if (n == 0 or n > std.math.maxInt(u32) or n % @as(u64, if (i % 3 == 0) 4 else 2) != 0) return error.InvalidExpertStride;
        stride = try std.math.add(u64, stride, n);
    }
    const count = @min(requested / stride, keys);
    if (count < max_union) return error.ExpertPoolTooSmall;
    if (count > std.math.maxInt(i32)) return error.InvalidPoolCapacity;
    const slabs = try std.math.mul(u64, count, stride);
    // CPU CLOCK slots and read tickets, the persistent CPU/GPU map, demand bits and owner bookkeeping.
    const per_slot = @sizeOf(cache.Slot) + @sizeOf(arena.Arena(void).Job) + @sizeOf(usize);
    const per_key = 9 * @sizeOf(reader.Span) + @sizeOf(i32) + @sizeOf(bool);
    const descriptors = try std.math.mul(u64, keys, per_key);
    const metadata = try std.math.add(u64, try std.math.mul(u64, count, per_slot), try std.math.add(u64, descriptors, arena.worker_stack_bytes + 4096));
    return .{ .requested = requested, .slots = @intCast(count), .stride = stride, .slabs = slabs, .metadata = metadata, .unused = requested - slabs };
}

pub fn pleRows(maximum: usize, width: usize, bits: usize, group: usize) !u64 {
    if (maximum == 0 or width == 0 or bits < 2 or bits > 8 or bits == 7 or group == 0 or width % group != 0)
        return error.InvalidPleRecipe;
    const code_bits = try std.math.mul(u64, width, bits);
    if (code_bits % 32 != 0) return error.InvalidPleRecipe;
    const row = try std.math.add(u64, code_bits / 8, try std.math.mul(u64, width / group, 4));
    // Three raw component slabs, dedup IDs, inverse indices and the immutable GPU lookup IDs.
    return std.math.mul(u64, maximum, try std.math.add(u64, row, 12));
}

/// Cold components leave the resident charge only when their loader path is replaced by bounded storage.
pub fn resident(raw: u64, routed: u64, ple: u64, expert: ?Pool, ple_stage: ?u64) !u64 {
    var out = raw;
    if (expert) |p| {
        out = try std.math.sub(u64, out, routed);
        out = try std.math.add(u64, out, try p.charged());
    }
    if (ple_stage) |stage| {
        out = try std.math.sub(u64, out, ple);
        out = try std.math.add(u64, out, stage);
    }
    return out;
}

test "FSR plan spends available slots and charges the full native storage path" {
    const parts: [9]u64 = @splat(4);
    const p = try pool(4 * 36 + 7, parts, 16, 4);
    try std.testing.expectEqual(@as(usize, 4), p.slots);
    try std.testing.expectEqual(@as(u64, 144), p.slabs);
    try std.testing.expectEqual(@as(u64, 7), p.unused);
    try std.testing.expect(p.metadata >= 16 * (9 * @sizeOf(reader.Span) + 5));
    var unaligned = parts;
    unaligned[0] = 6;
    try std.testing.expectError(error.InvalidExpertStride, pool(1024, unaligned, 16, 4));
    const q = try pool(8 * 36, parts, 16, 4);
    try std.testing.expectEqual(@as(usize, 8), q.slots);
    try std.testing.expectError(error.ExpertPoolTooSmall, pool(3 * 36, parts, 16, 4));
    try std.testing.expectEqual(@as(u64, 1000), try resident(1000, 800, 100, null, null));
    try std.testing.expectEqual(100 + try p.charged() + 64, try resident(1000, 800, 100, p, 64));
    try std.testing.expectError(error.Overflow, resident(10, 20, 0, p, null));
    try std.testing.expectError(error.BadExpertBudget, bytes(std.math.nan(f64)));
    try std.testing.expectError(error.BadExpertBudget, bytes(0));
    try std.testing.expectEqual(GiB / 2, try bytes(0.5));
    try std.testing.expectEqual(@as(u64, 152), try pleRows(1, 160, 6, 32));
    try std.testing.expectError(error.InvalidPleRecipe, pleRows(1, 160, 7, 32));
}
