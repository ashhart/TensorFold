//! Logical sparse-index groups shared by CUDA and Metal; token slots define group boundaries.
const std = @import("std");

pub const slots = 4;

pub const PoolShape = extern struct {
    groups: u32,
    width: u32,
};

pub const Append = struct {
    first_group: usize,
    new_groups: usize,
    pending_slots: usize,
};

/// Each completed group consumes four token slots and uses the first one's rotary position.
pub fn append(previous: usize, count: usize) !Append {
    const after = try std.math.add(usize, previous, count);
    return .{
        .first_group = previous / slots,
        .new_groups = after / slots - previous / slots,
        .pending_slots = after % slots,
    };
}

pub const Frontier = struct {
    visible_end: usize,
    indexed_groups: usize,
    current_group_start: usize,
    groups_to_recall: usize,
};

/// The sparse selector ranks only closed earlier groups. It always includes the causal part of the current group.
pub fn frontier(query_slot: usize, group_budget: usize) !Frontier {
    if (group_budget == 0) return error.ZeroGroupBudget;
    const end = try std.math.add(usize, query_slot, 1);
    const group = query_slot / slots;
    return .{
        .visible_end = end,
        .indexed_groups = group,
        .current_group_start = group * slots,
        .groups_to_recall = @min(group, group_budget - 1),
    };
}

test "prefix continuation closes the same groups after draft rejection" {
    const accepted = try append(2046, 3);
    try std.testing.expectEqual(Append{ .first_group = 511, .new_groups = 1, .pending_slots = 1 }, accepted);
    const provisioned = try append(2046, 6);
    try std.testing.expectEqual(@as(usize, 2), provisioned.new_groups);
    const after_rejection = try append(2046, 6 - 3);
    try std.testing.expectEqual(accepted, after_rejection);
}

test "recall excludes the query group from ranking around the 2048-token budget" {
    const last = try frontier(2047, 512);
    try std.testing.expectEqual(Frontier{ .visible_end = 2048, .indexed_groups = 511, .current_group_start = 2044, .groups_to_recall = 511 }, last);
    const first = try frontier(2048, 512);
    try std.testing.expectEqual(Frontier{ .visible_end = 2049, .indexed_groups = 512, .current_group_start = 2048, .groups_to_recall = 511 }, first);
}
