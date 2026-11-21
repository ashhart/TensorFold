//! Shapes for scoring completed index groups and expanding their token positions.
const std = @import("std");
const groups = @import("index_groups.zig");

pub const heads = 16;
pub const width = 128;
pub const head_scale: f32 = 0.02209708691207961; // 1 / sqrt(heads * width)

/// A group's location in the compressed key arena and the query's logical token position.
pub const Row = extern struct {
    group_base: u32,
    token: u32,
};

pub const Shape = extern struct {
    rows: u32,
    score_groups: u32,
    group_budget: u32,

    pub fn check(self: Shape) !void {
        if (self.rows == 0 or self.rows > std.math.maxInt(u16) or self.score_groups == 0 or
            self.score_groups > 65536 or self.group_budget < 2 or self.group_budget > 512)
            return error.BadHy4RecallShape;
    }

    pub fn tokenStride(self: Shape) u32 {
        return self.group_budget * groups.slots;
    }

    pub fn scoreBytes(self: Shape) usize {
        return @as(usize, self.rows) * self.score_groups * @sizeOf(f32);
    }

    pub fn tokenCount(self: Shape, token: u32) !u32 {
        try self.check();
        const recall = try groups.frontier(token, self.group_budget);
        if (recall.indexed_groups > self.score_groups) return error.ShortHy4RecallScores;
        return @intCast(recall.groups_to_recall * groups.slots + recall.visible_end - recall.current_group_start);
    }
};

test "the current group's first token consumes a new key at the recall edge" {
    const shape: Shape = .{ .rows = 1, .score_groups = 513, .group_budget = 512 };
    try std.testing.expectEqual(@as(u32, 2048), shape.tokenStride());
    try std.testing.expectEqual(@as(u32, 2048), try shape.tokenCount(2047));
    try std.testing.expectEqual(@as(u32, 2045), try shape.tokenCount(2048));
    try std.testing.expectEqual(@as(u32, 2048), try shape.tokenCount(2051));
}
