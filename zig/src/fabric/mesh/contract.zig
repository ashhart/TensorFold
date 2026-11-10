//! The reduction contract an output still owes, and what the transport must deliver before it can be resolved.
const std = @import("std");
const Axis = @import("mesh.zig").Axis;

/// disjoint: one contributor per element; rank_fold: partials added rank 0 first in FP32; canon_tree: canon's tree.
pub const Contract = enum { disjoint, rank_fold, canon_tree };

/// An output that is not complete yet: its sum over `axis`'s group under `contract` is still to run.
pub const Owes = struct { axis: Axis, contract: Contract };

/// What the transport moves before the family's epilogue resolves the output.
pub const Need = enum {
    /// every rank's partial, gathered in rank order; the family's fused kernel adds them rank 0 first
    gather_partials,
    /// every rank's disjoint part, gathered in rank order; nothing is added
    gather_disjoint,
    /// per-slice partials for canon.reduce
    slices,
};

pub fn need(o: Owes) Need {
    return switch (o.contract) {
        .rank_fold => .gather_partials,
        .disjoint => .gather_disjoint,
        .canon_tree => .slices,
    };
}

test "each contract names its transport" {
    try std.testing.expectEqual(Need.gather_partials, need(.{ .axis = .tensor, .contract = .rank_fold }));
    try std.testing.expectEqual(Need.gather_disjoint, need(.{ .axis = .tensor, .contract = .disjoint }));
    try std.testing.expectEqual(Need.slices, need(.{ .axis = .tensor, .contract = .canon_tree }));
}
