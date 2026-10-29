//! The family sets in the full kernel contract: Nemotron's pocket.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

pub const Sets = struct {
    pocket: *@import("contract_pocket.zig").Set,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !Sets {
        const pocket = try @import("contract_pocket.zig").Set.init(gpu, s, a, list);
        return .{ .pocket = pocket };
    }

    pub fn deinit(sets: Sets) void {
        sets.pocket.deinit();
    }
};
