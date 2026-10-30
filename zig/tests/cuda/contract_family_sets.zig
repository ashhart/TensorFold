//! The family sets in the full kernel contract: the bf16 experts, Nemotron's pocket.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

pub const Sets = struct {
    bf16_experts: *@import("contract_experts_bf16.zig").Set,
    pocket: *@import("contract_pocket.zig").Set,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !Sets {
        const bf16_experts = try @import("contract_experts_bf16.zig").Set.init(gpu, s, a, list);
        errdefer bf16_experts.deinit();
        const pocket = try @import("contract_pocket.zig").Set.init(gpu, s, a, list);
        return .{ .bf16_experts = bf16_experts, .pocket = pocket };
    }

    pub fn deinit(sets: Sets) void {
        sets.pocket.deinit();
        sets.bf16_experts.deinit();
    }
};
