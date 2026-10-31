//! The family sets in the full kernel contract: Nemotron's pocket, MiniMax-M3's, MOSS-Image's.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

pub const Sets = struct {
    pocket: *@import("contract_pocket.zig").Set,
    minimax: *@import("contract_minimax.zig").Set,
    moss: *@import("contract_moss_image.zig").Set,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !Sets {
        const pocket = try @import("contract_pocket.zig").Set.init(gpu, s, a, list);
        errdefer pocket.deinit();
        const minimax = try @import("contract_minimax.zig").Set.init(gpu, s, a, list);
        errdefer minimax.deinit();
        const moss = try @import("contract_moss_image.zig").Set.init(gpu, s, a, list);
        return .{ .pocket = pocket, .minimax = minimax, .moss = moss };
    }

    pub fn deinit(sets: Sets) void {
        sets.moss.deinit();
        sets.minimax.deinit();
        sets.pocket.deinit();
    }
};
