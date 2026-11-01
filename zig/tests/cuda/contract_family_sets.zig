//! The family sets in the full kernel contract: Nemotron's pocket, MiniMax-M3's, MOSS-Image's, Idol's and its DINOv3's.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

pub const Sets = struct {
    pocket: *@import("contract_pocket.zig").Set,
    minimax: *@import("contract_minimax.zig").Set,
    moss: *@import("contract_moss_image.zig").Set,
    idol: *@import("contract_idol.zig").Set,
    idol_dino: *@import("contract_idol_dino.zig").Set,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !Sets {
        const pocket = try @import("contract_pocket.zig").Set.init(gpu, s, a, list);
        errdefer pocket.deinit();
        const minimax = try @import("contract_minimax.zig").Set.init(gpu, s, a, list);
        errdefer minimax.deinit();
        const moss = try @import("contract_moss_image.zig").Set.init(gpu, s, a, list);
        errdefer moss.deinit();
        const idol = try @import("contract_idol.zig").Set.init(gpu, s, a, list);
        errdefer idol.deinit();
        const idol_dino = try @import("contract_idol_dino.zig").Set.init(gpu, s, a, list);
        return .{ .pocket = pocket, .minimax = minimax, .moss = moss, .idol = idol, .idol_dino = idol_dino };
    }

    pub fn deinit(sets: Sets) void {
        sets.idol_dino.deinit();
        sets.idol.deinit();
        sets.moss.deinit();
        sets.minimax.deinit();
        sets.pocket.deinit();
    }
};
