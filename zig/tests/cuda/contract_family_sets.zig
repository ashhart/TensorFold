//! The full kernel contract's family sets: Nemotron's pocket, MiniMax-M3, MOSS-Image, GB10 affine-4, MiMo-V2.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

const Grouped = @import("contract_gb10_many.zig").Registrations;

pub const Sets = struct {
    gpa: std.mem.Allocator,
    pocket: *@import("contract_pocket.zig").Set,
    minimax: *@import("contract_minimax.zig").Set,
    moss: *@import("contract_moss_image.zig").Set,
    grouped: *Grouped,
    mimo: *@import("contract_mimo.zig").Group,
    a: std.mem.Allocator,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), affine4: *cuda.qlinear.Affine4Kernels, cap: u32) !Sets {
        const pocket = try @import("contract_pocket.zig").Set.init(gpu, s, a, list);
        errdefer pocket.deinit();
        const minimax = try @import("contract_minimax.zig").Set.init(gpu, s, a, list);
        errdefer minimax.deinit();
        const moss = try @import("contract_moss_image.zig").Set.init(gpu, s, a, list);
        errdefer moss.deinit();
        const grouped = try gpu.gpa.create(Grouped);
        errdefer gpu.gpa.destroy(grouped);
        grouped.* = Grouped.init(gpu, s, affine4);
        errdefer grouped.deinit();
        try grouped.append(cap, a, list);
        const mimo = try @import("contract_mimo.zig").Group.create(a, gpu, s, list);
        return .{ .gpa = gpu.gpa, .pocket = pocket, .minimax = minimax, .moss = moss, .grouped = grouped, .mimo = mimo, .a = a };
    }

    pub fn deinit(sets: Sets) void {
        sets.mimo.destroy(sets.a);
        sets.grouped.deinit();
        sets.gpa.destroy(sets.grouped);
        sets.moss.deinit();
        sets.minimax.deinit();
        sets.pocket.deinit();
    }
};
