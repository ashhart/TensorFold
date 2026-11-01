//! Family contract sets: pocket, MiniMax-M3, MOSS-Image, grouped affine-4, Flash Next BF16 dense and FP8 expert downs.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

const Prompt = @import("contract_qmmf_prompt.zig").Registrations;
const SharedDown = @import("contract_qmmf_shared.zig").Registrations;
const Grouped = @import("contract_gb10_many.zig").Registrations;

pub const Sets = struct {
    gpa: std.mem.Allocator,
    pocket: *@import("contract_pocket.zig").Set,
    minimax: *@import("contract_minimax.zig").Set,
    moss: *@import("contract_moss_image.zig").Set,
    modelopt: *@import("contract_flashnext_modelopt.zig").Set,
    grouped: *Grouped,
    shared_down: *SharedDown,
    prompt: *Prompt,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), affine4: *cuda.qlinear.Affine4Kernels, cap: u32) !Sets {
        const pocket = try @import("contract_pocket.zig").Set.init(gpu, s, a, list);
        errdefer pocket.deinit();
        const minimax = try @import("contract_minimax.zig").Set.init(gpu, s, a, list);
        errdefer minimax.deinit();
        const moss = try @import("contract_moss_image.zig").Set.init(gpu, s, a, list);
        errdefer moss.deinit();
        const modelopt = try @import("contract_flashnext_modelopt.zig").Set.init(gpu, s, a, list);
        errdefer modelopt.deinit();
        const grouped = try gpu.gpa.create(Grouped);
        errdefer gpu.gpa.destroy(grouped);
        grouped.* = Grouped.init(gpu, s, affine4);
        errdefer grouped.deinit();
        try grouped.append(cap, a, list);
        const shared_down = try SharedDown.create(gpu, s, &modelopt.lane, a, list);
        errdefer shared_down.deinit();
        const prompt = try gpu.gpa.create(Prompt);
        errdefer gpu.gpa.destroy(prompt);
        prompt.* = Prompt.init(gpu, s, &modelopt.lane);
        errdefer prompt.deinit();
        try prompt.append(cap, a, list);
        return .{ .prompt = prompt, .shared_down = shared_down, .gpa = gpu.gpa, .pocket = pocket, .minimax = minimax, .moss = moss, .modelopt = modelopt, .grouped = grouped };
    }

    pub fn deinit(sets: Sets) void {
        sets.prompt.deinit();
        sets.gpa.destroy(sets.prompt);
        sets.shared_down.deinit();
        sets.grouped.deinit();
        sets.gpa.destroy(sets.grouped);
        sets.modelopt.deinit();
        sets.moss.deinit();
        sets.minimax.deinit();
        sets.pocket.deinit();
    }
};
