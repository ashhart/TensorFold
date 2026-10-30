//! Full-contract family sets: Transcribe audio, bf16 experts, Nemotron's pocket, MiniMax-M3, MOSS-Image, GB10 affine-4.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

const Grouped = @import("contract_gb10_many.zig").Registrations;

pub const Sets = struct {
    gpa: std.mem.Allocator,
    nemo_audio: *@import("contract_nemo_audio.zig").Set,
    bf16_experts: *@import("contract_experts_bf16.zig").Set,
    pocket: *@import("contract_pocket.zig").Set,
    minimax: *@import("contract_minimax.zig").Set,
    moss: *@import("contract_moss_image.zig").Set,
    grouped: *Grouped,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), affine4: *cuda.qlinear.Affine4Kernels, cap: u32) !Sets {
        const nemo_audio = try @import("contract_nemo_audio.zig").Set.init(gpu, s, a, list);
        errdefer nemo_audio.deinit();
        const bf16_experts = try @import("contract_experts_bf16.zig").Set.init(gpu, s, a, list);
        errdefer bf16_experts.deinit();
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
        return .{ .gpa = gpu.gpa, .nemo_audio = nemo_audio, .bf16_experts = bf16_experts, .pocket = pocket, .minimax = minimax, .moss = moss, .grouped = grouped };
    }

    pub fn deinit(sets: Sets) void {
        sets.grouped.deinit();
        sets.gpa.destroy(sets.grouped);
        sets.moss.deinit();
        sets.minimax.deinit();
        sets.pocket.deinit();
        sets.bf16_experts.deinit();
        sets.nemo_audio.deinit();
    }
};
