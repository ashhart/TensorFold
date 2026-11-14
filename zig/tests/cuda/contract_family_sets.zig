//! The CUDA contract's family sets and Clef's joint schema head.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

const Prompt = @import("contract_qmmf_prompt.zig").Registrations;
const SharedDown = @import("contract_qmmf_shared.zig").Registrations;
const Grouped = @import("contract_gb10_many.zig").Registrations;

pub const Sets = struct {
    gpa: std.mem.Allocator,
    bf16_experts: *@import("contract_experts_bf16.zig").Set,
    pocket: *@import("contract_pocket.zig").Set,
    minimax: *@import("contract_minimax.zig").Set,
    moss: *@import("contract_moss_image.zig").Set,
    modelopt: *@import("contract_flashnext_modelopt.zig").Set,
    grouped: *Grouped,
    flashnext: *@import("contract_flashnext_more.zig").Set,
    exl3: *@import("contract_exl3.zig").Set,
    shared_down: *SharedDown,
    prompt: *Prompt,
    joint_schema: *@import("contract_joint_schema.zig").Set,
    vonly: *@import("contract_window_vonly.zig").Set,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), affine4: *cuda.qlinear.Affine4Kernels, cap: u32) !Sets {
        const bf16_experts = try @import("contract_experts_bf16.zig").Set.init(gpu, s, a, list);
        errdefer bf16_experts.deinit();
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
        const flashnext = try @import("contract_flashnext_more.zig").Set.init(gpu, s, a, list);
        errdefer flashnext.deinit();
        const exl3 = try gpu.gpa.create(@import("contract_exl3.zig").Set);
        exl3.* = .{};
        errdefer {
            exl3.deinit();
            gpu.gpa.destroy(exl3);
        }
        try exl3.add(gpu, s, list);
        const shared_down = try SharedDown.create(gpu, s, &modelopt.lane, a, list);
        errdefer shared_down.deinit();
        const prompt = try gpu.gpa.create(Prompt);
        errdefer gpu.gpa.destroy(prompt);
        var prompt_lane = modelopt.lane; // the tile in this set's copy only, as Flash Next loads it
        try prompt_lane.loadPrompt(gpu.d);
        prompt.* = Prompt.init(gpu, s, prompt_lane);
        errdefer prompt.deinit();
        try prompt.append(cap, a, list);
        const joint_schema = try @import("contract_joint_schema.zig").Set.init(gpu, s, a, list);
        errdefer joint_schema.deinit();
        const vonly = try @import("contract_window_vonly.zig").Set.init(gpu, s, a, list);
        return .{
            .gpa = gpu.gpa,
            .bf16_experts = bf16_experts,
            .pocket = pocket,
            .minimax = minimax,
            .moss = moss,
            .modelopt = modelopt,
            .grouped = grouped,
            .flashnext = flashnext,
            .exl3 = exl3,
            .shared_down = shared_down,
            .prompt = prompt,
            .joint_schema = joint_schema,
            .vonly = vonly,
        };
    }

    pub fn deinit(sets: Sets) void {
        sets.vonly.deinit();
        sets.joint_schema.deinit();
        sets.prompt.deinit();
        sets.gpa.destroy(sets.prompt);
        sets.shared_down.deinit();
        sets.exl3.deinit();
        sets.gpa.destroy(sets.exl3);
        sets.flashnext.deinit();
        sets.grouped.deinit();
        sets.gpa.destroy(sets.grouped);
        sets.modelopt.deinit();
        sets.moss.deinit();
        sets.minimax.deinit();
        sets.pocket.deinit();
        sets.bf16_experts.deinit();
    }
};
