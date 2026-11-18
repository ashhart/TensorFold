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
    inkling2: *@import("contract_inkling2.zig").Set,
    moss: *@import("contract_moss_image.zig").Set,
    nemo_audio: *@import("contract_nemo_audio.zig").Set,
    qwen3_asr: *@import("contract_qwen3_asr.zig").Set,
    glm_vision: *@import("contract_glm_vision.zig").Set,
    modelopt: *@import("contract_flashnext_modelopt.zig").Set,
    grouped: *Grouped,
    flashnext: *@import("contract_flashnext_more.zig").Set,
    exl3: *@import("contract_exl3.zig").Set,
    shared_down: *SharedDown,
    prompt: *Prompt,
    muse: *@import("contract_muse.zig").Group,
    a: std.mem.Allocator,
    joint_schema: *@import("contract_joint_schema.zig").Set,
    idol: *@import("contract_idol.zig").Set,
    idol_decoder: *@import("contract_idol_decoder.zig").Set,
    glm_moe: *@import("contract_glm_moe_direct.zig").Set,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), affine4: *cuda.qlinear.Affine4Kernels, cap: u32) !Sets {
        const bf16_experts = try @import("contract_experts_bf16.zig").Set.init(gpu, s, a, list);
        errdefer bf16_experts.deinit();
        const pocket = try @import("contract_pocket.zig").Set.init(gpu, s, a, list);
        errdefer pocket.deinit();
        const minimax = try @import("contract_minimax.zig").Set.init(gpu, s, a, list);
        errdefer minimax.deinit();
        const inkling2 = try @import("contract_inkling2.zig").Set.init(gpu, s, a, list);
        errdefer inkling2.deinit();
        const moss = try @import("contract_moss_image.zig").Set.init(gpu, s, a, list);
        errdefer moss.deinit();
        const nemo_audio = try @import("contract_nemo_audio.zig").Set.init(gpu, s, a, list);
        errdefer nemo_audio.deinit();
        const qwen3_asr = try @import("contract_qwen3_asr.zig").Set.init(gpu, s, a, list);
        errdefer qwen3_asr.deinit();
        const glm_vision = try @import("contract_glm_vision.zig").Set.init(gpu, s, a, list);
        errdefer glm_vision.deinit();
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
        const muse = try @import("contract_muse.zig").Group.create(a, gpu, s, list);
        errdefer muse.destroy(a);
        const joint_schema = try @import("contract_joint_schema.zig").Set.init(gpu, s, a, list);
        errdefer joint_schema.deinit();
        const idol = try @import("contract_idol.zig").Set.init(gpu, s, a, list);
        errdefer idol.deinit();
        const idol_decoder = try @import("contract_idol_decoder.zig").Set.init(gpu, s, a, list);
        const glm_moe = try @import("contract_glm_moe_direct.zig").Set.init(gpu, s, a, list);
        return .{
            .gpa = gpu.gpa,
            .bf16_experts = bf16_experts,
            .pocket = pocket,
            .minimax = minimax,
            .inkling2 = inkling2,
            .moss = moss,
            .nemo_audio = nemo_audio,
            .qwen3_asr = qwen3_asr,
            .glm_vision = glm_vision,
            .modelopt = modelopt,
            .grouped = grouped,
            .flashnext = flashnext,
            .exl3 = exl3,
            .shared_down = shared_down,
            .prompt = prompt,
            .joint_schema = joint_schema,
            .idol = idol,
            .muse = muse,
            .a = a,
            .idol_decoder = idol_decoder,
            .glm_moe = glm_moe,
        };
    }

    pub fn deinit(sets: Sets) void {
        sets.glm_moe.deinit();
        sets.idol_decoder.deinit();
        sets.idol.deinit();
        sets.joint_schema.deinit();
        sets.muse.destroy(sets.a);
        sets.prompt.deinit();
        sets.gpa.destroy(sets.prompt);
        sets.shared_down.deinit();
        sets.exl3.deinit();
        sets.gpa.destroy(sets.exl3);
        sets.flashnext.deinit();
        sets.grouped.deinit();
        sets.gpa.destroy(sets.grouped);
        sets.modelopt.deinit();
        sets.qwen3_asr.deinit();
        sets.nemo_audio.deinit();
        sets.glm_vision.deinit();
        sets.moss.deinit();
        sets.inkling2.deinit();
        sets.minimax.deinit();
        sets.pocket.deinit();
        sets.bf16_experts.deinit();
    }
};
