//! The CUDA contract's family sets, Clef's joint schema head and ZONOS3's kernels.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

const Prompt = @import("contract_qmmf_prompt.zig").Registrations;
const Many = @import("contract_qmmf_many.zig").Registrations;
const SharedDown = @import("contract_qmmf_shared.zig").Registrations;
const Grouped = @import("contract_gb10_many.zig").Registrations;

pub const Sets = struct {
    gpa: std.mem.Allocator,
    bf16_experts: *@import("contract_experts_bf16.zig").Set,
    mxfp4_experts: *@import("contract_experts_mxfp4.zig").Set,
    ernie5_5_head: *@import("contract_ernie5_5_head.zig").Set,
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
    many: *Many,
    prompt: *Prompt,
    muse: *@import("contract_muse.zig").Group,
    a: std.mem.Allocator,
    joint_schema: *@import("contract_joint_schema.zig").Set,
    idol: *@import("contract_idol.zig").Set,
    idol_decoder: *@import("contract_idol_decoder.zig").Set,
    glm_moe: *@import("contract_glm_moe_direct.zig").Set,
    zonos3: *@import("contract_zonos3.zig").Set,
    hy4_ihc_hc: *@import("contract_hy4_hc.zig").Set,
    topk_rows: *@import("contract_topk_rows.zig").Set,
    sparse_index: *@import("contract_sparse_index.zig").Set,
    sparse_attn: *@import("contract_sparse_attn.zig").Set,
    drama3_attn: *@import("contract_drama3_attn.zig").Group,
    drama3_codec_attn: *@import("contract_drama3_codec_attn.zig").Group,
    qwen35_draft_attn: *@import("contract_qwen35_draft_attn.zig").Group,
    glm_dflash_attn: *@import("contract_glm_dflash_attn.zig").Group,
    glm_chunks_fp8: *@import("contract_glm_chunks_fp8.zig").Group,
    hy4_chunks: *@import("contract_hy4_chunks.zig").Group,
    prefill_full_ab: *@import("contract_prefill_full_ab.zig").Group,
    qwen_prompt_ab: *@import("contract_qwen_prompt_ab.zig").Group,
    flashnext_attn_ab: *@import("contract_flashnext_attn_ab.zig").Group,
    nemo_direct_attn: *@import("contract_nemo_direct_attn.zig").Group,
    qwen_pages_ab: *@import("contract_qwen_pages_ab.zig").Group,
    qwen_tc_ab: *@import("contract_qwen_tc_ab.zig").Group,
    qwen_tree_ab: *@import("contract_qwen_tree_ab.zig").Group,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel), affine4: *cuda.qlinear.Affine4Kernels, cap: u32) !Sets {
        const bf16_experts = try @import("contract_experts_bf16.zig").Set.init(gpu, s, a, list);
        errdefer bf16_experts.deinit();
        const mxfp4_experts = try @import("contract_experts_mxfp4.zig").Set.init(gpu, s, a, list);
        errdefer mxfp4_experts.deinit();
        const ernie5_5_head = try @import("contract_ernie5_5_head.zig").Set.init(gpu, s, a, list);
        errdefer ernie5_5_head.deinit();
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
        const many = try gpu.gpa.create(Many);
        errdefer gpu.gpa.destroy(many);
        many.* = .{ .gpu = gpu, .lane = &modelopt.lane, .stream = s };
        errdefer many.deinit();
        try many.append(list);
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
        errdefer idol_decoder.deinit();
        const glm_moe = try @import("contract_glm_moe_direct.zig").Set.init(gpu, s, a, list);
        errdefer glm_moe.deinit();
        const zonos3, const hy4_ihc_hc = try initPair(
            @import("contract_zonos3.zig").Set,
            @import("contract_hy4_hc.zig").Set,
            gpu,
            s,
            a,
            list,
        );
        errdefer hy4_ihc_hc.deinit();
        errdefer zonos3.deinit();
        const topk_rows, const sparse_index = try @import("contract_set_pair.zig").pair(@import("contract_topk_rows.zig").Set, @import("contract_sparse_index.zig").Set, .{ gpu, s, a, list });
        errdefer topk_rows.deinit();
        errdefer sparse_index.deinit();
        const sparse_attn = try @import("contract_sparse_attn.zig").Set.init(gpu, s, a, list);
        errdefer sparse_attn.deinit();
        const drama3_attn = try @import("contract_drama3_attn.zig").Group.create(a, gpu, s, list);
        errdefer drama3_attn.destroy();
        const drama3_codec_attn = try @import("contract_drama3_codec_attn.zig").Group.create(a, gpu, s, list);
        errdefer drama3_codec_attn.destroy();
        const qwen35_draft_attn = try @import("contract_qwen35_draft_attn.zig").Group.create(a, .{ .gpu = gpu, .s = s }, list);
        errdefer qwen35_draft_attn.destroy();
        const glm_dflash_attn = try @import("contract_glm_dflash_attn.zig").Group.create(a, .{ .gpu = gpu, .s = s }, list);
        errdefer glm_dflash_attn.destroy();
        const glm_chunks_fp8 = try @import("contract_glm_chunks_fp8.zig").Group.create(a, gpu, s, list);
        errdefer glm_chunks_fp8.destroy();
        const hy4_chunks = try @import("contract_hy4_chunks.zig").Group.create(a, gpu, s, list);
        errdefer hy4_chunks.destroy();
        const prefill_full_ab = try @import("contract_prefill_full_ab.zig").Group.create(a, gpu, s, list);
        errdefer prefill_full_ab.destroy();
        const qwen_prompt_ab = try @import("contract_qwen_prompt_ab.zig").Group.create(a, gpu, s, list);
        errdefer qwen_prompt_ab.destroy();
        const flashnext_attn_ab = try @import("contract_flashnext_attn_ab.zig").Group.create(a, gpu, s, list);
        errdefer flashnext_attn_ab.destroy();
        const nemo_direct_attn = try @import("contract_nemo_direct_attn.zig").Group.create(a, gpu, s, list);
        errdefer nemo_direct_attn.destroy();
        const qwen_pages_ab = try @import("contract_qwen_pages_ab.zig").Group.create(a, gpu, s, list);
        errdefer qwen_pages_ab.destroy();
        const qwen_tc_ab = try @import("contract_qwen_tc_ab.zig").Group.create(a, gpu, s, list);
        errdefer qwen_tc_ab.destroy();
        const qwen_tree_ab = try @import("contract_qwen_tree_ab.zig").Group.create(a, gpu, s, list);
        errdefer qwen_tree_ab.destroy();
        return .{
            .gpa = gpu.gpa,
            .bf16_experts = bf16_experts,
            .mxfp4_experts = mxfp4_experts,
            .ernie5_5_head = ernie5_5_head,
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
            .many = many,
            .prompt = prompt,
            .joint_schema = joint_schema,
            .idol = idol,
            .muse = muse,
            .a = a,
            .idol_decoder = idol_decoder,
            .glm_moe = glm_moe,
            .zonos3 = zonos3,
            .hy4_ihc_hc = hy4_ihc_hc,
            .topk_rows = topk_rows,
            .sparse_index = sparse_index,
            .sparse_attn = sparse_attn,
            .drama3_attn = drama3_attn,
            .drama3_codec_attn = drama3_codec_attn,
            .qwen35_draft_attn = qwen35_draft_attn,
            .glm_dflash_attn = glm_dflash_attn,
            .glm_chunks_fp8 = glm_chunks_fp8,
            .hy4_chunks = hy4_chunks,
            .prefill_full_ab = prefill_full_ab,
            .qwen_prompt_ab = qwen_prompt_ab,
            .flashnext_attn_ab = flashnext_attn_ab,
            .nemo_direct_attn = nemo_direct_attn,
            .qwen_pages_ab = qwen_pages_ab,
            .qwen_tc_ab = qwen_tc_ab,
            .qwen_tree_ab = qwen_tree_ab,
        };
    }

    pub fn deinit(sets: Sets) void {
        sets.sparse_attn.deinit();
        sets.sparse_index.deinit();
        sets.topk_rows.deinit();
        sets.hy4_ihc_hc.deinit();
        sets.zonos3.deinit();
        sets.glm_moe.deinit();
        sets.qwen_tree_ab.destroy();
        sets.qwen_tc_ab.destroy();
        sets.qwen_pages_ab.destroy();
        sets.nemo_direct_attn.destroy();
        sets.flashnext_attn_ab.destroy();
        sets.qwen_prompt_ab.destroy();
        sets.prefill_full_ab.destroy();
        sets.hy4_chunks.destroy();
        sets.glm_chunks_fp8.destroy();
        sets.glm_dflash_attn.destroy();
        sets.qwen35_draft_attn.destroy();
        sets.drama3_codec_attn.destroy();
        sets.drama3_attn.destroy();
        sets.idol_decoder.deinit();
        sets.idol.deinit();
        sets.joint_schema.deinit();
        sets.muse.destroy(sets.a);
        sets.prompt.deinit();
        sets.gpa.destroy(sets.prompt);
        sets.shared_down.deinit();
        sets.many.deinit();
        sets.gpa.destroy(sets.many);
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
        sets.ernie5_5_head.deinit();
        sets.bf16_experts.deinit();
        sets.mxfp4_experts.deinit();
    }

    /// Initializes `First` then `Second`, ensuring `First.deinit()` runs if `Second.init()` fails.
    pub fn initPair(comptime First: type, comptime Second: type, gpu: anytype, s: anytype, a: std.mem.Allocator, list: anytype) !struct { *First, *Second } {
        const first = try First.init(gpu, s, a, list);
        errdefer first.deinit();
        const second = try Second.init(gpu, s, a, list);
        return .{ first, second };
    }
};

test "contract family sets: ZONOS3 cleans up when the next family set fails to initialize" {
    const FakeFirst = struct {
        cleaned: *usize,
        pub fn init(cleaned: *usize, _: void, _: std.mem.Allocator, _: void) !*@This() {
            const p = try std.testing.allocator.create(@This());
            p.* = .{ .cleaned = cleaned };
            return p;
        }
        pub fn deinit(self: *@This()) void {
            self.cleaned.* += 1;
            std.testing.allocator.destroy(self);
        }
    };
    const FailingSecond = struct {
        pub fn init(_: *usize, _: void, _: std.mem.Allocator, _: void) !*@This() {
            return error.InitFailed;
        }
    };
    var cleaned: usize = 0;
    try std.testing.expectError(error.InitFailed, Sets.initPair(FakeFirst, FailingSecond, &cleaned, {}, std.testing.allocator, {}));
    try std.testing.expectEqual(@as(usize, 1), cleaned);
}
