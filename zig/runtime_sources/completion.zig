//! Production libraries whose loaders assemble sources outside the main family catalogs.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const Set = @import("set.zig").Set;

fn precise() mtl.CompileOptions {
    return .{ .language = mtl.mlx().language, .math_functions = .precise };
}

pub fn add(s: *Set, device: mtl.Device) !void {
    try direct(s);
    try deepseek(s);
    try hy4(s, device);
    try flashnext(s);
    try vision(s);
    try ricursive(s);
    try flint(s);
    try glm(s);
    try inkling2(s);
}

fn direct(s: *Set) !void {
    try s.raw("core", "core_exl3_rows", ks.core_exl3_rows);
    inline for (.{
        "hy4_flash_indexpool",
        "hy4_flash_index_recall",
        "hy4_flash_index_build",
        "hy4_flash_attention_sink",
        "hy4_flash_mla_project",
        "hy4_flash_mla_attention",
        "hy4_flash_ihc",
        "hy4_flash_engram",
        "hy4_flash_affine_experts",
        "hy4_flash_tensor_dispatch",
    }) |name| try s.raw("hy4-flash", name, @field(ks, name));
    try s.raw("lfm3", "lfm3_vision", ks.lfm3_vision);
    try s.raw("uocr", "uocr", ks.uocr);
    try s.raw("uocr", "uocr_decoder", ks.uocr_decoder);
    try s.add("qwen3", "qwen3_heads", ks.qwen3_heads, precise());
    inline for (.{ "glm_draft_topk", "glm_draft_accept", "glm_draft_follow", "glm_dflash_opt" }) |name|
        try s.raw("glm-dflash2", name, @field(ks, name));
}

fn deepseek(s: *Set) !void {
    try s.raw("deepseek-v41", "mv_mma", ks.dsv41.mv_mma);
    try s.raw("deepseek-v41", "vision", ks.dsv41.vision);
    try s.raw("deepseek-v41", "experts_exl3", @import("../src/families/deepseek_v41/metal/experts_exl3.zig").text);
    try s.raw("deepseek-v41", "early", ks.dsv41.keyed ++ ks.dsv41.early);
    try s.raw("deepseek-v41", "markov", ks.dsv41.keyed ++ ks.dsv41.markov);
}

fn hy4(s: *Set, device: mtl.Device) !void {
    try s.fragments("hy4-flash", "hy4_flash_mla_nax", ks.hy4_flash_mla_nax);
    try s.raw("hy4-flash", "rows", try std.mem.concat(s.a, u8, &.{
        @import("../src/core/row_projection.zig").source(true),
        ks.hy4_flash_rows,
    }));

    const contracts = @import("../src/families/hy4_flash/contract_exports.zig");
    const parsed = try @import("../src/families/hy4_flash/config.zig").parse(s.a, contracts.config_json);
    defer parsed.deinit();
    const t = parsed.value.text_config;
    const routed = @import("../src/families/hy4_flash/metal_routed.zig");
    const route = @import("../src/core/moe_route.zig");
    const core = try route.source(s.a, .{
        .hidden = t.hidden_size,
        .experts = t.n_routed_experts,
        .topk = t.num_experts_per_tok,
        .max_rows = routed.max_rows,
    });
    try s.raw("hy4-flash", "route-rows", try std.mem.concat(s.a, u8, &.{ core, ks.hy4_flash_route_rows }));
    for ([_]u8{ 4, 8 }) |bits|
        try s.raw("hy4-flash", try std.fmt.allocPrint(s.a, "tensor-experts-b{d}", .{bits}), try @import("../src/core/affine_mm.zig").source(device, s.a, .{
            .bits = bits,
            .group = 64,
            .out_f32 = true,
            .glu = .clipped_silu,
            .mapped_experts = true,
        }));
}

fn flashnext(s: *Set) !void {
    try s.raw("flashnext", "tfq3", try @import("../src/families/flashnext/tfq3.zig").buildSource(s.a, s.units));
    const vision_contract = @import("../src/families/flashnext/vision_contract.zig");
    for ([_]vision_contract.Which{ .mrope, .target, .pooled, .absolute, .positioned }) |which|
        try s.raw("flashnext-probe", @tagName(which), try std.mem.concat(s.a, u8, &.{ vision_contract.source(which), "\n", ks.flashnext_rotary_sweep }));
}

fn vision(s: *Set) !void {
    try s.fragments("idol", "idol", ks.idol);
    try s.fragments("idol", "idol_dino", ks.idol_dino);
    try s.fragments("idol", "idol_decoder", ks.idol_decoder);
    try s.raw("idol", "idol_vae", ks.idol_vae);
    try s.fragments("moss-image", "moss_image", ks.moss_image);
    try s.fragments("moss-image", "moss_image_decoder", ks.moss_image_decoder);
}

fn ricursive(s: *Set) !void {
    const contracts = @import("../src/families/ricursive/contracts.zig");
    const model = @import("../src/families/ricursive/model.zig");
    try s.add("ricursive", "ricursive_glue", try contracts.glueSource(s.a), precise());
    try s.add("ricursive", "llama_glue", try model.glueSource(s.a, "#define LL_DH {d}\n{s}", ks.llama_glue), precise());
    const head = try contracts.attentionHeader(s.a);
    try s.fragments("ricursive", "attention", try std.mem.concat(s.a, u8, &.{ head, ks.ricursive_attention_nax }));
    try s.add("ricursive", "ricursive_fp4", ks.ricursive_fp4, precise());
    try s.fragments("ricursive", "ricursive_fp4_gemm", ks.ricursive_fp4_gemm);
    try s.add("ricursive", "ricursive_dflash2", ks.ricursive_dflash2, precise());
}

fn flint(s: *Set) !void {
    const config = @import("../src/families/flint/config.zig");
    const state = @import("../src/families/flint/state.zig");
    const c: config.Config = .{};
    try s.add("flint", "flint_glue", ks.flint_glue, precise());
    try s.add("flint", "route", try @import("../src/core/moe_route.zig").source(s.a, .{
        .hidden = c.hidden,
        .experts = c.experts,
        .topk = c.topk,
        .round_logits = true,
    }), precise());
    for ([_]bool{ true, false }) |sliding| {
        const head = try std.fmt.allocPrint(s.a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define WINDOW {d}\n#define RING {d}\n#define SCALE {e}f\n", .{
            c.head_dim,
            c.heads,
            c.heads / c.kv_heads,
            c.kv_heads,
            if (sliding) c.window else 0,
            if (sliding) state.slidingRing(&c) else 0,
            1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim))),
        });
        try s.fragments("flint", if (sliding) "attention-sliding" else "attention-full", try std.mem.concat(s.a, u8, &.{ head, ks.flint_attention_nax }));
    }
    for (ks.nemotron.all) |kernel| if (std.mem.eql(u8, kernel.key, "sample")) {
        const vocab = try std.fmt.allocPrint(s.a, "constexpr int V = {d};", .{c.vocab});
        try s.raw("flint", "sampler", try std.mem.replaceOwned(u8, s.a, kernel.source, "constexpr int V = 131072;", vocab));
        break;
    };
    try s.raw("flint", "row-projection", @import("../src/core/row_projection.zig").source(s.units));
}

fn glm(s: *Set) !void {
    const stock = @import("../src/families/glm/metal_dflash_config.zig").stock;
    const attention = try @import("../src/families/glm/metal_dflash_attention.zig").source(s.a, .{
        .heads = stock.heads,
        .kv_heads = stock.kv_heads,
        .dim = stock.head_dim,
    });
    try s.add("glm-dflash2", "glm_draft_attention", attention, precise());
    const wide = @import("../src/families/glm/wide.zig");
    try s.raw("glm", "tfq3", try @import("../src/families/glm/tfq3.zig").buildSource(s.a, s.units, wide.glue_math));
    const m3_def = if (s.units) "" else "#define TF_GLM_M3_FAST 1\n";
    try s.raw("glm", "tfq3-kda-q8", try std.mem.concat(s.a, u8, &.{
        comptime @import("../src/families/glm/kernels.zig").generatedSource("kda_rows"),
        m3_def,
        ks.glm_kda_prompt,
        ks.glm_kda_window,
        ks.glm_kda_parts,
        ks.glm_tfq3.kda_q8,
    }));
}

fn inkling2(s: *Set) !void {
    const src = @import("inkling2_metal");
    try s.raw("inkling2", "attention", src.attention);
    try s.raw("inkling2", if (s.units) "prompt_attention" else "prompt_attention_pre_m5", if (s.units) src.prompt_attention else src.prompt_attention_pre_m5);
    try s.raw("inkling2", if (s.units) "dense_tiled" else "dense_mma", if (s.units) src.dense_tiled else src.dense_mma);
    inline for (.{ "glue", "experts", "experts_mma", "loop", "loop_mma", "tree", "runtime" }) |name|
        try s.raw("inkling2", name, @field(src, name));
    try s.fragments("inkling2", "vision", src.vision);
    try s.add("inkling2", "inkling2_dflash2", src.inkling2_dflash2, precise());
    try s.raw("inkling2", if (s.units) "inkling2_dspark" else "inkling2_dspark_pre_m5", if (s.units) src.inkling2_dspark else src.inkling2_dspark_pre_m5);
}
