//! Model geometry specializations call the same source factories as their loaders.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const Set = @import("set.zig").Set;
const g4 = @import("../src/families/gemma4/kernels.zig");
const gcfg = @import("../src/families/gemma4/config.zig");
const gmm = @import("../src/families/gemma4/affine_mm.zig");
pub fn add(s: *Set, device: mtl.Device) !void {
    const precise = mtl.CompileOptions{ .language = mtl.mlx().language, .math_functions = .precise };
    for ([_]gcfg.Geometry{ gcfg.dense31b, gcfg.unified12b }) |geo| {
        for ([_]bool{ false, true }) |global| {
            for ([_]bool{ false, true }) |assistant| {
                if (assistant and !g4.draftAvailable(device)) continue;
                for (gcfg.widths.bit_widths) |bits| try s.rawFrom("gemma4", try std.fmt.allocPrint(s.a, "{s}-{}-{}-embed{d}", .{ geo.name, global, assistant, bits }), &.{ks.gemma4_glue}, try g4.glueSource(s.a, geo, global, assistant, bits));
            }
            try s.rawFrom("gemma4", "prompt-attention", &.{ ks.gemma4_prompt_attention, ks.nax }, try g4.promptSource(device, s.a, geo, global));
        }
        try s.rawFrom("gemma4", "vision", &.{ ks.gemma4_vision, ks.nax }, try @import("../src/families/gemma4/vision_source.zig").source(device, s.a, geo.hidden));
    }
    for (gcfg.widths.bit_widths) |bits| for ([_]u16{ 32, 64 }) |group| for ([_]bool{ false, true }) |out_f32| for (std.enums.values(gmm.Glu)) |glu|
        try s.rawFrom("gemma4", "affine", &.{ ks.gemma4_affine_mm, ks.nax }, try gmm.source(device, s.a, .{ .bits = bits, .group = group, .out_f32 = out_f32, .glu = glu }));
    for (@import("../src/families/gemma4/contracts.zig").entries) |v| if (v.kind != 0 and @import("../src/families/gemma4/fused_lane.zig").available(device))
        try s.fragmentsFrom("gemma4", v.name, &.{ ks.core_lane_projection, ks.gemma4_fused, ks.gemma4_lane_projection }, try @import("../src/families/gemma4/fused_lane.zig").source(s.a, v.layout()));
    const ll = @import("../src/families/llama/contracts.zig");
    if (ll.available(device)) {
        try s.addFrom("llama", "glue", &.{ks.llama_glue}, try ll.glueSource(s.a), precise);
        for (ll.geometries) |geo| try s.rawFrom("llama", geo.name, &.{ ks.llama_attention_nax, ks.nax }, try ll.attentionSource(device, s.a, geo));
        try s.rawFrom("llama", "gemm", &.{ ks.llama_gemm, ks.nax }, try ll.gemmSource(device, s.a));
    }
    try s.addFrom("step5", "glue-sparse", &.{ ks.core_affine_embed, ks.step5_glue, ks.step5_sparse }, ks.step5_glue ++ ks.step5_sparse, precise);
    const step: @import("../src/families/step5/config.zig").Config = .{};
    const route = @import("../src/core/moe_route.zig");
    try s.addFrom("step5", "route", &.{ks.core_moe_route}, try route.source(s.a, .{ .hidden = step.hidden, .experts = step.experts, .topk = step.topk }), precise);
    const dsa: @import("../src/families/glm_dsa/config.zig").Config = .{};
    try s.rawFrom("glm-dsa", "route", &.{ks.core_moe_route}, try route.source(s.a, .{ .hidden = dsa.hidden, .experts = dsa.experts, .topk = dsa.topk }));
    const step_attn = try @import("../src/core/frags.zig").sourceMode(device, s.a, ks.step5_attend_nax, false);
    try s.addFrom("step5", "attention", &.{ ks.step5_attend_nax, ks.nax }, step_attn, precise);
    const sp = @import("../src/families/spark25/metal_model.zig");
    try s.addFrom("spark25", "glue", &.{ ks.core_affine_embed, ks.spark25_glue }, ks.spark25_glue, precise);
    for ([_]bool{ false, true }) |sliding| try s.rawFrom("spark25", "attention", &.{ ks.spark25_attention_nax, ks.nax }, try sp.attentionSource(device, s.a, sliding));
    try s.fragments("spark25", "gemm", ks.spark25_gemm);
    const mm = @import("../src/families/minimax_m3/model.zig");
    const c: @import("../src/families/minimax_m3/config.zig").Config = .{};
    try s.addFrom("minimax-m3", "glue", &.{ ks.core_affine_embed, ks.minimax_m3_glue }, ks.minimax_m3_glue, precise);
    try s.rawFrom("minimax-m3", "attention", &.{ ks.minimax_m3_attention_nax, ks.nax }, try mm.attentionSource(device, s.a, &c));
    try s.rawFrom("minimax-m3", "index", &.{ ks.minimax_m3_index_nax, ks.nax }, try mm.indexSource(device, s.a, &c));
    try s.rawFrom("minimax-m3", "sampler", &.{ks.nemotron_sample}, try mm.samplerSource(s.a, &c));
    try s.raw("minimax-m3", "tap-draft-vec", ks.minimax_m3_dspark);
    try s.add("minimax-m3", "dspark", ks.minimax_m3_dspark, precise);
    try s.add("minimax-m3", "dflash2", ks.minimax_m3_dflash2, precise);
    try s.rawFrom("minimax-m3", "experts", &.{ ks.core_row_projection, ks.minimax_m3_experts }, ks.core_row_projection ++ "\n" ++ ks.minimax_m3_experts);
    const wide = @import("../src/families/glm/wide.zig");
    const kda = comptime @import("../src/families/glm/kernels.zig").generatedSource("kda_rows");
    try s.rawFrom("glm", "wide-rows-gather", &.{ ks.core_row_projection, ks.glm_glue, ks.glm_rows, ks.glm_gather }, wide.gather_source);
    try s.rawFrom("glm", "wide-kda-pre", &.{kda}, try wide.kdaPreSource(s.a, kda));
    try s.rawFrom("glm", "wide-kda-fg", &.{ ks.core_row_projection, kda }, try wide.kdaFgSource(s.a, kda));
}
