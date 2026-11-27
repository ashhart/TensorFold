//! Canonical runtime catalogs and complete literal source sets.
const std = @import("std");
const ks = @import("kernel_sources");
const mtl = @import("metal");
const Set = @import("set.zig").Set;
const nk = @import("../src/families/nemotron/kernels.zig");

pub fn add(s: *Set, device: @import("metal").Device) !void {
    for (ks.nemotron.all) |k| {
        const text = if (!s.units and std.mem.startsWith(u8, k.key, "attn_partial"))
            try @import("../src/families/nemotron/simd_attention.zig").forGeneration(s.a, s.generation, k.source)
        else
            k.source;
        try s.raw("nemotron", k.key, text);
        if (!s.units and std.mem.startsWith(u8, k.key, "attn_partial")) {
            for ([_]u32{ 13, 15 }) |generation|
                try s.raw("nemotron", k.key, try @import("../src/families/nemotron/simd_attention.zig").forGeneration(s.a, generation, k.source));
        }
    }
    try s.raw("nemotron", "expert-geometries-up", nk.geo_up);
    try s.raw("nemotron", "expert-geometries-down", nk.geo_down);
    for (ks.prefill) |f| try s.fragments("nemotron-prompt", f.name, f.text);
    try s.fragments("nemotron-prompt", "masked-contiguous", ks.nemotron_prompt_masked.contiguous);
    try s.fragments("nemotron-prompt", "masked-paged", ks.nemotron_prompt_masked.paged_source);
    try s.raw("nemotron-paged", "glue", ks.nemotron_paged.glue);
    try s.raw("nemotron-paged", "partial", if (s.units) ks.nemotron_paged.partial else try @import("../src/families/nemotron/simd_attention.zig").rewrite(s.a, ks.nemotron_paged.partial));
    try s.raw("nemotron-paged", "tree", if (s.units) ks.nemotron_paged.tree else "#define TF_SIMD_LAYOUT 1\n" ++ ks.nemotron_paged.tree);
    try s.fragments("nemotron-paged", "prompt", ks.nemotron_paged.prefill);
    for (ks.qwen35.all) |k| try s.raw("qwen35", k.key, k.source);
    try s.raw("qwen35", "layout", ks.qwen35_layout);
    try s.rawFrom("qwen35", "paged", &.{ ks.qwen35_layout, ks.kv_pages, ks.qwen35_paged }, ks.qwen35_paged);
    inline for (.{ "nemotron_glue", "nemotron_experts", "nemotron_round", "nemotron_head", "nemotron_sample", "train", "train_mixers" }) |name|
        try s.raw("nemotron", name, @field(ks, name));
    try s.rawFrom("nemotron", "tree", &.{ks.nemotron_tree}, if (s.units) ks.nemotron_tree else "#define TF_SIMD_LAYOUT 1\n" ++ ks.nemotron_tree);
    inline for (.{ "glm_dsa_mla", "glm_dsa_index", "glm_dsa_glue", "glm_dsa_moe_jobs", "glm_ep" }) |name|
        try s.raw("glm-dsa", name, @field(ks, name));
    inline for (.{ "glm_vision", "core_qwen_vision", "qwen_image", "step5_vision" }) |name|
        try s.fragments("vision-image", name, @field(ks, name));
    if (@import("../src/families/moss_image/kernels.zig").available(device)) {
        try s.fragments("vision-image", "moss_image", ks.moss_image);
        try s.fragments("vision-image", "moss_image_decoder", ks.moss_image_decoder);
    }
    try @import("lfm2.zig").add(s, device);
    try s.rawFrom("lfm3", "base-attention", &.{ ks.lfm3, ks.lfm3_attention }, ks.lfm3 ++ "\n" ++ ks.lfm3_attention);
    try s.raw("lfm3", "image", ks.lfm3_image_rows);
    try s.rawFrom("lfm3", "row-projection", &.{ ks.core_row_projection, ks.lfm3_projection }, try @import("../src/families/lfm3/row_projection.zig").source(s.a));
    try s.rawFrom("lfm3", "dual-projection", &.{ ks.core_row_projection, ks.lfm3_projection_dual }, try @import("../src/families/lfm3/dual_projection.zig").source(s.a));
    const h3 = @import("../src/families/h3/source.zig");
    if (h3.available(device)) {
        const kernels = @import("h3_kernels");
        try s.rawFrom("h3", "rows-route-attention", &.{ gemmSource(), kernels.source, kernels.gemm_source, ks.nax }, try h3.library(s.a, 56, 18432, 288, ""));
    }

    try s.raw("hy4-flash", "elementwise", ks.elementwise);
    var precise = mtl.mlx();
    precise.math_functions = .precise;
    try s.add("h3-image", "encoder", @import("../kernels/metal/h3/image_encoder.zig").source, precise);
    try s.raw("h3-image", "prompt", @import("../kernels/metal/h3/prompt_kernels.zig").source);
    try s.raw("h3-image", "decoder", @import("../kernels/metal/h3/image_decoder.zig").source);
    const ling = @import("../kernels/metal/ling/sources.zig");
    try s.raw("ling", "route", ling.route);
    try s.raw("ling", "mla", ling.mla);
    try s.raw("ling", "lfm3", ks.lfm3);
}

fn gemmSource() []const u8 {
    inline for (ks.prefill) |f| if (comptime std.mem.eql(u8, f.name, "gemm_nax")) return f.text;
    @compileError("no gemm_nax source");
}
