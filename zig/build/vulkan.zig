//! Vulkan, built only with -Dvulkan=true: host tests in `test`, named GPU steps; options on every target (#1165).

const std = @import("std");
const vulkan_stages = @import("vulkan_stages.zig");

/// Vulkan's options as given, declared whether or not -Dvulkan=true builds anything with them.
pub const Options = struct { include: ?[]const u8, glslang: ?[]const u8, stage1_root: ?[]const u8, contract_root: ?[]const u8 };

pub fn options(b: *std.Build) Options {
    return .{
        .include = b.option([]const u8, "vulkan-include", "Directory holding vulkan/vulkan_core.h and vk_video/ (default /usr/include)"),
        .glslang = b.option([]const u8, "glslang", "glslang for the Vulkan shaders (default: glslang on PATH)"),
        .stage1_root = b.option([]const u8, "stage1-test-root", "Root .zig of the stage 1 GPU probe (default zig/tests/vulkan/stage1_probe.zig)"),
        .contract_root = b.option([]const u8, "contract-root", "Root .zig of the Vulkan kernel-contract runner (default zig/tests/vulkan/contract_main.zig)"),
    };
}

/// vulkan-host-test (part of `test`), vulkan-gpu-build and vulkan-gpu-test, vulkan-probe-build and vulkan-probe.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize, test_step: *std.Build.Step, o: Options) void {
    var ctx = vulkan_stages.Context{
        .b = b,
        .target = target,
        .optimize = optimize,
        .o = o,
        .glslang = o.glslang orelse "glslang",
        .kernel_dir = b.path("zig/kernels/vulkan"),
        .vulkan = module(b, target, optimize, o),
    };
    vulkan_stages.hostAdmission(&ctx);
    test_step.dependOn(ctx.host);
    vulkan_stages.probeSteps(&ctx);
    vulkan_stages.stage1(&ctx);
    vulkan_stages.stage2(&ctx);
    vulkan_stages.stage3(&ctx);
    vulkan_stages.stage4(&ctx);
    vulkan_stages.stage5(&ctx);
    vulkan_stages.contracts(&ctx);
    vulkan_stages.kernelContract(&ctx);
    // Family host admission: refAllDeclsRecursive over the family modules; house wires attach in build.zig.
    const model_host_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/vulkan_model_host_tests.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = module(b, b.graph.host, .debug, ctx.o) },
            .{ .name = "core", .module = coreModule(b, b.graph.host, .debug) },
        },
    }) });
    ctx.host.dependOn(&b.addRunArtifact(model_host_tests).step);
    b.step("model-host-test", "Qwen3.5 Vulkan family weights/state host tests (TF_QWEN_DIR real-load test stays skipped without the env var)").dependOn(&b.addRunArtifact(model_host_tests).step);
}

const ShaderVariant = struct { name: []const u8, src: []const u8, defines: []const []const u8 };

/// The stage 1 closure's shader images: one variant per op path the host code names in ops.Shaders.
pub const shader_variants = [_]ShaderVariant{
    .{ .name = "qmm_bf16", .src = "zig/kernels/vulkan/qmm.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "qmm_q4_vec4", .src = "zig/kernels/vulkan/qmm_q4_vec4.comp", .defines = &.{} },
    .{ .name = "dequant_bf16", .src = "zig/kernels/vulkan/dequant.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "gather_u32", .src = "zig/kernels/vulkan/gather_take.comp", .defines = &.{"-DUSE_U32"} },
    .{ .name = "gather_u16", .src = "zig/kernels/vulkan/gather_take.comp", .defines = &.{"-DUSE_U16"} },
    .{ .name = "cast_bf16_f32", .src = "zig/kernels/vulkan/cast.comp", .defines = &.{"-DSOURCE_BF16"} },
    .{ .name = "cast_f32_bf16", .src = "zig/kernels/vulkan/cast.comp", .defines = &.{"-DDEST_BF16"} },
    .{ .name = "copy_u32", .src = "zig/kernels/vulkan/copy_general.comp", .defines = &.{"-DUSE_U32"} },
    .{ .name = "copy_u16", .src = "zig/kernels/vulkan/copy_general.comp", .defines = &.{"-DUSE_U16"} },
};

pub const stage1_shaders_src =
    \\pub const qmm_bf16 align(4) = @embedFile("qmm_bf16.spv").*;
    \\pub const qmm_q4_vec4 align(4) = @embedFile("qmm_q4_vec4.spv").*;
    \\pub const dequant_bf16 align(4) = @embedFile("dequant_bf16.spv").*;
    \\pub const gather_u32 align(4) = @embedFile("gather_u32.spv").*;
    \\pub const gather_u16 align(4) = @embedFile("gather_u16.spv").*;
    \\pub const cast_bf16_f32 align(4) = @embedFile("cast_bf16_f32.spv").*;
    \\pub const cast_f32_bf16 align(4) = @embedFile("cast_f32_bf16.spv").*;
    \\pub const copy_u32 align(4) = @embedFile("copy_u32.spv").*;
    \\pub const copy_u16 align(4) = @embedFile("copy_u16.spv").*;
;

/// Stage 2's images: norm/swiglu in the ASH_QWEN35 oracle form; add_preserve_bf16 is add plus DenormPreserve (f32).
pub const stage2_variants = [_]ShaderVariant{
    .{ .name = "norm_bf16", .src = "zig/kernels/vulkan/fast_norm.comp", .defines = &.{ "-DUSE_BF16", "-DASH_QWEN35" } },
    .{ .name = "add_bf16", .src = "zig/kernels/vulkan/binary_vec.comp", .defines = &.{} },
    .{ .name = "add_preserve_bf16", .src = "zig/kernels/vulkan/binary_vec.comp", .defines = &.{"-DPRESERVE_F32"} },
    .{ .name = "swiglu_bf16", .src = "zig/kernels/vulkan/swiglu.comp", .defines = &.{ "-DUSE_BF16", "-DASH_QWEN35" } },
};

pub const stage2_shaders_src =
    \\pub const norm_bf16 align(4) = @embedFile("norm_bf16.spv").*;
    \\pub const add_bf16 align(4) = @embedFile("add_bf16.spv").*;
    \\pub const add_preserve_bf16 align(4) = @embedFile("add_preserve_bf16.spv").*;
    \\pub const swiglu_bf16 align(4) = @embedFile("swiglu_bf16.spv").*;
;

/// Stage 3's images: the ASH_QWEN35 rope variant, the causal multi-row SDPA decode arm and the attention output gate.
pub const stage3_variants = [_]ShaderVariant{
    .{ .name = "rope_bf16", .src = "zig/kernels/vulkan/fast_rope.comp", .defines = &.{ "-DUSE_BF16", "-DASH_QWEN35" } },
    .{ .name = "sdpa_rows_bf16", .src = "zig/kernels/vulkan/sdpa_decode_native.comp", .defines = &.{ "-DBF16_IO", "-DSDPA_DIM=256", "-DSDPA_ROWS" } },
    .{ .name = "gate_bf16", .src = "zig/kernels/vulkan/attention_gate.comp", .defines = &.{} },
};

pub const stage3_shaders_src =
    \\pub const rope_bf16 align(4) = @embedFile("rope_bf16.spv").*;
    \\pub const sdpa_rows_bf16 align(4) = @embedFile("sdpa_rows_bf16.spv").*;
    \\pub const gate_bf16 align(4) = @embedFile("gate_bf16.spv").*;
;

/// Stage 4's images: GDN conv, gate split, prefill scan, decode, RMSNorm epilogue and the bf16 copy, all -DUSE_BF16.
pub const stage4_variants = [_]ShaderVariant{
    .{ .name = "conv_dw1d_bf16", .src = "zig/kernels/vulkan/conv_dw1d.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "gdn_conv_decode_bf16", .src = "zig/kernels/vulkan/gdn_conv_decode.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "gdn_split_bf16", .src = "zig/kernels/vulkan/gdn_split.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "prefill_recur_bf16", .src = "zig/kernels/vulkan/gated_delta_prefill_ash.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "gated_delta_decode_bf16", .src = "zig/kernels/vulkan/gated_delta_decode.comp", .defines = &.{ "-DUSE_BF16", "-DASH_QWEN35" } },
    .{ .name = "fast_norm_gated_bf16", .src = "zig/kernels/vulkan/fast_norm_gated.comp", .defines = &.{ "-DUSE_BF16", "-DASH_QWEN35" } },
    .{ .name = "copy_u16", .src = "zig/kernels/vulkan/copy_general.comp", .defines = &.{"-DUSE_U16"} },
};

pub const stage4_shaders_src =
    \\pub const conv_dw1d_bf16 align(4) = @embedFile("conv_dw1d_bf16.spv").*;
    \\pub const gdn_conv_decode_bf16 align(4) = @embedFile("gdn_conv_decode_bf16.spv").*;
    \\pub const gdn_split_bf16 align(4) = @embedFile("gdn_split_bf16.spv").*;
    \\pub const prefill_recur_bf16 align(4) = @embedFile("prefill_recur_bf16.spv").*;
    \\pub const gated_delta_decode_bf16 align(4) = @embedFile("gated_delta_decode_bf16.spv").*;
    \\pub const fast_norm_gated_bf16 align(4) = @embedFile("fast_norm_gated_bf16.spv").*;
    \\pub const copy_u16 align(4) = @embedFile("copy_u16.spv").*;
;

/// Stage 5's images: the greedy gate's three passes; argreduce_chunk is -DUSE_BF16, combine reduces the f32 partials.
pub const stage5_variants = [_]ShaderVariant{
    .{ .name = "finite_logits", .src = "zig/kernels/vulkan/finite_logits.comp", .defines = &.{} },
    .{ .name = "argreduce_chunk", .src = "zig/kernels/vulkan/argreduce_chunk.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "argreduce_combine", .src = "zig/kernels/vulkan/argreduce_combine.comp", .defines = &.{} },
};

pub const stage5_shaders_src =
    \\pub const finite_logits align(4) = @embedFile("finite_logits.spv").*;
    \\pub const argreduce_chunk align(4) = @embedFile("argreduce_chunk.spv").*;
    \\pub const argreduce_combine align(4) = @embedFile("argreduce_combine.spv").*;
;

/// The qwen3.5 Vulkan server's modules: the family root, the embedded SPIR-V bundle, the native_engines root.
pub fn serverModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize, o: Options, lanes: *std.Build.Module) struct { qwen35_vulkan: *std.Build.Module, qwen35_shaders: *std.Build.Module, native_engines: *std.Build.Module, engine_api: *std.Build.Module, core: *std.Build.Module } {
    const vulkan = module(b, target, optimize, o);
    const core = coreModule(b, target, optimize);
    const qwen35_vulkan = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/vulkan.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "vulkan", .module = vulkan },
        .{ .name = "lanes", .module = lanes },
        .{ .name = "core", .module = core },
    } });
    // one glslang invocation per image, all embedded under their bundle names
    const glslang = o.glslang orelse "glslang";
    const kernel_dir = b.path("zig/kernels/vulkan");
    const files = b.addWriteFiles();
    var src: std.ArrayListUnmanaged(u8) = .empty;
    for (server_shader_images) |img| {
        const cmd = b.addSystemCommand(&.{ glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-I." });
        cmd.setCwd(kernel_dir);
        for (img.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{img.name}));
        cmd.addFileArg(b.path(img.src));
        _ = files.addCopyFile(spv, b.fmt("{s}.spv", .{img.name}));
        src.appendSlice(b.allocator, b.fmt("pub const {s} align(4) = @embedFile(\"{s}.spv\").*;\n", .{ img.name, img.name })) catch @panic("OOM");
    }
    const qwen35_shaders = b.createModule(.{ .root_source_file = files.add("qwen35_shaders.zig", src.items) });
    const engine_api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const native_engines = b.createModule(.{ .root_source_file = b.path("zig/src/native/vulkan.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "engine_api", .module = engine_api },
        .{ .name = "lanes", .module = lanes },
        .{ .name = "qwen35_vulkan", .module = qwen35_vulkan },
        .{ .name = "qwen35_shaders", .module = qwen35_shaders },
    } });
    return .{ .qwen35_vulkan = qwen35_vulkan, .qwen35_shaders = qwen35_shaders, .native_engines = native_engines, .engine_api = engine_api, .core = core };
}

/// Every image the bundle embeds: the real stage arrays, concatenated; names are what the adapter's bundle() reads.
const server_shader_images = shader_variants ++ stage2_variants ++ stage3_variants ++ stage4_variants[0 .. stage4_variants.len - 1] ++ stage5_variants;

/// The house core module the family loader's config/shard imports resolve through, built exactly as cuda.zig builds it.
fn coreModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize) *std.Build.Module {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", tokenizer);
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("living", living);
    return core;
}

/// The backend over `vk`, vulkan_core.h translated without prototypes: every entry point comes from the loader.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize, o: Options) *std.Build.Module {
    const header = b.addWriteFiles().add("vulkan_c.h", "#define VK_NO_PROTOTYPES\n#include <vulkan/vulkan_core.h>\n");
    const vk = b.addTranslateC(.{ .root_source_file = header, .target = target, .optimize = optimize });
    // after the target's libc headers, so a cross build never takes the host's stdint.h
    vk.addAfterIncludePath(.{ .cwd_relative = o.include orelse "/usr/include" });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/vulkan/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("vk", vk.createModule());
    m.addImport("stage1_params", b.createModule(.{ .root_source_file = b.path("zig/src/vulkan/params.zig"), .target = target, .optimize = optimize }));
    // Host-test-only .comp sources: flag-pin tests match push flags against literal masks; shipped ELFs embed SPIR-V.
    const comp_files = b.addWriteFiles();
    inline for (.{ "gated_delta_decode", "gated_delta_prefill_ash", "fast_rope", "gdn_conv_decode" }) |name|
        _ = comp_files.addCopyFile(b.path("zig/kernels/vulkan/" ++ name ++ ".comp"), name ++ ".comp");
    m.addImport("vulkan_comp_sources", b.createModule(.{ .root_source_file = comp_files.add("comp_sources.zig",
        \\pub const gated_delta_decode = @embedFile("gated_delta_decode.comp");
        \\pub const gated_delta_prefill_ash = @embedFile("gated_delta_prefill_ash.comp");
        \\pub const fast_rope = @embedFile("fast_rope.comp");
        \\pub const gdn_conv_decode = @embedFile("gdn_conv_decode.comp");
        \\
    ) }));
    return m;
}
