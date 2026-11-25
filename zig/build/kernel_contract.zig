//! The kernel contract's programs: tf-kernel-contract on Metal, and `zig build kernel-contract` that runs the gates.
const std = @import("std");

/// The contract module alone (its host tests run in `zig build test` through core/root.zig).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/core/contract.zig"), .target = target, .optimize = optimize });
    m.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    return m;
}

/// `test-kernel-contract` on every host: the gates' own tests and every registered launch against every chip's limits.
pub fn hostTests(b: *std.Build, family: *std.Build.Step) void {
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, .debug) }));
    run.has_side_effects = true;
    const attn = b.addTest(.{ .root_module = module(b, b.graph.host, .debug), .filters = &.{"XPU attention:"} });
    b.step("test-xpu-attn-contract-host", "Decode attention's float64 reference, tree, tail and sum fixtures (no GPU)").dependOn(&b.addRunArtifact(attn).step);
    b.step("test-kernel-seed-host", "Named contract fixtures keep their seed when entries are inserted").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = module(b, b.graph.host, .debug), .filters = &.{"contract seed:"} })).step);
    const step = b.step("test-kernel-contract", "The kernel contract's host tests: gate logic, controls, every launch against every chip");
    step.dependOn(&run.step);
    const q27_pages = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/q27_paged_cases.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "contract", .module = module(b, b.graph.host, .debug) },
            .{ .name = "kernel_sources", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") }) },
        },
    }) }));
    b.step("test-qwen27-paged-contract-host", "Paged prompt source controls, shuffled pages and boundary witnesses without a device").dependOn(&q27_pages.step);
    step.dependOn(&q27_pages.step);
    const inkling_controls = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/metal_inkling2_controls_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "contract", .module = module(b, b.graph.host, .debug) },
            .{ .name = "inkling2_metal", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/inkling2_sources.zig") }) },
            .{ .name = "inkling2_glue_plan", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/glue_plan.zig") }) },
            .{ .name = "inkling2_loop_plan", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/loop_plan.zig") }) },
            .{ .name = "inkling2_runtime_plan", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/runtime_plan.zig") }) },
            .{ .name = "inkling2_experts_plan", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/experts_plan.zig") }) },
        },
    }) }));
    b.step("test-inkling2-controls-host", "Inkling 2 Mini source controls plant one fault without a Metal compiler").dependOn(&inkling_controls.step);
    step.dependOn(&inkling_controls.step);
    const mapping_stub = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/metal_mapping_host.zig") });
    const llama_controls = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/metal_llama_controls.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "contract", .module = module(b, b.graph.host, .debug) },
            .{ .name = "metal", .module = mapping_stub },
            .{ .name = "tensorfold", .module = mapping_stub },
            .{ .name = "kernel_sources", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") }) },
        },
    }) }));
    b.step("test-llama-controls-host", "Llama's source controls plant one fault without a device or shader compiler").dependOn(&llama_controls.step);
    step.dependOn(&llama_controls.step);
    const ling_mla_controls = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/metal_ling_mla_controls.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "contract", .module = module(b, b.graph.host, .debug) },
            .{ .name = "ling_metal", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/ling/sources.zig") }) },
        },
    }) }));
    b.step("test-ling-mla-controls-host", "Ling MLA source controls plant one fault without a device or shader compiler").dependOn(&ling_mla_controls.step);
    step.dependOn(&ling_mla_controls.step);
    const mapping_tf = b.createModule(.{ .root_source_file = b.path("zig/lfm2_mapping_tf.zig"), .imports = &.{
        .{ .name = "metal", .module = mapping_stub },
        .{ .name = "kernel_sources", .module = mapping_stub },
    } });
    const mapping_root = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/metal_mapping_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "contract", .module = module(b, b.graph.host, .debug) },
            .{ .name = "inkling2_metal", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/inkling2_sources.zig") }) },
            .{ .name = "metal", .module = mapping_stub },
            .{ .name = "tensorfold", .module = mapping_tf },
            .{ .name = "kernel_sources", .module = mapping_stub },
        },
    });
    const mapping = b.addRunArtifact(b.addTest(.{ .root_module = mapping_root }));
    mapping.has_side_effects = true;
    b.step("test-metal-launch-map-host", "Every registered Metal launch maps without a device or shader compiler").dependOn(&mapping.step);
    const dspark_mapping = b.addRunArtifact(b.addTest(.{ .root_module = mapping_root, .filters = &.{"Metal launch map: every DSpark"} }));
    b.step("test-inkling2-launch-map-host", "Every Inkling DSpark launch resolves without other families' device checks").dependOn(&dspark_mapping.step);
    step.dependOn(&mapping.step);
    const precision = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/attention_precision.zig"),
        .target = b.graph.host,
        .optimize = .debug,
    }) }));
    b.step("test-attention-precision-host", "Shared attention precision mode, source and kept-state identity").dependOn(&precision.step);
    step.dependOn(&precision.step);
    const rig = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/contract_rig.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "cuda", .module = @import("cuda_runtime.zig").runtime(b, b.graph.host, .debug, &.{}) }},
    });
    const rig_test = b.addRunArtifact(b.addTest(.{ .root_module = rig, .filters = &.{"completed CUDA fixtures"} }));
    b.step("test-cuda-contract-lifetime-host", "Completed CUDA fixtures free buffers once and remain reusable").dependOn(&rig_test.step);
    step.dependOn(&rig_test.step);
    step.dependOn(family);
    const rotary = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/rotary_cache_positions.zig"),
        .target = b.graph.host,
        .optimize = .debug,
    }) }));
    b.step("test-rotary-cache-positions-host", "Cache-check positions match the consumed pooled axes without shaders").dependOn(&rotary.step);
    step.dependOn(&rotary.step);
    const zstd = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/contract/fatbin_zstd.zig"),
        .target = b.graph.host,
    }) }));
    b.step("test-fatbin-zstd-host", "Bounded compressed-image reader and encoded expansion controls").dependOn(&zstd.step);
    step.dependOn(&zstd.step);
    const delta = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/delta_slots_contract_test.zig"),
        .target = b.graph.host,
        .imports = &.{.{ .name = "contract", .module = module(b, b.graph.host, .debug) }},
    }) }));
    b.step("test-delta-slots-contract-host", "FP64 DeltaNet and slot corruption controls without shaders").dependOn(&delta.step);
    step.dependOn(&delta.step);
    const nvfp4_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/drama3_nvfp4_ref.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = module(b, b.graph.host, .debug) }},
    });
    const nvfp4 = b.addRunArtifact(b.addTest(.{ .root_module = nvfp4_module }));
    b.step("test-drama-nvfp4-reference-host", "NVFP4 scalar reference and reduced-accumulator discrimination").dependOn(&nvfp4.step);
    step.dependOn(&nvfp4.step);
    const nemotron = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/nemotron_experts_contract_host.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = module(b, b.graph.host, .debug) }},
    }) }));
    b.step("test-nemotron-experts-contract-host", "Nemotron routed-expert tables, float64 steps, stress and controls").dependOn(&nemotron.step);
    step.dependOn(&nemotron.step);
    const shared_gate = b.addRunArtifact(b.addTest(.{ .root_module = nvfp4_module, .filters = &.{"shared gate stress"} }));
    b.step("test-drama-combine-stress-host", "Shared-gate stress remains discriminating after nonlinear output").dependOn(&shared_gate.step);
}

/// A compile-only root of metal_main.zig gets tf-kernel-contract's Inkling 2 and H3 image modules, the engine's shared.
pub fn metalRootImports(b: *std.Build, engine: *std.Build.Module, root: *std.Build.Module) void {
    inline for (.{ "inkling2_metal", "inkling2_glue_plan", "inkling2_loop_plan" }) |name| root.addImport(name, engine.import_table.get(name).?);
    root.addImport("inkling2_runtime_plan", b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/runtime_plan.zig") }));
    for (@import("h3_image_contract.zig").sources(b)) |image| root.addImport(image.name, image.module);
}

/// macOS: tf-kernel-contract over the engine's Metal kernels; `kernel-contract` runs it on this Mac's GPU.
pub fn metal(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal_mod: *std.Build.Module, engine: *std.Build.Module, sources: *std.Build.Module, kimi_metallib: *std.Build.Module, test_step: *std.Build.Step) void {
    @import("rotary_sweep.zig").steps(b, target, metal_mod);
    @import("ling_forward.zig").steps(b, target, optimize, metal_mod, sources, kimi_metallib);
    const inkling2_metal = engine.import_table.get("inkling2_metal").?;
    const inkling2_glue_plan = engine.import_table.get("inkling2_glue_plan").?;
    const inkling2_loop_plan = engine.import_table.get("inkling2_loop_plan").?;
    const image = @import("h3_image_contract.zig").sources(b);
    const exe = b.addExecutable(.{ .name = "tf-kernel-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/metal_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal_mod },
            .{ .name = "tensorfold", .module = engine },
            .{ .name = "kernel_sources", .module = sources },
            .{ .name = "contract", .module = module(b, target, optimize) },
            .{ .name = "h3_kernels", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") }) },
            .{ .name = "h3_image_encoder_kernels", .module = image[0].module },
            .{ .name = "h3_prompt_kernels", .module = image[1].module },
            .{ .name = "h3_image_decoder_kernels", .module = image[2].module },
            .{ .name = "ling_metal", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/ling/sources.zig") }) },
            .{ .name = "ling_route", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/route.zig") }) },
            .{ .name = "masked_sources", .module = b.createModule(.{ .root_source_file = b.path("zig/masked_value_sources.zig"), .imports = &.{.{ .name = "kernel_sources", .module = sources }} }) },
            .{ .name = "inkling2_metal", .module = inkling2_metal },
            .{ .name = "inkling2_glue_plan", .module = inkling2_glue_plan },
            .{ .name = "inkling2_loop_plan", .module = inkling2_loop_plan },
            .{ .name = "inkling2_runtime_plan", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/inkling2/metal/runtime_plan.zig") }) },
        },
    }) });
    const install = b.addInstallArtifact(exe, .{});
    b.step("tf-kernel-contract", "Build the kernel contract's Metal gates and controls (no run)").dependOn(&install.step);
    const exl3_suites = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/metal_dsv41_exl3_row.zig"), .target = target, .optimize = optimize, .link_libc = true });
    for (exe.root_module.import_table.keys(), exe.root_module.import_table.values()) |name, m| exl3_suites.addImport(name, m);
    const exl3_suites_run = b.addRunArtifact(b.addTest(.{ .root_module = exl3_suites, .filters = &.{"dsv41 exl3 suite:"} }));
    b.step("test-dsv41-exl3-contract", "DeepSeek V4.1 EXL3 contract suites free all they took when an allocation fails").dependOn(&exl3_suites_run.step);
    const limits_run = b.addRunArtifact(exe);
    limits_run.addArg("--no-gates");
    limits_run.has_side_effects = true;
    b.step("metal-kernel-contract-limits", "Run all Metal launch limits on this GPU, without numerical gates").dependOn(&limits_run.step);
    const run = b.addRunArtifact(exe);
    b.step("kernel-contract", "Every registered kernel through the contract's gates on this GPU, each control red").dependOn(&run.step);
    const ling_gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/ling_route_gpu.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal_mod },
            .{ .name = "route", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/route.zig") }) },
            .{ .name = "ling_metal", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/ling/sources.zig") }) },
        },
    }) });
    b.step("test-ling-route-gpu", "Ling grouped route on this GPU against the host formula").dependOn(&b.addRunArtifact(ling_gpu).step);
    const ling_kda_gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/ling_kda_gpu.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal_mod },
            .{ .name = "kda", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/kda.zig") }) },
            .{ .name = "kimi_metallib", .module = kimi_metallib },
        },
    }) });
    b.step("test-ling-kda-gpu", "Ling KDA mixer on this GPU against kda.zig, using k3_kda").dependOn(&b.addRunArtifact(ling_kda_gpu).step);
    const ling_mla_gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/ling_mla_gpu.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal_mod },
            .{ .name = "mla", .module = b.createModule(.{ .root_source_file = b.path("zig/src/families/ling/mla.zig") }) },
            .{ .name = "kimi_metallib", .module = kimi_metallib },
            .{ .name = "ling_metal", .module = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/ling/sources.zig") }) },
        },
    }) });
    b.step("test-ling-mla-gpu", "Ling MLA mixer on this GPU around k3_mla, against mla.zig").dependOn(&b.addRunArtifact(ling_mla_gpu).step);
    // Inkling 2 Mini's dense-MMA tail: a partial row tile reads no row past its input, and its bits equal the matvec's.
    const dense_guard = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/contract/metal_inkling2_dense_guard.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "metal", .module = metal_mod },
            .{ .name = "tensorfold", .module = engine },
            .{ .name = "kernel_sources", .module = sources },
        },
    }) });
    const dense_guard_run = b.addRunArtifact(dense_guard);
    dense_guard_run.has_side_effects = true; // a device run: never cached
    // The GPU faults on an out-of-bounds read only with Metal's shader validation; it aborts on the fault.
    dense_guard_run.setEnvironmentVariable("MTL_SHADER_VALIDATION", "1");
    dense_guard_run.setEnvironmentVariable("MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "1");
    dense_guard_run.setEnvironmentVariable("MTL_SHADER_VALIDATION_ABORT_ON_FAULT", "1");
    b.step("test-inkling2-dense-guard", "Inkling 2 Mini's dense-MMA tail: no read past a partial tile's input, bits equal the matvec's").dependOn(&dense_guard_run.step);
    test_step.dependOn(&dense_guard_run.step);
}

/// Linux: `kernel-contract` runs tf-cuda-test's kernel-contract command (kernels from -Dnvcc or -Dfatbins).
pub fn cuda(b: *std.Build, runner: *std.Build.Step.Compile, controls: ?std.Build.LazyPath, h3: ?std.Build.LazyPath, h3_image: [2]?std.Build.LazyPath) *std.Build.Step {
    const opts = b.addOptions();
    opts.addOption(bool, "controls", controls != null);
    runner.root_module.addOptions("contract_options", opts);
    const empty = b.addWriteFiles().add("contract_controls.fatbin", "");
    runner.root_module.addAnonymousImport("fatbin_contract_controls", .{ .root_source_file = controls orelse empty });
    // MiniMax H3's image (the tf-h3 library's) for its kernels' registrations
    runner.root_module.addAnonymousImport("fatbin_h3", .{ .root_source_file = h3 orelse b.addWriteFiles().add("h3.fatbin", "") });
    const names = [2][]const u8{ "h3_image_decoder", "h3_image_prompt" };
    for (names, h3_image) |n, file| runner.root_module.addAnonymousImport(b.fmt("fatbin_{s}", .{n}), .{ .root_source_file = file orelse b.addWriteFiles().add(b.fmt("{s}.fatbin", .{n}), "") });
    const install = b.step("kernel-contract-fatbins", "Install all CUDA contract images (fatbin/)");
    if (controls) |file| install.dependOn(&b.addInstallFile(file, "fatbin/contract_controls.fatbin").step);
    if (h3) |file| install.dependOn(&b.addInstallFile(file, "fatbin/h3.fatbin").step);
    for (names, h3_image) |n, file| if (file) |f| install.dependOn(&b.addInstallFile(f, b.fmt("fatbin/{s}.fatbin", .{n})).step);
    const run = b.addRunArtifact(runner);
    run.addArg("kernel-contract");
    b.step("kernel-contract", "Every registered CUDA kernel through the contract's gates on this GPU, each control red").dependOn(&run.step);
    const prefill_many = b.addRunArtifact(runner);
    prefill_many.addArg("qmm-prefill-many-contract");
    b.step("qmm-prefill-many-contract", "GB10 bit-exact multi-projection prompt launch against separate launches").dependOn(&prefill_many.step);
    const prefill_tile22 = b.addRunArtifact(runner);
    prefill_tile22.addArg("qmm-prefill-tile22-contract");
    b.step("qmm-prefill-tile22-contract", "GB10 large-prompt 2x2 tile against the default tile").dependOn(&prefill_tile22.step);
    const prefill_many_bench = b.addRunArtifact(runner);
    prefill_many_bench.addArg("qmm-prefill-many-bench");
    b.step("qmm-prefill-many-bench", "Same-process A-B-A screening for the GB10 prompt launch fusion").dependOn(&prefill_many_bench.step);
    return install;
}
