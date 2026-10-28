//! Vulkan, built only with -Dvulkan=true: host tests in `test`, named GPU steps; options on every target (#1165).

const std = @import("std");

/// Vulkan's options as given, declared whether or not -Dvulkan=true builds anything with them.
pub const Options = struct { include: ?[]const u8, glslang: ?[]const u8, stage1_root: ?[]const u8 };

pub fn options(b: *std.Build) Options {
    return .{
        .include = b.option([]const u8, "vulkan-include", "Directory holding vulkan/vulkan_core.h and vk_video/ (default /usr/include)"),
        .glslang = b.option([]const u8, "glslang", "glslang for the Vulkan shaders (default: glslang on PATH)"),
        .stage1_root = b.option([]const u8, "stage1-test-root", "Root .zig of the stage 1 GPU probe (default zig/tests/vulkan/stage1_probe.zig)"),
    };
}

/// vulkan-host-test (part of `test`), vulkan-gpu-build and vulkan-gpu-test, vulkan-probe-build and vulkan-probe.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize, test_step: *std.Build.Step, o: Options) void {
    const vk_host_tests = b.addTest(.{ .root_module = module(b, b.graph.host, .debug, o) });
    const host = b.step("vulkan-host-test", "Vulkan loader admission, device selection, buffer geometry and the qmm fixture, without GPU work");
    host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    const fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/affine_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
            .{ .name = "stage1_params", .module = b.createModule(.{ .root_source_file = b.path("zig/src/vulkan/params.zig"), .target = b.graph.host, .optimize = .debug }) },
        },
    });
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fixture })).step);
    const stage1_fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage1_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }},
    });
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = stage1_fixture })).step);
    test_step.dependOn(host);
    b.step("stage1-host-test", "Stage 1 tensor, arena, params and operator-plan host tests, plus the qmm fixture suite").dependOn(host);

    // Stage 2 host admission: the vulkan module's own tests (norm_mlp's operator-plan cases among
    // them) plus the stage 2 fixture suite, without GPU work.
    const stage2_fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage2_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }},
    });
    const stage2_fixture_tests = b.addTest(.{ .root_module = stage2_fixture });
    host.dependOn(&b.addRunArtifact(stage2_fixture_tests).step);
    const stage2_host = b.step("stage2-host-test", "Stage 2 norm/MLP operator-plan host tests and the stage 2 fixture suite, without GPU work");
    stage2_host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    stage2_host.dependOn(&b.addRunArtifact(stage2_fixture_tests).step);
    // The stage 2 probe's host tests: the Bound16 view-derivation regression (binding lifetime),
    // pure host. `shaders` and `main` stay unanalyzed here; the SPIR-V assets belong to the GPU build.
    const stage2_probe_host_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage2_probe.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "vulkan", .module = module(b, b.graph.host, .debug, o) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
        },
    }) });
    stage2_host.dependOn(&b.addRunArtifact(stage2_probe_host_tests).step);
    host.dependOn(&b.addRunArtifact(stage2_probe_host_tests).step);

    // Stages 3 and 4 host admission: the vulkan module's own tests (attention's
    // and gdn's plan cases among them) plus each stage's fixture suite, without
    // GPU work.
    const stage3_fixture_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage3_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }},
    }) });
    host.dependOn(&b.addRunArtifact(stage3_fixture_tests).step);
    const stage3_host = b.step("stage3-host-test", "Stage 3 attention plan host tests and the stage 3 fixture suite, without GPU work");
    stage3_host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    stage3_host.dependOn(&b.addRunArtifact(stage3_fixture_tests).step);
    const stage4_fixture_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage4_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
            .{ .name = "vulkan", .module = module(b, b.graph.host, .debug, o) },
        },
    }) });
    host.dependOn(&b.addRunArtifact(stage4_fixture_tests).step);
    const stage4_host = b.step("stage4-host-test", "Stage 4 GDN plan host tests and the stage 4 fixture suite, without GPU work");
    stage4_host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    stage4_host.dependOn(&b.addRunArtifact(stage4_fixture_tests).step);

    // Stage 5 host admission: the vulkan module's own tests (greedy's plan
    // cases among them) plus the stage 5 fixture suite, without GPU
    // work. The fixture file lands with its owner; the step fails honestly until
    // then, and no other step waits on it.
    const stage5_fixture_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage5_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }},
    }) });
    host.dependOn(&b.addRunArtifact(stage5_fixture_tests).step);
    const stage5_host = b.step("stage5-host-test", "Stage 5 greedy plan host tests and the stage 5 fixture suite, without GPU work");
    stage5_host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    stage5_host.dependOn(&b.addRunArtifact(stage5_fixture_tests).step);

    const vulkan = module(b, target, optimize, o);
    const gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/vulkan/gpu_tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "vulkan", .module = vulkan }},
    }) });
    b.step("vulkan-gpu-build", "Compile the Vulkan device tests without running GPU work").dependOn(&gpu.step);
    b.step("vulkan-gpu-test", "Real Vulkan device selection and staged device-local buffer round trips").dependOn(&b.addRunArtifact(gpu).step);
    const glslang = o.glslang orelse "glslang";
    // vulkan1.2 emits SPIR-V 1.5: 1.6's LocalSizeId needs maintenance4, which Device does not enable
    const spirv = b.addSystemCommand(&.{ glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-o" });
    const qmm = spirv.addOutputFileArg("qmm.spv");
    spirv.addFileArg(b.path("zig/kernels/vulkan/qmm.comp"));
    const files = b.addWriteFiles();
    _ = files.addCopyFile(qmm, "qmm.spv");
    const shaders = b.createModule(.{ .root_source_file = files.add("shaders.zig", "pub const qmm align(4) = @embedFile(\"qmm.spv\").*;\n") });
    const probe = b.addExecutable(.{ .name = "tf-vulkan-probe", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "vulkan_shaders", .module = shaders },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .debug) },
            .{ .name = "stage1_params", .module = vulkan.import_table.get("stage1_params").? },
        },
    }) });
    const install = b.addInstallArtifact(probe, .{});
    b.step("vulkan-probe-build", "Build zig-out/bin/tf-vulkan-probe and its SPIR-V without running GPU work").dependOn(&install.step);
    const run = b.addRunArtifact(probe);
    run.addPassthruArgs();
    b.step("vulkan-probe", "Run the Vulkan qmm probe on the chosen GPU (args after --)").dependOn(&run.step);

    const contract_fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_qmm.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "vulkan", .module = module(b, b.graph.host, .debug, o) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
        },
    });
    host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = contract_fixture })).step);
    const contract_glslang = o.glslang orelse "glslang";
    const qmm_bf16_cmd = b.addSystemCommand(&.{ contract_glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-DUSE_BF16=1", "-o" });
    const qmm_bf16 = qmm_bf16_cmd.addOutputFileArg("qmm_bf16.spv");
    qmm_bf16_cmd.addFileArg(b.path("zig/kernels/vulkan/qmm.comp"));
    const variants = .{
        .{ "qmm_fmax", "-DTF_CTL_QMM_FMAX=1" },
        .{ "atomic_sum", "-DTF_CTL_ATOMIC_SUM=1" },
        .{ "scribble_sum", "-DTF_CTL_SCRIBBLE=1" },
        .{ "drop_tail", "-DTF_CTL_DROP_TAIL=1" },
        .{ "fp16_accum", "-DTF_CTL_FP16=1" },
        .{ "over_limit", "-DTF_CTL_OVER_LIMIT=1" },
    };
    const controls: [6]std.Build.LazyPath = blk: {
        var out: [6]std.Build.LazyPath = undefined;
        inline for (variants, 0..) |variant, i| {
            const command = b.addSystemCommand(&.{ contract_glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", variant[1], "-o" });
            out[i] = command.addOutputFileArg(variant[0] ++ ".spv");
            command.addFileArg(b.path("zig/kernels/vulkan/contract_controls.comp"));
        }
        break :blk out;
    };
    const contract_files = b.addWriteFiles();
    _ = contract_files.addCopyFile(qmm_bf16, "qmm_bf16.spv");
    inline for (variants, 0..) |variant, i| {
        _ = contract_files.addCopyFile(controls[i], variant[0] ++ ".spv");
    }
    const contract_shaders = b.createModule(.{ .root_source_file = contract_files.add("vulkan_contract_shaders.zig", "pub const qmm_bf16 align(4) = @embedFile(\"qmm_bf16.spv\").*;\n" ++
        "pub const qmm_fmax align(4) = @embedFile(\"qmm_fmax.spv\").*;\n" ++
        "pub const atomic_sum align(4) = @embedFile(\"atomic_sum.spv\").*;\n" ++
        "pub const scribble_sum align(4) = @embedFile(\"scribble_sum.spv\").*;\n" ++
        "pub const drop_tail align(4) = @embedFile(\"drop_tail.spv\").*;\n" ++
        "pub const fp16_accum align(4) = @embedFile(\"fp16_accum.spv\").*;\n" ++
        "pub const over_limit align(4) = @embedFile(\"over_limit.spv\").*;\n") });
    const contract_module = @import("kernel_contract.zig").module(b, target, .debug);
    const contract_exe = b.addExecutable(.{ .name = "tf-vulkan-kernel-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "vulkan_shaders", .module = contract_shaders },
            .{ .name = "core", .module = contract_module },
            .{ .name = "contract", .module = contract_module },
        },
    }) });
    const contract_install = b.addInstallArtifact(contract_exe, .{});
    b.step("tf-vulkan-kernel-contract", "Build Vulkan QMM contract runner and planted control shaders without running them").dependOn(&contract_install.step);
    b.step("vulkan-kernel-contract", "Run Vulkan QMM contract gates and planted controls on the selected device").dependOn(&b.addRunArtifact(contract_exe).step);

    // Stage 1: the operator variants as SPIR-V assets, and the GPU probe from the fixture owner's root.
    const stage1_files = b.addWriteFiles();
    const stage1_shaders_mod = b.createModule(.{ .root_source_file = stage1_files.add("stage1_shaders.zig", stage1_shaders_src) });
    const stage1_build = b.step("stage1-gpu-build", "Compile the stage 1 SPIR-V variants, install them under stage1/, and build the stage 1 GPU probe without running it");
    const stage1_run = b.step("stage1-gpu-run", "Run the stage 1 GPU probe on the chosen device (args after --)");
    for (shader_variants) |v| {
        const cmd = b.addSystemCommand(&.{ glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp" });
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        stage1_build.dependOn(&b.addInstallFile(spv, b.fmt("stage1/{s}.spv", .{v.name})).step);
        _ = stage1_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    const probe_root = b.path(o.stage1_root orelse "zig/tests/vulkan/stage1_probe.zig");
    const stage1_probe = b.addExecutable(.{ .name = "tf-stage1-gpu", .root_module = b.createModule(.{
        .root_source_file = probe_root,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "stage1_shaders", .module = stage1_shaders_mod },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .debug) },
        },
    }) });
    stage1_build.dependOn(&b.addInstallArtifact(stage1_probe, .{}).step);
    const stage1_probe_run = b.addRunArtifact(stage1_probe);
    stage1_probe_run.addPassthruArgs();
    stage1_run.dependOn(&stage1_probe_run.step);

    // Stage 2: the norm/MLP pointwise variants — norm and swiglu compiled as the Ash8c
    // Metal-oracle form (ASH_QWEN35), add the plain bf16 form; all with -DUSE_BF16 —
    // and the GPU probe over the production NormMlp operator. Compile only: no GPU work.
    const stage2_files = b.addWriteFiles();
    const stage2_build = b.step("stage2-gpu-build", "Compile the stage 2 SPIR-V variants, install them under stage2/, and build the stage 2 GPU probe without running it");
    const stage2_run = b.step("stage2-gpu-run", "Run the stage 2 GPU probe on the chosen device (args after --)");
    for (stage2_variants) |v| {
        const cmd = b.addSystemCommand(&.{ glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp" });
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        stage2_build.dependOn(&b.addInstallFile(spv, b.fmt("stage2/{s}.spv", .{v.name})).step);
        _ = stage2_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    const stage2_probe = b.addExecutable(.{ .name = "tf-stage2-gpu", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage2_probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "stage2_shaders", .module = b.createModule(.{ .root_source_file = stage2_files.add("stage2_shaders.zig", stage2_shaders_src) }) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .debug) },
        },
    }) });
    stage2_build.dependOn(&b.addInstallArtifact(stage2_probe, .{}).step);
    const stage2_probe_run = b.addRunArtifact(stage2_probe);
    stage2_probe_run.addPassthruArgs();
    stage2_run.dependOn(&stage2_probe_run.step);

    // Stages 3 and 4: attention (half-split RoPE, causal GQA SDPA, output gate)
    // and the GatedDeltaNet closure. Each stage compiles its SPIR-V variants
    // (vulkan1.2 -> SPIR-V 1.5), installs them under stageN/, and builds its
    // GPU probe without running it. The compile runs from the kernel directory
    // with -I. so fast_rope's repo-local #include "omarchy_trig.h" resolves.
    const kernel_dir = b.path("zig/kernels/vulkan");
    const stage3_files = b.addWriteFiles();
    const stage3_build = b.step("stage3-gpu-build", "Compile the stage 3 attention SPIR-V variants, install them under stage3/, and build the stage 3 GPU probe without running it");
    const stage3_run = b.step("stage3-gpu-run", "Run the stage 3 GPU probe on the chosen device (args after --)");
    for (stage3_variants) |v| {
        const cmd = b.addSystemCommand(&.{ glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-I." });
        cmd.setCwd(kernel_dir);
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        stage3_build.dependOn(&b.addInstallFile(spv, b.fmt("stage3/{s}.spv", .{v.name})).step);
        _ = stage3_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    const stage3_probe = b.addExecutable(.{ .name = "tf-stage3-gpu", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage3_probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "stage1_shaders", .module = stage1_shaders_mod },
            .{ .name = "stage3_shaders", .module = b.createModule(.{ .root_source_file = stage3_files.add("stage3_shaders.zig", stage3_shaders_src) }) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .debug) },
        },
    }) });
    stage3_build.dependOn(&b.addInstallArtifact(stage3_probe, .{}).step);
    const stage3_probe_run = b.addRunArtifact(stage3_probe);
    stage3_probe_run.addPassthruArgs();
    stage3_run.dependOn(&stage3_probe_run.step);

    const stage4_files = b.addWriteFiles();
    const stage4_build = b.step("stage4-gpu-build", "Compile the stage 4 GDN SPIR-V variants, install them under stage4/, and build the stage 4 GPU probe without running it");
    const stage4_run = b.step("stage4-gpu-run", "Run the stage 4 GPU probe on the chosen device (args after --)");
    for (stage4_variants) |v| {
        const cmd = b.addSystemCommand(&.{ glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-I." });
        cmd.setCwd(kernel_dir);
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        stage4_build.dependOn(&b.addInstallFile(spv, b.fmt("stage4/{s}.spv", .{v.name})).step);
        _ = stage4_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    const stage4_probe = b.addExecutable(.{ .name = "tf-stage4-gpu", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage4_probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "stage4_shaders", .module = b.createModule(.{ .root_source_file = stage4_files.add("stage4_shaders.zig", stage4_shaders_src) }) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .debug) },
        },
    }) });
    stage4_build.dependOn(&b.addInstallArtifact(stage4_probe, .{}).step);
    const stage4_probe_run = b.addRunArtifact(stage4_probe);
    stage4_probe_run.addPassthruArgs();
    stage4_run.dependOn(&stage4_probe_run.step);

    // Stage 5: the greedy gate (finite logits, chunked argreduce, combine).
    // The probe root lands with its owner; the target fails honestly until
    // then, no other step waits on it.
    const stage5_files = b.addWriteFiles();
    const stage5_build = b.step("stage5-gpu-build", "Compile the stage 5 greedy SPIR-V variants, install them under stage5/, and build the stage 5 GPU probe without running it");
    const stage5_run = b.step("stage5-gpu-run", "Run the stage 5 GPU probe on the chosen device (args after --)");
    for (stage5_variants) |v| {
        const cmd = b.addSystemCommand(&.{ glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-I." });
        cmd.setCwd(kernel_dir);
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        stage5_build.dependOn(&b.addInstallFile(spv, b.fmt("stage5/{s}.spv", .{v.name})).step);
        _ = stage5_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    const stage5_probe = b.addExecutable(.{ .name = "tf-stage5-gpu", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage5_probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan },
            .{ .name = "stage5_shaders", .module = b.createModule(.{ .root_source_file = stage5_files.add("stage5_shaders.zig", stage5_shaders_src) }) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, .debug) },
        },
    }) });
    stage5_build.dependOn(&b.addInstallArtifact(stage5_probe, .{}).step);
    const stage5_probe_run = b.addRunArtifact(stage5_probe);
    stage5_probe_run.addPassthruArgs();
    stage5_run.dependOn(&stage5_probe_run.step);

    // The qwen3.5 Vulkan family weights/state host admission (auditable in this
    // tree, unlike a child harness): refAllDeclsRecursive over the family
    // modules, with the production vulkan module. The house wires (sys,
    // checkpoint_config, save_guard) attach to this root module in build.zig's
    // post-declaration walk, so they are not named here. The shared loader's
    // TF_QWEN_DIR real test stays skipped: the parent runs the step with
    // `env -u TF_QWEN_DIR`. No engine/native adapter import (unfinished).
    const model_host_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/vulkan_model_host_tests.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = module(b, b.graph.host, .debug, o) },
            .{ .name = "core", .module = coreModule(b, b.graph.host, .debug) },
        },
    }) });
    host.dependOn(&b.addRunArtifact(model_host_tests).step);
    b.step("model-host-test", "Qwen3.5 Vulkan family weights/state host tests (TF_QWEN_DIR real-load test stays skipped without the env var)").dependOn(&b.addRunArtifact(model_host_tests).step);
}

const ShaderVariant = struct { name: []const u8, src: []const u8, defines: []const []const u8 };

/// The stage 1 closure's shader images: one variant per op path the host code names in ops.Shaders.
const shader_variants = [_]ShaderVariant{
    .{ .name = "qmm_bf16", .src = "zig/kernels/vulkan/qmm.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "dequant_bf16", .src = "zig/kernels/vulkan/dequant.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "gather_u32", .src = "zig/kernels/vulkan/gather_take.comp", .defines = &.{"-DUSE_U32"} },
    .{ .name = "gather_u16", .src = "zig/kernels/vulkan/gather_take.comp", .defines = &.{"-DUSE_U16"} },
    .{ .name = "cast_bf16_f32", .src = "zig/kernels/vulkan/cast.comp", .defines = &.{"-DSOURCE_BF16"} },
    .{ .name = "cast_f32_bf16", .src = "zig/kernels/vulkan/cast.comp", .defines = &.{"-DDEST_BF16"} },
    .{ .name = "copy_u32", .src = "zig/kernels/vulkan/copy_general.comp", .defines = &.{"-DUSE_U32"} },
    .{ .name = "copy_u16", .src = "zig/kernels/vulkan/copy_general.comp", .defines = &.{"-DUSE_U16"} },
};

const stage1_shaders_src =
    \\pub const qmm_bf16 align(4) = @embedFile("qmm_bf16.spv").*;
    \\pub const dequant_bf16 align(4) = @embedFile("dequant_bf16.spv").*;
    \\pub const gather_u32 align(4) = @embedFile("gather_u32.spv").*;
    \\pub const gather_u16 align(4) = @embedFile("gather_u16.spv").*;
    \\pub const cast_bf16_f32 align(4) = @embedFile("cast_bf16_f32.spv").*;
    \\pub const cast_f32_bf16 align(4) = @embedFile("cast_f32_bf16.spv").*;
    \\pub const copy_u32 align(4) = @embedFile("copy_u32.spv").*;
    \\pub const copy_u16 align(4) = @embedFile("copy_u16.spv").*;
;

/// The stage 2 closure's shader images: norm and swiglu carry the Ash8c Metal-oracle variant
/// (single store rounding; weight applied to the product), add the plain bf16 form.
const stage2_variants = [_]ShaderVariant{
    .{ .name = "norm_bf16", .src = "zig/kernels/vulkan/fast_norm.comp", .defines = &.{ "-DUSE_BF16", "-DASH_QWEN35" } },
    .{ .name = "add_bf16", .src = "zig/kernels/vulkan/binary_vec.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "swiglu_bf16", .src = "zig/kernels/vulkan/swiglu.comp", .defines = &.{ "-DUSE_BF16", "-DASH_QWEN35" } },
};

const stage2_shaders_src =
    \\pub const norm_bf16 align(4) = @embedFile("norm_bf16.spv").*;
    \\pub const add_bf16 align(4) = @embedFile("add_bf16.spv").*;
    \\pub const swiglu_bf16 align(4) = @embedFile("swiglu_bf16.spv").*;
;

/// The stage 3 closure's shader images: the ASH_QWEN35 rope variant (Metal-oracle
/// inverse frequencies, no intermediate bf16 rounding), the causal multi-row SDPA
/// decode arm and the attention output gate.
const stage3_variants = [_]ShaderVariant{
    .{ .name = "rope_bf16", .src = "zig/kernels/vulkan/fast_rope.comp", .defines = &.{ "-DUSE_BF16", "-DASH_QWEN35" } },
    .{ .name = "sdpa_rows_bf16", .src = "zig/kernels/vulkan/sdpa_decode_native.comp", .defines = &.{ "-DBF16_IO", "-DSDPA_DIM=256", "-DSDPA_ROWS" } },
    .{ .name = "gate_bf16", .src = "zig/kernels/vulkan/attention_gate.comp", .defines = &.{} },
};

const stage3_shaders_src =
    \\pub const rope_bf16 align(4) = @embedFile("rope_bf16.spv").*;
    \\pub const sdpa_rows_bf16 align(4) = @embedFile("sdpa_rows_bf16.spv").*;
    \\pub const gate_bf16 align(4) = @embedFile("gate_bf16.spv").*;
;

/// The stage 4 closure's shader images: the GatedDeltaNet conv, gate split,
/// gated-delta prefill scan (u8 mask binding removed) and decode, the gated
/// RMSNorm epilogue and the contiguous bf16 copy, all -DUSE_BF16.
const stage4_variants = [_]ShaderVariant{
    .{ .name = "conv_dw1d_bf16", .src = "zig/kernels/vulkan/conv_dw1d.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "gdn_conv_decode_bf16", .src = "zig/kernels/vulkan/gdn_conv_decode.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "gdn_split_bf16", .src = "zig/kernels/vulkan/gdn_split.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "prefill_recur_bf16", .src = "zig/kernels/vulkan/gated_delta_prefill_ash.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "gated_delta_decode_bf16", .src = "zig/kernels/vulkan/gated_delta_decode.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "fast_norm_gated_bf16", .src = "zig/kernels/vulkan/fast_norm_gated.comp", .defines = &.{ "-DUSE_BF16", "-DASH_QWEN35" } },
    .{ .name = "copy_u16", .src = "zig/kernels/vulkan/copy_general.comp", .defines = &.{"-DUSE_U16"} },
};

const stage4_shaders_src =
    \\pub const conv_dw1d_bf16 align(4) = @embedFile("conv_dw1d_bf16.spv").*;
    \\pub const gdn_conv_decode_bf16 align(4) = @embedFile("gdn_conv_decode_bf16.spv").*;
    \\pub const gdn_split_bf16 align(4) = @embedFile("gdn_split_bf16.spv").*;
    \\pub const prefill_recur_bf16 align(4) = @embedFile("prefill_recur_bf16.spv").*;
    \\pub const gated_delta_decode_bf16 align(4) = @embedFile("gated_delta_decode_bf16.spv").*;
    \\pub const fast_norm_gated_bf16 align(4) = @embedFile("fast_norm_gated_bf16.spv").*;
    \\pub const copy_u16 align(4) = @embedFile("copy_u16.spv").*;
;

/// The stage 5 closure's shader images: the greedy gate's three passes.
/// argreduce_chunk reads the bf16 logits (-DUSE_BF16; the default float32
/// build would read the wrong dtype and run out of bounds); finite_logits is
/// plain bf16-widening and needs no define; argreduce_combine reduces the f32
/// partials with no bf16 specialization.
const stage5_variants = [_]ShaderVariant{
    .{ .name = "finite_logits", .src = "zig/kernels/vulkan/finite_logits.comp", .defines = &.{} },
    .{ .name = "argreduce_chunk", .src = "zig/kernels/vulkan/argreduce_chunk.comp", .defines = &.{"-DUSE_BF16"} },
    .{ .name = "argreduce_combine", .src = "zig/kernels/vulkan/argreduce_combine.comp", .defines = &.{} },
};

const stage5_shaders_src =
    \\pub const finite_logits align(4) = @embedFile("finite_logits.spv").*;
    \\pub const argreduce_chunk align(4) = @embedFile("argreduce_chunk.spv").*;
    \\pub const argreduce_combine align(4) = @embedFile("argreduce_combine.spv").*;
;

/// The qwen3.5 Vulkan server's modules: the thin family root, the embedded
/// SPIR-V bundle (one decl per shader image, names the adapter's bundle()
/// reads), and the native_engines registry root. The caller (the server
/// target's build site) attaches them; nothing here builds a server binary.
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

/// Every SPIR-V image the server bundle embeds: the REAL stage variant arrays,
/// concatenated (stage4's trailing copy_u16 skipped — the bundle needs ONE
/// copy_u16 decl, already present from stage1; names are exactly what the
/// adapter's bundle() reads). Reusing the arrays means a define change (e.g.
/// cast's source/dest variants, argreduce_chunk's bf16) cannot drift between
/// the probe builds and the server bundle.
const server_shader_images = shader_variants ++ stage2_variants ++ stage3_variants ++ stage4_variants[0 .. stage4_variants.len - 1] ++ stage5_variants;

/// The house core module (core/root.zig) the family loader's config/shard/
/// slicing/table imports resolve through, built exactly as cuda.zig builds it.
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
fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.Optimize, o: Options) *std.Build.Module {
    const header = b.addWriteFiles().add("vulkan_c.h", "#define VK_NO_PROTOTYPES\n#include <vulkan/vulkan_core.h>\n");
    const vk = b.addTranslateC(.{ .root_source_file = header, .target = target, .optimize = optimize });
    // after the target's libc headers, so a cross build never takes the host's stdint.h
    vk.addAfterIncludePath(.{ .cwd_relative = o.include orelse "/usr/include" });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/vulkan/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("vk", vk.createModule());
    m.addImport("stage1_params", b.createModule(.{ .root_source_file = b.path("zig/src/vulkan/params.zig"), .target = target, .optimize = optimize }));
    return m;
}
