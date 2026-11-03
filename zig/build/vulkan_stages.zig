//! The Vulkan stages' build steps, split out of vulkan.zig: one function per stage, called from it in step order.

const std = @import("std");
const vulkan_build = @import("vulkan.zig");

/// What a stage's build needs: the graph, the target, the backend module, the host step and the shader compiler.
pub const Context = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Optimize,
    o: vulkan_build.Options,
    glslang: []const u8,
    kernel_dir: std.Build.LazyPath,
    vulkan: *std.Build.Module,
    host: *std.Build.Step = undefined,
    stage3_host: *std.Build.Step = undefined,
    stage5_host: *std.Build.Step = undefined,
    stage1_shaders_mod: *std.Build.Module = undefined,
    stage2_shaders_mod: *std.Build.Module = undefined,
    stage3_shaders_mod: *std.Build.Module = undefined,
    stage4_shaders_mod: *std.Build.Module = undefined,
    stage5_shaders_mod: *std.Build.Module = undefined,
    stage3_build: *std.Build.Step = undefined,
    stage4_build: *std.Build.Step = undefined,
};

/// vulkan-host-test and the five stage host suites: the host admission, without GPU work.
pub fn hostAdmission(ctx: *Context) void {
    const b = ctx.b;
    const vk_host_tests = b.addTest(.{ .root_module = vulkan_build.module(b, b.graph.host, .debug, ctx.o) });
    ctx.host = b.step("vulkan-host-test", "Vulkan loader admission, device selection, buffer geometry and the qmm fixture, without GPU work");
    ctx.host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    const fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/affine_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
            .{ .name = "stage1_params", .module = b.createModule(.{ .root_source_file = b.path("zig/src/vulkan/params.zig"), .target = b.graph.host, .optimize = .debug }) },
        },
    });
    ctx.host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fixture })).step);
    const stage1_fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage1_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }},
    });
    ctx.host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = stage1_fixture })).step);
    b.step("stage1-host-test", "Stage 1 tensor, arena, params and operator-plan host tests, plus the qmm fixture suite").dependOn(ctx.host);

    // Stage 2 host admission: the vulkan module's own tests plus the stage 2 fixture suite, without GPU work.
    const stage2_fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage2_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }},
    });
    const stage2_fixture_tests = b.addTest(.{ .root_module = stage2_fixture });
    ctx.host.dependOn(&b.addRunArtifact(stage2_fixture_tests).step);
    const stage2_host = b.step("stage2-host-test", "Stage 2 norm/MLP operator-plan host tests and the stage 2 fixture suite, without GPU work");
    stage2_host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    stage2_host.dependOn(&b.addRunArtifact(stage2_fixture_tests).step);
    // The stage 2 probe's host tests (the Bound16 view regression), pure host; `shaders` and `main` stay unanalyzed.
    const stage2_probe_host_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage2_probe.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan_build.module(b, b.graph.host, .debug, ctx.o) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
        },
    }) });
    stage2_host.dependOn(&b.addRunArtifact(stage2_probe_host_tests).step);
    ctx.host.dependOn(&b.addRunArtifact(stage2_probe_host_tests).step);

    // Stages 3 and 4 host admission: the vulkan module's own tests plus each stage's fixture suite, without GPU work.
    const stage3_fixture_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage3_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }},
    }) });
    ctx.host.dependOn(&b.addRunArtifact(stage3_fixture_tests).step);
    ctx.stage3_host = b.step("stage3-host-test", "Stage 3 attention plan host tests and the stage 3 fixture suite, without GPU work");
    ctx.stage3_host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    ctx.stage3_host.dependOn(&b.addRunArtifact(stage3_fixture_tests).step);
    const stage4_fixture_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage4_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
            .{ .name = "vulkan", .module = vulkan_build.module(b, b.graph.host, .debug, ctx.o) },
        },
    }) });
    ctx.host.dependOn(&b.addRunArtifact(stage4_fixture_tests).step);
    const stage4_host = b.step("stage4-host-test", "Stage 4 GDN plan host tests and the stage 4 fixture suite, without GPU work");
    stage4_host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    stage4_host.dependOn(&b.addRunArtifact(stage4_fixture_tests).step);

    // Stage 5 host admission: the vulkan module's tests plus its fixture suite; honest failure until the fixture lands.
    const stage5_fixture_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage5_fixture.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{.{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) }},
    }) });
    ctx.host.dependOn(&b.addRunArtifact(stage5_fixture_tests).step);
    ctx.stage5_host = b.step("stage5-host-test", "Stage 5 greedy plan host tests and the stage 5 fixture suite, without GPU work");
    ctx.stage5_host.dependOn(&b.addRunArtifact(vk_host_tests).step);
    ctx.stage5_host.dependOn(&b.addRunArtifact(stage5_fixture_tests).step);
}

/// vulkan-gpu-build, vulkan-gpu-test, vulkan-probe-build and vulkan-probe: the device tests and the qmm probe.
pub fn probeSteps(ctx: *Context) void {
    const b = ctx.b;
    const gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/vulkan/gpu_tests.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "vulkan", .module = ctx.vulkan }},
    }) });
    b.step("vulkan-gpu-build", "Compile the Vulkan device tests without running GPU work").dependOn(&gpu.step);
    b.step("vulkan-gpu-test", "Real Vulkan device selection and staged device-local buffer round trips").dependOn(&b.addRunArtifact(gpu).step);
    // vulkan1.2 emits SPIR-V 1.5: 1.6's LocalSizeId needs maintenance4, which Device does not enable
    const spirv = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-o" });
    const qmm = spirv.addOutputFileArg("qmm.spv");
    spirv.addFileArg(b.path("zig/kernels/vulkan/qmm.comp"));
    const files = b.addWriteFiles();
    _ = files.addCopyFile(qmm, "qmm.spv");
    const shaders = b.createModule(.{ .root_source_file = files.add("shaders.zig", "pub const qmm align(4) = @embedFile(\"qmm.spv\").*;\n") });
    const probe = b.addExecutable(.{ .name = "tf-vulkan-probe", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/probe.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = ctx.vulkan },
            .{ .name = "vulkan_shaders", .module = shaders },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, ctx.target, .debug) },
            .{ .name = "stage1_params", .module = ctx.vulkan.import_table.get("stage1_params").? },
        },
    }) });
    const install = b.addInstallArtifact(probe, .{});
    b.step("vulkan-probe-build", "Build zig-out/bin/tf-vulkan-probe and its SPIR-V without running GPU work").dependOn(&install.step);
    const run = b.addRunArtifact(probe);
    run.addPassthruArgs();
    b.step("vulkan-probe", "Run the Vulkan qmm probe on the chosen GPU (args after --)").dependOn(&run.step);
}

/// stage1-gpu-build, stage1-gpu-run and stage1-datamove: the stage 1 images and the datamove contract.
pub fn stage1(ctx: *Context) void {
    const b = ctx.b;
    // Stage 1: the operator variants as SPIR-V assets, and the GPU probe from the fixture owner's root.
    const stage1_files = b.addWriteFiles();
    ctx.stage1_shaders_mod = b.createModule(.{ .root_source_file = stage1_files.add("stage1_shaders.zig", vulkan_build.stage1_shaders_src) });
    const stage1_build = b.step("stage1-gpu-build", "Compile the stage 1 SPIR-V variants, install them under stage1/, and build the stage 1 GPU probe without running it");
    const stage1_run = b.step("stage1-gpu-run", "Run the stage 1 GPU probe on the chosen device (args after --)");
    for (vulkan_build.shader_variants) |v| {
        const cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp" });
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        stage1_build.dependOn(&b.addInstallFile(spv, b.fmt("stage1/{s}.spv", .{v.name})).step);
        _ = stage1_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    const stage1_imports: []const std.Build.Module.Import = &.{
        .{ .name = "vulkan", .module = ctx.vulkan },
        .{ .name = "stage1_shaders", .module = ctx.stage1_shaders_mod },
        .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, ctx.target, .debug) },
    };
    const probe_root = b.path(ctx.o.stage1_root orelse "zig/tests/vulkan/stage1_probe.zig");
    const stage1_probe = b.addExecutable(.{ .name = "tf-stage1-gpu", .root_module = b.createModule(.{
        .root_source_file = probe_root,
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = stage1_imports,
    }) });
    stage1_build.dependOn(&b.addInstallArtifact(stage1_probe, .{}).step);
    const stage1_probe_run = b.addRunArtifact(stage1_probe);
    stage1_probe_run.addPassthruArgs();
    stage1_run.dependOn(&stage1_probe_run.step);

    // The datamove group's contract: the seven exact kernels through gates 1-4.
    const datamove_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_datamove_main.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = stage1_imports,
    });
    const datamove = b.addExecutable(.{ .name = "tf-stage1-datamove", .root_module = datamove_module });
    stage1_build.dependOn(&b.addInstallArtifact(datamove, .{}).step);
    const datamove_run = b.addRunArtifact(datamove);
    datamove_run.addPassthruArgs();
    b.step("stage1-datamove", "Run the datamove group's contract gates on the chosen device (args after --)").dependOn(&datamove_run.step);
    // The datamove fixtures' host suite runs on the build host, whatever -Dtarget the runner is built for.
    const datamove_host_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_datamove_main.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan_build.module(b, b.graph.host, .debug, ctx.o) },
            .{ .name = "stage1_shaders", .module = ctx.stage1_shaders_mod },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
        },
    });
    ctx.host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = datamove_host_module })).step);
}

/// stage2-gpu-build and stage2-gpu-run: the stage 2 norm/swiglu images and the NormMlp probe.
pub fn stage2(ctx: *Context) void {
    const b = ctx.b;
    // Stage 2: norm/swiglu as the ASH_QWEN35 oracle form, plus plain bf16 and -DUSE_BF16; the NormMlp probe.
    const stage2_files = b.addWriteFiles();
    const stage2_build = b.step("stage2-gpu-build", "Compile the stage 2 SPIR-V variants, install them under stage2/, and build the stage 2 GPU probe without running it");
    const stage2_run = b.step("stage2-gpu-run", "Run the stage 2 GPU probe on the chosen device (args after --)");
    for (vulkan_build.stage2_variants) |v| {
        const cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp" });
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        stage2_build.dependOn(&b.addInstallFile(spv, b.fmt("stage2/{s}.spv", .{v.name})).step);
        _ = stage2_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    ctx.stage2_shaders_mod = b.createModule(.{ .root_source_file = stage2_files.add("stage2_shaders.zig", vulkan_build.stage2_shaders_src) });
    const stage2_probe = b.addExecutable(.{ .name = "tf-stage2-gpu", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage2_probe.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = ctx.vulkan },
            .{ .name = "stage2_shaders", .module = ctx.stage2_shaders_mod },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, ctx.target, .debug) },
        },
    }) });
    stage2_build.dependOn(&b.addInstallArtifact(stage2_probe, .{}).step);
    const stage2_probe_run = b.addRunArtifact(stage2_probe);
    stage2_probe_run.addPassthruArgs();
    stage2_run.dependOn(&stage2_probe_run.step);
}

/// stage3-gpu-build, stage3-gpu-run and attention-contract-run: the attention images and the attention contract.
pub fn stage3(ctx: *Context) void {
    const b = ctx.b;
    // Stages 3 and 4: attention and GatedDeltaNet; compiles run from the kernel dir with -I. for fast_rope's #include.
    const stage3_files = b.addWriteFiles();
    ctx.stage3_build = b.step("stage3-gpu-build", "Compile the stage 3 attention SPIR-V variants, install them under stage3/, and build the stage 3 GPU probe without running it");
    const stage3_run = b.step("stage3-gpu-run", "Run the stage 3 GPU probe on the chosen device (args after --)");
    for (vulkan_build.stage3_variants) |v| {
        const cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-I." });
        cmd.setCwd(ctx.kernel_dir);
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        ctx.stage3_build.dependOn(&b.addInstallFile(spv, b.fmt("stage3/{s}.spv", .{v.name})).step);
        _ = stage3_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    ctx.stage3_shaders_mod = b.createModule(.{ .root_source_file = stage3_files.add("stage3_shaders.zig", vulkan_build.stage3_shaders_src) });
    const stage3_probe = b.addExecutable(.{ .name = "tf-stage3-gpu", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage3_probe.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = ctx.vulkan },
            .{ .name = "stage1_shaders", .module = ctx.stage1_shaders_mod },
            .{ .name = "stage3_shaders", .module = ctx.stage3_shaders_mod },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, ctx.target, .debug) },
        },
    }) });
    ctx.stage3_build.dependOn(&b.addInstallArtifact(stage3_probe, .{}).step);
    const stage3_probe_run = b.addRunArtifact(stage3_probe);
    stage3_probe_run.addPassthruArgs();
    stage3_run.dependOn(&stage3_probe_run.step);

    // The attention contract: rope and sdpa_rows against the fixture's f64 refs; the GPU probe reuses stage 3 images.
    const attention_files = b.addWriteFiles();
    _ = attention_files.addCopyFile(b.path("zig/kernels/vulkan/sdpa_decode_native.comp"), "sdpa_decode_native.comp");
    const attention_sources = b.createModule(.{ .root_source_file = attention_files.add("attention_sources.zig",
        \\pub const sdpa_decode_native = @embedFile("sdpa_decode_native.comp");
        \\
    ) });
    const attention_contract_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_attention.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
            .{ .name = "attention_sources", .module = attention_sources },
        },
    }) });
    ctx.host.dependOn(&b.addRunArtifact(attention_contract_tests).step);
    ctx.stage3_host.dependOn(&b.addRunArtifact(attention_contract_tests).step);
    const attention_contract = b.addExecutable(.{ .name = "tf-attention-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_attention_main.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = ctx.vulkan },
            .{ .name = "stage1_shaders", .module = ctx.stage1_shaders_mod },
            .{ .name = "stage3_shaders", .module = ctx.stage3_shaders_mod },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, ctx.target, .debug) },
            .{ .name = "attention_sources", .module = attention_sources },
        },
    }) });
    ctx.stage3_build.dependOn(&b.addInstallArtifact(attention_contract, .{}).step);
    const attention_contract_run = b.addRunArtifact(attention_contract);
    attention_contract_run.addPassthruArgs();
    b.step("attention-contract-run", "Run the attention contract (rope_bf16 + sdpa_rows_bf16, served shapes) on the chosen device (args after --)").dependOn(&attention_contract_run.step);
}

/// stage4-gpu-build and stage4-gpu-run: the stage 4 GDN images and the probe.
pub fn stage4(ctx: *Context) void {
    const b = ctx.b;
    const stage4_files = b.addWriteFiles();
    ctx.stage4_build = b.step("stage4-gpu-build", "Compile the stage 4 GDN SPIR-V variants, install them under stage4/, and build the stage 4 GPU probe without running it");
    const stage4_run = b.step("stage4-gpu-run", "Run the stage 4 GPU probe on the chosen device (args after --)");
    for (vulkan_build.stage4_variants) |v| {
        const cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-I." });
        cmd.setCwd(ctx.kernel_dir);
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        ctx.stage4_build.dependOn(&b.addInstallFile(spv, b.fmt("stage4/{s}.spv", .{v.name})).step);
        _ = stage4_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    ctx.stage4_shaders_mod = b.createModule(.{ .root_source_file = stage4_files.add("stage4_shaders.zig", vulkan_build.stage4_shaders_src) });
    const stage4_probe = b.addExecutable(.{ .name = "tf-stage4-gpu", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage4_probe.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = ctx.vulkan },
            .{ .name = "stage4_shaders", .module = ctx.stage4_shaders_mod },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, ctx.target, .debug) },
        },
    }) });
    ctx.stage4_build.dependOn(&b.addInstallArtifact(stage4_probe, .{}).step);
    const stage4_probe_run = b.addRunArtifact(stage4_probe);
    stage4_probe_run.addPassthruArgs();
    stage4_run.dependOn(&stage4_probe_run.step);
}

/// stage5-gpu-build, stage5-gpu-run and greedy-contract-run: the greedy images and the greedy contract.
pub fn stage5(ctx: *Context) void {
    const b = ctx.b;
    // Stage 5: the greedy gate (finite logits, chunked argreduce, combine); fails honestly until its probe root lands.
    const stage5_files = b.addWriteFiles();
    const stage5_build = b.step("stage5-gpu-build", "Compile the stage 5 greedy SPIR-V variants, install them under stage5/, and build the stage 5 GPU probe without running it");
    const stage5_run = b.step("stage5-gpu-run", "Run the stage 5 GPU probe on the chosen device (args after --)");
    for (vulkan_build.stage5_variants) |v| {
        const cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-I." });
        cmd.setCwd(ctx.kernel_dir);
        for (v.defines) |d| cmd.addArg(d);
        cmd.addArg("-o");
        const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{v.name}));
        cmd.addFileArg(b.path(v.src));
        stage5_build.dependOn(&b.addInstallFile(spv, b.fmt("stage5/{s}.spv", .{v.name})).step);
        _ = stage5_files.addCopyFile(spv, b.fmt("{s}.spv", .{v.name}));
    }
    ctx.stage5_shaders_mod = b.createModule(.{ .root_source_file = stage5_files.add("stage5_shaders.zig", vulkan_build.stage5_shaders_src) });
    const stage5_probe = b.addExecutable(.{ .name = "tf-stage5-gpu", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/stage5_probe.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = ctx.vulkan },
            .{ .name = "stage5_shaders", .module = ctx.stage5_shaders_mod },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, ctx.target, .debug) },
        },
    }) });
    stage5_build.dependOn(&b.addInstallArtifact(stage5_probe, .{}).step);
    const stage5_probe_run = b.addRunArtifact(stage5_probe);
    stage5_probe_run.addPassthruArgs();
    stage5_run.dependOn(&stage5_probe_run.step);

    // The greedy contract: the composed draw through gates 1-4 plus the NaN control; the probe reuses stage 5 images.
    const greedy_files = b.addWriteFiles();
    const greedy_ctl_cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-o" });
    const greedy_ctl_spv = greedy_ctl_cmd.addOutputFileArg("contract_controls_greedy.spv");
    greedy_ctl_cmd.addFileArg(b.path("zig/kernels/vulkan/contract_controls_greedy.comp"));
    _ = greedy_files.addCopyFile(greedy_ctl_spv, "contract_controls_greedy.spv");
    const greedy_controls = b.createModule(.{ .root_source_file = greedy_files.add("greedy_controls.zig", "pub const fmax_max align(4) = @embedFile(\"contract_controls_greedy.spv\").*;\n") });
    const greedy_contract_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_greedy.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan_build.module(b, b.graph.host, .debug, ctx.o) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
        },
    }) });
    ctx.host.dependOn(&b.addRunArtifact(greedy_contract_tests).step);
    ctx.stage5_host.dependOn(&b.addRunArtifact(greedy_contract_tests).step);
    const greedy_contract = b.addExecutable(.{ .name = "tf-vulkan-greedy-contract", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_greedy_main.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = ctx.vulkan },
            .{ .name = "stage5_shaders", .module = ctx.stage5_shaders_mod },
            .{ .name = "greedy_controls", .module = greedy_controls },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, ctx.target, .debug) },
        },
    }) });
    stage5_build.dependOn(&b.addInstallArtifact(greedy_contract, .{}).step);
    const greedy_contract_run = b.addRunArtifact(greedy_contract);
    greedy_contract_run.addPassthruArgs();
    b.step("greedy-contract-run", "Run the greedy contract (the composed draw and its planted control) on the chosen device (args after --)").dependOn(&greedy_contract_run.step);
}

/// contract-pointwise-host-test, contract-pointwise-build and contract-pointwise: the pointwise contract.
pub fn contracts(ctx: *Context) void {
    const b = ctx.b;
    const contract_fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_qmm.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan_build.module(b, b.graph.host, .debug, ctx.o) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
        },
    });
    ctx.host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = contract_fixture })).step);
    // The gdn host proofs: walker vs the stage 4 fixture's, depth shape honesty, stress sums' f32 and fp16 margins
    const contract_gdn_fixture = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_gdn.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "vulkan", .module = vulkan_build.module(b, b.graph.host, .debug, ctx.o) },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
        },
    });
    ctx.host.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = contract_gdn_fixture })).step);
    // The pointwise contract: all six closures against one GPU probe, plus the fixtures' host suite in `test`.
    const pointwise_fixture_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_pointwise.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .imports = &.{
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, b.graph.host, .debug) },
            .{ .name = "stage2_shaders", .module = ctx.stage2_shaders_mod },
            .{ .name = "vulkan", .module = vulkan_build.module(b, b.graph.host, .debug, ctx.o) },
        },
    }) });
    ctx.host.dependOn(&b.addRunArtifact(pointwise_fixture_tests).step);
    b.step("contract-pointwise-host-test", "The pointwise contract fixtures' host suite (references and checkers), without GPU work").dependOn(ctx.host);
    const pointwise = b.addExecutable(.{ .name = "tf-contract-pointwise", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/vulkan/contract_pointwise_main.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = ctx.vulkan },
            .{ .name = "stage2_shaders", .module = ctx.stage2_shaders_mod },
            .{ .name = "stage3_shaders", .module = ctx.stage3_shaders_mod },
            .{ .name = "stage4_shaders", .module = ctx.stage4_shaders_mod },
            .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, ctx.target, .debug) },
        },
    }) });
    const pointwise_install = b.addInstallArtifact(pointwise, .{});
    ctx.stage3_build.dependOn(&pointwise_install.step);
    ctx.stage4_build.dependOn(&pointwise_install.step);
    b.step("contract-pointwise-build", "Compile the six pointwise SPIR-V images and the pointwise contract probe without running GPU work").dependOn(&pointwise_install.step);
    const pointwise_run = b.addRunArtifact(pointwise);
    pointwise_run.addPassthruArgs();
    b.step("contract-pointwise", "Run the pointwise shader contract on the chosen GPU (args after --)").dependOn(&pointwise_run.step);
}

/// tf-vulkan-kernel-contract and vulkan-kernel-contract: the QMM contract runner and its planted controls.
pub fn kernelContract(ctx: *Context) void {
    const b = ctx.b;
    const qmm_bf16_cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-DUSE_BF16=1", "-o" });
    const qmm_bf16 = qmm_bf16_cmd.addOutputFileArg("qmm_bf16.spv");
    qmm_bf16_cmd.addFileArg(b.path("zig/kernels/vulkan/qmm.comp"));
    const qmm_vec4_cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-o" });
    const qmm_vec4 = qmm_vec4_cmd.addOutputFileArg("qmm_q4_vec4.spv");
    qmm_vec4_cmd.addFileArg(b.path("zig/kernels/vulkan/qmm_q4_vec4.comp"));
    const qmm_tile_cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-o" });
    const qmm_tile = qmm_tile_cmd.addOutputFileArg("qmm_q4_tile.spv");
    qmm_tile_cmd.addFileArg(b.path("zig/kernels/vulkan/qmm_q4_tile.comp"));
    const qmm_tile_rows_cmd = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", "-o" });
    const qmm_tile_rows = qmm_tile_rows_cmd.addOutputFileArg("qmm_q4_tile_rows.spv");
    qmm_tile_rows_cmd.addFileArg(b.path("zig/kernels/vulkan/qmm_q4_tile_rows.comp"));
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
            const command = b.addSystemCommand(&.{ ctx.glslang, "-V", "--target-env", "vulkan1.2", "-S", "comp", variant[1], "-o" });
            out[i] = command.addOutputFileArg(variant[0] ++ ".spv");
            command.addFileArg(b.path("zig/kernels/vulkan/contract_controls.comp"));
        }
        break :blk out;
    };
    const contract_files = b.addWriteFiles();
    _ = contract_files.addCopyFile(qmm_bf16, "qmm_bf16.spv");
    _ = contract_files.addCopyFile(qmm_vec4, "qmm_q4_vec4.spv");
    _ = contract_files.addCopyFile(qmm_tile, "qmm_q4_tile.spv");
    _ = contract_files.addCopyFile(qmm_tile_rows, "qmm_q4_tile_rows.spv");
    inline for (variants, 0..) |variant, i| {
        _ = contract_files.addCopyFile(controls[i], variant[0] ++ ".spv");
    }
    const contract_shaders = b.createModule(.{ .root_source_file = contract_files.add("vulkan_contract_shaders.zig", "pub const qmm_bf16 align(4) = @embedFile(\"qmm_bf16.spv\").*;\n" ++
        "pub const qmm_q4_vec4 align(4) = @embedFile(\"qmm_q4_vec4.spv\").*;\n" ++
        "pub const qmm_q4_tile align(4) = @embedFile(\"qmm_q4_tile.spv\").*;\n" ++
        "pub const qmm_q4_tile_rows align(4) = @embedFile(\"qmm_q4_tile_rows.spv\").*;\n" ++
        "pub const qmm_fmax align(4) = @embedFile(\"qmm_fmax.spv\").*;\n" ++
        "pub const atomic_sum align(4) = @embedFile(\"atomic_sum.spv\").*;\n" ++
        "pub const scribble_sum align(4) = @embedFile(\"scribble_sum.spv\").*;\n" ++
        "pub const drop_tail align(4) = @embedFile(\"drop_tail.spv\").*;\n" ++
        "pub const fp16_accum align(4) = @embedFile(\"fp16_accum.spv\").*;\n" ++
        "pub const over_limit align(4) = @embedFile(\"over_limit.spv\").*;\n") });
    const contract_module = @import("kernel_contract.zig").module(b, ctx.target, .debug);
    const contract_exe = b.addExecutable(.{ .name = "tf-vulkan-kernel-contract", .root_module = b.createModule(.{
        .root_source_file = b.path(ctx.o.contract_root orelse "zig/tests/vulkan/contract_main.zig"),
        .target = ctx.target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "vulkan", .module = ctx.vulkan },
            .{ .name = "vulkan_shaders", .module = contract_shaders },
            .{ .name = "stage1_shaders", .module = ctx.stage1_shaders_mod },
            .{ .name = "stage2_shaders", .module = ctx.stage2_shaders_mod },
            .{ .name = "stage3_shaders", .module = ctx.stage3_shaders_mod },
            .{ .name = "stage4_shaders", .module = ctx.stage4_shaders_mod },
            .{ .name = "stage5_shaders", .module = ctx.stage5_shaders_mod },
            .{ .name = "core", .module = contract_module },
            .{ .name = "contract", .module = contract_module },
        },
    }) });
    const contract_install = b.addInstallArtifact(contract_exe, .{});
    b.step("tf-vulkan-kernel-contract", "Build Vulkan QMM contract runner and planted control shaders without running them").dependOn(&contract_install.step);
    b.step("vulkan-kernel-contract", "Run Vulkan QMM contract gates and planted controls on the selected device").dependOn(&b.addRunArtifact(contract_exe).step);
}
