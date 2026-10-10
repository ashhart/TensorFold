//! HIP: mock-backed admission tests join `zig build test`; GPU builds and runs are opt-in steps (-Dhipcc, -Dhip-include, -Dhip-arch).

const std = @import("std");
const caps = @import("../src/hip/caps.zig");

/// hip-host-test (part of `test`), hip-gpu-build and hip-gpu-test, hip-affine-build and hip-affine-test.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options, test_step: *std.Build.Step) void {
    const hip_fixtures = b.addOptions();
    for ([_][]const u8{ "success", "failed", "missing" }) |kind| {
        const fixture_module = b.createModule(.{ .target = b.graph.host, .link_libc = true });
        fixture_module.addCSourceFile(.{
            .file = b.path("zig/tests/hip_mock.c"),
            .flags = if (std.mem.eql(u8, kind, "missing")) &.{ "-DOMIT_INIT", "-DINIT_RESULT=0" } else if (std.mem.eql(u8, kind, "failed")) &.{"-DINIT_RESULT=1"} else &.{"-DINIT_RESULT=0"},
        });
        const fixture = b.addLibrary(.{ .name = b.fmt("hip-mock-{s}", .{kind}), .linkage = .dynamic, .root_module = fixture_module });
        hip_fixtures.addOptionPath(kind, fixture.getEmittedBin());
    }
    const hip_test_module = b.createModule(.{
        .root_source_file = b.path("zig/src/hip/admission_tests.zig"),
        .target = b.graph.host,
        .link_libc = true,
    });
    hip_test_module.addOptions("hip_fixtures", hip_fixtures);
    hip_test_module.addImport("hip_kernels", b.createModule(.{ .root_source_file = b.addWriteFiles().add("kernels.zig", stub) }));
    hip_test_module.addImport("core", coreModule(b, b.graph.host));
    const hip_tests = b.addTest(.{ .root_module = hip_test_module });
    const run_hip_tests = b.addRunArtifact(hip_tests);
    test_step.dependOn(&run_hip_tests.step);
    b.step("hip-host-test", "HIP admission tests without GPU work").dependOn(&run_hip_tests.step);
    const loader = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3_5/loader.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = coreModule(b, b.graph.host) }},
    }) }));
    loader.has_side_effects = true; // TF_QWEN_DIR is not a cached input
    test_step.dependOn(&loader.step);
    b.step("test-qwen35", "Qwen3.5 / 3.6 checkpoint loader tests; TF_QWEN_DIR indexes a real checkpoint").dependOn(&loader.step);
    const hipcc = b.option([]const u8, "hipcc", "HIP compiler for model-free tests") orelse "hipcc";
    const hipcc_resolved = b.findProgram(.{ .names = &.{hipcc} }) orelse hipcc;
    const hip_include = b.option([]const u8, "hip-include", "HIP header directory") orelse
        b.pathResolve(&.{ std.fs.path.dirname(hipcc_resolved) orelse "/opt/rocm/bin", "..", "include" });
    const hip_arch = b.option([]const u8, "hip-arch", "Exact GPU architecture for the probe code object") orelse "gfx1151";
    if (caps.Caps.of(hip_arch) == null or std.mem.indexOfScalar(u8, hip_arch, ':') != null)
        std.debug.panic("-Dhip-arch {s} is not an exact name in the caps table (zig/src/hip/caps.zig)", .{hip_arch});
    const hip_compile = b.addSystemCommand(&.{ hipcc, "--genco", b.fmt("--offload-arch={s}", .{hip_arch}), "-O2", "-ffp-contract=off" });
    hip_compile.addFileArg(b.path("zig/kernels/hip/runtime_tests.hip"));
    hip_compile.addArg("-o");
    const hip_object = hip_compile.addOutputFileArg("hip-runtime-probe.hsaco");
    const hip_files = b.addWriteFiles();
    _ = hip_files.addCopyFile(hip_object, "probe.hsaco");
    const hip_probe = b.createModule(.{ .root_source_file = hip_files.add("probe.zig", b.fmt("pub const arch = \"{s}\";\npub const bytes align(8) = @embedFile(\"probe.hsaco\").*;\n", .{hip_arch})) });
    const hip_gpu_module = b.createModule(.{ .root_source_file = b.path("zig/src/hip/runtime_tests.zig"), .target = target, .link_libc = true });
    hip_gpu_module.addIncludePath(.{ .cwd_relative = hip_include });
    hip_gpu_module.addCSourceFile(.{ .file = b.path("zig/src/hip/device_arch.c"), .flags = &.{"-D__HIP_PLATFORM_AMD__"} });
    hip_gpu_module.addImport("hip_probe", hip_probe);
    const hip_gpu_test = b.addTest(.{ .root_module = hip_gpu_module });
    b.step("hip-gpu-build", "Compile HIP runtime tests without running GPU work").dependOn(&hip_gpu_test.step);
    b.step("hip-gpu-test", "Real HIP copies, fills and architecture-selected module launches").dependOn(&b.addRunArtifact(hip_gpu_test).step);
    const affine_compile = b.addSystemCommand(&.{ hipcc, "--genco", b.fmt("--offload-arch={s}", .{hip_arch}), "-O2", "-ffp-contract=off" });
    affine_compile.addFileArg(b.path("zig/kernels/hip/affine.hip"));
    affine_compile.addArg("-o");
    const affine_image = affine_compile.addOutputFileArg("affine.hsaco");
    const affine_files = b.addWriteFiles();
    _ = affine_files.addCopyFile(affine_image, "affine.hsaco");
    _ = affine_files.addCopyFile(b.path("zig/tests/hip_affine_g64.hex"), "golden.hex");
    _ = affine_files.addCopyFile(b.path("zig/tests/hip_affine_sensitive.hex"), "sensitive.hex");
    _ = affine_files.addCopyFile(b.path("zig/tests/hip_affine_matrix.hex"), "matrix.hex");
    const affine_data = b.createModule(.{ .root_source_file = affine_files.add("data.zig", b.fmt("pub const arch = \"{s}\";\npub const image align(8) = @embedFile(\"affine.hsaco\").*;\npub const hex = @embedFile(\"golden.hex\");\npub const sensitive = @embedFile(\"sensitive.hex\");\npub const matrix = @embedFile(\"matrix.hex\");\n", .{hip_arch})) });
    const affine_module = b.createModule(.{ .root_source_file = b.path("zig/src/hip/affine_gpu_test.zig"), .target = target, .link_libc = true });
    affine_module.addIncludePath(.{ .cwd_relative = hip_include });
    affine_module.addImport("affine_data", affine_data);
    affine_module.addCSourceFile(.{ .file = b.path("zig/src/hip/device_arch.c"), .flags = &.{"-D__HIP_PLATFORM_AMD__"} });
    const affine_test = b.addTest(.{ .root_module = affine_module });
    b.step("hip-affine-build", "Compile affine golden GPU test without executing").dependOn(&affine_test.step);
    b.step("hip-affine-test", "Run exact affine golden GPU regression").dependOn(&b.addRunArtifact(affine_test).step);
    const device_lib = b.option([]const u8, "hip-device-lib", "ROCm device bitcode directory (default <rocm>/lib/llvm/amdgcn/bitcode; Arch: <rocm>/amdgcn/bitcode)");
    const objects = kernelObjects(b, hipcc, hipcc_resolved, device_lib, hip_arch);
    const kernel_module = b.createModule(.{ .root_source_file = b.path("zig/src/hip/kernel_tests.zig"), .target = target, .link_libc = true });
    kernel_module.addIncludePath(.{ .cwd_relative = hip_include });
    kernel_module.addCSourceFile(.{ .file = b.path("zig/src/hip/device_arch.c"), .flags = &.{"-D__HIP_PLATFORM_AMD__"} });
    kernel_module.addImport("hip_kernels", objects);
    kernel_module.addImport("core", coreModule(b, target));
    const kernel_test = b.addTest(.{ .root_module = kernel_module });
    b.step("hip-kernel-build", "Compile the model-free kernels and their GPU tests without running them").dependOn(&kernel_test.step);
    b.step("hip-kernel-test", "Every model-free kernel against its host reference on the GPU").dependOn(&b.addRunArtifact(kernel_test).step);
    const stub_kernels = b.createModule(.{ .root_source_file = b.addWriteFiles().add("kernels.zig", stub) });
    const host = engineModules(b, b.graph.host, .Debug, stub_kernels, null);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.hip })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.family })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.native })).step);
    // the GPU programs compile on any host against stub kernels, so `test` catches what only they reach
    test_step.dependOn(&checksExe(b, b.graph.host, .Debug, host).step);
    test_step.dependOn(&server(b, b.graph.host, host, build_options).step);
    const gpu = engineModules(b, target, optimize, objects, hip_include);
    b.step("tf-qwen35-test", "Qwen3.5 / 3.6 model checks on the GPU (-Dhip-arch)").dependOn(&b.addInstallArtifact(checksExe(b, target, optimize, gpu), .{}).step);
    const native = server(b, target, gpu, build_options);
    b.step("native-hip", "tensorfold-native with the HIP engines into zig-out/native-hip/bin (-Dhip-arch)").dependOn(&b.addInstallArtifact(native, .{ .dest_dir = .{ .override = .{ .custom = "native-hip/bin" } } }).step);
}

/// `tf-qwen35-test`: the model checks on a real checkpoint and GPU.
fn checksExe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, e: Engine) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "tf-qwen35-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/qwen35/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = e.hip }, .{ .name = "qwen3_5", .module = e.family }, .{ .name = "lanes", .module = e.lanes }, .{ .name = "npy", .module = b.createModule(.{ .root_source_file = b.path("zig/src/core/npy.zig"), .target = target, .optimize = optimize }) } },
    }) });
}

const Engine = struct { hip: *std.Build.Module, lanes: *std.Build.Module, api: *std.Build.Module, family: *std.Build.Module, native: *std.Build.Module };

/// The HIP backend, the Qwen3.5 family over it and the native engines, on `kernels` (a stub on the host).
fn engineModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, kernels: *std.Build.Module, hip_include: ?[]const u8) Engine {
    const core = coreModule(b, target);
    const hip = b.createModule(.{ .root_source_file = b.path("zig/src/hip/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    hip.addImport("core", core);
    hip.addImport("hip_kernels", kernels);
    hip.addImport("cuda_memory", b.createModule(.{ .root_source_file = b.path("zig/src/native/cuda_memory.zig"), .target = target, .optimize = optimize }));
    if (hip_include) |dir| {
        hip.addIncludePath(.{ .cwd_relative = dir });
        hip.addCSourceFile(.{ .file = b.path("zig/src/hip/device_arch.c"), .flags = &.{"-D__HIP_PLATFORM_AMD__"} });
    }
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3_5/hip.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "engine_api", .module = api } },
    });
    const native = b.createModule(.{
        .root_source_file = b.path("zig/src/native/hip.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "qwen3_5", .module = family } },
    });
    return .{ .hip = hip, .lanes = lanes, .api = api, .family = family, .native = native };
}

/// The server over the HIP engines, as the CUDA build makes it: the HTTP side keeps its safety checks.
fn server(b: *std.Build, target: std.Build.ResolvedTarget, e: Engine, build_options: *std.Build.Step.Options) *std.Build.Step.Compile {
    const api = e.api;
    const engines = e.native;
    const tokenizer = e.hip.import_table.get("core").?.import_table.get("tokenizer").?;
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = engines }} }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    return exe;
}

/// The model-free kernels' source groups, in zig/src/hip/kernels.zig's Group order.
const groups = [_][]const u8{ "ops/ops.hip", "ops/act.hip", "attention/attention.hip", "recurrence/gated_delta.hip", "attention/prefill.hip", "recurrence/gdn_prefill.hip", "decode/decode.hip", "decode/plan.hip", "tiles/dot2_tiles.hip", "tiles/dot2.hip" };

/// What the group sources include, so an edit to one rebuilds them.
const headers = [_][]const u8{
    "attention/attention_fa.hip", "decode/pages.hpp", "decode/plan.hpp",     "ops/attention.hpp",     "ops/common.hpp",
    "ops/draw.hpp",               "ops/tp.hpp",       "ops/elementwise.hpp", "ops/linear.hpp",        "ops/moe.hpp",
    "ops/norms.hpp",              "ops/rope.hpp",     "common/arch.hpp",     "common/dot2.hpp",       "common/vec.hpp",
    "common/wmma.hpp",            "quant/act.hpp",    "quant/mlx.hpp",       "quant/mlx_decoder.hpp", "quant/mlx_pieces.hpp",
    "quant/mlx_tiles.hpp",        "tiles/dot2.hpp",   "tiles/epilogue.hpp",  "tiles/gemm.hpp",        "tiles/gemm_kp.hpp",
    "tiles/matrix_gemm.hpp",      "tiles/plan.hpp",   "tiles/stream.hpp",
};

/// The flags of the kernels' first build: no contraction, wave32 on RDNA, C++20.
const flags = [_][]const u8{ "-D__HIP_PLATFORM_AMD__=1", "-DUSE_ROCM=1", "-DHIPBLAS_V2", "-fPIC", "-DCUDA_HAS_FP16=1", "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1", "-std=c++20", "-fno-gpu-rdc", "-mno-wavefrontsize64", "-ffp-contract=off" };

const stub = "pub const arch = \"\";\npub const images: [10][]align(8) const u8 = @splat(&.{});\n";

/// One code object a group for `arch`, built with the caps table's instruction switches, as the `hip_kernels` module.
fn kernelObjects(b: *std.Build, hipcc: []const u8, hipcc_resolved: []const u8, device_lib: ?[]const u8, arch: []const u8) *std.Build.Module {
    const c = caps.Caps.of(arch).?;
    const root = std.fs.path.dirname(std.fs.path.dirname(hipcc_resolved) orelse ".") orelse ".";
    const files = b.addWriteFiles();
    var decls: []const u8 = "";
    var refs: []const u8 = "";
    for (groups, 0..) |source, i| {
        const run = b.addSystemCommand(&.{ hipcc, "--genco" });
        run.addArgs(&flags);
        run.addArg(b.fmt("-DTF_WAVE={d}", .{c.wave}));
        run.addArg(b.fmt("-DTF_DOT2_F16={d}", .{@intFromBool(c.dot2_f16)}));
        run.addArg(b.fmt("-DTF_DOT2_BF16={d}", .{@intFromBool(c.dot2_bf16)}));
        run.addArg(b.fmt("-DTF_SDOT4={d}", .{@intFromBool(c.sdot4)}));
        run.addArg(b.fmt("-DTF_SDOT8={d}", .{@intFromBool(c.sdot8)}));
        run.addArg(b.fmt("-DTF_MATRIX={d}", .{@intFromBool(c.matrix != .none)}));
        run.addArg(b.fmt("--rocm-path={s}", .{root}));
        run.addArg(b.fmt("--rocm-device-lib-path={s}", .{device_lib orelse b.fmt("{s}/lib/llvm/amdgcn/bitcode", .{root})}));
        run.addArg(b.fmt("--offload-arch={s}", .{arch}));
        run.addPrefixedDirectoryArg("-I", b.path("zig/kernels/hip"));
        for (headers) |h| run.addFileInput(b.path(b.fmt("zig/kernels/hip/{s}", .{h})));
        run.addArg("-o");
        const object = run.addOutputFileArg(b.fmt("group{d}.hsaco", .{i}));
        run.addFileArg(b.path(b.fmt("zig/kernels/hip/{s}", .{source})));
        _ = files.addCopyFile(object, b.fmt("group{d}.hsaco", .{i}));
        decls = b.fmt("{s}const g{d} align(8) = @embedFile(\"group{d}.hsaco\").*;\n", .{ decls, i, i });
        refs = b.fmt("{s}&g{d}, ", .{ refs, i });
    }
    return b.createModule(.{ .root_source_file = files.add("kernels.zig", b.fmt("pub const arch = \"{s}\";\n{s}pub const images = [_][]align(8) const u8{{ {s}}};\n", .{ arch, decls, refs })) });
}

/// The backend-neutral core (checkpoint formats, the kernel registry), as the other backends import it.
fn coreModule(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .link_libc = true });
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .link_libc = true }));
    return core;
}
