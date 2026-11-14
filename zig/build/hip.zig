//! HIP, built only with -Dhip=true: admission tests in `test`, GPU steps; options on every target (#1165).
const std = @import("std");
const builtin = @import("builtin");
const caps = @import("../src/hip/caps.zig");
const exlr8_build = @import("exlr8.zig");
const LazyPath = std.Build.LazyPath;
/// Some HIP SDKs ship only `hipcc.bat`, which CreateProcess cannot start: run it through `cmd /c call`.
fn hipccRun(b: *std.Build, hipcc: []const u8, args: []const []const u8) *std.Build.Step.Run {
    const bat = builtin.os.tag == .windows and std.mem.endsWith(u8, hipcc, ".bat");
    const run = if (bat)
        b.addSystemCommand(&.{ "cmd", "/c", "call" })
    else
        b.addSystemCommand(&.{hipcc});
    if (bat) run.addArg(hipcc);
    run.addArgs(args);
    return run;
}
/// The HIP compiler: `-Dhipcc`, then PATH, then `HIP_PATH`/`ROCM_PATH`'s bin (`hipcc.exe` first on Windows).
fn resolveHipcc(b: *std.Build, given: ?[]const u8) []const u8 {
    if (given) |path| return b.findProgram(.{ .names = &.{path} }) orelse path;
    const names: []const []const u8 = if (builtin.os.tag == .windows)
        &.{ "hipcc.exe", "hipcc.bat", "hipcc" }
    else
        &.{"hipcc"};
    if (b.findProgram(.{ .names = names })) |found| return found;
    for ([_][]const u8{ "HIP_PATH", "ROCM_PATH" }) |root| {
        const root_path = b.graph.environ_map.get(root) orelse continue;
        for ([_][]const u8{ "hipcc.exe", "hipcc.bat" }) |name| {
            const candidate = b.pathResolve(&.{ root_path, "bin", name });
            if (std.Io.Dir.accessAbsolute(b.graph.io, candidate, .{})) |_| return candidate else |_| {}
        }
    }
    return names[0];
}
/// HIP's options as given, declared whether or not -Dhip=true builds anything with them.
pub const Options = struct { hipcc: ?[]const u8, hip_include: ?[]const u8, hip_arch: ?[]const u8, device_lib: ?[]const u8, relax_branches: bool };
pub fn options(b: *std.Build) Options {
    return .{
        .hipcc = b.option([]const u8, "hipcc", "HIP compiler for model-free tests"),
        .hip_include = b.option([]const u8, "hip-include", "HIP header directory"),
        .hip_arch = b.option([]const u8, "hip-arch", "Exact GPU architecture for the probe code object"),
        .device_lib = b.option([]const u8, "hip-device-lib", "ROCm device bitcode directory (default <rocm>/lib/llvm/amdgcn/bitcode; Arch: <rocm>/amdgcn/bitcode)"),
        .relax_branches = b.option(bool, "hip-relax-branches", "Relax gfx1100 branches earlier (-amdgpu-s-branch-bits=15), for clang 21 (#1521)") orelse false,
    };
}
/// The Qwen3.5 / 3.6 checkpoint loader tests: part of `test` on every build, HIP or not.
pub fn loaderTests(b: *std.Build, test_step: *std.Build.Step) void {
    const loader = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3_5/loader.zig"),
        .target = b.graph.host,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = coreModule(b, b.graph.host) }},
    }) }));
    loader.has_side_effects = true; // TF_QWEN_DIR is not a cached input
    test_step.dependOn(&loader.step);
    loader.step.dependOn(@import("mesh.zig").qwenDeclTest(b, coreModule(b, b.graph.host))); // its cuts as declarations
    b.step("test-qwen35", "Qwen3.5 / 3.6 checkpoint loader tests; TF_QWEN_DIR indexes a real checkpoint").dependOn(&loader.step);
}
/// hip-host-test (part of `test`), hip-gpu-build and hip-gpu-test, hip-affine-build and hip-affine-test.
pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options, test_step: *std.Build.Step, o: Options) void {
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
    const hipcc_resolved = resolveHipcc(b, o.hipcc);
    const hipcc = hipcc_resolved;
    const hip_include = o.hip_include orelse
        b.pathResolve(&.{ std.fs.path.dirname(hipcc_resolved) orelse "/opt/rocm/bin", "..", "include" });
    const hip_arch = o.hip_arch orelse "gfx1151";
    if (caps.Caps.of(hip_arch) == null or std.mem.indexOfScalar(u8, hip_arch, ':') != null)
        std.debug.panic("-Dhip-arch {s} is not an exact name in the caps table (zig/src/hip/caps.zig)", .{hip_arch});
    const hip_compile = hipccRun(b, hipcc, &.{ "--genco", b.fmt("--offload-arch={s}", .{hip_arch}), "-O2", "-ffp-contract=off" });
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
    const affine_compile = hipccRun(b, hipcc, &.{ "--genco", b.fmt("--offload-arch={s}", .{hip_arch}), "-O2", "-ffp-contract=off" });
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
    const device_lib = o.device_lib;
    @import("dots_mocr.zig").columnObjects(b, .{ .name = hipcc, .resolved = hipcc_resolved, .device_lib = device_lib, .relax_branches = o.relax_branches });
    const objects = kernelObjects(b, hipcc, hipcc_resolved, device_lib, o.relax_branches, hip_arch);
    const kernel_module = b.createModule(.{ .root_source_file = b.path("zig/src/hip/kernel_tests.zig"), .target = target, .link_libc = true });
    kernel_module.addIncludePath(.{ .cwd_relative = hip_include });
    kernel_module.addCSourceFile(.{ .file = b.path("zig/src/hip/device_arch.c"), .flags = &.{"-D__HIP_PLATFORM_AMD__"} });
    kernel_module.addImport("hip_kernels", objects);
    @import("flashnext_reach.zig").wire(b, target, kernel_module, coreModule(b, target));
    const kernel_test = b.addTest(.{ .root_module = kernel_module });
    b.step("hip-kernel-build", "Compile the model-free kernels and their GPU tests without running them").dependOn(&kernel_test.step);
    b.step("hip-kernel-test", "Every model-free kernel against its host reference on the GPU").dependOn(&b.addRunArtifact(kernel_test).step);
    const tune_module = b.createModule(.{ .root_source_file = b.path("zig/src/hip/affine_tune.zig"), .target = target, .link_libc = true });
    tune_module.addIncludePath(.{ .cwd_relative = hip_include });
    tune_module.addCSourceFile(.{ .file = b.path("zig/src/hip/device_arch.c"), .flags = &.{"-D__HIP_PLATFORM_AMD__"} });
    tune_module.addImport("hip_kernels", objects);
    tune_module.addImport("core", coreModule(b, target));
    const tune_test = b.addTest(.{ .root_module = tune_module });
    b.step("hip-affine-tune-build", "Compile the MLX affine prompt-tile timing step without running it").dependOn(&tune_test.step);
    const tune_run = b.addRunArtifact(tune_test);
    tune_run.has_side_effects = true; // timings: run every time, never from the cache
    b.step("hip-affine-tune", "Time every MLX affine prompt tile the GPU has and propose table rows (-Dhip-arch)").dependOn(&tune_run.step);
    const dense_tune_module = b.createModule(.{ .root_source_file = b.path("zig/src/hip/dense_tune.zig"), .target = target, .link_libc = true });
    dense_tune_module.addIncludePath(.{ .cwd_relative = hip_include });
    dense_tune_module.addCSourceFile(.{ .file = b.path("zig/src/hip/device_arch.c"), .flags = &.{"-D__HIP_PLATFORM_AMD__"} });
    dense_tune_module.addImport("hip_kernels", objects);
    dense_tune_module.addImport("core", coreModule(b, target));
    const dense_tune_run = b.addRunArtifact(b.addTest(.{ .root_module = dense_tune_module }));
    dense_tune_run.has_side_effects = true; // timings: run every time, never from the cache
    b.step("hip-dense-tune", "Time the dot2 and WMMA BF16 dense tiles at Qwen3-8B's shapes (-Dhip-arch)").dependOn(&dense_tune_run.step);
    const stub_kernels = b.createModule(.{ .root_source_file = b.addWriteFiles().add("kernels.zig", stub) });
    const host = engineModules(b, b.graph.host, .Debug, stub_kernels, null);
    b.step("test-environment-hip-host", "Named HIP budget refusals without GPU work").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.hip, .filters = &.{ "HIP environment:", "a carve-out needs the card" } })).step);
    b.step("test-slide-hip-host", "Unsupported HIP learning refuses before device or checkpoint access").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.native, .filters = &.{"slide admission:"} })).step);
    const blas_host = b.addRunArtifact(b.addTest(.{ .root_module = host.hip, .filters = &.{ "HIP BLAS:", "HIP fixed BLAS:", "HIP convolution:", "HIP AOTriton:" } }));
    b.step("test-hip-blas-host", "HIP BLAS, convolution and attention admission without device execution").dependOn(&blas_host.step);
    const device_checks = engineModules(b, target, .safe, stub_kernels, null);
    const blas_device = b.addTest(.{ .name = "tf-hip-blas-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/blas_test.zig"),
        .target = target,
        .optimize = .safe,
        .link_libc = true,
        .imports = &.{.{ .name = "hip", .module = device_checks.hip }},
    }), .filters = &.{"HIP BLAS device:"} });
    b.step("build-hip-blas-test", "Build BF16 projection device checks; execution requires device selection and admission").dependOn(&b.addInstallArtifact(blas_device, .{}).step);
    @import("glm_ocr.zig").hipTests(b, host.hip, device_checks.hip, test_step, o);
    const native_ocr = b.step("test-glm-ocr-native-host", "GLM-OCR native registry and preallocation refusals without devices");
    native_ocr.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.native, .filters = &.{"GLM-OCR native flags:"} })).step);
    native_ocr.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.native.import_table.get("glm_ocr").?, .filters = &.{"GLM-OCR native admission:"} })).step);
    b.step("test-hip-family-offers-host", "HIP optional native family metadata without device execution").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.native, .filters = &.{"HIP optional family offers:"} })).step);
    b.step("test-hip-logit-bias-host", "Host bias and ban draws without a HIP device").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.serve, .filters = &.{"logit_bias"} })).step);
    b.step("test-hip-native-options-host", "Shared HIP native open settings without device execution").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.serve, .filters = &.{"HIP native options:"} })).step);
    @import("dots_mocr.zig").hipTests(b, host.hip, test_step);
    b.step("test-dots-native-host", "dots.mocr registry, reasoning, and unsupported flags without GPU execution").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.native, .filters = &.{ "dots refuses", "every registered family", "a HIP family declares replies without reasoning" } })).step);
    @import("lighton_hip.zig").host(b, host.lighton, test_step);
    const qwen_vision_host = b.addRunArtifact(b.addTest(.{ .root_module = host.family.import_table.get("qwen_hip_vision").? }));
    b.step("test-qwen-vision-hip-host", "Qwen HIP vision settings, layouts and memory admission without devices").dependOn(&qwen_vision_host.step);
    test_step.dependOn(&qwen_vision_host.step);
    const vision_contract_host = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/contract_qwen_vision.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = host.hip }, .{ .name = "core", .module = host.hip.import_table.get("core").? }, .{ .name = "qwen_hip_vision", .module = host.family.import_table.get("qwen_hip_vision").? } },
    }) }));
    b.step("test-qwen-vision-contract-host", "Qwen HIP contract references and wrappers without a device").dependOn(&vision_contract_host.step);
    test_step.dependOn(&vision_contract_host.step);
    b.step("test-draft-hip-host", "HIP draft source metadata without devices").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.native, .filters = &.{"draft sources:"} })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.hip })).step);
    b.step("test-qwen-hip-formats-host", "Qwen HIP format offers: the BF16 refusal, and discovery against actual config and matrix admission").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.family, .filters = &.{ "a bf16 checkpoint without a joint head", "capability audit: HIP Qwen format" } })).step);
    const family_host = b.addRunArtifact(b.addTest(.{ .root_module = host.family }));
    test_step.dependOn(&family_host.step);
    b.step("test-qwen35-hip-host", "Qwen HIP family CPU tests with inert kernels").dependOn(&family_host.step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.flashnext })).step);
    b.step("test-flashnext-proposal-hip-host", "Original Flash Next HIP IDs and head-row spans without device runtime").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.flashnext, .filters = &.{"Flash Next proposal:"} })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.nemotron })).step);
    const qwen3_host = b.addRunArtifact(b.addTest(.{ .root_module = host.qwen3 }));
    test_step.dependOn(&qwen3_host.step);
    b.step("test-qwen3-hip-host", "Qwen3 dense config, checkpoint prefix and lm_head rows on the CPU").dependOn(&qwen3_host.step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.serve })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = host.parse })).step);
    const native_host_tests = b.addRunArtifact(b.addTest(.{ .root_module = host.native }));
    test_step.dependOn(&native_host_tests.step);
    b.step("test-native-hip-host", "Native HIP family offers and flags without a device").dependOn(&native_host_tests.step);
    // the GPU programs compile on any host against stub kernels, so `test` catches what only they reach
    test_step.dependOn(&checksExe(b, b.graph.host, .Debug, host).step);
    test_step.dependOn(&clefExe(b, b.graph.host, .Debug, host).step);
    test_step.dependOn(&server(b, b.graph.host, host, build_options).step);
    const check_target = if (b.graph.host.result.os.tag == .windows)
        b.graph.host
    else
        b.resolveTargetQuery(.{ .cpu_arch = b.graph.host.result.cpu.arch, .os_tag = .linux, .abi = .gnu });
    const check = engineModules(b, check_target, .Debug, stub_kernels, null);
    b.step("check-native-hip-host", "Compile the real native HIP server with inert kernels, without GPU execution").dependOn(&server(b, check_target, check, build_options).step);
    const gpu = engineModules(b, target, optimize, objects, hip_include);
    const qwen3_gpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3/forward_test.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = gpu.hip }, .{ .name = "qwen3", .module = gpu.qwen3 } },
    }) });
    const qwen3_run = b.addRunArtifact(qwen3_gpu);
    qwen3_run.has_side_effects = true; // the checkpoints and oracle files named by TF_QWEN3_* are not cached inputs
    b.step("qwen3-hip-gpu-build", "Compile the Qwen3 dense GPU checks without running them").dependOn(&qwen3_gpu.step);
    b.step("qwen3-hip-gpu-test", "Qwen3 lastRows on the GPU against transformers (TF_QWEN3_EMBED_*, TF_QWEN3_RERANK_*)").dependOn(&qwen3_run.step);
    const parse_object = compile(b, .{ .name = hipcc, .resolved = hipcc_resolved, .device_lib = device_lib }, hip_arch, b.path("zig/kernels/hip/ops/vision.hip"), "parse-vision.hsaco");
    @import("nemotron_parse_gpu.zig").gpuTests(b, target, host.hip, parse_object);
    @import("nemotron_parse_gpu.zig").modelTests(b, target, optimize, gpu.hip);
    b.step("tf-qwen35-test", "Qwen3.5 / 3.6 model checks on the GPU (-Dhip-arch)").dependOn(&b.addInstallArtifact(checksExe(b, target, optimize, gpu), .{}).step);
    b.step("tf-clef-test", "Clef's joint schema head on the GPU: the tiny fixture and gate 3 (-Dhip-arch)").dependOn(&b.addInstallArtifact(clefExe(b, target, optimize, gpu), .{}).step);
    const fx_test = b.addExecutable(.{ .name = "tf-flashnext-hip-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/flashnext_hip/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = gpu.hip },
            .{ .name = "flashnext", .module = gpu.flashnext },
            .{ .name = "tokenizer", .module = gpu.hip.import_table.get("core").?.import_table.get("tokenizer").? },
        },
    }) });
    b.step(
        "tf-flashnext-hip-test",
        "Flash Next model checks on the GPU (-Dhip-arch)",
    ).dependOn(&b.addInstallArtifact(fx_test, .{}).step);
    const fx_gpu = b.addTest(.{ .name = "tf-flashnext-hip-gpu-test", .root_module = gpu.flashnext, .filters = &.{"Flash Next fused prompt prep:"} });
    b.step("flashnext-hip-gpu-build", "Compile the Flash Next GPU checks without running them").dependOn(&fx_gpu.step);
    b.step("flashnext-hip-gpu-test", "Flash Next's DeltaNet through linear() on the GPU: the fused prompt prep against the unfused chain").dependOn(&b.addRunArtifact(fx_gpu).step);
    const nem_test = b.addExecutable(.{ .name = "tf-nemotron-hip-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/nemotron_hip/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = gpu.hip },
            .{ .name = "nemotron", .module = gpu.nemotron },
        },
    }) });
    b.step(
        "tf-nemotron-hip-test",
        "Nemotron-H load and skeleton checks on the GPU (-Dhip-arch)",
    ).dependOn(&b.addInstallArtifact(nem_test, .{}).step);
    @import("dots_mocr.zig").modelChecks(b, gpu.hip);
    const dots_owner = b.addTest(.{ .name = "tf-dots-hip-owner-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/hip/dots_owner_test.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = gpu.hip }, .{ .name = "dots_mocr", .module = gpu.native.import_table.get("dots_mocr").? }, .{ .name = "hip_serve", .module = gpu.native.import_table.get("hip_serve").? } },
    }), .filters = &.{"dots HIP device owner:"} });
    b.step("build-dots-hip-owner-test", "Build real owner close and failure cleanup checks without GPU execution").dependOn(&b.addInstallArtifact(dots_owner, .{}).step);
    const lighton_capture = b.addExecutable(.{ .name = "tf-lighton-hip-capture", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/lighton_hip/capture.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = gpu.hip }, .{ .name = "core", .module = gpu.hip.import_table.get("core").? }, .{ .name = "lighton_ocr", .module = gpu.lighton } },
    }) });
    b.step("tf-lighton-hip-capture", "Capture native HIP LightOnOCR stages against a local reference fixture").dependOn(&b.addInstallArtifact(lighton_capture, .{}).step);
    const qwen_capture_host = captureQwen(b, b.graph.host, host);
    b.step("check-qwen-hip-capture-host", "Compile the Qwen HIP fixture capture without device execution").dependOn(&qwen_capture_host.step);
    const qwen_capture_gpu = captureQwen(b, target, gpu);
    b.step("tf-qwen-hip-capture", "Capture native Qwen HIP stages against local reference fixtures").dependOn(&b.addInstallArtifact(qwen_capture_gpu, .{}).step);
    for ([_][]const u8{ "lighton", "qwen" }) |family_name| {
        const exact_host = exactTool(b, b.graph.host, host, family_name);
        b.step(b.fmt("check-{s}-hip-exact-host", .{family_name}), "Compile native HIP image consistency checks without execution").dependOn(&exact_host.step);
        const exact_gpu = exactTool(b, target, gpu, family_name);
        b.step(b.fmt("tf-{s}-hip-exact", .{family_name}), "Build native HIP image consistency checks").dependOn(&b.addInstallArtifact(exact_gpu, .{}).step);
    }
    const native = server(b, target, gpu, build_options);
    b.step("native-hip", "tensorfold-native with the HIP engines into zig-out/native-hip/bin (-Dhip-arch)").dependOn(&b.addInstallArtifact(native, .{ .dest_dir = .{ .override = .{ .custom = "native-hip/bin" } } }).step);
    @import("hip_contract.zig").steps(b, .{
        .target = target,
        .optimize = optimize,
        .hipcc = .{ .name = hipcc, .resolved = hipcc_resolved, .device_lib = device_lib, .relax_branches = o.relax_branches },
        .arch = hip_arch,
        .gpu = gpu.hip,
        .host = host.hip,
        .lighton_gpu = gpu.lighton,
        .lighton_host = host.lighton,
        .test_step = test_step,
    });
}
/// `tf-qwen35-test`: the model checks on a real checkpoint and GPU.
fn checksExe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, e: Engine) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "tf-qwen35-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/qwen35/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = e.hip }, .{ .name = "qwen3_5", .module = e.family }, .{ .name = "lanes", .module = e.lanes }, .{ .name = "npy", .module = b.createModule(.{ .root_source_file = b.path("zig/npy_from_core.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = e.hip.import_table.get("core").? }} }) } },
    }) });
}
fn clefExe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, e: Engine) *std.Build.Step.Compile {
    const core = e.hip.import_table.get("core").?;
    return b.addExecutable(.{ .name = "tf-clef-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/clef/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = e.hip }, .{ .name = "core", .module = core }, .{ .name = "npy", .module = b.createModule(.{ .root_source_file = b.path("zig/npy_from_core.zig"), .target = target, .optimize = optimize, .imports = &.{.{ .name = "core", .module = core }} }) } },
    }) });
}
pub const Engine = struct {
    hip: *std.Build.Module,
    lanes: *std.Build.Module,
    api: *std.Build.Module,
    serve: *std.Build.Module,
    family: *std.Build.Module,
    flashnext: *std.Build.Module,
    nemotron: *std.Build.Module,
    qwen3: *std.Build.Module,
    parse: *std.Build.Module,
    lighton: *std.Build.Module,
    native: *std.Build.Module,
};
/// The HIP backend, the Qwen3.5 family over it and the native engines, on `kernels` (a stub on the host).
pub fn engineModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, kernels: *std.Build.Module, hip_include: ?[]const u8) Engine {
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
    const serve = b.createModule(.{
        .root_source_file = b.path("zig/src/hip_serve/serve.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = hip },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "engine_api", .module = api },
        },
    });
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3_5/hip.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = hip },
            .{ .name = "core", .module = core },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "engine_api", .module = api },
            .{ .name = "hip_serve", .module = serve },
        },
    });
    const qwen_vision = @import("lighton_hip.zig").vision(b, target, optimize, hip, core, lanes, serve);
    family.addImport("qwen_hip_vision", qwen_vision);
    const lighton = @import("lighton_hip.zig").module(b, target, optimize, hip, core, lanes, api, serve);
    const flashnext = b.createModule(.{
        .root_source_file = b.path("zig/src/families/flashnext/hip.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = hip },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "core", .module = core },
            .{ .name = "engine_api", .module = api },
            .{ .name = "hip_serve", .module = serve },
        },
    });
    const nemotron = b.createModule(.{
        .root_source_file = b.path("zig/src/families/nemotron/hip.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = hip },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "core", .module = core },
            .{ .name = "engine_api", .module = api },
            .{ .name = "hip_serve", .module = serve },
        },
    });
    const ocr_kernels = b.createModule(.{ .root_source_file = b.addWriteFiles().add("glm_ocr_kernels.zig", "const hip = @import(\"hip\"); pub const arch = hip.kernels.arch; pub const bytes = hip.kernels.images[@intFromEnum(hip.kernels.Group.glm_ocr)];\n"), .imports = &.{.{ .name = "hip", .module = hip }} });
    const glm_ocr = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_ocr/hip.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "engine_api", .module = api }, .{ .name = "hip_serve", .module = serve }, .{ .name = "glm_ocr_kernels", .module = ocr_kernels } } });
    const qwen3 = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3/qwen3.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = hip },
            .{ .name = "core", .module = core },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "engine_api", .module = api },
            .{ .name = "hip_serve", .module = serve },
        },
    });
    const dots = @import("dots_mocr.zig").hipModule(b, hip);
    dots.addImport("lanes", lanes);
    dots.addImport("engine_api", api);
    dots.addImport("hip_serve", serve);
    const parse = b.createModule(.{
        .root_source_file = b.path("zig/src/families/nemotron_parse/hip.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = hip }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "engine_api", .module = api }, .{ .name = "hip_serve", .module = serve } },
    });
    const native = b.createModule(.{
        .root_source_file = b.path("zig/src/native/hip.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "hip", .module = hip },
            .{ .name = "engine_api", .module = api },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "qwen3_5", .module = family },
            .{ .name = "flashnext", .module = flashnext },
            .{ .name = "nemotron", .module = nemotron },
            .{ .name = "glm_ocr", .module = glm_ocr },
            .{ .name = "hip_serve", .module = serve },
            .{ .name = "qwen3", .module = qwen3 },
            .{ .name = "dots_mocr", .module = dots },
            .{ .name = "nemotron_parse", .module = parse },
            .{ .name = "lighton_ocr", .module = lighton },
        },
    });
    return .{
        .hip = hip,
        .lanes = lanes,
        .api = api,
        .serve = serve,
        .family = family,
        .flashnext = flashnext,
        .nemotron = nemotron,
        .qwen3 = qwen3,
        .parse = parse,
        .lighton = lighton,
        .native = native,
    };
}
/// The server over the HIP engines, as the CUDA build makes it: the HTTP side keeps its safety checks.
pub fn server(b: *std.Build, target: std.Build.ResolvedTarget, e: Engine, build_options: *std.Build.Step.Options) *std.Build.Step.Compile {
    const api = e.api;
    const engines = e.native;
    const tokenizer = e.hip.import_table.get("core").?.import_table.get("tokenizer").?;
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = e.hip.import_table.get("core").? }, .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "native_engines", .module = engines }, exlr8_build.cliImport(b, target, .ReleaseSafe, e.hip.import_table.get("core").?), .{ .name = "core", .module = e.hip.import_table.get("core").? } } }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    _ = @import("server.zig").addTo(b, exe.root_module, target, .ReleaseSafe);
    return exe;
}
fn captureQwen(b: *std.Build, target: std.Build.ResolvedTarget, e: Engine) *std.Build.Step.Compile {
    return b.addExecutable(.{ .name = "tf-qwen-hip-capture", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/lighton_hip/qwen_capture.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = e.hip }, .{ .name = "core", .module = e.hip.import_table.get("core").? }, .{ .name = "qwen3_5", .module = e.family }, .{ .name = "lanes", .module = e.lanes }, .{ .name = "qwen_hip_vision", .module = e.family.import_table.get("qwen_hip_vision").? } },
    }) });
}
fn exactTool(b: *std.Build, target: std.Build.ResolvedTarget, e: Engine, family_name: []const u8) *std.Build.Step.Compile {
    const file = if (std.mem.eql(u8, family_name, "lighton")) "exact.zig" else "qwen_exact.zig";
    return b.addExecutable(.{ .name = b.fmt("tf-{s}-hip-exact", .{family_name}), .root_module = b.createModule(.{
        .root_source_file = b.path(b.fmt("zig/tests/lighton_hip/{s}", .{file})),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "hip", .module = e.hip }, .{ .name = "core", .module = e.hip.import_table.get("core").? }, .{ .name = "lighton_ocr", .module = e.lighton }, .{ .name = "qwen3_5", .module = e.family }, .{ .name = "lanes", .module = e.lanes }, .{ .name = "qwen_hip_vision", .module = e.family.import_table.get("qwen_hip_vision").? } },
    }) });
}
/// A group's source and whether it compiles with IEEE arithmetic (`strict`: -O2 -fno-fast-math, IEEE denormals).
const Source = struct { path: []const u8, strict: bool = false };
/// The model-free kernels' source groups, in zig/src/hip/kernels.zig's Group order.
const groups = [_]Source{
    .{ .path = "ops/ops.hip" },
    .{ .path = "ops/act.hip" },
    .{ .path = "attention/attention.hip" },
    .{ .path = "recurrence/gated_delta.hip" },
    .{ .path = "attention/prefill.hip" },
    .{ .path = "decode/decode.hip" },
    .{ .path = "decode/plan.hip" },
    .{ .path = "tiles/dot2_tiles.hip" },
    .{ .path = "tiles/dot2.hip" },
    .{ .path = "flashnext/flashnext.hip" },
    .{ .path = "recurrence/mamba.hip" },
    .{ .path = "glm_ocr/pointwise.hip" },
    .{ .path = "joint_schema/joint_schema.hip" },
    .{ .path = "lighton_ocr.hip", .strict = true },
    .{ .path = "qwen_vision.hip", .strict = true },
    .{ .path = "draft/draft.hip" },
};
/// What the group sources include, so an edit to one rebuilds them.
const headers = [_][]const u8{
    "attention/attention_fa.hip", "attention/prefill.hpp", "decode/pages.hpp",      "decode/plan.hpp",               "ops/attention.hpp",
    "ops/common.hpp",             "ops/draw.hpp",          "ops/tp.hpp",            "ops/elementwise.hpp",           "ops/linear.hpp",
    "ops/moe.hpp",                "ops/norms.hpp",         "ops/rope.hpp",          "common/arch.hpp",               "common/dot2.hpp",
    "common/vec.hpp",             "common/wmma.hpp",       "quant/act.hpp",         "quant/mlx.hpp",                 "quant/mlx_decoder.hpp",
    "quant/mlx_pieces.hpp",       "quant/mlx_tiles.hpp",   "tiles/dot2.hpp",        "tiles/epilogue.hpp",            "tiles/gemm.hpp",
    "tiles/gemm_kp.hpp",          "tiles/matrix_gemm.hpp", "tiles/matrix_item.hpp", "tiles/plan.hpp",                "tiles/stream.hpp",
    "flashnext/streams.hpp",      "flashnext/ngram.hpp",   "flashnext/indexer.hpp", "flashnext/sparse.hpp",          "ops/nemotron_moe.hpp",
    "quant/bf16_decoder.hpp",     "quant/dense_tiles.hpp", "tiles/wmma12_gemm.hpp", "tiles/wmma12_gemm_k32_ref.hpp", "ops/columns.hpp",
    "columns_controls.hpp",       "ops/vision.hpp",        "ops/dense_text.hpp",    "ops/vl_position.hpp",           "flashnext/sparse_lanes.hpp",
};
/// The flags of the kernels' first build: no contraction, wave32 on RDNA, C++20.
const flags = [_][]const u8{ "-D__HIP_PLATFORM_AMD__=1", "-DUSE_ROCM=1", "-DHIPBLAS_V2", "-fPIC", "-DCUDA_HAS_FP16=1", "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1", "-std=c++20", "-fno-gpu-rdc", "-mno-wavefrontsize64", "-ffp-contract=off" };
const stub = std.fmt.comptimePrint("pub const arch = \"\";\npub const images: [{d}][]align(8) const u8 = @splat(&.{{}});\n", .{groups.len});
/// One code object a group for `arch`, built with the caps table's instruction switches, as the `hip_kernels` module.
pub fn kernelObjects(b: *std.Build, hipcc: []const u8, hipcc_resolved: []const u8, device_lib: ?[]const u8, relax_branches: bool, arch: []const u8) *std.Build.Module {
    const files = objectFiles(b, .{ .name = hipcc, .resolved = hipcc_resolved, .device_lib = device_lib, .relax_branches = relax_branches }, arch);
    var decls: []const u8 = "";
    var refs: []const u8 = "";
    for (0..groups.len) |i| {
        decls = b.fmt("{s}const g{d} align(8) = @embedFile(\"group{d}.hsaco\").*;\n", .{ decls, i, i });
        refs = b.fmt("{s}&g{d}, ", .{ refs, i });
    }
    return b.createModule(.{ .root_source_file = files.add("kernels.zig", b.fmt("pub const arch = \"{s}\";\n{s}pub const images = [_][]align(8) const u8{{ {s}}};\n", .{ arch, decls, refs })) });
}

/// The HIP compiler as the build found it, the device bitcode it links, and whether gfx1100 relaxes branches earlier.
pub const Hipcc = struct { name: []const u8, resolved: []const u8, device_lib: ?[]const u8, relax_branches: bool = false };

/// The groups' code objects for `arch` in one directory, group0.hsaco onward.
pub fn objectFiles(b: *std.Build, cc: Hipcc, arch: []const u8) *std.Build.Step.WriteFile {
    const files = b.addWriteFiles();
    for (groups, 0..) |g, i| {
        const name = b.fmt("group{d}.hsaco", .{i});
        const source = b.path(b.fmt("zig/kernels/hip/{s}", .{g.path}));
        _ = files.addCopyFile(compileSource(b, cc, arch, source, name, g.strict), name);
    }
    return files;
}

/// One source's code object for `arch`, with the kernels' flags and the caps table's instruction switches.
pub fn compile(b: *std.Build, cc: Hipcc, arch: []const u8, source: std.Build.LazyPath, out: []const u8) LazyPath {
    return compileSource(b, cc, arch, source, out, false);
}

/// `compile`, with `-O2 -fno-fast-math -fdenormal-fp-math=ieee` after the common flags when `strict`.
fn compileSource(b: *std.Build, cc: Hipcc, arch: []const u8, source: LazyPath, out: []const u8, strict: bool) LazyPath {
    const c = caps.Caps.of(arch).?;
    const root = std.fs.path.dirname(std.fs.path.dirname(cc.resolved) orelse ".") orelse ".";
    const run = hipccRun(b, cc.name, &.{"--genco"});
    run.addArgs(&flags);
    if (strict) run.addArgs(&.{ "-O2", "-fno-fast-math", "-fdenormal-fp-math=ieee" });
    run.addArg(b.fmt("-DTF_WAVE={d}", .{c.wave}));
    run.addArg(b.fmt("-DTF_DOT2_F16={d}", .{@intFromBool(c.dot2_f16)}));
    run.addArg(b.fmt("-DTF_DOT2_BF16={d}", .{@intFromBool(c.dot2_bf16)}));
    run.addArg(b.fmt("-DTF_SDOT4={d}", .{@intFromBool(c.sdot4)}));
    run.addArg(b.fmt("-DTF_SDOT8={d}", .{@intFromBool(c.sdot8)}));
    run.addArg(b.fmt("-DTF_MATRIX={d}", .{@intFromBool(c.matrix != .none)}));
    run.addArg(b.fmt("-DTF_DOT2_SIGNED_ZERO={d}", .{@intFromBool(c.dot2_signed_zero)}));
    run.addArg(b.fmt("--rocm-path={s}", .{root}));
    const lib = cc.device_lib orelse b.fmt("{s}/lib/llvm/amdgcn/bitcode", .{root});
    run.addArg(b.fmt("--rocm-device-lib-path={s}", .{lib}));
    run.addArg(b.fmt("--offload-arch={s}", .{arch}));
    // clang 21 leaves a branch of affine_dot2_stream_relu2_pair<Dot, 6, 1, 8> past simm16 on gfx1100 (#1521)
    if (cc.relax_branches and std.mem.eql(u8, arch, "gfx1100")) run.addArgs(&.{ "-mllvm", "-amdgpu-s-branch-bits=15" });
    run.addPrefixedDirectoryArg("-I", b.path("zig/kernels/hip"));
    for (headers) |h| run.addFileInput(b.path(b.fmt("zig/kernels/hip/{s}", .{h})));
    run.addArg("-o");
    const object = run.addOutputFileArg(out);
    run.addFileArg(source);
    return object;
}

/// The backend-neutral core (checkpoint formats, the kernel registry), as the other backends import it.
fn coreModule(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Module {
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .link_libc = true }));
    return core;
}
