const std = @import("std");
const glm_expiry_build = @import("zig/build/glm_expiry.zig");
const builtin = @import("builtin");
const cuda_build = @import("zig/build/cuda.zig");
const dist_build = @import("zig/build/dist.zig");
const qwen27_build = @import("zig/build/qwen27.zig");
const core_checks = @import("zig/build/core_checks.zig");
const glm_checks = @import("zig/build/glm_checks.zig");
const merge_build = @import("zig/build/merge.zig");
const flashnext_checks = @import("zig/build/flashnext_checks.zig");
const hip_build = @import("zig/build/hip.zig");
const xpu_build = @import("zig/build/xpu.zig");
const grammar_build = @import("zig/build/grammar.zig");
const counters_build = @import("zig/build/metal_counters.zig");
const exl3_build = @import("zig/build/exl3.zig");
const exlr8_build = @import("zig/build/exlr8.zig");
const server_build = @import("zig/build/server.zig");
const golden_build = @import("zig/build/golden.zig");
const contract_build = @import("zig/build/kernel_contract.zig");
const video_build = @import("zig/build/video.zig");
const sys_build = @import("zig/build/sys.zig");

comptime {
    const required = std.mem.trim(u8, @embedFile(".zig-version"), "\r\n");
    if (!std.mem.eql(u8, builtin.zig_version_string, required))
        @compileError("TensorFold's Zig engine requires Zig " ++ required);
}

const metal_lists = @import("zig/build/metal_source_lists.zig");
const kernels = metal_lists.kernels;
const generated = metal_lists.generated;
const kimi_dir = metal_lists.kimi_dir;
const kimi_kernels = metal_lists.kimi_kernels;
const nemotron_packed = metal_lists.nemotron_packed;

/// MLX's language version for mx.fast.metal_kernel on this macOS: 4.1 from 27, 4.0 on 26, 3.2 on 15, else 3.1.
fn mlxMetalStd(b: *std.Build) []const u8 {
    const os = b.graph.host.result.os;
    const major = if (os.tag == .macos) os.version_range.semver.min.major else 0;
    if (major >= 27) return "metal4.1";
    if (major >= 26) return "metal4.0";
    if (major >= 15) return "metal3.2";
    return "metal3.1";
}

/// xcrun metal -c each source to .air with `flags`, then link them into `name`.metallib.
fn metallib(b: *std.Build, name: []const u8, sources: []const []const u8, flags: []const []const u8) std.Build.LazyPath {
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const out = link.addOutputFileArg(b.fmt("{s}.metallib", .{name}));
    for (sources) |src| {
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal" });
        cc.addArgs(flags);
        cc.addArg("-c");
        cc.addFileArg(b.path(src));
        cc.addArg("-o");
        link.addFileArg(cc.addOutputFileArg(b.fmt("{s}.air", .{std.fs.path.stem(src)})));
    }
    return out;
}

/// The Kimi metallib: each source compiled with kimi_dir on the include path, the shared header a dependency.
fn kimiMetallib(b: *std.Build, flags: []const []const u8) std.Build.LazyPath {
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const out = link.addOutputFileArg("kimi.metallib");
    for (kimi_kernels) |name| {
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal" });
        cc.addArgs(flags);
        cc.addPrefixedDirectoryArg("-I", b.path(kimi_dir));
        cc.addFileInput(b.path(kimi_dir ++ "/kimi_common.h"));
        cc.addArg("-c");
        cc.addFileArg(b.path(b.fmt("{s}/{s}", .{ kimi_dir, name })));
        cc.addArg("-o");
        link.addFileArg(cc.addOutputFileArg(b.fmt("kimi_{s}.air", .{std.fs.path.stem(name)})));
    }
    return out;
}

/// A module whose `bytes` is the metallib at `lib`, embedded at compile time.
fn embedded(b: *std.Build, lib: std.Build.LazyPath, file: []const u8) *std.Build.Module {
    const wf = b.addWriteFiles();
    _ = wf.addCopyFile(lib, file);
    const src = wf.add("metallib.zig", b.fmt("pub const bytes = @embedFile(\"{s}\");\n", .{file}));
    return b.createModule(.{ .root_source_file = src });
}

/// The page pool's own tests, and its checks under engine_api's paged store and lane host (kv_pool a module of its own).
fn poolChecks(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const pool = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv_pool.zig"), .target = target, .optimize = .Debug });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const checks = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv_pool_checks.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "kv_pool", .module = pool }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes } } });
    const step = b.step("test-kv-pool", "The KV page pool: layout, leases, injected failures, the paged store and lane-host admission");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = pool })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = checks })).step);
    test_step.dependOn(step);
}

pub fn build(b: *std.Build) void {
    @import("zig/build/dsv41_shared.zig").steps(b);
    @import("zig/build/dsv41_cuda.zig").steps(b);
    @import("zig/build/dsv41_state.zig").steps(b);
    @import("zig/build/dsv41_draft.zig").steps(b);
    const target = b.standardTargetOptions(.{});
    @import("zig/build/copy_source.zig").steps(b);
    const release_version = b.option([]const u8, "version", "Release version, exactly MAJOR.MINOR.PATCH");
    if (release_version) |v| if (!dist_build.validVersion(v)) @panic("-Dversion must be exactly MAJOR.MINOR.PATCH");
    const build_options = b.addOptions();
    // Keep the existing manifest version for developer builds; releases require an explicit override.
    const manifest = @import("build.zig.zon");
    build_options.addOption([]const u8, "version", release_version orelse manifest.version);
    const slide_test_hooks = b.option(bool, "slide-test-hooks", "Living Weights test hooks, such as TF_SLIDE_FORGET_FAIL (never in a release)") orelse false;
    if (slide_test_hooks and release_version != null) @panic("-Dslide-test-hooks is for test builds: a -Dversion release never carries it");
    build_options.addOption(bool, "slide_test_hooks", slide_test_hooks);
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "debug, safe, fast or small (default fast)") orelse .fast;
    // Nemotron's draft vocabulary (zig/src/families/nemotron/draft_ids.txt), for every backend
    const ids_files = b.addWriteFiles();
    _ = ids_files.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
    const draft_ids = b.createModule(.{ .root_source_file = ids_files.add("draft_ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");\n") });
    const test_step = b.step("test", "Unit tests (macOS includes available Metal device checks)");
    if (b.graph.host.result.os.tag == .macos) {
        const step5_mods = metalEngineModules(b, b.graph.host, .Debug, draft_ids);
        const empty = b.addWriteFiles().add("step5-compile-only.metallib", "");
        step5_mods.sources.addImport("nemotron_packed_metallib", embedded(b, empty, "nemotron_packed.metallib"));
        @import("zig/build/step5_host.zig").steps(b, step5_mods.metal, step5_mods.engine, step5_mods.lanes);
    }
    @import("zig/build/h3_host.zig").tests(b, target, test_step);
    @import("zig/build/digest.zig").steps(b, test_step, draft_ids, build_options);
    @import("zig/build/flashnext_budget.zig").steps(b, test_step);
    @import("zig/build/flashnext_batch_host.zig").tests(b, test_step);
    if (b.graph.host.result.os.tag == .macos) {
        const budget_mods = metalEngineModules(b, b.graph.host, .Debug, draft_ids);
        const budget_empty = b.addWriteFiles().add("budget-compile-only.metallib", "");
        budget_mods.sources.addImport("nemotron_packed_metallib", embedded(b, budget_empty, "nemotron_packed.metallib"));
        @import("zig/build/flashnext_budget.zig").compileHost(b, budget_mods.metal, budget_mods.engine, budget_mods.lanes);
    }
    // every tracked path checks out on Windows too
    const names_run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/portable_names.zig"), .target = target }) }));
    names_run.setCwd(b.path("."));
    test_step.dependOn(&names_run.step);
    // the XPU server's stream admission, a pure function of the card's memory
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/native/xpu_admit.zig"), .target = target }) })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/safetensors.zig"), .target = target }) })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/limits.zig"), .target = target }) })).step);
    const cost_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/cost_rule_cases.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    }) });
    const cost_run = b.addRunArtifact(cost_cases);
    b.step("test-cost-rule", "CPU scripted window and interleaved head timing").dependOn(&cost_run.step);
    test_step.dependOn(&cost_run.step);
    merge_build.targets(b, target, optimize); // tensorfold merge's tests and tf-merge-check, host only on every OS
    switch (target.result.os.tag) {
        .macos => metalTargets(b, target, optimize, draft_ids, build_options, test_step),
        .linux, .windows => cuda_build.targets(b, target, optimize, draft_ids, build_options),
        else => {},
    }
    const gemma4_host_sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig"), .target = b.graph.host });
    gemma4_host_sources.addAnonymousImport("nemotron_packed_metallib", .{ .root_source_file = b.addWriteFiles().add("gemma4-host-empty.metallib", "") });
    const gemma4_contract_host = @import("zig/build/gemma4_host.zig").tests(b, gemma4_host_sources);
    if (b.graph.host.result.os.tag == .macos) {
        const gemma4_host_mods = metalEngineModules(b, b.graph.host, .Debug, draft_ids);
        gemma4_host_mods.sources.addImport("nemotron_packed_metallib", embedded(b, b.addWriteFiles().add("gemma4-compile-empty.metallib", ""), "nemotron_packed.metallib"));
        @import("zig/build/gemma4_host.zig").compileHost(b, gemma4_host_mods.metal, gemma4_host_mods.engine, gemma4_host_mods.lanes, gemma4_host_mods.sources);
        @import("zig/build/qwen27_host.zig").compileHost(b, gemma4_host_mods.metal, gemma4_host_mods.engine, gemma4_host_mods.lanes);
    }
    @import("zig/build/qwen27_host.zig").tests(b);
    const xgrammar = grammar_build.option(b);
    dist_build.targets(b, draft_ids, build_options, release_version, xgrammar);
    @import("zig/build/sessions.zig").add(b, test_step);
    @import("zig/build/tp.zig").add(b, target, optimize, test_step);
    @import("zig/build/gradient_host.zig").add(b);
    @import("zig/build/living_tools_host.zig").steps(b);
    cuda_build.hostTests(b, draft_ids, build_options, test_step, target.result.os.tag != .macos);
    xpu_build.hostTests(b, test_step);
    if (target.result.os.tag == .linux or target.result.os.tag == .windows) {
        // the engine seam's tests (lane_host.zig and its siblings), which the macOS branch runs from its own api module
        const lanes_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
        const api_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes_mod }} });
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api_mod })).step);
    }
    poolChecks(b, b.graph.host, test_step);
    hip_build.steps(b, target, optimize, build_options, test_step);
    exlr8_build.hostTests(b, target, test_step);
    xpu_build.steps(b, target, optimize, build_options);
    video_build.steps(b, target);
    const lanes_cpu = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const disk_cpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/glm_disk_tests.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes_cpu }},
    }) });
    const disk_run = b.addRunArtifact(disk_cpu);
    test_step.dependOn(&disk_run.step);
    b.step("test-learn-recovery", "CPU learned-state recovery and admission tests").dependOn(&disk_run.step);
    @import("zig/build/qwen27_required.zig").tests(b, lanes_cpu);
    glm_expiry_build.steps(b, test_step, lanes_cpu);
    const warning = b.addExecutable(.{ .name = "learned-warning-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/learned_warning_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes_cpu }},
    }) });
    const warning_run = b.addRunArtifact(warning);
    test_step.dependOn(&warning_run.step);
    b.step("test-learn-warning", "CPU production-log refusal, backoff and recovery test").dependOn(&warning_run.step);
    grammar_build.steps(b, target, xgrammar);
    contract_build.hostTests(b, gemma4_contract_host);
    sys_build.wire(b);
    @import("zig/build/checkpoint_config.zig").wire(b);
    @import("zig/build/save_guard.zig").wire(b);
}

/// `zig build native -Dcpu=apple_m1`: tensorfold-native with the Metal engines for the Python package's bundle (a native M5 build traps on M1-M4).
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module, living: *std.Build.Module, api: *std.Build.Module, engines: *std.Build.Module, build_options: *std.Build.Step.Options, test_step: ?*std.Build.Step) *std.Build.Step.Compile {
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize`
    const tokenizer = engines.import_table.get("tokenizer") orelse b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engines }, .{ .name = "living", .module = living }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{ .{ .name = "native_engines", .module = engines }, exlr8_build.cliImport(b, target, .ReleaseSafe, engine), .{ .name = "core", .module = engine } } }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    _ = server_build.addTo(b, exe.root_module, target, .ReleaseSafe);
    if (test_step == null) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the Metal engines into zig-out/native/bin (bundles: -Dcpu=apple_m1)").dependOn(&install.step);
    // libSystem's memcpy, not compiler-rt's: a quad-float or 128-bit helper links compiler-rt, whose weak memcpy then
    // wins
    const libc_mem = b.addSystemCommand(&.{ "sh", "-c", "nm -m \"$0\" | grep -q 'external _memcpy (from libSystem)' || { echo 'tensorfold-native links its own memcpy: find the typed std.json int parse or 128-bit float conversion that pulled in compiler-rt'; exit 1; }" });
    libc_mem.addArtifactArg(exe);
    test_step.?.dependOn(&libc_mem.step);
    const server_tests = b.createModule(.{
        .root_source_file = b.path("zig/src/server/root.zig"),
        .target = target,
        .optimize = .Debug, // the server's unit tests run with safety checks, as zig_test.sh builds them
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "living", .module = living } },
    });
    // the required-tools tests' 27B mask policy and lane fake, on the engine API's own lanes module
    if (api.import_table.get("lanes")) |lanes_mod| {
        const q27 = @import("zig/build/qwen27_required.zig").modules(b, lanes_mod);
        server_tests.addImport("qwen27_mask_fake", q27.fake);
        server_tests.addImport("qwen27_masks", q27.masks);
    }
    const server_json = server_build.addTo(b, server_tests, target, .Debug);
    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests })).step);
    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_json })).step);
    @import("zig/build/cortheon_fuzz.zig").add(b, target, api, tokenizer, template);
    // the parse half's tests, as their own test binary (zig/src/server/tool_parse_test.zig)
    const tool_parse_tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } } }) });
    _ = server_build.addTo(b, tool_parse_tests.root_module, target, .Debug);
    test_step.?.dependOn(&b.addRunArtifact(tool_parse_tests).step);
    // the chat-template engine's tests (zig/src/core/template/tests.zig), which no other test binary reaches
    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/tests.zig"), .target = target, .optimize = .Debug, .link_libc = true }) })).step);
    // the engine seam's own tests (lane_host.zig), as zig/tests/server/zig_test.sh runs them
    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    // the frozen server parity goldens, checked by Zig driving fake_serve
    golden_build.step(b, target, server_tests, api, tokenizer, template);
    b.step("test-native", "The Metal engines module's host-side tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engines })).step);
    const reuse = b.addExecutable(.{ .name = "tf-flashnext-reuse", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/flashnext_reuse.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "engine_api", .module = api } },
    }) });
    b.installArtifact(reuse);
    b.step("tf-flashnext-reuse", "Flash Next prompt reuse against fresh prompt passes: replies at every depth and kept states' bytes").dependOn(&b.addInstallArtifact(reuse, .{}).step);
    const batch = b.addExecutable(.{ .name = "tf-flashnext-batch", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/flashnext_batch.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "kernel_sources", .module = engine.import_table.get("kernel_sources").? } },
    }) });
    b.step("tf-flashnext-batch", "Flash Next shared rounds against fresh legacy replies with prompt admission and cancellation").dependOn(&b.addInstallArtifact(batch, .{}).step);
    return exe;
}

/// macOS: the metallib, the Metal runtime's test programs and the engine over Metal.
fn metalTargets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, test_step: *std.Build.Step) void {
    const metal_std = b.option([]const u8, "metal-std", "Metal language version (default: MLX's for this macOS)") orelse mlxMetalStd(b);

    // MLX's mx.fast.metal_kernel JIT: MTLCompileOptions mathMode safe, fp32 functions fast (the default), its language
    // version.
    const mlx_flags = [_][]const u8{ b.fmt("-std={s}", .{metal_std}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=fast" };
    const lib = metallib(b, "tensorfold", &kernels, &mlx_flags);
    const lib_install = b.addInstallFile(lib, "lib/tensorfold.metallib");
    b.step("metallib", "Compile zig/kernels/metal into tensorfold.metallib").dependOn(&lib_install.step);

    // Control: the same Nemotron kernels under fast math, which the bit-exact check must be able to tell apart.
    const fast_flags = [_][]const u8{ b.fmt("-std={s}", .{metal_std}), "-fmetal-math-mode=fast", "-fmetal-math-fp32-functions=fast" };
    const fast_lib = metallib(b, "nemotron_fastmath", generated, &fast_flags);
    b.getInstallStep().dependOn(&b.addInstallFile(fast_lib, "lib/nemotron_fastmath.metallib").step);
    b.getInstallStep().dependOn(&lib_install.step);

    const mods = metalEngineModules(b, target, optimize, draft_ids);
    const metal = mods.metal;
    const kernels_mod = embedded(b, lib, "tensorfold.metallib");
    counters_build.targets(b, target, optimize, metal, mods.sources, test_step);

    // The engines registry: the families this platform's binaries serve; the CLI reads it for models/info/pull.
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    const engines = b.createModule(.{
        .root_source_file = b.path("zig/src/native/metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "engine_api", .module = api }, .{ .name = "tensorfold", .module = mods.engine } },
    });
    @import("zig/build/drama3_metal.zig").addNative(b, target, optimize, metal, engines, metal_std);
    @import("zig/build/drama3_metal.zig").targets(b, target, optimize, metal, metal_std, test_step);

    const programs = [_]struct { name: []const u8, path: []const u8, about: []const u8 }{
        .{ .name = "tf-metal-tests", .path = "zig/tests/gpu_tests.zig", .about = "GPU runtime tests" },
        .{ .name = "tf-dispatch-bench", .path = "zig/tests/dispatch_bench.zig", .about = "Dispatch boundary benchmark" },
        .{ .name = "tf-kernel-exact", .path = "zig/tests/kernel_exact.zig", .about = "Bit-exact check of a kernel against an MLX oracle case" },
    };
    for (programs) |p| {
        const mod = b.createModule(.{
            .root_source_file = b.path(p.path),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "metallib", .module = kernels_mod } },
        });
        const exe = b.addExecutable(.{ .name = p.name, .root_module = mod });
        b.installArtifact(exe);
        b.step(p.name, p.about).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    // Kimi K3: the family module, its metallib, and its GPU check and benchmark programs.
    const kimi_lib = kimiMetallib(b, &mlx_flags);
    b.getInstallStep().dependOn(&b.addInstallFile(kimi_lib, "lib/kimi.metallib").step);
    const kimi = b.createModule(.{
        .root_source_file = b.path("zig/src/families/kimi_k3/kimi_k3.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "metal", .module = metal }},
    });
    const kimi_bytes = embedded(b, kimi_lib, "kimi.metallib");
    const kimi_programs = [_]struct { name: []const u8, path: []const u8, about: []const u8 }{
        .{ .name = "tf-k3-check", .path = "zig/tests/kimi_k3/check.zig", .about = "Kimi K3 kernels and layers on synthetic weights at real shapes" },
        .{ .name = "tf-k3-bench", .path = "zig/tests/kimi_k3/bench.zig", .about = "Kimi K3 kernel bandwidth" },
    };
    for (kimi_programs) |p| {
        const mod = b.createModule(.{
            .root_source_file = b.path(p.path),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kimi_k3", .module = kimi }, .{ .name = "kimi_metallib", .module = kimi_bytes } },
        });
        const exe = b.addExecutable(.{ .name = p.name, .root_module = mod });
        b.installArtifact(exe);
        b.step(p.name, p.about).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    // Kimi's tiktoken tokenizer and its parity program (tools/zig/k3_tokenizer_parity.py).
    const tiktoken = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tiktoken.zig"), .target = target, .optimize = optimize });
    const tok_mod = b.createModule(.{
        .root_source_file = b.path("zig/tests/kimi_k3/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "tiktoken", .module = tiktoken }},
    });
    const tok_exe = b.addExecutable(.{ .name = "tf-k3-tokenizer", .root_module = tok_mod });
    b.installArtifact(tok_exe);
    b.step("tf-k3-tokenizer", "Kimi K3's tiktoken tokenizer over parity cases").dependOn(&b.addInstallArtifact(tok_exe, .{}).step);

    // The cluster layer: inventory, membership, planner, loading and converged rounds over the lane core.
    const lanes = mods.lanes;
    const fabric = mods.fabric;
    const cluster = b.createModule(.{
        .root_source_file = b.path("zig/src/cluster/cluster.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric } },
    });
    // The engine: kernel sources compiled at run time (MLX's options), the families, the lane core.
    const sources = mods.sources;
    // Nemotron's packed kernels, compiled offline and embedded: the runtime fallback when this macOS's
    // compiler refuses uint4b_format (#451).
    const packed_lib = metallib(b, "nemotron_packed", &nemotron_packed, &mlx_flags);
    b.getInstallStep().dependOn(&b.addInstallFile(packed_lib, "lib/nemotron_packed.metallib").step);
    sources.addImport("nemotron_packed_metallib", embedded(b, packed_lib, "nemotron_packed.metallib"));
    const engine = mods.engine;
    // FlashNext's CPU contracts, with the family module's imports so the replay graph type-checks
    const flash_host = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/flashnext_host.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "nemotron_draft_ids", .module = draft_ids }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric } },
    }) });
    b.step("compile-flashnext-host", "Compile FlashNext CPU metadata contracts without running them").dependOn(&flash_host.step);
    const flash_host_run = b.addRunArtifact(flash_host);
    b.step("test-flashnext-host", "Run FlashNext CPU metadata contracts without Metal").dependOn(&flash_host_run.step);
    test_step.dependOn(&flash_host_run.step);
    // Pack builder cases that do not need the replay graph.
    const pack_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/flashnext_pack_cases.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    test_step.dependOn(&b.addRunArtifact(pack_cases).step);
    const load_admission_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/flashnext_load_admission_cases.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    const load_admission_run = b.addRunArtifact(load_admission_cases);
    b.step("test-flashnext-load-admission", "CPU-only Flash Next runtime quantization admission").dependOn(&load_admission_run.step);
    test_step.dependOn(&load_admission_run.step);
    const pack_cache_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/flashnext_pack_cache_cases.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    const pack_cache_run = b.addRunArtifact(pack_cache_cases);
    b.step("test-flashnext-pack-cache", "CPU-only atomic Flash Next pack cache lifecycle").dependOn(&pack_cache_run.step);
    test_step.dependOn(&pack_cache_run.step);
    const engine_programs = [_]struct { name: []const u8, path: []const u8, about: []const u8, c_source: ?[]const u8 = null }{
        .{ .name = "tf-qwen35-check", .path = "zig/tests/qwen35_check.zig", .about = "Qwen3.5-2B checkpoint and native operations" },
        .{ .name = "tf-qwen35-forward", .path = "zig/tests/qwen35_forward.zig", .about = "Qwen3.5-2B native teacher-forced logits" },
        .{ .name = "tf-qwen35-exact", .path = "zig/tests/qwen35_exact.zig", .about = "Qwen3.5-2B forward and committed-state exactness" },
        .{ .name = "tf-lfm2-forward", .path = "zig/tests/lfm2_forward.zig", .about = "LFM2-MoE native teacher-forced logits" },
        .{ .name = "tf-lfm2-bench", .path = "zig/tests/lfm2_bench.zig", .about = "LFM2-MoE prompt-chunk and one-row decode timing" },
        .{ .name = "tf-lfm2-exact", .path = "zig/tests/lfm2_exact.zig", .about = "LFM2-MoE prompt chunks, windows and committed-state exactness" },
        .{ .name = "tf-lfm2-fp64", .path = "zig/tests/lfm2_fp64.zig", .about = "LFM2 projection classes and attention on the tensor units, the tiles and the twin, each against float64" },
        .{ .name = "tensorfold", .path = "zig/src/main.zig", .about = "The native engine's command line" },
        .{ .name = "tf-nemotron-fixtures", .path = "zig/tests/nemotron_fixtures.zig", .about = "Nemotron kernels against the Python engine's captured ops" },
        .{ .name = "tf-nemotron-bench", .path = "zig/tests/nemotron_bench.zig", .about = "One-row projection kernels timed by tile count" },
        .{ .name = "tf-nemotron-wide", .path = "zig/tests/nemotron_wide.zig", .about = "Window costs by kernel class and expert overlap of drafted, sibling and stream rows" },
        .{ .name = "tf-nemotron-prefill-check", .path = "zig/tests/nemotron_prefill_check.zig", .about = "A prompt chunk layer by layer against prefill_dump.py's rows" },
        .{ .name = "tf-nemotron-dense", .path = "zig/tests/nemotron_dense.zig", .about = "Dense projection schedules bit-checked against the engine's kernels, then timed" },
        .{ .name = "tf-nemotron-experts", .path = "zig/tests/nemotron_experts.zig", .about = "Routed-expert pass shapes bit-checked against the engine's kernels, then timed by window width" },
        .{ .name = "tf-nemotron-sample-check", .path = "zig/tests/nemotron_sample_check.zig", .about = "tf_sample_full against its host reference on synthetic rows, timed beside tf_gpu_sample" },
        .{ .name = "tf-train-check", .path = "zig/tests/train_check.zig", .about = "The learning step's gradients against the engine's own forward by finite differences, on the real model" },
        .{ .name = "tf-train-pack", .path = "zig/tests/train_pack.zig", .about = "Training passes holding several sequences against one a pass, on the real model: differences and time" },
        .{ .name = "tf-pocket-fit-check", .path = "zig/tests/pocket_fit_check.zig", .about = "A dumped pocket lesson's read rows fitted as the learner fits them, timed, each fit's numbers and weights written" },
        .{ .name = "tf-flashnext-run", .path = "zig/tests/flashnext_run.zig", .about = "Flash Next one-row greedy steps on the Python engine's recorded kernels, against its tokens" },
        .{ .name = "tf-flashnext-pack", .path = "zig/tests/flashnext_pack.zig", .about = "Flash Next's three weight packs built from a checkpoint, byte-compared with a Python dump" },
        .{ .name = "tf-flashnext-think", .path = "zig/tests/flashnext_think.zig", .about = "Flash Next's verdict kernels under thinking budgets on a fake target, against the lane core's close" },
        .{ .name = "tf-grid-sync-bench", .path = "zig/tests/grid_sync_bench.zig", .about = "A GPU-wide barrier in one persistent dispatch against dependent relaunches" },
        .{ .name = "tf-weight-read-check", .path = "zig/tests/weight_read_check.zig", .about = "Check native file reads and failed-read cleanup", .c_source = "zig/tests/pread_fault.c" },
        .{ .name = "tf-glm-run", .path = "zig/tests/glm_run.zig", .about = "GLM-5.3-Flash greedy replies at each draft depth against depth 0 and reference tokens" },
        .{ .name = "tf-glm-vision", .path = "zig/tests/glm_vision.zig", .about = "GLM-5.3-Flash's image tower on an image or a reference dump's patches, compared with the dump" },
        .{ .name = "tf-glm-video", .path = "zig/tests/glm_video.zig", .about = "GLM-5.3-Flash's video preparation on a clip (libtfvideo), written for tools/zig/glm_video_fixtures.py" },
        .{ .name = "tf-qwen-vision", .path = "zig/tests/qwen_vision.zig", .about = "The Qwen-VL tower on picture files: patches, grids and merged rows written for a float32 reference" },
        .{ .name = "tf-qwen-image", .path = "zig/tests/qwen_image.zig", .about = "Qwen-Image-2.1's transformer on a reference case: each step compared, then the schedule timed" },
        .{ .name = "tf-glm-attn-probe", .path = "zig/tests/glm_attn_probe.zig", .about = "GLM-5.3-Flash's dense latent attention kernels on given inputs (scores, probabilities, outputs)" },
        .{ .name = "tf-moe-bench", .path = "zig/tests/moe_bench.zig", .about = "Routed-expert prompt kernels from .metal files on a skewed routing: chained speed, bits against the first" },
        .{ .name = "tf-mm-bench", .path = "zig/tests/mm_bench.zig", .about = "Prompt matmul kernels from .metal files: chained speed at one shape, bits against the first, error against fp64" },
        .{ .name = "tf-gemma4-kernels", .path = "zig/tests/gemma4/kernels.zig", .about = "Gemma 4 dense decode kernels against the reference's dumped fixtures, bit for bit by window width" },
        .{ .name = "tf-gemma4-glue", .path = "zig/tests/gemma4/glue.zig", .about = "Gemma 4 dense glue kernels against the reference's dumped fixtures: embed, norm, q/k/v, attention, tails, softcap" },
        .{ .name = "tf-gemma4-run", .path = "zig/tests/gemma4/run.zig", .about = "Gemma 4 dense greedy decode, tokens and top-two margins for the oracle's check" },
        .{ .name = "tf-gemma4-lanes", .path = "zig/tests/gemma4/lanes.zig", .about = "Gemma 4 dense on the lane core: plain, copy-drafted and shared-round streams against the oracle's tokens" },
        .{ .name = "tf-gemma4-vision", .path = "zig/tests/gemma4/vision.zig", .about = "Gemma 4's image tower against gemma4_vision_fixtures.py's dumps: patches bit for bit, each stage's error beside torch bf16's" },
        .{ .name = "tf-gemma4-bench", .path = "zig/tests/gemma4/bench.zig", .about = "Gemma 4 dense step costs: projections alone by rows against whole forwards" },
        .{ .name = "tf-qwen36-row-seam", .path = "zig/tests/qwen36_row_seam.zig", .about = "Model-free fixture: the row projection's bias sums stay fp32 at every width" },
    };
    for (engine_programs) |p| {
        const mod = b.createModule(.{
            .root_source_file = b.path(p.path),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "core", .module = engine }, .{ .name = "cluster", .module = cluster }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "native_engines", .module = engines }, .{ .name = "engine_api", .module = api }, exlr8_build.cliImport(b, target, optimize, engine) },
        });
        if (p.c_source) |file| {
            mod.addCSourceFile(.{ .file = b.path(file), .flags = &.{"-std=c11"} });
        }
        const exe = b.addExecutable(.{ .name = p.name, .root_module = mod });
        b.installArtifact(exe);
        b.step(p.name, p.about).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

    glm_checks.sampler(b, target, optimize, metal, lanes, sources);

    // MiniMax H3 / FastH3's transformer: the packed rows' denoising forwards against a case file (M5 tensor units).
    const h3_kernels = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/h3/kernels.zig") });
    const h3_safetensors = b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors.zig"), .target = target, .optimize = optimize });
    const h3_mod = b.createModule(.{ .root_source_file = b.path("zig/src/families/h3/h3.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "safetensors", .module = h3_safetensors }, .{ .name = "h3_kernels", .module = h3_kernels }, .{ .name = "kernel_sources", .module = sources } } });
    const h3_dit = b.addExecutable(.{ .name = "tf-h3-dit", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/h3_dit.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "h3", .module = h3_mod } },
    }) });
    b.step("tf-h3-dit", "FastH3's Metal transformer against a case file: velocities, every step, timings").dependOn(&b.addInstallArtifact(h3_dit, .{}).step);
    // The core row projection (chips without tensor units) on synthetic matrices; runs on any Mac's GPU
    const row_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/row_projection.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const row_check = b.addExecutable(.{ .name = "tf-row-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/row_projection_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "row_projection", .module = row_mod } },
    }) });
    b.step("tf-row-check", "The core row projection on synthetic 4-bit matrices: widths bit-equal, a CPU reference, indexed experts").dependOn(&b.addInstallArtifact(row_check, .{}).step);
    core_checks.rowWidths(b, target, optimize, metal, row_mod);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = row_mod })).step);
    core_checks.tileQmm(b, target, optimize, metal, sources, test_step);
    exl3_build.targets(b, target, optimize, metal, test_step);
    exlr8_build.metalTargets(b, target, optimize, metal, mods.sources, test_step);
    exlr8_build.glmTargets(b, target, optimize, metal, mods.engine);

    // The checkpoint CLI: models, info and pull, against a synthetic cache and a fake hub, at ReleaseSafe as it ships.
    const cli_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/tests.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "native_engines", .module = engines }, exlr8_build.cliImport(b, target, optimize, mods.engine), .{ .name = "core", .module = mods.engine } },
    }) });
    const run_cli_tests = b.addRunArtifact(cli_tests);
    test_step.dependOn(&run_cli_tests.step);
    b.step("test-cli", "The checkpoint CLI's host tests").dependOn(&run_cli_tests.step);

    // The core lane, shared-context and Qwen3.8-27B programs and host tests.
    qwen27_build.targets(b, target, optimize, metal, sources, engine, lanes, mods.qwen27, test_step);
    core_checks.targets(b, target, optimize, metal, engine);
    contract_build.metal(b, target, optimize, metal, engine, sources);
    glm_checks.targets(b, target, optimize, metal, engine, sources);
    flashnext_checks.targets(b, target, optimize, metal, engine);

    _ = nativeServer(b, target, optimize, metal, engine, mods.living, api, engines, build_options, test_step);

    const cluster_metal = b.createModule(.{
        .root_source_file = b.path("zig/src/cluster/metal_sink.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "cluster", .module = cluster } },
    });
    const cluster_exe = b.addExecutable(.{ .name = "tf-cluster", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/cluster/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "cluster", .module = cluster }},
    }) });
    b.installArtifact(cluster_exe);
    b.step("tf-cluster", "tensorfold cluster / serve --cluster / node").dependOn(&b.addInstallArtifact(cluster_exe, .{}).step);
    // Kimi K3's checkpoint validator: the cluster's header reader cross-checked with the family's own specs.
    const k3_validate = b.addExecutable(.{ .name = "tf-k3-validate", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/kimi_k3/validate.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "kimi_k3", .module = kimi }, .{ .name = "cluster", .module = cluster } },
    }) });
    b.installArtifact(k3_validate);
    b.step("tf-k3-validate", "Kimi K3's checkpoint: index, headers, coverage and every tensor against the family").dependOn(&b.addInstallArtifact(k3_validate, .{}).step);
    const k3_cluster = b.addExecutable(.{ .name = "tf-k3-cluster", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/kimi_k3/cluster.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kimi_k3", .module = kimi }, .{ .name = "kimi_metallib", .module = kimi_bytes }, .{ .name = "cluster", .module = cluster } },
    }) });
    b.installArtifact(k3_cluster);
    b.step("tf-k3-cluster", "Kimi K3 tensor- and expert-parallel ranks against one node, through the cluster's canon").dependOn(&b.addInstallArtifact(k3_cluster, .{}).step);
    const tp2 = b.addExecutable(.{ .name = "tf-tp2-bench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/tp2_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "fabric", .module = fabric } },
    }) });
    b.step("tf-tp2-bench", "TP=2's exchange as the GPU sees it, between two Macs over MCDMA").dependOn(&b.addInstallArtifact(tp2, .{}).step);
    const cluster_tests = b.step("test-cluster", "Cluster tests: fake nodes and fabric, K3 planning (TF_K3_DIR), Metal sink");
    cluster_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cluster })).step);
    cluster_tests.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cluster_metal })).step);

    // These module tests include Metal device checks when the macOS host has a GPU.
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = metal })).step);
    const helpers = b.createModule(.{
        .root_source_file = b.path("zig/tests/common.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "metallib", .module = kernels_mod } },
    });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = helpers })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = engine })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lanes })).step);
    // the fake fabric maps its windows with mmap: the Macs' RDMA, not Windows
    if (target.result.os.tag != .windows) test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = fabric })).step);
    const dashboard = b.createModule(.{ .root_source_file = b.path("zig/src/server/dashboard_test.zig"), .target = target, .optimize = optimize, .link_libc = true });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = dashboard })).step);
    const kimi_test = b.step("test-k3", "Kimi K3's host-side unit tests (no GPU work)");
    for ([_]*std.Build.Module{ kimi, tiktoken }) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
        kimi_test.dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
}

/// The server and distribution use the same embedded-source Metal engine.
fn metalEngineModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module) struct { metal: *std.Build.Module, lanes: *std.Build.Module, fabric: *std.Build.Module, sources: *std.Build.Module, engine: *std.Build.Module, living: *std.Build.Module, qwen27: qwen27_build.Shared } {
    const metal = b.createModule(.{
        .root_source_file = b.path("zig/src/metal/metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{}); // the Neural Engine's prompt share passes IOSurfaces
    metal.linkSystemLibrary("objc", .{});
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const fabric = b.createModule(.{ .root_source_file = b.path("zig/src/fabric/fabric.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    // Living Weights' family-neutral learner, which Nemotron's lessons and the server's view of them share
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const engine = b.createModule(.{
        .root_source_file = b.path("zig/src/tensorfold.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "nemotron_draft_ids", .module = draft_ids }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric }, .{ .name = "living", .module = living } },
    });
    return .{ .metal = metal, .lanes = lanes, .fabric = fabric, .sources = sources, .engine = engine, .living = living, .qwen27 = qwen27_build.engineModules(b, target, metal, sources, engine) };
}

pub fn distMetalServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, sdk: []const u8) *std.Build.Step.Compile {
    const m = metalEngineModules(b, target, .fast, draft_ids);
    // the release binary embeds the same prebuilt packed-kernel metallib as the development build
    const flags = [_][]const u8{ b.fmt("-std={s}", .{mlxMetalStd(b)}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=fast" };
    m.sources.addImport("nemotron_packed_metallib", embedded(b, metallib(b, "nemotron_packed", &nemotron_packed, &flags), "nemotron_packed.metallib"));
    // An explicit deployment target is a cross-build even on macOS. Supply the SDK search paths.
    m.metal.addSystemFrameworkPath(b.graph.cwdRelativePath(b.pathJoin(&.{ sdk, "System/Library/Frameworks" })));
    m.metal.addLibraryPath(b.graph.cwdRelativePath(b.pathJoin(&.{ sdk, "usr/lib" })));
    m.metal.addSystemIncludePath(b.graph.cwdRelativePath(b.pathJoin(&.{ sdk, "usr/include" })));
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .fast, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = m.lanes }} });
    const engines = b.createModule(.{ .root_source_file = b.path("zig/src/native/metal.zig"), .target = target, .optimize = .fast, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = m.metal }, .{ .name = "engine_api", .module = api }, .{ .name = "tensorfold", .module = m.engine } } });
    @import("zig/build/drama3_metal.zig").addNative(b, target, .fast, m.metal, engines, mlxMetalStd(b));
    return nativeServer(b, target, .fast, m.metal, m.engine, m.living, api, engines, build_options, null);
}

test {
    _ = dist_build;
}
