const std = @import("std");
const builtin = @import("builtin");
const cuda_build = @import("zig/build/cuda.zig");
const dist_build = @import("zig/build/dist.zig");
const qwen27_build = @import("zig/build/qwen27.zig");
const hip_build = @import("zig/build/hip.zig");

comptime {
    const required = std.mem.trim(u8, @embedFile(".zig-version"), "\r\n");
    if (!std.mem.eql(u8, builtin.zig_version_string, required))
        @compileError("TensorFold's Zig engine requires Zig " ++ required);
}

/// Our Metal sources, compiled into one metallib that executables embed.
const kernels = [_][]const u8{
    "zig/kernels/metal/runtime_tests.metal",
    "zig/kernels/metal/nemotron_expert_up.metal",
    "zig/kernels/metal/nemotron_expert_down.metal",
};

/// Kernels generated from our Python sources (tools/zig/gen_nemotron_metal.py), also built under fast math as a control.
const generated = kernels[1..];

/// Kimi K3's kernels: one metallib, sharing kimi_common.h.
const kimi_dir = "zig/kernels/metal/kimi";
const kimi_kernels = [_][]const u8{ "dense.metal", "dense_mma.metal", "norms.metal", "experts.metal", "kda.metal", "mla.metal", "synth.metal" };

/// Nemotron's packed kernels (generated, they use uint4b_format): one metallib built ahead of time for the
/// runtime compilers that refuse the type (macOS 26.3, #451). Keep in step with tools/zig/gen_nemotron_kernels.py.
const nemotron_packed = [_][]const u8{
    "zig/kernels/metal/nemotron/coop_down_1_0.metal",
    "zig/kernels/metal/nemotron/coop_down_2_0.metal",
    "zig/kernels/metal/nemotron/coop_down_2_1.metal",
    "zig/kernels/metal/nemotron/coop_draft_1_0.metal",
    "zig/kernels/metal/nemotron/coop_draft_2_0.metal",
    "zig/kernels/metal/nemotron/coop_draft_2_1.metal",
    "zig/kernels/metal/nemotron/coop_eh_1_0.metal",
    "zig/kernels/metal/nemotron/coop_eh_1_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_eh_2_0.metal",
    "zig/kernels/metal/nemotron/coop_eh_2_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_eh_2_1.metal",
    "zig/kernels/metal/nemotron/coop_eh_2_1_sk.metal",
    "zig/kernels/metal/nemotron/coop_head_1_0.metal",
    "zig/kernels/metal/nemotron/coop_head_2_0.metal",
    "zig/kernels/metal/nemotron/coop_head_2_1.metal",
    "zig/kernels/metal/nemotron/coop_in_1_0.metal",
    "zig/kernels/metal/nemotron/coop_in_1_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_in_2_0.metal",
    "zig/kernels/metal/nemotron/coop_in_2_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_in_2_1.metal",
    "zig/kernels/metal/nemotron/coop_in_2_1_sk.metal",
    "zig/kernels/metal/nemotron/coop_out_1_0.metal",
    "zig/kernels/metal/nemotron/coop_out_1_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_out_2_0.metal",
    "zig/kernels/metal/nemotron/coop_out_2_0_sk.metal",
    "zig/kernels/metal/nemotron/coop_out_2_1.metal",
    "zig/kernels/metal/nemotron/coop_out_2_1_sk.metal",
    "zig/kernels/metal/nemotron/coop_qkv_1_0.metal",
    "zig/kernels/metal/nemotron/coop_qkv_2_0.metal",
    "zig/kernels/metal/nemotron/coop_qkv_2_1.metal",
    "zig/kernels/metal/nemotron/coop_up_1_0.metal",
    "zig/kernels/metal/nemotron/coop_up_2_0.metal",
    "zig/kernels/metal/nemotron/coop_up_2_1.metal",
    "zig/kernels/metal/nemotron/up_relu2_1_0.metal",
    "zig/kernels/metal/nemotron/up_relu2_2_0.metal",
    "zig/kernels/metal/nemotron/up_relu2_2_1.metal",
};

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

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const release_version = b.option([]const u8, "version", "Release version, exactly MAJOR.MINOR.PATCH");
    if (release_version) |v| if (!dist_build.validVersion(v)) @panic("-Dversion must be exactly MAJOR.MINOR.PATCH");
    const build_options = b.addOptions();
    // Keep the existing manifest version for developer builds; releases require an explicit override.
    const manifest = @import("build.zig.zon");
    build_options.addOption([]const u8, "version", release_version orelse manifest.version);
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "debug, safe, fast or small (default fast)") orelse .fast;
    // Nemotron's draft vocabulary (zig/src/families/nemotron/draft_ids.txt), for every backend
    const ids_files = b.addWriteFiles();
    _ = ids_files.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
    const draft_ids = b.createModule(.{ .root_source_file = ids_files.add("draft_ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");\n") });
    const test_step = b.step("test", "Host-side unit tests (no GPU work)");
    // every tracked path checks out on Windows too
    const names_run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/portable_names.zig"), .target = target }) }));
    names_run.setCwd(b.path("."));
    test_step.dependOn(&names_run.step);
    const cost_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/cost_rule_cases.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    }) });
    const cost_run = b.addRunArtifact(cost_cases);
    b.step("test-cost-rule", "CPU scripted window and interleaved head timing").dependOn(&cost_run.step);
    test_step.dependOn(&cost_run.step);
    switch (target.result.os.tag) {
        .macos => metalTargets(b, target, optimize, draft_ids, build_options, test_step),
        .linux => cuda_build.targets(b, target, optimize, draft_ids, build_options),
        else => {},
    }
    dist_build.targets(b, draft_ids, build_options, release_version);
    cuda_build.hostTests(b, draft_ids, build_options, test_step);
    hip_build.steps(b, target, test_step);
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
}

/// `zig build native -Dcpu=apple_m1`: tensorfold-native with the Metal engines for the Python package's bundle (a native M5 build traps on M1-M4).
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, engine: *std.Build.Module, api: *std.Build.Module, engines: *std.Build.Module, build_options: *std.Build.Step.Options, test_step: ?*std.Build.Step) *std.Build.Step.Compile {
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize`
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = engines }} }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    if (test_step == null) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the Metal engines into zig-out/native/bin (bundles: -Dcpu=apple_m1)").dependOn(&install.step);
    // libSystem's memcpy, not compiler-rt's: a quad-float or 128-bit helper links compiler-rt, whose weak memcpy then wins
    const libc_mem = b.addSystemCommand(&.{ "sh", "-c", "nm -m \"$0\" | grep -q 'external _memcpy (from libSystem)' || { echo 'tensorfold-native links its own memcpy: find the typed std.json int parse or 128-bit float conversion that pulled in compiler-rt'; exit 1; }" });
    libc_mem.addArtifactArg(exe);
    test_step.?.dependOn(&libc_mem.step);
    const server_tests = b.createModule(.{
        .root_source_file = b.path("zig/src/server/root.zig"),
        .target = target,
        .optimize = .Debug, // the server's unit tests run with safety checks, as zig_test.sh builds them
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } },
    });
    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server_tests })).step);
    // the parse half's tests, as their own test binary (zig/src/server/tool_parse_test.zig)
    const tool_parse_tests = b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } } }) });
    test_step.?.dependOn(&b.addRunArtifact(tool_parse_tests).step);
    // the engine seam's own tests (lane_host.zig), as zig/tests/server/zig_test.sh runs them
    test_step.?.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    // the frozen server parity goldens, checked by Zig driving fake_serve
    const fake_serve = b.addExecutable(.{ .name = "fake_serve", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/fake_serve.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "server", .module = server_tests }, .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } },
    }) });
    const golden_check = b.addExecutable(.{ .name = "golden_check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/golden_check.zig"),
        .target = target,
        .optimize = .Debug,
    }) });
    const golden_run = b.addRunArtifact(golden_check);
    golden_run.addArtifactArg(fake_serve);
    golden_run.addFileArg(b.path("zig/tests/server/golden/cases.json"));
    golden_run.addDirectoryArg(b.path("zig/tests/server/golden"));
    golden_run.addDirectoryArg(b.path("zig/tests/server/fixtures"));
    golden_run.setCwd(b.path("."));
    b.step("test-golden", "The frozen server parity goldens checked by a Zig step driving fake_serve").dependOn(&golden_run.step);
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
    return exe;
}

/// macOS: the metallib, the Metal runtime's test programs and the engine over Metal.
fn metalTargets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, test_step: *std.Build.Step) void {
    const metal_std = b.option([]const u8, "metal-std", "Metal language version (default: MLX's for this macOS)") orelse mlxMetalStd(b);

    // MLX's mx.fast.metal_kernel JIT: MTLCompileOptions mathMode safe, fp32 functions fast (the default), its language version.
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

    // The engines registry: the families this platform's binaries serve; the CLI reads it for models/info/pull.
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    const engines = b.createModule(.{
        .root_source_file = b.path("zig/src/native/metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "engine_api", .module = api }, .{ .name = "tensorfold", .module = mods.engine } },
    });

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
        .{ .name = "tensorfold", .path = "zig/src/main.zig", .about = "The native engine's command line" },
        .{ .name = "tf-nemotron-fixtures", .path = "zig/tests/nemotron_fixtures.zig", .about = "Nemotron kernels against the Python engine's captured ops" },
        .{ .name = "tf-nemotron-bench", .path = "zig/tests/nemotron_bench.zig", .about = "One-row projection kernels timed by tile count" },
        .{ .name = "tf-nemotron-wide", .path = "zig/tests/nemotron_wide.zig", .about = "Window costs by kernel class and expert overlap of drafted, sibling and stream rows" },
        .{ .name = "tf-nemotron-prefill-check", .path = "zig/tests/nemotron_prefill_check.zig", .about = "A prompt chunk layer by layer against prefill_dump.py's rows" },
        .{ .name = "tf-nemotron-dense", .path = "zig/tests/nemotron_dense.zig", .about = "Dense projection schedules bit-checked against the engine's kernels, then timed" },
        .{ .name = "tf-nemotron-experts", .path = "zig/tests/nemotron_experts.zig", .about = "Routed-expert pass shapes bit-checked against the engine's kernels, then timed by window width" },
        .{ .name = "tf-nemotron-sample-check", .path = "zig/tests/nemotron_sample_check.zig", .about = "tf_sample_full against its host reference on synthetic rows, timed beside tf_gpu_sample" },
        .{ .name = "tf-train-check", .path = "zig/tests/train_check.zig", .about = "The learning step's gradients against the engine's own forward by finite differences, on the real model" },
        .{ .name = "tf-flashnext-run", .path = "zig/tests/flashnext_run.zig", .about = "Flash Next one-row greedy steps on the Python engine's recorded kernels, against its tokens" },
        .{ .name = "tf-flashnext-pack", .path = "zig/tests/flashnext_pack.zig", .about = "Flash Next's three weight packs built from a checkpoint, byte-compared with a Python dump" },
        .{ .name = "tf-grid-sync-bench", .path = "zig/tests/grid_sync_bench.zig", .about = "A GPU-wide barrier in one persistent dispatch against dependent relaunches" },
        .{ .name = "tf-weight-read-check", .path = "zig/tests/weight_read_check.zig", .about = "Check native file reads and failed-read cleanup", .c_source = "zig/tests/pread_fault.c" },
        .{ .name = "tf-glm-run", .path = "zig/tests/glm_run.zig", .about = "GLM-5.3-Flash greedy replies at each draft depth against depth 0 and reference tokens" },
        .{ .name = "tf-glm-attn-probe", .path = "zig/tests/glm_attn_probe.zig", .about = "GLM-5.3-Flash's dense latent attention kernels on given inputs (scores, probabilities, outputs)" },
        .{ .name = "tf-moe-bench", .path = "zig/tests/moe_bench.zig", .about = "Routed-expert prompt kernels from .metal files on a skewed routing: chained speed, bits against the first" },
        .{ .name = "tf-mm-bench", .path = "zig/tests/mm_bench.zig", .about = "Prompt matmul kernels from .metal files: chained speed at one shape, bits against the first, error against fp64" },
        .{ .name = "tf-qwen36-row-seam", .path = "zig/tests/qwen36_row_seam.zig", .about = "Model-free fixture: the row projection's bias sums stay fp32 at every width" },
    };
    for (engine_programs) |p| {
        const mod = b.createModule(.{
            .root_source_file = b.path(p.path),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "tensorfold", .module = engine }, .{ .name = "cluster", .module = cluster }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "native_engines", .module = engines } },
        });
        if (p.c_source) |file| {
            mod.addCSourceFile(.{ .file = b.path(file), .flags = &.{"-std=c11"} });
        }
        const exe = b.addExecutable(.{ .name = p.name, .root_module = mod });
        b.installArtifact(exe);
        b.step(p.name, p.about).dependOn(&b.addInstallArtifact(exe, .{}).step);
    }

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
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = row_mod })).step);

    // The checkpoint CLI: models, info and pull, against a synthetic cache and a fake hub.
    const cli_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "native_engines", .module = engines }},
    }) });
    const run_cli_tests = b.addRunArtifact(cli_tests);
    test_step.dependOn(&run_cli_tests.step);
    b.step("test-cli", "The checkpoint CLI's host tests").dependOn(&run_cli_tests.step);

    // The core lane, shared-context and Qwen3.8-27B programs and host tests.
    qwen27_build.targets(b, target, optimize, metal, sources, engine, lanes, mods.qwen27, test_step);

    _ = nativeServer(b, target, optimize, metal, engine, api, engines, build_options, test_step);

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

    // Host-only unit tests: no GPU work.
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
fn metalEngineModules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module) struct { metal: *std.Build.Module, lanes: *std.Build.Module, fabric: *std.Build.Module, sources: *std.Build.Module, engine: *std.Build.Module, qwen27: qwen27_build.Shared } {
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
    const engine = b.createModule(.{
        .root_source_file = b.path("zig/src/tensorfold.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "nemotron_draft_ids", .module = draft_ids }, .{ .name = "lanes", .module = lanes }, .{ .name = "fabric", .module = fabric } },
    });
    return .{ .metal = metal, .lanes = lanes, .fabric = fabric, .sources = sources, .engine = engine, .qwen27 = qwen27_build.engineModules(b, target, metal, sources, engine) };
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
    return nativeServer(b, target, .fast, m.metal, m.engine, api, engines, build_options, null);
}

test {
    _ = dist_build;
}
