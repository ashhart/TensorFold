//! `zig build -Dgpu=<gpu>`: one stripped server for one hardware GPU, every model family, nothing else.
const std = @import("std");
const catalog = @import("cuda_kernels.zig");
const caps = @import("../src/hip/caps.zig");
const gpu_builds = @import("../src/cuda/gpu_builds.zig");

/// The `-Dgpu` name, or null for the development build.
pub fn requested(b: *std.Build) ?[]const u8 {
    return b.option([]const u8, "gpu", "Build one server for one GPU: sm_80, sm_86, sm_89, sm_120, sm_121, spark, or a HIP arch (gfx1201 ...)");
}

/// A CUDA SM as `-Dsm` spells it, or an exact HIP architecture.
const Gpu = union(enum) { cuda: []const u8, hip: []const u8 };

/// The backend and architecture `name` means, or null.
fn parseGpu(name: []const u8) ?Gpu {
    if (std.mem.eql(u8, name, "spark")) return .{ .cuda = "121" };
    if (std.mem.startsWith(u8, name, "sm_")) {
        inline for (gpu_builds.sms) |sm| if (std.mem.eql(u8, name[3..], std.fmt.comptimePrint("{d}", .{sm}))) return .{ .cuda = name[3..] };
        return null;
    }
    if (std.mem.indexOfScalar(u8, name, ':') == null and caps.Caps.of(name) != null) return .{ .hip = name };
    return null;
}

/// Every option the product reads, declared before any refusal so that `zig build --help` with any option works.
const Options = struct {
    sm: bool,
    hip: bool,
    hip_options: @import("hip.zig").Options,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    strip: bool,
    version: ?[]const u8,
    nvcc: ?[]const u8,
    prebuilt: ?[]const u8,
    /// the first option given that the product does not read (a development build option, or none at all)
    other: ?[]const u8,

    fn declare(b: *std.Build) Options {
        var o: Options = .{
            .sm = b.option([]const u8, "sm", "Refused with -Dgpu") != null,
            .hip = b.option(bool, "hip", "Refused with -Dgpu") != null,
            .hip_options = @import("hip.zig").options(b), // the only declaration of the HIP options, so they parse once
            .target = b.standardTargetOptions(.{}),
            .optimize = b.option(std.builtin.OptimizeMode, "optimize", "debug, safe, fast or small (default fast)") orelse .fast,
            .strip = b.option(bool, "strip", "No debug info in the server (default true with -Dgpu)") orelse true,
            .version = b.option([]const u8, "version", "Release version, exactly MAJOR.MINOR.PATCH"),
            .nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the kernel images"),
            .prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files, each checked for its one SASS profile"),
            .other = null,
        };
        o.other = undeclared(b); // last: it declares only what the lines above did not
        return o;
    }
};

/// Declares each option given that the product does not read, so Zig accepts it; returns the first, for a refusal.
fn undeclared(b: *std.Build) ?[]const u8 {
    var first: ?[]const u8 = null;
    for (b.user_input_options.keys(), b.user_input_options.values()) |key, value| {
        if (b.available_options_map.contains(key)) continue;
        const what = "Refused: not an option of the -Dgpu build";
        switch (value) {
            .flag => _ = b.option(bool, key, what),
            .list => _ = b.option([]const []const u8, key, what),
            else => _ = b.option([]const u8, key, what),
        }
        first = first orelse key;
    }
    return first;
}

/// Why `-Dgpu=name` with these options builds nothing, or null when it builds.
fn refusal(b: *std.Build, name: []const u8, gpu: ?Gpu, o: Options) ?[]const u8 {
    const g = gpu orelse return b.fmt("-Dgpu={s}: not a GPU this build knows; use sm_80, sm_86, sm_89, sm_120, sm_121, spark, or an exact HIP arch from zig/src/hip/caps.zig", .{name});
    if (o.sm) return "-Dsm: -Dgpu sets the architecture";
    if (o.hip) return b.fmt("-Dhip: -Dgpu={s} chooses the backend", .{name});
    if (o.hip_options.hip_arch != null) return "-Dhip-arch: -Dgpu sets the architecture";
    if (o.other) |key| return b.fmt("-D{s}: not an option of the -Dgpu={s} build; drop it, or build without -Dgpu", .{ key, name });
    const os = o.target.result.os.tag;
    switch (g) {
        .cuda => {
            for ([_][]const u8{ "hipcc", "hip-include", "hip-device-lib", "hip-relax-branches" }) |option| {
                if (b.user_input_options.contains(option)) return b.fmt("-D{s}: -Dgpu={s} builds CUDA, where the HIP options do not apply", .{ option, name });
            }
            if ((o.nvcc == null) == (o.prebuilt == null)) return b.fmt("-Dgpu={s} needs exactly one of -Dnvcc and -Dfatbins", .{name});
            if (os != .linux and os != .windows) return b.fmt("-Dgpu={s}: CUDA servers build for Linux and Windows targets", .{name});
        },
        .hip => {
            for ([_][]const u8{ "nvcc", "fatbins" }) |option| {
                if (b.user_input_options.contains(option)) return b.fmt("-D{s}: -Dgpu={s} builds HIP, where the CUDA options do not apply", .{ option, name });
            }
            if (os != .linux) return b.fmt("-Dgpu={s}: HIP servers build for Linux targets", .{name});
        },
    }
    if (o.version) |v| if (!@import("dist.zig").validVersion(v)) return "-Dversion must be exactly MAJOR.MINOR.PATCH";
    return null;
}

/// Every refusal waits for the install step (a fail step it depends on), so `--help`, `-l` and `uninstall` always work.
pub fn build(b: *std.Build, name: []const u8) void {
    const gpu = parseGpu(name);
    const o = Options.declare(b);
    if (refusal(b, name, gpu, o)) |why| return b.getInstallStep().dependOn(&b.addFail(why).step);
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", o.version orelse @import("../../build.zig.zon").version);
    build_options.addOption(bool, "slide_test_hooks", false);
    const exe = switch (gpu.?) {
        .cuda => |sm| cudaServer(b, o, build_options, sm),
        .hip => |arch| hipServer(b, o, build_options, arch),
    };
    exe.root_module.strip = o.strip; // before the install step, which decides then whether a .pdb comes with it
    const dir = switch (gpu.?) {
        .cuda => "native/bin",
        .hip => "native-hip/bin",
    };
    b.getInstallStep().dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = dir } } }).step);
    @import("sys.zig").wire(b);
    @import("checkpoint_config.zig").wire(b);
    @import("save_guard.zig").wire(b);
    @import("bf16_round.zig").wire(b, null);
    @import("qwen35_layers.zig").wire(b, null);
}

/// The development `native` server for one SM, with every image built for it (nvcc) or checked (prebuilt).
fn cudaServer(b: *std.Build, o: Options, build_options: *std.Build.Step.Options, sm: []const u8) *std.Build.Step.Compile {
    const cuda_build = @import("cuda.zig");
    const version: ?std.Build.LazyPath = if (o.nvcc) |tool| blk: {
        const run = b.addSystemCommand(&.{ tool, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    const target_sm = catalog.Target.parse(sm);
    var images: [catalog.kernels.len]?std.Build.LazyPath = @splat(null);
    const check = b.addRunArtifact(b.addExecutable(.{ .name = "check-product-fatbins", .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_product_fatbins.zig"), .target = b.graph.host }) }));
    for (catalog.kernels, &images) |k, *image| {
        if (o.nvcc) |tool| {
            image.* = @import("cuda_fatbin.zig").fatbin(b, tool, version.?, k, sm);
        } else {
            const path = b.graph.cwdRelativePath(b.pathJoin(&.{ o.prebuilt.?, b.fmt("{s}.fatbin", .{k.name}) }));
            image.* = path;
            // an image below its kernel's min_sm is an inert empty file, as the nvcc path makes it
            if (catalog.profile(k, target_sm)) |p| {
                check.addArg(b.fmt("{d}{s}", .{ p.number, if (p.arch_specific) "a" else "" }));
                check.addFileArg(path);
            }
        }
    }
    const rt = @import("cuda_runtime.zig").runtimeFor(b, o.target, o.optimize, &images, std.fmt.parseInt(u32, sm, 10) catch unreachable);
    const ids_files = b.addWriteFiles();
    _ = ids_files.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
    const draft_ids = b.createModule(.{ .root_source_file = ids_files.add("draft_ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");\n") });
    const mods = cuda_build.family(b, o.target, o.optimize, rt, draft_ids);
    const exe = cuda_build.nativeServer(b, o.target, o.optimize, rt, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision, mods.tokenizer, mods.living, build_options, false, null);
    if (o.prebuilt != null) exe.step.dependOn(&check.step);
    return exe;
}

/// The development `native-hip` server for one arch.
fn hipServer(b: *std.Build, o: Options, build_options: *std.Build.Step.Options, arch: []const u8) *std.Build.Step.Compile {
    const hip_build = @import("hip.zig");
    const hipcc = o.hip_options.hipcc orelse "hipcc";
    const hipcc_resolved = b.findProgram(.{ .names = &.{hipcc} }) orelse hipcc;
    const hip_include = o.hip_options.hip_include orelse b.pathResolve(&.{ std.fs.path.dirname(hipcc_resolved) orelse "/opt/rocm/bin", "..", "include" });
    const objects = hip_build.kernelObjects(b, hipcc, hipcc_resolved, o.hip_options.device_lib, o.hip_options.relax_branches, arch);
    const e = hip_build.engineModules(b, o.target, o.optimize, objects, hip_include);
    return hip_build.server(b, o.target, e, build_options);
}
